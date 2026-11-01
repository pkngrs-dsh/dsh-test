"""Independent CPU provenance/routing audit for the frozen safe-crop study.

Never changes the frozen policy, selection, reconstructions or decisions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import selfieSafeCropPolicy as policy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ahash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_case(case, selected, root, log_lines, frozen_path):
    case_root = root / "cases" / case["caseId"]
    result = {"caseId": case["caseId"], "status": "checking", "photos": [], "variants": [], "checks": {}}
    require(case["status"] == "succeeded", "Case not succeeded")
    require(case["paths"] == selected["paths"], "Selected photo order changed")
    require(len(case["paths"]) == 4, "Expected four photos")
    for i, (path, expected, file) in enumerate(zip(case["paths"], selected["sourceHashes"], selected["files"])):
        require(path == file["path"], "Selection file order changed")
        actual = sha(path)
        require(actual == expected == file["sha256"] == case["photoSha256"][i], "Frozen source/manifest/original hash mismatch")
        with Image.open(path) as source:
            source.load()
            photo = ImageOps.exif_transpose(source).convert("RGB")
            oriented = list(photo.size)
            decoded = hashlib.sha256(str(oriented).encode() + photo.tobytes()).hexdigest()
        require(oriented == file["orientedSize"], "Oriented size changed")
        require(decoded == file["decodedOrientedRgbSha256"], "Decoded original RGB changed")
        result["photos"].append({"viewIndex": i, "path": path, "sha256": actual,
                                  "decodedOrientedRgbSha256": decoded, "frozenSelectionManifestOriginalAgree": True})
    decision_path = case_root / "policyDecision.json"
    decision = read(decision_path)
    regenerated = policy.evaluate_case(case_root, case_id=case["caseId"])
    require(regenerated == decision == case["policyDecision"], "Input-only policy decision not deterministic")
    require(sha(decision_path) == case["policyDecisionSha256"], "Frozen decision hash mismatch")
    for path, expected in decision["inputEvidenceSha256"].items():
        require(sha(path) == expected, "Policy input evidence changed")
    require(decision["config"] == policy.CONFIG and decision["implementationSha256"] == sha(policy.__file__), "Decision used different policy")
    expected_route = "m110_headGray" if all(v["accepted"] for v in decision["perViewChecks"]) else "m135_raw"
    require(decision["selectedVariantId"] == expected_route, "Whole-case route does not match all-four gate")
    route_lines = [i for i, line in enumerate(log_lines) if line.startswith(f"ROUTE {case['caseId']} ")]
    require(route_lines, "Route event absent from run log")
    done_lines = {v: [i for i, line in enumerate(log_lines) if line.startswith(f"DONE {case['caseId']} {v} ")]
                  for v in ("m135_raw", "m110_headGray")}
    require(all(lines and route_lines[0] < lines[0] for lines in done_lines.values()), "Route must precede first A/B DONE log events")
    first_decision_ns = decision_path.stat().st_mtime_ns
    geometry_times = {v: (case_root / v / "geometry.npz").stat().st_mtime_ns
                      for v in ("m135_raw", "m110_headGray", "m135_repeat")}
    require(frozen_path.stat().st_mtime_ns <= first_decision_ns, "Decision saved before protocol freeze")
    require(all(first_decision_ns < timestamp for timestamp in geometry_times.values()), "Decision not recorded before independent geometry save")
    result["decision"] = {"selectedVariantId": expected_route, "deterministicReevaluationEqual": True,
        "decisionSha256": sha(decision_path), "inputEvidenceFileCount": len(decision["inputEvidenceSha256"]),
        "retakeRequired": decision["retakeRequired"], "manualReviewRequired": decision["manualReviewRequired"],
        "retakeOrReviewRecommendation": decision["retakeOrReviewRecommendation"],
        "mtimeEvidence": {"policyFrozenNs": frozen_path.stat().st_mtime_ns, "decisionNs": first_decision_ns,
                          "independentGeometryNs": geometry_times},
        "logEvidence": {"firstRouteLineOneBased": route_lines[0] + 1,
                        "firstDoneLinesOneBased": {k: v[0] + 1 for k, v in done_lines.items()}}}
    arrays = {}
    for crop in ("m135", "m110"):
        cache = case_root / "cropInputs" / crop / "arrays.npz"
        metadata = read(cache.with_name("metadata.json"))
        require(sha(cache) == metadata["arraysSha256"], "Crop array cache hash changed")
        with np.load(cache, allow_pickle=False) as archive:
            arrays[crop] = {k: archive[k] for k in ("raw", "headGray", "uvMaps", "faceMasks")}
    variants = {v["id"]: v for v in case["variants"]}
    require(set(variants) == {"m135_raw", "m110_headGray", "m135_repeat", "guarded_selected"}, "Unexpected variant set")
    geometries = {}
    source_results = {}
    for name, variant in variants.items():
        directory = Path(variant["outputDirectory"])
        require(directory.resolve() == (case_root / name).resolve(), "Unexpected output directory")
        require(variant["status"] == "succeeded", "Variant not succeeded")
        require(sha(directory / "result.json") == variant["resultSha256"], "Result JSON hash changed")
        record = read(directory / "result.json")
        source_results[name] = record
        hashes = record["outputHashes"]
        for filename, expected in hashes.items():
            require(sha(directory / filename) == expected, "Saved output hash changed")
        inputs = read(directory / "inputs.json")
        require([item["sha256"] for item in inputs] == case["photoSha256"], "Variant original input photo hashes changed")
        crop = variant["cropKey"]
        cache = arrays[crop]
        expected_hashes = {"images": ahash(cache[variant["background"]]), "uvMaps": ahash(cache["uvMaps"]), "faceMasks": ahash(cache["faceMasks"])}
        require(record["inputArraysSha256"] == expected_hashes, "Inference input arrays differ from frozen cache")
        with np.load(directory / "geometry.npz", allow_pickle=False) as archive:
            geometry = {k: archive[k] for k in archive.files}
        require(geometry["vertices"].shape == (6986, 3) and geometry["faces"].shape == (13632, 3), "Unexpected topology size")
        require(all(np.isfinite(value).all() for value in geometry.values()), "Nonfinite geometry artifact")
        require(np.array_equal(geometry["uvMaps"], cache["uvMaps"]) and np.array_equal(geometry["faceMasks"], cache["faceMasks"]), "Frozen UV/masks changed during inference")
        geometries[name] = {k: geometry[k] for k in ("vertices", "faces", "uvs")}
        expected_inference = name != "guarded_selected"
        require(variant["inferencePerformed"] is expected_inference, "Independent inference/alias flag incorrect")
        result["variants"].append({"variantId": name, "inferencePerformed": expected_inference,
            "checkedOutputFileCount": len(hashes), "resultSha256": sha(directory / "result.json"),
            "geometrySha256": sha(directory / "geometry.npz"), "topology": {"vertices": 6986, "triangles": 13632},
            "finiteGeometry": True, "inputHashesAndUvMasksMatchFrozenCache": True})
    baseline = geometries["m135_raw"]
    require(all(np.array_equal(g["faces"], baseline["faces"]) and np.array_equal(g["uvs"], baseline["uvs"])
                for g in geometries.values()), "Topology or vertex UVs differ across variants")
    source, alias = source_results[expected_route], source_results["guarded_selected"]
    require(alias["selectedSourceVariantId"] == expected_route == variants["guarded_selected"]["selectedSourceVariantId"], "Alias selects wrong variant")
    require(alias["selectedSourceResultSha256"] == sha(case_root / expected_route / "result.json"), "Alias source result hash changed")
    require(alias["policyDecisionSha256"] == sha(decision_path), "Alias decision hash changed")
    require(alias["inferencePerformed"] is False and alias["geometrySecondsIncludingDiagnostics"] == 0 and alias["cameraFitSeconds"] == 0,
            "Alias falsely counted as inference")
    require(alias["outputHashes"] == source["outputHashes"] and alias["inputArraysSha256"] == source["inputArraysSha256"], "Alias differs from selected source assets")
    alias_count = 0
    for item in (case_root / expected_route).iterdir():
        if item.is_file() and item.name != "result.json":
            require(sha(item) == sha(case_root / "guarded_selected" / item.name), "Alias file is not an exact copy")
            alias_count += 1
    repeat = source_results["m135_repeat"]
    require(repeat["inputArraysSha256"] == source_results["m135_raw"]["inputArraysSha256"], "Repeat uses different inference inputs")
    require(read(case_root / "m135_repeat/inputs.json") == read(case_root / "m135_raw/inputs.json"), "Repeat input metadata changed")
    delta = geometries["m135_repeat"]["vertices"].astype(float) - baseline["vertices"].astype(float)
    norms = np.linalg.norm(delta, axis=1)
    extent = float(np.linalg.norm(np.ptp(baseline["vertices"], axis=0)))
    result["repeatControl"] = {"sameInputArraysAndMetadata": True, "vertexRmsDistance": float(np.sqrt(np.mean(norms**2))),
        "vertexMaxDistance": float(norms.max()), "vertexRmsDividedByBaselineDiagonal": float(np.sqrt(np.mean(norms**2)) / extent),
        "interpretation": "One repeat estimates observed numerical variation; no statistical significance claim."}
    result["alias"] = {"selectedSourceVariantId": expected_route, "allSelectedFilesExactCopies": True,
                         "exactCopiedFileCount": alias_count, "countsAsIndependentInference": False}
    result["checks"] = {"sourceIdentity": True, "policyFrozenAndDeterministic": True,
        "selectionBeforeSavedIndependentGeometry": True, "logOrderConsistent": True,
        "allOutputHashesValid": True, "repeatInputsEqual": True, "aliasExact": True}
    result["status"] = "passed"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs/uffSelfieSafeCropValidation20261012")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest_path = root / "cropManifest.json"
    manifest = read(manifest_path)
    require(manifest["status"] == "complete", "Wait for completed manifest; audit does not poll")
    selection_path = Path(manifest["selectionPath"])
    selection = read(selection_path)
    frozen_path = root / "policyFrozen.json"
    frozen = read(frozen_path)
    runner = ROOT / "tools/runUffSafeCropValidation.py"
    shared = ROOT / "tools/runUffSelfieCropStudy.py"
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "status": "checking",
        "auditImplementationSha256": sha(__file__), "manifestSha256": sha(manifest_path),
        "selectionSha256": sha(selection_path), "policyFrozenSha256": sha(frozen_path),
        "checks": {}, "cases": [], "failures": [],
        "limits": ["File hashes establish retained content and linkage; local mtimes/logs are corroborating evidence, not independently signed timestamps.",
            "The frozen runner source orders decision persistence before UV prediction/reconstruction; STFR caches are prepared earlier but policy never reads them.",
            "Alias copy2 preserves source mtimes; alias mtimes are excluded from inference chronology.",
            "Source file disjointness does not establish different people or held-out views; geometry accuracy is outside this audit.",
            "RetakeRequired indicates original boundary/padding risk; manualReviewRequired can include normal self-occlusion or uncertain parsing."]}
    try:
        require(frozen["configuration"] == policy.CONFIG and frozen["policyVersion"] == policy.POLICY_VERSION, "Frozen config changed")
        require(frozen["implementationSha256"] == sha(policy.__file__), "Frozen policy script changed")
        require(frozen["runnerSha256"] == sha(runner) and frozen["sharedRunnerSha256"] == sha(shared), "Frozen reconstruction code changed")
        require(sha(selection_path) == frozen["selectionSha256"] == manifest["selectionSha256"], "Frozen selection changed")
        require(sha(frozen_path) == manifest["policyFrozenSha256"], "Frozen protocol record changed")
        implementations = read(root / "implementationHashes.json")
        for relative, expected in implementations.items():
            require(sha(ROOT / relative) == expected == sha(root / "implementation" / relative), "Snapshot/source implementation mismatch")
        source = runner.read_text(encoding="utf-8")
        sequence = [source.index(text) for text in ("decision = policy.evaluate_case", "study.saveInputs(Path(case[\"policyDecisionPath\"])",
                                                   "arrays, info = study.cropArrays", "variant.update(study.reconstruct")]
        require(sequence == sorted(sequence), "Runner decision/inference source order unexpected")
        selected = {c["caseId"]: c for c in selection["cases"]}
        require(set(selected) == {c["caseId"] for c in manifest["cases"]}, "Selected case set changed")
        hashes = [value for c in selection["cases"] for value in c["sourceHashes"]]
        require(len(hashes) == len(set(hashes)) == 24, "Expected 24 unique selected photo files")
        overlap = {}
        for key, hash_key in (("priorSelectionPath", "priorSelectionSha256"), ("developmentSelectionPath", "developmentSelectionSha256")):
            prior_path = Path(selection[key])
            require(sha(prior_path) == selection[hash_key], "Prior/development selection record changed")
            prior = read(prior_path)
            prior_hashes = {f["sha256"] for c in prior["cases"] for f in c["files"]}
            overlap[key] = sorted(set(hashes) & prior_hashes)
            require(not overlap[key], "Photo file overlap with development/prior groups")
        report["checks"].update(frozenPolicyConfigAndCode=True, frozenSelection=True, sourceImplementationSnapshot=True,
            runnerDecisionBeforeInferenceSourceOrder=True, sourcePhotoFileDisjointFromPriorAndDevelopment=True)
        report["priorFileOverlap"] = overlap
        logs = (root / "run.log").read_text(encoding="utf-8").splitlines()
        report["runLogSha256"] = sha(root / "run.log")
        for case in manifest["cases"]:
            try:
                report["cases"].append(audit_case(case, selected[case["caseId"]], root, logs, frozen_path))
            except Exception as exc:
                report["cases"].append({"caseId": case["caseId"], "status": "failed", "error": str(exc)})
                report["failures"].append({"caseId": case["caseId"], "error": str(exc)})
        actual = [v for c in manifest["cases"] for v in c["variants"] if v["inferencePerformed"]]
        aliases = [v for c in manifest["cases"] for v in c["variants"] if not v["inferencePerformed"]]
        require(len(actual) == 18 and len(aliases) == 6, "Expected 18 real inferences and six selected aliases")
        report["summary"] = {"caseCount": len(manifest["cases"]), "checkedSourcePhotoCount": len(hashes),
            "independentInferenceCount": len(actual), "guardedAliasCount": len(aliases),
            "outputHashChecks": sum(v["checkedOutputFileCount"] for c in report["cases"] for v in c.get("variants", [])),
            "passedCaseCount": sum(c["status"] == "passed" for c in report["cases"]),
            "candidateSelectedCount": sum(c.get("decision", {}).get("selectedVariantId") == "m110_headGray" for c in report["cases"]),
            "fallbackSelectedCount": sum(c.get("decision", {}).get("selectedVariantId") == "m135_raw" for c in report["cases"]),
            "retakeRequiredCaseCount": sum(c.get("decision", {}).get("retakeRequired", False) for c in report["cases"]),
            "manualReviewRequiredCaseCount": sum(c.get("decision", {}).get("manualReviewRequired", False) for c in report["cases"])}
    except Exception as exc:
        report["failures"].append({"scope": "globalProtocol", "error": str(exc)})
    report["status"] = "passed" if not report["failures"] else "failed"
    destination = root / "protocolAudit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "status": report["status"], "summary": report.get("summary"), "failures": report["failures"]}, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
