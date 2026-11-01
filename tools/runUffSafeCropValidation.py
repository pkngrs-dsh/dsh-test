"""Validate a frozen input-only safe-crop rule on held-out photo groups.

Runs a fresh baseline, an ungated tight/head-gray candidate and one baseline
repeat. The guarded result is an exact asset copy selected before inference,
not an additional reconstruction or a selection using reconstruction scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import runUffSelfieCropStudy as study
from faceAppearance.artifacts import writeJson
from renderUffMultiCase import grid

RUN_VARIANTS = (
    ("m135_raw", "m135", 1.35, "raw", "基线：1.35 原背景"),
    ("m110_headGray", "m110", 1.10, "headGray", "无保护：1.10 头部灰底"),
    ("m135_repeat", "m135", 1.35, "raw", "1.35 原背景重复"),
)


def freeze_policy(root, selection_path, policy):
    record = {"policyVersion": policy.POLICY_VERSION, "configuration": policy.CONFIG,
              "implementationSha256": study.sha(policy.__file__),
              "runnerSha256": study.sha(__file__),
              "sharedRunnerSha256": study.sha(study.__file__),
              "selectionSha256": study.sha(selection_path),
              "developmentRoot": str(ROOT / "runs/uffSelfieCropStudy20261012"),
              "rule": "All four inputs must pass; otherwise route the entire group to m135_raw.",
              "decisionInputs": "Original detections, automatic semantics, and crop metadata only; no reconstruction metrics.",
              "heldOutMeaning": "Exact photo file-disjoint groups, not verified identity-disjoint participants or held-out views.",
              "retakeMeaning": "Source truncation/parser uncertainty is flagged; fallback is not asserted to repair missing content."}
    path = root / "policyFrozen.json"
    if path.exists():
        prior = study.read(path)
        for key, value in record.items():
            if prior.get(key) != value:
                raise ValueError(f"Frozen protocol changed: {key}; use a new experiment root")
    else:
        record["frozenAt"] = datetime.now(timezone.utc).isoformat()
        writeJson(path, record)
    return study.sha(path)


def guarded_assets(case, decision):
    source = next(v for v in case["variants"] if v["id"] == decision["selectedVariantId"])
    target = next(v for v in case["variants"] if v["id"] == "guarded_selected")
    src, dst = Path(source["outputDirectory"]), Path(target["outputDirectory"])
    dst.mkdir(parents=True, exist_ok=True)
    result = study.read(src / "result.json")
    if (dst / "result.json").exists():
        prior = study.read(dst / "result.json")
        if prior.get("selectedSourceResultSha256") != study.sha(src / "result.json"):
            raise ValueError("Guarded asset source changed")
        for name, expected in prior["outputHashes"].items():
            if study.sha(dst / name) != expected:
                raise ValueError("Guarded asset changed")
    else:
        # Copy the immutable assets, keeping the selected geometry exactly equal.
        for item in src.iterdir():
            if item.is_file() and item.name != "result.json":
                shutil.copy2(item, dst / item.name)
        result.update(variantId="guarded_selected", inferencePerformed=False,
                      selectedSourceVariantId=source["id"],
                      selectedSourceResultSha256=study.sha(src / "result.json"),
                      policyDecisionSha256=study.sha(case["policyDecisionPath"]),
                      geometrySecondsIncludingDiagnostics=0., cameraFitSeconds=0.)
        writeJson(dst / "result.json", result)
    target.update(status="succeeded", cropKey=source["cropKey"], margin=source["margin"],
                  background=source["background"], cropMode="perView",
                  selectedSourceVariantId=source["id"], inferencePerformed=False,
                  geometrySeconds=0., acceptedCameraCount=source["acceptedCameraCount"],
                  resultSha256=study.sha(dst / "result.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "runs/uffSelfieSafeCropValidation20261012")
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--case", action="append")
    args = ap.parse_args()
    import selfieSafeCropPolicy as policy
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selection_path = args.selection or root / "selection/selection.json"
    selection = study.read(selection_path)
    frozen_hash = freeze_policy(root, selection_path, policy)
    manifest_path = root / "cropManifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Existing validation: use --resume")
        manifest = study.read(manifest_path)
        if manifest["selectionSha256"] != study.sha(selection_path) or manifest["policyFrozenSha256"] != frozen_hash:
            raise ValueError("Frozen selection/protocol changed")
    else:
        cases = []
        for record in selection["cases"]:
            case_root = root / "cases" / record["caseId"]
            case = {**record, "status": "pending", "referenceViewIndex": 0,
                    "sourceDirectory": str(case_root / "m135_raw"),
                    "photoSha256": [study.sha(p) for p in record["paths"]],
                    "policyDecisionPath": str(case_root / "policyDecision.json"), "variants": []}
            for variant_id, crop, margin, bg, label in RUN_VARIANTS:
                case["variants"].append({"id": variant_id, "cropKey": crop, "margin": margin,
                    "background": bg, "cropMode": "perView", "label": label, "status": "pending",
                    "outputDirectory": str(case_root / variant_id), "inferencePerformed": True})
            case["variants"].append({"id": "guarded_selected", "cropKey": None,
                "label": "输入保护后选用", "status": "pending", "inferencePerformed": False,
                "outputDirectory": str(case_root / "guarded_selected")})
            cases.append(case)
        manifest = {"schemaVersion": 1, "status": "preparing", "createdAt": datetime.now(timezone.utc).isoformat(),
            "selectionPath": str(selection_path.resolve()), "selectionSha256": study.sha(selection_path),
            "policyFrozenSha256": frozen_hash, "seed": 2026, "referenceVariant": "m135_raw",
            "design": "6 held-out groups x baseline / ungated candidate / baseline repeat; input-only guarded route is an asset alias",
            "cases": cases, "limits": ["Input photos are file-disjoint from development; subject identity is unverified.",
                "Both eye proxies use reconstruction input photos, not independent ground truth.",
                "Guard handles input coverage, not guaranteed prediction accuracy or identity similarity.",
                "Retake flags do not stop offline diagnostic inference; flagged output is not recommended for automatic delivery."]}
        study.persist(manifest_path, manifest)
    study.CROPS = (("m135", 1.35, "perView"), ("m110", 1.10, "perView"))
    import torch
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    begin = perf_counter()
    adapter = study.GeometryAdapter(ROOT, "cuda", "staged", depthChunkSize=1).loadModels()
    parser = study.FaceParsingAdapter(ROOT, device="cuda", parserName="celebm").loadModels()
    eyelid = study.EyelidLandmarksAdapter(ROOT).loadModels()
    manifest["runtime"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "modelLoadSeconds": perf_counter()-begin,
        "cudnnDeterministic": torch.backends.cudnn.deterministic}
    manifest["modelHashes"] = {k: study.sha(v) for k, v in adapter.assetPaths().items()}
    manifest["status"] = "running"
    study.persist(manifest_path, manifest)
    for case in manifest["cases"]:
        if args.case and case["caseId"] not in args.case:
            continue
        case_root = root / "cases" / case["caseId"]
        case_root.mkdir(parents=True, exist_ok=True)
        print(f"CASE {case['caseId']} PREPARE", flush=True)
        try:
            if freeze_policy(root, selection_path, policy) != frozen_hash:
                raise ValueError("Frozen protocol changed during run")
            photos = study.originalPhotos(case)
            crops, detections = study.prepareMetadata(case, case_root, adapter, photos)
            study.cacheOriginalObservations(case, case_root, photos, detections, parser, eyelid)
            del photos
            decision = policy.evaluate_case(case_root, case_id=case["caseId"])
            # Freeze the route before reconstructing either candidate. The policy
            # reads input-only artifacts even on resumed runs.
            study.saveInputs(Path(case["policyDecisionPath"]), decision)
            case["policyDecision"] = decision
            case["policyDecisionSha256"] = study.sha(case["policyDecisionPath"])
            case["decisionRecordedBeforeInference"] = True
            case["status"] = "running"
            study.persist(manifest_path, manifest)
            print(f"ROUTE {case['caseId']} {decision['selectedVariantId']} retake={decision.get('retakeRequired')}", flush=True)
            for crop, _, _ in study.CROPS:
                images, inputs = crops[crop]
                arrays, info = study.cropArrays(case_root, crop, images, inputs, detections, adapter, manifest["seed"])
                for variant in case["variants"]:
                    if variant["cropKey"] != crop or variant["id"] in ("m135_repeat", "guarded_selected"):
                        continue
                    if variant["status"] == "succeeded":
                        result = study.read(Path(variant["outputDirectory"]) / "result.json")
                        for name, expected in result["outputHashes"].items():
                            if study.sha(Path(variant["outputDirectory"]) / name) != expected:
                                raise ValueError("Completed result changed")
                        continue
                    variant["status"] = "running"
                    study.persist(manifest_path, manifest)
                    variant.update(study.reconstruct(case, variant, arrays, adapter, manifest["seed"]))
                    print(f"DONE {case['caseId']} {variant['id']} {variant['geometrySeconds']:.2f}s", flush=True)
                    study.persist(manifest_path, manifest)
                del arrays
            repeat = next(v for v in case["variants"] if v["id"] == "m135_repeat")
            if repeat["status"] != "succeeded":
                with np.load(case_root / "cropInputs/m135/arrays.npz", allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                repeat.update(study.reconstruct(case, repeat, arrays, adapter, manifest["seed"]))
                del arrays
            guarded_assets(case, decision)
            panels, titles = [], []
            for key, label in (("m135_raw", "Baseline 1.35 raw"), ("m110_headGray", "Ungated 1.10 head gray"), ("guarded_selected", "Guarded selected")):
                panels.append(Image.open(case_root / key / "input00.png").convert("RGB")); titles.append(label)
            grid(panels, titles, 3, panel_size=320,
                 subtitle=f"{case['caseId']} | selected {decision['selectedVariantId']} | retake {decision.get('retakeRequired')}").save(case_root / "inputFronts.png")
            study.originalPhotos(case)
            case["status"] = "succeeded"
        except Exception as error:
            case.update(status="failed", error=str(error))
            (case_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
        study.persist(manifest_path, manifest)
        gc.collect()
        torch.cuda.empty_cache()
    manifest["status"] = "complete" if all(c["status"] == "succeeded" for c in manifest["cases"]) else "partial"
    manifest["lastRunSecondsIncludingLoad"] = perf_counter()-begin
    study.persist(manifest_path, manifest)
    source_paths = [Path(__file__), Path(policy.__file__), Path(study.__file__),
        ROOT / "src/faceAppearance/data.py", ROOT / "src/faceAppearance/faceCrop.py", ROOT / "src/faceAppearance/adapters/geometry.py"]
    for source in source_paths:
        target = root / "implementation" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    writeJson(root / "implementationHashes.json", {str(p.relative_to(ROOT)): study.sha(p) for p in source_paths})
    if any(c["status"] == "failed" for c in manifest["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
