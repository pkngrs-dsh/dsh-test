"""Render two unique reconstructions and reuse the selected one for the guard."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import traceback

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.artifacts import writeJson
import renderUffSelfieCrop as renderer
from renderUffMultiCase import grid
from evaluateUffSafeCropValidation import BASELINE, CANDIDATE, GUARDED, DEFAULT_ROOT, decisionFor, read

LABELS = {BASELINE: "Baseline / 1.35 original", CANDIDATE: "Unconditional / 1.10 head gray", GUARDED: "Guarded selection"}
UNIQUE = (BASELINE, CANDIDATE)
DISPLAY = (*UNIQUE, GUARDED)
ORIGINAL_SOURCE_HASHES = renderer.source_hashes


def configure():
    renderer.VARIANTS = UNIQUE
    renderer.GROUPS = {"validation": UNIQUE}
    renderer.LABELS = LABELS
    renderer.source_hashes = sourceHashes


def sourceHashes(case):
    hashes = ORIGINAL_SOURCE_HASHES(case)
    paths = [Path(__file__), ROOT / "tools/evaluateUffSafeCropValidation.py"]
    path = case.get("policyDecisionPath")
    if path:
        paths.append(Path(path))
    guarded = next((v for v in case["variants"] if v["id"] == GUARDED), None)
    if guarded:
        directory = Path(guarded["outputDirectory"])
        paths.extend(directory / name for name in ("geometry.npz", "cameras.json", "inputs.json", "result.json"))
    for path in paths:
        hashes[str(path.resolve())] = renderer.digest(path)
    return hashes


def addGuardedPanels(case, root, destination, report):
    decision = decisionFor(case, root)
    if decision["status"] != "available":
        raise ValueError("Frozen policy decision is unavailable")
    selected = decision["selectedVariantId"]
    entries = {v["id"]: v for v in case["variants"]}
    if GUARDED not in entries:
        raise ValueError("Manifest needs guarded_selected alias for viewer geometry")
    chosenDirectory = Path(entries[selected]["outputDirectory"])
    aliasDirectory = Path(entries[GUARDED]["outputDirectory"])
    identity = {}
    for name in ("geometry.npz", "cameras.json", "inputs.json"):
        originalHash, aliasHash = renderer.digest(chosenDirectory / name), renderer.digest(aliasDirectory / name)
        if originalHash != aliasHash:
            raise ValueError(f"Guarded alias is not the selected saved result: {name}")
        identity[name] = originalHash
    guardedLabel = "Guarded -> " + ("1.10 head gray" if selected == CANDIDATE else "1.35 original")
    for view in report["views"]:
        index = view["viewIndex"]
        directory = destination / f"view{index:02d}"
        for suffix in ("fixed_clay", "fixed_visible", "fixed_overlay", "gray", "eyes", "overlay"):
            shutil.copy2(directory / f"{selected}_{suffix}.png", directory / f"{GUARDED}_{suffix}.png")
        for key, suffix, height in (("fixedComparison", "gray", 360), ("fixedEyes", "eyes", 210), ("fixedOverlays", "overlay", 360)):
            paths = [directory / f"photo_{suffix}.png"] + [directory / f"{variant}_{suffix}.png" for variant in DISPLAY]
            panels = [Image.open(path).convert("RGB") for path in paths]
            grid(panels, ["Original photograph", LABELS[BASELINE], LABELS[CANDIDATE], guardedLabel], 4,
                 panel_size=360, panel_height=height,
                 subtitle=f"{case['caseId']} / view {index} | frozen baseline photo camera; guard reuses selected mesh").save(directory / f"{key}.png")
        panels = [Image.open(Path(entries[variant]["outputDirectory"]) / f"input{index:02d}.png").convert("RGB") for variant in DISPLAY]
        grid(panels, [LABELS[BASELINE], LABELS[CANDIDATE], guardedLabel], 3, panel_size=400,
             subtitle=f"{case['caseId']} / view {index} | actual model inputs; guard uses all four selected frames").save(directory / "inputContact.png")
        view["variants"][GUARDED] = {**view["variants"][selected], "reusedFrom": selected, "renderRepeated": False}
        for prefix, wanted in (("front", report["selectedFrontView"]), ("side", report["selectedSideView"])):
            if wanted == index:
                for key in renderer.PANEL_FILES:
                    shutil.copy2(directory / f"{key}.png", destination / f"{prefix}{key[0].upper()+key[1:]}.png")
    report.update({"variantIds": list(DISPLAY), "uniqueRenderedVariantIds": list(UNIQUE),
        "guardedSelectedVariantId": selected, "policyDecision": decision,
        "guardedGeometryAndCameraIdentity": identity, "guardedInferenceRepeated": False,
        "guardedRenderRepeated": False, "wrapperSha256": renderer.digest(__file__)})
    report["artifactHashes"] = {str(path.relative_to(destination)): renderer.digest(path) for path in sorted(destination.rglob("*.png"))}
    writeJson(destination / "diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximumPhotoSide", type=int, default=768)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(root / "cropManifest.json")
    configure()
    output = root / "preview"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    snapshots = output / "manifestSnapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f"{renderer.digest(root / 'cropManifest.json')}.json"
    if not snapshot.exists():
        shutil.copy2(root / "cropManifest.json", snapshot)
    reports = []
    for case in manifest["cases"]:
        if args.case and case["caseId"] not in args.case:
            continue
        destination = output / case["caseId"]
        try:
            report = renderer.resume_case(case, destination, args.maximumPhotoSide)
            if report is None:
                report = renderer.render_case(case, destination, args.device, args.maximumPhotoSide)
            if GUARDED not in report.get("variantIds", []):
                report = addGuardedPanels(case, root, destination, report)
        except Exception as error:
            report = {"caseId": case["caseId"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            print(report["traceback"], flush=True)
        reports.append(report)
        writeJson(output / "progress.json", reports)
    writeJson(output / "diagnostics.json", {"status": "completed" if all(r["status"] == "rendered" for r in reports) else "partial", "cases": reports,
        "manifestSha256": renderer.digest(root / "cropManifest.json"), "wrapperSha256": renderer.digest(__file__)})
    if any(r["status"] == "failed" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
