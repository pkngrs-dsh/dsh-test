"""Cache original-photo semantics and eyelids for successful UFF batch cases.

The parser and eyelid detector are loaded once. Cases run sequentially; existing
outputs are never replaced. All views, including rejected eye observations, are
retained so the downstream front-priority experiment can inspect its evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]

from cacheEyelidLandmarks import cacheEyelidLandmarks, readValidatedViews
from cacheFaceSemantics import cacheSemantics
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.adapters.faceParsing import FaceParsingAdapter
from faceAppearance.artifacts import jsonValue, writeJson
from renderCropComparison import fileHash


def cacheBatch(sourceManifest: Path, output: Path, *, device="cuda", minimumPeakScore=.2, resume=False):
    sourceManifest, output = sourceManifest.resolve(), output.resolve()
    source = json.loads(sourceManifest.read_text(encoding="utf-8"))
    selected = [case for case in source["cases"] if case["status"] == "succeeded"]
    manifestPath = output / "observationManifest.json"
    if manifestPath.exists() and not resume:
        raise FileExistsError(f"Existing batch observation report: {manifestPath}")
    for case in selected:
        for kind in ("semantics", "eyelids"):
            target = output / "cases" / case["caseId"] / kind
            if target.exists() and any(target.iterdir()) and not resume:
                raise FileExistsError(f"Existing observation cache: {target}")
    output.mkdir(parents=True, exist_ok=True)
    if manifestPath.exists():
        attempt = 1
        while (output / f"observationManifest.attempt{attempt}.json").exists():
            attempt += 1
        manifestPath.rename(output / f"observationManifest.attempt{attempt}.json")
    implementation = output / "observationImplementation"
    implementation.mkdir(exist_ok=True)
    implementationPath = implementation / (f"{Path(__file__).stem}-{fileHash(Path(__file__))[:12]}.py")
    if not implementationPath.exists():
        shutil.copy2(__file__, implementationPath)
    report = {
        "schemaVersion": 1, "status": "loadingModels", "startedAt": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": sourceManifest, "sourceManifestSha256": fileHash(sourceManifest),
        "device": device, "parserName": "celebm", "eyelidDevice": "cpu",
        "minimumPeakScore": minimumPeakScore, "selectedCases": [case["caseId"] for case in selected],
        "implementationSha256": fileHash(Path(__file__)), "cases": [],
        "limits": ["Automatic observations are not manual ground truth.",
                   "Shape and score gates do not independently establish eye visibility.",
                   "Image-side eye labels are assigned by semantic center proximity.",
                   "All source photographs, original UFF geometry, cameras and textures remain unchanged."]}
    writeJson(manifestPath, report)
    start = perf_counter()
    try:
        import cv2
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        begin = perf_counter()
        parser = FaceParsingAdapter(projectRoot, device=device, parserName="celebm").loadModels()
        report["parserLoadSeconds"] = perf_counter() - begin
        begin = perf_counter()
        eyelid = EyelidLandmarksAdapter(projectRoot, minimumPeakScore=minimumPeakScore).loadModels()
        report["eyelidLoadSeconds"] = perf_counter() - begin
        report["status"] = "running"
        writeJson(manifestPath, report)
        for case in selected:
            caseId = case["caseId"]
            root = output / "cases" / caseId
            root.mkdir(parents=True, exist_ok=True)
            sourceDirectory = Path(case["outputDirectory"])
            record = {"caseId": caseId, "label": case["label"], "status": "running",
                      "sourceDirectory": sourceDirectory,
                      "referenceViewIndex": case.get("referenceViewIndex", 0),
                      "semanticCache": root / "semantics", "eyelidCache": root / "eyelids"}
            report["cases"].append(record)
            writeJson(manifestPath, report)
            begin = perf_counter()
            print(f"CASE {caseId}: start", flush=True)
            try:
                if resume and (root / "semantics/manifest.json").exists():
                    _, semantic, _ = readValidatedViews(sourceDirectory, root / "semantics")
                    record["semanticCacheReused"] = True
                else:
                    semantic = cacheSemantics(sourceDirectory, root / "semantics", device=device,
                                              parserName="celebm", adapter=parser)
                record["semanticSeconds"] = semantic["totalSeconds"]
                if resume and (root / "eyelids/manifest.json").exists():
                    from runEyeRefinementExperiment import loadEyeCache
                    inputs = json.loads((sourceDirectory / "inputs.json").read_text(encoding="utf-8"))
                    _, eyes, _ = loadEyeCache(root / "eyelids", inputs,
                                             inputsSha256=fileHash(sourceDirectory / "inputs.json"))
                    if eyes["minimumPeakScore"] != minimumPeakScore:
                        raise ValueError("Existing eyelid cache uses a different threshold")
                    record["eyelidCacheReused"] = True
                else:
                    eyes = cacheEyelidLandmarks(sourceDirectory, root / "semantics", root / "eyelids",
                                               minimumPeakScore=minimumPeakScore, adapter=eyelid)
                record["eyelidSeconds"] = eyes["totalSeconds"]
                record["status"] = "complete"
                record["views"] = [{"viewIndex": view["viewIndex"],
                                    "imageSizeWh": view["imageSizeWh"],
                                    "quality": view["quality"],
                                    "semanticLabelAssignment": view["semanticLabelAssignment"],
                                    "imageSideToSemanticLabel": view["imageSideToSemanticLabel"]}
                                   for view in eyes["views"]]
                record["semanticManifestSha256"] = fileHash(root / "semantics/manifest.json")
                record["eyelidManifestSha256"] = fileHash(root / "eyelids/manifest.json")
                front = record["views"][record["referenceViewIndex"]]
                print(json.dumps({"caseId": caseId, "status": "complete", "referenceView": front},
                                 ensure_ascii=False, default=jsonValue), flush=True)
            except Exception as error:
                record.update(status="failed", errorType=type(error).__name__, error=str(error))
                (root / "observationFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"CASE {caseId}: FAILED {type(error).__name__}: {error}", flush=True)
            record["wallSeconds"] = perf_counter() - begin
            writeJson(manifestPath, report)
        report["status"] = "complete" if all(case["status"] == "complete" for case in report["cases"]) else "completeWithFailures"
    except Exception as error:
        report.update(status="failed", errorType=type(error).__name__, error=str(error))
        raise
    finally:
        report["wallSeconds"] = perf_counter() - start
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        writeJson(manifestPath, report)
    return report


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser(description=__doc__)
    argumentParser.add_argument("--manifest", type=Path,
                                default=projectRoot / "runs/uffMultiCase20261011/batchManifest.json")
    argumentParser.add_argument("--output", type=Path, default=projectRoot / "runs/uffFrontPriority20261011")
    argumentParser.add_argument("--device", default="cuda")
    argumentParser.add_argument("--minimumPeakScore", type=float, default=.2)
    argumentParser.add_argument("--resume", action="store_true", help="Reuse validated complete caches and retain prior reports")
    args = argumentParser.parse_args()
    cacheBatch(args.manifest, args.output, device=args.device, minimumPeakScore=args.minimumPeakScore, resume=args.resume)
