#!/usr/bin/env python3
"""Run the geometry adapter on a selected four-view preflight case.

Example: .venv/bin/python tools/smokeGeometry.py --repeat 2
Use --dryRun to validate/decode the selected inputs without loading models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def loadSelectedViews(projectRoot: Path, manifestPath: Path,
                      inputRoot: Path | None, caseId: str | None):
    from faceAppearance.data import readImage

    manifest = json.loads(manifestPath.read_text())
    cases = [case for case in manifest["cases"] if len(case.get("subsets", {}).get("4") or []) == 4]
    if caseId:
        cases = [case for case in cases if case["caseId"] == caseId]
    if not cases:
        raise ValueError("No matching four-view subset in the preflight manifest")
    case = cases[0]
    if inputRoot is None:
        localData = projectRoot / "TestData"
        inputRoot = localData if localData.is_dir() else Path(manifest["sourceRoot"])
    inputRoot = inputRoot.resolve()
    expectedHashes = {item["relativePath"]: item["sha256"] for item in case.get("files", [])}
    crops, metadata = [], []
    for filename in case["subsets"]["4"]:
        relativePath = str(Path(case["relativeGroupPath"]) / filename)
        path = inputRoot / relativePath
        fileBytes = path.read_bytes()
        digest = hashlib.sha256(fileBytes).hexdigest()
        if relativePath in expectedHashes and digest != expectedHashes[relativePath]:
            raise ValueError(f"Input changed since preflight: {path}")
        # Share the pipeline's EXIF handling, crop and resampling exactly.
        crop, imageMetadata = readImage(path)
        crops.append(crop)
        metadata.append(imageMetadata)
    return case, np.stack(crops), metadata


def main() -> None:
    projectRoot = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(projectRoot / "src"))
    from faceAppearance.runtime import configureRuntime

    configureRuntime(projectRoot)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=projectRoot / "reports/preflight/candidateCases.json")
    parser.add_argument("--inputRoot", type=Path)
    parser.add_argument("--caseId")
    parser.add_argument("--outputRoot", type=Path, default=projectRoot / "reports/geometrySmoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memoryPolicy", choices=("staged", "resident"), default="staged")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dryRun", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    case, images, inputMetadata = loadSelectedViews(projectRoot, args.manifest, args.inputRoot, args.caseId)
    args.outputRoot.mkdir(parents=True, exist_ok=True)
    report = {"caseId": case["caseId"], "inputs": inputMetadata,
              "imageShape": list(images.shape), "memoryPolicy": args.memoryPolicy,
              "device": args.device, "status": "inputValidated"}
    print(f"Selected {case['caseId']}: {len(images)} verified views", flush=True)
    if args.dryRun:
        (args.outputRoot / "inputReport.json").write_text(json.dumps(report, indent=2))
        print("Input validation passed; models were not loaded", flush=True)
        return

    import torch
    import trimesh
    from faceAppearance.adapters.geometry import GeometryAdapter

    adapter = GeometryAdapter(projectRoot, device=args.device, memoryPolicy=args.memoryPolicy)
    start = time.perf_counter()
    print("Loading local geometry weights", flush=True)
    adapter.loadModels()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    report["loadSeconds"] = time.perf_counter() - start
    modelIdentities = {name: id(model) for name, model in adapter.models.items()}
    report["inferenceSeconds"] = []
    for runIndex in range(args.repeat):
        print(f"Reconstruction {runIndex + 1}/{args.repeat}", flush=True)
        start = time.perf_counter()
        result = adapter.reconstruct(images)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        report["inferenceSeconds"].append(time.perf_counter() - start)
    assert modelIdentities == {name: id(model) for name, model in adapter.models.items()}, "Models were replaced"
    for name in ("vertices", "uvs", "uvMaps", "faceMasks", "cameraIntrinsics", "cameraExtrinsics"):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"Nonfinite output: {name}")
    arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    arrays["images512"] = images
    arrays["viewScales"] = np.array([transform["s"] for transform in result["viewTransforms"]])
    arrays["viewRotations"] = np.stack([transform["R"] for transform in result["viewTransforms"]])
    arrays["viewTranslations"] = np.stack([transform["t"] for transform in result["viewTransforms"]])
    np.savez_compressed(args.outputRoot / "geometry.npz", **arrays)
    trimesh.Trimesh(result["vertices"], result["faces"], process=False).export(args.outputRoot / "geometry.ply")
    for viewIndex, image in enumerate(images):
        cv2.imwrite(str(args.outputRoot / f"input{viewIndex:02d}.png"), np.rint(image[..., ::-1] * 255).astype(np.uint8))
        cv2.imwrite(str(args.outputRoot / f"mask{viewIndex:02d}.png"), np.rint(result["faceMasks"][viewIndex] * 255).astype(np.uint8))
    report.update(status="passed", vertexCount=len(result["vertices"]), faceCount=len(result["faces"]),
                  supportFraction=float(result["supportMask"].mean()), modelsReused=True,
                  modelDevices={name: str(next(model.parameters()).device) for name, model in adapter.models.items()})
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    (args.outputRoot / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
