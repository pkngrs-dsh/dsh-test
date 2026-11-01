"""Controlled VGGT RGB-context ablation with frozen UVs, masks and crops.

Loads only VGGT and UFF fusion. No new detection, UV prediction, delighting,
texture fusion, anatomical refinement or client assembly is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.adapters.geometry import GeometryAdapter
from faceAppearance.artifacts import exportMesh, writeImage, writeJson
from faceAppearance.camera import fitViewCameras
from faceAppearance.photoCameras import canonicalCamera
from runReferenceExperiment import readArrays, fileHash

VARIANTS = [("original", "baseline", "A 原始背景"),
            ("backgroundGray", "bgRemovedGray", "B 仅去环境背景 · 灰底"),
            ("headGray", "headOnlyGray", "C 保留头部 · 灰底"),
            ("headBlack", "headOnlyBlack", "D 保留头部 · 黑底"),
            ("originalRepeat", "baseline", "A′ 原输入重复")]


def arrayHash(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str((values.dtype.str, values.shape)).encode()+values.tobytes()).hexdigest()


def persist(path, manifest):
    temporary = path.with_suffix(".writing.json")
    writeJson(temporary, manifest)
    temporary.replace(path)


def runVariant(adapter, case, variant, images, original, output, seed):
    import torch
    output.mkdir(parents=True, exist_ok=False)
    source = Path(case["sourceDirectory"])
    for name in ("inputs.json",):
        shutil.copy2(source / name, output / name)
    for index, image in enumerate(images):
        writeImage(output / f"input{index:02d}.png", image)
    inputHashes = {"images": arrayHash(images), "uvMaps": arrayHash(original["uvMaps"]),
                   "faceMasks": arrayHash(original["faceMasks"])}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = perf_counter()
    geometry = adapter.reconstructFromCachedUvs(images, original["uvMaps"], original["faceMasks"],
                                                diagnosticsDirectory=output / "diagnostics")
    torch.cuda.synchronize()
    geometrySeconds = perf_counter()-started
    for name in ("uvMaps", "faceMasks", "faces", "uvs"):
        if not np.array_equal(geometry[name], original[name]):
            raise AssertionError(f"Frozen {name} changed")
    if (arrayHash(images) != inputHashes["images"] or arrayHash(original["uvMaps"]) != inputHashes["uvMaps"]
            or arrayHash(original["faceMasks"]) != inputHashes["faceMasks"]):
        raise AssertionError("Inference mutated inputs")
    np.savez_compressed(output / "geometry.npz", **{key: value for key, value in geometry.items()
                                                    if isinstance(value, np.ndarray)})
    writeJson(output / "viewTransforms.json", geometry["viewTransforms"])
    exportMesh(output, geometry, includeMaterial=False)
    cameraStart = perf_counter()
    cameras = fitViewCameras(geometry, maxErrorPixels=12., seed=seed)
    cameraSeconds = perf_counter()-cameraStart
    writeJson(output / "cameras.json", cameras)
    native = [{**canonicalCamera(extrinsics, transform), "intrinsics": intrinsics, "valid": True}
              for extrinsics, transform, intrinsics in zip(geometry["cameraExtrinsics"], geometry["viewTransforms"],
                                                          geometry["cameraIntrinsics"])]
    writeJson(output / "nativeCameras.json", native)
    report = {"status": "succeeded", "caseId": case["caseId"], "variantId": variant["id"],
        "label": variant["label"], "sourceDirectory": source, "seed": seed,
        "geometrySecondsIncludingDiagnosticWrites": geometrySeconds, "cameraFitSeconds": cameraSeconds,
        "wallSeconds": perf_counter()-started, "peakAllocatedGiB": torch.cuda.max_memory_allocated()/2**30,
        "acceptedCameraCount": sum(bool(camera.get("valid")) for camera in cameras),
        "inputArrayHashes": inputHashes, "topologyAndUVsFrozen": True, "samplingMasksFrozen": True,
        "geometricFusionInputImages": "Prepared float32 512 RGB, exact arrays in case inputs.npz; PNGs are previews",
        "sourceHashes": {name: fileHash(source / name) for name in ("geometry.npz", "inputs.json")},
        "limits": ["Only VGGT RGB context is altered; UV predictor and mask estimation are frozen upstream caches.",
            "Input photos participate in reconstruction, so reprojection/eye scores are proxies, not independent truth.",
            "Canonical mesh units and raw depth scale are uncalibrated.",
            "Own PnP fits use each variant's VGGT intrinsics and can absorb geometry differences.",
            "Repeated baseline estimates observed numerical variability, not statistical significance."]}
    report["outputHashes"] = {name: fileHash(output / name) for name in
        ("geometry.npz", "cameras.json", "nativeCameras.json", "viewTransforms.json", "diagnostics/manifest.json")}
    writeJson(output / "result.json", report)
    return {"status": "succeeded", "geometrySeconds": geometrySeconds, "cameraFitSeconds": cameraSeconds,
            "acceptedCameraCount": report["acceptedCameraCount"], "resultSha256": fileHash(output / "result.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/uffMultiCase20261011/batchManifest.json")
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "backgroundManifest.json"
    source = json.loads(args.source.read_text())
    records = [row for row in source["cases"] if row["status"] == "succeeded"]
    if path.exists():
        if not args.resume:
            raise FileExistsError("Existing experiment; use --resume to validate and skip completed variants")
        manifest = json.loads(path.read_text())
        if manifest["sourceManifestSha256"] != fileHash(args.source):
            raise ValueError("Source manifest changed")
    else:
        manifest = {"schemaVersion": 1, "status": "preparing", "createdAt": datetime.now(timezone.utc).isoformat(),
            "sourceManifest": args.source.resolve(), "sourceManifestSha256": fileHash(args.source),
            "seed": 2026, "design": "Frozen UV/mask; change RGB context only at VGGT input; A,B,C,D,A-repeat per case",
            "cases": [{"caseId": row["caseId"], "label": row["label"], "status": "pending",
                "captureModeHypothesis": row.get("captureModeHypothesis"), "sourceDirectory": row["outputDirectory"],
                "referenceViewIndex": 0, "inputsPath": str(root / "cases" / row["caseId"] / "inputs.npz"),
                "variants": [{"id": name, "inputKey": key, "label": label, "status": "pending",
                    "outputDirectory": str(root / "cases" / row["caseId"] / name)} for name, key, label in VARIANTS]}
                for row in records],
            "limits": ["Six prior successful cases, not a random population sample.",
                       "User labels the old control as camera orbit; other labels are batch-level predominantly head-turning.",
                       "This isolates VGGT sensitivity, not end-to-end UV-predictor sensitivity to matting.",
                       "No shape ground truth; output changes and proxy improvements are reported separately.",
                       "Masks approximate visible foreground/head from frozen semantic parsing, with protected face pixels."]}
        persist(path, manifest)
    selected = [case for case in manifest["cases"] if not args.case or case["caseId"] in args.case]
    for case in selected:
        if not Path(case["inputsPath"]).is_file():
            raise FileNotFoundError(case["inputsPath"])
        expected = case.get("preparedInputsSha256")
        actual = fileHash(Path(case["inputsPath"]))
        if expected and expected != actual:
            raise ValueError("Prepared input arrays changed")
        case["preparedInputsSha256"] = actual
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    manifest["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "depthChunkSize": 1,
        "cudnnDeterministic": torch.backends.cudnn.deterministic,
        "cudnnBenchmark": torch.backends.cudnn.benchmark}
    start = perf_counter()
    adapter = GeometryAdapter(ROOT, "cuda", "staged", depthChunkSize=1).loadDiagnosticModels()
    manifest.setdefault("loadAttempts", []).append({"seconds": perf_counter()-start, "selectedCases": [c["caseId"] for c in selected]})
    manifest["modelHashes"] = {key: fileHash(adapter.assetPaths()[key]) for key in ("vggt", "fusion", "mesh", "uvs")}
    manifest["status"] = "running"
    persist(path, manifest)
    for case in selected:
        case["status"] = "running"
        original = readArrays(Path(case["sourceDirectory"]) / "geometry.npz")
        images = readArrays(Path(case["inputsPath"]))
        protected = original["faceMasks"] > .5
        for variant in case["variants"]:
            output = Path(variant["outputDirectory"])
            if variant["status"] == "succeeded":
                report = json.loads((output / "result.json").read_text())
                for name, expected in report["outputHashes"].items():
                    if fileHash(output / name) != expected:
                        raise ValueError(f"Saved variant changed: {output / name}")
                continue
            variant["status"] = "running"
            persist(path, manifest)
            print(f"START {case['caseId']} {variant['id']}", flush=True)
            try:
                rgb = images[variant["inputKey"]]
                if not np.array_equal(rgb[protected], images["baseline"][protected]):
                    raise AssertionError("Face-supported RGB changed")
                variant.update(runVariant(adapter, case, variant, rgb, original, output, manifest["seed"]))
                print(f"DONE {case['caseId']} {variant['id']} {variant['geometrySeconds']:.2f}s cameras={variant['acceptedCameraCount']}/4", flush=True)
            except Exception as exc:
                variant.update(status="failed", error=repr(exc))
                output.mkdir(parents=True, exist_ok=True)
                (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc(), flush=True)
            persist(path, manifest)
            gc.collect()
            torch.cuda.empty_cache()
        case["status"] = "succeeded" if all(v["status"] == "succeeded" for v in case["variants"]) else "failed"
        persist(path, manifest)
    manifest["status"] = "complete" if all(c["status"] == "succeeded" for c in manifest["cases"]) else "partial"
    manifest["lastRunWallSecondsIncludingLoad"] = perf_counter()-start
    persist(path, manifest)
    implementation = root / "implementation"
    names = ["tools/runUffBackgroundAblation.py", "src/faceAppearance/adapters/geometry.py",
             "src/faceAppearance/vendor/uvFusion/inference.py", "src/faceAppearance/vendor/uvFusion/meshSampling.py",
             "src/faceAppearance/camera.py", "src/faceAppearance/photoCameras.py"]
    for name in names:
        target = implementation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    writeJson(root / "implementationHashes.json", {name: fileHash(ROOT / name) for name in names})
    if any(c["status"] == "failed" for c in selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
