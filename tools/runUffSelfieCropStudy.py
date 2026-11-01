"""Factorial crop/context study on additional four-view head-turn candidates.

Freeze original-image detections. Predict UV/masks anew for each crop, then
reuse those correspondences for three VGGT contexts. Preserve original inputs.
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
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.adapters.geometry import GeometryAdapter
from faceAppearance.adapters.faceParsing import FaceParsingAdapter
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson, writeImage, exportMesh, jsonValue
from faceAppearance.camera import fitViewCameras
from faceAppearance.data import readImage
from faceAppearance.faceCrop import workingImageValidMask
from faceAppearance.photoCameras import canonicalCamera
from faceAppearance.faceSemantics import semanticLabelNames
from cacheEyelidLandmarks import cacheEyelidLandmarks, readValidatedViews
from prepareUffBackgroundInputs import transform_labels, HEAD_NAMES, contact_sheet, rgb_image, PALETTE
from renderUffMultiCase import grid

CROPS = (("m135", 1.35, "perView"), ("m110", 1.10, "perView"),
         ("m165", 1.65, "perView"), ("shared135", 1.35, "sharedSequence"))
BACKGROUNDS = (("raw", "原背景"), ("backgroundGray", "环境灰底"), ("headGray", "头部灰底"))


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


def persist(path, value):
    temporary = Path(path).with_suffix(".writing.json")
    writeJson(temporary, value)
    temporary.replace(path)


def saveInputs(path, value):
    expected = json.loads(json.dumps(value, default=jsonValue))
    if path.exists():
        if read(path) != expected:
            raise ValueError(f"Refusing changed crop metadata: {path}")
    else:
        writeJson(path, value)


def seedAll(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def originalPhotos(case):
    photos = []
    for path, expected in zip(case["paths"], case["photoSha256"]):
        if sha(path) != expected:
            raise ValueError(f"Source photo changed: {path}")
        with Image.open(path) as source:
            photos.append(np.array(ImageOps.exif_transpose(source).convert("RGB"), copy=True))
    return photos


def prepareMetadata(case, root, adapter, photos):
    detectionPath = root / "detections.json"
    if detectionPath.exists():
        stored = read(detectionPath)
        if stored["photoSha256"] != case["photoSha256"]:
            raise ValueError("Detection photo identity changed")
        detections = stored["detections"]
    else:
        detections = [adapter.detectInputFace(photo) for photo in photos]
        writeJson(detectionPath, {"photoSha256": case["photoSha256"], "detections": detections,
            "implementation": "GeometryAdapter.detectInputFace; once per original image"})
    sameSize = len({photo.shape for photo in photos}) == 1
    boxes = np.asarray([d["boxXyxy"] for d in detections])
    points = np.concatenate([d["landmarks"] for d in detections])
    union = [min(boxes[:, 0].min(), np.floor(points[:, 0].min())),
             min(boxes[:, 1].min(), np.floor(points[:, 1].min())),
             max(boxes[:, 2].max(), np.floor(points[:, 0].max()) + 1),
             max(boxes[:, 3].max(), np.floor(points[:, 1].max()) + 1)]
    crops = {}
    for prefix, margin, mode in CROPS:
        if mode == "sharedSequence" and not sameSize:
            continue
        images, items = [], []
        for index, path in enumerate(case["paths"]):
            detection = dict(detections[index])
            if mode == "sharedSequence":
                detection["boxXyxy"] = union
            image, item = readImage(Path(path), cropPolicy="faceSquare", cropMargin=margin,
                                   faceDetector=lambda _, d=detection: d)
            item["cropStudyMode"] = mode
            item["originalFaceDetection"] = detections[index]
            images.append(image)
            items.append(item)
        if mode == "sharedSequence" and len({tuple(i["cropBoxXyxy"]) for i in items}) != 1:
            raise AssertionError("Shared sequence crop must have one identical crop box")
        crops[prefix] = (np.stack(images), items)
        for variant in case["variants"]:
            if variant["cropKey"] == prefix:
                directory = Path(variant["outputDirectory"])
                directory.mkdir(parents=True, exist_ok=True)
                saveInputs(directory / "inputs.json", items)
    case["sharedCropEligible"] = sameSize
    return crops, detections


def cacheOriginalObservations(case, caseRoot, photos, detections, parser, eyelid):
    source = Path(case["sourceDirectory"])
    destination = caseRoot / "semantics"
    if (destination / "manifest.json").exists():
        _, report, _ = readValidatedViews(source, destination)
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Partial semantic cache exists; preserve it")
        destination.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        inputs = read(source / "inputs.json")
        views = []
        panels, labels = [], []
        for index, (photo, detection, item) in enumerate(zip(photos, detections, inputs)):
            begin = perf_counter()
            result = parser.infer(photo, faceLandmarks=np.asarray(detection["landmarks"], np.float32))
            out = destination / f"view{index:02d}.npz"
            np.savez_compressed(out, labelMap=result.labelMap, confidence=result.confidence,
                                landmarks=result.landmarks)
            panels.extend([Image.fromarray(photo), Image.fromarray(PALETTE[result.labelMap])])
            labels.extend([f"view {index}: original", f"view {index}: frozen semantics"])
            views.append({"viewIndex": index, "photoSha256": item["sha256"],
                          "imageSizeWh": item["orientedSize"], "seconds": perf_counter() - begin,
                          "metadata": result.metadata, "cacheSha256": sha(out),
                          "pixelsPerLabel": np.bincount(result.labelMap.ravel(), minlength=19)})
        grid(panels, labels, 4, panel_size=240, subtitle=f"{case['caseId']} | original semantics").save(destination / "overview.png")
        report = {"status": "complete", "source": source, "device": "cuda", "parserName": "celebm",
            "inputsSha256": sha(source / "inputs.json"), "labelNames": result.labelNames,
            "views": views, "totalSeconds": perf_counter() - started,
            "modelWeightHashes": {k: sha(v) for k, v in parser.assetPaths().items()},
            "landmarkSource": "same frozen GeometryAdapter original detections as all crop variants"}
        writeJson(destination / "manifest.json", report)
    cache = caseRoot / "eyelids"
    if not (cache / "manifest.json").exists():
        cacheEyelidLandmarks(source, destination, cache, adapter=eyelid)
    else:
        from runEyeRefinementExperiment import loadEyeCache
        loadEyeCache(cache, read(source / "inputs.json"), inputsSha256=sha(source / "inputs.json"))
    return report


def cropArrays(caseRoot, prefix, images, items, detections, adapter, seed):
    directory = caseRoot / "cropInputs" / prefix
    path, infoPath = directory / "arrays.npz", directory / "metadata.json"
    sourceHashes = {"rgb": ahash(images), "inputs": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
                    "semanticManifest": sha(caseRoot / "semantics/manifest.json")}
    if path.exists() and infoPath.exists():
        info = read(infoPath)
        if info["sourceHashes"] != sourceHashes or sha(path) != info["arraysSha256"]:
            raise ValueError("Crop source/cache changed")
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}, info
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Partial crop arrays preserved")
    seedAll(seed)
    started = perf_counter()
    uv = adapter.predictUvs(images)
    masks = adapter.predictFaceMasks(images)
    valid = np.stack([workingImageValidMask(i) for i in items])
    masks *= valid
    predictionSeconds = perf_counter() - started
    headIds = [list(semanticLabelNames).index(name) for name in HEAD_NAMES]
    foregroundAlpha, headAlpha, labelMaps, checks = [], [], [], []
    for index, (item, detection) in enumerate(zip(items, detections)):
        with np.load(caseRoot / "semantics" / f"view{index:02d}.npz", allow_pickle=False) as source:
            labels, _ = transform_labels(source["labelMap"], item)
        protected = masks[index] > .5
        foreground = ((labels != 0) | protected).astype(np.float32)
        headCore = np.isin(labels, headIds) | protected
        originalBox = np.asarray(detection["boxXyxy"])
        faceScale = max(originalBox[2:] - originalBox[:2])
        workingScale = float(item["inputToWorking"][0][0])
        dilation, feather = .016 * faceScale * workingScale, .011 * faceScale * workingScale
        distance = distance_transform_edt(~headCore)
        alpha = np.clip((dilation + feather - distance) / feather, 0, 1).astype(np.float32)
        alpha[protected] = 1
        foregroundAlpha.append(foreground)
        headAlpha.append(alpha)
        labelMaps.append(labels)
        checks.append({"viewIndex": index, "paddingFraction": float((~valid[index]).mean()),
            "faceWidthWorkingPixels": float((originalBox[2] - originalBox[0]) * workingScale),
            "dilationWorkingPixels": dilation, "featherWorkingPixels": feather,
            "headReplacementFraction": float((alpha < 1).mean()),
            "backgroundReplacementFraction": float((foreground < 1).mean())})
    foregroundAlpha, headAlpha = np.stack(foregroundAlpha), np.stack(headAlpha)
    arrays = {"raw": images, "uvMaps": uv, "faceMasks": masks, "validImageMasks": valid,
              "labelMaps": np.stack(labelMaps), "alphaBackground": foregroundAlpha, "alphaHead": headAlpha}
    for name, alpha in (("backgroundGray", foregroundAlpha), ("headGray", headAlpha)):
        rgb = images * alpha[..., None] + np.float32(.5) * (1 - alpha[..., None])
        rgb[masks > .5] = images[masks > .5]
        assert np.array_equal(rgb[masks > .5], images[masks > .5])
        arrays[name] = rgb.astype(np.float32)
    np.savez_compressed(path, **arrays)
    info = {"sourceHashes": sourceHashes, "arraysSha256": sha(path), "predictionSeconds": predictionSeconds,
            "views": checks, "facePixelsUnchangedWithinCrop": True,
            "boundaryRule": "head dilation=.016, feather=.011 times original detected max face extent, mapped into each crop",
            "uvRule": "new raw-crop UV and parser prediction per crop; frozen across background variants"}
    writeJson(infoPath, info)
    contact_sheet([[rgb_image(arrays[name][i]) for name in ("raw", "backgroundGray", "headGray")] for i in range(len(images))],
        ["Original background", "Environment gray", "Head gray"], f"{caseRoot.name} / {prefix}",
        ["Only VGGT context is changed; UV/masks come from this raw crop.", "Face-supported RGB unchanged; fixed source-relative head feather."], directory / "inputContactSheet.png")
    return arrays, info


def reconstruct(case, variant, arrays, adapter, seed):
    import torch
    directory = Path(variant["outputDirectory"])
    if (directory / "geometry.npz").exists() or (directory / "diagnostics").exists():
        raise FileExistsError(f"Partial or existing reconstruction preserved: {directory}")
    images = arrays[variant["background"]]
    for index, image in enumerate(images):
        writeImage(directory / f"input{index:02d}.png", image)
    seedAll(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = perf_counter()
    geometry = adapter.reconstructFromCachedUvs(images, arrays["uvMaps"], arrays["faceMasks"],
                                                 diagnosticsDirectory=directory / "diagnostics")
    torch.cuda.synchronize()
    seconds = perf_counter() - started
    for name in ("uvMaps", "faceMasks"):
        if not np.array_equal(geometry[name], arrays[name]):
            raise AssertionError(f"Frozen {name} changed")
    np.savez_compressed(directory / "geometry.npz", **{k: v for k, v in geometry.items() if isinstance(v, np.ndarray)})
    writeJson(directory / "viewTransforms.json", geometry["viewTransforms"])
    exportMesh(directory, geometry, includeMaterial=False)
    fitStart = perf_counter()
    cameras = fitViewCameras(geometry, maxErrorPixels=12., seed=seed)
    fitSeconds = perf_counter() - fitStart
    writeJson(directory / "cameras.json", cameras)
    native = [{**canonicalCamera(e, t), "intrinsics": k, "valid": True} for e, t, k in
              zip(geometry["cameraExtrinsics"], geometry["viewTransforms"], geometry["cameraIntrinsics"])]
    writeJson(directory / "nativeCameras.json", native)
    result = {"status": "succeeded", "caseId": case["caseId"], "variantId": variant["id"],
        "geometrySecondsIncludingDiagnostics": seconds, "cameraFitSeconds": fitSeconds,
        "acceptedCameraCount": sum(bool(c.get("valid")) for c in cameras),
        "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
        "inputArraysSha256": {"images": ahash(images), "uvMaps": ahash(arrays["uvMaps"]), "faceMasks": ahash(arrays["faceMasks"])},
        "limits": ["Only VGGT sees modified RGB; UV/parser predictions are from each raw crop.",
                   "No calibrated 3D truth; own PnP is a diagnostic, not ranking evidence."]}
    result["outputHashes"] = {name: sha(directory / name) for name in ("geometry.npz", "cameras.json", "nativeCameras.json", "viewTransforms.json", "inputs.json")}
    writeJson(directory / "result.json", result)
    return {"status": "succeeded", "geometrySeconds": seconds, "acceptedCameraCount": result["acceptedCameraCount"], "resultSha256": sha(directory / "result.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "runs/uffSelfieCropStudy20261012")
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--case", action="append")
    args = ap.parse_args()
    root = args.root.resolve()
    selectionPath = args.selection or root / "selection/selection.json"
    selection = read(selectionPath)
    root.mkdir(parents=True, exist_ok=True)
    manifestPath = root / "cropManifest.json"
    if manifestPath.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new output root")
        manifest = read(manifestPath)
        if manifest["selectionSha256"] != sha(selectionPath):
            raise ValueError("Selection changed")
    else:
        cases = []
        for record in selection["cases"]:
            caseId = record["caseId"]
            caseRoot = root / "cases" / caseId
            case = {**record, "status": "pending", "referenceViewIndex": 0,
                    "sourceDirectory": str(caseRoot / "m135_raw"),
                    "photoSha256": [sha(p) for p in record["paths"]], "variants": []}
            for prefix, margin, mode in CROPS:
                for bg, bgLabel in BACKGROUNDS:
                    if prefix == "shared135" and bg == "backgroundGray":
                        continue
                    case["variants"].append({"id": f"{prefix}_{bg}", "cropKey": prefix,
                        "label": f"{'共享框' if mode == 'sharedSequence' else '逐帧框'} {margin:.2f} · {bgLabel}",
                        "margin": margin, "cropMode": mode, "background": bg, "status": "pending",
                        "outputDirectory": str(caseRoot / f"{prefix}_{bg}")})
            case["variants"].append({"id": "m135_repeat", "cropKey": "m135", "label": "1.35 原背景重复",
                "margin": 1.35, "cropMode": "perView", "background": "raw", "status": "pending",
                "outputDirectory": str(caseRoot / "m135_repeat")})
            cases.append(case)
        manifest = {"schemaVersion": 1, "status": "preparing", "createdAt": datetime.now(timezone.utc).isoformat(),
            "selectionPath": str(selectionPath.resolve()), "selectionSha256": sha(selectionPath), "seed": 2026,
            "design": "three per-view margins x three VGGT backgrounds, two shared-sequence crops, one cached-UV baseline repeat",
            "referenceVariant": "m135_raw", "cases": cases,
            "limits": ["Batch-reported mostly fixed-camera head turns, not independently motion-calibrated.",
                "Directory groups are not a verified count of independent people.",
                "Targets are automatic in-sample proxies; this is not a ground-truth 3D benchmark.",
                "Head matting uses imperfect automatic semantics; shared crop is retrospective sequence union.",
                "Background changes VGGT input only; each crop gets new UV and sampling masks from its raw RGB."]}
        persist(manifestPath, manifest)
    import torch
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    start = perf_counter()
    adapter = GeometryAdapter(ROOT, "cuda", "staged", depthChunkSize=1).loadModels()
    parser = FaceParsingAdapter(ROOT, device="cuda", parserName="celebm").loadModels()
    eyelid = EyelidLandmarksAdapter(ROOT).loadModels()
    manifest["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "modelLoadSeconds": perf_counter() - start, "cudnnDeterministic": torch.backends.cudnn.deterministic}
    manifest["modelHashes"] = {key: sha(value) for key, value in adapter.assetPaths().items()}
    manifest["status"] = "running"
    persist(manifestPath, manifest)
    for case in manifest["cases"]:
        if args.case and case["caseId"] not in args.case:
            continue
        caseRoot = root / "cases" / case["caseId"]
        caseRoot.mkdir(parents=True, exist_ok=True)
        case["status"] = "running"
        persist(manifestPath, manifest)
        print(f"CASE {case['caseId']} PREPARE", flush=True)
        try:
            photos = originalPhotos(case)
            crops, detections = prepareMetadata(case, caseRoot, adapter, photos)
            cacheOriginalObservations(case, caseRoot, photos, detections, parser, eyelid)
            del photos
            for prefix, _, _ in CROPS:
                if prefix not in crops:
                    for variant in case["variants"]:
                        if variant["cropKey"] == prefix:
                            variant.update(status="skipped", reason="Shared crop requires identical source dimensions")
                    continue
                images, inputs = crops[prefix]
                print(f"CROP {case['caseId']} {prefix} UV + masks", flush=True)
                arrays, info = cropArrays(caseRoot, prefix, images, inputs, detections, adapter, manifest["seed"])
                for variant in case["variants"]:
                    if variant["cropKey"] != prefix or variant["id"] == "m135_repeat":
                        continue
                    if variant["status"] == "succeeded":
                        result = read(Path(variant["outputDirectory"]) / "result.json")
                        for name, expected in result["outputHashes"].items():
                            if sha(Path(variant["outputDirectory"]) / name) != expected:
                                raise ValueError("Completed result changed")
                        continue
                    print(f"START {case['caseId']} {variant['id']}", flush=True)
                    variant["status"] = "running"
                    persist(manifestPath, manifest)
                    variant.update(reconstruct(case, variant, arrays, adapter, manifest["seed"]))
                    persist(manifestPath, manifest)
                    print(f"DONE {case['caseId']} {variant['id']} {variant['geometrySeconds']:.2f}s", flush=True)
                del arrays
            repeat = next(v for v in case["variants"] if v["id"] == "m135_repeat")
            if repeat["status"] != "succeeded":
                with np.load(caseRoot / "cropInputs/m135/arrays.npz", allow_pickle=False) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                repeat.update(reconstruct(case, repeat, arrays, adapter, manifest["seed"]))
                del arrays
            panels, titles = [], []
            for prefix, _, _ in CROPS:
                if prefix not in crops:
                    continue
                for bg, _ in BACKGROUNDS:
                    output = caseRoot / f"{prefix}_{bg}" / "input00.png"
                    if output.exists():
                        panels.append(Image.open(output).convert("RGB")); titles.append(f"{prefix} / {bg}")
                    else:
                        panels.append(Image.new("RGB", (512, 512), (20, 25, 29))); titles.append("not part of matrix")
            grid(panels, titles, 3, panel_size=300, subtitle=f"{case['caseId']} | front model inputs").save(caseRoot / "inputFronts.png")
            case["status"] = "succeeded" if all(v["status"] in ("succeeded", "skipped") for v in case["variants"]) else "failed"
            originalPhotos(case)
        except Exception as error:
            case.update(status="failed", error=str(error))
            (caseRoot / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
        persist(manifestPath, manifest)
        gc.collect()
        torch.cuda.empty_cache()
    manifest["status"] = "complete" if all(c["status"] == "succeeded" for c in manifest["cases"]) else "partial"
    manifest["lastRunSecondsIncludingLoad"] = perf_counter() - start
    persist(manifestPath, manifest)
    sources = [Path(__file__), ROOT / "src/faceAppearance/adapters/geometry.py", ROOT / "src/faceAppearance/data.py", ROOT / "src/faceAppearance/faceCrop.py"]
    for source in sources:
        target = root / "implementation" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    writeJson(root / "implementationHashes.json", {str(p.relative_to(ROOT)): sha(p) for p in sources})
    if any(c["status"] == "failed" for c in manifest["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
