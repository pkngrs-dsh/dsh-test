"""Cache dense STFR eyelids on hash-matched original photos for local experiments.

The saved RetinaFace anchors define the baseline crops. Small anchor translations
only measure crop sensitivity; they never replace or average the baseline points.
All original-image predictions and the exact inputToWorking mapping are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from cacheEyelidLandmarks import makeEyeQaImages


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transformPoints(points, matrix):
    """Preserve all leading dimensions while applying a homogeneous XY map."""
    points = np.asarray(points, np.float64)
    flat = points.reshape(-1, 2)
    transformed = np.c_[flat, np.ones(len(flat))] @ np.asarray(matrix, np.float64).T
    if np.any(np.abs(transformed[:, 2]) < 1e-10):
        raise ValueError("Image transform sends points to infinity")
    return (transformed[:, :2] / transformed[:, 2:]).reshape(points.shape)


def workingOverlay(photo, curves, corners, coarse):
    result = np.asarray(photo).copy()
    for eye in range(2):
        # Sparse face-model eye points provide a visible comparison of semantics.
        sparse = np.rint(coarse[36 + 6 * eye:42 + 6 * eye]).astype(np.int32)
        cv2.polylines(result, [sparse], True, (185, 130, 245), 1, cv2.LINE_AA)
        for curve, color in zip(curves[eye], ((0, 255, 255), (255, 125, 0))):
            cv2.polylines(result, [np.rint(curve).astype(np.int32)], False, color, 1, cv2.LINE_AA)
            for x, y in curve:
                cv2.circle(result, (round(float(x)), round(float(y))), 1, color, -1)
        for x, y in corners[eye]:
            cv2.circle(result, (round(float(x)), round(float(y))), 2, (255, 50, 50), -1)
    return Image.fromarray(result)


def cacheClientDenseEyes(source, photos, output, *, minimumPeakScore=.2,
                         cropOffsetPixels=2., maximumCropVariationWorkingPixels=3., referenceView=0):
    source, photos, output = (Path(path).resolve() for path in (source, photos, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty dense-eye observation directory")
    if (not np.isfinite(cropOffsetPixels) or cropOffsetPixels < 0
            or not np.isfinite(maximumCropVariationWorkingPixels) or maximumCropVariationWorkingPixels <= 0):
        raise ValueError("Crop offset must be nonnegative and working variation threshold positive")
    inputsPath, camerasPath = source / "inputs.json", source / "cameras.json"
    inputs = json.loads(inputsPath.read_text(encoding="utf-8"))
    cameras = json.loads(camerasPath.read_text(encoding="utf-8"))
    if not isinstance(inputs, list) or not inputs or len(cameras) != len(inputs) or not 0 <= referenceView < len(inputs):
        raise ValueError("Inputs, cameras and selected reference view must match")
    validated = []
    for index, item in enumerate(inputs):
        path = photos / f"{index:02d}.image"
        if fileHash(path) != item["sha256"]:
            raise ValueError(f"View {index} original photo hash differs from reconstruction")
        with Image.open(path) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        if list(photo.size) != item["orientedSize"]:
            raise ValueError(f"View {index} oriented image size differs from reconstruction")
        matrix = np.asarray(item["inputToWorking"], np.float64)
        inverse = np.asarray(item["workingToInput"], np.float64)
        anchors = np.asarray(item["faceDetection"]["landmarks"], np.float64)
        if (matrix.shape != (3, 3) or inverse.shape != (3, 3)
                or not np.isfinite(matrix).all() or not np.isfinite(inverse).all()
                or not np.allclose(matrix @ inverse, np.eye(3), atol=1e-9, rtol=0)
                or anchors.shape not in ((4, 2), (5, 2)) or not np.isfinite(anchors).all()):
            raise ValueError(f"View {index} has invalid saved image transforms or anchors")
        if len(item["workingSize"]) != 2 or item["workingSize"][0] != item["workingSize"][1]:
            raise ValueError("This cache requires the recorded square reconstruction crop")
        working, replay = readImage(path, imageSize=item["workingSize"][0],
            cropPolicy=item["requestedCropPolicy"], cropMargin=item["cropMargin"],
            faceDetector=lambda _, item=item: item["faceDetection"])
        for key in ("inputToWorking", "workingToInput", "cropBoxXyxy", "paddingLeftTopRightBottom", "paddingMode"):
            if replay[key] != item[key]:
                raise ValueError(f"View {index} working photo replay differs: {key}")
        validated.append((path, photo, Image.fromarray(np.rint(working * 255).astype(np.uint8)), matrix, inverse, anchors))
    started = perf_counter()
    cv2.setNumThreads(1)
    detector = EyelidLandmarksAdapter(ROOT, minimumPeakScore=minimumPeakScore).loadModels()
    loadSeconds = perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / "manifest.json", {"status": "running", "inputsSha256": fileHash(inputsPath)})
    offsets = np.array([[0., 0.]] if cropOffsetPixels == 0 else
        [[0., 0.], [cropOffsetPixels, 0.], [-cropOffsetPixels, 0.],
         [0., cropOffsetPixels], [0., -cropOffsetPixels]], np.float64)
    records, panels, combined = [], [], []
    for index, (path, photo, working, matrix, inverse, anchors) in enumerate(validated):
        tick = perf_counter()
        rgb = np.asarray(photo)
        inferred = [detector.infer(rgb, faceAnchors=anchors + offset) for offset in offsets]
        baseline = inferred[0]
        curves = transformPoints(baseline["eyelidCurves"], matrix)
        corners = transformPoints(baseline["corners"], matrix)
        coarse = transformPoints(baseline["faceLandmarks"], matrix)
        variants = np.stack([transformPoints(result["eyelidCurves"], matrix) for result in inferred])
        variation = np.linalg.norm(variants - variants[:1], axis=-1).max(0)
        pointMask = (baseline["pointValidMask"] & np.isfinite(curves).all(-1)
            & ((curves >= 0) & (curves < inputs[index]["workingSize"])).all(-1)
            & (variation <= maximumCropVariationWorkingPixels))
        eyeMask = (pointMask.all(axis=(1, 2)) & np.array([q["valid"] for q in baseline["quality"]])
            & bool(cameras[index].get("valid", False)))
        roundTrip = float(np.abs(transformPoints(curves, inverse) - baseline["eyelidCurves"]).max())
        arrays = {name: value for name, value in baseline.items() if isinstance(value, np.ndarray)}
        arrays.update(eyelidCurvesWorking=curves, cornersWorking=corners, faceLandmarksWorking=coarse,
            faceAnchorsOriginal=anchors, inputToWorking=matrix, workingToInput=inverse,
            cropVariantAnchorOffsetsOriginal=offsets, cropVariantEyelidCurvesWorking=variants,
            cropVariantRawPeakScores=np.stack([result["rawPeakScores"] for result in inferred]),
            maximumCropVariationWorkingPixels=variation, candidatePointValidMask=pointMask,
            candidateEyeValidMask=eyeMask)
        arrayPath, metaPath = output / f"view{index:02d}.npz", output / f"view{index:02d}.json"
        np.savez_compressed(arrayPath, **arrays)
        overlay, details = makeEyeQaImages(photo, baseline, index)
        overlay.save(output / f"view{index:02d}OriginalOverlay.png")
        details.save(output / f"view{index:02d}EyeDetail.png")
        working.save(output / f"view{index:02d}WorkingPhoto.png")
        workingOverlay(working, curves, corners, coarse).save(output / f"view{index:02d}WorkingOverlay.png")
        panels.append(details.resize((480, 361), Image.Resampling.LANCZOS))
        metadata = {name: value for name, value in baseline.items() if not isinstance(value, np.ndarray)}
        metadata.update(viewIndex=index, inputSha256=inputs[index]["sha256"], photoPath=path,
            originalSize=list(photo.size), workingSize=list(working.size), inputToWorking=matrix,
            anchorSource="unchangedReconstructionInputRetinaFaceFivePoints",
            selectedPrediction="baselineOriginalAnchors; no averaging or selection by geometry error",
            recommendedForFrontalFitting=index == referenceView,
            candidateEyeValidMask=eyeMask, candidatePointValidMask=pointMask,
            maximumCropVariationWorkingPixelsPerEye=variation.max(axis=(1, 2)),
            meanCropVariationWorkingPixelsPerEye=variation.mean(axis=(1, 2)),
            cropVariantAnchorOffsetsOriginal=offsets,
            cropVariantQuality=[result["quality"] for result in inferred],
            cornersOriginal=baseline["corners"], cornersWorking=corners,
            denseVersusSparseEyeNote="Dense curves and face68 eyelids have different model outputs and do not share point identities.",
            coordinateRoundTripMaximumOriginalPixels=roundTrip, seconds=perf_counter() - tick)
        writeJson(metaPath, metadata)
        record = {"viewIndex": index, "photoSha256": inputs[index]["sha256"],
            "arrays": arrayPath.name, "arraySha256": fileHash(arrayPath),
            "metadata": metaPath.name, "metadataSha256": fileHash(metaPath),
            "quality": baseline["quality"], "candidateEyeValidMask": eyeMask,
            "maximumCropVariationWorkingPixelsPerEye": variation.max(axis=(1, 2)),
            "cornersWorking": corners, "seconds": metadata["seconds"]}
        records.append(record)
        combined.append(arrays)
        print(json.dumps({"viewIndex": index, "candidateEyeValidMask": eyeMask.tolist(),
            "minimumRawPeakPerEye": baseline["rawPeakScores"].min(axis=(1, 2)).tolist(),
            "maximumCropVariationWorkingPixelsPerEye": variation.max(axis=(1, 2)).tolist(),
            "cornersWorking": corners.tolist(), "quality": baseline["quality"]}), flush=True)
    # The combined archive uses one leading view dimension for the fitting experiment.
    np.savez_compressed(output / "denseEyeObservations.npz",
        **{key: np.stack([arrays[key] for arrays in combined]) for key in combined[0]})
    columns = min(2, len(panels))
    overview = Image.new("RGB", (480 * columns, 361 * ((len(panels) + columns - 1) // columns)), (28, 28, 28))
    for index, panel in enumerate(panels):
        overview.paste(panel, ((index % columns) * 480, (index // columns) * 361))
    overview.save(output / "overview.png")
    report = {"schemaVersion": 1, "status": "complete", "referenceView": referenceView,
        "method": "STFR dense eye model on exact original photos with saved RetinaFace anchors",
        "device": "OpenCV DNN CPU, one thread", "model": detector.modelMetadata,
        "inputsSha256": fileHash(inputsPath), "camerasSha256": fileHash(camerasPath),
        "implementationSha256": fileHash(__file__),
        "adapterSha256": fileHash(ROOT / "src/faceAppearance/adapters/eyelidLandmarks.py"),
        "coordinateConvention": "Original EXIF-oriented image pixel XY; Working suffix uses exact saved inputToWorking",
        "curveOrder": ["lower", "upper"], "eyeOrder": ["imageLeft", "imageRight"],
        "pointOrder": "12 points per curve, image-left endpoint to image-right endpoint",
        "confidenceKind": "Clipped raw regressed heatmap peak for weighting, not calibrated probability",
        "candidateGate": {"minimumRawPeak": minimumPeakScore,
            "maximumCropVariationWorkingPixels": maximumCropVariationWorkingPixels,
            "insideOriginalAndWorkingImage": True, "baselineAdapterShapeGate": True, "acceptedCamera": True},
        "cropSensitivity": {"anchorOffsetsOriginalPixels": offsets,
            "meaning": "Translate the complete crop-anchor set only; unchanged original photo and baseline anchor semantics",
            "baselineSelection": "Original unperturbed anchors; sensitivity variants do not alter saved baseline predictions"},
        "loadSeconds": loadSeconds, "seconds": perf_counter() - started, "views": records,
        "combinedArrays": "denseEyeObservations.npz", "combinedArraySha256": fileHash(output / "denseEyeObservations.npz"),
        "limitations": ["Automatic confidence, shape and crop-stability gates do not establish eyelid-margin accuracy or visibility.",
            "Inspect the dense lines to distinguish true eyelid margin from eyelash or makeup contours.",
            "Only the selected frontal view is recommended for this fitting experiment; side views are inspection evidence.",
            "RetinaFace five-point anchors are retained for consistency with the current service; upstream STFR uses FAN68 anchors.",
            "No geometry, camera, texture, photo or production pipeline is modified."]}
    writeJson(output / "manifest.json", report)
    print(json.dumps({"status": "complete", "seconds": report["seconds"], "output": str(output)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-peak-score", type=float, default=.2)
    parser.add_argument("--crop-offset-pixels", type=float, default=2.)
    parser.add_argument("--maximum-crop-variation-working-pixels", type=float, default=3.)
    parser.add_argument("--reference-view", type=int, default=0)
    args = parser.parse_args()
    cacheClientDenseEyes(args.source, args.photos, args.output, minimumPeakScore=args.minimum_peak_score,
        cropOffsetPixels=args.crop_offset_pixels,
        maximumCropVariationWorkingPixels=args.maximum_crop_variation_working_pixels,
        referenceView=args.reference_view)
