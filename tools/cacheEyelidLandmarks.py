"""Cache offline eyelid landmarks from original photos and validated semantic anchors.

The cache keeps raw regressed heatmap peaks, point order, crop provenance and
per-eye inspection images. Automatic quality gates do not establish visibility.
No models are downloaded, and a nonempty output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]

from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson
from renderCropComparison import fileHash, validatePhotoPair


def readValidatedViews(source: Path, semanticCache: Path) -> tuple[Path, dict, list]:
    """Verify photo bytes, view order, semantic cache hashes and anchor geometry."""
    source, semanticCache = Path(source).resolve(), Path(semanticCache).resolve()
    inputsPath = source / "inputs.json" if source.is_dir() else source
    inputs = json.loads(inputsPath.read_text(encoding="utf-8"))
    manifest = json.loads((semanticCache / "manifest.json").read_text(encoding="utf-8"))
    views = manifest.get("views")
    if (not isinstance(inputs, list) or not inputs or manifest.get("status") != "complete"
            or manifest.get("inputsSha256") != fileHash(inputsPath)
            or not isinstance(views, list) or len(views) != len(inputs)):
        raise ValueError("Semantic cache must be complete and match the exact nonempty input manifest")
    validated = []
    for index, (item, view) in enumerate(zip(inputs, views)):
        if (view.get("viewIndex") != index or view.get("photoSha256") != item.get("sha256")
                or view.get("imageSizeWh") != item.get("orientedSize")):
            raise ValueError(f"Semantic view {index} has different photo identity, order or dimensions")
        photo = validatePhotoPair(item, item)
        path = semanticCache / f"view{index:02d}.npz"
        if view.get("cacheSha256") != fileHash(path):
            raise ValueError(f"Semantic view {index} cache content changed")
        with np.load(path, allow_pickle=False) as archive:
            if not {"labelMap", "confidence", "landmarks"}.issubset(archive.files):
                raise ValueError(f"Semantic view {index} is missing labels, confidence or original-image anchors")
            labels, confidence, anchors = (archive[key].copy() for key in ("labelMap", "confidence", "landmarks"))
        shape = (photo.height, photo.width)
        if (labels.shape != shape or labels.dtype.kind not in "iu" or np.any(labels > 18) or np.any(labels < 0)
                or confidence.shape != shape or not np.isfinite(confidence).all()
                or np.any(confidence < 0) or np.any(confidence > 1)
                or anchors.shape != (5, 2) or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors >= [photo.width, photo.height])
                or anchors[0, 0] >= anchors[1, 0] or np.linalg.norm(anchors[1]-anchors[0]) < 6):
            raise ValueError(f"Semantic view {index} has invalid image-space labels, confidence or RetinaFace5 anchors")
        validated.append({"photo": photo, "input": item, "semanticPath": path,
                          "labels": labels, "confidence": confidence, "anchors": anchors})
    return inputsPath, manifest, validated


def matchSemanticEyeLabels(result: dict, labels: np.ndarray, confidence: np.ndarray) -> dict:
    """Assign image-side detections to parser IDs by measured region centers.

    A missing or ambiguous pair stays unassigned. IDs 4/5 retain the parser's
    dataset convention and are never silently inferred from anatomical names.
    """
    centers = {}
    for labelId in (4, 5):
        yy, xx = np.nonzero((labels == labelId) & (confidence >= .65))
        if len(xx) >= 4:
            centers[labelId] = np.array([xx.mean(), yy.mean()])
    report = {"imageSideToSemanticLabel": {}, "semanticLabelAssignment": {"method": "minimumTwoEyeCenterDistance",
              "semanticConfidenceThreshold": .65, "semanticEyeCenters": centers}}
    if len(centers) != 2:
        report["semanticLabelAssignment"]["status"] = "missingSemanticEyeRegion"
        return report
    eyeCenters = result["corners"].mean(1)
    distances = np.linalg.norm(eyeCenters[:, None]-np.stack([centers[4], centers[5]])[None], axis=-1)
    identityCost, swappedCost = float(distances[0, 0]+distances[1, 1]), float(distances[0, 1]+distances[1, 0])
    report["semanticLabelAssignment"].update(identityCostPixels=identityCost, swappedCostPixels=swappedCost)
    if abs(identityCost-swappedCost) < 2:
        report["semanticLabelAssignment"]["status"] = "ambiguousSemanticEyeAssignment"
        return report
    order = [4, 5] if identityCost < swappedCost else [5, 4]
    report["imageSideToSemanticLabel"] = dict(zip(("imageLeft", "imageRight"), order))
    report["semanticLabelAssignment"].update(status="assigned", distancesPixels=[
        float(np.linalg.norm(eyeCenters[index]-centers[labelId])) for index, labelId in enumerate(order)])
    return report


def makeEyeQaImages(photo: Image.Image, result: dict, viewIndex: int) -> tuple[Image.Image, Image.Image]:
    """Original/curve/raw-index panels preserve visible evidence for every eye."""
    overlay = np.asarray(photo).copy()
    colors = ((0, 255, 255), (255, 125, 0))
    for eye in range(2):
        for curve, color in zip(result["eyelidCurves"][eye], colors):
            cv2.polylines(overlay, [np.rint(curve).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        for x, y in result["corners"][eye]:
            cv2.circle(overlay, (round(float(x)), round(float(y))), 2, (255, 0, 0), -1)
    size, rowHeight, header = 320, 346, 30
    panel = Image.new("RGB", (size*3, rowHeight*2+header), (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), f"View {viewIndex}: original | lower cyan / upper orange | raw eyelid indices 0-23", fill="white")
    for eye, box in enumerate(result["cropBoxes"]):
        x0, y0, x1, y1 = map(int, box)
        width, height = x1-x0, y1-y0
        crop = photo.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BILINEAR)
        offsetY = header + eye*rowHeight
        panel.paste(crop, (0, offsetY))
        quality = result["quality"][eye]
        draw.text((5, offsetY+size+4),
                  f"image {'left' if eye == 0 else 'right'} / minimum raw peak {quality['minimumRawPeakScore']:.3f} / shape gate {quality['valid']}",
                  fill="white")
        for column in (1, 2):
            marked = crop.copy()
            marks = ImageDraw.Draw(marked)
            if column == 1:
                for curve, color in zip(result["eyelidCurves"][eye], colors):
                    points = [((float(x)-x0)/width*size, (float(y)-y0)/height*size) for x, y in curve]
                    marks.line(points, fill=color, width=2)
                for x, y in result["corners"][eye]:
                    px, py = (float(x)-x0)/width*size, (float(y)-y0)/height*size
                    marks.ellipse((px-3, py-3, px+3, py+3), fill="red")
            else:
                for index, (x, y) in enumerate(result["rawEyePoints"][eye, 19:]):
                    px, py = (float(x)-x0)/width*size, (float(y)-y0)/height*size
                    color = colors[index >= 12]
                    marks.ellipse((px-1, py-1, px+1, py+1), fill=color)
                    marks.text((px+1, py+(2 if index < 12 else -13)), str(index), fill=color,
                               stroke_width=1, stroke_fill="black")
            panel.paste(marked, (column*size, offsetY))
    return Image.fromarray(overlay), panel


def cacheEyelidLandmarks(source: Path, semanticCache: Path, output: Path, *, minimumPeakScore: float = .2,
                        adapter=None) -> dict:
    """Run CPU inference only after all existing photo/semantic inputs validate."""
    output, semanticCache = Path(output).resolve(), Path(semanticCache).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose an empty eyelid cache directory; existing outputs are never overwritten")
    inputsPath, semanticManifest, inputs = readValidatedViews(source, semanticCache)
    start = perf_counter()
    if adapter is None:
        adapter = EyelidLandmarksAdapter(projectRoot, minimumPeakScore=minimumPeakScore).loadModels()
    elif not adapter.isLoaded or adapter.minimumPeakScore != minimumPeakScore:
        raise ValueError("Preloaded eyelid adapter must match the requested peak threshold")
    loadSeconds = perf_counter()-start
    implementationPaths = ("src/faceAppearance/adapters/eyelidLandmarks.py", "tools/cacheEyelidLandmarks.py",
                           "tools/renderCropComparison.py", "src/faceAppearance/artifacts.py")
    implementationHashes = {name: fileHash(projectRoot / name) for name in implementationPaths}
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / "manifest.json", {"status": "running", "inputsPath": inputsPath,
                                         "semanticCache": semanticCache, "device": "cpu"})
    views, panels = [], []
    for index, view in enumerate(inputs):
        begin = perf_counter()
        result = adapter.infer(np.asarray(view["photo"]), faceAnchors=view["anchors"])
        inferenceSeconds = perf_counter()-begin
        assignment = matchSemanticEyeLabels(result, view["labels"], view["confidence"])
        arraysPath = output / f"view{index:02d}.npz"
        metadataPath = output / f"view{index:02d}.json"
        np.savez_compressed(arraysPath, semanticFaceAnchors=view["anchors"],
                            **{name: value for name, value in result.items() if isinstance(value, np.ndarray)})
        metadata = {name: value for name, value in result.items() if not isinstance(value, np.ndarray)}
        metadata.update(assignment)
        metadata.update(inputPath=view["input"]["path"], inputSha256=view["input"]["sha256"],
                        seconds=inferenceSeconds, anchorSource="validatedSemanticCacheOriginalImageRetinaFaceFivePoints",
                        semanticCachePath=view["semanticPath"], semanticCacheSha256=fileHash(view["semanticPath"]))
        writeJson(metadataPath, metadata)
        overlay, detail = makeEyeQaImages(view["photo"], result, index)
        overlay.save(output / f"view{index:02d}Overlay.png")
        detail.save(output / f"view{index:02d}EyeDetail.png")
        panels.append(detail.resize((480, 361), Image.Resampling.LANCZOS))
        views.append({"viewIndex": index, "inputPath": view["input"]["path"],
                      "inputSha256": view["input"]["sha256"], "imageSizeWh": list(view["photo"].size),
                      "arrays": arraysPath.name, "metadata": metadataPath.name,
                      "cacheSha256": fileHash(arraysPath), "metadataSha256": fileHash(metadataPath),
                      "semanticCacheSha256": fileHash(view["semanticPath"]), "seconds": inferenceSeconds,
                      "quality": result["quality"], **assignment})
        print(f"Cached eyelid landmarks for view {index+1}/{len(inputs)}", flush=True)
    overview = Image.new("RGB", (480*min(2, len(panels)), 361*((len(panels)+1)//2)), (28, 28, 28))
    for index, panel in enumerate(panels):
        overview.paste(panel, ((index % 2)*480, (index//2)*361))
    overview.save(output / "overview.png")
    for name in implementationPaths:
        destination = output / "implementation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(projectRoot / name, destination)
    report = {"schemaVersion": 1, "status": "complete", "inputsPath": inputsPath,
              "inputsSha256": fileHash(inputsPath), "semanticCache": semanticCache,
              "semanticManifestSha256": fileHash(semanticCache / "manifest.json"),
              "semanticParserName": semanticManifest.get("parserName"), "device": "cpu",
              "model": adapter.modelMetadata, "modelWeightHashes": {str(adapter.assetPath()): fileHash(adapter.assetPath())},
              "implementationHashes": implementationHashes, "adapterSha256": implementationHashes[implementationPaths[0]],
              "anchorSource": "validatedSemanticCacheOriginalImageRetinaFaceFivePoints",
              "minimumPeakScore": minimumPeakScore, "loadSeconds": loadSeconds,
              "totalSeconds": perf_counter()-start, "views": views,
              "limits": ["Raw heatmap peaks and clipped confidence weights are not calibrated probabilities.",
                         "Automatic shape/confidence gates do not prove visibility; occluded eyes still require rejection.",
                         "Curves need visual inspection to distinguish the eyelid margin, eyeliner and eyelashes.",
                         "Crop anchors come from the validated face-semantic cache, not upstream independent FAN68 landmarks.",
                         "Semantic label assignment is a region-center match, not anatomical left/right ground truth.",
                         "No geometry, texture, camera, input photograph or pre-existing cache is modified."]}
    writeJson(output / "manifest.json", report)
    print(json.dumps({"status": "complete", "views": len(views), "seconds": report["totalSeconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment",
                        help="Experiment directory containing inputs.json, or the input manifest itself")
    parser.add_argument("--semantic-cache", type=Path, default=projectRoot / "runs/faceSemanticCache")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--minimum-peak-score", type=float, default=.2)
    arguments = parser.parse_args()
    cacheEyelidLandmarks(arguments.input, arguments.semantic_cache, arguments.output,
                         minimumPeakScore=arguments.minimum_peak_score)
