"""Fit separately identified eye corners and ordered eyelids with frozen cameras."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.eyeContours import (bindTemplateEyeContours, buildEyeContourConstraints,
                                        convertCachedEyeLandmarks, eyeContourMetrics)
from faceAppearance.eyeRefinement import refineEyeGeometry
from faceAppearance.semanticProjection import visibleVertices
from renderCropComparison import photoCamera, validatePhotoPair
from runReferenceExperiment import readArrays, fileHash


def loadEyeCache(cache, inputs, *, minimumRawPeak=.6, sampleCount=24, inputsSha256=None):
    """Validate photos/frame and keep target provenance alongside observations."""
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or len(manifest["views"]) != len(inputs)
            or (inputsSha256 is not None and manifest["inputsSha256"] != inputsSha256)):
        raise ValueError("A complete eye cache matching the input manifest is required")
    observations, hashes = [], {"manifest.json": fileHash(cache / "manifest.json")}
    for index, (record, item) in enumerate(zip(manifest["views"], inputs)):
        if record["viewIndex"] != index or record["inputSha256"] != item["sha256"]:
            raise ValueError("Eye cache uses different ordered photos")
        validatePhotoPair(item, item)
        arrays = readArrays(cache / record["arrays"])
        metadata = json.loads((cache / record["metadata"]).read_text(encoding="utf-8"))
        if metadata["inputSha256"] != item["sha256"]:
            raise ValueError("Cached eye observation metadata differs from input")
        for name in (record["arrays"], record["metadata"]):
            hashes[name] = fileHash(cache / name)
        if (hashes[record["arrays"]] != record["cacheSha256"]
                or hashes[record["metadata"]] != record["metadataSha256"]):
            raise ValueError("Cached eye arrays or metadata changed after inference")
        if (metadata.get("imageSideToSemanticLabel") == {}
                and metadata.get("semanticLabelAssignment", {}).get("status") in
                ("missingSemanticEyeRegion", "ambiguousSemanticEyeAssignment")):
            # A deliberately unassigned semantic pair is a rejected view, not
            # permission to infer image-side identity. Keep the non-identity
            # portion of convertCachedEyeLandmarks' contract strict as well.
            curves = np.asarray(arrays["eyelidCurves"], float)
            corners = np.asarray(arrays["corners"], float)
            confidence = np.asarray(arrays["curveConfidence"], float)
            peaks = np.asarray(arrays["rawPeakScores"], float)
            validPoints = np.asarray(arrays.get("pointValidMask", np.ones(confidence.shape, bool)))
            contract = metadata.get("metadata", {})
            if (curves.ndim != 4 or curves.shape[:2] != (2, 2) or curves.shape[-1] != 2
                    or curves.shape[2] < 3 or corners.shape != (2, 2, 2)
                    or confidence.shape != curves.shape[:-1] or peaks.shape != confidence.shape
                    or validPoints.shape != confidence.shape or validPoints.dtype != np.bool_
                    or not all(np.isfinite(value).all() for value in (curves, corners, confidence, peaks))
                    or np.any((confidence < 0) | (confidence > 1))
                    or not np.isfinite(minimumRawPeak) or minimumRawPeak < 0
                    or contract.get("eyeOrder") != ["imageLeft", "imageRight"]
                    or contract.get("curveOrder") != ["lower", "upper"]
                    or contract.get("cornerOrder") != ["imageLeftEndpoint", "imageRightEndpoint"]
                    or len(metadata.get("quality", [])) != 2
                    or len(contract.get("originalSize", [])) != 2):
                raise ValueError("Unsupported or invalid cached eye-landmark contract")
            # IDs name unavailable canonical observations only; neither ID is
            # associated with a detected curve or image side in this branch.
            observation = {
                "schemaVersion": 1,
                "eyes": {labelId: {"labelId": labelId, "valid": False,
                                    "reason": "missingSemanticIdentity",
                                    "source": "STFRMergeFanEye/OpenCvCpu"}
                         for labelId in (4, 5)},
                "imageShape": list(reversed(contract["originalSize"])),
                "coordinateFrame": "originalExifOrientedPixels", "source": "STFRMergeFanEye/OpenCvCpu",
                "diagnostics": {"minimumRawPeakGate": minimumRawPeak, "modelVisibilityNotAssumed": True,
                                "semanticLabelAssignment": metadata["semanticLabelAssignment"],
                                "detectorQuality": metadata["quality"], "imageSideIdentityInferred": False}}
        else:
            observation = convertCachedEyeLandmarks(arrays, metadata, sampleCount=sampleCount, minimumRawPeak=minimumRawPeak)
        if tuple(observation["imageShape"]) != tuple(reversed(item["orientedSize"])):
            raise ValueError("Cached eyes use a different image frame")
        observations.append(observation)
    return observations, manifest, hashes


def projectBinding(vertices, binding, camera):
    points = (vertices[np.asarray(binding["vertexIndices"])] * np.asarray(binding["barycentric"])[..., None]).sum(1)
    return projectPoints(points, camera["intrinsics"], camera["rotation"], camera["translation"])[0]


def eyeMetrics(vertices, binding, cameras, observations):
    """All fixed endpoints/curves; report unoccluded fit scores separately."""
    results = []
    for viewIndex, (camera, photo) in enumerate(zip(cameras, observations)):
        for eyeName, eye in binding["eyes"].items():
            target = photo["eyes"][eye["labelId"]]
            if not target["valid"]:
                continue
            projected = {part: projectBinding(vertices, eye[part], camera)
                         for part in ("innerCorner", "outerCorner", "upper", "lower")}
            metric = eyeContourMetrics(projected["innerCorner"][0], projected["outerCorner"][0],
                                       projected["upper"], projected["lower"])
            targetMetric = eyeContourMetrics(target["innerCorner"], target["outerCorner"],
                                             target["upperCurve"], target["lowerCurve"])
            angularKeys = [key for key in metric if key.endswith("Degrees")]
            angularError = {key: float(abs((metric[key]-targetMetric[key]+180) % 360-180)) for key in angularKeys}
            results.append({"viewIndex": viewIndex, "eye": eyeName, "labelId": eye["labelId"],
                "cornerErrorPixels": [float(np.linalg.norm(projected[part][0]-target[part]))
                                      for part in ("innerCorner", "outerCorner")],
                "upperMeanErrorPixels": float(np.linalg.norm(projected["upper"]-target["upperCurve"], axis=1).mean()),
                "lowerMeanErrorPixels": float(np.linalg.norm(projected["lower"]-target["lowerCurve"], axis=1).mean()),
                "angularErrorDegrees": angularError, "projectedMetrics": metric, "targetMetrics": targetMetric,
                "scope": "All fixed sample correspondences, including points omitted by visibility from the fit; detector targets are not ground truth"})
    return results


def drawComparison(output, inputs, cameras, geometry, fitted, binding, observations):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    rows = []
    for viewIndex, (item, camera, photo) in enumerate(zip(inputs, cameras, observations)):
        original = validatePhotoPair(item, item)
        allPoints = np.concatenate([projectBinding(geometry["vertices"], eye[part], camera)
                    for eye in binding["eyes"].values() for part in ("upper", "lower")])
        minimum, maximum = np.floor(allPoints.min(0)-18).astype(int), np.ceil(allPoints.max(0)+18).astype(int)
        crop = tuple(np.r_[np.maximum(minimum, 0), np.minimum(maximum, original.size)])
        scale = min(2.5, 590/(crop[2]-crop[0]))
        panels = []
        for vertices, title in ((None, "Original photo"), (geometry["vertices"], "Before: mesh + target"),
                                (fitted, "After: mesh + target")):
            tile = original.crop(crop).resize((round((crop[2]-crop[0])*scale), round((crop[3]-crop[1])*scale)), Image.Resampling.BICUBIC)
            draw = ImageDraw.Draw(tile)
            def coordinates(points):
                return [tuple(value) for value in (np.asarray(points)-np.array(crop[:2])+.5)*scale-.5]
            if vertices is not None:
                for eye in binding["eyes"].values():
                    target = photo["eyes"][eye["labelId"]]
                    if target["valid"]:
                        for part in ("upperCurve", "lowerCurve"):
                            draw.line(coordinates(target[part]), fill="#ffce4d", width=2)
                        for part in ("innerCorner", "outerCorner"):
                            x, y = coordinates([target[part]])[0]
                            draw.ellipse((x-4,y-4,x+4,y+4), outline="#ffce4d", width=2)
                    for part in ("upper", "lower"):
                        draw.line(coordinates(projectBinding(vertices, eye[part], camera)), fill="#40f2ed", width=2)
            panel = Image.new("RGB", (tile.width, tile.height+38), "#202027")
            panel.paste(tile, (0, 38))
            ImageDraw.Draw(panel).text((8, 8), f"View {viewIndex+1}: {title}", font=font, fill="white")
            panels.append(panel)
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height+30), "#202027")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        ImageDraw.Draw(row).text((8, row.height-25), "Cyan: fixed mesh paths | Yellow: detector targets | Rejected occluded eyes have no yellow curve", font=font, fill="white")
        row.save(output / f"view{viewIndex:02d}Contours.png")
        rows.append(row)
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "#202027")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output / "contoursOverview.png")


def run(source, cache, output, *, iterations=100, referenceWeight=12., supportRadiusFraction=.16,
        controlCount=24, cornerWeight=8., minimumRawPeak=.6):
    start = perf_counter()
    source, cache, output = source.resolve(), cache.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty eye refinement directory")
    geometry = readArrays(source / "geometry.npz")
    vertices, faces = geometry["vertices"], geometry["faces"]
    inputs = json.loads((source / "inputs.json").read_text())
    savedCameras = json.loads((source / "cameras.json").read_text())
    cameras = [photoCamera(camera, item, "fitted") for camera, item in zip(savedCameras, inputs)]
    implementation = {name: (projectRoot / name).read_bytes() for name in (
        "tools/runEyeRefinementExperiment.py", "src/faceAppearance/eyeContours.py",
        "src/faceAppearance/eyeRefinement.py", "src/faceAppearance/semanticProjection.py",
        "src/faceAppearance/surfaceVisibility.py", "src/faceAppearance/anatomicalBoundaries.py",
        "src/faceAppearance/geometryRefinement.py", "src/faceAppearance/camera.py", "src/faceAppearance/texture.py")}
    observations, manifest, cacheHashes = loadEyeCache(cache, inputs, minimumRawPeak=minimumRawPeak,
                                                      inputsSha256=fileHash(source / "inputs.json"))
    assetPath = projectRoot / "assets/anatomical/uff6986/boundaries.json"
    loops = validateBoundaryAsset(vertices, faces, json.loads(assetPath.read_text()))
    eyeLoops = {name: loop for name, loop in loops.items() if name.endswith("Eye")}
    binding = bindTemplateEyeContours(vertices, faces, eyeLoops, cameras[0], observations[0])
    print("Computing continuous-coordinate visibility for fixed eyelid bindings", flush=True)
    shapes = [tuple(reversed(item["orientedSize"])) for item in inputs]
    visibility = visibleVertices(vertices, faces, cameras, shapes, minimumCosine=.05, visibilityMode="exact")
    bound = buildEyeContourConstraints(vertices, faces, cameras, binding, observations, visibility, cornerWeight=cornerWeight)
    print(f"Fitting {len(bound['constraints'])} frozen corner/curve groups", flush=True)
    fitStart = perf_counter()
    fitted = refineEyeGeometry(vertices, faces, cameras, bound["constraints"], iterations=iterations,
                    supportRadius=supportRadiusFraction*float(np.ptp(vertices, axis=0).max()),
                    controlCount=controlCount, referenceWeight=referenceWeight)
    fitSeconds = perf_counter()-fitStart
    refined = fitted["vertices"]
    finalVisibility = visibleVertices(refined, faces, cameras, shapes, minimumCosine=.05, visibilityMode="exact")
    if not np.array_equal(vertices[~fitted["activeVertexMask"]], refined[~fitted["activeVertexMask"]]):
        raise AssertionError("Vertices outside eye support changed")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in implementation.items():
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    np.savez_compressed(output / "geometryRefined.npz", **{**geometry, "vertices": refined})
    np.savez_compressed(output / "fitState.npz", activeVertexMask=fitted["activeVertexMask"],
                        controlPoints=fitted["controlPoints"], controlOffsets=fitted["controlOffsets"],
                        initialVisibility=visibility, finalVisibility=finalVisibility)
    writeJson(output / "templateBinding.json", binding)
    writeJson(output / "photoContours.json", observations)
    writeJson(output / "constraints.json", bound)
    writeJson(output / "history.json", fitted["history"])
    metrics = {"before": eyeMetrics(vertices, binding, cameras, observations),
               "after": eyeMetrics(refined, binding, cameras, observations)}
    writeJson(output / "eyeMetrics.json", metrics)
    visibilityAudit = []
    for constraint in bound["constraints"]:
        indices, viewIndex = constraint["vertexIndices"], constraint["viewIndex"]
        retained = finalVisibility[viewIndex, indices].all(1)
        visibilityAudit.append({"name": constraint["name"], "viewIndex": viewIndex,
                                "sampleCount": len(indices), "visibleAfterFit": int(retained.sum())})
    report = {"status": "complete", "source": source, "eyeCache": cache, "eyeCacheHashes": cacheHashes,
              "detector": manifest["model"], "minimumRawPeakScore": minimumRawPeak,
              "sourceHashes": {name: fileHash(source / name) for name in ("geometry.npz", "cameras.json", "inputs.json")},
              "boundaryAssetSha256": fileHash(assetPath), "geometryDiagnostics": fitted["diagnostics"],
              "constraintDiagnostics": bound["diagnostics"], "eyeMetrics": metrics,
              "visibilityAfterFit": visibilityAudit, "fitSeconds": fitSeconds,
              "cameraParametersFrozen": True, "topologyAndUvsPreserved": True,
              "outsideEyeSupportExactlyUnchanged": True,
              "implementationHashes": {name: hashlib.sha256(content).hexdigest() for name, content in implementation.items()},
              "limits": fitted["diagnostics"]["limits"] + [
                  "Detector eye curves and projected template extrema are automatic correspondences, not anatomical ground truth",
                  "Raw heatmap score threshold is an experiment heuristic; it is not a calibrated occlusion probability",
                  "Different eyelid opening/gaze across photos is not modeled with per-view expression parameters",
                  "These geometry-only scores are in-sample; compare final original-camera grey and textured renders"]}
    drawComparison(output, inputs, cameras, geometry, refined, binding, observations)
    report["seconds"] = perf_counter()-start
    report["outputHashes"] = {name: fileHash(output / name) for name in (
        "geometryRefined.npz", "fitState.npz", "templateBinding.json", "photoContours.json", "constraints.json", "eyeMetrics.json")}
    writeJson(output / "report.json", report)
    print(json.dumps({"fitSeconds": fitSeconds, "totalSeconds": report["seconds"],
                      "diagnostics": fitted["diagnostics"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/exactVisibilityExperiment")
    parser.add_argument("--eye-cache", type=Path, default=projectRoot / "runs/eyeLandmarkValidatedCache")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/eyeCornerRefinementFinal")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-weight", type=float, default=12.)
    parser.add_argument("--support-radius-fraction", type=float, default=.16)
    parser.add_argument("--controls", type=int, default=24)
    parser.add_argument("--corner-weight", type=float, default=8.)
    parser.add_argument("--minimum-raw-peak", type=float, default=.6)
    args = parser.parse_args()
    run(args.input, args.eye_cache, args.output, iterations=args.iterations,
        referenceWeight=args.reference_weight, supportRadiusFraction=args.support_radius_fraction,
        controlCount=args.controls, cornerWeight=args.corner_weight, minimumRawPeak=args.minimum_raw_peak)
