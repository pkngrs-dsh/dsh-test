"""CPU diagnostics for selfie crop/context experiments in original photo pixels.

The m135_raw cameras, eye binding and original-photo targets are fixed. Learned
UV consistency, geometry displacement and independently fitted cameras remain
diagnostics; none is used as a truth-based reconstruction score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.eyeContours import bindTemplateEyeContours
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.texture import buildUvLookup, interpolateSurface
from diagnoseUffBackgroundAblation import (arrayHash, arrays, cameraAvailable,
    fileHash, geometryChange, project, resolvePath, rotationAngle, stats)
from renderCropComparison import photoCamera, validatePhotoPair
from runEyeRefinementExperiment import loadEyeCache

REFERENCE = "m135_raw"
REPEAT = "m135_repeat"
COMPLETE = {"succeeded", "complete", "completed", "measured"}
CORE = [f"m{margin}_{background}" for margin in (110, 135, 165)
        for background in ("raw", "backgroundGray", "headGray")]


def transformPixels(points, matrix):
    values = np.column_stack([np.asarray(points), np.ones(len(points))]) @ np.asarray(matrix).T
    return values[:, :2] / values[:, 2:]


def loadVariant(record, root):
    directory = resolvePath(record["outputDirectory"], root)
    resultPath = directory / "result.json"
    result = json.loads(resultPath.read_text(encoding="utf-8"))
    if record.get("resultSha256") and fileHash(resultPath) != record["resultSha256"]:
        raise ValueError("Result manifest changed after successful reconstruction")
    for name, expected in result.get("outputHashes", {}).items():
        if fileHash(directory / name) != expected:
            raise ValueError(f"Saved output differs from reconstruction provenance: {name}")
    geometry = arrays(directory / "geometry.npz")
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    cameras = json.loads((directory / "cameras.json").read_text(encoding="utf-8"))
    count = len(inputs)
    if (not count or len(cameras) != count or len(geometry["uvMaps"]) != count
            or len(geometry["faceMasks"]) != count):
        raise ValueError("Saved photo, camera, UV and mask view counts differ")
    if not np.isfinite(geometry["vertices"]).all():
        raise ValueError("Final vertices are not finite")
    photoCameras = [photoCamera(camera, item, "fitted") if cameraAvailable(camera) else None
                    for camera, item in zip(cameras, inputs)]
    nativePath = directory / "nativeCameras.json"
    native = json.loads(nativePath.read_text()) if nativePath.exists() else None
    nativePhoto = ([photoCamera(camera, item, "native") if cameraAvailable(camera) else None
                    for camera, item in zip(native, inputs)] if native else None)
    names = ("geometry.npz", "inputs.json", "cameras.json", "nativeCameras.json",
             "viewTransforms.json", "result.json")
    return {"record": record, "directory": directory, "geometry": geometry, "inputs": inputs,
            "cameras": cameras, "photoCameras": photoCameras, "nativePhotoCameras": nativePhoto,
            "hashes": {name: fileHash(directory / name) for name in names if (directory / name).exists()}}


def originalFaceScale(item):
    box = np.asarray(item.get("faceDetection", {}).get("boxXyxy",
                     item.get("cropReferenceBoxXyxy")), float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.min(box[2:]-box[:2]) <= 0:
        raise ValueError("Reference input lacks a valid original-photo face box")
    return float(np.max(box[2:] - box[:2]))


def uvSamples(geometry, view, lookup, seed):
    uv = geometry["uvMaps"][view]
    yy, xx = np.mgrid[0:uv.shape[0]:5, 0:uv.shape[1]:5]
    values = uv[yy, xx]
    valid = ((geometry["faceMasks"][view, yy, xx] > .5)
             & np.isfinite(values).all(-1) & (values >= 0).all(-1) & (values <= 1).all(-1))
    uvPixels = np.rint(np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0), 0, 1)*511).astype(int)
    valid &= lookup["validMask"][uvPixels[..., 1], uvPixels[..., 0]]
    pixels, uvPixels = np.stack([xx, yy], -1)[valid], uvPixels[valid]
    order = np.random.default_rng(seed+view).permutation(len(pixels))
    heldOut = order[:max(12, len(pixels)//5)] if len(pixels) >= 60 else np.empty(0, int)
    return pixels, uvPixels, heldOut


def residual(points, targets, camera, indices, originalTransform, faceScale):
    if camera is None or not cameraAvailable(camera):
        return {"status": "unavailable", "reason": "missingCameraMatrices"}
    pixels, depth = project(points, camera)
    valid = np.isfinite(pixels).all(1) & (depth > 1e-9)
    selected = indices[valid[indices]]
    error = np.linalg.norm(pixels[selected]-targets[selected], axis=-1)
    originalProjected = transformPixels(pixels[selected], originalTransform)
    originalTarget = transformPixels(targets[selected], originalTransform)
    originalError = np.linalg.norm(originalProjected-originalTarget, axis=-1)
    return {"status": "measured", "sampleCount": len(indices), "usableCount": len(selected),
            "workingErrorPixels": stats(error), "originalErrorPixels": stats(originalError),
            "errorOverReferenceOriginalFaceBoxMaxSide": stats(originalError/faceScale)}


def uvAndCameraDiagnostics(variant, reference, lookup, seed):
    geometry = variant["geometry"]
    surface = interpolateSurface(geometry["vertices"], geometry["faces"], lookup)
    result = []
    for view, (item, referenceInput) in enumerate(zip(variant["inputs"], reference["inputs"])):
        pixels, uv, heldOut = uvSamples(geometry, view, lookup, seed)
        points = surface[uv[:, 1], uv[:, 0]]
        originalPixels = transformPixels(pixels, item["workingToInput"])
        faceScale = originalFaceScale(referenceInput)
        own = variant["cameras"][view]
        fixed = reference["photoCameras"][view]
        row = {"viewIndex": view, "originalFaceScalePixels": faceScale,
               "pointCount": len(points), "heldOutCount": len(heldOut),
               "orderedWorkingPixelsAndUvSha256": arrayHash(np.column_stack([pixels, uv])),
               "heldOutIndicesSha256": arrayHash(heldOut),
               "ownFitted": residual(points, pixels, own, heldOut, item["workingToInput"], faceScale),
               "fixedReferencePhotoCamera": residual(points, originalPixels, fixed, heldOut, np.eye(3), faceScale),
               "savedFit": {key: own.get(key) for key in ("valid", "reason", "inlierFraction",
                   "positiveDepthFraction", "heldOutMedianErrorPixels", "heldOutP90ErrorPixels")},
               "ownToOriginalPixelTransform": item["workingToInput"],
               "limits": "Own crop UV targets/splits differ across margins; consistency diagnostics only; not photo holdout or quality ranking"}
        if variant["nativePhotoCameras"]:
            native = variant["nativePhotoCameras"][view]
            row["ownNative"] = residual(points, originalPixels, native, heldOut, np.eye(3), faceScale)
        camera, baselineCamera = variant["photoCameras"][view], reference["photoCameras"][view]
        if camera is not None:
            k = camera["intrinsics"]
            row["ownPhotoFxFyCxCy"] = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
        if camera is not None and baselineCamera is not None:
            row["ownFittedRotationDeltaFromReferenceDegrees"] = rotationAngle(camera["rotation"] @ baselineCamera["rotation"].T)
            row["ownPhotoIntrinsicsDeltaFromReference"] = camera["intrinsics"]-baselineCamera["intrinsics"]
        result.append(row)
    return result


def aperture(inner, outer, upper, lower, support=None):
    inner, outer, upper, lower = map(np.asarray, (inner, outer, upper, lower))
    chord = outer-inner
    width = float(np.linalg.norm(chord))
    if width <= 1e-8 or not np.isfinite(width):
        return {"status": "unavailable", "reason": "degenerateProjectedEyeWidth"}
    normal = np.array([-chord[1], chord[0]]) / width
    normal *= 1 if normal[1] >= 0 else -1
    heights = (lower-upper) @ normal
    support = np.ones(len(heights), bool) if support is None else np.asarray(support).copy()
    support[[0, -1]] = False
    if not support.any():
        return {"status": "unavailable", "reason": "noCommonInteriorUpperLowerSamples"}
    height = float(np.max(heights[support]))
    crossing = float(np.mean(heights[support] < -.01*width))
    return {"status": "measured", "widthPixels": width, "heightPixels": height,
            "heightOverWidth": height/width, "interiorPairCount": int(support.sum()),
            "crossingFraction": crossing, "plausibleProjectedOrdering": bool(height > 0 and crossing <= .1),
            "definition": "maximum signed upper/lower separation perpendicular to each projected eye corner chord; sampled paired curves"}


def eyeSamples(variant, binding, observation, view, camera):
    if camera is None:
        return None
    geometry = variant["geometry"]
    shape = tuple(observation["imageShape"])
    visible = visibleVertices(geometry["vertices"], geometry["faces"], [camera], [shape],
                              minimumCosine=.05, visibilityMode="exact")[0]
    result = {}
    for name, eye in binding["eyes"].items():
        target = observation["eyes"].get(eye["labelId"], {})
        if not target.get("valid"):
            continue
        width = float(np.linalg.norm(np.asarray(target["outerCorner"])-target["innerCorner"]))
        parts = {}
        for part, targetKey in (("upper", "upperCurve"), ("lower", "lowerCurve"),
                                ("innerCorner", "innerCorner"), ("outerCorner", "outerCorner")):
            definition = eye[part]
            indices = np.asarray(definition["vertexIndices"])
            weights = np.asarray(definition["barycentric"])
            points = (geometry["vertices"][indices]*weights[..., None]).sum(1)
            pixels, depth = project(points, camera)
            positive = (depth > 1e-9) & np.isfinite(pixels).all(1)
            selectedVisible = (visible[indices] | ~(weights > 1e-12)).all(1) & positive
            targetPixels = np.asarray(target[targetKey]).reshape(-1, 2)
            parts[part] = {"pixels": pixels, "positive": positive, "visible": selectedVisible,
                           "errors": np.linalg.norm(pixels-targetPixels, axis=-1)}
        result[name] = {"labelId": eye["labelId"], "width": width, "target": target, "parts": parts}
    return result


def summarizeEyeView(raw, supportIds):
    usable = {key: raw[key] for key in supportIds if raw.get(key) is not None}
    if not usable:
        return {"status": "unavailable", "reason": "noUsableFixedCamera"}
    commonNames = set.intersection(*(set(item) for item in usable.values()))
    if not commonNames:
        return {"status": "unavailable", "reason": "noReliablePhotoEyes"}
    common = {}
    for name in sorted(commonNames):
        common[name] = {part: {
            "visible": np.logical_and.reduce([item[name]["parts"][part]["visible"] for item in usable.values()]),
            "positive": np.logical_and.reduce([item[name]["parts"][part]["positive"] for item in usable.values()])}
            for part in ("upper", "lower", "innerCorner", "outerCorner")}
    rows = {}
    for key, item in usable.items():
        allVisible, allPositive, allVisibleNorm, allPositiveNorm = [], [], [], []
        eyes, enoughParts = {}, True
        for name in sorted(commonNames):
            eye = item[name]
            parts, projected = {}, {}
            for part, values in eye["parts"].items():
                positive, visible = common[name][part]["positive"], common[name][part]["visible"]
                errors = values["errors"]
                parts[part] = {"sampleCount": len(errors), "ownPositiveDepthCount": int(values["positive"].sum()),
                    "ownVisibleCount": int(values["visible"].sum()), "commonPositiveDepthCount": int(positive.sum()),
                    "commonVisibleCount": int(visible.sum()), "commonPositiveErrorPixels": stats(errors[positive]),
                    "commonVisibleErrorPixels": stats(errors[visible]),
                    "commonVisibleErrorOverTargetEyeWidth": stats(errors[visible]/eye["width"]),
                    "projectedOriginalPixels": values["pixels"], "ownVisibleMask": values["visible"]}
                projected[part] = values["pixels"]
                if part in ("upper", "lower"):
                    allVisible.extend(errors[visible]); allPositive.extend(errors[positive])
                    allVisibleNorm.extend(errors[visible]/eye["width"])
                    allPositiveNorm.extend(errors[positive]/eye["width"])
                    enoughParts &= int(visible.sum()) >= 3
            positivePairs = common[name]["upper"]["positive"] & common[name]["lower"]["positive"]
            visiblePairs = common[name]["upper"]["visible"] & common[name]["lower"]["visible"]
            cornersPositive = all(common[name][part]["positive"].all() for part in ("innerCorner", "outerCorner"))
            target = eye["target"]
            targetAperture = aperture(target["innerCorner"], target["outerCorner"], target["upperCurve"], target["lowerCurve"])
            apertureReports = {}
            for mode, support in (("commonPositive", positivePairs), ("commonVisible", visiblePairs)):
                measured = (aperture(projected["innerCorner"][0], projected["outerCorner"][0],
                            projected["upper"], projected["lower"], support) if cornersPositive else
                            {"status": "unavailable", "reason": "eyeCornerNonpositiveDepth"})
                targetMatched = aperture(target["innerCorner"], target["outerCorner"],
                                         target["upperCurve"], target["lowerCurve"], support)
                if measured["status"] == targetMatched["status"] == "measured":
                    measured["targetOnSameSamples"] = targetMatched
                    measured["absoluteHeightOverWidthError"] = abs(measured["heightOverWidth"]-targetMatched["heightOverWidth"])
                    measured["heightOverTargetWidth"] = measured["heightPixels"]/eye["width"]
                apertureReports[mode] = measured
            eyes[name] = {"labelId": eye["labelId"], "targetEyeWidthPixels": eye["width"], "parts": parts,
                          "targetApertureAllSamples": targetAperture, "projectedAperture": apertureReports}
        enough = bool(len(commonNames) == 2 and enoughParts and len(allVisible) >= 16)
        rows[key] = {"status": "measured", "eyes": eyes,
            "commonVisibleContourErrorPixels": stats(allVisible),
            "commonPositiveContourErrorPixels": stats(allPositive),
            "commonVisibleContourErrorOverTargetEyeWidth": stats(allVisibleNorm),
            "commonPositiveContourErrorOverTargetEyeWidth": stats(allPositiveNorm),
            "sufficientProxySupport": enough,
            "supportRule": "two reliable photo eyes; >=3 common-visible points for each upper/lower curve; >=16 total contour points"}
    return {"status": "measured", "supportVariantIds": list(usable), "commonMasks": common, "variants": rows}


def eyeDiagnostics(case, variants, reference, output, root):
    inputsPath = reference["directory"] / "inputs.json"
    cache = root / "cases" / case["caseId"] / "eyelids"
    observations, cacheManifest, hashes = loadEyeCache(cache, reference["inputs"],
        minimumRawPeak=.6, inputsSha256=fileHash(inputsPath))
    refView = int(case.get("referenceViewIndex", 0))
    if not all(observations[refView]["eyes"].get(label, {}).get("valid") for label in (4, 5)):
        return {"status": "unavailable", "reason": "referencePhotoRequiresTwoReliableEyes",
                "observations": observations, "eyeCacheHashes": hashes}
    camera = reference["photoCameras"][refView]
    if camera is None or not reference["cameras"][refView].get("valid"):
        return {"status": "unavailable", "reason": "referenceCameraNotAccepted"}
    geometry = reference["geometry"]
    asset = ROOT / "assets/anatomical/uff6986/boundaries.json"
    loops = validateBoundaryAsset(geometry["vertices"], geometry["faces"], json.loads(asset.read_text()))
    binding = bindTemplateEyeContours(geometry["vertices"], geometry["faces"],
        {key: value for key, value in loops.items() if key.endswith("Eye")}, camera, observations[refView])
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / "frozenReferenceEyeBinding.json", binding)
    writeJson(output / "originalEyeObservations.json", observations)
    views = []
    for view, observation in enumerate(observations):
        raw = {key: eyeSamples(variant, binding, observation, view, reference["photoCameras"][view])
               for key, variant in variants.items()}
        core = [key for key in CORE if key in variants]
        allIds = list(variants)
        views.append({"viewIndex": view, "fixedReferenceSavedCameraValid": reference["cameras"][view].get("valid", False),
            "photoObservationDiagnostics": {label: {"valid": eye.get("valid"), "reason": eye.get("reason"),
                "diagnostics": eye.get("diagnostics")} for label, eye in observation["eyes"].items()},
            "coreSupport": summarizeEyeView(raw, core), "allCompletedSupport": summarizeEyeView(raw, allIds)})
    return {"status": "measured", "coordinateFrame": "originalExifOrientedPixels", "referenceViewIndex": refView,
            "cameraSource": "frozen m135_raw fitted K converted once through saved inputToWorking inverse; R/t unchanged",
            "bindingPath": str(output / "frozenReferenceEyeBinding.json"),
            "bindingSha256": fileHash(output / "frozenReferenceEyeBinding.json"),
            "observationSource": cacheManifest.get("model"), "eyeCacheHashes": hashes,
            "boundaryAssetSha256": fileHash(asset), "views": views,
            "apertureLimits": "Projected template eye-hole proxy; upper/lower paired arc samples and hidden geometry are not exact anatomical eyelids; aperture is not used for winner ranking"}


def caseRankings(eyes, variants):
    if eyes.get("status") != "measured":
        return {"status": "unavailable", "reason": "noFrontalEyeProxy"}
    front = eyes["views"][eyes["referenceViewIndex"]]
    result = {"status": "measured", "metric": "fixed-reference original-photo common-visible contour error / target eye width",
              "notGroundTruth": True}
    for scope in ("coreSupport", "allCompletedSupport"):
        measurement = front[scope]
        if measurement.get("status") != "measured":
            result[scope] = {"status": "unavailable", "reason": measurement.get("reason")}
            continue
        rows = measurement["variants"]
        if not all(row["sufficientProxySupport"] for row in rows.values()):
            result[scope] = {"status": "unavailable", "reason": "insufficientCommonVisibleContourSupport"}
            continue
        values = {key: row["commonVisibleContourErrorOverTargetEyeWidth"]["mean"]
                  for key, row in rows.items() if key != REPEAT}
        if REFERENCE not in values or any(value is None for value in values.values()):
            result[scope] = {"status": "unavailable", "reason": "missingBaselineOrFiniteProxy"}
            continue
        baseline = values[REFERENCE]
        ranked = sorted(values, key=lambda key: values[key])
        result[scope] = {"status": "ranked", "availableVariantIds": list(values),
            "allCorePresent": all(key in values for key in CORE),
            "ranking": [{"variantId": key, "rank": index+1, "score": values[key],
                         "deltaFromReference": values[key]-baseline,
                         "fractionChangeFromReference": values[key]/baseline-1 if baseline > 0 else None}
                        for index, key in enumerate(ranked)],
            "winnerIds": [key for key in ranked if abs(values[key]-values[ranked[0]]) <= 1e-8],
            "tiedScoreTolerance": 1e-8,
            "limits": "Descriptive in-sample proxy ordering, conditional on this frozen camera/binding/visibility; no inferential significance or 3D quality claim"}
    return result


def diagnoseCase(case, root, output, seed):
    begin = perf_counter()
    variants, failures = {}, []
    for record in case["variants"]:
        if record.get("status") not in COMPLETE:
            failures.append({"id": record["id"], "status": record.get("status", "pending"), "reason": record.get("reason", record.get("error"))})
            continue
        try:
            variants[record["id"]] = loadVariant(record, root)
        except Exception as exc:
            failures.append({"id": record["id"], "status": "analysisFailed", "error": str(exc)})
    if REFERENCE not in variants:
        return {"caseId": case["caseId"], "status": "pending", "reason": "referenceMissing", "variants": failures}
    reference = variants[REFERENCE]
    for item in reference["inputs"]:
        validatePhotoPair(item, item)
    for key, variant in variants.items():
        for topology in ("faces", "uvs"):
            if not np.array_equal(variant["geometry"][topology], reference["geometry"][topology]):
                raise ValueError(f"Topology differs: {key}/{topology}")
        if len(variant["inputs"]) != len(reference["inputs"]):
            raise ValueError("View counts differ from reference")
        for item, base in zip(variant["inputs"], reference["inputs"]):
            if item["sha256"] != base["sha256"] or item["orientedSize"] != base["orientedSize"]:
                raise ValueError("Variants do not use identical ordered original photos")
        rawKey = REFERENCE if key == REPEAT else key.rsplit("_", 1)[0]+"_raw"
        if rawKey in variants:
            for name in ("uvMaps", "faceMasks"):
                if not np.array_equal(variant["geometry"][name], variants[rawKey]["geometry"][name]):
                    raise ValueError(f"Same-crop background/repeat changed frozen {name}: {key}")
            if variant["inputs"] != variants[rawKey]["inputs"]:
                raise ValueError(f"Same-crop background/repeat changed input metadata: {key}")
    geometry = reference["geometry"]
    diagonal = float(np.linalg.norm(np.ptp(geometry["vertices"], axis=0)))
    lookup = buildUvLookup(geometry["uvs"], geometry["faces"], 512)
    rows = []
    for key, variant in variants.items():
        row = {"id": key, "label": variant["record"].get("label", key), "status": "measured",
            "outputDirectory": str(variant["directory"]), "sourceHashes": variant["hashes"],
            "geometryChangeFromReference": geometryChange(variant["geometry"]["vertices"], geometry["vertices"], diagonal),
            "views": uvAndCameraDiagnostics(variant, reference, lookup, seed),
            "cropDiagnostics": [{"viewIndex": index, "cropSize": item.get("cropSize"),
                "cropBoxXyxy": item.get("cropBoxXyxy"), "margin": item.get("cropMargin"),
                "paddingLeftTopRightBottom": item.get("paddingLeftTopRightBottom"),
                "qualityFlags": item.get("qualityFlags"),
                "referenceFaceMaxSideInWorkingPixels": originalFaceScale(base)*float(np.asarray(item["inputToWorking"])[0, 0]),
                "uvMapSha256": arrayHash(variant["geometry"]["uvMaps"][index]),
                "faceMaskSha256": arrayHash(variant["geometry"]["faceMasks"][index])}
                for index, (item, base) in enumerate(zip(variant["inputs"], reference["inputs"]))]}
        cropPrefix = "m135" if key == REPEAT else key.rsplit("_", 1)[0]
        cropMetadataPath = root / "cases" / case["caseId"] / "cropInputs" / cropPrefix / "metadata.json"
        if cropMetadataPath.exists():
            cropMetadata = json.loads(cropMetadataPath.read_text(encoding="utf-8"))
            row["cropInputAudit"] = {"path": str(cropMetadataPath), "sha256": fileHash(cropMetadataPath),
                "views": cropMetadata.get("views"), "boundaryRule": cropMetadata.get("boundaryRule"),
                "facePixelsUnchangedWithinCrop": cropMetadata.get("facePixelsUnchangedWithinCrop"),
                "interpretation": "Larger crops may add replicated padding instead of real context; VGGT sees padding RGB while fusion masks exclude it"}
        rows.append(row)
    try:
        eyes = eyeDiagnostics(case, variants, reference, output, root)
    except Exception as exc:
        eyes = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    report = {"caseId": case["caseId"], "label": case.get("label", case["caseId"]), "status": "measured",
        "referenceVariantId": REFERENCE, "referenceBboxDiagonal": diagonal,
        "referenceViewIndex": int(case.get("referenceViewIndex", 0)), "variants": rows+failures,
        "eyes": eyes, "proxyRankings": caseRankings(eyes, variants), "wallSeconds": perf_counter()-begin}
    if REPEAT in variants:
        repeat = next(row for row in rows if row["id"] == REPEAT)
        report["observedRepeatVertexRms"] = repeat["geometryChangeFromReference"]["canonicalDisplacement"]["rms"]
        if eyes.get("status") == "measured":
            referenceView = eyes["views"][eyes["referenceViewIndex"]]
            values = referenceView["allCompletedSupport"].get("variants", {})
            if REPEAT in values and REFERENCE in values:
                a = values[REFERENCE]["commonVisibleContourErrorOverTargetEyeWidth"]["mean"]
                b = values[REPEAT]["commonVisibleContourErrorOverTargetEyeWidth"]["mean"]
                if a is not None and b is not None:
                    report["observedRepeatFixedEyeProxyDelta"] = b-a
    return report


def aggregate(cases):
    result = {"caseCount": len(cases), "measuredCaseCount": sum(c.get("status") == "measured" for c in cases),
              "eyeMeasuredCaseCount": sum(c.get("eyes", {}).get("status") == "measured" for c in cases),
              "coreVariantIds": CORE, "scopes": {},
              "limits": "Cases are observational selected groups, not verified independent identities; case means get equal weight; ranks are conditional eye proxies, not 3D accuracy"}
    for scope in ("coreSupport", "allCompletedSupport"):
        eligible = [(case, case.get("proxyRankings", {}).get(scope, {})) for case in cases]
        eligible = [(case, ranks) for case, ranks in eligible if ranks.get("status") == "ranked"
                    and (scope != "coreSupport" or ranks["allCorePresent"])]
        collected = {}
        for case, ranks in eligible:
            for row in ranks["ranking"]:
                item = collected.setdefault(row["variantId"], {"cases": [], "winCaseIds": [], "improvedCaseIds": [],
                                                              "worsenedCaseIds": [], "tiedReferenceCaseIds": []})
                item["cases"].append({"caseId": case["caseId"], **row})
                if row["variantId"] in ranks["winnerIds"]:
                    item["winCaseIds"].append(case["caseId"])
                delta = row["deltaFromReference"]
                item["improvedCaseIds" if delta < -1e-8 else "worsenedCaseIds" if delta > 1e-8 else "tiedReferenceCaseIds"].append(case["caseId"])
        for item in collected.values():
            item["eligibleCaseCount"] = len(item["cases"])
            item["winCount"] = len(item["winCaseIds"])
            item["meanCaseRank"] = float(np.mean([r["rank"] for r in item["cases"]]))
            item["caseScoreStats"] = stats([r["score"] for r in item["cases"]])
            item["pairedDeltaFromReferenceStats"] = stats([r["deltaFromReference"] for r in item["cases"]])
        result["scopes"][scope] = {"eligibleCaseIds": [case["caseId"] for case, _ in eligible],
                                   "eligibleCaseCount": len(eligible), "variants": collected}
    pairs = [("coreSupport", f"m135_{bg}", f"m{margin}_{bg}")
             for bg in ("raw", "backgroundGray", "headGray") for margin in (110, 165)]
    pairs += [("coreSupport", f"m{margin}_raw", f"m{margin}_{bg}")
              for margin in (110, 135, 165) for bg in ("backgroundGray", "headGray")]
    pairs += [("allCompletedSupport", f"m135_{bg}", f"shared135_{bg}") for bg in ("raw", "headGray")]
    contrasts = []
    for scope, first, second in pairs:
        rows = []
        for case in cases:
            ranking = case.get("proxyRankings", {}).get(scope, {})
            if ranking.get("status") != "ranked" or (scope == "coreSupport" and not ranking["allCorePresent"]):
                continue
            scores = {item["variantId"]: item["score"] for item in ranking["ranking"]}
            if first in scores and second in scores:
                rows.append({"caseId": case["caseId"], "firstScore": scores[first], "secondScore": scores[second],
                    "deltaSecondMinusFirst": scores[second]-scores[first],
                    "fractionChangeSecondFromFirst": scores[second]/scores[first]-1 if scores[first] > 0 else None})
        contrasts.append({"scope": scope, "first": first, "second": second, "cases": rows,
            "eligibleCaseCount": len(rows), "secondLowerCount": sum(row["deltaSecondMinusFirst"] < -1e-8 for row in rows),
            "secondHigherCount": sum(row["deltaSecondMinusFirst"] > 1e-8 for row in rows),
            "casePairedDeltaStats": stats([row["deltaSecondMinusFirst"] for row in rows]),
            "caseRelativeChangeStats": stats([row["fractionChangeSecondFromFirst"] for row in rows
                                              if row["fractionChangeSecondFromFirst"] is not None])})
    result["pairedFactorContrasts"] = contrasts
    return result


def renderOverview(report, destination):
    """Static scientific figure; missing or insufficient common support stays blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    values = np.full((len(cases), len(CORE)), np.nan)
    labels = []
    for index, case in enumerate(cases):
        ranking = case.get("proxyRankings", {}).get("coreSupport", {})
        n = None
        if ranking.get("status") == "ranked" and ranking["allCorePresent"]:
            for row in ranking["ranking"]:
                if row["variantId"] in CORE and row["fractionChangeFromReference"] is not None:
                    values[index, CORE.index(row["variantId"])] = row["fractionChangeFromReference"] * 100
            eyes = case["eyes"]
            n = eyes["views"][eyes["referenceViewIndex"]]["coreSupport"]["variants"][REFERENCE]["commonVisibleContourErrorPixels"]["count"]
        labels.append(case["caseId"] + (f"  (N={n})" if n is not None else "  (unavailable)"))
    finite = values[np.isfinite(values)]
    limit = max(10., min(50., float(np.max(abs(finite))))) if finite.size else 10.
    figure, (axis, counts) = plt.subplots(2, 1, figsize=(14.4, max(5.3, 2.8+.45*len(cases))),
        gridspec_kw={"height_ratios": [max(2.5, len(cases)*.4), .9]})
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e7e9ed")
    shown = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(np.arange(len(cases)), labels, fontsize=10)
    variantLabels = [f"{margin/100:.2f}\n{name}" for margin in (110, 135, 165)
                     for name in ("original", "environment gray", "head gray")]
    axis.set_xticks(np.arange(len(CORE)), variantLabels, fontsize=9)
    for row in range(len(cases)):
        for column in range(len(CORE)):
            value = values[row, column]
            text = f"{value:+.1f}%" if np.isfinite(value) else "n/a"
            color = "white" if np.isfinite(value) and abs(value) > .65*limit else "#17212b"
            axis.text(column, row, text, ha="center", va="center", fontsize=9, color=color)
    axis.set_xticks(np.arange(len(CORE)+1)-.5, minor=True)
    axis.set_yticks(np.arange(len(cases)+1)-.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(shown, ax=axis, pad=.02, fraction=.035,
                              extend="both" if finite.size and np.max(abs(finite)) > limit else "neither")
    colorbar.set_label("Relative to m135_raw (%)", fontsize=9)
    counts.set_axis_off()
    summary = report["summary"]["scopes"]["coreSupport"]
    line = []
    for key in CORE:
        item = summary["variants"].get(key, {})
        line.append(f"{len(item.get('improvedCaseIds', []))}/{item.get('eligibleCaseCount', 0)}")
    table = counts.table(cellText=[line], rowLabels=["Lower proxy / eligible cases"],
                         cellLoc="center", loc="upper center", bbox=[0, .5, .95, .5])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#cbd1d9"); cell.set_facecolor("#f2f4f7")
    counts.text(0, .1, "Lower means closer to automatic eye contours in this fixed-camera proxy. It does not establish more accurate 3D geometry.",
                transform=counts.transAxes, fontsize=9, color="#364454")
    figure.suptitle("Selfie crop study | frontal eye contour proxy", fontsize=17, y=.985)
    figure.text(.18, .915, "Frozen m135_raw camera, targets and binding | common visible samples per case | equal case weights", fontsize=10)
    figure.subplots_adjust(left=.18, right=.97, top=.855, bottom=.04, hspace=.45)
    figure.savefig(destination, dpi=170, facecolor="white")
    plt.close(figure)
    return {"path": str(destination), "sha256": fileHash(destination),
            "scope": "coreSupport; only complete nine-variant cases with sufficient common-visible frontal support"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/uffSelfieCropStudy20261012/cropManifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    output = args.output.resolve() if args.output else root / "metrics.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError("Existing metrics; pass --overwrite to replace analysis only")
    batch = json.loads(manifest.read_text(encoding="utf-8"))
    report = {"schemaVersion": 1, "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(manifest), "sourceManifestSha256": fileHash(manifest),
        "implementationSha256": fileHash(__file__), "referenceVariantId": REFERENCE,
        "implementationHashes": {name: fileHash(ROOT / name) for name in (
            "tools/diagnoseUffSelfieCrop.py", "tools/diagnoseUffBackgroundAblation.py",
            "tools/renderCropComparison.py", "tools/runEyeRefinementExperiment.py",
            "src/faceAppearance/eyeContours.py", "src/faceAppearance/semanticProjection.py",
            "src/faceAppearance/surfaceVisibility.py", "src/faceAppearance/photoCameras.py")},
        "notGroundTruth": True, "cases": [],
        "definitions": {"projection": "original EXIF-oriented image integer-center coordinates; frozen m135_raw photo camera",
            "eyes": "original-photo STFR target and A-bound mesh samples frozen; per-view common visibility/positive-depth intersections",
            "coreSupport": "intersection of completed 3 margins x 3 background variants; shared-crop and repeat excluded from this intersection",
            "allCompletedSupport": "intersection including completed shared-crop and repeat variants",
            "ranking": "front view only, common-visible eye-width-normalized contour error; sufficient two-eye support required; own PnP never ranked",
            "UV": "per-variant 5px grid and seed+view held-out split; original-photo residual normalized by baseline detector-box maximum side",
            "geometry": "raw and proper-similarity-aligned paired vertices relative to baseline bbox diagonal; change only, no quality meaning"},
        "limits": ["No calibrated cameras, independent 3D truth, or held-out photos",
            "Core ranking uses a common support within each case; different cases have different counts",
            "A frozen camera can favor its baseline; canonical geometry/frame drift is included in this comparison",
            "Original-photo automatic eye targets, proxy eye-hole binding and mesh visibility are imperfect",
            "Projected eye apertures include hidden/uncertain template boundaries unless marked commonVisible",
            "Own PnP uses crop-dependent UV and VGGT intrinsics and is unstable; excluded from ranking",
            "Different margins change source resolution and UV/mask predictions; this is an operational crop comparison",
            "A wider crop can contain more edge-replicated pixels, not more real scene context; VGGT sees those pixels although valid-image masks exclude them from fusion support",
            "Same-margin background variants freeze original RGB-derived UV/mask; only VGGT RGB changes",
            "One repeated baseline measures observed numerical variation, not a noise distribution or significance",
            "Missing eye caches/targets do not remove the case's geometry and UV diagnostics"]}
    start = perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in batch["cases"]:
        if args.case and case["caseId"] not in args.case:
            continue
        print(f"ANALYZE {case['caseId']}", flush=True)
        try:
            row = diagnoseCase(case, root, output.parent / "analysis" / case["caseId"], int(batch.get("seed", 2026)))
        except Exception as exc:
            row = {"caseId": case["caseId"], "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        report["cases"].append(row)
        writeJson(output.with_suffix(".pending.json"), report)
        output.with_suffix(".pending.json").replace(output)
        print(f"ANALYZED {case['caseId']} {row['status']} eyes={row.get('eyes', {}).get('status')}", flush=True)
    report["summary"] = aggregate(report["cases"])
    report["status"] = "complete" if all(row["status"] == "measured" and all(
        variant["status"] in {"measured", "skipped"} for variant in row["variants"]) for row in report["cases"]) else "partial"
    report["wallSeconds"] = perf_counter()-start
    report["completedAt"] = datetime.now(timezone.utc).isoformat()
    figurePath = output.with_name("metricsOverview.png" if output.name == "metrics.json" else output.stem+"Overview.png")
    try:
        report["overviewFigure"] = renderOverview(report, figurePath)
    except Exception as exc:
        report["overviewFigure"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    writeJson(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "wallSeconds": report["wallSeconds"],
        "cases": len(report["cases"]), "eyeMeasuredCases": report["summary"]["eyeMeasuredCaseCount"],
        "coreRankedCases": report["summary"]["scopes"]["coreSupport"]["eligibleCaseCount"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
