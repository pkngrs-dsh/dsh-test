"""CPU diagnostics for frozen-UV background ablations; no quality ground truth.

The freshly reconstructed ``original`` is the common reference. This tool does
not fit cameras, run models, alter geometry, or use the historical mesh as truth.
"""
from __future__ import annotations

import argparse
import hashlib
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
from faceAppearance.camera import projectPoints
from faceAppearance.eyeContours import bindTemplateEyeContours
from faceAppearance.photoCameras import canonicalCamera
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.texture import buildUvLookup, interpolateSurface
from runEyeRefinementExperiment import loadEyeCache
from runUffFrontPriority import workingObservations


COMPLETE = {"succeeded", "complete", "completed", "measured"}


def fileHash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrayHash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in (data.files if keys is None else keys) if key in data}


def stats(values):
    values = np.asarray(values, float).reshape(-1)
    finite = np.isfinite(values)
    result = {"count": int(len(values)), "finiteCount": int(finite.sum()),
              "mean": None, "rms": None, "p50": None, "p90": None,
              "p95": None, "maximum": None}
    if finite.any():
        valid = values[finite]
        result.update(mean=float(valid.mean()), rms=float(np.sqrt(np.mean(valid ** 2))),
                      p50=float(np.median(valid)), p90=float(np.percentile(valid, 90)),
                      p95=float(np.percentile(valid, 95)), maximum=float(valid.max()))
    return result


def resolvePath(value, manifestDirectory):
    path = Path(value)
    if path.is_absolute():
        return path
    return next((p for p in (manifestDirectory / path, ROOT / path, path) if p.exists()),
                manifestDirectory / path)


def cameraAvailable(camera):
    return all(key in camera and np.asarray(camera[key]).shape == shape and
               np.isfinite(camera[key]).all() for key, shape in
               (("intrinsics", (3, 3)), ("rotation", (3, 3)), ("translation", (3,))))


def project(vertices, camera):
    if not cameraAvailable(camera):
        return None, None
    return projectPoints(np.asarray(vertices), np.asarray(camera["intrinsics"]),
                         np.asarray(camera["rotation"]), np.asarray(camera["translation"]))


def rotationAngle(rotation):
    u, _, vt = np.linalg.svd(np.asarray(rotation))
    proper = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(proper) - 1) / 2, -1, 1))))


def alignSimilarity(moving, reference):
    """Least-squares proper similarity for row vectors; no reflection allowed."""
    moving, reference = np.asarray(moving, float), np.asarray(reference, float)
    mc, rc = moving.mean(0), reference.mean(0)
    x, y = moving - mc, reference - rc
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.array([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ np.diag(correction) @ vt
    scale = float(np.sum(singular * correction) / np.sum(x * x))
    translation = rc - scale * mc @ rotation
    return scale * moving @ rotation + translation, {
        "scale": scale, "rowRotation": rotation, "translation": translation,
        "rotationMagnitudeDegrees": rotationAngle(rotation),
        "convention": "aligned = scale * (moving @ rowRotation) + translation",
        "fitSupport": "all corresponding final vertices; identity-sensitive shape can also be absorbed"}


def geometryChange(moving, reference, faceSize):
    delta = np.asarray(moving, float) - reference
    distance = np.linalg.norm(delta, axis=1)
    aligned, transform = alignSimilarity(moving, reference)
    residual = np.linalg.norm(aligned - reference, axis=1)
    return {"vertexCount": len(delta), "canonicalDisplacement": stats(distance),
            "displacementOverOriginalBboxDiagonal": stats(distance / faceSize),
            "signedMeanDeltaXyz": delta.mean(0),
            "p95AbsoluteDeltaXyz": np.percentile(np.abs(delta), 95, axis=0),
            "similarityAlignedResidual": stats(residual),
            "alignedResidualOverOriginalBboxDiagonal": stats(residual / faceSize),
            "fittedSimilarityToReference": transform}


def surfaceProjection(vertices, reference, cameras):
    result = []
    for index, camera in enumerate(cameras):
        pixels, depth = project(vertices, camera)
        original, originalDepth = project(reference, camera)
        if pixels is None:
            result.append({"viewIndex": index, "status": "unavailable", "reason": "missingOriginalCamera"})
            continue
        valid = ((depth > 1e-9) & (originalDepth > 1e-9) &
                 np.isfinite(pixels).all(1) & np.isfinite(original).all(1))
        result.append({"viewIndex": index, "status": "measured", "vertexCount": len(vertices),
                       "commonPositiveDepthCount": int(valid.sum()),
                       "displacementPixels": stats(np.linalg.norm(pixels[valid] - original[valid], axis=1)),
                       "signedDepthChangeCanonical": stats((depth - originalDepth)[valid]),
                       "absoluteDepthChangeCanonical": stats(np.abs(depth - originalDepth)[valid]),
                       "visibilityFiltering": "none; common positive depth only",
                       "cameraSavedValid": camera.get("valid")})
    return result


def uvSampleDefinitions(geometry, seed):
    lookup = buildUvLookup(geometry["uvs"], geometry["faces"], 512)
    samples = []
    for viewIndex, uv in enumerate(geometry["uvMaps"]):
        yy, xx = np.mgrid[0:uv.shape[0]:5, 0:uv.shape[1]:5]
        values = uv[yy, xx]
        valid = ((geometry["faceMasks"][viewIndex, yy, xx] > .5) &
                 np.isfinite(values).all(-1) & (values >= 0).all(-1) & (values <= 1).all(-1))
        safe = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
        uvPixels = np.rint(np.clip(safe, 0, 1) * 511).astype(np.int32)
        valid &= lookup["validMask"][uvPixels[..., 1], uvPixels[..., 0]]
        pixels = np.stack([xx, yy], -1)[valid]
        uvPixels = uvPixels[valid]
        order = np.random.default_rng(seed + viewIndex).permutation(len(pixels))
        heldOut = order[:max(12, len(pixels) // 5)] if len(pixels) >= 60 else np.empty(0, np.int64)
        samples.append({"pixels": pixels, "uvPixels": uvPixels, "heldOut": heldOut})
    return lookup, samples


def projectionResidual(points, targets, camera, indices):
    pixels, depth = project(points, camera)
    if pixels is None:
        return {"status": "unavailable", "reason": "missingCameraMatrices"}
    valid = (depth > 1e-9) & np.isfinite(pixels).all(1)
    support = indices[valid[indices]]
    return {"status": "measured", "sampleCount": len(indices), "usableCount": len(support),
            "positiveDepthFraction": float(valid[indices].mean()) if len(indices) else None,
            "errorPixels": stats(np.linalg.norm(pixels[support] - targets[support], axis=1)),
            "savedCameraValid": camera.get("valid")}


def uvDiagnostics(variant, original, lookup, samples):
    surface = interpolateSurface(variant["geometry"]["vertices"], variant["geometry"]["faces"], lookup)
    result = []
    for viewIndex, sample in enumerate(samples):
        uv = sample["uvPixels"]
        points = surface[uv[:, 1], uv[:, 0]]
        fitted = variant["cameras"][viewIndex]
        result.append({"viewIndex": viewIndex, "pointCount": len(points),
            "heldOutCount": len(sample["heldOut"]),
            "heldOutIndicesSha256": arrayHash(sample["heldOut"]),
            "orderedPixelsAndUvSha256": arrayHash(np.column_stack([sample["pixels"], uv])),
            "fixedOriginalCamera": projectionResidual(points, sample["pixels"], original["cameras"][viewIndex], sample["heldOut"]),
            "ownFittedCamera": projectionResidual(points, sample["pixels"], fitted, sample["heldOut"]),
            "ownNativeCamera": projectionResidual(points, sample["pixels"], variant["native"][viewIndex], sample["heldOut"]),
            "savedFit": {key: fitted.get(key) for key in ("valid", "reason", "inlierFraction",
                "positiveDepthFraction", "heldOutMedianErrorPixels", "heldOutP90ErrorPixels")}})
    return result


def cameraDiagnostics(variant, original):
    result = []
    for index, (camera, reference) in enumerate(zip(variant["native"], original["native"])):
        k, ka = np.asarray(camera["intrinsics"]), np.asarray(reference["intrinsics"])
        r, ra = np.asarray(camera["rotation"]), np.asarray(reference["rotation"])
        t, ta = np.asarray(camera["translation"]), np.asarray(reference["translation"])
        relative = r @ np.asarray(variant["native"][0]["rotation"]).T
        referenceRelative = ra @ np.asarray(original["native"][0]["rotation"]).T
        fitted = variant["cameras"][index]
        row = {"viewIndex": index, "nativeFxFyCxCy": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
               "intrinsicsDeltaFromOriginal": k - ka,
               "focalFractionChange": [k[0, 0] / ka[0, 0] - 1, k[1, 1] / ka[1, 1] - 1],
               "nativeCanonicalRotationChangeDegrees": rotationAngle(r @ ra.T),
               "nativeRelativeToView0AngleDegrees": rotationAngle(relative),
               "nativeRelativePoseChangeFromOriginalDegrees": rotationAngle(relative @ referenceRelative.T),
               "nativeRelativeToView0Rotation": relative,
               "nativeRelativeToView0Translation": t - relative @ np.asarray(variant["native"][0]["translation"]),
               "nativeCameraCenterCanonical": -r.T @ t,
               "nativeCameraCenterDeltaFromOriginal": -r.T @ t + ra.T @ ta,
               "canonicalizationScale": variant["transforms"][index]["s"],
               "fittedCameraValid": fitted.get("valid")}
        if cameraAvailable(fitted):
            rf = np.asarray(fitted["rotation"])
            row["nativeVsOwnFittedAngleDegrees"] = rotationAngle(r @ rf.T)
            if cameraAvailable(original["cameras"][index]):
                row["fittedRotationChangeFromOriginalDegrees"] = rotationAngle(rf @ np.asarray(original["cameras"][index]["rotation"]).T)
        result.append(row)
    return result


def canonicalPointMaps(variant):
    path = variant["directory"] / "diagnostics" / "vggt.npz"
    if not path.exists():
        return None
    values = arrays(path, ["pointMaps512"])
    if "pointMaps512" not in values:
        return None
    result = []
    for points, transform in zip(values["pointMaps512"], variant["transforms"]):
        result.append(float(transform["s"]) * (points.astype(np.float64) @ np.asarray(transform["R"])) + np.asarray(transform["t"]))
    return np.stack(result)


def pointMapDiagnostics(variant, original, referencePoints, faceSize):
    points = canonicalPointMaps(variant)
    if points is None or referencePoints is None:
        return {"status": "unavailable", "reason": "missingPointMaps512"}
    result = []
    for index, (moving, reference) in enumerate(zip(points, referencePoints)):
        mask = (original["geometry"]["faceMasks"][index] > .5)
        valid = mask & np.isfinite(moving).all(-1) & np.isfinite(reference).all(-1)
        delta = moving[valid] - reference[valid]
        camera = original["cameras"][index]
        row = {"viewIndex": index, "cachedFaceSupportCount": int(mask.sum()),
               "commonFiniteCount": int(valid.sum()),
               "canonicalXyzDisplacement": stats(np.linalg.norm(delta, axis=1)),
               "displacementOverOriginalBboxDiagonal": stats(np.linalg.norm(delta, axis=1) / faceSize)}
        if cameraAvailable(camera):
            depthDelta = delta @ np.asarray(camera["rotation"])[2]
            row["signedDepthChangeInOriginalCamera"] = stats(depthDelta)
            row["absoluteDepthChangeInOriginalCamera"] = stats(np.abs(depthDelta))
        result.append(row)
    return {"status": "measured", "views": result,
            "coordinateRule": "Each world XYZ uses its own saved per-view similarity into canonical space; no coordinate clipping",
            "support": "unchanged original cached face mask > .5, common finite XYZ",
            "meaning": "Change of model-derived point maps and canonicalization, not true depth error"}


def fusionSupport(variant, original):
    result = {}
    path = variant["directory"] / "diagnostics" / "fusion.npz"
    referencePath = original["directory"] / "diagnostics" / "fusion.npz"
    if not path.exists() or not referencePath.exists():
        return {"status": "unavailable"}
    with np.load(path, allow_pickle=False) as data, np.load(referencePath, allow_pickle=False) as reference:
        for key in ("inputValidMasks", "viewValidVertexMasks", "partialValid"):
            if key in data and key in reference:
                value, before = np.asarray(data[key], bool), np.asarray(reference[key], bool)
                result[key] = {"shape": value.shape, "count": int(value.sum()),
                               "originalCount": int(before.sum()), "changedCount": int((value != before).sum())}
        if "inputUvPointMaps" in data and "inputValidMasks" in data:
            xyz, valid = data["inputUvPointMaps"], np.asarray(data["inputValidMasks"], bool)
            saturated = valid & (np.abs(xyz) >= 1).any(-1)
            result["atCoordinateClipBoundary"] = {"validCount": int(valid.sum()), "count": int(saturated.sum()),
                "fraction": float(saturated.sum() / max(valid.sum(), 1)),
                "meaning": "post-clip coordinates at +/-1, not an exact count of pre-clip outliers"}
    return {"status": "measured", **result}


def eyeSamples(variant, binding, observations, camera, imageShape):
    if not cameraAvailable(camera):
        return None
    geometry = variant["geometry"]
    visible = visibleVertices(geometry["vertices"], geometry["faces"], [camera], [imageShape],
                              minimumCosine=.05, visibilityMode="exact")[0]
    result = {}
    for name, eye in binding["eyes"].items():
        observed = observations[0]["eyes"][eye["labelId"]]
        if not observed.get("valid"):
            continue
        width = float(np.linalg.norm(np.asarray(observed["outerCorner"]) - observed["innerCorner"]))
        parts = {}
        for part, targetKey in (("upper", "upperCurve"), ("lower", "lowerCurve"),
                                ("innerCorner", "innerCorner"), ("outerCorner", "outerCorner")):
            definition = eye[part]
            indices = np.asarray(definition["vertexIndices"])
            weights = np.asarray(definition["barycentric"])
            points = (geometry["vertices"][indices] * weights[..., None]).sum(1)
            pixels, depth = project(points, camera)
            target = np.asarray(observed[targetKey]).reshape(-1, 2)
            positive = (depth > 1e-9) & np.isfinite(pixels).all(1)
            activeVertices = weights > 1e-12
            sampleVisible = (visible[indices] | ~activeVertices).all(1) & positive
            parts[part] = {"pixels": pixels, "targets": target, "depth": depth,
                           "positive": positive, "visible": sampleVisible,
                           "errors": np.linalg.norm(pixels - target, axis=1)}
        result[name] = {"targetEyeWidthPixels": width, "labelId": eye["labelId"], "parts": parts}
    return result


def eyeDiagnostics(variants, original, case, eyeCacheRoot, outputDirectory):
    sourceInputsPath = case["sourceDirectory"] / "inputs.json"
    inputs = json.loads(sourceInputsPath.read_text())
    cache = eyeCacheRoot / "cases" / case["caseId"] / "eyelids"
    observations, cacheManifest, hashes = loadEyeCache(cache, inputs, minimumRawPeak=.6,
                                                       inputsSha256=fileHash(sourceInputsPath))
    observations = workingObservations(observations, inputs)
    asset = ROOT / "assets/anatomical/uff6986/boundaries.json"
    geometry = original["geometry"]
    loops = validateBoundaryAsset(geometry["vertices"], geometry["faces"], json.loads(asset.read_text()))
    if not cameraAvailable(original["cameras"][0]):
        raise ValueError("Original frontal camera has no usable matrices")
    binding = bindTemplateEyeContours(geometry["vertices"], geometry["faces"],
        {key: value for key, value in loops.items() if key.endswith("Eye")},
        original["cameras"][0], observations[0])
    outputDirectory.mkdir(parents=True, exist_ok=True)
    writeJson(outputDirectory / "frozenOriginalEyeBinding.json", binding)
    writeJson(outputDirectory / "workingEyeObservations.json", observations)
    shape = tuple(reversed(inputs[0]["workingSize"]))
    result = {"status": "measured", "viewIndex": 0, "bindingReferenceVariant": "original",
              "bindingPath": str(outputDirectory / "frozenOriginalEyeBinding.json"),
              "bindingSha256": fileHash(outputDirectory / "frozenOriginalEyeBinding.json"),
              "observationSource": cacheManifest.get("model"), "eyeCacheHashes": hashes,
              "boundaryAssetSha256": fileHash(asset), "modes": {}}
    for mode in ("fixedOriginalCamera", "ownFittedCamera"):
        raw = {key: eyeSamples(value, binding, observations,
               original["cameras"][0] if mode == "fixedOriginalCamera" else value["cameras"][0], shape)
               for key, value in variants.items()}
        usable = {key: value for key, value in raw.items() if value is not None}
        common, commonPositive = {}, {}
        for name, eye in binding["eyes"].items():
            for part in ("upper", "lower", "innerCorner", "outerCorner"):
                key = name + "." + part
                common[key] = np.logical_and.reduce([item[name]["parts"][part]["visible"] for item in usable.values()])
                commonPositive[key] = np.logical_and.reduce([item[name]["parts"][part]["positive"] for item in usable.values()])
        modeResult = {"commonSupportAcrossVariants": list(usable), "commonVisibilityMasks": common,
                      "commonPositiveDepthMasks": commonPositive, "variants": {}}
        for variantId, item in raw.items():
            if item is None:
                modeResult["variants"][variantId] = {"status": "unavailable", "reason": "missingCamera"}
                continue
            eyes = {}
            allErrors, visibleErrors, normalizedVisible = [], [], []
            for name, eye in item.items():
                parts = {}
                for part, values in eye["parts"].items():
                    key = name + "." + part
                    selected, positive = common[key], commonPositive[key]
                    parts[part] = {"sampleCount": len(selected), "positiveDepthCount": int(values["positive"].sum()),
                        "ownVisibleCount": int(values["visible"].sum()), "commonVisibleCount": int(selected.sum()),
                        "commonPositiveDepthCount": int(positive.sum()),
                        "allFixedPositiveDepthErrorPixels": stats(values["errors"][positive]),
                        "commonVisibleErrorPixels": stats(values["errors"][selected]),
                        "commonVisibleErrorOverTargetEyeWidth": stats(values["errors"][selected] / eye["targetEyeWidthPixels"]),
                        "projectedPixels": values["pixels"], "visibleMask": values["visible"],
                        "positiveDepthMask": values["positive"]}
                    if part in ("upper", "lower"):
                        allErrors.extend(values["errors"][positive])
                        visibleErrors.extend(values["errors"][selected])
                        normalizedVisible.extend(values["errors"][selected] / eye["targetEyeWidthPixels"])
                eyes[name] = {"labelId": eye["labelId"], "targetEyeWidthPixels": eye["targetEyeWidthPixels"], "parts": parts}
            modeResult["variants"][variantId] = {"status": "measured", "eyes": eyes,
                "allFixedPositiveDepthContourErrorPixels": stats(allErrors),
                "commonVisibleContourErrorPixels": stats(visibleErrors),
                "commonVisibleContourErrorOverTargetEyeWidth": stats(normalizedVisible)}
        result["modes"][mode] = modeResult
    return result


def loadVariant(record, manifestDirectory):
    directory = resolvePath(record["outputDirectory"], manifestDirectory)
    geometry = arrays(directory / "geometry.npz")
    cameras = json.loads((directory / "cameras.json").read_text())
    cameras = [{**camera, **{key: np.asarray(camera[key], float) for key in
                 ("intrinsics", "rotation", "translation") if key in camera}} for camera in cameras]
    transforms = json.loads((directory / "viewTransforms.json").read_text())
    native = [{**canonicalCamera(extrinsic, transform), "intrinsics": intrinsic}
              for extrinsic, transform, intrinsic in zip(geometry["cameraExtrinsics"], transforms,
                                                          geometry["cameraIntrinsics"])]
    if not np.isfinite(geometry["vertices"]).all():
        raise ValueError("Final vertices must be finite")
    if not len(cameras) == len(native) == len(geometry["uvMaps"]):
        raise ValueError("Variant camera and view counts differ")
    return {"directory": directory, "geometry": geometry, "cameras": cameras,
            "transforms": transforms, "native": native}


def diagnoseCase(case, manifestDirectory, eyeCacheRoot, analysisDirectory, seed):
    start = perf_counter()
    case = dict(case, sourceDirectory=resolvePath(case["sourceDirectory"], manifestDirectory))
    records = {item["id"]: item for item in case["variants"]}
    if records.get("original", {}).get("status") not in COMPLETE:
        return {"caseId": case["caseId"], "status": "unavailable", "reason": "freshOriginalNotComplete"}
    variants = {key: loadVariant(record, manifestDirectory) for key, record in records.items()
                if record.get("status") in COMPLETE}
    original = variants["original"]
    reference = np.asarray(original["geometry"]["vertices"], float)
    faceSize = float(np.linalg.norm(np.ptp(reference, axis=0)))
    if faceSize <= 0:
        raise ValueError("Degenerate original face size")
    frozenKeys = ("faces", "uvs", "uvMaps", "faceMasks")
    sourceInputs = json.loads((case["sourceDirectory"] / "inputs.json").read_text())
    for key, variant in variants.items():
        for field in frozenKeys:
            if arrayHash(variant["geometry"][field]) != arrayHash(original["geometry"][field]):
                raise ValueError(f"{key}: frozen {field} differs from original")
        inputs = json.loads((variant["directory"] / "inputs.json").read_text())
        for a, b in zip(sourceInputs, inputs):
            if a["sha256"] != b["sha256"] or not np.array_equal(a["inputToWorking"], b["inputToWorking"]):
                raise ValueError(f"{key}: photo order or inputToWorking differs from frozen source")
        if len(inputs) != len(sourceInputs):
            raise ValueError(f"{key}: input count differs from source")
    lookup, samples = uvSampleDefinitions(original["geometry"], seed)
    referencePoints = canonicalPointMaps(original)
    result = {"caseId": case["caseId"], "status": "measured", "label": case.get("label"),
              "referenceVariant": "original", "originalBboxDiagonalCanonical": faceSize,
              "frozenArraysSha256": {key: arrayHash(original["geometry"][key]) for key in frozenKeys},
              "variants": [], "contrasts": []}
    for key, variant in variants.items():
        vertices = variant["geometry"]["vertices"]
        row = {"id": key, "status": "measured", "outputDirectory": str(variant["directory"]),
            "geometryChangeFromOriginal": geometryChange(vertices, reference, faceSize),
            "fixedOriginalCameraSurfaceProjection": surfaceProjection(vertices, reference, original["cameras"]),
            "uvHeldOutConsistency": uvDiagnostics(variant, original, lookup, samples),
            "cameras": cameraDiagnostics(variant, original),
            "canonicalPointMapChange": pointMapDiagnostics(variant, original, referencePoints, faceSize),
            "fusionSupport": fusionSupport(variant, original),
            "sourceHashes": {name: fileHash(variant["directory"] / name) for name in
                ("geometry.npz", "cameras.json", "viewTransforms.json", "inputs.json", "result.json")
                if (variant["directory"] / name).exists()}}
        result["variants"].append(row)
    for key, record in records.items():
        if key not in variants:
            result["variants"].append({"id": key, "status": "unavailable", "sourceStatus": record.get("status"),
                                       "reason": record.get("error", "variantNotComplete")})
    for moving, before, interpretation in (("originalRepeat", "original", "observedRepeatNoise"),
            ("backgroundGray", "original", "environmentReplacementIncludingMaskBoundary"),
            ("headGray", "backgroundGray", "additionalShoulderNeckClothingRemovalIncludingMaskBoundary"),
            ("headBlack", "headGray", "fillColorWithSameHeadMask")):
        if moving in variants and before in variants:
            result["contrasts"].append({"moving": moving, "reference": before, "interpretation": interpretation,
                "geometryChange": geometryChange(variants[moving]["geometry"]["vertices"],
                    variants[before]["geometry"]["vertices"], faceSize),
                "fixedOriginalCameraSurfaceProjection": surfaceProjection(variants[moving]["geometry"]["vertices"],
                    variants[before]["geometry"]["vertices"], original["cameras"])})
    repeat = next((row for row in result["variants"] if row["id"] == "originalRepeat" and row["status"] == "measured"), None)
    if repeat:
        noise = repeat["geometryChangeFromOriginal"]["canonicalDisplacement"]["rms"]
        for row in result["variants"]:
            if row["status"] == "measured":
                effect = row["geometryChangeFromOriginal"]["canonicalDisplacement"]["rms"]
                row["observedRepeatComparison"] = {"repeatRms": noise, "effectRms": effect,
                    "effectOverRepeatRms": effect / noise if noise > 0 else None,
                    "exceedsObservedRepeatRms": bool(effect > noise),
                    "meaning": "single observed repeat difference; no significance threshold or p-value"}
    try:
        result["frontalEyes"] = eyeDiagnostics(variants, original, case, eyeCacheRoot, analysisDirectory / case["caseId"])
    except Exception as exc:
        result["frontalEyes"] = {"status": "unavailable", "reason": str(exc), "traceback": traceback.format_exc()}
    result["seconds"] = perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eye-cache-root", type=Path, default=ROOT / "runs/uffFrontPriority20261011")
    parser.add_argument("--case", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    batch = json.loads(manifest.read_text())
    output = args.output.resolve() if args.output else manifest.parent / "metrics.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite to recompute analysis only")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = int(batch.get("seed", batch.get("config", {}).get("seed", 2026)))
    report = {"schemaVersion": 1, "status": "running", "cases": [], "seed": seed,
        "manifest": str(manifest), "sourceManifestSha256": fileHash(manifest),
        "implementationSha256": fileHash(__file__), "notGroundTruth": True,
        "definitions": {"reference": "fresh original from this batch; historical source geometry is not used",
            "lengthNormalization": "Euclidean diagonal of fresh original final-vertex bounding box",
            "projection": "512 working pixels; K/R/t fully frozen to fresh original fitted cameras",
            "uvHeldOut": "same fixed UV lookup, five-pixel grid and seed+view split as fitViewCameras",
            "eyes": "front view only; original binding and original-photo STFR targets frozen; all completed variants share each mode's visibility intersection",
            "eyeVisibility": "positive depth, active binding vertices visible with exact mesh depth and minimumCosine=.05; mesh visibility does not establish absence of real occluders",
            "pose": "proper rotation magnitudes in degrees, not signed yaw/pitch; native poses converted to canonical head space"},
        "limits": ["Six preselected successful cases are not an independent population or generalization study",
            "All photos participated in reconstruction; held-out UV points are not held-out photos or annotated truth",
            "STFR curves and automatic A-bound eye paths are in-sample proxies, not anatomical truth",
            "Lower UV/eye residuals or confidence changes do not prove improved 3D accuracy",
            "Native and fitted camera differences compare two estimates; own PnP uses variant-dependent VGGT intrinsics",
            "Canonical point maps include each variant's own similarity fit; units are uncalibrated, not metres",
            "Similarity alignment may absorb genuine shape-related global size changes; raw and aligned changes are reported separately",
            "Fixed-camera whole-surface projections include hidden/back-facing vertices; only positive-depth eligibility is applied",
            "Frontal eye common visibility excludes samples hidden in any measured variant and reports exact masks/counts; all common-positive fixed samples are also reported",
            "No eyeballs, complete head, expression or true camera motion are evaluated",
            "A/A-repeat is one observed numerical repeat comparison, not a statistical noise distribution or significance test",
            "Mask replacement includes edge and fill-color effects; it does not isolate semantic background content alone",
            "Historical PNG quantization belongs to the input audit and cannot be interpreted as a background effect"]}
    inputManifest = manifest.parent / "inputManifest.json"
    if inputManifest.exists():
        inputAudit = json.loads(inputManifest.read_text())
        report["inputAudit"] = {"path": str(inputManifest), "sha256": fileHash(inputManifest),
                                "automaticChecks": inputAudit.get("automaticChecks", {})}
    for case in batch["cases"]:
        if args.case and case["caseId"] not in args.case:
            continue
        print(f"ANALYZE {case['caseId']}", flush=True)
        try:
            result = diagnoseCase(case, manifest.parent, args.eye_cache_root.resolve(), output.parent / "analysis", seed)
        except Exception as exc:
            result = {"caseId": case["caseId"], "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        report["cases"].append(result)
        temporary = output.with_suffix(output.suffix + ".tmp")
        writeJson(temporary, report)
        temporary.replace(output)
        print(f"ANALYZED {case['caseId']} {result['status']} eyes={result.get('frontalEyes', {}).get('status', 'unavailable')}", flush=True)
    report["status"] = "complete" if all(row["status"] == "measured" and
        all(variant["status"] == "measured" for variant in row["variants"])
        for row in report["cases"]) else "partial"
    report["analysisFailures"] = [row["caseId"] for row in report["cases"] if row["status"] == "failed"]
    report["eyeUnavailableCases"] = [row["caseId"] for row in report["cases"] if row.get("frontalEyes", {}).get("status") != "measured"]
    writeJson(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "cases": len(report["cases"]),
                      "analysisFailures": report["analysisFailures"], "eyeUnavailableCases": report["eyeUnavailableCases"]}), flush=True)


if __name__ == "__main__":
    main()
