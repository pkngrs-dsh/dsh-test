"""Compare saved cameras against shared learned UV correspondences, without fitting.

These are consistency diagnostics, not ground truth pose measurements. Both
camera lists must project the final canonical geometry into the working images
at OpenCV integer pixel centers (before a renderer's half-pixel adjustment).
"""

from __future__ import annotations

import numpy as np

from .camera import projectPoints
from .texture import buildUvLookup, interpolateSurface


def _cameraMatrices(camera: dict) -> tuple | None:
    """An early failed PnP result can legitimately contain no saved pose."""
    try:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        translation = np.asarray(camera["translation"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if (intrinsics.shape != (3, 3) or rotation.shape != (3, 3)
            or translation.size != 3
            or not all(np.isfinite(value).all() for value in (intrinsics, rotation, translation))):
        return None
    return intrinsics, rotation, translation.reshape(3)


def _errorSummary(errors: np.ndarray) -> tuple[float | None, float | None]:
    if not len(errors):
        return None, None
    return float(np.median(errors)), float(np.percentile(errors, 90))


def _cameraSummary(camera: dict, matrices: tuple | None, points: np.ndarray,
                   pixels: np.ndarray, heldOut: np.ndarray, fallbackSource: str) -> dict:
    result = {
        "source": str(camera.get("source", fallbackSource)),
        "savedValid": bool(camera["valid"]) if "valid" in camera else None,
        "status": "unavailable",
        "allMedianErrorPixels": None,
        "allP90ErrorPixels": None,
        "heldOutMedianErrorPixels": None,
        "heldOutP90ErrorPixels": None,
        "positiveDepthFraction": None,
        "heldOutPositiveDepthFraction": None,
        "evaluatedPointCount": 0,
        "heldOutEvaluatedPointCount": 0,
    }
    if matrices is None:
        result["reason"] = "Saved camera matrices are missing, nonfinite, or malformed"
        return result
    if not len(points):
        result["status"] = "noCorrespondences"
        return result
    projected, depth = projectPoints(points, *matrices)
    positiveDepth = depth > 0
    usable = positiveDepth & np.isfinite(projected).all(axis=1) & (depth > 1e-9)
    errors = np.linalg.norm(projected - pixels, axis=1)
    usable &= np.isfinite(errors)
    heldOutUsable = heldOut[usable[heldOut]]
    result.update({
        "status": "measured" if usable.any() else "noPositiveDepthCorrespondences",
        "positiveDepthFraction": float(positiveDepth.mean()),
        "heldOutPositiveDepthFraction": float(positiveDepth[heldOut].mean()) if len(heldOut) else None,
        "evaluatedPointCount": int(usable.sum()),
        "heldOutEvaluatedPointCount": int(len(heldOutUsable)),
    })
    result["allMedianErrorPixels"], result["allP90ErrorPixels"] = _errorSummary(errors[usable])
    result["heldOutMedianErrorPixels"], result["heldOutP90ErrorPixels"] = _errorSummary(errors[heldOutUsable])
    return result


def _vertexComparison(vertices: np.ndarray, nativeMatrices: tuple | None,
                      fittedMatrices: tuple | None) -> dict:
    result = {"vertexCount": int(len(vertices)), "bothPositiveDepthVertexCount": 0,
              "medianDisplacementPixels": None, "p90DisplacementPixels": None,
              "maximumDisplacementPixels": None, "maximumDisplacementVertexIndex": None}
    if nativeMatrices is None or fittedMatrices is None:
        result["status"] = "unavailable"
        return result
    nativePixels, nativeDepth = projectPoints(vertices, *nativeMatrices)
    fittedPixels, fittedDepth = projectPoints(vertices, *fittedMatrices)
    usable = ((nativeDepth > 1e-9) & (fittedDepth > 1e-9)
              & np.isfinite(nativePixels).all(axis=1) & np.isfinite(fittedPixels).all(axis=1))
    displacements = np.linalg.norm(nativePixels - fittedPixels, axis=1)
    usable &= np.isfinite(displacements)
    result["bothPositiveDepthVertexCount"] = int(usable.sum())
    result["status"] = "measured" if usable.any() else "noSharedPositiveDepthVertices"
    if usable.any():
        result["medianDisplacementPixels"], result["p90DisplacementPixels"] = _errorSummary(displacements[usable])
        maximumIndex = int(np.flatnonzero(usable)[np.argmax(displacements[usable])])
        result["maximumDisplacementPixels"] = float(displacements[maximumIndex])
        result["maximumDisplacementVertexIndex"] = maximumIndex
    return result


def compareSavedCameras(geometry: dict, nativeCameras: list[dict],
                        fittedCameras: list[dict], seed: int = 2026) -> list[dict]:
    """Measure native and fitted cameras against exactly ``fitViewCameras`` samples.

    Correspondences use the same 512-pixel UV lookup, rounded UV coordinates,
    five-pixel image grid and mask threshold. For >=60 correspondences the
    shared held-out subset matches ``fitCamera``'s ``seed + viewIndex`` split;
    below its fitting threshold there is no original held-out subset.

    Error percentiles include finite projections with depth >1e-9 only, with
    explicit support counts and positive-depth fractions. Missing/failed saved
    poses remain unavailable; a saved ``valid=False`` pose with matrices can
    still be measured and retains that validity flag. No solver is invoked.
    Vertices are compared without visibility or occlusion filtering.
    """
    uvMaps = np.asarray(geometry["uvMaps"])
    masks = np.asarray(geometry["faceMasks"])
    if uvMaps.ndim != 4 or uvMaps.shape[-1] != 2 or masks.shape != uvMaps.shape[:3]:
        raise ValueError("Expected uvMaps (N,H,W,2) and matching faceMasks (N,H,W)")
    if len(nativeCameras) != len(uvMaps) or len(fittedCameras) != len(uvMaps):
        raise ValueError("Native and fitted camera counts must match geometry views")
    vertices = np.asarray(geometry["vertices"], dtype=np.float64)
    lookup = buildUvLookup(geometry["uvs"], geometry["faces"], 512)
    surface = interpolateSurface(vertices, geometry["faces"], lookup)
    comparisons = []
    for viewIndex, uvMap in enumerate(uvMaps):
        imageHeight, imageWidth = uvMap.shape[:2]
        yy, xx = np.mgrid[0:imageHeight:5, 0:imageWidth:5]
        predictedUv = uvMap[yy, xx]
        valid = masks[viewIndex, yy, xx] > 0.5
        valid &= np.isfinite(predictedUv).all(axis=-1) & (predictedUv >= 0).all(axis=-1)
        valid &= (predictedUv <= 1).all(axis=-1)
        safeUv = np.nan_to_num(predictedUv, nan=0.0, posinf=0.0, neginf=0.0)
        uvPixels = np.rint(np.clip(safeUv, 0, 1) * 511).astype(np.int32)
        valid &= lookup["validMask"][uvPixels[..., 1], uvPixels[..., 0]]
        points = surface[uvPixels[..., 1], uvPixels[..., 0]][valid].astype(np.float64)
        pixels = np.stack([xx, yy], axis=-1)[valid]
        pointCount = len(points)
        order = np.random.default_rng(seed + viewIndex).permutation(pointCount)
        heldOut = order[:max(12, pointCount // 5)] if pointCount >= 60 else np.empty(0, dtype=np.int64)
        native, fitted = nativeCameras[viewIndex], fittedCameras[viewIndex]
        nativeMatrices, fittedMatrices = _cameraMatrices(native), _cameraMatrices(fitted)
        comparisons.append({
            "viewIndex": viewIndex,
            "notGroundTruth": True,
            "reference": "learnedUvCorrespondencesOnFinalCanonicalMesh",
            "pixelConvention": "workingImageOpenCvIntegerCenters",
            "errorSupport": "finiteProjectionWithCameraDepthGreaterThan1e-9",
            "heldOutMeaning": "sameIndicesAsSavedPnPWhenItsFitSeedMatches; notIndependentGroundTruth",
            "seed": int(seed + viewIndex),
            "pointCount": pointCount,
            "heldOutPointCount": int(len(heldOut)),
            "native": _cameraSummary(native, nativeMatrices, points, pixels, heldOut, "savedNativeCamera"),
            "fitted": _cameraSummary(fitted, fittedMatrices, points, pixels, heldOut, "savedFittedCamera"),
            "nativeVsFittedVertices": _vertexComparison(vertices, nativeMatrices, fittedMatrices),
        })
    return comparisons
