"""Common, visibility-tested UV observations for controlled delighting experiments.

Raw photographs are sampled at their original EXIF-oriented resolution through
the saved crop affine. Geometry, PnP cameras and dense UV correspondences remain
fixed. All fitting/fusion uses linear RGB; gamma-2.2 is only an output contract
for the existing WildCap adapter and diagnostic renderer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from .imageColor import readSrgbImage

from .surfaceSampling import invertUvMap
from .texture import (buildUvLookup, interpolateSurface, rasterizeDepth, samplePixels,
                      vertexNormals, weightedColorMedian)


def srgbToLinear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected finite sRGB values in [0, 1]")
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linearToGamma22(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Linear colors must be finite")
    return np.clip(values, 0, 1) ** (1 / 2.2)


def workingToOriginal(points: np.ndarray, metadata: dict) -> np.ndarray:
    """Invert the saved integer-pixel-center transform, including half pixels."""
    points = np.asarray(points, np.float64)
    matrix = np.asarray(metadata["inputToWorking"], np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 working image positions")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Expected an invertible inputToWorking affine")
    projected = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(matrix).T
    return projected[:, :2] / projected[:, 2:3]


def readOriginalLinear(metadata: dict) -> np.ndarray:
    path = Path(metadata["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Source photograph changed: {path.name}")
    image, _ = readSrgbImage(path)
    with image:
        if list(image.size) != metadata["orientedSize"]:
            raise ValueError("Original photo size differs from the reconstruction metadata")
        return srgbToLinear(np.asarray(image, np.float32) / 255)


def collectAppearanceObservations(geometry: dict, cameras: list[dict], inputs: list[dict],
                                 delight: dict, resolution: int = 1024, *,
                                 samplingMode: str = "denseUv", visibilityMode: str = "raster") -> dict:
    """Sample raw and predicted albedo at identical, qualified UV correspondences.

    The OpenDelight valid region is shared by all branches for the controlled
    comparison. Normal directions stay in the UFF canonical object frame.
    """
    if samplingMode not in {"denseUv", "projection"}:
        raise ValueError("samplingMode must be denseUv or projection")
    if visibilityMode not in {"raster", "refineRejected"}:
        raise ValueError("visibilityMode must be raster or refineRejected")
    vertices = np.asarray(geometry["vertices"], np.float64)
    faces = np.asarray(geometry["faces"], np.int64)
    lookup = buildUvLookup(geometry["uvs"], faces, resolution)
    atlasMask = lookup["validMask"]
    surface = interpolateSurface(vertices, faces, lookup)[atlasMask].astype(np.float64)
    normalAtlas = interpolateSurface(vertexNormals(vertices, faces), faces, lookup)
    normalAtlas /= np.maximum(np.linalg.norm(normalAtlas, axis=-1, keepdims=True), 1e-12)
    normals = normalAtlas[atlasMask]
    masks = np.asarray(delight["validMasks"], bool)
    albedoImages = np.asarray(delight["albedoImages"], np.float32)
    count = len(cameras)
    if (len(inputs) != count or albedoImages.shape != (*masks.shape, 3)
            or len(masks) != count
            or (samplingMode == "denseUv" and np.shape(geometry.get("uvMaps")) != (*masks.shape, 2))):
        raise ValueError("Saved observations, cameras and masks must have matching view counts/shapes")
    yy, xx = np.nonzero(atlasMask)
    queryUvs = np.column_stack([xx, yy]) / (resolution - 1)
    raw = np.zeros((count, resolution, resolution, 3), np.float32)
    predicted = np.zeros_like(raw)
    weights = np.zeros(raw.shape[:3], np.float32)
    viewingCosines = np.zeros_like(weights)
    samplingPixels = np.full((*weights.shape, 2), -1, np.float32)
    meshExtent = float(np.ptp(vertices, axis=0).max())
    tolerance = max(meshExtent * 0.005, 1e-7)
    diagnostics = []
    for index, camera in enumerate(cameras):
        if not camera.get("valid", False):
            raise ValueError(f"Saved camera {index} is invalid; this experiment never refits cameras")
        height, width = masks[index].shape
        rotation = np.asarray(camera["rotation"])
        translation = np.asarray(camera["translation"])
        intrinsics = np.asarray(camera["intrinsics"])
        cameraSurface = surface @ rotation.T + translation
        projected = cameraSurface @ intrinsics.T
        points = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
        viewDirections = -cameraSurface / np.maximum(np.linalg.norm(cameraSurface, axis=-1, keepdims=True), 1e-12)
        cosine = np.maximum(np.sum((normals @ rotation.T) * viewDirections, axis=-1), 0)
        candidate = ((cameraSurface[:, 2] > 1e-8) & (cosine > 0.1)
                     & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                     & (points[:, 1] >= 0) & (points[:, 1] <= height - 1))
        indices = np.flatnonzero(candidate)
        if not len(indices):
            raise ValueError(f"View {index} has no front-facing projected surface")
        depth = rasterizeDepth(vertices @ rotation.T + translation, faces, intrinsics, height, width)
        nearest = np.rint(points[indices]).astype(int)
        sampledDepth = depth[nearest[:, 1], nearest[:, 0]]
        depthValid = (np.isfinite(sampledDepth)
                      & (np.abs(sampledDepth - cameraSurface[indices, 2]) <= tolerance))
        visibilityDiagnostics = {"mode": "raster", "recoveredQueries": 0}
        if visibilityMode == "refineRejected":
            from .surfaceVisibility import refineRejectedVisibility
            refinedVisibility = refineRejectedVisibility(
                vertices @ rotation.T + translation, faces, intrinsics,
                cameraSurface[indices], depthValid, imageShape=(height, width),
                depthTolerance=tolerance, candidateMask=np.ones(len(indices), bool))
            depthValid = refinedVisibility["visible"]
            visibilityDiagnostics = refinedVisibility["diagnostics"]
        if samplingMode == "denseUv":
            pixels, semantic = invertUvMap(geometry["uvMaps"][index], masks[index], queryUvs[indices], points[indices])
        else:
            pixels, semantic = points[indices], np.ones(len(indices), bool)
        originalPixels = workingToOriginal(pixels, inputs[index])
        original = readOriginalLinear(inputs[index])
        sourceHeight, sourceWidth = original.shape[:2]
        originalInside = ((originalPixels[:, 0] >= 0) & (originalPixels[:, 0] <= sourceWidth - 1)
                          & (originalPixels[:, 1] >= 0) & (originalPixels[:, 1] <= sourceHeight - 1))
        keep = depthValid & semantic & originalInside & (samplePixels(masks[index].astype(np.float32), pixels) >= 0.999)
        retained = indices[keep]
        rawColors = samplePixels(original, originalPixels[keep])
        # OpenDelight native encoding is not calibrated; keep the same explicit
        # gamma-2.2 interpretation as the previous WildCap experiments.
        predictedColors = samplePixels(albedoImages[index] ** 2.2, pixels[keep])
        y, x = yy[retained], xx[retained]
        raw[index, y, x] = rawColors
        predicted[index, y, x] = predictedColors
        quality = 1 / (1 + (float(camera["medianErrorPixels"]) / 4) ** 2)
        weights[index, y, x] = cosine[retained] ** 4 * quality
        viewingCosines[index, y, x] = np.clip(cosine[retained], 0, 1)
        samplingPixels[index, y, x] = originalPixels[keep]
        diagnostics.append({"viewIndex": index, "observedTexels": int(len(retained)),
                            "originalSizeWh": [sourceWidth, sourceHeight], "depthTolerance": tolerance,
                            "visibility": visibilityDiagnostics,
                            "medianDenseCorrectionWorkingPixels": float(np.median(np.linalg.norm(
                                pixels[keep] - points[retained], axis=-1))),
                            "photoSha256": inputs[index]["sha256"]})
    if not weights.any():
        raise ValueError("No common raw/OpenDelight observations survived visibility checks")
    return {"rawLinear": raw, "openDelightLinear": predicted, "weights": weights,
            "viewingCosines": viewingCosines,
            "normals": normalAtlas, "atlasMask": atlasMask, "originalSamplingPixels": samplingPixels,
            "diagnostics": {"views": diagnostics, "rawEncoding": "sRGB decoded before original-resolution sampling",
                            "openDelightEncoding": "native assumed gamma2.2; not color calibrated",
                            "samplingMode": samplingMode,
                            "visibilityMode": visibilityMode,
                            "correspondences": ("saved dense UV constrained by frozen PnP projection/depth"
                                                if samplingMode == "denseUv" else "frozen PnP geometry projection/depth; no dense UV correction"),
                            "sharedObservationRegion": "saved OpenDelight validMask and original photo support"}}


def fuseLinearObservations(observations: np.ndarray, weights: np.ndarray,
                           residualScale: float = 0.04) -> dict:
    """Identical robust linear-RGB fusion for all experiment branches."""
    values, weights = np.asarray(observations, np.float32), np.asarray(weights, np.float32)
    if (values.ndim != 4 or values.shape[-1] != 3 or weights.shape != values.shape[:3]
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(values < 0) or np.any(weights < 0) or residualScale <= 0):
        raise ValueError("Expected nonnegative finite NHW3 linear colors and NHW weights")
    count, height, width = weights.shape
    flat = values.reshape(count, -1, 3)
    flatWeights = weights.reshape(count, -1)
    median = weightedColorMedian(flat, flatWeights)
    residual = np.sqrt(np.mean((flat - median[None]) ** 2, axis=-1))
    robust = flatWeights / (1 + (residual / residualScale) ** 2)
    total = robust.sum(axis=0)
    valid = total > 1e-8
    fused = np.divide((flat * robust[..., None]).sum(axis=0), total[:, None],
                      out=np.zeros_like(median), where=valid[:, None])
    source = np.argmax(robust, axis=0).astype(np.int16)
    source[~valid] = -1
    clipped = np.any(fused > 1, axis=-1) & valid
    return {"linearAlbedo": np.clip(fused.reshape(height, width, 3), 0, 1),
            "coverageMask": valid.reshape(height, width), "sourceIndex": source.reshape(height, width),
            "confidence": np.clip(flatWeights.max(axis=0).reshape(height, width), 0, 1),
            "diagnostics": {"fusion": "linearRgbRobustWeightedAverage", "residualScale": residualScale,
                            "observedTexels": int(valid.sum()), "clippedTexels": int(clipped.sum()),
                            "colorBalance": "disabledForAllBranches; lighting fit includes RGB scale"}}
