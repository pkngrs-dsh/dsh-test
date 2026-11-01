"""Project library-independent semantic maps through saved photo coordinates.

Categorical labels and their confidence are sampled from the same nearest pixel;
interpolating label IDs would create nonexistent semantic correspondences.
"""
from __future__ import annotations

import numpy as np

from .camera import projectPoints
from .texture import rasterizeDepth, vertexNormals


def visibleVertices(vertices, faces, cameras, imageShapes, *, minimumCosine=.15,
                    depthToleranceFraction=.005, visibilityMode="raster") -> np.ndarray:
    """Conservative initial vertex visibility for fixed semantic attachments.

    Cameras are OpenCV integer-pixel centered in the same original-image frame
    as imageShapes. This mask is an initial gate, not dynamic visibility during
    deformation; use it only with bounded small refinements. ``exact`` checks
    continuous projected rays against the nearest mesh surface, avoiding an
    eye-rim vertex rounding into a neighboring empty depth-buffer pixel.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if len(cameras) != len(imageShapes) or not 0 <= minimumCosine < 1 or depthToleranceFraction <= 0:
        raise ValueError("Invalid visibility inputs")
    if visibilityMode not in ("raster", "exact"):
        raise ValueError("visibilityMode must be raster or exact")
    normals = vertexNormals(vertices, faces)
    tolerance = max(float(np.ptp(vertices, axis=0).max()) * depthToleranceFraction, 1e-8)
    result = np.zeros((len(cameras), len(vertices)), bool)
    for index, (camera, (height, width)) in enumerate(zip(cameras, imageShapes)):
        rotation, translation = np.asarray(camera["rotation"]), np.asarray(camera["translation"])
        intrinsics = np.asarray(camera["intrinsics"])
        points, z = projectPoints(vertices, intrinsics, rotation, translation)
        cameraVertices = vertices @ rotation.T + translation
        directions = -cameraVertices / np.maximum(np.linalg.norm(cameraVertices, axis=-1, keepdims=True), 1e-12)
        cosine = np.sum((normals @ rotation.T) * directions, axis=-1)
        valid = ((z > 0) & (cosine > minimumCosine) & np.isfinite(points).all(axis=-1)
                 & (points[:, 0] >= 0) & (points[:, 0] <= width-1)
                 & (points[:, 1] >= 0) & (points[:, 1] <= height-1))
        indices = np.flatnonzero(valid)
        if visibilityMode == "exact":
            from .surfaceVisibility import ProjectedSurfaceVisibility
            sampler = ProjectedSurfaceVisibility(cameraVertices, faces, intrinsics, (height, width))
            sampled = sampler.closestDepth(cameraVertices[indices])["depth"]
        else:
            depth = rasterizeDepth(cameraVertices, faces, intrinsics, height, width)
            pixels = np.floor(points[indices] + .5).astype(int)
            sampled = depth[pixels[:, 1], pixels[:, 0]]
        result[index, indices] = np.isfinite(sampled) & (np.abs(sampled-z[indices]) <= tolerance)
    return result


def sampleSemanticObservations(labelMaps, confidenceMaps, samplingPixels: np.ndarray,
                               observationWeights: np.ndarray) -> dict:
    """Return NHW labels/confidence without expanding qualified photo support.

    Coordinates are integer-centered, EXIF-oriented original-image pixels. A
    point outside the image is invalid even if rounding would land inside it.
    Unknown/unobserved values are background with zero confidence.
    """
    points = np.asarray(samplingPixels)
    weights = np.asarray(observationWeights)
    if (weights.ndim != 3 or points.shape != (*weights.shape, 2)
            or len(labelMaps) != len(weights) or len(confidenceMaps) != len(weights)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("Expected finite NHW2 photo coordinates and nonnegative NHW weights")
    labels = np.zeros(weights.shape, np.uint8)
    confidence = np.zeros(weights.shape, np.float32)
    for index, (labelMap, confidenceMap) in enumerate(zip(labelMaps, confidenceMaps)):
        labelMap, confidenceMap = np.asarray(labelMap), np.asarray(confidenceMap)
        if (labelMap.ndim != 2 or confidenceMap.shape != labelMap.shape
                or not np.issubdtype(labelMap.dtype, np.integer)
                or np.any(labelMap < 0) or np.any(labelMap > 18)
                or not np.isfinite(confidenceMap).all()
                or np.any(confidenceMap < 0) or np.any(confidenceMap > 1)):
            raise ValueError("Expected standard 19-label HW maps and confidence in [0, 1]")
        height, width = labelMap.shape
        x, y = points[index, ..., 0], points[index, ..., 1]
        valid = ((weights[index] > 0) & (x >= 0) & (y >= 0)
                 & (x <= width - 1) & (y <= height - 1))
        xs, ys = np.floor(x[valid] + .5).astype(int), np.floor(y[valid] + .5).astype(int)
        labels[index][valid] = labelMap[ys, xs]
        confidence[index][valid] = confidenceMap[ys, xs]
    return {"semanticLabels": labels, "semanticConfidence": confidence,
            "diagnostics": {"sampling": "sameNearestOriginalPixelForLabelAndConfidence",
                            "support": "subsetOfQualifiedPhotoObservations",
                            "observedTexelsPerView": (confidence > 0).sum(axis=(1, 2)).tolist()}}
