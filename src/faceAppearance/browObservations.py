"""Build conservative, fixed correspondences for local brow geometry fitting.

Normal vectors must already use the mesh's canonical axes. They are sampled in
the working crop, while independent semantics and cameras use original pixels.
The network is a learned prior, not a measurement of ground-truth surface shape.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .camera import projectPoints
from .texture import vertexNormals


def sampleNormalObservations(vertices, faces, cameras, inputToWorking,
                             normalMaps, validMaps, semanticMaps, confidenceMaps,
                             vertexVisibility, *, minimumConfidence=.8):
    """Sample visible skin only; exclude eyebrow hair, eyes, glasses and hair.

    Returns N×V vectors/weights with frozen baseline correspondences. All four
    bilinear neighbors must have valid normal vectors. Semantic labels and
    confidence use the same nearest original pixel. No pose/axis fitting occurs.
    """
    vertices = np.asarray(vertices, dtype=float)
    viewCount, vertexCount = len(cameras), len(vertices)
    if not 0 <= minimumConfidence <= 1:
        raise ValueError("minimumConfidence must be in [0,1]")
    if any(len(items) != viewCount for items in
           (inputToWorking, normalMaps, validMaps, semanticMaps, confidenceMaps, vertexVisibility)):
        raise ValueError("Observation view counts differ")
    normals = np.zeros((viewCount, vertexCount, 3), np.float64)
    weights = np.zeros((viewCount, vertexCount), np.float64)
    samplingPixels = np.zeros((viewCount, vertexCount, 2), np.float64)
    originalPixels = np.zeros_like(samplingPixels)
    meshNormals = vertexNormals(vertices, faces)
    for viewIndex, camera in enumerate(cameras):
        imageNormals = np.asarray(normalMaps[viewIndex], dtype=float)
        imageValid = np.asarray(validMaps[viewIndex], dtype=bool)
        labels = np.asarray(semanticMaps[viewIndex])
        confidence = np.asarray(confidenceMaps[viewIndex], dtype=float)
        if (imageNormals.ndim != 3 or imageNormals.shape[2] != 3
                or imageValid.shape != imageNormals.shape[:2] or labels.ndim != 2
                or confidence.shape != labels.shape or not np.isfinite(confidence).all()
                or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError("Invalid normal or semantic maps")
        pixels, depth = projectPoints(vertices, **camera)
        matrix = np.asarray(inputToWorking[viewIndex], dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError("inputToWorking must be a finite affine pixel transform")
        working = np.c_[pixels, np.ones(vertexCount)] @ matrix.T
        working = working[:, :2]
        samplingPixels[viewIndex], originalPixels[viewIndex] = working, pixels
        height, width = labels.shape
        normalHeight, normalWidth = imageValid.shape
        valid = (np.asarray(vertexVisibility[viewIndex], dtype=bool)
                 & (depth > 0) & np.isfinite(pixels).all(1)
                 & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)
                 & (working[:, 0] >= 0) & (working[:, 0] <= normalWidth - 1)
                 & (working[:, 1] >= 0) & (working[:, 1] <= normalHeight - 1))
        indices = np.flatnonzero(valid)
        x, y = np.floor(pixels[indices] + .5).astype(int).T
        selectedConfidence = confidence[y, x]
        skin = (labels[y, x] == 1) & (selectedConfidence >= minimumConfidence)
        indices, selectedConfidence = indices[skin], selectedConfidence[skin]
        locations = working[indices][:, ::-1].T
        support = map_coordinates(imageValid.astype(float), locations, order=1, mode="constant", cval=0.)
        sampled = np.stack([map_coordinates(imageNormals[..., channel], locations,
                                           order=1, mode="constant", cval=0.) for channel in range(3)], axis=1)
        length = np.linalg.norm(sampled, axis=1)
        supported = (support >= 1 - 1e-7) & np.isfinite(sampled).all(1) & (length > 1e-6)
        indices, selectedConfidence = indices[supported], selectedConfidence[supported]
        sampled = sampled[supported] / length[supported, None]
        cameraCenter = -np.asarray(camera["translation"]) @ np.asarray(camera["rotation"])
        towardCamera = cameraCenter - vertices[indices]
        towardCamera /= np.maximum(np.linalg.norm(towardCamera, axis=1, keepdims=True), 1e-12)
        cosine = np.clip(np.sum(meshNormals[indices] * towardCamera, axis=1), 0., 1.)
        normals[viewIndex, indices] = sampled
        weights[viewIndex, indices] = selectedConfidence * cosine ** 2
    return {"normalsByView": normals, "weightsByView": weights,
            "samplingPixels": samplingPixels, "originalPixels": originalPixels,
            "diagnostics": {"qualifiedVerticesPerView": (weights > 0).sum(1).tolist(),
                "coordinates": "canonical XYZ vectors; no camera rotation applied",
                "support": "visible semantic skin label 1; nearest label/confidence; all normal interpolation neighbors valid",
                "minimumSemanticConfidence": minimumConfidence,
                "correspondences": "frozen projections of baseline geometry"}}


def fuseNormalObservations(normalsByView, weightsByView, fittingViewIndices, *,
                           minimumViews=2, maxAgreementDegrees=25., referenceViewIndex=0,
                           referenceWeight=2.):
    """Medoid consensus rejects cross-view disagreement before weighted averaging.

    The held-out view cannot influence the medoid, support, weights or targets.
    Missing normals have zero weight; missing support is never filled from mesh.
    """
    normals = np.asarray(normalsByView, dtype=float)
    weights = np.asarray(weightsByView, dtype=float)
    indices = np.asarray(fittingViewIndices, dtype=int)
    if (normals.shape != (*weights.shape, 3) or weights.ndim != 2
            or not np.isfinite(normals).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not 0 < maxAgreementDegrees < 90
            or minimumViews < 1 or not np.isfinite(referenceWeight) or referenceWeight <= 0 or indices.ndim != 1
            or len(indices) < minimumViews or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(weights))):
        raise ValueError("Invalid normal consensus observations or options")
    targets = np.zeros(normals.shape[1:], float)
    fusedWeights = np.zeros(weights.shape[1], float)
    supportCount = np.zeros(weights.shape[1], np.int64)
    accepted = np.zeros(weights.shape, bool)
    cosineThreshold = np.cos(np.radians(maxAgreementDegrees))
    selectedNormals = normals[indices]
    selectedWeights = weights[indices].copy()
    selectedWeights[indices == referenceViewIndex] *= referenceWeight
    lengths = np.linalg.norm(selectedNormals, axis=2)
    selectedWeights[lengths < 1e-6] = 0
    selectedNormals = selectedNormals / np.maximum(lengths[..., None], 1e-12)
    for vertexIndex in np.flatnonzero((selectedWeights > 0).sum(0) >= minimumViews):
        available = np.flatnonzero(selectedWeights[:, vertexIndex] > 0)
        candidates = selectedNormals[available, vertexIndex]
        candidateWeights = selectedWeights[available, vertexIndex]
        dot = np.clip(candidates @ candidates.T, -1., 1.)
        medoid = np.argmin((np.arccos(dot) * candidateWeights[None]).sum(1))
        inliers = dot[medoid] >= cosineThreshold
        if inliers.sum() < minimumViews:
            continue
        mean = (candidates[inliers] * candidateWeights[inliers, None]).sum(0)
        mean /= np.linalg.norm(mean)
        # Require every retained observation to remain inside the consensus
        # radius after averaging. This is not a pairwise-angle bound.
        if np.any(candidates[inliers] @ mean < cosineThreshold):
            continue
        targets[vertexIndex] = mean
        fusedWeights[vertexIndex] = candidateWeights[inliers].sum() / len(indices)
        supportCount[vertexIndex] = inliers.sum()
        accepted[indices[available[inliers]], vertexIndex] = True
    return {"normalTargets": targets, "normalWeights": fusedWeights,
            "normalSupportCount": supportCount, "acceptedNormalViews": accepted,
            "diagnostics": {"fittingViewIndices": indices.tolist(), "minimumViews": minimumViews,
                "maxAgreementDegrees": maxAgreementDegrees, "referenceViewIndex": referenceViewIndex,
                "agreementMeaning": "angular radius around medoid and final mean; not a pairwise bound",
                "referenceWeight": referenceWeight, "supportedVertexCount": int((fusedWeights > 0).sum()),
                "heldOutIncluded": False}}


def fuseDepthObservations(vertices, directions, partialVertices, partialValid,
                          fittingViewIndices, *, minimumViews=2,
                          maxScatterFraction=.015, maxLateralFraction=.035,
                          maxOffsetFraction=.04):
    """Fuse canonical VGGT partial positions into bounded signed ray offsets.

    Partial positions must be real valid UV samples. This is reused VGGT evidence
    already seen by UFF, not an independent depth sensor or a confidence score.
    """
    vertices, directions, partials = [np.asarray(value, dtype=float) for value in
                                     (vertices, directions, partialVertices)]
    valid = np.asarray(partialValid, dtype=bool).copy()
    indices = np.asarray(fittingViewIndices, dtype=int)
    if (vertices.shape != directions.shape or partials.shape != (*valid.shape, 3)
            or partials.shape[1:] != vertices.shape or not np.isfinite(vertices).all()
            or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions, axis=1), 1.)
            or len(indices) < minimumViews or minimumViews < 1 or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(partials))
            or not np.isfinite([maxScatterFraction, maxLateralFraction, maxOffsetFraction]).all()
            or min(maxScatterFraction, maxLateralFraction, maxOffsetFraction) <= 0):
        raise ValueError("Invalid depth observations")
    extent = float(np.ptp(vertices, axis=0).max())
    if extent <= 0:
        raise ValueError("Degenerate geometry")
    difference = partials - vertices[None]
    offsets = np.sum(difference * directions[None], axis=2)
    lateral = np.linalg.norm(difference - offsets[..., None] * directions[None], axis=2)
    valid &= (np.isfinite(partials).all(2) & (lateral <= maxLateralFraction * extent)
              & (np.abs(offsets) <= maxOffsetFraction * extent))
    excluded = np.ones(len(partials), bool)
    excluded[indices] = False
    valid[excluded] = False
    targets, weights, scatter = [np.zeros(len(vertices), float) for _ in range(3)]
    count = valid.sum(0)
    for vertexIndex in np.flatnonzero(count >= minimumViews):
        values = offsets[valid[:, vertexIndex], vertexIndex]
        center = np.median(values)
        spread = np.max(np.abs(values - center))
        scatter[vertexIndex] = spread
        if spread > maxScatterFraction * extent:
            continue
        targets[vertexIndex] = center
        weights[vertexIndex] = (len(values) / len(indices)) / (1 + (spread / (.5 * maxScatterFraction * extent)) ** 2)
    return {"depthTargets": targets, "depthWeights": weights, "depthScatter": scatter,
            "depthSupportCount": count, "depthOffsetsByView": offsets,
            "depthQualifiedViews": valid,
            "diagnostics": {"fittingViewIndices": indices.tolist(), "supportedVertexCount": int((weights > 0).sum()),
                "minimumViews": minimumViews, "maxScatterFraction": maxScatterFraction,
                "maxLateralFraction": maxLateralFraction, "maxOffsetFraction": maxOffsetFraction,
                "units": "signed canonical distance along unit reference camera ray",
                "evidence": "same VGGT partial observations used by UFF initialization; not independent truth"}}
