"""Invert dense semantic UV predictions while rejecting holes and folded matches."""

import numpy as np
from scipy.spatial import cKDTree


def invertUvMap(uvMap: np.ndarray, validMask: np.ndarray, queryUvs: np.ndarray,
                projectedPixels: np.ndarray, maxCorrectionPixels: float = 24.0):
    """Map atlas samples to image pixels, bounded by the fitted geometry camera.

    Nearest semantic neighbors remove residual alignment errors from a coarse
    mesh. Widely separated image neighbors, UV holes, and corrections exceeding
    the geometry bound are rejected, rather than filled with arbitrary colors.
    """
    uvMap = np.asarray(uvMap)
    if uvMap.shape != (*validMask.shape, 2) or queryUvs.shape != projectedPixels.shape:
        raise ValueError("Dense UV map, mask and query coordinates have incompatible shapes")
    yy, xx = np.nonzero((validMask > 0.999) & np.isfinite(uvMap).all(axis=-1)
                        & (uvMap >= 0).all(axis=-1) & (uvMap <= 1).all(axis=-1))
    if len(xx) < 4:
        return np.zeros_like(queryUvs), np.zeros(len(queryUvs), dtype=bool)
    imagePixels = np.stack([xx, yy], axis=-1).astype(np.float64)
    tree = cKDTree(uvMap[yy, xx].astype(np.float64))
    distances, indices = tree.query(queryUvs, k=4, workers=1)
    neighbors = imagePixels[indices]
    weights = 1 / np.maximum(distances, 1e-8) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    correctedPixels = np.sum(neighbors * weights[..., None], axis=1)
    neighborSpread = np.max(np.linalg.norm(neighbors - correctedPixels[:, None], axis=-1), axis=1)
    correction = np.linalg.norm(correctedPixels - projectedPixels, axis=1)
    accepted = ((distances[:, 0] <= 2.0 / 511) & (neighborSpread <= 8)
                & (correction <= maxCorrectionPixels))
    return correctedPixels, accepted
