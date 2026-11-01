"""Exact closest-triangle correspondence for pre-registered template surfaces.

This module does not register meshes or prove anatomical correspondence. A
conservative sphere bound finds all candidate triangles before exact projection.
It avoids an optional spatial-index binary dependency and runs entirely on CPU.
"""

from __future__ import annotations

import hashlib
import numpy as np
from scipy.spatial import cKDTree


def arrayHash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class SurfaceIndex:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        from .texture import validateTriangles

        self.vertices, self.faces = validateTriangles(vertices, faces, 3)
        self.triangles = self.vertices[self.faces]
        self.centroids = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(self.triangles - self.centroids[:, None], axis=-1).max(axis=1)
        self.centroidTree = cKDTree(self.centroids)
        self.vertexTree = cKDTree(self.vertices[np.unique(self.faces)])
        cross = np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                         self.triangles[:, 2] - self.triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        if (lengths <= 1e-14).any():
            raise ValueError("Source surface contains degenerate triangles")
        self.faceNormals = cross / lengths[:, None]

    def query(self, points: np.ndarray, chunkSize: int = 512) -> dict:
        """Return exact closest points, triangle IDs, barycentrics and distances."""
        from trimesh.triangles import closest_point, points_to_barycentric

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite query points shaped (N,3)")
        if chunkSize < 1:
            raise ValueError("chunkSize must be positive")
        triangleIds = np.empty(len(points), dtype=np.int32)
        closestPoints = np.empty_like(points)
        distances = np.empty(len(points))
        maxRadius = self.radii.max()
        for start in range(0, len(points), chunkSize):
            batch = points[start:start + chunkSize]
            # A nearest vertex is on the source surface, so its distance is a
            # valid upper bound on distance to the nearest triangle.
            upperBounds = self.vertexTree.query(batch)[0]
            candidates = self.centroidTree.query_ball_point(batch, upperBounds + maxRadius + 1e-10)
            counts = np.fromiter((len(ids) for ids in candidates), dtype=np.int64)
            queryIds = np.repeat(np.arange(len(batch)), counts)
            faceIds = np.concatenate(candidates).astype(np.int64)
            valid = (np.linalg.norm(self.centroids[faceIds] - batch[queryIds], axis=1)
                     <= upperBounds[queryIds] + self.radii[faceIds] + 1e-10)
            queryIds, faceIds = queryIds[valid], faceIds[valid]
            projected = closest_point(self.triangles[faceIds], batch[queryIds])
            squared = np.sum((projected - batch[queryIds]) ** 2, axis=1)
            # Stable order makes shared-edge tie handling repeatable.
            order = np.lexsort((faceIds, squared, queryIds))
            sortedQueries = queryIds[order]
            best = order[np.r_[True, sortedQueries[1:] != sortedQueries[:-1]]]
            if len(best) != len(batch):
                raise RuntimeError("Spatial query failed to cover every requested point")
            triangleIds[start:start + len(batch)] = faceIds[best]
            closestPoints[start:start + len(batch)] = projected[best]
            distances[start:start + len(batch)] = np.sqrt(squared[best])
        barycentrics = points_to_barycentric(self.triangles[triangleIds], closestPoints)
        barycentrics = np.clip(barycentrics, 0, 1)
        barycentrics /= barycentrics.sum(axis=1, keepdims=True)
        return {"triangleIds": triangleIds, "barycentricWeights": barycentrics.astype(np.float32),
                "closestPoints": closestPoints.astype(np.float32), "distances": distances.astype(np.float32)}


def sampleSurface(values: np.ndarray, sourceFaces: np.ndarray, mapping: dict) -> np.ndarray:
    """Transfer vertex values without conflating the two directions of a map."""
    triangleIds = np.asarray(mapping["triangleIds"])
    weights = np.asarray(mapping["barycentricWeights"])
    values, sourceFaces = np.asarray(values), np.asarray(sourceFaces)
    if triangleIds.ndim != 1 or weights.shape != (len(triangleIds), 3):
        raise ValueError("Mapping has incompatible triangle/weight dimensions")
    if ((triangleIds < 0).any() or (triangleIds >= len(sourceFaces)).any()
            or not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1, atol=1e-5)
            or (weights < -1e-6).any()):
        raise ValueError("Mapping has invalid triangle IDs or barycentric weights")
    corners = values[sourceFaces[triangleIds]]
    expandedWeights = weights.reshape((*weights.shape, *((1,) * (corners.ndim - 2))))
    return (corners * expandedWeights).sum(axis=1)


def qualifyMapping(mapping: dict, source: SurfaceIndex, queryNormals: np.ndarray,
                   maxDistance: float, minNormalDot: float = 0.5) -> dict:
    queryNormals = np.asarray(queryNormals, dtype=np.float64)
    lengths = np.linalg.norm(queryNormals, axis=1)
    normalized = queryNormals / np.maximum(lengths[:, None], 1e-12)
    normalDots = np.sum(source.faceNormals[mapping["triangleIds"]] * normalized, axis=1)
    valid = (mapping["distances"] <= maxDistance) & (normalDots >= minNormalDot) & (lengths > 1e-10)
    return {**mapping, "normalDots": normalDots.astype(np.float32), "validMask": valid}
