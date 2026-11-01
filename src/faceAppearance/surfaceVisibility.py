"""Exact continuous photo-depth checks for rejected surface observations.

The initial raster gate remains the fast path. Only explicitly qualified points
that it rejected are reconsidered. A projected triangle bin index bounds CPU
work and memory; no model, renderer, torch, or optional dependency is loaded.
Both face windings occlude, matching the existing CPU depth rasterizer.
"""
from __future__ import annotations

import numpy as np


class ProjectedSurfaceVisibility:
    """Immutable per-camera triangle index with integer-centered photo pixels.

    Triangles with any vertex at/beyond the near-depth limit are omitted, as in
    ``texture.rasterizeDepth``. Perspective interpolation tests true query XY,
    instead of rounding query XY to a neighboring depth-buffer pixel. Degenerate
    projected triangles have no area and are ignored. Boundary containment uses
    the same 1e-7 barycentric tolerance as the existing CPU triangle rasterizer.
    """

    def __init__(self, cameraVertices, faces, intrinsics, imageShape, *, binSize=16,
                 nearDepth=1e-8):
        vertices, faces = np.asarray(cameraVertices, dtype=np.float64), np.asarray(faces)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                or not len(faces) or faces.dtype.kind not in 'iu'
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Expected finite camera-space Vx3 vertices and valid integer Fx3 faces')
        if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-12)
                or abs(np.linalg.det(intrinsics)) <= 1e-12):
            raise ValueError('Expected finite nonsingular OpenCV pinhole intrinsics')
        if (len(imageShape) != 2
                or any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1 for value in imageShape)
                or not isinstance(binSize, (int, np.integer)) or isinstance(binSize, bool) or binSize < 1
                or not np.isfinite(nearDepth) or nearDepth <= 0):
            raise ValueError('Invalid image shape, bin size or near-depth limit')
        self.imageShape = tuple(int(value) for value in imageShape)
        self.binSize, self.nearDepth = int(binSize), float(nearDepth)
        self.intrinsics = intrinsics.copy()
        height, width = self.imageShape
        self.binColumns = (width+self.binSize-1)//self.binSize
        self.binRows = (height+self.binSize-1)//self.binSize
        faceIds = np.flatnonzero((vertices[faces, 2] > nearDepth).all(1))
        triangles = vertices[faces[faceIds]]
        projected = triangles@intrinsics.T
        pixels = projected[..., :2]/projected[..., 2:]
        first, second, third = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        denominator = ((second[:, 1]-third[:, 1])*(first[:, 0]-third[:, 0])
                       + (third[:, 0]-second[:, 0])*(first[:, 1]-third[:, 1]))
        lower, upper = pixels.min(1), pixels.max(1)
        nondegenerate = abs(denominator) >= 1e-12
        inViewport = ((upper[:, 0] >= -1e-7) & (upper[:, 1] >= -1e-7)
                      & (lower[:, 0] <= width-1+1e-7) & (lower[:, 1] <= height-1+1e-7))
        retained = nondegenerate & inViewport
        self.faceIds = faceIds[retained]
        self.trianglePixels = pixels[retained]
        self.triangleInverseDepth = 1/triangles[retained, :, 2]
        self.denominator = denominator[retained]
        lower, upper = lower[retained], upper[retained]
        self.bins = {}
        for triangleIndex, (minimum, maximum) in enumerate(zip(lower, upper)):
            minimum = np.maximum(np.floor((minimum-1e-7)/self.binSize).astype(np.int64), 0)
            maximum = np.minimum(np.floor((maximum+1e-7)/self.binSize).astype(np.int64),
                                 [self.binColumns-1, self.binRows-1])
            for row in range(minimum[1], maximum[1]+1):
                for column in range(minimum[0], maximum[0]+1):
                    self.bins.setdefault(int(row*self.binColumns+column), []).append(triangleIndex)
        self.bins = {key: np.asarray(value, dtype=np.int64) for key, value in self.bins.items()}
        self.diagnostics = {'inputTriangleCount': len(faces), 'indexedTriangleCount': len(self.faceIds),
                            'nearPlaneSkippedTriangleCount': len(faces)-len(faceIds),
                            'degenerateProjectedTriangleCount': int((~nondegenerate).sum()),
                            'outsideViewportTriangleCount': int((nondegenerate & ~inViewport).sum()),
                            'occupiedBinCount': len(self.bins), 'binSize': self.binSize,
                            'triangleBinEntries': sum(len(value) for value in self.bins.values()),
                            'twoSidedOcclusion': True}

    def closestDepth(self, queryCameraPoints, *, chunkSize=128):
        """Return closest camera-space Z and original face ID for each query.

        Outside-photo/behind-camera points and rays without hits return inf/-1.
        Query point depth is used only to project it; the hit is independently
        the nearest triangle at that ray, so hidden points cannot self-validate.
        """
        points = np.asarray(queryCameraPoints, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or not isinstance(chunkSize, (int, np.integer)) or isinstance(chunkSize, bool) or chunkSize < 1):
            raise ValueError('Expected finite Nx3 query points and positive integer chunk size')
        depth = np.full(len(points), np.inf, dtype=np.float64)
        hitFaces = np.full(len(points), -1, dtype=np.int64)
        if not len(points):
            return {'depth': depth, 'faceIds': hitFaces, 'insidePhoto': np.zeros(len(points), bool)}
        projected = points@self.intrinsics.T
        positive = points[:, 2] > self.nearDepth
        pixels = np.zeros((len(points), 2), dtype=np.float64)
        pixels[positive] = projected[positive, :2]/projected[positive, 2:]
        height, width = self.imageShape
        inside = (positive & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width-1)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height-1))
        if not len(self.faceIds):
            return {'depth': depth, 'faceIds': hitFaces, 'insidePhoto': inside}
        queryIndices = np.flatnonzero(inside)
        binXy = np.floor(pixels[queryIndices]/self.binSize).astype(np.int64)
        binIds = binXy[:, 1]*self.binColumns+binXy[:, 0]
        order = np.argsort(binIds, kind='stable')
        sortedBins, sortedIndices = binIds[order], queryIndices[order]
        splits = np.r_[0, np.flatnonzero(np.diff(sortedBins))+1, len(order)]
        for firstQuery, lastQuery in zip(splits[:-1], splits[1:]):
            if firstQuery == lastQuery:
                continue
            triangleIndices = self.bins.get(int(sortedBins[firstQuery]))
            if triangleIndices is None:
                continue
            triangles = self.trianglePixels[triangleIndices]
            first, second, third = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            inverseDepth = self.triangleInverseDepth[triangleIndices]
            denominator = self.denominator[triangleIndices]
            for start in range(firstQuery, lastQuery, chunkSize):
                indices = sortedIndices[start:min(start+chunkSize, lastQuery)]
                query = pixels[indices]
                firstWeight = ((second[:, 1]-third[:, 1])*(query[:, None, 0]-third[:, 0])
                               + (third[:, 0]-second[:, 0])*(query[:, None, 1]-third[:, 1]))/denominator
                secondWeight = ((third[:, 1]-first[:, 1])*(query[:, None, 0]-third[:, 0])
                                + (first[:, 0]-third[:, 0])*(query[:, None, 1]-third[:, 1]))/denominator
                thirdWeight = 1-firstWeight-secondWeight
                contained = (firstWeight >= -1e-7) & (secondWeight >= -1e-7) & (thirdWeight >= -1e-7)
                inverse = (firstWeight*inverseDepth[:, 0]+secondWeight*inverseDepth[:, 1]
                           + thirdWeight*inverseDepth[:, 2])
                valid = contained & (inverse > 0)
                values = np.divide(1., inverse, out=np.full_like(inverse, np.inf), where=valid)
                nearest = values.argmin(1)
                nearestDepth = values[np.arange(len(indices)), nearest]
                depth[indices] = nearestDepth
                hitFaces[indices] = np.where(np.isfinite(nearestDepth), self.faceIds[triangleIndices[nearest]], -1)
        return {'depth': depth, 'faceIds': hitFaces, 'insidePhoto': inside}


def refineRejectedVisibility(cameraVertices, faces, intrinsics, queryCameraPoints, initialDepthValid, *,
                             imageShape, depthTolerance, candidateMask, binSize=16, chunkSize=128):
    """Recheck only qualified raster rejects against exact continuous depth.

    ``candidateMask`` must include the caller's existing cosine, positive-depth,
    and photo-bounds gates. This function never relaxes them or depthTolerance.
    Previously accepted depth bits remain unchanged; callers retain their other
    validity gates as before. Invalid/outside queries can never be recovered.
    ``exactDepth`` is NaN where no check was requested, inf for a tested ray with
    no intersection. Diagnostics distinguish retested/recovered/no-hit counts.
    """
    points = np.asarray(queryCameraPoints, dtype=np.float64)
    initial, candidate = np.asarray(initialDepthValid), np.asarray(candidateMask)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or initial.shape != (len(points),) or candidate.shape != initial.shape
            or initial.dtype != np.bool_ or candidate.dtype != np.bool_
            or not np.isfinite(depthTolerance) or depthTolerance <= 0):
        raise ValueError('Expected finite Nx3 queries, boolean masks and positive depth tolerance')
    index = ProjectedSurfaceVisibility(cameraVertices, faces, intrinsics, imageShape, binSize=binSize)
    selected = np.flatnonzero(candidate & ~initial)
    checked = index.closestDepth(points[selected], chunkSize=chunkSize)
    accepted = checked['insidePhoto'] & np.isfinite(checked['depth']) & (abs(checked['depth']-points[selected, 2]) <= depthTolerance)
    visible, recovered = initial.copy(), np.zeros(len(points), bool)
    visible[selected[accepted]], recovered[selected[accepted]] = True, True
    exactDepth = np.full(len(points), np.nan)
    exactDepth[selected] = checked['depth']
    return {'visible': visible, 'recoveredMask': recovered, 'exactDepth': exactDepth,
            'diagnostics': {**index.diagnostics, 'queryCount': len(points),
                'initialDepthAcceptedCount': int(initial.sum()), 'eligibleRejectedCount': len(selected),
                'exactHitCount': int(np.isfinite(checked['depth']).sum()),
                'noIntersectionCount': int((checked['insidePhoto'] & ~np.isfinite(checked['depth'])).sum()),
                'outsidePhotoOrBehindCameraCount': int((~checked['insidePhoto']).sum()),
                'recoveredCount': int(accepted.sum()), 'stillRejectedCount': int((~accepted).sum()),
                'depthTolerance': float(depthTolerance), 'toleranceChanged': False,
                'previousAcceptedMaskPreserved': True, 'method': 'continuousPerspectiveDepthForQualifiedRasterRejects'}}
