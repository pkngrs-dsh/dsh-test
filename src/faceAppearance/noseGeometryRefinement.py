"""Small multiview nose deformations from visible, frozen surface samples.

Camera calibration and topology are inputs, never optimization parameters.
Correspondences are view dependent: an alar contour is not a common 3D point.
The field has hard zero boundary conditions and no symmetry prior. Callers
must independently check cumulative geometry, intersections and fresh renders.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .eyeRefinement import _preparePointConstraints, _projectWithJacobian
from .geometryRefinement import _meshEdges, _validateCameras, _validateMesh
from .surfaceVisibility import ProjectedSurfaceVisibility


def eyeMouthAnchors(points, fractions):
    """Interpolate fixed facial anchors; does not assume bilateral symmetry."""
    points, fractions = np.asarray(points, float), np.asarray(fractions, float)
    eye = points[[36, 39, 42, 45]].mean(0)
    mouth = points[[48, 54]].mean(0)
    return eye + fractions[:, None] * (mouth - eye)


def noseAnchorFractions(photoPoints, semanticIds):
    """Freeze vertical position along the photo eye-to-mouth axis per sample."""
    points = np.asarray(photoPoints, float)
    eye = points[[36, 39, 42, 45]].mean(0)
    mouth = points[[48, 54]].mean(0)
    if not np.isfinite(points[np.r_[semanticIds, 36, 39, 42, 45, 48, 54]]).all() or mouth[1] - eye[1] <= 1:
        raise ValueError('Unavailable eye-mouth anchor frame')
    return np.clip((points[semanticIds, 1] - eye[1]) / (mouth[1] - eye[1]), -.15, .85)


def visibleNoseBindings(vertices, faces, camera, imageShape, pixels, *,
                        domainDistance, maximumDomainDistance, fixedVertexIds=(),
                        minimumViewCosine=.08):
    """Ray-bind the nearest visible skin, rejecting protected/outside surfaces.

    ``domainDistance`` is only a broad anatomical gate, such as geodesic
    distance to approximate nasal seeds. It never supplies target coordinates.
    """
    vertices, faces = _validateMesh(vertices, faces)
    camera = _validateCameras([camera])[0]
    pixels = np.asarray(pixels, float)
    distance = np.asarray(domainDistance, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError('Expected finite pixel rays')
    if (distance.shape != (len(vertices),) or np.isnan(distance).any() or (distance < 0).any()
            or not np.isfinite([maximumDomainDistance, minimumViewCosine]).all()
            or maximumDomainDistance <= 0 or not 0 <= minimumViewCosine < 1):
        raise ValueError('Invalid nasal anatomical domain')
    cameraVertices = vertices @ camera['rotation'].T + camera['translation']
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera['intrinsics']).T
    visibility = ProjectedSurfaceVisibility(cameraVertices, faces, camera['intrinsics'], imageShape)
    hits = visibility.closestDepth(rays)
    faceIds = hits['faceIds']
    indices = faces[np.maximum(faceIds, 0)]
    barycentric = np.full((len(pixels), 3), np.nan)
    canonical = np.full_like(barycentric, np.nan)
    cosine = np.full(len(pixels), np.nan)
    domain = np.full(len(pixels), np.inf)
    reasons = []
    fixed = np.asarray(fixedVertexIds, int)
    for row, faceId in enumerate(faceIds):
        why = []
        if faceId < 0:
            reasons.append(['noVisibleSkinHit'])
            continue
        triangle = cameraVertices[indices[row]]
        point = rays[row] * hits['depth'][row] / rays[row, 2]
        local = np.linalg.lstsq((triangle[1:] - triangle[0]).T, point - triangle[0], rcond=None)[0]
        barycentric[row] = np.r_[1 - local.sum(), local]
        canonical[row] = barycentric[row] @ vertices[indices[row]]
        # A disconnected geodesic component may legitimately have +inf
        # distance. Reject its interpolation explicitly, including 0 * inf
        # at triangle edges, rather than letting NaN bypass a comparison.
        triangleDistance = distance[indices[row]]
        domain[row] = (barycentric[row] @ triangleDistance
                       if np.isfinite(triangleDistance).all() else np.inf)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        cosine[row] = normal @ (-point) / (np.linalg.norm(normal) * np.linalg.norm(point))
        if barycentric[row].min() < -1e-7: why.append('invalidBarycentric')
        if not np.isfinite(domain[row]) or domain[row] > maximumDomainDistance:
            why.append('outsideBroadNoseDomain')
        if cosine[row] < minimumViewCosine: why.append('grazingOrBackFacing')
        if np.isin(indices[row], fixed).any(): why.append('protectedTriangle')
        reasons.append(why)
    return {'vertexIndices': indices, 'barycentric': barycentric, 'faceIds': faceIds,
            'canonicalPoints': canonical, 'viewNormalCosine': cosine,
            'domainDistance': domain, 'acceptedMask': np.asarray([not item for item in reasons]),
            'reasons': reasons}


def refineNoseGeometry(vertices, faces, cameras, constraints, *, activeMask,
                       maxDisplacement, smoothnessWeight=10., strainWeight=.1,
                       screenWeight=.002):
    """One linearized multiview solve for a smooth 3D displacement field.

    Unknowns are displacements normalized by ``maxDisplacement``. All inactive
    vertices are eliminated exactly from the linear system. A single uniform
    scale limits the final field; no pointwise clipping or iterative rebind is
    used. Positive screening makes unobserved directions prefer zero movement.
    This returns a proposal, not a quality-approved reconstruction.
    """
    original, faces = _validateMesh(vertices, faces)
    cameras = _validateCameras(cameras)
    active = np.asarray(activeMask)
    if active.dtype.kind != 'b' or active.shape != (len(original),) or not active.any():
        raise ValueError('Need a nonempty boolean active vertex mask')
    if (not np.isfinite([maxDisplacement, smoothnessWeight, strainWeight, screenWeight]).all()
            or min(maxDisplacement, smoothnessWeight, screenWeight) <= 0 or strainWeight < 0):
        raise ValueError('Invalid bounded field settings')
    prepared = _preparePointConstraints(constraints, len(original), cameras)
    if not prepared:
        raise ValueError('Need visible surface constraints')
    free = np.flatnonzero(active)
    freeIndex = np.full(len(original), -1, int)
    freeIndex[free] = np.arange(len(free))
    data, rowIds, columnIds, targets, weights, projections = [], [], [], [], [], []
    offset = 0
    for item in prepared:
        points = (original[item['vertexIndices']] * item['barycentric'][..., None]).sum(1)
        pixels, depth, jacobian = _projectWithJacobian(points, cameras[item['viewIndex']])
        if (depth <= 0).any():
            raise ValueError('Constraints require positive camera depth')
        if not active[item['vertexIndices']].any(1).all():
            raise ValueError('A constraint has no active surface support')
        for sample, ids in enumerate(item['vertexIndices']):
            for vertex, bary in zip(ids, item['barycentric'][sample]):
                if not active[vertex]: continue
                for axis in range(2):
                    rowIds.extend([offset + 2 * sample + axis] * 3)
                    columnIds.extend((3 * freeIndex[vertex] + np.arange(3)).tolist())
                    data.extend((bary * jacobian[sample, axis] * maxDisplacement).tolist())
        targets.extend((item['targetPixels'] - pixels).ravel())
        weights.extend(np.repeat(item['weights'], 2))
        projections.append(pixels)
        offset += 2 * len(points)
    matrix = sparse.coo_matrix((data, (rowIds, columnIds)), shape=(offset, 3 * len(free))).tocsr()
    weights = np.asarray(weights)
    weights /= weights.sum()
    edges = _meshEdges(faces)
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(original), len(original))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = (sparse.eye(len(original)) - sparse.diags(1 / np.maximum(degree, 1)) @ adjacency)[:, free]
    length = np.linalg.norm(original[edges[:, 0]] - original[edges[:, 1]], axis=1)
    shortWeight = np.median(length) / length
    gradient = sparse.coo_matrix((np.c_[-shortWeight, shortWeight].ravel(),
        (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), len(original))).tocsr()[:, free]
    regularizer = (smoothnessWeight / len(original) * (laplacian.T @ laplacian)
        + strainWeight / len(edges) * (gradient.T @ gradient)
        + screenWeight / len(original) * sparse.eye(len(free)))
    hessian = matrix.T @ sparse.diags(weights) @ matrix + sparse.kron(regularizer, sparse.eye(3))
    normalized = spsolve(hessian.tocsc(), matrix.T @ (weights * targets)).reshape(-1, 3)
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite nasal field solution')
    boundScale = min(1., 1. / max(float(np.linalg.norm(normalized, axis=1).max()), 1e-20))
    delta = np.zeros_like(original)
    delta[free] = normalized * (maxDisplacement * boundScale)
    proposal = original.copy()
    proposal[free] += delta[free]
    reports = []
    for item, beforePixels in zip(prepared, projections):
        points = (proposal[item['vertexIndices']] * item['barycentric'][..., None]).sum(1)
        afterPixels, _, _ = _projectWithJacobian(points, cameras[item['viewIndex']])
        reports.append({'viewIndex': item['viewIndex'], 'beforePixels': beforePixels,
            'proposalPixels': afterPixels, 'targetPixels': item['targetPixels'],
            'beforeErrorPixels': np.linalg.norm(beforePixels - item['targetPixels'], axis=1),
            'proposalErrorPixels': np.linalg.norm(afterPixels - item['targetPixels'], axis=1)})
    return proposal, {'freeVertexCount': len(free), 'boundScale': boundScale,
        'maximumDisplacement': float(np.linalg.norm(delta, axis=1).max()),
        'fixedVerticesExactlyUnchanged': bool(np.array_equal(proposal[~active], original[~active])),
        'pointReports': reports, 'linearizedSolveCount': 1, 'deformationFrame': 'world3d',
        'symmetryPrior': False, 'qualityAccepted': False}
