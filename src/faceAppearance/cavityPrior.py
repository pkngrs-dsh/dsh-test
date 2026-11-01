"""Conservative depth priors from an existing fixed-topology template cavity.

An explicit anatomical region is required. A photometric dark patch or a
nearest open hole in another mesh does not establish this correspondence.
Only missing inward depth is proposed; no topology or tangential position is
changed. The returned anchors are a prior, not reconstructed anatomy.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _indices(value, size, name):
    indices = np.asarray(value)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or indices.min() < 0 or indices.max() >= size or len(np.unique(indices)) != len(indices)):
        raise ValueError(f'{name} must be unique valid vertex indices')
    return indices


def _frame(points, inward):
    center = points.mean(0)
    _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
    if singular[1] <= 1e-10 * max(singular[0], 1e-30):
        raise ValueError('Cavity rim must span a nondegenerate plane')
    normal = axes[-1]
    if normal @ inward < 0:
        normal = -normal
    if normal @ inward < .25:
        raise ValueError('Inward direction is ambiguous relative to cavity rim')
    depth = (points-center) @ normal
    planar = points-center-depth[:, None]*normal
    radius = float(np.sqrt(np.mean(np.sum(planar**2, axis=1))))
    return center, normal, radius, float(np.sqrt(np.mean(depth**2)))


def buildCavityDepthPrior(vertices, restVertices, rimVertexIds, interiorVertexIds, *,
                         patchFaces, inwardDirection, maxAddedDepth, restorationFraction=.35):
    """Build bounded point-plane anchors for existing cavity interior vertices.

    Both meshes must share vertex order, frame and units. Boundary planes are
    fitted independently and oriented by the caller's inward direction. Rest
    depths scale by the ratio of boundary RMS radii; ratios outside [0.5, 2]
    are rejected. A harmonic extension of each nonplanar rim's signed depth
    is subtracted before comparing cavity depths. Targets retain input
    tangential coordinates and add only ``restorationFraction`` of missing
    positive template depth. A single uniform scale enforces ``maxAddedDepth``.
    Current deeper cavities are never flattened.

    Region selection, boundary identity, shape plausibility and geometric
    safety remain caller responsibilities. This does not infer an aperture.
    """
    current, rest = np.asarray(vertices, float), np.asarray(restVertices, float)
    inward = np.asarray(inwardDirection, float)
    if (current.ndim != 2 or current.shape[1] != 3 or rest.shape != current.shape
            or not np.isfinite(current).all() or not np.isfinite(rest).all()
            or inward.shape != (3,) or not np.isfinite(inward).all()
            or np.linalg.norm(inward) <= 1e-12):
        raise ValueError('Expected finite shared-topology coordinates and nonzero inward direction')
    if (isinstance(maxAddedDepth, (bool, np.bool_)) or isinstance(restorationFraction, (bool, np.bool_))
            or not np.isfinite([maxAddedDepth, restorationFraction]).all()
            or maxAddedDepth <= 0 or not 0 < restorationFraction <= 1):
        raise ValueError('Expected positive depth limit and restoration fraction in (0,1]')
    rim = _indices(rimVertexIds, len(current), 'rimVertexIds')
    interior = _indices(interiorVertexIds, len(current), 'interiorVertexIds')
    if len(rim) < 3 or np.intersect1d(rim, interior).size:
        raise ValueError('Cavity requires a separate rim of at least three vertices')
    faces = np.asarray(patchFaces)
    members = np.r_[rim, interior]
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or not np.array_equal(np.unique(faces), np.sort(members))):
        raise ValueError('patchFaces must cover exactly the supplied rim and interior')
    mapping = np.full(len(current), -1, int)
    mapping[members] = np.arange(len(members))
    localFaces = mapping[faces]
    edges = np.sort(np.concatenate([localFaces[:, [0, 1]], localFaces[:, [1, 2]], localFaces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if (np.any(counts > 2) or not np.array_equal(np.unique(edges[counts == 1]), np.arange(len(rim)))
            or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('patchFaces need a manifold boundary matching rimVertexIds')
    adjacency = sparse.coo_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(members), len(members))).tocsr()
    if sparse.csgraph.connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('Cavity patch must be connected to its rim')
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    interiorBlock = laplacian[len(rim):, len(rim):].tocsc()
    rimBlock = laplacian[len(rim):, :len(rim)]
    inward = inward / np.linalg.norm(inward)
    center, normal, radius, rimRms = _frame(current[rim], inward)
    restCenter, restNormal, restRadius, restRms = _frame(rest[rim], inward)
    scale = radius/restRadius
    if not .5 <= scale <= 2.:
        raise ValueError('Current/rest cavity rim scale differs excessively')
    beforeRaw = (current[interior]-center) @ normal
    restRaw = (rest[interior]-restCenter) @ restNormal
    currentExtension = spsolve(interiorBlock, -rimBlock @ ((current[rim]-center) @ normal))
    restExtension = spsolve(interiorBlock, -rimBlock @ ((rest[rim]-restCenter) @ restNormal))
    beforeDepth = beforeRaw-currentExtension
    restDepth = (restRaw-restExtension)*scale
    missing = np.maximum(restDepth-beforeDepth, 0.)
    increment = np.where(restDepth > 0, restorationFraction*missing, 0.)
    priorScale = min(1., maxAddedDepth/max(float(increment.max()), 1e-30))
    increment *= priorScale
    targets = current[interior] + increment[:, None]*normal
    return {
        'vertexIndices': interior[:, None], 'barycentric': np.ones((len(interior), 1)),
        'targetPoints': targets, 'targetNormals': np.tile(normal, (len(interior), 1)),
        'weights': np.ones(len(interior)), 'beforeDepth': beforeDepth,
        'scaledRestDepth': restDepth, 'addedDepth': increment,
        'beforeRawDepth': beforeRaw, 'currentBoundaryExtension': currentExtension,
        'restBoundaryExtension': restExtension,
        'diagnostics': {'method': 'boundedMissingTemplateCavityDepth', 'qualityAccepted': False,
            'rimVertexIds': rim, 'interiorVertexIds': interior,
            'currentRimCenter': center, 'currentInwardNormal': normal,
            'restRimCenter': restCenter, 'restInwardNormal': restNormal,
            'boundaryRadiusScale': scale, 'currentRimPlaneRms': rimRms, 'restRimPlaneRms': restRms,
            'currentRimPlaneRmsOverRadius': rimRms/radius, 'restRimPlaneRmsOverRadius': restRms/restRadius,
            'inwardNormalDot': float(normal @ restNormal), 'priorUniformScale': priorScale,
            'nonplanarBoundaryCorrection': 'Fixed-topology harmonic rim-depth extension, zero corrected rim depth',
            'restorationFraction': restorationFraction, 'maxAddedDepth': maxAddedDepth,
            'maximumProposedDepthIncrement': float(increment.max()),
            'limitations': ['Template cavity depth is a shape prior, not photo-measured subject anatomy.',
                'Anatomical region and direction must be verified by the caller.',
                'These targets do not enforce boundary continuity or mesh safety.']}}
