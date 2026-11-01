"""One local geometry proposal from frozen image contours and 3D anchors.

An image contour constrains motion only along its supplied image normal. The
caller must establish anatomical meaning, visibility and trustworthy bindings;
neither a dark pixel boundary nor a parser label establishes those properties.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .eyeRefinement import _projectWithJacobian
from .geometryRefinement import _buildSmoothDeformationBasis, _meshEdges, _validateCameras, _validateMesh
from .localSurfaceRegistration import _array, _scalar


def _prepareSamples(constraint, vertexCount, *, dimensions, targetKey, normalKey):
    if not isinstance(constraint, Mapping):
        raise ValueError('Each constraint must be a mapping')
    required = ('vertexIndices', 'barycentric', targetKey, normalKey, 'weights')
    if any(key not in constraint for key in required):
        raise ValueError(f'Each constraint requires {required}')
    indices = np.asarray(constraint['vertexIndices'])
    if (indices.ndim != 2 or not len(indices) or indices.shape[1] not in (1, 2, 3)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= vertexCount):
        raise ValueError('vertexIndices must be valid nonempty integer Mx1/2/3 indices')
    count = len(indices)
    barycentric = _array(constraint['barycentric'], indices.shape, 'barycentric')
    if (np.any(barycentric < 0) or np.any(barycentric > 1)
            or not np.allclose(barycentric.sum(1), 1., atol=1e-7, rtol=0)):
        raise ValueError('barycentric must contain convex weights summing to one')
    targets = _array(constraint[targetKey], (count, dimensions), targetKey)
    normals = _array(constraint[normalKey], (count, dimensions), normalKey)
    normalLengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normalLengths).all() or np.any(normalLengths <= 1e-12):
        raise ValueError(f'{normalKey} must contain finite nonzero directions')
    weights = _array(constraint['weights'], (count,), 'weights')
    weightSum = float(weights.sum())
    if np.any(weights < 0) or not np.isfinite(weightSum) or weightSum <= 0:
        raise ValueError('weights must have nonnegative values and positive finite total support')
    return {'vertexIndices': indices.copy(), 'barycentric': barycentric.copy(),
            'targets': targets.copy(), 'normals': normals/normalLengths[:, None],
            'normalInputLengths': normalLengths, 'weights': weights.copy()}


def _bindSamples(item, vertices, basis, *, requireSupport=True):
    indices, barycentric = item['vertexIndices'], item['barycentric']
    item['beforePoints'] = (vertices[indices]*barycentric[..., None]).sum(1)
    item['sampleBasis'] = (basis[indices]*barycentric[..., None]).sum(1)
    if requireSupport and np.any((item['weights'] > 0) & (np.linalg.norm(item['sampleBasis'], axis=1) <= 1e-12)):
        raise ValueError('A positive-weight sample has no movable surface support')


def _contourMetrics(vertices, item, camera):
    points = (vertices[item['vertexIndices']]*item['barycentric'][..., None]).sum(1)
    pixels, depths, _ = _projectWithJacobian(points, camera)
    valid = (depths > 1e-12) & np.isfinite(pixels).all(1)
    positive = item['weights'] > 0
    validSupport = bool(valid[positive].all())
    normalResidual = np.einsum('ij,ij->i', pixels-item['targets'], item['normals'])
    normalizedResidual = normalResidual/item['pixelScales']
    weight = item['weights']/item['weights'].sum()
    return {
        'weightedNormalRmsPixels': float(np.sqrt(np.sum(weight*normalResidual**2))) if validSupport else None,
        'weightedNormalizedNormalRms': float(np.sqrt(np.sum(weight*normalizedResidual**2))) if validSupport else None,
        'minimumPositiveWeightDepth': float(depths[positive].min()),
        'positiveWeightProjectionsValid': validSupport,
        'points': points, 'pixels': pixels, 'validProjectionMask': valid,
        'normalResidualPixels': normalResidual, 'normalizedNormalResidual': normalizedResidual,
    }


def _depthMetrics(vertices, item, maximum):
    points = (vertices[item['vertexIndices']]*item['barycentric'][..., None]).sum(1)
    residual = np.einsum('ij,ij->i', points-item['targets'], item['normals'])
    weight = item['weights']/item['weights'].sum()
    rms = float(np.sqrt(np.sum(weight*residual**2)))
    return {'weightedPlaneRms': rms, 'weightedNormalizedPlaneRms': rms/maximum,
            'points': points, 'planeResidual': residual}


def refineLocalContours(
    vertices, faces, cameras, constraints, *, activeMask, maxDisplacement,
    controlCount=32, controlRadius, displacementWeight=.01, strainWeight=.1,
    depthConstraints=None, depthWeight=1.,
):
    """Return ``(proposalVertices, diagnostics)`` after one linearized solve.

    Each contour dictionary supplies ``viewIndex``, original ``vertexIndices``
    (Mx1/2/3), convex ``barycentric``, ``targetPixels`` and ``normalPixels``
    (both Mx2), and nonnegative ``weights`` with positive total support. Image
    normals are normalized; their sign is immaterial. Tangential target motion
    has no effect. Cameras follow the frozen OpenCV pinhole convention.

    Optional ``depthConstraints`` supplies ``vertexIndices``, ``barycentric``,
    ``targetPoints``/``targetNormals`` (Mx3), and ``weights``. It adds a 3D
    point-plane anchor; it does not establish cavity or surface semantics.

    Coordinates, radii and displacement caps share one mesh length unit. With
    J the initial projection Jacobian, every photo residual n dot (p-target)
    is divided by the frozen scale ``maxDisplacement * ||n.T @ J||`` in pixels.
    Thus photo residuals approximate normal-direction displacements relative
    to the cap. All photo weights share one total normalization across views.
    Depth residuals are divided by the cap and independently weight-normalized
    before multiplication by ``depthWeight``. Regularizers are weighted means
    of active-vertex ||delta/cap||^2 and affected-edge
    ||delta_i-delta_j||^2/originalEdgeLength^2.

    Gaussian RBF controls and a smooth envelope give coherent displacements.
    One global scaling enforces the cap; inactive vertices remain exact. No
    visibility, correspondences, cameras, normals, scales or targets update.
    This function returns a proposal without geometric or likeness acceptance.
    """
    original, triangles = _validateMesh(vertices, faces)
    original = original.copy()
    frozenCameras = _validateCameras(cameras)
    active = np.asarray(activeMask)
    if active.dtype != np.bool_ or active.shape != (len(original),) or not active.any():
        raise ValueError('activeMask must be a nonempty boolean vertex mask')
    if (isinstance(controlCount, (bool, np.bool_))
            or not isinstance(controlCount, (int, np.integer)) or controlCount < 1):
        raise ValueError('controlCount must be a positive integer')
    maximum = _scalar(maxDisplacement, 'maxDisplacement', positive=True)
    radius = _scalar(controlRadius, 'controlRadius', positive=True)
    displacementWeight = _scalar(displacementWeight, 'displacementWeight')
    strainWeight = _scalar(strainWeight, 'strainWeight')
    depthWeight = _scalar(depthWeight, 'depthWeight')
    if not isinstance(constraints, (list, tuple)) or not constraints:
        raise ValueError('constraints must be a nonempty list or tuple of contour groups')

    activeBasis, basisReport = _buildSmoothDeformationBasis(original, active, int(controlCount), radius)
    basis = np.zeros((len(original), activeBasis.shape[1]), dtype=np.float64)
    basis[active] = activeBasis
    controlSize = basis.shape[1]
    prepared = []
    for constraint in constraints:
        item = _prepareSamples(constraint, len(original), dimensions=2,
                               targetKey='targetPixels', normalKey='normalPixels')
        view = constraint.get('viewIndex')
        if (isinstance(view, (bool, np.bool_)) or not isinstance(view, (int, np.integer))
                or not 0 <= view < len(frozenCameras)):
            raise ValueError('viewIndex must be a valid non-boolean integer camera index')
        item['viewIndex'] = int(view)
        item['name'] = str(constraint.get('name', f'contour{len(prepared)}'))
        _bindSamples(item, original, basis)
        pixels, depths, jacobian = _projectWithJacobian(item['beforePoints'], frozenCameras[view])
        normalJacobian = np.einsum('mi,mij->mj', item['normals'], jacobian)
        gradientLengths = np.linalg.norm(normalJacobian, axis=1)
        scales = maximum*gradientLengths
        if (np.any(depths <= 1e-12) or not np.isfinite(pixels).all()
                or not np.isfinite(scales).all() or np.any(scales <= 0)):
            raise ValueError('Contour samples require positive camera depths and finite nonzero projection scales')
        item['beforePixels'], item['pixelScales'] = pixels, scales
        item['normalizedGradient'] = normalJacobian/gradientLengths[:, None]
        prepared.append(item)
    photoWeightSum = float(sum(item['weights'].sum() for item in prepared))
    if not np.isfinite(photoWeightSum):
        raise ValueError('Combined photo weights must have a finite total')

    matrices, targets = [], []
    for item in prepared:
        count = len(item['weights'])
        scale = np.sqrt(item['weights']/photoWeightSum)
        matrix = (item['sampleBasis'][:, :, None]*item['normalizedGradient'][:, None, :]).reshape(count, 3*controlSize)
        wanted = np.einsum('ij,ij->i', item['targets']-item['beforePixels'], item['normals'])/item['pixelScales']
        matrices.append(matrix*scale[:, None])
        targets.append(wanted*scale)

    depth = None
    if depthConstraints is not None:
        depth = _prepareSamples(depthConstraints, len(original), dimensions=3,
                                targetKey='targetPoints', normalKey='targetNormals')
        _bindSamples(depth, original, basis, requireSupport=depthWeight > 0)
        if depthWeight:
            count = len(depth['weights'])
            scale = np.sqrt(depthWeight*depth['weights']/depth['weights'].sum())
            matrix = (depth['sampleBasis'][:, :, None]*depth['normals'][:, None, :]).reshape(count, 3*controlSize)
            wanted = np.einsum('ij,ij->i', depth['targets']-depth['beforePoints'], depth['normals'])/maximum
            matrices.append(matrix*scale[:, None])
            targets.append(wanted*scale)
    if displacementWeight:
        matrices.append(np.kron(activeBasis, np.eye(3))*np.sqrt(displacementWeight/int(active.sum())))
        targets.append(np.zeros(3*int(active.sum())))
    edges = _meshEdges(triangles)
    edges = edges[active[edges].any(1)]
    edgeLengths = np.linalg.norm(original[edges[:, 1]]-original[edges[:, 0]], axis=1)
    if np.any(edgeLengths <= 0) or not np.isfinite(edgeLengths).all():
        raise ValueError('Affected edges must have finite positive original lengths')
    if strainWeight and len(edges):
        edgeBasis = (basis[edges[:, 1]]-basis[edges[:, 0]])*(maximum/edgeLengths)[:, None]
        matrices.append(np.kron(edgeBasis, np.eye(3))*np.sqrt(strainWeight/len(edges)))
        targets.append(np.zeros(3*len(edges)))
    matrix, target = np.vstack(matrices), np.concatenate(targets)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError('Normalized contour system is nonfinite; check camera and coordinate units')
    controls, _, rank, singularValues = np.linalg.lstsq(matrix, target, rcond=None)
    controls = controls.reshape(controlSize, 3)
    rawDelta = maximum*(basis@controls)
    if not np.isfinite(rawDelta).all():
        raise ValueError('Local contour solution is nonfinite')
    rawMaximum = float(np.linalg.norm(rawDelta, axis=1).max())
    boundScale = min(1., maximum/max(rawMaximum, 1e-30))
    delta = rawDelta*boundScale
    proposal = original.copy()
    proposal[active] += delta[active]
    relativeEdgeDisplacement = np.linalg.norm(delta[edges[:, 1]]-delta[edges[:, 0]], axis=1)/edgeLengths
    changedEdgeLengths = np.linalg.norm(proposal[edges[:, 1]]-proposal[edges[:, 0]], axis=1)
    groups = []
    for item in prepared:
        groups.append({'name': item['name'], 'viewIndex': item['viewIndex'],
                       'sampleCount': len(item['weights']), 'positiveWeightSampleCount': int((item['weights'] > 0).sum()),
                       'objectiveWeightFraction': float(item['weights'].sum()/photoWeightSum),
                       'frozenPixelScales': item['pixelScales'].copy(),
                       'before': _contourMetrics(original, item, frozenCameras[item['viewIndex']]),
                       'proposal': _contourMetrics(proposal, item, frozenCameras[item['viewIndex']])})
    diagnostics = {
        'method': 'fixedContourNormalGaussianRbfIncrement', 'qualityAccepted': False,
        'linearizedSolveCount': 1, 'correspondencesUpdated': False, 'scalesFrozen': True,
        'vertexCount': len(original), 'activeVertexCount': int(active.sum()),
        'photoSampleCount': sum(len(item['weights']) for item in prepared),
        'depthSampleCount': 0 if depth is None else len(depth['weights']),
        'affectedEdgeCount': len(edges), 'basis': basisReport,
        'units': 'Mesh coordinates/radius/cap share length units; targets are camera pixels; objective residuals are dimensionless.',
        'photoScaleDefinition': 'maxDisplacement * norm(unitImageNormal @ initialProjectionJacobian)',
        'objectiveWeights': {'photo': 1., 'depth': depthWeight, 'displacement': displacementWeight, 'strain': strainWeight},
        'rawMaximumDisplacement': rawMaximum, 'maximumDisplacement': float(np.linalg.norm(delta, axis=1).max()),
        'maxDisplacementLimit': maximum, 'boundScale': boundScale,
        'fixedVerticesExactlyUnchanged': bool(np.array_equal(proposal[~active], original[~active])),
        'maximumRelativeEdgeDisplacement': float(relativeEdgeDisplacement.max(initial=0)),
        'maximumRelativeEdgeLengthChange': float(np.max(np.abs(changedEdgeLengths/edgeLengths-1), initial=0)),
        'contourGroups': groups,
        'depth': None if depth is None else {'frozenLengthScale': maximum,
            'before': _depthMetrics(original, depth, maximum), 'proposal': _depthMetrics(proposal, depth, maximum)},
        'rawNormalizedControlOffsets': controls, 'proposalNormalizedControlOffsets': controls*boundScale,
        'linearSystem': {'rows': matrix.shape[0], 'columns': matrix.shape[1], 'rank': int(rank),
                         'largestSingularValue': float(singularValues[0]), 'smallestSingularValue': float(singularValues[-1])},
        'limitations': [
            'Photo contours and 3D anchors require caller-qualified anatomy, visibility and original-surface bindings.',
            'A contour constrains only its image-normal direction; a single view does not determine unique 3D geometry.',
            'The frozen Jacobian describes a local linear approximation; proposal metrics use fresh projection.',
            'Displacement capping and strain regularization do not certify area, orientation, collisions, likeness or rig validity.',
            'Independent held-out observations and fresh multiview renders are required before adoption.',
        ],
    }
    return proposal, diagnostics
