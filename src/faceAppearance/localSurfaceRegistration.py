"""One coherent local surface increment from frozen 3D correspondences.

The caller must qualify target surfaces, normals, anatomical regions and
visibility before calling this solver. This module does not find or update
correspondences, alter topology, or approve a reconstruction for use.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from .geometryRefinement import _buildSmoothDeformationBasis, _meshEdges, _validateMesh


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def _surfaceMetrics(points, targets, normals, weights):
    residual = points - targets
    normal = np.einsum('ij,ij->i', residual, normals)
    tangent = residual - normal[:, None] * normals
    return {
        'weightedPointRms': float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)))),
        'weightedNormalRms': float(np.sqrt(np.sum(weights * normal**2))),
        'weightedTangentRms': float(np.sqrt(np.sum(weights * np.sum(tangent**2, axis=1)))),
    }


def solveLocalSurfaceIncrement(
    vertices, faces, sampleVertexIndices, sampleBarycentric, targetPoints,
    targetNormals, weights, *, activeMask, maxDisplacement, controlCount=32,
    controlRadius, normalWeight=1., tangentWeight=.05,
    displacementWeight=.01, strainWeight=.1,
):
    """Return ``(proposal, diagnostics)`` from one linear least-squares solve.

    All coordinates, ``controlRadius`` and ``maxDisplacement`` share one length
    unit. Samples use convex barycentrics over 1--3 original vertex indices.
    Normals are normalized internally; weights are nonnegative with positive
    total support. Positive-weight samples require movable surface support.

    Unknown Kx3 control offsets U parameterize delta/maxDisplacement through a
    fixed Gaussian RBF basis with a smooth envelope toward inactive vertices.
    The dimensionless objective is the sum of:

    * weighted mean normal residual squared / maxDisplacement**2;
    * weighted mean tangential residual squared / maxDisplacement**2;
    * mean active-vertex ||delta/maxDisplacement||**2;
    * mean affected-edge ||delta_i-delta_j||**2 / originalEdgeLength**2.

    The four respective coefficients are the named weight arguments. Both
    ends of affected edges participate, including a zero inactive endpoint.
    One uniform scale enforces the displacement cap after solving; there is
    no per-vertex clipping or target update. Inactive vertices remain exact.
    Diagnostics report a proposal, never geometric or anatomical acceptance.
    """
    original, triangles = _validateMesh(vertices, faces)
    original = original.copy()
    active = np.asarray(activeMask)
    if active.dtype != np.bool_ or active.shape != (len(original),) or not active.any():
        raise ValueError('activeMask must be a nonempty boolean vertex mask')
    if (isinstance(controlCount, (bool, np.bool_))
            or not isinstance(controlCount, (int, np.integer)) or controlCount < 1):
        raise ValueError('controlCount must be a positive integer')
    maximum = _scalar(maxDisplacement, 'maxDisplacement', positive=True)
    radius = _scalar(controlRadius, 'controlRadius', positive=True)
    normalWeight = _scalar(normalWeight, 'normalWeight', positive=True)
    tangentWeight = _scalar(tangentWeight, 'tangentWeight')
    displacementWeight = _scalar(displacementWeight, 'displacementWeight')
    strainWeight = _scalar(strainWeight, 'strainWeight')

    indices = np.asarray(sampleVertexIndices)
    if (indices.ndim != 2 or not len(indices) or indices.shape[1] not in (1, 2, 3)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= len(original)):
        raise ValueError('sampleVertexIndices must be valid nonempty integer Mx1/2/3 indices')
    count = len(indices)
    barycentric = _array(sampleBarycentric, indices.shape, 'sampleBarycentric')
    if (np.any(barycentric < 0) or np.any(barycentric > 1)
            or not np.allclose(barycentric.sum(1), 1., atol=1e-7, rtol=0)):
        raise ValueError('sampleBarycentric must be convex weights summing to one')
    targets = _array(targetPoints, (count, 3), 'targetPoints')
    normals = _array(targetNormals, (count, 3), 'targetNormals')
    normalLengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normalLengths).all() or np.any(normalLengths <= 1e-12):
        raise ValueError('targetNormals must be nonzero finite directions')
    normals = normals / normalLengths[:, None]
    sampleWeights = _array(weights, (count,), 'weights')
    weightSum = float(sampleWeights.sum())
    if np.any(sampleWeights < 0) or not np.isfinite(weightSum) or weightSum <= 0:
        raise ValueError('weights must be nonnegative with positive finite total support')
    normalizedWeights = sampleWeights / weightSum

    activeBasis, basisReport = _buildSmoothDeformationBasis(original, active, int(controlCount), radius)
    basis = np.zeros((len(original), activeBasis.shape[1]), dtype=np.float64)
    basis[active] = activeBasis
    sampleBasis = (basis[indices] * barycentric[..., None]).sum(1)
    if np.any((sampleWeights > 0) & (np.linalg.norm(sampleBasis, axis=1) <= 1e-12)):
        raise ValueError('A positive-weight sample has no movable surface support')
    beforePoints = (original[indices] * barycentric[..., None]).sum(1)
    wanted = (targets - beforePoints) / maximum
    controlSize = activeBasis.shape[1]

    # A row is n dot (sum_k B_k U_k). Control ordering is [k, xyz].
    normalMatrix = (sampleBasis[:, :, None] * normals[:, None, :]).reshape(count, 3*controlSize)
    normalScale = np.sqrt(normalWeight * normalizedWeights)
    matrices = [normalMatrix * normalScale[:, None]]
    rightHandSides = [np.einsum('ij,ij->i', wanted, normals) * normalScale]
    if tangentWeight:
        projector = np.eye(3)[None] - normals[:, :, None]*normals[:, None, :]
        tangentMatrix = (sampleBasis[:, None, :, None]*projector[:, :, None, :]).reshape(3*count, 3*controlSize)
        tangentScale = np.repeat(np.sqrt(tangentWeight*normalizedWeights), 3)
        matrices.append(tangentMatrix*tangentScale[:, None])
        rightHandSides.append(np.einsum('mij,mj->mi', projector, wanted).ravel()*tangentScale)
    if displacementWeight:
        matrices.append(np.kron(activeBasis, np.eye(3))*np.sqrt(displacementWeight/int(active.sum())))
        rightHandSides.append(np.zeros(3*int(active.sum())))

    edges = _meshEdges(triangles)
    edges = edges[active[edges].any(1)]
    edgeLengths = np.linalg.norm(original[edges[:, 1]]-original[edges[:, 0]], axis=1)
    if np.any(edgeLengths <= 0) or not np.isfinite(edgeLengths).all():
        raise ValueError('Affected edges require finite positive original lengths')
    if strainWeight and len(edges):
        edgeBasis = (basis[edges[:, 1]]-basis[edges[:, 0]]) * (maximum/edgeLengths)[:, None]
        matrices.append(np.kron(edgeBasis, np.eye(3))*np.sqrt(strainWeight/len(edges)))
        rightHandSides.append(np.zeros(3*len(edges)))
    matrix, target = np.vstack(matrices), np.concatenate(rightHandSides)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError('Normalized surface system is nonfinite; check coordinate units')
    controls, _, rank, singularValues = np.linalg.lstsq(matrix, target, rcond=None)
    controls = controls.reshape(controlSize, 3)
    rawDelta = (basis @ controls) * maximum
    if not np.isfinite(rawDelta).all():
        raise ValueError('Local surface solution is nonfinite')
    rawMaximum = float(np.linalg.norm(rawDelta, axis=1).max())
    boundScale = min(1., maximum / max(rawMaximum, 1e-30))
    delta = rawDelta * boundScale
    proposal = original.copy()
    proposal[active] += delta[active]
    proposalPoints = (proposal[indices] * barycentric[..., None]).sum(1)
    relativeEdgeDisplacement = np.linalg.norm(delta[edges[:, 1]]-delta[edges[:, 0]], axis=1)/edgeLengths
    changedEdgeLengths = np.linalg.norm(proposal[edges[:, 1]]-proposal[edges[:, 0]], axis=1)
    diagnostics = {
        'method': 'fixedCorrespondenceGaussianRbfSurfaceIncrement',
        'qualityAccepted': False, 'linearizedSolveCount': 1, 'correspondencesUpdated': False,
        'vertexCount': len(original), 'activeVertexCount': int(active.sum()),
        'sampleCount': count, 'positiveWeightSampleCount': int((sampleWeights>0).sum()),
        'affectedEdgeCount': len(edges), 'basis': basisReport,
        'units': 'All coordinates/radii/displacements share input mesh length units; objective residuals are dimensionless.',
        'objectiveWeights': {'normal': normalWeight, 'tangent': tangentWeight,
                            'displacement': displacementWeight, 'strain': strainWeight},
        'normalInputLengths': {'minimum': float(normalLengths.min()), 'maximum': float(normalLengths.max())},
        'rawMaximumDisplacement': rawMaximum, 'maximumDisplacement': float(np.linalg.norm(delta,axis=1).max()),
        'maxDisplacementLimit': maximum, 'boundScale': boundScale,
        'fixedVerticesExactlyUnchanged': bool(np.array_equal(proposal[~active], original[~active])),
        'maximumRelativeEdgeDisplacement': float(relativeEdgeDisplacement.max(initial=0)),
        'maximumRelativeEdgeLengthChange': float(np.max(np.abs(changedEdgeLengths/edgeLengths-1), initial=0)),
        'before': _surfaceMetrics(beforePoints, targets, normals, normalizedWeights),
        'proposal': _surfaceMetrics(proposalPoints, targets, normals, normalizedWeights),
        'beforePoints': beforePoints, 'proposalPoints': proposalPoints,
        'rawNormalizedControlOffsets': controls, 'proposalNormalizedControlOffsets': controls*boundScale,
        'linearSystem': {'rows': matrix.shape[0], 'columns': matrix.shape[1], 'rank': int(rank),
                         'largestSingularValue': float(singularValues[0]), 'smallestSingularValue': float(singularValues[-1])},
        'limitations': ['Target correspondence, normals, visibility, holes and anatomical region must be gated by the caller.',
            'Displacement capping and strain regularization do not certify area, orientation, collisions, likeness or rig validity.',
            'Independent held-out surface samples and fresh multiview renders are required before adopting the proposal.'],
    }
    return proposal, diagnostics
