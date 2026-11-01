"""Generic, observation-weighted 3D shape prior for a bounded template patch.

This opt-in builder returns a sparse quadratic term. It does not detect an
anatomical region, fit vertices, change defaults, or approve geometry. Template
shape is prior knowledge, not a measurement of an individual subject.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .cavityPrior import _indices
from .geometryRefinement import _validateMesh
from .localSurfaceRegistration import _array, _scalar
from .unsupportedCavityPrior import _harmonicBoundaryOperator, _sampleSupport


def buildUnsupportedShapePrior(
    alignedRestVertices, faces, boundaryVertexIds, interiorVertexIds,
    sampleVertexIndices, sampleBarycentric, sampleWeights, *, normalDirection,
    lengthScale, referenceSupport=None, supportThreshold=.15,
    normalTargetFraction=1., tangentTargetFraction=1., normalPenalty=1., tangentPenalty=1.,
):
    """Prepare a target-frame harmonic-relative vector prior, without fitting.

    ``faces`` is the complete template topology. Boundary and interior IDs must
    describe one connected manifold disk whose interior has no outside
    neighbors. The supplied axis need not be a fitted boundary-plane normal;
    it is a caller-selected direction for anisotropic target scaling. Its sign
    has no effect. This function validates topology, not anatomical identity.

    The observation matrix C has CLIENT vertex indices and convex barycentrics
    for positive-weight samples. Zero-weight rows may contain finite legacy
    extrapolations. Support is s = diag(C.T @ diag(sampleWeights) @ C), combining
    duplicate indices in each row before squaring. For the existing direct
    registration objective, use 1 for valid samples and 0 for excluded ones.
    Other positive values describe a different caller-supplied support model.

    Reference support defaults to the median positive boundary support. An
    explicit positive value is required when the boundary is unobserved.
    Interior influence is a_i = clip(1-s_i/(reference*supportThreshold), 0, 1).
    Every supplied interior vertex participates according to support, including
    vertices with zero or negative normal offset. No geometric sign test, face
    ID, individual sample ID, or handpicked bad vertex changes that influence.

    Let W be the convex harmonic extension from the boundary, H V=V_I-W V_B,
    r=H alignedRest, P=n n.T, T=I-P, and

        q = normalTargetFraction * r @ P + tangentTargetFraction * r @ T
        K = sqrt(normalPenalty) * P + sqrt(tangentPenalty) * T.

    Fractions in [0,1] scale absolute TEMPLATE relative vectors, never missing
    depth from a previously collapsed solution. Both fractions default to one.
    Nonnegative penalties choose directional costs; at least one must be
    positive. The returned weightedOperator and weightedTargets encode

        ||A @ V.ravel()-b||^2 = sum_i a_i*||(H V-q)_i @ K||^2 / (M*lengthScale^2).

    M is the number of ALL supplied interior vertices, not 3M or sum(a). The
    caller supplies the global prior weight, without modifying its data-term
    normalization. A is sparse 3M x 3N; rows are interior-major XYZ and columns
    vertex-major XYZ. These arrays can be passed directly to the generic
    ``registerDirectTemplate(..., quadraticPrior=...)`` exact solver. To merge
    several patches with a global interior-count normalization, multiply each
    patch's A and b by sqrt(Mpatch/Mtotal) before stacking them.

    All vertices, directions and targets are already in the registration target
    frame. Translation and arbitrary harmonic boundary displacement do not
    change residuals. Rebuilding with jointly rotated/rest-scaled inputs and a
    correspondingly scaled lengthScale preserves the normalized objective.
    Uniform replication or rescaling of observations also preserves influence
    when the default reference is used; explicit references must be rescaled
    by the caller. The fixed operator is not rotation-adaptive during a solve.
    It leaves global translation unconstrained and cannot alone guarantee
    noncollapsed edges, orientation, intersections or subject likeness.
    """
    rest, triangles = _validateMesh(alignedRestVertices, faces)
    boundary = _indices(boundaryVertexIds, len(rest), 'boundaryVertexIds').copy()
    interior = _indices(interiorVertexIds, len(rest), 'interiorVertexIds').copy()
    if len(boundary) < 3 or np.intersect1d(boundary, interior).size:
        raise ValueError('Shape patch requires disjoint interior and at least three boundary vertices')
    # The shared harmonic helper solves for a dense M-by-B convex extension;
    # reject unbounded allocation before invoking it. This is a memory limit,
    # independent of subject identity, anatomical position or observed failures.
    if len(boundary) * len(interior) > 1_000_000:
        raise ValueError('Harmonic boundary extension exceeds the dense storage budget')
    scale = _scalar(lengthScale, 'lengthScale', positive=True)
    threshold = _scalar(supportThreshold, 'supportThreshold', positive=True)
    normalFraction = _scalar(normalTargetFraction, 'normalTargetFraction')
    tangentFraction = _scalar(tangentTargetFraction, 'tangentTargetFraction')
    normalCost = _scalar(normalPenalty, 'normalPenalty')
    tangentCost = _scalar(tangentPenalty, 'tangentPenalty')
    if threshold > 1 or normalFraction > 1 or tangentFraction > 1:
        raise ValueError('supportThreshold must be in (0,1] and target fractions in [0,1]')
    if normalCost == 0 and tangentCost == 0:
        raise ValueError('At least one directional penalty must be positive')
    normal = _array(normalDirection, (3,), 'normalDirection')
    normalLength = float(np.linalg.norm(normal))
    if not np.isfinite(normalLength) or normalLength <= 1e-12:
        raise ValueError('normalDirection must be a finite nonzero axis')
    normal = normal / normalLength
    normalProjector = np.outer(normal, normal)
    tangentProjector = np.eye(3) - normalProjector
    metricRoot = np.sqrt(normalCost) * normalProjector + np.sqrt(tangentCost) * tangentProjector
    relative, extension, topology = _harmonicBoundaryOperator(len(rest), triangles, boundary, interior)
    support, validSampleCount = _sampleSupport(len(rest), sampleVertexIndices, sampleBarycentric, sampleWeights)
    positiveBoundary = support[boundary][support[boundary] > 0]
    if referenceSupport is None:
        if not len(positiveBoundary):
            raise ValueError('No positive boundary support; supply an explicit positive referenceSupport')
        reference = float(np.median(positiveBoundary))
        referenceMethod = 'medianPositiveBoundaryDataNormalEquationDiagonal'
    else:
        reference = _scalar(referenceSupport, 'referenceSupport', positive=True)
        referenceMethod = 'explicitCallerSupportReference'
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        supportRatio = support[interior] / reference
        priorWeights = np.clip(1. - supportRatio / threshold, 0., 1.)
        relativeRest = relative @ rest
        targetOffsets = normalFraction * (relativeRest @ normalProjector) + tangentFraction * (relativeRest @ tangentProjector)
        vertexScales = np.sqrt(priorWeights / len(interior)) / scale
        rowScales = np.repeat(vertexScales, 3)
        vectorOperator = sparse.kron(relative, sparse.eye(3), format='csr')
        weightedOperator = (sparse.diags(rowScales) @ sparse.kron(relative, metricRoot, format='csr')).tocsr()
        weightedOperator.eliminate_zeros()
        weightedTargets = (targetOffsets @ metricRoot).ravel() * rowScales
    if (not all(np.isfinite(value).all() for value in (supportRatio, relativeRest, targetOffsets,
            rowScales, weightedTargets, weightedOperator.data))):
        raise ValueError('Shape prior overflowed; check support and coordinate/length units')
    return {'relativeOperator': relative, 'harmonicBoundaryWeights': extension,
        'vectorOperator': vectorOperator, 'restRelativeOffsets': relativeRest, 'targetOffsets': targetOffsets,
        'normalProjector': normalProjector, 'tangentProjector': tangentProjector, 'metricRoot': metricRoot,
        'weightedOperator': weightedOperator, 'weightedTargets': weightedTargets,
        'priorWeights': priorWeights, 'vertexScales': vertexScales, 'rowScales': rowScales,
        'vertexSupport': support, 'interiorSupportRatio': supportRatio,
        'boundaryVertexIds': boundary, 'interiorVertexIds': interior, 'normalDirection': normal,
        'diagnostics': {'method': 'unsupportedInteriorHarmonicRelativeVectorPrior',
            'qualityAccepted': False, 'registrationDefaultsChanged': False,
            'sourceDataSupport': 'diagonal(C.T @ diag(sampleWeights) @ C)',
            'validSampleCount': validSampleCount, 'referenceSupport': reference,
            'referenceSupportMethod': referenceMethod, 'positiveBoundarySupportCount': len(positiveBoundary),
            'supportThreshold': threshold, 'interiorCount': len(interior),
            'unsupportedInteriorCount': int((priorWeights > 0).sum()),
            'zeroDataSupportInteriorCount': int((support[interior] == 0).sum()),
            'geometricSignUsedToSelectPrior': False, 'priorWeightSum': float(priorWeights.sum()),
            'normalizationInteriorCount': len(interior), 'lengthScale': scale,
            'normalTargetFraction': normalFraction, 'tangentTargetFraction': tangentFraction,
            'normalPenalty': normalCost, 'tangentPenalty': tangentCost,
            'topology': topology, 'columnOrder': 'vertexMajorXYZ', 'rowOrder': 'interiorMajorXYZ',
            'coordinateFrame': 'alreadyAlignedTargetFrameAndUnits',
            'normalDirectionSignInvariant': True,
            'objective': 'shapePriorWeight * sum(priorWeights * ||(H@vertices-targetOffsets)@metricRoot||^2) / (interiorCount*lengthScale^2)',
            'limitations': [
                'Template shape is prior knowledge, not measured individual anatomy; support is not visibility or independent-view confidence.',
                'The caller must provide a valid region and common target coordinate frame; no anatomical region or local direction is inferred.',
                'Boundary coordinates remain variable; the operator does not anchor world translation.',
                'Harmonic coordinates, directional projectors and target offsets are frozen and do not adapt to local rotation during fitting.',
                'Uniform observation replication leaves normalized influence unchanged, but changes the underlying data objective unless its caller normalizes it.',
                'No edge, triangle, intersection, rig, texture or photo-likeness acceptance is performed.']}}
