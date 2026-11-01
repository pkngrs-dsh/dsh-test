"""Relative-depth regularization for an explicitly audited template cavity.

This module only prepares an opt-in quadratic prior. It does not find cavities,
fit a mesh, change registration defaults, or certify anatomical correctness.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .cavityPrior import _frame, _indices
from .geometryRefinement import _meshEdges, _validateMesh
from .localSurfaceRegistration import _array, _scalar


def _sampleSupport(vertexCount, indices, barycentric, sampleWeights):
    indices = np.asarray(indices)
    if (indices.ndim != 2 or not len(indices) or indices.shape[1] not in (1, 2, 3)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= vertexCount):
        raise ValueError('sampleVertexIndices must be valid nonempty integer Mx1/2/3 indices')
    barycentric = _array(barycentric, indices.shape, 'sampleBarycentric')
    weights = _array(sampleWeights, (len(indices),), 'sampleWeights')
    if np.any(weights < 0) or not np.isfinite(weights.sum()):
        raise ValueError('sampleWeights must have nonnegative values and a finite sum')
    included = weights > 0
    if (not np.allclose(barycentric.sum(1), 1., rtol=0, atol=1e-7)
            or np.any(barycentric[included] < 0) or np.any(barycentric[included] > 1)):
        raise ValueError('Sample barycentrics must sum to one; positive-weight rows must be convex')
    keptIndices, keptBary = indices[included], barycentric[included]
    matrix = sparse.csr_matrix((keptBary.ravel(),
        (np.repeat(np.arange(included.sum()), indices.shape[1]), keptIndices.ravel())),
        shape=(int(included.sum()), vertexCount))
    # Duplicate indices in a sample must be combined before squaring: this is
    # the true diagonal of C.T @ W @ C, not the sum of separate barycentric squares.
    matrix.sum_duplicates()
    support = np.asarray(matrix.power(2).T @ weights[included]).ravel()
    if not np.isfinite(support).all():
        raise ValueError('Observation support overflowed')
    return support, int(included.sum())


def _harmonicBoundaryOperator(vertexCount, faces, rim, interior):
    members = np.r_[rim, interior]
    memberMask = np.zeros(vertexCount, bool)
    memberMask[members] = True
    interiorMask = np.zeros(vertexCount, bool)
    interiorMask[interior] = True
    allEdges = _meshEdges(faces)
    touchesInterior = interiorMask[allEdges].any(1)
    if np.any(~memberMask[allEdges[touchesInterior]].all(1)):
        raise ValueError('Cavity interior connects outside the supplied rim and interior region')
    patchFaces = faces[memberMask[faces].all(1)]
    if not len(patchFaces) or not np.array_equal(np.unique(patchFaces), np.sort(members)):
        raise ValueError('Induced cavity patch must cover exactly its rim and interior vertices')
    mapping = np.full(vertexCount, -1, int)
    mapping[members] = np.arange(len(members))
    localFaces = mapping[patchFaces]
    edges = np.sort(np.concatenate([localFaces[:, [0, 1]], localFaces[:, [1, 2]], localFaces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    if (np.any(counts > 2) or not np.array_equal(np.unique(boundary), np.arange(len(rim)))
            or np.any(np.bincount(boundary.ravel(), minlength=len(members))[:len(rim)] != 2)):
        raise ValueError('Induced cavity patch must have a manifold boundary matching its rim')
    adjacency = sparse.coo_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(members), len(members))).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('Cavity patch must be connected to its rim')
    boundaryAdjacency = sparse.coo_matrix((np.ones(2*len(boundary)),
        (np.r_[boundary[:, 0], boundary[:, 1]], np.r_[boundary[:, 1], boundary[:, 0]])),
        shape=(len(rim), len(rim))).tocsr()
    if (connected_components(boundaryAdjacency, directed=False, return_labels=False) != 1
            or len(members)-len(edges)+len(localFaces) != 1):
        raise ValueError('Cavity patch must be a single topological disk')
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    interiorBlock = laplacian[len(rim):, len(rim):].tocsc()
    rimBlock = laplacian[len(rim):, :len(rim)].toarray()
    try:
        extension = splu(interiorBlock).solve(-rimBlock)
    except RuntimeError as error:
        raise ValueError('Cavity harmonic interior is not anchored to its boundary') from error
    partitionError = float(np.max(np.abs(extension.sum(1)-1.)))
    harmonicError = float(np.max(np.abs(interiorBlock @ extension+rimBlock)))
    if (not np.isfinite(extension).all() or extension.min() < -1e-10
            or partitionError > 1e-10 or harmonicError > 1e-9):
        raise ValueError('Harmonic boundary coordinates are not a finite convex partition')
    rows = np.repeat(np.arange(len(interior)), len(rim))
    operator = sparse.csr_matrix((np.r_[np.ones(len(interior)), -extension.ravel()],
        (np.r_[np.arange(len(interior)), rows], np.r_[interior, np.tile(rim, len(interior))])),
        shape=(len(interior), vertexCount))
    operator.eliminate_zeros()
    return operator, extension, {'patchVertexCount': len(members), 'patchFaceCount': len(localFaces),
        'patchEdgeCount': len(edges), 'harmonicPartitionMaximumError': partitionError,
        'harmonicEquationMaximumError': harmonicError, 'minimumHarmonicWeight': float(extension.min()),
        'maximumHarmonicWeight': float(extension.max())}


def buildUnsupportedCavityPrior(
    alignedRestVertices, faces, rimVertexIds, interiorVertexIds,
    sampleVertexIndices, sampleBarycentric, sampleWeights, *, inwardDirection,
    depthScale, referenceSupport=None, supportThreshold=.15, targetDepthFraction=1.,
):
    """Build a frozen relative-depth quadratic prior without locking rest XYZ.

    Vertex arrays must be in the final registration coordinate frame. ``faces``
    is the full client topology; supplied region IDs must identify one audited
    closed cavity disk, whose interior has no neighbors outside that disk.
    This validates topology, not the anatomical identity of the selected region.

    Observation indices/barycentrics locate data samples on this CLIENT mesh,
    not UFF vertices. Positive-weight samples require convex barycentrics; zero
    weights exclude finite legacy extrapolations. Support is the diagonal of
    C.T @ diag(sampleWeights) @ C. To represent registerDirectTemplate's current
    objective, pass 1 for valid observations and 0 for excluded observations;
    that solver presently uses mapping weights only as a validity mask.

    The reference support defaults to the median positive rim support. Supply
    a positive explicit reference when the rim is unobserved. For interior i,
    a_i = clip(1-support_i/(referenceSupport*supportThreshold), 0, 1).
    Only positive template relative depths receive nonzero a_i. These values
    are mathematical data influence, not probabilities or independent views.

    Harmonic coordinates W interpolate boundary motion, and H V = V_I-W V_B.
    With the frozen unit inward direction n, A vec(V) = (H V) dot n, where vec
    is vertex-major xyz order. Targets t equal targetDepthFraction*A vec(rest).
    The returned weightedOperator/weightedTargets encode exactly

        ||weightedOperator @ vec(V)-weightedTargets||^2
        = sum_i a_i*(A_i vec(V)-t_i)^2 / (M*depthScale^2).

    M is ALL supplied interior vertices, never sum(a_i), so confidence amplitude
    is retained. A caller adds its explicit depthPriorWeight to this separate
    objective; no legacy data/smooth normalization is altered. To use custom
    targets, rebuild weightedTargets = rowScales*customTargetDepths.

    ``targetDepthFraction`` in (0,1] retains that fraction of TEMPLATE depth;
    it is not a fraction of missing depth. A baseline-plus-missing-depth target
    must be constructed explicitly by the caller. All lengths share mesh units.
    Translation and harmonic boundary displacement leave residuals unchanged.
    The fixed direction and template offsets do not adapt to local rotations,
    subject-specific cavity shape, or depth scaling during registration.
    """
    rest, triangles = _validateMesh(alignedRestVertices, faces)
    rim = _indices(rimVertexIds, len(rest), 'rimVertexIds').copy()
    interior = _indices(interiorVertexIds, len(rest), 'interiorVertexIds').copy()
    if len(rim) < 3 or np.intersect1d(rim, interior).size:
        raise ValueError('Cavity requires disjoint interior vertices and a rim with at least three vertices')
    lengthScale = _scalar(depthScale, 'depthScale', positive=True)
    threshold = _scalar(supportThreshold, 'supportThreshold', positive=True)
    fraction = _scalar(targetDepthFraction, 'targetDepthFraction', positive=True)
    if threshold > 1 or fraction > 1:
        raise ValueError('supportThreshold and targetDepthFraction must lie in (0, 1]')
    inward = _array(inwardDirection, (3,), 'inwardDirection')
    inwardLength = float(np.linalg.norm(inward))
    if not np.isfinite(inwardLength) or inwardLength <= 1e-12:
        raise ValueError('inwardDirection must be a finite nonzero direction')
    inward = inward/inwardLength
    _, planeNormal, rimRadius, rimPlaneRms = _frame(rest[rim], inward)
    relative, extension, topology = _harmonicBoundaryOperator(len(rest), triangles, rim, interior)
    support, validSampleCount = _sampleSupport(len(rest), sampleVertexIndices, sampleBarycentric, sampleWeights)
    positiveRim = support[rim][support[rim] > 0]
    if referenceSupport is None:
        if not len(positiveRim):
            raise ValueError('No positive rim support; supply an explicit positive referenceSupport')
        reference = float(np.median(positiveRim))
        referenceMethod = 'medianPositiveRimDataNormalEquationDiagonal'
    else:
        reference = _scalar(referenceSupport, 'referenceSupport', positive=True)
        referenceMethod = 'explicitCallerSupportReference'
    supportRatio = support[interior]/reference
    confidence = np.clip(1.-supportRatio/threshold, 0., 1.)
    relativeRest = relative @ rest
    restDepth = relativeRest @ inward
    if not np.isfinite(restDepth).all() or not np.isfinite(supportRatio).all():
        raise ValueError('Relative depth or support normalization overflowed')
    positiveDepth = restDepth > 0
    priorWeights = confidence*positiveDepth
    targetDepths = fraction*restDepth
    depthOperator = sparse.kron(relative, inward.reshape(1, 3), format='csr')
    rowScales = np.sqrt(priorWeights/len(interior))/lengthScale
    if not np.isfinite(rowScales).all():
        raise ValueError('Prior scaling overflowed; check depthScale units')
    weightedOperator = (sparse.diags(rowScales) @ depthOperator).tocsr()
    weightedOperator.eliminate_zeros()
    weightedTargets = rowScales*targetDepths
    if not np.isfinite(weightedTargets).all() or not np.isfinite(weightedOperator.data).all():
        raise ValueError('Weighted cavity objective is nonfinite; check length units')
    return {'relativeOperator': relative, 'harmonicBoundaryWeights': extension,
        'depthOperator': depthOperator, 'targetDepths': targetDepths,
        'restRelativeDepths': restDepth, 'restRelativeOffsets': relativeRest,
        'priorWeights': priorWeights, 'rowScales': rowScales,
        'weightedOperator': weightedOperator, 'weightedTargets': weightedTargets,
        'vertexSupport': support, 'interiorSupportRatio': supportRatio,
        'rimVertexIds': rim, 'interiorVertexIds': interior, 'inwardDirection': inward,
        'diagnostics': {'method': 'unsupportedInteriorHarmonicRelativeDepthPrior',
            'qualityAccepted': False, 'registrationDefaultsChanged': False,
            'sourceDataSupport': 'diagonal(C.T @ diag(sampleWeights) @ C)',
            'validSampleCount': validSampleCount, 'referenceSupport': reference,
            'referenceSupportMethod': referenceMethod, 'positiveRimSupportCount': len(positiveRim),
            'supportThreshold': threshold, 'interiorCount': len(interior),
            'unsupportedInteriorCount': int((priorWeights > 0).sum()),
            'zeroDataSupportInteriorCount': int((support[interior] == 0).sum()),
            'nonpositiveTemplateDepthCount': int((~positiveDepth).sum()),
            'priorWeightSum': float(priorWeights.sum()), 'normalizationInteriorCount': len(interior),
            'depthScale': lengthScale, 'targetDepthFraction': fraction,
            'rimPlaneRmsOverRadius': rimPlaneRms/rimRadius,
            'inwardRimPlaneNormalDot': float(inward @ planeNormal),
            'topology': topology, 'flatteningOrder': 'vertexMajorXYZ',
            'objective': 'depthPriorWeight * sum(priorWeights * (depthOperator @ vertices.ravel() - targetDepths)^2) / (interiorCount * depthScale^2)',
            'limitations': [
                'Template depth is a shape prior; support does not certify visibility, independent views, or anatomical correctness.',
                'Rim coordinates remain variable; the prior is relative and does not lock aligned-rest XYZ.',
                'The inward direction and harmonic coordinates are frozen; local rotation and subject depth scale are not inferred.',
                'Target depth fraction is a fraction of template depth, not a fraction of missing depth.',
                'Additional legacy absolute Laplacian smoothing still competes with this prior.',
                'No triangle, collision, rig, texture or photo-likeness acceptance is performed.']}}
