"""Protect unsupported template regions while retaining supported shape fitting.

Support means a nonzero column of the supplied fixed sample matrix. It is
binary availability, not confidence. No anatomy, sample ID or failed element
is inferred here; separately guarded domains are supplied explicitly.
"""
from __future__ import annotations

from numbers import Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .meshShapeConstraints import _triangleMetrics, _crossGradient


DEFAULT_POLICY = {
    'maximumTransitionDisplacement': .5,
    'maximumProtectedDisplacement': .1,
    'minimumEdgeRatio': .9,
    'maximumEdgeRatio': 1.1,
    'minimumAreaRatio': .8,
    'maximumAreaRatio': 1.25,
    'minimumNormalCosine': .95,
}
_LIMITS = {'vertices': 50_000, 'faces': 100_000, 'samples': 50_000, 'sampleNonzeros': 1_000_000}


def _vertices(value, name, shape=None):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1:] != (3,) or not 3 <= len(raw) <= _LIMITS['vertices']
            or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all()
            or (shape is not None and raw.shape != shape)):
        raise ValueError(f'{name} must be a finite real Nx3 array matching the bounded topology')
    return np.array(raw, dtype=np.float64, copy=True)


def _ids(value, name, count):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or np.any(raw < 0)
            or np.any(raw >= count) or len(np.unique(raw)) != len(raw)):
        raise ValueError(f'{name} must contain unique valid integer vertex IDs')
    return np.array(raw, dtype=np.int64, copy=True)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def buildSupportedShapeConstraintSpec(
    referenceVertices, faces, sampleMatrix, staticVertexIds, *,
    delegatedVertexIds, protectedVertexIds, lengthScale, policy=None,
    safetyReferenceVertices=None,
):
    """Freeze support protection, fixed coordinates and local shape constraints.

    The caller retains ORIGINAL C: a finite, nonnegative, affine sparse MxN
    sample matrix, already filtered to its valid observation rows. Explicit
    zeros and duplicate CSR entries are canonicalized, without renormalizing
    any row. A supported vertex has at least one remaining nonzero coefficient;
    the threshold is exactly zero and no confidence is estimated. Repeating or
    permuting observations does not change support or the resulting domains.

    Unsupported free vertices outside delegatedVertexIds are HARD FIXED at
    referenceVertices. Components are derived on the induced face-edge graph,
    including isolated vertices. Their one-ring supported, nonstatic neighbors
    outside the delegated domain form transitionVertexIds. Protected vertices
    are an explicit caller semantic support (e.g. original eye/mouth bindings),
    never chosen from candidate errors. Both sets form retained controls.

    Every edge/face touching a newly fixed unsupported vertex is guarded unless
    it touches the caller's delegated domain. Its remaining endpoints are fixed
    or transition controls. This is a LOCAL constraint set: unguarded exterior
    edges/faces and complete collision/data/asset checks remain mandatory at
    the caller. In particular, delegation does not waive any acceptance gate.

    Displacement and shape bounds refer to safetyReferenceVertices (defaults to
    referenceVertices). Pass the ORIGINAL pureDirect safety reference when the
    starting reference already includes accepted cavity fitting, so limits are
    not reset or accumulated. All hard-fixed positions retain referenceVertices
    exactly; original static positions must agree between both references.

    lengthScale is the target-frame size of ONE physical unit (normally 1 mm).
    Default limits are transition .5 units, protected .1 units, reference edge
    ratios [.9,1.1], area ratios [.8,1.25] and normal cosine >=.95. Protected
    bounds take precedence when a point also lies in the transition ring.

    Pass hardFixedVertexIds to exact coordinate elimination and controlVertexIds
    to the retained solver, together with all controls of delegated guards.
    This builder never solves, blends, changes C or mutates coordinates. An
    empty control set is valid data but cannot be passed blindly to a solver
    requiring free controls. Solver workspace budgets must be checked by caller.
    """
    reference = _vertices(referenceVertices, 'referenceVertices')
    safety = reference.copy() if safetyReferenceVertices is None else _vertices(
        safetyReferenceVertices, 'safetyReferenceVertices', reference.shape)
    count = len(reference)
    rawFaces = np.asarray(faces)
    if (rawFaces.ndim != 2 or rawFaces.shape[1:] != (3,) or not 1 <= len(rawFaces) <= _LIMITS['faces']
            or rawFaces.dtype.kind not in 'iu' or np.any(rawFaces < 0) or np.any(rawFaces >= count)
            or np.any(np.diff(np.sort(rawFaces, axis=1), axis=1) == 0)):
        raise ValueError('faces must be bounded valid integer triangles without repeated corners')
    triangles = np.array(rawFaces, dtype=np.int64, copy=True)
    static = _ids(staticVertexIds, 'staticVertexIds', count)
    delegated = _ids(delegatedVertexIds, 'delegatedVertexIds', count)
    protected = _ids(protectedVertexIds, 'protectedVertexIds', count)
    if not np.array_equal(reference[static], safety[static]):
        raise ValueError('Original static coordinates must agree exactly between references')
    if (not sparse.issparse(sampleMatrix) or sampleMatrix.format not in ('csr', 'csc', 'coo')
            or len(sampleMatrix.shape) != 2 or sampleMatrix.shape[1] != count
            or sampleMatrix.shape[0] > _LIMITS['samples'] or sampleMatrix.nnz > _LIMITS['sampleNonzeros']
            or sampleMatrix.dtype.kind not in 'fiu' or not np.isfinite(sampleMatrix.data).all()
            or np.any(sampleMatrix.data < 0)):
        raise ValueError('sampleMatrix must be a bounded finite nonnegative sparse MxN surface sampling matrix')
    sampling = sparse.csr_matrix(sampleMatrix, dtype=np.float64, copy=True)
    sampling.sum_duplicates(); sampling.eliminate_zeros()
    if (not np.isfinite(sampling.data).all()
            or np.max(abs(np.asarray(sampling.sum(1)).ravel() - 1), initial=0) > 1e-12):
        raise ValueError('sampleMatrix rows must remain affine without renormalization')
    scale = _positive(lengthScale, 'lengthScale')
    effective = DEFAULT_POLICY.copy()
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(effective):
            raise ValueError('Unknown supported shape constraint policy entries')
        effective.update(policy)
    for name in effective:
        if name == 'minimumNormalCosine':
            value = effective[name]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not -1 <= value <= 1):
                raise ValueError('minimumNormalCosine must be finite in [-1,1]')
            effective[name] = float(value)
        else:
            effective[name] = _positive(effective[name], name)
    if (not effective['minimumEdgeRatio'] <= 1 <= effective['maximumEdgeRatio']
            or not effective['minimumAreaRatio'] <= 1 <= effective['maximumAreaRatio']
            or effective['maximumProtectedDisplacement'] > effective['maximumTransitionDisplacement']):
        raise ValueError('Reference-inclusive shape bounds and a no-looser protected displacement are required')
    supported = np.zeros(count, bool); supported[sampling.indices] = True
    staticMask = np.zeros(count, bool); staticMask[static] = True
    delegatedMask = np.zeros(count, bool); delegatedMask[delegated] = True
    protectedMask = np.zeros(count, bool); protectedMask[protected] = True
    unsupported = ~supported & ~staticMask & ~delegatedMask
    unsupportedIds = np.flatnonzero(unsupported)
    fixedMask = staticMask | unsupported
    # These rows are removed by exact elimination, but their inherited values
    # must not silently start outside the original physical safety envelope.
    referenceShift = np.linalg.norm(reference - safety, axis=1) / scale
    if (np.any(referenceShift[unsupported] > effective['maximumTransitionDisplacement'] + 1e-8)
            or np.any(referenceShift[protected] > effective['maximumProtectedDisplacement'] + 1e-8)):
        raise ValueError('Fixed or protected reference positions exceed original displacement limits')
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
        triangles[:, [2, 0]])), axis=1), axis=0)
    incidentEdges = unsupported[edges].any(1)
    transitionMask = np.zeros(count, bool)
    transitionMask[np.unique(edges[incidentEdges])] = True
    transitionMask &= supported & ~fixedMask & ~delegatedMask
    controls = np.flatnonzero((transitionMask | protectedMask) & ~fixedMask)
    guardedEdges = incidentEdges & ~delegatedMask[edges].any(1)
    incidentFaces = unsupported[triangles].any(1)
    guardedFaces = incidentFaces & ~delegatedMask[triangles].any(1)
    faceIds = np.flatnonzero(guardedFaces)
    shapeEdges, shapeFaces = edges[guardedEdges], triangles[guardedFaces]
    allowed = fixedMask.copy(); allowed[controls] = True
    if not allowed[shapeEdges].all() or not allowed[shapeFaces].all():
        raise ValueError('Local guard domain escaped fixed coordinates and retained controls')
    localEdges = edges[unsupported[edges].all(1)]
    localIndices = np.searchsorted(unsupportedIds, localEdges)
    graph = sparse.csr_matrix((np.ones(2 * len(localEdges)),
        (localIndices.ravel(), localIndices[:, ::-1].ravel())), shape=(len(unsupportedIds),) * 2)
    componentCount, componentLabels = connected_components(graph, directed=False)
    components = [unsupportedIds[componentLabels == i].copy() for i in range(componentCount)]
    lengths = np.linalg.norm(safety[shapeEdges[:, 1]] - safety[shapeEdges[:, 0]], axis=1)
    metrics = _triangleMetrics(safety, shapeFaces)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Guarded reference edges must have positive finite length')
    limits = np.where(protectedMask[controls], effective['maximumProtectedDisplacement'],
        effective['maximumTransitionDisplacement']) * scale
    spec = {'schemaVersion': 'supportedShapeConstraintsV1', 'vertexCount': count,
        'referenceVertices': reference, 'safetyReferenceVertices': safety, 'faces': triangles,
        'sampleMatrix': sampling, 'supportedMask': supported, 'unsupportedFreeMask': unsupported,
        'unsupportedComponentVertexIds': components, 'originalStaticVertexIds': static,
        'delegatedVertexIds': delegated, 'protectedVertexIds': protected,
        'hardFixedVertexIds': np.flatnonzero(fixedMask), 'transitionVertexIds': np.flatnonzero(transitionMask),
        'controlVertexIds': controls, 'edgeVertexIds': shapeEdges, 'faceVertexIds': shapeFaces, 'faceIds': faceIds,
        'edgeLower': effective['minimumEdgeRatio'] * lengths, 'edgeUpper': effective['maximumEdgeRatio'] * lengths,
        'areaLower': effective['minimumAreaRatio'] * metrics['area'],
        'areaUpper': effective['maximumAreaRatio'] * metrics['area'], 'safetyReferenceNormals': metrics['normal'],
        'displacementLimits': limits, 'lengthScale': scale, 'policy': effective,
        'diagnostics': {'method': 'hardUnsupportedDomainWithSupportedOneRingTransition',
            'supportedVertexCount': int(supported.sum()), 'unsupportedFreeVertexCount': len(unsupportedIds),
            'unsupportedComponentCount': componentCount, 'unsupportedComponentSizes': [len(ids) for ids in components],
            'newHardFixedVertexCount': len(unsupportedIds), 'totalHardFixedVertexCount': int(fixedMask.sum()),
            'delegatedUnsupportedFreeCount': int((~supported & ~staticMask & delegatedMask).sum()),
            'transitionVertexCount': int(transitionMask.sum()), 'controlVertexCount': len(controls),
            'edgeCount': len(shapeEdges), 'faceCount': len(shapeFaces),
            'delegatedIncidentEdgesNotGuardedHere': int((incidentEdges & ~guardedEdges).sum()),
            'delegatedIncidentFacesNotGuardedHere': int((incidentFaces & ~guardedFaces).sum()),
            'supportDefinition': 'A column has at least one exactly nonzero coefficient after sparse duplicate summation and zero elimination',
            'supportThreshold': 0., 'supportMeaning': 'Binary sampling availability, not confidence',
            'transitionRule': 'One face-edge ring of supported nonstatic nondelegated neighbors of newly fixed unsupported vertices',
            'fixedCoordinateFrame': 'Exact caller referenceVertices', 'safetyCoordinateFrame': 'Original caller safetyReferenceVertices',
            'sampleMatrixRenormalized': False, 'sourcePriorChanged': False, 'candidateDependentSelection': False,
            'geometryFitted': False, 'qualityAccepted': False, 'allExteriorConstraintsIncluded': False,
            'limitations': ['Caller must eliminate hardFixedVertexIds exactly; this evaluator alone does not enforce fixed coordinates.',
                'Unlisted exterior edge, area, normal, global displacement, data, collision, binding and likeness gates remain separate and unchanged.',
                'Unsupported geometry is preserved from a reference, not reconstructed as anatomical truth.',
                'Delegation only assigns responsibility to another guard; it never waives final safety checks.']}}
    values = evaluateSupportedShapeConstraints(reference, spec, withJacobian=False)['values']
    if len(values) and values.min() < -1e-8:
        raise ValueError('Starting reference violates the unchanged original safety reference bounds')
    return spec


def evaluateSupportedShapeConstraints(vertices, guardSpec, withJacobian=True):
    """Return dimensionless ``values >= 0`` and optional full-domain CSR J.

    Hard-fixed coordinates are enforced by the caller's exact elimination, not
    a penalty. Sparse Jacobian support stays in controlVertexIds plus fixed IDs.
    Empty local domains return zero rows; zero-area guarded faces are rejected.
    """
    if not isinstance(guardSpec, dict) or guardSpec.get('schemaVersion') != 'supportedShapeConstraintsV1':
        raise ValueError('Expected a supported shape constraint specification')
    if type(withJacobian) is not bool:
        raise ValueError('withJacobian must be boolean')
    spec = guardSpec
    current = _vertices(vertices, 'vertices', (spec['vertexCount'], 3))
    scale = spec['lengthScale']
    normalized = current / scale
    edges, faces, controls = spec['edgeVertexIds'], spec['faceVertexIds'], spec['controlVertexIds']
    edgeVectors = normalized[edges[:, 1]] - normalized[edges[:, 0]]
    lengths = np.linalg.norm(edgeVectors, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Guarded candidate edges must have positive finite length')
    metrics = _triangleMetrics(normalized, faces)
    values, rows, columns, entries, groups = [], [], [], [], {}
    offset = 0

    def append(name, value, ids, gradient):
        nonlocal offset
        value = np.asarray(value, float)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError('Constraint values must remain finite')
        groups[name] = {'start': offset, 'stop': offset + len(value)}
        values.append(value)
        if withJacobian:
            if gradient.shape != (*ids.shape, 3) or not np.isfinite(gradient).all():
                raise ValueError('Constraint Jacobian must remain finite')
            rows.append(np.repeat(np.arange(len(value)) + offset, ids.shape[1] * 3))
            columns.append((3 * ids[:, :, None] + np.arange(3)).ravel())
            entries.append(gradient.ravel())
        offset += len(value)

    edgeGradient = np.stack((-edgeVectors / lengths[:, None], edgeVectors / lengths[:, None]), axis=1) / scale if withJacobian else None
    append('edgeLower', lengths - spec['edgeLower'] / scale, edges, edgeGradient)
    append('edgeUpper', spec['edgeUpper'] / scale - lengths, edges, -edgeGradient if withJacobian else None)
    areaGradient = _crossGradient(metrics, .5 * metrics['normal']) / scale if withJacobian else None
    append('areaLower', metrics['area'] - spec['areaLower'] / scale ** 2, faces, areaGradient)
    append('areaUpper', spec['areaUpper'] / scale ** 2 - metrics['area'], faces, -areaGradient if withJacobian else None)
    cosine = np.sum(metrics['normal'] * spec['safetyReferenceNormals'], axis=1)
    normalGradient = None
    if withJacobian:
        crossGradient = (spec['safetyReferenceNormals'] - cosine[:, None] * metrics['normal']) / metrics['crossLength'][:, None]
        normalGradient = _crossGradient(metrics, crossGradient) / scale
    append('normalToSafetyReference', cosine - spec['policy']['minimumNormalCosine'], faces, normalGradient)
    delta = (current[controls] - spec['safetyReferenceVertices'][controls]) / scale
    append('controlDisplacement', (spec['displacementLimits'] / scale) ** 2 - np.sum(delta ** 2, axis=1),
        controls[:, None], -2 * delta[:, None, :] / scale if withJacobian else None)
    result = {'values': np.concatenate(values), 'groups': groups}
    if withJacobian:
        jacobian = sparse.csr_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(offset, current.size))
        jacobian.sum_duplicates(); jacobian.eliminate_zeros()
        if not np.isfinite(jacobian.data).all():
            raise ValueError('Constraint Jacobian overflowed')
        result['jacobian'] = jacobian
    return result
