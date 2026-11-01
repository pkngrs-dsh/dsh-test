"""Optional source-observation differentials for the fixed client topology.

This builder never fits a mesh. Source geometry is model evidence, not measured
anatomical truth; binary coverage is explicitly separate from reliability.
"""
from __future__ import annotations

from numbers import Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


_LIMITS = {'vertices': 50_000, 'faces': 100_000, 'samples': 20_000,
           'graphNodes': 50_000, 'graphNonzeros': 1_000_000,
           'distanceVectorWork': 100_000_000, 'priorNonzeros': 1_000_000,
           'normalProductWork': 20_000_000, 'neighbors': 32}


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} scalar')
    return float(value)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _floatArray(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'fiu' or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite real array of shape {shape}')
    return np.array(result, dtype=np.float64, copy=True)


def _ids(value, shape, bound, name):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in 'iu'
            or np.any(result < 0) or np.any(result >= bound)):
        raise ValueError(f'{name} must have integer shape {shape} with values in [0, {bound})')
    return np.array(result, dtype=np.int64, copy=True)


def _mask(value, count, name):
    result = np.asarray(value)
    if result.shape != (count,) or result.dtype.kind != 'b':
        raise ValueError(f'{name} must be a boolean vector of length {count}')
    return result.copy()


def _faces(value, vertexCount, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or not 1 <= len(raw) <= _LIMITS['faces']:
        raise ValueError(f'{name} must be a nonempty Fx3 array within the face budget')
    result = _ids(raw, raw.shape, vertexCount, name)
    if np.any(np.diff(np.sort(result, axis=1), axis=1) == 0):
        raise ValueError(f'{name} may not repeat a vertex inside a triangle')
    return result


def _binding(faceIds, barycentric, faces, count, valid, name):
    ids = _ids(faceIds, (count,), len(faces), name + 'FaceIds')
    weights = _floatArray(barycentric, (count, 3), name + 'Barycentric')
    if (np.max(abs(weights.sum(1) - 1), initial=0) > 1e-6
            or np.any(weights[valid] < 0) or np.any(weights[valid] > 1)):
        raise ValueError(f'{name} weights must sum to one; valid rows must be convex')
    if name == 'sampleClient' and np.max(abs(weights[valid].sum(1) - 1), initial=0) > 1e-12:
        raise ValueError('Valid client barycentrics must form affine rows without renormalizing the data matrix')
    return ids, weights, faces[ids]


def _rowKey(indices, coefficients):
    return tuple(sorted((int(i), float(w)) for i, w in zip(indices, coefficients) if w != 0))


def _uniqueSamples(valid, clientIndices, clientWeights, sourceIndices, sourceWeights, coverage, reliability):
    groups = {}
    for row in np.flatnonzero(valid):
        key = (_rowKey(clientIndices[row], clientWeights[row]), _rowKey(sourceIndices[row], sourceWeights[row]))
        groups.setdefault(key, []).append(int(row))
    representatives, covered, scores, inputToUnique = [], [], [], np.full(len(valid), -1, np.int64)
    for group in groups.values():
        inputToUnique[group] = len(representatives)
        representatives.append(group[0])
        covered.append(bool(coverage[group].all()))
        scores.append(float(reliability[group].min()))
    return (np.asarray(representatives, np.int64), np.asarray(covered, bool),
            np.asarray(scores, np.float64), inputToUnique)


def _sourceEdges(vertices, faces):
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    areas = np.linalg.norm(cross, axis=1)
    if not np.isfinite(lengths).all() or not np.isfinite(areas).all() or np.any(lengths <= 0) or np.any(areas <= 0):
        raise ValueError('Source mesh must have finite nonzero edges and nondegenerate triangles')
    return edges, lengths


def _sourceGraph(vertices, edges, lengths, samples, sampleTriangles):
    sampleIds = np.repeat(np.arange(len(samples)) + len(vertices), 3)
    cornerIds = sampleTriangles.ravel()
    cornerDistances = np.linalg.norm((samples[:, None] - vertices[sampleTriangles]).reshape(-1, 3), axis=1)
    rows = np.r_[edges[:, 0], edges[:, 1], sampleIds, cornerIds]
    columns = np.r_[edges[:, 1], edges[:, 0], cornerIds, sampleIds]
    values = np.r_[lengths, lengths, cornerDistances, cornerDistances]
    if not np.isfinite(values).all() or len(values) > _LIMITS['graphNonzeros']:
        raise ValueError('Source topology graph exceeds its finite sparse budget')
    # Explicit zero edges attach exact vertex samples. Do NOT eliminate them:
    # scipy.csgraph recognizes stored CSR zero entries as real zero-cost edges.
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(vertices) + len(samples),) * 2)


def buildSourceShapePreservationPrior(
    clientFaces, sampleClientFaceIds, sampleClientBarycentric, targetPoints,
    sourceVertices, sourceFaces, sampleSourceFaceIds, sampleSourceBarycentric, *,
    clientVertexCount, validMask, coverageMask=None, reliabilityWeights=None,
    reliabilityProvenance=None, lengthScale, neighborhoodRadius,
    maxNeighbors=12, minNeighbors=4, weight=1.,
):
    """Build ``weight * ||A @ V.ravel() - t||²`` from fixed source samples.

    C is the ORIGINAL client barycentric sampling matrix. Q=targetPoints must
    match the supplied source triangle bindings in the registration TARGET
    frame; source vertices, Q, radius and lengthScale use those same units.
    No similarity is inferred or applied. Client barycentrics are not changed.

    Distances follow source mesh EDGES plus edges connecting each sample to
    its own source-triangle corners. Thus Euclidean proximity of disconnected
    or folded surfaces cannot create shortcuts between them. These are upper
    approximations of continuous surface geodesics, not exact geodesics. The
    full source topology remains traversable; sample coverage does not certify
    coverage along every path. No anatomical seam or region is guessed.

    Each accepted center has minNeighbors..maxNeighbors covered, positive-
    reliability neighbors inside neighborhoodRadius. Its D row is center
    minus a normalized Gaussian average, exp(-.5*(distance/radius)**2). D has
    zero row sum and acts on the same original samples in C and Q. It is a
    graph differential, not a calibrated mean-curvature estimator. The
    shortest paths and D are frozen before optimization, never rebound on V.
    A cutoff tie is retained or discarded as a WHOLE distance shell; indices
    never break metric ties. Insufficient neighborhoods produce no prior row.

    coverageMask is binary availability, NOT confidence. None means unknown
    coverage, not measured observation support. reliabilityWeights is an
    optional [0,1] engineering weight, and requires descriptive provenance.
    None gives uniform weights, explicitly without a confidence estimate.
    Row reliability is the minimum of its center and contributing neighbors.

        A = kron(diag(sqrt(rowReliability/(3*M))/lengthScale) @ D @ C, I3)
        t = vec(diag(sqrt(rowReliability/(3*M))/lengthScale) @ D @ Q)

    M counts ALL distinct valid source/client binding PAIRS, including those
    with zero reliability or missing coverage. It does not shrink when rows
    are rejected. Lower reliability therefore weakens the cost, never boosts
    remaining rows. Exact duplicate binding pairs are deduplicated; their
    coverage/reliability is conservatively AND/min combined. D columns still
    index the original samples. Uniform observation replication or ordering
    changes do not change the prior objective. This does NOT alter the main
    direct-registration data normalization, which can depend on replication.

    Joint translation/rotation and uniform scaling (including lengthScale and
    radius) preserve the normalized objective. Source float32 barycentric
    partition rounding is allowed when checking Q, but Q itself is retained.
    Source geometry is a prediction, not personal anatomy ground truth. This
    term does not enforce mesh safety, static supports or likeness: use the
    exact solver's hard-static elimination and independent existing gates.

    The result's quadraticPrior can be passed to registerDirectTemplate.
    An empty/inactive prior is a single zero row with weight=0, retaining its
    legacy no-prior solver path. Diagnostics never indicate asset acceptance.
    """
    clientCount = _integer(clientVertexCount, 'clientVertexCount', 3, _LIMITS['vertices'])
    clientTriangles = _faces(clientFaces, clientCount, 'clientFaces')
    rawSource = np.asarray(sourceVertices)
    if rawSource.ndim != 2 or rawSource.shape[1:] != (3,) or not 3 <= len(rawSource) <= _LIMITS['vertices']:
        raise ValueError('sourceVertices must be Nx3 within the vertex budget')
    source = _floatArray(rawSource, rawSource.shape, 'sourceVertices')
    sourceTriangles = _faces(sourceFaces, len(source), 'sourceFaces')
    sourceEdges, sourceEdgeLengths = _sourceEdges(source, sourceTriangles)
    rawTargets = np.asarray(targetPoints)
    if rawTargets.ndim != 2 or rawTargets.shape[1:] != (3,) or not 1 <= len(rawTargets) <= _LIMITS['samples']:
        raise ValueError('targetPoints must be Mx3 within the sample budget')
    targets = _floatArray(rawTargets, rawTargets.shape, 'targetPoints')
    count = len(targets)
    valid = _mask(validMask, count, 'validMask')
    coverage = np.ones(count, bool) if coverageMask is None else _mask(coverageMask, count, 'coverageMask')
    if reliabilityWeights is None:
        if reliabilityProvenance is not None:
            raise ValueError('reliabilityProvenance without reliabilityWeights is ambiguous')
        reliability = np.ones(count)
        provenance = 'Uniform engineering weights; no reliability/confidence estimate supplied'
    else:
        if not isinstance(reliabilityProvenance, str) or not reliabilityProvenance.strip():
            raise ValueError('Explicit reliabilityWeights require nonempty reliabilityProvenance')
        reliability = _floatArray(reliabilityWeights, (count,), 'reliabilityWeights')
        if np.any(reliability < 0) or np.any(reliability > 1):
            raise ValueError('reliabilityWeights must lie in [0,1], not be silently clipped')
        provenance = reliabilityProvenance
    scale = _scalar(lengthScale, 'lengthScale', positive=True)
    radius = _scalar(neighborhoodRadius, 'neighborhoodRadius', positive=True)
    priorWeight = _scalar(weight, 'weight')
    maximum = _integer(maxNeighbors, 'maxNeighbors', 1, _LIMITS['neighbors'])
    minimum = _integer(minNeighbors, 'minNeighbors', 1, maximum)
    clientIds, clientWeights, clientCorners = _binding(sampleClientFaceIds, sampleClientBarycentric,
        clientTriangles, count, valid, 'sampleClient')
    sourceIds, sourceWeights, sourceCorners = _binding(sampleSourceFaceIds, sampleSourceBarycentric,
        sourceTriangles, count, valid, 'sampleSource')
    expected = np.sum(source[sourceCorners] * sourceWeights[..., None], axis=1)
    partitionError = abs(sourceWeights.sum(1) - 1)
    coordinateScale = max(float(abs(source).max(initial=0)), float(abs(targets).max(initial=0)), scale)
    tolerance = (partitionError + 256 * np.finfo(float).eps) * coordinateScale
    if np.any(np.max(abs(expected[valid] - targets[valid]), axis=1) > tolerance[valid]):
        raise ValueError('targetPoints do not match their source bindings in the same coordinate frame')
    sampleMatrix = sparse.csr_matrix((clientWeights.ravel(),
        (np.repeat(np.arange(count), 3), clientCorners.ravel())), shape=(count, clientCount))
    sampleMatrix.eliminate_zeros()
    representatives, uniqueCoverage, uniqueReliability, inputToUnique = _uniqueSamples(
        valid, clientCorners, clientWeights, sourceCorners, sourceWeights, coverage, reliability)
    uniqueCount = len(representatives)
    eligible = uniqueCoverage & (uniqueReliability > 0)
    nodeCount = len(source) + uniqueCount
    if (nodeCount > _LIMITS['graphNodes'] or int(eligible.sum()) * nodeCount > _LIMITS['distanceVectorWork']
            or 6 * len(sourceTriangles) + 6 * uniqueCount > 2 * _LIMITS['graphNonzeros']):
        raise ValueError('Source neighborhood construction exceeds the bounded graph/work budget')
    rows, columns, values, centers, rowReliability, neighborRecords, distanceRecords = [], [], [], [], [], [], []
    insufficient, tieLimited = 0, 0
    if np.any(eligible):
        graph = _sourceGraph(source, sourceEdges, sourceEdgeLengths,
            targets[representatives], sourceCorners[representatives])
        distanceTolerance = radius * 1e-10
        for center in np.flatnonzero(eligible):
            distances = dijkstra(graph, directed=True, indices=int(len(source) + center), limit=radius + distanceTolerance)[len(source):]
            allowed = eligible & np.isfinite(distances) & (distances <= radius + distanceTolerance)
            allowed[center] = False
            neighbors = np.flatnonzero(allowed)
            if len(neighbors) > maximum:
                cutoff = np.partition(distances[neighbors], maximum - 1)[maximum - 1]
                inclusive = neighbors[distances[neighbors] <= cutoff + distanceTolerance]
                if len(inclusive) <= maximum:
                    neighbors = inclusive
                else:
                    neighbors = neighbors[distances[neighbors] < cutoff - distanceTolerance]
                    tieLimited += 1
            if len(neighbors) < minimum:
                insufficient += 1
                continue
            neighborWeights = np.exp(-.5 * (distances[neighbors] / radius) ** 2)
            neighborWeights /= neighborWeights.sum()
            row = len(centers)
            rows.extend([row] * (len(neighbors) + 1))
            columns.extend([int(representatives[center]), *representatives[neighbors].tolist()])
            values.extend([1., *(-neighborWeights).tolist()])
            centers.append(int(representatives[center]))
            rowReliability.append(float(min(uniqueReliability[center], uniqueReliability[neighbors].min())))
            neighborRecords.append(representatives[neighbors].copy())
            distanceRecords.append(distances[neighbors].copy())
    differential = sparse.csr_matrix((values, (rows, columns)), shape=(len(centers), count))
    reliabilityRows = np.asarray(rowReliability, float)
    rowScales = np.sqrt(reliabilityRows / (3 * max(uniqueCount, 1))) / scale
    scalarOperator = (sparse.diags(rowScales) @ differential @ sampleMatrix).tocsr()
    scalarOperator.sum_duplicates(); scalarOperator.eliminate_zeros()
    rowCounts = np.diff(scalarOperator.indptr).astype(float)
    if (3 * scalarOperator.nnz > _LIMITS['priorNonzeros']
            or 3 * np.dot(rowCounts, rowCounts) > _LIMITS['normalProductWork']):
        raise ValueError('Source differential prior exceeds the direct solver sparse product budget')
    weightedOperator = sparse.kron(scalarOperator, sparse.eye(3), format='csr')
    weightedTargets = ((differential @ targets) * rowScales[:, None]).ravel()
    if not np.isfinite(weightedOperator.data).all() or not np.isfinite(weightedTargets).all():
        raise ValueError('Source prior overflowed; check coordinate and length units')
    active = bool(len(centers) and priorWeight > 0
                  and (scalarOperator.nnz or np.any(weightedTargets != 0)))
    if not active:
        quadraticPrior = {'operator': sparse.csr_matrix((1, 3 * clientCount)), 'targets': np.zeros(1), 'weight': 0.}
    else:
        quadraticPrior = {'operator': weightedOperator, 'targets': weightedTargets, 'weight': priorWeight}
    return {'quadraticPrior': quadraticPrior, 'sampleMatrix': sampleMatrix,
        'differentialOperator': differential, 'weightedOperator': weightedOperator,
        'weightedTargets': weightedTargets, 'rowScales': rowScales,
        'rowReliability': reliabilityRows, 'centerSampleIds': np.asarray(centers, np.int64),
        'neighborSampleIds': neighborRecords, 'neighborDistances': distanceRecords,
        'uniqueRepresentativeSampleIds': representatives, 'inputToUniqueSample': inputToUnique,
        'diagnostics': {'method': 'sourceTopologySampleDifferentialPreservation', 'active': active,
            'requestedWeight': priorWeight, 'effectiveWeight': quadraticPrior['weight'],
            'sampleCount': count, 'validSampleCount': int(valid.sum()), 'uniqueValidSampleCount': uniqueCount,
            'normalizationSampleCount': uniqueCount, 'duplicateValidSampleCount': int(valid.sum()) - uniqueCount,
            'coveredUniqueValidCount': int(uniqueCoverage.sum()), 'eligibleCenterCount': int(eligible.sum()),
            'acceptedCenterCount': len(centers), 'insufficientNeighborhoodCount': insufficient,
            'tieLimitedNeighborhoodCount': tieLimited, 'neighborhoodRadius': radius,
            'minNeighbors': minimum, 'maxNeighbors': maximum, 'lengthScale': scale,
            'coverageMeaning': 'Binary availability only; not reliability or calibrated confidence',
            'coverageProvided': coverageMask is not None, 'reliabilityProvided': reliabilityWeights is not None,
            'reliabilityProvenance': provenance, 'rowReliabilityRule': 'minimum of center and positive-contribution neighbors',
            'normalization': 'sqrt(rowReliability/(3*allDistinctValidBindingPairs))/lengthScale',
            'duplicateRule': 'exact source/client sparse binding pairs; coverage AND, reliability minimum',
            'neighborMetric': 'source topology edges plus within-source-triangle sample attachments; radius-limited shortest paths',
            'candidateRebinding': False, 'targetsAreTemplateDifferentials': False,
            'columnOrder': 'clientVertexMajorXYZ', 'rowOrder': 'acceptedSampleMajorXYZ',
            'coordinateFrame': 'callerProvidedRegistrationTargetFrame', 'sparseLimits': dict(_LIMITS),
            'normalProductWork': int(3 * np.dot(rowCounts, rowCounts)),
            'clientVertexOrderChanged': False, 'geometryFitted': False, 'qualityAccepted': False,
            'staticConstraintEnforcement': 'caller must use exact registration hard-static elimination',
            'limitations': ['Source reconstruction is learned evidence, not anatomical ground truth.',
                'No reliability is inferred from binary coverage; uniform weights are not confidence.',
                'Geodesic paths use the complete source topology and are not required to be covered along every edge.',
                'Same-source-point but different client bindings remain separate constraints; only exact binding pairs are deduplicated.',
                'Source/data normalization outside this prior is not changed.',
                'No edge, area, normal, collision, binding or photo-likeness safety acceptance is performed.']}}
