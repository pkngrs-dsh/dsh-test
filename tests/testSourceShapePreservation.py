"""Source differential contracts on synthetic surfaces, without real case IDs."""
from copy import deepcopy
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import floyd_warshall

import faceAppearance.sourceShapePreservation as module
from faceAppearance.sourceShapePreservation import buildSourceShapePreservationPrior
from faceAppearance.directTemplateRegistration import registerDirectTemplate


def grid(width=5, height=5):
    yy, xx = np.meshgrid(np.arange(height, dtype=float), np.arange(width, dtype=float), indexing='ij')
    vertices = np.column_stack((xx.ravel(), yy.ravel(), .17 * np.sin(xx.ravel()) * np.cos(yy.ravel() * .6)))
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            faces += [[i, i + 1, i + width], [i + 1, i + width + 1, i + width]]
    return vertices, np.asarray(faces)


def vertexBindings(vertices, faces):
    ids, weights = [], []
    for vertex in range(len(vertices)):
        index = int(np.flatnonzero(np.any(faces == vertex, axis=1))[0])
        ids.append(index)
        weights.append((faces[index] == vertex).astype(float))
    return np.asarray(ids), np.asarray(weights)


def problem(*, centroid=False):
    vertices, faces = grid()
    if centroid:
        faceIds = np.arange(len(faces))
        bary = np.full((len(faces), 3), 1. / 3)
    else:
        faceIds, bary = vertexBindings(vertices, faces)
    targets = (vertices[faces[faceIds]] * bary[..., None]).sum(1)
    return {'clientFaces': faces.copy(), 'sampleClientFaceIds': faceIds.copy(),
        'sampleClientBarycentric': bary.copy(), 'targetPoints': targets,
        'sourceVertices': vertices.copy(), 'sourceFaces': faces.copy(),
        'sampleSourceFaceIds': faceIds.copy(), 'sampleSourceBarycentric': bary.copy(),
        'clientVertexCount': len(vertices), 'validMask': np.ones(len(targets), bool),
        'coverageMask': np.ones(len(targets), bool), 'lengthScale': 1.3,
        'neighborhoodRadius': 2.6, 'maxNeighbors': 12, 'minNeighbors': 4, 'weight': 1.7}


def loss(prior, vertices):
    term = prior['quadraticPrior']
    residual = term['operator'] @ vertices.ravel() - term['targets']
    return float(term['weight'] * residual @ residual)


def independentSourceDistances(args):
    source, faces = args['sourceVertices'], args['sourceFaces']
    sample = args['targetPoints']
    n = len(source) + len(sample)
    graph = np.full((n, n), np.inf)
    np.fill_diagonal(graph, 0)
    for face in faces:
        for i in face:
            for j in face:
                graph[i, j] = min(graph[i, j], np.linalg.norm(source[i] - source[j]))
    for i, (point, faceId) in enumerate(zip(sample, args['sampleSourceFaceIds'])):
        for j in faces[faceId]:
            graph[len(source) + i, j] = graph[j, len(source) + i] = np.linalg.norm(point - source[j])
    # Sparse explicit zero entries preserve true vertex/sample connections.
    row, col = np.where(np.isfinite(graph) & ~np.eye(n, dtype=bool))
    distances = floyd_warshall(sparse.csr_matrix((graph[row, col], (row, col)), shape=(n, n)), directed=True)
    return distances[len(source):, len(source):]


def testIndependentGraphDistancesAndDenseObjective():
    args = problem(centroid=True)
    prior = buildSourceShapePreservationPrior(**args)
    distances = independentSourceDistances(args)
    C = np.zeros((len(args['targetPoints']), args['clientVertexCount']))
    for row, (face, bary) in enumerate(zip(args['sampleClientFaceIds'], args['sampleClientBarycentric'])):
        C[row, args['clientFaces'][face]] += bary
    vertices = args['sourceVertices'] + np.random.default_rng(4).normal(size=args['sourceVertices'].shape) * .13
    residual = C @ vertices - args['targetPoints']
    expected = 0.
    for row, center in enumerate(prior['centerSampleIds']):
        neighbors = prior['neighborSampleIds'][row]
        np.testing.assert_allclose(prior['neighborDistances'][row], distances[center, neighbors], atol=1e-12)
        w = np.exp(-.5 * (distances[center, neighbors] / args['neighborhoodRadius']) ** 2)
        w /= w.sum()
        delta = residual[center] - w @ residual[neighbors]
        expected += args['weight'] * np.dot(delta, delta) / (3 * len(C) * args['lengthScale'] ** 2)
    assert prior['diagnostics']['active']
    assert loss(prior, vertices) == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(prior['sampleMatrix'].toarray(), C, atol=0)
    np.testing.assert_allclose(prior['differentialOperator'].sum(1), 0., atol=2e-15)
    assert loss(prior, args['sourceVertices']) < 1e-27
    assert sparse.isspmatrix_csr(prior['quadraticPrior']['operator'])


def testIndependentDenseRegistrationOptimumWithExactHardStatic():
    args = problem(centroid=True)
    prior = buildSourceShapePreservationPrior(**args)
    rest = args['sourceVertices'].copy(); rest[:, 2] = .03 * rest[:, 0]
    static = np.array([0, 4, 20, 24])
    options = {'dataWeight': 9., 'smoothWeight': 1.8, 'restWeight': .2, 'staticConstraint': 'hard'}
    result = registerDirectTemplate(rest, args['clientFaces'], args['sampleClientFaceIds'],
        args['sampleClientBarycentric'], args['targetPoints'], args['validMask'], static,
        quadraticPrior=prior['quadraticPrior'], **options)
    # Rebuild the objective independently, including the unchanged joint data
    # denominator and elimination of every XYZ coordinate at hard-static rows.
    C = prior['sampleMatrix'].toarray()
    n, m = len(rest), len(C)
    adjacency = np.zeros((n, n))
    for face in args['clientFaces']:
        for i in face:
            for j in face:
                if i != j:
                    adjacency[i, j] = 1
    L = np.diag(adjacency.sum(1)) - adjacency
    anchors = np.eye(n)[static]
    design = np.r_[C, anchors]
    cd, cs, cr = 9 / (3 * (m + len(static))), 1.8 / (3 * n), .2 / (3 * n)
    H = np.kron(cd * design.T @ design + cs * L.T @ L + cr * np.eye(n), np.eye(3))
    rhs = (cd * design.T @ np.r_[args['targetPoints'], result['alignedRest'][static]] + cr * result['alignedRest']).ravel()
    A, target = prior['quadraticPrior']['operator'].toarray(), prior['quadraticPrior']['targets']
    H += args['weight'] * A.T @ A
    rhs += args['weight'] * A.T @ target
    fixed = (3 * static[:, None] + np.arange(3)).ravel()
    free = np.setdiff1d(np.arange(3 * n), fixed)
    expected = result['alignedRest'].ravel().copy()
    expected[free] = np.linalg.solve(H[np.ix_(free, free)], rhs[free] - H[np.ix_(free, fixed)] @ expected[fixed])
    np.testing.assert_allclose(result['rawVertices'].ravel(), expected, atol=2e-13)
    np.testing.assert_array_equal(result['rawVertices'][static], result['alignedRest'][static])


def testSourceShapeTargetsAreNotClientTemplateCoordinates():
    args = problem()
    first = buildSourceShapePreservationPrior(**args)
    args['sourceVertices'][12, 2] += .4
    args['targetPoints'] = args['sourceVertices'].copy()
    second = buildSourceShapePreservationPrior(**args)
    assert loss(second, args['sourceVertices']) < 1e-27
    assert loss(first, args['sourceVertices']) > 1e-5
    assert second['diagnostics']['targetsAreTemplateDifferentials'] is False


def testDisconnectedAndFoldedNearbySheetsCannotCreateEuclideanShortcut():
    first, faces = grid(5, 3)
    first[:, 2] = 0
    source = np.r_[first, first + [0, 0, .03]]
    # Connect the sheets only at their far x=4 end. Points at x=0 are close
    # in Euclidean space but more than eight units apart along the surface.
    foldedFaces = np.r_[faces, faces + len(first), [[4, 9, 19], [9, 24, 19]]]
    for currentFaces in (np.r_[faces, faces + len(first)], foldedFaces):
        faceIds, weights = vertexBindings(source, currentFaces)
        args = problem(); args.update(sourceVertices=source, sourceFaces=currentFaces,
            clientFaces=currentFaces, clientVertexCount=len(source), targetPoints=source,
            sampleClientFaceIds=faceIds, sampleSourceFaceIds=faceIds,
            sampleClientBarycentric=weights, sampleSourceBarycentric=weights,
            validMask=np.ones(len(source), bool), coverageMask=np.ones(len(source), bool),
            neighborhoodRadius=2.2, minNeighbors=2)
        prior = buildSourceShapePreservationPrior(**args)
        row = int(np.flatnonzero(prior['centerSampleIds'] == 0)[0])
        assert np.all(prior['neighborSampleIds'][row] < len(first))
        assert np.min(np.linalg.norm(source[len(first):] - source[0], axis=1)) < .04


def testSimilarityCovarianceAndUniformWeightAmplitude():
    args = problem(centroid=True)
    prior = buildSourceShapePreservationPrior(**args)
    vertices = args['sourceVertices'] + np.random.default_rng(8).normal(size=args['sourceVertices'].shape) * .2
    rotation = np.linalg.qr(np.array([[.3, -.2, .9], [.7, .8, .1], [.5, -.6, .2]]))[0]
    scaled = deepcopy(args)
    scale, translation = 4.3, np.array([7., -3., 11.])
    for key in ('sourceVertices', 'targetPoints'):
        scaled[key] = args[key] @ rotation * scale + translation
    scaled['lengthScale'] *= scale; scaled['neighborhoodRadius'] *= scale
    changed = buildSourceShapePreservationPrior(**scaled)
    assert loss(changed, vertices @ rotation * scale + translation) == pytest.approx(loss(prior, vertices), rel=2e-12)
    assert loss(prior, vertices + [31., -18., 22.]) == pytest.approx(loss(prior, vertices), abs=2e-15)
    weak = buildSourceShapePreservationPrior(**args, reliabilityWeights=np.full(len(args['targetPoints']), .2),
        reliabilityProvenance='synthetic common reliability decrease')
    assert loss(weak, vertices) == pytest.approx(.2 * loss(prior, vertices), rel=1e-12)
    assert weak['diagnostics']['normalizationSampleCount'] == len(args['targetPoints'])


def testPointOrderingReplicationAndMeshRenumberingPreserveObjective():
    args = problem(centroid=True)
    vertices = args['sourceVertices'] + np.random.default_rng(8).normal(size=args['sourceVertices'].shape) * .2
    baseline = buildSourceShapePreservationPrior(**args)
    random = np.random.default_rng(3)
    samples = random.permutation(np.tile(np.arange(len(args['targetPoints'])), 3))
    shuffled = deepcopy(args)
    for key in ('targetPoints', 'sampleClientFaceIds', 'sampleClientBarycentric', 'sampleSourceFaceIds',
                'sampleSourceBarycentric', 'validMask', 'coverageMask'):
        shuffled[key] = args[key][samples]
    repeated = buildSourceShapePreservationPrior(**shuffled)
    assert loss(repeated, vertices) == pytest.approx(loss(baseline, vertices), rel=1e-12)
    assert repeated['diagnostics']['duplicateValidSampleCount'] == 2 * len(args['targetPoints'])
    changed = deepcopy(args)
    sourceOrder, clientOrder = random.permutation(len(vertices)), random.permutation(len(vertices))
    sourceInverse, clientInverse = np.argsort(sourceOrder), np.argsort(clientOrder)
    changed['sourceVertices'] = args['sourceVertices'][sourceOrder]
    changed['sourceFaces'] = sourceInverse[args['sourceFaces']]
    changed['clientFaces'] = clientInverse[args['clientFaces']]
    for faceKey, sampleKey in (('clientFaces', 'sampleClientFaceIds'), ('sourceFaces', 'sampleSourceFaceIds')):
        order = random.permutation(len(changed[faceKey]))
        changed[faceKey] = changed[faceKey][order]
        changed[sampleKey] = np.argsort(order)[changed[sampleKey]]
    renumbered = buildSourceShapePreservationPrior(**changed)
    assert loss(renumbered, vertices[clientOrder]) == pytest.approx(loss(baseline, vertices), rel=1e-12)


def testDistanceTiesDoNotSelectByVertexNumber():
    args = problem(); args['sourceVertices'][:, 2] = 0; args['targetPoints'] = args['sourceVertices'].copy()
    args.update(maxNeighbors=5, minNeighbors=2)
    first = buildSourceShapePreservationPrior(**args)
    assert first['diagnostics']['tieLimitedNeighborhoodCount'] > 0
    order = np.arange(len(args['targetPoints']))[::-1]
    secondArgs = deepcopy(args)
    for key in ('targetPoints', 'sampleClientFaceIds', 'sampleClientBarycentric', 'sampleSourceFaceIds',
                'sampleSourceBarycentric', 'validMask', 'coverageMask'):
        secondArgs[key] = args[key][order]
    second = buildSourceShapePreservationPrior(**secondArgs)
    vertices = args['sourceVertices'].copy(); vertices[12, 2] = 1.
    assert loss(first, vertices) == pytest.approx(loss(second, vertices), rel=1e-12)


@pytest.mark.parametrize('mode', ['noCoverage', 'zeroReliability', 'noValidSamples', 'noNeighbors', 'zeroGlobalWeight'])
def testInactiveReturnsValidatedZeroWeightLegacyPath(mode):
    args = problem()
    if mode == 'noCoverage': args['coverageMask'][:] = False
    if mode == 'zeroReliability': args.update(reliabilityWeights=np.zeros(len(args['targetPoints'])), reliabilityProvenance='synthetic unavailable evidence')
    if mode == 'noValidSamples': args['validMask'][:] = False
    if mode == 'noNeighbors': args['neighborhoodRadius'] = .01
    if mode == 'zeroGlobalWeight': args['weight'] = 0.
    prior = buildSourceShapePreservationPrior(**args)
    assert prior['diagnostics']['active'] is False
    assert prior['quadraticPrior']['weight'] == 0
    assert prior['quadraticPrior']['operator'].shape == (1, 3 * args['clientVertexCount'])
    rest = args['sourceVertices'].copy(); rest[:, 2] = 0
    inputs = (rest, args['clientFaces'], args['sampleClientFaceIds'], args['sampleClientBarycentric'],
              args['targetPoints'], np.ones(len(args['targetPoints']), bool), np.array([0, 4, 20, 24]))
    baseline = registerDirectTemplate(*inputs, staticConstraint='hard')
    candidate = registerDirectTemplate(*inputs, staticConstraint='hard', quadraticPrior=prior['quadraticPrior'])
    np.testing.assert_array_equal(candidate['rawVertices'], baseline['rawVertices'])


def testCoverageIsNotNamedConfidenceAndReliabilityNeedsProvenance():
    args = problem()
    prior = buildSourceShapePreservationPrior(**args)
    assert 'no reliability/confidence estimate' in prior['diagnostics']['reliabilityProvenance']
    assert not prior['diagnostics']['reliabilityProvided']
    assert prior['diagnostics']['coverageProvided']
    with pytest.raises(ValueError, match='Provenance'):
        buildSourceShapePreservationPrior(**args, reliabilityWeights=np.ones(len(args['targetPoints'])))
    with pytest.raises(ValueError, match='ambiguous'):
        buildSourceShapePreservationPrior(**args, reliabilityProvenance='unsupported claim')


def testExplicitClientCountPreservesUnusedTrailingVertices():
    args = problem(); args['clientVertexCount'] += 3
    prior = buildSourceShapePreservationPrior(**args)
    assert prior['quadraticPrior']['operator'].shape[1] == 3 * args['clientVertexCount']
    assert prior['quadraticPrior']['operator'][:, -9:].nnz == 0


def testLengthWeightContractKeepsPhysicalObjectiveUnits():
    args = problem()
    vertices = args['sourceVertices'].copy(); vertices[12, 2] += .3
    args['weight'] = 10000 * args['lengthScale'] ** 2
    first = buildSourceShapePreservationPrior(**args)
    args['lengthScale'] *= 7
    args['weight'] = 10000 * args['lengthScale'] ** 2
    second = buildSourceShapePreservationPrior(**args)
    assert loss(first, vertices) == pytest.approx(loss(second, vertices), rel=1e-12)


def testFloat32SourcePartitionsArePreservedAcrossCoordinateTransform():
    args = problem(centroid=True)
    args['sampleSourceBarycentric'] = args['sampleSourceBarycentric'].astype(np.float32)
    weights = args['sampleSourceBarycentric'].astype(float)
    args['targetPoints'] = np.sum(args['sourceVertices'][args['sourceFaces'][args['sampleSourceFaceIds']]] * weights[..., None], axis=1)
    assert np.max(abs(weights.sum(1) - 1)) > 1e-12
    prior = buildSourceShapePreservationPrior(**args)
    shifted = deepcopy(args)
    shifted['sourceVertices'] += [17., -21., 13.]
    shifted['targetPoints'] += [17., -21., 13.]
    changed = buildSourceShapePreservationPrior(**shifted)
    vertices = args['sourceVertices'].copy(); vertices[12, 2] += .3
    assert loss(changed, vertices + [17., -21., 13.]) == pytest.approx(loss(prior, vertices), rel=1e-12)
    np.testing.assert_array_equal(args['sampleSourceBarycentric'], weights.astype(np.float32))


def testAllStaticRegistrationRetainsExactAlignedTemplate():
    args = problem()
    prior = buildSourceShapePreservationPrior(**args)
    rest = args['sourceVertices'].copy(); rest[:, 2] = 0
    result = registerDirectTemplate(rest, args['clientFaces'], args['sampleClientFaceIds'],
        args['sampleClientBarycentric'], args['targetPoints'], args['validMask'], np.arange(len(rest)),
        quadraticPrior=prior['quadraticPrior'], staticConstraint='hard')
    np.testing.assert_array_equal(result['rawVertices'], result['alignedRest'])


def testCoverageAndRowReliabilityStaySeparateAndDoNotAmplifyRemainingRows():
    args = problem()
    scores = np.ones(len(args['targetPoints'])); scores[12] = .05
    prior = buildSourceShapePreservationPrior(**args, reliabilityWeights=scores,
        reliabilityProvenance='synthetic independently provided weights, not a probability')
    for row, center in enumerate(prior['centerSampleIds']):
        expected = min(scores[center], scores[prior['neighborSampleIds'][row]].min())
        assert prior['rowReliability'][row] == expected
        assert prior['rowScales'][row] == pytest.approx(np.sqrt(expected / (3 * len(scores))) / args['lengthScale'])
    partial = deepcopy(args); partial['coverageMask'][12] = False
    covered = buildSourceShapePreservationPrior(**partial)
    assert covered['diagnostics']['normalizationSampleCount'] == len(scores)
    assert 12 not in covered['centerSampleIds']
    assert all(12 not in group for group in covered['neighborSampleIds'])
    assert covered['diagnostics']['coverageMeaning'].startswith('Binary availability only')


def testInvalidSourceStillRejectedWhenPriorHasNoCoveredRows():
    args = problem(); args['coverageMask'][:] = False
    args['sourceVertices'][1] = args['sourceVertices'][0]
    args['targetPoints'] = args['sourceVertices'].copy()
    with pytest.raises(ValueError, match='nonzero edges'):
        buildSourceShapePreservationPrior(**args)


@pytest.mark.parametrize('change', [
    {'clientVertexCount': True}, {'clientVertexCount': 100_000}, {'lengthScale': 0.},
    {'neighborhoodRadius': np.inf}, {'weight': -1.}, {'maxNeighbors': 33}, {'minNeighbors': 13},
    {'coverageMask': np.ones(25)}, {'validMask': np.ones(24, bool)},
    {'targetPoints': np.full((25, 3), np.nan)},
    {'reliabilityWeights': np.full(25, 1.1), 'reliabilityProvenance': 'bad bound'},
])
def testInvalidInputsRejected(change):
    args = problem(); args.update(change)
    with pytest.raises(ValueError):
        buildSourceShapePreservationPrior(**args)


def testWrongFrameDegenerateSourceAndNonAffineClientBindingsRejected():
    args = problem(); args['targetPoints'] += [0, 0, .01]
    with pytest.raises(ValueError, match='coordinate frame'):
        buildSourceShapePreservationPrior(**args)
    args = problem(); args['sourceVertices'][1] = args['sourceVertices'][0]; args['targetPoints'] = args['sourceVertices'].copy()
    with pytest.raises(ValueError, match='nonzero edges'):
        buildSourceShapePreservationPrior(**args)
    args = problem(); args['sampleClientBarycentric'][0, 0] -= 1e-8
    with pytest.raises(ValueError, match='affine rows'):
        buildSourceShapePreservationPrior(**args)


def testGraphBudgetFailsBeforeShortestPaths(monkeypatch):
    args = problem()
    monkeypatch.setitem(module._LIMITS, 'distanceVectorWork', 1)
    monkeypatch.setattr(module, 'dijkstra', lambda *a, **k: pytest.fail('Shortest paths ran before budget check'))
    with pytest.raises(ValueError, match='bounded graph/work budget'):
        buildSourceShapePreservationPrior(**args)


def testNormalProductBudgetIsExplicit(monkeypatch):
    monkeypatch.setitem(module._LIMITS, 'normalProductWork', 1)
    with pytest.raises(ValueError, match='sparse product budget'):
        buildSourceShapePreservationPrior(**problem())


def testBuilderDoesNotModifyCallerArrays():
    args = problem(centroid=True)
    snapshot = deepcopy(args)
    buildSourceShapePreservationPrior(**args)
    for key, value in args.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, snapshot[key])
