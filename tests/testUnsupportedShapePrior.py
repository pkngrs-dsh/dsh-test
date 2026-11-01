"""Synthetic shape-prior contracts; no production identities or bad-point IDs."""
from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

from faceAppearance.directTemplateRegistration import registerDirectTemplate
from faceAppearance.unsupportedShapePrior import buildUnsupportedShapePrior


def problem():
    yy, xx = np.meshgrid(np.linspace(-1., 1., 7), np.linspace(-1., 1., 7), indexing='ij')
    rest = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(49)])
    faces = []
    for row in range(6):
        for column in range(6):
            i = row*7 + column
            faces.extend([[i, i+1, i+7], [i+1, i+8, i+7]])
    grid = np.arange(49).reshape(7, 7)
    interior = grid[2:5, 2:5].ravel()
    boundary = np.setdiff1d(grid[1:6, 1:6].ravel(), interior)
    t = np.arange(len(interior), dtype=float)
    rest[interior, 0] += .028*np.cos(t)
    rest[interior, 1] += .035*np.sin(t*.7)
    rest[interior, 2] = [-.12, 0., .18, .27, .39, .29, .13, .21, .17]
    return {'alignedRestVertices': rest, 'faces': np.asarray(faces),
        'boundaryVertexIds': boundary, 'interiorVertexIds': interior,
        'sampleVertexIndices': boundary[:, None], 'sampleBarycentric': np.ones((len(boundary), 1)),
        'sampleWeights': np.ones(len(boundary)), 'normalDirection': np.array([0., 0., 1.]), 'lengthScale': .2}


def priorLoss(prior, vertices):
    residual = prior['weightedOperator'] @ vertices.ravel() - prior['weightedTargets']
    return float(residual @ residual)


def independentRelativeOperator(args):
    boundary, interior = args['boundaryVertexIds'], args['interiorVertexIds']
    members = np.r_[boundary, interior]
    adjacency = np.zeros((len(members), len(members)))
    lookup = {int(vertex): index for index, vertex in enumerate(members)}
    for face in args['faces']:
        if all(int(vertex) in lookup for vertex in face):
            for i in face:
                for j in face:
                    if i != j:
                        adjacency[lookup[int(i)], lookup[int(j)]] = 1.
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    extension = np.linalg.solve(laplacian[len(boundary):, len(boundary):], -laplacian[len(boundary):, :len(boundary)])
    relative = np.zeros((len(interior), len(args['alignedRestVertices'])))
    relative[:, boundary] = -extension
    relative[np.arange(len(interior)), interior] = 1.
    return relative, extension


def testIndependentDenseVectorObjectiveAndTargetShape():
    args = problem()
    args.update(normalDirection=np.array([.2, -.3, .9]), normalTargetFraction=.35,
                tangentTargetFraction=.8, normalPenalty=2.3, tangentPenalty=.6)
    prior = buildUnsupportedShapePrior(**args)
    H, extension = independentRelativeOperator(args)
    rest = args['alignedRestVertices']
    n = args['normalDirection']/np.linalg.norm(args['normalDirection'])
    P, T = np.outer(n, n), np.eye(3)-np.outer(n, n)
    desired = .35*(H@rest@P) + .8*(H@rest@T)
    vertices = rest + np.random.default_rng(51).normal(size=rest.shape)*.06
    residual = H@vertices - desired
    expected = np.mean(2.3*np.sum((residual@P)**2, axis=1) + .6*np.sum((residual@T)**2, axis=1))/.2**2
    np.testing.assert_allclose(prior['relativeOperator'].toarray(), H, atol=1e-14)
    np.testing.assert_allclose(prior['harmonicBoundaryWeights'], extension, atol=1e-14)
    np.testing.assert_allclose(prior['vectorOperator']@vertices.ravel(), (H@vertices).ravel(), atol=1e-14)
    np.testing.assert_allclose(prior['targetOffsets'], desired, atol=1e-14)
    assert priorLoss(prior, vertices) == pytest.approx(expected, rel=1e-13)
    assert sparse.isspmatrix_csr(prior['weightedOperator'])
    assert prior['weightedOperator'].shape == (3*len(args['interiorVertexIds']), 3*len(rest))
    targetMesh = rest.copy()
    targetMesh[args['interiorVertexIds']] = extension@rest[args['boundaryVertexIds']] + desired
    assert priorLoss(prior, targetMesh) < 1e-26


def testAllUnsupportedSignsReceiveShapePriorWithoutAnatomicalSelection():
    args = problem()
    prior = buildUnsupportedShapePrior(**args)
    offsets = prior['restRelativeOffsets']
    assert np.any(offsets[:, 2] < 0) and np.any(offsets[:, 2] == 0) and np.any(offsets[:, 2] > 0)
    np.testing.assert_array_equal(prior['priorWeights'], np.ones(len(offsets)))
    assert prior['diagnostics']['geometricSignUsedToSelectPrior'] is False
    assert priorLoss(prior, args['alignedRestVertices']) < 1e-26
    for interiorIndex in np.flatnonzero(offsets[:, 2] <= 0):
        changed = args['alignedRestVertices'].copy()
        changed[args['interiorVertexIds'][interiorIndex], 0] += .2
        assert priorLoss(prior, changed) == pytest.approx(1/len(offsets))


def testTangentialAndNormalChangesBothCostButHarmonicBoundaryMotionDoesNot():
    args = problem()
    prior = buildUnsupportedShapePrior(**args, normalTargetFraction=.35)
    rest, boundary, interior = args['alignedRestVertices'], args['boundaryVertexIds'], args['interiorVertexIds']
    target = rest.copy()
    target[interior] = prior['harmonicBoundaryWeights']@rest[boundary] + prior['targetOffsets']
    for direction in np.eye(3):
        displaced = target.copy()
        displaced[interior] += .08*direction
        assert priorLoss(prior, displaced) == pytest.approx(.08**2/.2**2)
    assert priorLoss(prior, target+[17., -21., 9.]) < 1e-23
    boundaryMotion = np.random.default_rng(21).normal(size=(len(boundary), 3))*.3
    motion = np.zeros_like(rest)
    motion[boundary] = boundaryMotion
    motion[interior] = prior['harmonicBoundaryWeights']@boundaryMotion
    assert priorLoss(prior, target+motion) < 1e-25
    outside = np.setdiff1d(np.arange(len(rest)), np.r_[boundary, interior])
    assert prior['relativeOperator'][:, outside].nnz == 0
    altered = target.copy(); altered[outside] += [500., -200., 100.]
    assert priorLoss(prior, altered) == pytest.approx(priorLoss(prior, target), abs=1e-26)


def testSupportMatchesDenseDesignAndNotSeparateDuplicateBarycentricSquares():
    args = problem()
    boundary, interior = args['boundaryVertexIds'], args['interiorVertexIds']
    sampleIds = np.vstack([np.repeat(boundary[:, None], 2, axis=1),
                          [interior[0], interior[0]], [interior[1], interior[2]], [interior[3], interior[4]]])
    bary = np.vstack([np.tile([.3, .7], (len(boundary), 1)), [.2, .8], [.4, .6], [1.4, -.4]])
    weights = np.r_[np.ones(len(boundary)), .05, .1, 0.]
    args.update(sampleVertexIndices=sampleIds, sampleBarycentric=bary, sampleWeights=weights)
    prior = buildUnsupportedShapePrior(**args)
    design = np.zeros((len(weights), len(args['alignedRestVertices'])))
    for row in range(len(weights)):
        for vertex, coefficient in zip(sampleIds[row], bary[row]):
            design[row, vertex] += coefficient
    support = np.diag(design.T@np.diag(weights)@design)
    expected = np.clip(1-support[interior]/.15, 0, 1)
    np.testing.assert_allclose(prior['vertexSupport'], support, atol=1e-14)
    np.testing.assert_allclose(prior['priorWeights'], expected, atol=1e-14)
    assert prior['priorWeights'][0] == pytest.approx(2/3)
    displaced = args['alignedRestVertices'].copy(); displaced[interior, 0] += .2
    assert priorLoss(prior, displaced) == pytest.approx(float(expected.mean()), rel=1e-13)


def testStrongDataSupportDisablesPriorAndConfidenceAmplitudeIsNotRenormalized():
    args = problem()
    interior, boundary = args['interiorVertexIds'], args['boundaryVertexIds']
    unsupported = buildUnsupportedShapePrior(**args)
    args.update(sampleVertexIndices=np.r_[boundary, interior][:, None],
                sampleBarycentric=np.ones((len(boundary)+len(interior), 1)),
                sampleWeights=np.r_[np.ones(len(boundary)), np.full(len(interior), .075)])
    weak = buildUnsupportedShapePrior(**args)
    displaced = args['alignedRestVertices'].copy(); displaced[interior] += [.05, -.03, .08]
    assert priorLoss(weak, displaced) == pytest.approx(.5*priorLoss(unsupported, displaced), rel=1e-13)
    args['sampleWeights'][-len(interior):] = 1.
    strong = buildUnsupportedShapePrior(**args)
    assert strong['weightedOperator'].nnz == 0
    assert priorLoss(strong, displaced) == 0


def testObservationOrderReplicationAndUniformWeightScaleDoNotSelectSpecialSamples():
    args = problem()
    boundary, interior = args['boundaryVertexIds'], args['interiorVertexIds']
    args.update(sampleVertexIndices=np.r_[boundary, interior][:, None],
                sampleBarycentric=np.ones((len(boundary)+len(interior), 1)),
                sampleWeights=np.r_[np.linspace(.8, 1.2, len(boundary)), np.linspace(0., .15, len(interior))])
    base = buildUnsupportedShapePrior(**args)
    rng = np.random.default_rng(41)
    for replication, scale in ((1, 1.), (4, 1.), (3, 1/3), (1, 37.)):
        changed = deepcopy(args)
        order = rng.permutation(np.repeat(np.arange(len(args['sampleWeights'])), replication))
        for key in ('sampleVertexIndices', 'sampleBarycentric', 'sampleWeights'):
            changed[key] = changed[key][order]
        changed['sampleWeights'] *= scale
        actual = buildUnsupportedShapePrior(**changed)
        np.testing.assert_allclose(actual['priorWeights'], base['priorWeights'], atol=1e-14)
        np.testing.assert_allclose(actual['weightedOperator'].toarray(), base['weightedOperator'].toarray(), atol=2e-14)
        np.testing.assert_allclose(actual['weightedTargets'], base['weightedTargets'], atol=2e-14)


@pytest.mark.parametrize('scale', [.001, 1., 1000.])
def testJointSimilarityCovarianceAndNormalAxisSignInvariance(scale):
    args = problem()
    args.update(normalDirection=np.array([.23, -.31, .91]), normalTargetFraction=.35,
                tangentTargetFraction=.9, normalPenalty=.7, tangentPenalty=1.8)
    original = buildUnsupportedShapePrior(**args)
    trial = args['alignedRestVertices'] + np.random.default_rng(19).normal(size=(49, 3))*.04
    rotation, _ = np.linalg.qr(np.random.default_rng(31).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = np.array([.7, -.2, .3])*scale
    transformed = deepcopy(args)
    transformed['alignedRestVertices'] = args['alignedRestVertices']@rotation*scale + translation
    transformed['normalDirection'] = args['normalDirection']@rotation
    transformed['lengthScale'] *= scale
    result = buildUnsupportedShapePrior(**transformed)
    np.testing.assert_allclose(result['targetOffsets'], original['targetOffsets']@rotation*scale, rtol=1e-12, atol=1e-12*scale)
    assert priorLoss(result, trial@rotation*scale+translation) == pytest.approx(priorLoss(original, trial), rel=1e-12)
    transformed['normalDirection'] *= -1
    reversedAxis = buildUnsupportedShapePrior(**transformed)
    np.testing.assert_array_equal(reversedAxis['weightedOperator'].toarray(), result['weightedOperator'].toarray())
    np.testing.assert_array_equal(reversedAxis['weightedTargets'], result['weightedTargets'])


def testVertexAndRegionReindexingAndFaceOrderCannotChangePhysicalPrior():
    args = problem()
    args.update(normalTargetFraction=.35, normalPenalty=.9, tangentPenalty=2.)
    original = buildUnsupportedShapePrior(**args)
    rng = np.random.default_rng(22)
    vertexOrder = rng.permutation(49)
    oldToNew = np.argsort(vertexOrder)
    boundaryOrder = rng.permutation(len(args['boundaryVertexIds']))
    interiorOrder = rng.permutation(len(args['interiorVertexIds']))
    changed = deepcopy(args)
    changed['alignedRestVertices'] = args['alignedRestVertices'][vertexOrder]
    changed['faces'] = oldToNew[args['faces'][rng.permutation(len(args['faces']))]][:, ::-1]
    changed['boundaryVertexIds'] = oldToNew[args['boundaryVertexIds'][boundaryOrder]]
    changed['interiorVertexIds'] = oldToNew[args['interiorVertexIds'][interiorOrder]]
    changed['sampleVertexIndices'] = oldToNew[args['sampleVertexIndices']]
    result = buildUnsupportedShapePrior(**changed)
    np.testing.assert_allclose(result['targetOffsets'], original['targetOffsets'][interiorOrder], atol=1e-14)
    np.testing.assert_allclose(result['harmonicBoundaryWeights'], original['harmonicBoundaryWeights'][interiorOrder][:, boundaryOrder], atol=1e-14)
    trial = args['alignedRestVertices'] + rng.normal(size=(49, 3))*.07
    assert priorLoss(result, trial[vertexOrder]) == pytest.approx(priorLoss(original, trial), rel=1e-13)


def testAbsentBoundarySupportNeedsExplicitReferenceAndInputsAreNotMutated():
    args = problem(); args['sampleWeights'][:] = 0
    before = deepcopy(args)
    with pytest.raises(ValueError, match='No positive boundary support'):
        buildUnsupportedShapePrior(**args)
    prior = buildUnsupportedShapePrior(**args, referenceSupport=1.)
    assert prior['diagnostics']['unsupportedInteriorCount'] == len(args['interiorVertexIds'])
    for key, value in before.items():
        np.testing.assert_array_equal(args[key], value)
    prior['targetOffsets'][:] = 0
    prior['boundaryVertexIds'][:] = 0
    np.testing.assert_array_equal(args['alignedRestVertices'], before['alignedRestVertices'])
    np.testing.assert_array_equal(args['boundaryVertexIds'], before['boundaryVertexIds'])


@pytest.mark.parametrize('normalCost,tangentCost', [(0., 1.), (1., 0.)])
def testDirectionalPenaltiesAreExplicitAndMayDisableOneSubspace(normalCost, tangentCost):
    args = problem()
    prior = buildUnsupportedShapePrior(**args, normalPenalty=normalCost, tangentPenalty=tangentCost)
    target = args['alignedRestVertices'].copy()
    target[args['interiorVertexIds']] += [.06, 0., .08]
    assert priorLoss(prior, target) == pytest.approx((normalCost*.08**2+tangentCost*.06**2)/.2**2)


def testSparseOutputIntegratesWithCoupledExactRegistrationAndPreservesHardBoundary():
    args = problem()
    # Vertex samples covering the outer mesh provide a non-collinear alignment;
    # only the explicit patch interior lacks observations. No face IDs encode
    # special treatment in either the prior or the registration solver.
    rest, faces = args['alignedRestVertices'], args['faces']
    observed = np.setdiff1d(np.arange(len(rest)), args['interiorVertexIds'])
    faceIds, bary = [], []
    for vertex in observed:
        faceId, corner = np.argwhere(faces == vertex)[0]
        faceIds.append(faceId)
        row = np.zeros(3); row[corner] = 1.
        bary.append(row)
    direct = dict(restVertices=rest, faces=faces, sampleFaceIds=np.asarray(faceIds),
        sampleBarycentrics=np.asarray(bary), targetPoints=rest[observed], validMask=np.ones(len(observed), bool),
        staticVertexIds=args['boundaryVertexIds'], dataWeight=100., smoothWeight=3., restWeight=0., staticConstraint='hard')
    baseline = registerDirectTemplate(**direct)
    shapeArgs = deepcopy(args)
    shapeArgs.update(alignedRestVertices=baseline['alignedRest'], sampleVertexIndices=observed[:, None],
        sampleBarycentric=np.ones((len(observed), 1)), sampleWeights=np.ones(len(observed)))
    prior = buildUnsupportedShapePrior(**shapeArgs, normalTargetFraction=.35)
    result = registerDirectTemplate(**direct, quadraticPrior={
        'operator': prior['weightedOperator'], 'targets': prior['weightedTargets'], 'weight': .2})
    np.testing.assert_array_equal(result['rawVertices'][direct['staticVertexIds']], baseline['alignedRest'][direct['staticVertexIds']])
    assert priorLoss(prior, result['rawVertices']) < priorLoss(prior, baseline['rawVertices'])
    assert result['diagnostics']['quadraticPrior']['rowCount'] == 3*len(args['interiorVertexIds'])
    assert result['diagnostics']['normalEquationRelativeResidual'] < 1e-12


@pytest.mark.parametrize('damage', ['overlap', 'missingBoundary', 'duplicateBoundary', 'outsideInterior',
    'nanRest', 'invalidSampleIndex', 'negativeWeight', 'nonconvexPositive', 'wrongBarySum',
    'nanAxis', 'zeroAxis', 'zeroScale', 'nanScale', 'zeroThreshold', 'largeThreshold',
    'largeNormalFraction', 'negativeTangentFraction', 'nanPenalty', 'zeroPenalties',
    'negativePenalty', 'zeroReference', 'floatIds', 'booleanFraction'])
def testInvalidInputsRegionsAndParametersAreRejected(damage):
    args = problem()
    if damage == 'overlap': args['interiorVertexIds'][0] = args['boundaryVertexIds'][0]
    elif damage == 'missingBoundary': args['boundaryVertexIds'] = np.delete(args['boundaryVertexIds'], 1)
    elif damage == 'duplicateBoundary': args['boundaryVertexIds'][0] = args['boundaryVertexIds'][1]
    elif damage == 'outsideInterior': args['interiorVertexIds'][0] = 0
    elif damage == 'nanRest': args['alignedRestVertices'][0, 0] = np.nan
    elif damage == 'invalidSampleIndex': args['sampleVertexIndices'][0] = 100
    elif damage == 'negativeWeight': args['sampleWeights'][0] = -1
    elif damage == 'nonconvexPositive':
        args['sampleVertexIndices'] = np.repeat(args['sampleVertexIndices'], 2, axis=1)
        args['sampleBarycentric'] = np.tile([1.2, -.2], (len(args['sampleWeights']), 1))
    elif damage == 'wrongBarySum': args['sampleBarycentric'][0] = .5
    elif damage == 'nanAxis': args['normalDirection'] = [0., np.nan, 1.]
    elif damage == 'zeroAxis': args['normalDirection'] = np.zeros(3)
    elif damage == 'zeroScale': args['lengthScale'] = 0.
    elif damage == 'nanScale': args['lengthScale'] = np.nan
    elif damage == 'zeroThreshold': args['supportThreshold'] = 0.
    elif damage == 'largeThreshold': args['supportThreshold'] = 1.1
    elif damage == 'largeNormalFraction': args['normalTargetFraction'] = 1.1
    elif damage == 'negativeTangentFraction': args['tangentTargetFraction'] = -.1
    elif damage == 'nanPenalty': args['normalPenalty'] = np.nan
    elif damage == 'zeroPenalties': args.update(normalPenalty=0., tangentPenalty=0.)
    elif damage == 'negativePenalty': args['tangentPenalty'] = -.1
    elif damage == 'zeroReference': args['referenceSupport'] = 0.
    elif damage == 'floatIds': args['boundaryVertexIds'] = args['boundaryVertexIds'].astype(float)
    elif damage == 'booleanFraction': args['normalTargetFraction'] = True
    with pytest.raises(ValueError):
        buildUnsupportedShapePrior(**args)
