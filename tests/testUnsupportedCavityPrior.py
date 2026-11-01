"""Relative cavity prior contracts independent of any face reconstruction."""
import copy

import numpy as np
import pytest
from scipy import sparse

from faceAppearance.unsupportedCavityPrior import buildUnsupportedCavityPrior


def problem():
    yy, xx = np.meshgrid(np.linspace(-1., 1., 7), np.linspace(-1., 1., 7), indexing='ij')
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(49)])
    faces = []
    for row in range(6):
        for column in range(6):
            i = row*7+column
            faces.extend([[i, i+1, i+7], [i+1, i+8, i+7]])
    grid = np.arange(49).reshape(7, 7)
    interior = grid[2:5, 2:5].ravel()
    rim = np.setdiff1d(grid[1:6, 1:6].ravel(), interior)
    vertices[interior, 2] = .4*(1-(vertices[interior, 0]/(2/3))**2)*(1-(vertices[interior, 1]/(2/3))**2)
    return {'alignedRestVertices': vertices, 'faces': np.asarray(faces),
        'rimVertexIds': rim, 'interiorVertexIds': interior,
        'sampleVertexIndices': rim[:, None], 'sampleBarycentric': np.ones((len(rim), 1)),
        'sampleWeights': np.ones(len(rim)), 'inwardDirection': [0., 0., 1.], 'depthScale': .2}


def loss(prior, vertices):
    residual = prior['weightedOperator']@vertices.ravel()-prior['weightedTargets']
    return float(residual@residual)


def testPriorPreservesTemplateDepthAndHasNoSupportOutsideAuditedRegion():
    args = problem(); snapshot = copy.deepcopy(args)
    prior = buildUnsupportedCavityPrior(**args)
    assert sparse.isspmatrix_csr(prior['relativeOperator'])
    assert sparse.isspmatrix_csr(prior['depthOperator'])
    assert loss(prior, args['alignedRestVertices']) < 1e-25
    np.testing.assert_allclose(prior['priorWeights'], 1.)
    np.testing.assert_allclose(prior['relativeOperator']@np.ones(49), 0., atol=1e-14)
    outside = np.setdiff1d(np.arange(49), np.r_[args['rimVertexIds'], args['interiorVertexIds']])
    assert prior['relativeOperator'][:, outside].nnz == 0
    assert prior['diagnostics']['qualityAccepted'] is False
    for key in args:
        np.testing.assert_array_equal(args[key], snapshot[key])


def testGlobalTranslationAndArbitraryHarmonicBoundaryMotionCostNothing():
    args = problem(); prior = buildUnsupportedCavityPrior(**args)
    rest = args['alignedRestVertices']
    assert loss(prior, rest+[4., -3., 2.]) < 1e-25
    rim = args['rimVertexIds']; interior = args['interiorVertexIds']
    delta = np.zeros_like(rest)
    t = np.arange(len(rim))
    delta[rim] = np.column_stack([.2*np.sin(t), .1*np.cos(t), .3*np.sin(t*.7)])
    delta[interior] = prior['harmonicBoundaryWeights']@delta[rim]
    assert np.linalg.norm(delta[interior]) > .1
    assert loss(prior, rest+delta) < 1e-25
    np.testing.assert_allclose(prior['depthOperator']@(rest+delta).ravel(), prior['targetDepths'], atol=1e-14)


def testTangentialMotionIsFreeButFlatteningRelativeDepthIsPenalized():
    args = problem(); prior = buildUnsupportedCavityPrior(**args)
    moved = args['alignedRestVertices'].copy()
    moved[args['interiorVertexIds'], :2] += [.12, -.3]
    assert loss(prior, moved) < 1e-25
    moved[args['interiorVertexIds']] = prior['harmonicBoundaryWeights']@moved[args['rimVertexIds']]
    expected = np.mean((prior['restRelativeDepths']/args['depthScale'])**2)
    assert loss(prior, moved) == pytest.approx(expected)
    assert expected > .5


def testSupportMatchesTrueNormalEquationDiagonalIncludingRepeatedIndices():
    args = problem()
    extra = np.array([[24, 24], [16, 17], [23, 24]])
    bary = np.array([[.2, .8], [.3, .7], [1.4, -.4]])
    ids = np.concatenate([np.repeat(args['sampleVertexIndices'], 2, axis=1), extra])
    weights = np.r_[args['sampleWeights'], .05, .1, 0.]
    bary = np.concatenate([np.tile([.25, .75], (len(args['sampleWeights']), 1)), bary])
    args.update(sampleVertexIndices=ids, sampleBarycentric=bary, sampleWeights=weights, supportThreshold=.2)
    prior = buildUnsupportedCavityPrior(**args)
    matrix = np.zeros((len(ids), 49))
    for row in range(len(ids)):
        for vertex, coefficient in zip(ids[row], bary[row]):
            matrix[row, vertex] += coefficient
    expected = np.diag(matrix.T@np.diag(weights)@matrix)
    np.testing.assert_allclose(prior['vertexSupport'], expected, atol=1e-14)
    np.testing.assert_allclose(prior['priorWeights'],
        np.clip(1-expected[args['interiorVertexIds']]/.2, 0, 1))
    assert prior['priorWeights'][np.flatnonzero(args['interiorVertexIds'] == 24)[0]] == pytest.approx(.75)


def testConfidenceAmplitudeIsNotCancelledByItsNormalization():
    args = problem(); ids = args['interiorVertexIds']
    strong = buildUnsupportedCavityPrior(**args)
    args['sampleVertexIndices'] = np.r_[args['rimVertexIds'], ids][:, None]
    args['sampleBarycentric'] = np.ones((len(args['sampleVertexIndices']), 1))
    args['sampleWeights'] = np.r_[np.ones(len(args['rimVertexIds'])), np.full(len(ids), .075)]
    weaker = buildUnsupportedCavityPrior(**args)
    flattened = args['alignedRestVertices'].copy(); flattened[ids, 2] = 0.
    assert loss(weaker, flattened) == pytest.approx(.5*loss(strong, flattened))
    args['sampleWeights'][-len(ids):] = 1.
    supported = buildUnsupportedCavityPrior(**args)
    assert supported['weightedOperator'].nnz == 0
    assert loss(supported, flattened) == 0


def testFractionIsAbsoluteTemplateDepthAndCustomTargetScalingIsExplicit():
    args = problem(); prior = buildUnsupportedCavityPrior(**args, targetDepthFraction=.35)
    np.testing.assert_allclose(prior['targetDepths'], .35*prior['restRelativeDepths'])
    candidate = args['alignedRestVertices'].copy(); candidate[args['interiorVertexIds'], 2] *= .35
    assert loss(prior, candidate) < 1e-25
    assert loss(prior, args['alignedRestVertices']) > .1


def testJointSimilarityAndObservationWeightScalingPreserveNormalizedObjective():
    args = problem(); prior = buildUnsupportedCavityPrior(**args)
    candidate = args['alignedRestVertices'].copy(); candidate[args['interiorVertexIds'], 2] *= .6
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    transformed = copy.deepcopy(args)
    transformed['alignedRestVertices'] = args['alignedRestVertices']@rotation.T*1000+[2., 4., 6.]
    transformed['inwardDirection'] = np.asarray(args['inwardDirection'])@rotation.T
    transformed['depthScale'] *= 1000
    transformed['sampleWeights'] *= 17
    other = buildUnsupportedCavityPrior(**transformed)
    assert loss(other, candidate@rotation.T*1000+[2., 4., 6.]) == pytest.approx(loss(prior, candidate), rel=1e-12)


def testUnobservedRimRequiresExplicitReferenceAndNonpositiveDepthIsDisabled():
    args = problem(); args['sampleWeights'][:] = 0
    with pytest.raises(ValueError, match='No positive rim support'):
        buildUnsupportedCavityPrior(**args)
    args['alignedRestVertices'][args['interiorVertexIds'][0], 2] = -.1
    prior = buildUnsupportedCavityPrior(**args, referenceSupport=1.)
    assert prior['priorWeights'][0] == 0
    assert prior['diagnostics']['validSampleCount'] == 0
    assert prior['diagnostics']['unsupportedInteriorCount'] == 8


@pytest.mark.parametrize('damage', ['overlap', 'missingRim', 'duplicateRim', 'outsideInterior', 'nanRest',
    'invalidIndex', 'negativeWeight', 'nonconvexPositive', 'wrongBarySum', 'nanDirection',
    'tangentDirection', 'zeroScale', 'nanScale', 'zeroThreshold', 'largeThreshold',
    'largeFraction', 'zeroReference', 'floatIds'])
def testInvalidRegionsInputsAndSettingsAreRejected(damage):
    args = problem()
    if damage == 'overlap': args['interiorVertexIds'][0] = args['rimVertexIds'][0]
    elif damage == 'missingRim': args['rimVertexIds'] = np.delete(args['rimVertexIds'], 1)
    elif damage == 'duplicateRim': args['rimVertexIds'][0] = args['rimVertexIds'][1]
    elif damage == 'outsideInterior': args['interiorVertexIds'][0] = 0
    elif damage == 'nanRest': args['alignedRestVertices'][0, 0] = np.nan
    elif damage == 'invalidIndex': args['sampleVertexIndices'][0] = 100
    elif damage == 'negativeWeight': args['sampleWeights'][0] = -1
    elif damage == 'nonconvexPositive':
        args['sampleVertexIndices'] = np.repeat(args['sampleVertexIndices'], 2, axis=1)
        args['sampleBarycentric'] = np.tile([1.2, -.2], (len(args['sampleWeights']), 1))
    elif damage == 'wrongBarySum': args['sampleBarycentric'][0] = .5
    elif damage == 'nanDirection': args['inwardDirection'] = [0., np.nan, 1.]
    elif damage == 'tangentDirection': args['inwardDirection'] = [1., 0., 0.]
    elif damage == 'zeroScale': args['depthScale'] = 0.
    elif damage == 'nanScale': args['depthScale'] = np.nan
    elif damage == 'zeroThreshold': args['supportThreshold'] = 0.
    elif damage == 'largeThreshold': args['supportThreshold'] = 1.1
    elif damage == 'largeFraction': args['targetDepthFraction'] = 1.1
    elif damage == 'zeroReference': args['referenceSupport'] = 0.
    elif damage == 'floatIds': args['interiorVertexIds'] = args['interiorVertexIds'].astype(float)
    with pytest.raises(ValueError):
        buildUnsupportedCavityPrior(**args)
