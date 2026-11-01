import copy

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from faceAppearance.surfaceContactConstraints import (
    buildSurfaceContactSpec, evaluateSurfaceContacts,
    evaluateSurfaceContactResidual, evaluateSurfaceContactSafety,
)


def fixture():
    vertices = np.asarray([[-3., -2., 0.], [3., -2., 0.], [0., 4., 0.],
        [-.5, 0., -.2], [.6, .1, .25], [0., .6, 1.5]])
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    groups = [{'name': 'firstPatch', 'sourceVertexIds': np.asarray([3, 4, 5]),
        'targetFaceIds': np.asarray([0]), 'normalSign': 1,
        'orientationProvenance': 'Synthetic target winding points along positive Z.'}]
    return dict(baselineVertices=vertices, faces=faces, contactGroups=groups,
        lengthScale=1., searchRadius=2., activationDistance=1., weight=1.)


def build(**changes):
    arguments = fixture()
    arguments.update(changes)
    return buildSurfaceContactSpec(**arguments)


def testNearestFeatureSignedOrientationAndInheritedBudgetAreExplicit():
    spec = build()
    np.testing.assert_array_equal(spec['sourceVertexIds'], [3, 4])
    np.testing.assert_allclose(spec['baselineGaps'], [-.2, .25])
    np.testing.assert_allclose(spec['lowerGaps'], [-.2, 0])
    np.testing.assert_array_equal(spec['inheritedPenetrationMask'], [True, False])
    assert spec['diagnostics']['inheritedNegativeGapCount'] == 1
    assert spec['diagnostics']['insideSearchButInactiveCount'] == 1
    assert not spec['diagnostics']['orientationFromBaselineGap']
    assert not spec['diagnostics']['isCollisionCertificate']
    initial = evaluateSurfaceContactSafety(spec['baselineVertices'], spec)
    assert initial['accepted'] and initial['baselineNegativeGapCount'] == 1
    assert initial['strictFullIntersectionGateRequired']
    deeper = spec['baselineVertices'].copy(); deeper[3, 2] -= .01
    assert not evaluateSurfaceContactSafety(deeper, spec)['accepted']
    repaired = spec['baselineVertices'].copy(); repaired[3, 2] = -.01
    assert evaluateSurfaceContactSafety(repaired, spec)['accepted']
    assert evaluateSurfaceContacts(repaired, spec)['energy'] < evaluateSurfaceContacts(spec['baselineVertices'], spec)['energy']


def testAnalyticHardResidualAndEnergyJacobiansAgainstFiniteDifferences():
    spec = build(weight=2.3)
    vertices = spec['baselineVertices'].copy()
    vertices[4, 2] = -.3
    vertices[:3, 2] += [.01, -.03, .02]
    checked = evaluateSurfaceContacts(vertices, spec)
    assert sparse.isspmatrix_csr(checked['jacobian'])
    assert sparse.isspmatrix_csr(checked['residualJacobian'])
    numericHard = np.empty(checked['jacobian'].shape)
    numericResidual = np.empty_like(numericHard)
    numericEnergy = np.empty(vertices.size)
    epsilon = 1e-6
    for column in range(vertices.size):
        plus, minus = vertices.copy(), vertices.copy()
        plus.ravel()[column] += epsilon; minus.ravel()[column] -= epsilon
        a, b = evaluateSurfaceContacts(plus, spec, False), evaluateSurfaceContacts(minus, spec, False)
        numericHard[:, column] = (a['values']-b['values'])/(2*epsilon)
        numericResidual[:, column] = (a['residual']-b['residual'])/(2*epsilon)
        numericEnergy[column] = (a['energy']-b['energy'])/(2*epsilon)
    np.testing.assert_allclose(checked['jacobian'].toarray(), numericHard, atol=5e-11)
    np.testing.assert_allclose(checked['residualJacobian'].toarray(), numericResidual, atol=5e-11)
    np.testing.assert_allclose(checked['energyGradient'].ravel(), numericEnergy, atol=5e-11)
    # Translation cancels exactly in the two-sided relative coordinate rows.
    np.testing.assert_allclose(checked['jacobian'] @ np.tile([.4, -.2, .7], len(vertices)), 0, atol=1e-15)
    assert 'jacobian' not in evaluateSurfaceContacts(vertices, spec, False)
    hook = evaluateSurfaceContactResidual(vertices, spec)
    np.testing.assert_array_equal(hook['residual'], checked['residual'])
    np.testing.assert_array_equal(hook['jacobian'].toarray(), checked['residualJacobian'].toarray())
    assert set(evaluateSurfaceContactResidual(vertices, spec, False)) == {'residual'}


def testIndependentResidualNormalizationAndWeightZeroDoesNotDisableHardGate():
    spec = build(weight=7.)
    current = spec['baselineVertices']
    expected = np.sqrt(7./2)*np.asarray([-.2, 0.])
    np.testing.assert_allclose(evaluateSurfaceContactResidual(current, spec)['residual'], expected)
    zero = build(weight=0.)
    deeper = current.copy(); deeper[3, 2] -= 1.
    checked = evaluateSurfaceContacts(deeper, zero)
    assert checked['energy'] == 0 and checked['residualJacobian'].nnz == 0
    assert not evaluateSurfaceContactSafety(deeper, zero)['accepted']


def testTwoMovableOpposingLipPatchesRestoreGapWithBothJacobians():
    lower = np.asarray([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]])
    vertices = np.vstack((lower, lower+[0, 0, -.2]))
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    groups = [dict(name='upperToLower', sourceVertexIds=np.arange(3, 6), targetFaceIds=np.asarray([0]),
                   normalSign=1, orientationProvenance='Lower opposing surface outward into gap +Z.'),
              dict(name='lowerToUpper', sourceVertexIds=np.arange(3), targetFaceIds=np.asarray([1]),
                   normalSign=-1, orientationProvenance='Upper opposing surface outward into gap -Z.')]
    spec = buildSurfaceContactSpec(vertices, faces, groups, lengthScale=1., searchRadius=2., activationDistance=1., weight=30.)
    np.testing.assert_allclose(spec['baselineGaps'], -.2)
    evaluated = evaluateSurfaceContacts(vertices, spec)
    assert np.linalg.norm(evaluated['energyGradient'][:3]) > 0
    assert np.linalg.norm(evaluated['energyGradient'][3:]) > 0

    def objective(flat):
        moved = flat.reshape(vertices.shape)
        contact = evaluateSurfaceContacts(moved, spec)
        delta = moved-vertices
        return .5*np.sum(delta**2)+contact['energy'], (delta+contact['energyGradient']).ravel()

    solved = minimize(objective, vertices.ravel(), jac=True, method='L-BFGS-B', options={'gtol': 1e-12, 'ftol': 1e-15})
    assert solved.success
    moved = solved.x.reshape(vertices.shape)
    np.testing.assert_allclose(evaluateSurfaceContacts(moved, spec)['gaps'], -.2/(1+2*30/3), atol=1e-10)
    np.testing.assert_allclose(moved.mean(axis=0), vertices.mean(axis=0), atol=1e-12)
    assert np.all(moved[:3, 2] < vertices[:3, 2])
    assert np.all(moved[3:, 2] > vertices[3:, 2])
    assert evaluateSurfaceContactSafety(moved, spec)['accepted']
    # An existing signed violation is still present: the optimization did not
    # convert grandfathering into an anatomical/collision acceptance claim.
    assert evaluateSurfaceContactSafety(moved, spec)['currentNegativeGapCount'] == 6


def testFixedTargetVerticesCanBeEliminatedExactlyByCaller():
    spec = build(weight=10.)
    baseline = spec['baselineVertices']
    source = int(spec['sourceVertexIds'][0])

    def objective(point):
        current = baseline.copy(); current[source] = point
        checked = evaluateSurfaceContacts(current, spec)
        delta = point-baseline[source]
        return .5*(delta @ delta)+checked['energy'], delta+checked['energyGradient'][source]

    solved = minimize(objective, baseline[source], jac=True, method='BFGS', options={'gtol': 1e-12})
    current = baseline.copy(); current[source] = solved.x
    np.testing.assert_array_equal(current[:3], baseline[:3])
    assert abs(evaluateSurfaceContacts(current, spec)['gaps'][0]) < abs(spec['baselineGaps'][0])


def testRigidTransformUnitCovarianceAndOrientationWinding():
    args = fixture()
    original = buildSurfaceContactSpec(**args)
    rotation, _ = np.linalg.qr(np.asarray([[1., 2., 3.], [-2., 1., 4.], [3., -1., 2.]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    unit = 17.3
    translation = np.asarray([11., -7., 2.])
    args['baselineVertices'] = unit*(args['baselineVertices'] @ rotation.T)+translation
    for name in ('lengthScale', 'searchRadius', 'activationDistance'):
        args[name] *= unit
    transformed = buildSurfaceContactSpec(**args)
    current = original['baselineVertices'].copy(); current[3, 2] -= .02
    before = evaluateSurfaceContacts(current, original)
    after = evaluateSurfaceContacts(unit*current @ rotation.T+translation, transformed)
    np.testing.assert_allclose(after['values'], before['values'], atol=3e-15)
    np.testing.assert_allclose(after['residual'], before['residual'], atol=3e-15)
    np.testing.assert_allclose(after['energyGradient'], before['energyGradient'] @ rotation.T/unit, atol=3e-15)
    # Winding reversal is safe only with the corresponding EXPLICIT sign flip.
    reversedArgs = fixture(); reversedArgs['faces'] = reversedArgs['faces'][:, [0, 2, 1]]
    reversedArgs['contactGroups'][0]['normalSign'] = -1
    reversedSpec = buildSurfaceContactSpec(**reversedArgs)
    np.testing.assert_allclose(reversedSpec['gapOperator'].toarray(), original['gapOperator'].toarray(), atol=1e-15)
    reversedArgs['contactGroups'][0]['normalSign'] = 1
    wrongSign = buildSurfaceContactSpec(**reversedArgs)
    # Incorrect provenance/sign is not silently repaired from point position.
    np.testing.assert_allclose(wrongSign['baselineGaps'][:2], -original['baselineGaps'])


def testVertexFaceAndSourceOrderPermutationPreservesPhysicalRows():
    args = fixture()
    original = buildSurfaceContactSpec(**args)
    newToOld = np.asarray([4, 2, 5, 0, 3, 1]); oldToNew = np.argsort(newToOld)
    args['baselineVertices'] = args['baselineVertices'][newToOld]
    args['faces'] = oldToNew[args['faces'][::-1]]
    args['contactGroups'][0]['sourceVertexIds'] = oldToNew[args['contactGroups'][0]['sourceVertexIds'][::-1]]
    args['contactGroups'][0]['targetFaceIds'] = np.asarray([1])
    permuted = buildSurfaceContactSpec(**args)
    order = np.argsort(newToOld[permuted['sourceVertexIds']])
    columns = (3*oldToNew[:, None]+np.arange(3)).ravel()
    np.testing.assert_allclose(permuted['gapOperator'][order][:, columns].toarray(), original['gapOperator'].toarray(), atol=1e-15)
    np.testing.assert_allclose(permuted['baselineGaps'][order], original['baselineGaps'])


def testSearchIsExactForLongTriangleInsteadOfCentroidNearestHeuristic():
    vertices = np.asarray([[-100., -100., 0.], [100., -100., 0.], [0., 100., 0.],
        [-.1, -.1, .7], [.1, -.1, .7], [0., .1, .7], [0., 0., .1]])
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    groups = [dict(name='largeFace', sourceVertexIds=np.asarray([6]), targetFaceIds=np.asarray([0, 1]),
                   normalSign=1, orientationProvenance='Both planes use positive Z winding.')]
    spec = buildSurfaceContactSpec(vertices, faces, groups, lengthScale=1., searchRadius=2., activationDistance=1.)
    assert spec['targetFaceIds'][0] == 0
    np.testing.assert_allclose(spec['targetBarycentric'][0], [.25, .25, .5])
    assert spec['diagnostics']['allTargetFacesEnumerated']


@pytest.mark.parametrize('point, expectedBary', [([0., -3., .2], [.5, .5, 0.]),
    ([-4., -3., .2], [1., 0., 0.]), ([0., 0., .2], [1/3, 1/3, 1/3])])
def testClosestEdgeCornerAndInteriorFeatures(point, expectedBary):
    args = fixture(); args['baselineVertices'][3] = point
    spec = buildSurfaceContactSpec(**args)
    np.testing.assert_allclose(spec['selectionEvidence'][0]['nearestBarycentric'][0], expectedBary, atol=1e-15)


def testNoFarComponentFallbackAndNegativeGapActivationInsideSearch():
    args = fixture(); args['baselineVertices'][3, 2] = -1.8; args['baselineVertices'][4, 2] = -2.1
    spec = buildSurfaceContactSpec(**args)
    np.testing.assert_array_equal(spec['sourceVertexIds'], [3])
    assert spec['diagnostics']['outsideSearchRadiusCount'] == 1
    assert spec['diagnostics']['insideSearchButInactiveCount'] == 1
    evidence = spec['selectionEvidence'][0]
    np.testing.assert_array_equal(evidence['activeMask'], [True, False, False])


def testIncidentFaceExclusionEmptyGroupsAndNoActiveRowsAreExplicit():
    args = fixture(); args['contactGroups'][0]['sourceVertexIds'] = np.asarray([0])
    spec = buildSurfaceContactSpec(**args)
    assert spec['diagnostics']['noEligibleTargetCount'] == 1
    assert spec['selectionEvidence'][0]['nearestTargetFaceIds'][0] == -1
    for empty in (spec, build(contactGroups=[]), build(activationDistance=0., baselineVertices=fixture()['baselineVertices']+[0., 0., 0.])):
        if empty['diagnostics']['activeRowCount']:
            continue
        evaluated = evaluateSurfaceContacts(empty['baselineVertices'], empty)
        assert evaluated['values'].shape == (0,)
        assert evaluated['residualJacobian'].shape == (0, empty['baselineVertices'].size)
        assert evaluated['energy'] == 0
        safety = evaluateSurfaceContactSafety(empty['baselineVertices'], empty)
        assert safety['accepted'] and safety['noActiveContacts'] and safety['minimumValue'] is None


def testFrozenCorrespondenceDoesNotPretendToTrackSlidingOrNewContacts():
    args = fixture(); args['baselineVertices'][3, 2] = .1
    spec = buildSurfaceContactSpec(**args)
    beforeFace = spec['targetFaceIds'].copy()
    moved = spec['baselineVertices'].copy(); moved[3] = [12., 0., .1]
    assert evaluateSurfaceContactSafety(moved, spec)['accepted']  # a halfspace, not proximity/attachment
    rebuiltArgs = fixture(); rebuiltArgs['baselineVertices'] = moved
    rebuilt = buildSurfaceContactSpec(**rebuiltArgs)
    assert 3 not in rebuilt['sourceVertexIds']
    np.testing.assert_array_equal(spec['targetFaceIds'], beforeFace)
    # An inactive baseline point may later cross the target while no frozen row
    # exists for it. Trust regions and independent full collision checks matter.
    moved = spec['baselineVertices'].copy(); moved[5, 2] = -.5
    assert evaluateSurfaceContactSafety(moved, spec)['accepted']
    assert not spec['diagnostics']['isCollisionCertificate']


def testInputsAreCopiedAndEvaluationNeverMutatesFrozenEvidence():
    args = fixture(); original = copy.deepcopy(args)
    spec = buildSurfaceContactSpec(**args)
    saved = copy.deepcopy(spec)
    args['baselineVertices'][:] = 100.; args['faces'][:] = 0
    args['contactGroups'][0]['sourceVertexIds'][:] = 0
    np.testing.assert_array_equal(spec['baselineVertices'], original['baselineVertices'])
    np.testing.assert_array_equal(spec['faces'], original['faces'])
    evaluated = evaluateSurfaceContacts(spec['baselineVertices'], spec)
    evaluated['jacobian'].data[:] = 7
    np.testing.assert_array_equal(spec['gapOperator'].toarray(), saved['gapOperator'].toarray())
    np.testing.assert_array_equal(spec['baselineGaps'], saved['baselineGaps'])


@pytest.mark.parametrize('changes', [dict(lengthScale=0), dict(searchRadius=-1), dict(activationDistance=3),
    dict(minimumGap=2), dict(weight=-1), dict(weight=np.inf), dict(weight=True), dict(activationDistance=np.nan),
    dict(contactGroups={}), dict(baselineVertices=np.zeros((2, 3))),
    dict(faces=np.asarray([[0, 0, 1]])), dict(faces=np.asarray([[0, 1, 9]]))])
def testInvalidGlobalInputsRejected(changes):
    with pytest.raises(ValueError):
        build(**changes)


@pytest.mark.parametrize('field, value', [('normalSign', 0), ('normalSign', True), ('normalSign', np.nan),
    ('orientationProvenance', ''), ('sourceVertexIds', np.asarray([3, 3])),
    ('sourceVertexIds', np.asarray([3.])), ('targetFaceIds', np.asarray([-1]))])
def testInvalidGroupsRejected(field, value):
    args = fixture(); args['contactGroups'][0][field] = value
    with pytest.raises(ValueError):
        buildSurfaceContactSpec(**args)


def testDegenerateTargetsBudgetAndNonfiniteEvaluationRejected():
    args = fixture(); args['baselineVertices'][2] = args['baselineVertices'][1]
    with pytest.raises(ValueError, match='nonzero area'):
        buildSurfaceContactSpec(**args)
    args = fixture()
    args['baselineVertices'] = np.vstack((args['baselineVertices'], np.zeros((4097, 3))))
    args['contactGroups'][0]['sourceVertexIds'] = np.arange(6, len(args['baselineVertices']))
    with pytest.raises(ValueError, match='budget'):
        buildSurfaceContactSpec(**args)
    spec = build(); bad = spec['baselineVertices'].copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluateSurfaceContacts(bad, spec)
    with pytest.raises(ValueError):
        evaluateSurfaceContactSafety(spec['baselineVertices'], spec, tolerance=-1.)
