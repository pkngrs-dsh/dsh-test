"""Residual fitting starts after coarse alignment and keeps contact in the solve."""
from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

import faceAppearance.clientResidualEyeRefinement as residual
import faceAppearance.fastFrontEyeGeometry as fast
from faceAppearance.camera import projectPoints
from faceAppearance.surfaceContactConstraints import buildSurfaceContactSpec
from testClientVisibleEyeRefinement import rasterInputs, mockFastInternals, invokeFast


def coarseFor(scene, camera):
    skin = np.zeros(len(scene['vertices']), bool); skin[scene['nicpToFullIds']] = True
    candidates = scene['faces'][skin[scene['faces']].all(1)]
    output = []
    for name, sign, start in [('imageLeftEye', -1, 36), ('imageRightEye', 1, 42)]:
        selected = candidates[sign*scene['vertices'][candidates].mean(1)[:, 0] > 0]
        selected = selected[np.linspace(0, len(selected)-1, 6).astype(int)]
        bary = np.full((6, 3), 1/3)
        points = (scene['vertices'][selected]*bary[..., None]).sum(1)
        pixels = projectPoints(points, camera['intrinsics'], camera['rotation'], camera['translation'])[0]
        output.append(dict(name=name, ibugIds=np.arange(start, start+6), vertexIndices=selected,
            barycentric=bary, targetPixels=pixels, weights=np.ones(6), confidence=np.ones(6)))
    return output


def prepare(policy=None):
    scene, camera, eyes, corners = rasterInputs()
    return residual.prepareClientResidualEyes(scene, camera, eyes, coarseEyeConstraints=coarseFor(scene, camera),
        imageShape=(256, 256), cornerFaceIds=corners,
        policy=policy or {'rounds': 1, 'iterationsPerRound': 1})


def specFor(prepared, *, radius=2.):
    scene = prepared['scene']
    source = np.flatnonzero(prepared['support']['movableMask'])
    # The synthetic octahedra have known outward winding. This is independent
    # of which side their surrounding skin happens to occupy.
    groups = [dict(name='syntheticOpaqueEyes', sourceVertexIds=source,
        targetFaceIds=np.flatnonzero(scene['materialIds'] == 2), normalSign=1,
        orientationProvenance='known outward-wound synthetic octahedra')]
    return buildSurfaceContactSpec(scene['vertices'], scene['faces'], groups,
        lengthScale=1., searchRadius=radius, activationDistance=radius, minimumGap=0., weight=1.)


def testPreparationKeepsCompletedBaselineAndAllAccessoriesIncludingCanthi():
    p = prepare(); scene = p['scene']
    nonSkin = np.ones(len(scene['vertices']), bool); nonSkin[scene['nicpToFullIds']] = False
    assert p['support']['diagnostics']['status'] == 'supported'
    assert p['support']['fixedMask'][nonSkin].all()
    assert p['support']['fixedMask'][np.unique(scene['faces'][p['cornerFaceIds']])].all()
    assert not p['diagnostics']['coarseStagesRerun']
    assert not p['diagnostics']['coarseStagesSkipped']
    assert not p['diagnostics']['coarseStagesRestricted']
    assert residual.evaluateResidualCoarseAlignment(p, scene['vertices'])['accepted']


def testPreparationOwnsItsCopies():
    scene, camera, eyes, corners = rasterInputs(); constraints = coarseFor(scene, camera)
    p = residual.prepareClientResidualEyes(scene, camera, eyes, coarseEyeConstraints=constraints,
        imageShape=(256, 256), cornerFaceIds=corners)
    old = deepcopy(p)
    scene['vertices'][:] += 1.; camera['translation'][:] += 1.
    eyes['imageLeft']['upper'][:] += 1.; constraints[0]['targetPixels'][:] += 1.
    for key in ('vertices', 'faces', 'uvs', 'uvFaces'):
        np.testing.assert_array_equal(p['scene'][key], old['scene'][key])
    np.testing.assert_array_equal(p['camera']['translation'], old['camera']['translation'])
    np.testing.assert_array_equal(p['photoEyes']['imageLeft']['upper'], old['photoEyes']['imageLeft']['upper'])
    np.testing.assert_array_equal(p['coarseEyeConstraints'][0]['targetPixels'], old['coarseEyeConstraints'][0]['targetPixels'])


@pytest.mark.parametrize('field', ['vertices', 'photoEyes', 'support', 'camera', 'coarse'])
def testChangingPreparationCannotSilentlyChangeBaselineOrTargets(field):
    p = prepare()
    if field == 'vertices': p['scene']['vertices'][0, 0] += .01
    elif field == 'photoEyes': p['photoEyes']['imageLeft']['upper'][1, 0] += .1
    elif field == 'support': p['support']['fixedMask'][0] ^= True
    elif field == 'camera': p['camera']['translation'][0] += .01
    else: p['coarseEyeConstraints'][0]['targetPixels'][0, 0] += .1
    with pytest.raises(ValueError, match='Prepared baseline'):
        residual.refineClientResidualEyes(p)


@pytest.mark.parametrize('change', ['missingEye', 'wrongEyeIds', 'accessoryBinding', 'wrongBarycentric', 'outsidePhoto'])
def testOriginalCoarseBindingAndPhotoContractIsRequired(change):
    scene, camera, eyes, corners = rasterInputs(); constraints = coarseFor(scene, camera)
    if change == 'missingEye': constraints.pop()
    elif change == 'wrongEyeIds': constraints[0]['ibugIds'][0] = 42
    elif change == 'accessoryBinding': constraints[0]['vertexIndices'][0] = scene['faces'][corners[0]]
    elif change == 'wrongBarycentric': constraints[0]['barycentric'][0] *= .5
    else: constraints[0]['targetPixels'][0, 0] = 256.
    with pytest.raises(ValueError):
        residual.prepareClientResidualEyes(scene, camera, eyes, coarseEyeConstraints=constraints,
            imageShape=(256, 256), cornerFaceIds=corners)


def testOriginalCoarseMeanAndMaximumRemainHardGates():
    p = prepare(); proposed = p['scene']['vertices'].copy()
    proposed[:, 0] += .01
    check = residual.evaluateResidualCoarseAlignment(p, proposed)
    assert not check['accepted']
    assert check['tolerancePixels'] == .025
    for eye in check['eyes'].values():
        assert eye['after']['meanErrorPixels'] > eye['before']['meanErrorPixels']+.025
        assert eye['after']['maximumErrorPixels'] > eye['before']['maximumErrorPixels']+.025


def testImprovingCoarseMeanCannotHideOneWorseCoarsePoint():
    vertices = np.c_[np.arange(36)*.01, np.zeros(36), np.full(36, 5.)]
    camera = dict(intrinsics=np.array([[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]]),
        rotation=np.eye(3), translation=np.zeros(3))
    constraints = []
    candidate = vertices.copy()
    for name, start in zip(residual.EYE_NAMES, (0, 18)):
        indices = np.arange(start, start+18).reshape(6, 3)
        bary = np.full((6, 3), 1/3)
        target = projectPoints(vertices[indices].mean(1), camera['intrinsics'], camera['rotation'], camera['translation'])[0]
        target[1:, 0] += 1.
        candidate[indices[1:].ravel(), 0] += .05
        candidate[indices[0], 0] += .1
        constraints.append(dict(name=name, vertexIndices=indices, barycentric=bary, targetPixels=target, ibugIds=np.arange(6)))
    p = dict(scene={'vertices': vertices}, camera=camera, coarseEyeConstraints=constraints,
        beforeCoarseAlignment=residual._coarseScores(vertices, constraints, camera), imageShape=(256, 256))
    check = residual.evaluateResidualCoarseAlignment(p, candidate)
    assert not check['accepted']
    for row in check['eyes'].values():
        assert row['after']['meanErrorPixels'] < row['before']['meanErrorPixels']
        assert row['after']['maximumErrorPixels'] > row['before']['maximumErrorPixels']+.025


def testNoActiveContactsSkipWithoutInvokingAnyOptimizer(monkeypatch):
    p = prepare(); spec = specFor(p, radius=1e-6)
    assert not len(spec['sourceVertexIds'])
    monkeypatch.setattr(residual, 'refineFastFrontEyes', lambda *a, **k: pytest.fail('Empty contact rows must skip'))
    result = residual.refineClientResidualEyes(p, spec)
    assert result['diagnostics']['status'] == 'skipNoActiveContactRows'
    assert result['diagnostics']['contactOptimizationEnabled']
    np.testing.assert_array_equal(result['vertices'], p['scene']['vertices'])


def testMissingApertureMaskIsExplicitlySkippedWithoutInventingMeasurement(monkeypatch):
    scene, camera, eyes, corners = rasterInputs()
    monkeypatch.setattr(residual, 'extractVisibleEyeContours', lambda **kw: {
        'eyes': {name: {'status': 'rejected', 'reasons': ['noVisibleAperture']} for name in eyes}})
    monkeypatch.setattr(residual, 'refineFastFrontEyes', lambda *a, **k: pytest.fail('No visible mask cannot be fitted'))
    p = residual.prepareClientResidualEyes(scene, camera, eyes, coarseEyeConstraints=coarseFor(scene, camera),
        imageShape=(256, 256), cornerFaceIds=corners)
    result = residual.refineClientResidualEyes(p)
    assert result['diagnostics']['status'] == 'skippedInsufficientVisibleSupport'
    assert result['beforeApertureMetrics'] is None
    assert result['diagnostics']['apertureMeasurementFailure']
    np.testing.assert_array_equal(result['vertices'], scene['vertices'])


@pytest.mark.parametrize('change', ['baseline', 'faces', 'weight', 'outsideSupport', 'movingTarget'])
def testContactSpecificationMustMatchExactBaselineAndFrozenDomains(change):
    p = prepare(); spec = specFor(p)
    if change == 'baseline': spec['baselineVertices'][0, 0] += .01
    elif change == 'faces': spec['faces'][0] = spec['faces'][1]
    elif change == 'weight': spec['weight'] = 0.
    elif change == 'outsideSupport': spec['contactGroups'][0]['sourceVertexIds'] = np.array([len(p['scene']['vertices'])-1])
    else: spec['contactGroups'][0]['targetFaceIds'] = np.array([0])
    with pytest.raises(ValueError): residual.refineClientResidualEyes(p, spec)


def testCompleteSyntheticRunCallsContactInsideOptimizerAndKeepsStrictFullGate():
    p = prepare(); spec = specFor(p)
    assert len(spec['sourceVertexIds'])
    result = residual.refineClientResidualEyes(p, spec)
    calls = result['diagnostics']['contactEvaluationCalls']
    assert result['diagnostics']['fastSolverInvoked']
    assert calls['residual'] > 0 and calls['jacobian'] > 0 and calls['safety'] > 1
    assert result['candidateProposals']
    assert all('fullAssembly' in row['strict'] or not row['strict']['geometry']['accepted']
        for row in result['candidateProposals'])
    assert all(row['contactEvaluation'] is not None for row in result['candidateProposals'])
    assert result['diagnostics']['allNonSkinExactlyFixed']
    assert result['diagnostics']['fixedDomainExactlyPreserved']
    assert result['diagnostics']['completeApertureRoisExactlyPreserved']
    np.testing.assert_array_equal(p['scene']['vertices'], spec['baselineVertices'])


def testNoContactIsExplicitAblationButStillPreservesCoarseAlignmentAndAccessories():
    p = prepare(); result = residual.refineClientResidualEyes(p)
    assert not result['diagnostics']['contactOptimizationEnabled']
    assert result['diagnostics']['contactEvaluationCalls']['residual'] == 0
    assert result['diagnostics']['contactEvaluationCalls']['safety'] > 0
    assert result['diagnostics']['strict']['coarseAlignment']['accepted']
    assert result['contactBefore'] is None
    assert result['diagnostics']['allNonSkinExactlyFixed']


def testFastMapsFullContactJacobianToLocalSolverIdsAndLeavesOthersFixed(monkeypatch):
    original = mockFastInternals(monkeypatch)
    # Add a fixed eyeball vertex that is absent from the local skin/canthus set.
    original = np.r_[original, [[20., 0., 5.]]]
    captured = {}
    def extra(vertices, withJacobian):
        assert not vertices.flags.writeable
        np.testing.assert_array_equal(vertices[6], original[6])
        captured['full'] = vertices.copy()
        return dict(residual=np.array([vertices[0, 0]-vertices[6, 0]]),
            jacobian=sparse.csr_matrix(([1., -1.], ([0, 0], [0, 18])), shape=(1, 21)))
    def fit(vertices, *a, **kw):
        proposal = vertices.copy(); proposal[0, 0] += .001
        report = kw['additionalResidual'](proposal, True)
        assert report['jacobian'].shape == (1, 18)
        assert report['jacobian'][0, 0] == 1.
        assert report['jacobian'].nnz == 1
        assert kw['additionalSafety'](proposal)['accepted']
        return dict(vertices=proposal, diagnostics={})
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    result = invokeFast(original, additionalResidual=extra,
        additionalSafety=lambda full: {'accepted': bool(np.array_equal(full[6], original[6]))})
    assert 'full' in captured
    np.testing.assert_array_equal(result['vertices'][6], original[6])


def testFixedOnlyBoundaryOmissionKeepsPartialBindingsAndAllAperturePixels(monkeypatch):
    vertices = np.array([[0., 0., 5.], [1., 0., 5.], [0., 1., 5.]])
    faces = np.array([[0, 1, 2]])
    camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
    aperture = np.zeros((9, 9), bool); aperture[2:7, 2:7] = True
    originalMask = aperture.copy()
    measurement = {'eyes': {'imageLeft': {'roi': {'origin': [0., 0.], 'scale': 1.},
        'roiMasks': {'eye': aperture, 'eligibleBindableSkin': ~aperture}, 'diagnostics': {}}}}
    photo = {'imageLeft': {'upper': np.array([[2., 4.], [3., 2.], [5., 2.], [6., 4.]]),
        'lower': np.array([[2., 4.], [3., 6.], [5., 6.], [6., 4.]])}}
    monkeypatch.setattr(fast, '_raster', lambda *a, **k: (np.zeros((9, 9), int), np.full((9, 9, 3), 1/3), None, None))
    arguments = (vertices, faces, np.array([0]), camera, photo, measurement, np.arange(3), np.array([], int), None, None, None, None)
    legacy, legacyStats = fast._constraints(*arguments)
    fixed, fixedStats = fast._constraints(*arguments, fixedVertexIds=np.arange(3))
    partial, partialStats = fast._constraints(*arguments, fixedVertexIds=np.array([0, 1]))
    assert legacy and partial and not fixed
    assert fixedStats['imageLeft']['fullyFixedBoundarySamplesRemoved'] == legacyStats['imageLeft']['kept']
    assert partialStats['imageLeft']['fullyFixedBoundarySamplesRemoved'] == 0
    np.testing.assert_array_equal(partial[0]['targetPixels'], legacy[0]['targetPixels'])
    np.testing.assert_array_equal(aperture, originalMask)


@pytest.mark.parametrize('option', ['additionalResidual', 'additionalSafety'])
def testFastAdditionalCallbacksMustBeCallable(option):
    with pytest.raises(ValueError, match='callbacks'):
        invokeFast(np.zeros((6, 3)), **{option: True})


def testFastFixedOnlyOptionRejectsAmbiguousTruthValues():
    with pytest.raises(ValueError, match='boolean'):
        invokeFast(np.zeros((6, 3)), omitFullyFixedBoundaryConstraints=1)


def testAdditionalOuterGateReceivesEveryRawProposalAndCannotBeBypassed():
    p = prepare(); calls = []
    def gate(vertices):
        assert not vertices.flags.writeable
        calls.append(vertices.copy())
        return dict(accepted=False, legacySideView={'accepted': False})
    result = residual.refineClientResidualEyes(p, candidateGate=gate)
    assert len(calls) == len(result['candidateProposals'])+1
    assert result['candidateProposals']
    assert all(not row['strict']['externalCandidateGate']['accepted'] for row in result['candidateProposals'])
    assert not result['diagnostics']['accepted']
    np.testing.assert_array_equal(result['vertices'], p['scene']['vertices'])


@pytest.mark.parametrize('value', [None, {}, {'accepted': 1}, {'accepted': None}])
def testOuterGateMustReturnAnExplicitBoolean(value):
    p = prepare()
    with pytest.raises(ValueError, match='explicit boolean'):
        residual.refineClientResidualEyes(p, candidateGate=lambda vertices: value)
