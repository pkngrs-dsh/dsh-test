"""Dependent attachments are assembled before checks, without new variables.

The synthetic mapping isolates solver integration from template binding. The
binding module has separate template/domain/identity tests; these tests inspect
the exact callback arrays, derivative chain and multi-round assembly behavior.
"""
from copy import deepcopy
import hashlib

import numpy as np
import pytest
from scipy import sparse

import faceAppearance.fastFrontEyeGeometry as fast
import faceAppearance.clientResidualEyeRefinement as residual
from testClientVisibleEyeRefinement import mockFastInternals
from testClientResidualEyeRefinement import prepare
from testEyeAccessoryFollowing import binding, template, makeSpec


def installSyntheticFollowing(monkeypatch, original, faces, skin, fixed, followers, rows, *, movable=None):
    import faceAppearance.eyeAccessoryFollowing as module
    operator = sparse.lil_matrix((len(original), len(original)))
    operator[skin, skin] = 1.
    for follower, row in zip(followers, rows):
        operator[follower] = row
    operator = operator.tocsr()
    spec = dict(baselineVertices=original.copy(), faces=faces.copy(), skinVertexIds=skin.copy(),
        originalFixedVertexIds=fixed.copy(), followerVertexIds=np.asarray(followers),
        effectiveFixedVertexIds=np.setdiff1d(fixed, followers), specSha256='syntheticFrozenMapping',
        movableSkinVertexIds=np.setdiff1d(skin, fixed) if movable is None else movable.copy())
    calls = []
    def validate(candidate, **expected):
        for name, value in expected.items():
            np.testing.assert_array_equal(candidate[name], value)
        calls.append(tuple(expected))
    def apply(candidate, vertices):
        # Ignore an independently edited follower row. Its only driver is skin.
        return candidate['baselineVertices'] + operator @ (vertices-candidate['baselineVertices'])
    monkeypatch.setattr(module, 'validateEyeAccessoryFollowingSpec', validate)
    monkeypatch.setattr(module, 'applyEyeAccessoryFollowing', apply)
    monkeypatch.setattr(module, 'eyeAccessoryDisplacementOperator', lambda _: operator.copy())
    return spec, operator, apply, calls


def fastFixture(monkeypatch):
    original = np.r_[mockFastInternals(monkeypatch), [[3., .1, 5.], [20., 0., 5.]]]
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    skin, fixed, followers = np.arange(3), np.arange(3, 8), np.array([3, 4, 6])
    rows = np.zeros((3, len(original)))
    rows[0, :2] = [.25, .75]; rows[1, :2] = [.5, .5]; rows[2, 2] = 1.
    spec, operator, apply, calls = installSyntheticFollowing(
        monkeypatch, original, faces, skin, fixed, followers, rows)
    def invoke(**kwargs):
        return fast.refineFastFrontEyes(original, faces, np.array([0, 0]),
            dict(rotation=np.eye(3)), {'imageLeft': {}, 'imageRight': {}},
            skinVertexIds=skin, cornerFaceIds=np.array([1]), fixedVertexIds=fixed,
            imageShape=(256, 256), **kwargs)
    return original, spec, operator, apply, invoke, calls


def testContactJacobianIncludesFollowerTargetAndFullCallbacksSeeAssembledSkin(monkeypatch):
    original, spec, operator, apply, invoke, validationCalls = fastFixture(monkeypatch)
    captured = []
    fullJacobian = sparse.csr_matrix(([1., -1., 2., -1.], ([0]*4, [0, 9, 19, 23])),
        shape=(1, 3*len(original)))
    def extra(full, withJacobian):
        assert not full.flags.writeable
        np.testing.assert_allclose(full, apply(spec, full), rtol=0, atol=0)
        np.testing.assert_array_equal(full[7], original[7])
        output = dict(residual=np.asarray(fullJacobian @ full.ravel()).ravel())
        if withJacobian:
            output['jacobian'] = fullJacobian
        return output
    def fit(local, *args, **options):
        np.testing.assert_array_equal(options['fixedVertexIds'], [3, 4, 5])
        proposal = local.copy(); proposal[0, 0] += .01; proposal[2, 1] += .002
        value = options['additionalResidual'](proposal, True)
        derivative = value['jacobian'].toarray()
        assert derivative.shape == (1, 18)
        assert derivative[0, 0] == .75 and derivative[0, 3] == -.75 and derivative[0, 7] == 2.
        assert not np.any(derivative[:, 9:])
        step = 1e-5
        numeric = np.zeros_like(derivative)
        for column in range(proposal.size):
            plus, minus = proposal.copy(), proposal.copy()
            plus.flat[column] += step; minus.flat[column] -= step
            numeric[:, column] = (options['additionalResidual'](plus, False)['residual']
                - options['additionalResidual'](minus, False)['residual'])/(2*step)
        np.testing.assert_allclose(derivative, numeric, rtol=1e-7, atol=1e-8)
        safetyResult = options['additionalSafety'](proposal)
        assert safetyResult['accepted']
        expectedFull = original.copy(); expectedFull[:6] = proposal
        expectedFull = apply(spec, expectedFull)
        assert safetyResult['accessoryFollowingSnapshot'] == dict(
            specSha256=spec['specSha256'],
            fullVerticesSha256=hashlib.sha256(expectedFull.tobytes()).hexdigest(),
            changedFollowerVertexCount=3, effectiveFixedExact=True)
        captured.append(proposal.copy())
        return dict(vertices=proposal, diagnostics={})
    def safety(full):
        np.testing.assert_allclose(full, apply(spec, full), rtol=0, atol=0)
        return dict(accepted=True)
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    result = invoke(rounds=1, additionalResidual=extra, additionalSafety=safety, accessoryFollowingSpec=spec)
    assert captured and validationCalls
    assert result['diagnostics']['accessoryDegreesOfFreedom'] == 0
    np.testing.assert_allclose(result['vertices'], apply(spec, result['vertices']), rtol=0, atol=0)
    np.testing.assert_array_equal(result['vertices'][[5, 7]], original[[5, 7]])


def testTwoRoundsAndBacktrackingAlwaysUseGlobalBaselineWithoutDoubleFollowing(monkeypatch):
    original, spec, operator, apply, invoke, _ = fastFixture(monkeypatch)
    solverReferences, gateCalls, measured = [], [], []
    oldMeasure = fast.extractVisibleEyeContours
    def measure(full, *args, **options):
        np.testing.assert_allclose(full, apply(spec, full), rtol=0, atol=0)
        measured.append(full.copy())
        return oldMeasure(full, *args, **options)
    def fit(local, *args, **options):
        solverReferences.append(local.copy())
        np.testing.assert_array_equal(options['fixedVertexIds'], [3, 4, 5])
        candidate = local.copy(); candidate[0, 0] += .02
        return dict(vertices=candidate, diagnostics={})
    def geometry(reference, candidate, *args, **options):
        np.testing.assert_array_equal(options['fixedVertexIds'], [5])
        np.testing.assert_array_equal(reference, original[:6])
        return dict(accepted=True, reasons=[])
    def gate(full, context):
        np.testing.assert_allclose(full, apply(spec, full), rtol=0, atol=0)
        gateCalls.append((context['round'], context['scale'], full.copy()))
        return dict(accepted=context['scale'] == (.5 if context['round'] == 1 else .25))
    monkeypatch.setattr(fast, 'extractVisibleEyeContours', measure)
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    monkeypatch.setattr(fast, 'checkFrontGeometry', geometry)
    result = invoke(rounds=2, candidateGate=gate, accessoryFollowingSpec=spec)
    assert len(solverReferences) == 2
    np.testing.assert_allclose(solverReferences[1][3, 0], original[3, 0]+.0025)
    np.testing.assert_allclose(result['vertices'][0, 0], .015)
    np.testing.assert_allclose(result['vertices'][3, 0], original[3, 0]+.00375)
    np.testing.assert_allclose(result['vertices'][4, 0], original[4, 0]+.0075)
    assert [(roundIndex, scale) for roundIndex, scale, _ in gateCalls] == [(1, 1.), (1, .5), (2, 1.), (2, .5), (2, .25)]
    assert len(measured) >= 4
    np.testing.assert_array_equal(spec['baselineVertices'], original)


def testFullCandidateRejectionReturnsExactBaselineIncludingDependentAccessories(monkeypatch):
    original, spec, _, apply, invoke, _ = fastFixture(monkeypatch)
    def fit(local, *args, **options):
        candidate = local.copy(); candidate[0, 0] += .01
        return dict(vertices=candidate, diagnostics={})
    seen = []
    def gate(full, context):
        np.testing.assert_allclose(full, apply(spec, full), rtol=0, atol=0)
        assert full[3, 0] != original[3, 0]
        seen.append(full.copy())
        return dict(accepted=False)
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    result = invoke(rounds=2, candidateGate=gate, accessoryFollowingSpec=spec)
    assert len(seen) == 3
    np.testing.assert_array_equal(result['vertices'], original)


def testNoneFollowingRetainsLegacyPathWithoutNewEvidenceKeys(monkeypatch):
    _, _, _, _, invoke, calls = fastFixture(monkeypatch)
    def fit(local, *args, **options):
        assert options['additionalSafety'](local) == {'accepted': True}
        proposal = local.copy(); proposal[0, 0] += .01
        return dict(vertices=proposal, diagnostics={})
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    implicit = invoke(rounds=1, additionalSafety=lambda full: {'accepted': True})
    explicit = invoke(rounds=1, additionalSafety=lambda full: {'accepted': True}, accessoryFollowingSpec=None)
    np.testing.assert_array_equal(implicit['vertices'], explicit['vertices'])
    assert implicit['diagnostics']['stages'][0]['selectedScale'] == explicit['diagnostics']['stages'][0]['selectedScale']
    assert 'accessoryFollowingEnabled' not in explicit['diagnostics']
    assert not calls


def testResidualWrapperSeparatesPhysicalFixedDomainFromFrozenSolverSupport(monkeypatch):
    prepared = prepare()
    original = prepared['scene']['vertices']
    originalSupport = deepcopy(prepared['support'])
    followers = np.unique(prepared['scene']['faces'][prepared['cornerFaceIds']])
    movable = np.flatnonzero(originalSupport['movableMask'])
    rows = np.zeros((len(followers), len(original))); rows[:, movable[0]] = 1.
    spec, _, apply, _ = installSyntheticFollowing(monkeypatch, original,
        prepared['scene']['faces'], prepared['scene']['nicpToFullIds'], originalSupport['fixedVertexIds'],
        followers, rows, movable=movable)
    seen = []
    strictOld = residual._strictCandidateChecks
    def strict(reference, candidate, scene, camera, support, pairs):
        assert not np.intersect1d(support['fixedVertexIds'], followers).size
        np.testing.assert_array_equal(support['fixedVertexIds'], spec['effectiveFixedVertexIds'])
        seen.append(support)
        return strictOld(reference, candidate, scene, camera, support, pairs)
    def fit(vertices, *args, **options):
        np.testing.assert_array_equal(options['fixedVertexIds'], originalSupport['fixedVertexIds'])
        assert options['accessoryFollowingSpec']['specSha256'] == spec['specSha256']
        options['candidateGate'](vertices.copy(), dict(round=1, scale=1., geometry={'accepted': True}))
        return dict(vertices=vertices.copy(), beforeContours=prepared['beforeContours'],
            afterContours=prepared['beforeContours'], diagnostics=dict(status='noImprovingSafeCandidate',
                after=prepared['beforeApertureMetrics'], stages=[]))
    monkeypatch.setattr(residual, '_strictCandidateChecks', strict)
    monkeypatch.setattr(residual, 'refineFastFrontEyes', fit)
    result = residual.refineClientResidualEyes(prepared, accessoryFollowingSpec=spec)
    assert len(seen) == 2
    np.testing.assert_array_equal(result['vertices'], original)
    for key in ('movableMask', 'fixedMask', 'fixedVertexIds'):
        np.testing.assert_array_equal(prepared['support'][key], originalSupport[key])
        np.testing.assert_array_equal(result['support'][key], originalSupport[key])
    assert result['diagnostics']['effectiveFixedDomainExactlyPreserved']
    assert result['diagnostics']['frozenSolverSupportPreserved']


def testResidualFollowingCannotReplaceTheFrozenMovableSkinSupport(monkeypatch):
    prepared = prepare(); scene = prepared['scene']; support = prepared['support']
    followers = np.unique(scene['faces'][prepared['cornerFaceIds']])
    movable = np.flatnonzero(support['movableMask'])
    rows = np.zeros((len(followers), len(scene['vertices']))); rows[:, movable[0]] = 1.
    spec, _, _, _ = installSyntheticFollowing(monkeypatch, scene['vertices'], scene['faces'],
        scene['nicpToFullIds'], support['fixedVertexIds'], followers, rows, movable=movable[:-1])
    monkeypatch.setattr(residual, 'refineFastFrontEyes', lambda *a, **k: pytest.fail('Invalid support must fail before fitting'))
    with pytest.raises(ValueError, match='exact frozen movable skin support'):
        residual.refineClientResidualEyes(prepared, accessoryFollowingSpec=spec)


def testRealTemplateSpecPassesThroughFastAssemblyAndContactDerivative(monkeypatch, binding, template):
    """Exercise the real 264-row mapping and validation with a synthetic solve."""
    from faceAppearance.eyeAccessoryFollowing import applyEyeAccessoryFollowing, eyeAccessoryDisplacementOperator
    mockFastInternals(monkeypatch)
    spec = makeSpec(binding); original = spec['baselineVertices']; skin = spec['skinVertexIds']
    arrays, components = template
    cornerFaces = np.concatenate([part['faceIndices'] for part in components
        if part['name'] in ('male00SG.component001', 'male00SG.component002')])
    localIds = np.union1d(skin, np.unique(spec['faces'][cornerFaces]))
    driverMask = np.isin(localIds, skin)
    first = binding['components'][0]
    target = int(first['targetVertexIds'][0]); source = int(first['sourceVertexIndices'][0, 0])
    fullJacobian = sparse.csr_matrix(([1., -1.], ([0, 0], [3*source, 3*target])),
        shape=(1, original.size))
    expected = fullJacobian @ sparse.kron(eyeAccessoryDisplacementOperator(spec)[:, localIds],
        sparse.eye(3, format='csr'), format='csr')
    def extra(full, withJacobian):
        np.testing.assert_array_equal(full, applyEyeAccessoryFollowing(spec, full))
        result = dict(residual=np.array([full[source, 0]-full[target, 0]]))
        if withJacobian: result['jacobian'] = fullJacobian
        return result
    seen = []
    def fit(local, *args, **options):
        candidate = local.copy(); candidate[driverMask, 0] += .01
        actual = options['additionalResidual'](candidate, True)
        np.testing.assert_allclose(actual['jacobian'].toarray(), expected.toarray(), rtol=0, atol=0)
        seen.append(candidate)
        return dict(vertices=candidate, diagnostics={})
    monkeypatch.setattr(fast, 'refineFrontLandmarks', fit)
    result = fast.refineFastFrontEyes(original, spec['faces'], arrays['materialIds'],
        dict(rotation=np.eye(3)), {'imageLeft': {}, 'imageRight': {}}, skinVertexIds=skin,
        cornerFaceIds=cornerFaces, fixedVertexIds=spec['originalFixedVertexIds'], imageShape=(256, 256),
        rounds=1, additionalResidual=extra, accessoryFollowingSpec=spec)
    assert seen
    np.testing.assert_array_equal(result['vertices'], applyEyeAccessoryFollowing(spec, result['vertices']))
    np.testing.assert_array_equal(result['vertices'][spec['effectiveFixedVertexIds']],
        original[spec['effectiveFixedVertexIds']])
    np.testing.assert_allclose(result['vertices'][spec['followerVertexIds'], 0]
        - original[spec['followerVertexIds'], 0], .01, rtol=0, atol=1e-13)
