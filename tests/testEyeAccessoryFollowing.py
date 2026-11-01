"""Fixed binding invariants and the full skin-to-accessory derivative chain."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from faceAppearance.eyeAccessoryFollowing import (
    _closestOnTriangles, _equivalentRows, applyEyeAccessoryFollowing,
    buildEyeAccessoryBinding, eyeAccessoryDisplacementOperator,
    prepareEyeAccessoryFollowing, validateEyeAccessoryFollowingSpec,
)


@pytest.fixture(scope='module')
def template():
    root = Path(__file__).resolve().parents[1]/'assets/clientTemplates/legacyMaleDirectServiceV1'
    with np.load(root/'template.npz') as archive:
        arrays = {name: archive[name] for name in archive.files}
    components = json.loads((root/'components.json').read_text())
    return arrays, components


@pytest.fixture(scope='module')
def binding(template):
    return buildEyeAccessoryBinding(*template)


def makeSpec(binding, *, fixSource=False):
    baseline = binding['templateVertices'].copy()
    skin = binding['skinVertexIds']; fixed = np.setdiff1d(np.arange(len(baseline)), skin)
    if fixSource:
        fixed = np.union1d(fixed, binding['components'][0]['sourceVertexIndices'][0, :1])
    return prepareEyeAccessoryFollowing(binding, baseline, binding['faces'], skin, fixed)


def testTemplateWhitelistAndFixedSemanticDomains(binding, template):
    assert len(binding['followerVertexIds']) == 264
    assert {row['componentName'] for row in binding['components']} == {
        'male00SG.component001', 'male00SG.component002', 'male00SG3.component000',
        'male00SG3.component001', 'male00SG3.component002', 'male00SG3.component003'}
    assert set(binding['templateArraySha256']) == set(template[0])
    for eye in binding['eyeDomains']:
        assert not np.intersect1d(eye['sourceFaceDomains']['upper'], eye['sourceFaceDomains']['lower']).size
        assert eye['upperSampleMask'].any() and eye['lowerSampleMask'].any()
    for row in binding['components']:
        assert np.isin(row['sourceVertexIndices'], binding['skinVertexIds']).all()
        assert np.isin(row['sourceFaceIds'], row['allowedFaceIds']).all()
        actual = (binding['templateVertices'][row['sourceVertexIndices']]*row['barycentricWeights'][..., None]).sum(1)
        np.testing.assert_allclose(actual, row['closestRestPoints'], atol=1e-12, rtol=0)
        assert (row['barycentricWeights'] >= 0).all()
        np.testing.assert_allclose(row['barycentricWeights'].sum(1), 1., atol=1e-13, rtol=0)


def testBuilderAndPrepareDoNotMutateInputs(binding, template):
    arrays, components = template
    hashes = {name: value.tobytes() for name, value in arrays.items()}
    originalComponents = deepcopy(components)
    again = buildEyeAccessoryBinding(arrays, components)
    assert again['bindingSha256'] == binding['bindingSha256']
    assert components == originalComponents
    assert hashes == {name: value.tobytes() for name, value in arrays.items()}
    makeSpec(binding)
    assert again['bindingSha256'] == binding['bindingSha256']


def testZeroDisplacementExactAndCandidateIsNotMutated(binding):
    spec = makeSpec(binding); baseline = spec['baselineVertices']
    before = baseline.copy(); output = applyEyeAccessoryFollowing(spec, baseline)
    assert np.array_equal(output, baseline)
    assert np.array_equal(before, baseline)
    assert not np.shares_memory(output, baseline)
    validateEyeAccessoryFollowingSpec(spec, baselineVertices=baseline, faces=spec['faces'],
        skinVertexIds=spec['skinVertexIds'], originalFixedVertexIds=spec['originalFixedVertexIds'])


def testUniformSkinTranslationAndFixedEyes(binding, template):
    spec = makeSpec(binding); baseline = spec['baselineVertices']; candidate = baseline.copy()
    translation = np.array([.04, -.02, .03]); candidate[spec['skinVertexIds']] += translation
    result = applyEyeAccessoryFollowing(spec, candidate)
    np.testing.assert_allclose(result[spec['followerVertexIds']]-baseline[spec['followerVertexIds']],
        np.broadcast_to(translation, (len(spec['followerVertexIds']), 3)), atol=5e-14, rtol=0)
    assert np.array_equal(result[spec['skinVertexIds']], candidate[spec['skinVertexIds']])
    assert np.array_equal(result[spec['effectiveFixedVertexIds']], baseline[spec['effectiveFixedVertexIds']])
    for component in template[1]:
        if component['label'] in ('leftEye', 'rightEye', 'eyebrow', 'upperTeeth', 'lowerTeeth', 'tongue'):
            assert np.array_equal(result[component['vertexIndices']], baseline[component['vertexIndices']])


def testRepeatedApplicationUsesGlobalTotalSkinDelta(binding):
    spec = makeSpec(binding); candidate = spec['baselineVertices'].copy()
    rng = np.random.default_rng(420)
    candidate[spec['skinVertexIds']] += rng.normal(size=(len(spec['skinVertexIds']), 3))*.01
    first = applyEyeAccessoryFollowing(spec, candidate)
    assert np.array_equal(first, applyEyeAccessoryFollowing(spec, first))
    candidate[spec['followerVertexIds']] += 100.
    assert np.array_equal(first, applyEyeAccessoryFollowing(spec, candidate))


def testFixedSkinWeightsAreNotRenormalized(binding):
    spec = makeSpec(binding, fixSource=True); baseline = spec['baselineVertices']; candidate = baseline.copy()
    candidate[spec['movableSkinVertexIds'], 0] += .1
    result = applyEyeAccessoryFollowing(spec, candidate)
    operator = eyeAccessoryDisplacementOperator(spec)
    expected = operator @ (candidate-baseline)
    np.testing.assert_allclose(result-baseline, expected, atol=2e-14, rtol=0)
    partial = expected[spec['followerVertexIds'], 0]
    assert ((partial > 0) & (partial < .1-1e-12)).any()


def testUnauthorizedAccessoryOrFixedSkinChangesRejected(binding):
    spec = makeSpec(binding, fixSource=True)
    fixedSkin = np.intersect1d(spec['originalFixedVertexIds'], spec['skinVertexIds'])
    fixedOther = np.setdiff1d(spec['effectiveFixedVertexIds'], spec['skinVertexIds'])
    for index in (fixedSkin[0], fixedOther[0]):
        candidate = spec['baselineVertices'].copy(); candidate[index, 0] += 1e-10
        with pytest.raises(ValueError, match='fixed vertex'):
            applyEyeAccessoryFollowing(spec, candidate)


def testSparseOperatorNoIndependentAccessoryColumns(binding):
    spec = makeSpec(binding); operator = eyeAccessoryDisplacementOperator(spec)
    assert sparse.isspmatrix_csr(operator)
    nonSkin = np.setdiff1d(np.arange(operator.shape[0]), spec['skinVertexIds'])
    assert operator[:, nonSkin].nnz == 0
    assert operator[np.setdiff1d(nonSkin, spec['followerVertexIds'])].nnz == 0
    identityDifference = operator[spec['skinVertexIds']][:, spec['skinVertexIds']]-sparse.eye(len(spec['skinVertexIds']))
    assert identityDifference.nnz == 0
    np.testing.assert_allclose(np.asarray(operator[spec['followerVertexIds']].sum(1)).ravel(), 1., atol=1e-13, rtol=0)


def testFullXyzJacobianChainMatchesFiniteDifference(binding):
    spec = makeSpec(binding, fixSource=True); baseline = spec['baselineVertices']
    rng = np.random.default_rng(1789); skinJacobian = np.zeros((baseline.size, 4))
    ids = (3*spec['movableSkinVertexIds'][:, None]+np.arange(3)).ravel()
    skinJacobian[ids] = rng.normal(size=(len(ids), 4))*.1
    xyzMap = sparse.kron(eyeAccessoryDisplacementOperator(spec), sparse.eye(3), format='csr')
    analytical = xyzMap @ skinJacobian
    epsilon = 1e-5
    for column in range(4):
        direction = skinJacobian[:, column].reshape(baseline.shape)
        plus = applyEyeAccessoryFollowing(spec, baseline+epsilon*direction)
        minus = applyEyeAccessoryFollowing(spec, baseline-epsilon*direction)
        numeric = ((plus-minus)/(2*epsilon)).ravel()
        np.testing.assert_allclose(analytical[:, column], numeric, atol=3e-9, rtol=0)
    # A residual whose source and target both move needs both chain terms.
    part = spec['binding']['components'][0]
    target = int(part['targetVertexIds'][0]); source = int(part['sourceVertexIndices'][0, 0])
    normal = np.array([.2, -.7, .3]); gapJacobian = np.zeros(baseline.size)
    gapJacobian[3*source:3*source+3] = normal
    gapJacobian[3*target:3*target+3] = -normal
    expected = gapJacobian @ analytical
    def gap(vertices): return (vertices[source]-vertices[target]) @ normal
    for column in range(4):
        direction = skinJacobian[:, column].reshape(baseline.shape)
        numerical = (gap(applyEyeAccessoryFollowing(spec, baseline+epsilon*direction))
            -gap(applyEyeAccessoryFollowing(spec, baseline-epsilon*direction)))/(2*epsilon)
        assert abs(expected[column]-numerical) < 3e-9


def testBindingAndSpecTamperRejected(binding):
    changed = deepcopy(binding); changed['components'][0]['barycentricWeights'][0, 0] += .01
    with pytest.raises(ValueError, match='binding changed'): makeSpec(changed)
    spec = makeSpec(binding); spec['baselineVertices'][0, 0] += 1e-9
    with pytest.raises(ValueError, match='specification changed'): validateEyeAccessoryFollowingSpec(spec)


def testTopologyOrderAndSupportMustStayOriginal(binding):
    spec = makeSpec(binding)
    with pytest.raises(ValueError, match='topology'):
        prepareEyeAccessoryFollowing(binding, spec['baselineVertices'], spec['faces'][::-1],
            spec['skinVertexIds'], spec['originalFixedVertexIds'])
    with pytest.raises(ValueError, match='support'):
        prepareEyeAccessoryFollowing(binding, spec['baselineVertices'], spec['faces'],
            spec['skinVertexIds'], spec['originalFixedVertexIds'], movableSkinVertexIds=spec['movableSkinVertexIds'][:-1])
    changed = spec['baselineVertices'].copy(); changed[0, 0] += 1e-9
    with pytest.raises(ValueError, match='baselineVertices'):
        validateEyeAccessoryFollowingSpec(spec, baselineVertices=changed)


@pytest.mark.parametrize('policy', [{'typo': 1}, {'innerCornerWidthFraction': .5},
    {'rimResolution': True}, {'maximumBindingDistanceEyeWidthFraction': np.nan}])
def testInvalidPoliciesRejected(template, policy):
    with pytest.raises(ValueError): buildEyeAccessoryBinding(*template, policy=policy)


def testMissingDuplicateAndCrossSideComponentsRejected(template):
    arrays, components = template
    for wrong in (components[:-3], components+[deepcopy(components[0])]):
        with pytest.raises(ValueError): buildEyeAccessoryBinding(arrays, wrong)
    changed = {name: value.copy() for name, value in arrays.items()}
    canthus = next(part for part in components if part['name'] == 'male00SG.component001')
    changed['vertices'][canthus['vertexIndices'][0], 0] *= -1
    with pytest.raises(ValueError, match='straddles'): buildEyeAccessoryBinding(changed, components)


def testClosestTriangleCoversInteriorEdgesAndVertices():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    for point, expected in [([.2, .3, 1.], [.2, .3, 0.]), ([.7, .7, .2], [.5, .5, 0.]),
            ([-2., -.4, .3], [0., 0., 0.]), ([.2, -.4, 0.], [.2, 0., 0.])]:
        bary, closest, distances = _closestOnTriangles(np.array(point), triangle)
        np.testing.assert_allclose(closest[0], expected, atol=1e-13)
        assert (bary >= 0).all() and abs(bary.sum()-1) < 1e-13
        assert abs(distances[0]-np.linalg.norm(np.asarray(point)-expected)) < 1e-13
    with pytest.raises(ValueError, match='Degenerate'):
        _closestOnTriangles(np.zeros(3), np.zeros((1, 3, 3)))


def testEquivalentTieRequiresSameNumberedSkinInterpolation():
    assert _equivalentRows(np.array([0, 1, 2]), np.array([.4, .6, 0.]),
        np.array([1, 0, 3]), np.array([.6, .4, 0.]), 1e-10)
    assert not _equivalentRows(np.array([0, 1, 2]), np.array([.4, .6, 0.]),
        np.array([3, 4, 5]), np.array([.4, .6, 0.]), 1e-10)
