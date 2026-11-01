import copy

import numpy as np
import pytest

import faceAppearance.clientEyelidAppearance as appearance


def sceneInputs(accessory=False):
    vertices = np.array([[0., 0., 1.], [255., 0., 1.], [255., 255., 1.], [0., 255., 1.]])
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    uvs = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    if accessory:
        uvs = np.vstack([uvs, [[.20, .20], [.38, .20], [.20, .38]]])
    geometry = dict(vertices=vertices.copy(), faces=faces.copy(), uvs=uvs.copy(),
        uvFaces=faces.copy(), uvConvention='topLeftPixelEdges')
    fullVertices, fullFaces, uvFaces = vertices.copy(), faces.copy(), faces.copy()
    if accessory:
        fullVertices = np.vstack([vertices, [[400., 400., 2.], [430., 400., 2.], [400., 430., 2.]]])
        fullFaces = np.vstack([faces, [4, 6, 5]])
        uvFaces = np.vstack([faces, [4, 6, 5]])
    scene = dict(vertices=fullVertices, faces=fullFaces, uvs=uvs.copy(), uvFaces=uvFaces,
        materialIds=np.zeros(len(fullFaces), np.int64), nicpToFullIds=np.arange(4),
        materials=[{'alphaMode': 'OPAQUE'}], alphaTextures={})
    camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
    eyes = {}
    for name, center in [('imageLeft', 70.), ('imageRight', 170.)]:
        eyes[name] = {'upper': np.array([[center-16, 80.], [center, 70.], [center+16, 80.]]),
            'lower': np.array([[center-16, 80.], [center, 90.], [center+16, 80.]])}
    source = np.full((256, 256, 3), .65, np.float32)
    valid = np.ones((256, 256), bool)
    baseline = np.full((64, 64, 3), .2, np.float32)
    return geometry, scene, camera, source, valid, baseline, eyes


def measured(eyes, rejected=False):
    return {'eyes': {name: {'status': 'rejected' if rejected else 'accepted',
        'reasons': ['occluded'] if rejected else [],
        **{part: {'sourcePixels': curve + [0., 3.], 'targetPixels': curve.copy()}
            for part, curve in eye.items()}} for name, eye in eyes.items()}, 'diagnostics': {}}


def testSafeSamplingPreservesInputsAndOutsideDomain(monkeypatch):
    args = sceneInputs()
    snapshots = copy.deepcopy(args)
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: measured(kwargs['photoEyes']))
    result = appearance.refineClientEyelidAppearance(*args)
    assert result['diagnostics']['status'] == 'applied'
    maps = result['maps']
    changed = np.any(result['linearRGB'] != args[5], axis=-1)
    assert changed.any()
    assert not np.any(changed & ~maps['writableMask'])
    assert np.array_equal(result['linearRGB'][~maps['modifiedMask']], args[5][~maps['modifiedMask']])
    assert result['diagnostics']['strictSourceSupportChecks']['invalidSourceSupportQueryCount'] == 0
    assert not maps['filterPadding'].any()
    for key in ('vertices', 'faces', 'uvs', 'uvFaces'):
        np.testing.assert_array_equal(args[0][key], snapshots[0][key])
        np.testing.assert_array_equal(args[1][key], snapshots[1][key])
    for index in (3, 4, 5):
        np.testing.assert_array_equal(args[index], snapshots[index])
    for name in args[-1]:
        for part in args[-1][name]:
            np.testing.assert_array_equal(args[-1][name][part], snapshots[-1][name][part])


def testSameMaterialAccessoryUvsAreProtectedWithPadding(monkeypatch):
    args = sceneInputs(accessory=True)
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: measured(kwargs['photoEyes']))
    result = appearance.refineClientEyelidAppearance(*args)
    protected = result['maps']['protectedAccessoryUvs']
    assert protected.any()
    assert np.array_equal(result['linearRGB'][protected], args[5][protected])
    assert not (result['maps']['modifiedMask'] & protected).any()
    assert protected[14, 14]  # outside triangle interior but inside the two-texel padding


def testRejectedMeasurementSkipsWithoutSampling(monkeypatch):
    args = sceneInputs()
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: measured(kwargs['photoEyes'], True))
    monkeypatch.setattr(appearance, 'reprojectEyelidTexture', lambda *a, **k: pytest.fail('Rejected contours must not sample'))
    result = appearance.refineClientEyelidAppearance(*args)
    assert result['diagnostics']['status'] == 'skippedUnreliableVisibleContours'
    np.testing.assert_array_equal(result['linearRGB'], args[5])
    assert not result['maps']['modifiedMask'].any()


def testMatchedResampledTargetsAreUsedRatherThanRawIndices(monkeypatch):
    args = sceneInputs()
    prediction = measured(args[-1])
    for name in prediction['eyes']:
        for part in ('upper', 'lower'):
            original = prediction['eyes'][name][part]
            original['targetPixels'] = np.stack([original['targetPixels'][0],
                (original['targetPixels'][0] + original['targetPixels'][1])*.5,
                original['targetPixels'][1],
                (original['targetPixels'][1] + original['targetPixels'][2])*.5,
                original['targetPixels'][2]])
            original['sourcePixels'] = original['targetPixels'] + [0., 3.]
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: prediction)
    result = appearance.refineClientEyelidAppearance(*args)
    assert result['maps']['rawPhotoCurves'].shape[-2] == 3
    assert result['maps']['resampledPhotoCurves'].shape[-2] == 5
    np.testing.assert_array_equal(result['maps']['modelCurves'], result['maps']['resampledPhotoCurves'] + [0., 3.])


def testAllFourContributorsAndBothRawAndResampledApertures():
    eyes = sceneInputs()[-1]
    queries = np.array([[[70.25, 60.25]]])
    mask = np.ones((1, 1), bool)
    valid = np.ones((256, 256), bool)
    result = appearance.validateEyelidSourceFootprints(queries, mask, valid, eyes, eyes)
    assert result['invalidSourceSupportQueryCount'] == 0
    valid[61, 71] = False
    assert appearance.validateEyelidSourceFootprints(queries, mask, valid, eyes, eyes)['invalidSourceSupportQueryCount'] == 1
    valid[:] = True
    raw = copy.deepcopy(eyes)
    raw['imageLeft']['upper'][1, 1] = 50.
    assert appearance.validateEyelidSourceFootprints(queries, mask, valid, eyes, raw)['invalidSourceSupportQueryCount'] == 1
    assert appearance.validateEyelidSourceFootprints(queries, mask, valid, raw, eyes)['invalidSourceSupportQueryCount'] == 1


def testSourceFailureRaisesRatherThanReturningPartialEdit(monkeypatch):
    args = sceneInputs()
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: measured(kwargs['photoEyes']))
    monkeypatch.setattr(appearance, 'validateEyelidSourceFootprints', lambda *a, **k: {'invalidSourceSupportQueryCount': 1})
    with pytest.raises(ValueError, match='four-contributor'):
        appearance.refineClientEyelidAppearance(*args)


@pytest.mark.parametrize('policy', [{'filterFootprint': True}, {'pointsPerLid': True},
    {'destinationBoxFeatherWorkingPixels': 0}, {'coreWorkingPixels': 30}, {'unknown': 2},
    {'sourceSafetyWorkingPixels': float('nan')}, {'accessoryUvPaddingPixels': 0}])
def testInvalidPolicyRejected(policy):
    with pytest.raises(ValueError):
        appearance.refineClientEyelidAppearance(*sceneInputs(), policy=policy)


@pytest.mark.parametrize('change', ['geometry', 'uv', 'order', 'material', 'mask'])
def testMismatchedSceneOrSourceRejected(change):
    args = list(sceneInputs())
    if change == 'geometry': args[1]['vertices'][0, 0] += .01
    elif change == 'uv': args[1]['uvs'][0, 0] += .01
    elif change == 'order': args[1]['faces'][0] = args[1]['faces'][0][::-1]
    elif change == 'material': args[1]['materialIds'][0] = 2
    elif change == 'mask': args[4] = args[4].astype(np.float32)
    with pytest.raises(ValueError):
        appearance.refineClientEyelidAppearance(*args)


def testZeroSourceSupportLeavesBaselineExactly(monkeypatch):
    args = list(sceneInputs());args[4][:] = False
    monkeypatch.setattr(appearance, 'extractVisibleEyeContours', lambda **kwargs: measured(kwargs['photoEyes']))
    result = appearance.refineClientEyelidAppearance(*args)
    np.testing.assert_array_equal(result['linearRGB'], args[5])
    assert result['diagnostics']['status'] == 'noSafeVisibleColorChange'


def testMissingUvConventionCannotSilentlyUsePixelCenters():
    args = list(sceneInputs())
    args[0].pop('uvConvention')
    with pytest.raises(ValueError, match='explicit topLeftPixelEdges'):
        appearance.refineClientEyelidAppearance(*args)


def testInvalidSourceCoordinatesAndBounds():
    eyes = sceneInputs()[-1]
    mask, valid = np.ones((1, 1), bool), np.ones((256, 256), bool)
    with pytest.raises(ValueError, match='nonfinite'):
        appearance.validateEyelidSourceFootprints(np.full((1, 1, 2), np.nan), mask, valid, eyes, eyes)
    checks = appearance.validateEyelidSourceFootprints(np.array([[[255., 255.]]]), mask, valid, eyes, eyes)
    assert checks['invalidSourceSupportQueryCount'] == 1
