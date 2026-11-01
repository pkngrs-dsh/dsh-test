import numpy as np
import pytest

from faceAppearance.clientShDelight import applyPhotoShLighting, renderClientPhotoNormals


def constantLight(value):
    coefficients = np.zeros((9, 3))
    coefficients[0] = np.asarray(value) / .28209479177387814
    return coefficients


def testKnownLightingRecoversLinearReflectanceWithoutColorClipping():
    albedo = np.array([[[.12, .18, .24], [.2, .15, .1]]], np.float32)
    lighting = np.array([2., 1., .5])
    raw = (albedo * lighting).astype(np.float32)
    normals = np.broadcast_to([0., 0., -1.], raw.shape)
    result = applyPhotoShLighting(raw, normals, np.ones(raw.shape[:2], bool), constantLight(lighting))
    np.testing.assert_allclose(result['linearRGB'], albedo, atol=1e-7)
    assert result['correctionMask'].all()


def testUnsafeOrUnsupportedCorrectionRetainsExactInputAndMarksFallback():
    raw = np.array([[[.9, .8, .7], [.2, .2, .2], [.3, .4, .5]]], np.float32)
    normals = np.broadcast_to([0., 0., 1.], raw.shape)
    valid = np.array([[True, True, False]])
    result = applyPhotoShLighting(raw, normals, valid, constantLight(.5))
    assert result['correctionMask'].tolist() == [[False, True, False]]
    np.testing.assert_array_equal(result['linearRGB'][~result['correctionMask']], raw[~result['correctionMask']])
    np.testing.assert_allclose(result['linearRGB'][0, 1], [.4, .4, .4])
    for value in [-1., .1, 5.]:
        unsafe = applyPhotoShLighting(raw, normals, valid, constantLight(value))
        assert not unsafe['correctionMask'].any()
        np.testing.assert_array_equal(unsafe['linearRGB'], raw)
    failed = applyPhotoShLighting(raw, normals, valid, constantLight(.5), fitted=False)
    assert not failed['correctionMask'].any()
    np.testing.assert_array_equal(failed['linearRGB'], raw)


def testPhotoNormalsUseDepthOcclusionAndObjectFrame():
    # Two overlapping front-facing squares, nearer surface must win.
    vertices = np.array([[-1,-1,2],[-1,1,2],[1,1,2],[1,-1,2]], np.float64)
    far = np.array([[-2,-2,3.5],[-2,2,3.5],[2,2,4.5],[2,-2,4.5]], np.float64)
    geometry = {'vertices': np.vstack([vertices, far]),
                'faces': np.array([[4,5,6],[4,6,7],[0,1,2],[0,2,3]])}
    camera = {'rotation': np.eye(3), 'translation': np.zeros(3),
              'intrinsics': np.array([[4,0,4],[0,4,4],[0,0,1]])}
    normals, valid = renderClientPhotoNormals(geometry, camera, (9,9))
    assert valid[4,4]
    np.testing.assert_allclose(normals[4,4], [0,0,-1], atol=1e-7)
    assert not valid[0,0]


def testNonFiniteLightingFailsInsteadOfProducingInvisibleNaNs():
    raw = np.ones((2,2,3), np.float32) * .2
    coefficients = constantLight(1.)
    coefficients[1,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        applyPhotoShLighting(raw, raw, np.ones((2,2),bool), coefficients)


def testPhotoFallbackCountsExcludeBackgroundWithoutChangingCorrectedPixels():
    raw = np.full((2, 4, 3), .9, np.float32)
    raw[0, :2] = .2
    normals = np.broadcast_to([0., 0., -1.], raw.shape)
    normalMask = np.ones((2, 4), bool)
    normalMask[0, 1] = normalMask[1, 0] = False
    photoMask = np.array([[True, True, True, False], [False, False, False, False]])
    kwargs = dict(rawLinear=raw, normals=normals, normalMask=normalMask,
                  coefficients=constantLight(.5))
    allPixels = applyPhotoShLighting(**kwargs)
    scoped = applyPhotoShLighting(**kwargs, photoMask=photoMask)
    np.testing.assert_array_equal(scoped['linearRGB'], allPixels['linearRGB'])
    np.testing.assert_array_equal(scoped['correctionMask'], allPixels['correctionMask'])
    report = scoped['diagnostics']
    assert report['reasonCountsDomain'] == 'validPhotoMask'
    assert report['fullImage']['pixels'] == 8
    assert report['fullImage']['outOfRangePixels'] == 5
    assert report['validPhoto']['pixels'] == 3
    assert report['validPhoto']['correctedPixels'] == 1
    assert report['validPhoto']['rawFallbackPixels'] == 2
    assert report['validPhoto']['rawFallbackFraction'] == pytest.approx(2 / 3)
    assert report['missingNormalPixels'] == report['outOfRangePixels'] == 1
    assert sum(report['validPhoto'][name] for name in (
        'missingNormalPixels', 'unsafeLightingPixels', 'outOfRangePixels', 'unfittedPixels')) == 2
    failed = applyPhotoShLighting(**kwargs, photoMask=photoMask, fitted=False)
    assert failed['diagnostics']['validPhoto']['rawFallbackFraction'] == 1
    assert failed['diagnostics']['validPhoto']['unfittedPixels'] == 1
    np.testing.assert_array_equal(failed['linearRGB'], raw)


def testClientApplicationSummaryUsesPixelWeightedValidPhotoFallback(monkeypatch):
    import faceAppearance.clientShDelight as module

    raw = np.full((2, 2, 3, 3), .2, np.float32)
    raw[0, 0, 0] = .9
    raw[1, 1, 2] = .9  # Outside this view's valid photo support.
    masks = np.array([[[True, True, True], [True, True, True]],
                      [[True, False, False], [False, False, False]]])
    normals = np.broadcast_to([0., 0., -1.], raw.shape[1:])
    monkeypatch.setattr(module, 'collectClientPhotoObservations', lambda *args: {
        'raw': raw, 'openDelight': raw, 'weights': masks, 'normals': normals})
    monkeypatch.setattr(module, 'solveShDelight', lambda *args, **kwargs: {
        'coefficients': np.asarray([constantLight(.5)] * 2),
        'diagnostics': {'views': [{'status': 'fitted'}, {'status': 'fitted'}]}})
    monkeypatch.setattr(module, 'renderClientPhotoNormals', lambda *args: (
        normals, np.ones(masks.shape[1:], bool)))
    result = module.delightClientPhotos({}, [{}, {}], raw, masks, referenceViewIndex=0, fitResolution=32)
    summary = result['diagnostics']['applicationSummary']
    assert summary['validPhotoPixels'] == 7
    assert summary['fallbackValidPhotoPixels'] == 1
    assert summary['correctedValidPhotoPixels'] == 6
    assert summary['rawFallbackFraction'] == pytest.approx(1 / 7)
    assert result['diagnostics']['application'][1]['outOfRangePixels'] == 0
    assert result['diagnostics']['application'][1]['fullImage']['outOfRangePixels'] == 1
    np.testing.assert_array_equal(result['validMasks'], masks)
    np.testing.assert_array_equal(result['linearImages'][0, 0, 0], raw[0, 0, 0])
