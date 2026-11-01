import numpy as np
import pytest

from faceAppearance.noseTextureRefinement import refineNoseTexture, encodeNoseTextureEdit, evaluateNoseViewQuality


def scene():
    vertices = np.array([[-1., -1., 2.], [-1., 1., 2.], [1., 1., 2.], [1., -1., 2.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    geometry = {'vertices': vertices, 'faces': faces, 'uvs': np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]),
                'uvFaces': faces, 'uvConvention': 'topLeftPixelEdges'}
    camera = {'rotation': np.eye(3), 'translation': np.zeros(3), 'intrinsics': np.array([[4., 0., 4.], [0., 4., 4.], [0., 0., 1.]]),
              'viewIndex': 3, 'valid': True}
    yy, xx = np.indices((9, 9))
    source = np.stack([xx/8., yy/8., np.full_like(xx, .3, dtype=float)], axis=-1)
    return dict(geometry=geometry, camera=camera, sourceLinear=source,
        sourceValidMask=np.ones((9, 9), bool), sourceCorrectionMask=np.ones((9, 9), bool),
        baselineLinear=np.full((16, 16, 3), .05), noseUvMask=np.ones((16, 16), bool),
        protectedUvMask=np.zeros((16, 16), bool), visibleScene={'vertices': vertices, 'faces': faces})


def testDirectPerspectiveSamplingUsesActualSurfaceAndPreservesProtectedPixels():
    args = scene(); args['protectedUvMask'][7:9, 7:9] = True
    result = refineNoseTexture(**args, featherTexels=2)
    # UV pixel (x=4, y=5) is world [-.4375, -.3125, 2], photo [3.125, 3.375].
    np.testing.assert_allclose(result['sourceXY'][5, 4], [3.125, 3.375], atol=1e-12)
    np.testing.assert_allclose(result['linearRGB'][5, 4], [3.125/8, 3.375/8, .3], atol=1e-12)
    np.testing.assert_array_equal(result['linearRGB'][7:9, 7:9], args['baselineLinear'][7:9, 7:9])
    assert not result['directReprojectionSupportMask'][7:9, 7:9].any()
    assert (result['sourceIndex'][result['alpha'] > 0] == 3).all()
    assert np.isnan(result['sourceXY'][7:9, 7:9]).all()
    assert 'coverageMask' not in result


def testTiltedSurfaceProjectsAfterBarycentricInterpolation():
    args = scene()
    args['geometry']['vertices'][:, 2] = [2., 2., 4., 4.]
    result = refineNoseTexture(**args, featherTexels=2)
    # The plane is z=3+x, so interpolating projected corner positions would
    # be wrong. Evaluate the perspective ray at the known world-space point.
    point = np.array([-.4375, -.3125, 2.5625])
    expected = 4 + 4*point[:2]/point[2]
    np.testing.assert_allclose(result['sourceXY'][5, 4], expected, atol=1e-12)
    np.testing.assert_allclose(result['linearRGB'][5, 4], [expected[0]/8, expected[1]/8, .3], atol=1e-12)


@pytest.mark.parametrize('maskName,reasonCode', [('sourceValidMask', 7), ('sourceCorrectionMask', 8)])
def testEveryBilinearContributorMustBeValidAndCorrected(maskName, reasonCode):
    args = scene(); args[maskName][4, 4] = False
    result = refineNoseTexture(**args)
    assert not result['directReprojectionSupportMask'][8, 8]
    assert result['reasonCode'][8, 8] == reasonCode
    np.testing.assert_array_equal(result['linearRGB'][8, 8], args['baselineLinear'][8, 8])
    assert result['directReprojectionSupportMask'][1, 1]


def testHiddenSurfaceCannotSampleThroughAnOpaqueAccessory():
    args = scene()
    occluder = args['geometry']['vertices'].copy(); occluder[:, 2] = 1.
    args['visibleScene'] = {'vertices': np.vstack([args['geometry']['vertices'], occluder]),
        'faces': np.vstack([args['geometry']['faces'], args['geometry']['faces']+4])}
    result = refineNoseTexture(**args)
    assert not result['directReprojectionSupportMask'].any()
    np.testing.assert_array_equal(result['linearRGB'], args['baselineLinear'])
    assert (result['reasonCode'] == 6).all()


def testConflictingUvIslandsAreRejectedInsteadOfPaintingBothSurfaces():
    args = scene(); geometry = args['geometry']
    far = geometry['vertices'].copy(); far[:, 2] += 1
    geometry['vertices'] = np.vstack([geometry['vertices'], far])
    geometry['faces'] = np.vstack([geometry['faces'], geometry['faces']+4])
    geometry['uvFaces'] = np.vstack([geometry['uvFaces'], geometry['uvFaces']])
    args['visibleScene'] = {'vertices': geometry['vertices'], 'faces': geometry['faces']}
    result = refineNoseTexture(**args)
    assert result['uvAliasMask'].all()
    assert not result['directReprojectionSupportMask'].any()
    np.testing.assert_array_equal(result['linearRGB'], args['baselineLinear'])


def testFeatherReducesContributionAtDomainEdgesWithoutChangingSourceQuery():
    args = scene(); args['noseUvMask'][:2] = False
    result = refineNoseTexture(**args, featherTexels=4)
    assert 0 < result['alpha'][2, 8] < result['alpha'][3, 8] < result['alpha'][6, 8] == 1.
    np.testing.assert_array_equal(result['linearRGB'][:2], args['baselineLinear'][:2])
    np.testing.assert_array_equal(result['sourceXY'][result['alpha'] > 0], result['projectedXY'][result['alpha'] > 0])


def testEncodingPreservesAlphaAndUnmodifiedPixelsExactly():
    original = np.arange(4*5*4, dtype=np.uint8).reshape(4, 5, 4)
    colors = np.full((4, 5, 3), .5)
    alpha = np.zeros((4, 5)); alpha[1, 2] = .7
    protected = np.zeros((4, 5), bool); protected[2:] = True
    pixels = encodeNoseTextureEdit(original, colors, alpha, protected)
    np.testing.assert_array_equal(pixels[1, 2, :3], [188, 188, 188])
    np.testing.assert_array_equal(pixels[..., 3], original[..., 3])
    np.testing.assert_array_equal(pixels[alpha == 0], original[alpha == 0])
    alpha[2, 2] = .1
    with pytest.raises(ValueError, match='protected'):
        encodeNoseTextureEdit(original, colors, alpha, protected)


def testIncidenceQualityPrefersActuallyAvailableBetterView():
    args = scene()
    front = dict(args['camera'], translation=np.array([-3., 0., 0.]),
        intrinsics=np.array([[1., 0., 4.], [0., 1., 4.], [0., 0., 1.]]))
    better = dict(front, viewIndex=0, translation=np.zeros(3))
    valid = np.ones((9, 9), bool)
    kwargs = dict(referenceViewIndex=3, visibleScene=args['visibleScene'])
    quality = evaluateNoseViewQuality(args['geometry'], [front, better], [valid, valid], args['noseUvMask'], **kwargs)
    assert quality['viewAvailableMask'][:, 8, 8].all()
    assert quality['bestViewIndex'][8, 8] == 0
    assert quality['facingCosines'][0, 8, 8] > .5
    assert quality['relativeFacing'][8, 8] < .85
    assert quality['sourceQuality'][8, 8] == 0
    # A missing photograph cannot act as a superior available observation.
    missing = evaluateNoseViewQuality(args['geometry'], [front, better], [valid, ~valid], args['noseUvMask'], **kwargs)
    assert missing['bestViewIndex'][8, 8] == 3
    assert missing['relativeFacing'][8, 8] == 1
    assert 0 < missing['sourceQuality'][8, 8] < 1
    # Reference absence forbids any contribution even if another view exists.
    noFront = evaluateNoseViewQuality(args['geometry'], [front, better], [~valid, valid], args['noseUvMask'], **kwargs)
    assert not noFront['sourceQuality'].any()


def testSurfaceQualityOnlyReducesDirectContributionAndPreservesRejectedPixels():
    args = scene()
    quality = np.ones((16, 16)); quality[:8] = 0; quality[8:] = .25
    ordinary = refineNoseTexture(**args, featherTexels=2)
    result = refineNoseTexture(**args, featherTexels=2, sourceQuality=quality)
    np.testing.assert_array_equal(result['linearRGB'][:8], args['baselineLinear'][:8])
    np.testing.assert_allclose(result['alpha'][10:], ordinary['alpha'][10:]*.25)
    assert (result['alpha'][8:] <= ordinary['alpha'][8:]*.25).all()
    assert result['alpha'][8, 8] < result['alpha'][9, 8]
    np.testing.assert_array_equal(result['sourceXY'][8:], ordinary['sourceXY'][8:])
    assert (result['reasonCode'][:8] == 10).all()


@pytest.mark.parametrize('holeKind', ['quality', 'photoValidity'])
def testFinalAcceptedSupportFeathersInternalRejectedHolesWithoutExpandingSupport(holeKind):
    args = scene()
    if holeKind == 'quality':
        quality = np.ones((16, 16)); quality[7:9, 7:9] = 0
        args['sourceQuality'] = quality
    else:
        args['sourceValidMask'][4, 4] = False
    # Feathering changes weights, never which source queries are supported.
    unfeathered = refineNoseTexture(**args, featherTexels=1)
    result = refineNoseTexture(**args, featherTexels=3)
    support = result['directReprojectionSupportMask']
    np.testing.assert_array_equal(support, unfeathered['directReprojectionSupportMask'])
    np.testing.assert_array_equal(result['linearRGB'][~support], args['baselineLinear'][~support])
    np.testing.assert_array_equal(result['sourceXY'][support], unfeathered['sourceXY'][support])
    from scipy.ndimage import binary_dilation
    # Select accepted pixels adjacent to an INTERNAL rejected area, not the
    # requested nose ROI edge; these were previously given full contribution.
    adjacent = binary_dilation(~support) & support
    assert adjacent.any()
    assert (result['acceptedSupportBoundaryDistance'][adjacent] == 1).all()
    assert (result['alpha'][adjacent] < unfeathered['alpha'][adjacent]).all()
    assert (result['alpha'][adjacent] < .27).all()


@pytest.mark.parametrize('badValue', [np.nan, np.inf, -.1, 1.1])
def testRejectsInvalidQualityWeightsAndThresholds(badValue):
    args = scene(); quality = np.ones((16, 16)); quality[8, 8] = badValue
    with pytest.raises(ValueError, match='sourceQuality'):
        refineNoseTexture(**args, sourceQuality=quality)
    with pytest.raises(ValueError, match='thresholds'):
        evaluateNoseViewQuality(args['geometry'], [args['camera']], [args['sourceValidMask']],
            args['noseUvMask'], referenceViewIndex=3, visibleScene=args['visibleScene'],
            minimumFacingCosine=badValue)


@pytest.mark.parametrize('damage', ['nonfinite', 'camera', 'softMask', 'emptyScene'])
def testRejectsInvalidInputs(damage):
    args = scene()
    if damage == 'nonfinite': args['sourceLinear'][0, 0, 0] = np.nan
    elif damage == 'camera': args['camera']['rotation'][0, 0] = -1.
    elif damage == 'softMask': args['sourceCorrectionMask'] = args['sourceCorrectionMask'].astype(float)
    else: args['visibleScene'] = None
    with pytest.raises(ValueError): refineNoseTexture(**args)
