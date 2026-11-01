import copy

import numpy as np
import pytest

from faceAppearance.referenceFeaturePriority import applyReferenceFeaturePriority


def inputs(shape=(21, 25)):
    y, x = np.indices(shape)
    reference = np.empty((*shape, 3), np.float32)
    reference[..., 0], reference[..., 1], reference[..., 2] = .7, .4, .2
    alpha = np.full(shape, .25, np.float32)
    auxiliary = np.full_like(reference, .1)
    colors = alpha[..., None]*reference+(1-alpha[..., None])*auxiliary
    fusion = {'linearAlbedo': colors, 'coverageMask': np.ones(shape, bool),
              'referenceAlpha': alpha, 'sourceIndex': np.full(shape, -2, np.int32),
              'confidence': np.full(shape, .4, np.float32), 'referenceCoreMask': np.zeros(shape, bool),
              'protectionMask': np.zeros(shape, bool), 'albedo': colors**(1/2.2),
              'blendAllowedMask': np.zeros(shape, bool), 'auxiliarySourceIndex': np.ones(shape, np.int32)}
    pixels = np.stack([x, y], -1).astype(np.float32)
    return fusion, reference, np.full(shape, .8, np.float32), pixels


def apply(values, curves=None, **options):
    return applyReferenceFeaturePriority(*values, featureCurves=[np.array([[4, 10], [20, 10.]])] if curves is None else curves,
                                         **options)


def testCoreCopiesExistingReferenceExactlyAndKeepsCoverage():
    values = inputs()
    result = apply(values)
    core = result['priorityCoreMask']
    assert core.any()
    np.testing.assert_array_equal(result['linearAlbedo'][core], values[1][core])
    np.testing.assert_array_equal(result['referenceAlpha'][core], 1.)
    np.testing.assert_array_equal(result['sourceIndex'][core], 0)
    np.testing.assert_array_equal(result['auxiliarySourceIndex'][core], -1)
    np.testing.assert_array_equal(result['coverageMask'], values[0]['coverageMask'])
    assert not result['referenceCoreMask'].any()
    assert result['protectionMask'][core].all()


def testCosineFalloffUsesActualComposedAlphaAndConfidence():
    values = inputs()
    result = apply(values, coreRadiusPixels=2., falloffRadiusPixels=8.)
    # Five pixels from the line is the midpoint of the falloff: w=.5.
    pixel = (5, 12)
    assert result['priorityWeight'][pixel] == pytest.approx(.5)
    assert result['referenceAlpha'][pixel] == pytest.approx(1-(1-.25)*(1-.5))
    np.testing.assert_allclose(result['linearAlbedo'][pixel], .5*values[0]['linearAlbedo'][pixel]+.5*values[1][pixel])
    assert result['confidence'][pixel] == pytest.approx(.6)
    assert result['sourceIndex'][pixel] == -2
    assert result['blendAllowedMask'][pixel]


def testOutsideFeatureAndInvalidPixelsAreBitIdenticalIncludingCachedAlbedo():
    values = inputs()
    result = apply(values, coreRadiusPixels=1., falloffRadiusPixels=3.)
    outside = ~result['priorityMask']
    for field in ('linearAlbedo', 'referenceAlpha', 'sourceIndex', 'confidence', 'albedo'):
        assert result[field][outside].tobytes() == values[0][field][outside].tobytes()
    changed = result['priorityMask']
    np.testing.assert_allclose(result['albedo'][changed], result['linearAlbedo'][changed]**(1/2.2), atol=1e-7)


def testUnobservedReferenceAndUncoveredGapsAreNeverFilled():
    fusion, reference, weights, pixels = inputs()
    weights[10, 9:12] = 0
    pixels[10, 9:12] = -1
    fusion['coverageMask'][10, 13:16] = False
    fusion['sourceIndex'][10, 13:16] = -1
    fusion['linearAlbedo'][10, 13:16] = 0
    fusion['referenceAlpha'][10, 13:16] = 0
    before = copy.deepcopy(fusion)
    result = apply((fusion, reference, weights, pixels))
    forbidden = (weights == 0) | ~fusion['coverageMask']
    assert not result['priorityMask'][forbidden].any()
    assert not result['priorityCoreMask'][forbidden].any()
    np.testing.assert_array_equal(result['linearAlbedo'][forbidden], before['linearAlbedo'][forbidden])
    np.testing.assert_array_equal(result['coverageMask'], before['coverageMask'])


def testOriginalTrustedAndOwnedReferenceRemainBitExact():
    fusion, reference, weights, pixels = inputs()
    fusion['referenceCoreMask'][8:13, 8:13] = True
    mask = fusion['referenceCoreMask']
    fusion['linearAlbedo'][mask] = reference[mask]
    fusion['referenceAlpha'][mask] = 1
    fusion['sourceIndex'][mask] = 0
    before = copy.deepcopy(fusion)
    result = apply((fusion, reference, weights, pixels))
    for field in ('linearAlbedo', 'referenceAlpha', 'confidence', 'sourceIndex', 'referenceCoreMask', 'albedo'):
        assert result[field][mask].tobytes() == before[field][mask].tobytes()
    assert result['diagnostics']['referenceFeaturePriority']['originalTrustedCorePreserved']


def testSeparateCurvesDoNotCreateImaginaryConnectionAcrossGap():
    values = inputs((25, 40))
    curves = [np.array([[4., 10], [8, 10]]), np.array([[28., 10], [32, 10]])]
    result = apply(values, curves, coreRadiusPixels=1., falloffRadiusPixels=3.)
    assert result['priorityCoreMask'][10, 6] and result['priorityCoreMask'][10, 30]
    assert not result['priorityMask'][10, 18]


def testLongSparseSegmentsAreMeasuredContinuouslyNotByEndpointDistance():
    values = inputs((25, 40))
    result = apply(values, [np.array([[2., 10], [36, 10]])], coreRadiusPixels=1., falloffRadiusPixels=3.)
    assert result['priorityCoreMask'][10, 19]
    assert result['priorityMask'][12, 19]
    assert not result['priorityMask'][13, 19]


def testDistancesUsePhotoCoordinatesRatherThanUvPixelCoordinates():
    fusion, reference, weights, pixels = inputs()
    pixels = pixels*2+np.array([100., 200.])
    result = apply((fusion, reference, weights, pixels), [np.array([[108., 220], [140, 220]])],
                   coreRadiusPixels=2., falloffRadiusPixels=4.)
    assert result['priorityCoreMask'][9, 12]
    assert not result['priorityMask'][8, 12]


def testNoCurvesAreANoopAndInputsAreNotMutated():
    values = inputs()
    before = copy.deepcopy(values)
    result = apply(values, [])
    for field, array in values[0].items():
        np.testing.assert_array_equal(array, before[0][field])
        np.testing.assert_array_equal(result[field], array)
    assert not result['priorityMask'].any()
    apply(values)
    for field, array in values[0].items():
        np.testing.assert_array_equal(array, before[0][field])


def testAlphaAndConfidenceStayBoundedAndSourceIsKnown():
    values = inputs()
    result = apply(values)
    for field in ('referenceAlpha', 'confidence', 'priorityWeight'):
        assert np.isfinite(result[field]).all()
        assert result[field].min() >= 0 and result[field].max() <= 1
    assert set(np.unique(result['sourceIndex'])) <= {-2, 0}
    assert np.all(result['referenceAlpha'] >= values[0]['referenceAlpha'])


def testOwnershipDiagnosticsAreRecomputedAndPartitionCoverageAfterPriority():
    fusion, reference, weights, pixels = inputs()
    fusion['referenceAlpha'][:] = 0
    fusion['sourceIndex'][:] = 1
    fusion['diagnostics'] = {'auxiliaryOnlyTexels': int(fusion['coverageMask'].sum())}
    result = apply((fusion, reference, weights, pixels))
    alpha, coverage, diagnostics = result['referenceAlpha'], result['coverageMask'], result['diagnostics']
    expectedAuxiliary = int((coverage & (alpha == 0)).sum())
    assert 0 < expectedAuxiliary < fusion['diagnostics']['auxiliaryOnlyTexels']
    assert diagnostics['auxiliaryOnlyTexels'] == expectedAuxiliary
    assert diagnostics['referenceOwnedTexels'] == int((coverage & (alpha == 1)).sum())
    assert diagnostics['blendedTexels'] == int((coverage & (alpha > 0) & (alpha < 1)).sum())
    assert sum(diagnostics[key] for key in ('referenceOwnedTexels', 'blendedTexels', 'auxiliaryOnlyTexels')) == int(coverage.sum())


def testFloat32RoundingDoesNotTurnTransitionIntoFalsePureReferenceOwnership():
    fusion, reference, weights, pixels = inputs()
    fusion['referenceAlpha'][:] = np.nextafter(np.float32(1), np.float32(0))
    fusion['linearAlbedo'][:] = fusion['referenceAlpha'][..., None]*reference+(1-fusion['referenceAlpha'][..., None])*.1
    # Tiny positive distance beyond the core would round cosine weight to one
    # in float32 without an explicit distinction from the true geometric core.
    pixels[5, 12] = [12., 13.000001]
    result = apply((fusion, reference, weights, pixels))
    assert not result['priorityCoreMask'][5, 12]
    assert result['priorityWeight'][5, 12] < 1
    assert result['referenceAlpha'][5, 12] < 1
    assert result['sourceIndex'][5, 12] == -2
    pure = result['referenceAlpha'] == 1
    np.testing.assert_array_equal(result['linearAlbedo'][pure], reference[pure])


@pytest.mark.parametrize('invalid', ['negativeWeight', 'unknownSource', 'negativeRadius', 'equalRadii',
                                   'nanCurve', 'collapsedCurve', 'wrongReference', 'invalidQualifiedPixel'])
def testInvalidInputsFailClearly(invalid):
    values = inputs()
    kwargs, curves = {}, None
    if invalid == 'negativeWeight':
        values[2][0, 0] = -1
    elif invalid == 'unknownSource':
        values[0]['sourceIndex'][0, 0] = -3
    elif invalid == 'negativeRadius':
        kwargs['coreRadiusPixels'] = -1
    elif invalid == 'equalRadii':
        kwargs.update(coreRadiusPixels=3., falloffRadiusPixels=3.)
    elif invalid == 'nanCurve':
        curves = [np.array([[0., 0], [np.nan, 1]])]
    elif invalid == 'collapsedCurve':
        curves = [np.array([[1., 1], [1, 1]])]
    elif invalid == 'wrongReference':
        values[0]['referenceAlpha'][0, 0] = 1
        values[0]['sourceIndex'][0, 0] = 0
    else:
        values[3][0, 0] = -1
    with pytest.raises(ValueError):
        apply(values, curves, **kwargs)
