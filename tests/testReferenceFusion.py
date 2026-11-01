import numpy as np
import pytest

from faceAppearance.referenceFusion import fuseReferenceObservations


def makeObservations(count=3, size=17):
    values = np.full((count, size, size, 3), 0.2, np.float32)
    weights = np.ones((count, size, size), np.float32)
    cosines = np.ones_like(weights)
    return values, weights, cosines


def testTrustedReferenceIsExactDespiteStrongerDisagreeingAuxiliaries():
    values, weights, cosines = makeObservations()
    values[0, 5:12, 6:8] = [0.11, 0.09, 0.07]
    values[1] = 0.8
    values[2] = 0.95
    weights[0] = 0.001
    weights[1:] = 100
    originals = [array.copy() for array in (values, weights, cosines)]
    result = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=3)
    core = result["referenceCoreMask"]
    assert core.any()
    np.testing.assert_array_equal(result["linearAlbedo"][core], values[0][core])
    assert (result["referenceAlpha"][core] == 1).all()
    assert (result["sourceIndex"][core] == 0).all()
    for original, current in zip(originals, (values, weights, cosines)):
        np.testing.assert_array_equal(original, current)


def testInvalidReferenceUsesBestAuxiliaryInsteadOfColorAverage():
    values, weights, cosines = makeObservations()
    values[1] = 0.4
    values[2] = 0.8
    weights[0] = 0
    weights[1] = 0.1
    weights[2] = 0.9
    weights[2, 4, 4] = 0
    result = fuseReferenceObservations(values, weights, cosines, matchAuxiliaryColors=False)
    assert result["linearAlbedo"][3, 3, 0] == values[2, 3, 3, 0]
    assert result["linearAlbedo"][4, 4, 0] == values[1, 4, 4, 0]
    assert result["sourceIndex"][3, 3] == 2
    assert result["auxiliarySourceIndex"][4, 4] == 1
    assert (result["referenceAlpha"] == 0).all()
    assert not result["referenceCoreMask"].any()


def testSoleReferencePreservesEdgesWithoutInventingSupport():
    values, weights, cosines = makeObservations(count=1)
    cosines[:] = 0.2
    weights[:, 0, 0] = 0
    result = fuseReferenceObservations(values, weights, cosines)
    valid = weights[0] > 0
    np.testing.assert_array_equal(result["linearAlbedo"][valid], values[0][valid])
    assert (result["referenceAlpha"][valid] == 1).all()
    assert not result["referenceCoreMask"].any()
    assert (result["auxiliarySourceIndex"] == -1).all()
    assert not result["coverageMask"][0, 0]
    assert result["sourceIndex"][0, 0] == -1
    assert result["confidence"][0, 0] == 0
    assert not result["linearAlbedo"][0, 0].any()


def testFeatherIsMonotonicFromActualSupportBoundary():
    values, weights, cosines = makeObservations(count=2)
    values[0] = 0.8
    values[1] = 0.2
    weights[0, :, :3] = 0
    result = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=3,
                                       matchAuxiliaryColors=False)
    alpha = result["referenceAlpha"][8]
    np.testing.assert_allclose(alpha[2:7], [0, 0, 7 / 27, 20 / 27, 1], atol=1e-6)
    assert result["sourceIndex"][8, 4] == -2
    assert result["sourceIndex"][8, 3] == 1
    assert result["sourceIndex"][8, 6] == 0
    np.testing.assert_allclose(result["linearAlbedo"][8, 4], 0.8 * (7 / 27) + 0.2 * (20 / 27), atol=1e-6)


def testAngleTrustDoesNotDependOnDisagreeingAuxiliaryWeights():
    values, weights, cosines = makeObservations(count=2)
    values[1] = 0.8
    cosines[0, 8, 6:9] = [0.35, 0.5, 0.65]
    result = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=0,
                                       matchAuxiliaryColors=False)
    np.testing.assert_allclose(result["referenceAlpha"][8, 6:9], [0, 0.5, 1], atol=1e-6)
    assert not result["referenceCoreMask"][8, 7]
    assert result["referenceCoreMask"][8, 8]
    weights[1] *= 100
    stronger = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=0,
                                         matchAuxiliaryColors=False)
    np.testing.assert_array_equal(stronger["referenceAlpha"], result["referenceAlpha"])


def testRgbGainUsesRobustOverlapAndNeverChangesReference():
    values, weights, cosines = makeObservations(count=2)
    values[0] = [0.3, 0.2, 0.4]
    values[1] = [0.2, 0.25, 0.2]
    # A minority of mismatched brow pixels must not dominate calibration.
    values[1, :2] = 0.8
    weights[0, 8, 8] = 0
    result = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=0)
    gains = result["diagnostics"]["colorMatching"]["views"][1]["gainsRgb"]
    np.testing.assert_allclose(gains, [1.5, 0.8, 1.5], atol=1e-6)
    np.testing.assert_allclose(result["linearAlbedo"][8, 8], [0.3, 0.2, 0.3], atol=1e-6)
    np.testing.assert_array_equal(result["linearAlbedo"][7, 7], values[0, 7, 7])
    assert result["diagnostics"]["colorMatching"]["views"][0]["gainsRgb"] == [1, 1, 1]


def testInsufficientOrPoorAngleOverlapDoesNotCalibrateGains():
    values, weights, cosines = makeObservations(count=2, size=5)
    values[1] = 0.1
    insufficient = fuseReferenceObservations(values, weights, cosines)
    assert insufficient["diagnostics"]["colorMatching"]["views"][1]["gainsRgb"] == [1, 1, 1]
    values, weights, cosines = makeObservations(count=2)
    values[1] = 0.1
    cosines[1] = 0.4
    angle = fuseReferenceObservations(values, weights, cosines)
    assert angle["diagnostics"]["colorMatching"]["views"][1]["overlapSamplesRgb"] == [0, 0, 0]


def testConfidenceTracksActualSourcesAndMissingSupport():
    values, weights, cosines = makeObservations(count=2)
    weights[0] = 0.2
    weights[1] = 0.8
    weights[:, 5, 5] = 0
    cosines[0, 8, 8] = 0.5
    result = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=0,
                                       matchAuxiliaryColors=False)
    assert result["confidence"][8, 7] == np.float32(0.2)
    assert result["confidence"][0, 0] == np.float32(0.2)
    np.testing.assert_allclose(result["confidence"][8, 8], 0.5, atol=1e-6)
    assert result["confidence"][5, 5] == 0
    assert result["auxiliarySourceIndex"][5, 5] == -1
    assert result["sourceIndex"][5, 5] == -1


def testNonzeroReferenceIndexAndDeterministicAuxiliaryTies():
    values, weights, cosines = makeObservations()
    values[0] = 0.3
    values[1] = 0.6
    values[2] = 0.9
    weights[2, 8, 8] = 0
    result = fuseReferenceObservations(values, weights, cosines, referenceViewIndex=2,
                                       referenceBlendWidth=0, matchAuxiliaryColors=False)
    assert result["sourceIndex"][7, 7] == 2
    assert result["sourceIndex"][8, 8] == 0
    np.testing.assert_array_equal(result["linearAlbedo"][8, 8], values[0, 8, 8])


def testEntirelyMissingSupportStaysEmpty():
    values, weights, cosines = makeObservations()
    weights[:] = 0
    result = fuseReferenceObservations(values, weights, cosines)
    assert not result["linearAlbedo"].any()
    assert not result["referenceAlpha"].any()
    assert not result["coverageMask"].any()
    assert not result["confidence"].any()
    assert (result["sourceIndex"] == -1).all()
    assert (result["auxiliarySourceIndex"] == -1).all()


def testDarkAndSaturatedChannelsDoNotBiasAuxiliaryCalibration():
    values, weights, cosines = makeObservations(count=2)
    values[0] = [0.3, 0.2, 0.1]
    values[1] = [0.25, 0.25, 0.1]
    values[1, :8, :, 0] = 1
    values[1, :8, :, 1] = 0
    result = fuseReferenceObservations(values, weights, cosines)
    record = result["diagnostics"]["colorMatching"]["views"][1]
    np.testing.assert_allclose(record["gainsRgb"], [1.2, 0.8, 1], atol=1e-6)
    assert record["overlapSamplesRgb"] == [9 * 17, 9 * 17, 17 * 17]


@pytest.mark.parametrize("parameter,value", [("referenceViewIndex", -1), ("referenceViewIndex", 3),
                                             ("referenceViewIndex", True), ("referenceBlendWidth", -1),
                                             ("referenceBlendWidth", np.nan), ("referenceMinCosine", 0.65),
                                             ("referenceFullCosine", 1.1), ("matchAuxiliaryColors", 1)])
def testInvalidParametersFail(parameter, value):
    values, weights, cosines = makeObservations()
    with pytest.raises(ValueError):
        fuseReferenceObservations(values, weights, cosines, **{parameter: value})


@pytest.mark.parametrize("arrayIndex,invalid", [(0, -0.1), (0, 1.1), (0, np.nan),
                                               (1, -1), (1, np.inf), (2, -0.1), (2, 1.1)])
def testInvalidObservationArraysFail(arrayIndex, invalid):
    arrays = list(makeObservations())
    arrays[arrayIndex].flat[0] = invalid
    with pytest.raises(ValueError):
        fuseReferenceObservations(*arrays)


def testShapeMismatchAndEmptyViewsFail():
    values, weights, cosines = makeObservations()
    with pytest.raises(ValueError):
        fuseReferenceObservations(values, weights[..., :4], cosines)
    with pytest.raises(ValueError):
        fuseReferenceObservations(values[:0], weights[:0], cosines[:0])
