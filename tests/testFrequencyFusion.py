import numpy as np
import pytest

from faceAppearance.frequencyFusion import decomposeObservationFrequency, fuseFrequencyObservations
from faceAppearance.semanticFusion import fuseSemanticObservations


def makeObservations(count=2, height=19, width=23):
    values = np.full((count, height, width, 3), 0.3, np.float32)
    weights = np.ones((count, height, width), np.float32)
    cosines = np.ones_like(weights)
    cosines[0] = 0.5
    labels = np.ones(weights.shape, np.int16)
    confidence = np.ones_like(weights)
    return values, weights, cosines, labels, confidence


def fuse(*arrays, **kwargs):
    return fuseFrequencyObservations(*arrays, referenceBlendWidth=0, matchAuxiliaryColors=False,
                                      sourceIterations=0, **kwargs)


def testConstantLowFrequencyColorsKeepSoftTransitions():
    values, *arrays = makeObservations()
    values[0] = 0.2
    values[1] = 0.8
    result = fuse(values, *arrays, protectFeatureRegions=False)
    np.testing.assert_allclose(result["linearAlbedo"], 0.5, atol=2e-6)
    assert (result["highFrequencySourceIndex"] == 0).all()
    assert (result["sourceIndex"] == -2).all()
    assert result["diagnostics"]["highFrequencyMixedTexels"] == 0


@pytest.mark.parametrize("sigma", [0.0, 0.0001, 0.01, 0.1, 0.25, 0.5])
def testSmallSigmaUnitImpulseHasRequestedVarianceAndApproachesIdentity(sigma):
    values = np.zeros((1, 9, 9, 3), np.float32)
    values[0, 4, 4] = 1
    valid = np.ones(values.shape[:3], bool)
    result = decomposeObservationFrequency(values, valid, lowFrequencySigma=sigma)
    low = result["lowFrequency"][0, ..., 0]
    yy, xx = np.indices(low.shape)
    # One full step has per-axis variance .25. A fractional step must produce
    # sigma**2, rather than the same full-strength blur for every positive sigma.
    np.testing.assert_allclose((low * (xx - 4) ** 2).sum(), sigma ** 2, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose((low * (yy - 4) ** 2).sum(), sigma ** 2, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(low.sum(), 1, atol=1e-7)
    np.testing.assert_allclose(low[4, 4], 1 - 2 * sigma ** 2, atol=1e-7)
    assert np.count_nonzero(low) <= 5
    if sigma == 0:
        np.testing.assert_array_equal(result["lowFrequency"], values)


def testIntegerDiffusionSigmaThreeMatchesThirtySixExactFullSteps():
    rng = np.random.default_rng(3)
    values = rng.random((2, 13, 17, 3), dtype=np.float32)
    valid = np.ones(values.shape[:3], bool)
    valid[:, 4:8, 8] = False
    charts = np.zeros(values.shape[1:3], np.int32)
    charts[:, 10:] = 1
    direct = decomposeObservationFrequency(values, valid, lowFrequencySigma=3, uvChartLabels=charts)
    repeated = values
    for _ in range(36):
        repeated = decomposeObservationFrequency(repeated, valid, lowFrequencySigma=.5,
                                                 uvChartLabels=charts)["lowFrequency"]
    np.testing.assert_array_equal(direct["lowFrequency"], repeated)
    assert direct["diagnostics"]["fullDiffusionSteps"] == 36
    assert direct["diagnostics"]["finalStepWeight"] == 0


def testHighFrequencyDetailIsCopiedFromOneSourceWithoutAlphaAttenuation():
    values, weights, cosines, labels, confidence = makeObservations()
    yy, xx = np.indices(weights.shape[1:])
    values[0] = (0.3 + 0.15 * ((yy + xx) % 2))[..., None]
    values[1] = (0.45 - 0.15 * ((yy + xx) % 2))[..., None]
    decomposed = decomposeObservationFrequency(values, weights > 0, lowFrequencySigma=2)
    result = fuse(values, weights, cosines, labels, confidence,
                  lowFrequencySigma=2, protectFeatureRegions=False)
    np.testing.assert_array_equal(result["highFrequencyResidual"], decomposed["highFrequency"][0])
    expected = ((decomposed["lowFrequency"][0] + decomposed["lowFrequency"][1]) / 2
                + decomposed["highFrequency"][0])
    np.testing.assert_allclose(result["unclippedLinearAlbedo"], expected, atol=1e-7)
    assert np.std(result["linearAlbedo"][4:-4, 4:-4]) > 0.06


def testShiftedFeatureUnionUsesWholeReferenceAndRemovesAuxiliaryDuplicate():
    values, weights, cosines, labels, confidence = makeObservations(height=25, width=31)
    values[:] = 0.6
    values[0, 10:13, 9:20] = 0.1
    values[1, 13:16, 9:20] = 0.1
    labels[0, 10:13, 9:20] = 12
    labels[1, 13:16, 9:20] = 12
    cosines[0, 11, 14] = 0.8  # A reliable seed; remaining region has soft alpha.
    result = fuse(values, weights, cosines, labels, confidence, featureHaloWidth=2)
    region = result["featureOwnershipMask"]
    assert region[10:16, 9:20].all()
    assert (result["featureSourceIndex"][region] == 0).all()
    np.testing.assert_array_equal(result["linearAlbedo"][region], values[0][region])
    assert (result["linearAlbedo"][13:16, 9:20] == 0.6).all()
    assert result["diagnostics"]["featureOwnership"]["ownedRegionCount"] == 1


def testWholeFeatureCannotCrossOneMissingReferenceTexelOrOverwriteProtectedReference():
    values, weights, cosines, labels, confidence = makeObservations()
    labels[:, 6:12, 8:14] = 12
    cosines[0, 8, 10] = 0.9
    weights[0, 10, 12] = 0
    values[1] = 0.8
    result = fuse(values, weights, cosines, labels, confidence, featureHaloWidth=1)
    assert not result["featureOwnershipMask"].any()
    assert result["highFrequencySourceIndex"][10, 12] == 1
    assert result["highFrequencySourceIndex"][8, 10] == 0
    assert result["diagnostics"]["featureOwnership"]["unownedRegionCount"] == 1
    np.testing.assert_array_equal(result["linearAlbedo"][8, 10], values[0, 8, 10])


def testAbsentReferenceFeatureCanSelectOneEntireAuxiliaryRegion():
    values, weights, cosines, labels, confidence = makeObservations(count=3)
    weights[0] = 0
    labels[1:, 6:12, 8:14] = 2
    weights[1] = 0.8
    weights[2] = 0.5
    values[1] = 0.4
    values[2] = 0.7
    result = fuse(values, weights, cosines, labels, confidence, featureHaloWidth=1)
    owned = result["featureOwnershipMask"]
    assert owned.any()
    assert (result["featureSourceIndex"][owned] == 1).all()
    np.testing.assert_array_equal(result["linearAlbedo"][owned], values[1][owned])


def testUncertainSemanticHaloDoesNotImplyInvisibleSource():
    arrays = makeObservations()
    values, weights, cosines, labels, confidence = arrays
    labels[:, 6:12, 8:14] = 2
    cosines[0, 8, 10] = 0.8
    confidence[0, 5, 10] = 0.1
    result = fuse(*arrays, featureHaloWidth=1)
    assert result["featureSourceIndex"][5, 10] == 0
    np.testing.assert_array_equal(result["linearAlbedo"][5, 10], values[0, 5, 10])
    # Changing the same halo texel's actual geometric visibility must reject
    # the complete region, despite all the trustworthy semantic seeds.
    weights[0, 5, 10] = 0
    incomplete = fuse(*arrays, featureHaloWidth=1)
    assert not incomplete["featureOwnershipMask"].any()


def testConfidentOtherFeatureInHaloRejectsWholeRegionOwnership():
    arrays = makeObservations()
    values, weights, cosines, labels, confidence = arrays
    labels[:, 6:12, 8:14] = 2
    cosines[0, 8, 10] = 0.8
    labels[0, 5, 10] = 4  # A confident eye must not become eyebrow halo.
    result = fuse(*arrays, featureHaloWidth=1)
    assert not result["featureOwnershipMask"].any()
    assert any(0 in region["semanticConflictSources"]
               for region in result["diagnostics"]["featureOwnership"]["regions"])


def testCoreAndSemanticProtectionAreBitExactAndInputsUnchanged():
    arrays = makeObservations()
    values, weights, cosines, labels, confidence = arrays
    rng = np.random.default_rng(81)
    values[:] = rng.random(values.shape)
    cosines[0, 2:9, 2:9] = 0.9
    labels[:, 11:14, 13:18] = 12
    cosines[0, 11:14, 13:18] = 0.8
    before = [array.copy() for array in arrays]
    result = fuse(*arrays)
    baseline = fuseSemanticObservations(*arrays, referenceBlendWidth=0, matchAuxiliaryColors=False,
                                       sourceIterations=0)
    core = baseline["referenceCoreMask"] | baseline["protectionMask"]
    assert core.any()
    np.testing.assert_array_equal(result["linearAlbedo"][core], values[0][core])
    np.testing.assert_array_equal(result["referenceCoreMask"], baseline["referenceCoreMask"])
    assert result["diagnostics"]["referenceCoreAndProtectionMaximumError"] == 0
    for previous, current in zip(before, arrays):
        np.testing.assert_array_equal(previous, current)


def testNoObservationDoesNotInventColorOrSource():
    arrays = makeObservations()
    arrays[1][:, 5:9, 6:10] = 0
    arrays[0][:, 5:9, 6:10] = 1
    result = fuse(*arrays)
    missing = ~result["coverageMask"]
    assert missing.any()
    for key in ("linearAlbedo", "unclippedLinearAlbedo", "lowFrequencyAlbedo", "highFrequencyResidual"):
        assert (result[key][missing] == 0).all()
    assert (result["highFrequencySourceIndex"][missing] == -1).all()
    assert (result["confidence"][missing] == 0).all()
    arrays[1][:] = 0
    empty = fuse(*arrays)
    assert not empty["coverageMask"].any()
    assert empty["diagnostics"]["unclippedMaximum"] == 0


def testInvalidWallPreventsLowFrequencyColorExchange():
    values = np.zeros((1, 15, 17, 3), np.float32)
    values[:, :, :8] = 0.1
    values[:, :, 8:] = 0.9
    valid = np.ones(values.shape[:3], bool)
    valid[:, :, 8] = False
    result = decomposeObservationFrequency(values, valid, lowFrequencySigma=4)
    np.testing.assert_allclose(result["lowFrequency"][0, :, :8], 0.1, atol=1e-6)
    np.testing.assert_allclose(result["lowFrequency"][0, :, 9:], 0.9, atol=1e-6)
    assert (result["lowFrequency"][0, :, 8] == 0).all()
    assert (result["highFrequency"][0, :, 8] == 0).all()


def testTouchingChartBoundaryBlocksLowFrequencyAndFeatureOwnershipExchange():
    arrays = makeObservations(height=17, width=20)
    values, weights, cosines, labels, confidence = arrays
    values[:, :, :10] = 0.1
    values[:, :, 10:] = 0.9
    charts = np.zeros((17, 20), np.int32)
    charts[:, 10:] = 1
    result = decomposeObservationFrequency(values, weights > 0, uvChartLabels=charts)
    np.testing.assert_allclose(result["lowFrequency"], values, atol=1e-6)
    labels[:, 7:10, 8:12] = 12
    cosines[0, 8, 9] = 0.8
    cosines[0, :, 10:] = 0.1
    fused = fuse(*arrays, uvChartLabels=charts, featureHaloWidth=1)
    assert fused["featureSourceIndex"][8, 9] == 0
    assert fused["featureSourceIndex"][8, 10] == 1
    assert fused["diagnostics"]["featureOwnership"]["regionCount"] == 2


def testUnknownChartTexelsAreIsolatedAndCannotBecomeWholeFeatureRegion():
    arrays = makeObservations()
    charts = np.full(arrays[1].shape[1:], -1, np.int32)
    arrays[3][:] = 12
    arrays[2][:] = 0.8
    result = fuse(*arrays, uvChartLabels=charts)
    assert not result["featureOwnershipMask"].any()
    assert (result["highFrequencyResidual"] == 0).all()
    assert result["coverageMask"].all()


def testRecombinationClippingIsReportedAndUnclippedColorsAreRetained():
    arrays = makeObservations(height=15, width=15)
    values = arrays[0]
    values[0] = 0
    values[0, 7, 7] = 1
    values[1] = 1
    result = fuse(*arrays, protectFeatureRegions=False)
    assert result["unclippedLinearAlbedo"][7, 7, 0] > 1
    assert result["linearAlbedo"][7, 7, 0] == 1
    assert result["clippingMask"][7, 7]
    assert result["diagnostics"]["recombinationClippedTexels"] > 0


def testSingleViewOutputIsExactIncludingSupportEdges():
    arrays = makeObservations(count=1)
    arrays[0][:] = np.random.default_rng(7).random(arrays[0].shape)
    arrays[1][:, 0] = 0
    result = fuse(*arrays)
    covered = result["coverageMask"]
    np.testing.assert_array_equal(result["linearAlbedo"][covered], arrays[0][0][covered])


@pytest.mark.parametrize("arguments", [
    {"lowFrequencySigma": -1}, {"lowFrequencySigma": np.inf}, {"lowFrequencySigma": True},
    {"featureHaloWidth": -1}, {"featureHaloWidth": True}, {"featureHaloWidth": 1.2},
    {"featureConfidence": 0}, {"featureSeedCosine": 0.1}, {"protectFeatureRegions": 1},
    {"uvChartLabels": np.zeros((4, 4), int)},
])
def testInvalidConfigurationIsRejected(arguments):
    with pytest.raises(ValueError):
        fuse(*makeObservations(), **arguments)


def testEdgeJumpDiagnosticsUseSameSourceBoundaryForBaselineAndResult():
    arrays = makeObservations()
    arrays[2][0, :, :10] = 0.4
    arrays[2][0, :, 10:] = 0.6
    arrays[0][1] = 0.5
    result = fuse(*arrays, protectFeatureRegions=False)
    output = result["diagnostics"]["outputEdgeJump"]
    baseline = result["diagnostics"]["baselineAtSameEdges"]
    assert output["sourceSwitchEdges"] == baseline["sourceSwitchEdges"] == 19
    assert output["meanLinearRgbJump"] >= 0
    assert np.isfinite(list(output.values())).all()
