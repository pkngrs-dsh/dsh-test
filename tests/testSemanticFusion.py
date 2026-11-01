import numpy as np
import pytest

from faceAppearance.referenceFusion import fuseReferenceObservations
from faceAppearance.semanticFusion import (buildSemanticProtection, fuseSemanticObservations,
                                            selectContinuousSources)


def makeObservations(count=3, height=15, width=17):
    values = np.full((count, height, width, 3), 0.2, np.float32)
    weights = np.ones((count, height, width), np.float32)
    cosines = np.ones_like(weights)
    labels = np.ones_like(weights, np.int16)
    confidence = np.ones_like(weights)
    return values, weights, cosines, labels, confidence


def testDisabledFusionExactlyMatchesBaselineIncludingDiagnostics():
    values, weights, cosines, labels, confidence = makeObservations()
    values[1] = 0.35
    values[2] = 0.7
    weights[:, 0, 0] = 0
    cosines[0, 7, 7] = 0.5
    arguments = {"referenceViewIndex": 2, "referenceBlendWidth": 3, "matchAuxiliaryColors": False}
    baseline = fuseReferenceObservations(values, weights, cosines, **arguments)
    result = fuseSemanticObservations(values, weights, cosines, enabled=False, **arguments)
    assert result.keys() == baseline.keys()
    for key in result:
        if isinstance(result[key], np.ndarray):
            np.testing.assert_array_equal(result[key], baseline[key])
        else:
            assert result[key] == baseline[key]
    defaultBaseline = fuseReferenceObservations(values, weights, cosines)
    defaultDisabled = fuseSemanticObservations(values, weights, cosines, enabled=False)
    np.testing.assert_array_equal(defaultDisabled["linearAlbedo"], defaultBaseline["linearAlbedo"])


def testProtectionRetainsReferenceExactlyAgainstArbitraryAuxiliaryWeight():
    values, weights, cosines, labels, confidence = makeObservations()
    labels[:, 1:5, 6:10] = 12
    values[0, 1:5, 6:10] = [0.3, 0.11, 0.1]
    values[1:] = 0.9
    weights[0] = 0.001
    weights[1:] = 10000
    before = [array.copy() for array in (values, weights, cosines, labels, confidence)]
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                       referenceBlendWidth=12, matchAuxiliaryColors=False)
    mask = result["protectionMask"]
    assert mask[1:5, 6:10].all()
    assert (result["referenceAlpha"][mask] == 1).all()
    assert (result["sourceIndex"][mask] == 0).all()
    np.testing.assert_array_equal(result["linearAlbedo"][mask], values[0][mask])
    assert result["diagnostics"]["protectedReferenceMaximumError"] == 0
    for old, current in zip(before, (values, weights, cosines, labels, confidence)):
        np.testing.assert_array_equal(current, old)


def testHaloIsBoundedAndCannotJumpAcrossDifferentLabelsOrInvalidSupport():
    _, weights, cosines, labels, confidence = makeObservations(count=1, height=9, width=11)
    labels[:] = 12
    confidence[:] = 0.7
    confidence[0, 4, 3] = 0.95
    labels[0, 4, 4] = 11  # An inner-mouth texel cannot be crossed.
    weights[0, 3, 3] = 0
    result = buildSemanticProtection(labels, confidence, weights, cosines, haloWidth=2)
    mask = result["protectionMask"]
    assert mask[4, 3] and mask[4, 1]
    assert not mask[4, 0]
    assert not mask[4, 4] and not mask[4, 5]
    assert not mask[3, 3] and not mask[2, 3]
    assert result["diagnostics"]["seedTexels"] == 1
    assert result["diagnostics"]["haloTexels"] > 0


def testProjectionHolesPoorCosinesAndLowConfidenceAreNeverProtected():
    _, weights, cosines, labels, confidence = makeObservations()
    labels[:] = 10
    weights[0, 6, 6] = 0
    cosines[0, 6, 7] = 0.1
    confidence[0, 6, 8] = 0.2
    result = buildSemanticProtection(labels, confidence, weights, cosines, haloWidth=10)
    assert not result["protectionMask"][6, 6:9].any()
    assert result["protectionMask"][7, 7]


def testCoherentSourceSelectionReducesWeakCheckerboardAndKeepsEnergyMonotonic():
    values, weights, _, labels, confidence = makeObservations(height=16, width=18)
    weights[0] = 0
    yy, xx = np.indices(weights.shape[1:])
    checkerboard = (yy + xx) % 2
    weights[1] = 0.5 + 0.01 * checkerboard
    weights[2] = 0.5 + 0.01 * (1 - checkerboard)
    result = selectContinuousSources(values, weights, labels, confidence, smoothness=0.3)
    stats = result["diagnostics"]
    assert stats["initialContinuity"]["sourceSwitchFraction"] == 1
    assert stats["finalContinuity"]["sourceSwitchFraction"] < 0.1
    assert stats["changedTexels"] > 0
    assert (np.diff(stats["energyHistory"]) <= 1e-5).all()
    assert np.isin(result["sourceIndex"], [1, 2]).all()


def testSpatialRegularizationCannotSelectAnUnobservedSource():
    values, weights, _, labels, confidence = makeObservations()
    weights[0] = 0
    weights[1] = 0.9
    weights[2] = 0.1
    weights[1, 5:9, 5:9] = 0
    weights[:, 7, 7] = 0
    result = selectContinuousSources(values, weights, labels, confidence, smoothness=100)
    source = result["sourceIndex"]
    yy, xx = np.indices(source.shape)
    covered = source >= 0
    assert (weights[np.maximum(source, 0), yy, xx][covered] > 0).all()
    assert source[7, 7] == -1
    assert source[6, 6] == 2


def testColorAgreementCanChooseTheBetterAuxiliaryDespiteModerateWeightAdvantage():
    values, weights, _, labels, confidence = makeObservations()
    values[0] = 0.2
    values[1] = 0.8
    values[2] = 0.21
    weights[1] = 1
    weights[2] = 0.8
    result = selectContinuousSources(values, weights, labels, confidence, smoothness=0)
    assert (result["sourceIndex"] == 2).all()


def testReliableSemanticAgreementWinsAndMissingAgreementRetainsCoverage():
    values, weights, _, labels, confidence = makeObservations()
    labels[0] = 12
    labels[1] = 11
    labels[2] = 12
    weights[1] = 100
    weights[2] = 0.01
    weights[2, 4, 4] = 0
    result = selectContinuousSources(values, weights, labels, confidence, smoothness=0)
    assert result["sourceIndex"][5, 5] == 2
    assert result["sourceIndex"][4, 4] == 1
    assert result["coverageMask"].all()
    assert result["diagnostics"]["semanticAgreementFallbackTexels"] == 1


def testConfidentSemanticDisagreementNeverMixesAndUsesReliableReference():
    values, weights, cosines, labels, confidence = makeObservations(count=2)
    values[0] = 0.2
    values[1] = 0.8
    labels[0] = 1
    labels[1] = 11
    cosines[0] = 0.5
    cosines[0, 6, 6] = 0.1
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                       referenceBlendWidth=0, matchAuxiliaryColors=False,
                                       narrowBoundaryBlend=False)
    assert result["referenceAlpha"][7, 7] == 1
    assert result["referenceAlpha"][6, 6] == 0
    assert (result["sourceIndex"] != -2).all()
    np.testing.assert_array_equal(result["linearAlbedo"][7, 7], values[0, 7, 7])
    np.testing.assert_array_equal(result["linearAlbedo"][6, 6], values[1, 6, 6])


def testUnknownSemanticsDoNotCreateHolesOrRejectGeometricFeather():
    values, weights, cosines, labels, confidence = makeObservations(count=2)
    confidence[:] = 0
    labels[0] = 12
    labels[1] = 1
    cosines[0] = 0.5
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                       referenceBlendWidth=0, matchAuxiliaryColors=False,
                                       narrowBoundaryBlend=False)
    assert result["coverageMask"].all()
    assert not result["protectionMask"].any()
    assert (result["referenceAlpha"] == 0.5).all()


def testIntermediateAlphaIsConfinedToNarrowOwnershipBoundary():
    values, weights, cosines, labels, confidence = makeObservations(count=2, height=31, width=35)
    cosines[0] = 0.5
    cosines[0, :, :15] = 0.45
    confidence[:] = 0
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                       referenceBlendWidth=2, matchAuxiliaryColors=False,
                                       narrowBoundaryBlend=True)
    mixed = (result["referenceAlpha"] > 0) & (result["referenceAlpha"] < 1)
    assert mixed.any()
    assert result["blendAllowedMask"][mixed].all()
    assert result["referenceAlpha"][15, 6] == 0
    assert result["referenceAlpha"][15, 25] == 1
    assert result["diagnostics"]["boundaryBlending"]["snappedOutsideBandTexels"] > 0


def testDefaultPreservesSoftAngleRampAndExactlyMatchesExplicitFalse():
    values, weights, cosines, labels, confidence = makeObservations(count=2, height=31, width=35)
    values[1] = 0.8
    cosines[0] = 0.5
    cosines[0, :, :15] = 0.45
    confidence[:] = 0
    arguments = {"referenceBlendWidth": 2, "matchAuxiliaryColors": False}
    default = fuseSemanticObservations(values, weights, cosines, labels, confidence, **arguments)
    explicit = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                         narrowBoundaryBlend=False, **arguments)
    assert default.keys() == explicit.keys()
    for key in default:
        if isinstance(default[key], np.ndarray):
            np.testing.assert_array_equal(default[key], explicit[key])
        else:
            assert default[key] == explicit[key]
    baseline = fuseReferenceObservations(values, weights, cosines, **arguments)
    np.testing.assert_array_equal(default["referenceAlpha"], baseline["referenceAlpha"])
    assert 0 < default["referenceAlpha"][15, 6] < 1
    assert default["referenceAlpha"][15, 25] == 0.5
    assert not default["diagnostics"]["boundaryBlending"]["enabled"]
    assert default["diagnostics"]["boundaryBlending"]["snappedOutsideBandTexels"] == 0


@pytest.mark.parametrize("narrowBoundaryBlend", [False, True])
def testOriginalReferenceCoreRemainsBitExactWithNarrowBlendingAndSemantics(narrowBoundaryBlend):
    values, weights, cosines, labels, confidence = makeObservations(height=31, width=35)
    values[1:] = 0.8
    labels[1:] = 11
    baseline = fuseReferenceObservations(values, weights, cosines, referenceBlendWidth=3)
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence,
                                       referenceBlendWidth=3, narrowBoundaryBlend=narrowBoundaryBlend)
    core = baseline["referenceCoreMask"]
    assert core.any()
    np.testing.assert_array_equal(result["linearAlbedo"][core], baseline["linearAlbedo"][core])


def testFusionPreservesMissingSupportAndSoleReferenceFallback():
    values, weights, cosines, labels, confidence = makeObservations(count=1)
    labels[:] = 10
    weights[:, 4, 4] = 0
    cosines[:] = 0.2
    result = fuseSemanticObservations(values, weights, cosines, labels, confidence)
    valid = weights[0] > 0
    assert (result["sourceIndex"][valid] == 0).all()
    assert result["sourceIndex"][4, 4] == -1
    assert not result["coverageMask"][4, 4]
    assert not result["linearAlbedo"][4, 4].any()
    assert result["confidence"][4, 4] == 0
    assert not result["protectionMask"].any()
    np.testing.assert_array_equal(result["linearAlbedo"][valid], values[0][valid])


@pytest.mark.parametrize("badLabels", [-1, 19, 100])
def testInvalidLabelsFail(badLabels):
    values, weights, cosines, labels, confidence = makeObservations()
    labels.flat[0] = badLabels
    with pytest.raises(ValueError, match="semantic labels"):
        fuseSemanticObservations(values, weights, cosines, labels, confidence)


@pytest.mark.parametrize("badConfidence", [-0.1, 1.1, np.nan, np.inf])
def testInvalidSemanticConfidenceFails(badConfidence):
    values, weights, cosines, labels, confidence = makeObservations()
    confidence.flat[0] = badConfidence
    with pytest.raises(ValueError, match="confidence"):
        fuseSemanticObservations(values, weights, cosines, labels, confidence)


def testFloatingLabelsShapeMismatchAndAbsentSemanticDataFail():
    values, weights, cosines, labels, confidence = makeObservations()
    for badLabels, badConfidence in ((labels.astype(np.float32), confidence),
                                     (labels[:, :, :3], confidence), (labels, confidence[:, :2])):
        with pytest.raises(ValueError):
            fuseSemanticObservations(values, weights, cosines, badLabels, badConfidence)
    with pytest.raises(ValueError, match="requires"):
        fuseSemanticObservations(values, weights, cosines)


@pytest.mark.parametrize("arguments", [{"referenceViewIndex": True}, {"haloWidth": -1},
                                      {"haloWidth": 1.5}, {"confidenceThreshold": 0},
                                      {"seedConfidenceThreshold": 0.5}, {"seedMinimumCosine": -1},
                                      {"protectedLabelIds": (0, 2)}, {"protectedLabelIds": (2, 2)}])
def testInvalidProtectionParametersFail(arguments):
    _, weights, cosines, labels, confidence = makeObservations()
    with pytest.raises(ValueError):
        buildSemanticProtection(labels, confidence, weights, cosines, **arguments)


@pytest.mark.parametrize("arguments", [{"smoothness": np.nan}, {"colorWeight": -1},
                                      {"iterations": True}, {"iterations": -1},
                                      {"excludeReference": 1}, {"confidenceThreshold": 1.1}])
def testInvalidSelectionParametersFail(arguments):
    values, weights, _, labels, confidence = makeObservations()
    with pytest.raises(ValueError):
        selectContinuousSources(values, weights, labels, confidence, **arguments)
