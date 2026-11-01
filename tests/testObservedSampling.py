import numpy as np
import pytest

from faceAppearance.observedSampling import normalizeObservedSamples, sampleObservedAtlas


def testExactDivisionRemovesZeroNeighborDarkeningOnlyWithObservedSupport():
    coverage = np.array([[1., .5, .75, 0.]], np.float32)
    color = np.array([.8, .4, .2], np.float32)
    premultiplied = coverage[..., None] * color
    originalColors, originalCoverage = premultiplied.copy(), coverage.copy()
    result = normalizeObservedSamples(premultiplied, coverage)
    np.testing.assert_array_equal(result["linearColors"][0, :3], np.broadcast_to(color, (3, 3)))
    np.testing.assert_array_equal(result["linearColors"][0, 3], 0)
    np.testing.assert_array_equal(result["validMask"], [[True, True, True, False]])
    np.testing.assert_array_equal(result["partialMask"], [[False, True, True, False]])
    np.testing.assert_array_equal(premultiplied, originalColors)
    np.testing.assert_array_equal(coverage, originalCoverage)
    result["sampleCoverage"][:] = 0
    np.testing.assert_array_equal(coverage, originalCoverage)


def testThresholdEqualityIsAcceptedAndImmediatelyLowerValueRejected():
    below = np.nextafter(np.float32(.5), np.float32(0))
    above = np.nextafter(np.float32(.5), np.float32(1))
    coverage = np.array([[below, .5, above]], np.float32)
    colors = np.repeat((coverage * .25)[..., None], 3, axis=-1)
    result = normalizeObservedSamples(colors, coverage, minimumCoverage=.5)
    np.testing.assert_array_equal(result["validMask"], [[False, True, True]])
    np.testing.assert_array_equal(result["partialMask"], [[False, True, True]])
    assert np.all(result["linearColors"][0, 0] == 0)
    assert np.all(result["linearColors"][0, 1:] == .25)
    np.testing.assert_array_equal(result["sampleCoverage"], coverage)


def testTinyPositiveCoverageIsDividedWithoutEpsilonAndZeroStaysInvalid():
    coverage = np.array([[1e-12, 0]], np.float64)
    color = np.array([.75, .5, .25])
    result = normalizeObservedSamples(coverage[..., None] * color, coverage, minimumCoverage=1e-12)
    np.testing.assert_array_equal(result["linearColors"][0, 0], color)
    assert not result["validMask"][0, 1]
    assert np.all(result["linearColors"][0, 1] == 0)


def testMinimumCoverageOneAcceptsOnlyCompleteFootprints():
    coverage = np.array([[1, .999, .5]], np.float64)
    result = normalizeObservedSamples(np.repeat(coverage[..., None], 3, axis=-1), coverage,
                                      minimumCoverage=1)
    np.testing.assert_array_equal(result["validMask"], [[True, False, False]])
    assert not result["partialMask"].any()


class AverageSampler:
    def __init__(self):
        self.inputs = []

    def sampleAtlas(self, values):
        self.inputs.append(values.copy())
        sampled = values.mean(axis=(0, 1), keepdims=True)
        if values.ndim == 2:
            sampled = sampled[..., None]
        return sampled


def testWrapperPremultipliesBeforeSamplingAndCannotUseInvalidAtlasColor():
    atlas = np.array([[[.8, .4, .2], [1., 1., 1.]]], np.float32)
    coverage = np.array([[True, False]])
    sampler = AverageSampler()
    result = sampleObservedAtlas(sampler, atlas, coverage)
    np.testing.assert_array_equal(sampler.inputs[0][0, 1], [0, 0, 0])
    np.testing.assert_array_equal(sampler.inputs[1], [[1, 0]])
    np.testing.assert_array_equal(result["linearColors"][0, 0], atlas[0, 0])
    assert result["validMask"][0, 0] and result["partialMask"][0, 0]
    assert result["sampleCoverage"][0, 0] == .5
    np.testing.assert_array_equal(atlas[0, 1], [1, 1, 1])


def testBareCallableSamplerAndNoSupportNeverInventColors():
    atlas = np.ones((2, 2, 3), np.float32)
    coverage = np.zeros((2, 2), bool)
    result = sampleObservedAtlas(lambda values: values, atlas, coverage)
    assert not result["validMask"].any()
    assert not result["partialMask"].any()
    assert not result["linearColors"].any()


def testNonuniformFootprintPreservesRelativeWeightsOfValidNeighbors():
    atlas = np.array([[[.8, .2, .1], [.1, .6, .4], [1., 1., 1.]]], np.float64)
    mask = np.array([[True, True, False]])
    footprint = np.array([[.2, .3, .5]])
    def weightedSampler(values):
        expandedWeights = footprint[..., None] if values.ndim == 3 else footprint
        return (values * expandedWeights).sum(axis=1, keepdims=True)
    result = sampleObservedAtlas(weightedSampler, atlas, mask)
    expected = (.2 * atlas[0, 0] + .3 * atlas[0, 1]) / .5
    np.testing.assert_array_equal(result["linearColors"][0, 0], expected)
    assert result["sampleCoverage"][0, 0] == .5


def testTensorLikeSamplerNeedsNoTorchDependency():
    class TensorLike:
        def __init__(self, array):
            self.array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array
    atlas = np.full((2, 2, 3), .4, np.float32)
    result = sampleObservedAtlas(lambda values: TensorLike(values), atlas, np.ones((2, 2), bool))
    np.testing.assert_array_equal(result["linearColors"], atlas)


def testForeignSamplerCannotMutateCallerArrays():
    atlas = np.full((2, 2, 3), .4, np.float32)
    coverage = np.ones((2, 2), bool)
    def destructiveSampler(values):
        values[:] = 0
        return values
    result = sampleObservedAtlas(destructiveSampler, atlas, coverage)
    assert np.all(atlas == .4) and coverage.all()
    assert not result["validMask"].any()


@pytest.mark.parametrize("minimumCoverage", [0, -1, 1.001, np.nan, np.inf, True, [.5], ".5"])
def testInvalidThresholdsRejected(minimumCoverage):
    with pytest.raises(ValueError):
        normalizeObservedSamples(np.zeros((1, 1, 3)), np.zeros((1, 1)), minimumCoverage=minimumCoverage)


@pytest.mark.parametrize("color,coverage", [
    ([.1, 0, 0], 0), ([.6, 0, 0], .5), ([-.1, 0, 0], 1),
    ([np.nan, 0, 0], 1), ([np.inf, 0, 0], 1), ([0, 0, 0], -.1),
    ([0, 0, 0], 1.1), ([0, 0, 0], np.nan), ([0, 0, 0], np.inf),
])
def testInvalidOrNonPremultipliedSamplesRejected(color, coverage):
    with pytest.raises(ValueError):
        normalizeObservedSamples(np.array([[color]]), np.array([[coverage]]))


def testShapesAndNonBooleanAtlasCoverageRejected():
    with pytest.raises(ValueError):
        normalizeObservedSamples(np.zeros((2, 3, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        normalizeObservedSamples(np.zeros((2, 3, 3)), np.ones((2, 3, 1)))
    with pytest.raises(ValueError):
        sampleObservedAtlas(lambda values: values, np.zeros((2, 3, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        sampleObservedAtlas(object(), np.zeros((2, 3, 3)), np.ones((2, 3), bool))
