import cv2
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from faceAppearance.uvAlignment import (OpenCvUvFlowAdapter, alignUvObservations,
                                         sampleUvArray, _jacobianDeterminant)


class ConstantFlowAdapter:
    def __init__(self, dx=2, dy=0, inconsistent=False):
        self.dx, self.dy, self.inconsistent = dx, dy, inconsistent
        self.calls = 0

    def estimateFlow(self, referenceStructure, sourceStructure):
        sign = 1 if self.calls % 2 == 0 or self.inconsistent else -1
        self.calls += 1
        flow = np.zeros(referenceStructure.shape + (2,), np.float32)
        flow[:] = (sign * self.dx, sign * self.dy)
        return flow


def makeTranslation(dx=2, dy=0, size=96):
    rng = np.random.default_rng(28)
    texture = gaussian_filter(rng.random((size, size)).astype(np.float32), 1)
    texture = .1 + .7 * (texture - texture.min()) / np.ptp(texture)
    first = np.repeat(texture[..., None], 3, axis=-1)
    second = cv2.warpAffine(first, np.array([[1, 0, dx], [0, 1, dy]], np.float32), (size, size))
    colors = np.stack([first, second])
    weights = np.ones(colors.shape[:3], np.float32)
    labels = np.ones(weights.shape, np.uint8)
    confidence = np.ones_like(weights)
    return colors, weights, labels, confidence


def testSamplingConventionUsesSourcePlusFlowAndPreservesInput():
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    yy, xx = np.indices(values.shape, dtype=np.float32)
    grid = np.stack([xx + 1, yy], axis=-1)
    sampled = sampleUvArray(values, grid)
    np.testing.assert_array_equal(sampled[:, :-1], values[:, 1:])
    assert np.all(sampled[:, -1] == 0)


@pytest.mark.parametrize("dx,dy", [(2, 0), (-2, 1), (0, -3)])
def testKnownTranslationCorrectDirectionAndReferenceExact(dx, dy):
    inputs = makeTranslation(dx, dy)
    originals = [value.copy() for value in inputs]
    result = alignUvObservations(*inputs, flowAdapter=ConstantFlowAdapter(dx, dy))
    center = np.s_[25:70, 25:70]
    np.testing.assert_allclose(result["samplingFlow"][1][center], np.broadcast_to([dx, dy], (45, 45, 2)), atol=1e-5)
    np.testing.assert_allclose(result["alignedObservations"][1][center], inputs[0][0][center], atol=1e-5)
    np.testing.assert_array_equal(result["alignedObservations"][0], inputs[0][0])
    assert not result["acceptedMask"][0].any()
    assert result["diagnostics"]["views"][1]["structureAfter"] < result["diagnostics"]["views"][1]["structureBefore"]
    assert result["diagnostics"]["views"][1]["maximumFinalForwardBackwardError"] <= 1.5
    assert np.linalg.norm(result["samplingFlow"], axis=-1).max() <= 6
    for old, current in zip(originals, inputs):
        np.testing.assert_array_equal(old, current)


def testOpenCvAdapterRecoversSyntheticShiftWithoutDirectionInversion():
    colors, *_ = makeTranslation(2, 1, size=96)
    flow = OpenCvUvFlowAdapter().estimateFlow(colors[0, ..., 0], colors[1, ..., 0])
    np.testing.assert_allclose(np.median(flow[20:-20, 20:-20], axis=(0, 1)), [2, 1], atol=.3)


def testNoSupportExpansionAndNoCrossingAHoleOrSemanticStrip():
    colors, weights, labels, confidence = makeTranslation(4)
    weights[:, :, 46] = 0
    labels[:, :, 65] = 12
    result = alignUvObservations(colors, weights, labels, confidence, flowAdapter=ConstantFlowAdapter(4))
    assert not result["acceptedMask"][:, :, 46].any()
    assert not result["acceptedMask"][:, :, 65].any()
    assert result["diagnostics"]["supportExpandedTexels"] == 0
    assert not (result["alignedWeights"] > weights + 1e-7).any()
    coordinates = result["samplingCoordinates"][1]
    yy, xx = np.indices(weights.shape[1:])
    accepted = result["acceptedMask"][1]
    assert not (accepted & (xx < 46) & (coordinates[..., 0] >= 46)).any()
    assert not (accepted & (xx < 65) & (coordinates[..., 0] >= 65)).any()
    np.testing.assert_array_equal(result["alignedObservations"][1][~accepted], colors[1][~accepted])


def testUnknownAndDisagreeingSemanticsRemainUnchanged():
    colors, weights, labels, confidence = makeTranslation()
    confidence[:, 30:50] = .2
    labels[1, 60:75] = 12
    result = alignUvObservations(colors, weights, labels, confidence, flowAdapter=ConstantFlowAdapter())
    assert not result["acceptedMask"][:, 30:50].any()
    assert not result["acceptedMask"][:, 60:75].any()
    np.testing.assert_array_equal(result["alignedObservations"][:, 30:50], colors[:, 30:50])


def testExcessMotionCannotBeMadeAcceptableByClipping():
    inputs = makeTranslation()
    result = alignUvObservations(*inputs, flowAdapter=ConstantFlowAdapter(20))
    assert not result["acceptedMask"].any()
    assert result["diagnostics"]["views"][1]["rejectedExcessDisplacementTexels"] == 96 * 96
    np.testing.assert_array_equal(result["alignedObservations"], inputs[0])


def testForwardBackwardInconsistencyRejectsInteriorMotion():
    inputs = makeTranslation()
    result = alignUvObservations(*inputs, flowAdapter=ConstantFlowAdapter(2, inconsistent=True))
    assert not result["acceptedMask"][1, 25:70, 25:70].any()
    assert result["diagnostics"]["views"][1]["rejectedInconsistentTexels"] > 0


@pytest.mark.parametrize("failure", ["nan", "shape", "exception"])
def testAdapterFailureIsExplicitAndFallsBack(failure):
    class BadAdapter:
        def estimateFlow(self, first, second):
            if failure == "exception":
                raise RuntimeError("test failure")
            return np.full(first.shape + ((2,) if failure == "nan" else (3,)), np.nan if failure == "nan" else 0, np.float32)
    inputs = makeTranslation()
    result = alignUvObservations(*inputs, flowAdapter=BadAdapter())
    assert result["diagnostics"]["views"][1]["status"] == "estimatorFailed"
    np.testing.assert_array_equal(result["alignedObservations"], inputs[0])


def testFinalExportedFieldHasNoFoldIncludingRejectedBoundaries():
    inputs = makeTranslation(4)
    inputs[1][:, 40:55, 25:70] = 0
    result = alignUvObservations(*inputs, flowAdapter=ConstantFlowAdapter(4))
    for flow in result["samplingFlow"]:
        determinant = _jacobianDeterminant(flow)
        assert determinant.min() >= .35
        assert determinant.max() <= 2.5


def testExportedCoordinatesAndAcceptedColorUseExactlyTheSameMap():
    inputs = makeTranslation(2, 1)
    result = alignUvObservations(*inputs, flowAdapter=ConstantFlowAdapter(2, 1))
    sampled = sampleUvArray(inputs[0][1], result["samplingCoordinates"][1])
    accepted = result["acceptedMask"][1]
    np.testing.assert_array_equal(sampled[accepted], result["alignedObservations"][1][accepted])
    assert np.all(result["samplingFlow"][1][~accepted] == 0)


def testChangingReferenceIndexDoesNotMoveTheNewReference():
    colors, weights, labels, confidence = makeTranslation(2)
    result = alignUvObservations(colors, weights, labels, confidence,
                                  referenceViewIndex=1, flowAdapter=ConstantFlowAdapter(-2))
    np.testing.assert_array_equal(result["alignedObservations"][1], colors[1])
    assert not result["acceptedMask"][1].any()
    np.testing.assert_allclose(result["samplingFlow"][0, 30:60, 30:60],
                               np.broadcast_to([-2, 0], (30, 30, 2)), atol=1e-5)


def testSingleViewAndInsufficientOverlapAreExactNoOps():
    inputs = makeTranslation()
    single = alignUvObservations(*(value[:1] for value in inputs))
    np.testing.assert_array_equal(single["alignedObservations"], inputs[0][:1])
    inputs[1][1] = 0
    result = alignUvObservations(*inputs)
    assert result["diagnostics"]["views"][1]["status"] == "insufficientSemanticOverlap"
    np.testing.assert_array_equal(result["alignedObservations"], inputs[0])


@pytest.mark.parametrize("arguments", [{"referenceViewIndex": True}, {"referenceViewIndex": 2},
                                        {"maxDisplacement": 0}, {"maxDisplacement": np.nan},
                                        {"taperWidth": 1}, {"minimumOverlap": 1.5},
                                        {"confidenceThreshold": 0}, {"minimumJacobian": 0}])
def testInvalidConfigurationRejected(arguments):
    with pytest.raises(ValueError):
        alignUvObservations(*makeTranslation(), **arguments)


def testMalformedDataRejected():
    colors, weights, labels, confidence = makeTranslation()
    colors[1, 20, 20] = np.nan
    with pytest.raises(ValueError):
        alignUvObservations(colors, weights, labels, confidence)
