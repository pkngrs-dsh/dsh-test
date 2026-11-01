import numpy as np
import pytest

from faceAppearance.adapters.eyelidLandmarks import (EyelidLandmarksAdapter,
                                                     _cropWithPadding, _makeInputs,
                                                     decodeEyelidHeatmaps)


def makeImage():
    image = np.zeros((100, 120, 3), np.uint8)
    image[..., 0] = np.arange(120, dtype=np.uint8)
    image[..., 1] = 170
    anchors = np.array([[35, 35], [85, 35], [60, 55], [60, 75]], np.float32)
    return image, anchors


class FakeAdapter(EyelidLandmarksAdapter):
    def __init__(self, projectRoot):
        super().__init__(projectRoot)
        self.backendLoads = 0

    def loadBackend(self, path):
        self.backendLoads += 1
        return object()

    def runBackend(self, inputs):
        face = np.zeros((1, 68, 64, 64), np.float32)
        face[:, :, 32, 32] = .9
        eyes = np.zeros((2, 43, 32, 32), np.float32)
        eyes[:, :19, 16, 16] = .9
        # Both canonical crops travel right -> left; eye zero is flipped back.
        for point in range(12):
            x = 26 - point * 2
            lowerY = 16 if point in (0, 11) else 20
            upperY = 16 if point in (0, 11) else 12
            eyes[:, 19 + point, lowerY, x] = .7 + .01 * point
            eyes[:, 31 + point, upperY, x] = .7 + .01 * point
        return face, eyes


def loadedAdapter(tmp_path):
    adapter = FakeAdapter(tmp_path)
    path = adapter.assetPath()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test model")
    adapter.loadModels()
    return adapter


def testLocalHeatmapCentroidUsesUpstreamHalfPixelAndKeepsRawPeaks():
    values = np.zeros((1, 2, 10, 12), np.float32)
    values[0, 0, 3, 5] = 2
    values[0, 0, 3, 6] = 1
    values[0, 1] = -1
    result = decodeEyelidHeatmaps(values, radiusFraction=.2)
    np.testing.assert_allclose(result["points"][0, 0], [5.5 + 1/3, 3.5], atol=1e-6)
    assert result["rawPeakScores"][0, 0] == 2
    assert result["rawPeakScores"][0, 1] == -1
    assert not result["validMask"][0, 1]
    np.testing.assert_array_equal(result["points"][0, 1], [0, 0])


def testLocalHeatmapDecodeRejectsDistantSecondPeak():
    values = np.zeros((1, 1, 32, 32), np.float32)
    values[0, 0, 10, 10] = 1
    values[0, 0, 25, 25] = .9
    result = decodeEyelidHeatmaps(values)
    np.testing.assert_array_equal(result["points"][0, 0], [10.5, 10.5])


def testPaddingAndCropHaveExactOriginalImageCoordinates():
    image = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
    result = _cropWithPadding(image, np.array([-2, -1]), np.array([4, 3]))
    assert result.shape == (4, 6, 3)
    np.testing.assert_array_equal(result[1:, 2:], image[:3, :4])
    assert not result[0].any() and not result[:, :2].any()


def testModelInputsAreRgbNormalizedAndOnlyFirstEyeIsFlipped():
    image, anchors = makeImage()
    inputs, crop = _makeInputs(image, anchors)
    assert inputs["input.1"].shape == (1, 3, 256, 256)
    assert inputs["leye_img_flip"].shape == (1, 3, 128, 128)
    assert inputs["reye_img"].shape == (1, 3, 128, 128)
    assert inputs["leye_img_flip"][0, 0, 50, 20] > inputs["leye_img_flip"][0, 0, 50, 100]
    assert inputs["reye_img"][0, 0, 50, 20] < inputs["reye_img"][0, 0, 50, 100]
    np.testing.assert_allclose(inputs["reye_img"][0, 1], 170/255)
    assert crop["eyeSize"] == 32


def testPreloadingIsIdempotentAndMissingAssetsFailClosed(tmp_path):
    adapter = FakeAdapter(tmp_path)
    with pytest.raises(FileNotFoundError):
        adapter.loadModels()
    assert not adapter.isLoaded
    adapter = loadedAdapter(tmp_path)
    adapter.loadModels()
    assert adapter.backendLoads == 1
    assert adapter.modelMetadata["weightSha256"]


def testInferenceRequiresPreload(tmp_path):
    image, anchors = makeImage()
    with pytest.raises(RuntimeError):
        FakeAdapter(tmp_path).infer(image, faceAnchors=anchors)


def testCurveOrderFlipBackConfidenceAndOriginalCoordinates(tmp_path):
    adapter = loadedAdapter(tmp_path)
    image, anchors = makeImage()
    imageBefore, anchorsBefore = image.copy(), anchors.copy()
    result = adapter.infer(image, faceAnchors=anchors)
    curves = result["eyelidCurves"]
    assert curves.shape == (2, 2, 12, 2)
    assert (np.diff(curves[..., 0], axis=-1) > 0).all()
    assert (curves[:, 0, 1:-1, 1] > curves[:, 1, 1:-1, 1]).all()
    # Eye zero: crop origin 19, radius 16, first model coordinate 26.5.
    assert curves[0, 0, 0, 0] == 19 + 32 - 1 - 26.5
    # Eye one is unflipped, then sequence-reversed (original point 11 x=4.5).
    assert curves[1, 0, 0, 0] == 69 + 4.5
    np.testing.assert_allclose(result["rawPeakScores"][0, 0], .7 + .01*np.arange(12))
    np.testing.assert_allclose(result["rawPeakScores"][1, 0], (.7 + .01*np.arange(12))[::-1])
    np.testing.assert_array_equal(result["corners"][:, 0], curves[:, 0, 0])
    assert all(item["valid"] for item in result["quality"])
    np.testing.assert_array_equal(image, imageBefore)
    np.testing.assert_array_equal(anchors, anchorsBefore)


def testRetinaFaceMouthMeanIsExplicitlyRecorded(tmp_path):
    image, anchors = makeImage()
    five = np.concatenate([anchors[:3], [[55, 75], [65, 75]]])
    result = loadedAdapter(tmp_path).infer(image, faceAnchors=five)
    np.testing.assert_array_equal(result["faceAnchors"], anchors)
    assert result["metadata"]["anchorSource"] == "retinaFaceFivePointsWithMeanMouthCorners"


def testRawPeakAboveOneRetainedButNotClaimedProbability(tmp_path):
    adapter = loadedAdapter(tmp_path)
    original = adapter.runBackend
    def highPeak(inputs):
        face, eyes = original(inputs)
        return face, eyes*2
    adapter.runBackend = highPeak
    image, anchors = makeImage()
    result = adapter.infer(image, faceAnchors=anchors)
    assert result["rawPeakScores"].max() > 1
    assert result["curveConfidence"].max() == 1
    assert "not a calibrated probability" in result["metadata"]["confidenceKind"]


def testNonpositiveMapsProduceInvalidEyeQuality(tmp_path):
    adapter = loadedAdapter(tmp_path)
    adapter.runBackend = lambda inputs: (np.zeros((1, 68, 64, 64), np.float32), np.zeros((2, 43, 32, 32), np.float32))
    image, anchors = makeImage()
    result = adapter.infer(image, faceAnchors=anchors)
    assert not result["pointValidMask"].any()
    assert not result["curveConfidence"].any()
    assert all(not item["valid"] for item in result["quality"])


@pytest.mark.parametrize("change", ["wrongImageType", "wrongShape", "outOfBounds", "eyeOrder", "nonfinite", "tooClose"])
def testInvalidInputsRejected(tmp_path, change):
    adapter = loadedAdapter(tmp_path)
    image, anchors = makeImage()
    if change == "wrongImageType": image = image.astype(np.float32)
    if change == "wrongShape": anchors = anchors[:3]
    if change == "outOfBounds": anchors[2] = [-1, 50]
    if change == "eyeOrder": anchors[[0, 1]] = anchors[[1, 0]]
    if change == "nonfinite": anchors[0, 0] = np.nan
    if change == "tooClose": anchors[1] = anchors[0] + [1, 0]
    with pytest.raises(ValueError): adapter.infer(image, faceAnchors=anchors)
