import json

import numpy as np
from PIL import Image
import pytest

from faceAppearance.config import PipelineConfig
from faceAppearance.data import readImage
from faceAppearance.faceCrop import cropFaceSquare, workingImageValidMask


def detection(box=(20, 20, 70, 70), landmarks=((30, 35), (60, 35), (45, 55)), score=0.98):
    return {"boxXyxy": list(box), "landmarks": np.asarray(landmarks).tolist(), "score": score}


def testFaceCropRetainsEyesLostByFixedCenterCrop(tmp_path):
    pixels = np.zeros((1280, 720, 3), dtype=np.uint8)
    pixels[90:115, 245:275] = [255, 0, 0]
    pixels[90:115, 425:455] = [255, 0, 0]
    path = tmp_path / "topFace.png"
    Image.fromarray(pixels).save(path)
    face = detection((160, 20, 550, 610), ((260, 102), (440, 102), (350, 400)))
    center, centerRecord = readImage(path)
    cropped, record = readImage(path, cropPolicy="faceSquare", cropMargin=1.1,
                                faceDetector=lambda rgb: face)
    assert centerRecord["cropBoxXyxy"] == [0, 280, 720, 1000]
    assert not center.any()
    assert cropped[..., 0].max() > 0.95
    assert record["cropBoxXyxy"][1] == 0
    assert record["paddingLeftTopRightBottom"] == [0, 0, 0, 0]
    assert np.all((np.asarray(record["faceLandmarksWorking"]) >= 0)
                  & (np.asarray(record["faceLandmarksWorking"]) < 512))
    assert record["faceDetection"] == face
    json.dumps(record, allow_nan=False)


def testSelectionIsTraceableWithoutChangingDetectionOrCrop(tmp_path):
    path = tmp_path / "selected.png"
    Image.new("RGB", (100, 160), (128, 64, 32)).save(path)
    face = detection()
    baseline, baselineRecord = readImage(path, cropPolicy="faceSquare", faceDetector=lambda rgb: face)
    selection = {"strategy": "largestArea", "detectedCount": 2, "selectedIndex": 1}
    withSelection = {**face, "selection": selection}
    image, record = readImage(path, cropPolicy="faceSquare", faceDetector=lambda rgb: withSelection)
    np.testing.assert_array_equal(image, baseline)
    assert record["faceDetection"] == face
    assert record["faceSelection"] == selection
    assert {key: value for key, value in record.items() if key != "faceSelection"} == baselineRecord
    selection["selectedIndex"] = 0
    assert record["faceSelection"]["selectedIndex"] == 1
    json.dumps(record, allow_nan=False)


def testCropShiftsInsideOriginalBeforeConsideringPadding():
    pixels = np.zeros((120, 160, 3), np.uint8)
    cropped, record = cropFaceSquare(pixels, detection((2, 3, 62, 63), ((20, 25), (40, 25))), 1.5)
    assert cropped.shape == (90, 90, 3)
    assert record["cropBoxXyxy"] == [0, 0, 90, 90]
    assert record["paddingMode"] == "none"
    assert not record["qualityFlags"]


def testOutsideOriginalUsesExactEdgeReplicationAndReportsTruncation():
    pixels = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    cropped, record = cropFaceSquare(pixels, detection((-1, -1, 4, 3), ((0, 0), (3, 2))), 1)
    assert cropped.shape == (5, 5, 3)
    assert record["cropBoxXyxy"] == [-1, -2, 4, 3]
    assert record["paddingLeftTopRightBottom"] == [1, 2, 0, 0]
    np.testing.assert_array_equal(cropped, np.pad(pixels, ((2, 0), (1, 0), (0, 0)), mode="edge"))
    assert {"faceBoxTouchesOriginalBoundary", "faceBoxOutsideOriginalImage", "edgePaddingApplied"} <= set(record["qualityFlags"])


def testFractionalBoxIsNotClippedAtUnitMargin():
    cropped, record = cropFaceSquare(np.zeros((150, 150, 3), np.uint8),
                                     detection((10.8, 20.2, 110.8, 120.2)), 1)
    left, top, right, bottom = record["cropBoxXyxy"]
    assert left <= 10.8 and top <= 20.2 and right >= 110.8 and bottom >= 120.2
    assert cropped.shape[0] == cropped.shape[1] == 101


def testLandmarksOutsideBoxAreRetainedAndFlagged():
    cropped, record = cropFaceSquare(np.zeros((120, 160, 3), np.uint8),
                                     detection((20, 20, 70, 70), ((10, 10), (80, 80))), 1.35)
    assert "landmarksExpandDetectorBox" in record["qualityFlags"]
    left, top, right, bottom = record["cropBoxXyxy"]
    assert left <= 10 and top <= 10 and right > 80 and bottom > 80
    assert cropped.shape[0] == cropped.shape[1]


def testLandmarkExactlyAtExclusiveBoxEdgeRemainsInsideCrop():
    _, record = cropFaceSquare(np.zeros((120, 160, 3), np.uint8),
                               detection((20, 20, 70, 70), ((20, 20), (70, 70))), 1)
    assert record["cropBoxXyxy"][2] > 70
    assert record["cropBoxXyxy"][3] > 70


def testValidPixelMaskExcludesEdgeReplicas(tmp_path):
    path = tmp_path / "padded.png"
    Image.new("RGB", (100, 160), (128, 128, 128)).save(path)
    _, record = readImage(path, imageSize=100, cropPolicy="faceSquare", cropMargin=1,
                          faceDetector=lambda rgb: detection((-10, 0, 90, 100)))
    assert record["cropBoxXyxy"] == [-10, 0, 90, 100]
    mask = workingImageValidMask(record)
    expected = np.zeros((100, 100), dtype=bool)
    expected[:, 10:] = True
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == bool


def testValidPixelMaskUsesHalfPixelMappingAndSupportsRectangularOutput():
    record = {"orientedSize": [2, 3], "workingSize": [4, 6],
              "inputToWorking": [[2, 0, .5], [0, 2, .5], [0, 0, 1]]}
    expected = np.zeros((6, 4), dtype=bool)
    expected[1:5, 1:3] = True
    np.testing.assert_array_equal(workingImageValidMask(record), expected)
    record["inputToWorking"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="invertible"):
        workingImageValidMask(record)


@pytest.mark.parametrize("face, message", [
    (None, "dictionary"),
    ({}, "requires"),
    (detection((0, 0, np.nan, 70)), "finite"),
    (detection((10, 10, 9, 70)), "positive width"),
    (detection(landmarks=((np.inf, 5),)), "finite Nx2"),
    (detection(landmarks=()), "finite Nx2"),
    (detection(score=np.nan), "score"),
    (detection(score=1.1), "score"),
    (detection((200, 200, 220, 220)), "overlap"),
    (detection((-1e8, -1e8, 1e8, 1e8)), "exceeds"),
])
def testInvalidDetectionFailsExplicitly(face, message):
    with pytest.raises(ValueError, match=message):
        cropFaceSquare(np.zeros((120, 160, 3), np.uint8), face)


@pytest.mark.parametrize("margin", [0.5, 0, -1, np.nan, np.inf, True, "1.35"])
def testCropMarginRejectsAmbiguousOrTruncatingValues(margin):
    with pytest.raises(ValueError, match="cropMargin"):
        cropFaceSquare(np.zeros((120, 160, 3), np.uint8), detection(), margin)


def testPipelineDefaultsToFaceCropButRetainsExplicitCenterAndSerialization(tmp_path):
    config = PipelineConfig(tmp_path)
    assert config.inputCropPolicy == "faceSquare"
    assert config.faceCropMargin == 1.35
    center = PipelineConfig(tmp_path, inputCropPolicy="centerSquare", faceCropMargin=1)
    configPath = tmp_path / "config.json"
    configPath.write_text(json.dumps(center.toDict()))
    assert PipelineConfig.fromFile(configPath) == center
    for values in ({"inputCropPolicy": "auto"}, {"faceCropMargin": 0.8}, {"faceCropMargin": float("nan")},
                   {"faceCropMargin": True}, {"faceCropMargin": "1.35"}):
        with pytest.raises(ValueError):
            PipelineConfig(tmp_path, **values)
