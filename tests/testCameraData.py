from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from faceAppearance.camera import fitCamera, projectPoints
from faceAppearance.data import loadViews, readImage


def testCameraGeneralizesToWithheldCorrespondences():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(500, 3)) * [0.4, 0.5, 0.1]
    intrinsics = np.array([[600, 0, 255.5], [0, 600, 255.5], [0, 0, 1.]])
    pixels, _ = projectPoints(points, intrinsics, np.eye(3), np.array([0.1, 0.2, 3]))
    pixels += rng.normal(scale=0.3, size=pixels.shape)
    pixels[:70] = rng.uniform(0, 512, (70, 2))
    camera = fitCamera(points, pixels, intrinsics, 3)
    assert camera["valid"]
    assert camera["heldOutMedianErrorPixels"] < 1
    assert np.linalg.norm(camera["translation"] - [0.1, 0.2, 3]) < 0.01


def testUnrelatedPointsDoNotProduceAcceptedCamera():
    rng = np.random.default_rng(5)
    camera = fitCamera(rng.normal(size=(200, 3)), rng.uniform(0, 512, (200, 2)),
                       np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1.]]), 3)
    assert not camera["valid"]


def testIngestionPixelCenterAndDuplicates(tmp_path: Path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (800, 1000), (0, 0, 0)).save(path)
    image, record = readImage(path)
    transform = np.asarray(record["inputToWorking"])
    assert np.allclose(transform @ [399.5, 499.5, 1], [255.5, 255.5, 1])
    assert image.shape == (512, 512, 3)
    assert not image.any()
    with pytest.raises(ValueError, match="Duplicate"):
        loadViews([path] * 4)


def testTruncatedFileIsRejected(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(OSError):
        readImage(path)


def testFaceCropHalfPixelCoordinatesRoundTripWithPadding(tmp_path):
    path = tmp_path / "padded.png"
    Image.new("RGB", (100, 160), (25, 50, 75)).save(path)
    landmarks = np.array([[15., 20.], [90., 25.], [52.5, 90.]])
    face = {"boxXyxy": [-5, 2, 98, 118], "landmarks": landmarks, "score": 0.99}
    image, record = readImage(path, imageSize=512, cropPolicy="faceSquare", faceDetector=lambda rgb: face)
    transform = np.asarray(record["inputToWorking"])
    inverse = np.asarray(record["workingToInput"])
    original = np.column_stack([landmarks, np.ones(len(landmarks))])
    mapped = original @ transform.T
    np.testing.assert_allclose(mapped @ inverse.T, original, atol=1e-12)
    np.testing.assert_allclose(record["faceLandmarksWorking"], mapped[:, :2])
    left, top, right, bottom = record["cropBoxXyxy"]
    # Continuous pixel-center edges of the crop must map exactly to image edges.
    np.testing.assert_allclose(transform @ [left - 0.5, top - 0.5, 1], [-0.5, -0.5, 1], atol=1e-12)
    np.testing.assert_allclose(transform @ [right - 0.5, bottom - 0.5, 1], [511.5, 511.5, 1], atol=1e-12)
    assert any(record["paddingLeftTopRightBottom"])
    np.testing.assert_allclose(image, np.broadcast_to(np.array([25, 50, 75]) / 255, image.shape), atol=1e-7)


def testDetectorReceivesExifOrientedOriginalRgbPixels(tmp_path):
    pixels = np.zeros((120, 80, 3), dtype=np.uint8)
    pixels[:30, :20] = [255, 0, 0]
    source = Image.fromarray(pixels)
    exif = source.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.png"
    source.save(path, exif=exif)
    with Image.open(path) as saved:
        expected = np.asarray(ImageOps.exif_transpose(saved).convert("RGB"))
    observed = []

    def detector(rgb):
        assert rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb, expected)
        observed.append(rgb.shape)
        return {"boxXyxy": [20, 10, 90, 70], "landmarks": [[30, 25], [70, 25]], "score": 0.95}

    _, record = readImage(path, cropPolicy="faceSquare", faceDetector=detector)
    assert observed == [(80, 120, 3)]
    assert record["orientedSize"] == [120, 80]
    assert record["requestedCropPolicy"] == "faceSquare"


def testFaceCropRequiresDetectorAndNeverFallsBack(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (100, 160)).save(path)
    with pytest.raises(ValueError, match="requires a faceDetector"):
        readImage(path, cropPolicy="faceSquare")

    def missingFace(rgb):
        raise ValueError("Expected one face, detected 0")

    with pytest.raises(ValueError, match="detected 0"):
        readImage(path, cropPolicy="faceSquare", faceDetector=missingFace)
    # Cheap validation remains model-free even if an unused detector is supplied.
    _, record = readImage(path, faceDetector=missingFace)
    assert record["cropPolicy"] == "exifTransposeThenCenterSquare"


def testLoadViewsForwardsFaceCropAndPreservesDuplicateValidation(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"photo{index}.png"
        Image.new("RGB", (100, 160), (index * 10, 20, 30)).save(path)
        paths.append(path)
    detected = []

    def detector(rgb):
        detected.append(rgb.shape)
        return {"boxXyxy": [10, 5, 80, 75], "landmarks": [[20, 20], [60, 20]], "score": 0.95}

    images, records = loadViews(paths, cropPolicy="faceSquare", faceDetector=detector, cropMargin=1.2)
    assert images.shape == (4, 512, 512, 3)
    assert len(detected) == 4
    assert all(record["cropMargin"] == 1.2 and record["requestedCropPolicy"] == "faceSquare" for record in records)
    with pytest.raises(ValueError, match="Duplicate"):
        loadViews([paths[0]] * 4, cropPolicy="faceSquare", faceDetector=detector)
