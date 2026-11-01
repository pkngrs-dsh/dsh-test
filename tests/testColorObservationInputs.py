"""Original-photo color contracts at the two geometric observation entry points."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
import pytest

import faceAppearance.clientGeometryRefinement as refinement
import faceAppearance.renderedJawRuntime as jaw
from faceAppearance.data import readImage
from faceAppearance.imageColor import readSrgbImage


PROFILE = Path(__file__).with_name("fixtures") / "displayP3.icc"
FACE = {"boxXyxy": [12, 12, 80, 68],
        "landmarks": [[30, 30], [65, 30], [48, 42], [34, 55], [62, 55]],
        "score": .99}


def _photo(tmp_path, *, tagged, working=False, invalid=False):
    height, width = (64, 64) if working else (96, 80)
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([130 + x // 2, 70 + y // 3, 45 + x // 4], axis=-1).astype(np.uint8)
    source = Image.fromarray(pixels)
    kwargs = {}
    if not working:
        exif = source.getexif()
        exif[274] = 6
        kwargs["exif"] = exif
    if tagged or invalid:
        kwargs["icc_profile"] = b"invalid ICC profile" if invalid else PROFILE.read_bytes()
    path = tmp_path / ("working.png" if working else "original.png")
    source.save(path, **kwargs)
    with Image.open(path) as saved:
        oriented = ImageOps.exif_transpose(saved).convert("RGB")
        raw = np.asarray(oriented).copy()
        if tagged and not invalid:
            converted = ImageCms.profileToProfile(
                oriented, ImageCms.ImageCmsProfile(io.BytesIO(PROFILE.read_bytes())),
                ImageCms.createProfile("sRGB"), outputMode="RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, flags=0)
            expected = np.asarray(converted).copy()
            assert not np.array_equal(expected, raw)
        else:
            expected = raw
    return path, expected


def _record(path):
    _, record = readImage(path, imageSize=64, cropPolicy="faceSquare",
                          faceDetector=lambda _: FACE)
    return record


class _StfrDetector:
    def __init__(self):
        self.images = []

    def infer(self, image, *, faceAnchors):
        self.images.append(np.array(image, copy=True))
        return {"faceLandmarks": np.tile([48., 40.], (68, 1)),
                "faceRawPeakScores": np.ones(68), "metadata": {"backend": "test"}}


class _FanDetector:
    config = SimpleNamespace(crop_ratio=.8, input_size=256)
    faceAppearanceWeightPath = "test-no-model"
    faceAppearanceWeightSha256 = "test-no-model"

    def __init__(self):
        self.images = []

    def __call__(self, image, boxes, *, rgb):
        assert rgb is True
        self.images.append(np.array(image, copy=True))
        return np.tile([48., 40.], (3, 68, 1)), np.ones((3, 68))


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "displayP3"])
def testRefinementOriginalFeedsOrientedSrgbAndPreservesSourceIdentity(tmp_path, monkeypatch, tagged):
    path, expected = _photo(tmp_path, tagged=tagged)
    record = _record(path)
    detector = _StfrDetector()
    monkeypatch.setattr(refinement, "_adapter", lambda _: detector)
    monkeypatch.setattr(refinement, "mouthObservationQuality", lambda *args: {"test": True})
    observations, reports = refinement._observations(tmp_path, [path], None, [record], [{}])
    np.testing.assert_array_equal(detector.images[0], expected)
    np.testing.assert_array_equal(observations[0]["workingRgb"], cv2.warpPerspective(
        expected, np.asarray(record["inputToWorking"]), tuple(record["workingSize"])))
    assert reports[0]["sourceSha256"] == record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reports[0]["colorManagement"] == readSrgbImage(path)[1]
    assert reports[0]["source"] == "originalPhotograph"


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "displayP3"])
def testRefinementWorkingPhotoUsesOnlyItsOwnProfile(tmp_path, monkeypatch, tagged):
    path, expected = _photo(tmp_path, tagged=tagged, working=True)
    record = {"workingSize": [64, 64], "inputToWorking": np.eye(3).tolist(),
              "faceDetection": FACE, "colorManagement": {"sourceProfile": "unrelated original profile"}}
    detector = _StfrDetector()
    monkeypatch.setattr(refinement, "_adapter", lambda _: detector)
    monkeypatch.setattr(refinement, "mouthObservationQuality", lambda *args: {"test": True})
    observations, reports = refinement._observations(tmp_path, [None], [path], [record], [{}])
    np.testing.assert_array_equal(detector.images[0], expected)
    np.testing.assert_array_equal(observations[0]["workingRgb"], expected)
    assert reports[0]["colorManagement"] == readSrgbImage(path)[1]
    assert reports[0]["colorManagement"] != record["colorManagement"]
    assert reports[0]["source"] == "providedWorkingPhotograph"


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "displayP3"])
def testRenderedJawFeedsOrientedSrgbWithRealCropReplay(tmp_path, monkeypatch, tagged):
    path, expected = _photo(tmp_path, tagged=tagged)
    record = _record(path)
    detector = _FanDetector()
    monkeypatch.setattr(jaw, "getDetector", lambda _: detector)
    result = jaw.observePhotos(tmp_path, [path], [record])
    np.testing.assert_array_equal(detector.images[0], expected)
    assert result["acceptedViewMask"].tolist() == [True]
    view = result["diagnostics"]["views"][0]
    assert view["photoSha256"] == record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert view["colorManagement"] == readSrgbImage(path)[1]
    assert view["inputToWorking"] == record["inputToWorking"]


@pytest.mark.parametrize("entrypoint", ["refinementOriginal", "refinementWorking", "renderedJaw"])
def testInvalidIccPropagatesBeforeObservationInference(tmp_path, monkeypatch, entrypoint):
    path, _ = _photo(tmp_path, tagged=False, invalid=True,
                     working=entrypoint == "refinementWorking")
    record = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "orientedSize": [64, 64] if entrypoint == "refinementWorking" else [96, 80],
              "workingSize": [64, 64], "inputToWorking": np.eye(3).tolist(),
              "workingToInput": np.eye(3).tolist(), "faceDetection": FACE}
    detector = _FanDetector() if entrypoint == "renderedJaw" else _StfrDetector()
    if entrypoint == "renderedJaw":
        monkeypatch.setattr(jaw, "getDetector", lambda _: detector)
        with pytest.raises(ValueError):
            jaw.observePhotos(tmp_path, [path], [record])
    else:
        monkeypatch.setattr(refinement, "_adapter", lambda _: detector)
        with pytest.raises(ValueError):
            refinement._observations(tmp_path,
                [None] if entrypoint == "refinementWorking" else [path],
                [path] if entrypoint == "refinementWorking" else None, [record], [{}])
    assert not detector.images
