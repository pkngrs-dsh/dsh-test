"""CPU checks for frame preservation and rejection; no model weights needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

from faceAppearance.adapters.delight import (
    DelightAdapter, computeAlignment, loadArchitectureModule, restoreToInput,
)


def makeLandmarks():
    landmarks = np.full((68, 2), 256, dtype=np.float32)
    landmarks[36:42] = [206, 206]
    landmarks[42:48] = [306, 206]
    landmarks[48] = [216, 316]
    landmarks[54] = [296, 316]
    return landmarks


def testAlignmentInvertsPixelCoordinates():
    inputToAligned = computeAlignment(makeLandmarks())
    points = np.array([[123, 255, 1], [420, 180, 1], [255, 388, 1]], np.float64).T
    aligned = inputToAligned @ points
    recovered = np.linalg.inv(inputToAligned) @ aligned
    np.testing.assert_allclose(recovered / recovered[2], points, atol=1e-8)


def testIdentityRestorationPreservesRgbAndMask():
    image = np.zeros((512, 512, 3), np.float32)
    image[:] = [0.1, 0.4, 0.9]
    mask = np.zeros((512, 512), np.float32)
    mask[40:460, 70:420] = 1
    restored, validMask = restoreToInput(image, mask, mask, np.eye(3))
    np.testing.assert_array_equal(validMask, mask.astype(bool))
    np.testing.assert_allclose(restored[validMask], image[validMask], atol=1e-7)
    assert np.count_nonzero(restored[~validMask]) == 0


def testTranslatedInverseWarpDoesNotShiftFinalOutput():
    import cv2

    image = np.zeros((512, 512, 3), np.float32)
    image[100:300, 130:310] = [0.2, 0.6, 0.8]
    mask = image.sum(axis=-1) > 0
    inputToAligned = np.array([[1, 0, 31], [0, 1, -17], [0, 0, 1]], np.float64)
    aligned = cv2.warpPerspective(image, inputToAligned, (512, 512))
    alignedMask = cv2.warpPerspective(mask.astype(np.float32), inputToAligned, (512, 512))
    restored, validMask = restoreToInput(aligned, alignedMask, mask, inputToAligned)
    np.testing.assert_allclose(restored, image, atol=1e-7)
    np.testing.assert_array_equal(validMask, mask)


def testDegenerateLandmarksFailClearly():
    with pytest.raises(ValueError, match="Degenerate"):
        computeAlignment(np.zeros((68, 2), np.float32))


def testInvalidInputFrameAndRangesFailBeforeLoading(tmp_path):
    adapter = DelightAdapter(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="512"):
        adapter.predict(np.zeros((1, 518, 518, 3)), np.zeros((1, 518, 518)))
    with pytest.raises(ValueError, match="finite"):
        adapter.predict(np.full((1, 512, 512, 3), np.nan), np.ones((1, 512, 512)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adapter.predict(np.full((1, 512, 512, 3), 255), np.ones((1, 512, 512)))


def testPredictRequiresExplicitPreload(tmp_path):
    adapter = DelightAdapter(tmp_path, device="cpu")
    with pytest.raises(RuntimeError, match="loadModels"):
        adapter.predict(np.zeros((1, 512, 512, 3)), np.ones((1, 512, 512)))


def testSmallFaceMaskIsReportedWithoutDetectorCall(tmp_path):
    adapter = DelightAdapter(tmp_path, device="cpu")
    albedo, validMask, diagnostic = adapter.predictView(
        np.ones((512, 512, 3), np.float32), np.zeros((512, 512), np.float32))
    assert diagnostic["status"] == "rejected"
    assert diagnostic["reason"] == "insufficientInputFaceCoverage"
    assert not validMask.any()
    assert not albedo.any()


def testNoFaceDetectionIsReportedWithZeroSupport(tmp_path):
    adapter = DelightAdapter(tmp_path, device="cpu")
    adapter.faceDetector = lambda image, rgb: np.empty((0, 15), np.float32)
    albedo, validMask, diagnostic = adapter.predictView(
        np.ones((512, 512, 3), np.float32), np.ones((512, 512), np.float32))
    assert diagnostic["reason"] == "ibugFaceNotDetected"
    assert not validMask.any()
    assert not albedo.any()


def testPredictViewReturnsOriginalFrameWithExplicitTransforms(tmp_path, monkeypatch):
    adapter = DelightAdapter(tmp_path, device="cpu")
    adapter.faceDetector = lambda image, rgb: np.array([[100, 100, 410, 410, 0.99]])
    adapter.landmarkDetector = lambda image, faces, rgb: (
        makeLandmarks()[None], np.ones((1, 68), np.float32))
    monkeypatch.setattr(adapter, "runNetwork", lambda image, mask: image)
    image = np.zeros((512, 512, 3), np.float32)
    image[:] = [0.2, 0.5, 0.8]
    mask = np.zeros((512, 512), np.float32)
    mask[130:360, 150:365] = 1
    albedo, validMask, diagnostic = adapter.predictView(image, mask)
    assert diagnostic["status"] == "ok"
    assert validMask.sum() > 0.99 * mask.sum()
    np.testing.assert_allclose(albedo[validMask], image[validMask], atol=1e-5)
    np.testing.assert_allclose(np.array(diagnostic["alignedToInput"])
                               @ np.array(diagnostic["inputToAligned"]), np.eye(3), atol=1e-8)
    assert np.all(albedo[mask == 0] == 0)


def testMissingAssetsNeverAutoDownload(tmp_path):
    with pytest.raises(FileNotFoundError, match="2dfan2_alt.pth"):
        DelightAdapter(tmp_path, device="cpu").loadModels()


def testArchitectureLoaderDoesNotPolluteSearchPath(tmp_path):
    sourceFile = tmp_path / "architecture.py"
    sourceFile.write_text(
        "import sys\nsys.path.append('UNWANTED')\n"
        "from utils import count_parameters\nanswer = 42\n", encoding="utf-8")
    originalPath = list(sys.path)
    module = loadArchitectureModule(sourceFile)
    assert module.answer == 42
    assert sys.path == originalPath
    assert module.__name__.startswith("faceAppearanceOpenDelight_")


def testRejectedViewKeepsBatchOrderAndStagedOffload(tmp_path, monkeypatch):
    adapter = DelightAdapter(tmp_path, device="cpu")
    adapter.loaded = True
    moves = []
    monkeypatch.setattr(adapter, "moveModels", moves.append)
    masks = np.ones((2, 512, 512), np.float32)
    masks[0] = 0

    def fakePredictView(image, mask):
        rejected = not mask.any()
        return image.copy(), mask.astype(bool), {"status": "rejected" if rejected else "ok"}

    monkeypatch.setattr(adapter, "predictView", fakePredictView)
    result = adapter.predict(np.zeros((2, 512, 512, 3), np.float32), masks)
    assert [item["viewIndex"] for item in result["diagnostics"]] == [0, 1]
    assert [item["status"] for item in result["diagnostics"]] == ["rejected", "ok"]
    assert moves == ["cpu", "cpu"]


def testStagedOffloadRunsAfterFailure(tmp_path, monkeypatch):
    adapter = DelightAdapter(tmp_path, device="cpu")
    adapter.loaded = True
    moves = []
    monkeypatch.setattr(adapter, "moveModels", moves.append)

    def failPredictView(image, mask):
        raise RuntimeError("network failure")

    monkeypatch.setattr(adapter, "predictView", failPredictView)
    with pytest.raises(RuntimeError, match="network failure"):
        adapter.predict(np.zeros((1, 512, 512, 3), np.float32), np.ones((1, 512, 512)))
    assert moves == ["cpu", "cpu"]
