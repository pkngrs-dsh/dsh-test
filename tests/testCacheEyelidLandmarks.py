import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from testEyelidLandmarks import FakeAdapter


spec = importlib.util.spec_from_file_location("cacheEyelidLandmarksTool", Path(__file__).resolve().parents[1]/"tools/cacheEyelidLandmarks.py")
cacheTool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cacheTool)


def makeInputs(tmp_path):
    source, semantics = tmp_path/"source", tmp_path/"semanticCache"
    source.mkdir()
    semantics.mkdir()
    photoPath = tmp_path/"photo.png"
    Image.fromarray(np.full((100, 120, 3), 150, np.uint8)).save(photoPath)
    item = {"path": str(photoPath), "sha256": cacheTool.fileHash(photoPath), "orientedSize": [120, 100],
            "faceDetection": {"landmarks": [[1, 1]]*5}}  # Deliberately not the trusted cache anchors.
    (source/"inputs.json").write_text(json.dumps([item]))
    labels = np.ones((100, 120), np.uint8)
    labels[30:40, 30:40], labels[30:40, 80:90] = 4, 5
    confidence = np.ones((100, 120), np.float32)
    anchors = np.array([[35, 35], [85, 35], [60, 55], [55, 75], [65, 75]], np.float32)
    np.savez_compressed(semantics/"view00.npz", labelMap=labels, confidence=confidence, landmarks=anchors)
    manifest = {"status": "complete", "inputsSha256": cacheTool.fileHash(source/"inputs.json"), "parserName": "celebm",
                "views": [{"viewIndex": 0, "photoSha256": item["sha256"], "imageSizeWh": [120, 100],
                           "cacheSha256": cacheTool.fileHash(semantics/"view00.npz")}]}
    (semantics/"manifest.json").write_text(json.dumps(manifest))
    return source, semantics, anchors


def testCacheUsesValidatedSemanticAnchorsAndRecordsHashes(tmp_path, monkeypatch):
    source, semantics, anchors = makeInputs(tmp_path)
    adapter = FakeAdapter(tmp_path)
    adapter.assetPath().parent.mkdir(parents=True)
    adapter.assetPath().write_bytes(b"fixture model")
    monkeypatch.setattr(cacheTool, "EyelidLandmarksAdapter", lambda projectRoot, minimumPeakScore: adapter)
    output = tmp_path/"output"
    report = cacheTool.cacheEyelidLandmarks(source, semantics, output)
    assert report["status"] == "complete" and report["device"] == "cpu"
    assert report["semanticManifestSha256"] == cacheTool.fileHash(semantics/"manifest.json")
    assert report["modelWeightHashes"][str(adapter.assetPath())] == cacheTool.fileHash(adapter.assetPath())
    assert adapter.backendLoads == 1
    with np.load(output/"view00.npz") as arrays:
        np.testing.assert_array_equal(arrays["semanticFaceAnchors"], anchors)
        np.testing.assert_array_equal(arrays["faceAnchors"], np.concatenate([anchors[:3], anchors[3:].mean(0, keepdims=True)]))
    assert report["views"][0]["cacheSha256"] == cacheTool.fileHash(output/"view00.npz")
    assert report["views"][0]["imageSideToSemanticLabel"] == {"imageLeft": 4, "imageRight": 5}
    assert (output/"view00EyeDetail.png").is_file() and (output/"view00Overlay.png").is_file()
    assert (output/"overview.png").is_file()
    assert (output/"implementation/tools/cacheEyelidLandmarks.py").is_file()


def testProducerNeverOverwritesExistingOutput(tmp_path):
    source, semantics, _ = makeInputs(tmp_path)
    output = tmp_path/"output"
    output.mkdir()
    (output/"sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        cacheTool.cacheEyelidLandmarks(source, semantics, output)
    assert (output/"sentinel").read_text() == "keep"


@pytest.mark.parametrize("mutation", ["photo", "cache", "viewOrder", "inputManifest", "incomplete"])
def testChangedOrMismatchedProvenanceRejectedBeforeOutput(tmp_path, mutation):
    source, semantics, _ = makeInputs(tmp_path)
    manifest = json.loads((semantics/"manifest.json").read_text())
    if mutation == "photo": (tmp_path/"photo.png").write_bytes(b"changed")
    if mutation == "cache": (semantics/"view00.npz").write_bytes(b"changed")
    if mutation == "viewOrder": manifest["views"][0]["viewIndex"] = 1
    if mutation == "inputManifest": manifest["inputsSha256"] = "different"
    if mutation == "incomplete": manifest["status"] = "running"
    (semantics/"manifest.json").write_text(json.dumps(manifest))
    output = tmp_path/"output"
    with pytest.raises(ValueError):
        cacheTool.cacheEyelidLandmarks(source, semantics, output)
    assert not output.exists()


def testUntrustedAnchorGeometryRejectedEvenWithValidCacheHash(tmp_path):
    source, semantics, _ = makeInputs(tmp_path)
    with np.load(semantics/"view00.npz") as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["landmarks"][0, 0] = -1
    np.savez_compressed(semantics/"view00.npz", **arrays)
    manifest = json.loads((semantics/"manifest.json").read_text())
    manifest["views"][0]["cacheSha256"] = cacheTool.fileHash(semantics/"view00.npz")
    (semantics/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="anchors"):
        cacheTool.readValidatedViews(source/"inputs.json", semantics)


def testMissingEyeSemanticRegionRemainsUnassigned():
    labels = np.ones((30, 40), np.uint8)
    confidence = np.ones_like(labels, np.float32)
    result = {"corners": np.array([[[5, 10], [15, 10]], [[25, 10], [35, 10]]])}
    assigned = cacheTool.matchSemanticEyeLabels(result, labels, confidence)
    assert not assigned["imageSideToSemanticLabel"]
    assert assigned["semanticLabelAssignment"]["status"] == "missingSemanticEyeRegion"
