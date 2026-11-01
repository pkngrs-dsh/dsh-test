"""Persistent queue contracts exercised with a small CPU pipeline."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import faceAppearance.jobs as jobsModule
from faceAppearance.jobs import JobQueue, timestamp, writeJobRecord


class CpuPipeline:
    def __init__(self, root):
        self.loaded = True
        self.config = SimpleNamespace(projectRoot=root)
        self.runCount = 0
        self.fail = False

    def run(self, paths, outputDir, stopAfter):
        self.runCount += 1
        if self.fail:
            raise RuntimeError("Failed to reconstruct supplied images")
        outputDir.mkdir()
        (outputDir / "result.json").write_text('{"complete": true}', encoding="utf-8")
        return {"viewCount": len(paths), "stopAfter": stopAfter}


def incoming(root, jobId):
    directory = root / f".incoming-{jobId}"
    (directory / "inputs").mkdir(parents=True)
    (directory / "inputs/view.image").write_bytes(b"image")
    return directory


def savedJob(root, index, status="succeeded"):
    jobId = f"{index:032x}"
    directory = root / jobId
    directory.mkdir(parents=True)
    record = {"jobId": jobId, "status": status, "stopAfter": "fusion",
              "createdAt": timestamp(), "startedAt": timestamp(), "finishedAt": timestamp(),
              "numImages": 1, "inputNames": ["view.png"],
              "result": {"viewCount": 1}, "artifacts": [], "error": None}
    writeJobRecord(directory, record)
    return record


def clientOutput(outputDir):
    """Export a complete client archive from a tiny CPU mesh fixture."""
    from faceAppearance.clientAsset import exportClientAsset
    from faceAppearance.clientContract import validateClientOutput

    template = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.]], np.float32)
    vertices = template.copy()
    vertices[2, 2] = 2.
    asset = {"geometry": {
        "vertices": vertices, "faces": np.array([[0, 1, 2]], np.int32),
        "uvs": np.array([[0., 0.], [1., 0.], [0., 1.]], np.float32),
        "uvFaces": np.array([[0, 1, 2]], np.int32), "materialIds": np.array([0], np.uint8)},
        "templateVertices": template, "templateId": "fixture-template",
        "templateHashes": {"originalTemplateObj": "a" * 64},
        "materials": [{"name": "skin", "baseColor": np.full((2, 2, 3), .25, np.float32),
                       "roughness": .6, "metallic": 0., "alphaMode": "OPAQUE"}],
        "components": [], "diagnostics": {"fixture": True}}
    exportClientAsset(outputDir / "assembly", asset)
    return validateClientOutput(outputDir)


def testRecoveryRetainsTerminalJobsAndFailsInterruptedWithoutReplaying(tmp_path):
    records = [savedJob(tmp_path, index, status)
               for index, status in enumerate(("succeeded", "failed", "queued", "running"), 1)]
    pipeline = CpuPipeline(tmp_path)
    queue = JobQueue(pipeline, tmp_path, 2)
    queue.start()
    try:
        assert queue.get(records[0]["jobId"])["result"] == {"viewCount": 1}
        assert queue.get(records[1]["jobId"])["status"] == "failed"
        for original in records[2:]:
            recovered = queue.get(original["jobId"])
            assert recovered["status"] == "failed"
            assert recovered["error"]["code"] == "service_restarted"
            assert recovered["result"] is None
            assert recovered["artifacts"] == []
            assert recovered["clientAsset"] is None
            assert json.loads((tmp_path / original["jobId"] / "job.json").read_text()) == recovered
        assert pipeline.runCount == 0
        assert queue.pending.empty()
    finally:
        queue.close()


def testRecoveryRejectsCorruptRecordsUnsafePathsAndSymlinks(tmp_path):
    root = tmp_path / "jobs"
    good = savedJob(root, 1)
    mismatch = savedJob(root, 2)
    mismatch["jobId"] = "../escape"
    writeJobRecord(root / f"{2:032x}", mismatch)
    corrupt = savedJob(root, 3)
    (root / corrupt["jobId"] / "job.json").write_text("{bad JSON")
    unsafe = savedJob(root, 4)
    unsafe["artifacts"] = [{"filename": "../job.json", "sizeBytes": 1}]
    writeJobRecord(root / unsafe["jobId"], unsafe)
    external = tmp_path / "outside"
    savedJob(external, 5)
    (root / f"{5:032x}").symlink_to(external / f"{5:032x}", target_is_directory=True)
    linkedRecord = root / f"{6:032x}"
    linkedRecord.mkdir()
    (linkedRecord / "job.json").symlink_to(external / f"{5:032x}" / "job.json")
    invalidResult = savedJob(root, 7)
    invalidResult["result"] = ["not a result mapping"]
    writeJobRecord(root / invalidResult["jobId"], invalidResult)
    invalidNumber = savedJob(root, 8)
    invalidNumber["result"] = {"value": float("nan")}
    (root / invalidNumber["jobId"] / "job.json").write_text(json.dumps(invalidNumber))
    queue = JobQueue(CpuPipeline(root), root, 1)
    queue.start()
    try:
        assert set(queue.jobs) == {good["jobId"]}
    finally:
        queue.close()


def testOutputOwnershipPrecedesRecoveryAndIsReleasedOnClose(tmp_path):
    record = savedJob(tmp_path, 1, "running")
    first = JobQueue(CpuPipeline(tmp_path), tmp_path, 1)
    second = JobQueue(CpuPipeline(tmp_path), tmp_path, 1)
    first.acquireOwnership()
    first.acquireOwnership()  # Model loading can reserve before start().
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second.start()
        assert json.loads((tmp_path / record["jobId"] / "job.json").read_text())["status"] == "running"
        with pytest.raises(RuntimeError, match="requires service ownership"):
            second.recover()
    finally:
        first.close()
    second.start()
    try:
        assert second.get(record["jobId"])["error"]["code"] == "service_restarted"
        with pytest.raises(RuntimeError, match="worker is running"):
            second.releaseOwnership()
    finally:
        second.close()
        second.close()


def testFailedRecoveryReleasesOwnershipForNextStartup(tmp_path, monkeypatch):
    savedJob(tmp_path, 1, "queued")
    queue = JobQueue(CpuPipeline(tmp_path), tmp_path, 1)
    with monkeypatch.context() as patch:
        patch.setattr(jobsModule, "writeJobRecord", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            queue.start()
    following = JobQueue(CpuPipeline(tmp_path), tmp_path, 1)
    following.start()
    following.close()


def testSubmitWriteFailureDoesNotPublishOrStrandJob(tmp_path, monkeypatch):
    pipeline = CpuPipeline(tmp_path)
    queue = JobQueue(pipeline, tmp_path, 1)
    jobId = "1" * 32
    directory = incoming(tmp_path, jobId)
    queue.start()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(jobsModule, "writeJobRecord", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
            with pytest.raises(OSError, match="disk full"):
                queue.submit(jobId, directory, ["view.png"], "fusion")
        assert queue.get(jobId) is None
        assert queue.pending.empty()
        assert not (tmp_path / jobId).exists()
        assert directory.is_dir()
        assert pipeline.runCount == 0
        queue.submit(jobId, directory, ["view.png"], "fusion")
        queue.pending.join()
        record = queue.get(jobId)
        assert record["status"] == "succeeded"
        artifact = record["artifacts"][0]
        assert artifact["sha256"] == hashlib.sha256(b'{"complete": true}').hexdigest()
        assert artifact["mediaType"] == "application/json"
        assert artifact["url"] == f"/jobs/{jobId}/artifacts/result.json"
    finally:
        queue.close()


def testPersistenceFailureKeepsConsumerAliveAndClearsPublishedResults(tmp_path, monkeypatch):
    pipeline = CpuPipeline(tmp_path)
    queue = JobQueue(pipeline, tmp_path, 1)
    queue.start()
    failingId, followingId = "1" * 32, "2" * 32
    originalWriter = jobsModule.writeJobRecord

    def failTerminalWrites(directory, record):
        if record["jobId"] == failingId and record["status"] in {"succeeded", "failed"}:
            raise OSError("disk full")
        originalWriter(directory, record)

    monkeypatch.setattr(jobsModule, "writeJobRecord", failTerminalWrites)
    try:
        queue.submit(failingId, incoming(tmp_path, failingId), ["view.png"], "fusion")
        queue.pending.join()
        failed = queue.get(failingId)
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "job_storage_error"
        assert failed["result"] is None
        assert failed["artifacts"] == []
        assert queue.activeJobId is None
        queue.submit(followingId, incoming(tmp_path, followingId), ["view.png"], "fusion")
        queue.pending.join()
        assert queue.get(followingId)["status"] == "succeeded"
        assert pipeline.runCount == 2
        assert queue.worker.is_alive()
    finally:
        queue.close()


def testTracebackFailureDoesNotPreventPersistingTerminalFailure(tmp_path, monkeypatch):
    pipeline = CpuPipeline(tmp_path)
    pipeline.fail = True
    queue = JobQueue(pipeline, tmp_path, 1)
    queue.start()
    originalWriteText = Path.write_text

    def failTraceback(path, *args, **kwargs):
        if path.name == "traceback.txt":
            raise OSError("cannot write traceback")
        return originalWriteText(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failTraceback)
    jobId = "1" * 32
    try:
        queue.submit(jobId, incoming(tmp_path, jobId), ["view.png"], "fusion")
        queue.pending.join()
        failed = queue.get(jobId)
        assert failed["error"]["code"] == "reconstruction_failed"
        assert failed["status"] == "failed"
        assert json.loads((tmp_path / jobId / "job.json").read_text()) == failed
    finally:
        queue.close()


def testAtomicRecordFailureKeepsPreviousRecord(tmp_path, monkeypatch):
    writeJobRecord(tmp_path, {"status": "queued"})

    def failReplace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", failReplace)
    with pytest.raises(OSError, match="replace failed"):
        writeJobRecord(tmp_path, {"status": "running"})
    assert json.loads((tmp_path / "job.json").read_text()) == {"status": "queued"}
    assert list(tmp_path.glob(".job-*.tmp")) == []


@pytest.mark.parametrize("damage", [None, "missing_archive", "changed_summary"])
def testRecoveryRevalidatesClientAssetsWithoutUsingCurrentTemplate(tmp_path, damage):
    record = savedJob(tmp_path, 1)
    directory = tmp_path / record["jobId"]
    record["clientContract"] = True
    record["clientAsset"] = clientOutput(directory / "output")
    if damage == "missing_archive":
        (directory / "output/assembly/clientAsset.zip").unlink()
    elif damage == "changed_summary":
        record["clientAsset"]["templateId"] = "wrong-template"
    writeJobRecord(directory, record)
    pipeline = CpuPipeline(tmp_path)
    pipeline.assemblyAssets = SimpleNamespace(manifest={"templateId": "different-current-template"})
    queue = JobQueue(pipeline, tmp_path, 1)
    queue.start()
    try:
        recovered = queue.get(record["jobId"])
        if damage is None:
            assert recovered["status"] == "succeeded"
            assert recovered["clientAsset"]["templateId"] == "fixture-template"
        else:
            assert recovered["status"] == "failed"
            assert recovered["error"]["code"] == "invalid_client_asset"
            assert recovered["artifacts"] == []
            assert recovered["clientAsset"] is None
            assert recovered["result"] is None
            assert json.loads((directory / "job.json").read_text()) == recovered
        assert pipeline.runCount == 0
    finally:
        queue.close()


def testClientJobFailsIfPackageUsesDifferentConfiguredTemplate(tmp_path):
    class WrongTemplatePipeline(CpuPipeline):
        assemblyAssets = SimpleNamespace(manifest={"templateId": "expected-template", "templateHashes": {}})

        def run(self, paths, outputDir, stopAfter):
            result = super().run(paths, outputDir, stopAfter)
            clientOutput(outputDir)
            return result

    queue = JobQueue(WrongTemplatePipeline(tmp_path), tmp_path, 1)
    queue.start()
    jobId = "1" * 32
    try:
        queue.submit(jobId, incoming(tmp_path, jobId), ["view.png"], "assembly", clientContract=True)
        queue.pending.join()
        assert queue.get(jobId)["status"] == "failed"
        assert queue.get(jobId)["error"]["code"] == "invalid_client_asset"
    finally:
        queue.close()
