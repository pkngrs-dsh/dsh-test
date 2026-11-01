"""Service contracts using fake CPU pipelines, without loading model weights."""

import io
import json
import threading
import time
from pathlib import Path

import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from faceAppearance.config import PipelineConfig
from faceAppearance.service import createApp


def makeUploads(count=4):
    files = []
    for imageIndex in range(count):
        stream = io.BytesIO()
        Image.new("RGB", (80, 90), (20 + imageIndex * 7, 100, 150)).save(stream, format="PNG")
        files.append(("images", (f"view{imageIndex}.png", stream.getvalue(), "image/png")))
    return files


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.loaded = False
        self.loadCount = 0
        self.runCount = 0
        self.activeCount = 0
        self.maxActiveCount = 0
        self.fail = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def loadModels(self):
        self.loadCount += 1
        self.loaded = True

    def run(self, paths, outputDir, stopAfter="fusion"):
        assert self.loaded
        assert all(Path(path).is_file() for path in paths)
        self.runCount += 1
        self.activeCount += 1
        self.maxActiveCount = max(self.maxActiveCount, self.activeCount)
        self.started.set()
        try:
            if not self.release.wait(timeout=10):
                raise RuntimeError("Test worker timed out")
            if self.fail:
                raise RuntimeError("Geometry alignment could not be validated")
            Path(outputDir).mkdir(parents=True)
            (Path(outputDir) / "result.json").write_text(json.dumps({"stopAfter": stopAfter}))
            return {"viewCount": len(paths), "stopAfter": stopAfter}
        finally:
            self.activeCount -= 1


def waitForJob(client, jobId, expected="succeeded"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{jobId}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] == expected:
            return record
        if record["status"] in {"succeeded", "failed"}:
            pytest.fail(f"Expected {expected}, got {record}")
        time.sleep(0.01)
    pytest.fail(f"Job {jobId} did not reach {expected}")


def testLifecycleLoadsOnceAndPersistsOutput(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    pipeline = FakePipeline(config)
    app = createApp(config, pipelineFactory=lambda config: pipeline)
    assert pipeline.loadCount == 0
    with TestClient(app) as client:
        assert client.get("/health").json()["modelsLoaded"]
        for stopAfter in ("geometry", "delight"):
            response = client.post("/jobs", files=makeUploads(), data={"stopAfter": stopAfter})
            assert response.status_code == 202
            jobId = response.json()["jobId"]
            job = waitForJob(client, jobId)
            assert job["result"]["stopAfter"] == stopAfter
            assert job["artifacts"][0]["filename"] == "result.json"
            download = client.get(job["artifacts"][0]["url"])
            assert download.json()["stopAfter"] == stopAfter
            saved = json.loads((tmp_path / "runs/service" / jobId / "job.json").read_text())
            assert saved["status"] == "succeeded"
        assert pipeline.loadCount == 1
    assert not app.state.jobQueue.worker.is_alive()


def testActualCameraArrayContractCanBeReturnedAndPersisted(tmp_path):
    class ArrayPipeline(FakePipeline):
        def run(self, paths, outputDir, stopAfter="fusion"):
            result = super().run(paths, outputDir, stopAfter)
            result["cameraDiagnostics"] = [{"rotation": np.eye(3), "valid": np.bool_(True),
                                            "medianErrorPixels": np.float32(2.5)}]
            return result

    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with TestClient(createApp(config, pipelineFactory=ArrayPipeline)) as client:
        jobId = client.post("/jobs", files=makeUploads()).json()["jobId"]
        result = waitForJob(client, jobId)["result"]
        assert result["cameraDiagnostics"][0]["rotation"] == np.eye(3).tolist()
        assert result["cameraDiagnostics"][0]["valid"] is True


def testConfiguredWildCapStageCanBeQueued(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableWildCap=True)
    with TestClient(createApp(config, pipelineFactory=FakePipeline)) as client:
        response = client.post("/jobs", files=makeUploads(), data={"stopAfter": "wildcap"})
        assert response.status_code == 202
        result = waitForJob(client, response.json()["jobId"])["result"]
        assert result["stopAfter"] == "wildcap"


@pytest.mark.parametrize("enabled,preloaded", [(False, False), (False, True), (True, False)])
def testAssemblyRequiresEnabledAndPreloadedAssetsBeforeQueue(tmp_path, enabled, preloaded):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableAssembly=enabled)
    pipeline = FakePipeline(config)
    pipeline.assemblyAssets = object() if preloaded else None
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        response = client.post("/jobs", files=makeUploads(), data={"stopAfter": "assembly"})
        assert response.status_code == 422
        assert "preloaded" in response.json()["detail"]
        assert pipeline.runCount == 0
        assert not list((tmp_path / "runs/service").glob(".incoming-*"))


def testAssemblyExportsDownloadableNestedAssetsAndKeepsDefaultFusion(tmp_path):
    class AssemblyPipeline(FakePipeline):
        def loadModels(self):
            super().loadModels()
            self.assemblyAssets = object()

        def run(self, paths, outputDir, stopAfter="fusion"):
            result = super().run(paths, outputDir, stopAfter)
            if stopAfter == "assembly":
                assemblyDir = Path(outputDir) / "assembly"
                assemblyDir.mkdir()
                (assemblyDir / "manifest.json").write_text(json.dumps({"status": "exported"}))
                np.savez(assemblyDir / "legacyMesh.npz", vertices=np.eye(3, dtype=np.float32))
            return result

    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableAssembly=True)
    pipeline = AssemblyPipeline(config)
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        response = client.post("/jobs", files=makeUploads(), data={"stopAfter": "assembly"})
        assert response.status_code == 202
        job = waitForJob(client, response.json()["jobId"])
        assert job["result"]["stopAfter"] == "assembly"
        artifacts = {entry["filename"]: entry["url"] for entry in job["artifacts"]}
        assert client.get(artifacts["assembly/manifest.json"]).json() == {"status": "exported"}
        downloadedMesh = client.get(artifacts["assembly/legacyMesh.npz"])
        assert downloadedMesh.status_code == 200
        with np.load(io.BytesIO(downloadedMesh.content), allow_pickle=False) as mesh:
            np.testing.assert_array_equal(mesh["vertices"], np.eye(3))
        defaultResponse = client.post("/jobs", files=makeUploads())
        defaultJob = waitForJob(client, defaultResponse.json()["jobId"])
        assert defaultJob["result"]["stopAfter"] == "fusion"
        assert all(not entry["filename"].startswith("assembly/") for entry in defaultJob["artifacts"])
        assert pipeline.loadCount == 1


@pytest.mark.parametrize("wildCapPreloaded", [False, True])
def testAssemblyWildCapSourceRequiresPreloadedWildCap(tmp_path, wildCapPreloaded):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableAssembly=True,
                            enableWildCap=True, assemblySourceStage="wildcap")
    pipeline = FakePipeline(config)
    pipeline.assemblyAssets = object()
    pipeline.wildCapAdapter = object() if wildCapPreloaded else None
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        response = client.post("/jobs", files=makeUploads(), data={"stopAfter": "assembly"})
        if wildCapPreloaded:
            assert response.status_code == 202
            assert waitForJob(client, response.json()["jobId"])["result"]["stopAfter"] == "assembly"
        else:
            assert response.status_code == 422
            assert "WildCap" in response.json()["detail"]
            assert pipeline.runCount == 0


def testQueueIsBoundedAndExecutionIsSerial(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    pipeline = FakePipeline(config)
    pipeline.release.clear()
    app = createApp(config, pipelineFactory=lambda config: pipeline, queueCapacity=1)
    with TestClient(app) as client:
        try:
            first = client.post("/jobs", files=makeUploads()).json()["jobId"]
            assert pipeline.started.wait(timeout=5)
            assert client.get(f"/jobs/{first}").json()["status"] == "running"
            secondResponse = client.post("/jobs", files=makeUploads())
            assert secondResponse.status_code == 202
            second = secondResponse.json()["jobId"]
            assert client.get(f"/jobs/{second}").json()["status"] == "queued"
            rejected = client.post("/jobs", files=makeUploads())
            assert rejected.status_code == 429
            assert client.get("/health").json()["queueDepth"] == 1
            assert client.get(f"/jobs/{first}/artifacts/result.json").status_code == 409
        finally:
            pipeline.release.set()
        waitForJob(client, first)
        waitForJob(client, second)
        assert pipeline.maxActiveCount == 1


def testFailureHasMessageAndLocalTracebackThenQueueContinues(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    pipeline = FakePipeline(config)
    pipeline.fail = True
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        jobId = client.post("/jobs", files=makeUploads()).json()["jobId"]
        job = waitForJob(client, jobId, expected="failed")
        assert job["error"]["message"] == "Geometry alignment could not be validated"
        assert "Traceback" not in json.dumps(job)
        tracePath = tmp_path / "runs/service" / jobId / "traceback.txt"
        assert "Traceback" in tracePath.read_text()
        assert client.get(f"/jobs/{jobId}/artifacts/traceback.txt").status_code != 200
        pipeline.fail = False
        nextJob = client.post("/jobs", files=makeUploads()).json()["jobId"]
        waitForJob(client, nextJob)


@pytest.mark.parametrize("count", [3, 17])
def testWrongImageCountRejected(tmp_path, count):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with TestClient(createApp(config, pipelineFactory=FakePipeline)) as client:
        response = client.post("/jobs", files=makeUploads(count))
        assert response.status_code == 422
        assert "Expected 4–16" in response.json()["detail"]


def testDuplicateCorruptAndInvalidStageRejectedBeforeQueue(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    pipeline = FakePipeline(config)
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        files = makeUploads()
        files[1] = files[0]
        response = client.post("/jobs", files=files)
        assert response.status_code == 422
        assert "duplicates" in response.json()["detail"]
        files = makeUploads()
        files[2] = ("images", ("broken.png", b"broken image data", "image/png"))
        response = client.post("/jobs", files=files)
        assert response.status_code == 422
        assert "Invalid image set" in response.json()["detail"]
        assert client.post("/jobs", files=makeUploads(), data={"stopAfter": "wildcap"}).status_code == 422
        assert pipeline.runCount == 0
        assert not list((tmp_path / "runs/service").glob(".incoming-*"))


def testUploadLimitsAreEnforcedBeforeQueue(tmp_path, monkeypatch):
    import faceAppearance.service as serviceModule

    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with TestClient(createApp(config, pipelineFactory=FakePipeline)) as client:
        monkeypatch.setattr(serviceModule, "maxImageBytes", 5)
        assert client.post("/jobs", files=makeUploads()).status_code == 413
        monkeypatch.setattr(serviceModule, "maxImageBytes", 10000)
        monkeypatch.setattr(serviceModule, "maxTotalBytes", 5)
        assert client.post("/jobs", files=makeUploads()).status_code == 413


def testArtifactPathsCannotEscapeOutputFolder(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with TestClient(createApp(config, pipelineFactory=FakePipeline)) as client:
        jobId = client.post("/jobs", files=makeUploads()).json()["jobId"]
        waitForJob(client, jobId)
        assert client.get("/jobs/unknown").status_code == 404
        assert client.get("/jobs/unknown/artifacts/result.json").status_code == 404
        assert client.get(f"/jobs/{jobId}/artifacts/%2E%2E%2Fjob.json").status_code == 404
        outputDir = tmp_path / "runs/service" / jobId / "output"
        secretPath = tmp_path / "private.txt"
        secretPath.write_text("private")
        (outputDir / "escape.txt").symlink_to(secretPath)
        assert client.get(f"/jobs/{jobId}/artifacts/escape.txt").status_code == 404
        assert client.get(f"/jobs/{jobId}/artifacts/absent.txt").status_code == 404


def testStartupFailureIsNotReportedAsHealthy(tmp_path):
    class BrokenPipeline(FakePipeline):
        def loadModels(self):
            raise RuntimeError("Missing model weights")

    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with pytest.raises(RuntimeError, match="Missing model weights"):
        with TestClient(createApp(config, pipelineFactory=BrokenPipeline)):
            pytest.fail("Startup should fail")


def testShutdownDrainsAcceptedJobs(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    pipeline = FakePipeline(config)
    pipeline.release.clear()
    app = createApp(config, pipelineFactory=lambda config: pipeline)
    with TestClient(app) as client:
        jobIds = [client.post("/jobs", files=makeUploads()).json()["jobId"] for _ in range(2)]
        assert pipeline.started.wait(timeout=5)
        assert pipeline.runCount == 1
        releaseTimer = threading.Timer(0.1, pipeline.release.set)
        releaseTimer.start()
        # Exiting lifespan must wait for both the running and queued jobs.
    releaseTimer.join()
    assert pipeline.runCount == 2
    assert all(app.state.jobQueue.get(jobId)["status"] == "succeeded" for jobId in jobIds)
    assert not app.state.jobQueue.worker.is_alive()
