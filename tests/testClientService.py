"""Client HTTP contract using a real asset export and a small CPU pipeline."""

import hashlib
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from faceAppearance.clientAsset import exportClientAsset
from faceAppearance.config import PipelineConfig
from faceAppearance.service import createApp


TEMPLATE_ID = "clientServiceFixtureV1"
TEMPLATE_HASHES = {"originalTemplateObj": "a" * 64}


def uploads(count=4):
    result = []
    for index in range(count):
        content = io.BytesIO()
        Image.new("RGB", (80, 90), (20 + index * 7, 100, 150)).save(content, format="PNG")
        result.append(("images", (f"view{index}.png", content.getvalue(), "image/png")))
    return result


def tinyAssembly():
    template = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.]], np.float32)
    vertices = template.copy()
    vertices[2, 2] = 2.
    return {
        "geometry": {
            "vertices": vertices,
            "faces": np.array([[0, 1, 2]], np.int32),
            "uvs": np.array([[0., 0.], [1., 0.], [0., 1.]], np.float32),
            "uvFaces": np.array([[0, 1, 2]], np.int32),
            "materialIds": np.array([0], np.uint8),
        },
        "templateVertices": template,
        "templateId": TEMPLATE_ID,
        "templateHashes": TEMPLATE_HASHES,
        "materials": [{"name": "skin", "baseColor": np.full((2, 2, 3), .25, np.float32),
                       "roughness": .6, "metallic": 0., "alphaMode": "OPAQUE"}],
        "components": [],
        "diagnostics": {"fixture": True},
    }


class ClientPipeline:
    def __init__(self, config):
        self.config = config
        self.loaded = False
        self.loadCount = 0
        self.runStages = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.damage = None

    def loadModels(self):
        self.loadCount += 1
        self.loaded = True
        self.assemblyAssets = SimpleNamespace(manifest={"schemaVersion": 1,
            "templateId": TEMPLATE_ID, "templateHashes": TEMPLATE_HASHES})

    def run(self, paths, outputDir, stopAfter="fusion"):
        self.runStages.append(stopAfter)
        self.started.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("Test pipeline did not receive release")
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True)
        result = {"status": "succeeded", "stopAfter": stopAfter,
                  "config": self.config.toDict(),
                  "inputs": [{"path": str(path)} for path in paths]}
        if stopAfter == "assembly":
            result["clientAssetManifest"] = exportClientAsset(outputDir / "assembly", tinyAssembly())
            result["assemblySourceStage"] = "fusion"
            result["pendingStages"] = ["clientRigValidation", "validatedNormalConvention"]
            package = outputDir / "assembly/clientAsset.zip"
            if self.damage == "missing":
                package.unlink()
            elif self.damage == "corrupt":
                package.write_bytes(b"not a ZIP file")
        (outputDir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return result


def clientApp(tmp_path, **kwargs):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableAssembly=True)
    pipeline = ClientPipeline(config)
    app = createApp(config, pipelineFactory=lambda config: pipeline, **kwargs)
    return app, pipeline


def waitForClientJob(client, jobId, expected="succeeded"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{jobId}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] in {"succeeded", "failed"}:
            assert record["status"] == expected, record
            return record
        time.sleep(.01)
    pytest.fail(f"Client job {jobId} did not finish")


def assertError(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["apiVersion"] == "1"
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert isinstance(body["error"]["retryable"], bool)
    return body["error"]


def testDecodedPixelLimitIsAnInputError(tmp_path, monkeypatch):
    app, pipeline = clientApp(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with TestClient(app) as client:
        assertError(client.post("/v1/jobs", files=uploads()), 422, "validation_error")
        assert pipeline.runStages == []
        assert not list((tmp_path / "runs/service").glob(".incoming-*"))


def testClientLifecycleReturnsCompatibleAssetAndVerifiedDownload(tmp_path):
    app, pipeline = clientApp(tmp_path)
    assert pipeline.loadCount == 0
    with TestClient(app) as client:
        capability = client.get("/v1/capabilities").json()
        assert capability["apiVersion"] == "1"
        assert capability["ready"] and capability["acceptingJobs"]
        assert capability["template"] == {"templateId": TEMPLATE_ID, "templateHashes": TEMPLATE_HASHES}
        assert capability["limits"]["minImages"] == 4
        assert capability["limits"]["maxImages"] == 16
        assert capability["limits"]["maxImageBytes"] > 0
        assert capability["limits"]["maxTotalBytes"] >= capability["limits"]["maxImageBytes"]
        assert capability["output"] == {"format": "zip", "schemaVersion": 1}
        submission = client.post("/v1/jobs", files=uploads(),
                                 data={"templateId": TEMPLATE_ID, "stopAfter": "geometry"})
        assert submission.status_code == 202, submission.text
        initial = submission.json()
        assert initial["apiVersion"] == "1"
        assert initial["status"] == "queued"
        assert initial["download"] is None and initial["asset"] is None and initial["error"] is None
        assert initial["statusUrl"] == f"/v1/jobs/{initial['jobId']}"
        assert submission.headers["location"] == initial["statusUrl"]
        assert int(submission.headers["retry-after"]) > 0
        job = waitForClientJob(client, initial["jobId"])
        assert job["numImages"] == 4
        assert job["createdAt"] and job["startedAt"] and job["finishedAt"]
        assert isinstance(job["pollAfterSeconds"], (int, float))
        assert job["error"] is None
        assert isinstance(job["warnings"], list)
        assert all(isinstance(warning, str) for warning in job["warnings"])
        assert job["asset"]["schemaVersion"] == 1
        assert job["asset"]["templateId"] == TEMPLATE_ID
        assert job["asset"]["templateHashes"] == TEMPLATE_HASHES
        assert job["asset"]["coordinates"]["exportLengthUnit"] == "meter"
        assert job["asset"]["binding"]["originalBindingIndicesRemainValid"]
        assert job["asset"]["counts"]["vertices"] == 3
        assert str(tmp_path) not in json.dumps(job)
        assert "result" not in job and "config" not in job and "inputs" not in job
        package = client.get(job["download"]["url"])
        assert package.status_code == 200
        assert package.headers["content-type"] == "application/zip"
        assert "attachment" in package.headers["content-disposition"]
        assert job["download"]["sizeBytes"] == len(package.content)
        assert job["download"]["sha256"] == hashlib.sha256(package.content).hexdigest()
        manifest = client.get(job["download"]["manifestUrl"])
        assert manifest.status_code == 200
        assert manifest.json()["templateId"] == TEMPLATE_ID
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            assert json.loads(archive.read("manifest.json")) == manifest.json()
            assert "clientMesh.npz" in archive.namelist()
        assert pipeline.runStages == ["assembly"]
        assert pipeline.loadCount == 1


@pytest.mark.parametrize("invalid", ["count", "duplicate", "corrupt", "missing"])
def testClientInvalidUploadsAreRejectedWithStructuredError(tmp_path, invalid):
    app, pipeline = clientApp(tmp_path)
    with TestClient(app) as client:
        files = uploads(3 if invalid == "count" else 4)
        if invalid == "duplicate":
            files[1] = files[0]
        elif invalid == "corrupt":
            files[1] = ("images", ("bad.png", b"broken", "image/png"))
        elif invalid == "missing":
            files = []
        assertError(client.post("/v1/jobs", files=files), 422, "validation_error")
        assert not pipeline.runStages
        assert not list((tmp_path / "runs/service").glob(".incoming-*"))


def testClientTemplateMismatchFailsBeforePipelineExecution(tmp_path):
    app, pipeline = clientApp(tmp_path)
    with TestClient(app) as client:
        assertError(client.post("/v1/jobs", files=uploads(), data={"templateId": "anotherRig"}),
                    422, "template_mismatch")
        assert not pipeline.runStages


def testClientDisabledAssemblyIsUnavailable(tmp_path):
    config = PipelineConfig(projectRoot=tmp_path, device="cpu", enableAssembly=False)
    pipeline = ClientPipeline(config)
    with TestClient(createApp(config, pipelineFactory=lambda config: pipeline)) as client:
        capability = client.get("/v1/capabilities").json()
        assert not capability["ready"]
        assert not capability["acceptingJobs"]
        assertError(client.post("/v1/jobs", files=uploads()), 503, "client_service_unavailable")
        assert not pipeline.runStages


def testClientQueueBackpressureAndDownloadsWaitForCompletion(tmp_path):
    app, pipeline = clientApp(tmp_path, queueCapacity=1)
    pipeline.release.clear()
    with TestClient(app) as client:
        try:
            first = client.post("/v1/jobs", files=uploads()).json()["jobId"]
            assert pipeline.started.wait(timeout=5)
            response = client.get(f"/v1/jobs/{first}")
            assert response.json()["status"] == "running"
            assert response.json()["download"] is None
            assert int(response.headers["retry-after"]) > 0
            assert client.get(f"/v1/jobs/{first}/download").status_code == 409
            assert client.get(f"/v1/jobs/{first}/manifest").status_code == 409
            secondResponse = client.post("/v1/jobs", files=uploads())
            assert secondResponse.status_code == 202
            second = secondResponse.json()["jobId"]
            rejected = client.post("/v1/jobs", files=uploads())
            error = assertError(rejected, 429, "queue_full")
            assert error["retryable"]
            assert int(rejected.headers["retry-after"]) > 0
        finally:
            pipeline.release.set()
        waitForClientJob(client, first)
        waitForClientJob(client, second)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def testClientMissingOrInvalidPackageCannotSucceed(tmp_path, damage):
    app, pipeline = clientApp(tmp_path)
    pipeline.damage = damage
    with TestClient(app) as client:
        response = client.post("/v1/jobs", files=uploads())
        assert response.status_code == 202
        jobId = response.json()["jobId"]
        failed = waitForClientJob(client, jobId, expected="failed")
        assert failed["error"]["code"] == "invalid_client_asset"
        assert failed["download"] is None and failed["asset"] is None
        assert str(tmp_path) not in json.dumps(failed)
        assert client.get(f"/v1/jobs/{jobId}/download").status_code == 409


def testClientCompletedJobSurvivesServiceRestart(tmp_path):
    firstApp, _ = clientApp(tmp_path)
    with TestClient(firstApp) as client:
        jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
        firstJob = waitForClientJob(client, jobId)
        package = client.get(firstJob["download"]["url"]).content
    secondApp, pipeline = clientApp(tmp_path)
    with TestClient(secondApp) as client:
        restored = client.get(f"/v1/jobs/{jobId}")
        assert restored.status_code == 200
        assert restored.json()["status"] == "succeeded"
        assert restored.json()["download"] == firstJob["download"]
        assert client.get(restored.json()["download"]["url"]).content == package
        assert not pipeline.runStages


@pytest.mark.parametrize("interruptedStatus", ["queued", "running"])
def testClientInterruptedJobIsReportedAfterRestart(tmp_path, interruptedStatus):
    firstApp, _ = clientApp(tmp_path)
    with TestClient(firstApp) as client:
        jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
        waitForClientJob(client, jobId)
    recordPath = tmp_path / "runs/service" / jobId / "job.json"
    record = json.loads(recordPath.read_text(encoding="utf-8"))
    record.update(status=interruptedStatus, finishedAt=None)
    recordPath.write_text(json.dumps(record), encoding="utf-8")
    secondApp, pipeline = clientApp(tmp_path)
    with TestClient(secondApp) as client:
        restored = client.get(f"/v1/jobs/{jobId}")
        assert restored.status_code == 200
        job = restored.json()
        assert job["status"] == "failed"
        assert job["error"]["code"] and job["error"]["message"]
        assert job["download"] is None and job["asset"] is None
        assert client.get(f"/v1/jobs/{jobId}/download").status_code == 409
        assert not pipeline.runStages


def testClientRoutesDoNotTreatResearchJobsAsClientJobs(tmp_path):
    app, _ = clientApp(tmp_path)
    with TestClient(app) as client:
        research = client.post("/jobs", files=uploads(), data={"stopAfter": "geometry"})
        assert research.status_code == 202
        for jobId in (research.json()["jobId"], "unknown"):
            for suffix in ("", "/download", "/manifest"):
                response = client.get(f"/v1/jobs/{jobId}{suffix}")
                assert response.status_code == 404
                assert response.json()["apiVersion"] == "1"
                assert response.json()["error"]["code"]


def testBearerAuthenticationCoversClientLegacyAndSchemaRoutes(tmp_path):
    app, pipeline = clientApp(tmp_path, apiToken="fixture-service-token")
    authorization = {"Authorization": "Bearer fixture-service-token"}
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer incorrect-token"}):
            for path in ("/v1/capabilities", "/health", "/openapi.json"):
                response = client.get(path, headers=headers)
                assertError(response, 401, "unauthorized")
                assert response.headers["www-authenticate"] == "Bearer"
            for path in ("/v1/jobs", "/jobs"):
                assertError(client.post(path, files=uploads(), headers=headers), 401, "unauthorized")
        assert not pipeline.runStages
        client.headers.update(authorization)
        assert client.get("/health").status_code == 200
        assert client.get("/v1/capabilities").json()["ready"]
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert "ClientJob" in schema.json()["components"]["schemas"]
        assert "Capabilities" in schema.json()["components"]["schemas"]
        assert "/v1/jobs" in schema.json()["paths"]
        assert schema.json()["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
        assert schema.json()["paths"]["/v1/jobs"]["post"]["security"] == [{"HTTPBearer": []}]
        assert "application/zip" in schema.json()["paths"]["/v1/jobs/{jobId}/download"]["get"]["responses"]["200"]["content"]
        submitted = client.post("/v1/jobs", files=uploads())
        assert submitted.status_code == 202
        job = waitForClientJob(client, submitted.json()["jobId"])
        protectedPaths = [job["statusUrl"], job["download"]["url"], job["download"]["manifestUrl"],
                          f"/jobs/{job['jobId']}",
                          f"/jobs/{job['jobId']}/artifacts/assembly/clientAsset.zip"]
        for path in protectedPaths:
            assert client.get(path).status_code == 200
            assertError(client.get(path, headers={"Authorization": "Bearer incorrect-token"}),
                        401, "unauthorized")
        assert client.post("/jobs", files=uploads(), data={"stopAfter": "geometry"}).status_code == 202
