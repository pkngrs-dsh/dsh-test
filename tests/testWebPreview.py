"""CPU checks for browser geometry, packaged textures and access boundaries."""

import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from faceAppearance.clientAsset import exportClientAsset
from faceAppearance.clientImport import loadClientAsset
from faceAppearance.webPreview import buildPreview
from testClientService import (TEMPLATE_HASHES, TEMPLATE_ID, assertError, clientApp,
                               tinyAssembly, uploads, waitForClientJob)


def testPreviewPreservesSourceNormalsAcrossUvAndMaterialSeams(tmp_path):
    assembly = tinyAssembly()
    vertices = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    assembly["templateVertices"] = vertices
    assembly["geometry"] = {
        "vertices": vertices,
        "faces": np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]], dtype=np.int64),
        "uvs": np.array([[0., 0.], [1., 0.], [0., 1.], [.3, .7]]),
        "uvFaces": np.array([[0, 1, 2], [3, 2, 1], [0, 2, 1]], dtype=np.int64),
        "materialIds": np.array([0, 1, 0], dtype=np.int64)}
    assembly["materials"].append({**assembly["materials"][0], "name": "eyes", "alphaMode": "MASK"})
    exportClientAsset(tmp_path / "asset", assembly)
    preview = buildPreview(loadClientAsset(tmp_path / "asset/clientAsset.zip"), "fixture")
    mesh = preview["mesh"]
    sourceIds = np.asarray(mesh["sourceVertexIds"])
    renderNormals = np.asarray(mesh["normals"]).reshape(-1, 3)
    positions = np.asarray(mesh["positions"]).reshape(-1, 3)
    indices = np.asarray(mesh["indices"]).reshape(-1, 3)
    assert preview["counts"]["renderVertices"] > preview["counts"]["sourceVertices"]
    np.testing.assert_array_equal(positions, vertices[sourceIds] * .01)
    np.testing.assert_array_equal(sourceIds[indices], assembly["geometry"]["faces"])
    expectedNormals = np.array([[1., 1., 1.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    expectedNormals /= np.linalg.norm(expectedNormals, axis=1)[:, None]
    np.testing.assert_allclose(renderNormals, expectedNormals[sourceIds])
    np.testing.assert_array_equal(np.asarray(mesh["uvs"]).reshape(-1, 2)[indices],
                                  assembly["geometry"]["uvs"][assembly["geometry"]["uvFaces"]])
    assert mesh["groups"] == [{"start": 0, "count": 3, "materialIndex": 0},
                              {"start": 3, "count": 3, "materialIndex": 1},
                              {"start": 6, "count": 3, "materialIndex": 0}]
    assert preview["coordinates"] == {"lengthUnit": "meter", "uvOrigin": "vDown"}
    assert preview["materials"][1]["alphaMode"] == "MASK"


def testPreviewAndTextureComeFromVerifiedClientPackage(tmp_path):
    app, _ = clientApp(tmp_path)
    with TestClient(app) as client:
        jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
        waitForClientJob(client, jobId)
        response = client.get(f"/v1/jobs/{jobId}/preview")
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["schemaVersion"] == 1 and preview["jobId"] == jobId
        assert preview["templateId"] == TEMPLATE_ID and preview["templateHashes"] == TEMPLATE_HASHES
        assert preview["counts"] == {"sourceVertices": 3, "renderVertices": 3, "faces": 1, "materials": 1}
        assert str(tmp_path) not in json.dumps(preview)
        texture = preview["materials"][0]["texture"]
        image = client.get(texture["url"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(image.content).hexdigest() == texture["sha256"]
        assert len(image.content) == texture["sizeBytes"]
        assert image.headers["etag"] == '"' + texture["sha256"] + '"'
        assert response.headers["cache-control"] == image.headers["cache-control"] == "no-store"
        for index in (-1, 1, 999):
            assertError(client.get(f"/v1/jobs/{jobId}/textures/{index}"), 404, "not_found")


def testViewerIsPublicButPreviewAndAllApisRequireBearerToken(tmp_path):
    app, _ = clientApp(tmp_path, apiToken="viewer-test-token")
    with TestClient(app) as client:
        assert client.get("/viewer", follow_redirects=False).headers["location"] == "/viewer/"
        shell = client.get("/viewer/")
        assert shell.status_code == 200 and "text/html" in shell.headers["content-type"]
        for path in ("/v1/jobs/unknown/preview", "/v1/jobs/unknown/textures/0", "/v1/capabilities",
                     "/jobs/unknown", "/health", "/viewer-not-public", "/openapi.json"):
            assertError(client.get(path), 401, "unauthorized")
        for path in ("/viewer/%2e%2e/service.py", "/viewer/%2e%2e/v1/capabilities"):
            assert client.get(path).status_code in (401, 404)
        assert client.get("/viewer/no-such-file.py").status_code == 404
        client.headers["Authorization"] = "Bearer viewer-test-token"
        jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
        waitForClientJob(client, jobId)
        for suffix in ("preview", "textures/0"):
            path = f"/v1/jobs/{jobId}/{suffix}"
            assert client.get(path).status_code == 200
            assertError(client.get(path, headers={"Authorization": "Bearer wrong-token"}), 401, "unauthorized")


def testPreviewWaitsForSuccessfulClientJob(tmp_path):
    app, pipeline = clientApp(tmp_path)
    pipeline.release.clear()
    with TestClient(app) as client:
        try:
            jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
            assert pipeline.started.wait(timeout=5)
            for suffix in ("preview", "textures/0"):
                assertError(client.get(f"/v1/jobs/{jobId}/{suffix}"), 409, "result_not_ready")
        finally:
            pipeline.release.set()
        waitForClientJob(client, jobId)


@pytest.mark.parametrize("damage", ["bytes", "template"])
def testPreviewRejectsPackageChangedAfterJobCompletion(tmp_path, damage):
    app, _ = clientApp(tmp_path)
    with TestClient(app) as client:
        jobId = client.post("/v1/jobs", files=uploads()).json()["jobId"]
        waitForClientJob(client, jobId)
        archive = tmp_path / "runs/service" / jobId / "output/assembly/clientAsset.zip"
        if damage == "bytes":
            archive.write_bytes(archive.read_bytes() + b"tampered")
        else:
            # Even a valid replacement ZIP cannot substitute a different template.
            from shutil import rmtree
            replacement = tinyAssembly()
            replacement["templateId"] = "anotherTemplate"
            rmtree(archive.parent)
            exportClientAsset(archive.parent, replacement)
        for suffix in ("preview", "textures/0"):
            response = client.get(f"/v1/jobs/{jobId}/{suffix}")
            assertError(response, 409, "invalid_client_asset")
            assert str(tmp_path) not in response.text


def testPreviewDoesNotExposeResearchOrUnknownJobs(tmp_path):
    app, _ = clientApp(tmp_path)
    with TestClient(app) as client:
        jobId = client.post("/jobs", files=uploads(), data={"stopAfter": "geometry"}).json()["jobId"]
        for candidate in (jobId, "unknown"):
            for suffix in ("preview", "textures/0"):
                assertError(client.get(f"/v1/jobs/{candidate}/{suffix}"), 404, "not_found")
