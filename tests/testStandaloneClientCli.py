"""Standalone CLI lifecycle using real assets and a network-only stub."""

from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from faceAppearance import clientCli
from faceAppearance.clientAsset import exportClientAsset
from faceAppearance.clientHttp import JobFailedError


JOB_ID = "0123456789abcdef0123456789abcdef"
ORIGIN = "http://127.0.0.1:12345"


@pytest.fixture
def integration(tmp_path, monkeypatch):
    templateVertices = np.array([[0., 0., 0.], [100., 0., 0.], [100., 200., 0.], [0., 200., 0.]])
    vertices = templateVertices.copy()
    vertices[2, 2] = 3.
    geometry = {"vertices": vertices, "faces": np.array([[0, 1, 2], [0, 2, 3]], np.int64),
                "uvs": np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.2, .1]]),
                "uvFaces": np.array([[0, 1, 2], [4, 2, 3]], np.int64),
                "materialIds": np.array([0, 0], np.int64)}
    identity = {"templateId": "standaloneFixtureV1", "templateHashes": {"originalTemplateObj": "a" * 64}}
    assembly = {"geometry": geometry, "templateVertices": templateVertices,
                "materials": [{"name": "skin", "baseColor": np.ones((2, 2, 3)) * .5,
                               "roughness": .6, "metallic": 0, "alphaMode": "OPAQUE"}],
                "components": [], "diagnostics": {"fixture": True}, **identity}
    exported = tmp_path / "exported"
    exportClientAsset(exported, assembly)
    archive = exported / "clientAsset.zip"
    template = tmp_path / "template"
    template.mkdir()
    np.savez_compressed(template / "template.npz", vertices=templateVertices,
                        **{name: geometry[name] for name in ("faces", "uvs", "uvFaces", "materialIds")},
                        nicpToFullIds=np.arange(4), staticMask=np.array([True, True, False, True]))
    manifest = {"schemaVersion": 1, **identity, "templateUnits": "centimeters", "outputUnits": "meters",
                "outputUnitScale": .01, "arrayUvOrigin": "vDown", "files": {
                    "template.npz": hashlib.sha256((template / "template.npz").read_bytes()).hexdigest()}}
    (template / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    photos = tmp_path / "photos"
    photos.mkdir()
    for number in range(4):
        (photos / f"view-{number}.png").write_bytes(f"stub upload {number}".encode())
    (photos / "ignore.txt").write_text("not an image", encoding="utf-8")
    queued = {"apiVersion": "1", "jobId": JOB_ID, "status": "queued", "error": None, "download": None}
    succeeded = {**queued, "status": "succeeded", "download": {
        "url": f"/v1/jobs/{JOB_ID}/download", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "sizeBytes": archive.stat().st_size}}
    scenario = SimpleNamespace(template=template, photos=photos, archive=archive, identity=identity,
                               geometry=geometry, output=tmp_path / "client", queued=queued,
                               succeeded=succeeded, calls=[], waitError=None, capabilityIdentity=identity)

    class StubClient:
        def __init__(self, baseUrl, **kwargs):
            self.baseUrl = baseUrl.rstrip("/")
            scenario.calls.append(("construct", self.baseUrl, kwargs))

        def getCapabilities(self):
            scenario.calls.append(("capabilities",))
            return {"apiVersion": "1", "template": deepcopy(scenario.capabilityIdentity)}

        def submit(self, imagePaths, *, templateId):
            scenario.calls.append(("submit", list(imagePaths), templateId))
            return deepcopy(queued)

        def waitForJob(self, jobId, *, timeout):
            scenario.calls.append(("wait", jobId, timeout))
            if scenario.waitError:
                raise scenario.waitError
            return deepcopy(succeeded)

        def download(self, job, destination):
            scenario.calls.append(("download", deepcopy(job), destination))
            with destination.open("xb") as stream:
                stream.write(archive.read_bytes())
            return destination

    monkeypatch.setattr(clientCli, "FaceAppearanceClient", StubClient)
    monkeypatch.delenv("FACE_APPEARANCE_API_TOKEN", raising=False)
    return scenario


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["faceAppearance.clientCli", *map(str, arguments)])
    clientCli.main()


def submit(monkeypatch, scenario, *extra):
    invoke(monkeypatch, "--url", ORIGIN, "submit", "--input", scenario.photos,
           "--output", scenario.output, "--template", scenario.template, *extra)


def resume(monkeypatch, scenario):
    invoke(monkeypatch, "resume", "--output", scenario.output, "--template", scenario.template)


def readState(scenario):
    return json.loads((scenario.output / "clientJob.json").read_text(encoding="utf-8"))


def testSubmitNoWaitPersistsIdentityAndNeverToken(integration, monkeypatch, capsys):
    monkeypatch.setenv("FACE_APPEARANCE_API_TOKEN", "do-not-persist-this-token")
    submit(monkeypatch, integration, "--noWait")
    output = capsys.readouterr()
    state = readState(integration)
    assert state["jobId"] == JOB_ID
    assert state["baseUrl"] == ORIGIN
    assert state["expectedTemplate"] == integration.identity
    assert state["lastJob"]["status"] == "queued"
    saved = (integration.output / "clientJob.json").read_text(encoding="utf-8")
    assert "do-not-persist-this-token" not in saved + output.out + output.err
    assert integration.calls[0][2]["token"] == "do-not-persist-this-token"
    submitted = [call for call in integration.calls if call[0] == "submit"]
    assert len(submitted) == 1 and len(submitted[0][1]) == 4
    assert all(path.suffix == ".png" for path in submitted[0][1])
    assert not any(call[0] in {"wait", "download"} for call in integration.calls)
    assert sorted(path.name for path in integration.output.iterdir()) == ["clientJob.json"]


def testResumeDownloadsAuditsAndReusesValidAssetWithoutPost(integration, monkeypatch):
    submit(monkeypatch, integration, "--noWait")
    integration.calls.clear()
    resume(monkeypatch, integration)
    assert not any(call[0] == "submit" for call in integration.calls)
    assert sum(call[0] == "download" for call in integration.calls) == 1
    assert readState(integration)["lastJob"]["status"] == "succeeded"
    report = json.loads((integration.output / "importReport.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert not report["rigValidated"] and not report["blendshapesValidated"]
    assert report["renderVertexCount"] > report["sourceVertexCount"]
    with np.load(integration.output / "renderMesh.npz", allow_pickle=False) as mesh:
        np.testing.assert_array_equal(mesh["sourceVertexIds"][mesh["faces"]], integration.geometry["faces"])
        np.testing.assert_array_equal(mesh["sourceUvIds"][mesh["faces"]], integration.geometry["uvFaces"])
    archiveBytes = (integration.output / "clientAsset.zip").read_bytes()
    resume(monkeypatch, integration)
    assert sum(call[0] == "download" for call in integration.calls) == 1
    assert not any(call[0] == "submit" for call in integration.calls)
    assert (integration.output / "clientAsset.zip").read_bytes() == archiveBytes


def testTimeoutLeavesAcceptedJobForResume(integration, monkeypatch, capsys):
    integration.waitError = TimeoutError("Polling timed out")
    with pytest.raises(SystemExit) as raised:
        submit(monkeypatch, integration, "--timeout", "0.01")
    assert raised.value.code == 1
    assert readState(integration)["jobId"] == JOB_ID
    assert readState(integration)["lastJob"]["status"] == "queued"
    assert JOB_ID in capsys.readouterr().err
    assert sum(call[0] == "submit" for call in integration.calls) == 1
    assert not any(call[0] == "download" for call in integration.calls)


def testFailedTaskIsPersistedBeforeCliExit(integration, monkeypatch):
    failed = {**integration.queued, "status": "failed", "error": {
        "code": "reconstruction_failed", "message": "Bad face input", "retryable": False}}
    integration.waitError = JobFailedError(failed)
    with pytest.raises(SystemExit) as raised:
        submit(monkeypatch, integration)
    assert raised.value.code == 1
    assert readState(integration)["lastJob"] == failed
    assert not any(call[0] == "download" for call in integration.calls)


@pytest.mark.parametrize("mismatch", ["template", "origin"])
def testResumeRejectsTemplateOrOriginBeforeConstructingHttp(integration, monkeypatch, mismatch):
    submit(monkeypatch, integration, "--noWait")
    integration.calls.clear()
    prefix = []
    if mismatch == "template":
        path = integration.template / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["templateId"] = "anotherTemplate"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        prefix = ["--url", "http://127.0.0.1:54321"]
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, *prefix, "resume", "--output", integration.output, "--template", integration.template)
    assert raised.value.code == 1
    assert not integration.calls


def testSubmitRejectsServiceTemplateMismatchBeforeUpload(integration, monkeypatch):
    integration.capabilityIdentity = {**integration.identity, "templateId": "different"}
    with pytest.raises(SystemExit) as raised:
        submit(monkeypatch, integration)
    assert raised.value.code == 1
    assert [call[0] for call in integration.calls] == ["construct", "capabilities"]
    assert not integration.output.exists()


def testInspectWorksOfflineWithoutConstructingHttp(integration, monkeypatch, tmp_path):
    report = tmp_path / "offline-report.json"
    render = tmp_path / "offline-render.npz"
    invoke(monkeypatch, "inspect", "--archive", integration.archive, "--template", integration.template,
           "--report", report, "--renderMesh", render)
    assert not integration.calls
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "passed"
    with np.load(render, allow_pickle=False) as mesh:
        assert len(mesh["sourceVertexIds"]) == 5


@pytest.mark.parametrize("useAlias", [False, True])
def testInspectRejectsCollidingOutputPaths(integration, monkeypatch, tmp_path, useAlias):
    destination = tmp_path / "result"
    render = tmp_path / "alias" / ".." / "result" if useAlias else destination
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "inspect", "--archive", integration.archive, "--template", integration.template,
               "--report", destination, "--renderMesh", render)
    assert raised.value.code == 1
    assert not destination.exists()
    assert not integration.calls


@pytest.mark.parametrize("timeout", ["-1", "nan"])
def testInvalidPollingDeadlineRejectedBeforeSubmitting(integration, monkeypatch, timeout):
    with pytest.raises(SystemExit) as raised:
        submit(monkeypatch, integration, "--timeout", timeout)
    assert raised.value.code == 1
    assert not integration.calls


@pytest.mark.parametrize("existing", ["report", "render"])
def testInspectRefusesOverwrite(integration, monkeypatch, tmp_path, existing):
    report = tmp_path / "offline-report.json"
    render = tmp_path / "offline-render.npz"
    target = report if existing == "report" else render
    target.write_bytes(b"keep existing output")
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "inspect", "--archive", integration.archive, "--template", integration.template,
               "--report", report, "--renderMesh", render)
    assert raised.value.code == 1
    assert target.read_bytes() == b"keep existing output"
    assert not (render if existing == "report" else report).exists()
    assert not integration.calls


def testSubmitRefusesNonemptyOutputWithoutHttp(integration, monkeypatch):
    integration.output.mkdir()
    existing = integration.output / "keep.txt"
    existing.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        submit(monkeypatch, integration)
    assert raised.value.code == 1
    assert existing.read_text(encoding="utf-8") == "existing"
    assert not integration.calls


def testResumeRejectsCorruptExistingZipWithoutOverwritingIt(integration, monkeypatch):
    submit(monkeypatch, integration, "--noWait")
    existing = integration.output / "clientAsset.zip"
    existing.write_bytes(b"damaged package")
    integration.calls.clear()
    with pytest.raises(SystemExit) as raised:
        resume(monkeypatch, integration)
    assert raised.value.code == 1
    assert existing.read_bytes() == b"damaged package"
    assert not any(call[0] in {"submit", "download"} for call in integration.calls)
    assert not (integration.output / "renderMesh.npz").exists()
