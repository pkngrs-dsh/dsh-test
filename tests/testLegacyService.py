"""Legacy OSS result URL contracts with fake storage and CPU-only startup."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from faceAppearance.config import PipelineConfig
from faceAppearance.legacyService import LegacyQueue, createLegacyApp


@pytest.fixture
def fakeOss(monkeypatch):
    signedUrl = ("https://cdn.example:9443/avatars/a%2Fb%20%23%3F.zip"
                 "?OSSAccessKeyId=fake&Expires=1&Signature=fake#fragment")
    calls = []

    class Bucket:
        def __init__(self, auth, endpoint, bucketName, *, connect_timeout):
            calls.append(("bucket", endpoint, bucketName, connect_timeout))

        def put_object_from_file(self, key, archive):
            calls.append(("upload", key, archive))
            return SimpleNamespace(status=200)

        def sign_url(self, method, key, expires):
            calls.append(("sign", method, key, expires))
            return signedUrl

    monkeypatch.setitem(sys.modules, "oss2", SimpleNamespace(Auth=lambda *args: object(), Bucket=Bucket))
    return signedUrl, calls


@pytest.mark.parametrize("mode", [None, "signed", "public"])
def testOssResultUrlModeFromEnvironment(tmp_path, monkeypatch, fakeOss, mode):
    signedUrl, calls = fakeOss
    if mode is None:
        monkeypatch.delenv("FACE_APPEARANCE_OSS_RESULT_URL_MODE", raising=False)
    else:
        monkeypatch.setenv("FACE_APPEARANCE_OSS_RESULT_URL_MODE", mode)
    monkeypatch.setattr("faceAppearance.legacyService._ossConfiguration",
                        lambda: ("fake-id", "fake-secret", "https://oss.example", "fake-bucket"))

    class Pipeline:
        def __init__(self, config):
            self.loaded = False

        def loadModels(self):
            self.loaded = True

    class Assets:
        def __init__(self, path):
            self.manifest = {"gender": "female" if "Female" in str(path) else "male"}

    monkeypatch.setattr("faceAppearance.headAssembly.AssemblyAssets", Assets)
    async def runImmediately(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("faceAppearance.legacyService.run_in_threadpool", runImmediately)
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    app = createLegacyApp(config, pipelineFactory=Pipeline)
    archive = tmp_path / "asset.zip"
    archive.write_bytes(b"fake-archive")
    async def upload():
        async with app.router.lifespan_context(app):
            return app.state.legacyQueue._upload(archive, {"upload_dir": "avatars/", "task_id": "task-one"})

    result = asyncio.run(upload())
    expected = "https://cdn.example:9443/avatars/a%2Fb%20%23%3F.zip" if mode == "public" else signedUrl
    assert result == expected
    assert calls == [("bucket", "https://oss.example", "fake-bucket", 30),
                     ("upload", "avatars/task-one.zip", str(archive)),
                     ("sign", "GET", "avatars/task-one.zip", 3600)]


def testLegacyQueueExistingConstructorKeepsSignedUrls(tmp_path, fakeOss):
    signedUrl, _ = fakeOss
    worker = LegacyQueue(None, {}, None, tmp_path, 2, None, False,
                         ("fake-id", "fake-secret", "https://oss.example", "fake-bucket"))
    assert worker._upload(tmp_path / "asset.zip", {"upload_dir": "avatars", "task_id": "old-caller"}) == signedUrl


def testPublicOssModeKeepsServiceDownloadWithoutOss(tmp_path):
    worker = LegacyQueue(None, {}, None, tmp_path, 2, "https://service.example", False, None,
                         ossResultUrlMode="public")
    assert worker._upload(tmp_path / "asset.zip", {"task_id": "local-result"}) == (
        "https://service.example/tasks/local-result/download")


@pytest.mark.parametrize("mode", ["", "unsigned", "PUBLIC"])
def testInvalidOssResultUrlModeFailsBeforeStartup(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("FACE_APPEARANCE_OSS_RESULT_URL_MODE", mode)
    monkeypatch.setattr("faceAppearance.legacyService._ossConfiguration", lambda: None)
    config = PipelineConfig(projectRoot=tmp_path, device="cpu")
    with pytest.raises(ValueError, match="FACE_APPEARANCE_OSS_RESULT_URL_MODE must be signed or public"):
        createLegacyApp(config, publicBaseUrl="https://service.example")
