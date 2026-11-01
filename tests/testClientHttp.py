"""Reference HTTP consumer against a local server, without model or GPU work."""

from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from faceAppearance.clientHttp import (
    ClientHttpError,
    ClientProtocolError,
    FaceAppearanceClient,
    JobFailedError,
)


JOB_ID = "0123456789abcdef0123456789abcdef"
TEMPLATE_ID = "legacyFemaleV1"
JOB_PATH = f"/v1/jobs/{JOB_ID}"
PAYLOAD = b"client archive bytes for integrity checks"


def capabilities():
    return {
        "apiVersion": "1", "ready": True, "acceptingJobs": True,
        "template": {"templateId": TEMPLATE_ID, "templateHashes": {"fullMesh": "a" * 64}},
        "limits": {"minImages": 4, "maxImages": 16, "maxImageBytes": 1024 * 1024,
                   "maxTotalBytes": 4 * 1024 * 1024},
        "output": {"format": "zip", "schemaVersion": 1},
    }


def job(status="succeeded"):
    return {
        "apiVersion": "1", "jobId": JOB_ID, "status": status,
        "statusUrl": JOB_PATH, "pollAfterSeconds": 0.001,
        "createdAt": "2026-10-10T00:00:00+00:00", "startedAt": None,
        "finishedAt": None, "numImages": 4, "metrics": {}, "warnings": [],
        "asset": {"schemaVersion": 1, "templateId": TEMPLATE_ID},
        "error": {"code": "reconstruction_failed", "message": "Face detection failed",
                  "retryable": False} if status == "failed" else None,
        "download": {"url": JOB_PATH + "/download", "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
                     "sizeBytes": len(PAYLOAD)} if status == "succeeded" else None,
    }


@contextmanager
def localServer():
    state = SimpleNamespace(requests=[], routes={})

    class Handler(BaseHTTPRequestHandler):
        def handleRequest(self):
            count = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(count)
            state.requests.append({"method": self.command, "path": self.path,
                                   "headers": dict(self.headers), "body": body})
            response = state.routes.get((self.command, self.path),
                                        (404, {"error": {"code": "not_found", "message": "Missing", "retryable": False}}, {}))
            if isinstance(response, list):
                response = response.pop(0) if len(response) > 1 else response[0]
            status, content, headers = response
            if isinstance(content, (dict, list)):
                content = json.dumps(content).encode("utf-8")
                headers = {"Content-Type": "application/json", **headers}
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content)

        do_GET = do_POST = handleRequest

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.baseUrl = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def server():
    with localServer() as running:
        running.routes[("GET", "/v1/capabilities")] = (200, capabilities(), {})
        yield running


@pytest.fixture
def photos(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"view-{index}.png"
        Image.new("RGB", (10, 12), (index * 20, 100, 150)).save(path)
        paths.append(path)
    return paths


def testUploadPollAndVerifiedDownload(server, photos, tmp_path):
    server.routes[("POST", "/v1/jobs")] = (202, job("queued"), {})
    server.routes[("GET", JOB_PATH)] = [(200, job("queued"), {}), (200, job(), {})]
    server.routes[("GET", JOB_PATH + "/download")] = (200, PAYLOAD, {"Content-Type": "application/zip"})
    client = FaceAppearanceClient(server.baseUrl, token="fixture-token")
    assert client.getCapabilities()["template"]["templateId"] == TEMPLATE_ID
    submitted = client.submit(photos, templateId=TEMPLATE_ID)
    assert submitted["jobId"] == JOB_ID
    completed = client.waitForJob(JOB_ID, timeout=5, pollInterval=0.001)
    destination = tmp_path / "download.zip"
    assert client.download(completed, destination) == destination
    assert destination.read_bytes() == PAYLOAD
    assert all(request["headers"]["Authorization"] == "Bearer fixture-token"
               for request in server.requests)
    uploads = [request for request in server.requests if request["method"] == "POST"]
    assert len(uploads) == 1
    uploaded = uploads[0]
    assert int(uploaded["headers"]["Content-Length"]) == len(uploaded["body"])
    assert "Transfer-Encoding" not in uploaded["headers"]
    multipart = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + uploaded["headers"]["Content-Type"] + "\r\n\r\n").encode()
        + uploaded["body"])
    fields = list(multipart.iter_parts())
    images = [part for part in fields if part.get_param("name", header="content-disposition") == "images"]
    assert [part.get_payload(decode=True) for part in images] == [path.read_bytes() for path in photos]
    template = [part for part in fields if part.get_param("name", header="content-disposition") == "templateId"]
    assert len(template) == 1 and template[0].get_payload(decode=True).decode() == TEMPLATE_ID


@pytest.mark.parametrize("case", ["tooFew", "missing", "empty", "tooLarge", "templateMismatch"])
def testInvalidUploadNeverPosts(server, photos, tmp_path, case):
    selected = photos[:]
    templateId = TEMPLATE_ID
    if case == "tooFew":
        selected.pop()
    elif case == "missing":
        selected[-1] = tmp_path / "missing.png"
    elif case == "empty":
        selected[-1].write_bytes(b"")
    elif case == "tooLarge":
        limits = capabilities()
        limits["limits"]["maxImageBytes"] = 1
        server.routes[("GET", "/v1/capabilities")] = (200, limits, {})
    else:
        templateId = "wrongTemplate"
    with pytest.raises((ValueError, OSError)):
        FaceAppearanceClient(server.baseUrl).submit(selected, templateId=templateId)
    assert all(request["method"] == "GET" for request in server.requests)


def testStructuredHttpErrorDoesNotRetryPost(server, photos):
    server.routes[("POST", "/v1/jobs")] = (429, {
        "apiVersion": "1", "error": {"code": "queue_full", "message": "Retry later", "retryable": True}},
        {"Retry-After": "2"})
    with pytest.raises(ClientHttpError) as raised:
        FaceAppearanceClient(server.baseUrl).submit(photos, templateId=TEMPLATE_ID)
    assert raised.value.statusCode == 429
    assert raised.value.code == "queue_full"
    assert raised.value.message == "Retry later"
    assert raised.value.retryable is True
    assert sum(request["method"] == "POST" for request in server.requests) == 1


def testFailedTaskPreservesRecord(server):
    failed = job("failed")
    server.routes[("GET", JOB_PATH)] = (200, failed, {})
    with pytest.raises(JobFailedError) as raised:
        FaceAppearanceClient(server.baseUrl).waitForJob(JOB_ID, timeout=5)
    assert raised.value.record == failed
    assert len(server.requests) == 1


def testPollingDeadlineStopsWithoutResubmitting(server):
    server.routes[("GET", JOB_PATH)] = (200, job("running"), {})
    with pytest.raises(TimeoutError):
        FaceAppearanceClient(server.baseUrl).waitForJob(JOB_ID, timeout=0.025, pollInterval=0.01)
    assert server.requests
    assert all(request["method"] == "GET" and request["path"] == JOB_PATH for request in server.requests)


def testPollingRetriesTransientGetWithoutSubmittingAgain(server):
    server.routes[("GET", JOB_PATH)] = [
        (503, {"apiVersion": "1", "error": {"code": "service_unavailable",
                "message": "Temporarily busy", "retryable": True}}, {"Retry-After": "1"}),
        (200, job(), {}),
    ]
    result = FaceAppearanceClient(server.baseUrl).waitForJob(JOB_ID, timeout=5, pollInterval=0.001)
    assert result["status"] == "succeeded"
    assert [(request["method"], request["path"]) for request in server.requests] == [("GET", JOB_PATH)] * 2


def testPollingDoesNotRetryPermanentHttpError(server):
    server.routes[("GET", JOB_PATH)] = (404, {"apiVersion": "1", "error": {
        "code": "not_found", "message": "Unknown job", "retryable": False}}, {})
    with pytest.raises(ClientHttpError) as raised:
        FaceAppearanceClient(server.baseUrl).waitForJob(JOB_ID, timeout=5, pollInterval=0.001)
    assert raised.value.code == "not_found"
    assert len(server.requests) == 1


def testReturnedJobMustMatchRequestedIdentity(server):
    different = job()
    different["jobId"] = "a" * 32
    server.routes[("GET", JOB_PATH)] = (200, different, {})
    with pytest.raises(ClientProtocolError):
        FaceAppearanceClient(server.baseUrl).getJob(JOB_ID)


@pytest.mark.parametrize("baseUrl", ["file:///tmp", "ftp://localhost", "https://user:password@example.com",
                                   "https://example.com/prefix", "https://example.com?token=x",
                                   "https://example.com#fragment"])
def testBaseUrlMustBeHttpOrigin(baseUrl):
    with pytest.raises(ValueError):
        FaceAppearanceClient(baseUrl)


@pytest.mark.parametrize("jobId", ["../capabilities", "short", "g" * 32, JOB_ID + "?x=1"])
def testInvalidJobIdNeverRequests(server, jobId):
    with pytest.raises(ValueError):
        FaceAppearanceClient(server.baseUrl).getJob(jobId)
    assert not server.requests


def testRedirectIsNotFollowed(server):
    server.routes[("GET", "/v1/capabilities")] = (302, b"redirect", {"Location": server.baseUrl + "/other"})
    with pytest.raises((ClientHttpError, ClientProtocolError)):
        FaceAppearanceClient(server.baseUrl, token="fixture-token").getCapabilities()
    assert [request["path"] for request in server.requests] == ["/v1/capabilities"]


def testInvalidJsonIsProtocolError(server):
    server.routes[("GET", "/v1/capabilities")] = (200, b"not json", {"Content-Type": "application/json"})
    with pytest.raises(ClientProtocolError):
        FaceAppearanceClient(server.baseUrl).getCapabilities()


@pytest.mark.parametrize("field", ["sha256", "tooFewBytes", "tooManyBytes"])
def testDownloadIntegrityFailureLeavesNoOutput(server, tmp_path, field):
    record = job()
    if field == "sha256":
        record["download"][field] = "0" * 64
    else:
        record["download"]["sizeBytes"] = len(PAYLOAD) + (1 if field == "tooFewBytes" else -1)
    server.routes[("GET", JOB_PATH + "/download")] = (200, PAYLOAD, {})
    destination = tmp_path / "result.zip"
    before = set(tmp_path.iterdir())
    with pytest.raises(ClientProtocolError):
        FaceAppearanceClient(server.baseUrl).download(record, destination)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("url", ["https://untrusted.invalid/v1/jobs/file/download",
                               "//untrusted.invalid/v1/jobs/file/download", "/outside/download"])
def testDownloadRejectsUntrustedMetadataUrlBeforeRequest(server, tmp_path, url):
    record = job()
    record["download"]["url"] = url
    with pytest.raises(ValueError):
        FaceAppearanceClient(server.baseUrl, token="fixture-token").download(record, tmp_path / "result.zip")
    assert not server.requests
    assert not list(tmp_path.iterdir())


def testDownloadDoesNotOverwriteExistingAsset(server, tmp_path):
    destination = tmp_path / "result.zip"
    destination.write_bytes(b"existing asset")
    with pytest.raises(FileExistsError):
        FaceAppearanceClient(server.baseUrl).download(job(), destination)
    assert destination.read_bytes() == b"existing asset"
    assert not server.requests
