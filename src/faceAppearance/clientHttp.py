"""Independent HTTP client: no model imports, no automatic submission retries."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
import uuid


class ClientProtocolError(ValueError):
    """The server response does not match the supported client contract."""


class ClientHttpError(RuntimeError):
    def __init__(self, statusCode, code, message, retryable=False, retryAfter=None):
        super().__init__(message)
        self.statusCode = statusCode
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retryAfter = retryAfter


class JobFailedError(RuntimeError):
    def __init__(self, record):
        self.record = record
        error = record.get("error") or {}
        super().__init__(f"Job {record['jobId']} failed: {error.get('code', 'unknown')}: "
                         f"{error.get('message', 'No detail supplied')}")


def checkedJobId(jobId):
    if not isinstance(jobId, str) or re.fullmatch(r"[0-9a-f]{32}", jobId) is None:
        raise ClientProtocolError("Expected a lowercase 32-character hexadecimal job ID")
    return jobId


def positiveSeconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


class FaceAppearanceClient:
    def __init__(self, baseUrl: str, *, token: str | None = None, requestTimeout: float = 30,
                 maxDownloadBytes: int = 512 * 1024 * 1024):
        parsed = urlsplit(baseUrl)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("baseUrl must be an HTTP(S) origin without credentials, path, query or fragment")
        if token is not None and (not token or "\n" in token or "\r" in token):
            raise ValueError("token must be a nonempty single line")
        if type(maxDownloadBytes) is not int or maxDownloadBytes <= 0:
            raise ValueError("maxDownloadBytes must be a positive integer")
        self.baseUrl = baseUrl.rstrip("/")
        self.scheme, self.host, self.port = parsed.scheme, parsed.hostname, parsed.port
        self.token = token
        self.requestTimeout = positiveSeconds(requestTimeout, "requestTimeout")
        self.maxDownloadBytes = maxDownloadBytes

    def _open(self, method, path, *, body=None, headers=None, timeout=None):
        if not path.startswith("/v1/") or urlsplit(path).netloc or "\r" in path or "\n" in path:
            raise ClientProtocolError("Request path must stay under this service's /v1/ API")
        connectionClass = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        connection = connectionClass(self.host, self.port, timeout=timeout or self.requestTimeout)
        requestHeaders = {"Accept": "application/json", "User-Agent": "FaceAppearanceClient/1"}
        if self.token is not None:
            requestHeaders["Authorization"] = "Bearer " + self.token
        requestHeaders.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=requestHeaders)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                payload = response.read(2 * 1024 * 1024)
                try:
                    detail = json.loads(payload).get("error", {})
                    if not isinstance(detail, dict):
                        detail = {}
                except (ValueError, AttributeError):
                    detail = {}
                retry = response.getheader("Retry-After")
                retryAfter = float(retry) if retry and retry.isdigit() else None
                raise ClientHttpError(response.status, detail.get("code", "http_error"),
                    detail.get("message", f"HTTP {response.status} {response.reason}"),
                    bool(detail.get("retryable", response.status in {429, 502, 503, 504})), retryAfter)
            return connection, response
        except BaseException:
            connection.close()
            raise

    def _json(self, method, path, **kwargs):
        connection, response = self._open(method, path, **kwargs)
        try:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ClientProtocolError("JSON response exceeds 2 MiB")
            try:
                data = json.loads(raw)
            except ValueError as error:
                raise ClientProtocolError("Expected a JSON response") from error
            if not isinstance(data, dict) or data.get("apiVersion") != "1":
                raise ClientProtocolError("Expected API version 1")
            return data
        finally:
            connection.close()

    def getCapabilities(self):
        return self._json("GET", "/v1/capabilities")

    @staticmethod
    def _job(record, expectedId=None):
        jobId = checkedJobId(record.get("jobId"))
        if expectedId is not None and jobId != expectedId:
            raise ClientProtocolError("Response belongs to a different job")
        if record.get("status") not in {"queued", "running", "succeeded", "failed"}:
            raise ClientProtocolError("Unknown job status")
        return record

    def submit(self, imagePaths: list[Path], *, templateId: str):
        if not isinstance(templateId, str) or not templateId or "\r" in templateId or "\n" in templateId:
            raise ValueError("templateId must be nonempty")
        caps = self.getCapabilities()
        if not caps.get("ready"):
            raise ClientProtocolError("Client assembly is not ready")
        if (caps.get("template") or {}).get("templateId") != templateId:
            raise ClientProtocolError("Service template differs from the requested template")
        limits = caps.get("limits", {})
        for key in ("minImages", "maxImages", "maxImageBytes", "maxTotalBytes"):
            if type(limits.get(key)) is not int or limits[key] <= 0:
                raise ClientProtocolError("Invalid service upload limits")
        paths = [Path(path) for path in imagePaths]
        if not limits["minImages"] <= len(paths) <= limits["maxImages"]:
            raise ValueError(f"Expected {limits['minImages']}–{limits['maxImages']} images")
        sizes = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if not 0 < size <= limits["maxImageBytes"]:
                raise ValueError(f"Image is empty or exceeds service byte limit: {path}")
            sizes.append(size)
        if sum(sizes) > limits["maxTotalBytes"]:
            raise ValueError("Combined image data exceeds service byte limit")
        boundary = "FaceAppearance" + uuid.uuid4().hex
        prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="templateId"\r\n\r\n'
                  f'{templateId}\r\n').encode("utf-8")
        parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="images"; '
                  f'filename="image{index:02d}.image"\r\nContent-Type: application/octet-stream\r\n\r\n'
                  ).encode("ascii") for index in range(len(paths))]
        suffix = f"--{boundary}--\r\n".encode("ascii")
        length = len(prefix) + len(suffix) + sum(size + len(part) + 2 for size, part in zip(sizes, parts))

        def body():
            yield prefix
            for path, expectedSize, part in zip(paths, sizes, parts):
                yield part
                count = 0
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        count += len(chunk)
                        if count > expectedSize:
                            raise ValueError("Input image changed during upload")
                        yield chunk
                if count != expectedSize:
                    raise ValueError("Input image changed during upload")
                yield b"\r\n"
            yield suffix

        # A lost POST response is ambiguous; never repeat a reconstruction automatically.
        record = self._json("POST", "/v1/jobs", body=body(), headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(length)})
        return self._job(record)

    def getJob(self, jobId: str):
        jobId = checkedJobId(jobId)
        return self._job(self._json("GET", f"/v1/jobs/{jobId}"), jobId)

    def waitForJob(self, jobId: str, *, timeout: float = 600, pollInterval: float | None = None):
        jobId = checkedJobId(jobId)
        deadline = time.monotonic() + positiveSeconds(timeout, "timeout")
        if pollInterval is not None:
            positiveSeconds(pollInterval, "pollInterval")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Polling timed out; resume existing job {jobId}")
            delay = pollInterval or 2
            try:
                record = self._job(self._json("GET", f"/v1/jobs/{jobId}",
                    timeout=min(self.requestTimeout, remaining)), jobId)
                if record["status"] == "failed":
                    raise JobFailedError(record)
                if record["status"] == "succeeded":
                    return record
                requested = record.get("pollAfterSeconds", 2)
                if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not math.isfinite(requested):
                    raise ClientProtocolError("Invalid polling interval")
                delay = pollInterval if pollInterval is not None else max(.1, min(30, requested))
            except ClientHttpError as error:
                if not error.retryable:
                    raise
                delay = pollInterval or min(30, error.retryAfter or 2)
            except (OSError, http.client.HTTPException):
                # GET requests can safely be retried within the original deadline.
                pass
            time.sleep(min(delay, max(0, deadline - time.monotonic())))

    def download(self, job: dict, destination: Path) -> Path:
        self._job(job)
        if job["status"] != "succeeded":
            raise ClientProtocolError("Only successful jobs have an asset download")
        metadata = job.get("download")
        if not isinstance(metadata, dict):
            raise ClientProtocolError("Missing download metadata")
        expectedUrl = f"/v1/jobs/{job['jobId']}/download"
        if metadata.get("url") != expectedUrl:
            raise ClientProtocolError("Download URL does not match this service and job")
        digest = metadata.get("sha256")
        size = metadata.get("sizeBytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ClientProtocolError("Invalid archive SHA-256")
        if type(size) is not int or not 0 < size <= self.maxDownloadBytes:
            raise ClientProtocolError("Invalid archive size or download limit exceeded")
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        connection, response = self._open("GET", expectedUrl, headers={"Accept": "application/zip"})
        temporary = None
        try:
            received = 0
            checksum = hashlib.sha256()
            with tempfile.NamedTemporaryFile(prefix=".client-", suffix=".part", dir=destination.parent,
                                             delete=False) as output:
                temporary = Path(output.name)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > size:
                        raise ClientProtocolError("Downloaded archive exceeds declared size")
                    checksum.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if received != size or checksum.hexdigest() != digest:
                raise ClientProtocolError("Downloaded archive size or SHA-256 differs")
            # An exclusive hard link publishes complete bytes without overwriting a raced destination.
            os.link(temporary, destination)
            return destination
        finally:
            connection.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
