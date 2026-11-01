"""Persistent jobs and one reconstruction consumer per output directory.

The directory lock is held for the complete worker lifetime. On restart,
unfinished jobs are failed explicitly; reconstruction is never replayed.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import mimetypes
import os
import queue
import re
import stat
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from .artifacts import jsonValue


logger = logging.getLogger(__name__)
jobIdPattern = re.compile(r"[0-9a-f]{32}\Z")
jobStatuses = {"queued", "running", "succeeded", "failed"}
jobStages = {"geometry", "delight", "fusion", "wildcap", "assembly"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def writeJobRecord(jobDir: Path, record: dict[str, Any]) -> None:
    """Replace the record atomically, leaving the old record intact on failure."""
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, temporaryName = tempfile.mkstemp(prefix=".job-", suffix=".tmp", dir=jobDir)
    temporaryPath = Path(temporaryName)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporaryPath.replace(jobDir / "job.json")
    finally:
        temporaryPath.unlink(missing_ok=True)


def validJobId(jobId: str) -> bool:
    return isinstance(jobId, str) and jobIdPattern.fullmatch(jobId) is not None


def validArtifactName(filename: Any) -> bool:
    if not isinstance(filename, str) or not filename or "\\" in filename:
        return False
    path = PurePosixPath(filename)
    return filename != "." and not path.is_absolute() and ".." not in path.parts and path.as_posix() == filename


def rejectJsonConstant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def validateRecord(record: Any, jobId: str) -> None:
    """Reject damaged records instead of making their IDs or paths public."""
    if not isinstance(record, dict) or record.get("jobId") != jobId or not validJobId(jobId):
        raise ValueError("Invalid persisted job ID")
    if record.get("status") not in jobStatuses or record.get("stopAfter") not in jobStages:
        raise ValueError("Invalid persisted job status or stage")
    if not isinstance(record.get("createdAt"), str):
        raise ValueError("Missing persisted creation time")
    for key in ("startedAt", "finishedAt"):
        if key not in record or record[key] is not None and not isinstance(record[key], str):
            raise ValueError("Invalid persisted job time")
    names, count = record.get("inputNames"), record.get("numImages")
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or type(count) is not int or count != len(names)):
        raise ValueError("Invalid persisted input metadata")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Invalid persisted artifacts")
    for artifact in artifacts:
        if (not isinstance(artifact, dict) or not validArtifactName(artifact.get("filename"))
                or type(artifact.get("sizeBytes")) is not int or artifact["sizeBytes"] < 0):
            raise ValueError("Invalid persisted artifact")
        if "sha256" in artifact and (not isinstance(artifact["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None):
            raise ValueError("Invalid persisted artifact digest")
        if "mediaType" in artifact and not isinstance(artifact["mediaType"], str):
            raise ValueError("Invalid persisted artifact media type")
        artifact["url"] = f"/jobs/{jobId}/artifacts/{quote(artifact['filename'], safe='/')}"
    if record.get("error") is not None and not isinstance(record["error"], dict):
        raise ValueError("Invalid persisted job error")
    if record.get("result") is not None and not isinstance(record["result"], dict):
        raise ValueError("Invalid persisted job result")
    if "clientContract" in record and type(record["clientContract"]) is not bool:
        raise ValueError("Invalid persisted client contract")
    if record.get("clientAsset") is not None and not isinstance(record["clientAsset"], dict):
        raise ValueError("Invalid persisted client asset")


class JobQueue:
    """Exactly one service owner and one consumer execute jobs in outputRoot."""

    def __init__(self, pipeline: Any, outputRoot: Path, queueCapacity: int):
        if type(queueCapacity) is not int or queueCapacity < 1:
            raise ValueError("queueCapacity must be a positive integer")
        self.pipeline = pipeline
        self.outputRoot = Path(outputRoot).resolve()
        self.pending: queue.Queue[str | None] = queue.Queue(maxsize=queueCapacity)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.accepting = False
        self.activeJobId: str | None = None
        self.worker = threading.Thread(target=self.work, name="faceAppearanceGpuWorker", daemon=False)
        self._ownerDescriptor: int | None = None
        self._started = False
        self._closed = False
        self._closeLock = threading.Lock()

    def acquireOwnership(self) -> None:
        """Reserve outputRoot before loading models; idempotent for this queue."""
        with self.lock:
            if self._closed:
                raise RuntimeError("Job queue has already been closed")
            if self._ownerDescriptor is not None:
                return
            self.outputRoot.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.outputRoot / ".service.lock",
                                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RuntimeError("Service ownership lock must be a regular file")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("Another service already owns this outputRoot") from error
                self._ownerDescriptor = descriptor
            except BaseException:
                self._ownerDescriptor = None
                os.close(descriptor)
                raise

    def releaseOwnership(self) -> None:
        """Release an unused reservation, or ownership after the worker exits."""
        with self.lock:
            if self.worker.is_alive():
                raise RuntimeError("Cannot release ownership while the worker is running")
            if self._ownerDescriptor is not None:
                fcntl.flock(self._ownerDescriptor, fcntl.LOCK_UN)
                os.close(self._ownerDescriptor)
                self._ownerDescriptor = None

    def start(self) -> None:
        with self.lock:
            if self._started or self._closed:
                raise RuntimeError("Job queue has already been started or closed")
            self.acquireOwnership()
            try:
                self.recover()
                self.worker.start()
                self._started = True
                self.accepting = True
            except BaseException:
                self.releaseOwnership()
                raise

    def recover(self) -> None:
        """Load safe records while holding the exclusive service ownership lock."""
        if self._ownerDescriptor is None:
            raise RuntimeError("Job recovery requires service ownership")
        recovered = {}
        for jobDir in sorted(self.outputRoot.iterdir()):
            if jobDir.is_symlink() or not jobDir.is_dir() or not validJobId(jobDir.name):
                continue
            recordPath = jobDir / "job.json"
            if recordPath.is_symlink() or not recordPath.is_file():
                continue
            try:
                record = json.loads(recordPath.read_text(encoding="utf-8"), parse_constant=rejectJsonConstant)
                validateRecord(record, jobDir.name)
            except (OSError, ValueError, TypeError):
                logger.warning("Ignoring unreadable or invalid job record for %s", jobDir.name)
                continue
            record.setdefault("clientContract", False)
            record.setdefault("clientAsset", None)
            if record["status"] in {"queued", "running"}:
                record.update(status="failed", finishedAt=timestamp(), result=None, artifacts=[],
                              clientAsset=None, error={"code": "service_restarted",
                              "type": "ServiceRestarted",
                              "message": "Service restarted before this job completed; submit a new job"})
                # Fail startup if interruption cannot be persisted consistently.
                writeJobRecord(jobDir, record)
            elif record["status"] == "failed":
                record.update(result=None, artifacts=[], clientAsset=None)
            elif record["clientContract"]:
                from .clientContract import InvalidClientAssetError, validateClientOutput

                outputDir = jobDir / "output"
                try:
                    if outputDir.is_symlink() or not outputDir.resolve().is_relative_to(jobDir.resolve()):
                        raise InvalidClientAssetError("Client output escaped the job directory")
                    clientAsset = validateClientOutput(outputDir)
                    if clientAsset != record["clientAsset"]:
                        raise InvalidClientAssetError("Persisted client asset identity differs from its package")
                except (OSError, ValueError, TypeError, KeyError) as error:
                    record.update(status="failed", finishedAt=timestamp(), result=None, artifacts=[],
                                  clientAsset=None, error={"code": "invalid_client_asset",
                                  "type": "InvalidClientAssetError",
                                  "message": "Saved client asset package is missing or failed integrity checks"})
                    writeJobRecord(jobDir, record)
            recovered[jobDir.name] = record
        self.jobs = recovered

    def close(self) -> None:
        """Reject submissions, drain accepted work, and release directory ownership."""
        with self._closeLock:
            with self.lock:
                if self._closed:
                    return
                self.accepting = False
            if self._started:
                self.pending.join()
                self.pending.put(None)
                self.worker.join()
            with self.lock:
                self.releaseOwnership()
                self._closed = True

    def submit(self, jobId: str, incomingDir: Path, originalNames: list[str],
               stopAfter: str, *, clientContract: bool = False) -> dict[str, Any]:
        with self.lock:
            if not self.accepting:
                raise RuntimeError("Service is not accepting new jobs")
            if not validJobId(jobId):
                raise ValueError("Invalid job ID")
            if stopAfter not in jobStages:
                raise ValueError("Invalid job stage")
            if self.pending.full():
                raise queue.Full
            incomingDir = Path(incomingDir)
            if (incomingDir.is_symlink() or not incomingDir.is_dir()
                    or incomingDir.resolve().parent != self.outputRoot
                    or incomingDir.name != f".incoming-{jobId}"):
                raise ValueError("Invalid incoming job directory")
            jobDir = self.outputRoot / jobId
            if jobId in self.jobs or jobDir.exists() or jobDir.is_symlink():
                raise FileExistsError("Job ID already exists")
            record = {"jobId": jobId, "status": "queued", "createdAt": timestamp(),
                      "startedAt": None, "finishedAt": None, "stopAfter": stopAfter,
                      "numImages": len(originalNames), "inputNames": list(originalNames),
                      "artifacts": [], "result": None, "error": None,
                      "clientContract": clientContract, "clientAsset": None}
            # Persist before publishing the directory or in-memory job. A failed
            # write leaves only the incoming directory, which the uploader owns.
            writeJobRecord(incomingDir, record)
            incomingDir.replace(jobDir)
            self.jobs[jobId] = record
            try:
                self.pending.put_nowait(jobId)
            except BaseException:
                self.jobs.pop(jobId, None)
                jobDir.replace(incomingDir)
                raise
            return copy.deepcopy(record)

    def get(self, jobId: str) -> dict[str, Any] | None:
        with self.lock:
            record = self.jobs.get(jobId)
            return copy.deepcopy(record) if record else None

    def update(self, jobId: str, **values: Any) -> None:
        with self.lock:
            record = {**self.jobs[jobId], **values}
            writeJobRecord(self.outputRoot / jobId, record)
            self.jobs[jobId] = record

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {"status": "ready" if self.accepting else "draining",
                    "modelsLoaded": bool(self.pipeline.loaded),
                    "queueDepth": self.pending.qsize(), "queueCapacity": self.pending.maxsize,
                    "activeJobId": self.activeJobId, "concurrency": 1}

    def work(self) -> None:
        while True:
            jobId = self.pending.get()
            try:
                if jobId is None:
                    return
                self.runJob(jobId)
            except Exception:
                # Keep the consumer alive even if failure reporting itself fails.
                logger.exception("Unexpected worker failure for job %s", jobId)
            finally:
                self.pending.task_done()

    def runJob(self, jobId: str) -> None:
        jobDir = self.outputRoot / jobId
        outputDir = jobDir / "output"
        try:
            with self.lock:
                self.activeJobId = jobId
            self.update(jobId, status="running", startedAt=timestamp())
            imagePaths = sorted((jobDir / "inputs").iterdir())
            result = self.pipeline.run(imagePaths, outputDir=outputDir,
                                       stopAfter=self.jobs[jobId]["stopAfter"])
            encodedResult = json.loads(json.dumps(result, default=jsonValue, allow_nan=False))
            resolvedOutput = outputDir.resolve()
            if outputDir.is_symlink() or not resolvedOutput.is_relative_to(jobDir.resolve()):
                raise RuntimeError("Pipeline output directory escaped its job directory")
            clientAsset = None
            if self.jobs[jobId].get("clientContract", False):
                from .clientContract import InvalidClientAssetError, validateClientOutput

                clientAsset = validateClientOutput(outputDir)
                manifest = self.pipeline.assemblyAssets.manifest
                if any(clientAsset[key] != manifest[key] for key in ("templateId", "templateHashes")):
                    raise InvalidClientAssetError("Exported asset differs from the configured template")
            artifacts = []
            for artifactPath in sorted(outputDir.rglob("*")):
                if (artifactPath.is_file() and not artifactPath.is_symlink()
                        and artifactPath.resolve().is_relative_to(resolvedOutput)):
                    relativeName = artifactPath.relative_to(outputDir).as_posix()
                    digest = hashlib.sha256()
                    with artifactPath.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    artifacts.append({"filename": relativeName, "sizeBytes": artifactPath.stat().st_size,
                                      "sha256": digest.hexdigest(),
                                      "mediaType": mimetypes.guess_type(relativeName)[0] or "application/octet-stream",
                                      "url": f"/jobs/{jobId}/artifacts/{quote(relativeName, safe='/')}"})
            self.update(jobId, status="succeeded", finishedAt=timestamp(), error=None,
                        result=encodedResult, artifacts=artifacts, clientAsset=clientAsset)
        except Exception as error:
            traceText = traceback.format_exc()
            logger.error("Reconstruction job %s failed: %s", jobId, type(error).__name__)
            try:
                (jobDir / "traceback.txt").write_text(traceText, encoding="utf-8")
            except OSError:
                logger.exception("Could not save traceback for job %s", jobId)
            try:
                message = str(error).replace(str(self.outputRoot), "<jobs>")
                projectRoot = getattr(getattr(self.pipeline, "config", None), "projectRoot", None)
                if projectRoot is not None:
                    message = message.replace(str(projectRoot), "<project>")
                self.update(jobId, status="failed", finishedAt=timestamp(), result=None,
                            artifacts=[], clientAsset=None, error={
                                "code": getattr(error, "code", "reconstruction_failed"),
                                "type": type(error).__name__, "message": message[:1000]})
            except Exception:
                logger.exception("Could not persist failure for job %s", jobId)
                with self.lock:
                    self.jobs[jobId].update(status="failed", finishedAt=timestamp(), result=None,
                                           artifacts=[], clientAsset=None, error={
                                               "code": "job_storage_error", "type": "JobStorageError",
                                               "message": "Unable to save the job failure record"})
        finally:
            with self.lock:
                self.activeJobId = None
