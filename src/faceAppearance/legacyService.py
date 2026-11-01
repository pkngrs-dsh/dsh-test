"""Persistent notebook-compatible HTTP service with one GPU pipeline.

The request contract is the old JSON /generate/ API. Gender selects an explicitly
preloaded client template after shared UFF reconstruction; it never silently
changes the requested template. Secrets and source URLs remain in private state.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import stat
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .jobs import timestamp, writeJobRecord
from .legacyTransport import TransportError, downloadImage, notifyResult, resolveUrl


logger = logging.getLogger(__name__)
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
DIRECTORY_ID = re.compile(r"[a-f0-9]{64}\Z")
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_JSON_BYTES = 256 * 1024
MAX_CALLBACK_ATTEMPTS = 3
REQUIRED_FIELDS = ("task_id", "gender", "upload_dir", "notify_url", "image_urls")


def _directoryId(taskId: str) -> str:
    return hashlib.sha256(taskId.encode("utf-8")).hexdigest()


def _fileHash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _traceLocations(error):
    """Return stack locations without exception text, source lines or locals."""
    return [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)]


def _validateRequest(data, config, *, allowInternal):
    if not isinstance(data, dict):
        raise ValueError("请求体必须为 JSON 对象")
    for name in REQUIRED_FIELDS:
        if not data.get(name):
            raise ValueError(f"缺少 {name}")
    taskId = data["task_id"]
    if not isinstance(taskId, str) or TASK_ID.fullmatch(taskId) is None:
        raise ValueError("task_id 必须为 1–128 位字母、数字、下划线或连字符，并以字母或数字开头")
    gender = data["gender"]
    if not isinstance(gender, str) or gender.lower() not in {"male", "female"}:
        raise ValueError("性别必须是'male'或'female'")
    prefix = data["upload_dir"]
    if (not isinstance(prefix, str) or len(prefix) > 512 or prefix.startswith("/")
            or "\\" in prefix or any(ord(char) < 32 for char in prefix)
            or any(part in {".", ".."} for part in prefix.split("/"))):
        raise ValueError("upload_dir 必须为有效的 OSS 相对目录")
    images = data["image_urls"]
    if not isinstance(images, list) or not config.minViews <= len(images) <= config.maxViews:
        raise ValueError(f"当前 UFF 服务要求 {config.minViews}–{config.maxViews} 张不同视角照片")
    for url in [*images, data["notify_url"]]:
        resolveUrl(url, allowInternal=allowInternal)
    return {"task_id": taskId, "gender": gender.lower(), "upload_dir": prefix,
            "notify_url": data["notify_url"], "image_urls": list(images)}


def _ossConfiguration():
    names = ("FACEV2_OSS_ACCESS_KEY_ID", "FACEV2_OSS_ACCESS_KEY_SECRET",
             "FACEV2_OSS_ENDPOINT", "FACEV2_OSS_BUCKET")
    values = tuple(os.environ.get(name, "") for name in names)
    if any(values) and not all(values):
        raise ValueError("OSS 配置不完整；四个 FACEV2_OSS_* 环境变量须同时配置")
    return values if all(values) else None


class LegacyQueue:
    def __init__(self, pipeline, assets, config, root, capacity, publicBaseUrl, allowInternal, ossConfig,
                 ossResultUrlMode="signed"):
        self.pipeline, self.assets, self.config = pipeline, assets, config
        self.root = Path(root).resolve()
        self.pending = queue.Queue(maxsize=capacity)
        self.publicBaseUrl, self.allowInternal, self.ossConfig = publicBaseUrl, allowInternal, ossConfig
        self.ossResultUrlMode = ossResultUrlMode
        self.records = {}
        self.lock = threading.RLock()
        self.owner = None
        self.accepting = False
        self.started = False
        self.active = None
        self.recoveredCallbacks = []
        self.worker = threading.Thread(target=self._work, name="faceAppearanceLegacyGpu", daemon=False)

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.root / ".legacy-service.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError("Service lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        self.owner = descriptor

    def release(self):
        if self.owner is not None:
            fcntl.flock(self.owner, fcntl.LOCK_UN)
            os.close(self.owner)
            self.owner = None

    def _directory(self, taskId):
        result = self.root / _directoryId(taskId)
        if result.is_symlink() or result.resolve().parent != self.root:
            raise ValueError("Invalid task directory")
        return result

    def _save(self, record):
        writeJobRecord(self._directory(record["task_id"]), record)

    def recover(self):
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or DIRECTORY_ID.fullmatch(directory.name) is None:
                continue
            recordPath = directory / "job.json"
            if recordPath.is_symlink() or not recordPath.is_file():
                continue
            try:
                record = json.loads(recordPath.read_text(encoding="utf-8"))
                taskId = record["task_id"]
                if (not isinstance(taskId, str) or TASK_ID.fullmatch(taskId) is None
                        or _directoryId(taskId) != directory.name
                        or record.get("status") not in {"pending", "processing", "completed", "failed"}
                        or not isinstance(record.get("request"), dict)
                        or not isinstance(record.get("callback"), dict)):
                    raise ValueError
                callback = record["callback"]
                if type(callback.get("attempts")) is not int or not 0 <= callback["attempts"] <= MAX_CALLBACK_ATTEMPTS:
                    raise ValueError
                if record["status"] in {"pending", "processing"}:
                    record.update(status="failed", error="service_restarted: 服务重启中断了此任务，请使用新 task_id 重试",
                                  finishedAt=timestamp(), result_url=None, archive=None)
                    self._save(record)
                elif record["status"] == "completed":
                    try:
                        self._archivePath(record, verify=True)
                    except (ValueError, OSError, KeyError, TypeError):
                        record.update(status="failed", error="saved_asset_invalid: 已保存的结果包缺失或损坏",
                                      result_url=None, archive=None)
                        self._save(record)
                self.records[taskId] = record
                if callback.get("state") != "delivered" and callback["attempts"] < MAX_CALLBACK_ATTEMPTS:
                    self.recoveredCallbacks.append(taskId)
            except (ValueError, TypeError, KeyError, OSError):
                logger.warning("Ignoring invalid legacy record directory %s", directory.name)

    def start(self):
        self.recover()
        self.accepting = True
        self.worker.start()
        self.started = True

    def close(self):
        with self.lock:
            self.accepting = False
        if self.started:
            self.pending.join()
            self.pending.put(None)
            self.worker.join()
        self.release()

    def submit(self, request):
        taskId = request["task_id"]
        with self.lock:
            if not self.accepting:
                raise ValueError("服务尚未就绪或正在停止")
            if self.pending.full():
                raise ValueError("任务队列已满，请稍后重试")
            directory = self._directory(taskId)
            if taskId in self.records or directory.exists():
                raise ValueError("task_id 已存在，请勿重复提交")
            directory.mkdir(mode=0o700)
            record = {"schemaVersion": 1, "task_id": taskId, "status": "pending", "request": request,
                      "createdAt": timestamp(), "startedAt": None, "finishedAt": None,
                      "result_url": None, "error": None, "archive": None,
                      "callback": {"state": "pending", "attempts": 0, "lastError": None}}
            try:
                self._save(record)
                self.records[taskId] = record
                self.pending.put_nowait(taskId)
            except BaseException:
                self.records.pop(taskId, None)
                shutil.rmtree(directory)
                raise

    def get(self, taskId):
        with self.lock:
            return copy.deepcopy(self.records.get(taskId))

    def _update(self, taskId, **values):
        with self.lock:
            previous = self.records[taskId]
            # A caller may subsequently mutate its local callback dictionary.
            # Publish only detached values after the durable write succeeds.
            updated = {**previous, **copy.deepcopy(values)}
            self._save(updated)
            self.records[taskId] = updated
            return updated

    def _images(self, record):
        from PIL import Image
        directory = self._directory(record["task_id"]) / "inputs"
        directory.mkdir(mode=0o700)
        paths, hashes, total = [], set(), 0
        for index, url in enumerate(record["request"]["image_urls"]):
            path = directory / f"image{index:02d}.image"
            total += downloadImage(url, path, maxBytes=min(MAX_IMAGE_BYTES, MAX_TOTAL_BYTES - total),
                                   allowInternal=self.allowInternal)
            digest = _fileHash(path)
            if digest in hashes:
                raise TransportError("输入照片重复，请提供不同视角")
            hashes.add(digest)
            try:
                with Image.open(path) as source:
                    width, height = source.size
                    if min(width, height) < 64 or width * height > 32_000_000:
                        raise ValueError
                    source.verify()
                with Image.open(path) as source:
                    source.load()
            except (OSError, ValueError, Image.DecompressionBombError):
                raise TransportError(f"第 {index + 1} 张照片无效、过小或超过 3200 万像素") from None
            paths.append(path)
        return paths

    def _upload(self, archive, request):
        if self.ossConfig is None:
            return f"{self.publicBaseUrl}/tasks/{request['task_id']}/download"
        import oss2
        accessId, secret, endpoint, bucketName = self.ossConfig
        bucket = oss2.Bucket(oss2.Auth(accessId, secret), endpoint, bucketName, connect_timeout=30)
        key = f"{request['upload_dir'].rstrip('/')}/{request['task_id']}.zip"
        try:
            result = bucket.put_object_from_file(key, str(archive))
            if not 200 <= result.status < 300:
                raise ValueError
            resultUrl = bucket.sign_url("GET", key, 3600)
            if self.ossResultUrlMode == "public":
                parts = urlsplit(resultUrl)
                resultUrl = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            return resultUrl
        except Exception:
            raise RuntimeError("OSS 上传失败") from None

    def _process(self, taskId):
        import numpy as np
        from .legacyAsset import exportLegacyAsset
        record = self._update(taskId, status="processing", startedAt=timestamp())
        directory = self._directory(taskId)
        try:
            paths = self._images(record)
            output = directory / "reconstruction"
            report = self.pipeline.run(paths, output, stopAfter="fusion")
            with np.load(output / "geometry.npz", allow_pickle=False) as saved:
                geometry = {key: saved[key].copy() for key in saved.files}
            with np.load(output / "texture.npz", allow_pickle=False) as saved:
                texture = {key: saved[key].copy() for key in saved.files}
            gender = record["request"]["gender"]
            encoding = "linear" if "linearAlbedo" in texture else self.config.assemblyColorEncoding
            if gender == "male" and self.config.legacyMaleAssemblyMode == "directColmap":
                from .directClientService import assembleDirectClient
                assembled, clientReport = assembleDirectClient(
                    self.assets[gender], geometry, texture, output, paths, self.config)
                report["clientPipeline"] = clientReport["pipeline"]
                report["clientSeconds"] = clientReport["seconds"]
                report["stageSeconds"].update(clientReport["stageSeconds"])
                report["inferenceSeconds"] += clientReport["seconds"]
                from .artifacts import writeJson
                writeJson(output / "result.json", report)
            else:
                assembled = self.assets[gender].assemble(geometry, texture, colorEncoding=encoding)
            exported = exportLegacyAsset(directory / "asset", assembled, gender=gender)
            archivePath = Path(exported["archivePath"]).resolve()
            if not archivePath.is_relative_to(directory) or not archivePath.is_file():
                raise ValueError("Invalid legacy export location")
            archive = {"path": archivePath.relative_to(directory).as_posix(),
                       "sha256": exported["archiveSha256"], "sizeBytes": exported["sizeBytes"],
                       "files": exported["files"]}
            resultUrl = self._upload(archivePath, record["request"])
            self._update(taskId, status="completed", result_url=resultUrl, archive=archive,
                         metrics={key: report[key] for key in ("loadSeconds", "inferenceSeconds", "stageSeconds",
                                  "peakAllocatedGiB", "acceptedCameraCount", "clientPipeline", "clientSeconds") if key in report},
                         finishedAt=timestamp())
        except Exception as error:
            # URL/token-bearing library exception strings never enter responses
            # or logs. Known input errors have deliberately URL-free messages.
            message = str(error) if isinstance(error, TransportError) else "重建或资产交付失败，请检查照片或联系维护者"
            locations = _traceLocations(error)
            logger.error("Legacy task %s failed (%s), stack locations: %s", taskId, type(error).__name__, locations)
            self._update(taskId, status="failed", error=message, result_url=None, archive=None,
                         finishedAt=timestamp(), failureType=type(error).__name__, failureLocations=locations)
        finally:
            # Pipeline diagnostics include derived face crops; keep these with
            # private task state. Delete downloaded original files after use.
            shutil.rmtree(directory / "inputs", ignore_errors=True)

    def _notify(self, taskId):
        record = self.get(taskId)
        if record["status"] not in {"completed", "failed"} or record["callback"].get("state") == "delivered":
            return
        payload = {"task_id": taskId, "status": record["status"], "sh_coeffs": [0.0] * 16,
                   "result_url": record["result_url"] if record["status"] == "completed" else "error"}
        attempts = record["callback"]["attempts"]
        for attempt in range(attempts + 1, MAX_CALLBACK_ATTEMPTS + 1):
            if attempt > attempts + 1:
                time.sleep(min(2 ** (attempt - 1), 4))
            callback = {"state": "sending", "attempts": attempt, "lastError": None, "attemptedAt": timestamp()}
            self._update(taskId, callback=callback)
            try:
                code = notifyResult(record["request"]["notify_url"], payload, allowInternal=self.allowInternal)
                callback.update(state="delivered", httpStatus=code, deliveredAt=timestamp())
                self._update(taskId, callback=callback)
                return
            except Exception as error:
                callback.update(state="failed" if attempt == MAX_CALLBACK_ATTEMPTS else "retrying",
                                lastError=str(error) if isinstance(error, TransportError) else "回调连接失败")
                self._update(taskId, callback=callback)
                logger.warning("Legacy callback for %s failed on attempt %s", taskId, attempt)

    def _work(self):
        for taskId in self.recoveredCallbacks:
            try:
                self._notify(taskId)
            except Exception:
                logger.error("Could not persist recovered callback for %s", taskId)
                self.accepting = False
        while True:
            taskId = self.pending.get()
            try:
                if taskId is None:
                    return
                self.active = taskId
                self._process(taskId)
                self._notify(taskId)
            except Exception as error:
                # Storage failure stops future acceptance. Accepted records stay
                # persistent and restart recovery explicitly fails active jobs.
                logger.error("Legacy worker persistence failure for %s (%s)", taskId, type(error).__name__)
                self.accepting = False
            finally:
                self.active = None
                self.pending.task_done()

    def _archivePath(self, record, *, verify=False):
        directory = self._directory(record["task_id"])
        metadata = record["archive"]
        path = directory / metadata["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(directory)
                or not path.is_file() or path.stat().st_size != metadata["sizeBytes"]):
            raise ValueError("Saved archive is missing or invalid")
        if verify and _fileHash(path) != metadata["sha256"]:
            raise ValueError("Saved archive hash differs")
        return path


def createLegacyApp(config, *, outputRoot=None, queueCapacity=2, publicBaseUrl=None,
                    maleAssemblyAsset="assets/clientTemplates/legacyMaleV1",
                    femaleAssemblyAsset="assets/clientTemplates/legacyFemaleV1", pipelineFactory=None):
    """Create an app without importing inference models or initializing CUDA."""
    if type(queueCapacity) is not int or queueCapacity < 1:
        raise ValueError("queueCapacity must be a positive integer")
    publicBaseUrl = publicBaseUrl or os.environ.get("FACE_APPEARANCE_PUBLIC_BASE_URL") or os.environ.get("FACEV2_SERVICE__PUBLIC_BASE_URL")
    if publicBaseUrl:
        parts = urlsplit(publicBaseUrl)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or any(ord(char) <= 32 for char in publicBaseUrl)):
            raise ValueError("publicBaseUrl must be an HTTP(S) service base without credentials, query or fragment")
        publicBaseUrl = publicBaseUrl.rstrip("/")
    ossConfig = _ossConfiguration()
    ossResultUrlMode = os.environ.get("FACE_APPEARANCE_OSS_RESULT_URL_MODE", "signed")
    if ossResultUrlMode not in {"signed", "public"}:
        raise ValueError("FACE_APPEARANCE_OSS_RESULT_URL_MODE must be signed or public")
    if not publicBaseUrl and ossConfig is None:
        raise ValueError("Configure publicBaseUrl or complete FACEV2_OSS_* settings so clients can download results")
    allowInternal = os.environ.get("FACE_APPEARANCE_ALLOW_INTERNAL_URLS", "0") == "1"
    root = Path(outputRoot or config.projectRoot / "runs/legacyService").resolve()
    inferenceConfig = replace(config, enableAssembly=False)

    @asynccontextmanager
    async def lifespan(app):
        from .headAssembly import AssemblyAssets
        factory = pipelineFactory
        if factory is None:
            from .pipeline import Pipeline
            factory = Pipeline
        pipeline = factory(inferenceConfig)
        worker = LegacyQueue(pipeline, {}, config, root, queueCapacity, publicBaseUrl, allowInternal, ossConfig,
                             ossResultUrlMode=ossResultUrlMode)
        worker.acquire()
        try:
            for gender, asset in (("male", maleAssemblyAsset), ("female", femaleAssemblyAsset)):
                loaded = await run_in_threadpool(AssemblyAssets, config.projectRoot / asset)
                if loaded.manifest.get("gender") != gender:
                    raise ValueError(f"{gender} assembly asset has a different declared gender")
                worker.assets[gender] = loaded
            if config.legacyMaleAssemblyMode == "directColmap":
                from .directClientService import checkDirectRuntime
                worker.directRuntime = await run_in_threadpool(checkDirectRuntime, config, worker.assets["male"])
            await run_in_threadpool(pipeline.loadModels)
            if not pipeline.loaded:
                raise RuntimeError("Pipeline did not finish loading models")
            from .startupWarmup import runStartupWarmup
            app.state.startupWarmup = await run_in_threadpool(runStartupWarmup, pipeline, config,
                                                             legacyAssets=worker.assets)
            worker.start()
            app.state.legacyQueue = worker
            yield
        finally:
            await run_in_threadpool(worker.close)

    app = FastAPI(title="FaceAppearance notebook-compatible reconstruction", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def getQueue():
        worker = getattr(app.state, "legacyQueue", None)
        if worker is None:
            raise HTTPException(503, "服务尚未就绪")
        return worker

    @app.get("/hello/")
    def hello():
        return {"result": "hello"}

    @app.get("/health")
    def health():
        worker = getQueue()
        return {"status": "ok" if worker.accepting else "not_accepting", "models": "loaded",
                "queue_depth": worker.pending.qsize(), "concurrency": 1, "active_task": worker.active,
                "accepting_jobs": worker.accepting, "protocol": "notebook-v1",
                "templates": {gender: asset.manifest["templateId"] for gender, asset in worker.assets.items()},
                "assembly_modes": {"male": config.legacyMaleAssemblyMode, "female": "uvTransfer"},
                "male_jaw_postprocessing": ("frontWeightedSafeV1"
                    if config.enableRenderedJawRefinement else "disabled"),
                "startupWarmup": app.state.startupWarmup,
                "image_limits": {"min": config.minViews, "max": config.maxViews}}

    @app.post("/generate/")
    async def generate(request: Request):
        data = None
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_JSON_BYTES:
                    raise ValueError("JSON 请求体超过 256 KiB")
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError):
                raise ValueError("请求体必须为有效 JSON") from None
            validated = await run_in_threadpool(_validateRequest, data, config, allowInternal=allowInternal)
            await run_in_threadpool(getQueue().submit, validated)
            return {"status": "queued", "task_id": validated["task_id"], "message": "重建任务已加入处理队列"}
        except Exception as error:
            taskId = data.get("task_id", "unknown") if isinstance(data, dict) else "unknown"
            if not isinstance(taskId, str) or len(taskId) > 128:
                taskId = "unknown"
            message = str(error) if isinstance(error, ValueError) else "服务暂时无法接受任务"
            return JSONResponse(status_code=400, content={"status": "error", "task_id": taskId, "message": message})

    @app.get("/task/status/{task_id}")
    def taskStatus(task_id: str):
        record = getQueue().get(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"message": "任务不存在"})
        response = {"task_id": task_id, "status": record["status"]}
        if record["status"] == "completed":
            response["result_url"] = record["result_url"]
        elif record["status"] == "failed":
            response["error"] = record["error"]
        return response

    def completed(taskId):
        record = getQueue().get(taskId)
        if record is None:
            raise HTTPException(404, "任务不存在")
        if record["status"] != "completed":
            raise HTTPException(409, "任务尚未成功完成")
        return record

    @app.get("/tasks/{task_id}/download")
    def download(task_id: str):
        record = completed(task_id)
        try:
            path = getQueue()._archivePath(record, verify=True)
        except (ValueError, OSError, KeyError, TypeError):
            raise HTTPException(410, "结果包缺失或完整性校验失败") from None
        return FileResponse(path, media_type="application/zip", filename=f"{task_id}.zip",
                            headers={"ETag": '"' + record["archive"]["sha256"] + '"'})

    @app.get("/tasks/{task_id}/files")
    def files(task_id: str):
        record = completed(task_id)
        return {"task_id": task_id, "files": record["archive"]["files"]}

    from .integrationRelay import registerIntegrationRelay
    registerIntegrationRelay(app)
    return app
