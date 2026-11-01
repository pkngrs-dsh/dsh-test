"""Opt-in callback receiver for local previews of the deployed legacy service.

The SSH integration helper registers one random token and task ID in private
storage before submission. This receiver accepts only that task's legacy
callback; it neither executes tasks nor proxies arbitrary destinations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from threading import Lock
import time

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse


TOKEN = re.compile(r"[a-f0-9]{64}\Z")
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
INPUT_INDEX = re.compile(r"(?:[0-9]|1[0-5])\Z")
FIELDS = {"task_id", "status", "sh_coeffs", "result_url"}
MAX_BODY_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _rejectConstant(_):
    raise ValueError("Non-finite JSON number")


def _readJson(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_rejectConstant)


def _registration(root, token):
    if TOKEN.fullmatch(token) is None:
        raise HTTPException(404, "联调登记不存在")
    directory = root / token
    registration = directory / "registration.json"
    if directory.is_symlink() or registration.is_symlink():
        raise HTTPException(404, "联调登记不存在")
    try:
        if not stat.S_ISREG(registration.lstat().st_mode):
            raise ValueError
        registered = _readJson(registration)
    except (OSError, ValueError):
        raise HTTPException(404, "联调登记不存在") from None
    if (not isinstance(registered, dict) or not isinstance(registered.get("task_id"), str)
            or TASK_ID.fullmatch(registered["task_id"]) is None):
        raise HTTPException(404, "联调登记不存在")
    return directory, registered


def _validatePayload(payload, taskId):
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise HTTPException(400, "回调字段必须符合旧协议")
    if payload["task_id"] != taskId or not isinstance(payload["task_id"], str):
        raise HTTPException(400, "回调任务编号不匹配")
    if payload["status"] not in ("completed", "failed"):
        raise HTTPException(400, "回调状态无效")
    coefficients = payload["sh_coeffs"]
    if (not isinstance(coefficients, list) or len(coefficients) != 16
            or any(type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))
                   for value in coefficients)):
        raise HTTPException(400, "回调光照系数必须包含 16 个有限数值")
    if not isinstance(payload["result_url"], str):
        raise HTTPException(400, "回调结果地址必须为字符串")


def _writePrivateJson(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".callback-", suffix=".json", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def registerIntegrationRelay(app):
    """Mount the receiver only when its private storage directory is configured."""
    configured = os.environ.get("FACE_APPEARANCE_INTEGRATION_ROOT")
    if not configured:
        return
    root = Path(configured).expanduser().resolve()
    writeLock = Lock()

    @app.get("/_integration/inputs/{token}/{index}", include_in_schema=False)
    async def readIntegrationInput(token: str, index: str):
        directory, registered = _registration(root, token)
        expiresAt = registered.get("inputExpiresAt")
        if (INPUT_INDEX.fullmatch(index) is None or type(expiresAt) not in (int, float)
                or (type(expiresAt) is float and not math.isfinite(expiresAt))
                or expiresAt <= time.time()):
            raise HTTPException(404, "联调照片不存在")
        inputs = directory / "inputs"
        path = inputs / f"{int(index):02d}.image"
        try:
            metadata = path.lstat()
            if (inputs.is_symlink() or not stat.S_ISREG(metadata.st_mode)
                    or not 0 < metadata.st_size <= MAX_IMAGE_BYTES
                    or path.resolve(strict=True).parent != inputs
                    or not path.resolve(strict=True).is_relative_to(root)):
                raise ValueError
        except (OSError, ValueError):
            raise HTTPException(404, "联调照片不存在") from None
        return FileResponse(path, media_type="application/octet-stream", stat_result=metadata,
                            headers={"Cache-Control": "private, no-store"})

    @app.post("/_integration/callback/{token}", include_in_schema=False)
    async def receiveIntegrationCallback(token: str, request: Request):
        directory, registered = _registration(root, token)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                raise HTTPException(413, "回调内容超过 64 KiB")
        try:
            payload = json.loads(body, parse_constant=_rejectConstant)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "回调必须为有效 JSON") from None
        _validatePayload(payload, registered["task_id"])
        callbackPath = directory / "callback.json"
        with writeLock:
            if callbackPath.is_symlink():
                raise HTTPException(409, "回调记录冲突")
            if callbackPath.exists():
                try:
                    saved = _readJson(callbackPath)
                except (OSError, ValueError):
                    raise HTTPException(409, "回调记录冲突") from None
                if not isinstance(saved, dict) or saved.get("payload") != payload:
                    raise HTTPException(409, "回调记录冲突")
            else:
                _writePrivateJson(callbackPath, {
                    "receivedAt": datetime.now(timezone.utc).isoformat(), "payload": payload,
                })
        return {"status": "received", "task_id": registered["task_id"]}
