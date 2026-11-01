"""Read-only local comparison of complete client assets, including eyeballs."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from faceAppearance.legacyPreviewAsset import loadLegacyPreview, readLegacyTexture

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = {"baseline": "修正前结果", "eyes": "仅修正眼球", "texture": "仅修正贴图边界", "combined": "两项修正"}
DEFAULT_CONFIG = {
    "title": "眼球位置与贴图接缝",
    "subtitle": "同一次线上重建结果，分开查看眼球配准与边界融合的影响。拖动、缩放两个窗口会同步。",
    "brand": "外观修正对照",
    "badge": "同一任务 · 四分支对照",
    "branches": [{"id": key, "label": value} for key, value in BRANCHES.items()],
    "defaultLeft": "baseline", "defaultRight": "combined",
    "defaultViewerMode": "textured", "defaultRenderMode": "lit",
    "photoDescription": "保留原照片的拍摄光照，按重建时的同一裁剪比较。模型沿用估计相机；可叠加原照观察轮廓与五官位置。",
    "overviewDescription": "同一组照片与固定相机下的修正前后结果。",
    "notes": [
        {"title": "眼球", "text": "依据眼缘移动球心并约束大小，保留模板虹膜方向；灰模和侧面便于观察穿插。"},
        {"title": "贴图", "text": "在真实覆盖外边界用安全肤色外扩，再逐渐过渡到模板；眼睛、眉毛和嘴部单独保护。"},
        "四个分支使用相同皮肤几何、相机和客户端 UV。此页包含完整头部附件；尚未覆盖其他样本和表情。",
    ],
}


def readConfig(directory: Path) -> dict:
    """Read display text and an explicit asset allowlist; never accept paths from config."""
    path = directory / "config.json"
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("comparison/config.json 必须是对象")
    config = dict(DEFAULT_CONFIG)
    if "branches" in raw:
        config.update({
            "title": "客户端资产对照", "brand": "分支对照",
            "subtitle": "同一组输入下的各分支结果。拖动、缩放两个窗口会同步。",
            "badge": "同一任务 · 分支对照",
            "overviewDescription": "同一组照片与固定相机下的各分支结果。",
        })
    for key in ("title", "subtitle", "brand", "badge", "photoDescription", "overviewDescription"):
        if key in raw:
            if not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 2000:
                raise ValueError(f"config.json 的 {key} 必须是非空文字")
            config[key] = raw[key]
    options = raw.get("branches", config["branches"])
    if not isinstance(options, list) or not 2 <= len(options) <= 16:
        raise ValueError("config.json 的 branches 应包含 2–16 个分支")
    ids, branches = set(), []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("config.json 的分支必须包含 id 与 label")
        key, label = option.get("id"), option.get("label")
        if (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key)
                or key in ids or not isinstance(label, str) or not label.strip() or len(label) > 160):
            raise ValueError("config.json 的分支 id 或 label 无效")
        ids.add(key)
        branches.append({"id": key, "label": label})
    config["branches"] = branches
    for field, index in (("defaultLeft", 0), ("defaultRight", 1)):
        selected = raw.get(field, config[field] if config[field] in ids else branches[index]["id"])
        if not isinstance(selected, str) or selected not in ids:
            raise ValueError(f"config.json 的 {field} 必须在 branches 中")
        config[field] = selected
    for field, modes in (("defaultViewerMode", ("textured", "baseColor", "clay")),
                         ("defaultRenderMode", ("lit", "baseColor", "clay"))):
        if field in raw:
            if raw[field] not in modes:
                raise ValueError(f"config.json 的 {field} 无效")
            config[field] = raw[field]
    if "notes" in raw:
        notes = raw["notes"]
        if not isinstance(notes, list) or len(notes) > 20:
            raise ValueError("config.json 的 notes 应为文字或 title/text 对象数组")
        for note in notes:
            if isinstance(note, str) and len(note) <= 4000:
                continue
            if (isinstance(note, dict) and isinstance(note.get("title"), str)
                    and isinstance(note.get("text"), str) and len(note["title"]) <= 160 and len(note["text"]) <= 4000):
                continue
            raise ValueError("config.json 的 notes 内容无效")
        config["notes"] = notes
    elif "branches" in raw:
        config["notes"] = []
    return config


def createApp(directory: Path) -> FastAPI:
    directory = directory.resolve()
    config = readConfig(directory)
    branch_ids = {option["id"] for option in config["branches"]}
    app = FastAPI(title=config["title"], docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        return response

    def archive(branch):
        if branch not in branch_ids:
            raise HTTPException(404, "未知分支")
        path = (directory / branch / "legacyAsset.zip").resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(503, "本分支尚未生成")
        return path

    @lru_cache(maxsize=24)
    def preview(path, modified, branch):
        return loadLegacyPreview(path, branch)

    @lru_cache(maxsize=32)
    def texture(path, modified, index):
        return readLegacyTexture(path, index)

    @app.get("/api/branches")
    def branches():
        return config["branches"]

    @app.get("/api/config")
    def read_config():
        return config

    @app.get("/api/jobs/{branch}/preview")
    def read_preview(branch: str):
        path = archive(branch)
        return preview(path, path.stat().st_mtime_ns, branch)

    @app.get("/api/jobs/{branch}/textures/{index}")
    def read_texture(branch: str, index: int):
        if not 0 <= index < 4:
            raise HTTPException(404, "未知贴图")
        path = archive(branch)
        return Response(texture(path, path.stat().st_mtime_ns, index), media_type="image/png")

    @app.get("/api/jobs/{branch}/asset.zip")
    def download(branch: str):
        return FileResponse(archive(branch), media_type="application/zip", filename=branch+".zip")

    def evidence_manifest():
        try:
            data = json.loads((directory/"evidence/manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(503, "固定视角渲染正在生成，请稍后刷新对照。") from None
        if (not isinstance(data, dict) or data.get("schemaVersion") != 1
                or not isinstance(data.get("views"), list) or not data["views"]):
            raise HTTPException(409, "照片对照清单无效")
        files, ids = set(), set()

        def add_file(name):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.png", name):
                raise HTTPException(409, "照片对照文件名无效")
            files.add(name)

        for view in data["views"]:
            if (not isinstance(view, dict) or not re.fullmatch(r"view[0-9]+", str(view.get("id", "")))
                    or view["id"] in ids or not isinstance(view.get("renders"), dict)):
                raise HTTPException(409, "照片视角信息无效")
            ids.add(view["id"])
            add_file(view.get("photo"))
            for optional in ("original", "comparison"):
                if view.get(optional):
                    add_file(view[optional])
            for branch in branch_ids:
                render = view["renders"].get(branch, {})
                if not isinstance(render, dict):
                    raise HTTPException(409, "照片对照分支无效")
                for mode in ("baseColor", "lit", "clay"):
                    add_file(render.get(mode))
        if data.get("overview"):
            add_file(data["overview"])
        return data, files

    @app.get("/api/evidence")
    def read_evidence():
        return evidence_manifest()[0]

    @app.get("/evidence/{filename}")
    def evidence_file(filename: str):
        _, files = evidence_manifest()
        if filename not in files:
            raise HTTPException(404, "未知渲染图")
        root = (directory/"evidence").resolve()
        path = (root/filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "渲染图尚未生成")
        return FileResponse(path, media_type="image/png")

    app.mount("/viewer-common", StaticFiles(directory=ROOT/"src/faceAppearance/web"), name="common")
    app.mount("/", StaticFiles(directory=ROOT/"src/faceAppearance/clientAppearanceWeb", html=True), name="comparison")
    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"runs/clientAppearanceFix20261012/comparison")
    parser.add_argument("--port", type=int, default=18099)
    args = parser.parse_args()
    uvicorn.run(createApp(args.directory), host="127.0.0.1", port=args.port, access_log=False)
