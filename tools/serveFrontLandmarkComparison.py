"""Read-only localhost viewer for client frontal-landmark postprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
ARTIFACTS = {"preview.json", "frontOverlay.png", "mouth.png", "face.png", "side.png",
             "report.json", "meshes.npz", "appearance.json", "skinTexture.png", "sourceTexture.png",
             "texturedComparison.png", "texturedLitComparison.png"}


def createApp(experiment: Path = ROOT / "runs/frontLandmark20261012") -> FastAPI:
    experiment = Path(experiment).resolve()
    app = FastAPI(title="客户端正面 landmark 后处理", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "same-origin"})
        return response

    def manifest():
        try:
            result = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(503, "实验结果尚未就绪，请稍后刷新。") from None
        if not isinstance(result.get("cases"), list):
            raise HTTPException(409, "实验清单格式无效。")
        return result

    @app.get("/api/experiment")
    def experiment_data():
        data = manifest()
        # Do not publish cached input paths or legacy task transport metadata.
        keys = ("id", "gender", "label", "captureMode", "previewUrl", "status", "qualityApproved")
        return {key: data.get(key) for key in ("status", "objective", "solver")} | {
            "cases": [{key: case[key] for key in keys if key in case} for case in data["cases"]]}

    @app.get("/data/{case_id}/{filename}")
    def artifact(case_id: str, filename: str):
        if not CASE_ID.fullmatch(case_id) or filename not in ARTIFACTS:
            raise HTTPException(404, "未知实验资源。")
        if case_id not in {case.get("id") for case in manifest()["cases"]}:
            raise HTTPException(404, "未知实验组。")
        path = (experiment / case_id / filename).resolve()
        if not path.is_relative_to(experiment) or not path.is_file():
            raise HTTPException(404, "实验资源尚未生成。")
        return FileResponse(path, filename=filename if filename.endswith(".npz") else None)

    app.mount("/viewer-common", StaticFiles(directory=ROOT / "src/faceAppearance/web"), name="common")
    app.mount("/", StaticFiles(directory=ROOT / "src/faceAppearance/frontLandmarkWeb", html=True), name="comparison")
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/frontLandmark20261012")
    parser.add_argument("--port", type=int, default=18095)
    args = parser.parse_args()
    uvicorn.run(createApp(args.experiment), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
