"""Read-only localhost comparison of the frozen male frontal mouth texture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "maleExactRegression"
ARTIFACTS = {
    "preview.json", "appearance.json", "report.json", "overview.png",
    "mouth.png", "side.png", "diagnostic.png", "coverage.png", "white.png",
    *(f"branch{index}.png" for index in range(6)),
}


def createApp(experiment: Path = ROOT / "runs/maleMouthSeam20261012/final") -> FastAPI:
    experiment = Path(experiment).resolve()
    app = FastAPI(title="男生样本 · 正面嘴部取色对照", docs_url=None, redoc_url=None, openapi_url=None)
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
        if not isinstance(result, dict) or not isinstance(result.get("cases"), list):
            raise HTTPException(409, "实验清单格式无效。")
        return result

    @app.get("/api/experiment")
    def experiment_data():
        data = manifest()
        keys = ("id", "label", "previewUrl", "status", "captureMode")
        return {key: data.get(key) for key in ("status", "objective", "method")} | {
            "cases": [{key: case[key] for key in keys if key in case}
                      for case in data["cases"] if isinstance(case, dict) and case.get("id") == CASE_ID]}

    @app.get("/data/{case_id}/{filename}")
    def artifact(case_id: str, filename: str):
        if case_id != CASE_ID or filename not in ARTIFACTS:
            raise HTTPException(404, "未知实验资源。")
        if case_id not in {case.get("id") for case in manifest()["cases"] if isinstance(case, dict)}:
            raise HTTPException(404, "未知实验组。")
        case_path = (experiment / case_id).resolve()
        path = (case_path / filename).resolve()
        if not case_path.is_relative_to(experiment) or not path.is_relative_to(case_path) or not path.is_file():
            raise HTTPException(404, "实验资源尚未生成。")
        return FileResponse(path)

    app.mount("/viewer-common", StaticFiles(directory=ROOT / "src/faceAppearance/web"), name="common")
    app.mount("/", StaticFiles(directory=ROOT / "src/faceAppearance/maleMouthWeb", html=True), name="comparison")
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/maleMouthSeam20261012/final")
    parser.add_argument("--port", type=int, default=18097)
    args = parser.parse_args()
    uvicorn.run(createApp(args.experiment), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
