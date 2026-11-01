"""Read-only localhost viewer for cached direct-registration comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
ARTIFACTS = {"preview.json", "mouth.png", "mouthSelected.png", "face.png", "meshes.npz",
             "legacyPrior.json", "weakPrior.json", "noPrior.json", "noPriorHard.json", "noPriorHardSmooth300.json"}


def createApp(experiment: Path = ROOT / "runs/directRegistration20261012") -> FastAPI:
    experiment = Path(experiment).resolve()
    app = FastAPI(title="直接配准 · 本地实验对照", docs_url=None, redoc_url=None, openapi_url=None)
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
        # Only publish the intended experiment summary, never nearby task state.
        data = manifest()
        allowed = ("id", "gender", "captureMode", "previewUrl", "mouthImage", "branches", "baseline")
        return {key: data.get(key) for key in ("status", "objective", "solver")} | {
            "cases": [{key: case[key] for key in allowed if key in case} for case in data["cases"]]}

    @app.get("/api/safety")
    def safety_data():
        try:
            data = json.loads((ROOT / "reports/directRegistration20261012/safety.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(503, "几何交叠审计尚未就绪。") from None
        summaries = []
        for case in data.get("cases", []):
            case_id = case.get("caseId", "")
            if not CASE_ID.fullmatch(case_id):
                continue
            mesh = (experiment / case_id / "meshes.npz").resolve()
            matched = mesh.is_relative_to(experiment) and mesh.is_file() and hashlib.sha256(mesh.read_bytes()).hexdigest() == case.get("meshSha256")
            branches = {}
            for branch_id, branch in case.get("branches", {}).items():
                comparison = branch.get("comparison", {})
                branches[branch_id] = {
                    "summary": branch.get("summary", {}),
                    "newPositivePairsVsRest": len(comparison.get("newPositivePairsVsRest", [])),
                    "newPositivePairsVsBaseline": len(comparison.get("newPositivePairsVsBaseline", [])),
                    "removedPositivePairsVsBaseline": len(comparison.get("removedPositivePairsVsBaseline", [])),
                    "newVsBaselineMouthFrontExposed": comparison.get("newVsBaselineMouthFrontExposed"),
                    "zeroAreaFaceCount": len(branch.get("degeneracy", {}).get("zeroAreaFaceIds", [])),
                }
            summaries.append({"caseId": case_id, "meshMatchesAudit": matched, "branches": branches})
        increases = sum(
            case["meshMatchesAudit"] and case["branches"].get("noPriorHardSmooth300", {}).get("summary", {}).get("positiveLengthOrAreaPairs", 0)
            > case["branches"].get("baseline", {}).get("summary", {}).get("positiveLengthOrAreaPairs", 0)
            for case in summaries)
        return {"status": data.get("status"), "createdAt": data.get("createdAt"), "method": data.get("method"),
                "limitations": data.get("limitations", []), "deploymentStatus": "not_ready",
                "caseCount": len(summaries), "fIntersectionIncreaseCaseCount": increases, "cases": summaries}

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
    app.mount("/", StaticFiles(directory=ROOT / "src/faceAppearance/directRegistrationWeb", html=True), name="comparison")
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/directRegistration20261012")
    parser.add_argument("--port", type=int, default=18094)
    args = parser.parse_args()
    uvicorn.run(createApp(args.experiment), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
