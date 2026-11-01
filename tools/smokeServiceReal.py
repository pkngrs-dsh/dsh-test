"""Exercise FastAPI startup, GPU worker, JSON results and artifact downloads."""

import io
import argparse
import json
import logging
import sys
import time
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from fastapi.testclient import TestClient
from PIL import Image
from faceAppearance.artifacts import writeJson
from faceAppearance.config import PipelineConfig
from faceAppearance.service import createApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stopAfter", choices=("fusion", "wildcap"), default="fusion")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/serviceSmoke")
    parser.add_argument("--report", type=Path, default=projectRoot / "reports/serviceSmoke.json")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    config = (PipelineConfig.fromFile(args.config) if args.config else
              PipelineConfig(projectRoot, workingResolution=512))
    logging.basicConfig(level=logging.WARNING)
    manifest = json.loads((projectRoot / "reports/preflight/candidateCases.json").read_text())
    case = manifest["cases"][0]
    paths = [Path(manifest["sourceRoot"]) / case["relativeGroupPath"] / name
             for name in case["subsets"]["4"]]
    app = createApp(config, outputRoot=args.output)
    print("Starting real preloaded GPU service in TestClient", flush=True)
    with TestClient(app) as client:
        assert client.get("/health").json()["modelsLoaded"]
        identities = {key: id(value) for key, value in app.state.pipeline.geometryAdapter.models.items()}
        uploads = [("images", (path.name, path.read_bytes(), "image/png")) for path in paths]
        response = client.post("/jobs", files=uploads, data={"stopAfter": args.stopAfter})
        assert response.status_code == 202, response.text
        jobId = response.json()["jobId"]
        print(f"Submitted {jobId}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            record = client.get(f"/jobs/{jobId}").json()
            if record["status"] in {"succeeded", "failed"}:
                break
            time.sleep(1)
        assert record["status"] == "succeeded", record.get("error") or record["status"]
        downloaded = client.get(f"/jobs/{jobId}/artifacts/albedo.png")
        assert downloaded.status_code == 200
        with Image.open(io.BytesIO(downloaded.content)) as image:
            image.load()
            assert image.size == (config.workingResolution, config.workingResolution)
        geometryResponse = client.get(f"/jobs/{jobId}/artifacts/face.obj")
        assert geometryResponse.status_code == 200 and "v " in geometryResponse.text
        assert identities == {key: id(value) for key, value in app.state.pipeline.geometryAdapter.models.items()}
        materialDownloaded = False
        if args.stopAfter == "wildcap":
            material = client.get(f"/jobs/{jobId}/artifacts/wildcapUff/diffuse.png")
            assert material.status_code == 200
            with Image.open(io.BytesIO(material.content)) as image:
                image.load()
                assert image.size == (config.workingResolution, config.workingResolution)
            materialDownloaded = True
        summary = {"status": "passed", "jobId": jobId, "modelsReused": True,
                   "stopAfter": args.stopAfter, "wildCapIterations": config.wildCapIterations,
                   "wildCapMaterialDownloaded": materialDownloaded,
                   "outputResolution": config.workingResolution, "downloadedImageDecoded": True,
                   "downloadedMesh": True, "realGpuInference": True,
                   "inferenceSeconds": record["result"]["inferenceSeconds"],
                   "loadSeconds": record["result"]["loadSeconds"],
                   "peakAllocatedGiB": record["result"]["peakAllocatedGiB"]}
    assert not app.state.jobQueue.worker.is_alive()
    summary["workerStoppedCleanly"] = True
    writeJson(args.report, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
