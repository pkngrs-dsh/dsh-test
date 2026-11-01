"""Exercise the authenticated client API with real GPU reconstruction and ZIP downloads."""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
import zipfile

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from fastapi.testclient import TestClient
from faceAppearance.config import PipelineConfig
from faceAppearance.service import createApp


def modelIdentities(pipeline):
    identities = {f"geometry.{key}": id(value)
                  for key, value in pipeline.geometryAdapter.models.items()}
    identities.update({f"delight.{key}": id(getattr(pipeline.delightAdapter, key))
                       for key in ("model", "enhancer", "faceDetector", "landmarkDetector")})
    identities["assemblyAssets"] = id(pipeline.assemblyAssets)
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=projectRoot / "configs/clientService.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240, help="Job polling deadline, at most 240 seconds")
    args = parser.parse_args()
    outputRoot, reportPath = args.output.resolve(), args.report.resolve()
    if outputRoot.exists() and (not outputRoot.is_dir() or any(outputRoot.iterdir())):
        parser.error("--output must be a new or empty directory")
    if reportPath.exists() or args.report.is_symlink():
        parser.error("--report must name a new file; existing reports are never overwritten")
    if reportPath == outputRoot or not 0 < args.timeout <= 240:
        parser.error("Use distinct output/report paths and a timeout in (0, 240]")
    config = PipelineConfig.fromFile(args.config)
    if not config.device.startswith("cuda") or not config.enableAssembly:
        parser.error("The real smoke requires CUDA and enableAssembly=true")
    logging.basicConfig(level=logging.WARNING)
    candidatePath = projectRoot / "reports/preflight/candidateCases.json"
    candidates = json.loads(candidatePath.read_text(encoding="utf-8"))
    case = candidates["cases"][0]
    sourceRoot = Path(candidates["sourceRoot"])
    relativePaths = [Path(case["relativeGroupPath"]) / name for name in case["subsets"]["4"]]
    expectedHashes = {item["relativePath"]: item["sha256"] for item in case["files"]}
    assert len(relativePaths) == 4
    uploads, inputEvidence = [], []
    for relativePath in relativePaths:
        path = sourceRoot / relativePath
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == expectedHashes[relativePath.as_posix()], f"Input hash changed: {path}"
        uploads.append(("images", (path.name, payload, "image/png")))
        inputEvidence.append({"path": str(path), "sha256": digest})

    fixtureToken = "local-client-service-smoke-fixture-token"
    app = createApp(config, outputRoot=outputRoot, apiToken=fixtureToken)
    print("Starting real client service smoke with four fixed views", flush=True)
    with TestClient(app) as client:
        assert client.get("/v1/capabilities").status_code == 401
        client.headers["Authorization"] = f"Bearer {fixtureToken}"
        response = client.get("/v1/capabilities")
        assert response.status_code == 200, response.text
        capabilities = response.json()
        assert capabilities["ready"] and capabilities["acceptingJobs"]
        template = capabilities["template"]
        identities = modelIdentities(app.state.pipeline)
        assert app.state.pipeline.loaded
        response = client.post("/v1/jobs", files=uploads, data={"templateId": template["templateId"]})
        assert response.status_code == 202, response.text
        record = response.json()
        jobId, statusUrl = record["jobId"], record["statusUrl"]
        print(f"Submitted real client job {jobId}", flush=True)
        deadline, lastStatus = time.monotonic() + args.timeout, None
        while time.monotonic() < deadline:
            response = client.get(statusUrl)
            assert response.status_code == 200, response.text
            record = response.json()
            if record["status"] != lastStatus:
                print(f"Job {jobId}: {record['status']}", flush=True)
                lastStatus = record["status"]
            if record["status"] in {"succeeded", "failed"}:
                break
            time.sleep(min(1., max(0., deadline - time.monotonic())))
        assert record["status"] == "succeeded", record.get("error") or "Job polling timed out"
        assert record["numImages"] == 4
        download, asset = record["download"], record["asset"]
        assert {key: asset[key] for key in template} == template
        packageResponse = client.get(download["url"])
        manifestResponse = client.get(download["manifestUrl"])
        assert packageResponse.status_code == 200, packageResponse.text[:200]
        assert manifestResponse.status_code == 200, manifestResponse.text[:200]
        package = packageResponse.content
        digest = hashlib.sha256(package).hexdigest()
        assert digest == download["sha256"] and len(package) == download["sizeBytes"]
        assert packageResponse.headers["etag"] == f'"{digest}"'
        assetManifest = manifestResponse.json()
        assert assetManifest["status"] == "exported"
        assert {key: assetManifest[key] for key in template} == template
        assert assetManifest["counts"] == asset["counts"]
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            assert archive.testzip() is None
            assert archive.read("manifest.json") == manifestResponse.content
            assert set(archive.namelist()) == set(assetManifest["files"]) | {"manifest.json"}
            for name, identity in assetManifest["files"].items():
                payload = archive.read(name)
                assert len(payload) == identity["bytes"]
                assert hashlib.sha256(payload).hexdigest() == identity["sha256"]
        assert modelIdentities(app.state.pipeline) == identities
        saved = json.loads((outputRoot / jobId / "job.json").read_text(encoding="utf-8"))
        result = saved["result"]
        assert saved["status"] == result["status"] == "succeeded"
        assert saved["clientContract"] and result["stopAfter"] == "assembly"
        assert {"geometry", "delight", "cameraAndFusion", "assemblyAndExport"} <= set(result["stageSeconds"])
        import torch

        summary = {"status": "passed", "apiVersion": "1", "jobId": jobId,
                   "realGpuInference": True, "gpu": torch.cuda.get_device_name(config.device),
                   "bearerAuthenticationChecked": True, "modelsReused": True,
                   "caseId": case["caseId"], "inputs": inputEvidence,
                   "metrics": record["metrics"], "outputResolution": config.workingResolution,
                   "stageStatus": {stage: "succeeded" for stage in result["stageSeconds"]},
                   "assemblyStatus": assetManifest["status"], "asset": asset,
                   "archive": {"sha256": digest, "sizeBytes": len(package), "downloadVerified": True},
                   "warnings": record["warnings"],
                   "paths": {"config": str(args.config.resolve()), "candidateManifest": str(candidatePath),
                             "outputRoot": str(outputRoot), "jobRecord": str(outputRoot / jobId / "job.json"),
                             "archive": str(outputRoot / jobId / "output/assembly/clientAsset.zip"),
                             "manifest": str(outputRoot / jobId / "output/assembly/manifest.json"),
                             "report": str(reportPath)}}
    assert not app.state.jobQueue.worker.is_alive()
    summary["workerStoppedCleanly"] = True
    reportPath.parent.mkdir(parents=True, exist_ok=True)
    with reportPath.open("x", encoding="utf-8") as report:
        json.dump(summary, report, ensure_ascii=False, indent=2, allow_nan=False)
        report.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
