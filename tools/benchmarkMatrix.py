"""Real-photo 4/16 view test; one model load, 1024 UV outputs, no external assets."""

import json
import logging
import sys
import traceback
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeJson
from faceAppearance.config import PipelineConfig
from faceAppearance.pipeline import Pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manifest = json.loads((projectRoot / "reports/preflight/candidateCases.json").read_text())
    case = manifest["cases"][0]
    pipeline = Pipeline(PipelineConfig(projectRoot, workingResolution=1024)).loadModels()
    modelIdentities = {key: id(value) for key, value in pipeline.geometryAdapter.models.items()}
    reports = []
    for viewCount in (4, 16):
        outputDir = projectRoot / f"runs/quality1024_{viewCount}views"
        outputDir.mkdir(parents=True, exist_ok=True)
        paths = [Path(manifest["sourceRoot"]) / case["relativeGroupPath"] / filename
                 for filename in case["subsets"][str(viewCount)]]
        try:
            result = pipeline.run(paths, outputDir)
            summary = {key: result[key] for key in ("inferenceSeconds", "peakAllocatedGiB", "stageSeconds",
                                                    "acceptedCameraCount", "loadSeconds")}
            summary.update(viewCount=viewCount, status="succeeded", resolution=1024,
                           atlasCoverage=result["textureDiagnostics"]["atlasCoverage"])
            summary["modelsReused"] = modelIdentities == {
                key: id(value) for key, value in pipeline.geometryAdapter.models.items()}
        except Exception as error:
            summary = {"viewCount": viewCount, "status": "failed", "error": str(error)}
            writeJson(outputDir / "failure.json", {**summary, "traceback": traceback.format_exc()})
        reports.append(summary)
        writeJson(projectRoot / "reports/benchmarkMatrix.json", reports)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
