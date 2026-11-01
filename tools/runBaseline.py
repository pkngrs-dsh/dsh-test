"""Run one reproducible, selected real-photo baseline without moving source data."""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeJson
from faceAppearance.config import PipelineConfig
from faceAppearance.data import loadViews
from faceAppearance.pipeline import Pipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--views", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--caseId", default="diffuseLightNestedViews")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/baseline512")
    parser.add_argument("--stopAfter", choices=("geometry", "delight", "fusion", "wildcap"), default="fusion")
    args = parser.parse_args()
    manifest = json.loads((projectRoot / "reports/preflight/candidateCases.json").read_text())
    case = next(item for item in manifest["cases"] if item["caseId"] == args.caseId)
    paths = [Path(manifest["sourceRoot"]) / case["relativeGroupPath"] / name
             for name in case["subsets"][str(args.views)]]
    loadViews(paths)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        config = (PipelineConfig.fromFile(args.config) if args.config else
                  PipelineConfig(projectRoot, workingResolution=args.resolution))
        pipeline = Pipeline(config)
        print("Loading local weights once", flush=True)
        pipeline.loadModels()
        print(f"Models loaded in {pipeline.loadSeconds:.1f}s; reconstructing {args.views} views", flush=True)
        result = pipeline.run(paths, args.output, args.stopAfter)
        print(json.dumps({key: result[key] for key in ("status", "stageSeconds", "inferenceSeconds",
                                                       "peakAllocatedGiB")}, indent=2), flush=True)
    except Exception as error:
        writeJson(args.output / "failure.json", {"status": "failed", "error": str(error),
                  "traceback": traceback.format_exc(), "timestamp": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
