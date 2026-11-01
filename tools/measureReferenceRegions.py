"""Summarize saved fitted-photo predictions on shared, predeclared face regions.

No inference or fitting. These are in-sample photograph appearance errors,
not reflectance ground truth or an identity-recognition score.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeJson
from renderReferenceExperiment import photoMetrics
from diagnoseBrowAlignment import makeStrips


def measure(directory):
    report = json.loads((directory / "comparisonReport.json").read_text())
    if report.get("status") != "rendered":
        raise ValueError("Only completed comparisons can be measured")
    index = report["referenceViewIndex"]
    view = directory / f"view{index:02d}"
    with np.load(view / "commonPhotoData.npz", allow_pickle=False) as data:
        photo, common = data["observedLinear"], data["evaluationMask"]
    if photo.shape[:2] != (1280, 720):
        raise ValueError("The inspection boxes are defined only for this 720x1280 source")
    regions = {"browRegion": (65, 370, 405, 485), "mouthRegion": (120, 610, 330, 690)}
    summary = {"comparison": str(directory), "metricScope": "in-sample fitted-photo appearance on common support",
               "manualRegionsXyxy": regions, "metrics": {}}
    for name in ("denseBaseline", "projectionMean", "referenceFirst"):
        with np.load(view / f"{name}PhotoFit.npz", allow_pickle=False) as data:
            prediction = data["renderedLinear"]
            if not np.array_equal(data["evaluationMask"], common):
                raise ValueError("Branch evaluation masks must match exactly")
        summary["metrics"][name] = {"face": photoMetrics(prediction, photo, common)}
        for region, (left, top, right, bottom) in regions.items():
            mask = np.zeros_like(common)
            mask[top:bottom, left:right] = common[top:bottom, left:right]
            summary["metrics"][name][region] = photoMetrics(prediction, photo, mask)
    labels = ["Original", "Dense UV + mean", "Projection + mean", "Reference first"]
    imagePaths = [view / "original.png"] + [view / f"{name}Matched.png" for name in (
        "denseBaseline", "projectionMean", "referenceFirst")]
    for region, crop in regions.items():
        makeStrips([Image.open(path).crop(crop) for path in imagePaths], labels,
                   directory / f"{region}Detail.png", columns=4)
    writeJson(directory / "regionMetrics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    measure(parser.parse_args().input.resolve())
