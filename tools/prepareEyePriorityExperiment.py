"""Isolate local front-eye texture priority on already refined eye geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeJson
from faceAppearance.referenceFeaturePriority import applyReferenceFeaturePriority
from runReferenceExperiment import readArrays, fileHash, exportBranch


def prepare(source, fit, output, *, coreRadiusPixels=3., falloffRadiusPixels=8.):
    start = perf_counter()
    source, fit, output = source.resolve(), fit.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty eye priority experiment directory")
    report = json.loads((source / "experimentReport.json").read_text())
    for name, expected in report["outputHashes"].items():
        if fileHash(source / name) != expected:
            raise ValueError(f"Prepared eye appearance changed: {name}")
    fitReport = json.loads((fit / "report.json").read_text())
    for name in ("geometryRefined.npz", "photoContours.json"):
        if fileHash(fit / name) != fitReport["outputHashes"][name]:
            raise ValueError(f"Eye fit output changed: {name}")
    if fileHash(fit / "geometryRefined.npz") != fileHash(source / "geometryRefined.npz"):
        raise ValueError("Eye priority must use the same refined geometry")
    contours = json.loads((fit / "photoContours.json").read_text())
    reference = report["referenceViewIndex"]
    curves = [eye[part] for eye in contours[reference]["eyes"].values() if eye["valid"]
              for part in ("upperCurve", "lowerCurve")]
    if len(curves) != 4:
        raise ValueError("Two reliable front eye contours are required")
    baseline, refined = readArrays(source / "branches/baseline.npz"), readArrays(source / "branches/eyeRefined.npz")
    for name, values in (("baseline", baseline), ("eyeRefined", refined)):
        diagnosticsPath = source / f"branches/{name}.json"
        values["diagnostics"] = json.loads(diagnosticsPath.read_text()) if diagnosticsPath.exists() else {}
    observations, sh = readArrays(source / "observations.npz"), readArrays(source / "shFit.npz")
    priority = applyReferenceFeaturePriority(refined, sh["correctedObservations"][reference],
        sh["validWeights"][reference], observations["originalSamplingPixels"][reference], curves,
        referenceViewIndex=reference, coreRadiusPixels=coreRadiusPixels, falloffRadiusPixels=falloffRadiusPixels)
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "geometryRefined.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    geometry, refinedGeometry = readArrays(output / "geometry.npz"), readArrays(output / "geometryRefined.npz")
    for name, values, mesh in (("baseline", baseline, geometry), ("eyeRefined", refined, refinedGeometry),
                               ("eyePriority", priority, refinedGeometry)):
        exportBranch(output, name, values, mesh)
    branches = {"baseline": {"label": "A: Current geometry", "geometry": "geometry.npz", "texture": "branches/baseline.npz"},
                "eyeRefined": {"label": "B: Refined geometry", "geometry": "geometryRefined.npz", "texture": "branches/eyeRefined.npz"},
                "eyePriority": {"label": "C: Refined + front eye texture", "geometry": "geometryRefined.npz", "texture": "branches/eyePriority.npz"}}
    paths = ["geometry.npz", "geometryRefined.npz", "inputs.json", "cameras.json"] + [b["texture"] for b in branches.values()]
    implementations = ("src/faceAppearance/referenceFeaturePriority.py", "tools/prepareEyePriorityExperiment.py")
    for name in implementations:
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(projectRoot / name, target)
    summary = {**report, "source": source, "fit": fit, "branches": branches,
        "sourceReportSha256": fileHash(source / "experimentReport.json"),
        "inputHashes": {name: fileHash(source / name) for name in ("observations.npz", "shFit.npz")},
        "photoContoursSha256": fileHash(fit / "photoContours.json"),
        "outputHashes": {name: fileHash(output / name) for name in paths},
        "implementationHashes": {name: fileHash(projectRoot / name) for name in implementations},
        "diagnostics": priority["diagnostics"], "seconds": perf_counter()-start,
        "limits": report["limits"] + ["Local eye texture priority uses only qualified front-photo SH observations",
            "It may trade side-view consistency for front-photo matching; camera, mesh, coverage and SH remain fixed between B and C"]}
    # Parent implementationSha256 describes a different preparation script.
    summary.pop("implementationSha256", None)
    writeJson(output / "experimentReport.json", summary)
    print(json.dumps({"seconds": summary["seconds"], "diagnostics": priority["diagnostics"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/eyeCornerAppearanceFinal")
    parser.add_argument("--fit", type=Path, default=projectRoot / "runs/eyeCornerRefinementFinal")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/eyeCornerPriorityExperiment")
    parser.add_argument("--core-radius", type=float, default=3.)
    parser.add_argument("--falloff-radius", type=float, default=8.)
    args = parser.parse_args()
    prepare(args.input, args.fit, args.output, coreRadiusPixels=args.core_radius, falloffRadiusPixels=args.falloff_radius)
