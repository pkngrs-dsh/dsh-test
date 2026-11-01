"""Frozen-geometry front-reference texture experiment, using cached photos.

No geometry, OpenDelight or WildCap network runs. The old dense-UV SH branch is
retained as a historical workflow baseline, not a single-variable ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.appearanceObservations import (collectAppearanceObservations, fuseLinearObservations,
                                                  linearToGamma22)
from faceAppearance.artifacts import exportMesh, writeImage, writeJson
from faceAppearance.referenceFusion import fuseReferenceObservations
from faceAppearance.shDelight import solveShDelight


def readArrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exportBranch(output, name, result, geometry):
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["albedo"] = linearToGamma22(arrays["linearAlbedo"])
    np.savez_compressed(output / "branches" / f"{name}.npz", **arrays)
    if "diagnostics" in result:
        writeJson(output / "branches" / f"{name}.json", result["diagnostics"])
    writeImage(output / "branches" / f"{name}.png", arrays["albedo"])
    directory = output / "meshes" / name
    directory.mkdir(parents=True)
    preview = arrays["albedo"].copy()
    mask = arrays["coverageMask"]
    _, nearest = distance_transform_edt(~mask, return_indices=True)
    preview[~mask] = preview[nearest[0][~mask], nearest[1][~mask]]
    writeImage(directory / "albedo.png", arrays["albedo"])
    writeImage(directory / "albedoPreview.png", preview)
    writeImage(directory / "coverage.png", mask)
    exportMesh(directory, geometry)


def runExperiment(source: Path, baseline: Path, output: Path, *, referenceViewIndex=0,
                  resolution=1024, shIterations=8, referenceBlendWidth=12.):
    start = perf_counter()
    source, baseline, output = (Path(path).resolve() for path in (source, baseline, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reference experiment requires an empty output directory")
    if resolution not in (512, 1024):
        raise ValueError("resolution must be 512 or 1024")
    sourceResult = json.loads((source / "result.json").read_text())
    if sourceResult.get("status") != "succeeded":
        raise ValueError("Source reconstruction must have succeeded")
    cameras = json.loads((source / "cameras.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    if isinstance(referenceViewIndex, bool) or not isinstance(referenceViewIndex, int) or not 0 <= referenceViewIndex < len(inputs):
        raise ValueError("referenceViewIndex must select a source photo")
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        if fileHash(source / name) != fileHash(baseline / name):
            raise ValueError(f"Historical baseline differs from source {name}")
    geometry, delight = readArrays(source / "geometry.npz"), readArrays(source / "delight.npz")
    oldTexture = readArrays(baseline / "texture.npz")
    if oldTexture["albedo"].shape != (resolution, resolution, 3):
        raise ValueError("Historical baseline and new experiment must share atlas resolution")
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    parameters = {"referenceViewIndex": referenceViewIndex, "atlasResolution": resolution,
                  "samplingMode": "projection", "shIterations": shIterations,
                  "referenceBlendWidth": referenceBlendWidth, "referenceMinCosine": .35,
                  "referenceFullCosine": .65, "wildCapEnabled": False,
                  "geometryAndCamerasFrozen": True, "rawSampling": "original EXIF-oriented sRGB decoded before sampling",
                  "source": str(source), "historicalBaseline": str(baseline)}
    writeJson(output / "experimentReport.json", {"status": "preparing", **parameters})
    print("Collecting full-resolution projected observations without dense UV correction", flush=True)
    samplingStart = perf_counter()
    observations = collectAppearanceObservations(geometry, cameras, inputs, delight,
                                                 resolution, samplingMode="projection")
    samplingSeconds = perf_counter() - samplingStart
    np.savez_compressed(output / "projectionObservations.npz", **{
        key: value for key, value in observations.items() if isinstance(value, np.ndarray)})
    writeJson(output / "projectionObservations.json", observations["diagnostics"])
    print("Fitting SH with the reference photo fixing the RGB scale", flush=True)
    fitStart = perf_counter()
    fitted = solveShDelight(observations["rawLinear"], observations["weights"], observations["normals"],
                            iterations=shIterations, gaugeReferenceView=referenceViewIndex)
    fitSeconds = perf_counter() - fitStart
    np.savez_compressed(output / "shFit.npz", **{
        key: value for key, value in fitted.items() if isinstance(value, np.ndarray)})
    writeJson(output / "shFit.json", fitted["diagnostics"])
    colors, weights = fitted["correctedObservations"], fitted["validWeights"]
    fusionOptions = {"referenceViewIndex": referenceViewIndex, "referenceBlendWidth": referenceBlendWidth,
                     "referenceMinCosine": .35, "referenceFullCosine": .65, "matchAuxiliaryColors": True}
    reference = fuseReferenceObservations(colors, weights, observations["viewingCosines"], **fusionOptions)
    # Give the average-fusion control the identical auxiliary gains, so the
    # new two-branch comparison isolates reference ownership/blending policy.
    gains = np.array([item["gainsRgb"] for item in reference["diagnostics"]["colorMatching"]["views"]], np.float32)
    matchedColors = np.clip(colors * gains[:, None, None], 0, 1)
    average = fuseLinearObservations(matchedColors, weights)
    average["diagnostics"]["colorBalance"] = "sameReferenceOverlapGainsAsReferenceFirst"
    average["diagnostics"]["colorGains"] = gains.tolist()
    # Lit control shares SH-qualified support but uses original RGB, not albedo.
    raw = fuseReferenceObservations(observations["rawLinear"], weights, observations["viewingCosines"], **fusionOptions)
    raw["diagnostics"]["notAlbedo"] = True
    baselineResult = {"linearAlbedo": oldTexture.get("linearAlbedo", oldTexture["albedo"] ** 2.2),
                      "coverageMask": oldTexture["coverageMask"],
                      "diagnostics": {"source": str(baseline), "kind": "historicalDenseUvShOnlyBaseline",
                                      "notSingleVariableAblation": True}}
    branches = {"denseBaseline": baselineResult, "projectionMean": average,
                "referenceFirst": reference, "rawReference": raw}
    for name, result in branches.items():
        exportBranch(output, name, result, geometry)
    core = reference["referenceCoreMask"]
    if not core.any():
        raise ValueError("No trusted reference core survived; do not claim reference protection")
    coreError = float(np.max(np.abs(reference["linearAlbedo"][core] - colors[referenceViewIndex][core])))
    if coreError != 0:
        raise AssertionError("Reference core colors were not preserved exactly")
    np.savez_compressed(output / "referenceOwnership.npz", referenceAlpha=reference["referenceAlpha"],
                        referenceCoreMask=core, sourceIndex=reference["sourceIndex"])
    writeImage(output / "referenceAlpha.png", reference["referenceAlpha"])
    writeImage(output / "referenceCore.png", core)
    common = np.logical_and.reduce([branches[name]["coverageMask"] for name in (
        "denseBaseline", "projectionMean", "referenceFirst")])
    writeImage(output / "commonCoverage.png", common)
    descriptions = {"denseBaseline": "Previous dense-UV SH-only + robust multi-view average",
                    "projectionMean": "Projection SH + same calibrated colors + robust multi-view average",
                    "referenceFirst": "Projection SH + protected front core + best auxiliary completion",
                    "rawReference": "LIT PHOTO CONTROL, NOT ALBEDO: front projection with auxiliary completion"}
    implementationFiles = ("src/faceAppearance/appearanceObservations.py", "src/faceAppearance/shDelight.py",
                           "src/faceAppearance/referenceFusion.py", "tools/runReferenceExperiment.py")
    report = {"status": "prepared", **parameters, "branchDescriptions": descriptions,
              "samplingSeconds": samplingSeconds, "shSeconds": fitSeconds, "seconds": perf_counter() - start,
              "referencePhotoSha256": inputs[referenceViewIndex]["sha256"],
              "referenceCoreMaxColorChange": coreError,
              "branchCoverageTexels": {name: int(result["coverageMask"].sum()) for name, result in branches.items()},
              "commonCoverageTexels": int(common.sum()), "fusionDiagnostics": reference["diagnostics"],
              "sourceHashes": {str(source / name): fileHash(source / name) for name in (
                  "geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json", "delight.npz")},
              "baselineTextureSha256": fileHash(baseline / "texture.npz"),
              "implementationHashes": {name: fileHash(projectRoot / name) for name in implementationFiles},
              "limits": ["All photographs are reconstruction/fitting inputs, not held-out evaluation",
                         "Geometry, PnP camera and silhouette errors remain fixed",
                         "Reference protection follows geometric visibility/angle, not annotated brow/lip landmarks",
                         "Historical baseline differs in sampling, gauge, support and color matching; not a one-factor ablation",
                         "Projection mean and reference first share projected observations, SH fit, safety support and gains",
                         "Original photo tone mapping and true illuminant are uncalibrated",
                         "Reference core copies SH-corrected pixels exactly but may retain specular/shadow artifacts",
                         "Best auxiliary source switches may produce seams; no secondary-view feature registration",
                         "Mesh previews fill holes for viewing only; coverage identifies actual observations"]}
    writeJson(output / "experimentReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds", "commonCoverageTexels", "referenceCoreMaxColorChange")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/faceCrop1024_4viewsObserved")
    parser.add_argument("--baseline", type=Path, default=projectRoot / "runs/shDelightExperiment/shOnly")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/referenceViewExperiment")
    parser.add_argument("--referenceViewIndex", type=int, default=0)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--shIterations", type=int, default=8)
    parser.add_argument("--referenceBlendWidth", type=float, default=12.)
    args = parser.parse_args()
    if not 1 <= args.shIterations <= 50 or not np.isfinite(args.referenceBlendWidth) or args.referenceBlendWidth < 0:
        parser.error("Require 1..50 SH iterations and nonnegative finite blend width")
    runExperiment(args.input, args.baseline, args.output, referenceViewIndex=args.referenceViewIndex,
                  resolution=args.resolution, shIterations=args.shIterations, referenceBlendWidth=args.referenceBlendWidth)


if __name__ == "__main__":
    main()
