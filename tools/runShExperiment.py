"""Compare cached OpenDelight, SH-only and SH-prior, optionally with WildCap.

No reconstruction models are rerun. Input images/geometry/cameras stay immutable.
Use --stage prepare on CPU, then --stage wildcap for the three GPU refinements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.appearanceObservations import (collectAppearanceObservations, fuseLinearObservations,
                                                   linearToGamma22)
from faceAppearance.artifacts import exportMesh, writeImage, writeJson

methodNames = ("openDelight", "shOnly", "shPrior")


def readArrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def hashFile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validateCompletedWildCap(branch: Path, branchReport: dict, planned: dict, atlasResolution: int) -> dict:
    """Certify a saved completion before resume/display, without loading models.

    A crash can leave material files before the final result marker, or a result
    marker whose files were not durably written. Neither is a completed branch.
    """
    branch = Path(branch)
    diagnostic = branchReport.get("wildCapDiagnostics", {})
    if (branchReport.get("status") != "succeeded" or branchReport.get("stopAfter") != "wildcap"
            or diagnostic.get("status") != "experimental"):
        raise ValueError(f"WildCap requires a committed completion marker for every branch: {branch}")
    for key, expected in (("iterations", planned["iterations"]), ("seed", planned["seed"]),
                          ("workingResolution", planned["resolution"]),
                          ("outputResolution", planned["resolution"]), ("inputMethod", branch.name)):
        if diagnostic.get(key) != expected:
            raise ValueError(f"Completed WildCap {key} differs from the experiment plan: {branch}")

    diagnosticPath = branch / "wildcapIct/diagnostics.json"
    required = (diagnosticPath, branch / "wildcapIct/materials.npz", branch / "wildcapUff/materials.npz")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Completed WildCap artifacts required for every branch: {path}")
    try:
        savedDiagnostic = json.loads(diagnosticPath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable completed WildCap diagnostics: {diagnosticPath}") from error

    def requireFiniteJson(value):
        if isinstance(value, dict):
            for child in value.values():
                requireFiniteJson(child)
        elif isinstance(value, list):
            for child in value:
                requireFiniteJson(child)
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Nonfinite completed WildCap diagnostics: {diagnosticPath}")

    requireFiniteJson(savedDiagnostic)
    if savedDiagnostic != diagnostic:
        raise ValueError(f"Completed WildCap diagnostics disagree with the result marker: {branch}")
    fields = {"diffuse": 3, "specular": None, "validMask": None}
    archives = ((required[1], planned["resolution"], {**fields, "tangentNormal": 3}),
                (required[2], atlasResolution, {**fields, "baseWithWildCap": 3}))
    for path, resolution, expectedFields in archives:
        try:
            with np.load(path, allow_pickle=False) as saved:
                if set(expectedFields) - set(saved.files):
                    raise ValueError("Required material fields are missing")
                for key in saved.files:
                    values = saved[key]
                    if values.dtype.kind not in "bifu" or not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite or nonnumeric material field {key}")
                    if key in expectedFields:
                        channels = expectedFields[key]
                        shape = (resolution, resolution) + (() if channels is None else (channels,))
                        if values.shape != shape:
                            raise ValueError(f"Material field {key} has the wrong resolution/shape")
                mask = saved["validMask"]
                if not mask.any() or np.any((mask != 0) & (mask != 1)):
                    raise ValueError("Material validMask must contain observed binary support")
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError(f"Invalid completed WildCap materials: {path}: {error}") from error
    return savedDiagnostic


def detailMetric(linear, mask):
    """Descriptive texture energy, never a correctness or identity score."""
    luminance = np.asarray(linear) @ np.array([.2126, .7152, .0722])
    safe = binary_erosion(mask, iterations=5)
    residual = luminance - gaussian_filter(luminance, 1.2)
    localMean = gaussian_filter(luminance, 1.2)
    relative = residual / np.maximum(localMean, 0.03)
    return {"highPassRmsLinear": float(np.sqrt(np.mean(residual[safe] ** 2))) if safe.any() else None,
            "relativeHighPassRms": float(np.sqrt(np.mean(relative[safe] ** 2))) if safe.any() else None,
            "pixelCount": int(safe.sum()), "sigmaUvPixels": 1.2,
            "meaning": "texture energy only; includes shading/noise, not accuracy"}


def exportBranch(source, output, name, geometry, fusion, commonMask, parameters, diagnostics):
    branch = output / name
    branch.mkdir()
    for filename in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / filename, branch / filename)
    for path in source.glob("input*.png"):
        shutil.copy2(path, branch / path.name)
    config = json.loads((source / "config.json").read_text())
    config.update(enableWildCap=False, workingResolution=int(commonMask.shape[0]))
    writeJson(branch / "config.json", config)
    linear = fusion["linearAlbedo"].copy()
    linear[~commonMask] = 0
    albedo = linearToGamma22(linear)
    preview = albedo.copy()
    _, nearest = distance_transform_edt(~commonMask, return_indices=True)
    preview[~commonMask] = albedo[nearest[0][~commonMask], nearest[1][~commonMask]]
    confidence = fusion["confidence"].copy()
    confidence[~commonMask] = 0
    sourceIndex = fusion["sourceIndex"].copy()
    sourceIndex[~commonMask] = -1
    np.savez_compressed(branch / "texture.npz", albedo=albedo, linearAlbedo=linear,
                        previewAlbedo=preview, coverageMask=commonMask,
                        confidence=confidence, sourceIndex=sourceIndex)
    writeImage(branch / "albedo.png", albedo)
    writeImage(branch / "albedoPreview.png", preview)
    writeImage(branch / "coverage.png", commonMask)
    exportMesh(branch, geometry)
    report = {"status": "succeeded", "stopAfter": "fusion", "method": name, "config": config,
              "geometrySpace": "uvFaceFusionCanonical6986", "sourceRun": str(source),
              "inputColorEncoding": "raw=sRGB; OpenDelight=native assumed gamma2.2",
              "colorSpace": "gamma2.2 encoded linear RGB; not physically calibrated",
              "experiment": parameters, "shDiagnostics": diagnostics,
              "textureDiagnostics": fusion["diagnostics"], "detailMetric": detailMetric(linear, commonMask),
              "geometryAndCamerasRefitted": False, "commonCoverageTexels": int(commonMask.sum()),
              "sourceHashes": {name: hashFile(source / name) for name in
                               ("geometry.npz", "cameras.json", "inputs.json", "delight.npz")}}
    writeJson(branch / "result.json", report)
    return report


def prepareExperiment(args):
    from faceAppearance.shDelight import solveShDelight

    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preparation requires a new output directory")
    sourceReport = json.loads((source / "result.json").read_text())
    if sourceReport.get("status") != "succeeded":
        raise ValueError("Source reconstruction must have succeeded")
    start = perf_counter()
    geometry, delight = readArrays(source / "geometry.npz"), readArrays(source / "delight.npz")
    cameras = json.loads((source / "cameras.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    print("Sampling original photos and cached OpenDelight through frozen UV correspondences", flush=True)
    observations = collectAppearanceObservations(geometry, cameras, inputs, delight, args.resolution)
    rawSampling = getattr(args, "rawSampling", "original")
    if rawSampling == "working":
        from runShBandwidthControl import sampleSavedWorkingImages
        sampled = sampleSavedWorkingImages(observations, inputs, source)
        if np.any((observations["weights"] > 0) & ~sampled["samplingValidMask"]):
            raise ValueError("Working images cannot supply all shared UV observations")
        observations["rawOriginalLinear"] = observations["rawLinear"]
        observations["rawLinear"] = sampled["raw512Linear"]
        observations["diagnostics"]["originalRawEncoding"] = observations["diagnostics"].get("rawEncoding")
        observations["diagnostics"]["rawEncoding"] = sampled["diagnostics"]["encoding"]
        observations["diagnostics"]["rawSampling"] = "savedWorkingImages"
        observations["diagnostics"]["workingSampling"] = sampled["diagnostics"]
        print("SH uses saved working images, matching cached OpenDelight input bandwidth", flush=True)
    output.mkdir(parents=True)
    np.savez_compressed(output / "observations.npz", **{
        key: value for key, value in observations.items() if isinstance(value, np.ndarray)})
    writeJson(output / "observationDiagnostics.json", observations["diagnostics"])
    weights, raw = observations["weights"], observations["rawLinear"]
    prior = fuseLinearObservations(observations["openDelightLinear"], weights)["linearAlbedo"]
    corrections = {}
    for name in ("shOnly", "shPrior"):
        print(f"Fitting {name} on low-frequency observations; retaining {rawSampling} source samples", flush=True)
        fitStart = perf_counter()
        result = solveShDelight(raw, weights, observations["normals"], mode=name,
                               priorAlbedo=prior if name == "shPrior" else None,
                               iterations=args.shIterations)
        result["diagnostics"]["seconds"] = perf_counter() - fitStart
        corrections[name] = result
        np.savez_compressed(output / f"{name}Fit.npz", **{
            key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        writeJson(output / f"{name}Fit.json", result["diagnostics"])
    # The same individual view/texel weights feed every final fusion, so a
    # rejected dark pixel cannot improve one branch by changing its coverage.
    commonWeights = np.minimum(corrections["shOnly"]["validWeights"], corrections["shPrior"]["validWeights"])
    commonWeights = np.minimum(commonWeights, weights)
    if not commonWeights.any():
        raise ValueError("No mutually safe observations survived the two SH branches")
    branchValues = {"openDelight": observations["openDelightLinear"],
                    **{name: result["correctedObservations"] for name, result in corrections.items()}}
    fusions = {name: fuseLinearObservations(values, commonWeights) for name, values in branchValues.items()}
    commonMask = np.logical_and.reduce([item["coverageMask"] for item in fusions.values()])
    np.savez_compressed(output / "commonSupport.npz", weights=commonWeights, coverageMask=commonMask)
    parameters = {"atlasResolution": args.resolution, "shIterations": args.shIterations,
                  "geometryAndCamerasFrozen": True, "colorBalance": False,
                  "originalResolutionRawSampling": rawSampling == "original", "rawSampling": rawSampling,
                  "workingImageSizesWh": [item["workingSize"] for item in inputs],
                  "commonViewTexelWeights": True,
                  "basisFrame": "UFF canonical object frame; independent SH per view",
                  "openDelightBaseline": "cached predictions re-fused in linear RGB with shared experiment support",
                  "comparisonToOldPipeline": "not identical to original native-encoded fusion",
                  "photometricCalibration": "approximate; SH-only gauge is raw-derived, prior gauge is OD-derived",
                  "wildCapPlanned": {"resolution": args.wildCapResolution, "iterations": args.wildCapIterations,
                                     "seed": args.seed, "artifactMask": "none; globalOnly for all branches"}}
    parameters["implementationHashes"] = {
        filename: hashFile(projectRoot / filename) for filename in (
            "src/faceAppearance/shDelight.py", "src/faceAppearance/appearanceObservations.py")}
    parameters["runnerSha256"] = hashFile(Path(__file__))
    if rawSampling == "working":
        parameters["workingSamplerSha256"] = hashFile(projectRoot / "tools/runShBandwidthControl.py")
        parameters["workingImageHashes"] = {f"input{index:02d}.png": hashFile(source / f"input{index:02d}.png")
                                             for index in range(len(inputs))}
    reports = {}
    for name in methodNames:
        reports[name] = exportBranch(source, output, name, geometry, fusions[name], commonMask, parameters,
                                     corrections[name]["diagnostics"] if name in corrections else None)
    report = {"status": "prepared", "sourceRun": str(source), "parameters": parameters,
              "seconds": perf_counter() - start, "commonCoverageTexels": int(commonMask.sum()),
              "initialCoverageTexels": int((weights.sum(axis=0) > 0).sum()),
              "commonObservationCount": int(np.count_nonzero(commonWeights)),
              "initialObservationCount": int(np.count_nonzero(weights)),
              "branchDetailMetrics": {name: report["detailMetric"] for name, report in reports.items()},
              "limits": ["No ground-truth albedo or calibrated lighting; increased high-frequency energy is not accuracy",
                         "Original geometry/camera/dense-UV errors remain fixed and can contaminate recovered colors",
                         "SH does not explicitly model cast shadows, specular reflection or subsurface scattering",
                         "Skin tone and lighting scale/color cannot be uniquely separated from these photos",
                         "Original photographs may contain unknown tone mapping or beauty processing"]}
    writeJson(output / "experimentReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds", "commonCoverageTexels")}), flush=True)


def refineExperiment(args):
    import torch
    from faceAppearance.runtime import configureRuntime
    configureRuntime(projectRoot)
    from faceAppearance.adapters.wildcap import WildCapAdapter
    from faceAppearance.atlasTransfer import transferTexture
    from faceAppearance.registration import RegistrationAssets

    output = args.output.resolve()
    report = json.loads((output / "experimentReport.json").read_text())
    planned = report["parameters"]["wildCapPlanned"]
    if (planned["iterations"] != args.wildCapIterations or planned["resolution"] != args.wildCapResolution
            or planned["seed"] != args.seed):
        raise ValueError("WildCap options must match the saved experiment plan")
    completed = {}
    for name in methodNames:
        branch = output / name
        branchReport = json.loads((branch / "result.json").read_text())
        if branchReport.get("wildCapDiagnostics", {}).get("status") == "experimental":
            completed[name] = validateCompletedWildCap(
                branch, branchReport, planned, report["parameters"]["atlasResolution"])
        elif any((branch / sub).exists() for sub in ("wildcapIct", "wildcapUff")):
            raise FileExistsError(f"Partial WildCap result in {branch}; preserve it and choose a fresh experiment")
    # Recover the aggregate if interruption happened after a branch commit but
    # before the experiment report update. Validated files remain untouched.
    report.setdefault("wildCapBranches", {}).update(completed)
    if len(completed) == len(methodNames):
        report["status"] = "completedWithWildCap"
        writeJson(output / "experimentReport.json", report)
        print("All WildCap branches already completed and validated", flush=True)
        return
    forward = RegistrationAssets(projectRoot / "assets/registration/wrapV1", args.wildCapResolution)
    back = RegistrationAssets(projectRoot / "assets/registration/wrapV1", report["parameters"]["atlasResolution"])
    geometry = readArrays(output / "openDelight/geometry.npz")
    ictGeometry = forward.toIctGeometry(geometry)
    loadStart = perf_counter()
    adapter = WildCapAdapter(projectRoot, args.device).loadModels()
    report["wildCapLoadSeconds"] = perf_counter() - loadStart
    writeJson(output / "experimentReport.json", report)
    for name in methodNames:
        branch = output / name
        branchReport = json.loads((branch / "result.json").read_text())
        if name in completed:
            print(f"Skipping completed WildCap branch {name}", flush=True)
            continue
        if any((branch / sub).exists() for sub in ("wildcapIct", "wildcapUff")):
            raise FileExistsError(f"Partial WildCap result in {branch}; preserve it and choose a fresh experiment")
        texture = readArrays(branch / "texture.npz")
        ictInput = transferTexture(texture["albedo"], texture["coverageMask"], forward.atlasMap("uffToIct"),
                                    materialKind="color", colorEncoding="gamma2.2")
        ictDir, uffDir = branch / "wildcapIct", branch / "wildcapUff"
        ictDir.mkdir()
        uffDir.mkdir()
        writeImage(ictDir / "input.png", ictInput["texture"])
        writeImage(ictDir / "inputObserved.png", ictInput["validMask"])
        torch.cuda.reset_peak_memory_stats()
        print(f"WildCap {name}: {args.wildCapIterations} updates, {args.wildCapResolution} atlas", flush=True)
        result = adapter.optimize(ictGeometry["vertices"], ictGeometry["faces"], ictGeometry["uvs"],
                                  ictInput["texture"], ictInput["validMask"], uvFaces=ictGeometry["uvFaces"],
                                  geometryFrame=ictGeometry["geometryFrame"], workingResolution=args.wildCapResolution,
                                  iterations=args.wildCapIterations, seed=args.seed)
        np.savez_compressed(ictDir / "materials.npz", **{
            key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        returned = {}
        for key in ("diffuse", "specular"):
            baked = transferTexture(result[key], result["validMask"], back.atlasMap("ictToUff"),
                                     materialKind="color" if key == "diffuse" else "scalar",
                                     colorEncoding="gamma2.2" if key == "diffuse" else "native")
            returned[key] = baked["texture"]
            returned["validMask"] = baked["validMask"]
            writeImage(uffDir / f"{key}.png", baked["texture"])
        composite = texture["albedo"].copy()
        valid = returned["validMask"] & texture["coverageMask"]
        composite[valid] = returned["diffuse"][valid]
        returned["validMask"], returned["baseWithWildCap"] = valid, composite
        np.savez_compressed(uffDir / "materials.npz", **returned)
        writeImage(uffDir / "baseWithWildCap.png", composite)
        diagnostic = {**result["diagnostics"], "registrationResultSha256": forward.manifest["resultSha256"],
                      "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 1024 ** 3,
                      "inputMethod": name, "inputColorInterpretation": "explicit gamma2.2 encoded linear RGB",
                      "forwardAndReverseResampling": "linearLight", "roughnessStatus": "notEstimated",
                      "originalUffGeometryPreserved": True,
                      "normalTransferStatus": "keptInIct; not used for controlled diffuse comparison"}
        writeJson(ictDir / "diagnostics.json", diagnostic)
        branchReport.update(wildCapDiagnostics=diagnostic, stopAfter="wildcap",
                            wildCapDetailMetric=detailMetric(composite ** 2.2, texture["coverageMask"]))
        branchReport["config"].update(enableWildCap=True, wildCapResolution=args.wildCapResolution,
                                      wildCapIterations=args.wildCapIterations, seed=args.seed)
        writeJson(branch / "config.json", branchReport["config"])
        writeJson(branch / "result.json", branchReport)
        report.setdefault("wildCapBranches", {})[name] = diagnostic
        writeJson(output / "experimentReport.json", report)
        print(f"Completed {name}: {diagnostic['seconds']:.1f}s", flush=True)
    report["status"] = "completedWithWildCap"
    writeJson(output / "experimentReport.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/faceCrop1024_4viewsObserved")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/shDelightExperiment")
    parser.add_argument("--stage", choices=("prepare", "wildcap", "all"), default="all")
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--shIterations", type=int, default=8)
    parser.add_argument("--rawSampling", choices=("original", "working"), default="original",
                        help="SH source bandwidth: full original photos or cached working input PNGs")
    parser.add_argument("--wildCapResolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--wildCapIterations", type=int, default=599)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.shIterations <= 50 or not 1 <= args.wildCapIterations <= 1000:
        parser.error("SH iterations must be 1..50 and WildCap iterations 1..1000")
    if not 0 <= args.seed < 2 ** 63:
        parser.error("seed must be in [0, 2**63)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.stage in ("prepare", "all"):
        prepareExperiment(args)
    if args.stage in ("wildcap", "all"):
        refineExperiment(args)


if __name__ == "__main__":
    main()
