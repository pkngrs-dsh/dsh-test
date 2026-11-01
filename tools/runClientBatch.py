"""Run a fixed-parameter photo batch through the existing modules.

The selection is explicit and every failure stays in the batch manifest. This
is an offline experiment coordinator, not a new reconstruction implementation.
GPU reconstruction and appearance run in separate processes to release model
memory. Within each phase the applicable neural models/assets load only once.
Completed phase boundaries can be resumed; interrupted partial case outputs
remain intact for diagnosis and require a fresh output location to retry.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeJson
from runReferenceExperiment import exportBranch, fileHash, readArrays


parameters = {"resolution": 1024, "referenceViewIndex": 0, "shIterations": 8,
              "referenceBlendWidth": 4., "referenceMinCosine": .35, "referenceFullCosine": .65,
              "samplingMode": "projection", "visibilityMode": "refineRejected",
              "narrowBoundaryBlend": False, "eyeIterations": 100, "eyeReferenceWeight": 12.,
              "eyeSupportRadiusFraction": .16, "eyeControlCount": 24, "eyeCornerWeight": 8.,
              "minimumRawPeak": .6, "coreRadiusPixels": 4., "falloffRadiusPixels": 18.}
assetNames = {"A": "legacyFemaleV1", "B": "legacyFemaleBarycentricV1", "C": "legacyFemaleBarycentricUvV1"}


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initializeBatch(selection, output):
    selection, output = selection.resolve(), output.resolve()
    cases = readJson(selection)["cases"]
    identities = [case["caseId"] for case in cases]
    if len(set(identities)) != len(cases) or not cases:
        raise ValueError("Expected unique nonempty case IDs")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", case["caseId"]):
            raise ValueError("Case IDs must be simple directory names")
        if case.get("referenceViewIndex", 0) != 0 or not 4 <= len(case["paths"]) <= 16:
            raise ValueError("This frozen experiment requires the reference first and 4–16 photos")
    manifestPath = output / "batchManifest.json"
    sourceHashes = {path: fileHash(path) for case in cases for path in case["paths"]}
    if manifestPath.exists():
        report = readJson(manifestPath)
        if (report["selectionSha256"] != fileHash(selection) or report["parameters"] != parameters
                or report["photoHashes"] != sourceHashes):
            raise ValueError("Resume inputs or settings differ from the frozen batch")
        return report, manifestPath
    output.mkdir(parents=True, exist_ok=True)
    report = {"schemaVersion": 1, "status": "preparing", "selection": str(selection),
              "selectionSha256": fileHash(selection), "photoHashes": sourceHashes,
              "parameters": parameters, "clientAssets": assetNames,
              "clientTemplatePolicy": "All cases use the same female/260429 binding identity; no gender inference or template switching",
              "cases": [{**case, "outputDir": case["caseId"], "status": "pending", "stages": {}} for case in cases],
              "limits": ["Photos are reconstruction inputs, not held-out evaluation or manual ground truth",
                         "No per-case parameter tuning; original client binding identity is fixed across cases",
                         "Unreliable front eyes skip both eye geometry refinement and local eye texture priority, explicitly recorded",
                         "WildCap is disabled, matching the prior client A/B/C experiment"]}
    writeJson(manifestPath, report)
    return report, manifestPath


def recordFailure(case, stage, error, directory):
    case["status"] = "failed"
    case["error"] = f"{stage}: {type(error).__name__}: {error}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stage}Failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    print(case["caseId"], case["error"], flush=True)


def runReconstruction(report, manifestPath):
    from faceAppearance.config import PipelineConfig
    from faceAppearance.pipeline import Pipeline

    config = PipelineConfig(projectRoot=projectRoot, workingResolution=1024, enableAssembly=False)
    candidates = [case for case in report["cases"] if "reconstruction" not in case["stages"]]
    if not candidates:
        return
    pipeline = Pipeline(config).loadModels()
    report["reconstructionPreloadSeconds"] = pipeline.loadSeconds
    for case in candidates:
        directory = manifestPath.parent / case["outputDir"]
        try:
            print(f"RECONSTRUCTION {case['caseId']}", flush=True)
            case["status"] = "reconstructing"
            writeJson(manifestPath, report)
            result = pipeline.run([Path(path) for path in case["paths"]], directory / "reconstruction", "fusion")
            case["stages"]["reconstruction"] = {"status": "succeeded", "seconds": result["inferenceSeconds"],
                "peakAllocatedGiB": result["peakAllocatedGiB"], "resultSha256": fileHash(directory / "reconstruction/result.json")}
            case["status"] = "reconstructed"
        except Exception as error:
            recordFailure(case, "reconstruction", error, directory)
        writeJson(manifestPath, report)
        gc.collect()
    report["status"] = "reconstructionComplete"
    writeJson(manifestPath, report)


def prepareSemanticAppearance(source, semanticCache, output):
    """Same effective pre-eye branch as the previous final experiment."""
    from faceAppearance.appearanceObservations import collectAppearanceObservations
    from faceAppearance.semanticProjection import sampleSemanticObservations
    from faceAppearance.semanticFusion import fuseSemanticObservations
    from faceAppearance.shDelight import solveShDelight
    from runSemanticRefinementExperiment import loadSemanticCache, saveArrays

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Do not overwrite a semantic appearance stage")
    geometry, delight = readArrays(source / "geometry.npz"), readArrays(source / "delight.npz")
    inputs, cameras = readJson(source / "inputs.json"), readJson(source / "cameras.json")
    semantics, _ = loadSemanticCache(semanticCache, inputs)
    observations = collectAppearanceObservations(geometry, cameras, inputs, delight, parameters["resolution"],
                    samplingMode=parameters["samplingMode"], visibilityMode=parameters["visibilityMode"])
    fitted = solveShDelight(observations["rawLinear"], observations["weights"], observations["normals"],
                            iterations=parameters["shIterations"], gaugeReferenceView=0)
    projected = sampleSemanticObservations([item["labelMap"] for item in semantics],
        [item["confidence"] for item in semantics], observations["originalSamplingPixels"], fitted["validWeights"])
    fusion = fuseSemanticObservations(fitted["correctedObservations"], fitted["validWeights"],
        observations["viewingCosines"], projected["semanticLabels"], projected["semanticConfidence"],
        referenceViewIndex=0, referenceBlendWidth=parameters["referenceBlendWidth"],
        referenceMinCosine=parameters["referenceMinCosine"], referenceFullCosine=parameters["referenceFullCosine"],
        narrowBoundaryBlend=False)
    (output / "branches").mkdir(parents=True)
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    exportBranch(output, "semanticProtected", fusion, geometry)
    for name, arrays in (("observations", observations), ("shFit", fitted), ("semantics", projected)):
        saveArrays(output / f"{name}.npz", arrays)
    writeJson(output / "experimentReport.json", {"status": "prepared", "source": source,
        "referenceViewIndex": 0, "shIterations": parameters["shIterations"],
        "semanticCache": semanticCache, "semanticManifestSha256": fileHash(semanticCache / "manifest.json"),
        "parameters": parameters, "effectiveBranch": "Direct composition of the prior exactVisibility semanticProtected branch"})


def prepareUnrefinedAppearance(source, output, reason):
    """Publish a clearly declared fallback; never label rejected eyes as fitted."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Do not overwrite an appearance stage")
    (output / "branches").mkdir(parents=True)
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "geometry.npz", output / "geometryRefined.npz")
    shutil.copy2(source / "branches/semanticProtected.npz", output / "branches/eyePriority.npz")
    shutil.copy2(source / "branches/semanticProtected.json", output / "branches/eyePriority.json")
    writeJson(output / "experimentReport.json", {"status": "preparedWithSkippedEyeRefinement", "source": source,
        "eyeRefinementApplied": False, "eyePriorityApplied": False, "reason": reason,
        "geometryRefinedNameIsCompatibilityAlias": True, "parameters": parameters})


def finishCase(case, directory, parser, eyelid, assets):
    from cacheFaceSemantics import cacheSemantics
    from cacheEyelidLandmarks import cacheEyelidLandmarks
    from runEyeRefinementExperiment import loadEyeCache, run as refineEyes
    from prepareEyeAppearance import prepare as prepareEyeAppearance
    from prepareEyePriorityExperiment import prepare as prepareEyePriority
    from faceAppearance.clientAsset import exportClientAsset
    from evaluateClientEyeTransfer import evaluate

    source, semanticCache = directory / "reconstruction", directory / "semantics"
    start = perf_counter()
    # Reuse only completed neural caches whose input and payload hashes agree.
    # Partially prepared downstream stages still fail closed rather than being
    # overwritten or silently treated as successful.
    semanticManifestPath = semanticCache / "manifest.json"
    if semanticManifestPath.exists():
        from runSemanticRefinementExperiment import loadSemanticCache
        manifest = readJson(semanticManifestPath)
        if manifest.get("inputsSha256") != fileHash(source / "inputs.json"):
            raise ValueError("Existing semantics use a different ordered input manifest")
        if manifest.get("modelWeightHashes") != {name: fileHash(path) for name, path in parser.assetPaths().items()}:
            raise ValueError("Existing semantics use different weights")
        loadSemanticCache(semanticCache, readJson(source / "inputs.json"))
    else:
        cacheSemantics(source, semanticCache, adapter=parser)
    eyeManifestPath = directory / "eyelids/manifest.json"
    if eyeManifestPath.exists():
        manifest = readJson(eyeManifestPath)
        if (manifest.get("status") != "complete" or manifest.get("inputsSha256") != fileHash(source / "inputs.json")
                or manifest.get("semanticManifestSha256") != fileHash(semanticManifestPath)
                or manifest.get("modelWeightHashes") != {str(eyelid.assetPath()): fileHash(eyelid.assetPath())}):
            raise ValueError("Existing eyelid cache has different provenance or is incomplete")
        for view in manifest["views"]:
            if (fileHash(directory / "eyelids" / view["arrays"]) != view["cacheSha256"]
                    or fileHash(directory / "eyelids" / view["metadata"]) != view["metadataSha256"]):
                raise ValueError("Existing eyelid cache payload changed")
    else:
        cacheEyelidLandmarks(source, semanticCache, directory / "eyelids", adapter=eyelid)
    prepareSemanticAppearance(source, semanticCache, directory / "semanticAppearance")
    inputs = readJson(source / "inputs.json")
    reason = None
    try:
        observations, _, _ = loadEyeCache(directory / "eyelids", inputs,
                            minimumRawPeak=parameters["minimumRawPeak"], inputsSha256=fileHash(source / "inputs.json"))
        if not all(eye["valid"] for eye in observations[0]["eyes"].values()):
            reason = "Front eyelid observations did not pass the unchanged quality threshold"
    except ValueError as error:
        reason = f"Eye observation validation rejected the cache: {error}"
    if reason is None:
        fitReport = refineEyes(directory / "semanticAppearance", directory / "eyelids", directory / "eyeFit",
            iterations=parameters["eyeIterations"], referenceWeight=parameters["eyeReferenceWeight"],
            supportRadiusFraction=parameters["eyeSupportRadiusFraction"], controlCount=parameters["eyeControlCount"],
            cornerWeight=parameters["eyeCornerWeight"], minimumRawPeak=parameters["minimumRawPeak"])
        fitDiagnostics = fitReport["geometryDiagnostics"]
        case["eyeFitGate"] = {"status": fitDiagnostics["status"],
                             "inputTargetGate": fitDiagnostics["inputTargetGate"]}
        if fitDiagnostics["inputTargetGate"]["entireFitRejected"]:
            reason = "Eye fit rejected by the unchanged 40-original-pixel initial-target gate"
        elif fitDiagnostics["status"] != "accepted":
            reason = "Eye solver retained the original geometry; no accepted safe improvement"
    if reason is not None:
        case["eyeRefinement"] = {"status": "skipped", "reason": reason}
        prepareUnrefinedAppearance(directory / "semanticAppearance", directory / "appearance", reason)
    else:
        prepareEyeAppearance(directory / "semanticAppearance", directory / "eyeFit", directory / "eyeAppearance")
        prepareEyePriority(directory / "eyeAppearance", directory / "eyeFit", directory / "appearance",
            coreRadiusPixels=parameters["coreRadiusPixels"], falloffRadiusPixels=parameters["falloffRadiusPixels"])
        case["eyeRefinement"] = {"status": "applied"}
    geometry = readArrays(directory / "appearance/geometryRefined.npz")
    texture = readArrays(directory / "appearance/branches/eyePriority.npz")
    results = {}
    for label, asset in assets.items():
        assembled = asset.assemble(geometry, texture, colorEncoding="linear")
        exportClientAsset(directory / "assembly" / label, assembled)
        results[label] = readArrays(directory / "assembly" / label / "clientMesh.npz")
    for label in ("B", "C"):
        for name in ("faces", "uvs", "uvFaces", "materialIds", "templateVertices"):
            if not np.array_equal(results["A"][name], results[label][name]):
                raise AssertionError(f"Binding identity changed in {label}/{name}")
    if not np.array_equal(results["B"]["vertices"], results["C"]["vertices"]):
        raise AssertionError("B and C geometry must be identical")
    case["validation"] = {"bindingArraysEqual": True, "bcVerticesExactlyEqual": True}
    if reason is None:
        metrics = evaluate({label: directory / "assembly" / label for label in assets}, directory / "appearance",
            projectRoot / "assets/clientTemplates/legacyFemaleV1", directory / "eyeFit/photoContours.json",
            directory / "assembly/A/diagnostics.json", directory / "metrics")
        case["eyeProxyMeanErrorPixels"] = {label: value["meanErrorPixels"] for label, value in metrics["branches"].items()}
    case["stages"]["appearanceAndAssembly"] = {"status": "succeeded", "seconds": perf_counter()-start}
    case["status"] = "succeeded"


def runAppearance(report, manifestPath, *, cachedOnly=False):
    from faceAppearance.adapters.faceParsing import FaceParsingAdapter
    from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
    from faceAppearance.headAssembly import AssemblyAssets

    candidates = [case for case in report["cases"] if case["status"] == "reconstructed"]
    if not candidates:
        return
    start = perf_counter()
    assets = {label: AssemblyAssets(projectRoot / "assets/clientTemplates" / name) for label, name in assetNames.items()}
    parser = FaceParsingAdapter(projectRoot, device="cuda", parserName="celebm")
    eyelid = EyelidLandmarksAdapter(projectRoot, minimumPeakScore=.2)
    if cachedOnly:
        for case in candidates:
            for name in ("semantics", "eyelids"):
                path = manifestPath.parent / case["outputDir"] / name / "manifest.json"
                if not path.is_file() or readJson(path).get("status") != "complete":
                    raise ValueError("cachedOnly requires completed semantic and eyelid caches for every case")
    else:
        parser.loadModels()
        eyelid.loadModels()
    report["appearanceCachedOnly"] = cachedOnly
    report["appearancePreloadSeconds"] = perf_counter()-start
    for case in candidates:
        directory = manifestPath.parent / case["outputDir"]
        try:
            print(f"APPEARANCE {case['caseId']}", flush=True)
            case["status"] = "preparingAppearance"
            writeJson(manifestPath, report)
            finishCase(case, directory, parser, eyelid, assets)
        except Exception as error:
            recordFailure(case, "appearance", error, directory)
        writeJson(manifestPath, report)
        gc.collect()
    report["status"] = "completed" if all(case["status"] == "succeeded" for case in report["cases"]) else "completedWithFailures"
    writeJson(manifestPath, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("reconstruction", "appearance"), required=True)
    parser.add_argument("--cachedOnly", action="store_true", help="Appearance only, verified neural caches required; no model/GPU loading")
    args = parser.parse_args()
    report, manifestPath = initializeBatch(args.selection, args.output)
    if args.cachedOnly and args.phase != "appearance":
        parser.error("--cachedOnly applies only to appearance")
    if args.phase == "reconstruction":
        runReconstruction(report, manifestPath)
    else:
        runAppearance(report, manifestPath, cachedOnly=args.cachedOnly)
    print(json.dumps({case["caseId"]: case["status"] for case in report["cases"]}), flush=True)


if __name__ == "__main__":
    main()
