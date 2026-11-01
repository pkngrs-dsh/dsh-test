"""Three controlled stages: front reference, semantic protection, local geometry.

Photographs and cameras are identical across branches. Geometry refinement uses
fixed semantic boundary attachments, not RGB texture reprojection as evidence.
All illumination/observations are recomputed after changing geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.appearanceObservations import collectAppearanceObservations
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.semanticProjection import sampleSemanticObservations, visibleVertices
from faceAppearance.shDelight import solveShDelight
from renderCropComparison import fileHash, photoCamera, validatePhotoPair
from runReferenceExperiment import exportBranch, readArrays


def saveArrays(path, result):
    np.savez_compressed(path, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})


def loadSemanticCache(directory, inputs):
    from faceAppearance.faceSemantics import semanticLabelNames

    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest["views"]) != len(inputs):
        raise ValueError("Complete, matching semantic cache required")
    if tuple(manifest["labelNames"]) != semanticLabelNames:
        raise ValueError("Semantic cache must use the standard label contract")
    maps = []
    for index, (record, item) in enumerate(zip(manifest["views"], inputs)):
        path = directory / f"view{index:02d}.npz"
        if (record["viewIndex"] != index or record["photoSha256"] != item["sha256"]
                or fileHash(path) != record["cacheSha256"]):
            raise ValueError("Semantic cache source or contents changed")
        values = readArrays(path)
        if list(values["labelMap"].shape) != list(reversed(item["orientedSize"])):
            raise ValueError("Semantic coordinates differ from original photograph")
        maps.append(values)
    return maps, manifest


def prepareExperiment(source: Path, semanticCache: Path, output: Path, *, iterations=160,
                      maxDisplacementFraction=.01, referenceWeight=4.,
                      deformationMode="smooth", referenceAnchorWeight=.1,
                      narrowBoundaryBlend=False):
    from faceAppearance.geometryRefinement import buildSemanticBoundaryConstraints, refineLocalGeometry
    from faceAppearance.semanticFusion import fuseSemanticObservations

    start = perf_counter()
    source, semanticCache, output = (Path(p).resolve() for p in (source, semanticCache, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty experiment directory")
    baselineReport = json.loads((source / "experimentReport.json").read_text())
    if baselineReport.get("status") != "prepared":
        raise ValueError("Prepared reference experiment required")
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    for item in inputs:
        validatePhotoPair(item, item)
    referenceIndex = baselineReport["referenceViewIndex"]
    semantics, semanticManifest = loadSemanticCache(semanticCache, inputs)
    labelMaps, confidenceMaps = ([m[key] for m in semantics] for key in ("labelMap", "confidence"))
    geometry = readArrays(source / "geometry.npz")
    observations = readArrays(source / "projectionObservations.npz")
    shFit = readArrays(source / "shFit.npz")
    baseline = readArrays(source / "branches/referenceFirst.npz")
    resolution = baseline["linearAlbedo"].shape[0]
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    writeJson(output / "experimentReport.json", {"status": "preparing", "source": source})
    options = {"referenceViewIndex": referenceIndex,
               "referenceBlendWidth": baselineReport["referenceBlendWidth"],
               "referenceMinCosine": baselineReport["referenceMinCosine"],
               "referenceFullCosine": baselineReport["referenceFullCosine"], "matchAuxiliaryColors": True,
               "narrowBoundaryBlend": narrowBoundaryBlend}

    def fuseBranch(currentObservations, currentFit, name, currentGeometry):
        projected = sampleSemanticObservations(labelMaps, confidenceMaps,
                        currentObservations["originalSamplingPixels"], currentFit["validWeights"])
        result = fuseSemanticObservations(currentFit["correctedObservations"], currentFit["validWeights"],
                    currentObservations["viewingCosines"], projected["semanticLabels"],
                    projected["semanticConfidence"], **options)
        exportBranch(output, name, result, currentGeometry)
        saveArrays(output / f"{name}Semantics.npz", projected)
        writeImage(output / f"{name}Protection.png", result["protectionMask"])
        saveArrays(output / f"{name}Observations.npz", currentObservations)
        saveArrays(output / f"{name}ShFit.npz", currentFit)
        return result

    print("Preparing semantic-protected fusion on unchanged geometry", flush=True)
    exportBranch(output, "baseline", baseline, geometry)
    semanticResult = fuseBranch(observations, shFit, "semanticProtected", geometry)

    print("Binding fixed semantic boundaries and fitting local vertex offsets", flush=True)
    photoCameras = [photoCamera(camera, item, "fitted") for camera, item in zip(cameras, inputs)]
    visibility = visibleVertices(geometry["vertices"], geometry["faces"], photoCameras,
                                  [labelMap.shape for labelMap in labelMaps])
    groups = {"leftEye": {"labelIds": [4], "weight": 1.},
              "rightEye": {"labelIds": [5], "weight": 1.},
              "lips": {"labelIds": [12, 13], "weight": 1.},
              "innerMouth": {"labelIds": [11], "weight": .5},
              "nose": {"labelIds": [10], "weight": .15}}
    bindings = buildSemanticBoundaryConstraints(geometry["vertices"], geometry["faces"], photoCameras,
                    labelMaps, confidenceMaps, visibility, groups, referenceViewIndex=referenceIndex)
    writeJson(output / "constraintBinding.json", bindings["diagnostics"])
    packed = {"vertexVisibility": visibility}
    constraintMeta = []
    for index, constraint in enumerate(bindings["constraints"]):
        record = {}
        for key, value in constraint.items():
            if isinstance(value, np.ndarray):
                packed[f"constraint{index:02d}_{key}"] = value
            else:
                record[key] = value
        constraintMeta.append(record)
    np.savez_compressed(output / "geometryConstraints.npz", **packed)
    writeJson(output / "geometryConstraints.json", constraintMeta)
    extent = float(np.ptp(geometry["vertices"], axis=0).max())
    fitStart = perf_counter()
    refined = refineLocalGeometry(geometry["vertices"], geometry["faces"], photoCameras,
                bindings["constraints"], referenceViewIndex=referenceIndex, referenceWeight=referenceWeight,
                maxDisplacement=extent * maxDisplacementFraction, iterations=iterations,
                deformationMode=deformationMode, referenceAnchorWeight=referenceAnchorWeight)
    geometrySeconds = perf_counter()-fitStart
    newGeometry = {**geometry, "vertices": refined["vertices"]}
    if not np.isfinite(newGeometry["vertices"]).all() or newGeometry["vertices"].shape != geometry["vertices"].shape:
        raise ValueError("Refinement must preserve vertex count and finite geometry")
    np.savez_compressed(output / "geometryRefined.npz", **newGeometry)
    writeJson(output / "geometryRefinement.json", {"diagnostics": refined["diagnostics"],
                                                 "history": refined["history"], "seconds": geometrySeconds})
    print("Recomputing geometry visibility, photo sampling, SH and semantic fusion", flush=True)
    reconstruction = Path(baselineReport["source"])
    for name in ("inputs.json", "cameras.json", "geometry.npz"):
        if fileHash(reconstruction / name) != fileHash(source / name):
            raise ValueError("Original reconstruction no longer matches reference experiment")
    delight = readArrays(reconstruction / "delight.npz")
    newObservations = collectAppearanceObservations(newGeometry, cameras, inputs, delight,
                                                     resolution, samplingMode="projection")
    newFit = solveShDelight(newObservations["rawLinear"], newObservations["weights"], newObservations["normals"],
                           iterations=baselineReport["shIterations"], gaugeReferenceView=referenceIndex)
    refinedResult = fuseBranch(newObservations, newFit, "geometryRefined", newGeometry)
    branchInfo = {"baseline": {"label": "A: Front reference", "geometry": "geometry.npz"},
                  "semanticProtected": {"label": "B: Semantic protection", "geometry": "geometry.npz"},
                  "geometryRefined": {"label": "C: Local geometry + B", "geometry": "geometryRefined.npz"}}
    for name, value in branchInfo.items():
        value["texture"] = f"branches/{name}.npz"
    implementationFiles = ("src/faceAppearance/adapters/faceParsing.py", "src/faceAppearance/faceSemantics.py",
                           "src/faceAppearance/semanticFusion.py", "src/faceAppearance/semanticProjection.py",
                           "src/faceAppearance/geometryRefinement.py", "tools/runSemanticRefinementExperiment.py")
    report = {"status": "prepared", "source": source, "semanticCache": semanticCache,
              "referenceViewIndex": referenceIndex, "branches": branchInfo, "atlasResolution": resolution,
              "semanticModel": semanticManifest["parserName"], "semanticManifestSha256": fileHash(semanticCache / "manifest.json"),
              "cameraParametersFrozen": True, "vertexFaceUvOrderPreserved": True, "wildCapEnabled": False,
              "maxDisplacementFraction": maxDisplacementFraction, "referenceWeight": referenceWeight,
              "deformationMode": deformationMode, "referenceAnchorWeight": referenceAnchorWeight,
              "narrowBoundaryBlend": narrowBoundaryBlend,
              "geometryRefinementSeconds": geometrySeconds, "seconds": perf_counter()-start,
              "geometryDiagnostics": refined["diagnostics"],
              "semanticFusionDiagnostics": semanticResult["diagnostics"],
              "refinedFusionDiagnostics": refinedResult["diagnostics"],
              "sourceHashes": {name: fileHash(source / name) for name in (
                  "geometry.npz", "cameras.json", "inputs.json", "shFit.npz", "projectionObservations.npz", "branches/referenceFirst.npz")},
              "implementationHashes": {name: fileHash(projectRoot / name) for name in implementationFiles},
              "outputHashes": {name: fileHash(output / name) for name in (
                  "geometry.npz", "geometryRefined.npz", "cameras.json", "inputs.json",
                  "branches/baseline.npz", "branches/semanticProtected.npz", "branches/geometryRefined.npz")},
              "limits": ["Automatic segmentation and initial reference attachments are not anatomical ground truth",
                         "Initial visibility gates are frozen during bounded local fitting; final photo sampling recomputes occlusion",
                         "Different expressions and segmentation errors may conflict across views",
                         "RGB photo scores are in-sample; front texture copies reference observations",
                         "Fixed-topology UFF mesh is not validated standard FLAME or an assembled animated head",
                         "SH-only diffuse illumination; specular, SSS, eyes and oral components are not fitted"]}
    writeJson(output / "experimentReport.json", report)
    for name in implementationFiles:
        snapshot = output / "implementation" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(projectRoot / name, snapshot)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "geometrySeconds": geometrySeconds}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/referenceViewBlend4Experiment")
    parser.add_argument("--semanticCache", type=Path, default=projectRoot / "runs/faceSemanticCache")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/semanticRefinementExperiment")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--maxDisplacementFraction", type=float, default=.01)
    parser.add_argument("--referenceWeight", type=float, default=4.)
    parser.add_argument("--deformationMode", choices=("vertex", "smooth"), default="smooth")
    parser.add_argument("--referenceAnchorWeight", type=float, default=.1)
    parser.add_argument("--narrowBoundaryBlend", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000 or not 0 < args.maxDisplacementFraction <= .03 or args.referenceWeight <= 0:
        parser.error("Expected 1..1000 iterations, displacement fraction in (0,.03] and positive reference weight")
    prepareExperiment(args.input, args.semanticCache, args.output, iterations=args.iterations,
                      maxDisplacementFraction=args.maxDisplacementFraction, referenceWeight=args.referenceWeight,
                      deformationMode=args.deformationMode, referenceAnchorWeight=args.referenceAnchorWeight,
                      narrowBoundaryBlend=args.narrowBoundaryBlend)
