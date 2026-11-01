"""Recompute observed eye appearance after a validated fixed-topology fit."""
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
from faceAppearance.artifacts import writeJson
from faceAppearance.semanticProjection import sampleSemanticObservations
from faceAppearance.semanticFusion import fuseSemanticObservations
from faceAppearance.shDelight import solveShDelight
from runReferenceExperiment import readArrays, exportBranch, fileHash
from runSemanticRefinementExperiment import loadSemanticCache, saveArrays


def findReconstruction(source):
    """Follow saved provenance, rejecting cycles and mismatched photo/camera frames."""
    current, visited, shIterations = Path(source).resolve(), set(), None
    while current not in visited:
        visited.add(current)
        for name in ("inputs.json", "cameras.json"):
            if fileHash(current / name) != fileHash(source / name):
                raise ValueError(f"Provenance frame differs at {current}: {name}")
        if (current / "delight.npz").exists():
            if shIterations is None:
                raise ValueError("No recorded SH iteration count in experiment provenance")
            return current, shIterations
        report = json.loads((current / "experimentReport.json").read_text())
        if "shIterations" in report and shIterations is None:
            shIterations = int(report["shIterations"])
        current = Path(report["source"]).resolve()
    raise ValueError("Cycle in reconstruction provenance")


def prepare(source: Path, candidate: Path, output: Path):
    start = perf_counter()
    source, candidate, output = source.resolve(), candidate.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty eye appearance directory")
    prior = json.loads((source / "experimentReport.json").read_text())
    candidateReport = json.loads((candidate / "report.json").read_text())
    for name in ("geometry.npz", "inputs.json", "cameras.json"):
        if fileHash(source / name) != candidateReport["sourceHashes"][name]:
            raise ValueError(f"Eye fit used a different source: {name}")
    geometry = readArrays(source / "geometry.npz")
    refined = readArrays(candidate / "geometryRefined.npz")
    for name in ("faces", "uvs"):
        if not np.array_equal(geometry[name], refined[name]):
            raise ValueError(f"Eye refinement changed {name}")
    if refined["vertices"].shape != geometry["vertices"].shape or not np.isfinite(refined["vertices"]).all():
        raise ValueError("Invalid eye geometry")
    reconstruction, shIterations = findReconstruction(source)
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    semantics, _ = loadSemanticCache(Path(prior["semanticCache"]), inputs)
    baseline = readArrays(source / "branches/semanticProtected.npz")
    options = json.loads((source / "branches/semanticProtected.json").read_text())
    output.mkdir(parents=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(candidate / "geometryRefined.npz", output / "geometryRefined.npz")
    exportBranch(output, "baseline", baseline, geometry)
    print("Recomputing exact-qualified visibility and original-photo sampling for fitted eyes", flush=True)
    observations = collectAppearanceObservations(refined, cameras, inputs, readArrays(reconstruction / "delight.npz"),
                    baseline["linearAlbedo"].shape[0], samplingMode="projection", visibilityMode="refineRejected")
    reference = prior["referenceViewIndex"]
    fitted = solveShDelight(observations["rawLinear"], observations["weights"], observations["normals"],
                           iterations=shIterations, gaugeReferenceView=reference)
    projected = sampleSemanticObservations([item["labelMap"] for item in semantics], [item["confidence"] for item in semantics],
                                            observations["originalSamplingPixels"], fitted["validWeights"])
    fusion = fuseSemanticObservations(fitted["correctedObservations"], fitted["validWeights"], observations["viewingCosines"],
                    projected["semanticLabels"], projected["semanticConfidence"], referenceViewIndex=reference,
                    referenceBlendWidth=options["referenceBlendWidthUvPixels"], referenceMinCosine=options["referenceMinCosine"],
                    referenceFullCosine=options["referenceFullCosine"], narrowBoundaryBlend=False)
    exportBranch(output, "eyeRefined", fusion, refined)
    for name, result in (("observations", observations), ("shFit", fitted), ("semantics", projected)):
        saveArrays(output / f"{name}.npz", result)
    saved = ("geometry.npz", "geometryRefined.npz", "inputs.json", "cameras.json", "branches/baseline.npz", "branches/eyeRefined.npz")
    summary = {"status": "prepared", "source": source, "candidate": candidate, "referenceViewIndex": reference,
               "branches": {"baseline": {"label": "A: Current eyes", "geometry": "geometry.npz", "texture": "branches/baseline.npz"},
                            "eyeRefined": {"label": "B: Corners + eyelid curves", "geometry": "geometryRefined.npz", "texture": "branches/eyeRefined.npz"}},
               "semanticCache": prior["semanticCache"], "semanticManifestSha256": prior["semanticManifestSha256"],
               "cameraParametersFrozen": True, "uvTopologyFrozen": True, "visibilityAndTextureResampled": True,
               "candidateReportSha256": fileHash(candidate / "report.json"), "candidateGeometrySha256": fileHash(candidate / "geometryRefined.npz"),
               "outputHashes": {name: fileHash(output / name) for name in saved},
               "implementationSha256": fileHash(Path(__file__)), "seconds": perf_counter()-start,
               "limits": candidateReport["limits"] + ["Appearance and SH were resampled and refitted after geometry changes",
                     "Shared pixel photo errors are in-sample diagnostics and must be assessed alongside eye-corner/curve residuals and support counts",
                     "Eye sockets remain without eyeball/tearline/eyelash component assembly"]}
    writeJson(output / "experimentReport.json", summary)
    print(json.dumps({"status": "prepared", "seconds": summary["seconds"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/exactVisibilityExperiment")
    parser.add_argument("--candidate", type=Path, default=projectRoot / "runs/eyeCornerRefinementFinal")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/eyeCornerAppearanceFinal")
    args = parser.parse_args()
    prepare(args.input, args.candidate, args.output)
