"""Resample photographs after a saved anatomical geometry refinement.

Old UV colors must not simply move with edited vertices: recompute visibility,
sampling, SH and semantic fusion using unchanged original-photo cameras.
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
from faceAppearance.artifacts import writeJson
from faceAppearance.semanticProjection import sampleSemanticObservations
from faceAppearance.semanticFusion import fuseSemanticObservations
from faceAppearance.shDelight import solveShDelight
from runReferenceExperiment import readArrays, exportBranch, fileHash
from runSemanticRefinementExperiment import loadSemanticCache, saveArrays


def prepare(source: Path, candidate: Path, output: Path):
    start = perf_counter()
    source, candidate, output = source.resolve(), candidate.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    previous = json.loads((source / "experimentReport.json").read_text())
    candidateReport = json.loads((candidate / "report.json").read_text())
    for name, digest in candidateReport["sourceHashes"].items():
        if fileHash(source / name) != digest:
            raise ValueError(f"Anatomical candidate used different input: {name}")
    geometry = readArrays(source / "geometry.npz")
    refined = readArrays(candidate / "geometryRefined.npz")
    assert np.array_equal(geometry["faces"], refined["faces"])
    assert np.array_equal(geometry["uvs"], refined["uvs"])
    assert geometry["vertices"].shape == refined["vertices"].shape
    referenceSource = Path(previous["source"])
    reconstructionReport = json.loads((referenceSource / "experimentReport.json").read_text())
    reconstruction = Path(reconstructionReport["source"])
    for name in ("geometry.npz", "cameras.json", "inputs.json"):
        assert fileHash(reconstruction / name) == fileHash(source / name)
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    semantics, _ = loadSemanticCache(Path(previous["semanticCache"]), inputs)
    baseline = readArrays(source / "branches/semanticProtected.npz")
    diagnostics = json.loads((source / "branches/semanticProtected.json").read_text())
    output.mkdir(parents=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(candidate / "geometryRefined.npz", output / "geometryRefined.npz")
    exportBranch(output, "baseline", baseline, geometry)
    print("Resampling original photos and fitting SH for the anatomical candidate", flush=True)
    observations = collectAppearanceObservations(refined, cameras, inputs, readArrays(reconstruction / "delight.npz"),
                                                  baseline["linearAlbedo"].shape[0], samplingMode="projection")
    reference = previous["referenceViewIndex"]
    fitted = solveShDelight(observations["rawLinear"], observations["weights"], observations["normals"],
                           iterations=reconstructionReport["shIterations"], gaugeReferenceView=reference)
    projected = sampleSemanticObservations([item["labelMap"] for item in semantics],
                    [item["confidence"] for item in semantics], observations["originalSamplingPixels"], fitted["validWeights"])
    fusion = fuseSemanticObservations(fitted["correctedObservations"], fitted["validWeights"], observations["viewingCosines"],
                    projected["semanticLabels"], projected["semanticConfidence"], referenceViewIndex=reference,
                    referenceBlendWidth=diagnostics["referenceBlendWidthUvPixels"],
                    referenceMinCosine=diagnostics["referenceMinCosine"], referenceFullCosine=diagnostics["referenceFullCosine"],
                    narrowBoundaryBlend=False)
    exportBranch(output, "anatomical", fusion, refined)
    for name, result in (("observations", observations), ("shFit", fitted), ("semantics", projected)):
        saveArrays(output / f"{name}.npz", result)
    files = ("geometry.npz", "geometryRefined.npz", "cameras.json", "inputs.json", "branches/baseline.npz", "branches/anatomical.npz")
    report = {"status": "prepared", "referenceViewIndex": reference, "source": source, "candidate": candidate,
              "branches": {"baseline": {"label": "A: Current geometry", "geometry": "geometry.npz", "texture": "branches/baseline.npz"},
                           "anatomical": {"label": "B: Anatomical eye fit", "geometry": "geometryRefined.npz", "texture": "branches/anatomical.npz"}},
              "semanticCache": previous["semanticCache"], "semanticManifestSha256": previous["semanticManifestSha256"],
              "cameraParametersFrozen": True, "photoSamplingRecomputed": True, "wildCapEnabled": False,
              "candidateReportSha256": fileHash(candidate / "report.json"),
              "candidateGeometrySha256": fileHash(candidate / "geometryRefined.npz"),
              "outputHashes": {name: fileHash(output / name) for name in files},
              "implementationSha256": fileHash(Path(__file__)), "seconds": perf_counter()-start,
              "limits": candidateReport["limits"] + ["Texture and visibility recomputed after editing; shared photo support is required for comparisons",
                     "Photo error measures reproduction of fitting images, not independent reconstruction accuracy"]}
    writeJson(output / "experimentReport.json", report)
    print(json.dumps({"status": "prepared", "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--candidate", type=Path, default=projectRoot / "runs/anatomicalEyeBoundaryExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/anatomicalEyeAppearanceExperiment")
    args = parser.parse_args()
    prepare(args.input, args.candidate, args.output)
