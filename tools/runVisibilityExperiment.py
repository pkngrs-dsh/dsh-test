"""Compare cached nearest-depth observation gates with exact rejected-ray review."""
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
from faceAppearance.semanticProjection import sampleSemanticObservations
from faceAppearance.semanticFusion import fuseSemanticObservations
from faceAppearance.shDelight import solveShDelight
from runReferenceExperiment import readArrays, exportBranch, fileHash
from runSemanticRefinementExperiment import loadSemanticCache, saveArrays


def runExperiment(source: Path, output: Path):
    start = perf_counter()
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    previous = json.loads((source / "experimentReport.json").read_text())
    referenceReport = json.loads((Path(previous["source"]) / "experimentReport.json").read_text())
    reconstruction = Path(referenceReport["source"])
    for name in ("geometry.npz", "cameras.json", "inputs.json"):
        if fileHash(reconstruction / name) != fileHash(source / name):
            raise ValueError(f"Reconstruction frame differs: {name}")
    geometry = readArrays(source / "geometry.npz")
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    semantics, _ = loadSemanticCache(Path(previous["semanticCache"]), inputs)
    baseline = readArrays(source / "branches/semanticProtected.npz")
    oldObservations = readArrays(source / "semanticProtectedObservations.npz")
    diagnostics = json.loads((source / "branches/semanticProtected.json").read_text())
    output.mkdir(parents=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    exportBranch(output, "baseline", baseline, geometry)
    print("Rechecking only rejected depth queries at continuous projected positions", flush=True)
    observations = collectAppearanceObservations(geometry, cameras, inputs, readArrays(reconstruction / "delight.npz"),
                    baseline["linearAlbedo"].shape[0], samplingMode="projection", visibilityMode="refineRejected")
    oldValid = oldObservations["weights"] > 0
    assert not (oldValid & (observations["weights"] <= 0)).any(), "Existing support must not disappear"
    for name in ("rawLinear", "weights", "viewingCosines", "originalSamplingPixels"):
        assert np.array_equal(oldObservations[name][oldValid], observations[name][oldValid]), f"Existing sample changed: {name}"
    recovered = (observations["weights"] > 0) & ~oldValid
    reference = previous["referenceViewIndex"]
    print("Refitting the same SH model and semantic fusion on verified observations", flush=True)
    fitted = solveShDelight(observations["rawLinear"], observations["weights"], observations["normals"],
                           iterations=referenceReport["shIterations"], gaugeReferenceView=reference)
    projected = sampleSemanticObservations([item["labelMap"] for item in semantics], [item["confidence"] for item in semantics],
                                            observations["originalSamplingPixels"], fitted["validWeights"])
    fusion = fuseSemanticObservations(fitted["correctedObservations"], fitted["validWeights"], observations["viewingCosines"],
                    projected["semanticLabels"], projected["semanticConfidence"], referenceViewIndex=reference,
                    referenceBlendWidth=diagnostics["referenceBlendWidthUvPixels"], referenceMinCosine=diagnostics["referenceMinCosine"],
                    referenceFullCosine=diagnostics["referenceFullCosine"], narrowBoundaryBlend=False)
    exportBranch(output, "semanticProtected", fusion, geometry)
    for name, result in (("Observations", observations), ("ShFit", fitted), ("Semantics", projected)):
        saveArrays(output / f"semanticProtected{name}.npz", result)
    for index, mask in enumerate(recovered):
        writeImage(output / f"view{index:02d}Recovered.png", mask)
    writeJson(output / "visibility.json", observations["diagnostics"])
    implementation = ("src/faceAppearance/appearanceObservations.py", "src/faceAppearance/surfaceVisibility.py",
                      "src/faceAppearance/semanticFusion.py", "src/faceAppearance/shDelight.py", "tools/runVisibilityExperiment.py")
    for name in implementation:
        destination = output / "implementation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(projectRoot / name, destination)
    files = ("geometry.npz", "cameras.json", "inputs.json", "branches/baseline.npz", "branches/semanticProtected.npz")
    summary = {"status": "prepared", "referenceViewIndex": reference, "source": source,
              "branches": {"baseline": {"label": "A: Nearest-depth gate", "geometry": "geometry.npz", "texture": "branches/baseline.npz"},
                           "semanticProtected": {"label": "B: Exact rejected-ray review", "geometry": "geometry.npz", "texture": "branches/semanticProtected.npz"}},
              "semanticCache": previous["semanticCache"], "semanticManifestSha256": previous["semanticManifestSha256"],
              "geometryAndCamerasFrozen": True, "existingRawSamplesBitExact": True,
              "recoveredObservationTexelsByView": recovered.sum(axis=(1, 2)).tolist(),
              "outputHashes": {name: fileHash(output / name) for name in files},
              "implementationHashes": {name: fileHash(projectRoot / name) for name in implementation},
              "seconds": perf_counter()-start,
              "limits": ["Only previously rejected depth queries are rechecked; accepted coarse-depth queries are not revalidated",
                         "Original cosine, photo/mask, positive-depth and depth-tolerance gates remain in effect",
                         "Recovered support is supported by exact mesh intersection, not ground-truth geometry or correspondence",
                         "SH and semantic fusion are recomputed with identical settings; global albedo changes can follow new observations",
                         "Photographs are fitting inputs, not held-out evaluation; compare pre-fit common support",
                         "This does not align anatomical features or repair missing eyeballs and oral components"]}
    writeJson(output / "experimentReport.json", summary)
    print(json.dumps({"seconds": summary["seconds"], "recoveredObservationTexelsByView": summary["recoveredObservationTexelsByView"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/exactVisibilityExperiment")
    args = parser.parse_args()
    runExperiment(args.input, args.output)
