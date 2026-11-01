"""Cached four-way ablation of bounded UV alignment and frequency fusion.

All branches reuse the same geometry, cameras, SH fit and semantic cache.
No learned models run. The reference observation remains an exact anchor.
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
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.frequencyFusion import fuseFrequencyObservations
from faceAppearance.semanticFusion import fuseSemanticObservations
from faceAppearance.uvAlignment import alignUvObservations, sampleUvArray
from runReferenceExperiment import exportBranch, fileHash, readArrays


def runExperiment(source: Path, output: Path, *, maxDisplacement=6., lowFrequencySigma=3.):
    start = perf_counter()
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty experiment directory")
    sourceReport = json.loads((source / "experimentReport.json").read_text())
    sourceDiagnostics = json.loads((source / "branches/semanticProtected.json").read_text())
    observations = readArrays(source / "semanticProtectedObservations.npz")
    fitted = readArrays(source / "semanticProtectedShFit.npz")
    semantics = readArrays(source / "semanticProtectedSemantics.npz")
    geometry = readArrays(source / "geometry.npz")
    oldBranch = readArrays(source / "branches/semanticProtected.npz")
    reference = sourceReport["referenceViewIndex"]
    options = {"referenceViewIndex": reference,
               "referenceBlendWidth": sourceDiagnostics["referenceBlendWidthUvPixels"],
               "referenceMinCosine": sourceDiagnostics["referenceMinCosine"],
               "referenceFullCosine": sourceDiagnostics["referenceFullCosine"]}
    colors, weights = fitted["correctedObservations"], fitted["validWeights"]
    cosines = observations["viewingCosines"]
    labels, confidence = semantics["semanticLabels"], semantics["semanticConfidence"]
    baseline = fuseSemanticObservations(colors, weights, cosines, labels, confidence, **options)
    for name in ("linearAlbedo", "coverageMask", "referenceCoreMask", "protectionMask"):
        if not np.array_equal(baseline[name], oldBranch[name]):
            raise ValueError(f"Baseline failed exact historical reproduction: {name}")
    output.mkdir(parents=True)
    (output / "branches").mkdir()
    for name in ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    print("Aligning auxiliary observations with fixed reference and bounded residual flow", flush=True)
    alignmentStart = perf_counter()
    alignment = alignUvObservations(colors, weights, labels, confidence, referenceViewIndex=reference,
                                    maxDisplacement=maxDisplacement, taperWidth=max(12., maxDisplacement*2))
    alignmentSeconds = perf_counter()-alignmentStart
    alignedCosines, alignedPixels = cosines.copy(), observations["originalSamplingPixels"].copy()
    for index, coordinates in enumerate(alignment["samplingCoordinates"]):
        accepted = alignment["acceptedMask"][index]
        alignedCosines[index][accepted] = np.minimum(
            cosines[index], sampleUvArray(cosines[index], coordinates))[accepted]
        alignedPixels[index][accepted] = sampleUvArray(observations["originalSamplingPixels"][index], coordinates)[accepted]
    alignedArguments = (alignment["alignedObservations"], alignment["alignedWeights"], alignedCosines,
                        alignment["alignedSemanticLabels"], alignment["alignedSemanticConfidence"])
    print("Preparing independent alignment, frequency, and combined branches", flush=True)
    branches = {"baseline": baseline,
                "aligned": fuseSemanticObservations(*alignedArguments, **options),
                "frequency": fuseFrequencyObservations(colors, weights, cosines, labels, confidence,
                                                        lowFrequencySigma=lowFrequencySigma, **options),
                "combined": fuseFrequencyObservations(*alignedArguments, lowFrequencySigma=lowFrequencySigma, **options)}
    branchLabels = {"baseline": "A: Current semantic", "aligned": "B: UV alignment",
                    "frequency": "C: Frequency fusion", "combined": "D: Alignment + frequency"}
    immutable = baseline["referenceCoreMask"] | baseline["protectionMask"]
    invariants = {"baselineBitExact": True,
                  "referenceObservationBitExact": np.array_equal(alignment["alignedObservations"][reference], colors[reference]),
                  "referenceWeightsBitExact": np.array_equal(alignment["alignedWeights"][reference], weights[reference]),
                  "supportExpandedTexels": int(((alignment["alignedWeights"] > 0) & (weights <= 0)).sum()),
                  "branches": {}}
    for name, result in branches.items():
        invariants["branches"][name] = {
            "referenceCoreAndProtectionMaxError": float(np.abs(result["linearAlbedo"][immutable]-colors[reference][immutable]).max()),
            "coverageBitExact": np.array_equal(result["coverageMask"], baseline["coverageMask"]),
            "observedTexels": int(result["coverageMask"].sum())}
        if (invariants["branches"][name]["referenceCoreAndProtectionMaxError"] != 0
                or not invariants["branches"][name]["coverageBitExact"]):
            raise AssertionError(f"Reference or support contract failed: {name}")
        exportBranch(output, name, result, geometry)
    np.savez_compressed(output / "alignment.npz", **{key: value for key, value in alignment.items() if isinstance(value, np.ndarray)},
                        alignedViewingCosines=alignedCosines, alignedOriginalSamplingPixels=alignedPixels)
    writeJson(output / "alignment.json", alignment["diagnostics"])
    for index in range(len(colors)):
        writeImage(output / f"view{index:02d}Accepted.png", alignment["acceptedMask"][index])
        magnitude = np.linalg.norm(alignment["samplingFlow"][index], axis=-1)
        writeImage(output / f"view{index:02d}FlowMagnitude.png", magnitude/maxDisplacement)
    implementation = ("src/faceAppearance/uvAlignment.py", "src/faceAppearance/frequencyFusion.py",
                      "src/faceAppearance/semanticFusion.py", "src/faceAppearance/referenceFusion.py",
                      "tools/runGhostingExperiment.py")
    for name in implementation:
        destination = output / "implementation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(projectRoot / name, destination)
    sourceFiles = ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json", "semanticProtectedObservations.npz",
                   "semanticProtectedShFit.npz", "semanticProtectedSemantics.npz", "branches/semanticProtected.npz")
    outputFiles = ("geometry.npz", "inputs.json", "cameras.json", "alignment.npz") + tuple(f"branches/{name}.npz" for name in branches)
    report = {"status": "prepared", "source": source, "referenceViewIndex": reference,
              "semanticCache": sourceReport["semanticCache"], "semanticManifestSha256": sourceReport["semanticManifestSha256"],
              "branches": {name: {"label": branchLabels[name], "geometry": "geometry.npz", "texture": f"branches/{name}.npz"} for name in branches},
              "cameraParametersFrozen": True, "geometryFrozen": True, "shFitFrozen": True,
              "wildCapEnabled": False, "maxDisplacementUvPixels": maxDisplacement,
              "lowFrequencySigmaUvPixels": lowFrequencySigma, "fusionOptions": options,
              "invariants": invariants, "alignmentSeconds": alignmentSeconds, "seconds": perf_counter()-start,
              "sourceHashes": {name: fileHash(source / name) for name in sourceFiles},
              "outputHashes": {name: fileHash(output / name) for name in outputFiles},
              "implementationHashes": {name: fileHash(projectRoot / name) for name in implementation},
              "limits": ["All photographs are fitting inputs, not held-out validation",
                         "Reference anchoring is exact in the atlas; original-camera renders still have sampling and mesh errors",
                         "Residual alignment only changes accepted auxiliary samples, not cameras, geometry, or occluded support",
                         "Aligned photo coordinates interpolate the prior correspondence map; colors resample corrected UV observations, not original photos directly",
                         "Alignment structural error is its own fitting objective; it cannot establish true correspondence",
                         "Frequency fusion uses a single high-frequency source but can retain low-frequency duplication and source seams",
                         "Whole-feature ownership requires reliable coverage; unsupported or ambiguous regions are not forced",
                         "Geometry openings, residual baked lighting and missing eyeballs/teeth remain",
                         "Cached CPU preparation and diagnostic rendering are not full pipeline runtime"]}
    writeJson(output / "experimentReport.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"], "invariants": invariants}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/ghostingRefinementExperiment")
    parser.add_argument("--max-displacement", type=float, default=6.)
    parser.add_argument("--low-frequency-sigma", type=float, default=3.)
    args = parser.parse_args()
    runExperiment(args.input, args.output, maxDisplacement=args.max_displacement, lowFrequencySigma=args.low_frequency_sigma)
