"""Local frontal OpenDelight reprojection on already refined UFF geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.appearanceObservations import collectAppearanceObservations
from faceAppearance.artifacts import writeJson
from faceAppearance.referenceFeaturePriority import _curveDistances
from runReferenceExperiment import readArrays, fileHash


def prepare(record):
    started = perf_counter()
    fit = Path(record["resultPath"]).parent
    target = fit / "texturePriority.npz"
    if target.exists():
        raise FileExistsError(target)
    source = Path(record["sourceDirectory"])
    geometry, texture = readArrays(Path(record["geometryPath"])), readArrays(Path(record["texturePath"]))
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    reference = record["frontViewIndex"]
    delight = readArrays(source / "delight.npz")
    # Use just the qualified reference photo, at the new geometry's projected
    # coordinates; dense UFF UV correspondences are deliberately not consulted.
    selectedDelight = {name: delight[name][reference:reference+1] for name in ("albedoImages", "validMasks")}
    observations = collectAppearanceObservations(geometry, [cameras[reference]], [inputs[reference]],
        selectedDelight, texture["albedo"].shape[0], samplingMode="projection", visibilityMode="refineRejected")
    curves = json.loads((fit / "workingPhotoContours.json").read_text())[reference]
    featureCurves = [np.asarray(eye[part], float) for eye in curves["eyes"].values() if eye["valid"]
                     for part in ("upperCurve", "lowerCurve")]
    if len(featureCurves) != 4:
        raise ValueError("Two reliable reference eyes required")
    pixels = observations["originalSamplingPixels"][0]
    affine = np.asarray(inputs[reference]["inputToWorking"])
    projected = np.concatenate([pixels, np.ones((*pixels.shape[:2], 1))], -1)@affine.T
    workPixels = projected[..., :2]/projected[..., 2:]
    eligible = texture["coverageMask"] & (observations["weights"][0] > 0)
    distances = _curveDistances(workPixels[eligible], featureCurves, 10.)
    alpha = np.zeros_like(texture["confidence"])
    alpha[eligible] = .5+.5*np.cos(np.pi*np.clip((distances-3.)/7., 0, 1))
    # Clamp remote floating-point cosine residue to an exact unchanged exterior.
    alpha[alpha < 1e-7] = 0
    changed = alpha > 0
    linear = texture["albedo"]**2.2
    referenceColors = observations["openDelightLinear"][0]
    linear[changed] = ((1-alpha[changed, None])*linear[changed]
                       + alpha[changed, None]*referenceColors[changed])
    result = {key: value.copy() for key, value in texture.items()}
    result["albedo"][changed] = np.clip(linear[changed], 0, 1)**(1/2.2)
    if "previewAlbedo" in result:
        result["previewAlbedo"][changed] = result["albedo"][changed]
    if "sourceIndex" in result:
        result["sourceIndex"][changed] = -2
        result["sourceIndex"][alpha == 1] = reference
    result["confidence"][changed] = ((1-alpha[changed])*result["confidence"][changed]
        +alpha[changed]*np.clip(observations["weights"][0][changed], 0, 1))
    result["frontProjectionAlpha"] = alpha
    if not np.array_equal(texture["albedo"][~changed], result["albedo"][~changed]):
        raise AssertionError("Pixels outside local projection changed")
    np.savez_compressed(target, **result)
    report = {"caseId": record["caseId"], "seconds": perf_counter()-started,
        "referenceViewIndex": reference, "geometryFrozen": True, "camerasFrozen": True,
        "coverageExactlyUnchanged": bool(np.array_equal(texture["coverageMask"], result["coverageMask"])),
        "outsideLocalBlendExactlyUnchanged": True, "changedTexels": int(changed.sum()),
        "coreTexels": int((alpha == 1).sum()), "coreRadiusWorkingPixels": 3., "falloffRadiusWorkingPixels": 10.,
        "sourceHashes": {str(path): fileHash(path) for path in (Path(record["geometryPath"]),
            Path(record["texturePath"]), source / "cameras.json", source / "inputs.json", source / "delight.npz",
            fit / "workingPhotoContours.json")},
        "outputSha256": fileHash(target), "observationDiagnostics": observations["diagnostics"],
        "limits": ["Experimental frontal eyelid texture reprojection, not a whole-face identity correction.",
            "OpenDelight native encoding is interpreted as gamma2.2, matching the existing viewer contract.",
            "No new coverage, eyeball geometry, inpainting, or illumination model is introduced.",
            "Local front texture may trade side appearance for front matching; compare fixed side renders."]}
    writeJson(fit / "texturePriorityReport.json", report)
    record["texturePriorityPath"] = str(target)
    print(json.dumps({"caseId": record["caseId"], "seconds": report["seconds"], "changedTexels": report["changedTexels"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/uffFrontPriority20261011")
    args = parser.parse_args()
    path = args.root / "frontManifest.json"
    manifest = json.loads(path.read_text())
    for case in manifest["cases"]:
        if case["status"] == "succeeded" and not case.get("texturePriorityPath"):
            prepare(case)
            writeJson(path, manifest)


if __name__ == "__main__":
    main()
