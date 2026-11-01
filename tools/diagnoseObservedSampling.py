"""Inspect partial atlas footprints without filling entirely unobserved areas.

This is a separate display/sampling diagnostic. Historical fusion, lighting
fits and strict evaluation support remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.observedSampling import sampleObservedAtlas
from renderCropComparison import asImage, fileHash, makeGrid, photoCamera, prepareGeometry, validatePhotoPair
from renderReferenceExperiment import displayLinear
from renderSemanticRefinementExperiment import regionMasks
from runReferenceExperiment import readArrays
from runSemanticRefinementExperiment import loadSemanticCache


def diagnose(source: Path, output: Path, *, device="cuda"):
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    report = json.loads((source / "experimentReport.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    semantics, _ = loadSemanticCache(Path(report["semanticCache"]), inputs)
    observation = readArrays(source / "semanticProtectedObservations.npz")
    branch = readArrays(source / "branches/semanticProtected.npz")
    geometry = prepareGeometry(readArrays(source / "geometry.npz"), branch["linearAlbedo"].shape[0])
    index = report["referenceViewIndex"]
    photo = validatePhotoPair(inputs[index], inputs[index])
    camera = photoCamera(cameras[index], inputs[index], "fitted")
    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=(photo.height, photo.width), frameGeometry=False)
    renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
    visible = renderer.visible.cpu().numpy()
    output.mkdir(parents=True)
    regions = regionMasks(semantics[index]["labelMap"], visible)
    reports = {}
    for name, colors, coverage in (("ownRaw", observation["rawLinear"][index], observation["weights"][index] > 0),
                                   ("fusedSh", branch["linearAlbedo"], branch["coverageMask"])):
        sampled = renderer.sampleAtlas(colors).cpu().numpy()
        normalized = sampleObservedAtlas(renderer, colors, coverage, minimumCoverage=.5)
        strictMask = visible & (normalized["sampleCoverage"] >= .999)
        usable = visible & normalized["validMask"]
        strict, corrected = sampled.copy(), normalized["linearColors"].copy()
        strict[~strictMask] = .07
        corrected[~usable] = .07
        strict, corrected = displayLinear(strict, visible), displayLinear(corrected, visible)
        writeImage(output / f"{name}Strict.png", strict)
        writeImage(output / f"{name}Normalized.png", corrected)
        np.savez_compressed(output / f"{name}Support.npz", sampleCoverage=normalized["sampleCoverage"],
                            strictMask=strictMask, normalizedMask=usable, visible=visible)
        panels, labels = [], []
        regionReports = {}
        for regionName in ("mouth", "nose", "eyes"):
            region = regions[regionName]
            yy, xx = np.nonzero(region)
            crop = (int(xx.min())-10, int(yy.min())-10, int(xx.max())+11, int(yy.max())+11)
            panels.extend([photo.crop(crop), asImage(strict).crop(crop), asImage(corrected).crop(crop)])
            labels.extend([f"{regionName}: original", "Strict 99.9% footprint", "Normalized >=50% footprint"])
            regionReports[regionName] = {"meshVisiblePixels": int(region.sum()),
                "strictUnsupportedPixels": int((region & ~strictMask).sum()),
                "normalizedUnsupportedPixels": int((region & ~usable).sum()),
                "completelyUnobservedPixels": int((region & (normalized["sampleCoverage"] == 0)).sum()),
                "acceptedPartialFootprintPixels": int((region & usable & ~strictMask).sum())}
        makeGrid(panels, labels, columns=3, panelSize=300,
                 subtitle=("Own lit-photo roundtrip; gray is missing observation, NOT a geometric hole" if name == "ownRaw" else
                           "Fused unlit albedo; original lit photo is a positional reference only")).save(output / f"{name}Details.png")
        reports[name] = regionReports
    summary = {"status": "complete", "source": source, "minimumCoverage": .5, "regions": reports,
               "geometrySha256": fileHash(source / "geometry.npz"), "implementationSha256": fileHash(Path(__file__)),
               "limits": ["Normalize only actual observed bilinear contributors; zero-coverage footprints remain missing",
                          "At least half of the interpolation footprint must be observed; this threshold is a diagnostic choice",
                          "No reconstruction, geometry, fusion or historical evaluation mask was changed",
                          "Partial footprint acceptance removes a display discontinuity; it does not recover new measurements"]}
    writeJson(output / "report.json", summary)
    print(json.dumps(summary["regions"]))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/observedSamplingDiagnostic")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args.input, args.output, device=args.device)
