"""Separate texture mixing from geometric openings in frozen photo cameras.

The original-photo round trip is a sampling diagnostic, not recovered albedo
or an independent geometry accuracy measurement. Unsupported pixels stay gray.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeImage, writeJson
from renderCropComparison import (asImage, fileHash, makeGrid, originalMask, photoCamera,
                                  prepareGeometry, sharedPhotoCrop, validatePhotoPair)
from renderReferenceExperiment import displayLinear
from renderSemanticRefinementExperiment import regionMasks
from runReferenceExperiment import readArrays
from runSemanticRefinementExperiment import loadSemanticCache


def diagnose(source: Path, output: Path, *, device="cuda"):
    import torch
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    start = perf_counter()
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty diagnostic directory")
    sourceReport = json.loads((source / "experimentReport.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    semantics, _ = loadSemanticCache(Path(sourceReport["semanticCache"]), inputs)
    saved = readArrays(source / "geometry.npz")
    observations = readArrays(source / "semanticProtectedObservations.npz")
    fitted = readArrays(source / "semanticProtectedShFit.npz")
    fused = readArrays(source / "branches/semanticProtected.npz")
    geometry = prepareGeometry(saved, fused["linearAlbedo"].shape[0])
    output.mkdir(parents=True)
    rows, labels, reports = [], [], []
    referenceIndex = sourceReport["referenceViewIndex"]
    for index, item in enumerate(inputs):
        print(f"Diagnosing source / geometry {index+1}/{len(inputs)}", flush=True)
        photo = validatePhotoPair(item, item)
        camera = photoCamera(cameras[index], item, "fitted")
        renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=(photo.height, photo.width),
                                        frameGeometry=False)
        renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
        visible = renderer.visible.cpu().numpy()
        directory = output / f"view{index:02d}"
        directory.mkdir()
        photo.save(directory / "original.png")
        shown = {}
        supportCounts = {}
        variants = {
            "ownRaw": (observations["rawLinear"][index], observations["weights"][index] > 0),
            "ownSh": (fitted["correctedObservations"][index], fitted["validWeights"][index] > 0),
            "frontSh": (fitted["correctedObservations"][referenceIndex], fitted["validWeights"][referenceIndex] > 0),
            "fusedSh": (fused["linearAlbedo"], fused["coverageMask"]),
        }
        for name, (color, valid) in variants.items():
            sampled = renderer.sampleAtlas(color).cpu().numpy()
            supported = visible & (renderer.sampleAtlas(valid).squeeze(-1).cpu().numpy() >= .999)
            sampled[~supported] = .07
            shown[name] = displayLinear(sampled, visible)
            writeImage(directory / f"{name}.png", shown[name])
            writeImage(directory / f"{name}Support.png", supported)
            supportCounts[name] = int(supported.sum())
        material = renderer.prepareMaterial(fused["albedo"], fused["coverageMask"])
        gray = {"albedo": torch.full_like(material["albedo"], .35), "normals": renderer.normals,
                "specular": torch.zeros_like(material["specular"])}
        light = camera["rotation"].T @ np.array([-.5, -.5, -1.])
        shown["gray"] = renderer.render(gray, light)
        writeImage(directory / "gray.png", shown["gray"])
        writeImage(directory / "meshVisible.png", visible)
        faceMask = originalMask(saved["faceMasks"][index], item["inputToWorking"], photo.size)
        crop = sharedPhotoCrop(faceMask, faceMask)
        rowLabels = [f"View {index+1}: original", "Own photo roundtrip (LIT)", "Own SH / unlit",
                     "Front SH / unlit", "Fused SH / unlit", "Geometry only"]
        panels = [photo.crop(crop)] + [asImage(values).crop(crop) for values in shown.values()]
        rows.extend(panels)
        labels.extend(rowLabels)
        caption = "Same frozen camera / geometry | unlit colors are NOT photo-matched | gray = unsupported, dark = mesh holes"
        makeGrid(panels, rowLabels, columns=6, panelSize=270, subtitle=caption).save(directory / "comparison.png")
        if index == referenceIndex:
            makeGrid(panels, rowLabels, columns=6, panelSize=270, subtitle=caption).save(output / "frontDiagnostic.png")
            from diagnoseBrowAlignment import makeStrips
            openingPanels, openingLabels = [], []
            for name, mask in regionMasks(semantics[index]["labelMap"], np.ones_like(visible)).items():
                yy, xx = np.nonzero(mask)
                detailCrop = (int(xx.min())-10, int(yy.min())-10, int(xx.max())+11, int(yy.max())+11)
                details = [photo.crop(detailCrop)] + [asImage(value).crop(detailCrop) for value in shown.values()]
                makeStrips(details, rowLabels, output / f"{name}Detail.png", columns=6)
                if name in ("eyes", "nose", "mouth"):
                    openingPanels.extend([photo.crop(detailCrop), asImage(shown["ownRaw"]).crop(detailCrop),
                                          asImage(shown["gray"]).crop(detailCrop)])
                    openingLabels.extend([f"{name}: original", "One photo only (LIT)", "Geometry only"])
            makeGrid(openingPanels, openingLabels, columns=3, panelSize=300,
                     subtitle="Same camera | gray = unsupported observation | internal mouth rim is not the visible lip seam").save(output / "openingDetail.png")
        reports.append({"viewIndex": index, "supportedPixels": supportCounts,
                        "visibleMeshPixels": int(visible.sum()), "camera": camera})
    makeGrid(rows, labels, columns=6, panelSize=270,
             subtitle="Source isolation: lit-photo roundtrip, unlit single/fused albedo, and geometry").save(output / "overview.png")
    used = ["geometry.npz", "inputs.json", "cameras.json", "semanticProtectedObservations.npz",
            "semanticProtectedShFit.npz", "branches/semanticProtected.npz"]
    report = {"status": "complete", "source": source, "seconds": perf_counter()-start, "views": reports,
              "sourceHashes": {name: fileHash(source / name) for name in used},
              "implementationSha256": fileHash(Path(__file__)),
              "limits": ["Own-photo roundtrip reuses the same camera for projection and rendering; it does not validate geometry",
                         "Unlit albedo is not expected to match lit photographs in color",
                         "Eye/mouth/nose openings remain empty; no eyeballs, teeth or oral cavity have been installed",
                         "A single-source gray gap must be checked against visibility/support masks; it does not prove a displaced geometry hole",
                         "Different single-source support is explicitly gray; no support-conditioned quality score is used"]}
    writeJson(output / "diagnosticReport.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/ghostingSourceDiagnostic")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args.input, args.output, device=args.device)
