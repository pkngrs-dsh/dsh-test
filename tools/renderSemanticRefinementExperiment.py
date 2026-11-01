"""Compare different geometry/texture branches with frozen photo cameras.

Evaluation support is the intersection of all branch rasters and observations,
computed before illumination fitting. Grey-mesh views and semantic curve errors
are kept separate from in-sample RGB reproduction scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.appearanceObservations import srgbToLinear
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.photoLighting import fitPhotoShLighting
from renderCropComparison import (asImage, compareProjection, fileHash, makeGrid, originalMask,
                                  photoCamera, prepareGeometry, sharedPhotoCrop, validatePhotoPair)
from renderReferenceExperiment import (commonEvaluationMask, displayLinear, photoMetrics,
                                       validateBranch, canonicalLights)
from runReferenceExperiment import readArrays
from runSemanticRefinementExperiment import loadSemanticCache


def regionMasks(labelMap, common):
    """Predetermined automatic photo regions, independent of branch predictions."""
    groups = {"brows": [2, 3], "eyes": [4, 5], "nose": [10], "mouth": [11, 12, 13]}
    regions = {}
    for name, labels in groups.items():
        mask = np.isin(labelMap, labels).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8)).astype(bool) & common
        if mask.any():
            regions[name] = mask
    return regions


def renderComparison(source: Path, output: Path, *, device="cuda", panelSize=360):
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
    import torch

    start = perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty comparison directory")
    report = json.loads((source / "experimentReport.json").read_text())
    if report.get("status") != "prepared":
        raise ValueError("Prepared refinement experiment required")
    for name, digest in report["outputHashes"].items():
        if fileHash(source / name) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    photos = [validatePhotoPair(item, item) for item in inputs]
    if fileHash(Path(report["semanticCache"]) / "manifest.json") != report["semanticManifestSha256"]:
        raise ValueError("Semantic cache changed after experiment preparation")
    semantics, _ = loadSemanticCache(Path(report["semanticCache"]), inputs)
    branches, geometries, savedGeometries = {}, {}, {}
    branchInfo = report["branches"]
    for name, info in branchInfo.items():
        branch = readArrays(source / info["texture"])
        validateBranch(branch, name)
        branches[name] = branch
        if info["geometry"] not in geometries:
            saved = readArrays(source / info["geometry"])
            savedGeometries[info["geometry"]] = saved
            geometries[info["geometry"]] = prepareGeometry(saved, branch["albedo"].shape[0])
    baselineGeometry = savedGeometries[branchInfo["baseline"]["geometry"]]
    output.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    allPanels = {mode: [] for mode in ("matched", "fixed", "gray", "overlay", "unlit")}
    comparisonColumns = len(branchInfo) + 1
    allLabels, viewReports = [], []
    for index, photo in enumerate(photos):
        print(f"Rendering same-camera comparison {index+1}/{len(photos)}", flush=True)
        directory = output / f"view{index:02d}"
        directory.mkdir()
        photo.save(directory / "original.png")
        rgb = np.asarray(photo, np.float32)/255
        observed = srgbToLinear(rgb)
        faceMask = originalMask(baselineGeometry["faceMasks"][index], inputs[index]["inputToWorking"], photo.size)
        crop = sharedPhotoCrop(faceMask, faceMask)
        camera = photoCamera(cameras[index], inputs[index], "fitted")
        renderers, projectionChecks = {}, {}
        for name, geometry in geometries.items():
            renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=rgb.shape[:2], frameGeometry=False)
            renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
            renderers[name] = renderer
            projectionChecks[name] = compareProjection(renderer, geometry, camera)
        visibleMasks = {name: renderers[info["geometry"]].visible.cpu().numpy() for name, info in branchInfo.items()}
        sampledCoverage = {name: renderers[info["geometry"]].sampleAtlas(branches[name]["coverageMask"]).squeeze(-1).cpu().numpy()
                           for name, info in branchInfo.items()}
        commonVisible = np.logical_and.reduce(list(visibleMasks.values()))
        common = commonEvaluationMask(commonVisible, faceMask, list(sampledCoverage.values()))
        regions = regionMasks(semantics[index]["labelMap"], common)
        writeImage(directory / "commonEvaluationMask.png", common)
        np.savez_compressed(directory / "commonPhotoData.npz", observedLinear=observed,
                            evaluationMask=common, **{f"region_{key}": value for key, value in regions.items()})
        images, metrics = {}, {}
        light = camera["rotation"].T @ np.array([-.5, -.5, -1.])
        for name, info in branchInfo.items():
            renderer, branch = renderers[info["geometry"]], branches[name]
            albedo = renderer.sampleAtlas(branch["linearAlbedo"]).cpu().numpy()
            normals = renderer.normals.cpu().numpy()
            fitted = fitPhotoShLighting(albedo, observed, normals, common,
                                        initialCoefficients=None, maxSamples=20000, ridge=.02)
            prediction = fitted["renderedLinear"]
            visible = visibleMasks[name]
            supported = visible & (sampledCoverage[name] >= .999)
            shown = prediction.copy()
            shown[~supported] = .07
            material = renderer.prepareMaterial(branch["albedo"], branch["coverageMask"])
            grayMaterial = {"albedo": torch.full_like(material["albedo"], .35),
                            "normals": renderer.normals, "specular": torch.zeros_like(material["specular"])}
            matched = displayLinear(shown, visible)
            unlit = albedo.copy()
            unlit[~supported] = .07
            overlay = rgb.copy()
            overlay[visible] = .5 * rgb[visible] + .5 * matched[visible]
            images[name] = {"matched": matched, "overlay": overlay, "unlit": displayLinear(unlit, visible),
                            "fixed": renderer.render(material, light, roughness=.45, keyIntensity=2.5, ambient=.18),
                            "gray": renderer.render(grayMaterial, light, roughness=.45, keyIntensity=2.5, ambient=.18)}
            for mode, values in images[name].items():
                writeImage(directory / f"{name}{mode.title()}.png", values)
            np.savez_compressed(directory / f"{name}PhotoFit.npz", renderedLinear=prediction,
                                coefficients=fitted["coefficients"], evaluationMask=common, fitMask=fitted["fitMask"])
            metrics[name] = {"face": photoMetrics(prediction, observed, common),
                             "regions": {key: photoMetrics(prediction, observed, mask) for key, mask in regions.items()},
                             "lighting": fitted["diagnostics"],
                             "supportedPhotoPixels": int((supported & (faceMask >= .999)).sum()),
                             "visibleMeshPixels": int(visible.sum())}
        labels = [f"View {index+1}: original"] + [info["label"] for info in branchInfo.values()]
        allLabels.extend(labels)
        for mode in allPanels:
            panels = [photo.crop(crop)] + [asImage(images[name][mode]).crop(crop) for name in branches]
            allPanels[mode].extend(panels)
            subtitle = ("Frozen photo cameras | same 27-parameter SH lighting fit | common evaluation pixels"
                        if mode == "matched" else
                        "50% original photo / fitted SH render | shared original-image crop"
                        if mode == "overlay" else "Unlit albedo | lit original is a positional reference only"
                        if mode == "unlit" else "Frozen photo cameras | identical display crop and fixed GGX light")
            makeGrid(panels, labels, columns=comparisonColumns, panelSize=panelSize, subtitle=subtitle).save(directory / f"{mode}Comparison.png")
            if index == report["referenceViewIndex"]:
                makeGrid(panels, labels, columns=comparisonColumns, panelSize=panelSize, subtitle=subtitle).save(output / f"front{mode.title()}.png")
        if index == report["referenceViewIndex"]:
            for regionName, mask in regionMasks(semantics[index]["labelMap"], np.ones_like(common)).items():
                yy, xx = np.nonzero(mask)
                detailCrop = (int(xx.min())-8, int(yy.min())-8, int(xx.max())+9, int(yy.max())+9)
                panels = [photo.crop(detailCrop)] + [asImage(images[name]["matched"]).crop(detailCrop) for name in branches]
                from diagnoseBrowAlignment import makeStrips
                detailScale = min(2., 150/panels[0].height, 410/panels[0].width)
                if detailScale > 1:
                    panels = [panel.resize((round(panel.width*detailScale), round(panel.height*detailScale)),
                                            Image.Resampling.BICUBIC) for panel in panels]
                makeStrips(panels, ["Original"]+[info["label"] for info in branchInfo.values()],
                           output / f"{regionName}Detail.png", columns=comparisonColumns)
        viewReports.append({"viewIndex": index, "originalPhotoCamera": camera,
                            "projectionChecks": projectionChecks, "commonEvaluationPixels": int(common.sum()),
                            "branches": metrics, "sharedDisplayCropXyxy": crop})
        del renderer, renderers
    for mode, panels in allPanels.items():
        makeGrid(panels, allLabels, columns=comparisonColumns, panelSize=panelSize,
                 subtitle=f"Refinement comparison: {mode} | all original photo cameras fixed").save(output / f"{mode}Overview.png")

    # Normalize every branch with the baseline center/scale, never independent framing.
    base = geometries[branchInfo["baseline"]["geometry"]]["vertices"]
    center = (base.max(axis=0)+base.min(axis=0))/2
    scale = 1.9 / np.ptp(base[:, :2], axis=0).max()
    panels, labels = [], []
    for name, info in branchInfo.items():
        geometry = {**geometries[info["geometry"]]}
        geometry["vertices"] = (geometry["vertices"]-center)*scale
        renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=640, frameGeometry=False)
        renderer.setView(azimuth=0, elevation=0)
        material = renderer.prepareMaterial(branches[name]["albedo"], branches[name]["coverageMask"])
        for lightName, light in canonicalLights.items():
            rendered = renderer.render(material, light, roughness=.45, keyIntensity=2.5, ambient=.18)
            writeImage(output / f"canonical{name.title()}{lightName.title()}.png", rendered)
            panels.append(asImage(rendered))
            labels.append(f"{info['label']} / {lightName}")
    makeGrid(panels, labels, columns=3, panelSize=panelSize,
             subtitle="Same baseline framing and three GGX lights | detail normals/specular are not reconstructed").save(output / "canonicalRelight.png")
    summary = {"status": "rendered", "source": source, "experimentReportSha256": fileHash(source / "experimentReport.json"),
               "renderScriptSha256": fileHash(Path(__file__)),
               "referenceViewIndex": report["referenceViewIndex"], "views": viewReports,
               "seconds": perf_counter()-start,
               "peakGpuAllocatedGiB": torch.cuda.max_memory_allocated()/2**30 if device.startswith("cuda") else None,
               "cameraParametersFrozen": True, "sameLightingParameterCount": 27,
               "evaluation": "Shared pre-fit intersection of all mesh visibility and observed atlas coverage, eroded 3 original pixels",
               "regionEvaluation": "Original-photo automatic semantic regions dilated by 6 pixels, intersected with common support",
               "limits": report["limits"], "canonicalFraming": {"center": center, "scale": scale}}
    writeJson(output / "comparisonReport.json", summary)
    print(json.dumps({"status": summary["status"], "seconds": summary["seconds"],
                      "frontMae255": {key: value["face"]["srgbMae255"] for key, value in viewReports[report["referenceViewIndex"]]["branches"].items()}}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticRefinementExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/semanticRefinementComparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    renderComparison(args.input, args.output, device=args.device)
