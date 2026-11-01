"""Render the frozen-geometry reference-view experiment against original photos.

Only 27 global RGB SH illumination parameters are fitted per branch and photo.
Textures, geometry, cameras, coverage and evaluation masks remain fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
for importPath in (projectRoot / "src", Path(__file__).resolve().parent):
    if str(importPath) not in sys.path:
        sys.path.insert(0, str(importPath))

from faceAppearance.appearanceObservations import srgbToLinear
from faceAppearance.artifacts import writeImage, writeJson
from renderCropComparison import (asImage, compareProjection, fileHash, makeGrid,
                                  originalMask, photoCamera, prepareGeometry,
                                  sharedPhotoCrop, validatePhotoPair)

branchNames = ("denseBaseline", "projectionMean", "referenceFirst")
branchLabels = {"denseBaseline": "Dense UV + mean", "projectionMean": "Projection + mean",
                "referenceFirst": "Reference first"}
canonicalLights = {"front": (0., .3, 1.), "left": (-.75, .45, .85), "right": (.75, .45, .85)}
background = np.array([.105, .12, .14], np.float32)


def linearToSrgb(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values), 0)
    return np.clip(np.where(positive <= .0031308, positive * 12.92,
                           1.055 * positive ** (1 / 2.4) - .055), 0, 1)


def validateBranch(values: dict, name: str) -> None:
    for key in ("albedo", "linearAlbedo", "coverageMask"):
        if key not in values or not np.isfinite(values[key]).all():
            raise ValueError(f"Missing or nonfinite {key}: {name}")
    albedo, linear, coverage = values["albedo"], values["linearAlbedo"], values["coverageMask"]
    if (albedo.ndim != 3 or albedo.shape[-1] != 3 or albedo.shape[0] != albedo.shape[1]
            or linear.shape != albedo.shape or coverage.shape != albedo.shape[:2]
            or any(np.any(value < 0) or np.any(value > 1) for value in (albedo, linear, coverage))):
        raise ValueError(f"Invalid RGB atlas or coverage: {name}")
    if not np.allclose(albedo ** 2.2, linear, atol=2e-5, rtol=2e-5):
        raise ValueError(f"gamma-2.2 albedo disagrees with linearAlbedo: {name}")


def loadExperiment(directory: Path) -> dict:
    directory = Path(directory).resolve()
    names = ("geometry.npz", "inputs.json", "cameras.json", "viewTransforms.json", "experimentReport.json")
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete reference experiment: {directory / name}")
    readJson = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    report = readJson("experimentReport.json")
    if report.get("status") != "prepared":
        raise ValueError("Reference experiment must have status=prepared before rendering")
    with np.load(directory / "geometry.npz", allow_pickle=False) as saved:
        savedGeometry = dict(saved)
    inputs, cameras, transforms = readJson("inputs.json"), readJson("cameras.json"), readJson("viewTransforms.json")
    if not inputs or len(cameras) != len(inputs) or len(transforms) != len(inputs):
        raise ValueError("Input, camera and transform counts must match")
    referenceIndex = report["referenceViewIndex"]
    if isinstance(referenceIndex, bool) or not isinstance(referenceIndex, int) or not 0 <= referenceIndex < len(inputs):
        raise ValueError("referenceViewIndex must identify an existing input")
    for name in ("vertices", "faces", "uvs", "faceMasks"):
        if name not in savedGeometry or not np.isfinite(savedGeometry[name]).all():
            raise ValueError(f"Missing or nonfinite saved geometry field: {name}")
    if len(savedGeometry["faceMasks"]) != len(inputs):
        raise ValueError("Saved geometry segmentation view count differs")
    for index, (camera, item, mask) in enumerate(zip(cameras, inputs, savedGeometry["faceMasks"])):
        if not camera.get("valid", False) or camera.get("viewIndex", index) != index:
            raise ValueError("Accepted saved PnP camera required for every ordered view; no refitting")
        for key, shape in (("intrinsics", (3, 3)), ("rotation", (3, 3)), ("translation", (3,))):
            values = np.asarray(camera[key])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved camera {key}")
        if list(mask.shape) != list(reversed(item["workingSize"])):
            raise ValueError("Saved face-mask size differs from working input dimensions")
    photos = [validatePhotoPair(item, item) for item in inputs]
    branches, hashes = {}, {name: fileHash(directory / name) for name in names}
    for name in (*branchNames, "rawReference"):
        path = directory / "branches" / f"{name}.npz"
        with np.load(path, allow_pickle=False) as saved:
            branches[name] = dict(saved)
        validateBranch(branches[name], name)
        hashes[str(path.relative_to(directory))] = fileHash(path)
    shape = branches[branchNames[0]]["linearAlbedo"].shape
    if any(branch["linearAlbedo"].shape != shape for branch in branches.values()):
        raise ValueError("Every branch must share atlas dimensions")
    commonCoverage = np.logical_and.reduce([branches[name]["coverageMask"] >= .999 for name in branchNames])
    if not commonCoverage.any():
        raise ValueError("Three albedo branches have no common observed support")
    return {"directory": directory, "report": report, "inputs": inputs, "cameras": cameras,
            "transforms": transforms, "savedGeometry": savedGeometry, "branches": branches,
            "photos": photos, "referenceViewIndex": referenceIndex, "sourceHashes": hashes,
            "commonCoverage": commonCoverage}


def commonEvaluationMask(visible: np.ndarray, faceMask: np.ndarray,
                         coverageMasks: list[np.ndarray], erosionPixels: int = 3) -> np.ndarray:
    """Define evaluation support before lighting; fit rejection never changes it."""
    visible, faceMask = np.asarray(visible), np.asarray(faceMask)
    if (visible.ndim != 2 or faceMask.shape != visible.shape or not coverageMasks
            or isinstance(erosionPixels, bool) or not isinstance(erosionPixels, int) or erosionPixels < 0):
        raise ValueError("Expected shared 2D raster masks and nonnegative integer erosion")
    mask = (visible > .5) & (faceMask >= .999)
    for coverage in coverageMasks:
        coverage = np.asarray(coverage)
        if coverage.shape != mask.shape or not np.isfinite(coverage).all():
            raise ValueError("Every sampled coverage must share the raster shape")
        mask &= coverage >= .999
    if erosionPixels:
        kernel = np.ones((2 * erosionPixels + 1, 2 * erosionPixels + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not mask.any():
        raise ValueError("No common pixels remain for photo evaluation")
    return mask


def photoMetrics(renderedLinear: np.ndarray, observedLinear: np.ndarray,
                 evaluationMask: np.ndarray) -> dict:
    """In-sample appearance error on all predetermined support, never a fit mask."""
    renderedLinear, observedLinear = np.asarray(renderedLinear), np.asarray(observedLinear)
    mask = np.asarray(evaluationMask, bool)
    if (renderedLinear.shape != observedLinear.shape or renderedLinear.shape != (*mask.shape, 3)
            or not mask.any() or not np.isfinite(renderedLinear).all() or not np.isfinite(observedLinear).all()):
        raise ValueError("Metrics require finite RGB images and nonempty common support")
    prediction, observation = renderedLinear[mask], observedLinear[mask]
    srgbDifference = linearToSrgb(prediction) - linearToSrgb(observation)
    return {"pixelCount": int(mask.sum()),
            "linearRmse": float(np.sqrt(np.mean((prediction - observation) ** 2))),
            "srgbMae": float(np.mean(np.abs(srgbDifference))),
            "srgbMae255": float(np.mean(np.abs(srgbDifference)) * 255),
            "negativePredictionPixelFraction": float(np.any(prediction < 0, axis=-1).mean()),
            "displayClippedPixelFraction": float(np.any((prediction < 0) | (prediction > 1), axis=-1).mean()),
            "evaluation": "All independently defined common pixels, including fit-rejected observations"}


def displayLinear(values: np.ndarray, visible: np.ndarray) -> np.ndarray:
    result = linearToSrgb(values)
    result[~visible] = background
    return result


def makeStrip(panels: list[Image.Image], labels: list[str], path: Path, subtitle: str, panelSize: int):
    makeGrid(panels, labels, columns=len(panels), panelSize=panelSize, subtitle=subtitle).save(path)


def renderPhotoView(experiment: dict, geometry: dict, index: int, output: Path, *,
                    device: str, panelSize: int, maxSamples: int, ridge: float) -> tuple[dict, dict]:
    from faceAppearance.photoLighting import fitPhotoShLighting
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    directory = output / f"view{index:02d}"
    directory.mkdir()
    photoImage = experiment["photos"][index]
    photoImage.save(directory / "original.png")
    photo = np.asarray(photoImage, np.float32) / 255
    observed = srgbToLinear(photo)
    faceMask = originalMask(experiment["savedGeometry"]["faceMasks"][index],
                            experiment["inputs"][index]["inputToWorking"], photoImage.size)
    crop = sharedPhotoCrop(faceMask, faceMask)
    camera = photoCamera(experiment["cameras"][index], experiment["inputs"][index], "fitted")
    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=photo.shape[:2], frameGeometry=False)
    renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
    projection = compareProjection(renderer, geometry, camera)
    visible = renderer.visible.detach().cpu().numpy().copy()
    normals = renderer.normals.detach().cpu().numpy().copy()
    coverages = {name: renderer.sampleAtlas(experiment["branches"][name]["coverageMask"]).squeeze(-1).cpu().numpy()
                 for name in branchNames}
    evaluationMask = commonEvaluationMask(visible, faceMask, list(coverages.values()))
    light = camera["rotation"].T @ np.array([-.5, -.5, -1.])
    writeImage(directory / "commonEvaluationMask.png", evaluationMask)
    writeImage(directory / "faceMask.png", faceMask)
    writeImage(directory / "rasterMask.png", visible)
    images, fits = {}, {}
    for name in branchNames:
        branch = experiment["branches"][name]
        sampledAlbedo = renderer.sampleAtlas(branch["linearAlbedo"]).cpu().numpy()
        fitted = fitPhotoShLighting(sampledAlbedo, observed, normals, evaluationMask,
                                    initialCoefficients=None, maxSamples=maxSamples, ridge=ridge)
        prediction = np.asarray(fitted["renderedLinear"])
        if prediction.shape != photo.shape or not np.isfinite(prediction).all():
            raise FloatingPointError(f"Invalid fitted-light image: view {index}, {name}")
        supported = visible & (coverages[name] >= .999)
        # Missing observations use neutral gray only for display; no hidden skin fill.
        unlitLinear, matchedLinear = sampledAlbedo.copy(), prediction.copy()
        unlitLinear[~supported], matchedLinear[~supported] = .07, .07
        material = renderer.prepareMaterial(branch["albedo"], branch["coverageMask"])
        images[name] = {"matched": displayLinear(matchedLinear, visible),
                        "unlit": displayLinear(unlitLinear, visible),
                        "fixed": renderer.render(material, light, roughness=.45, keyIntensity=2.5, ambient=.18)}
        images[name]["overlay"] = photo.copy()
        images[name]["overlay"][visible] = .5 * photo[visible] + .5 * images[name]["matched"][visible]
        for mode, image in images[name].items():
            writeImage(directory / f"{name}{mode.title()}.png", image)
        np.savez_compressed(directory / f"{name}PhotoFit.npz", coefficients=fitted["coefficients"],
                            renderedLinear=prediction, sampledAlbedoLinear=sampledAlbedo,
                            fitMask=fitted["fitMask"], validMask=fitted["validMask"],
                            evaluationMask=evaluationMask)
        metrics = photoMetrics(prediction, observed, evaluationMask)
        fits[name] = {"metrics": metrics, "diagnostics": fitted["diagnostics"],
                      "fitMaskPixelsInEvaluation": int((np.asarray(fitted["fitMask"], bool) & evaluationMask).sum()),
                      "initialCoefficients": "Same neutral SH for every branch; initialization is not branch-specific"}
        writeJson(directory / f"{name}PhotoFit.json", fits[name])
    np.savez_compressed(directory / "commonPhotoData.npz", observedLinear=observed,
                        normalsCanonical=normals, evaluationMask=evaluationMask,
                        rasterVisible=visible, faceMask=faceMask)
    panelGroups = {}
    for mode, subtitle in (("matched", "Frozen geometry + camera | only 27 global SH lighting parameters fitted per image"),
                            ("fixed", "Same camera and fixed GGX light | source photo has different illumination"),
                            ("unlit", "Unlit albedo projection | original photo still contains illumination"),
                            ("overlay", "50% original / fitted SH render | same camera and shared display crop")):
        panels = [photoImage.crop(crop)] + [asImage(images[name][mode]).crop(crop) for name in branchNames]
        labels = [f"View {index + 1}: original"] + [branchLabels[name] for name in branchNames]
        panelGroups[mode] = {"panels": panels, "labels": labels}
        makeStrip(panels, labels, directory / f"{mode}Comparison.png", subtitle, panelSize)
    report = {"viewIndex": index, "sourceSha256": experiment["inputs"][index]["sha256"],
              "originalImageSizeWh": list(photoImage.size), "originalPhotoCamera": camera,
              "projectionCheck": projection, "frameGeometry": False, "rasterPrincipalPointShift": [.5, .5],
              "commonEvaluationPixels": int(evaluationMask.sum()), "evaluationMaskErosionPixels": 3,
              "sharedDisplayCropXyxy": crop, "fixedLightDirectionCanonical": light, "branches": fits}
    if index == experiment["referenceViewIndex"]:
        makeStrip(**panelGroups["matched"], path=output / "frontOverview.png",
                  subtitle="Reference photo | camera and geometry frozen | independently fitted SH illumination", panelSize=panelSize)
        # This explicit diagnostic crop belongs to the current 720x1280 source.
        if photoImage.size == (720, 1280):
            browCrop = (65, 370, 405, 485)
            panels = [photoImage.crop(browCrop)] + [asImage(images[name]["matched"]).crop(browCrop) for name in branchNames]
            makeStrip(panels, ["Original brow region"] + [branchLabels[name] for name in branchNames],
                      output / "browComparison.png", "Same original-pixel crop; manually chosen inspection region", panelSize)
            report["browCropXyxy"] = browCrop
        raw = experiment["branches"]["rawReference"]
        rawLinear = renderer.sampleAtlas(raw["linearAlbedo"]).cpu().numpy()
        rawCoverage = renderer.sampleAtlas(raw["coverageMask"]).squeeze(-1).cpu().numpy()
        rawSupported = visible & (rawCoverage >= .999)
        rawLinear[~rawSupported] = .07
        rawImage = displayLinear(rawLinear, visible)
        writeImage(directory / "rawReferenceUnlit.png", rawImage)
        makeStrip([photoImage.crop(crop), asImage(rawImage).crop(crop)],
                  ["Original reference", "Lit-photo texture / unlit projection"], output / "referenceProjection.png",
                  "Coordinate diagnostic only: lit source-photo colors are NOT reflectance / PBR albedo", panelSize)
        reference = experiment["branches"]["referenceFirst"]
        if "referenceAlpha" in reference:
            alpha = renderer.sampleAtlas(reference["referenceAlpha"]).squeeze(-1).cpu().numpy()
            heatmap = np.stack([alpha, .25 * np.ones_like(alpha), 1 - alpha], axis=-1)
            heatmap[~visible] = background
            writeImage(directory / "referenceContribution.png", heatmap)
    writeJson(directory / "comparisonReport.json", report)
    del renderer
    return panelGroups, report


def renderCanonical(experiment: dict, geometry: dict, output: Path, *,
                    device: str, canonicalSize: int, panelSize: int) -> dict:
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=canonicalSize, frameGeometry=True)
    renderer.setView(azimuth=0, elevation=0)
    panels, labels = [], []
    for name in branchNames:
        branch = experiment["branches"][name]
        material = renderer.prepareMaterial(branch["albedo"], experiment["commonCoverage"])
        for lightName, light in canonicalLights.items():
            image = renderer.render(material, light, roughness=.45, keyIntensity=2.5, ambient=.18)
            writeImage(output / f"canonical{name[0].upper() + name[1:]}{lightName.title()}.png", image)
            panels.append(asImage(image))
            labels.append(f"{branchLabels[name]} / {lightName}")
    makeGrid(panels, labels, columns=3, panelSize=panelSize,
             subtitle="Same canonical front camera | identical front / left / right GGX light").save(output / "canonicalRelight.png")
    return {"projection": "orthographic", "imageSizeHw": [canonicalSize, canonicalSize],
            "azimuth": 0, "elevation": 0, "frameGeometry": True,
            "center": renderer.center.cpu().numpy(), "scale": float(renderer.scale),
            "lightsCanonical": canonicalLights, "commonCoverageOnly": True}


def runComparison(inputRoot: Path, output: Path, *, device: str = "cuda", panelSize: int = 360,
                  canonicalSize: int = 768, maxSamples: int = 20000, ridge: float = .02) -> dict:
    start = perf_counter()
    inputRoot, output = Path(inputRoot).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty comparison output directory; partial runs are preserved")
    if panelSize < 32 or canonicalSize < 32 or maxSamples < 27 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Invalid display dimensions or lighting fit parameters")
    experiment = loadExperiment(inputRoot)
    geometry = prepareGeometry(experiment["savedGeometry"], experiment["branches"][branchNames[0]]["albedo"].shape[0])
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import pytorch3d
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    allPanels = {mode: {"panels": [], "labels": []} for mode in ("matched", "fixed", "unlit", "overlay")}
    reports = []
    for index in range(len(experiment["photos"])):
        panels, report = renderPhotoView(experiment, geometry, index, output, device=device,
                                         panelSize=panelSize, maxSamples=maxSamples, ridge=ridge)
        reports.append(report)
        for mode in allPanels:
            for key in ("panels", "labels"):
                allPanels[mode][key].extend(panels[mode][key])
        print(f"Rendered original-size photo {index + 1}/{len(experiment['photos'])}; cameras and textures frozen", flush=True)
    for mode, filename, subtitle in (("matched", "overview.png", "Photo appearance fit: 27 global lighting parameters; not ground-truth albedo"),
                                     ("fixed", "fixedLightOverview.png", "Fixed GGX preview lights; same geometry and cameras"),
                                     ("unlit", "unlitOverview.png", "Unlit albedo projection; original source contains illumination"),
                                     ("overlay", "overlayOverview.png", "50% overlays of fitted-SH images and original photos")):
        makeGrid(**allPanels[mode], columns=4, panelSize=panelSize, subtitle=subtitle).save(output / filename)
    canonical = renderCanonical(experiment, geometry, output, device=device,
                                 canonicalSize=canonicalSize, panelSize=panelSize)
    atlasPanels = []
    for name in branchNames:
        preview = linearToSrgb(experiment["branches"][name]["linearAlbedo"])
        preview[~experiment["commonCoverage"]] = .16
        writeImage(output / f"atlas{name[0].upper() + name[1:]}.png", preview)
        atlasPanels.append(asImage(preview))
    makeStrip(atlasPanels, [branchLabels[name] for name in branchNames], output / "atlasOverview.png",
              "Shared observed UV support | linear atlas displayed in exact sRGB", panelSize)
    report = {"status": "rendered", "inputRoot": inputRoot, "output": output,
              "device": device, "pytorch3dVersion": pytorch3d.__version__, "seconds": perf_counter() - start,
              "referenceViewIndex": experiment["referenceViewIndex"], "sourceHashes": experiment["sourceHashes"],
              "implementationHashes": {str(path.relative_to(projectRoot)): fileHash(path) for path in (
                  Path(__file__), projectRoot / "src/faceAppearance/photoLighting.py",
                  projectRoot / "src/faceAppearance/pytorch3dRenderer.py")},
              "sharedGeometryAndCameraVerified": True, "cameraFittingRepeated": False,
              "geometryOptimized": False, "textureOptimized": False, "wildCapEnabled": False,
              "lightingFittedPerBranchPerView": True, "lightingParameterCount": 27,
              "lightFit": {"maxSamples": maxSamples, "ridge": ridge, "sameNeutralInitialization": True},
              "commonCoverageTexels": int(experiment["commonCoverage"].sum()),
              "views": reports, "canonicalRender": canonical,
              "fixedShading": {"specularMultiplier": .22, "roughness": .45, "keyIntensity": 2.5,
                               "ambient": .18, "normals": "shared smooth geometry normals"},
              "color": {"photos": "Exact sRGB decode before fitting", "atlas": "Saved linear RGB sampled directly",
                        "display": "Exact piecewise sRGB; metrics also retain unclipped linear error"},
              "limitations": ["Metrics are in-sample photo appearance, not ground-truth reflectance or held-out accuracy",
                              "All branches fit the same 27 global SH coefficients; no local or per-pixel light fitting",
                              "Lighting fit rejection does not shrink the independent common evaluation mask",
                              "RGB SH can absorb exposure and white-balance differences; decomposition remains ambiguous",
                              "Diffuse SH does not model specular reflection, cast shadows, skin scattering or occlusion",
                              "Raw-reference unlit projection deliberately retains source illumination and is not PBR albedo",
                              "OpenCV agreement checks coordinate conversion, not accuracy of the saved camera or geometry"]}
    if device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    writeJson(output / "comparisonReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/referenceViewExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/referenceViewComparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--panelSize", type=int, default=360)
    parser.add_argument("--canonicalSize", type=int, default=768)
    parser.add_argument("--maxSamples", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=.02)
    args = parser.parse_args()
    runComparison(args.input, args.output, device=args.device, panelSize=args.panelSize,
                  canonicalSize=args.canonicalSize, maxSamples=args.maxSamples, ridge=args.ridge)


if __name__ == "__main__":
    main()
