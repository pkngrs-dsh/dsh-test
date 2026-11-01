"""Render saved OpenDelight / SH-only / SH-prior experiments under shared controls.

No reconstruction, camera fitting, illumination fitting or WildCap inference is
performed here. All texture inputs are gamma-2.2 encoded; renders are sRGB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
for importPath in (projectRoot / "src", Path(__file__).resolve().parent):
    if str(importPath) not in sys.path:
        sys.path.insert(0, str(importPath))

from faceAppearance.artifacts import writeImage, writeJson
from renderCropComparison import (asImage, compareProjection, fileHash, makeGrid,
                                  originalMask, overlayPhoto, photoCamera,
                                  prepareGeometry, sharedPhotoCrop, validatePhotoPair)
from runShExperiment import validateCompletedWildCap

branchNames = ("openDelight", "shOnly", "shPrior")
branchLabels = {"openDelight": "OpenDelight", "shOnly": "SH only", "shPrior": "SH + albedo prior"}
canonicalLights = {"front": (0., .3, 1.), "left": (-.75, .45, .85), "right": (.75, .45, .85)}


def linearToSrgb(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values), 0)
    return np.clip(np.where(values <= 0.0031308, 12.92 * values,
                           1.055 * np.power(values, 1 / 2.4) - .055), 0, 1)


def validateTexture(albedo: np.ndarray, mask: np.ndarray, name: str) -> None:
    if (albedo.ndim != 3 or albedo.shape[-1] != 3 or albedo.shape[0] != albedo.shape[1]
            or mask.shape != albedo.shape[:2] or not np.isfinite(albedo).all()
            or not np.isfinite(mask).all() or np.any(albedo < 0) or np.any(albedo > 1)
            or np.any(mask < 0) or np.any(mask > 1)):
        raise ValueError(f"Invalid gamma-2.2 RGB atlas or coverage mask: {name}")


def loadBranch(directory: Path, withWildCap: bool = False) -> dict:
    directory = Path(directory).resolve()
    names = ("geometry.npz", "texture.npz", "inputs.json", "cameras.json", "viewTransforms.json", "result.json")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete branch {directory}: {', '.join(missing)}")
    readJson = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    result = readJson("result.json")
    if result.get("status") != "succeeded":
        raise ValueError(f"Branch must have status=succeeded: {directory}")
    with np.load(directory / "geometry.npz", allow_pickle=False) as saved:
        geometry = dict(saved)
    with np.load(directory / "texture.npz", allow_pickle=False) as saved:
        albedo, coverage = saved["albedo"].copy(), saved["coverageMask"].copy()
    validateTexture(albedo, coverage, str(directory))
    inputs, cameras, transforms = readJson("inputs.json"), readJson("cameras.json"), readJson("viewTransforms.json")
    if not inputs or len(cameras) != len(inputs) or len(transforms) != len(inputs):
        raise ValueError(f"Input/camera/transform view counts differ: {directory}")
    for key in ("vertices", "faces", "uvs", "faceMasks"):
        if key not in geometry or not np.isfinite(geometry[key]).all():
            raise ValueError(f"Missing or nonfinite geometry field {key}: {directory}")
    if len(geometry["faceMasks"]) != len(inputs):
        raise ValueError(f"Segmentation view count differs: {directory}")
    for index, (item, camera, mask) in enumerate(zip(inputs, cameras, geometry["faceMasks"])):
        if not camera.get("valid", False) or camera.get("viewIndex", index) != index:
            raise ValueError(f"View {index} lacks an accepted, ordered saved PnP camera; no refitting")
        for key, shape in (("intrinsics", (3, 3)), ("rotation", (3, 3)), ("translation", (3,))):
            values = np.asarray(camera[key])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved camera {key}: {directory}, view {index}")
        if list(mask.shape) != list(reversed(item["workingSize"])):
            raise ValueError(f"Working face-mask size disagrees with input metadata: {directory}")
    branch = {"directory": directory, "report": result, "inputs": inputs, "cameras": cameras,
              "transforms": transforms, "savedGeometry": geometry, "albedo": albedo, "coverage": coverage,
              "sourceHashes": {name: fileHash(directory / name) for name in names}}
    if withWildCap:
        experiment = result.get("experiment", {})
        if "wildCapPlanned" not in experiment or "atlasResolution" not in experiment:
            raise ValueError(f"WildCap requires saved experiment parameters: {directory}")
        validateCompletedWildCap(directory, result, experiment["wildCapPlanned"], experiment["atlasResolution"])
        materialPath = directory / "wildcapUff/materials.npz"
        if not materialPath.is_file():
            raise FileNotFoundError(f"--withWildCap requires completed materials for every branch: {materialPath}")
        with np.load(materialPath, allow_pickle=False) as saved:
            diffuse, valid = saved["diffuse"].copy(), saved["validMask"].copy()
            composite = saved["baseWithWildCap"].copy()
        validateTexture(diffuse, valid, str(materialPath))
        validateTexture(composite, coverage, str(materialPath) + " baseWithWildCap")
        if diffuse.shape != albedo.shape:
            raise ValueError("WildCap UFF diffuse and base atlas resolutions must match")
        branch.update(wildCap=composite, wildCapDiffuse=diffuse, wildCapValidMask=valid)
        branch["sourceHashes"]["wildcapUff/materials.npz"] = fileHash(materialPath)
    return branch


def requireSameArray(left, right, label: str) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise ValueError(f"Experiment must share identical {label}; compare only the delight frontends")


def validateSharedExperiment(branches: dict) -> tuple[list[Image.Image], np.ndarray]:
    """Reject accidental geometry/camera/input changes before any rendering."""
    reference = branches[branchNames[0]]
    photos = [validatePhotoPair(item, item) for item in reference["inputs"]]
    commonCoverage = reference["coverage"] >= .999
    for name in branchNames[1:]:
        branch = branches[name]
        if len(branch["inputs"]) != len(photos):
            raise ValueError("Branches must share the same source view count")
        for key in ("vertices", "faces", "uvs", "faceMasks", "cameraIntrinsics", "cameraExtrinsics"):
            left, right = reference["savedGeometry"], branch["savedGeometry"]
            if key in left or key in right:
                if key not in left or key not in right:
                    raise ValueError(f"Branches disagree on geometry field {key}")
                requireSameArray(left[key], right[key], key)
        requireSameArray(reference["savedGeometry"].get("uvFaces", reference["savedGeometry"]["faces"]),
                         branch["savedGeometry"].get("uvFaces", branch["savedGeometry"]["faces"]), "UV faces")
        if reference["transforms"] != branch["transforms"]:
            raise ValueError("Branches must share identical saved view transforms")
        if branch["albedo"].shape != reference["albedo"].shape:
            raise ValueError("Branches must share atlas resolution")
        for index, (left, right) in enumerate(zip(reference["inputs"], branch["inputs"])):
            validatePhotoPair(left, right)
            for key in ("inputToWorking", "workingSize"):
                requireSameArray(left[key], right[key], key)
            for key in ("intrinsics", "rotation", "translation"):
                requireSameArray(reference["cameras"][index][key], branch["cameras"][index][key], f"PnP {key}")
        commonCoverage &= branch["coverage"] >= .999
    if not commonCoverage.any():
        raise ValueError("The three branches have no common observed UV support")
    return photos, commonCoverage


def materialVariants(branches: dict, withWildCap: bool) -> list[dict]:
    variants = [{"name": name, "label": branchLabels[name], "albedo": branches[name]["albedo"], "branch": name}
                for name in branchNames]
    if withWildCap:
        variants.extend({"name": f"{name}WildCap", "label": f"{branchLabels[name]} + WC diffuse",
                         "albedo": branches[name]["wildCap"], "branch": name} for name in branchNames)
    return variants


def atlasPreview(albedo: np.ndarray, commonCoverage: np.ndarray) -> np.ndarray:
    preview = linearToSrgb(np.clip(albedo, 0, 1) ** 2.2)
    preview[~commonCoverage] = .16
    return preview


def renderVariant(renderer, variant: dict, coverage: np.ndarray, light: np.ndarray) -> dict:
    # Omitting specular/normal maps deliberately selects constant 0.22 specular
    # and the shared geometry normals, including in all WildCap comparisons.
    material = renderer.prepareMaterial(variant["albedo"], coverage)
    rendered = renderer.render(material, light, roughness=.45, keyIntensity=2.5, ambient=.18)
    unlit = linearToSrgb(material["albedo"].detach().cpu().numpy())
    visible = renderer.visible.detach().cpu().numpy()
    unlit[~visible] = [.105, .12, .14]
    return {"render": rendered, "unlit": unlit}


def viewPanels(photo: Image.Image, rendered: dict, variants: list[dict],
               crop: tuple, mode: str, index: int) -> tuple[list, list]:
    panels = [photo.crop(crop)] + [asImage(rendered[item["name"]][mode]).crop(crop) for item in variants]
    labels = [f"View {index + 1}: original"] + [item["label"] for item in variants]
    return panels, labels


def detailCrops(mask: np.ndarray) -> dict:
    """Approximate image regions, not detected/annotated facial landmarks."""
    ys, xs = np.nonzero(mask > .5)
    left, top, width, height = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    regions = {"leftMidface": (.03, .32, .47, .66), "rightMidface": (.53, .32, .97, .66),
               "lowerFace": (.20, .55, .80, .88)}
    return {name: (int(left + box[0] * width), int(top + box[1] * height),
                   int(left + box[2] * width), int(top + box[3] * height)) for name, box in regions.items()}


def renderPhotoView(reference: dict, geometry: dict, variants: list[dict], coverage: np.ndarray,
                    photoImage: Image.Image, index: int, output: Path, device: str,
                    panelSize: int, includeDetails: bool) -> tuple[dict, dict]:
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    viewDir = output / f"view{index:02d}"
    viewDir.mkdir()
    photoImage.save(viewDir / "original.png")
    photo = np.asarray(photoImage, np.float32) / 255
    mask = originalMask(reference["savedGeometry"]["faceMasks"][index],
                        reference["inputs"][index]["inputToWorking"], photoImage.size)
    crop = sharedPhotoCrop(mask, mask)
    writeImage(viewDir / "sharedFaceMaskOriginal.png", mask)
    camera = photoCamera(reference["cameras"][index], reference["inputs"][index], "fitted")
    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=photo.shape[:2], frameGeometry=False)
    renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
    projection = compareProjection(renderer, geometry, camera)
    visible = renderer.visible.detach().cpu().numpy()
    if not visible.any():
        raise ValueError(f"Saved PnP view {index} has no visible mesh")
    light = camera["rotation"].T @ np.array([-.5, -.5, -1.])
    rendered = {}
    for variant in variants:
        images = renderVariant(renderer, variant, coverage, light)
        if images["render"].shape != photo.shape:
            raise AssertionError("Original photo dimensions were not preserved")
        images["overlay"] = overlayPhoto(photo, images["render"], visible)
        rendered[variant["name"]] = images
        for mode, values in images.items():
            writeImage(viewDir / f"{variant['name']}{mode.title()}.png", values)
    writeImage(viewDir / "sharedRenderedMask.png", visible)
    panels = {}
    for mode, title in (("render", "Fixed preview light; not estimated photo illumination"),
                        ("overlay", "50% overlay; shared saved PnP; same display crop"),
                        ("unlit", "Unlit material projection; original photos still contain illumination")):
        images, labels = viewPanels(photoImage, rendered, variants, crop, mode, index)
        panels[mode] = {"panels": images, "labels": labels}
        makeGrid(images, labels, columns=len(images), panelSize=panelSize, subtitle=title).save(viewDir / f"{mode}Comparison.png")
    details = detailCrops(mask) if includeDetails else {}
    for region, detailCrop in details.items():
        for mode in ("unlit", "render"):
            images, labels = viewPanels(photoImage, rendered, variants, detailCrop, mode, index)
            makeGrid(images, labels, columns=len(images), panelSize=panelSize,
                     subtitle=f"{region} / {mode}: approximate region; shared pixels; no landmark fitting").save(
                         output / f"detail{region.title()}{mode.title()}.png")
    report = {"viewIndex": index, "sourceSha256": reference["inputs"][index]["sha256"],
              "orientedImageSizeWh": list(photoImage.size), "sharedDisplayCropXyxy": crop,
              "detailCropsXyxy": details, "inputMetadata": reference["inputs"][index],
              "savedViewTransform": reference["transforms"][index], "savedPnpCamera": reference["cameras"][index],
              "originalPhotoCamera": camera, "projectionCheck": projection,
              "rasterPrincipalPointShift": [.5, .5], "renderedFacePixels": int(visible.sum()),
              "lightDirectionCanonical": light, "imageSizeHw": list(photo.shape[:2]), "frameGeometry": False}
    writeJson(viewDir / "comparisonReport.json", report)
    del renderer
    return panels, report


def renderCanonical(geometry: dict, variants: list[dict], coverage: np.ndarray,
                    output: Path, device: str, imageSize: int, panelSize: int,
                    withWildCap: bool) -> dict:
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=imageSize, frameGeometry=True)
    renderer.setView(azimuth=0, elevation=0)
    rendered, panels, labels = {}, [], []
    for variant in variants:
        for lightName, light in canonicalLights.items():
            images = renderVariant(renderer, variant, coverage, np.asarray(light))
            panel = asImage(images["render"])
            panel.save(output / f"canonical{variant['name'].title()}{lightName.title()}.png")
            rendered[(variant["name"], lightName)] = panel
            panels.append(panel)
            labels.append(f"{variant['label']} / {lightName}")
    makeGrid(panels, labels, columns=3, panelSize=panelSize,
             subtitle="Shared front orthographic camera | front / left / right fixed key lights").save(output / "canonicalRelight.png")
    if withWildCap:
        pairs = [rendered[(key, "front")] for name in branchNames for key in (name, f"{name}WildCap")]
        pairLabels = [f"{branchLabels[name]} / {stage}" for name in branchNames for stage in ("before WC", "after WC diffuse")]
        makeGrid(pairs, pairLabels, columns=2, panelSize=panelSize,
                 subtitle="Same front light and geometry normals | only diffuse changes").save(output / "beforeAfterWildCap.png")
    report = {"imageSizeHw": [imageSize, imageSize], "projection": "orthographic",
              "azimuth": 0, "elevation": 0, "frameGeometry": True,
              "framing": "Shared translation and uniform scale; no branch-specific framing",
              "center": renderer.center.detach().cpu().numpy(), "scale": float(renderer.scale),
              "lightDirectionsCanonical": canonicalLights}
    del renderer
    return report


def runComparison(inputRoot: Path, output: Path, *, device: str = "cuda", withWildCap: bool = False,
                  imageSize: int = 768, panelSize: int = 384, detailView: int = 0) -> dict:
    start = perf_counter()
    inputRoot, output = Path(inputRoot).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty comparison output directory")
    if imageSize < 32 or panelSize < 32:
        raise ValueError("imageSize and panelSize must be at least 32")
    print("Validating shared source photos, geometry, PnP cameras and saved transforms", flush=True)
    branches = {name: loadBranch(inputRoot / name, withWildCap) for name in branchNames}
    photos, coverage = validateSharedExperiment(branches)
    if not 0 <= detailView < len(photos):
        raise ValueError("detailView must identify an existing saved source view")
    reference = branches[branchNames[0]]
    sharedWorkingInputs = reference["report"].get("experiment", {}).get("rawSampling") == "working"
    bandwidthNote = ("SH and OpenDelight use the same saved working input images; internal model resampling still differs"
                     if sharedWorkingInputs else
                     "This compares complete frontends: original-resolution SH sampling versus cached OpenDelight predictions; detail gains are not isolated SH gains")
    geometry = prepareGeometry(reference["savedGeometry"], reference["albedo"].shape[0])
    variants = materialVariants(branches, withWildCap)
    output.mkdir(parents=True, exist_ok=True)
    writeImage(output / "commonUvCoverage.png", coverage)
    atlasPanels, atlasLabels = [], []
    for variant in variants:
        preview = atlasPreview(variant["albedo"], coverage)
        writeImage(output / f"atlas{variant['name'].title()}.png", preview)
        atlasPanels.append(asImage(preview))
        atlasLabels.append(variant["label"])
    makeGrid(atlasPanels, atlasLabels, columns=3, panelSize=512,
             subtitle="Same UV support | gamma 2.2 decoded to linear, displayed in sRGB").save(output / "atlasOverview.png")
    import torch
    import pytorch3d
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    allPanels = {mode: {"panels": [], "labels": []} for mode in ("render", "overlay", "unlit")}
    viewReports = []
    for index, photo in enumerate(photos):
        panels, viewReport = renderPhotoView(reference, geometry, variants, coverage, photo, index,
                                             output, device, panelSize, index == detailView)
        for mode in allPanels:
            for key in ("panels", "labels"):
                allPanels[mode][key].extend(panels[mode][key])
        viewReports.append(viewReport)
        print(f"Rendered source view {index + 1}/{len(photos)} at original dimensions", flush=True)
    for mode, filename, subtitle in (
            ("render", "overview.png", "Frozen PnP | fixed light | " +
             ("shared working input bandwidth" if sharedWorkingInputs else "frontend source bandwidth differs")),
            ("overlay", "overlayOverview.png", "50% overlays | same full-resolution raster and display crop"),
            ("unlit", "unlitOverview.png", "Unlit material projection | source photos retain illumination")):
        makeGrid(**allPanels[mode], columns=len(variants) + 1, panelSize=panelSize, subtitle=subtitle).save(output / filename)
    canonicalReport = renderCanonical(geometry, variants, coverage, output, device, imageSize, panelSize, withWildCap)
    branchReports = {}
    for name, branch in branches.items():
        branchReports[name] = {"directory": branch["directory"], "sourceHashes": branch["sourceHashes"],
                               "sourceResult": branch["report"], "coverageTexels": int((branch["coverage"] >= .999).sum()),
                               "excludedByCommonSupport": int(((branch["coverage"] >= .999) & ~coverage).sum())}
        if withWildCap:
            mask = branch["wildCapValidMask"] >= .999
            writeImage(output / f"wildCapValid{name.title()}.png", mask)
            branchReports[name]["wildCapValidTexelsInCommonSupport"] = int((mask & coverage).sum())
    report = {"status": "rendered", "inputRoot": inputRoot, "output": output, "device": device,
              "pytorch3dVersion": pytorch3d.__version__, "seconds": perf_counter() - start,
              "withWildCap": withWildCap, "branches": branchReports, "views": viewReports,
              "sharedGeometryAndCameraVerified": True, "cameraFittingRepeated": False,
              "modelInferenceRepeated": False, "illuminationFittingRepeated": False,
              "commonCoverageTexels": int(coverage.sum()), "detailView": detailView,
              "canonicalRender": canonicalReport,
              "shading": {"specularMultiplier": .22, "roughness": .45, "f0": .08,
                          "keyIntensity": 2.5, "ambient": .18, "normalSource": "shared geometry normals",
                          "photoLight": "saved fitted camera R.T @ [-0.5, -0.5, -1]; identical for all variants",
                          "lightIsEstimatedPhotoIllumination": False},
              "color": {"inputAlbedoEncoding": "gamma2.2 for every branch and WildCap diffuse",
                        "linearDecode": "encoded ** 2.2 before texture sampling and shading",
                        "renderAndAtlasDisplayEncoding": "exact piecewise sRGB",
                        "sourcePhotoDisplay": "EXIF-oriented original RGB bytes; no delight or exposure normalization here",
                        "wildCapDisplay": "saved baseWithWildCap composite; common base coverage; no WC specular or normal maps"},
              "limitations": ["Fixed preview lights are not estimated source illumination; RGB differences are not reflectance error",
                              bandwidthNote,
                              "No calibrated albedo ground truth or held-out identity claim",
                              "Original photos participated in reconstruction; saved PnP is reused without refitting",
                              "OpenCV agreement validates projection conversion, not correspondence accuracy",
                              "Common UV support excludes branch-only observations; missing support is neutral gray",
                              "WildCap composites can retain base albedo outside their valid transferred patch",
                              "No cast shadows, SSS, eyes, teeth or reconstructed roughness; normals and specular held fixed",
                              "Detail regions are approximate fractions of the face-mask box, not facial landmark annotations"]}
    if device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    writeJson(output / "comparisonReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds", "withWildCap")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/shDelightExperiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/shDelightComparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--withWildCap", action="store_true")
    parser.add_argument("--imageSize", type=int, default=768, help="Canonical render size; source-photo renders always use original size")
    parser.add_argument("--panelSize", type=int, default=384)
    parser.add_argument("--detailView", type=int, default=0, help="Zero-based source view for approximate face detail crops")
    args = parser.parse_args()
    runComparison(args.input, args.output, device=args.device, withWildCap=args.withWildCap,
                  imageSize=args.imageSize, panelSize=args.panelSize, detailView=args.detailView)


if __name__ == "__main__":
    main()
