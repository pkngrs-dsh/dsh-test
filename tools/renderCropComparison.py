"""Compare two saved input-crop experiments in the original photo coordinates.

Only the saved base OpenDelight atlas and geometry normals are rendered. This
tool never loads WildCap materials, refits cameras, or reruns reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot / "src") not in sys.path:
    sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.cameraDiagnostics import compareSavedCameras
from faceAppearance.normalTransfer import convertUvs
from faceAppearance.photoCameras import canonicalCamera, originalImageIntrinsics
from faceAppearance.renderMaterials import buildRenderBasis


def fileHash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asImage(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


def makeGrid(panels: list[Image.Image], labels: list[str], *, columns: int = 5,
             panelSize: int = 384, subtitle: str = "") -> Image.Image:
    if not panels or len(panels) != len(labels):
        raise ValueError("Expected nonempty panels with one label per panel")
    header, labelHeight = (48 if subtitle else 0), 36
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * panelSize, rows * (panelSize + labelHeight) + header),
                       (22, 27, 33))
    draw = ImageDraw.Draw(canvas)
    fontPath = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(fontPath), 16) if fontPath.exists() else ImageFont.load_default(size=16)
    if subtitle:
        draw.text((14, 14), subtitle, font=font, fill=(200, 214, 226))
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index % columns * panelSize
        y = index // columns * (panelSize + labelHeight) + header
        draw.text((x + 10, y + 10), label, font=font, fill=(235, 239, 244))
        # Photo overview crops are square. Contain rather than stretch remains
        # correct if a future saved working input has a non-square image size.
        resized = ImageOps.contain(panel.convert("RGB"), (panelSize, panelSize), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + (panelSize - resized.width) // 2,
                              y + labelHeight + (panelSize - resized.height) // 2))
    return canvas


def validatePhotoPair(before: dict, after: dict) -> Image.Image:
    """Require identical raw content at this view index and verify both disk paths."""
    if before["sha256"] != after["sha256"]:
        raise ValueError("Before/after source photo SHA-256 differs; compare the same view ordering")
    if before["orientedSize"] != after["orientedSize"]:
        raise ValueError("Before/after EXIF-oriented source dimensions differ")
    for item in (before, after):
        path = Path(item["path"])
        if fileHash(path) != item["sha256"]:
            raise ValueError(f"Source photo changed since reconstruction: {path}")
    with Image.open(before["path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if list(image.size) != before["orientedSize"]:
        raise ValueError("Saved EXIF-oriented photo dimensions do not match the current file")
    return image


def originalMask(mask: np.ndarray, inputToWorking: np.ndarray,
                 imageSize: tuple[int, int]) -> np.ndarray:
    """Inverse project the saved working segmentation onto the oriented source."""
    affine = np.asarray(inputToWorking, np.float64)
    mask = np.asarray(mask, np.float32)
    if (mask.ndim != 2 or not np.isfinite(mask).all() or affine.shape != (3, 3)
            or not np.isfinite(affine).all() or abs(np.linalg.det(affine)) < 1e-12):
        raise ValueError("Expected a finite saved mask and invertible inputToWorking")
    return cv2.warpPerspective(mask, np.linalg.inv(affine), imageSize,
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def sharedPhotoCrop(beforeMask: np.ndarray, afterMask: np.ndarray,
                    margin: float = 0.20) -> tuple[int, int, int, int]:
    """Square display crop around both masks; render cameras never use this crop.

    The square is not clamped independently at image edges: PIL pads any area
    outside the original frame, preserving equal scale across all five panels.
    """
    if beforeMask.shape != afterMask.shape or beforeMask.ndim != 2 or margin < 0:
        raise ValueError("Masks must share the original photo shape and margin must be nonnegative")
    ys, xs = np.nonzero((beforeMask > 0.5) | (afterMask > 0.5))
    if not len(xs):
        raise ValueError("Neither saved face mask survives conversion to original photo coordinates")
    center = np.array([(xs.min() + xs.max()) * 0.5, (ys.min() + ys.max()) * 0.5])
    side = max(1, int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * (1 + margin))))
    left, top = np.floor(center - (side - 1) * 0.5).astype(int)
    return int(left), int(top), int(left + side), int(top + side)


def prepareGeometry(savedGeometry: dict, resolution: int) -> dict:
    geometry = {key: savedGeometry[key] for key in ("vertices", "faces", "uvs")}
    geometry["uvFaces"] = savedGeometry.get("uvFaces", geometry["faces"])
    basis = buildRenderBasis(geometry, resolution, uvOrigin="vDown")
    return {**geometry,
            "objUvs": convertUvs(geometry["uvs"], sourceOrigin="vDown", targetOrigin="vUp"),
            "vertexNormals": basis["vertexNormals"],
            "cornerTangentFrames": basis["corners"]["cornerFrames"],
            "faceTangentHandedness": basis["corners"]["faceHandedness"],
            "faceTangentValidMask": basis["corners"]["faceValidMask"],
            "geometryFrame": "uvFaceFusionCanonical", "uvOrigin": "vDown"}


def loadRun(directory: Path) -> dict:
    directory = Path(directory).resolve()
    requiredFiles = ("result.json", "inputs.json", "viewTransforms.json", "cameras.json",
                     "geometry.npz", "texture.npz")
    missing = [name for name in requiredFiles if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete saved run {directory}: {', '.join(missing)}")
    readJson = lambda name: json.loads((directory / name).read_text())
    report = readJson("result.json")
    if report.get("status") != "succeeded":
        raise ValueError(f"Run must have status=succeeded: {directory}")
    inputs, transforms, fitted = readJson("inputs.json"), readJson("viewTransforms.json"), readJson("cameras.json")
    with np.load(directory / "geometry.npz", allow_pickle=False) as data:
        savedGeometry = dict(data)
    with np.load(directory / "texture.npz", allow_pickle=False) as data:
        albedo, coverage = data["albedo"].copy(), data["coverageMask"].copy()
    if (albedo.ndim != 3 or albedo.shape[2] != 3 or albedo.shape[0] != albedo.shape[1]
            or coverage.shape != albedo.shape[:2] or not np.isfinite(albedo).all()
            or not np.isfinite(coverage).all() or albedo.min() < 0 or albedo.max() > 1):
        raise ValueError(f"Invalid saved base albedo or coverage: {directory}")
    count = len(inputs)
    if not count or any(len(value) != count for value in (
            transforms, fitted, savedGeometry["cameraIntrinsics"], savedGeometry["cameraExtrinsics"],
            savedGeometry["faceMasks"])):
        raise ValueError(f"Saved camera/input/segmentation view counts disagree: {directory}")
    for index, camera in enumerate(fitted):
        if not camera.get("valid", False):
            raise ValueError(f"View {index} has no accepted saved PnP camera in {directory}; no camera will be refitted")
        if "viewIndex" in camera and camera["viewIndex"] != index:
            raise ValueError(f"Saved PnP view ordering is inconsistent in {directory}")
    native = [{**canonicalCamera(savedGeometry["cameraExtrinsics"][index], transform),
               "intrinsics": savedGeometry["cameraIntrinsics"][index], "viewIndex": index}
              for index, transform in enumerate(transforms)]
    config = report.get("config", {})
    cameraDiagnostics = compareSavedCameras(savedGeometry, native, fitted, seed=config.get("seed", 2026))
    inputImages, inputHashes = [], {}
    for index, item in enumerate(inputs):
        path = directory / f"input{index:02d}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")
        if list(image.size) != item["workingSize"]:
            raise ValueError(f"Saved input image dimensions do not match its metadata: {path}")
        if tuple(savedGeometry["faceMasks"][index].shape) != tuple(reversed(image.size)):
            raise ValueError(f"Saved face segmentation dimensions do not match working image: {path}")
        inputImages.append(image)
        inputHashes[path.name] = fileHash(path)
    return {"directory": directory, "report": report, "inputs": inputs, "transforms": transforms,
            "savedGeometry": savedGeometry, "geometry": prepareGeometry(savedGeometry, albedo.shape[0]),
            "albedo": albedo, "coverage": coverage, "nativeCameras": native, "fittedCameras": fitted,
            "cameraDiagnostics": cameraDiagnostics, "inputImages": inputImages,
            "sourceHashes": {**{name: fileHash(directory / name) for name in requiredFiles}, **inputHashes}}


def photoCamera(workingCamera: dict, metadata: dict, cameraKind: str) -> dict:
    return {"intrinsics": originalImageIntrinsics(workingCamera["intrinsics"], metadata["inputToWorking"]),
            "rotation": np.asarray(workingCamera["rotation"], np.float64),
            "translation": np.asarray(workingCamera["translation"], np.float64).reshape(3),
            "source": ("uvFaceFusionVggtNative; exact canonical and saved pixel conversions"
                       if cameraKind == "native" else "savedUvCorrespondencePnP; no refitting")}


def compareProjection(renderer, geometry: dict, camera: dict) -> dict:
    """Certify coordinate conversion against OpenCV integer-centered pixels."""
    expected, depth = projectPoints(geometry["vertices"], camera["intrinsics"],
                                    camera["rotation"], camera["translation"])
    actual = renderer.projectionCamera.transform_points_screen(renderer.mesh.verts_packed())[
        :, :2].detach().cpu().numpy()
    valid = (depth > 1e-9) & np.isfinite(expected).all(axis=1) & np.isfinite(actual).all(axis=1)
    if not valid.any():
        raise ValueError("No positive-depth vertices available for projection validation")
    maximum = float(np.abs(expected[valid] - actual[valid]).max())
    if maximum > 0.005:
        raise AssertionError(f"PyTorch3D/OpenCV projection mismatch: {maximum:.6f} pixels")
    return {"maximumAbsoluteErrorPixels": maximum, "testedPositiveDepthVertexCount": int(valid.sum()),
            "totalVertexCount": len(depth), "tolerancePixels": 0.005}


def overlayPhoto(photo: np.ndarray, rendered: np.ndarray, visible: np.ndarray) -> np.ndarray:
    overlay = photo.copy()
    overlay[visible] = 0.5 * photo[visible] + 0.5 * rendered[visible]
    return overlay


def renderRunView(run: dict, viewIndex: int, photo: np.ndarray, outputDir: Path,
                  group: str, device: str) -> tuple[dict, dict]:
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    height, width = photo.shape[:2]
    renderer = Pytorch3dFaceRenderer(run["geometry"], device=device,
                                   imageSize=(height, width), frameGeometry=False)
    light = np.asarray(run["nativeCameras"][viewIndex]["rotation"]).T @ np.array([-.5, -.5, -1.])
    result, cameras = {}, {}
    for cameraKind, workingCamera in (("native", run["nativeCameras"][viewIndex]),
                                       ("fitted", run["fittedCameras"][viewIndex])):
        camera = photoCamera(workingCamera, run["inputs"][viewIndex], cameraKind)
        renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
        projection = compareProjection(renderer, run["geometry"], camera)
        # Deliberately omit specular/tangentNormal: both groups use the same
        # constant preview specular and their own smooth geometry normals.
        material = renderer.prepareMaterial(run["albedo"], run["coverage"])
        rendered = renderer.render(material, light)
        visible = renderer.visible.detach().cpu().numpy().copy()
        if rendered.shape != photo.shape or visible.shape != photo.shape[:2] or not visible.any():
            raise ValueError(f"Invalid or empty original-size raster: {group} view {viewIndex} {cameraKind}")
        overlay = overlayPhoto(photo, rendered, visible)
        prefix = f"{group}{cameraKind.title()}"
        writeImage(outputDir / f"{prefix}Render.png", rendered)
        writeImage(outputDir / f"{prefix}Overlay50.png", overlay)
        writeImage(outputDir / f"{prefix}Mask.png", visible)
        result[cameraKind] = {"render": rendered, "overlay": overlay}
        cameras[cameraKind] = {**camera, "imageSizeHw": [height, width], "projectionCheck": projection,
                               "rasterPrincipalPointShift": [.5, .5], "renderedFacePixels": int(visible.sum()),
                               "lightDirectionCanonical": light}
        if cameraKind == "native":
            # Full geometric support for clay makes input UV coverage incapable
            # of concealing a shape difference. This is not texture completion.
            clay = renderer.prepareMaterial(np.full_like(run["albedo"], 0.6),
                                             np.ones_like(run["coverage"], dtype=np.float32))
            clayRendered = renderer.render(clay, light)
            clayOverlay = overlayPhoto(photo, clayRendered, visible)
            writeImage(outputDir / f"{group}ClayRender.png", clayRendered)
            writeImage(outputDir / f"{group}ClayOverlay50.png", clayOverlay)
            result["clay"] = {"render": clayRendered, "overlay": clayOverlay}
        del material
    # The caller renders the other group sequentially to avoid two full-frame
    # meshes/fragments/materials occupying GPU memory at once.
    del renderer
    return result, {"cameras": cameras, "lightDirectionCanonical": light,
                    "inputMetadata": run["inputs"][viewIndex],
                    "savedNativeCanonicalization": run["transforms"][viewIndex],
                    "savedPnpDiagnostics": run["fittedCameras"][viewIndex],
                    "workingPixelCameraDiagnostics": run["cameraDiagnostics"][viewIndex]}


def comparisonPanels(photo: Image.Image, before: dict, after: dict,
                     crop: tuple[int, int, int, int], viewIndex: int, mode: str) -> tuple[list, list]:
    panels = [photo, asImage(before[mode]["render"]), asImage(after[mode]["render"]),
              asImage(before[mode]["overlay"]), asImage(after[mode]["overlay"])]
    labels = [f"View {viewIndex + 1}: original", "Before / fixed preview light", "After / fixed preview light",
              "Before / 50% overlay", "After / 50% overlay"]
    return [panel.crop(crop) for panel in panels], labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=projectRoot / "runs/quality1024_4views")
    parser.add_argument("--after", type=Path, default=projectRoot / "runs/faceCrop1024_4viewsObserved")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/cropComparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty comparison output directory")
    if args.before.resolve() == args.after.resolve():
        raise ValueError("Choose two distinct saved runs")
    start = perf_counter()
    print("Reading saved crop experiments and preparing geometry-only tangent frames", flush=True)
    before, after = loadRun(args.before), loadRun(args.after)
    if len(before["inputs"]) != len(after["inputs"]):
        raise ValueError("Before/after runs have different view counts")
    photos = [validatePhotoPair(left, right) for left, right in zip(before["inputs"], after["inputs"])]
    preparationSeconds = perf_counter() - start
    args.output.mkdir(parents=True, exist_ok=True)
    inputPanels, inputLabels = [], []
    for index in range(len(photos)):
        inputPanels.extend([before["inputImages"][index], after["inputImages"][index]])
        inputLabels.extend([f"View {index + 1}: before saved input", f"View {index + 1}: after saved input"])
    makeGrid(inputPanels, inputLabels, columns=2, panelSize=512,
             subtitle="Saved model inputs | original source hashes match | no new image preprocessing").save(
                 args.output / "inputsBeforeAfter.png")
    import torch
    import pytorch3d
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    panelSets = {mode: {"panels": [], "labels": []} for mode in ("native", "fitted", "clay")}
    views = []
    for index, photoImage in enumerate(photos):
        photo = np.asarray(photoImage, np.float32) / 255
        viewDir = args.output / f"view{index:02d}"
        viewDir.mkdir()
        photoImage.save(viewDir / "original.png")
        masks = [originalMask(run["savedGeometry"]["faceMasks"][index],
                              run["inputs"][index]["inputToWorking"], photoImage.size)
                 for run in (before, after)]
        crop = sharedPhotoCrop(*masks)
        writeImage(viewDir / "beforeFaceMaskOriginal.png", masks[0])
        writeImage(viewDir / "afterFaceMaskOriginal.png", masks[1])
        beforeImages, beforeReport = renderRunView(before, index, photo, viewDir, "before", args.device)
        afterImages, afterReport = renderRunView(after, index, photo, viewDir, "after", args.device)
        for mode in panelSets:
            panels, labels = comparisonPanels(photoImage, beforeImages, afterImages, crop, index, mode)
            panelSets[mode]["panels"].extend(panels)
            panelSets[mode]["labels"].extend(labels)
            makeGrid(panels, labels, subtitle=f"{mode.title()} cameras | shared crop after original-size rendering").save(
                viewDir / f"{mode}Comparison.png")
        viewReport = {"viewIndex": index, "photoSha256": before["inputs"][index]["sha256"],
                      "orientedImageSizeWh": list(photoImage.size), "comparisonCropXyxy": crop,
                      "comparisonCropPolicy": "unionOfBothInverseWarpedFaceMasks; square; 20% size margin",
                      "before": beforeReport, "after": afterReport}
        writeJson(viewDir / "comparison.json", viewReport)
        views.append(viewReport)
        print(f"Rendered view {index + 1}/{len(photos)} at {photoImage.width}x{photoImage.height}", flush=True)
    names = {"native": "overview.png", "fitted": "fittedOverview.png", "clay": "clayOverview.png"}
    subtitles = {"native": "Native UFF/VGGT cameras | base OpenDelight color + geometry normals | fixed camera-relative preview light",
                 "fitted": "Saved UV-PnP cameras | no refitting | same per-group lights as native comparison",
                 "clay": "Native cameras + constant clay | geometry normals only | full geometric support; no appearance maps"}
    for mode, panels in panelSets.items():
        makeGrid(panels["panels"], panels["labels"], subtitle=subtitles[mode]).save(args.output / names[mode])

    focusIndices = sorted({0, len(photos) - 1})
    focusPanels, focusLabels = [], []
    for index in focusIndices:
        focusPanels.extend(panelSets["native"]["panels"][index * 5:index * 5 + 3])
        focusLabels.extend(panelSets["native"]["labels"][index * 5:index * 5 + 3])
    makeGrid(focusPanels, focusLabels, columns=3,
             subtitle="Original | before crop fix | after crop fix - native cameras, fixed preview lighting").save(
                 args.output / "focus.png")

    def runReport(run):
        source = run["report"]
        return {"directory": str(run["directory"]), "sourceHashes": run["sourceHashes"],
                "config": source.get("config"), "stopAfter": source.get("stopAfter"),
                "stageSeconds": source.get("stageSeconds"), "inferenceSeconds": source.get("inferenceSeconds"),
                "textureDiagnostics": source.get("textureDiagnostics"),
                "delightDiagnostics": source.get("delightDiagnostics"),
                "savedCameraDiagnostics": source.get("cameraDiagnostics"),
                "cameraComparisonDiagnostics": run["cameraDiagnostics"],
                "geometrySpace": source.get("geometrySpace"),
                "inputCropMetadata": run["inputs"], "albedoResolution": run["albedo"].shape[0]}

    report = {"status": "rendered", "before": runReport(before), "after": runReport(after),
              "seconds": perf_counter() - start, "preparationSeconds": preparationSeconds,
              "renderer": "PyTorch3D MeshRasterizer + fixed GGX preview shader",
              "pytorch3dVersion": pytorch3d.__version__, "device": args.device,
              "viewCount": len(views), "views": views, "frameGeometry": False,
              "renderResolution": "each EXIF-oriented original photo height/width; never display crop size",
              "modelInferenceRepeated": False, "cameraFittingRepeated": False,
              "wildCapMaterialsUsed": False, "normalMapsUsed": False,
              "shading": {"material": "each run's saved fused OpenDelight base albedo",
                          "normals": "PyTorch3D smooth geometry normals", "specularMultiplier": 0.22,
                          "roughness": 0.45, "keyIntensity": 2.5, "ambient": 0.18,
                          "displayEncoding": "sRGB", "baseAlbedoInterpretation": "gamma2.2 assumed",
                          "lightRule": "each group's native camera R.T @ [-0.5,-0.5,-1]; shader normalizes",
                          "nativeAndFittedShareLightWithinGroup": True,
                          "estimatedOriginalPhotoIllumination": False},
              "comparisonLimits": [
                  "Fixed camera-relative preview lighting is not estimated original-photo illumination",
                  "Differences include geometry, native camera, delight prediction and fusion caused by input crop",
                  "Other configuration/stochastic changes must be checked in the two saved run configs",
                  "Raw photo RGB includes lighting, highlights, eyes, hair and background; no raw RGB quality score is asserted",
                  "Saved UV-PnP errors use learned correspondences, not annotated camera/shape ground truth",
                  "Source photos participated in reconstruction; this is not held-out identity validation",
                  "Overlay retains source eyes/background outside the visible mesh; no eyeball or teeth assembly",
                  "Missing observed base-color regions render neutral gray; clay uses full geometric support",
                  "No fitted roughness, SSS, cast shadows or calibrated physical reflectance"]}
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    writeJson(args.output / "comparisonReport.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "output": str(args.output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
