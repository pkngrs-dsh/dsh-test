"""Render native UFF/VGGT cameras against source photos, with separate PnP controls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.cameraDiagnostics import compareSavedCameras
from faceAppearance.photoCameras import canonicalCamera, originalImageIntrinsics
from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
from faceAppearance.renderMaterials import prepareRenderMaterials
from renderPytorch3d import asImage, makeGrid


def compareProjection(renderer, geometry, camera):
    import torch

    expected, depth = projectPoints(geometry["vertices"], camera["intrinsics"], camera["rotation"], camera["translation"])
    actual = renderer.projectionCamera.transform_points_screen(renderer.mesh.verts_packed())[:, :2].cpu().numpy()
    maximum = float(np.abs(expected[depth > 0] - actual[depth > 0]).max())
    if maximum > 0.005:
        raise AssertionError(f"PyTorch3D/OpenCV point projection mismatch: {maximum:.6f}px")
    return maximum


def photoCrop(mask):
    yy, xx = np.nonzero(mask > .5)
    if len(xx) == 0:
        raise ValueError("No saved face segmentation survives original-photo coordinate conversion")
    center = np.array([(xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2])
    side = int(np.ceil(max(np.ptp(xx), np.ptp(yy)) * 1.20))
    left, top = np.floor(center - side / 2).astype(int)
    return (int(left), int(top), int(left + side), int(top + side))


def overlayPhoto(photo, rendered, visible, weight=.5):
    result = photo.copy()
    result[visible] = photo[visible] * (1 - weight) + rendered[visible] * weight
    return result


def fullFramePanel(panels, labels):
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height + 44), (22, 27, 33))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (index * width, 44))
        draw.text((index * width + 16, 15), label, fill="white", font=ImageFont.load_default(size=20))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/registeredEndToEndVerified")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/photoCameraComparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    sourceReport = json.loads((args.input / "result.json").read_text())
    inputs = json.loads((args.input / "inputs.json").read_text())
    transforms = json.loads((args.input / "viewTransforms.json").read_text())
    fittedCameras = json.loads((args.input / "cameras.json").read_text())
    with np.load(args.input / "geometry.npz", allow_pickle=False) as data:
        savedGeometry = dict(data)
    materials = prepareRenderMaterials(projectRoot, args.input, sourceConvention="openGL")
    geometry = materials["geometry"]
    nativeCameras = [{**canonicalCamera(savedGeometry["cameraExtrinsics"][index], transform),
                      "intrinsics": savedGeometry["cameraIntrinsics"][index], "viewIndex": index}
                     for index, transform in enumerate(transforms)]
    diagnostics = compareSavedCameras(savedGeometry, nativeCameras, fittedCameras,
                                      seed=sourceReport["config"]["seed"])
    allPanels, allLabels, viewReports, blinkFrames = [], [], [], []
    print(f"Rendering {len(inputs)} source-photo cameras, preserving original image dimensions", flush=True)
    import torch
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for index, item in enumerate(inputs):
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source photo changed since reconstruction: {path.name}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if list(image.size) != item["orientedSize"]:
            raise ValueError("Saved oriented photo dimensions no longer match")
        width, height = image.size
        photo = np.asarray(image).astype(np.float32) / 255
        pixelAffine = np.asarray(item["inputToWorking"])
        originalMask = cv2.warpPerspective(savedGeometry["faceMasks"][index], np.linalg.inv(pixelAffine),
                                           (width, height), flags=cv2.INTER_LINEAR)
        crop = photoCrop(originalMask)
        viewDir = args.output / f"view{index:02d}"
        viewDir.mkdir()
        image.save(viewDir / "original.png")
        renderer = Pytorch3dFaceRenderer(geometry, device=args.device, imageSize=(height, width), frameGeometry=False)
        cameras = {}
        # A single fixed light for this view, derived from its native camera.
        # PnP changes only the control camera, never the light, shape or texture.
        light = np.asarray(nativeCameras[index]["rotation"]).T @ np.array([-.5, -.5, -1.])
        results = {}
        for name, workingCamera in (("native", nativeCameras[index]), ("fitted", fittedCameras[index])):
            if name == "fitted" and not workingCamera.get("valid", False):
                raise ValueError(f"View {index} has no accepted fitted comparison camera")
            camera = {"intrinsics": originalImageIntrinsics(workingCamera["intrinsics"], pixelAffine),
                      "rotation": np.asarray(workingCamera["rotation"]),
                      "translation": np.asarray(workingCamera["translation"]),
                      "source": nativeCameras[index]["source"] if name == "native" else "postFusionUvCorrespondencePnP"}
            renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
            maximumProjectionError = compareProjection(renderer, geometry, camera)
            material = renderer.prepareMaterial(materials["wildCapAlbedo"], materials["baseCoverageMask"],
                                                 specular=materials["specular"], tangentNormal=materials["tangentNormal"],
                                                 normalValidMask=materials["normalValidMask"])
            rendered = renderer.render(material, light)
            visible = renderer.visible.cpu().numpy()
            overlay = overlayPhoto(photo, rendered, visible)
            unlit = renderer.sampleAtlas(materials["wildCapAlbedo"]).cpu().numpy()
            unlit[~visible] = [.105, .12, .14]
            results[name] = {"render": rendered, "visible": visible, "overlay": overlay}
            writeImage(viewDir / f"{name}Render.png", rendered)
            writeImage(viewDir / f"{name}Overlay50.png", overlay)
            writeImage(viewDir / f"{name}Unlit.png", unlit)
            writeImage(viewDir / f"{name}Mask.png", visible)
            cameras[name] = {**camera, "imageSizeHw": [height, width],
                             "opencvVsPytorch3dMaximumErrorPixels": maximumProjectionError,
                             "rasterPrincipalPointShift": [.5, .5],
                             "renderedFacePixels": int(visible.sum())}
        fullFramePanel([image, asImage(results["native"]["overlay"]), asImage(results["fitted"]["overlay"])],
                        ["Original photo", "Native UFF/VGGT camera / 50% overlay", "UV-PnP camera control / 50% overlay"]).save(viewDir / "fullFrameComparison.png")
        cropped = [image.crop(crop), asImage(results["native"]["render"]).crop(crop),
                   asImage(results["native"]["overlay"]).crop(crop), asImage(results["fitted"]["overlay"]).crop(crop)]
        labels = [f"View {index+1}: original", "Native camera / fixed preview light", "Native camera / 50% overlay", "PnP control / 50% overlay"]
        makeGrid(cropped, labels, columns=4, panelSize=384,
                 subtitle="Identical crop after full-resolution rendering; mesh/camera not adjusted to improve the overlay").save(viewDir / "comparison.png")
        allPanels.extend(cropped); allLabels.extend(labels)
        blinkFrames.append(makeGrid([cropped[0], cropped[0]], [f"View {index+1}: source", "Source / native-render toggle"], columns=2, panelSize=384))
        nativeComposite = overlayPhoto(photo, results["native"]["render"], results["native"]["visible"], weight=1.)
        blinkFrames.append(makeGrid([cropped[0], asImage(nativeComposite).crop(crop)],
                                    [f"View {index+1}: source", "Source / native-render toggle"], columns=2, panelSize=384))
        viewReport = {"viewIndex": index, "photoName": item["name"], "photoSha256": item["sha256"],
                      "comparisonCropXyxy": crop, "inputToWorking": pixelAffine,
                      "cameras": cameras, "workingPixelDiagnostics": diagnostics[index],
                      "lightDirectionCanonical": light}
        writeJson(viewDir / "cameraComparison.json", viewReport)
        viewReports.append(viewReport)
        print(f"Finished view {index+1}/{len(inputs)}", flush=True)
    makeGrid(allPanels, allLabels, columns=4, panelSize=384,
             subtitle="Photo-camera comparison | native UFF/VGGT vs separately labeled UV-PnP control").save(args.output / "overview.png")
    blinkFrames[0].save(args.output / "nativeCameraBlink.gif", save_all=True, append_images=blinkFrames[1:], duration=850, loop=0)
    report = {"status": "rendered", "input": str(args.input.resolve()), "seconds": perf_counter() - start,
              "modelInferenceRepeated": False, "cameraFittingRepeated": False, "views": viewReports,
              "sourceHashes": {name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
                               for name in ("geometry.npz", "viewTransforms.json", "cameras.json", "inputs.json")},
              "normalSourceConvention": "experimentalOpenGLPositiveY; unverified source baker",
              "comparisonLimits": ["Lighting is a fixed preview, not estimated original illumination",
                                   "No RGB reconstruction score: photo includes lighting, specular, hair, eyes and background",
                                   "Source photos participated in reconstruction; this is not held-out likeness evaluation",
                                   "UV-based reprojection errors measure learned internal correspondences, not annotated ground truth",
                                   "Photo overlays retain original eyes/background outside the rendered mesh; no eye assembly"]}
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024**3
    writeJson(args.output / "comparisonReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
