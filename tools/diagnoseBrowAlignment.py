"""Trace cached brow sampling without fitting lighting, cameras or geometry.

The small image-space/UV crops are manually chosen inspection regions for the
current four-photo sample, not annotated eyebrow ground truth.
"""
from pathlib import Path
import argparse
import sys
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.appearanceObservations import (fuseLinearObservations, linearToGamma22,
                                                  readOriginalLinear, workingToOriginal)
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
from faceAppearance.texture import buildUvLookup, interpolateSurface, samplePixels
from renderCropComparison import asImage, photoCamera, prepareGeometry, validatePhotoPair
from renderShExperiment import linearToSrgb


def makeStrips(panels, labels, path, columns=3):
    width, height, header = 410, 150, 34
    rows = (len(panels) + columns - 1) // columns
    image = Image.new("RGB", (columns * width, rows * (height + header)), (25, 29, 35))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = index % columns * width, index // columns * (height + header)
        draw.text((x + 8, y + 8), label, font=font, fill=(230, 235, 240))
        panel.thumbnail((width, height), Image.Resampling.LANCZOS)
        image.paste(panel, (x + (width-panel.width)//2, y + header + (height-panel.height)//2))
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/browAlignmentDiagnostic")
    args = parser.parse_args()
    experiment = projectRoot / "runs/shDelightExperiment"
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    reference = experiment / "openDelight"
    inputs = json.loads((reference / "inputs.json").read_text())
    cameras = json.loads((reference / "cameras.json").read_text())
    photo = validatePhotoPair(inputs[0], inputs[0])
    with np.load(reference / "geometry.npz") as data:
        savedGeometry = dict(data)
    with np.load(experiment / "observations.npz") as data:
        raw = data["rawLinear"].copy()
    with np.load(experiment / "commonSupport.npz") as data:
        weights, coverage = data["weights"].copy(), data["coverageMask"].copy()
    fused = fuseLinearObservations(raw, weights)
    geometry = prepareGeometry(savedGeometry, raw.shape[1])
    renderer = Pytorch3dFaceRenderer(geometry, device="cuda", imageSize=(photo.height, photo.width), frameGeometry=False)
    camera = photoCamera(cameras[0], inputs[0], "fitted")
    renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
    # Geometry-only self-projection control. It should approximately reproduce
    # the same source image in the same camera, but proves no multi-view quality.
    lookup = buildUvLookup(savedGeometry["uvs"], savedGeometry["faces"], raw.shape[1])
    surface = interpolateSurface(savedGeometry["vertices"], savedGeometry["faces"], lookup)
    support = weights[0] > 0
    rotation = np.asarray(cameras[0]["rotation"])
    translation = np.asarray(cameras[0]["translation"])
    cameraPoints = surface[support] @ rotation.T + translation
    projectedPoints = cameraPoints @ np.asarray(cameras[0]["intrinsics"]).T
    workingPoints = projectedPoints[:, :2] / projectedPoints[:, 2:3]
    originalPoints = workingToOriginal(workingPoints, inputs[0])
    original = readOriginalLinear(inputs[0])
    inside = ((originalPoints[:, 0] >= 0) & (originalPoints[:, 0] <= photo.width - 1)
              & (originalPoints[:, 1] >= 0) & (originalPoints[:, 1] <= photo.height - 1))
    yy, xx = np.nonzero(support)
    projectionOnly = np.zeros_like(raw[0])
    projectionOnly[yy[inside], xx[inside]] = samplePixels(original, originalPoints[inside])
    projectionSupport = np.zeros_like(support)
    projectionSupport[yy[inside], xx[inside]] = True
    variants = [(f"rawView{index + 1}", raw[index], weights[index] > 0) for index in range(len(raw))]
    variants.append(("rawFused", fused["linearAlbedo"], coverage))
    variants.append(("rawProjectionOnlyView1", projectionOnly, projectionSupport))
    projected = {}
    for name, colors, support in variants:
        material = renderer.prepareMaterial(linearToGamma22(colors), support)
        image = linearToSrgb(material["albedo"].detach().cpu().numpy())
        image[~renderer.visible.detach().cpu().numpy()] = .1
        projected[name] = asImage(image)
        projected[name].save(output / f"{name}InFrontCamera.png")
        uvImage = linearToSrgb(colors)
        uvImage[~support] = .1
        writeImage(output / f"{name}Uv.png", uvImage)
    # All panels use identical original-image pixels, without alignment fixes.
    crop = (65, 370, 405, 485)
    panels = [photo.crop(crop), projected["rawProjectionOnlyView1"].crop(crop),
              projected["rawView1"].crop(crop), projected["rawFused"].crop(crop)]
    labels = ["Original photo 1", "Photo 1: geometric self-projection", "Photo 1: current dense-UV mapping",
              "4 raw photos fused (no delight)"]
    for name, label in [("shOnly", "SH-only fused (unlit)"),
                        ("shOnlyWildCap", "SH-only + WildCap (unlit)")]:
        image = Image.open(projectRoot / "runs/shDelightComparisonWildCap/view00" / f"{name}Unlit.png")
        panels.append(image.crop(crop)); labels.append(label)
    makeStrips(panels, labels, output / "browStages.png")
    perViewPanels = [projected[f"rawView{index + 1}"].crop(crop) for index in range(len(raw))]
    makeStrips(perViewPanels, [f"Raw photo {index + 1} mapped to front view" for index in range(len(raw))],
               output / "browPerView.png", columns=2)
    uvCrop = (210, 235, 820, 390)
    uvPanels = [Image.open(output / f"{name}Uv.png").crop(uvCrop) for name, _, _ in variants]
    makeStrips(uvPanels, [name for name, _, _ in variants], output / "browUvPerView.png", columns=2)
    # Compare dominant source changes; this is not a semantic brow mask.
    writeJson(output / "diagnostic.json", {
        "status": "completed", "originalImageCropXyxy": crop, "uvCropXyxy": uvCrop,
        "geometryAndCameraRefitted": False, "inputCorrespondences": "cached constrained dense UV",
        "rawReferenceIsAlbedo": False, "render": "unlit; gamma2.2 decoded then displayed sRGB",
        "scope": "Manually selected inspection crops; no brow landmark labels or claimed accuracy score",
        "projectionControl": "Same photo and camera, same geometry, no dense UV correction; diagnostic self-projection only, not a multi-view reconstruction result",
        "sourceCameraErrors512Pixels": [{key: item[key] for key in (
            "viewIndex", "heldOutMedianErrorPixels", "heldOutP90ErrorPixels")} for item in cameras],
        "rawFusion": fused["diagnostics"],
    })
    print(output)


if __name__ == "__main__":
    main()
