"""Render saved face materials with PyTorch3D, without re-running reconstruction."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
from faceAppearance.renderMaterials import prepareRenderMaterials


def asImage(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


def makeGrid(panels, labels, columns=3, panelSize=512, subtitle=""):
    rows = math.ceil(len(panels) / columns)
    rowHeight, header = panelSize + 36, 44 if subtitle else 0
    canvas = Image.new("RGB", (panelSize * columns, rowHeight * rows + header), (22, 27, 33))
    draw = ImageDraw.Draw(canvas)
    fontPath = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(fontPath), 16) if fontPath.exists() else ImageFont.load_default(size=16)
    if subtitle:
        draw.text((14, 12), subtitle, fill=(200, 214, 226), font=font)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = index % columns * panelSize, index // columns * rowHeight + header
        canvas.paste(panel.resize((panelSize, panelSize), Image.Resampling.LANCZOS), (x, y + 36))
        draw.text((x + 12, y + 10), label, fill=(235, 239, 244), font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/registeredEndToEndVerified")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/pytorch3dPreview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imageSize", type=int, default=768)
    parser.add_argument("--roughness", type=float, default=0.45)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty preview output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    print("Preparing materials and PyTorch3D tangent frames for +Y assumption", flush=True)
    positive = prepareRenderMaterials(projectRoot, args.input, sourceConvention="openGL")
    print("Preparing the alternative -Y normal assumption", flush=True)
    negative = prepareRenderMaterials(projectRoot, args.input, sourceConvention="directX")
    preparationSeconds = perf_counter() - start
    writeImage(args.output / "normalUffPositiveY.png", (positive["tangentNormal"] + 1) / 2)
    writeImage(args.output / "normalUffNegativeY.png", (negative["tangentNormal"] + 1) / 2)
    writeImage(args.output / "normalCoverage.png", positive["normalValidMask"])
    np.savez_compressed(args.output / "experimentalNormalMaps.npz",
                        positiveY=positive["tangentNormal"], negativeY=negative["tangentNormal"],
                        validMask=positive["normalValidMask"])
    import torch
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    renderer = Pytorch3dFaceRenderer(positive["geometry"], device=args.device, imageSize=args.imageSize)
    renderer.setView()
    prepare = renderer.prepareMaterial
    coverage = positive["baseCoverageMask"]

    def materialSet():
        return {
            "base": prepare(positive["baseAlbedo"], coverage),
            "smooth": prepare(positive["wildCapAlbedo"], coverage, specular=positive["specular"]),
            "positive": prepare(positive["wildCapAlbedo"], coverage, specular=positive["specular"],
                                tangentNormal=positive["tangentNormal"], normalValidMask=positive["normalValidMask"]),
            "negative": prepare(negative["wildCapAlbedo"], coverage, specular=negative["specular"],
                                tangentNormal=negative["tangentNormal"], normalValidMask=negative["normalValidMask"])}

    materials = materialSet()
    render = lambda material, light: asImage(renderer.render(material, light, roughness=args.roughness))
    lights = {"Left": (-0.75, 0.45, 0.85), "Front": (0, 0.3, 1), "Right": (0.75, 0.45, 0.85)}
    print("Rendering controlled material and relighting comparisons", flush=True)
    rendered = {}
    for variant, material in materials.items():
        for lightName, direction in lights.items():
            panel = render(material, direction)
            panel.save(args.output / f"{variant}{lightName}.png")
            rendered[(variant, lightName)] = panel
    makeGrid([rendered[(variant, "Left")] for variant in ("base", "smooth", "positive")],
             ["Base color / smooth normals", "WildCap color + spec / smooth", "WildCap + detail normal (+Y assumption)"],
             subtitle=f"PyTorch3D | same geometry, camera and light | roughness {args.roughness:g} (preview)").save(args.output / "overview.png")
    makeGrid([rendered[(variant, light)] for variant in ("smooth", "positive", "negative") for light in lights],
             [f"{variant} / key {light.lower()}" for variant in ("Smooth normals", "Detail +Y", "Detail -Y") for light in lights],
             subtitle="Controlled relighting | +Y and -Y are source-normal assumptions, not certification").save(args.output / "lightingComparison.png")
    # Clay comparisons remove albedo features that could conceal normal defects.
    clay = np.full_like(positive["wildCapAlbedo"], 0.6)
    clayPanels = []
    for source in (None, positive, negative):
        kwargs = {} if source is None else {"tangentNormal": source["tangentNormal"], "normalValidMask": source["normalValidMask"]}
        material = prepare(clay, coverage, **kwargs)
        clayPanels.append(render(material, lights["Left"]))
    makeGrid(clayPanels, ["Geometry normals", "Transferred normal / source +Y", "Transferred normal / source -Y"],
             subtitle="Clay diagnostic | same albedo, specular and lighting").save(args.output / "normalConvention.png")
    frames = []
    for angle in np.linspace(-70, 70, 15):
        radians = np.deg2rad(angle)
        direction = (float(np.sin(radians)), 0.35, float(np.cos(radians)))
        frames.append(makeGrid([render(materials[name], direction) for name in ("smooth", "positive")],
                               ["WildCap / geometry normals", "WildCap / detail +Y (assumed)"], columns=2, panelSize=384))
    frames += frames[-2:0:-1]
    frames[0].save(args.output / "relighting.gif", save_all=True, append_images=frames[1:],
                   duration=110, loop=0)
    viewPanels = []
    for angle in (-25, 0, 25):
        renderer.setView(azimuth=angle)
        material = materialSet()["positive"]
        panel = render(material, lights["Left"])
        panel.save(args.output / f"view{angle:+d}.png")
        viewPanels.append(panel)
    makeGrid(viewPanels, ["View -25 degrees", "Front", "View +25 degrees"],
             subtitle="PyTorch3D orthographic cameras | WildCap + detail normal (+Y assumed)").save(args.output / "views.png")
    import pytorch3d
    report = {"status": "rendered", "renderer": "PyTorch3D MeshRasterizer + explicit GGX preview shader",
              "pytorch3dVersion": pytorch3d.__version__, "device": args.device,
              "input": str(args.input.resolve()), "imageSize": args.imageSize,
              "seconds": perf_counter() - start, "materialPreparationSeconds": preparationSeconds,
              "sourceHashes": {name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
                               for name in ("geometry.npz", "texture.npz", "wildcapIct/materials.npz")},
              "normalTransfer": {"positiveY": positive["metadata"], "negativeY": negative["metadata"]},
              "camera": "PyTorch3D orthographic look_at; canonical object frame preserved; framing only uses translation/uniform scale",
              "shading": {"roughness": args.roughness, "f0": 0.08, "specular": "WildCap scalar multiplier; base uses constant 0.22",
                           "keyIntensity": 2.5, "ambient": 0.18, "displayEncoding": "sRGB",
                           "inputInterpretation": "gamma2.2 assumed for OpenDelight, documented for WildCap"},
              "limitations": ["Source tangent-basis/green convention is an experimental choice",
                              "No fitted roughness, cast shadows, SSS, eyes or teeth",
                              "No identity accuracy or calibrated reflectance claim",
                              "Geometrically visible missing observations use neutral gray; no generated skin fill"]}
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    writeJson(args.output / "renderReport.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds", "materialPreparationSeconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
