"""Render three texture branches on one saved UFF mesh and fitted cameras.

Photo-frame images are unlit native encoded RGB. Their shared-support error is
an in-sample image-alignment diagnostic, not a physical-albedo quality score.
The optional canonical preview uses the existing fixed GGX diagnostic shader.
No reconstruction, camera fitting, image enhancement or texture filling runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot / "src") not in sys.path:
    sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeImage, writeJson
from renderCropComparison import asImage, compareProjection, makeGrid, prepareGeometry


BRANCHES = ("denseBaseline", "projectionMean", "colmap")
LABELS = {"denseBaseline": "Dense UV baseline", "projectionMean": "Projection + robust mean",
          "colmap": "COLMAP mesh texturer"}


def fileHash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readArrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def readRgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), np.float32) / 255.


def loadBranch(path: Path) -> dict:
    saved = readArrays(path)
    color, mask = np.asarray(saved["albedo"], np.float32), np.asarray(saved["coverageMask"])
    if (color.ndim != 3 or color.shape[-1] != 3 or min(color.shape[:2]) < 2
            or mask.shape != color.shape[:2] or not np.isfinite(color).all()
            or not np.isfinite(mask).all() or np.any((color < 0) | (color > 1))
            or np.any((mask < 0) | (mask > 1))):
        raise ValueError(f"Invalid branch albedo or coverage: {path}")
    return {"albedo": color, "coverage": mask.astype(np.float32)}


def focusCrops(mask: np.ndarray) -> dict:
    """Choose approximate inspection bands; no semantic landmark claim is made."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot choose close-up crops from an empty visible face")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    return {"eyeBrowBand": (x0, y0 + round(.20 * height), x1, y0 + round(.49 * height)),
            "mouthBand": (x0 + round(.18 * width), y0 + round(.56 * height),
                          x1 - round(.18 * width), y0 + round(.85 * height))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Saved successful job output")
    parser.add_argument("--experiment", type=Path, required=True, help="Contains branches/*.npz")
    parser.add_argument("--output", type=Path, required=True, help="New or empty comparison directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=512, help="Display panel and canonical render size")
    parser.add_argument("--skip-canonical", action="store_true", help="Only render saved photo cameras")
    args = parser.parse_args()
    source, experiment, output = (getattr(args, name).resolve() for name in ("source", "experiment", "output"))
    if not 128 <= args.size <= 1024:
        raise ValueError("--size must be between 128 and 1024")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty comparison output directory")
    if output == source or output == experiment:
        raise ValueError("The comparison directory must differ from its sources")
    start = perf_counter()
    cameras = json.loads((source / "cameras.json").read_text())
    for camera in cameras:
        if camera.get("valid", False):
            for key in ("intrinsics", "rotation", "translation"):
                camera[key] = np.asarray(camera[key], np.float64)
    if (source / "result.json").exists():
        result = json.loads((source / "result.json").read_text())
        if result.get("status") != "succeeded":
            raise ValueError("The saved reconstruction did not succeed")
    savedGeometry = readArrays(source / "geometry.npz")
    delight = readArrays(source / "delight.npz")
    albedoImages, validMasks = delight["albedoImages"], delight["validMasks"]
    if (albedoImages.ndim != 4 or albedoImages.shape[-1] != 3
            or validMasks.shape != albedoImages.shape[:-1]
            or len(cameras) != len(albedoImages) or not len(cameras)
            or not np.isfinite(albedoImages).all() or not np.isfinite(validMasks).all()
            or np.any((albedoImages < 0) | (albedoImages > 1))
            or np.any((validMasks < 0) | (validMasks > 1))):
        raise ValueError("Saved OpenDelight arrays do not match the camera count or RGB contract")
    active = [index for index, camera in enumerate(cameras) if camera.get("valid", False)]
    if not active:
        raise ValueError("No accepted saved cameras; this tool does not refit cameras")
    for index in active:
        if cameras[index].get("viewIndex", index) != index:
            raise ValueError("Saved cameras have inconsistent view ordering")
    branchPaths = {name: experiment / "branches" / f"{name}.npz" for name in BRANCHES}
    branches = {name: loadBranch(path) for name, path in branchPaths.items()}
    tracked = [source / "geometry.npz", source / "cameras.json", source / "delight.npz", *branchPaths.values()]
    photos, delightPng = {}, {}
    for index in active:
        photoPath, delightPath = (source / f"{stem}{index:02d}.png" for stem in ("input", "delight"))
        photos[index], delightPng[index] = readRgb(photoPath), readRgb(delightPath)
        if photos[index].shape != albedoImages[index].shape or delightPng[index].shape != photos[index].shape:
            raise ValueError(f"Saved image and OpenDelight array sizes disagree for view {index}")
        tracked.extend((photoPath, delightPath))
    hashes = {str(path): fileHash(path) for path in tracked}
    import torch
    import pytorch3d
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # Tangent frames are resolution-independent. The helper's unused UV frame
    # atlas is kept small; actual texture sampling uses each full branch atlas.
    geometry = prepareGeometry(savedGeometry, 32)
    output.mkdir(parents=True, exist_ok=True)
    panels, labels, viewReports = [], [], []
    totalAbsolute = dict.fromkeys(BRANCHES, 0.)
    totalChannels = 0
    center = (np.min(geometry["vertices"], axis=0) + np.max(geometry["vertices"], axis=0)) / 2

    def frontalScore(index):
        camera = cameras[index]
        direction = -np.asarray(camera["translation"]) @ np.asarray(camera["rotation"]) - center
        return float(direction[2] / max(np.linalg.norm(direction), 1e-12))

    frontalIndex = max(active, key=frontalScore)
    closeupPanels, closeupLabels, cropMetadata = [], [], {}
    for index in active:
        print(f"Rendering original working frame {index + 1}/{len(cameras)}", flush=True)
        photo, target = photos[index], np.asarray(albedoImages[index], np.float32)
        viewDir = output / f"view{index:02d}"
        viewDir.mkdir()
        camera = cameras[index]
        renderer = Pytorch3dFaceRenderer(geometry, device=args.device, imageSize=photo.shape[:2], frameGeometry=False)
        renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
        projection = compareProjection(renderer, geometry, camera)
        visible = renderer.visible.detach().cpu().numpy()
        if not visible.any():
            raise ValueError(f"No visible geometry in saved camera {index}")
        colors, support, rendered = {}, {}, {}
        rowPanels = [asImage(photo), asImage(delightPng[index])]
        rowLabels = [f"View {index + 1}: input", "OpenDelight input"]
        for name, branch in branches.items():
            colors[name] = renderer.sampleAtlas(branch["albedo"]).detach().cpu().numpy()
            support[name] = visible & (renderer.sampleAtlas(branch["coverage"]).squeeze(-1).detach().cpu().numpy() >= .999)
            rendered[name] = colors[name].copy()
            rendered[name][visible & ~support[name]] = .32
            rendered[name][~visible] = [.105, .12, .14]
            writeImage(viewDir / f"{name}.png", rendered[name])
            writeImage(viewDir / f"{name}Support.png", support[name])
            rowPanels.append(asImage(rendered[name]))
            rowLabels.append(LABELS[name])
        shared = visible & (validMasks[index] >= .999)
        for mask in support.values():
            shared &= mask
        writeImage(viewDir / "sharedSupport.png", shared)
        pixelCount = int(shared.sum())
        metrics = {}
        for name in BRANCHES:
            absolute = np.abs(colors[name][shared] - target[shared]).astype(np.float64)
            metrics[name] = {"meanAbsoluteNativeRgbError": float(absolute.mean()) if pixelCount else None,
                             "meanAbsoluteError255": float(absolute.mean() * 255) if pixelCount else None,
                             "observedVisiblePixelCount": int(support[name].sum())}
            totalAbsolute[name] += float(absolute.sum())
        totalChannels += pixelCount * 3
        panels.extend(rowPanels)
        labels.extend(rowLabels)
        makeGrid(rowPanels, rowLabels, columns=5, panelSize=args.size,
                 subtitle="Frozen geometry + cameras | unlit native RGB | gray = unsupported").save(viewDir / "comparison.png")
        viewReports.append({"viewIndex": index, "frameSizeHw": list(photo.shape[:2]), "projectionCheck": projection,
                            "rasterPrincipalPointShiftPixels": [.5, .5], "sharedSupportPixelCount": pixelCount,
                            "visibleFacePixelCount": int(visible.sum()), "branches": metrics})
        if index == frontalIndex:
            cropMetadata = {"viewIndex": index, "regionsXyxy": focusCrops(visible),
                            "selection": "closest saved camera to canonical +Z front",
                            "cropPolicy": "approximate fractional bands of visible face bounds; no saved semantic landmarks"}
            for region, crop in cropMetadata["regionsXyxy"].items():
                closeupPanels.extend(panel.crop(crop) for panel in rowPanels)
                closeupLabels.extend(f"{region}: {label}" for label in ("Input", "OpenDelight", "Dense UV", "Projection", "COLMAP"))
        del renderer
    makeGrid(panels, labels, columns=5, panelSize=args.size,
             subtitle="Same mesh / saved cameras | native RGB, no shading | gray = missing observations").save(output / "overview.png")
    makeGrid(closeupPanels, closeupLabels, columns=5, panelSize=args.size,
             subtitle=f"View {frontalIndex + 1} close-ups | identical pixel crops; display enlargement only").save(output / "closeups.png")

    canonicalMetadata = None
    if not args.skip_canonical:
        print("Rendering fixed-light canonical previews", flush=True)
        renderer = Pytorch3dFaceRenderer(geometry, device=args.device, imageSize=args.size, frameGeometry=True)
        canonicalPanels, canonicalLabels = [], []
        for azimuth in (-35, 0, 35):
            renderer.setView(azimuth=azimuth)
            for name, branch in branches.items():
                material = renderer.prepareMaterial(branch["albedo"], branch["coverage"])
                rendered = renderer.render(material, (-.6, .5, 1))
                writeImage(output / f"canonical{azimuth:+03d}_{name}.png", rendered)
                canonicalPanels.append(asImage(rendered))
                canonicalLabels.append(f"{azimuth:+d} deg: {LABELS[name]}")
        makeGrid(canonicalPanels, canonicalLabels, columns=3, panelSize=args.size,
                 subtitle="Fixed GGX light | gamma 2.2 assumed | geometry normals | missing support = gray").save(output / "canonical.png")
        canonicalMetadata = {"azimuthDegrees": [-35, 0, 35], "elevationDegrees": 0,
                             "lightDirection": [-.6, .5, 1], "keyIntensity": 2.5, "ambient": .18,
                             "roughness": .45, "specularMultiplier": .22, "baseColorEncoding": "gamma2.2 assumed",
                             "displayEncoding": "sRGB", "normalMapsUsed": False}
        del renderer
    changed = [str(path) for path in tracked if fileHash(path) != hashes[str(path)]]
    if changed:
        raise RuntimeError(f"Experiment inputs changed while rendering: {changed}")
    report = {"status": "rendered", "source": str(source), "experiment": str(experiment),
              "sourceHashes": hashes, "sourcesUnchangedAfterRendering": True,
              "renderer": "PyTorch3D MeshRasterizer; exact saved perspective cameras; unlit native RGB",
              "pytorch3dVersion": pytorch3d.__version__, "device": args.device, "seconds": perf_counter() - start,
              "photoFrameResolution": "saved working input dimensions; --size changes presentation only",
              "modelInferenceRepeated": False, "cameraFittingRepeated": False, "textureHoleFillingUsed": False,
              "activeViewIndices": active, "skippedInvalidCameraIndices": [i for i in range(len(cameras)) if i not in active],
              "views": viewReports, "closeups": cropMetadata, "canonicalPreview": canonicalMetadata,
              "metrics": {"target": "saved delight.npz albedoImages, native encoded RGB",
                          "support": "intersection of visible mesh, all three branch coverage masks and per-view OpenDelight valid mask",
                          "totalSharedPixels": totalChannels // 3,
                          "perBranch": {name: {"meanAbsoluteNativeRgbError": totalAbsolute[name] / totalChannels if totalChannels else None,
                                                "meanAbsoluteError255": 255 * totalAbsolute[name] / totalChannels if totalChannels else None}
                                        for name in BRANCHES}},
              "comparisonLimits": [
                  "This is in-sample appearance/alignment error; all source images participated in reconstruction and texture fusion",
                  "Lower error does not certify physical albedo, identity preservation or unseen-view appearance",
                  "Saved working image frames are used, not the original uncropped uploaded image dimensions",
                  "Dense UV and projection methods use different sampling coordinates despite the same mesh and fitted cameras",
                  "Canonical lighting and gamma 2.2 are preview assumptions, not fitted physical material measurements",
                  "Close-up bands are approximate display crops, not annotated semantic regions"]}
    writeJson(output / "comparisonReport.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "seconds": report["seconds"],
                      "sharedPixels": totalChannels // 3, "metrics": report["metrics"]["perBranch"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
