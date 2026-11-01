#!/usr/bin/env python3
"""Render a frozen-geometry three-branch eyelid view-prior comparison.

The original, soft, and strong view-selection branches are rendered with one
saved frontal camera. Rasterization is restricted to a four-times-resolution
eye ROI; no texture filtering, eye alignment, or geometry adjustment is applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderClientAssembly import fileHash

BRANCHES = (("baseline", "原选图"), ("soft", "温和正面优先"), ("strong", "强正面优先"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    start = perf_counter()
    out = args.output or args.case_dir / "eyes"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Eye evidence output must be empty: {out}")
    observation_manifest = json.loads((args.observations / "manifest.json").read_text())
    for manifest_key, filename in (("inputsSha256", "inputs.json"), ("camerasSha256", "cameras.json")):
        if observation_manifest.get(manifest_key) != fileHash(args.source / filename):
            raise ValueError(f"Eye observations do not match {filename}")
    cameras = json.loads((args.source / "cameras.json").read_text())
    camera = cameras[0]
    if camera.get("valid") is not True or camera.get("viewIndex") != 0:
        raise ValueError("Expected valid saved view 0 camera")
    archive_paths = {branch: args.case_dir / branch / "legacyAsset.zip" for branch, _ in BRANCHES}
    assets = {branch: _load(path) for branch, path in archive_paths.items()}
    reference = assets["baseline"]
    geometry_checks = {}
    for branch, asset in assets.items():
        if set(asset.mesh) != set(reference.mesh):
            raise ValueError(f"Different mesh arrays in {branch}")
        checks = {name: bool(np.array_equal(value, asset.mesh[name])) for name, value in reference.mesh.items()}
        if not all(checks.values()):
            raise ValueError(f"This comparison requires identical geometry and UVs: {branch}: {checks}")
        if asset.materials != reference.materials:
            raise ValueError(f"This comparison requires identical materials: {branch}")
        geometry_checks[branch] = checks
    similarity = reference.manifest["assemblyDiagnostics"]["templateToUff"]
    with np.load(args.observations / "view00.npz", allow_pickle=False) as cache:
        curves = np.asarray(cache["eyelidCurvesWorking"], np.float64)
    if curves.shape[0] != 2 or curves.shape[-1] != 2 or not np.isfinite(curves).all():
        raise ValueError("Expected finite working-image curves for two eyes")
    original_photo = Image.open(args.observations / "view00WorkingPhoto.png").convert("RGB")
    boxes = []
    for points in curves:
        points = points.reshape(-1, 2)
        box = np.r_[np.floor(points.min(0) - [15, 17]), np.ceil(points.max(0) + [15, 17])].astype(int)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, original_photo.width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, original_photo.height)
        if np.any(box[2:] <= box[:2]):
            raise ValueError("Empty eye ROI")
        boxes.append(box)
    boxes = np.asarray(boxes)
    roi = np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]
    factor = 4
    # Pixel-center mapping: integer camera coordinates describe source centers;
    # PIL crop/resize maps center x to factor * (x - left + .5) - .5.
    pixel_map = np.array([[factor, 0, factor * (.5 - roi[0]) - .5],
                          [0, factor, factor * (.5 - roi[1]) - .5], [0, 0, 1]])
    zoom = {key: np.asarray(camera[key], float) for key in ("intrinsics", "rotation", "translation")}
    zoom["intrinsics"] = pixel_map @ zoom["intrinsics"]
    width, height = map(int, (roi[2:] - roi[:2]) * factor)
    photo = original_photo.crop(tuple(roi)).resize((width, height), Image.Resampling.LANCZOS)
    out.mkdir(parents=True, exist_ok=True)
    photo.save(out / "photo.png")
    images, records = {}, []
    for branch, label in BRANCHES:
        renderer = EvidenceRenderer(archive_paths[branch], similarity, device=args.device)
        images[branch], record = renderer.variants(zoom, (height, width))
        for mode, value in images[branch].items():
            value.save(out / f"{branch}-{mode}.png")
        records.append({"branch": branch, **record})
        print(json.dumps({"branch": branch, "seconds": round(perf_counter() - start, 2)}), flush=True)
    panels, labels = [], []
    for i, box in enumerate(boxes):
        crop = np.r_[(box[:2] - roi[:2]) * factor, (box[2:] - roi[:2]) * factor]
        eye = "图左眼" if i == 0 else "图右眼"
        for mode, title in (("baseColor", "贴图基色"), ("lit", "统一光照")):
            panels.append(photo.crop(tuple(crop)))
            labels.append(f"{eye} · 原照片")
            for branch, label in BRANCHES:
                panels.append(solid(images[branch][mode]).crop(tuple(crop)))
                labels.append(f"{eye} · {label} · {title}")
    evidenceGrid(panels, labels, columns=4, panelSize=380,
                 subtitle="同一几何与相机 · 眼睑选图优先级对照 · 本地实验").save(out / "comparison.png")
    files = {}
    for path in sorted(out.glob("*.png")):
        with Image.open(path) as image:
            files[path.name] = {"sha256": fileHash(path), "size": list(image.size), "mode": image.mode}
    writeJson(out / "report.json", {
        "status": "complete", "seconds": perf_counter() - start,
        "sourceArchives": {branch: fileHash(path) for branch, path in archive_paths.items()},
        "cameraSourceSha256": fileHash(args.source / "cameras.json"),
        "inputsSourceSha256": fileHash(args.source / "inputs.json"),
        "observationsSha256": fileHash(args.observations / "view00.npz"),
        "workingPhotoSha256": fileHash(args.observations / "view00WorkingPhoto.png"),
        "implementationSha256": fileHash(Path(__file__)), "geometryIdentical": True,
        "geometryArraysExactlyEqual": geometry_checks, "materialsIdentical": True,
        "roiXyxyWorking": roi.tolist(), "eyeBoxesXyxyWorking": boxes.tolist(),
        "roiRasterScale": factor, "roiPixelMap": pixel_map.tolist(), "records": records,
        "cameraRefitted": False, "geometryModified": False, "textureModified": False,
        "photoUpsampling": "Pillow LANCZOS for display only; no added source detail",
        "files": files,
    })
    print(json.dumps({"output": str(out.resolve()), "seconds": perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
