"""Render original/refined native UFF at unchanged saved photo cameras.

Input manifest: {cases: [{caseId, label, sourceDirectory, geometryPath,
texturePath (optional), texturePriorityPath (optional), resultPath (optional),
frontViewIndex (optional)}]}.
No geometry inference, camera fitting, or texture fusion happens here.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import traceback

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.photoCameras import originalImageIntrinsics
from renderUffMultiCase import (as_image, box_from_points, digest, expand_eye_crop,
    grid, overlay, photo_projection_check, render_passes, renderer_geometry)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_source_hashes(case):
    source = Path(case["sourceDirectory"]).resolve()
    paths = [source / name for name in ("geometry.npz", "texture.npz", "cameras.json", "inputs.json", "result.json")]
    paths.extend(Path(case[key]).resolve() for key in ("geometryPath", "texturePath", "texturePriorityPath") if case.get(key))
    return {str(path): digest(path) for path in paths}


def resume_case(case, output, maximum_side):
    """Skip only a complete, hash-identical render; never replace a partial one."""
    if not output.exists() or not any(output.iterdir()):
        return None
    report_path = output / "diagnostics.json"
    if not report_path.is_file():
        raise FileExistsError(f"Existing render is incomplete; preserve it and choose a new output: {output}")
    report = read_json(report_path)
    if (report.get("status") != "rendered" or report.get("sourceHashes") != case_source_hashes(case)
            or report.get("maximumPhotoSide") != maximum_side
            or ("frontViewIndex" in case and report.get("selectedFrontView") != case["frontViewIndex"])):
        raise ValueError(f"Existing render inputs/settings changed: {output}")
    if not report.get("artifactHashes"):
        raise ValueError("Existing render lacks output hash validation")
    for relative, expected in report["artifactHashes"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Existing output changed: {path}")
    for view in report["views"]:
        if digest(view["sourcePhoto"]) != view["photoSha256"]:
            raise ValueError("Existing render's source photograph changed")
    print(f"{case['caseId']} unchanged complete render verified; skipped", flush=True)
    return report


def read_case(case):
    """Resolve explicit input paths and enforce unchanged topology/UVs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case["caseId"]):
        raise ValueError("Unsafe case identifier")
    source = Path(case["sourceDirectory"]).resolve()
    refined_path = Path(case["geometryPath"]).resolve()
    with np.load(source / "geometry.npz", allow_pickle=False) as arrays:
        original = dict(arrays)
    with np.load(refined_path, allow_pickle=False) as arrays:
        refined = dict(arrays)
    # A refinement may store only positions; topology remains the source's.
    for name in ("faces", "uvs"):
        if name in refined and not np.array_equal(refined[name], original[name]):
            raise ValueError(f"Refinement changed {name}")
        refined[name] = original[name]
    if (refined["vertices"].shape != original["vertices"].shape
            or not np.isfinite(refined["vertices"]).all()):
        raise ValueError("Refined positions are invalid")
    texture_path = Path(case["texturePath"]).resolve() if case.get("texturePath") else None
    textures = {}
    priority_path = Path(case["texturePriorityPath"]).resolve() if case.get("texturePriorityPath") else None
    for name, path in (("original", source / "texture.npz"), ("refused", texture_path), ("priority", priority_path)):
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as arrays:
            textures[name] = {"albedo": arrays["albedo"], "coverageMask": arrays["coverageMask"]}
        if not np.isfinite(textures[name]["albedo"]).all():
            raise ValueError("Nonfinite saved albedo")
    return source, original, refined, textures


def render_case(case, output, device, maximum_side):
    import torch
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    started = perf_counter()
    source, original, refined, textures = read_case(case)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs, cameras = read_json(source / "inputs.json"), read_json(source / "cameras.json")
    accepted = np.array([i for i, c in enumerate(cameras) if c.get("valid", False)])
    if not len(accepted):
        raise ValueError("No accepted source cameras")
    centers = np.array([-np.asarray(c["rotation"]).T @ np.asarray(c["translation"]) for c in cameras])
    yaws = np.degrees(np.arctan2(centers[:, 0], centers[:, 2]))
    front = int(case.get("frontViewIndex", accepted[np.argmin(np.abs(yaws[accepted]))]))
    if front not in accepted:
        raise ValueError("Selected front is not an accepted saved camera")
    side = int(accepted[np.argmax(np.abs(yaws[accepted]))])
    boundary_path = ROOT / "assets/anatomical/uff6986/boundaries.json"
    boundaries = validateBoundaryAsset(original["vertices"], original["faces"], read_json(boundary_path))
    eye_ids = np.concatenate([ids for name, ids in boundaries.items() if "eye" in name.lower()])
    geometry = {name: renderer_geometry(values) for name, values in (("original", original), ("refined", refined))}
    hashes = case_source_hashes(case)
    reports, front_panels, front_labels, front_eyes = [], [], [], []
    for index, item in enumerate(inputs):
        photo_path = Path(item["path"])
        if digest(photo_path) != item["sha256"]:
            raise ValueError(f"Source photo changed: {photo_path}")
        with Image.open(photo_path) as photo_image:
            photo_image = ImageOps.exif_transpose(photo_image).convert("RGB")
        if list(photo_image.size) != item["orientedSize"]:
            raise ValueError("Source image dimensions changed")
        factor = min(1., maximum_side / max(photo_image.size))
        size = tuple(max(32, round(v * factor)) for v in photo_image.size)
        sx, sy = size[0] / photo_image.width, size[1] / photo_image.height
        resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        image = photo_image.resize(size, Image.Resampling.LANCZOS)
        photo = np.asarray(image, np.float32) / 255.
        saved_camera = cameras[index]
        camera = {"intrinsics": resize @ originalImageIntrinsics(saved_camera["intrinsics"], item["inputToWorking"]),
                  "rotation": np.asarray(saved_camera["rotation"]), "translation": np.asarray(saved_camera["translation"])}
        light = camera["rotation"].T @ np.array([-.45, -.55, -1.])
        passes, errors = {}, {}
        for name in ("original", "refined"):
            renderer = Pytorch3dFaceRenderer(geometry[name], device=device, imageSize=(size[1], size[0]), frameGeometry=False)
            renderer.setPerspectiveCamera(**camera)
            errors[name] = photo_projection_check(renderer, geometry[name]["vertices"], camera)
            passes[name] = render_passes(renderer, textures["original"]["albedo"], textures["original"]["coverageMask"], light)
            passes[name]["overlay"] = overlay(photo, passes[name]["clay"], passes[name]["visible"])
            if name == "refined" and "refused" in textures:
                passes["refused"] = render_passes(renderer, textures["refused"]["albedo"], textures["refused"]["coverageMask"], light)
            if name == "refined" and "priority" in textures:
                passes["priority"] = render_passes(renderer, textures["priority"]["albedo"], textures["priority"]["coverageMask"], light)
            del renderer
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        view_dir = output / f"view{index:02d}"
        view_dir.mkdir()
        image.save(view_dir / "photo.png")
        for name, values in passes.items():
            for pass_name in ("clay", "baseColor", "overlay", "visible"):
                if pass_name in values:
                    writeImage(view_dir / f"{name}{pass_name[0].upper()+pass_name[1:]}.png", values[pass_name])
        xy_a, depths = projectPoints(original["vertices"], **camera)
        xy_b, _ = projectPoints(refined["vertices"], **camera)
        points = np.r_[xy_a[depths > 0], xy_b[depths > 0]]
        if "faceMasks" in original:
            mask = cv2.warpPerspective(original["faceMasks"][index], resize @ np.linalg.inv(np.asarray(item["inputToWorking"])), size)
            yy, xx = np.nonzero(mask > .5)
            if len(xx):
                points = np.r_[points, np.c_[xx, yy]]
        crop = box_from_points(points, size, square=True)
        eye_crop = expand_eye_crop(box_from_points(np.r_[xy_a[eye_ids], xy_b[eye_ids]], size, padding=.3), size)
        panels = [image, as_image(passes["original"]["clay"]), as_image(passes["refined"]["clay"]),
                  as_image(passes["original"]["baseColor"]), as_image(passes["refined"]["baseColor"])]
        labels = ["Source photo", "A original / gray", "B refined / gray", "A original / color", "B refined / original color"]
        if "refused" in passes:
            panels.append(as_image(passes["refused"]["baseColor"]))
            labels.append("C refined / recomputed color")
        if "priority" in passes:
            panels.append(as_image(passes["priority"]["baseColor"]))
            labels.append("D refined / front-priority color")
        full = [panel.crop(crop) for panel in panels]
        eyes = [panel.crop(eye_crop) for panel in panels]
        title = f"{case['caseId']} / view {index} | SAME saved photo camera and light; source photos are fitting inputs"
        grid(full, labels, len(panels), panel_size=300, subtitle=title).save(view_dir / "comparison.png")
        grid(eyes, labels, len(panels), panel_size=360, panel_height=200,
             subtitle="Native UFF eye openings; no eyeballs or client template").save(view_dir / "eyes.png")
        overlays = [image, as_image(passes["original"]["overlay"]), as_image(passes["refined"]["overlay"])]
        grid([p.crop(crop) for p in overlays], ["Source photo", "A gray overlay 50%", "B gray overlay 50%"], 3,
             panel_size=400, subtitle="Holes retain source photo in overlays; inspect standalone gray renders as well").save(view_dir / "overlays.png")
        if index == front:
            front_panels, front_labels, front_eyes = full, labels, eyes
            for name, dest in (("comparison.png", "frontComparison.png"), ("eyes.png", "frontEyes.png"), ("overlays.png", "frontOverlays.png")):
                shutil.copy2(view_dir / name, output / dest)
        if index == side:
            shutil.copy2(view_dir / "comparison.png", output / "sideComparison.png")
        reports.append({"viewIndex": index, "cameraAccepted": bool(saved_camera.get("valid")),
                        "photoSha256": item["sha256"], "sourcePhoto": str(photo_path), "renderSize": size,
                        "originalToRender": resize, "camera": camera, "lightCanonical": light,
                        "projectionAgreementMaximumPixels": errors, "faceCropXyxy": crop, "eyeCropXyxy": eye_crop,
                        "savedCameraCenterYawDegrees": float(yaws[index])})
        print(f"{case['caseId']} view {index + 1}/{len(inputs)} rendered", flush=True)
    report = {"caseId": case["caseId"], "label": case.get("label", case["caseId"]), "status": "rendered",
              "seconds": perf_counter() - started, "sourceDirectory": str(source), "sourceHashes": hashes,
              "maximumPhotoSide": maximum_side,
              "selectedFrontView": front, "selectedSideView": side, "views": reports,
              "vertexCount": len(original["vertices"]), "faceCount": len(original["faces"]),
              "topologyAndUvsUnchanged": True, "savedCamerasUnchanged": True, "hasRecomputedTexture": "refused" in textures,
              "hasFrontPriorityTexture": "priority" in textures,
              "limits": ["All photographs were reconstruction/refinement inputs, not independent ground truth",
                         "Fixed-light gray preview uses geometry normals; not fitted original lighting",
                         "Base color is saved encoded RGB shown unlit; unknown texels are gray",
                         "Only native UFF face surface, without eyeballs, lashes, oral cavity or client template",
                         "Photo overlays retain source photo behind geometric holes"]}
    report["artifactHashes"] = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*.png"))}
    writeJson(output / "diagnostics.json", report)
    return report, front_panels, front_labels, front_eyes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/uffFrontPriority20261011/frontManifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/uffFrontPriority20261011/preview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximumPhotoSide", type=int, default=1024)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if args.maximumPhotoSide < 256:
        parser.error("maximumPhotoSide must be at least 256")
    manifest = read_json(args.manifest)
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    if args.case:
        cases = [case for case in cases if case["caseId"] in args.case]
    if not cases:
        raise ValueError("No selected cases")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    snapshots = args.output / "manifestSnapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot_path = snapshots / (digest(args.manifest) + ".json")
    if not snapshot_path.exists():
        shutil.copy2(args.manifest, snapshot_path)
    records = []
    for case in cases:
        if not case.get("geometryPath") or not Path(case["geometryPath"]).is_file():
            records.append({"caseId": case["caseId"], "status": "notRendered", "reason": "No refined geometry candidate",
                            "candidateStatus": case.get("status"), "error": case.get("error")})
            writeJson(args.output / "progress.json", records)
            continue
        try:
            record = resume_case(case, args.output / case["caseId"], args.maximumPhotoSide)
            if record is None:
                record, _, _, _ = render_case(case, args.output / case["caseId"], args.device, args.maximumPhotoSide)
        except Exception as error:
            record = {"caseId": case["caseId"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            print(record["traceback"], flush=True)
        records.append(record)
        writeJson(args.output / "progress.json", records)
    # Rebuild aggregate contact sheets from complete per-case images, including
    # verified resumed cases; each row retains its own column labels.
    for source_name, output_name in (("frontComparison.png", "frontOverview.png"), ("frontEyes.png", "eyesOverview.png")):
        rows = []
        for record in records:
            if record["status"] == "rendered":
                with Image.open(args.output / record["caseId"] / source_name) as image:
                    rows.append(image.convert("RGB"))
        if rows:
            canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (22, 27, 33))
            y = 0
            for row in rows:
                canvas.paste(row, (0, y)); y += row.height
            canvas.save(args.output / output_name)
    writeJson(args.output / "diagnostics.json", {"status": "completed" if all(r["status"] == "rendered" for r in records) else "partial",
              "manifestPath": str(args.manifest.resolve()), "manifestSha256": digest(args.manifest), "cases": records})
    if any(r["status"] == "failed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
