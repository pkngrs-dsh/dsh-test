"""Render saved background-ablation geometry with frozen-A and own-PnP cameras.

Manifest cases contain caseId, label, sourceDirectory, referenceViewIndex (0),
and variants [{id, label, status, outputDirectory}]. Each completed variant
contains geometry.npz, cameras.json, inputs.json and result.json. inputs.json
describes the original photographs; inputNN.png stores the modified model input.
The freshly reconstructed original variant supplies the frozen camera. This
tool performs no inference, camera fitting, texture reading or texture fusion.
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

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.photoCameras import originalImageIntrinsics
from renderUffMultiCase import (as_image, box_from_points, digest, expand_eye_crop,
    grid, overlay, photo_projection_check, renderer_geometry)


VARIANTS = ("original", "backgroundGray", "headGray", "headBlack")
VARIANT_LABELS = {"original": "A original", "backgroundGray": "B background gray",
                  "headGray": "C head gray", "headBlack": "D head black"}
PANEL_FILES = ("fixedComparison", "fixedEyes", "fixedOverlays", "ownOverlays", "ownEyes")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variant_entries(case):
    values = {item["id"]: item for item in case["variants"]}
    return {name: values[name] for name in VARIANTS if name in values}


def variant_complete(item):
    if item.get("status") not in {"succeeded", "completed"}:
        return False
    source = Path(item["outputDirectory"])
    return all((source / name).is_file() for name in
               ("geometry.npz", "cameras.json", "inputs.json", "result.json"))


def contact_paths(case):
    directory = Path(case["inputsPath"]).resolve().parent if case.get("inputsPath") else None
    candidates = {"inputContact": case.get("inputContactPath") or (directory / "inputContactSheet.png" if directory else None),
                  "maskContact": case.get("maskContactPath") or (directory / "maskContactSheet.png" if directory else None)}
    return {name: Path(path).resolve() for name, path in candidates.items() if path and Path(path).is_file()}


def case_source_hashes(case):
    paths = []
    for item in variant_entries(case).values():
        source = Path(item["outputDirectory"]).resolve()
        paths.extend(source / name for name in
                     ("geometry.npz", "cameras.json", "inputs.json", "result.json"))
        paths.extend(sorted(source.glob("input[0-9][0-9].png")))
    paths.extend(contact_paths(case).values())
    paths.append(ROOT / "assets/anatomical/uff6986/boundaries.json")
    return {str(path): digest(path) for path in paths}


def load_case(case):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case["caseId"]):
        raise ValueError("Unsafe case identifier")
    entries = variant_entries(case)
    if set(entries) != set(VARIANTS) or not all(variant_complete(v) for v in entries.values()):
        raise ValueError("All four reconstructed variants must be complete")
    variants = {}
    for name, item in entries.items():
        source = Path(item["outputDirectory"]).resolve()
        result = read_json(source / "result.json")
        if result.get("status") != "succeeded":
            raise ValueError(f"Variant {name} did not succeed")
        with np.load(source / "geometry.npz", allow_pickle=False) as arrays:
            geometry = {key: arrays[key] for key in ("vertices", "faces", "uvs")}
        vertices, faces, uvs = (geometry[key] for key in ("vertices", "faces", "uvs"))
        if (vertices.shape != (6986, 3) or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0
                or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)
                or uvs.shape != (6986, 2) or not np.isfinite(uvs).all()):
            raise ValueError(f"Invalid native UFF geometry: {name}")
        inputs, cameras = read_json(source / "inputs.json"), read_json(source / "cameras.json")
        if len(cameras) != len(inputs) or not inputs:
            raise ValueError(f"Input/camera count differs: {name}")
        variants[name] = {"source": source, "geometry": geometry, "inputs": inputs, "cameras": cameras}
    original = variants["original"]
    for name, data in variants.items():
        for key in ("faces", "uvs"):
            if not np.array_equal(data["geometry"][key], original["geometry"][key]):
                raise ValueError(f"Variant {name} changed {key}")
        if len(data["inputs"]) != len(original["inputs"]):
            raise ValueError("Variant photograph count differs")
        for a, b in zip(original["inputs"], data["inputs"]):
            if any(a[key] != b[key] for key in ("path", "sha256", "orientedSize")):
                raise ValueError("Variant inputs do not reference the same original photograph")
    return variants


def converted_camera(saved, item, resize):
    camera = {"intrinsics": resize @ originalImageIntrinsics(saved["intrinsics"], item["inputToWorking"]),
              "rotation": np.asarray(saved["rotation"], dtype=np.float64),
              "translation": np.asarray(saved["translation"], dtype=np.float64).reshape(3)}
    if camera["rotation"].shape != (3, 3) or not all(np.isfinite(v).all() for v in camera.values()):
        raise ValueError("Saved camera contains invalid values")
    return camera


def gray_pass(renderer, light):
    import torch
    from faceAppearance.pytorch3dRenderer import linearToSrgb

    direction = torch.as_tensor(light, dtype=renderer.normals.dtype, device=renderer.normals.device)
    direction /= torch.linalg.vector_norm(direction)
    diffuse = (renderer.normals * direction).sum(-1).clamp(0, 1)
    clay = linearToSrgb((.13 + .57 * diffuse)[..., None].expand(-1, -1, 3)).cpu().numpy()
    visible = renderer.visible.cpu().numpy()
    clay[~visible] = [.105, .12, .14]
    return {"clay": clay, "visible": visible}


def resume_case(case, output, maximum_side):
    if not output.exists() or not any(output.iterdir()):
        return None
    report_path = output / "diagnostics.json"
    if not report_path.is_file():
        raise FileExistsError(f"Incomplete preview exists; preserve it and select a new output: {output}")
    report = read_json(report_path)
    if (report.get("status") != "rendered" or report.get("sourceHashes") != case_source_hashes(case)
            or report.get("maximumPhotoSide") != maximum_side
            or report.get("selectedFrontView") != int(case.get("referenceViewIndex", 0))
            or report.get("rendererSha256") != digest(__file__)):
        raise ValueError(f"Existing preview inputs/settings changed: {output}")
    if not report.get("artifactHashes"):
        raise ValueError("Existing preview has no artifact hashes")
    for relative, expected in report["artifactHashes"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Existing preview output changed: {path}")
    for view in report["views"]:
        if digest(view["sourcePhoto"]) != view["photoSha256"]:
            raise ValueError("Source photograph changed")
    print(f"{case['caseId']} unchanged complete preview verified; skipped", flush=True)
    return report


def stack_images(paths, output):
    rows = []
    for path in paths:
        if path.is_file():
            with Image.open(path) as source:
                rows.append(source.convert("RGB"))
    if not rows:
        return
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (22, 27, 33))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output)


def render_case(case, output, device, maximum_side):
    import torch
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    started = perf_counter()
    variants = load_case(case)
    original = variants["original"]
    inputs, cameras = original["inputs"], original["cameras"]
    front = int(case.get("referenceViewIndex", 0))
    if not 0 <= front < len(inputs):
        raise ValueError("Reference photo index outside input list")
    centers = np.array([-np.asarray(c["rotation"]).T @ np.asarray(c["translation"]).reshape(3) for c in cameras])
    yaws = np.degrees(np.arctan2(centers[:, 0], centers[:, 2]))
    accepted = [i for i, camera in enumerate(cameras) if camera.get("valid", False)]
    if not accepted:
        raise ValueError("Fresh A has no accepted saved PnP camera")
    side = max(accepted, key=lambda index: abs(yaws[index]))
    boundaries = validateBoundaryAsset(original["geometry"]["vertices"], original["geometry"]["faces"],
        read_json(ROOT / "assets/anatomical/uff6986/boundaries.json"))
    loops = [ids for name, ids in boundaries.items() if "eye" in name.lower()]
    if len(loops) != 2:
        raise ValueError("Expected two annotated native UFF eye openings")
    eye_ids = np.concatenate(loops)
    geometry = {name: renderer_geometry(data["geometry"]) for name, data in variants.items()}
    hashes = case_source_hashes(case)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reports, input_panels, input_labels = [], [], []
    for index, item in enumerate(inputs):
        photo_path = Path(item["path"])
        if digest(photo_path) != item["sha256"]:
            raise ValueError(f"Source photograph changed: {photo_path}")
        with Image.open(photo_path) as source:
            original_photo = ImageOps.exif_transpose(source).convert("RGB")
        if list(original_photo.size) != item["orientedSize"]:
            raise ValueError("Source photograph dimensions changed")
        factor = min(1., maximum_side / max(original_photo.size))
        size = tuple(max(32, int(round(value * factor))) for value in original_photo.size)
        sx, sy = size[0] / original_photo.width, size[1] / original_photo.height
        resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        image = original_photo.resize(size, Image.Resampling.LANCZOS)
        photo = np.asarray(image, np.float32) / 255.
        fixed = converted_camera(cameras[index], item, resize)
        light = fixed["rotation"].T @ np.array([-.45, -.55, -1.])
        passes, camera_reports, projected, projected_eyes = {}, {}, [], []
        for name, data in variants.items():
            renderer = Pytorch3dFaceRenderer(geometry[name], device=device, imageSize=(size[1], size[0]), frameGeometry=False)
            own = converted_camera(data["cameras"][index], data["inputs"][index], resize)
            passes[name], camera_reports[name] = {}, {}
            for camera_name, camera in (("fixed", fixed), ("own", own)):
                renderer.setPerspectiveCamera(**camera)
                error = photo_projection_check(renderer, geometry[name]["vertices"], camera)
                rendered = gray_pass(renderer, light)
                rendered["overlay"] = overlay(photo, rendered["clay"], rendered["visible"])
                passes[name][camera_name] = rendered
                xy, depth = projectPoints(geometry[name]["vertices"], **camera)
                projected.append(xy[depth > 0])
                projected_eyes.append(xy[eye_ids][depth[eye_ids] > 0])
                camera_reports[name][camera_name] = {**camera,
                    "source": "fresh A saved PnP" if camera_name == "fixed" else f"{name} saved PnP",
                    "acceptedByPipeline": bool((cameras if camera_name == "fixed" else data["cameras"])[index].get("valid")),
                    "maximumProjectionDifferencePixels": error,
                    "visiblePixels": int(rendered["visible"].sum())}
            del renderer
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        crop = box_from_points(np.concatenate(projected), size, square=True)
        eye_crop = expand_eye_crop(box_from_points(np.concatenate(projected_eyes), size, padding=.3), size)
        view_dir = output / f"view{index:02d}"
        view_dir.mkdir()
        image.save(view_dir / "photo.png")
        for name, values in passes.items():
            for camera_name, rendered in values.items():
                for pass_name in ("clay", "overlay", "visible"):
                    writeImage(view_dir / f"{name}_{camera_name}_{pass_name}.png", rendered[pass_name])
        specifications = (("fixedComparison", "fixed", "clay", crop), ("fixedEyes", "fixed", "clay", eye_crop),
                          ("fixedOverlays", "fixed", "overlay", crop), ("ownOverlays", "own", "overlay", crop),
                          ("ownEyes", "own", "overlay", eye_crop))
        for filename, camera_name, pass_name, framing in specifications:
            panels = [image.crop(framing)] + [as_image(passes[name][camera_name][pass_name]).crop(framing) for name in VARIANTS]
            labels = ["Original photograph"] + [VARIANT_LABELS[name] for name in VARIANTS]
            own_note = "OWN PnP (geometry + camera)" if camera_name == "own" else "FROZEN freshly reconstructed A PnP"
            title = f"{case['caseId']} / view {index} | {own_note} | shared light"
            if pass_name == "overlay":
                title += " | gray overlay 50%"
            grid(panels, labels, 5, panel_size=360, panel_height=210 if "Eyes" in filename else 360,
                 subtitle=title).save(view_dir / f"{filename}.png")
        for prefix, selected in (("front", front), ("side", side)):
            if index == selected:
                for name in PANEL_FILES:
                    shutil.copy2(view_dir / f"{name}.png", output / f"{prefix}{name[0].upper() + name[1:]}.png")
        for name, data in variants.items():
            path = data["source"] / f"input{index:02d}.png"
            if path.is_file():
                with Image.open(path) as source:
                    input_panels.append(source.convert("RGB"))
                input_labels.append(f"view {index} / {VARIANT_LABELS[name]}")
        reports.append({"viewIndex": index, "sourcePhoto": str(photo_path), "photoSha256": item["sha256"],
                        "originalSize": original_photo.size, "renderSize": size, "originalToRender": resize,
                        "faceCropXyxy": crop, "eyeCropXyxy": eye_crop, "cameras": camera_reports,
                        "fixedLightCanonical": light, "freshACameraCenterYawDegrees": float(yaws[index])})
        print(f"{case['caseId']} view {index + 1}/{len(inputs)} rendered", flush=True)
    for name in PANEL_FILES:
        stack_images([output / f"view{index:02d}" / f"{name}.png" for index in range(len(inputs))], output / f"all{name[0].upper() + name[1:]}.png")
    contacts = contact_paths(case)
    for name, path in contacts.items():
        with Image.open(path) as source:
            source.convert("RGB").save(output / f"{name}.png")
    if "inputContact" not in contacts and len(input_panels) == len(inputs) * len(VARIANTS):
        grid(input_panels, input_labels, 4, panel_size=300,
             subtitle=f"{case['caseId']} | saved model inputs; A / B / C / D").save(output / "inputContact.png")
    report = {"caseId": case["caseId"], "status": "rendered", "seconds": perf_counter() - started,
              "sourceHashes": hashes, "rendererSha256": digest(__file__), "maximumPhotoSide": maximum_side,
              "selectedFrontView": front, "selectedSideView": side, "views": reports,
              "vertexCount": 6986, "faceCount": len(original["geometry"]["faces"]),
              "fixedCameraVariant": "original", "usesFreshOriginalGeometry": True,
              "topologyAndUvsUnchanged": True, "savedCamerasUnchanged": True,
              "textureRead": False, "modelInferenceRepeated": False, "cameraFittingRepeated": False,
              "limits": ["Photographs are reconstruction inputs, not independent geometry ground truth",
                         "Own-PnP overlays combine changes in geometry and camera; use frozen-A panels for comparison",
                         "One chosen light uses geometry normals only; it is not fitted original illumination",
                         "Native UFF face surface only, without eyeballs, lashes, oral cavity or client head",
                         "Overlay holes retain original photograph; inspect standalone gray panels as well",
                         "Each inference uses native canonical coordinates; no inter-variant registration is fitted"]}
    report["artifactHashes"] = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*.png"))}
    writeJson(output / "diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012/backgroundManifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012/preview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximumPhotoSide", type=int, default=1024)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if args.maximumPhotoSide < 256:
        parser.error("maximumPhotoSide must be at least 256")
    manifest = read_json(args.manifest)
    cases = [case for case in manifest["cases"] if not args.case or case["caseId"] in args.case]
    if not cases:
        raise ValueError("No selected cases")
    if len({case["caseId"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identifier")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    snapshots = args.output / "manifestSnapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f"{digest(args.manifest)}.json"
    if not snapshot.exists():
        shutil.copy2(args.manifest, snapshot)
    records = []
    for case in cases:
        variants = variant_entries(case)
        if set(variants) != set(VARIANTS) or not all(variant_complete(item) for item in variants.values()):
            records.append({"caseId": case["caseId"], "status": "notRendered", "reason": "Four variants are not complete",
                            "variants": {name: item.get("status") for name, item in variants.items()}})
            writeJson(args.output / "progress.json", records)
            continue
        try:
            output = args.output / case["caseId"]
            record = resume_case(case, output, args.maximumPhotoSide)
            if record is None:
                record = render_case(case, output, args.device, args.maximumPhotoSide)
        except Exception as error:
            record = {"caseId": case["caseId"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            print(record["traceback"], flush=True)
        records.append(record)
        writeJson(args.output / "progress.json", records)
    for name in PANEL_FILES:
        filename = f"front{name[0].upper() + name[1:]}.png"
        stack_images([args.output / record["caseId"] / filename for record in records if record["status"] == "rendered"],
                     args.output / f"overview{name[0].upper() + name[1:]}.png")
    writeJson(args.output / "diagnostics.json", {"status": "completed" if all(r["status"] == "rendered" for r in records) else "partial",
              "manifestPath": str(args.manifest.resolve()), "manifestSha256": digest(args.manifest), "cases": records})
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
