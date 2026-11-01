"""Inspect saved native UFF runs at unchanged fitted and native photo cameras.

No inference, camera fitting, client assembly or anatomical refinement occurs.
Photos are reconstruction inputs, not held-out geometry or reflectance truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.normalTransfer import buildCornerTangentFrames, convertUvs
from faceAppearance.photoCameras import canonicalCamera, originalImageIntrinsics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_image(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


def grid(panels, labels, columns, panel_size=320, subtitle="", panel_height=None):
    panel_height = panel_height or panel_size
    row_height, header = panel_height + 34, 44 if subtitle else 0
    canvas = Image.new("RGB", (columns * panel_size, math.ceil(len(panels) / columns) * row_height + header), (22, 27, 33))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw = ImageDraw.Draw(canvas)
    if subtitle:
        draw.text((12, 12), subtitle, fill=(208, 220, 232), font=font)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (index % columns) * panel_size, (index // columns) * row_height + header
        draw.text((x + 8, y + 8), label, fill="white", font=font)
        resized = ImageOps.contain(panel, (panel_size, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + (panel_size - resized.width) // 2, y + 34 + (panel_height - resized.height) // 2))
    return canvas


def box_from_points(points, size, *, padding=.12, square=False):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=-1)]
    if len(points) == 0:
        raise ValueError("Cannot crop empty projected geometry")
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = np.maximum(upper - lower, 24.) * (1 + 2 * padding)
    if square:
        extent[:] = extent.max()
    center = (lower + upper) / 2
    # PIL supplies a black border for any out-of-photo area; the same crop is
    # applied to all compared panels, retaining aspect ratio and alignment.
    lower, upper = np.floor(center - extent / 2).astype(int), np.ceil(center + extent / 2).astype(int)
    if square:
        return tuple(np.r_[lower, upper].tolist())
    lower, upper = np.maximum(lower, 0), np.minimum(upper, size)
    return tuple(np.r_[lower, upper].tolist())


def renderer_geometry(saved):
    geometry = {key: saved[key] for key in ("vertices", "faces", "uvs")}
    frames = buildCornerTangentFrames(geometry["vertices"], geometry["faces"], geometry["uvs"],
                                     geometry["faces"], uvOrigin="vDown")
    return {**geometry, "objUvs": convertUvs(geometry["uvs"], sourceOrigin="vDown", targetOrigin="vUp"),
            "uvFaces": geometry["faces"], "cornerTangentFrames": frames["cornerFrames"],
            "faceTangentValidMask": frames["faceValidMask"]}


def expand_eye_crop(crop, size):
    """Include eyelids/brows rather than only a narrow hole-height strip."""
    left, top, right, bottom = crop
    center_y = (top + bottom) / 2
    height = max(bottom - top, (right - left) * .4)
    return (left, max(0, int(math.floor(center_y - height / 2))), right,
            min(size[1], int(math.ceil(center_y + height / 2))))


def photo_projection_check(renderer, vertices, camera):
    expected, depth = projectPoints(vertices, camera["intrinsics"], camera["rotation"], camera["translation"])
    actual = renderer.projectionCamera.transform_points_screen(renderer.mesh.verts_packed())[:, :2].cpu().numpy()
    valid = (depth > 0) & np.isfinite(expected).all(axis=-1)
    if not valid.any():
        raise ValueError("No positive-depth mesh vertices at saved camera")
    error = float(np.abs(expected[valid] - actual[valid]).max())
    if error > .005:
        raise ValueError(f"OpenCV/PyTorch3D projection difference {error:.6f}px")
    return error


def render_passes(renderer, albedo, coverage, light):
    import torch
    from faceAppearance.pytorch3dRenderer import linearToSrgb

    visible = renderer.visible.cpu().numpy()
    normals = renderer.normals
    direction = torch.as_tensor(light, dtype=torch.float32, device=normals.device)
    direction = direction / torch.linalg.vector_norm(direction)
    # An ordinary matte gray preview exposes geometry without albedo or normal
    # maps; its chosen light is not an estimate of original illumination.
    diffuse = (normals * direction).sum(-1).clamp(0, 1)
    clay = linearToSrgb((.13 + .57 * diffuse)[..., None].expand(-1, -1, 3)).cpu().numpy()
    sampled = renderer.sampleAtlas(albedo).cpu().numpy()
    observed = renderer.sampleAtlas(coverage).squeeze(-1).cpu().numpy() >= .999
    sampled[~observed] = [.32, .32, .32]
    # Native OpenDelight encoded RGB is displayed directly. No gamma decode /
    # relighting is needed for this unlit preview; encoding remains uncalibrated.
    clay[~visible] = [.105, .12, .14]
    sampled[~visible] = [.105, .12, .14]
    return {"clay": clay, "baseColor": sampled, "visible": visible,
            "observedPixels": int((visible & observed).sum()), "unobservedPixels": int((visible & ~observed).sum())}


def overlay(photo, render, mask):
    result = photo.copy()
    result[mask] = .5 * photo[mask] + .5 * render[mask]
    return result


def failed_input_case(case, output):
    """Keep failed inputs visible; never manufacture a missing reconstruction."""
    output.mkdir(parents=True)
    photos, labels, hashes = [], [], {}
    for index, path_string in enumerate(case.get("paths", [])):
        path = Path(path_string)
        hashes[str(path)] = digest(path)
        recorded = next((item["sha256"] for item in case.get("files", []) if item["path"] == str(path)), None)
        if recorded is not None and hashes[str(path)] != recorded:
            raise ValueError("Failed-case input changed since selection")
        with Image.open(path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        photo.save(output / f"input{index:02d}.png")
        photos.append(photo); labels.append(f"{case['caseId']} input {index+1}")
    blank = Image.new("RGB", (400, 400), (35, 35, 40))
    draw = ImageDraw.Draw(blank)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    draw.multiline_text((16, 125), "Input validation failed\nNo reconstructed geometry\nNo material output", fill="white", font=font, spacing=12)
    while len(photos) < 6:
        photos.append(blank); labels.append(f"{case['caseId']} / failed")
    grid(photos, labels, 6, subtitle="FAILED INPUT | expected one face; original inputs retained; no reconstruction fabricated").save(output / "photoOverview.png")
    record = {"caseId": case["caseId"], "status": "inputFailure", "reconstructionStatus": case.get("status"),
              "error": case.get("error"), "sourceHashes": hashes,
              "geometryRendered": False, "reason": "No succeeded reconstruction result or saved geometry"}
    writeJson(output / "diagnostics.json", record)
    return record, photos, labels


def render_case(case, output, device, maximum_side):
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
    import torch

    start = perf_counter()
    source = Path(case["outputDirectory"]).resolve()
    paths = [source / name for name in ("geometry.npz", "texture.npz", "cameras.json", "viewTransforms.json", "inputs.json", "result.json")]
    hashes = {path.name: digest(path) for path in paths}
    result = json.loads((source / "result.json").read_text())
    if result.get("status") != "succeeded":
        raise ValueError("Run result is not succeeded")
    inputs = json.loads((source / "inputs.json").read_text())
    fitted = json.loads((source / "cameras.json").read_text())
    transforms = json.loads((source / "viewTransforms.json").read_text())
    with np.load(source / "geometry.npz", allow_pickle=False) as saved:
        saved_geometry = dict(saved)
    with np.load(source / "texture.npz", allow_pickle=False) as saved:
        albedo, coverage = saved["albedo"], saved["coverageMask"]
    if not np.isfinite(saved_geometry["vertices"]).all() or not np.isfinite(albedo).all():
        raise ValueError("Nonfinite geometry or base color")
    geometry = renderer_geometry(saved_geometry)
    boundary_path = ROOT / "assets/anatomical/uff6986/boundaries.json"
    boundaries = validateBoundaryAsset(geometry["vertices"], geometry["faces"], json.loads(boundary_path.read_text()))
    eye_paths = {name: values for name, values in boundaries.items() if "eye" in name.lower()}
    if len(eye_paths) != 2:
        raise ValueError("Expected two explicitly annotated UFF eye-hole loops for crop framing")
    eye_ids = np.concatenate(list(eye_paths.values()))
    output.mkdir(parents=True)
    records, all_panels, all_labels, selected_panels, selected_labels, eye_panels, eye_labels = [], [], [], [], [], [], []
    projection_centers = [-np.asarray(camera["rotation"]).T @ np.asarray(camera["translation"]) for camera in fitted]
    angles = np.array([math.degrees(math.atan2(center[0], center[2])) for center in projection_centers])
    accepted_ids = np.array([index for index, camera in enumerate(fitted) if camera.get("valid", False)])
    if not len(accepted_ids):
        raise ValueError("No accepted saved fitted cameras")
    front_index = int(accepted_ids[np.argmin(np.abs(angles[accepted_ids]))])
    side_index = int(accepted_ids[np.argmax(np.abs(angles[accepted_ids]))])
    for view_index, item in enumerate(inputs):
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"Photo differs from saved reconstruction: {path}")
        with Image.open(path) as source_image:
            original = ImageOps.exif_transpose(source_image).convert("RGB")
        if list(original.size) != item["orientedSize"]:
            raise ValueError("Photo oriented dimensions changed")
        factor = min(1., maximum_side / max(original.size))
        size = tuple(max(32, int(round(value * factor))) for value in original.size)
        # Exact resize pixel affine: integer-center coordinates map with the
        # half-pixel convention, independent of the source crop affine.
        sx, sy = size[0] / original.width, size[1] / original.height
        resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        image = original.resize(size, Image.Resampling.LANCZOS) if original.size != size else original
        photo = np.asarray(image, np.float32) / 255.
        native = {**canonicalCamera(saved_geometry["cameraExtrinsics"][view_index], transforms[view_index]),
                  "intrinsics": saved_geometry["cameraIntrinsics"][view_index]}
        converted = {}
        for name, camera in (("fitted", fitted[view_index]), ("native", native)):
            converted[name] = {"intrinsics": resize @ originalImageIntrinsics(camera["intrinsics"], item["inputToWorking"]),
                               "rotation": np.asarray(camera["rotation"]), "translation": np.asarray(camera["translation"]),
                               "source": "postFusionUvCorrespondencePnP; saved unchanged" if name == "fitted" else native["source"],
                               "acceptedByPipeline": bool(camera.get("valid", False)) if name == "fitted" else None}
        light = converted["fitted"]["rotation"].T @ np.array([-.45, -.55, -1.])
        renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=(size[1], size[0]), frameGeometry=False)
        passes, camera_reports = {}, {}
        for name, camera in converted.items():
            renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
            error = photo_projection_check(renderer, geometry["vertices"], camera)
            passes[name] = render_passes(renderer, albedo, coverage, light)
            passes[name]["overlay"] = overlay(photo, passes[name]["clay"], passes[name]["visible"])
            camera_reports[name] = {**camera, "maximumOpenCvPytorch3dProjectionErrorPixels": error,
                                    "visiblePixels": int(passes[name]["visible"].sum()),
                                    "observedPixels": passes[name]["observedPixels"],
                                    "unobservedPixels": passes[name]["unobservedPixels"],
                                    "rasterPrincipalPointShiftPixels": [.5, .5]}
        view_dir = output / f"view{view_index:02d}"
        view_dir.mkdir()
        image.save(view_dir / "photo.png")
        for name, passes_for_camera in passes.items():
            for pass_name in ("clay", "baseColor", "overlay", "visible"):
                writeImage(view_dir / f"{name}{pass_name[0].upper()+pass_name[1:]}.png", passes_for_camera[pass_name])
        xy, depth = projectPoints(geometry["vertices"], **{key: converted["fitted"][key] for key in ("intrinsics", "rotation", "translation")})
        # Source segmentation and projected face both participate in crop
        # framing only; they do not change the camera or geometry.
        segmentation = cv2.warpPerspective(saved_geometry["faceMasks"][view_index], resize @ np.linalg.inv(np.asarray(item["inputToWorking"])), size)
        yy, xx = np.nonzero(segmentation > .5)
        points = xy[depth > 0]
        if len(xx):
            points = np.r_[points, np.c_[xx, yy]]
        crop = box_from_points(points, size, square=True)
        eye_crop = expand_eye_crop(box_from_points(xy[eye_ids], size, padding=.30), size)
        panels = [image, as_image(passes["fitted"]["clay"]), as_image(passes["fitted"]["baseColor"]),
                  as_image(passes["fitted"]["overlay"]), as_image(passes["native"]["clay"]), as_image(passes["native"]["overlay"])]
        titles = ["Input photo", "Saved PnP / gray", "Saved PnP / native color", "Saved PnP / overlay 50%", "Native VGGT / gray", "Native VGGT / overlay 50%"]
        labels = [f"v{view_index:02d} {label}" for label in titles]
        cropped = [panel.crop(crop) for panel in panels]
        eye_cropped = [panel.crop(eye_crop) for panel in panels]
        grid(cropped, labels, 6, subtitle=f"{case['caseId']} | saved input cameras; one fixed light shared by PnP/native").save(view_dir / "comparison.png")
        grid(eye_cropped, labels, 6, panel_height=190, subtitle="UFF eye region | geometric openings; no eyeballs, lashes or client template").save(view_dir / "eyes.png")
        all_panels.extend(cropped); all_labels.extend(labels)
        if view_index in {front_index, side_index}:
            selected_panels.extend(cropped)
            selected_labels.extend([f"{case['caseId']} v{view_index} / {label}" for label in ("photo", "PnP gray", "base color", "PnP overlay", "native gray", "native overlay")])
        if view_index == front_index:
            eye_panels = [eye_cropped[index] for index in (0, 1, 2, 3)]
            eye_labels = [f"{case['caseId']} / {label}" for label in ("photo", "PnP gray", "base color", "PnP overlay")]
        records.append({"viewIndex": view_index, "sourcePhoto": str(path), "photoSha256": item["sha256"],
                        "originalSize": original.size, "renderSize": size, "originalToRender": resize,
                        "comparisonCropXyxy": crop, "eyeCropXyxy": eye_crop, "cameras": camera_reports,
                        "fixedLightCanonical": light, "fittedCameraCenterYawDegrees": float(angles[view_index]),
                        "savedCameraDiagnostic": {key: value for key, value in fitted[view_index].items()
                                                  if key not in {"intrinsics", "rotation", "translation", "diagnosticImagePoints", "diagnosticProjectedPoints"}}})
        del renderer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"{case['caseId']} photo view {view_index+1}/{len(inputs)} rendered", flush=True)
    grid(all_panels, all_labels, 6, subtitle=f"{case['caseId']} | reconstruction inputs; comparison is not held-out validation").save(output / "photoOverview.png")
    ortho = Pytorch3dFaceRenderer(geometry, device=device, imageSize=512)
    ortho_panels, ortho_labels = [], []
    for angle in (-55, 0, 55):
        ortho.setView(azimuth=angle)
        rendered = render_passes(ortho, albedo, coverage, [-.45, .55, 1.])
        for pass_name in ("clay", "baseColor"):
            panel = as_image(rendered[pass_name])
            panel.save(output / f"canonical{angle:+03d}{pass_name}.png")
            ortho_panels.append(panel); ortho_labels.append(f"Canonical {angle:+d} deg / {pass_name}")
    grid(ortho_panels, ortho_labels, 2, subtitle="Uniform framing only; geometry untouched; fixed light; gray = missing observed color").save(output / "canonicalViews.png")
    report = {"caseId": case["caseId"], "label": case.get("label"), "status": "rendered", "seconds": perf_counter() - start,
              "sourceDirectory": str(source), "sourceHashes": hashes,
              "vertexCount": len(geometry["vertices"]), "faceCount": len(geometry["faces"]),
              "selectedFrontView": front_index, "selectedSideView": side_index, "views": records,
              "texture": {"meaning": "Native fused OpenDelight encoded RGB displayed unlit; no physical reflectance claim",
                          "missingObservations": "neutral gray, no preview nearest fill or template blending",
                          "observedUvTexels": int(coverage.sum()), "totalUvTexels": int(coverage.size)},
              "eyeCrop": {"annotationAsset": str(boundary_path), "annotationAssetSha256": digest(boundary_path),
                          "usage": "UFF annotated eye-hole vertices frame crop only; no anatomical fitting or eye geometry generated"},
              "modelInferenceRepeated": False, "cameraFittingRepeated": False, "clientAssemblyRendered": False}
    writeJson(output / "diagnostics.json", report)
    del ortho
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report, selected_panels, selected_labels, eye_panels, eye_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/uffMultiCase20261011/batchManifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/uffMultiCase20261011/preview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximumPhotoSide", type=int, default=1536)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty preview output directory")
    if args.maximumPhotoSide < 256:
        raise ValueError("maximumPhotoSide must be at least 256")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    records, panels, labels, eye_panels, eye_labels = [], [], [], [], []
    start = perf_counter()
    for case in cases:
        result_path = Path(case["outputDirectory"]) / "result.json"
        if not result_path.exists() or json.loads(result_path.read_text()).get("status") != "succeeded":
            record, failed_panels, failed_labels = failed_input_case(case, args.output / case["caseId"])
            records.append(record); panels.extend(failed_panels); labels.extend(failed_labels)
            writeJson(args.output / "progress.json", records)
            continue
        try:
            record, selected, selected_labels, eyes, labels_for_eyes = render_case(case, args.output / case["caseId"], args.device, args.maximumPhotoSide)
            panels.extend(selected); labels.extend(selected_labels)
            eye_panels.extend(eyes); eye_labels.extend(labels_for_eyes)
        except Exception as error:
            record = {"caseId": case["caseId"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            print(record["traceback"], flush=True)
        records.append(record)
        writeJson(args.output / "progress.json", records)
    if panels:
        grid(panels, labels, 6, panel_size=256, subtitle="Native UFF | saved PnP vs native VGGT cameras | reconstruction-input views").save(args.output / "overview.png")
        grid(eye_panels, eye_labels, 4, panel_size=400, panel_height=200,
             subtitle="Native UFF eye regions | input photo, gray geometry, fused base color, 50% gray overlay").save(args.output / "eyesOverview.png")
    report = {"status": "completed" if all(record["status"] == "rendered" for record in records) else "partial",
              "seconds": perf_counter() - start, "manifest": str(args.manifest.resolve()), "manifestSha256": digest(args.manifest),
              "device": args.device, "cases": records,
              "limits": ["All photo views participated in reconstruction; not held-out identity evaluation",
                         "Saved post-fusion PnP and native VGGT cameras are diagnostic alternatives, not ground truth",
                         "Native cameras converted separately through each saved world-to-canonical transform",
                         "Gray previews use chosen light with geometry normals only; original lighting not estimated",
                         "Native OpenDelight encoded color is unlit and uncalibrated, not measured reflectance",
                         "Only UFF face surface rendered; no eyeballs, oral cavity, hair or client full-head template",
                         "Transparent geometry holes in overlays retain the source photo; inspect gray standalone panels too",
                         "Geometry is arbitrary-scale canonical UFF; no metric 3D accuracy claim"]}
    writeJson(args.output / "diagnostics.json", report)
    print(json.dumps({key: report[key] for key in ("status", "seconds")}), flush=True)
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
