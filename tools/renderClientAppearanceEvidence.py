"""Render all delivered appearance branches using the saved estimated photo cameras.

This is a scientific inspection export, not image synthesis or camera fitting.
RGBA images retain a transparent background for pixel-aligned photo overlays.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from faceAppearance.legacyPreviewAsset import _load
from renderClientAssembly import ClientPreviewRenderer, fileHash, linearFromSrgb


DEFAULT_BRANCHES = (
    {"id": "baseline", "label": "修正前"},
    {"id": "eyes", "label": "仅修正眼球"},
    {"id": "texture", "label": "仅修正贴图边界"},
    {"id": "combined", "label": "眼球与贴图均修正"},
)
MODES = ("baseColor", "lit", "clay")


def comparisonConfig(directory):
    """Read branch names without allowing them to become arbitrary paths."""
    path = directory / "config.json"
    if not path.is_file():
        return {"branches": list(DEFAULT_BRANCHES), "defaultLeft": "baseline", "defaultRight": "combined"}, None
    config = json.loads(path.read_text(encoding="utf-8"))
    branches = config.get("branches")
    if not isinstance(branches, list) or not 2 <= len(branches) <= 8:
        raise ValueError("Comparison config requires 2 to 8 branches")
    ids, normalized = set(), []
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("Comparison branches must be objects")
        name, label = branch.get("id"), branch.get("label")
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None
                or name in ids or not isinstance(label, str) or not label.strip() or len(label) > 160):
            raise ValueError("Comparison branches require unique safe IDs and nonempty labels")
        ids.add(name)
        normalized.append({"id": name, "label": label})
    left = config.get("defaultLeft", normalized[0]["id"])
    right = config.get("defaultRight", normalized[-1]["id"])
    if left not in ids or right not in ids:
        raise ValueError("Comparison default branches must be in branches")
    return {"branches": normalized, "defaultLeft": left, "defaultRight": right}, fileHash(path)


def evidenceGrid(panels, labels, *, columns, panelSize, subtitle):
    """Build a contact sheet with readable Chinese and English branch labels."""
    if not panels or len(panels) != len(labels):
        raise ValueError("Expected nonempty panels with one label per panel")
    rows = math.ceil(len(panels) / columns)
    header, labelHeight = 48, 40
    canvas = Image.new("RGB", (columns * panelSize, rows * (panelSize + labelHeight) + header), (22, 27, 33))
    draw = ImageDraw.Draw(canvas)
    fontPaths = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    fontPath = next((path for path in fontPaths if path.is_file()), None)
    font = ImageFont.truetype(str(fontPath), 16) if fontPath else ImageFont.load_default(size=16)
    draw.text((14, 12), subtitle, font=font, fill=(200, 214, 226))
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index % columns * panelSize
        y = index // columns * (panelSize + labelHeight) + header
        labelFont = font
        if fontPath:
            for size in range(16, 9, -1):
                labelFont = ImageFont.truetype(str(fontPath), size)
                if draw.textlength(label, font=labelFont) <= panelSize - 20:
                    break
        draw.text((x + 10, y + 10), label, font=labelFont, fill=(235, 239, 244))
        resized = ImageOps.contain(panel.convert("RGB"), (panelSize, panelSize), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + (panelSize - resized.width) // 2,
                              y + labelHeight + (panelSize - resized.height) // 2))
    return canvas


class EvidenceRenderer(ClientPreviewRenderer):
    """Reuse the audited photo-camera/depth-layer method without atlas resampling."""

    def __init__(self, archivePath: Path, similarity: dict, *, device="cpu", facesPerPixel=16):
        import torch
        from pytorch3d.structures import Meshes

        source = _load(archivePath)
        self.source, self.torch, self.device = source, torch, device
        self.facesPerPixel = facesPerPixel
        tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, device=device, dtype=dtype)
        actual = source.manifest["assemblyDiagnostics"]["templateToUff"]
        for key in ("scale", "rotation", "translation"):
            if not np.array_equal(np.asarray(actual[key]), np.asarray(similarity[key])):
                raise ValueError(f"Export and saved geometry have different {key}")
        vertices = source.mesh["vertices"] * 100
        vertices = vertices @ np.asarray(similarity["rotation"]) * similarity["scale"] + similarity["translation"]
        self.vertices, self.faces = tensor(vertices), tensor(source.mesh["faces"], torch.int64)
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces])
        self.faceUvs = tensor(source.mesh["uvs"][source.mesh["uvFaces"]])
        self.materialIds = tensor(source.mesh["materialIds"], torch.int64)
        self.materials = source.materials["materials"]
        self.maps = []
        for material in self.materials:
            image = Image.open(io.BytesIO(source.payloads[material["baseColorTexture"]["path"]])).convert("RGBA")
            rgba = np.asarray(image, np.float32) / 255
            rgba[..., :3] = linearFromSrgb(rgba[..., :3])
            if material["alphaMode"] == "OPAQUE":
                rgba[..., 3] = 1
            self.maps.append(tensor(rgba).permute(2, 0, 1)[None])
        self.roughness = tensor([max(.05, m["roughnessFactor"]) for m in self.materials])
        self.metallic = tensor([m["metallicFactor"] for m in self.materials])

    def variants(self, camera: dict, imageShape: tuple):
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.ops import interpolate_face_attributes
        from faceAppearance.pytorch3dRenderer import linearToSrgb
        import torch.nn.functional as F

        torch = self.torch
        with torch.no_grad():
            camera = {key: np.asarray(camera[key], np.float64) for key in ("intrinsics", "rotation", "translation")}
            rasterCamera, projectionCheck = self.photoCamera(camera, imageShape)
            settings = RasterizationSettings(image_size=imageShape, blur_radius=0,
                faces_per_pixel=self.facesPerPixel, cull_backfaces=True, perspective_correct=True,
                bin_size=0 if self.device == "cpu" else None)
            fragments = MeshRasterizer(cameras=rasterCamera, raster_settings=settings)(self.mesh)
            faceLayers = fragments.pix_to_face[0]
            materialLayers = self.materialIds[faceLayers.clamp_min(0)]
            accepted = faceLayers >= 0
            # Alpha is tested on every depth candidate, before the nearest visible
            # surface is selected. This reveals the eyes under transparent cards.
            uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, self.faceUvs)[0]
            for i, material in enumerate(self.materials):
                if material["alphaMode"] != "MASK":
                    continue
                mask = accepted & (materialLayers == i)
                if mask.any():
                    grid = (uvs[mask] * 2 - 1).reshape(1, 1, -1, 2)
                    alpha = F.grid_sample(self.maps[i][:, 3:4], grid, mode="bilinear",
                        padding_mode="border", align_corners=False).reshape(-1)
                    accepted[mask] = alpha >= material["alphaCutoff"]
            visible = accepted.any(-1)
            exhausted = ~visible & (faceLayers[..., -1] >= 0)
            if exhausted.any():
                raise RuntimeError("Alpha-rejected depth queue exhausted; increase facesPerPixel")
            selected = accepted.to(torch.int64).argmax(-1)
            rows, cols = torch.meshgrid(torch.arange(imageShape[0], device=self.device),
                                       torch.arange(imageShape[1], device=self.device), indexing="ij")
            chosenFaces = faceLayers[rows, cols, selected]
            chosenBary = fragments.bary_coords[0, rows, cols, selected]
            chosenUvs = uvs[rows, cols, selected]
            chosenMaterial = materialLayers[rows, cols, selected]
            interpolate = lambda value: interpolate_face_attributes(chosenFaces[None, ..., None],
                chosenBary[None, ..., None, :], value)[0, ..., 0, :]
            normals = F.normalize(interpolate(self.mesh.verts_normals_packed()[self.faces]), dim=-1)
            positions = interpolate(self.vertices[self.faces])
            view = F.normalize(rasterCamera.get_camera_center()[0] - positions, dim=-1)
            color = torch.zeros((*imageShape, 3), device=self.device)
            for i, texture in enumerate(self.maps):
                mask = visible & (chosenMaterial == i)
                if mask.any():
                    grid = (chosenUvs[mask] * 2 - 1).reshape(1, 1, -1, 2)
                    color[mask] = F.grid_sample(texture[:, :3], grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[0, :, 0].T
            # The key is camera-relative, so all views and all branches use the
            # same inspection illumination. It is not fitted to the photograph.
            lightWorld = np.asarray(camera["rotation"]).T @ np.array([-.5, -.5, -1.])
            light = F.normalize(torch.as_tensor(lightWorld, device=self.device, dtype=torch.float32), dim=-1)
            half = F.normalize(light + view, dim=-1)
            ndl = (normals * light).sum(-1, keepdim=True).clamp(0, 1)
            ndv = (normals * view).sum(-1, keepdim=True).clamp(0, 1)
            ndh = (normals * half).sum(-1, keepdim=True).clamp(0, 1)
            vdh = (view * half).sum(-1, keepdim=True).clamp(0, 1)

            def shade(base, roughness, metallic):
                alpha2 = roughness ** 4
                distribution = alpha2 / (math.pi * (ndh.square() * (alpha2 - 1) + 1).square()).clamp_min(1e-8)
                smith = lambda cosine: 2 * cosine / (cosine + torch.sqrt(alpha2 + (1 - alpha2) * cosine.square())).clamp_min(1e-8)
                f0 = .04 * (1 - metallic) + base * metallic
                fresnel = f0 + (1 - f0) * (1 - vdh) ** 5
                specular = distribution * smith(ndl) * smith(ndv) * fresnel / (4 * ndl * ndv).clamp_min(1e-8)
                return .22 * base + ((1 - fresnel) * (1 - metallic) * base / math.pi + specular) * 2.5 * ndl

            linear = {"baseColor": color,
                "lit": shade(color, self.roughness[chosenMaterial, None], self.metallic[chosenMaterial, None]),
                "clay": shade(torch.full_like(color, .42), torch.full_like(ndl, .65), torch.zeros_like(ndl))}
            images = {}
            for name, values in linear.items():
                rgb = linearToSrgb(values).clamp(0, 1)
                rgb[~visible] = 0
                rgba = torch.cat((rgb, visible[..., None]), dim=-1)
                if not torch.isfinite(rgba).all():
                    raise FloatingPointError("Nonfinite render")
                images[name] = Image.fromarray(np.rint(rgba.cpu().numpy() * 255).astype(np.uint8), "RGBA")
            stats = {"visiblePixels": int(visible.sum()),
                "alphaRevealedDeeperPixels": int((visible & (selected > 0)).sum()),
                "alphaRejectedCandidates": int(((faceLayers >= 0) & ~accepted).sum()),
                "maximumSelectedDepthLayer": int(selected[visible].max()),
                "unresolvedPixels": 0, "projectionCheck": projectionCheck,
                "lightDirectionCanonical": lightWorld.tolist()}
            return images, stats


def solid(image):
    result = Image.new("RGBA", image.size, (27, 33, 40, 255))
    result.alpha_composite(image)
    return result.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=ROOT / "runs/clientAppearanceFix20261012/comparison")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/clientAppearanceFix20261012/deployedSource")
    parser.add_argument("--photos", type=Path, default=ROOT / "runs/legacyPreview/preview-fb21ed26fc4b41969d581365622054e5")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--facesPerPixel", type=int, default=16)
    args = parser.parse_args()
    if not 2 <= args.facesPerPixel <= 128:
        raise ValueError("facesPerPixel must be in [2,128]")
    config, configSha256 = comparisonConfig(args.comparison)
    args.output = args.output or args.comparison / "evidence"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evidence output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    inputs = json.loads((args.source / "inputs.json").read_text())
    cameras = json.loads((args.source / "cameras.json").read_text())
    geometryPath = args.source / "directClient/refinement/clientGeometry.npz"
    with np.load(geometryPath, allow_pickle=False) as geom:
        similarity = {"rotation": geom["templateToUffRotation"].tolist(),
            "translation": geom["templateToUffTranslation"].tolist(), "scale": float(geom["templateToUffScale"])}
    if len(inputs) != len(cameras) or len(inputs) != 4:
        raise ValueError("Expected four saved input/camera pairs")
    views, photoImages = [], []
    labels = ["照片 1 · 正面", "照片 2 · 转头侧面 A", "照片 3 · 转头侧面 B", "照片 4 · 抬头"]
    photoRecords = []
    for i, (record, camera) in enumerate(zip(inputs, cameras)):
        if camera.get("valid") is not True or camera.get("viewIndex") != i:
            raise ValueError("Saved camera invalid or view order mismatch")
        photoPath = args.photos / f"{i:02d}.image"
        if fileHash(photoPath) != record["sha256"]:
            raise ValueError("Photo bytes differ from the actual deployed input")
        # Replay original ingestion with its saved detection, not a new detector.
        working, replay = readImage(photoPath, imageSize=record["workingSize"][0],
            cropPolicy=record["requestedCropPolicy"], cropMargin=record["cropMargin"],
            faceDetector=lambda _, saved=record: saved["faceDetection"])
        for key in ("inputToWorking", "orientedSize", "workingSize", "cropBoxXyxy", "paddingLeftTopRightBottom"):
            if replay[key] != record[key]:
                raise ValueError(f"Photo preprocessing differs from stored {key}")
        photo = Image.fromarray(np.rint(working * 255).astype(np.uint8))
        original = ImageOps.exif_transpose(Image.open(photoPath)).convert("RGB")
        photo.save(args.output / f"view{i}-photo.png")
        original.save(args.output / f"view{i}-original.png")
        photoImages.append(photo)
        views.append({"id": f"view{i}", "viewIndex": i, "label": labels[i],
            "photo": f"view{i}-photo.png", "original": f"view{i}-original.png",
            "width": photo.width, "height": photo.height, "renders": {},
            "comparison": f"view{i}-comparison.png"})
        photoRecords.append({"viewIndex": i, "inputSha256": record["sha256"],
            "originalSize": original.size, "workingSize": record["workingSize"],
            "inputToWorking": record["inputToWorking"], "workingToInput": record["workingToInput"],
            "cropBoxXyxy": record["cropBoxXyxy"], "paddingLeftTopRightBottom": record["paddingLeftTopRightBottom"],
            "paddingMode": record["paddingMode"], "resize": "Pillow LANCZOS; exact original ingestion replay",
            "camera": {key: camera[key] for key in ("valid", "viewIndex", "intrinsics", "rotation", "translation")}})
    renderRecords, sourceHashes = [], {}
    for branchRecord in config["branches"]:
        branch = branchRecord["id"]
        archive = args.comparison / branch / "legacyAsset.zip"
        renderer = EvidenceRenderer(archive, similarity, device=args.device, facesPerPixel=args.facesPerPixel)
        sourceHashes[branch] = fileHash(archive)
        for i, camera in enumerate(cameras):
            images, stats = renderer.variants(camera, (photoImages[i].height, photoImages[i].width))
            views[i]["renders"][branch] = {}
            for mode, image in images.items():
                filename = f"view{i}-{branch}-{mode}.png"
                image.save(args.output / filename)
                views[i]["renders"][branch][mode] = filename
            renderRecords.append({"viewIndex": i, "branch": branch, **stats})
            print(json.dumps({"branch": branch, "view": i, "seconds": round(perf_counter() - start, 2),
                              "visiblePixels": stats["visiblePixels"]}), flush=True)
    overviewPanels, overviewLabels = [], []
    for i, view in enumerate(views):
        panels, names = [], []
        for mode, modeLabel in (("baseColor", "贴图基色"), ("lit", "共同光照")):
            panels.append(photoImages[i])
            names.append(f"{view['label']} · 原照片")
            for branchRecord in config["branches"]:
                branch = branchRecord["id"]
                with Image.open(args.output / view["renders"][branch][mode]) as image:
                    panels.append(solid(image))
                names.append(f"{branchRecord['label']} · {modeLabel}")
        evidenceGrid(panels, names, columns=1 + len(config["branches"]), panelSize=512,
            subtitle=f"{view['label']} | 所有分支共用估计相机与裁剪；共同光照未拟合照片").save(args.output / view["comparison"])
        overviewPanels.extend(panels)
        overviewLabels.extend(names)
    evidenceGrid(overviewPanels, overviewLabels, columns=1 + len(config["branches"]), panelSize=384,
        subtitle="四视角 · 每视角上行基色、下行共同光照 · 所有分支共用相机与原始裁剪").save(args.output / "overview.png")
    manifest = {"schemaVersion": 1, "status": "complete", "views": views, "overview": "overview.png",
        **config, "report": "renderReport.json", "notes": [
            "各视角使用同一次重建保存的估计相机；所有分支共用相机和裁剪，未为对照重新对齐。",
            "原照片沿用重建时的裁剪、边缘填充与 512×512 缩放，未经去光或美化。",
            "基色不加预览光照；统一光照与灰模使用同一灯光，未拟合照片中的真实光照。",
            "完整头部包含眼球与附件，保留原 UV，并按材质透明度选择可见表面。",
            "这是投影与观感检查；估计相机误差、表情差异及模板后脑/脖子仍会影响叠加。"]}
    files = {}
    for path in sorted(args.output.glob("*.png")):
        with Image.open(path) as image:
            files[path.name] = {"sha256": fileHash(path), "bytes": path.stat().st_size,
                "width": image.width, "height": image.height, "mode": image.mode}
    report = {"schemaVersion": 1, "status": "complete", "seconds": perf_counter() - start,
        "device": args.device, "cameraRefitted": False, "calibratedCamera": False,
        "meshModified": False, "textureModified": False, "photoModifiedBeyondOriginalPreprocessing": False,
        "templateToUff": similarity, "sourceArchiveSha256": sourceHashes,
        "cameraSourceSha256": fileHash(args.source / "cameras.json"),
        "inputsSourceSha256": fileHash(args.source / "inputs.json"),
        "geometrySourceSha256": fileHash(geometryPath), "implementationSha256": fileHash(Path(__file__)),
        "comparisonConfigSha256": configSha256, **config,
        "photos": photoRecords, "renders": renderRecords, "files": files,
        "textureSampling": "Original full-resolution sRGB PNG decoded to linear; perspective UV interpolation; bilinear pixel-edge sampler; border clamp; no atlas resizing",
        "lighting": {"baseColor": "No lighting", "lit": "linear Lambert + GGX/Smith/Schlick; material roughness",
            "clay": "same shader, linear gray 0.42, roughness 0.65, metallic 0; original alpha cutouts retained",
            "keyDirectionCamera": [-.5, -.5, -1.], "keyIntensity": 2.5, "ambient": .22,
            "exposureAdjustment": False, "photoIlluminationFitted": False},
        "raster": {"facesPerPixel": args.facesPerPixel, "cullBackfaces": True, "perspectiveCorrect": True,
            "supersampling": False, "alpha": "Binary MASK cutoff before nearest accepted depth; OPAQUE ignores alpha"},
        "limitations": manifest["notes"] + ["No cast shadows, ambient occlusion, subsurface scattering, environment lighting or estimated normal maps."]}
    writeJson(args.output / "renderReport.json", report)
    writeJson(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "output": str(args.output), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
