"""Render an exported client head with saved photo and full-head preview cameras.

This is a fixed-light inspection, not a photo-error evaluation or a fitted-light
reconstruction. Exported arrays remain untouched; only an in-memory preview UV
atlas is built. Alpha MASK surfaces are rejected before choosing the first
visible depth layer, so transparent eyelash cards reveal the surfaces behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot / "src") not in sys.path:
    sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.photoCameras import canonicalCamera, originalImageIntrinsics
from renderCropComparison import makeGrid


def fileHash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linearFromSrgb(values: np.ndarray) -> np.ndarray:
    return np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)


def assetFile(directory: Path, relativeName: str) -> Path:
    path = (directory/relativeName).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or nonlocal client asset: {relativeName}")
    return path


def loadClientAsset(directory: Path, maxTextureSize: int = 1024) -> dict:
    """Validate exported files and form a linear-RGBA atlas for preview only."""
    if not 32 <= maxTextureSize <= 4096:
        raise ValueError("maxTextureSize must be between 32 and 4096")
    directory = Path(directory).resolve()
    paths = {name: assetFile(directory, name) for name in
             ("manifest.json", "clientMesh.npz", "materials.json", "diagnostics.json")}
    manifest = json.loads(paths["manifest.json"].read_text())
    if manifest.get("status") != "exported" or manifest.get("coordinates", {}).get("exportLengthUnit") != "meter":
        raise ValueError("Expected an exported client asset in meter units")
    diagnostics = json.loads(paths["diagnostics.json"].read_text())
    similarity = diagnostics["templateToUff"]
    scale = float(similarity["scale"])
    rotation, translation = np.asarray(similarity["rotation"], float), np.asarray(similarity["translation"], float)
    if (not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("templateToUff must be a finite proper similarity")
    with np.load(paths["clientMesh.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("vertices", "faces", "uvs", "uvFaces", "materialIds")}
    vertices, faces, uvs, uvFaces, materialIds = (arrays[key] for key in
                                               ("vertices", "faces", "uvs", "uvFaces", "materialIds"))
    materials = json.loads(paths["materials.json"].read_text())["materials"]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or uvFaces.shape != faces.shape or uvs.ndim != 2 or uvs.shape[1] != 2
            or not np.isfinite(uvs).all() or materialIds.shape != (len(faces),) or not materials):
        raise ValueError("Invalid exported geometry dimensions or nonfinite coordinates")
    for name, values, upperBound in (("faces", faces, len(vertices)), ("uvFaces", uvFaces, len(uvs)),
                                     ("materialIds", materialIds, len(materials))):
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= upperBound:
            raise ValueError(f"Invalid exported {name}")
    # Linear RGB is resized/interpolated; alpha stays linear. A duplicated border
    # separates material tiles even when a sample lies exactly at a UV boundary.
    tileSize, padding = maxTextureSize, 2
    gridSize = math.ceil(math.sqrt(len(materials)))
    tileStride = tileSize+padding*2
    atlasSize = gridSize*tileStride
    atlas = np.zeros((atlasSize, atlasSize, 4), np.float32)
    previewCornerUvs = np.empty((len(faces), 3, 2), np.float32)
    cutoff, roughness, metallic = [], [], []
    materialDiagnostics = []
    for index, material in enumerate(materials):
        if material.get("index") != index or material.get("alphaMode") not in {"OPAQUE", "MASK"}:
            raise ValueError("Expected sequential material indices and OPAQUE/MASK alpha")
        texture = material["baseColorTexture"]
        if texture.get("colorSpace") != "sRGB" or texture.get("alphaColorSpace") != "linear":
            raise ValueError("Expected sRGB base color with linear alpha")
        texturePath = assetFile(directory, texture["path"])
        paths[texture["path"]] = texturePath
        with Image.open(texturePath) as source:
            rgba = np.asarray(source.convert("RGBA"), np.float32)/255.
        factor = np.asarray(material.get("baseColorFactor", [1., 1., 1., 1.]), np.float32)
        if factor.shape != (4,) or not np.isfinite(factor).all() or np.any((factor < 0) | (factor > 1)):
            raise ValueError("Invalid baseColorFactor")
        rgba[..., :3] = linearFromSrgb(rgba[..., :3])
        rgba *= factor
        if material["alphaMode"] == "OPAQUE":
            rgba[..., 3] = 1.
        resized = np.stack([np.asarray(Image.fromarray(rgba[..., channel], mode="F").resize(
            (tileSize, tileSize), Image.Resampling.BILINEAR), np.float32) for channel in range(4)], axis=-1)
        row, column = divmod(index, gridSize)
        y, x = row*tileStride, column*tileStride
        atlas[y:y+tileStride, x:x+tileStride] = np.pad(resized, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
        selectedFaces = materialIds == index
        corners = np.clip(uvs[uvFaces[selectedFaces]], 0, 1)
        previewCornerUvs[selectedFaces] = (corners*(tileSize-1)+[x+padding, y+padding])/(atlasSize-1)
        alphaCutoff = float(material.get("alphaCutoff", .5))
        roughnessFactor, metallicFactor = float(material["roughnessFactor"]), float(material["metallicFactor"])
        if not all(np.isfinite(item) and 0 <= item <= 1 for item in (alphaCutoff, roughnessFactor, metallicFactor)):
            raise ValueError("Invalid alpha/roughness/metallic factor")
        cutoff.append(alphaCutoff if material["alphaMode"] == "MASK" else 0.)
        roughness.append(max(.05, roughnessFactor))
        metallic.append(metallicFactor)
        materialDiagnostics.append({"index": index, "name": material["name"], "alphaMode": material["alphaMode"],
                                    "alphaCutoff": alphaCutoff, "roughnessPreset": roughnessFactor,
                                    "previewRoughness": roughness[-1], "metallicPreset": metallicFactor,
                                    "sourceSizeHw": list(rgba.shape[:2]), "previewSizeHw": [tileSize, tileSize]})
    for name, path in paths.items():
        if name == "manifest.json":
            continue
        recorded = manifest.get("files", {}).get(name)
        if recorded is None or recorded["sha256"] != fileHash(path):
            raise ValueError(f"Exported file differs from its manifest: {name}")
    return {"vertices": (vertices*100)@rotation*scale+translation, "faces": faces,
            "previewUvsVDown": previewCornerUvs.reshape(-1, 2),
            "previewUvFaces": np.arange(len(faces)*3).reshape(-1, 3), "atlasLinearRgba": atlas,
            "faceMaterialIds": materialIds, "alphaCutoff": np.asarray(cutoff, np.float32),
            "roughness": np.asarray(roughness, np.float32), "metallic": np.asarray(metallic, np.float32),
            "diagnostics": {"templateId": manifest["templateId"], "templateToUff": similarity,
                            "exportVerticesToUff": "(verticesMeters * 100) @ rotation * scale + translation",
                            "previewAtlasSize": atlasSize, "materials": materialDiagnostics,
                            "uvSampling": "UV corners clamped to [0,1] before preview atlas packing; original UVs unchanged",
                            "outOfRangeUvCorners": int(np.any((uvs[uvFaces] < 0) | (uvs[uvFaces] > 1), axis=-1).sum()),
                            "sourceHashes": {name: fileHash(path) for name, path in paths.items()}}}


class ClientPreviewRenderer:
    """Depth-layer alpha testing plus a geometry-normal dielectric PBR preview."""

    def __init__(self, asset: dict, *, device: str = "cuda", facesPerPixel: int = 16):
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesUV

        if not 2 <= facesPerPixel <= 128:
            raise ValueError("facesPerPixel must be between 2 and 128 for cutout materials")
        self.torch, self.device, self.facesPerPixel = torch, device, facesPerPixel
        tensor = lambda values, dtype=torch.float32: torch.as_tensor(values, device=device, dtype=dtype)
        self.vertices, self.faces = tensor(asset["vertices"]), tensor(asset["faces"], torch.int64)
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces])
        objUvs = np.asarray(asset["previewUvsVDown"]).copy()
        objUvs[:, 1] = 1-objUvs[:, 1]
        self.textures = TexturesUV(maps=tensor(asset["atlasLinearRgba"])[None],
                                  faces_uvs=[tensor(asset["previewUvFaces"], torch.int64)],
                                  verts_uvs=[tensor(objUvs)], align_corners=True, padding_mode="border")
        self.materialIds = tensor(asset["faceMaterialIds"], torch.int64)
        self.alphaCutoff = tensor(asset["alphaCutoff"])
        self.roughness, self.metallic = tensor(asset["roughness"]), tensor(asset["metallic"])

    def photoCamera(self, camera: dict, imageShape: tuple) -> tuple:
        from pytorch3d.utils import cameras_from_opencv_projection

        tensor = lambda value: self.torch.as_tensor(value, device=self.device, dtype=self.torch.float32)
        cameraArgs = {"R": tensor(camera["rotation"])[None], "tvec": tensor(camera["translation"]).reshape(1, 3),
                      "image_size": tensor([imageShape])}
        intrinsics = tensor(camera["intrinsics"]).clone()
        projection = cameras_from_opencv_projection(camera_matrix=intrinsics[None], **cameraArgs)
        expected, depth = projectPoints(self.vertices.cpu().numpy(), camera["intrinsics"],
                                       camera["rotation"], camera["translation"])
        actual = projection.transform_points_screen(self.vertices)[:, :2].cpu().numpy()
        valid = (depth > 1e-9) & np.isfinite(expected).all(-1)
        if not valid.any():
            raise ValueError("Saved photo camera has no positive-depth client vertices")
        error = float(np.abs(expected[valid]-actual[valid]).max())
        if error > .005:
            raise AssertionError(f"PyTorch3D/OpenCV projection mismatch: {error} pixels")
        intrinsics[:2, 2] += .5
        raster = cameras_from_opencv_projection(camera_matrix=intrinsics[None], **cameraArgs)
        return raster, {"maximumProjectionErrorPixels": error, "rasterPrincipalPointShift": [.5, .5]}

    def render(self, cameras, imageShape: tuple, lightDirection, *, perspective: bool) -> tuple:
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.ops import interpolate_face_attributes
        from faceAppearance.pytorch3dRenderer import linearToSrgb
        import torch.nn.functional as functional

        torch = self.torch
        with torch.no_grad():
            settings = RasterizationSettings(image_size=imageShape, blur_radius=0, faces_per_pixel=self.facesPerPixel,
                                              cull_backfaces=True, perspective_correct=perspective,
                                              bin_size=0 if self.device == "cpu" else None)
            fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(self.mesh)
            faceLayers = fragments.pix_to_face[0]
            sampled = self.textures.sample_textures(fragments)[0]
            materialLayers = self.materialIds[faceLayers.clamp_min(0)]
            accepted = (faceLayers >= 0) & (sampled[..., 3] >= self.alphaCutoff[materialLayers])
            visible = accepted.any(-1)
            unresolved = ~visible & (faceLayers[..., -1] >= 0)
            if unresolved.any():
                raise RuntimeError(f"{int(unresolved.sum())} pixels exhaust all alpha-rejected depth layers; increase --facesPerPixel")
            selected = accepted.to(torch.int64).argmax(-1)
            rows, columns = torch.meshgrid(torch.arange(imageShape[0], device=self.device),
                                          torch.arange(imageShape[1], device=self.device), indexing="ij")
            selectedFaces = faceLayers[rows, columns, selected]
            selectedBary = fragments.bary_coords[0, rows, columns, selected]
            # Only selected attributes are interpolated; the K layers exist just
            # long enough to alpha-test and uncover any underlying surfaces.
            interpolate = lambda values: interpolate_face_attributes(selectedFaces[None, ..., None],
                selectedBary[None, ..., None, :], values)[0, ..., 0, :]
            normals = functional.normalize(interpolate(self.mesh.verts_normals_packed()[self.faces]), dim=-1)
            positions = interpolate(self.vertices[self.faces])
            cameraCenter = cameras.get_camera_center()[0]
            if perspective:
                view = functional.normalize(cameraCenter-positions, dim=-1)
            else:
                # The view's positive Z axis points from camera into the scene.
                view = functional.normalize(-cameras.R[0, :, 2], dim=-1).expand_as(positions)
            color = sampled[rows, columns, selected, :3]
            material = materialLayers[rows, columns, selected]
            roughness, metallic = self.roughness[material, None], self.metallic[material, None]
            light = functional.normalize(torch.as_tensor(lightDirection, device=self.device, dtype=torch.float32), dim=-1)
            half = functional.normalize(light+view, dim=-1)
            ndl, ndv = (normals*light).sum(-1, keepdim=True).clamp(0, 1), (normals*view).sum(-1, keepdim=True).clamp(0, 1)
            ndh, vdh = (normals*half).sum(-1, keepdim=True).clamp(0, 1), (view*half).sum(-1, keepdim=True).clamp(0, 1)
            alphaSquared = roughness**4
            distribution = alphaSquared/(math.pi*(ndh.square()*(alphaSquared-1)+1).square()).clamp_min(1e-8)
            smith = lambda cosine: 2*cosine/(cosine+torch.sqrt(alphaSquared+(1-alphaSquared)*cosine.square())).clamp_min(1e-8)
            f0 = .04*(1-metallic)+color*metallic
            fresnel = f0+(1-f0)*(1-vdh)**5
            specular = distribution*smith(ndl)*smith(ndv)*fresnel/(4*ndl*ndv).clamp_min(1e-8)
            linear = .22*color+((1-fresnel)*(1-metallic)*color/math.pi+specular)*2.5*ndl
            image = linearToSrgb(linear).clamp(0, 1)
            image[~visible] = torch.tensor([.105, .12, .14], device=self.device)
            if not torch.isfinite(image).all():
                raise FloatingPointError("Nonfinite client preview pixels")
            stats = {"visiblePixels": int(visible.sum()), "alphaRevealedDeeperPixels": int((visible & (selected > 0)).sum()),
                     "maximumSelectedDepthLayer": int(selected[visible].max()) if visible.any() else 0,
                     "alphaRejectedCandidates": int(((faceLayers >= 0) & ~accepted).sum()),
                     "depthQueueFullPixels": int((faceLayers[..., -1] >= 0).sum()), "unresolvedPixels": 0}
            return image.cpu().numpy(), visible.cpu().numpy(), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=projectRoot/"runs/eyeCornerPriorityFinal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewIndex", type=int, default=0)
    parser.add_argument("--cameraKind", choices=("fitted", "native"), default="fitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--maxTextureSize", type=int, default=1024)
    parser.add_argument("--facesPerPixel", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("Choose an empty preview output directory")
    if not 64 <= args.size <= 2048:
        raise ValueError("size must be between 64 and 2048")
    start = perf_counter()
    asset = loadClientAsset(args.asset, args.maxTextureSize)
    sourcePaths = {name: args.source/name for name in ("inputs.json", "cameras.json", "viewTransforms.json", "geometry.npz")}
    inputs, fitted, transforms = (json.loads(sourcePaths[name].read_text()) for name in
                                 ("inputs.json", "cameras.json", "viewTransforms.json"))
    if not 0 <= args.viewIndex < len(inputs) or len(fitted) != len(inputs) or len(transforms) != len(inputs):
        raise ValueError("Invalid source view index or metadata counts")
    metadata = inputs[args.viewIndex]
    photoPath = Path(metadata["path"])
    if fileHash(photoPath) != metadata["sha256"]:
        raise ValueError("Original source photo no longer matches its saved hash")
    with Image.open(photoPath) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    if list(photo.size) != metadata["orientedSize"]:
        raise ValueError("Original EXIF-oriented photo dimensions changed")
    if args.cameraKind == "native":
        with np.load(sourcePaths["geometry.npz"], allow_pickle=False) as geometry:
            camera = {**canonicalCamera(geometry["cameraExtrinsics"][args.viewIndex], transforms[args.viewIndex]),
                      "intrinsics": geometry["cameraIntrinsics"][args.viewIndex]}
    else:
        camera = fitted[args.viewIndex]
        if not camera.get("valid") or camera.get("viewIndex", args.viewIndex) != args.viewIndex:
            raise ValueError("The saved reference camera is invalid or mismatched")
    camera = {"intrinsics": originalImageIntrinsics(camera["intrinsics"], metadata["inputToWorking"]),
              "rotation": np.asarray(camera["rotation"]), "translation": np.asarray(camera["translation"])}
    import torch
    import pytorch3d
    from pytorch3d.renderer import FoVOrthographicCameras, look_at_view_transform

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    renderer = ClientPreviewRenderer(asset, device=args.device, facesPerPixel=args.facesPerPixel)
    photoShape = (photo.height, photo.width)
    photoRaster, projectionCheck = renderer.photoCamera(camera, photoShape)
    light = camera["rotation"].T@np.array([-.5, -.5, -1.])
    rendered, mask, photoStats = renderer.render(photoRaster, photoShape, light, perspective=True)
    if not mask.any():
        raise ValueError("Client mesh is not visible from the saved photo camera")
    args.output.mkdir(parents=True, exist_ok=True)
    photo.save(args.output/"sourcePhoto.png")
    writeImage(args.output/"photoCameraRender.png", rendered)
    writeImage(args.output/"photoCameraMask.png", mask)
    overlay = np.asarray(photo, np.float32)/255
    overlay[mask] = .5*overlay[mask]+.5*rendered[mask]
    writeImage(args.output/"photoCameraOverlay50.png", overlay)
    # Same display crop for photo/render/overlay; full original frames are saved.
    crop = metadata.get("cropBoxXyxy", [0, 0, photo.width, photo.height])
    panels = [photo.crop(crop), Image.fromarray(np.rint(rendered*255).astype(np.uint8)).crop(crop),
              Image.fromarray(np.rint(overlay*255).astype(np.uint8)).crop(crop)]
    labels = ["Reference photo", "Same saved camera / preset light", "50% overlay / preset light"]
    center = (asset["vertices"].min(0)+asset["vertices"].max(0))*.5
    radius = float(np.linalg.norm(asset["vertices"]-center, axis=-1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid client head framing radius")
    orthoExtent = radius*1.08
    fullHeadViews = []
    for name, azimuth in (("front", 0.), ("leftThreeQuarter", -40.), ("rightThreeQuarter", 40.)):
        rotation, translation = look_at_view_transform(dist=4*radius, elev=0., azim=azimuth,
                                                       at=(tuple(center),), device=args.device)
        cameras = FoVOrthographicCameras(device=args.device, R=rotation, T=translation,
                    min_x=-orthoExtent, max_x=orthoExtent, min_y=-orthoExtent, max_y=orthoExtent,
                    znear=.01*radius, zfar=10*radius)
        image, visible, stats = renderer.render(cameras, (args.size, args.size), [-.5, .6, 1.], perspective=False)
        if not visible.any():
            raise ValueError(f"Empty full-head raster: {name}")
        writeImage(args.output/f"head{name[0].upper()+name[1:]}.png", image)
        panels.append(Image.fromarray(np.rint(image*255).astype(np.uint8)))
        labels.append(f"Full head / {name} / preset light")
        fullHeadViews.append({"name": name, "azimuthDegrees": azimuth, "elevationDegrees": 0.,
                              "R": rotation.cpu().numpy(), "T": translation.cpu().numpy(), **stats})
    makeGrid(panels, labels, columns=3, panelSize=512,
             subtitle="Client head assembly | fixed-light previews; not a photo-error evaluation").save(args.output/"overview.png")
    report = {"status": "rendered", "asset": str(args.asset.resolve()), "source": str(args.source.resolve()),
        "device": args.device, "pytorch3dVersion": pytorch3d.__version__, "seconds": perf_counter()-start,
        "previewOnly": True, "photoErrorEvaluation": False, "exportedArraysModified": False,
        "assetDiagnostics": asset["diagnostics"], "sourceHashes": {name: fileHash(path) for name, path in sourcePaths.items()},
        "implementationSha256": fileHash(Path(__file__)),
        "photoCamera": {**camera, "kind": args.cameraKind, "viewIndex": args.viewIndex, "refitted": False,
                        "sourcePhotoSha256": metadata["sha256"], "imageSizeHw": list(photoShape),
                        "displayCropXyxy": crop, "projectionCheck": projectionCheck, **photoStats},
        "fullHeadCameras": {"center": center, "boundingSphereRadius": radius, "orthographicExtent": orthoExtent,
                            "imageSizeHw": [args.size, args.size], "views": fullHeadViews},
        "lightingPreset": {"shader": "linear Lambert + isotropic GGX/Smith/Schlick; geometry normals only",
                           "f0Dielectric": .04, "keyIntensity": 2.5, "ambient": .22,
                           "photoLightDirectionUff": light, "fullHeadLightDirectionUff": [-.5, .6, 1.],
                           "roughnessAndMetallic": "exported material presets, not estimated; roughness minimum .05",
                           "rgbDecode": "standard sRGB EOTF", "displayEncode": "standard sRGB OETF"},
        "alpha": {"mode": "binary MASK cutoff before first accepted depth selection; OPAQUE ignores alpha",
                  "facesPerPixel": args.facesPerPixel, "cullBackfaces": True, "supersampling": False},
        "limits": ["Fixed preview lighting is not matched to photo illumination; no photo metrics are reported.",
                   "No shadow maps, ambient occlusion, SSS, environment lighting or measured normal maps.",
                   "Alpha masks are binary and may alias; BLEND materials are not supported.",
                   "Textures are resized for preview; out-of-range UV corners clamp before interpolation (preview approximation).",
                   "Assembly/rig suitability and absence of geometric self-intersection are not certified by this preview."]}
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated()/1024**3
    writeJson(args.output/"renderReport.json", report)
    print(json.dumps({"status": "rendered", "output": str(args.output.resolve()), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
