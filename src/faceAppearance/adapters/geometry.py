"""Reusable, offline UVFaceFusion geometry adapter.

The adapter owns model lifetime. Upstream inference scripts are never imported.
Inputs already share one 512-pixel crop; output cameras refer to that crop and
remain in the VGGT world frame. A view transform follows the row-vector rule
``canonical = s * (world @ R) + t``. These initial transforms do not make the
fused mesh and cameras exactly consistent; downstream camera fitting is required.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from ..faceSelection import selectLargestFaceIndex


modelLoadLock = RLock()


def _fileSha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _writeDiagnosticStage(directory: Path, stage: str, arrays: dict[str, np.ndarray],
                          metadata: dict[str, Any]) -> None:
    """Write lossless numeric arrays and explicit provenance, never overwrite a stage.

    Files are deliberately uncompressed: writing a stage releases its CPU buffers
    before the next model is staged on the GPU. A partial run stays marked partial.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}.npz"
    manifestPath = directory / "manifest.json"
    manifest = json.loads(manifestPath.read_text()) if manifestPath.exists() else {
        "schemaVersion": 1, "status": "partial", "stages": {},
        "notes": ["VGGT depth is model evidence, not independent 3D ground truth.",
                  "Canonical geometry has uncalibrated units; it is not millimetres.",
                  "Native UFF UV coordinates index arrays with v increasing downwards."],
        "implementationHashes": {
            "geometry.py": _fileSha256(Path(__file__)),
            "uvFusion/inference.py": _fileSha256(Path(__file__).parents[1] / "vendor/uvFusion/inference.py"),
        },
    }
    numericArrays = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in numericArrays.values()):
        raise ValueError("Geometry diagnostic arrays must not contain Python objects")
    # Exclusive creation also prevents accidentally mixing a rerun into a cache.
    with path.open("xb") as stream:
        np.savez(stream, **numericArrays)
    manifest["stages"][stage] = {
        "file": path.name, "sha256": _fileSha256(path), "metadata": metadata,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "dataSha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()}
                   for name, value in numericArrays.items()},
    }
    temporaryPath = directory / "manifest.json.tmp"
    temporaryPath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporaryPath.replace(manifestPath)


def sampleDiagnosticPartials(uvPointMaps: np.ndarray, validMasks: np.ndarray,
                             vertexUvs: np.ndarray) -> dict[str, np.ndarray]:
    """CPU sampling of actual canonical network inputs, with explicit missing data.

    The conservative threshold matches upstream similarity fitting (>1 accumulated
    Gaussian weight). It is a support threshold, not VGGT confidence. The sampler
    uses the same 3x3, sigma=.8 kernel as reconstruction. Invalid XYZ is NaN.
    """
    import torch
    from ..vendor.uvFusion.meshSampling import sample_uv_pointmap_local_masked

    maps = np.asarray(uvPointMaps)
    masks = np.asarray(validMasks)
    uvs = np.asarray(vertexUvs)
    if maps.ndim != 4 or maps.shape[-1] != 3 or masks.shape != maps.shape[:3]:
        raise ValueError("Expected UV point maps (N,H,W,3) and matching valid masks")
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("vertexUvs must be (V,2)")
    vertices, supports, counts = [], [], []
    with torch.inference_mode():
        for pointMap, mask in zip(maps, masks):
            sampled, support, count = sample_uv_pointmap_local_masked(
                torch.as_tensor(np.ascontiguousarray(pointMap), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.bool),
                torch.as_tensor(np.ascontiguousarray(uvs), dtype=torch.float32),
                radius=1, sigma=0.8, min_valid_neighbors=1,
            )
            vertices.append(sampled.numpy())
            supports.append(support.numpy())
            counts.append(count.numpy())
    points = np.stack(vertices)
    weights, neighbors = np.stack(supports), np.stack(counts)
    valid = (weights > 1.0) & (neighbors >= 1) & np.isfinite(points).all(axis=-1)
    points[~valid] = np.nan
    return {"partialVertices": points, "partialValid": valid,
            "partialSampleWeight": weights, "partialNeighborCount": neighbors}


def resizeIntrinsics(cameraIntrinsics: np.ndarray, sourceSize: int = 518,
                     targetSize: int = 512) -> np.ndarray:
    """Apply OpenCV resize's half-pixel mapping to camera image coordinates.

    Pixel centres obey ``new = scale * (old + 0.5) - 0.5``. Keeping the
    translation distinguishes this from simply multiplying K's first two rows.
    """
    if sourceSize <= 0 or targetSize <= 0:
        raise ValueError("Image sizes must be positive")
    intrinsics = np.asarray(cameraIntrinsics, dtype=np.float32)
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("cameraIntrinsics must end in shape (3, 3)")
    scale = targetSize / sourceSize
    imageTransform = np.array([[scale, 0, (scale - 1) / 2],
                               [0, scale, (scale - 1) / 2],
                               [0, 0, 1]], dtype=np.float32)
    return imageTransform @ intrinsics


def validateImages(images: np.ndarray) -> np.ndarray:
    """Reject ambiguous layouts, integer RGB and nonfinite/out-of-range values."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (512, 512, 3):
        raise ValueError("Expected RGB images shaped (N, 512, 512, 3)")
    if not 4 <= len(images) <= 16:
        raise ValueError("Geometry reconstruction requires 4 to 16 views")
    if not np.issubdtype(images.dtype, np.floating):
        raise ValueError("Expected floating RGB values in [0, 1], not uint8")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must contain finite RGB values in [0, 1]")
    return np.ascontiguousarray(images, dtype=np.float32)


@contextmanager
def sourcePath(sourceRoot: Path) -> Iterator[None]:
    """Temporarily expose an explicitly selected upstream source directory."""
    path = str(sourceRoot)
    alreadyPresent = path in sys.path
    if not alreadyPresent:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if not alreadyPresent:
            sys.path.remove(path)


def buildDinoOffline(modelName: str, **unusedOptions: Any) -> tuple[Any, Any]:
    """Construct the backbone without fetching weights; uv.ckpt supplies them."""
    import timm
    from torchvision.transforms import Normalize

    model = timm.create_model(modelName, pretrained=False, dynamic_img_size=True)
    config = timm.data.resolve_model_data_config(model)
    return model, Normalize(mean=config["mean"], std=config["std"])


class GeometryAdapter:
    """Load once, then reconstruct repeatedly with explicit GPU staging.

    ``staged`` keeps all model weights in CPU memory and moves just the active
    stage to the selected device. ``resident`` retains models on that device.
    A lock serializes inference because staged device movement is mutable state.

    Pixel3DMM is the only separately configured source tree. Set
    FACE_APPEARANCE_PIXEL3DMM_ROOT (or PIXEL3DMM_CODE_BASE) to its checkout, or
    supply ``pixel3dmmRoot``. By default the project's thirdParty/pixel3dmm
    checkout is used. There is no dependency on FaceReconstruction/v2.
    """

    def __init__(self, projectRoot: Path, device: str = "cuda",
                 memoryPolicy: str = "staged", *,
                 pixel3dmmRoot: Path | None = None, depthChunkSize: int = 1) -> None:
        if memoryPolicy not in {"staged", "resident"}:
            raise ValueError("memoryPolicy must be 'staged' or 'resident'")
        self.projectRoot = Path(projectRoot).resolve()
        self.device = str(device)
        self.memoryPolicy = memoryPolicy
        self.pixel3dmmRoot = pixel3dmmRoot
        if depthChunkSize < 1:
            raise ValueError("depthChunkSize must be positive")
        self.depthChunkSize = depthChunkSize
        self.models: dict[str, Any] = {}
        self.isLoaded = False
        self.diagnosticModelsLoaded = False
        self.lock = RLock()
        self.canonicalVertices: np.ndarray | None = None
        self.canonicalFaces: np.ndarray | None = None
        self.canonicalUvs: np.ndarray | None = None
        self.templateMask: np.ndarray | None = None
        self.rasterContext: Any = None

    def assetPaths(self) -> dict[str, Path]:
        root = self.projectRoot / "01_UVFaceFusion"
        weights = root / "weights"
        return {
            "vggt": weights / "vggt_weights.pt",
            "uvPredictor": weights / "uv.ckpt",
            "fusion": weights / "fusion_net.pth",
            # This local collection contains ResNet50, not MobileNet RetinaFace.
            "faceDetector": weights / "facer_hub_checkpoints/Resnet50_Final.pth",
            "faceParser": weights / "facer_hub_checkpoints/face_parsing.farl.celebm.main_ema_181500_jit.pt",
            "mesh": root / "uvfacefusion/assets/canonical_mesh_flame.ply",
            "uvs": root / "uvfacefusion/assets/canonical_mesh_flame_uvs.npy",
        }

    def loadModels(self) -> GeometryAdapter:
        """Load local weights exactly once; missing assets fail before inference."""
        with self.lock, modelLoadLock:
            if self.isLoaded:
                return self
            paths = self.assetPaths()
            missingPaths = [str(path) for path in paths.values() if not path.is_file()]
            if missingPaths:
                raise FileNotFoundError("Missing geometry assets:\n" + "\n".join(missingPaths))
            import torch

            targetDevice = torch.device(self.device)
            if targetDevice.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; full geometry inference requires a CUDA device")
            if not self.diagnosticModelsLoaded:
                self.loadTemplate(paths)
            previousModels = dict(self.models)
            try:
                self.models["uvPredictor"] = self.loadUvPredictor(paths["uvPredictor"])
                self.models["faceDetector"], self.models["faceParser"] = self.loadFaceModels(paths)
                if "vggt" not in self.models:
                    self.models["vggt"] = self.loadVggt(paths["vggt"])
                if "fusion" not in self.models:
                    self.models["fusion"] = self.loadFusion(paths["fusion"])
                for model in self.models.values():
                    model.eval().requires_grad_(False)
                    if self.memoryPolicy == "resident":
                        model.to(targetDevice)
                self.isLoaded = True
                self.diagnosticModelsLoaded = True
            except Exception:
                self.models = previousModels
                raise
        return self

    def loadDiagnosticModels(self) -> GeometryAdapter:
        """Load only canonical template, VGGT and fusion for cached-UV experiments.

        This does not authorize full reconstruct(): UV prediction and FACER are
        intentionally absent. A subsequent loadModels() fills missing models
        without reloading these two checkpoints.
        """
        with self.lock, modelLoadLock:
            if self.diagnosticModelsLoaded:
                return self
            paths = self.assetPaths()
            requiredNames = ("mesh", "uvs", "vggt", "fusion")
            missing = [str(paths[name]) for name in requiredNames if not paths[name].is_file()]
            if missing:
                raise FileNotFoundError("Missing diagnostic geometry assets:\n" + "\n".join(missing))
            import torch

            targetDevice = torch.device(self.device)
            if targetDevice.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; geometry inference requires a CUDA device")
            self.loadTemplate(paths)
            previousModels = dict(self.models)
            try:
                for name, loader in (("vggt", self.loadVggt), ("fusion", self.loadFusion)):
                    if name not in self.models:
                        self.models[name] = loader(paths[name])
                    self.models[name].eval().requires_grad_(False)
                    if self.memoryPolicy == "resident":
                        self.models[name].to(targetDevice)
                self.diagnosticModelsLoaded = True
            except Exception:
                self.models = previousModels
                raise
        return self

    def loadTemplate(self, paths: dict[str, Path]) -> None:
        import trimesh

        mesh = trimesh.load(str(paths["mesh"]), process=False, maintain_order=True)
        self.canonicalVertices = np.array(mesh.vertices, dtype=np.float32, copy=True)
        self.canonicalFaces = np.array(mesh.faces, dtype=np.int64, copy=True)
        self.canonicalUvs = np.load(paths["uvs"]).astype(np.float32)
        if self.canonicalUvs.shape != (len(self.canonicalVertices), 2):
            raise ValueError("Canonical mesh and per-vertex UV asset do not match")
        if self.device.startswith("cuda"):
            import nvdiffrast.torch as dr
            from ..vendor.uvFusion.inference import build_template_uv_valid_mask

            self.rasterContext = dr.RasterizeCudaContext(device=self.device)
            mask = build_template_uv_valid_mask(
                self.canonicalVertices, self.canonicalFaces, self.canonicalUvs,
                uv_size=512, device=self.device, ctx=self.rasterContext,
            )
            self.templateMask = mask.cpu().numpy()
        else:
            raise RuntimeError("Full geometry reconstruction requires CUDA for the upstream UV rasterizer")

    def resolvePixel3dmmRoot(self) -> Path:
        configuredRoot = self.pixel3dmmRoot or os.environ.get("FACE_APPEARANCE_PIXEL3DMM_ROOT")
        if configuredRoot is None:
            for localName in ("thirdParty/pixel3dmm", "third_party/pixel3dmm"):
                localRoot = self.projectRoot / localName
                if localRoot.is_dir():
                    configuredRoot = localRoot
                    break
        if configuredRoot is None:
            configuredRoot = os.environ.get("PIXEL3DMM_CODE_BASE")
        if configuredRoot is None:
            raise FileNotFoundError("Set FACE_APPEARANCE_PIXEL3DMM_ROOT to the Pixel3DMM source checkout")
        root = Path(configuredRoot).expanduser().resolve()
        if not (root / "src/pixel3dmm/lightning/p3dmm_network.py").is_file():
            raise FileNotFoundError(f"Pixel3DMM source checkout is invalid: {root}")
        return root

    def loadUvPredictor(self, checkpointPath: Path) -> Any:
        """Restore Network only, excluding Lightning's training-only assets.

        The released checkpoint includes the DINO backbone. Temporarily replace
        its pretrained constructor to avoid a redundant Hugging Face download,
        then require every network tensor to load strictly from the local file.
        """
        import torch

        root = self.resolvePixel3dmmRoot()
        with sourcePath(root / "src"):
            networkModule = importlib.import_module("pixel3dmm.lightning.p3dmm_network")
        checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
        config = checkpoint["hyper_parameters"]["cfg"]
        if config.model.feature_map_type != "DINO":
            raise ValueError("This offline adapter supports the released DINO UV checkpoint only")
        with patch.object(networkModule.DinoWrapper, "_build_dino", staticmethod(buildDinoOffline)):
            model = networkModule.Network(config)
        state = {key.removeprefix("net."): value for key, value in checkpoint["state_dict"].items()
                 if key.startswith("net.")}
        if not state:
            raise ValueError("Pixel3DMM checkpoint contains no net.* weights")
        model.load_state_dict(state, strict=True)
        return model.cpu()

    def loadFaceModels(self, paths: dict[str, Path]) -> tuple[Any, Any]:
        localRoot = self.projectRoot / "thirdParty/facer"
        with sourcePath(localRoot):
            facer = importlib.import_module("facer")

        detector = facer.face_detector("retinaface/resnet50", device="cpu",
                                       model_path=str(paths["faceDetector"]))
        parser = facer.face_parser("farl/celebm/448", device="cpu",
                                   model_path=str(paths["faceParser"]))
        return detector, parser

    def loadVggt(self, checkpointPath: Path) -> Any:
        import torch

        vendorRoot = self.projectRoot / "01_UVFaceFusion/uvfacefusion/third_party"
        with sourcePath(vendorRoot):
            modelClass = importlib.import_module("vggt.models.vggt").VGGT
        model = modelClass()
        model.load_state_dict(torch.load(checkpointPath, map_location="cpu", weights_only=True), strict=True)
        return model

    def loadFusion(self, checkpointPath: Path) -> Any:
        import torch
        from ..vendor.uvFusion.models import MultiViewFusionCompletionNet

        model = MultiViewFusionCompletionNet(
            in_channels=3, out_channels=3, base_channels=48,
            use_mask=False, norm_type="gn", num_groups=8, use_output_residual=True,
        )
        checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        return model

    @contextmanager
    def activateModels(self, *modelNames: str) -> Iterator[None]:
        """Release the active stage even when inference raises an exception."""
        import torch

        activeModels = [self.models[name] for name in modelNames]
        try:
            for model in activeModels:
                model.to(self.device)
            yield
        finally:
            if self.memoryPolicy == "staged":
                for model in activeModels:
                    model.to("cpu")
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    def predictUvs(self, images: np.ndarray) -> np.ndarray:
        import torch

        uvMaps = []
        with self.activateModels("uvPredictor"), torch.inference_mode():
            for image in images:
                rgb = torch.from_numpy(image).to(self.device)[None, None]
                output, _ = self.models["uvPredictor"]({"tar_rgb": rgb})
                mirroredOutput, _ = self.models["uvPredictor"]({"tar_rgb": torch.flip(rgb, dims=[3])})
                mirroredUv = torch.flip(mirroredOutput["uv_map"], dims=[4])
                mirroredUv[:, :, 0] = -mirroredUv[:, :, 0] + 0.015
                uv = torch.clamp(((output["uv_map"] + mirroredUv) / 2 + 1) / 2, 0, 1)
                uvMaps.append(uv[0, 0].permute(1, 2, 0).cpu().numpy())
        return np.stack(uvMaps).astype(np.float32)

    def predictFaceMasks(self, images: np.ndarray) -> np.ndarray:
        """Parse only the largest detected face in each working image."""
        import torch

        excludedLabels = {0, 1, 3, 4, 5, 8, 9, 14, 16, 17, 18}
        masks = []
        with self.activateModels("faceDetector", "faceParser"), torch.inference_mode():
            for viewIndex, image in enumerate(images):
                rgb = torch.from_numpy(np.rint(image * 255).astype(np.uint8)).permute(2, 0, 1)[None].to(self.device)
                faces = self.models["faceDetector"](rgb)
                selected = selectLargestFaceIndex(faces["rects"].detach().cpu().numpy(),
                                                  faces["scores"].detach().cpu().numpy())
                selectedFaces = {**faces, **{key: faces[key][selected:selected + 1]
                                            for key in ("rects", "points", "scores", "image_ids")}}
                parsedFaces = self.models["faceParser"](rgb, selectedFaces)
                probabilities = parsedFaces["seg"]["logits"].softmax(dim=1)[0]
                includedLabels = [index for index in range(len(probabilities)) if index not in excludedLabels]
                masks.append(probabilities[includedLabels].sum(dim=0).cpu().numpy())
        # Summing softmax channels can exceed one by float32 rounding.
        return np.clip(np.stack(masks), 0, 1).astype(np.float32)

    def detectInputFace(self, image: np.ndarray) -> dict:
        """Locate the largest face before cropping with the loaded detector.

        Return coordinates in the EXIF-oriented original image. Bounding the
        detector input size limits temporary memory for high-resolution photos;
        the crop itself is always sampled from the original pixels. Area takes
        priority over confidence, and selection metadata retains its provenance.
        """
        import cv2
        import torch

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Face detection expects original RGB uint8 pixels")
        if not self.isLoaded:
            raise RuntimeError("Call loadModels() before detecting input faces")
        height, width = image.shape[:2]
        scale = min(1.0, 1024 / max(height, width))
        resizedSize = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(image, resizedSize, interpolation=cv2.INTER_AREA) if scale < 1 else image
        with self.lock, self.activateModels("faceDetector"), torch.inference_mode():
            rgb = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1)[None].to(self.device)
            faces = self.models["faceDetector"](rgb)
            count = len(faces["scores"])
            selected = selectLargestFaceIndex(faces["rects"].detach().cpu().numpy(),
                                              faces["scores"].detach().cpu().numpy())
            box = faces["rects"][selected].float().cpu().numpy()
            landmarks = faces["points"][selected].float().cpu().numpy()
            score = float(faces["scores"][selected].cpu())
        xyScale = np.array([resizedSize[0] / width, resizedSize[1] / height])
        return {"boxXyxy": (box / np.tile(xyScale, 2)).tolist(),
                "landmarks": ((landmarks + 0.5) / xyScale - 0.5).tolist(), "score": score,
                "selection": {"strategy": "largestArea", "detectedCount": count, "selectedIndex": selected}}

    def predictPointMaps(self, images: np.ndarray, *,
                         diagnosticsDirectory: Path | None = None) -> dict[str, np.ndarray]:
        import cv2
        import torch

        # The package is already imported by loadVggt, and has its own __path__.
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        resizedImages = np.stack([cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
                                  for image in images])
        with self.activateModels("vggt"), torch.inference_mode():
            rgb = torch.from_numpy(resizedImages).permute(0, 3, 1, 2)[None].to(self.device)
            if self.device.startswith("cuda"):
                capability = torch.cuda.get_device_capability(torch.device(self.device))[0]
                dtype = torch.bfloat16 if capability >= 8 else torch.float16
                precisionContext = torch.autocast(device_type="cuda", dtype=dtype)
            else:
                precisionContext = nullcontext()
            model = self.models["vggt"]
            with precisionContext:
                tokens, patchStart = model.aggregator(rgb)
            pose = model.camera_head(tokens)[-1]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, rgb.shape[-2:])
            # Dense DPT prediction is independent per view after joint VGGT
            # aggregation. Chunking keeps 4–16 views within the 24 GiB target.
            depths, depthConfidence = model.depth_head(tokens, rgb, patchStart,
                                                       frames_chunk_size=self.depthChunkSize)
            pointMaps = unproject_depth_map_to_point_map(depths[0], extrinsics[0], intrinsics[0])
            intrinsics = intrinsics[0].float().cpu().numpy()
            extrinsics = extrinsics[0].float().cpu().numpy()
            if diagnosticsDirectory is not None:
                # DPT is outside aggregator autocast and yields float32. Keep
                # the returned layouts: depth N,H,W,1; confidence N,H,W.
                diagnosticDepths = depths[0].detach().cpu().numpy()
                diagnosticConfidence = depthConfidence[0].detach().cpu().numpy()
            # Free the large token pyramid before moving another stage to CUDA.
            del tokens, rgb, depths, depthConfidence, pose
        predictions = {
            "pointMaps": np.stack([cv2.resize(pointMap, (512, 512), interpolation=cv2.INTER_LINEAR)
                                    for pointMap in pointMaps]).astype(np.float32),
            "cameraIntrinsics": resizeIntrinsics(intrinsics),
            "cameraExtrinsics": extrinsics,
        }
        if diagnosticsDirectory is not None:
            confidence512 = np.stack([
                cv2.resize(confidence, (512, 512), interpolation=cv2.INTER_LINEAR)
                for confidence in diagnosticConfidence
            ]).astype(np.float32)
            _writeDiagnosticStage(diagnosticsDirectory, "vggt", {
                "depthNative518": diagnosticDepths,
                "depthConfidenceNative518": diagnosticConfidence,
                "depthConfidence512": confidence512,
                "pointMapsNative518": pointMaps,
                "pointMaps512": predictions["pointMaps"],
                "cameraIntrinsicsNative518": intrinsics,
                "cameraIntrinsics512": predictions["cameraIntrinsics"],
                "cameraExtrinsics": extrinsics,
            }, {"pointMapSource": "unproject_depth_map_to_point_map(depth_head, camera_head)",
                "notAvailable": ["VGGT direct point_head output (not used by this pipeline)"],
                "depthConfidence": "Actual depth_head confidence; uncalibrated, not a probability; unused by reconstruction",
                "resize": "OpenCV INTER_LINEAR, 518 to 512, intrinsics use half-pixel resize mapping",
                "depthChunkSize": self.depthChunkSize})
            predictions["depthConfidenceMaps"] = confidence512
        return predictions

    def fuseGeometry(self, uvMaps: np.ndarray, faceMasks: np.ndarray,
                     pointMaps: np.ndarray, *, diagnosticsDirectory: Path | None = None,
                     depthConfidenceMaps: np.ndarray | None = None) -> dict[str, Any]:
        import torch
        from ..vendor.uvFusion.inference import infer_uv_point_map_with_model
        from ..vendor.uvFusion.meshSampling import sample_uv_pointmap_local_masked
        from ..vendor.uvFusion.utilities import masked_gaussian_blur

        with self.activateModels("fusion"), torch.inference_mode():
            diagnosticOptions = {"capture_input_fusion": True} if diagnosticsDirectory is not None else {}
            result = infer_uv_point_map_with_model(
                self.models["fusion"], uvMaps, faceMasks, pointMaps,
                self.canonicalUvs, self.canonicalVertices,
                template_valid_mask=self.templateMask, device=self.device,
                uv_size=512, random_sample_views=False,
                partial_mesh_radius=1, partial_mesh_sigma=0.8,
                partial_mesh_min_valid_neighbors=1, partial_mesh_valid_thresh=1.0,
                **diagnosticOptions,
            )
            pointMap = masked_gaussian_blur(result["pred_uv_point_map"], self.templateMask, ksize=3, sigma=0)
            vertices, finalSampleWeight, finalNeighborCount = sample_uv_pointmap_local_masked(
                torch.from_numpy(pointMap).to(self.device),
                torch.from_numpy(self.templateMask).to(self.device),
                torch.from_numpy(self.canonicalUvs).to(self.device), radius=1, sigma=0.8,
            )
            vertices = vertices.cpu().numpy().astype(np.float32)
            if diagnosticsDirectory is not None:
                finalSampleWeight = finalSampleWeight.cpu().numpy()
                finalNeighborCount = finalNeighborCount.cpu().numpy()
        transforms = [{"s": float(transform["c"]),
                       "R": np.asarray(transform["R"], dtype=np.float32),
                       "t": np.asarray(transform["t"], dtype=np.float32),
                       "validVertexMask": np.asarray(transform["vertices_mask"], dtype=bool)}
                      for transform in result["transforms"]]
        if not np.isfinite(vertices).all():
            raise RuntimeError("UVFaceFusion produced nonfinite geometry")
        if diagnosticsDirectory is not None:
            diagnosticArrays = {
                "inputUvPointMaps": result["input_uv_point_maps"],
                "inputValidMasks": result["input_valid_masks"],
                "inputSupportMask": result["input_support_mask"],
                "selectedViewIndices": np.asarray(result["selected_view_indices"], dtype=np.int64),
                "modelInputMean": result["model_input_mean"],
                "modelInputCoverage": result["model_input_coverage"],
                "predUvPointMap": result["pred_uv_point_map"],
                "predValidMask": result["pred_valid_mask"],
                "blurredUvPointMap": pointMap,
                "vertices": vertices, "faces": self.canonicalFaces, "uvs": self.canonicalUvs,
                "canonicalVertices": self.canonicalVertices, "templateMask": self.templateMask,
                "finalSampleWeight": finalSampleWeight, "finalNeighborCount": finalNeighborCount,
                "viewScales": np.asarray([value["c"] for value in result["transforms"]]),
                "viewRotations": np.stack([value["R"] for value in result["transforms"]]),
                "viewTranslations": np.stack([value["t"] for value in result["transforms"]]),
                "viewValidVertexMasks": np.stack([value["vertices_mask"] for value in result["transforms"]]),
            }
            diagnosticArrays.update(sampleDiagnosticPartials(
                result["input_uv_point_maps"], result["input_valid_masks"], self.canonicalUvs))
            confidenceDescription = "Unavailable: depthConfidenceMaps not supplied"
            if depthConfidenceMaps is not None:
                diagnosticArrays.update(self.projectDiagnosticConfidence(
                    uvMaps, faceMasks, depthConfidenceMaps, result["input_valid_masks"],
                    result["selected_view_indices"], diagnosticArrays["partialValid"]))
                confidenceDescription = "Raw depth_head scores, resized to512 and UV-splatted with binary face mask; not used by fusion"
            _writeDiagnosticStage(diagnosticsDirectory, "fusion", diagnosticArrays, {
                "canonicalInputStage": "Per-view similarity, template-mask multiplication, then coordinate clip [-1,1]",
                "viewTransformConvention": "canonical = scale * (world @ R) + t; stored at vendor-return precision",
                "modelInputMean": "model.compute_input_fusion(x * mask, mask), mask clamped [0,1]; actual residual base",
                "blur": {"method": "masked_gaussian_blur", "ksize": 3, "sigma": 0},
                "sampling": {"method": "sample_uv_pointmap_local_masked", "radius": 1, "sigma": 0.8,
                             "partialValid": "sampleWeight > 1 and neighborCount >= 1 and finite XYZ; invalid XYZ is NaN",
                             "finalVertices": "Original default sampler thresholds, unchanged"},
                "confidence": confidenceDescription,
                "notAvailable": ["Per-view UV XYZ before canonicalization/clip; can be recomputed from saved inputs and pointMaps512"],
            })
        return {"vertices": vertices, "faces": self.canonicalFaces.copy(),
                "uvs": self.canonicalUvs.copy(), "viewTransforms": transforms,
                "supportMask": np.asarray(result["input_support_mask"], dtype=bool)}

    def projectDiagnosticConfidence(self, uvMaps: np.ndarray, faceMasks: np.ndarray,
                                    depthConfidenceMaps: np.ndarray, validUvMasks: np.ndarray,
                                    selectedViews: list[int], partialValid: np.ndarray) -> dict[str, np.ndarray]:
        """CPU-only projection of uncalibrated confidence; never changes inference."""
        import torch
        from ..vendor.uvFusion.inference import fuse_uv_single_view

        confidence = np.asarray(depthConfidenceMaps)
        if confidence.shape != np.asarray(faceMasks).shape or not np.isfinite(confidence).all():
            raise ValueError("depthConfidenceMaps must be finite and shaped like faceMasks")
        uvScores, splatWeights = [], []
        with torch.inference_mode():
            for index, view in enumerate(selectedViews):
                channels = np.repeat(confidence[view, ..., None], 3, axis=-1)
                scores, weights = fuse_uv_single_view(
                    torch.as_tensor(np.ascontiguousarray(uvMaps[view]), dtype=torch.float32),
                    torch.as_tensor(channels, dtype=torch.float32),
                    mask=torch.as_tensor(np.asarray(faceMasks[view]) > 0.5),
                    S=validUvMasks.shape[-1],
                )
                scoreMap = scores.numpy()
                scoreMap[~np.asarray(validUvMasks[index], dtype=bool)] = 0
                uvScores.append(scoreMap)
                splatWeights.append(weights.numpy())
        scores = np.stack(uvScores)
        sampled = sampleDiagnosticPartials(scores, validUvMasks, self.canonicalUvs)
        partialScores = sampled["partialVertices"][..., 0]
        partialScores[~partialValid] = np.nan
        return {"inputDepthConfidenceUv": scores[..., 0],
                "depthConfidenceSplatWeight": np.stack(splatWeights),
                "partialDepthConfidence": partialScores}

    def _reconstructFromUvs(self, images: np.ndarray, uvMaps: np.ndarray, faceMasks: np.ndarray,
                            diagnosticsDirectory: Path | None, *, inputSource: str) -> dict[str, Any]:
        if diagnosticsDirectory is not None:
            _writeDiagnosticStage(diagnosticsDirectory, "inputs", {
                "images512": images, "uvMaps": uvMaps, "faceMasks": faceMasks,
                "canonicalVertices": self.canonicalVertices, "canonicalFaces": self.canonicalFaces,
                "canonicalUvs": self.canonicalUvs, "templateMask": self.templateMask,
            }, {"inputSource": inputSource,
                "faceMasks": "Effective masks already include observed-image/padding exclusions; used unchanged",
                "assetPaths": {name: str(path) for name, path in self.assetPaths().items()
                               if name in {"vggt", "fusion", "mesh", "uvs"}}})
            predictions = self.predictPointMaps(images, diagnosticsDirectory=diagnosticsDirectory)
            result = self.fuseGeometry(uvMaps, faceMasks, predictions["pointMaps"],
                                       diagnosticsDirectory=diagnosticsDirectory,
                                       depthConfidenceMaps=predictions.get("depthConfidenceMaps"))
            manifestPath = Path(diagnosticsDirectory) / "manifest.json"
            manifest = json.loads(manifestPath.read_text())
            manifest["status"] = "complete"
            manifestPath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        else:
            # No additional keyword calls in the default path: retain adapter
            # subclasses used by clients and the original inference computation.
            predictions = self.predictPointMaps(images)
            result = self.fuseGeometry(uvMaps, faceMasks, predictions["pointMaps"])
        result.update(uvMaps=uvMaps, faceMasks=faceMasks,
                      cameraIntrinsics=predictions["cameraIntrinsics"],
                      cameraExtrinsics=predictions["cameraExtrinsics"])
        return result

    def reconstructFromCachedUvs(self, images: np.ndarray, uvMaps: np.ndarray, faceMasks: np.ndarray,
                                 *, diagnosticsDirectory: Path | None = None) -> dict[str, Any]:
        """Replay VGGT + fusion with cached 512px UV maps and effective face masks.

        The images must be exactly the original preprocessed crops, in the same
        view order. Caller supplies provenance tying them to the cache. Masks must
        already exclude padded/unobserved pixels, just as in reconstruct's return.
        No UV predictor, detector or face parser is called. loadDiagnosticModels()
        is sufficient. Output keys match reconstruct(); caches are optional.
        """
        images = validateImages(images)
        uvs, masks = np.asarray(uvMaps), np.asarray(faceMasks)
        if uvs.shape != (*images.shape[:3], 2) or not np.issubdtype(uvs.dtype, np.floating):
            raise ValueError("Cached uvMaps must be floating (N,512,512,2)")
        if masks.shape != images.shape[:3] or not (np.issubdtype(masks.dtype, np.floating) or masks.dtype == np.bool_):
            raise ValueError("Cached faceMasks must be floating or boolean (N,512,512)")
        if not np.isfinite(uvs).all() or uvs.min() < 0 or uvs.max() > 1:
            raise ValueError("Cached uvMaps must be finite in [0,1]")
        if not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
            raise ValueError("Cached faceMasks must be finite in [0,1]")
        if not (masks > 0.5).any(axis=(1, 2)).all():
            raise ValueError("Every cached view must have face-mask support above 0.5")
        with self.lock:
            if not self.diagnosticModelsLoaded:
                raise RuntimeError("Call loadDiagnosticModels() or loadModels() before cached-UV reconstruction")
            return self._reconstructFromUvs(images, uvs, masks, diagnosticsDirectory,
                                            inputSource="caller-supplied cached UVs and effective face masks")

    def reconstruct(self, images: np.ndarray, *, validImageMasks: np.ndarray | None = None,
                    diagnosticsDirectory: Path | None = None) -> dict[str, Any]:
        """Reconstruct canonical geometry; call loadModels during service startup."""
        images = validateImages(images)
        if validImageMasks is not None:
            validImageMasks = np.asarray(validImageMasks)
            if validImageMasks.shape != images.shape[:3] or validImageMasks.dtype != np.bool_:
                raise ValueError("validImageMasks must be boolean (N, 512, 512) observation masks")
            if not validImageMasks.any(axis=(1, 2)).all():
                raise ValueError("Every view must have observed image pixels")
        with self.lock:
            if not self.isLoaded:
                raise RuntimeError("Call loadModels() before reconstruct()")
            uvMaps = self.predictUvs(images)
            faceMasks = self.predictFaceMasks(images)
            if validImageMasks is not None:
                # Edge replication only helps network input boundaries. It
                # must never supply 3D samples or downstream texture evidence.
                faceMasks *= validImageMasks
            return self._reconstructFromUvs(images, uvMaps, faceMasks, diagnosticsDirectory,
                                            inputSource="UV predictor and FACER from this reconstruction")
