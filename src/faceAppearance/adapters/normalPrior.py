"""Independent offline Pixel3DMM normal prior with explicit FLAME axes.

Only the released normal Network is loaded. No UV predictor, fusion network,
segmentation, FLAME model or upstream inference script is imported. Predictions
are screen-space samples of FLAME-frame vectors, not camera-frame normals.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from unittest.mock import patch

import numpy as np

from .geometry import GeometryAdapter, buildDinoOffline, modelLoadLock, sourcePath


maximumViews = 16
imageResolution = 512
minimumNormalLength = 1e-6


def _fileHash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validateNormalImages(images):
    """Accept 1–16 already cropped, display-encoded float RGB images in [0,1]."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (imageResolution, imageResolution, 3):
        raise ValueError("Expected RGB images shaped (N, 512, 512, 3)")
    if not 1 <= len(images) <= maximumViews:
        raise ValueError("Normal prediction requires 1 to 16 images per call")
    if images.dtype.kind != "f":
        raise ValueError("Expected floating RGB values in [0,1]")
    # Avoid allocating a batch-sized validation mask or a float32 batch copy.
    for image in images:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Images must contain finite RGB values in [0,1]")
    return images


def _option(config, name, default=None):
    return config.get(name, default) if hasattr(config, "get") else getattr(config, name, default)


def _removeAbsentDisabledUvEncoding(model, modelConfig, state):
    """Match the old normal checkpoint without ignoring any active weights.

    Current upstream creates these modules unconditionally, but calls them only
    when use_uv_enc is true. The released normal checkpoint predates them. Only
    the complete, known six-tensor branch may be removed, and only when disabled
    and wholly absent from the checkpoint. Partial branches still fail strict
    loading, as do all unrelated missing or unexpected tensors.
    """
    moduleNames = ("dir_norm_uv", "uv_enc_mlp")
    prefixes = tuple(name + "." for name in moduleNames)
    expectedKeys = {"dir_norm_uv.norm.weight", "dir_norm_uv.norm.bias",
                    "dir_norm_uv.mlp.1.weight", "dir_norm_uv.mlp.1.bias",
                    "uv_enc_mlp.1.weight", "uv_enc_mlp.1.bias"}
    if _option(modelConfig, "use_uv_enc", False) or any(name.startswith(prefixes) for name in state):
        return []
    branchKeys = {name for name in model.state_dict() if name.startswith(prefixes)}
    if branchKeys != expectedKeys:
        return []
    for name in moduleNames:
        delattr(model, name)
    return list(moduleNames)


class NormalPriorAdapter:
    """Preload one normal model; process one view at a time with flip averaging.

    ``staged`` always preloads onto CPU, even when the inference device is CUDA.
    It moves the model for ``predict`` and restores CPU residency on success or
    failure. ``resident`` retains the model on the requested device. No returned
    image arrays are cached inside the adapter. The maximum returned cache is
    16*(512*512)*(3*4+1) bytes = 52 MiB; activations and model storage are extra.
    """

    def __init__(self, projectRoot, device="cuda", checkpointPath=None, memoryPolicy="staged"):
        if memoryPolicy not in {"staged", "resident"}:
            raise ValueError("memoryPolicy must be 'staged' or 'resident'")
        self.projectRoot = Path(projectRoot).expanduser().resolve()
        self.device = str(device)
        self.memoryPolicy = memoryPolicy
        path = Path(checkpointPath).expanduser() if checkpointPath is not None else Path("assets/weights/pixel3dmm/normals.ckpt")
        self.checkpointPath = (path if path.is_absolute() else self.projectRoot / path).resolve()
        self.model = None
        self.isLoaded = False
        self.lock = RLock()
        self.provenance = {}
        self.loadSeconds = 0.

    def assetPaths(self):
        return {"normalPredictor": self.checkpointPath}

    def resolvePixel3dmmRoot(self):
        # This only resolves paths; it never loads any GeometryAdapter model.
        return GeometryAdapter(self.projectRoot, device="cpu").resolvePixel3dmmRoot()

    def _checkpointProvenance(self):
        path = self.checkpointPath
        if not path.is_file():
            raise FileNotFoundError(f"Missing normal prior checkpoint: {path}")
        digest = _fileHash(path)
        result = {"checkpointPath": str(path), "checkpointSha256": digest,
                  "checkpointBytes": path.stat().st_size, "manifestVerified": False}
        manifestPath = path.parent / "manifest.json"
        if manifestPath.is_file():
            manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
            checkpoint = manifest.get("checkpoint", {})
            if checkpoint.get("path") == path.name:
                if checkpoint.get("sha256") != digest or checkpoint.get("sizeBytes") != path.stat().st_size:
                    raise ValueError("Normal prior checkpoint differs from its provenance manifest")
                result.update(manifestVerified=True, manifestPath=str(manifestPath),
                              manifestSha256=_fileHash(manifestPath), checkpointSource=deepcopy(checkpoint),
                              upstream=deepcopy(manifest.get("upstream", {})))
        return result

    def loadNormalPredictor(self, checkpointPath):
        """Restore the complete net.* state strictly, with no backbone download."""
        import torch

        root = self.resolvePixel3dmmRoot()
        with sourcePath(root / "src"):
            networkModule = importlib.import_module("pixel3dmm.lightning.p3dmm_network")
        expectedSource = (root / "src/pixel3dmm/lightning/p3dmm_network.py").resolve()
        actualSource = Path(networkModule.__file__).resolve()
        if actualSource != expectedSource:
            raise RuntimeError("An already imported Pixel3DMM Network uses a different source checkout")
        # Official Lightning checkpoints contain OmegaConf metadata. Verify the
        # file manifest before this trusted-local checkpoint deserialization.
        checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
        config = checkpoint["hyper_parameters"]["cfg"]
        modelConfig = _option(config, "model")
        predictionTypes = _option(modelConfig, "prediction_type")
        predictionTypes = predictionTypes.split(",") if isinstance(predictionTypes, str) else list(predictionTypes or [])
        if (_option(modelConfig, "feature_map_type") != "DINO" or predictionTypes != ["normals"]
                or _option(modelConfig, "pred_disentangled", False)):
            raise ValueError("Expected the released independent DINO normals checkpoint without disentangled outputs")
        state = {name.removeprefix("net."): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith("net.")}
        if not state:
            raise ValueError("Normal prior checkpoint contains no net.* weights")
        with patch.object(networkModule.DinoWrapper, "_build_dino", staticmethod(buildDinoOffline)):
            model = networkModule.Network(config)
        removedModules = _removeAbsentDisabledUvEncoding(model, modelConfig, state)
        model.load_state_dict(state, strict=True)
        self.provenance.update(pixel3dmmRoot=str(root), networkSourceSha256=_fileHash(expectedSource),
                               networkTensorCount=len(state), featureMapType="DINO", predictionType="normals",
                               strictNetworkState=True, pretrainedBackboneDownload=False,
                               removedAbsentDisabledModules=removedModules,
                               uvEncodingEnabled=bool(_option(modelConfig, "use_uv_enc", False)))
        return model.cpu()

    def loadModels(self):
        """Idempotent preload; staged loading does not require an available GPU."""
        with self.lock, modelLoadLock:
            if self.isLoaded:
                return self
            started = perf_counter()
            self.provenance = self._checkpointProvenance()
            import torch

            targetDevice = torch.device(self.device)
            if self.memoryPolicy == "resident" and targetDevice.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for resident normal model loading")
            try:
                model = self.loadNormalPredictor(self.checkpointPath)
                model.eval().requires_grad_(False)
                model.to(targetDevice if self.memoryPolicy == "resident" else "cpu")
                self.model = model
                self.provenance["modelStorageBytes"] = sum(tensor.numel() * tensor.element_size()
                    for tensor in list(model.parameters()) + list(model.buffers()))
                self.isLoaded = True
                self.loadSeconds = perf_counter() - started
            except Exception:
                self.model = None
                self.isLoaded = False
                raise
        return self

    @contextmanager
    def _activateModel(self):
        import torch

        target = torch.device(self.device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for normal prediction")
        try:
            self.model.to(target)
            yield
        finally:
            if self.memoryPolicy == "staged":
                self.model.to("cpu")
                if target.type == "cuda":
                    torch.cuda.empty_cache()

    def _predictOne(self, image):
        import torch

        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True, order="C")).to(self.device)[None, None]
        direct, _ = self.model({"tar_rgb": tensor})
        mirrored, _ = self.model({"tar_rgb": torch.flip(tensor, dims=[3])})
        if not isinstance(direct, dict) or not isinstance(mirrored, dict):
            raise ValueError("Normal predictor must return a dictionary containing normals")
        first, second = direct.get("normals"), mirrored.get("normals")
        expectedShape = (1, 1, 3, imageResolution, imageResolution)
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or tuple(first.shape) != expectedShape or tuple(second.shape) != expectedShape
                or not first.is_floating_point() or not second.is_floating_point()):
            raise ValueError("Normal predictor must return floating normals shaped (1,1,3,512,512)")
        first = first[0, 0].float()
        second = torch.flip(second[0, 0].float(), dims=[2])
        second[0] *= -1
        averaged = (first + second) * .5
        magnitude = torch.linalg.vector_norm(averaged, dim=0)
        finite = torch.isfinite(averaged).all(dim=0) & torch.isfinite(magnitude)
        valid = finite & (magnitude > minimumNormalLength)
        safeMagnitude = torch.where(valid, magnitude, torch.ones_like(magnitude))
        normalized = torch.where(valid[None], averaged / safeMagnitude[None], torch.zeros_like(averaged))
        # Official network_inference.py: after normalizing, the display encoding
        # stacks [encodedX, 1-encodedZ, 1-encodedY]. In signed vector coordinates
        # that is exactly (rawX, -rawZ, -rawY), with no camera rotation.
        canonical = torch.stack([normalized[0], -normalized[2], -normalized[1]], dim=0)
        diagnostics = {"validPixelCount": int(valid.sum().item()), "nonfinitePixelCount": int((~finite).sum().item()),
                       "zeroOrCancelledPixelCount": int((finite & ~valid).sum().item())}
        return (canonical.permute(1, 2, 0).cpu().numpy().copy(), valid.cpu().numpy().copy(), diagnostics)

    def predict(self, images):
        """Return unit FLAME vectors and a numerical-validity mask, never face support.

        Images retain the caller's crop/camera pixel frame. The network consumes
        [0,1] RGB; DINO's own processor performs backbone normalization. Returned
        signed normals are not RGB encodings. Invalid/cancelled vectors are zero.
        Consumers must intersect validMask with independent face/occlusion and
        confidence gates; finite predictions outside the face are not evidence.
        """
        images = validateNormalImages(images)
        with self.lock:
            if not self.isLoaded or self.model is None:
                raise RuntimeError("Call loadModels before normal prediction")
            import torch

            started = perf_counter()
            normals = np.empty(images.shape, np.float32)
            validMask = np.empty(images.shape[:3], bool)
            viewDiagnostics = []
            with self._activateModel(), torch.inference_mode():
                for index, image in enumerate(images):
                    normal, valid, diagnostics = self._predictOne(image)
                    normals[index], validMask[index] = normal, valid
                    viewDiagnostics.append({"viewIndex": index, **diagnostics})
                    del normal, valid
            return {"normals": normals, "validMask": validMask,
                    "diagnostics": {"coordinateSystem": "FLAME canonical; not camera space",
                        "networkToFlameAxes": "(rawX, -rawZ, -rawY)", "cameraRotationApplied": False,
                        "augmentation": "original + horizontal mirror, spatial unflip and raw X sign inversion, average then unit normalize",
                        "inputEncoding": "display-encoded float RGB [0,1]; DINO normalization inside network",
                        "outputEncoding": "signed unit XYZ float32; invalid vectors zero; no PNG quantization",
                        "validMaskMeaning": "finite nonzero averaged prediction only; not semantic support or confidence",
                        "minimumNormalLength": minimumNormalLength, "views": viewDiagnostics,
                        "viewBatchSize": 1, "networkCalls": 2 * len(images), "memoryPolicy": self.memoryPolicy,
                        "retainedPredictionCacheBytes": 0, "returnedArrayBytes": normals.nbytes + validMask.nbytes,
                        "maximumReturnedArrayBytes": maximumViews * imageResolution ** 2 * 13,
                        "memoryLimitScope": "Returned arrays only; model weights and one-view network activations are additional, not an asserted GPU peak",
                        "loadSeconds": self.loadSeconds, "inferenceSeconds": perf_counter() - started,
                        "provenance": deepcopy(self.provenance)}}
