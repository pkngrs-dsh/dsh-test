"""Offline, preloadable FACER/FaRL adapter with an original-image contract.

Only this module knows the external library. It uses the configuration-specific
alignment and inverse warp directly because the local FACER forward wrapper was
modified to always use CelebM alignment, which is incorrect for LAPA. Upstream
algorithms and model implementations remain in thirdParty/facer.
"""

from __future__ import annotations

import hashlib
import importlib
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from ..faceSelection import selectLargestFaceIndex
from ..faceSemantics import FaceSemanticResult, normalizeParserLabels
from ..runtime import configureRuntime


modelLoadingLock = RLock()
parserWeightNames = {
    "celebm": "face_parsing.farl.celebm.main_ema_181500_jit.pt",
    "lapa": "face_parsing.farl.lapa.main_ema_136500_jit191.pt",
}


class FaceParsingAdapter:
    """One resident detector/parser pair; load once, serialize inference calls.

    Input is an EXIF-oriented RGB uint8 HWC image. Detection may be downsampled
    for memory, but parsing samples the original image and returns original HW.
    Detection selects the largest face, breaking ties by confidence and input order.
    No implicit model downloads or device fallbacks.
    """

    def __init__(self, projectRoot: Path, device: str = "cpu", parserName: str = "celebm",
                 detectionMaxSize: int = 1024, detectionThreshold: float = 0.8) -> None:
        if parserName not in parserWeightNames:
            raise ValueError("parserName must be celebm or lapa")
        if (not isinstance(device, str) or not
                (device == "cpu" or device == "cuda" or
                 (device.startswith("cuda:") and device[5:].isdigit()))):
            raise ValueError("device must be cpu, cuda or cuda:<index>")
        if (isinstance(detectionMaxSize, bool) or not isinstance(detectionMaxSize, int)
                or detectionMaxSize < 32):
            raise ValueError("detectionMaxSize must be an integer >= 32")
        if (isinstance(detectionThreshold, (bool, np.bool_)) or not isinstance(detectionThreshold, Real)
                or not np.isfinite(detectionThreshold)
                or not 0 < detectionThreshold < 1):
            raise ValueError("detectionThreshold must be finite and in (0, 1)")
        self.projectRoot = Path(projectRoot).resolve()
        self.device = device
        self.parserName = parserName
        self.detectionMaxSize = detectionMaxSize
        self.detectionThreshold = float(detectionThreshold)
        self.lock = RLock()
        self.isLoaded = False
        self.models: dict[str, Any] = {}
        self.parserSettings: dict[str, Any] = {}
        self.modelMetadata: dict[str, Any] = {}

    def assetPaths(self) -> dict[str, Path]:
        weights = self.projectRoot / "01_UVFaceFusion/weights/facer_hub_checkpoints"
        return {"faceDetector": weights / "Resnet50_Final.pth",
                "faceParser": weights / parserWeightNames[self.parserName]}

    def loadModels(self) -> FaceParsingAdapter:
        """Idempotently preload local weights; failure leaves no partial state."""
        with self.lock, modelLoadingLock:
            if self.isLoaded:
                return self
            paths = self.assetPaths()
            missingPaths = [str(path) for path in paths.values() if not path.is_file()]
            if missingPaths:
                raise FileNotFoundError("Missing face parsing assets:\n" + "\n".join(missingPaths))
            import torch

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for the requested face parsing device")
            try:
                models, settings, metadata = self.loadBackend(paths)
                for model in models.values():
                    model.eval()
                    # FaRL weights are TorchScript modules, which do not expose
                    # nn.Module.requires_grad_. Their parameters still do.
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                    model.to(self.device)
                # Validate label compatibility before declaring the adapter ready.
                normalizeParserLabels(np.zeros((1, 1), np.uint8), settings["label_names"])
                self.models, self.parserSettings, self.modelMetadata = models, settings, metadata
                self.isLoaded = True
            except Exception:
                self.models.clear()
                self.parserSettings.clear()
                self.modelMetadata.clear()
                raise
        return self

    def loadBackend(self, paths: dict[str, Path]) -> tuple[dict[str, Any], dict, dict]:
        """External dependency seam, replaceable in CPU lifecycle tests."""
        configureRuntime(self.projectRoot)
        facer = importlib.import_module("facer")
        sourceRoot = self.projectRoot / "thirdParty/facer/facer"
        if Path(facer.__file__).resolve().parent != sourceRoot.resolve():
            raise RuntimeError("An unexpected facer installation is already imported")
        farl = importlib.import_module("facer.face_parsing.farl")
        version = importlib.import_module("facer.version").__version__
        detector = facer.face_detector("retinaface/resnet50", device="cpu",
                                       model_path=str(paths["faceDetector"]),
                                       threshold=self.detectionThreshold)
        parser = facer.face_parser(f"farl/{self.parserName}/448", device="cpu",
                                   model_path=str(paths["faceParser"]))
        sourceHashes = {name: hashlib.sha256((sourceRoot / name).read_bytes()).hexdigest()
                        for name in ("face_parsing/farl.py", "face_detection/retinaface.py", "transform.py")}
        metadata = {
            "backend": "facer/FaRL", "backendVersion": version,
            "sourceUrl": "https://github.com/FacePerceiver/facer",
            "sourceHashes": sourceHashes,
            "parserName": self.parserName, "parserResolution": 448,
            "weightPaths": {name: str(path) for name, path in paths.items()},
            "alignmentImplementation": "configurationSpecificFarlWarp",
        }
        return {"faceDetector": detector, "faceParser": parser.net}, dict(
            farl.pretrain_settings[f"{self.parserName}/448"]), metadata

    def infer(self, imageRgb: np.ndarray, *, faceLandmarks: np.ndarray | None = None) -> FaceSemanticResult:
        """Parse one image; optional five-point anchors bypass detection only."""
        image = np.asarray(imageRgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("infer requires an EXIF-oriented RGB uint8 HWC image")
        if faceLandmarks is not None:
            faceLandmarks = np.asarray(faceLandmarks, dtype=np.float32)
            if faceLandmarks.shape != (5, 2) or not np.isfinite(faceLandmarks).all():
                raise ValueError("faceLandmarks must contain five finite XY coordinates")
        with self.lock:
            if not self.isLoaded:
                raise RuntimeError("Call loadModels before face parsing inference")
            import torch

            with torch.inference_mode():
                imageTensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].to(self.device)
                if faceLandmarks is None:
                    points, box, score, selection = self.detectFace(imageTensor, includeSelection=True)
                else:
                    points = torch.from_numpy(faceLandmarks.copy())[None].to(self.device)
                    box, score = None, None
                    selection = None
                if torch.linalg.vector_norm(points[0, 1] - points[0, 0]).item() < 2:
                    raise ValueError("Degenerate face landmarks: eye anchors are less than two pixels apart")
                sourceLabels, confidence, validMask = self.parseImage(imageTensor, points)
                standardLabels = normalizeParserLabels(sourceLabels, self.parserSettings["label_names"])
                standardLabels[~validMask] = 0
                confidence[~validMask] = 0
                metadata = {
                    **self.modelMetadata, "device": self.device,
                    "originalSize": [image.shape[1], image.shape[0]],
                    "sourceLabelNames": list(self.parserSettings["label_names"]),
                    "sideConvention": "sourceDatasetLabels",
                    "landmarkConvention": "retinaFaceFivePointOrderOriginalPixelXY",
                    "landmarkSource": "provided" if faceLandmarks is not None else "retinaFace",
                    "detectionScore": score, "confidenceKind": "softmaxMaximumUncalibrated",
                }
                if selection is not None:
                    metadata["selection"] = selection
                return FaceSemanticResult(
                    labelMap=standardLabels, confidence=confidence,
                    landmarks=points[0].cpu().numpy().astype(np.float32),
                    validMask=validMask, boxXyxy=box, metadata=metadata)

    def detectFace(self, imageTensor: Any, *, includeSelection: bool = False
                   ) -> tuple[Any, np.ndarray, float] | tuple[Any, np.ndarray, float, dict[str, Any]]:
        """Return selected points, box and score, optionally followed by selection metadata."""
        import torch
        import torch.nn.functional as functional

        height, width = imageTensor.shape[-2:]
        scale = min(1.0, self.detectionMaxSize / max(height, width))
        targetHeight, targetWidth = max(1, round(height * scale)), max(1, round(width * scale))
        if (targetHeight, targetWidth) == (height, width):
            detectionImage = imageTensor
        else:
            detectionImage = functional.interpolate(imageTensor.float(), size=(targetHeight, targetWidth),
                                                     mode="bilinear", align_corners=False).round().to(torch.uint8)
        detected = self.models["faceDetector"](detectionImage)
        selectedIndex = selectLargestFaceIndex(detected["rects"].detach().cpu().numpy(),
                                               detected["scores"].detach().cpu().numpy())
        count = len(detected["scores"])
        if detected["points"].shape != (count, 5, 2):
            raise ValueError("Face detector must return five points per detected face")
        points = detected["points"][selectedIndex:selectedIndex + 1].float().clone()
        sizeRatio = points.new_tensor([width / targetWidth, height / targetHeight])
        points = (points + 0.5) * sizeRatio - 0.5
        boxTensor = detected["rects"][selectedIndex].float() * sizeRatio.repeat(2)
        if (not torch.isfinite(points).all() or not torch.isfinite(boxTensor).all()
                or not torch.all(boxTensor[2:] > boxTensor[:2])):
            raise ValueError("Face detector returned invalid coordinates")
        score = float(detected["scores"][selectedIndex].item())
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Face detector returned an invalid score")
        result = (points, boxTensor.cpu().numpy().astype(np.float32), score)
        if includeSelection:
            selection = {"strategy": "largestArea", "detectedCount": count, "selectedIndex": selectedIndex}
            return (*result, selection)
        return result

    def parseImage(self, imageTensor: Any, points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        height, width = imageTensor.shape[-2:]
        settings = self.parserSettings
        matrix = settings["get_matrix_fn"](points)
        grid = settings["get_grid_fn"](matrix=matrix, orig_shape=(height, width))
        inverseGrid = settings["get_inv_grid_fn"](matrix=matrix, orig_shape=(height, width))
        aligned = functional.grid_sample(imageTensor.float() / 255.0, grid,
                                         mode="bilinear", align_corners=False)
        alignedLogits, _ = self.models["faceParser"](aligned)
        if alignedLogits.shape[1] != len(settings["label_names"]):
            raise ValueError("Parser output channels do not match its configured labels")
        logits = functional.grid_sample(alignedLogits, inverseGrid, mode="bilinear", align_corners=False)
        if tuple(logits.shape[-2:]) != (height, width) or not torch.isfinite(logits).all():
            raise ValueError("Parser returned invalid original-image logits")
        confidence, sourceLabels = torch.softmax(logits, dim=1).max(dim=1)
        # Reject inverse samples requiring zero padding instead of reporting a
        # uniform zero-logit softmax as observed background with confidence.
        bounds = inverseGrid.new_tensor([1 - 1 / alignedLogits.shape[3], 1 - 1 / alignedLogits.shape[2]])
        validMask = (inverseGrid.abs() <= bounds + 1e-6).all(dim=-1)
        return (sourceLabels[0].cpu().numpy(), confidence[0].cpu().numpy().astype(np.float32),
                validMask[0].cpu().numpy().astype(bool))
