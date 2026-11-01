"""In-memory OpenDelight adapter with explicit input/aligned image transforms.

The inference sequence follows ``02_OpenDelight/opendelight/test.py``; the FFHQ
quad follows ``synthetic/align_utils.py``. Upstream code and its license remain
in that source tree. Network layers are loaded from that tree, not copied from
the legacy v2 project. Neither upstream test.py nor its matting script is run.

``albedoImages`` preserves the network's native RGB output encoding. Calling it
physically calibrated linear reflectance requires separate validation; this
adapter deliberately applies no additional gamma transform or uint8 rounding.
"""

from __future__ import annotations

import ast
import hashlib
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np


def computeAlignment(landmarks: np.ndarray, imageSize: int = 512,
                     borderScale: float = 0.85) -> np.ndarray:
    """Return HInputToAligned, mapping input pixel (x, y) to FFHQ pixels.

    Adapted from OpenDelight's synthetic/align_utils.py. Pixel coordinates and
    OpenCV's (width, height) convention are retained, including the upstream
    destination corners at ``imageSize`` rather than ``imageSize - 1``.
    """
    import cv2

    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
        raise ValueError("Expected 68 finite ibug landmarks in input pixel coordinates")
    leftEye = landmarks[36:42].mean(axis=0)
    rightEye = landmarks[42:48].mean(axis=0)
    eyeCenter = (leftEye + rightEye) * 0.5
    eyeVector = rightEye - leftEye
    mouthCenter = (landmarks[48] + landmarks[54]) * 0.5
    mouthVector = mouthCenter - eyeCenter
    if np.linalg.norm(eyeVector) < 4 or np.linalg.norm(mouthVector) < 4:
        raise ValueError("Degenerate eye/mouth landmarks; face cannot be aligned reliably")
    xAxis = eyeVector - np.flipud(mouthVector) * np.array([-1, 1], np.float32)
    axisLength = float(np.linalg.norm(xAxis))
    if axisLength < 1e-6:
        raise ValueError("Degenerate FFHQ alignment axis")
    xAxis = xAxis / axisLength * max(np.linalg.norm(eyeVector) * 2,
                                   np.linalg.norm(mouthVector) * 1.8) * borderScale
    yAxis = np.flipud(xAxis) * np.array([-1, 1], np.float32)
    center = eyeCenter + mouthVector * 0.1
    sourceQuad = np.array([center - xAxis - yAxis, center - xAxis + yAxis,
                           center + xAxis + yAxis, center + xAxis - yAxis], np.float32)
    targetQuad = np.array([[0, 0], [0, imageSize], [imageSize, imageSize],
                           [imageSize, 0]], np.float32)
    inputToAligned = cv2.getPerspectiveTransform(sourceQuad, targetQuad)
    if not np.isfinite(inputToAligned).all() or abs(np.linalg.det(inputToAligned)) < 1e-10:
        raise ValueError("Singular FFHQ alignment transform")
    return inputToAligned


def restoreToInput(alignedAlbedo: np.ndarray, alignedMask: np.ndarray,
                   inputMask: np.ndarray, inputToAligned: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-warp albedo and valid support into the original 512 pixel frame."""
    import cv2

    imageHeight, imageWidth = inputMask.shape
    alignedToInput = np.linalg.inv(inputToAligned)
    restored = cv2.warpPerspective(
        alignedAlbedo.astype(np.float32), alignedToInput, (imageWidth, imageHeight),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    restoredMask = cv2.warpPerspective(
        alignedMask.astype(np.float32), alignedToInput, (imageWidth, imageHeight),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    support = cv2.warpPerspective(
        np.ones(alignedMask.shape, np.float32), alignedToInput, (imageWidth, imageHeight),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    validMask = (inputMask > 0.5) & (restoredMask > 0.5) & (support > 0.999)
    restored = np.clip(restored, 0, 1).astype(np.float32)
    restored[~validMask] = 0
    return restored, validMask


def loadArchitectureModule(sourcePath: Path) -> types.ModuleType:
    """Load one isolated architecture file, excluding upstream debug imports.

    These three upstream files contain no cross-file runtime imports except an
    unused ``utils.count_parameters`` debug helper. Removing that import and
    ``sys.path.append`` avoids the generic model/encoder/utils module collision.
    All actual layer definitions execute directly from the checked-in source.
    """
    sourcePath = sourcePath.resolve()
    syntaxTree = ast.parse(sourcePath.read_text(encoding="utf-8"), filename=str(sourcePath))
    keptNodes = []
    for node in syntaxTree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "utils":
            if [alias.name for alias in node.names] != ["count_parameters"]:
                raise RuntimeError(f"Unexpected upstream runtime utils import: {sourcePath}")
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "sys.path.append"):
            continue
        keptNodes.append(node)
    syntaxTree.body = keptNodes
    moduleHash = hashlib.sha256(str(sourcePath).encode()).hexdigest()[:12]
    moduleName = f"faceAppearanceOpenDelight_{sourcePath.stem}_{moduleHash}"
    module = types.ModuleType(moduleName)
    module.__file__ = str(sourcePath)
    exec(compile(syntaxTree, str(sourcePath), "exec"), module.__dict__)
    return module


class DelightAdapter:
    """Reusable, serialized model owner; rejected views have zero valid support.

    Call loadModels() once before predict(). ``staged`` keeps loaded weights on
    CPU between calls; ``resident`` keeps them on the requested inference device.
    Input shape is (N, 512, 512, 3), RGB floats in [0, 1], with matching masks.
    """

    def __init__(self, projectRoot: Path, device: str = "cuda",
                 memoryPolicy: str = "staged", useEnhancer: bool = True):
        if memoryPolicy not in {"staged", "resident"}:
            raise ValueError("memoryPolicy must be 'staged' or 'resident'")
        self.projectRoot = Path(projectRoot).resolve()
        self.device = str(device)
        self.memoryPolicy = memoryPolicy
        self.useEnhancer = bool(useEnhancer)
        self.imageSize = 512
        self.borderScale = 0.85
        self.minInputCoverage = 0.01
        self.minAlignedCoverage = 0.03
        self.model: Any = None
        self.enhancer: Any = None
        self.faceDetector: Any = None
        self.landmarkDetector: Any = None
        self.loaded = False
        self.modelDevice = "cpu"
        self.lock = threading.RLock()

    def loadModels(self) -> "DelightAdapter":
        """Load local weights onto CPU exactly once; never download weights."""
        with self.lock:
            if self.loaded:
                return self
            sourceRoot = self.projectRoot / "02_OpenDelight/opendelight"
            weightsRoot = self.projectRoot / "02_OpenDelight/weights/opendelight"
            ibugRoot = self.projectRoot / "assets/models/ibug"
            requiredPaths = [sourceRoot / "config/delight_base_network.yaml",
                             weightsRoot / "base_delight_network.pth",
                             ibugRoot / "Resnet50_Final.pth", ibugRoot / "2dfan2_alt.pth"]
            if self.useEnhancer:
                requiredPaths.append(weightsRoot / "unet_enhancer.pth")
            missingPaths = [str(path) for path in requiredPaths if not path.is_file()]
            if missingPaths:
                raise FileNotFoundError("Missing local OpenDelight assets: " + ", ".join(missingPaths))

            import torch
            import yaml
            from ibug.face_alignment import FANPredictor
            from ibug.face_detection import RetinaFacePredictor

            with requiredPaths[0].open(encoding="utf-8") as configFile:
                modelConfig = yaml.safe_load(configFile)
            encoderConfig = modelConfig["encoder"]
            if (encoderConfig.get("type") != "mae_mix"
                    or encoderConfig.get("model_type") != "base"
                    or encoderConfig.get("img_size") != self.imageSize
                    or modelConfig["decoder"].get("type") != "sapiens"):
                raise ValueError("This adapter supports the bundled mae_mix/base/512 + sapiens model")

            encoderModule = loadArchitectureModule(sourceRoot / "encoder/mae_mix_encoder.py")
            decoderModule = loadArchitectureModule(sourceRoot / "decoder/sapiens_decoder.py")

            class DelightNetwork(torch.nn.Module):
                """Minimal composition matching upstream DelightBaseModel.forward_mix."""

                def __init__(self):
                    super().__init__()
                    self.encoder = encoderModule.mae_vit_base_patch16(
                        img_size=512, device="cpu", num_reg_tokens=encoderConfig["num_reg_tokens"])
                    self.decoder = decoderModule.SapiensDecoder(**modelConfig["decoder"])

                def forward(self, imageTensor, dataSource):
                    features = self.encoder(imageTensor, dataSource)
                    features = features[:, 1 + self.encoder.num_reg_tokens:, :]
                    patchSize = int(features.shape[1] ** 0.5)
                    features = features.reshape(features.shape[0], patchSize, patchSize, -1)
                    return torch.sigmoid(self.decoder(features.permute(0, 3, 1, 2).contiguous()))

            # The full checkpoint supplies all encoder weights. The upstream
            # MAE warm-start checkpoint is redundant and has a cwd-relative path.
            model = DelightNetwork()
            model.load_state_dict(torch.load(requiredPaths[1], map_location="cpu", weights_only=True),
                                  strict=True)
            # Upstream stores these as ordinary tensors, so .to() would miss them.
            # Nonpersistent buffers preserve the original checkpoint key contract.
            for tensorName in ("pos_embed", "imagenet_mean", "imagenet_std"):
                tensorValue = getattr(model.encoder, tensorName)
                delattr(model.encoder, tensorName)
                model.encoder.register_buffer(tensorName, tensorValue, persistent=False)
            model.eval().requires_grad_(False)

            enhancer = None
            if self.useEnhancer:
                enhancerModule = loadArchitectureModule(sourceRoot / "model/detail_enhance_unet.py")
                enhancer = enhancerModule.UNet_Enhancer(
                    n_channels=6, n_classes=3, output_confidence=False, output_shadow_mask=False)
                enhancer.load_state_dict(torch.load(weightsRoot / "unet_enhancer.pth",
                                                     map_location="cpu", weights_only=True), strict=True)
                enhancer.eval().requires_grad_(False)

            retinaModel = RetinaFacePredictor.get_model("resnet50")
            retinaModel.weights = str(ibugRoot / "Resnet50_Final.pth")
            fanModel = FANPredictor.get_model("2dfan2_alt")
            fanModel.weights = str(ibugRoot / "2dfan2_alt.pth")
            faceDetector = RetinaFacePredictor(threshold=0.8, device="cpu", model=retinaModel)
            landmarkDetector = FANPredictor(device="cpu", model=fanModel,
                                             config=FANPredictor.create_config(use_jit=False))
            self.model, self.enhancer = model, enhancer
            self.faceDetector, self.landmarkDetector = faceDetector, landmarkDetector
            if self.memoryPolicy == "resident":
                self.moveModels(self.device)
            self.loaded = True
            return self

    def moveModels(self, targetDevice: str) -> None:
        """Move weights and ibug's cached prior tensor together."""
        for model in (self.model, self.enhancer):
            if model is not None:
                model.to(targetDevice)
        for predictor in (self.faceDetector, self.landmarkDetector):
            if predictor is not None:
                predictor.net.to(targetDevice)
                predictor.device = targetDevice
        if self.faceDetector is not None:
            self.faceDetector.priors = None
            self.faceDetector.previous_size = None
        self.modelDevice = str(targetDevice)

    def predict(self, images: np.ndarray, faceMasks: np.ndarray) -> dict[str, Any]:
        """Return albedoImages, boolean validMasks, and per-view diagnostics.

        Bad view geometry is reported as status='rejected', with zero albedo and
        mask for that view. Invalid batch input or a model execution error raises
        an exception instead of silently replacing an algorithm failure.
        """
        images, faceMasks = self.validateInputs(images, faceMasks)
        with self.lock:
            if not self.loaded:
                raise RuntimeError("Call DelightAdapter.loadModels() before predict()")
            import torch

            albedoImages = np.zeros_like(images, dtype=np.float32)
            validMasks = np.zeros(faceMasks.shape, dtype=bool)
            diagnostics = []
            try:
                self.moveModels(self.device)
                with torch.inference_mode():
                    for viewIndex, (image, faceMask) in enumerate(zip(images, faceMasks)):
                        try:
                            albedo, validMask, diagnostic = self.predictView(image, faceMask)
                        except Exception as error:
                            raise RuntimeError(f"OpenDelight failed for view {viewIndex}: {error}") from error
                        albedoImages[viewIndex] = albedo
                        validMasks[viewIndex] = validMask
                        diagnostic["viewIndex"] = viewIndex
                        diagnostics.append(diagnostic)
            finally:
                if self.memoryPolicy == "staged":
                    self.moveModels("cpu")
                    if self.device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return {"albedoImages": albedoImages, "validMasks": validMasks,
                    "diagnostics": diagnostics}

    def validateInputs(self, images: np.ndarray, faceMasks: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        images, faceMasks = np.asarray(images), np.asarray(faceMasks)
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:] != (512, 512, 3):
            raise ValueError("images must have shape (N, 512, 512, 3), with N >= 1")
        if faceMasks.shape != images.shape[:3]:
            raise ValueError("faceMasks must have shape (N, 512, 512) matching images")
        for value, name in ((images, "images"), (faceMasks, "faceMasks")):
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite numeric values")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"{name} values must lie in [0, 1]")
        return images.astype(np.float32, copy=False), faceMasks.astype(np.float32, copy=False)

    def predictView(self, image: np.ndarray, faceMask: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        import cv2

        inputCoverage = float((faceMask > 0.5).mean())
        diagnostic: dict[str, Any] = {"status": "ok", "warnings": [],
                                     "inputCoverage": inputCoverage,
                                     "colorSpace": "opendelightNative"}

        def rejected(reason: str):
            diagnostic.update(status="rejected", reason=reason, validCoverage=0.0)
            return np.zeros_like(image), np.zeros(faceMask.shape, bool), diagnostic

        if inputCoverage < self.minInputCoverage:
            return rejected("insufficientInputFaceCoverage")
        detectorImage = np.round(image * 255).astype(np.uint8)
        faces = self.faceDetector(detectorImage, rgb=True)
        diagnostic["detectedFaceCount"] = len(faces)
        if len(faces) == 0:
            return rejected("ibugFaceNotDetected")
        # Select the face best supported by the caller's semantic mask.
        overlaps = []
        for face in faces:
            left, top, right, bottom = np.clip(face[:4], 0, self.imageSize).astype(int)
            overlaps.append(float(faceMask[top:bottom, left:right].sum()))
        selectedFace = int(np.argmax(overlaps))
        if overlaps[selectedFace] <= 0:
            return rejected("detectedFaceOutsideInputMask")
        landmarks, landmarkScores = self.landmarkDetector(
            detectorImage, faces[selectedFace:selectedFace + 1], rgb=True)
        if len(landmarks) == 0:
            return rejected("ibugLandmarksNotDetected")
        try:
            inputToAligned = computeAlignment(landmarks[0], self.imageSize, self.borderScale)
        except ValueError as error:
            return rejected(str(error))
        diagnostic["inputToAligned"] = inputToAligned.tolist()
        diagnostic["alignedToInput"] = np.linalg.inv(inputToAligned).tolist()
        diagnostic["meanLandmarkScore"] = float(np.asarray(landmarkScores).mean())
        leftEye = landmarks[0, 36:42].mean(axis=0)
        rightEye = landmarks[0, 42:48].mean(axis=0)
        eyeVector = rightEye - leftEye
        noseOffset = landmarks[0, 30] - (leftEye + rightEye) * 0.5
        # A diagnostic heuristic, not a calibrated yaw estimate or confidence.
        diagnostic["noseOffsetEyeRatio"] = float(abs(np.dot(noseOffset, eyeVector))
                                                 / np.dot(eyeVector, eyeVector))
        if diagnostic["noseOffsetEyeRatio"] > 0.65:
            diagnostic["warnings"].append("possibleProfileViewCheckCrossViewConsistency")
        alignedImage = cv2.warpPerspective(image, inputToAligned, (512, 512),
                                            flags=cv2.INTER_CUBIC)
        alignedMask = cv2.warpPerspective(faceMask, inputToAligned, (512, 512),
                                           flags=cv2.INTER_LINEAR)
        alignedMask = np.clip(alignedMask, 0, 1)
        alignedCoverage = float((alignedMask > 0.5).mean())
        diagnostic["alignedCoverage"] = alignedCoverage
        if alignedCoverage < self.minAlignedCoverage:
            return rejected("insufficientAlignedFaceCoverage")
        alignedAlbedo = self.runNetwork(np.clip(alignedImage, 0, 1), alignedMask)
        if alignedAlbedo.shape != (512, 512, 3) or not np.isfinite(alignedAlbedo).all():
            raise RuntimeError("OpenDelight produced an invalid shape or nonfinite albedo")
        albedo, validMask = restoreToInput(alignedAlbedo, alignedMask, faceMask, inputToAligned)
        diagnostic["validCoverage"] = float(validMask.mean())
        diagnostic["retainedFaceFraction"] = float(validMask.sum() / (faceMask > 0.5).sum())
        if diagnostic["retainedFaceFraction"] < 0.5:
            diagnostic["warnings"].append("alignmentRetainsLessThanHalfOfFaceMask")
        if len(faces) > 1:
            diagnostic["warnings"].append("multipleFacesSelectedByMaskOverlap")
        if not validMask.any():
            return rejected("noValidInverseWarpSupport")
        return albedo, validMask, diagnostic

    def runNetwork(self, alignedImage: np.ndarray, alignedMask: np.ndarray) -> np.ndarray:
        """Run preloaded networks without changing coordinates or reloading weights."""
        import torch

        imageTensor = torch.from_numpy(np.ascontiguousarray(alignedImage)).permute(2, 0, 1)[None]
        maskTensor = torch.from_numpy(np.ascontiguousarray(alignedMask))[None, None]
        inputTensor = (imageTensor * maskTensor).to(self.device)
        outputTensor = self.model(inputTensor, torch.zeros(1, dtype=torch.long, device=self.device))
        if self.enhancer is not None:
            outputTensor = self.enhancer(torch.cat([outputTensor, inputTensor], dim=1))["diffuse"]
        return outputTensor[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
