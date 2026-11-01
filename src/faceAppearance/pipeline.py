"""Quality-first composition of independently callable reconstruction stages."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from .artifacts import exportMesh, saveCameraOverlays, writeImage, writeJson
from .config import PipelineConfig
from .data import loadViews
from .faceCrop import workingImageValidMask

logger = logging.getLogger(__name__)


class MissingCorrespondenceError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loaded = False
        self.geometryAdapter = None
        self.delightAdapter = None
        self.lock = RLock()
        self.loadSeconds = 0.0
        self.uvLookups = {}
        self.registrationAssets = {}
        self.wildCapAdapter = None
        self.assemblyAssets = None
        self.assemblyAssetLoadSeconds = 0.0

    def loadModels(self):
        """One explicit initialization shared by all jobs in this process."""
        with self.lock:
            if self.loaded:
                return self
            if self.config.enableAssembly:
                # Validate and preload the complete CPU asset package before
                # initializing any inference model or GPU allocation.
                from .headAssembly import AssemblyAssets
                assetStart = perf_counter()
                self.assemblyAssets = AssemblyAssets(self.config.projectRoot / self.config.assemblyAsset)
                self.assemblyAssetLoadSeconds = perf_counter() - assetStart
            if self.config.enableWildCap:
                assetRoot = self.config.projectRoot / self.config.registrationAsset
                if not (assetRoot / "manifest.json").is_file():
                    raise MissingCorrespondenceError(
                        "A reviewed registered ICT template and compiled correspondence assets are required; "
                        "see assets/wrapPreparation/README.md.")
                from .registration import RegistrationAssets
                for resolution in {self.config.workingResolution, self.config.wildCapResolution}:
                    self.registrationAssets[resolution] = RegistrationAssets(assetRoot, resolution)
            from .runtime import configureRuntime
            configureRuntime(self.config.projectRoot)
            from .adapters.geometry import GeometryAdapter
            from .adapters.delight import DelightAdapter

            start = perf_counter()
            self.geometryAdapter = GeometryAdapter(self.config.projectRoot, self.config.device,
                                                   self.config.memoryPolicy,
                                                   depthChunkSize=self.config.depthChunkSize)
            self.delightAdapter = DelightAdapter(self.config.projectRoot, self.config.device,
                                                 self.config.memoryPolicy, self.config.useEnhancer)
            self.geometryAdapter.loadModels()
            self.delightAdapter.loadModels()
            if self.config.enableWildCap:
                from .adapters.wildcap import WildCapAdapter
                self.wildCapAdapter = WildCapAdapter(self.config.projectRoot, self.config.device,
                                                     self.config.memoryPolicy).loadModels()
            from .texture import buildUvLookup
            for resolution in {512, self.config.workingResolution}:
                self.uvLookups[resolution] = buildUvLookup(self.geometryAdapter.canonicalUvs,
                                                           self.geometryAdapter.canonicalFaces, resolution)
            self.loadSeconds = perf_counter() - start
            self.loaded = True
            return self

    def run(self, paths: list[Path], outputDir: Path, stopAfter: str = "fusion") -> dict:
        if stopAfter not in {"geometry", "delight", "fusion", "wildcap", "assembly"}:
            raise ValueError("stopAfter must be geometry, delight, fusion, wildcap or assembly")
        if stopAfter == "assembly" and not self.config.enableAssembly:
            raise ValueError("The assembly stage requires enableAssembly=true in the configuration")
        runWildCap = stopAfter == "wildcap" or (stopAfter == "assembly" and self.config.assemblySourceStage == "wildcap")
        if runWildCap and not self.config.enableWildCap:
            raise ValueError("The wildcap stage requires enableWildCap=true in the configuration")
        if not self.loaded:
            raise RuntimeError("Call loadModels() before run()")
        if stopAfter == "assembly" and self.assemblyAssets is None:
            raise RuntimeError("Assembly assets were not preloaded")
        with self.lock:
            outputDir = Path(outputDir)
            if outputDir.is_dir() and any(outputDir.iterdir()):
                raise FileExistsError("Output directory is not empty; choose a new directory")
            outputDir.mkdir(parents=True, exist_ok=True)
            start = perf_counter()
            if self.config.device.startswith("cuda"):
                import torch
                torch.cuda.reset_peak_memory_stats()
            inputStart = perf_counter()
            images, inputs = loadViews(
                paths, self.config.minViews, self.config.maxViews,
                cropPolicy=self.config.inputCropPolicy, cropMargin=self.config.faceCropMargin,
                faceDetector=(self.geometryAdapter.detectInputFace
                              if self.config.inputCropPolicy == "faceSquare" else None))
            inputSeconds = perf_counter() - inputStart
            report = {"status": "running", "stopAfter": stopAfter, "inputs": inputs,
                      "config": self.config.toDict(), "loadSeconds": self.loadSeconds,
                      "stageSeconds": {"inputPreparation": inputSeconds}, "availableStages": ["geometry", "delight", "fusion"]
                          + (["wildcap"] if self.config.enableWildCap else [])
                          + (["assembly"] if self.config.enableAssembly else []),
                      "pendingStages": ["rigAndFullHeadAssembly", "validatedNormalConvention"],
                      "geometrySpace": "uvFaceFusionCanonical6986",
                      "metricUnits": "uncalibrated", "colorSpace": "opendelightNative",
                      "materialStatus": "experimentalDelightedColor; PBR calibration pending",
                      "arrayUvConvention": "u right, v down; pixel=u/v*(size-1)",
                      "objUvConvention": "u right, v up; exportedV=1-nativeV"}
            writeJson(outputDir / "inputs.json", inputs)
            writeJson(outputDir / "config.json", self.config.toDict())
            for index, image in enumerate(images):
                writeImage(outputDir / f"input{index:02d}.png", image)
            stageStart = perf_counter()
            logger.info("Reconstructing geometry from %s views", len(images))
            observedMasks = np.stack([workingImageValidMask(item) for item in inputs])
            geometry = self.geometryAdapter.reconstruct(images, validImageMasks=observedMasks)
            report["stageSeconds"]["geometry"] = perf_counter() - stageStart
            logger.info("Geometry completed in %.2fs", report["stageSeconds"]["geometry"])
            geometryArrays = {key: value for key, value in geometry.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(outputDir / "geometry.npz", **geometryArrays)
            writeJson(outputDir / "viewTransforms.json", geometry["viewTransforms"])
            exportMesh(outputDir, geometry, includeMaterial=stopAfter in {"fusion", "wildcap", "assembly"})
            report["vertexCount"], report["faceCount"] = len(geometry["vertices"]), len(geometry["faces"])
            report["topologySha256"] = hashlib.sha256(
                np.asarray(geometry["faces"], dtype="<i8").tobytes()
                + np.asarray(geometry["uvs"], dtype="<f4").tobytes()).hexdigest()
            if stopAfter != "geometry":
                stageStart = perf_counter()
                delight = self.delightAdapter.predict(images, geometry["faceMasks"])
                report["stageSeconds"]["delight"] = perf_counter() - stageStart
                logger.info("Delighting completed in %.2fs", report["stageSeconds"]["delight"])
                np.savez_compressed(outputDir / "delight.npz", albedoImages=delight["albedoImages"],
                                    validMasks=delight["validMasks"])
                report["delightDiagnostics"] = delight["diagnostics"]
                for index, albedo in enumerate(delight["albedoImages"]):
                    writeImage(outputDir / f"delight{index:02d}.png", albedo)
                    writeImage(outputDir / f"delightMask{index:02d}.png", delight["validMasks"][index])
                if stopAfter in {"fusion", "wildcap", "assembly"}:
                    stageStart = perf_counter()
                    report.update(self.finishFusion(geometry, delight, images, outputDir))
                    report["stageSeconds"]["cameraAndFusion"] = perf_counter() - stageStart
                    if runWildCap or stopAfter == "assembly":
                        with np.load(outputDir / "texture.npz", allow_pickle=False) as savedTexture:
                            texture = {key: savedTexture[key] for key in savedTexture.files}
                    if runWildCap:
                        stageStart = perf_counter()
                        report["wildCapDiagnostics"] = self.refineMaterials(geometry, texture, outputDir)
                        report["stageSeconds"]["wildCapAndTransfer"] = perf_counter() - stageStart
                    if stopAfter == "assembly":
                        assemblyTexture = texture
                        encoding = self.config.assemblyColorEncoding
                        if self.config.assemblySourceStage == "wildcap":
                            with np.load(outputDir / "wildcapUff/materials.npz", allow_pickle=False) as materials:
                                assemblyTexture = {**texture, "albedo": materials["baseWithWildCap"].copy()}
                            # The explicitly selected WildCap/base mixture is
                            # encoded under the existing gamma-2.2 assumption;
                            # an inherited baseline linear cache must not win.
                            assemblyTexture.pop("linearAlbedo", None)
                            encoding = "gamma2.2"
                            report["assemblyColorInterpretation"] = (
                                "WildCap/base preview mixture interpreted as gamma2.2; "
                                "OpenDelight source color is not physically calibrated")
                        stageStart = perf_counter()
                        assembled = self.assembleAsset(geometry, assemblyTexture, outputDir,
                                                       colorEncoding=encoding)
                        report["assemblyDiagnostics"] = assembled["diagnostics"]
                        report["clientAssetManifest"] = assembled["manifest"]
                        report["assemblySourceStage"] = self.config.assemblySourceStage
                        report["assemblyAssetLoadSeconds"] = self.assemblyAssetLoadSeconds
                        report["stageSeconds"]["assemblyAndExport"] = perf_counter() - stageStart
                        report["pendingStages"] = ["clientRigValidation", "validatedNormalConvention"]
            report["status"] = "succeeded"
            report["inferenceSeconds"] = perf_counter() - start
            if self.config.device.startswith("cuda"):
                import torch
                report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            writeJson(outputDir / "result.json", report)
            return report

    def assembleAsset(self, geometry: dict, texture: dict, outputDir: Path, *,
                      colorEncoding: str | None = None) -> dict:
        """Independently assemble/export saved arrays with preloaded CPU assets.

        No inference models or Pipeline.run call are required. The primary
        legacy-template vertex/index order is the asset/exporter's contract;
        the original UFF geometry and material files remain unchanged.
        A supplied linearAlbedo takes priority over encoded albedo.
        """
        if self.assemblyAssets is None:
            raise RuntimeError("Assembly assets were not preloaded")
        from .clientAsset import exportClientAsset

        encoding = "linear" if "linearAlbedo" in texture else (colorEncoding or self.config.assemblyColorEncoding)
        assembled = self.assemblyAssets.assemble(geometry, texture, colorEncoding=encoding)
        manifest = exportClientAsset(Path(outputDir) / "assembly", assembled)
        return {"diagnostics": assembled.get("diagnostics", {}), "manifest": manifest}

    def finishFusion(self, geometry: dict, delight: dict, images: np.ndarray, outputDir: Path):
        """Independent downstream entry point for inspecting saved stage arrays."""
        from .camera import fitViewCameras
        from .texture import fuseTexture

        cameras = fitViewCameras(geometry, self.config.maxCameraErrorPixels, self.config.seed,
                                 uvLookup=self.uvLookups.get(512))
        writeJson(outputDir / "cameras.json", cameras)
        saveCameraOverlays(outputDir, images, cameras)
        acceptedCount = sum(camera["valid"] for camera in cameras)
        if acceptedCount < 2:
            raise ValueError(f"Only {acceptedCount} camera(s) passed reprojection checks; at least 2 required")
        texture = fuseTexture(geometry, delight, cameras, self.config.workingResolution,
                              self.config.colorBalance,
                              uvLookup=self.uvLookups.get(self.config.workingResolution),
                              samplingMode=self.config.textureSampling)
        if not texture["coverageMask"].any():
            raise ValueError("No observed texture pixels survived mask and visibility checks")
        np.savez_compressed(outputDir / "texture.npz", **{
            key: value for key, value in texture.items() if isinstance(value, np.ndarray)})
        writeImage(outputDir / "albedo.png", texture["albedo"])
        writeImage(outputDir / "albedoPreview.png", texture["previewAlbedo"])
        writeImage(outputDir / "coverage.png", texture["coverageMask"])
        writeImage(outputDir / "confidence.png", texture["confidence"])
        return {"acceptedCameraCount": acceptedCount, "cameraDiagnostics": cameras,
                "textureDiagnostics": texture["diagnostics"]}

    def refineMaterials(self, geometry: dict, texture: dict, outputDir: Path,
                        artifactMaskIct: np.ndarray | None = None) -> dict:
        """Independent ICT refinement, preserving the original UFF geometry/albedo.

        Source tangent convention is not yet certified. The original ICT normal
        vectors are exported with that limitation; they are never relabelled as
        a UFF PBR normal map. Diffuse and scalar specular can be transferred now.
        artifactMaskIct is an optional binary mask in the fixed ICT image atlas
        at wildCapResolution, supplied by a caller or an upstream detector.
        No mask is inferred here: the default uses global lighting only.
        """
        from .atlasTransfer import transferTexture

        if self.wildCapAdapter is None:
            raise RuntimeError("WildCap models were not preloaded")
        forwardAssets = self.registrationAssets[self.config.wildCapResolution]
        backAssets = self.registrationAssets[self.config.workingResolution]
        ictGeometry = forwardAssets.toIctGeometry(geometry)
        ictInput = transferTexture(texture["albedo"], texture["coverageMask"],
                                    forwardAssets.atlasMap("uffToIct"))
        ictDir, uffDir = outputDir / "wildcapIct", outputDir / "wildcapUff"
        ictDir.mkdir(parents=True, exist_ok=True)
        uffDir.mkdir(parents=True, exist_ok=True)
        writeImage(ictDir / "input.png", ictInput["texture"])
        writeImage(ictDir / "inputObserved.png", ictInput["validMask"])
        result = self.wildCapAdapter.optimize(
            ictGeometry["vertices"], ictGeometry["faces"], ictGeometry["uvs"],
            ictInput["texture"], ictInput["validMask"], artifactMask=artifactMaskIct,
            uvFaces=ictGeometry["uvFaces"],
            geometryFrame=ictGeometry["geometryFrame"], workingResolution=self.config.wildCapResolution,
            iterations=self.config.wildCapIterations, seed=self.config.seed)
        if not np.asarray(result["validMask"], dtype=bool).any():
            raise ValueError("WildCap returned no valid observed ICT material pixels")
        np.savez_compressed(ictDir / "materials.npz", **{
            key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        writeImage(ictDir / "normalEncodedUnverifiedConvention.png", (result["tangentNormal"] + 1) / 2)
        backMaterials = {}
        transferDiagnostics = {"uffToIct": ictInput["diagnostics"], "ictToUff": {}}
        for key in ("diffuse", "specular"):
            writeImage(ictDir / f"{key}.png", result[key])
            back = transferTexture(result[key], result["validMask"], backAssets.atlasMap("ictToUff"),
                                    materialKind="scalar" if key == "specular" else "color",
                                    colorEncoding="gamma2.2" if key == "diffuse" else "native")
            transferDiagnostics["ictToUff"][key] = back["diagnostics"]
            backMaterials[key] = back["texture"]
            backMaterials["validMask"] = back["validMask"]
            writeImage(uffDir / f"{key}.png", back["texture"])
        if not backMaterials["validMask"].any():
            raise ValueError("No observed WildCap material pixels survived the ICT-to-UFF transfer")
        # Retain measured base color outside the qualified common surface. Keep
        # a separate coverage map so this fallback cannot be mistaken for WC.
        blended = texture["albedo"].copy()
        blended[backMaterials["validMask"]] = backMaterials["diffuse"][backMaterials["validMask"]]
        backMaterials["baseWithWildCap"] = blended
        np.savez_compressed(uffDir / "materials.npz", **backMaterials)
        writeImage(uffDir / "baseWithWildCap.png", blended)
        writeImage(uffDir / "refinedCoverage.png", backMaterials["validMask"])
        diagnostics = {**result["diagnostics"],
                       "registrationResultSha256": forwardAssets.manifest["resultSha256"],
                       "refinedUffPixelCount": int(backMaterials["validMask"].sum()),
                       "atlasTransfer": transferDiagnostics,
                       "inputColorInterpretation": "OpenDelight native RGB assumed gamma-2.2 for this experiment",
                       "normalTransferStatus": "blockedOnSourceConventionVerification",
                       "roughnessStatus": "notEstimated", "originalUffGeometryPreserved": True,
                       "fallbackColorEncoding": "OpenDelightNative; mixed base is a preview, not calibrated PBR"}
        writeJson(ictDir / "diagnostics.json", diagnostics)
        return diagnostics
