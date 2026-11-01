"""Explicit, serializable settings shared by CLI, experiments and service."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json

from .faceCrop import validateCropOptions


@dataclass(frozen=True)
class PipelineConfig:
    projectRoot: Path
    workingResolution: int = 1024
    device: str = "cuda"
    memoryPolicy: str = "staged"
    depthChunkSize: int = 1
    useEnhancer: bool = True
    colorBalance: bool = True
    textureSampling: str = "denseUv"
    maxCameraErrorPixels: float = 12.0
    maxViews: int = 16
    minViews: int = 4
    seed: int = 2026
    enableWildCap: bool = False
    registrationAsset: str = "assets/registration/wrapV1"
    wildCapResolution: int = 512
    wildCapIterations: int = 599
    inputCropPolicy: str = "faceSquare"
    faceCropMargin: float = 1.35
    enableAssembly: bool = False
    assemblyAsset: str = "assets/clientTemplates/legacyFemaleV1"
    assemblySourceStage: str = "fusion"
    assemblyColorEncoding: str = "gamma2.2"
    legacyMaleAssemblyMode: str = "uvTransfer"
    enableRenderedJawRefinement: bool = False
    colmapExecutable: str = ".runtimes/colmap/bin/colmap"
    warmupEnabled: bool = False
    warmupImages: tuple[str, ...] = ()
    warmupGender: str = "male"

    def __post_init__(self):
        object.__setattr__(self, "projectRoot", Path(self.projectRoot).resolve())
        if not isinstance(self.warmupEnabled, bool):
            raise ValueError("warmupEnabled must be boolean")
        if (not isinstance(self.warmupImages, (tuple, list))
                or any(not isinstance(path, str) or not path.strip() for path in self.warmupImages)):
            raise ValueError("warmupImages must be a list of local image path strings")
        object.__setattr__(self, "warmupImages", tuple(self.warmupImages))
        if self.warmupEnabled and (len(self.warmupImages) != 4 or len(set(self.warmupImages)) != 4):
            raise ValueError("Enabled startup warmup requires four distinct local image paths")
        if self.warmupGender not in ("male", "female"):
            raise ValueError("warmupGender must be male or female")
        validateCropOptions(self.inputCropPolicy, self.faceCropMargin)
        if self.workingResolution not in (512, 1024):
            raise ValueError("workingResolution must be 512 or 1024")
        if self.wildCapResolution not in (512, 1024):
            raise ValueError("wildCapResolution must be 512 or 1024")
        if not 1 <= self.wildCapIterations <= 1000:
            raise ValueError("wildCapIterations must be between 1 and 1000")
        if self.memoryPolicy not in ("staged", "resident"):
            raise ValueError("memoryPolicy must be staged or resident")
        if not 1 <= self.depthChunkSize <= 16:
            raise ValueError("depthChunkSize must be between 1 and 16")
        if self.textureSampling not in {"projection", "denseUv"}:
            raise ValueError("textureSampling must be projection or denseUv")
        if not 4 <= self.minViews <= self.maxViews <= 16:
            raise ValueError("The prototype supports 4–16 views")
        if not 0 < self.maxCameraErrorPixels <= 100:
            raise ValueError("maxCameraErrorPixels must be in (0, 100]")
        if not isinstance(self.enableAssembly, bool):
            raise ValueError("enableAssembly must be boolean")
        if not isinstance(self.assemblyAsset, str) or not self.assemblyAsset.strip():
            raise ValueError("assemblyAsset must name an asset directory")
        if self.assemblySourceStage not in {"fusion", "wildcap"}:
            raise ValueError("assemblySourceStage must be fusion or wildcap")
        if self.assemblyColorEncoding not in {"linear", "srgb", "gamma2.2"}:
            raise ValueError("assemblyColorEncoding must be linear, srgb or gamma2.2")
        if self.legacyMaleAssemblyMode not in {"uvTransfer", "directColmap"}:
            raise ValueError("legacyMaleAssemblyMode must be uvTransfer or directColmap")
        if not isinstance(self.enableRenderedJawRefinement, bool):
            raise ValueError("enableRenderedJawRefinement must be boolean")
        if self.enableRenderedJawRefinement and self.legacyMaleAssemblyMode != "directColmap":
            raise ValueError("Rendered jaw refinement requires directColmap")
        if not isinstance(self.colmapExecutable, str) or not self.colmapExecutable.strip():
            raise ValueError("colmapExecutable must name an executable")
        if self.enableAssembly and self.assemblySourceStage == "wildcap" and not self.enableWildCap:
            raise ValueError("WildCap assembly source requires enableWildCap=true")

    def toDict(self):
        result = asdict(self)
        result["projectRoot"] = str(self.projectRoot)
        return result

    @classmethod
    def fromFile(cls, path: Path, projectRoot: Path | None = None):
        values = json.loads(Path(path).read_text())
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if projectRoot is not None:
            values["projectRoot"] = projectRoot
        elif "projectRoot" not in values:
            values["projectRoot"] = Path(path).resolve().parent.parent
        elif not Path(values["projectRoot"]).is_absolute():
            values["projectRoot"] = Path(path).resolve().parent / values["projectRoot"]
        return cls(**values)
