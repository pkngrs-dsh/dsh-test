"""Explicit, idempotent runtime setup for local upstream dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configureRuntime(projectRoot: Path) -> None:
    """Expose curated vendor packages and use project-local offline caches.

    Call before importing model packages. Existing caller environment variables
    take precedence. No working-directory changes, model downloads, cache file
    writes, or imports with model construction side effects happen here.
    """
    projectRoot = Path(projectRoot).resolve()
    if not projectRoot.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {projectRoot}")
    dependencyPaths = [projectRoot / "thirdParty/facer",
                       projectRoot / "thirdParty/pixel3dmm/src",
                       projectRoot / "thirdParty/ibug"]
    # Reverse insertion preserves the declared search order without duplicates.
    for dependencyPath in reversed(dependencyPaths):
        pathText = str(dependencyPath)
        if pathText not in sys.path:
            sys.path.insert(0, pathText)

    cacheRoot = projectRoot / ".cache"
    runtimeDefaults = {
        "FACE_APPEARANCE_ROOT": str(projectRoot),
        "PIXEL3DMM_CODE_BASE": str(projectRoot / "thirdParty/pixel3dmm"),
        "PIXEL3DMM_PREPROCESSED_DATA": str(cacheRoot / "pixel3dmm/preprocessed"),
        "PIXEL3DMM_TRACKING_OUTPUT": str(cacheRoot / "pixel3dmm/tracking"),
        "XDG_CACHE_HOME": str(cacheRoot),
        "TORCH_HOME": str(cacheRoot / "torch"),
        "HF_HOME": str(cacheRoot / "huggingface"),
        "MPLCONFIGDIR": str(cacheRoot / "matplotlib"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    for variableName, variableValue in runtimeDefaults.items():
        os.environ.setdefault(variableName, variableValue)
    # Derive the hub path from an explicit caller HF_HOME when one was supplied.
    os.environ.setdefault("HF_HUB_CACHE", os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    # Local prototype venv inherits its CUDA toolchain from the base conda
    # environment. Discover it through Python, without hard-coding that host's
    # path or replacing an explicitly configured system CUDA installation.
    basePrefix = Path(sys.base_prefix)
    baseBin = basePrefix / "bin"
    if (baseBin / "nvcc").is_file():
        pathEntries = os.environ.get("PATH", "").split(os.pathsep)
        if str(baseBin) not in pathEntries:
            os.environ["PATH"] = os.pathsep.join([str(baseBin), *pathEntries])
        os.environ.setdefault("CUDA_HOME", str(basePrefix))
        os.environ.setdefault("CUB_HOME", str(basePrefix / "include"))
        for variableName, executable in (("CC", "x86_64-conda-linux-gnu-gcc"),
                                          ("CXX", "x86_64-conda-linux-gnu-g++")):
            compilerPath = baseBin / executable
            if compilerPath.is_file():
                os.environ.setdefault(variableName, str(compilerPath))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cacheRoot / "torch_extensions"))
    os.environ.setdefault("MAX_JOBS", "2")
