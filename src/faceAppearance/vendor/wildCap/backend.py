"""Load only inference dependencies without importing the upstream CLI.

All source modules stay in 03_WildCap. Private package names prevent collisions
with OpenDelight's and UVFaceFusion's generic ``utils``/``models`` imports.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

loadLock = threading.RLock()


def writeUtf8IfChanged(path, content):
    """Torch 2.3's JIT helper uses the process locale for source-path text.

    Some native vision libraries change LC_CTYPE to C during inference. Keep
    Chinese workspace paths valid without changing the application's locale.
    """
    path = Path(path)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def makePackage(moduleName: str, sourcePath: Path) -> types.ModuleType:
    if moduleName not in sys.modules:
        module = types.ModuleType(moduleName)
        module.__path__ = [str(sourcePath)]
        module.__package__ = moduleName
        module.__spec__ = importlib.util.spec_from_loader(moduleName, loader=None, is_package=True)
        sys.modules[moduleName] = module
    return sys.modules[moduleName]


def loadBackend(sourceRoot: Path) -> types.SimpleNamespace:
    """Resolve local network/rendering code; no model loading or CUDA calls."""
    sourceRoot = Path(sourceRoot).resolve()
    sourceId = hashlib.sha256(str(sourceRoot).encode()).hexdigest()[:12]
    packageName = f"faceAppearanceWildCap_{sourceId}"
    with loadLock:
        makePackage(packageName, sourceRoot)
        # Bypass package __init__ files that import evaluation-only CLIP/FID and
        # transformer architectures. The bundled checkpoint uses image_v1 only.
        for suffix in ("k_diffusion", "k_diffusion.models", "utils", "splitsum"):
            makePackage(f"{packageName}.{suffix}", sourceRoot / suffix.replace(".", "/"))
        models = sys.modules[f"{packageName}.k_diffusion.models"]
        imageModel = importlib.import_module(f"{packageName}.k_diffusion.models.image_v1")
        models.ImageDenoiserModelV1 = imageModel.ImageDenoiserModelV1
        config = importlib.import_module(f"{packageName}.k_diffusion.config")
        layers = importlib.import_module(f"{packageName}.k_diffusion.layers")
        sampling = importlib.import_module(f"{packageName}.k_diffusion.sampling")
        resizer = importlib.import_module(f"{packageName}.utils.resizer")
        renderer = importlib.import_module(f"{packageName}.utils.mesh_renderer")

        def prepareRenderPlugin():
            import torch.utils.cpp_extension as extension

            operations = importlib.import_module(f"{packageName}.splitsum.renderutils.ops")
            # Scope the encoding compatibility patch strictly to serialized JIT
            # preparation, restoring the Torch helper even if compilation fails.
            with loadLock, patch.object(extension, "_maybe_write", writeUtf8IfChanged):
                operations._get_plugin()

        lightingName = f"{packageName}.utils.lighting"
        if lightingName not in sys.modules:
            lightingPath = sourceRoot / "utils/lighting.py"
            syntaxTree = ast.parse(lightingPath.read_text(), filename=str(lightingPath))
            expectedImports = {"splitsum": 2, "utils.resizer": 1}
            seenImports = {name: 0 for name in expectedImports}
            for node in syntaxTree.body:
                if isinstance(node, ast.ImportFrom) and node.module in expectedImports:
                    seenImports[node.module] += 1
                    node.module = f"{packageName}.{node.module}"
            if seenImports != expectedImports:
                raise RuntimeError("WildCap lighting imports changed; audit the backend adapter")
            module = types.ModuleType(lightingName)
            module.__file__ = str(lightingPath)
            module.__package__ = f"{packageName}.utils"
            exec(compile(syntaxTree, str(lightingPath), "exec"), module.__dict__)
            sys.modules[lightingName] = module
        return types.SimpleNamespace(config=config, layers=layers, sampling=sampling,
                                     Resizer=resizer.Resizer, MeshRenderer=renderer.MeshRenderer,
                                     TexelGridLighting=sys.modules[lightingName].TexelGridLighting,
                                     prepareRenderPlugin=prepareRenderPlugin)
