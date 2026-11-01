"""Assembly orchestration contracts, without inference models or GPU loading."""
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image
import pytest

from faceAppearance.config import PipelineConfig
from faceAppearance.pipeline import Pipeline


def installModule(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


def geometryFixture():
    return {"vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
            "faces": np.array([[0, 1, 2]]), "uvs": np.array([[0, 0], [1, 0], [0, 1]], np.float32),
            "faceMasks": np.ones((4, 4, 4), bool), "viewTransforms": [{"index": index} for index in range(4)]}


def testAssemblyAssetsPreloadBeforeModelsAndOnlyOnce(monkeypatch, tmp_path):
    events = []

    class Assets:
        def __init__(self, path):
            assert path == tmp_path/"assets/clientTemplates/legacyFemaleV1"
            events.append("assets")

    class Geometry:
        canonicalUvs = geometryFixture()["uvs"]
        canonicalFaces = geometryFixture()["faces"]

        def __init__(self, *args, **kwargs):
            events.append("geometryConstructor")

        def loadModels(self):
            events.append("geometryModels")

    class Delight:
        def __init__(self, *args, **kwargs):
            events.append("delightConstructor")

        def loadModels(self):
            events.append("delightModels")

    installModule(monkeypatch, "faceAppearance.headAssembly", AssemblyAssets=Assets)
    installModule(monkeypatch, "faceAppearance.adapters.geometry", GeometryAdapter=Geometry)
    installModule(monkeypatch, "faceAppearance.adapters.delight", DelightAdapter=Delight)
    monkeypatch.setattr("faceAppearance.runtime.configureRuntime", lambda root: events.append("runtime"))
    monkeypatch.setattr("faceAppearance.texture.buildUvLookup", lambda *args: {})
    pipeline = Pipeline(PipelineConfig(tmp_path, device="cpu", enableAssembly=True))
    pipeline.loadModels()
    firstAssets = pipeline.assemblyAssets
    pipeline.loadModels()
    assert pipeline.assemblyAssets is firstAssets
    assert events == ["assets", "runtime", "geometryConstructor", "delightConstructor", "geometryModels", "delightModels"]


def testMissingAssemblyAssetFailsBeforeModelInitialization(monkeypatch, tmp_path):
    class MissingAssets:
        def __init__(self, path):
            raise FileNotFoundError("Client template manifest missing")

    installModule(monkeypatch, "faceAppearance.headAssembly", AssemblyAssets=MissingAssets)
    pipeline = Pipeline(PipelineConfig(tmp_path, device="cpu", enableAssembly=True))
    with pytest.raises(FileNotFoundError, match="manifest"):
        pipeline.loadModels()
    assert not pipeline.loaded
    assert pipeline.geometryAdapter is None and pipeline.delightAdapter is None


@pytest.fixture
def assemblyPipeline(monkeypatch, tmp_path):
    events, calls = [], []
    config = PipelineConfig(tmp_path, device="cpu", inputCropPolicy="centerSquare",
                            enableWildCap=True, enableAssembly=True)
    pipeline = Pipeline(config)
    geometry = geometryFixture()
    texture = {"albedo": np.full((3, 3, 3), .5, np.float32),
               "linearAlbedo": np.full((3, 3, 3), .25, np.float32),
               "coverageMask": np.array([[True, False, True], [True, True, True], [False, True, True]])}

    class Geometry:
        def reconstruct(self, images, **kwargs):
            events.append("geometry")
            return geometry

    class Delight:
        def predict(self, images, mask):
            events.append("delight")
            return {"albedoImages": images[:, ::128, ::128]*.5,
                    "validMasks": np.ones((len(images), 4, 4), bool), "diagnostics": {}}

    class Assets:
        def assemble(self, receivedGeometry, receivedTexture, *, colorEncoding):
            events.append("assembly")
            calls.append((receivedGeometry, receivedTexture, colorEncoding))
            return {"diagnostics": {"templateVertexOrderPreserved": True}, "templateId": "fixtureTemplate"}

    def export(output, assembled):
        events.append("export")
        assert output.name == "assembly"
        output.mkdir(parents=True)
        manifest = {"status": "exported", "templateId": assembled["templateId"]}
        (output/"manifest.json").write_text(json.dumps(manifest))
        return manifest

    def finishFusion(geometry, delight, images, output):
        events.append("fusion")
        np.savez_compressed(output/"texture.npz", **texture)
        return {"textureDiagnostics": {"fixture": True}}

    def refineMaterials(geometry, texture, output):
        events.append("wildcap")
        (output/"wildcapUff").mkdir()
        np.savez_compressed(output/"wildcapUff/materials.npz",
                            baseWithWildCap=np.full((3, 3, 3), .75, np.float32),
                            validMask=np.zeros((3, 3), bool))
        return {"fixture": True}

    pipeline.loaded = True
    pipeline.geometryAdapter, pipeline.delightAdapter = Geometry(), Delight()
    pipeline.assemblyAssets, pipeline.wildCapAdapter = Assets(), object()
    pipeline.finishFusion, pipeline.refineMaterials = finishFusion, refineMaterials
    installModule(monkeypatch, "faceAppearance.clientAsset", exportClientAsset=export)
    paths = []
    for index in range(4):
        path = tmp_path/f"photo{index}.png"
        Image.new("RGB", (80, 90), (20+index*15, 40, 60)).save(path)
        paths.append(path)
    return pipeline, paths, geometry, texture, events, calls


@pytest.mark.parametrize("stopAfter,sourceStage,expected", [
    ("fusion", "fusion", ["geometry", "delight", "fusion"]),
    ("wildcap", "fusion", ["geometry", "delight", "fusion", "wildcap"]),
    ("assembly", "fusion", ["geometry", "delight", "fusion", "assembly", "export"]),
    ("assembly", "wildcap", ["geometry", "delight", "fusion", "wildcap", "assembly", "export"]),
])
def testAssemblyRoutingPreservesExistingStagesAndUsesExplicitMaterialSource(assemblyPipeline, tmp_path,
                                                                          stopAfter, sourceStage, expected):
    pipeline, paths, geometry, texture, events, calls = assemblyPipeline
    pipeline.config = PipelineConfig(tmp_path, device="cpu", inputCropPolicy="centerSquare",
                                    enableWildCap=True, enableAssembly=True, assemblySourceStage=sourceStage)
    result = pipeline.run(paths, tmp_path/"result", stopAfter)
    assert events == expected
    assert "assembly" in result["availableStages"]
    with np.load(tmp_path/"result/texture.npz") as saved:
        for key in texture:
            np.testing.assert_array_equal(saved[key], texture[key])
    if stopAfter != "assembly":
        assert not calls
        assert "rigAndFullHeadAssembly" in result["pendingStages"]
        return
    usedGeometry, usedTexture, encoding = calls[0]
    assert usedGeometry is geometry
    np.testing.assert_array_equal(usedTexture["coverageMask"], texture["coverageMask"])
    if sourceStage == "fusion":
        assert encoding == "linear"
        np.testing.assert_array_equal(usedTexture["linearAlbedo"], texture["linearAlbedo"])
    else:
        assert encoding == "gamma2.2"
        assert "linearAlbedo" not in usedTexture
        np.testing.assert_array_equal(usedTexture["albedo"], .75)
        assert "not physically calibrated" in result["assemblyColorInterpretation"]
    assert result["clientAssetManifest"]["templateId"] == "fixtureTemplate"
    assert "clientRigValidation" in result["pendingStages"]
    assert "rigAndFullHeadAssembly" not in result["pendingStages"]


def testIndependentAssemblyUsesLoadedAssetsWithoutLoadingInferenceModels(assemblyPipeline, tmp_path):
    pipeline, _, geometry, texture, events, calls = assemblyPipeline
    pipeline.loaded = False
    report = pipeline.assembleAsset(geometry, texture, tmp_path/"offline", colorEncoding="srgb")
    assert events == ["assembly", "export"]
    assert calls[0][2] == "linear"
    assert report["manifest"]["status"] == "exported"
    assert not pipeline.loaded


def testDisabledOrUnloadedAssemblyFailsBeforeRunningModels(tmp_path):
    pipeline = Pipeline(PipelineConfig(tmp_path, device="cpu"))
    pipeline.loaded = True
    with pytest.raises(ValueError, match="enableAssembly"):
        pipeline.run([], tmp_path/"disabled", "assembly")
    pipeline.config = PipelineConfig(tmp_path, device="cpu", enableAssembly=True)
    with pytest.raises(RuntimeError, match="preloaded"):
        pipeline.run([], tmp_path/"notLoaded", "assembly")
    assert not (tmp_path/"disabled").exists() and not (tmp_path/"notLoaded").exists()


@pytest.mark.parametrize("settings", [{"enableAssembly": "yes"}, {"assemblyAsset": ""},
    {"assemblySourceStage": "postprocess"}, {"assemblyColorEncoding": "RGB"},
    {"enableAssembly": True, "assemblySourceStage": "wildcap", "enableWildCap": False}])
def testInvalidAssemblyConfigurationFailsClearly(tmp_path, settings):
    with pytest.raises(ValueError):
        PipelineConfig(tmp_path, **settings)


def testAssemblyConfigurationRoundTripsWithExplicitDefaults(tmp_path):
    config = PipelineConfig(tmp_path, enableAssembly=True, assemblyAsset="client/custom",
                            assemblyColorEncoding="srgb")
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config.toDict()))
    assert PipelineConfig.fromFile(path) == config
    default = PipelineConfig(tmp_path)
    assert not default.enableAssembly and default.assemblySourceStage == "fusion"


def testStandaloneAssembleCliLoadsOnlyCpuAssets(monkeypatch, tmp_path, capsys):
    from faceAppearance.cli import main
    config = PipelineConfig(tmp_path, device="cuda")
    configPath = tmp_path/"settings.json"
    configPath.write_text(json.dumps(config.toDict()))
    geometryPath, texturePath = tmp_path/"geometry.npz", tmp_path/"texture.npz"
    np.savez(geometryPath, vertices=np.eye(3))
    np.savez(texturePath, linearAlbedo=np.full((2, 2, 3), .2), coverageMask=np.ones((2, 2), bool))
    events = []

    class Assets:
        def __init__(self, directory):
            events.append("assets")

        def assemble(self, geometry, texture, *, colorEncoding):
            assert colorEncoding == "linear"
            events.append("assemble")
            return {}

    def export(output, assembled):
        assert output == tmp_path/"client"
        events.append("export")
        return {"status": "exported"}

    installModule(monkeypatch, "faceAppearance.headAssembly", AssemblyAssets=Assets)
    installModule(monkeypatch, "faceAppearance.clientAsset", exportClientAsset=export)
    monkeypatch.setattr(Pipeline, "loadModels", lambda self: pytest.fail("Offline assembly must not load inference models"))
    monkeypatch.setattr(sys, "argv", ["face-appearance", "--config", str(configPath), "assemble",
        "--geometry", str(geometryPath), "--texture", str(texturePath), "--output", str(tmp_path/"client")])
    main()
    assert events == ["assets", "assemble", "export"]
    assert json.loads(capsys.readouterr().out)["status"] == "exported"
