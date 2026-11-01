"""CPU mocks exercise normal axes, augmentation and strict model lifecycle."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import faceAppearance.adapters.normalPrior as normalModule
from faceAppearance.adapters.normalPrior import NormalPriorAdapter, validateNormalImages


class ProbeModel(torch.nn.Module):
    def __init__(self, outputs=None, failCall=None):
        super().__init__()
        self.register_buffer("probe", torch.zeros(1))
        self.outputs = outputs
        self.failCall = failCall
        self.calls = []
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        assert str(device) == "cpu", "Tests must not initialize a GPU"
        return super().to(device)

    def forward(self, batch):
        self.calls.append(batch["tar_rgb"].clone())
        if self.failCall == len(self.calls):
            raise RuntimeError("synthetic prediction failure")
        if self.outputs is None:
            normals = torch.zeros((1, 1, 3, 512, 512))
            normals[:, :, 1] = -2
        else:
            normals = self.outputs[(len(self.calls) - 1) % len(self.outputs)].clone()
        return {"normals": normals}, None


class StubAdapter(NormalPriorAdapter):
    def __init__(self, projectRoot, model=None, **options):
        super().__init__(projectRoot, **options)
        self.testModel = model if model is not None else ProbeModel()
        self.loads = 0

    def loadNormalPredictor(self, checkpointPath):
        self.loads += 1
        return self.testModel


def makeCheckpoint(directory):
    path = Path(directory) / "assets/weights/pixel3dmm/normals.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mock normal checkpoint")
    return path


def encodedMirrored(raw):
    result = torch.flip(raw, dims=[4]).clone()
    result[:, :, 0] *= -1
    return result


def testFlipUsesSpatialWidthAndXSignBeforeFlameAxisTransform(tmp_path):
    makeCheckpoint(tmp_path)
    raw = torch.empty((1, 1, 3, 512, 512))
    raw[:, :, 0] = torch.linspace(-2, 2, 512)[None, :]
    raw[:, :, 1] = -3
    raw[:, :, 2] = torch.linspace(1, 2, 512)[:, None]
    model = ProbeModel([raw, encodedMirrored(raw)])
    adapter = StubAdapter(tmp_path, model, device="cpu").loadModels()
    image = np.zeros((1, 512, 512, 3), np.float32)
    image[0, :, :, 0] = np.linspace(0, 1, 512)[None, :]
    before = image.copy()
    result = adapter.predict(image)
    expected = raw[0, 0].numpy().transpose(1, 2, 0).copy()
    expected = expected[..., [0, 2, 1]] * [1, -1, -1]
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(result["normals"][0], expected, atol=2e-7)
    np.testing.assert_array_equal(model.calls[0][0, 0].numpy(), image[0])
    np.testing.assert_array_equal(model.calls[1][0, 0].numpy(), image[0, :, ::-1])
    np.testing.assert_array_equal(image, before)
    assert result["validMask"].all()
    assert result["diagnostics"]["cameraRotationApplied"] is False


def testCanonicalAxisSignsMatchOfficialPngDecodeWithoutQuantization(tmp_path):
    makeCheckpoint(tmp_path)
    raw = torch.zeros((1, 1, 3, 512, 512))
    raw[:, :, 0, :, :170] = 1
    raw[:, :, 1, :, 170:340] = 1
    raw[:, :, 2, :, 340:] = 1
    result = StubAdapter(tmp_path, ProbeModel([raw, encodedMirrored(raw)]), device="cpu").loadModels().predict(
        np.zeros((1, 512, 512, 3), np.float32))
    np.testing.assert_array_equal(result["normals"][0, 0, [0, 200, 400]], [[1, 0, 0], [0, 0, -1], [0, -1, 0]])


def testZeroNonfiniteAndFlipCancellationAreInvalidRatherThanManufacturedNormals(tmp_path):
    makeCheckpoint(tmp_path)
    raw = torch.zeros((1, 1, 3, 512, 512))
    raw[:, :, 1] = -1
    raw[:, :, :, 2, 3] = 0
    raw[:, :, 0, 4, 5] = float("nan")
    secondUnflipped = raw.clone()
    secondUnflipped[:, :, :, 6, 7] *= -1
    second = encodedMirrored(secondUnflipped)
    result = StubAdapter(tmp_path, ProbeModel([raw, second]), device="cpu").loadModels().predict(
        np.zeros((1, 512, 512, 3), np.float32))
    for row, column in [(2, 3), (4, 5), (6, 7)]:
        assert not result["validMask"][0, row, column]
        np.testing.assert_array_equal(result["normals"][0, row, column], 0)
    assert np.isfinite(result["normals"]).all()
    assert result["diagnostics"]["views"][0]["nonfinitePixelCount"] == 1
    assert result["diagnostics"]["views"][0]["zeroOrCancelledPixelCount"] == 2


def testPreloadIsIdempotentAndPredictionsUseSingleViewsWithoutRetainedCache(tmp_path):
    makeCheckpoint(tmp_path)
    adapter = StubAdapter(tmp_path, device="cpu")
    images = np.zeros((2, 512, 512, 3), np.float64)
    with pytest.raises(RuntimeError, match="loadModels"):
        adapter.predict(images)
    adapter.loadModels().loadModels()
    result = adapter.predict(images)
    assert adapter.loads == 1
    assert len(adapter.model.calls) == 4
    assert all(tuple(item.shape) == (1, 1, 512, 512, 3) for item in adapter.model.calls)
    assert all(item.dtype == torch.float32 for item in adapter.model.calls)
    assert result["normals"].dtype == np.float32 and result["validMask"].dtype == bool
    assert result["diagnostics"]["returnedArrayBytes"] == 2 * 512 * 512 * 13
    assert result["diagnostics"]["maximumReturnedArrayBytes"] == 52 * 1024 * 1024
    assert result["diagnostics"]["retainedPredictionCacheBytes"] == 0
    result["normals"][:] = 8
    result["diagnostics"]["provenance"]["checkpointSha256"] = "changed"
    repeated = adapter.predict(images[:1])
    np.testing.assert_array_equal(repeated["normals"][0, 0, 0], [0, 0, 1])
    assert repeated["diagnostics"]["provenance"]["checkpointSha256"] != "changed"


def testStagedCpuPreloadDoesNotRequireCudaButInferenceChecksAvailability(tmp_path, monkeypatch):
    makeCheckpoint(tmp_path)
    def forbidProbe():
        pytest.fail("Staged preload must not inspect or initialize CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidProbe)
    adapter = StubAdapter(tmp_path, device="cuda").loadModels()
    assert adapter.model.moves == ["cpu"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        adapter.predict(np.zeros((1, 512, 512, 3), np.float32))


def testPredictionFailureRestoresStagedResidency(tmp_path):
    makeCheckpoint(tmp_path)
    model = ProbeModel(failCall=2)
    adapter = StubAdapter(tmp_path, model, device="cpu").loadModels()
    with pytest.raises(RuntimeError, match="synthetic"):
        adapter.predict(np.zeros((1, 512, 512, 3), np.float32))
    assert model.moves == ["cpu", "cpu", "cpu"]
    assert adapter.isLoaded


def testResidentModeDoesNotRestageAfterPrediction(tmp_path):
    makeCheckpoint(tmp_path)
    model = ProbeModel()
    StubAdapter(tmp_path, model, device="cpu", memoryPolicy="resident").loadModels().predict(
        np.zeros((1, 512, 512, 3), np.float32))
    assert model.moves == ["cpu", "cpu"]


def testMissingAndCorruptedAssetsFailBeforeModelLoad(tmp_path):
    adapter = StubAdapter(tmp_path, device="cpu")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        adapter.loadModels()
    assert adapter.loads == 0
    path = makeCheckpoint(tmp_path)
    manifest = {"checkpoint": {"path": path.name, "sha256": "wrong", "sizeBytes": path.stat().st_size}}
    path.with_name("manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        adapter.loadModels()
    assert adapter.loads == 0 and not adapter.isLoaded
    manifest["checkpoint"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("manifest.json").write_text(json.dumps(manifest))
    adapter.loadModels()
    assert adapter.provenance["manifestVerified"]


@pytest.fixture
def strictLoadSetup(tmp_path, monkeypatch):
    source = tmp_path / "thirdParty/pixel3dmm/src/pixel3dmm/lightning/p3dmm_network.py"
    source.parent.mkdir(parents=True)
    source.write_text("# mocked independently loaded network\n")
    path = makeCheckpoint(tmp_path)
    class FakeNetwork(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(3))
    class FakeDino:
        @staticmethod
        def _build_dino(name):
            raise AssertionError("No pretrained download is allowed")
    module = SimpleNamespace(__file__=str(source), Network=FakeNetwork, DinoWrapper=FakeDino)
    importModule = normalModule.importlib.import_module
    monkeypatch.setattr(normalModule.importlib, "import_module", lambda name: module if name == "pixel3dmm.lightning.p3dmm_network" else importModule(name))
    checkpoint = {"hyper_parameters": {"cfg": {"model": {"feature_map_type": "DINO", "prediction_type": "normals"}}},
                  "state_dict": {"net.weight": torch.tensor([1., 2., 3.]), "trainingOnly": torch.tensor(42.)}}
    return tmp_path, path, checkpoint, module


def testOnlyNormalNetworkStateLoadsStrictlyAndModelIsFrozen(strictLoadSetup):
    root, path, checkpoint, module = strictLoadSetup
    originalBuilder = module.DinoWrapper._build_dino
    torch.save(checkpoint, path)
    adapter = NormalPriorAdapter(root, device="cpu").loadModels()
    np.testing.assert_array_equal(adapter.model.weight.detach().numpy(), [1., 2., 3.])
    assert not adapter.model.training and not adapter.model.weight.requires_grad
    assert adapter.provenance["strictNetworkState"] and adapter.provenance["networkTensorCount"] == 1
    assert adapter.provenance["modelStorageBytes"] == 12
    assert module.DinoWrapper._build_dino is originalBuilder


def addUnusedUvBranch(module):
    baseNetwork = module.Network
    class NetworkWithUvEncoding(baseNetwork):
        def __init__(self, config):
            super().__init__(config)
            self.dir_norm_uv = torch.nn.Module()
            self.dir_norm_uv.norm = torch.nn.LayerNorm(3)
            self.dir_norm_uv.mlp = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(3, 6))
            self.uv_enc_mlp = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(24, 3))

        def forward(self, values):
            # This represents the real forward's disabled UV branch: accessing
            # removed modules would raise instead of silently producing output.
            if getattr(self, "useUv", False):
                values = self.dir_norm_uv.norm(values)
            return values * self.weight
    module.Network = NetworkWithUvEncoding


@pytest.mark.parametrize("explicitDisabled", [False, True])
def testCompleteAbsentDisabledUvBranchIsRemovedBeforeStrictLoad(strictLoadSetup, explicitDisabled):
    root, path, checkpoint, module = strictLoadSetup
    addUnusedUvBranch(module)
    if explicitDisabled:
        checkpoint["hyper_parameters"]["cfg"]["model"]["use_uv_enc"] = False
    torch.save(checkpoint, path)
    adapter = NormalPriorAdapter(root, device="cpu").loadModels()
    assert not hasattr(adapter.model, "dir_norm_uv") and not hasattr(adapter.model, "uv_enc_mlp")
    assert adapter.provenance["removedAbsentDisabledModules"] == ["dir_norm_uv", "uv_enc_mlp"]
    assert adapter.provenance["strictNetworkState"]
    torch.testing.assert_close(adapter.model(torch.ones(3)), torch.tensor([1., 2., 3.]))


@pytest.mark.parametrize("issue", ["enabled", "partialBranch", "missingActive", "unexpectedActive", "unknownBranch"])
def testUvCompatibilityNeverSuppressesIncompleteOrActiveStateErrors(strictLoadSetup, issue):
    root, path, checkpoint, module = strictLoadSetup
    addUnusedUvBranch(module)
    if issue == "enabled":
        checkpoint["hyper_parameters"]["cfg"]["model"]["use_uv_enc"] = True
    elif issue == "partialBranch":
        checkpoint["state_dict"]["net.uv_enc_mlp.1.bias"] = torch.zeros(3)
    elif issue == "missingActive":
        checkpoint["state_dict"].pop("net.weight")
    elif issue == "unexpectedActive":
        checkpoint["state_dict"]["net.extra"] = torch.zeros(1)
    else:
        baseNetwork = module.Network
        class UnknownBranchNetwork(baseNetwork):
            def __init__(self, config):
                super().__init__(config)
                self.uv_enc_mlp.register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))
        module.Network = UnknownBranchNetwork
    torch.save(checkpoint, path)
    adapter = NormalPriorAdapter(root, device="cpu")
    with pytest.raises((ValueError, RuntimeError)):
        adapter.loadModels()
    assert not adapter.isLoaded and adapter.model is None


def testCompleteSavedDisabledUvBranchIsStillStrictlyRestored(strictLoadSetup):
    root, path, checkpoint, module = strictLoadSetup
    addUnusedUvBranch(module)
    expectedModel = module.Network(checkpoint["hyper_parameters"]["cfg"])
    expectedState = expectedModel.state_dict()
    checkpoint["state_dict"].update({"net." + name: value for name, value in expectedState.items()
                                     if name.startswith(("dir_norm_uv.", "uv_enc_mlp."))})
    torch.save(checkpoint, path)
    adapter = NormalPriorAdapter(root, device="cpu").loadModels()
    assert adapter.provenance["removedAbsentDisabledModules"] == []
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"]["net." + name])


@pytest.mark.parametrize("issue", ["missing", "extra", "wrongShape", "uvCheckpoint", "disentangled", "wrongBackbone"])
def testIncompatibleCheckpointCannotBecomeLoaded(strictLoadSetup, issue):
    root, path, checkpoint, module = strictLoadSetup
    if issue == "missing":
        checkpoint["state_dict"].pop("net.weight")
    elif issue == "extra":
        checkpoint["state_dict"]["net.extra"] = torch.ones(1)
    elif issue == "wrongShape":
        checkpoint["state_dict"]["net.weight"] = torch.ones(4)
    elif issue == "uvCheckpoint":
        checkpoint["hyper_parameters"]["cfg"]["model"]["prediction_type"] = "uv_map"
    elif issue == "disentangled":
        checkpoint["hyper_parameters"]["cfg"]["model"]["pred_disentangled"] = True
    else:
        checkpoint["hyper_parameters"]["cfg"]["model"]["feature_map_type"] = "sapiens"
    torch.save(checkpoint, path)
    adapter = NormalPriorAdapter(root, device="cpu")
    with pytest.raises((ValueError, RuntimeError)):
        adapter.loadModels()
    assert not adapter.isLoaded and adapter.model is None


def testWrongCachedSourceCheckoutFailsInsteadOfSilentlyUsingAnotherNetwork(strictLoadSetup):
    root, path, checkpoint, module = strictLoadSetup
    torch.save(checkpoint, path)
    module.__file__ = str(root / "otherNetwork.py")
    with pytest.raises(RuntimeError, match="different source checkout"):
        NormalPriorAdapter(root, device="cpu").loadModels()


@pytest.mark.parametrize("shape,dtype,value", [((0, 512, 512, 3), np.float32, 0),
    ((17, 512, 512, 3), np.float32, 0), ((1, 3, 512, 512), np.float32, 0),
    ((1, 256, 256, 3), np.float32, 0), ((1, 512, 512, 3), np.uint8, 0),
    ((1, 512, 512, 3), np.float32, np.nan), ((1, 512, 512, 3), np.float32, 1.01),
    ((1, 512, 512, 3), np.float32, -.01)])
def testInvalidInputsAreRejected(shape, dtype, value):
    with pytest.raises(ValueError):
        validateNormalImages(np.full(shape, value, dtype=dtype))


def testBadOutputShapeFailsWithCleanup(tmp_path):
    makeCheckpoint(tmp_path)
    model = ProbeModel([torch.zeros((1, 1, 2, 512, 512))])
    adapter = StubAdapter(tmp_path, model, device="cpu").loadModels()
    with pytest.raises(ValueError, match="shaped"):
        adapter.predict(np.zeros((1, 512, 512, 3), np.float32))
    assert model.moves[-1] == "cpu"
