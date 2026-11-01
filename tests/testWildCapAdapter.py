"""CPU checks of the WildCap contract, atlas invariants and request isolation.

These tests load no model weights and execute no CUDA work. Real GPU quality,
peak memory and upstream parity require an integration run on the local assets.
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceAppearance.adapters.wildcap import (
    WildCapAdapter, buildCoordinates, cropForResolution, readTemplate,
    lightingOptimizerForStep, validateAtlas, validateLightingSchedule, validateRequest,
)


def requestArguments():
    return {"vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
            "faces": np.array([[0, 1, 2]], np.int64),
            "uvs": np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32),
            "texture": np.full((512, 512, 3), 0.5, np.float32),
            "validMask": np.ones((512, 512), bool), "artifactMask": None,
            "workingResolution": 512, "iterations": 2, "seed": 7}


def templateFor(arguments):
    return {name: arguments[name].copy() for name in ("vertices", "faces", "uvs")} | {
        "uvFaces": arguments["faces"].copy()}


def testFixedCropScalesWithoutMovingAtlas():
    assert cropForResolution(1024) == (128, 896, 64, 832)
    assert cropForResolution(512) == (64, 448, 32, 416)
    with pytest.raises(ValueError, match="512 or 1024"):
        cropForResolution(768)


@pytest.mark.parametrize("field,value,message", [
    ("iterations", 0, "iterations"), ("iterations", 1001, "iterations"),
    ("iterations", True, "iterations"), ("seed", -1, "seed"),
    ("seed", 2 ** 63, "seed"), ("skinTone", [0, 0.5, 0.5], "skinTone"),
    ("lightingSchedule", "globalAlways", "lightingSchedule"),
])
def testInvalidOptionsFailBeforeModels(field, value, message):
    arguments = requestArguments()
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        validateRequest(**arguments)


def testImageRangeAndSoftMasksAreExplicit():
    arguments = requestArguments()
    arguments["texture"][0, 0] = 255
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validateRequest(**arguments)
    arguments["texture"][:] = 0.5
    arguments["validMask"] = np.full((512, 512), 0.8, np.float32)
    with pytest.raises(ValueError, match="binary"):
        validateRequest(**arguments)
    arguments["validMask"][:] = 0
    with pytest.raises(ValueError, match="no observations"):
        validateRequest(**arguments)


def testRequestOwnsItsArrays():
    arguments = requestArguments()
    request = validateRequest(**arguments)
    request["vertices"][0] = 100
    request["texture"][:] = 0
    request["validMask"][:] = False
    np.testing.assert_array_equal(arguments["vertices"][0], [0, 0, 0])
    assert arguments["texture"].mean() == 0.5
    assert arguments["validMask"].all()


@pytest.mark.parametrize("schedule,expected", [
    ("upstream", (101, 200, 298)), ("global301", (301, 0, 298)),
])
def testLightingScheduleBoundariesAndCounts(schedule, expected):
    selected = [lightingOptimizerForStep(step, schedule) for step in range(599)]
    assert tuple(selected.count(name) for name in ("global", "local", "frozen")) == expected
    assert selected[0] == selected[100] == "global"
    assert selected[101] == selected[300] == ("local" if schedule == "upstream" else "global")
    assert selected[301] == selected[-1] == "frozen"
    with pytest.raises(ValueError, match="nonnegative integer"):
        lightingOptimizerForStep(-1, schedule)
    assert validateRequest(**requestArguments())["lightingSchedule"] == "upstream"


def testGlobal301RejectsEffectiveLocalMaskBeforeCuda(tmp_path):
    pytest.importorskip("torch")
    arguments = requestArguments()
    arguments["artifactMask"] = np.zeros((512, 512), bool)
    arguments["artifactMask"][100, 100] = True
    adapter = WildCapAdapter(tmp_path, device="cpu")
    adapter.template, adapter.loaded = templateFor(arguments), True
    adapter.assets["shadowPrior"] = np.ones((512, 512, 3), np.float32)
    with pytest.raises(ValueError, match="empty effective local"):
        adapter.optimize(**arguments, lightingSchedule="global301")
    # The prior and fixed crop, rather than the raw mask alone, define activity.
    adapter.assets["shadowPrior"][100, 100] = 0
    with pytest.raises(ValueError, match="CUDA"):
        adapter.optimize(**arguments, lightingSchedule="global301")
    arguments["artifactMask"][:] = False
    arguments["artifactMask"][0, 0] = True
    with pytest.raises(ValueError, match="CUDA"):
        adapter.optimize(**arguments, lightingSchedule="global301")
    validateLightingSchedule("upstream", np.ones((2, 2), bool))
    validateLightingSchedule("global301", np.zeros((2, 2), bool))


def testAtlasRejectsFlippedVButAllowsIndependentUvIndices():
    arguments = requestArguments()
    template = templateFor(arguments)
    request = validateRequest(**arguments)
    validateAtlas(request, template)
    request["uvs"][:, 1] = 1 - request["uvs"][:, 1]
    with pytest.raises(ValueError, match="UV corners"):
        validateAtlas(request, template)
    arguments["uvs"] = arguments["uvs"][[2, 0, 1]]
    arguments["uvFaces"] = np.array([[1, 2, 0]])
    validateAtlas(validateRequest(**arguments), template)


def testAtlasAllowsShapeChangeButRejectsTopologyMutation():
    arguments = requestArguments()
    template = templateFor(arguments)
    request = validateRequest(**arguments)
    request["vertices"] = request["vertices"] * 10 + 25
    validateAtlas(request, template)
    request["faces"] = request["faces"][:, ::-1]
    with pytest.raises(ValueError, match="ordering"):
        validateAtlas(request, template)


def testTemplatePreservesObjSeamIndices(tmp_path):
    path = tmp_path / "triangle.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                    "vt 0.9 0.1\nvt 0.1 0.9\nvt 0.1 0.1\n"
                    "f 1/3 2/1 3/2\n")
    template = readTemplate(path)
    np.testing.assert_array_equal(template["faces"], [[0, 1, 2]])
    np.testing.assert_array_equal(template["uvFaces"], [[2, 0, 1]])
    np.testing.assert_allclose(template["uvs"][template["uvFaces"]],
                               [[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]])


def testCoordinateEncodingMatchesTrainingPixelDefinition():
    torch = pytest.importorskip("torch")
    coordinates = buildCoordinates(512)
    assert coordinates.shape == (1, 18, 512, 512)
    points = torch.tensor([-0.5, -0.75])
    expected = torch.cat([points, *[values for frequency in (1, 2, 4, 8)
                                   for values in (torch.sin(points * frequency),
                                                  torch.cos(points * frequency))]])
    torch.testing.assert_close(coordinates[0, :, 64, 128], expected)
    assert coordinates[0, 0, 0, -1] == 1 - 2 / 512


def testMissingAssetsNeverDownload(tmp_path):
    with pytest.raises(FileNotFoundError, match="ckpt.pth"):
        WildCapAdapter(tmp_path, device="cpu").loadModels()


def testPreloadIsExplicitAndIdempotent(tmp_path):
    adapter = WildCapAdapter(tmp_path, device="cpu")
    with pytest.raises(RuntimeError, match="loadModels"):
        adapter.optimize(**requestArguments())
    sentinel = object()
    adapter.model, adapter.loaded = sentinel, True
    assert adapter.loadModels() is adapter
    assert adapter.model is sentinel


def testFrameMustBeDeclaredAndCudaIsRequired(tmp_path):
    adapter = WildCapAdapter(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="geometryFrame"):
        adapter.optimize(**requestArguments(), geometryFrame="estimatedHiddenTransform")
    adapter.template, adapter.loaded = templateFor(requestArguments()), True
    with pytest.raises(ValueError, match="CUDA"):
        adapter.optimize(**requestArguments(), geometryFrame="uvFaceFusionCanonical")


@pytest.mark.parametrize("shouldFail", [False, True])
def testRequestStateAndStagedOffloadAreIndependent(tmp_path, monkeypatch, shouldFail):
    torch = pytest.importorskip("torch")
    # Replace CUDA context entry, not the sampler: this test exercises only the
    # request owner lifecycle and never dispatches a tensor to the accelerator.
    monkeypatch.setattr(torch.cuda, "device", lambda value: nullcontext())
    monkeypatch.setattr(torch.random, "fork_rng", lambda: nullcontext())
    monkeypatch.setattr(torch, "manual_seed", lambda value: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    adapter = WildCapAdapter(tmp_path, device="cuda", memoryPolicy="staged")
    adapter.template, adapter.loaded = templateFor(requestArguments()), True
    adapter.backend = SimpleNamespace(prepareRenderPlugin=lambda: None)
    movements, requests = [], []
    monkeypatch.setattr(adapter, "moveModels", movements.append)

    def run(request):
        requests.append(request)
        request["texture"][:] = 0
        if shouldFail:
            raise RuntimeError("optimization failed")
        return {"diagnostics": {"frame": request["geometryFrame"]}}

    monkeypatch.setattr(adapter, "runOptimization", run)
    arguments = requestArguments()
    if shouldFail:
        with pytest.raises(RuntimeError, match="optimization failed"):
            adapter.optimize(**arguments)
    else:
        adapter.optimize(**arguments, geometryFrame="uvFaceFusionCanonical")
        adapter.optimize(**arguments, geometryFrame="uvFaceFusionCanonical")
        assert requests[0] is not requests[1]
    assert arguments["texture"].mean() == 0.5
    assert movements == (["cuda", "cpu"] if shouldFail else ["cuda", "cpu", "cuda", "cpu"])


def testOutputColorNormalAndCoverageConventions(tmp_path):
    torch = pytest.importorskip("torch")
    adapter = WildCapAdapter(tmp_path, device="cpu")
    arguments = requestArguments()
    request = validateRequest(**arguments)
    request["geometryFrame"] = "ictNative"
    x1, x2, y1, y2 = cropForResolution(512)
    shape = (1, 1, y2 - y1, x2 - x1)
    faceMask = torch.ones(shape)
    observedMask = faceMask.clone()
    observedMask[..., :10, :] = 0
    state = {"faceMask": faceMask, "observedMask": observedMask,
             "skinMask": observedMask, "skinTone": torch.tensor([0.4, 0.5, 0.6]),
             "shadowMask": torch.zeros(shape), "edgeMask": torch.zeros(shape),
             "crop": (x1, x2, y1, y2)}
    prediction = torch.zeros((1, 7, shape[-2], shape[-1]))
    prediction[:, 5] = 2  # Encoded tangent +Z may have arbitrary length before normalization.
    result = adapter.finishResult(prediction, state, request, [])
    np.testing.assert_allclose(result["diffuse"][100, 100], [0.4, 0.5, 0.6], atol=1e-5)
    np.testing.assert_array_equal(result["tangentNormal"][100, 100], [0, 0, 1])
    assert result["specular"][100, 100] == 0.5
    assert result["diffuse"].shape == (512, 512, 3)
    assert result["inferredMask"][y1:y1 + 10, x1:x2].all()
    assert not result["validMask"][y1:y1 + 10, x1:x2].any()
    assert not result["generatedMask"][:y1].any()
    assert result["diagnostics"]["geometryTransformsApplied"] == []
    assert result["diagnostics"]["lightingMode"] == "globalOnly"
    assert result["diagnostics"]["localLightingTexelCount"] == 0
    assert result["effectiveLocalMask"].shape == (512, 512)
    assert not result["effectiveLocalMask"].any()
    assert result["skinMask"].shape == (512, 512)
    assert "localTexelGridLightingInactiveWithoutEffectiveArtifactMask" in result["diagnostics"]["warnings"]
    state["shadowMask"][..., 20:30, 20:30] = 1
    localResult = adapter.finishResult(prediction, state, request, [])
    assert localResult["diagnostics"]["lightingMode"] == "globalAndLocal"
    assert localResult["diagnostics"]["localLightingTexelCount"] == 100
    assert localResult["effectiveLocalMask"].sum() == 100
    assert "localTexelGridLightingInactiveWithoutEffectiveArtifactMask" not in localResult["diagnostics"]["warnings"]
    state["observedMask"] = torch.zeros(shape)
    with pytest.raises(ValueError, match="no valid observed"):
        adapter.finishResult(prediction, state, request, [])


def testActualLocalAtlasLoadsWithoutChangingIndices():
    projectRoot = Path(__file__).resolve().parents[1]
    path = projectRoot / "assets/wrapPreparation/originals/open_eye_template.obj"
    if not path.exists():
        pytest.skip("Local ICT asset is optional outside this workspace")
    template = readTemplate(path)
    assert template["vertices"].shape == (6274, 3)
    assert template["faces"].shape == (12256, 3)
    assert template["uvFaces"].shape == template["faces"].shape


def testDppsExecutesRealGradientsAndIgnoresUnobservedGuidance(tmp_path, monkeypatch):
    """Exercise the real loop on a small CPU likelihood, without the giant prior."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from faceAppearance.vendor.wildCap.backend import loadBackend

    projectRoot = Path(__file__).resolve().parents[1]
    sourceRoot = projectRoot / "03_WildCap/wildcap"
    if not sourceRoot.is_dir():
        pytest.skip("Local WildCap equations are optional outside this workspace")
    backend = loadBackend(sourceRoot)

    class ScalarLight(torch.nn.Module):
        def __init__(self, **options):
            super().__init__()
            self.main_light = torch.nn.Linear(1, 1, bias=False)
            self.main_light.weight.data.fill_(0.8)
            self.sp_light = torch.nn.Parameter(torch.zeros(1))
            self.localMask = options["seg"]

        def forward(self, diffuse, normal, detach=False):
            shade = (self.main_light.weight.reshape(1, 1, 1, 1) + self.sp_light * self.localMask).expand_as(diffuse)
            if detach:
                shade = shade.detach()
            return diffuse * shade, shade, torch.full_like(shade, 0.8)

    adapter = WildCapAdapter(tmp_path, device="cpu")
    adapter.backend = SimpleNamespace(sampling=backend.sampling, TexelGridLighting=ScalarLight)
    adapter.modelConfig = {"sigma_min": 0.01, "sigma_max": 160}
    adapter.model = lambda image, sigma: image[:, :7] * 0.25
    observedMask = torch.ones(1, 1, 8, 8)
    observedMask[:, :, :4] = 0
    state = {"initial": torch.full((1, 7, 8, 8), 0.3), "coordinates": torch.zeros(1, 18, 8, 8),
             "objectNormal": torch.zeros(1, 3, 8, 8), "guidance": torch.full((1, 3, 8, 8), 0.5),
             "observedMask": observedMask, "faceMask": torch.ones(1, 1, 8, 8),
             "edgeMask": torch.zeros(1, 1, 8, 8), "shadowMask": torch.zeros(1, 1, 8, 8),
             "initialShade": torch.ones(3)}
    monkeypatch.setattr(adapter, "prepareState", lambda request: state)
    monkeypatch.setattr(adapter, "finishResult", lambda prediction, state, request, trace: {
        "prediction": prediction.clone(), "trace": trace})
    request = {"iterations": 3, "seed": 19}
    first = adapter.runOptimization(request)
    state["guidance"][:, :, :4] = 1000
    second = adapter.runOptimization(request)
    torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
    assert torch.isfinite(second["prediction"]).all()
    assert first["trace"][-1]["iteration"] == 3
    assert first["trace"][0]["photometricSum"] > 0
    assert first["trace"][-1]["optimizer"] == "global"
    assert first["trace"][-1]["globalOptimizerSteps"] == 3
    assert first["trace"][-1]["localOptimizerSteps"] == 0
    assert first["trace"][-1]["localGradientMaxAbs"] == 0


@pytest.mark.parametrize("withLocalMask", [False, True])
def testUpstreamTexelGridLocalGradientsRespectMask(monkeypatch, withLocalMask):
    """Use the real upstream SH/local composition, replacing only CUDA Split-Sum."""
    torch = pytest.importorskip("torch")
    from faceAppearance.vendor.wildCap.backend import loadBackend

    sourceRoot = Path(__file__).resolve().parents[1] / "03_WildCap/wildcap"
    if not sourceRoot.is_dir():
        pytest.skip("Local WildCap equations are optional outside this workspace")
    backend = loadBackend(sourceRoot)

    class CpuMainLight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.color = torch.nn.Parameter(torch.full((1, 3, 1, 1), 0.8))

        def build_mips(self):
            pass

        def shade_diffuse(self, normal):
            return self.color.expand_as(normal)

    monkeypatch.setattr(backend.TexelGridLighting, "compute_sh_ss_scale", lambda self: 1.0)
    monkeypatch.setattr(backend.TexelGridLighting, "set_main_light",
                        lambda self: setattr(self, "main_light", CpuMainLight()))
    mask = torch.full((1, 1, 8, 8), float(withLocalMask))
    lighting = backend.TexelGridLighting(device="cpu", seg=mask, resolution=4,
                                        main_light_type="splitsum")
    normal = torch.zeros(1, 3, 8, 8)
    normal[:, 2] = 1
    diffuse = torch.full_like(normal, 0.5)
    rendered, shade, mainShade = lighting(diffuse, normal)
    loss = (rendered - 0.5).square().sum() + torch.relu(shade - mainShade).square().sum()
    loss.backward()
    gradient = lighting.sp_light.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool(torch.any(gradient != 0)) is withLocalMask
    previous = lighting.sp_light.detach().clone()
    torch.optim.Adam([lighting.sp_light], lr=0.01).step()
    assert bool(torch.any(lighting.sp_light.detach() != previous)) is withLocalMask


def testLightingAuditCannotMutateMaterialOutputs(tmp_path):
    torch = pytest.importorskip("torch")
    adapter = WildCapAdapter(tmp_path, device="cpu")
    request = validateRequest(**requestArguments())
    request["geometryFrame"] = "ictNative"
    x1, x2, y1, y2 = cropForResolution(512)
    mask = torch.ones(1, 1, y2 - y1, x2 - x1)
    state = {"faceMask": mask, "observedMask": mask, "skinMask": mask,
             "skinTone": torch.tensor([0.4, 0.5, 0.6]), "shadowMask": torch.zeros_like(mask),
             "edgeMask": torch.zeros_like(mask), "crop": (x1, x2, y1, y2),
             "objectNormal": torch.zeros(1, 3, y2 - y1, x2 - x1),
             "guidance": torch.full((1, 3, y2 - y1, x2 - x1), 0.5)}
    prediction = torch.zeros((1, 7, y2 - y1, x2 - x1))
    prediction[:, 5] = 1
    reference = adapter.finishResult(prediction, state, request, [])

    def auditLight(diffuse, normal, detach=False):
        shade = torch.full_like(diffuse, 0.8)
        return diffuse * shade, shade, shade

    state["lightingForAudit"] = auditLight
    audited = adapter.finishResult(prediction, state, request, [])
    for name in ("diffuse", "specular", "tangentNormal", "validMask", "generatedMask", "inferredMask"):
        np.testing.assert_array_equal(audited[name], reference[name])
    for name in ("fittedShading", "fittedMainShading", "renderedGuidance", "guidance"):
        assert audited[name].shape == (512, 512, 3)
        assert np.isfinite(audited[name]).all()
    np.testing.assert_allclose(audited["renderedGuidance"][100, 100], 0.4)
