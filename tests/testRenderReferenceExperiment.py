"""Independent evaluation support and immutable-source safeguards for reference rendering."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

toolPath = Path(__file__).resolve().parents[1] / "tools/renderReferenceExperiment.py"
specification = importlib.util.spec_from_file_location("renderReferenceExperimentTestTool", toolPath)
comparison = importlib.util.module_from_spec(specification)
specification.loader.exec_module(comparison)


@pytest.fixture
def savedReferenceExperiment(tmp_path):
    root = tmp_path / "experiment"
    (root / "branches").mkdir(parents=True)
    photo = tmp_path / "source.png"
    Image.new("RGB", (96, 64), (110, 90, 70)).save(photo)
    inputs = [{"path": str(photo), "sha256": comparison.fileHash(photo), "orientedSize": [96, 64],
               "workingSize": [96, 64], "inputToWorking": np.eye(3).tolist()}]
    cameras = [{"valid": True, "viewIndex": 0, "intrinsics": [[80., 0, 47.5], [0, 80., 31.5], [0, 0, 1]],
                "rotation": np.diag([-1., 1., -1.]).tolist(), "translation": [0., 0., 3.]}]
    mask = np.ones((1, 64, 96), np.float32)
    geometry = {"vertices": np.array([[-.6, -.4, 0], [.6, -.4, 0], [0, .6, 0]], np.float32),
                "faces": np.array([[0, 1, 2]], np.int64),
                "uvs": np.array([[0, 0], [1, 0], [.5, 1]], np.float32), "faceMasks": mask}
    for name, value in (("inputs.json", inputs), ("cameras.json", cameras),
                        ("viewTransforms.json", [{"preserved": True}]),
                        ("experimentReport.json", {"status": "prepared", "referenceViewIndex": 0})):
        (root / name).write_text(json.dumps(value))
    np.savez(root / "geometry.npz", **geometry)
    for index, name in enumerate((*comparison.branchNames, "rawReference")):
        linear = np.full((32, 32, 3), .12 + index * .03, np.float32)
        fields = {"linearAlbedo": linear, "albedo": linear ** (1 / 2.2), "coverageMask": np.ones((32, 32), bool)}
        if name == "referenceFirst":
            fields["referenceAlpha"] = np.ones((32, 32), np.float32)
        np.savez(root / "branches" / f"{name}.npz", **fields)
    return root


def testCommonEvaluationDoesNotFollowFitRejectionOrAnyOneBranch():
    shape = (16, 20)
    visible = np.ones(shape, bool)
    segmentation = np.ones(shape, np.float32)
    coverage = [np.ones(shape, np.float32) for _ in range(3)]
    coverage[0][:, :2] = 0
    coverage[1][:, -3:] = 0
    coverage[2][:2, :] = 0
    segmentation[-2:, :] = 0
    common = comparison.commonEvaluationMask(visible, segmentation, coverage, erosionPixels=1)
    assert common.sum() == 10 * 13
    assert not common[:3].any()
    # A hypothetical fit could reject almost all observations. Evaluation must
    # still include their error: that mask is never an argument to photoMetrics.
    observed = np.full((*shape, 3), .2)
    predicted = observed.copy()
    predicted[common] += .2
    metric = comparison.photoMetrics(predicted, observed, common)
    assert metric["pixelCount"] == 130
    assert metric["linearRmse"] == pytest.approx(.2)
    assert coverage[0][3, 3] == 1


def testMetricsExposeClippingRatherThanMakingBadFitsAppearAccurate():
    observed = np.ones((4, 4, 3), np.float32)
    predicted = np.full_like(observed, 2.)
    metric = comparison.photoMetrics(predicted, observed, np.ones((4, 4), bool))
    assert metric["srgbMae"] == pytest.approx(0, abs=1e-7)
    assert metric["linearRmse"] == 1
    assert metric["displayClippedPixelFraction"] == 1
    predicted[:, :, 0] = -.1
    assert comparison.photoMetrics(predicted, observed, np.ones((4, 4), bool))["negativePredictionPixelFraction"] == 1


def testSavedSourceHashAndGammaContractAreChecked(savedReferenceExperiment):
    experiment = comparison.loadExperiment(savedReferenceExperiment)
    assert experiment["referenceViewIndex"] == 0
    assert experiment["photos"][0].size == (96, 64)
    invalid = dict(experiment["branches"]["projectionMean"])
    invalid["albedo"] = invalid["linearAlbedo"].copy()
    with pytest.raises(ValueError, match="gamma-2.2"):
        comparison.validateBranch(invalid, "encoded-as-linear")
    source = Path(experiment["inputs"][0]["path"])
    Image.new("RGB", (96, 64), (120, 90, 70)).save(source)
    with pytest.raises(ValueError, match="changed"):
        comparison.loadExperiment(savedReferenceExperiment)


def testActualCpuReferenceComparisonHasIdenticalEvalMasksAndFrozenInputs(savedReferenceExperiment, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch3d")
    pytest.importorskip("faceAppearance.photoLighting")
    output = tmp_path / "rendered"
    before = {path: comparison.fileHash(path) for path in savedReferenceExperiment.rglob("*") if path.is_file()}
    report = comparison.runComparison(savedReferenceExperiment, output, device="cpu", panelSize=64,
                                      canonicalSize=32, maxSamples=2000)
    assert report["sharedGeometryAndCameraVerified"]
    assert report["lightingParameterCount"] == 27
    assert not report["cameraFittingRepeated"]
    assert not report["geometryOptimized"]
    assert not report["textureOptimized"]
    assert not report["wildCapEnabled"]
    view = report["views"][0]
    assert view["projectionCheck"]["maximumAbsoluteErrorPixels"] < .005
    assert view["frameGeometry"] is False
    assert view["commonEvaluationPixels"] > 80
    masks = []
    for name in comparison.branchNames:
        with np.load(output / "view00" / f"{name}PhotoFit.npz") as saved:
            assert saved["renderedLinear"].shape == (64, 96, 3)
            assert saved["coefficients"].shape == (9, 3)
            masks.append(saved["evaluationMask"].copy())
        assert view["branches"][name]["metrics"]["pixelCount"] == view["commonEvaluationPixels"]
        with Image.open(output / "view00" / f"{name}Matched.png") as image:
            assert image.size == (96, 64)
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    for path, digest in before.items():
        assert comparison.fileHash(path) == digest
    for filename in ("overview.png", "frontOverview.png", "fixedLightOverview.png", "unlitOverview.png",
                     "referenceProjection.png", "canonicalRelight.png", "comparisonReport.json"):
        assert (output / filename).is_file()
    with pytest.raises(FileExistsError, match="partial runs"):
        comparison.runComparison(savedReferenceExperiment, output, device="cpu")
