"""Saved-data safeguards and a tiny real CPU rendering path for the SH comparison."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

toolPath = Path(__file__).resolve().parents[1] / "tools/renderShExperiment.py"
specification = importlib.util.spec_from_file_location("renderShExperimentTestTool", toolPath)
comparison = importlib.util.module_from_spec(specification)
specification.loader.exec_module(comparison)


@pytest.fixture
def savedExperiment(tmp_path):
    photo = tmp_path / "source.png"
    Image.new("RGB", (96, 64), (70, 90, 110)).save(photo)
    inputs = [{"path": str(photo), "sha256": comparison.fileHash(photo), "orientedSize": [96, 64],
               "workingSize": [96, 64], "inputToWorking": np.eye(3).tolist()}]
    cameras = [{"valid": True, "viewIndex": 0, "intrinsics": [[80., 0, 47.5], [0, 80., 31.5], [0, 0, 1]],
                "rotation": np.diag([-1., 1., -1.]).tolist(), "translation": [0., 0., 3.]}]
    mask = np.zeros((1, 64, 96), np.float32)
    mask[0, 12:53, 25:71] = 1
    geometry = {"vertices": np.array([[-.6, -.4, 0], [.6, -.4, 0], [0, .6, 0]], np.float32),
                "faces": np.array([[0, 1, 2]], np.int64),
                "uvs": np.array([[0, 0], [1, 0], [.5, 1]], np.float32), "faceMasks": mask}
    root = tmp_path / "experiment"
    for index, name in enumerate(comparison.branchNames):
        directory = root / name
        directory.mkdir(parents=True)
        diagnostic = {"status": "experimental", "iterations": 599, "seed": 2026,
                      "workingResolution": 32, "outputResolution": 32, "inputMethod": name}
        result = {"status": "succeeded", "method": name, "stopAfter": "wildcap",
                  "wildCapDiagnostics": diagnostic,
                  "experiment": {"atlasResolution": 32,
                                 "wildCapPlanned": {"iterations": 599, "resolution": 32, "seed": 2026}}}
        for filename, value in (("inputs.json", inputs), ("cameras.json", cameras),
                                ("viewTransforms.json", [{"preserved": True}]),
                                ("result.json", result)):
            (directory / filename).write_text(json.dumps(value))
        np.savez(directory / "geometry.npz", **geometry)
        albedo = np.full((32, 32, 3), .5 + index * .1, np.float32)
        coverage = np.ones((32, 32), bool)
        np.savez(directory / "texture.npz", albedo=albedo, coverageMask=coverage)
        (directory / "wildcapIct").mkdir()
        (directory / "wildcapIct/diagnostics.json").write_text(json.dumps(diagnostic))
        np.savez(directory / "wildcapIct/materials.npz", diffuse=albedo * .9, validMask=coverage,
                 specular=np.full((32, 32), .22), tangentNormal=np.dstack(
                     [np.zeros((32, 32)), np.zeros((32, 32)), np.ones((32, 32))]))
        (directory / "wildcapUff").mkdir()
        np.savez(directory / "wildcapUff/materials.npz", diffuse=albedo * .9,
                 baseWithWildCap=albedo * .9, validMask=coverage,
                 specular=np.full((32, 32), 999), tangentNormal=np.full((32, 32, 3), 999))
    return root


def loadBranches(root, withWildCap=False):
    return {name: comparison.loadBranch(root / name, withWildCap) for name in comparison.branchNames}


def testSharedInputsValidatedWithoutRequiringSavedWorkingPngs(savedExperiment):
    branches = loadBranches(savedExperiment)
    photos, coverage = comparison.validateSharedExperiment(branches)
    assert photos[0].size == (96, 64)
    assert coverage.all()
    assert not (savedExperiment / "openDelight/input00.png").exists()
    changed = copy.deepcopy(branches)
    changed["shOnly"]["cameras"][0]["intrinsics"][0][0] += .01
    with pytest.raises(ValueError, match="PnP intrinsics"):
        comparison.validateSharedExperiment(changed)
    changed = copy.deepcopy(branches)
    changed["shPrior"]["savedGeometry"]["vertices"][0, 0] += .001
    with pytest.raises(ValueError, match="vertices"):
        comparison.validateSharedExperiment(changed)
    changed = copy.deepcopy(branches)
    changed["shOnly"]["inputs"][0]["inputToWorking"][0][2] += 1
    with pytest.raises(ValueError, match="inputToWorking"):
        comparison.validateSharedExperiment(changed)


def testChangedSourceContentAndUnavailableWildCapAreRejected(savedExperiment):
    branches = loadBranches(savedExperiment)
    photo = Path(branches["openDelight"]["inputs"][0]["path"])
    Image.new("RGB", (96, 64), (10, 20, 30)).save(photo)
    with pytest.raises(ValueError, match="changed"):
        comparison.validateSharedExperiment(branches)
    (savedExperiment / "shOnly/wildcapUff/materials.npz").unlink()
    with pytest.raises(FileNotFoundError, match="every branch"):
        loadBranches(savedExperiment, withWildCap=True)


def testWildCapMaterialsWithoutCommittedCompletionAreRejected(savedExperiment):
    path = savedExperiment / "shOnly/result.json"
    result = json.loads(path.read_text())
    result["stopAfter"] = "fusion"
    result.pop("wildCapDiagnostics")
    path.write_text(json.dumps(result))
    # Material files may have been written immediately before a crash.
    assert (path.parent / "wildcapUff/materials.npz").is_file()
    with pytest.raises(ValueError, match="completion marker"):
        loadBranches(savedExperiment, withWildCap=True)
    assert loadBranches(savedExperiment, withWildCap=False)


def testCommonUvSupportUsesIntersectionWithoutAlteringSourceMasks(savedExperiment):
    branches = loadBranches(savedExperiment)
    branches["shOnly"]["coverage"][:4] = False
    branches["shPrior"]["coverage"][:, :3] = False
    _, common = comparison.validateSharedExperiment(branches)
    assert common.sum() == 28 * 29
    assert branches["openDelight"]["coverage"].all()
    preview = comparison.atlasPreview(branches["openDelight"]["albedo"], common)
    np.testing.assert_allclose(preview[~common], .16)


def testAllMaterialsHaveExplicitGammaDecodeAndSrgbDisplay(savedExperiment):
    branches = loadBranches(savedExperiment, withWildCap=True)
    variants = comparison.materialVariants(branches, True)
    assert [item["name"] for item in variants] == [
        "openDelight", "shOnly", "shPrior", "openDelightWildCap", "shOnlyWildCap", "shPriorWildCap"]
    encoded = np.full((2, 2, 3), .5)
    expected = 1.055 * (.5 ** 2.2) ** (1 / 2.4) - .055
    np.testing.assert_allclose(comparison.atlasPreview(encoded, np.ones((2, 2), bool)), expected)
    assert expected != pytest.approx(.5, abs=.001)
    with pytest.raises(ValueError, match="Invalid"):
        comparison.validateTexture(encoded * 3, np.ones((2, 2)), "bad")


def testActualCpuComparisonPreservesOriginalSizeProjectionAndDiffuseOnly(savedExperiment, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch3d")
    output = tmp_path / "rendered"
    report = comparison.runComparison(savedExperiment, output, device="cpu", withWildCap=True,
                                      imageSize=32, panelSize=64)
    assert report["sharedGeometryAndCameraVerified"]
    assert not report["cameraFittingRepeated"]
    assert report["views"][0]["projectionCheck"]["maximumAbsoluteErrorPixels"] < .005
    assert report["views"][0]["frameGeometry"] is False
    assert report["shading"]["specularMultiplier"] == .22
    assert report["shading"]["normalSource"] == "shared geometry normals"
    assert report["color"]["renderAndAtlasDisplayEncoding"] == "exact piecewise sRGB"
    for name in ("overview.png", "overlayOverview.png", "unlitOverview.png", "atlasOverview.png",
                 "canonicalRelight.png", "beforeAfterWildCap.png", "comparisonReport.json"):
        assert (output / name).is_file()
    for variant in ("openDelight", "shOnly", "shPrior", "openDelightWildCap"):
        with Image.open(output / "view00" / f"{variant}Render.png") as image:
            assert image.size == (96, 64)
    unlit = np.asarray(Image.open(output / "view00/openDelightUnlit.png")) / 255
    expected = comparison.linearToSrgb(np.array(.5 ** 2.2))
    np.testing.assert_allclose(unlit[32, 48], expected, atol=1 / 255)
    # The fixture stores nonsensical WC specular/normals. Successful finite
    # renders verify that this diffuse comparison never consumes those maps.
    with pytest.raises(FileExistsError, match="empty"):
        comparison.runComparison(savedExperiment, output, device="cpu")
