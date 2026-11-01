import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


toolPath = Path(__file__).resolve().parents[1] / "tools/runShBandwidthControl.py"
specification = importlib.util.spec_from_file_location("shBandwidthControlTestTool", toolPath)
control = importlib.util.module_from_spec(specification)
specification.loader.exec_module(control)


def testSavedSamplingTransformsBackToWorkingAndLinearizesBeforeInterpolation(tmp_path):
    image = np.zeros((5, 5, 3), np.uint8)
    image[:, 1::2] = 255
    Image.fromarray(image).save(tmp_path / "input00.png")
    weights = np.ones((1, 2, 2), np.float32)
    weights[0, 1, 1] = 0
    original = np.array([[[[21.5, 41.5], [25.5, 41.5]], [[21.5, 45.5], [-1, -1]]]], np.float32)
    record = {"workingSize": [5, 5], "orientedSize": [80, 80], "sha256": "original-photo-id",
              "inputToWorking": [[.5, 0, -10.25], [0, .5, -20.25], [0, 0, 1]]}
    result = control.sampleSavedWorkingImages({"weights": weights, "originalSamplingPixels": original}, [record], tmp_path)
    np.testing.assert_allclose(result["raw512Linear"][weights > 0], .5, atol=1e-6)
    np.testing.assert_array_equal(result["samplingValidMask"], weights > 0)
    assert not result["raw512Linear"][0, 1, 1].any()
    record["workingSize"] = [6, 5]
    with pytest.raises(ValueError, match="dimensions"):
        control.sampleSavedWorkingImages({"weights": weights, "originalSamplingPixels": original}, [record], tmp_path)


def testMetricScaleNormalizationAndSharedSupport():
    yy, xx = np.mgrid[:24, :24]
    base = .25 + .03 * (2 * ((xx + yy) % 2) - 1)
    values = np.repeat(base[None, ..., None], 3, axis=0)
    values = np.repeat(values, 3, axis=-1).astype(np.float32)
    values[0] *= 1.1
    weights = np.ones(values.shape[:3], np.float32)
    mask = np.ones(base.shape, bool)
    fused = control.fuseLinearObservations(values, weights)
    first = control.comparisonMetrics(values, fused, weights, mask)
    scaled = {**fused, "linearAlbedo": fused["linearAlbedo"] * .7}
    second = control.comparisonMetrics(values * .7, scaled, weights, mask)
    assert first["relativeHighPassRms"] == pytest.approx(second["relativeHighPassRms"], rel=1e-5)
    assert first["relativeCrossViewRmse"] == pytest.approx(second["relativeCrossViewRmse"], rel=1e-5)
    assert first["multiViewMetricTexels"] == 24 * 24
    assert "not accuracy" in first["meaning"]
    weights[1:] = 0
    single = control.comparisonMetrics(values, fused, weights, mask)
    assert single["crossViewRmseLinear"] is None
    assert single["multiViewMetricTexels"] == 0


def testSmallCompleteCpuExperimentKeepsMainArtifactsImmutable(tmp_path):
    source, experiment = tmp_path / "source", tmp_path / "experiment"
    source.mkdir(); experiment.mkdir(); (experiment / "openDelight").mkdir()
    inputs = []
    for index in range(4):
        Image.new("RGB", (32, 32), (140 + index * 5, 145 + index * 5, 150 + index * 5)).save(source / f"input{index:02d}.png")
        inputs.append({"workingSize": [32, 32], "orientedSize": [32, 32], "sha256": f"source-{index}",
                       "inputToWorking": np.eye(3).tolist()})
    content = json.dumps(inputs)
    (source / "inputs.json").write_text(content)
    (experiment / "openDelight/inputs.json").write_text(content)
    report = {"sourceRun": str(source), "parameters": {"shIterations": 2}}
    (experiment / "experimentReport.json").write_text(json.dumps(report))
    yy, xx = np.mgrid[:16, :16]
    points = np.repeat(np.stack([xx * 2, yy * 2], axis=-1)[None], 4, axis=0)
    weights = np.ones((4, 16, 16), np.float32)
    normals = np.zeros((16, 16, 3), np.float32); normals[..., 2] = 1
    raw = np.full((4, 16, 16, 3), .3, np.float32)
    predicted = np.full_like(raw, .25)
    np.savez_compressed(experiment / "observations.npz", rawLinear=raw, openDelightLinear=predicted,
                        weights=weights, normals=normals, originalSamplingPixels=points, atlasMask=np.ones((16, 16), bool))
    np.savez_compressed(experiment / "commonSupport.npz", weights=weights, coverageMask=np.ones((16, 16), bool))
    for mode, colors in (("shOnly", raw), ("shPrior", predicted)):
        np.savez_compressed(experiment / f"{mode}Fit.npz", correctedObservations=colors)
    tracked = list(experiment.glob("*.npz")) + list(source.glob("*"))
    hashes = {path: control.fileHash(path) for path in tracked}
    output = experiment / "bandwidthControl"
    result = control.runBandwidthControl(experiment, source, output, iterations=2)
    assert result["status"] == "completed"
    assert result["commonCoverageTexels"] == 16 * 16
    assert not result["mainExperimentWasModified"] and not result["wildCapWasRun"]
    assert set(result["metrics"]) == set(control.branchLabels)
    assert all("LIT, NOT ALBEDO" in result["branchLabels"][name] for name in ("rawOriginalLit", "raw512Lit"))
    for path, digest in hashes.items():
        assert control.fileHash(path) == digest
    masks = []
    for name in control.branchLabels:
        with np.load(output / f"{name}.npz") as data:
            masks.append(data["coverageMask"])
        with Image.open(output / f"{name}.png") as image:
            image.verify()
    assert all(np.array_equal(mask, masks[0]) for mask in masks)
    with Image.open(output / "atlasComparison.png") as image:
        image.verify()
    with pytest.raises(FileExistsError):
        control.runBandwidthControl(experiment, source, output, iterations=2)
