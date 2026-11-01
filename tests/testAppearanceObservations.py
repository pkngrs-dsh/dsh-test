import hashlib

import numpy as np
import pytest
from PIL import Image

from faceAppearance.appearanceObservations import (collectAppearanceObservations, fuseLinearObservations,
                                                   linearToGamma22, readOriginalLinear,
                                                   srgbToLinear, workingToOriginal)


def testColorContractsPreserveLinearLight():
    values = np.array([0, 0.04045, 0.5, 1], np.float32)
    expected = np.array([0, 0.003130805, 0.21404114, 1])
    np.testing.assert_allclose(srgbToLinear(values), expected, atol=1e-7)
    np.testing.assert_allclose(linearToGamma22(expected) ** 2.2, expected, atol=1e-7)


def testOriginalSamplingPreservesHalfPixelOffsets():
    record = {"inputToWorking": [[0.5, 0, -10.25], [0, 0.5, -20.25], [0, 0, 1]]}
    np.testing.assert_array_equal(workingToOriginal(np.array([[0, 0], [10, 20]]), record),
                                  [[20.5, 40.5], [40.5, 80.5]])


def testCommonUvSamplesUseOriginalLinearPixelsAndRejectChangedPhotos(tmp_path):
    photo = np.zeros((10, 10, 3), np.uint8)
    photo[:, 1::2] = 255
    path = tmp_path / "photo.png"
    Image.fromarray(photo).save(path)
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "orientedSize": [10, 10], "inputToWorking": [[0.5, 0, -.25], [0, 0.5, -.25], [0, 0, 1]]}
    yy, xx = np.mgrid[:5, :5]
    geometry = {"vertices": np.array([[0, 0, 2], [0, 4, 2], [4, 0, 2]], np.float32),
                "faces": np.array([[0, 1, 2]]), "uvs": np.array([[0, 0], [0, 1], [1, 0]]),
                "uvMaps": (np.stack([xx, yy], axis=-1) / 4)[None]}
    camera = {"valid": True, "rotation": np.eye(3), "translation": np.zeros(3),
              "intrinsics": np.diag([2, 2, 1]), "medianErrorPixels": 0}
    delight = {"validMasks": np.ones((1, 5, 5), bool), "albedoImages": np.full((1, 5, 5, 3), .4)}
    result = collectAppearanceObservations(geometry, [camera], [record], delight, resolution=5)
    assert result["weights"][0, 0, 0] > 0
    np.testing.assert_allclose(result["originalSamplingPixels"][0, 0, 0], [.5, .5], atol=1e-5)
    # Linearize before interpolating alternating black/white source pixels.
    np.testing.assert_allclose(result["rawLinear"][0, 0, 0], .5, atol=1e-5)
    np.testing.assert_allclose(result["openDelightLinear"][0, 0, 0], .4 ** 2.2, atol=1e-6)
    Image.new("RGB", (10, 10), (100, 100, 100)).save(path)
    with pytest.raises(ValueError, match="changed"):
        readOriginalLinear(record)


def testFusionIgnoresUnobservedColorsAndDownweightsOutliers():
    values = np.full((4, 3, 3, 3), .2, np.float32)
    values[2] = .9
    values[3] = 4
    weights = np.ones((4, 3, 3), np.float32)
    weights[3] = 0
    weights[:, 0, 0] = 0
    result = fuseLinearObservations(values, weights)
    assert not result["coverageMask"][0, 0]
    assert result["sourceIndex"][0, 0] == -1
    assert not result["linearAlbedo"][0, 0].any()
    assert .2 <= result["linearAlbedo"][1, 1, 0] < .21


def testInvalidColorArraysFailBeforeFitting():
    with pytest.raises(ValueError):
        srgbToLinear(np.array([np.nan]))
    with pytest.raises(ValueError):
        fuseLinearObservations(np.zeros((4, 5, 5, 3)), np.full((4, 5, 5), -1))


def testProjectionSamplingBypassesSemanticWarpAndDoesNotNeedDenseMaps(tmp_path, monkeypatch):
    import faceAppearance.appearanceObservations as module
    photo = np.zeros((10, 10, 3), np.uint8)
    photo[:, 1::2] = 255
    path = tmp_path / "photo.png"
    Image.fromarray(photo).save(path)
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "orientedSize": [10, 10], "inputToWorking": [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]]}
    geometry = {"vertices": np.array([[0, 0, 2], [0, 4, 2], [4, 0, 2]], np.float32),
                "faces": np.array([[0, 1, 2]]), "uvs": np.array([[0, 0], [0, 1], [1, 0]])}
    camera = {"valid": True, "rotation": np.eye(3), "translation": np.zeros(3),
              "intrinsics": np.diag([2, 2, 1]), "medianErrorPixels": 0}
    delight = {"validMasks": np.ones((1, 5, 5), bool), "albedoImages": np.full((1, 5, 5, 3), .4)}
    def refuseSemanticWarp(*args, **kwargs):
        raise AssertionError("Projection must not use dense UV")
    monkeypatch.setattr(module, "invertUvMap", refuseSemanticWarp)
    result = collectAppearanceObservations(geometry, [camera], [record], delight,
                                           resolution=5, samplingMode="projection")
    np.testing.assert_allclose(result["originalSamplingPixels"][0, 0, 0], [.5, .5], atol=1e-6)
    np.testing.assert_allclose(result["rawLinear"][0, 0, 0], .5, atol=1e-6)
    assert result["viewingCosines"][0, 0, 0] == pytest.approx(1)
    assert result["diagnostics"]["views"][0]["medianDenseCorrectionWorkingPixels"] == 0
    assert result["diagnostics"]["samplingMode"] == "projection"
    with pytest.raises(ValueError, match="samplingMode"):
        collectAppearanceObservations(geometry, [camera], [record], delight, samplingMode="guess")
    with pytest.raises(ValueError, match="visibilityMode"):
        collectAppearanceObservations(geometry, [camera], [record], delight, visibilityMode="guess")


def testExactVisibilityRecoversSlopedPhotoSamplesWithoutChangingExistingOnes(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (32, 32), (150, 120, 100)).save(path)
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "orientedSize": [32, 32], "inputToWorking": np.eye(3).tolist()}
    intrinsics = np.array([[16., 0, 16], [0, 16, 16], [0, 0, 1]])
    pixels = np.array([[4., 4.], [28., 4.], [28., 28.], [4., 28.]])
    depths = np.array([2., 4., 4., 2.])
    vertices = np.column_stack([(pixels-16)*depths[:, None]/16, depths])
    # Reversed screen winding faces the camera, retaining the original cosine gate.
    geometry = {"vertices": vertices, "faces": np.array([[2, 1, 0], [3, 2, 0]]),
                "uvs": np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])}
    camera = {"valid": True, "rotation": np.eye(3), "translation": np.zeros(3),
              "intrinsics": intrinsics, "medianErrorPixels": 0}
    delight = {"validMasks": np.ones((1, 32, 32), bool), "albedoImages": np.full((1, 32, 32, 3), .4)}
    coarse = collectAppearanceObservations(geometry, [camera], [record], delight, resolution=31, samplingMode="projection")
    exact = collectAppearanceObservations(geometry, [camera], [record], delight, resolution=31,
                                          samplingMode="projection", visibilityMode="refineRejected")
    old = coarse["weights"] > 0
    added = (exact["weights"] > 0) & ~old
    assert added.any()
    assert not (old & (exact["weights"] == 0)).any()
    for name in ("rawLinear", "weights", "originalSamplingPixels"):
        np.testing.assert_array_equal(exact[name][old], coarse[name][old])
    np.testing.assert_allclose(exact["rawLinear"][added],
                              np.broadcast_to(srgbToLinear(np.array([150, 120, 100], np.float32)/255),
                                               (int(added.sum()), 3)), atol=1e-7)
    assert exact["diagnostics"]["views"][0]["visibility"]["toleranceChanged"] is False
