"""Small CPU fixtures for the standalone crop comparison tool."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

toolPath = Path(__file__).resolve().parents[1] / "tools/renderCropComparison.py"
specification = importlib.util.spec_from_file_location("renderCropComparisonTestTool", toolPath)
comparison = importlib.util.module_from_spec(specification)
specification.loader.exec_module(comparison)


def testSourceHashAndOrientedDimensionsMustMatch(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (80, 120), (70, 90, 110)).save(path)
    record = {"path": str(path), "sha256": comparison.fileHash(path), "orientedSize": [80, 120]}
    assert comparison.validatePhotoPair(record, dict(record)).size == (80, 120)
    with pytest.raises(ValueError, match="SHA-256"):
        comparison.validatePhotoPair(record, {**record, "sha256": "incorrect"})
    with pytest.raises(ValueError, match="dimensions"):
        comparison.validatePhotoPair(record, {**record, "orientedSize": [120, 80]})
    Image.new("RGB", (80, 120), (10, 20, 30)).save(path)
    with pytest.raises(ValueError, match="changed"):
        comparison.validatePhotoPair(record, dict(record))


def testMasksUseEachSavedInversePixelTransformAndSharedUnionCrop():
    working = np.zeros((10, 10), np.float32)
    working[2:5, 2:5] = 1
    before = comparison.originalMask(working, np.array([[1, 0, -20], [0, 1, -30], [0, 0, 1]]), (80, 60))
    after = comparison.originalMask(working, np.array([[1, 0, -30], [0, 1, -30], [0, 0, 1]]), (80, 60))
    assert before[32:35, 22:25].all()
    assert after[32:35, 32:35].all()
    assert before.sum() == after.sum() == 9
    crop = comparison.sharedPhotoCrop(before, after)
    left, top, right, bottom = crop
    assert right - left == bottom - top == 16
    assert left <= 22 and right > 34 and top <= 32 and bottom > 34
    assert comparison.sharedPhotoCrop(after, before) == crop


def testCropPadsOutsideFrameWithoutAsymmetricClamping():
    mask = np.zeros((20, 20), np.float32)
    mask[:4, :4] = 1
    left, top, right, bottom = comparison.sharedPhotoCrop(mask, np.zeros_like(mask))
    assert left < 0 and top < 0
    assert right - left == bottom - top
    with pytest.raises(ValueError, match="Neither"):
        comparison.sharedPhotoCrop(np.zeros_like(mask), np.zeros_like(mask))


def testEachGroupsOriginalIntrinsicsUndoItsOwnCrop():
    original = np.array([[80., 0, 47.5], [0, 82, 31.5], [0, 0, 1]])
    affines = (np.array([[2., 0, -5], [0, 2, -9], [0, 0, 1]]),
               np.array([[4., 0, -30], [0, 4, -19], [0, 0, 1]]))
    for affine in affines:
        working = {"intrinsics": affine @ original, "rotation": np.eye(3), "translation": [0, 0, 3]}
        camera = comparison.photoCamera(working, {"inputToWorking": affine}, "native")
        np.testing.assert_allclose(camera["intrinsics"], original, atol=1e-12)
        np.testing.assert_array_equal(camera["translation"], [0, 0, 3])


def testOverlayOnlyChangesVisibleMeshPixels():
    photo = np.full((4, 5, 3), 0.2, np.float32)
    rendered = np.full_like(photo, 0.8)
    visible = np.zeros((4, 5), bool)
    visible[1:3, 2:4] = True
    overlay = comparison.overlayPhoto(photo, rendered, visible)
    np.testing.assert_allclose(overlay[visible], 0.5)
    np.testing.assert_array_equal(overlay[~visible], photo[~visible])
    assert photo.mean() == pytest.approx(0.2)


def testActualPytorch3dCpuUsesOriginalNonSquareFrameAndGeometryNormals():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch3d")
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    saved = {"vertices": np.array([[-0.6, -0.4, 0], [0.6, -0.4, 0], [0, 0.6, 0]], np.float32),
             "faces": np.array([[0, 2, 1]], np.int64),
             "uvs": np.array([[0, 0], [1, 0], [0.5, 1]], np.float32)}
    geometry = comparison.prepareGeometry(saved, 32)
    renderer = Pytorch3dFaceRenderer(geometry, device="cpu", imageSize=(64, 96), frameGeometry=False)
    camera = {"intrinsics": np.array([[80., 0, 47.5], [0, 80., 31.5], [0, 0, 1]]),
              "rotation": np.eye(3), "translation": np.array([0., 0., 3.])}
    renderer.setPerspectiveCamera(camera["intrinsics"], camera["rotation"], camera["translation"])
    check = comparison.compareProjection(renderer, geometry, camera)
    assert check["maximumAbsoluteErrorPixels"] < 0.005
    torch.testing.assert_close(renderer.mesh.verts_packed(), torch.from_numpy(saved["vertices"]))
    material = renderer.prepareMaterial(np.full((32, 32, 3), 0.5, np.float32), np.ones((32, 32), np.float32))
    torch.testing.assert_close(material["normals"], renderer.normals)
    rendered = renderer.render(material, camera["rotation"].T @ np.array([-.5, -.5, -1.]))
    assert rendered.shape == (64, 96, 3)
    assert renderer.visible.any()
    assert np.isfinite(rendered).all()
