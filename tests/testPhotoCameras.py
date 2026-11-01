import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from faceAppearance.camera import projectPoints
from faceAppearance.photoCameras import canonicalCamera, originalImageIntrinsics
from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer


def testNativeCameraCanonicalizationPreservesProjectionAndDepth():
    world = np.random.default_rng(22).normal(size=(20, 3)) * .1
    canonicalRotation = Rotation.from_euler('xyz', [20, -35, 12], degrees=True).as_matrix()
    scale, offset = 5.7, np.array([.3, -.5, 2.])
    viewRotation = Rotation.from_euler('xyz', [5, 10, -3], degrees=True).as_matrix()
    viewTranslation = np.array([.2, .4, 3])
    extrinsics = np.c_[viewRotation, viewTranslation]
    camera = canonicalCamera(extrinsics, {"s": scale, "R": canonicalRotation, "t": offset})
    canonical = scale * (world @ canonicalRotation) + offset
    intrinsics = np.array([[850, 0, 248], [0, 790, 259], [0, 0, 1]])
    expected, depth = projectPoints(world, intrinsics, viewRotation, viewTranslation)
    actual, canonicalDepth = projectPoints(canonical, intrinsics, camera['rotation'], camera['translation'])
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(canonicalDepth, scale * depth, atol=1e-10)


def testPhotoIntrinsicsReverseCropHalfPixelTransform():
    affine = np.array([[.5, 0, -.25], [0, .5, -100.25], [0, 0, 1]])
    original = np.array([[1400, 0, 320], [0, 1410, 600], [0, 0, 1]])
    np.testing.assert_allclose(originalImageIntrinsics(affine @ original, affine), original)


def makeGeometry():
    vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [1., 1., 2.], [-1., 1., 2.]])
    # OpenCV camera looks +Z, so visible front faces point -Z.
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    frames = np.tile(np.diag([1., -1., -1.]), (2, 3, 1, 1))
    return {"vertices": vertices, "faces": faces, "objUvs": (vertices[:, :2] + 1) / 2,
            "uvFaces": faces, "cornerTangentFrames": frames}


def testPerspectiveProjectionAndRasterPixelsMatchOpenCvCenters():
    geometry = makeGeometry()
    renderer = Pytorch3dFaceRenderer(geometry, device="cpu", imageSize=(48, 64), frameGeometry=False)
    intrinsics = np.array([[30., 0, 31.], [0, 28., 23.], [0, 0, 1.]])
    renderer.setPerspectiveCamera(intrinsics, np.eye(3), np.zeros(3))
    expected, _ = projectPoints(geometry['vertices'], intrinsics, np.eye(3), np.zeros(3))
    actual = renderer.projectionCamera.transform_points_screen(torch.tensor(geometry['vertices'], dtype=torch.float32))
    np.testing.assert_allclose(actual[:, :2].numpy(), expected, atol=1e-5)
    # Sample triangle interiors, avoiding exact shared-edge raster tie breaks.
    assert renderer.visible[22, 31]
    torch.testing.assert_close(renderer.positions[22, 31], torch.tensor([0., -2 / 28, 2.]), atol=1e-5, rtol=0)
    expectedRay = torch.nn.functional.normalize(torch.tensor([-2 / 3, 0., -2.]), dim=0)
    torch.testing.assert_close(renderer.viewDirections[23, 41], expectedRay, atol=1e-5, rtol=0)


def testPhotoCameraRejectsAutoFramedMesh():
    renderer = Pytorch3dFaceRenderer(makeGeometry(), device="cpu", imageSize=32)
    with pytest.raises(ValueError, match="frameGeometry=False"):
        renderer.setPerspectiveCamera(np.eye(3), np.eye(3), np.zeros(3))
