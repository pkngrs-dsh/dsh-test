import numpy as np
import torch

from faceAppearance.pytorch3dRenderer import (Pytorch3dFaceRenderer, applyTangentNormals,
                                            linearToSrgb, shadeGgx)


def testNeutralNormalAndMirroredUvHandedness():
    frames = torch.eye(3).repeat(2, 1, 1)
    frames[1, :, 1] *= -1
    neutral = torch.tensor([[0., 0., 1.]]).repeat(2, 1)
    torch.testing.assert_close(applyTangentNormals(frames, neutral), neutral)
    directions = torch.tensor([[0., 1., 0.]]).repeat(2, 1)
    torch.testing.assert_close(applyTangentNormals(frames, directions),
                               torch.tensor([[0., 1., 0.], [0., -1., 0.]]))


def testBackLightAndSpecularAreFiniteAndLinear():
    color = torch.full((1, 3), .4)
    normals = torch.tensor([[0., 0., 1.]])
    strength = torch.zeros((1, 1))
    dark = shadeGgx(color, normals, normals, strength, [0, 0, -1], ambient=0)
    torch.testing.assert_close(dark, torch.zeros_like(dark))
    once = shadeGgx(color, normals, normals, strength, [0, 0, 1], ambient=0, keyIntensity=1)
    twice = shadeGgx(color, normals, normals, strength, [0, 0, 1], ambient=0, keyIntensity=2)
    torch.testing.assert_close(twice, 2 * once)
    grazing = shadeGgx(color, normals, torch.tensor([[1., 0., 0.]]), torch.ones((1, 1)), [1, 0, 0])
    assert torch.isfinite(grazing).all()
    torch.testing.assert_close(linearToSrgb(torch.tensor([0., 1.])), torch.tensor([0., 1.]))


def testPytorch3dUvSamplingAndCameraDoNotMirrorOrFlipTexture():
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    objUvs = vertices[:, :2] + .5
    geometry = {"vertices": vertices, "faces": faces, "objUvs": objUvs, "uvFaces": faces,
                "cornerTangentFrames": np.tile(np.eye(3), (2, 3, 1, 1))}
    renderer = Pytorch3dFaceRenderer(geometry, device="cpu", imageSize=32)
    renderer.setView()
    texture = np.zeros((8, 8, 3), np.float32)
    texture[:4, :4] = [1, 0, 0]
    texture[:4, 4:] = [0, 1, 0]
    texture[4:, :4] = [0, 0, 1]
    texture[4:, 4:] = [1, 1, 0]
    sampled = renderer.sampleAtlas(texture)
    for y, x, color in ((8, 8, [1, 0, 0]), (8, 24, [0, 1, 0]),
                        (24, 8, [0, 0, 1]), (24, 24, [1, 1, 0])):
        assert renderer.visible[y, x]
        torch.testing.assert_close(sampled[y, x], torch.tensor(color, dtype=torch.float32))


def testInvalidTangentBasisFallsBackToGeometryNormal():
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2]])
    geometry = {"vertices": vertices, "faces": faces, "objUvs": vertices[:, :2] + .5,
                "uvFaces": faces, "cornerTangentFrames": np.zeros((1, 3, 3, 3)),
                "faceTangentValidMask": np.array([False])}
    renderer = Pytorch3dFaceRenderer(geometry, device="cpu", imageSize=32)
    renderer.setView()
    normalMap = np.zeros((8, 8, 3), np.float32); normalMap[..., 0] = 1
    material = renderer.prepareMaterial(np.full((8, 8, 3), .5), np.ones((8, 8)),
                                        tangentNormal=normalMap, normalValidMask=np.ones((8, 8)))
    torch.testing.assert_close(material["normals"][16, 16], torch.tensor([0., 0., 1.]))
    assert renderer.render(material, [0, 0, 1])[16, 16].mean() > .3
