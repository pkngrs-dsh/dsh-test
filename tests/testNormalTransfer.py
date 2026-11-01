"""Analytic normal transport tests, independent of models and GPUs."""

import numpy as np
import pytest

from faceAppearance.normalTransfer import (
    buildCornerTangentFrames, buildTangentAtlas, convertUvs, objectToTangent,
    tangentToObject, transferTangentNormals,
)
from faceAppearance.texture import buildUvLookup


def planeMesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    faces = np.array([[0, 1, 2]], np.int64)
    uvs = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85]], np.float64)
    return vertices, faces, uvs


def makeTransfer(sourceUvs, targetUvs, *, sourceOrigin="vUp", targetOrigin="vUp", resolution=64):
    vertices, faces, _ = planeMesh()
    sourceFrames = buildTangentAtlas(vertices, faces, sourceUvs, faces, resolution, uvOrigin=sourceOrigin)
    targetFrames = buildTangentAtlas(vertices, faces, targetUvs, faces, resolution, uvOrigin=targetOrigin)
    targetRasterUvs = convertUvs(targetUvs, sourceOrigin=targetOrigin, targetOrigin="vDown")
    sourceRasterUvs = convertUvs(sourceUvs, sourceOrigin=sourceOrigin, targetOrigin="vDown")
    lookup = buildUvLookup(targetRasterUvs, faces, resolution)
    sourceUvMap = np.zeros((resolution, resolution, 2), np.float32)
    mask = lookup["validMask"]
    sourceUvMap[mask] = lookup["barycentrics"][mask] @ sourceRasterUvs
    mapping = {"sourceUvMap": sourceUvMap, "validMask": mask}
    return sourceFrames, targetFrames, mapping


@pytest.mark.parametrize("variant", ["rotated", "mirrored"])
def testFlatNormalRemainsSameObjectNormal(variant):
    _, _, sourceUvs = planeMesh()
    if variant == "rotated":
        rotation = np.array([[0, -1], [1, 0]], np.float64)
        targetUvs = (sourceUvs - 0.5) @ rotation.T + 0.5
    else:
        targetUvs = sourceUvs.copy()
        targetUvs[:, 0] = 1 - targetUvs[:, 0]
    sourceFrames, targetFrames, mapping = makeTransfer(sourceUvs, targetUvs)
    normalMap = np.zeros((64, 64, 3), np.float32)
    normalMap[..., 2] = 1
    result = transferTangentNormals(normalMap, sourceFrames["validMask"], mapping,
                                    sourceFrames, targetFrames, sourceConvention="positiveV")
    assert result["validMask"].sum() > 700
    np.testing.assert_allclose(result["tangentNormals"][result["validMask"]],
                               np.tile([0, 0, 1], (result["validMask"].sum(), 1)), atol=1e-6)
    np.testing.assert_allclose(result["objectNormals"][result["validMask"]],
                               np.tile([0, 0, 1], (result["validMask"].sum(), 1)), atol=1e-6)


def testNonFlatNormalAcross45DegreeUvRotation():
    _, _, sourceUvs = planeMesh()
    angle = np.deg2rad(45)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    targetUvs = (sourceUvs - 0.5) @ rotation.T * 0.8 + 0.5
    sourceFrames, targetFrames, mapping = makeTransfer(sourceUvs, targetUvs)
    normalMap = np.empty((64, 64, 3), np.float32)
    normalMap[:] = [np.sqrt(0.5), 0, np.sqrt(0.5)]
    result = transferTangentNormals(normalMap, sourceFrames["validMask"], mapping,
                                    sourceFrames, targetFrames, sourceConvention="positiveV")
    mask = result["validMask"]
    assert mask.sum() > 400
    np.testing.assert_allclose(result["tangentNormals"][mask],
                               np.tile([0.5, 0.5, np.sqrt(0.5)], (mask.sum(), 1)), atol=2e-6)
    np.testing.assert_allclose(result["objectNormals"][mask], normalMap[mask], atol=2e-6)


def testMirroredUvPreservesBitangentHandedness():
    _, _, sourceUvs = planeMesh()
    targetUvs = sourceUvs.copy()
    targetUvs[:, 0] = 1 - targetUvs[:, 0]
    sourceFrames, targetFrames, mapping = makeTransfer(sourceUvs, targetUvs)
    assert np.all(sourceFrames["handedness"][sourceFrames["validMask"]] == 1)
    assert np.all(targetFrames["handedness"][targetFrames["validMask"]] == -1)
    normalMap = np.empty((64, 64, 3), np.float32)
    normalMap[:] = [0.3, 0.4, np.sqrt(0.75)]
    result = transferTangentNormals(normalMap, sourceFrames["validMask"], mapping,
                                    sourceFrames, targetFrames, sourceConvention="positiveV")
    mask = result["validMask"]
    np.testing.assert_allclose(result["tangentNormals"][mask],
                               np.tile([-0.3, 0.4, np.sqrt(0.75)], (mask.sum(), 1)), atol=2e-6)
    np.testing.assert_allclose(result["objectNormals"][mask], normalMap[mask], atol=2e-6)


def testNativeVDownIsConvertedBeforeDerivatives():
    vertices, faces, objUvs = planeMesh()
    nativeUvs = convertUvs(objUvs, sourceOrigin="vUp", targetOrigin="vDown")
    objFrames = buildTangentAtlas(vertices, faces, objUvs, faces, 32, uvOrigin="vUp")
    nativeFrames = buildTangentAtlas(vertices, faces, nativeUvs, faces, 32, uvOrigin="vDown")
    np.testing.assert_array_equal(objFrames["validMask"], nativeFrames["validMask"])
    np.testing.assert_allclose(objFrames["frames"], nativeFrames["frames"], atol=1e-7)


def testExplicitNegativeGreenAndUnormDecode():
    _, _, uvs = planeMesh()
    sourceFrames, targetFrames, mapping = makeTransfer(uvs, uvs)
    signedNormal = np.array([0, 0.6, 0.8], np.float32)
    normalMap = np.broadcast_to((signedNormal + 1) / 2, (64, 64, 3))
    result = transferTangentNormals(normalMap, sourceFrames["validMask"], mapping,
                                    sourceFrames, targetFrames, sourceConvention="directX",
                                    targetConvention="openGL", inputEncoding="unorm")
    mask = result["validMask"]
    np.testing.assert_allclose(result["tangentNormals"][mask],
                               np.tile([0, -0.6, 0.8], (mask.sum(), 1)), atol=1e-6)


def testSourceHoleAndZeroNormalRemainInvalid():
    _, _, uvs = planeMesh()
    sourceFrames, targetFrames, mapping = makeTransfer(uvs, uvs)
    normalMap = np.zeros((64, 64, 3), np.float32)
    normalMap[..., 2] = 1
    coverage = sourceFrames["validMask"].copy()
    coverage[35:45, 15:25] = False
    normalMap[45:50, 15:20] = 0
    result = transferTangentNormals(normalMap, coverage, mapping, sourceFrames, targetFrames,
                                    sourceConvention="positiveV")
    assert not result["validMask"][35:45, 15:25].any()
    assert not result["validMask"][45:50, 15:20].any()
    assert np.all(result["tangentNormals"][~result["validMask"]] == 0)
    assert np.all(result["objectNormals"][~result["validMask"]] == 0)


def testDegenerateUvFaceProducesNoValidFrames():
    vertices, faces, _ = planeMesh()
    degenerateUvs = np.array([[0, 0], [0.5, 0.5], [1, 1]], np.float64)
    corners = buildCornerTangentFrames(vertices, faces, degenerateUvs, faces, uvOrigin="vUp")
    assert not corners["faceValidMask"].any()
    assert np.isfinite(corners["cornerFrames"]).all()
    atlas = buildTangentAtlas(vertices, faces, degenerateUvs, faces, 32, uvOrigin="vUp")
    assert not atlas["validMask"].any()


def testSplitMirroredUvSeamKeepsSeparateCornerFrames():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    uvs = np.array([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4],
                    [0.6, 0.6], [0.9, 0.9], [0.9, 0.6]])
    uvFaces = np.array([[0, 1, 2], [3, 4, 5]])
    corners = buildCornerTangentFrames(vertices, faces, uvs, uvFaces, uvOrigin="vUp")
    np.testing.assert_array_equal(corners["faceHandedness"], [1, -1])
    assert corners["faceValidMask"].all()
    frames = corners["cornerFrames"]
    np.testing.assert_allclose(np.einsum("...ji,...jk->...ik", frames, frames),
                               np.broadcast_to(np.eye(3), frames.shape), atol=1e-6)


def testSlantedFrameRoundTripAndFaceExclusion():
    vertices, faces, uvs = planeMesh()
    vertices[:, 2] = vertices[:, 0] * 0.4 + vertices[:, 1] * 0.2
    atlas = buildTangentAtlas(vertices, faces, uvs, faces, 16, uvOrigin="vUp")
    frames = atlas["frames"][atlas["validMask"]]
    local = np.broadcast_to([0.3, 0.4, np.sqrt(0.75)], (len(frames), 3))
    result = objectToTangent(tangentToObject(local, frames, convention="positiveV"),
                             frames, convention="positiveV")
    np.testing.assert_allclose(result, local, atol=1e-6)
    excluded = buildTangentAtlas(vertices, faces, uvs, faces, 16, uvOrigin="vUp",
                                 validFaces=np.array([False]))
    assert not excluded["validMask"].any()


def testUnknownWildCapConventionIsNotGuessed():
    with pytest.raises(ValueError, match="not verified"):
        tangentToObject(np.array([[0, 0, 1]]), np.eye(3)[None], convention="wildcap")


def testSourceConventionIsRequired():
    with pytest.raises(TypeError, match="sourceConvention"):
        transferTangentNormals(None, None, None, None, None)
