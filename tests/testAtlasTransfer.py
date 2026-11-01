import numpy as np
import pytest

from faceAppearance.atlasTransfer import compileAtlasMap, transferTexture


def planeMap(resolution=16):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = vertices[:, :2]
    return compileAtlasMap(vertices, faces, uvs, faces, vertices, faces, uvs, faces,
                           resolution, np.ones(2, bool), np.ones(2, bool))


def testIdenticalAtlasPreservesGradientsAndBlackPixels():
    mapping = planeMap()
    yy, xx = np.mgrid[:16, :16]
    texture = np.stack([xx / 15, yy / 15, np.zeros_like(xx)], axis=-1)
    result = transferTexture(texture, np.ones((16, 16)), mapping)
    assert result["validMask"].all()
    np.testing.assert_allclose(result["texture"], texture, atol=2e-7)


def testMissingObservationsStayMissingAndNormalCopyIsForbidden():
    mapping = planeMap()
    coverage = np.ones((16, 16)); coverage[3:7, 3:7] = 0
    result = transferTexture(np.zeros((16, 16, 3)), coverage, mapping)
    assert not result["validMask"][4:6, 4:6].any()
    with pytest.raises(ValueError, match="basis"):
        transferTexture(np.zeros((16, 16, 3)), coverage, mapping, materialKind="tangentNormal")


def testSourceUvSeamUsesFaceCornerIndices():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    faces = np.array([[0, 1, 2]])
    uvSource = np.array([[0, 0], [0, 0], [0, 0], [.2, .3], [.8, .3], [.2, .9]])
    mapping = compileAtlasMap(vertices, faces, uvSource, np.array([[3, 4, 5]]),
                              vertices, faces, vertices[:, :2], faces, 11,
                              np.ones(1, bool), np.ones(1, bool))
    np.testing.assert_allclose(mapping["sourceUvMap"][0, 0], [.2, .3], atol=1e-6)
    np.testing.assert_allclose(mapping["sourceUvMap"][0, 10], [.8, .3], atol=1e-6)


def blackWhiteSamples():
    texture = np.repeat(np.array([[0, 1], [0, 1]], np.float32)[..., None], 3, axis=-1)
    mapping = {"sourceUvMap": np.array([[[0, 0.5], [0.5, 0.5], [1, 0.5]]], np.float32),
               "validMask": np.ones((1, 3), bool)}
    return texture, mapping


def testGammaColorInterpolatesInLinearLightAndNativeStaysCompatible():
    texture, mapping = blackWhiteSamples()
    coverage = np.ones((2, 2), bool)
    native = transferTexture(texture, coverage, mapping)
    gamma = transferTexture(texture, coverage, mapping, colorEncoding="gamma2.2")
    np.testing.assert_allclose(native["texture"][0, 1], [0.5] * 3, atol=1e-7)
    np.testing.assert_allclose(gamma["texture"][0, 1], [0.5 ** (1 / 2.2)] * 3, atol=1e-7)
    np.testing.assert_array_equal(gamma["texture"][0, 0], [0, 0, 0])
    np.testing.assert_array_equal(gamma["texture"][0, 2], [1, 1, 1])
    assert gamma["validMask"].all()  # A measured black pixel remains valid.
    assert gamma["diagnostics"]["interpolationSpace"] == "linearLight"
    assert gamma["diagnostics"]["colorEncoding"] == "gamma2.2"
    assert native["diagnostics"]["interpolationSpace"] == "native"


def testGammaInterpolationPreservesCoverageHoles():
    texture, mapping = blackWhiteSamples()
    coverage = np.array([[True, False], [True, False]])
    gamma = transferTexture(texture, coverage, mapping, colorEncoding="gamma2.2")
    np.testing.assert_array_equal(gamma["validMask"], [[True, False, False]])
    assert not gamma["texture"].any()


def testScalarNeverReceivesColorGammaConversion():
    texture, mapping = blackWhiteSamples()
    scalar = transferTexture(texture[..., 0], np.ones((2, 2)), mapping, materialKind="scalar")
    assert scalar["texture"][0, 1] == 0.5
    with pytest.raises(ValueError, match="only for color"):
        transferTexture(texture[..., 0], np.ones((2, 2)), mapping,
                        materialKind="scalar", colorEncoding="gamma2.2")


def testColorEncodingMustBeExplicitAndGammaInputNonnegative():
    texture, mapping = blackWhiteSamples()
    with pytest.raises(ValueError, match="colorEncoding"):
        transferTexture(texture, np.ones((2, 2)), mapping, colorEncoding="srgb")
    texture[0, 0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        transferTexture(texture, np.ones((2, 2)), mapping, colorEncoding="gamma2.2")
