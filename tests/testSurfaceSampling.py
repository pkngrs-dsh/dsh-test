import numpy as np

from faceAppearance.surfaceSampling import invertUvMap


def testSemanticInverseCorrectsGeometryOffsetWithoutFlippingV():
    yy, xx = np.mgrid[:64, :64]
    uvMap = np.stack([xx, yy], axis=-1) / 63
    queries = np.array([[20, 30], [40, 15]], dtype=float)
    corrected, accepted = invertUvMap(uvMap, np.ones((64, 64)), queries / 63, queries + [5, -3])
    assert accepted.all()
    np.testing.assert_allclose(corrected, queries, atol=1e-6)


def testSemanticInverseRejectsUnrelatedProjectionAndUnobservedUv():
    yy, xx = np.mgrid[:64, :64]
    uvMap = np.stack([xx, yy], axis=-1) / 63
    mask = xx < 32
    _, accepted = invertUvMap(uvMap, mask, np.array([[20, 30], [60, 30]]) / 63,
                              np.array([[60, 30], [60, 30]]))
    assert not accepted.any()


def testFoldedSemanticMapDoesNotMixDistantImageRegions():
    uvMap = np.zeros((32, 32, 2))
    mask = np.zeros((32, 32))
    for y, x in ((0, 0), (0, 31), (31, 0), (31, 31)):
        uvMap[y, x] = [0.5, 0.5]
        mask[y, x] = 1
    _, accepted = invertUvMap(uvMap, mask, np.array([[0.5, 0.5]]), np.array([[15.5, 15.5]]))
    assert not accepted.any()
