import numpy as np
import pytest

from faceAppearance.correspondence import SurfaceIndex, qualifyMapping, sampleSurface


def testExactProjectionIncludesLargeTriangleWithDistantCentroid():
    # Centroid-kNN alone would choose the small elevated triangle incorrectly.
    vertices = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    index = SurfaceIndex(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    result = index.query(np.array([[0.1, 0.1, 0.2]]))
    assert result["triangleIds"].tolist() == [0]
    np.testing.assert_allclose(result["distances"], [0.2], atol=1e-6)
    np.testing.assert_allclose(sampleSurface(vertices, index.faces, result), [[0.1, 0.1, 0]], atol=1e-5)


def testSpatialPruningMatchesIndependentExhaustiveProjection():
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1)
    index = SurfaceIndex(mesh.vertices, mesh.faces)
    points = np.random.default_rng(42).normal(size=(70, 3))
    exhaustive, distances, _ = trimesh.proximity.closest_point_naive(mesh, points)
    actual = index.query(points, chunkSize=11)
    np.testing.assert_allclose(actual["distances"], distances, atol=1e-6)
    np.testing.assert_allclose(actual["closestPoints"], exhaustive, atol=1e-6)


def testOppositeNormalAndOutsideCommonSurfaceStayInvalid():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    index = SurfaceIndex(vertices, np.array([[0, 1, 2]]))
    mapping = index.query(np.array([[0.2, 0.2, 0], [0.2, 0.2, 0], [0.2, 0.2, 2]]))
    result = qualifyMapping(mapping, index, np.array([[0, 0, 1], [0, 0, -1], [0, 0, 1]]), 0.05)
    assert result["validMask"].tolist() == [True, False, False]


def testTriangleEdgesAreValidAndWeightsReproduceSurface():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    index = SurfaceIndex(vertices, np.array([[0, 1, 2]]))
    result = index.query(np.array([[1, 1, 0], [-1, 0, 0]]))
    np.testing.assert_allclose(result["closestPoints"], [[0.5, 0.5, 0], [0, 0, 0]])
    np.testing.assert_allclose(result["barycentricWeights"].sum(axis=1), 1)
    with pytest.raises(ValueError):
        index.query(np.full((1, 3), np.nan))
