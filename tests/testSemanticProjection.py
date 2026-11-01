import numpy as np
import pytest

from faceAppearance.semanticProjection import sampleSemanticObservations, visibleVertices


def testCategoricalSamplingKeepsConfidencePairedAndRejectsOutsideSupport():
    labels = np.array([[2, 12], [10, 13]], np.uint8)
    confidence = np.array([[.9, .8], [.7, .6]], np.float32)
    points = np.array([[[[.51, .1], [-.01, 0], [0, 1], [1, 1]]]], np.float32)
    weights = np.array([[[1, 1, 0, 1]]], np.float32)
    result = sampleSemanticObservations([labels], [confidence], points, weights)
    np.testing.assert_array_equal(result["semanticLabels"], [[[12, 0, 0, 13]]])
    np.testing.assert_allclose(result["semanticConfidence"], [[[.8, 0, 0, .6]]])


@pytest.mark.parametrize("invalid", [np.array([[19]]), np.array([[1.2]])])
def testRejectsAmbiguousLabels(invalid):
    with pytest.raises(ValueError):
        sampleSemanticObservations([invalid], [np.ones((1, 1))],
                                   np.zeros((1, 1, 1, 2)), np.ones((1, 1, 1)))


def testExactVertexVisibilityRecoversUnroundedBoundaryWithoutRevealingHiddenSurface():
    intrinsics = np.array([[16., 0, 16], [0, 16, 16], [0, 0, 1.]])
    pixels = np.array([[8.2, 8.2], [24.2, 8.2], [8.2, 24.2],
                       [12.2, 12.2], [13.2, 12.2], [12.2, 13.2]])
    depths = np.array([2., 2., 2., 3., 3., 3.])
    vertices = np.column_stack([(pixels-16)*depths[:, None]/16, depths])
    faces = np.array([[0, 2, 1], [3, 5, 4]])
    camera = {"intrinsics": intrinsics, "rotation": np.eye(3), "translation": np.zeros(3)}
    coarse = visibleVertices(vertices, faces, [camera], [(32, 32)])
    exact = visibleVertices(vertices, faces, [camera], [(32, 32)], visibilityMode="exact")
    assert not coarse[0, 0]  # The rounded sample lies outside the front triangle.
    assert exact[0, :3].all()
    assert not exact[0, 3:].any()  # A surface point cannot validate against its own hidden layer.
    with pytest.raises(ValueError, match="visibilityMode"):
        visibleVertices(vertices, faces, [camera], [(32, 32)], visibilityMode="unknown")
