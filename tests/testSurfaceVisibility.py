"""Visibility checks exercise independent analytic scenes, not implementation copies."""
import numpy as np
import pytest

from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility, refineRejectedVisibility
from faceAppearance.texture import rasterizeDepth


intrinsics = np.array([[16., 0, 16], [0, 16, 16], [0, 0, 1.]])


def cameraPoint(x, y, z):
    return np.array([(x-16)*z/16, (y-16)*z/16, z], dtype=float)


def screenSquare(depths=3., bounds=(4., 4., 28., 28.)):
    left, top, right, bottom = bounds
    depths = np.broadcast_to(depths, (4,))
    vertices = np.array([cameraPoint(x, y, z) for (x, y), z in zip(
        [(left, top), (right, top), (right, bottom), (left, bottom)], depths)])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def refine(vertices, faces, points, *, initial=None, candidate=None, tolerance=.001, **kwargs):
    points = np.asarray(points, float)
    if initial is None:
        initial = np.zeros(len(points), bool)
    if candidate is None:
        candidate = np.ones(len(points), bool)
    return refineRejectedVisibility(vertices, faces, intrinsics, points, initial,
        imageShape=(32, 32), depthTolerance=tolerance, candidateMask=candidate, **kwargs)


def testSteepSurfaceRasterNeighborRejectIsRecoveredAtSameTolerance():
    vertices, faces = screenSquare([2., 4., 4., 2.])
    x, y = 10.4, 14.2
    expectedDepth = 1/((28-x)/24/2+(x-4)/24/4)
    point = cameraPoint(x, y, expectedDepth)
    raster = rasterizeDepth(vertices, faces, intrinsics, 32, 32)
    assert abs(raster[round(y), round(x)]-point[2]) > .001
    result = refine(vertices, faces, [point], tolerance=.001)
    assert result['visible'].tolist() == [True]
    assert result['diagnostics']['recoveredCount'] == 1
    assert result['diagnostics']['depthTolerance'] == .001
    np.testing.assert_allclose(result['exactDepth'], [expectedDepth], atol=1e-12)


def testHiddenBackLayerCannotValidateAgainstItself():
    back, faces = screenSquare(4.)
    front, frontFaces = screenSquare(2., bounds=(10, 10, 22, 22))
    vertices = np.concatenate([back, front])
    triangles = np.concatenate([faces, frontFaces+4])
    points = [cameraPoint(16, 16, 4), cameraPoint(6, 16, 4), cameraPoint(16, 16, 2)]
    result = refine(vertices, triangles, points)
    assert result['visible'].tolist() == [False, True, True]
    np.testing.assert_allclose(result['exactDepth'], [2, 4, 2])


def testBothWindingsOccludeAsExistingDepthRasterizerDoes():
    back, faces = screenSquare(4.)
    front, frontFaces = screenSquare(2., bounds=(10, 10, 22, 22))
    vertices = np.concatenate([back, front])
    triangles = np.concatenate([faces, frontFaces[:, ::-1]+4])
    result = refine(vertices, triangles, [cameraPoint(16, 16, 4)])
    assert not result['visible'][0]
    assert result['exactDepth'][0] == pytest.approx(2.)


def testNormalCandidateGateIsNotRelaxedAndOldAcceptedMaskIsUntouched():
    vertices, faces = screenSquare()
    points = [cameraPoint(16, 16, 3), cameraPoint(17, 15, 3), cameraPoint(18, 15, 3)]
    initial = np.array([False, True, False])
    candidate = np.array([False, True, True])
    result = refine(vertices, faces, points, initial=initial, candidate=candidate)
    assert result['visible'].tolist() == [False, True, True]
    assert result['recoveredMask'].tolist() == [False, False, True]
    assert np.isnan(result['exactDepth'][:2]).all()
    np.testing.assert_array_equal(initial, [False, True, False])


@pytest.mark.parametrize('pixel', [(4., 4.), (16., 16.), (28., 28.), (28., 16.), (16., 4.)])
def testBoundarySharedEdgeAndBinBorderQueriesAreSupported(pixel):
    vertices, faces = screenSquare()
    result = refine(vertices, faces, [cameraPoint(*pixel, 3)], binSize=4)
    assert result['visible'][0]
    assert result['exactDepth'][0] == pytest.approx(3.)


def testQueriesOutsidePhotoBehindCameraAndEmptyRaysRemainRejected():
    vertices, faces = screenSquare()
    points = [cameraPoint(-.001, 16, 3), cameraPoint(32., 16, 3), cameraPoint(16, -.1, 3),
              cameraPoint(16, 32., 3), cameraPoint(16, 16, -3), cameraPoint(0, 0, 3)]
    result = refine(vertices, faces, points)
    assert not result['visible'].any()
    assert np.isinf(result['exactDepth']).all()
    assert result['diagnostics']['outsidePhotoOrBehindCameraCount'] == 5
    assert result['diagnostics']['noIntersectionCount'] == 1


def testProjectedDegenerateFacesAndNearPlaneCrossingsAreIgnored():
    vertices, faces = screenSquare()
    extraVertices = np.array([cameraPoint(16, 16, 1), cameraPoint(16, 16, 2), cameraPoint(16, 16, 3),
                              cameraPoint(5, 5, -1), cameraPoint(26, 5, 1), cameraPoint(16, 26, 1)])
    vertices = np.concatenate([vertices, extraVertices])
    faces = np.concatenate([faces, [[4, 5, 6], [7, 8, 9]]])
    result = refine(vertices, faces, [cameraPoint(16, 16, 3)])
    assert result['visible'][0]
    assert result['diagnostics']['degenerateProjectedTriangleCount'] == 1
    assert result['diagnostics']['nearPlaneSkippedTriangleCount'] == 1


def testNoIndexedTrianglesProducesNoHitRatherThanFakeSupport():
    vertices = np.array([cameraPoint(16, 16, 1), cameraPoint(16, 16, 2), cameraPoint(16, 16, 3)])
    result = refine(vertices, np.array([[0, 1, 2]]), [cameraPoint(16, 16, 3)])
    assert not result['visible'][0]
    assert result['diagnostics']['noIntersectionCount'] == 1
    assert result['diagnostics']['outsidePhotoOrBehindCameraCount'] == 0


def testChunkSizeBinSizeAndFaceOrderDoNotChangeDepthOrVisibility():
    vertices, faces = screenSquare([2., 4., 4., 2.])
    xs, ys = np.meshgrid(np.linspace(4.1, 27.9, 39), np.linspace(4.1, 27.9, 21))
    z = 1/((28-xs)/24/2+(xs-4)/24/4)
    points = np.array([cameraPoint(x, y, depth) for x, y, depth in zip(xs.ravel(), ys.ravel(), z.ravel())])
    small = refine(vertices, faces, points, chunkSize=1, binSize=3)
    large = refine(vertices, faces[::-1, ::-1], points, chunkSize=256, binSize=32)
    assert small['visible'].all()
    np.testing.assert_allclose(small['exactDepth'], z.ravel(), atol=1e-12)
    np.testing.assert_array_equal(small['visible'], large['visible'])
    np.testing.assert_allclose(small['exactDepth'], large['exactDepth'], atol=1e-12)


def testEmptyQueryBatchIsValid():
    vertices, faces = screenSquare()
    result = refine(vertices, faces, np.empty((0, 3)))
    assert result['visible'].shape == (0,)
    assert result['diagnostics']['eligibleRejectedCount'] == 0


@pytest.mark.parametrize('badArgument', ['candidateType', 'candidateShape', 'nonfinite', 'tolerance', 'binSize', 'chunkSize'])
def testInvalidInputsAreRejected(badArgument):
    vertices, faces = screenSquare()
    points = np.array([cameraPoint(16, 16, 3)])
    kwargs = {}
    if badArgument == 'candidateType':
        kwargs['candidate'] = np.ones(1, int)
    elif badArgument == 'candidateShape':
        kwargs['candidate'] = np.ones(2, bool)
    elif badArgument == 'nonfinite':
        points[0, 0] = np.nan
    elif badArgument == 'tolerance':
        kwargs['tolerance'] = 0
    else:
        kwargs[badArgument] = 0
    with pytest.raises(ValueError):
        refine(vertices, faces, points, **kwargs)
