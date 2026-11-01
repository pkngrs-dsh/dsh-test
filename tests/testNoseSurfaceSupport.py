"""Original-photo support gates on small meshes, without model inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('noseSurfaceRegistrationSupport',
    Path(__file__).resolve().parents[1] / 'tools/experimentNoseSurfaceRegistration.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plane(depth, lower=-4., upper=4.):
    vertices = np.array([[lower, lower, depth], [upper, lower, depth],
                         [upper, upper, depth], [lower, upper, depth]])
    # Both triangles face the cameras at the origin (negative Z normals).
    return vertices, np.array([[0, 2, 1], [0, 3, 2]])


def supportFixture():
    vertices, faces = plane(2.)
    source = {'vertices': vertices, 'faces': faces, 'faceMasks': np.ones((2, 8, 8))}
    cameras = [{'viewIndex': view, 'valid': True,
                'intrinsics': np.array([[2., 0., 3.], [0., 2., 3.], [0., 0., 1.]]),
                'rotation': np.eye(3), 'translation': np.array([view * .1, 0., 0.])}
               for view in range(2)]
    inputs = [{'workingSize': [8, 8]} for _ in cameras]
    return source, cameras, inputs


def support(source, cameras, inputs, points, targetNormals=None):
    points = np.asarray(points, dtype=float)
    if targetNormals is None:
        targetNormals = np.tile([0., 0., -1.], (len(points), 1))
    return MODULE.sourceVisibility(source, cameras, inputs, points,
                                   np.asarray(targetNormals, dtype=float))


def testUnobstructedFacingSurfaceHasSupportFromTwoDistinctViews():
    source, cameras, inputs = supportFixture()
    assert not np.array_equal(cameras[0]['translation'], cameras[1]['translation'])
    visible = support(source, cameras, inputs, [[.25, .25, 2.], [-.75, .5, 2.]])
    np.testing.assert_array_equal(visible, np.ones((2, 2), bool))
    np.testing.assert_array_equal(visible.sum(axis=0), [2, 2])


def testMatchingNormalCannotMakeAnOccludedSurfaceCountAsSupport():
    source, cameras, inputs = supportFixture()
    points = [[.25, .25, 2.], [-.75, .5, 2.]]
    np.testing.assert_array_equal(support(source, cameras, inputs, points), True)

    # The first ray now hits a nearer patch; the second remains unobstructed.
    # The patch and queried surface have identical normals, so only depth
    # visibility distinguishes them. A cosine-only implementation must fail.
    blockerVertices, blockerFaces = plane(1., lower=-.2, upper=.6)
    source['faces'] = np.r_[source['faces'], blockerFaces + len(source['vertices'])]
    source['vertices'] = np.r_[source['vertices'], blockerVertices]
    visible = support(source, cameras, inputs, points)
    np.testing.assert_array_equal(visible, [[False, True], [False, True]])


@pytest.mark.parametrize('tapX,tapY', [(3, 3), (4, 3), (3, 4), (4, 4)])
def testEveryMaskFootprintTapMustIndividuallyMeetSupportThreshold(tapX, tapY):
    source, cameras, inputs = supportFixture()
    # Query projects to (3.25, 3.25): all four taps have nonzero weights.
    # Their interpolated average is still > .5; testing just that average
    # would incorrectly admit a footprint containing this unsupported tap.
    source['faceMasks'][0, tapY, tapX] = .49
    visible = support(source, cameras, inputs, [[.25, .25, 2.], [-1.75, -1.75, 2.]])
    np.testing.assert_array_equal(visible, [[False, True], [True, True]])


@pytest.mark.parametrize('pixel', [(7., 3.25), (3.25, 7.)])
def testPhotoEdgeWithoutCompleteFourTapFootprintIsRejected(pixel):
    source, cameras, inputs = supportFixture()
    # At Z=2, this K maps world XY to pixel XY plus (3, 3).
    edgePoint = [pixel[0] - 3., pixel[1] - 3., 2.]
    interiorPoint = [min(pixel[0], 6.75) - 3., min(pixel[1], 6.75) - 3., 2.]
    visible = support(source, cameras, inputs, [edgePoint, interiorPoint])
    np.testing.assert_array_equal(visible, [[False, True], [False, True]])


def testBehindCameraPointIsRejectedEvenWhenItsNormalFacesTheCamera():
    source, cameras, inputs = supportFixture()
    behindVertices, behindFaces = plane(-2.)
    source['faces'] = np.r_[source['faces'], behindFaces[:, ::-1] + len(source['vertices'])]
    source['vertices'] = np.r_[source['vertices'], behindVertices]
    visible = support(source, cameras, inputs, [[.25, .25, 2.], [.25, .25, -2.]],
                      targetNormals=[[0., 0., -1.], [0., 0., 1.]])
    np.testing.assert_array_equal(visible, [[True, False], [True, False]])


def testBackFacingNormalIsRejectedDespiteAVisibleDepthHit():
    source, cameras, inputs = supportFixture()
    visible = support(source, cameras, inputs, [[.25, .25, 2.]],
                      targetNormals=[[0., 0., 1.]])
    np.testing.assert_array_equal(visible, [[False], [False]])


@pytest.mark.parametrize('maskCount,cameraCount,inputCount', [
    (1, 2, 2), (3, 2, 2), (2, 2, 1), (1, 1, 1)])
def testMissingOrMismatchedViewEvidenceFailsExplicitly(maskCount, cameraCount, inputCount):
    source, cameras, inputs = supportFixture()
    source['faceMasks'] = np.ones((maskCount, 8, 8))
    with pytest.raises(ValueError, match='finite face masks for every source camera'):
        support(source, cameras[:cameraCount], inputs[:inputCount], [[.25, .25, 2.]])
