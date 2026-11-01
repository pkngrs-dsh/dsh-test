"""Contracts for visible, bounded multiview nasal deformation."""
import numpy as np
import pytest

from faceAppearance.eyeRefinement import _projectWithJacobian
from faceAppearance.noseGeometryRefinement import (eyeMouthAnchors, noseAnchorFractions,
    refineNoseGeometry, visibleNoseBindings)


def fixtureMesh():
    y, x = np.mgrid[-.2:.21:.1, -.2:.21:.1]
    vertices = np.c_[x.ravel(), y.ravel(), np.full(x.size, 2.)]
    faces = []
    for row in range(4):
        for col in range(4):
            first = row * 5 + col
            faces.extend([[first, first + 5, first + 1], [first + 1, first + 5, first + 6]])
    camera = {'intrinsics': np.array([[500., 0, 256], [0, 500, 256], [0, 0, 1]]),
              'rotation': np.eye(3), 'translation': np.zeros(3)}
    angle = .4
    other = dict(camera, rotation=np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]))
    other['translation'] = np.array([-2 * np.sin(angle), 0, 2 - 2 * np.cos(angle)])
    active = np.zeros((5, 5), bool)
    active[1:4, 1:4] = True
    return vertices, np.asarray(faces), [camera, other], active.ravel()


def observations(vertices, cameras, delta):
    return [{'viewIndex': view, 'vertexIndices': np.array([[12]]), 'barycentric': np.ones((1, 1)),
             'targetPixels': _projectWithJacobian(vertices[[12]] + delta, camera)[0]}
            for view, camera in enumerate(cameras)]


def testMultiviewSupportsDepthWithoutMovingHardFixedVertices():
    vertices, faces, cameras, active = fixtureMesh()
    desired = np.array([.003, -.002, .006])
    constraints = observations(vertices, cameras, desired)
    candidate, diagnostics = refineNoseGeometry(vertices, faces, cameras, constraints,
        activeMask=active, maxDisplacement=.02, smoothnessWeight=.02, strainWeight=.001)
    np.testing.assert_array_equal(vertices[~active], candidate[~active])
    np.testing.assert_allclose(candidate[12] - vertices[12], desired, atol=.001)
    assert diagnostics['linearizedSolveCount'] == 1
    assert not diagnostics['qualityAccepted']
    for item in diagnostics['pointReports']:
        assert item['proposalErrorPixels'].mean() < item['beforeErrorPixels'].mean()


def testUniformBoundAndZeroTarget():
    vertices, faces, cameras, active = fixtureMesh()
    candidate, diagnostics = refineNoseGeometry(vertices, faces, cameras,
        observations(vertices, cameras, np.array([.3, -.2, .1])), activeMask=active, maxDisplacement=.003)
    assert diagnostics['boundScale'] < 1
    assert np.linalg.norm(candidate - vertices, axis=1).max() <= .003 + 1e-14
    np.testing.assert_array_equal(vertices[~active], candidate[~active])
    unchanged, _ = refineNoseGeometry(vertices, faces, cameras, observations(vertices, cameras, np.zeros(3)),
        activeMask=active, maxDisplacement=.02)
    np.testing.assert_array_equal(unchanged, vertices)


def testSurfaceBindingsCannotMoveUnavailableSupport():
    vertices, faces, cameras, active = fixtureMesh()
    constraints = observations(vertices, cameras, np.ones(3) * .01)
    active[12] = False
    with pytest.raises(ValueError, match='no active surface support'):
        refineNoseGeometry(vertices, faces, cameras, constraints, activeMask=active, maxDisplacement=.02)


def testNearestVisibleSurfaceIsUsedRatherThanHiddenNasalSeed():
    vertices, faces, cameras, _ = fixtureMesh()
    back = vertices.copy(); back[:, 2] += .3
    both = np.r_[vertices, back]
    joined = np.r_[faces, faces + len(vertices)]
    result = visibleNoseBindings(both, joined, cameras[0], (512, 512), np.array([[258., 259.]]),
        domainDistance=np.zeros(len(both)), maximumDomainDistance=.1)
    assert result['acceptedMask'].all()
    assert result['vertexIndices'].max() < len(vertices)
    np.testing.assert_allclose(result['canonicalPoints'][:, 2], 2.)
    pixels, _, _ = _projectWithJacobian(result['canonicalPoints'], cameras[0])
    np.testing.assert_allclose(pixels, [[258., 259.]], atol=1e-10)


def testRayGatesRejectProtectedBackFacingAndOutsideDomain():
    vertices, faces, cameras, _ = fixtureMesh()
    kwargs = dict(camera=cameras[0], imageShape=(512, 512), pixels=np.array([[258., 259.]]),
                  domainDistance=np.zeros(len(vertices)), maximumDomainDistance=.1)
    ordinary = visibleNoseBindings(vertices, faces, **kwargs)
    protected = visibleNoseBindings(vertices, faces, fixedVertexIds=ordinary['vertexIndices'].ravel(), **kwargs)
    assert 'protectedTriangle' in protected['reasons'][0]
    reversedFaces = visibleNoseBindings(vertices, faces[:, ::-1], **kwargs)
    assert 'grazingOrBackFacing' in reversedFaces['reasons'][0]
    kwargs['domainDistance'] = np.ones(len(vertices))
    outside = visibleNoseBindings(vertices, faces, **kwargs)
    assert 'outsideBroadNoseDomain' in outside['reasons'][0]


def testRelativeNoseCoordinatesIgnoreSharedImageTranslationButKeepAsymmetry():
    points = np.zeros((68, 2))
    points[[36, 39]] = [200., 200.]; points[[42, 45]] = [300., 200.]
    points[[48, 54]] = [250., 360.]
    points[27:31] = [[253., 202.], [254., 230.], [256., 255.], [258., 286.]]
    ids = np.arange(27, 31)
    fractions = noseAnchorFractions(points, ids)
    relative = points[ids] - eyeMouthAnchors(points, fractions)
    translated = points + [12., -9.]
    np.testing.assert_allclose(relative, translated[ids] - eyeMouthAnchors(translated, fractions))
    assert relative[-1, 0] == 8.  # The method does not force a symmetric nose.


def testConstraintOrderDoesNotChangeTheField():
    vertices, faces, cameras, active = fixtureMesh()
    constraints = observations(vertices, cameras, np.array([.004, .001, .003]))
    forward, _ = refineNoseGeometry(vertices, faces, cameras, constraints, activeMask=active, maxDisplacement=.02)
    reverse, _ = refineNoseGeometry(vertices, faces, cameras, constraints[::-1], activeMask=active, maxDisplacement=.02)
    np.testing.assert_allclose(forward, reverse, atol=1e-12)


@pytest.mark.parametrize('maximumDistance,minimumCosine', [
    (np.nan, .08), (np.inf, .08), (-1., .08), (0., .08),
    (.1, np.nan), (.1, np.inf), (.1, -.1), (.1, 1.)])
def testVisibleBindingsRejectNonfiniteOrOutOfRangeThresholds(maximumDistance, minimumCosine):
    vertices, faces, cameras, _ = fixtureMesh()
    with pytest.raises(ValueError, match='nasal anatomical domain'):
        visibleNoseBindings(vertices, faces, cameras[0], (512, 512), np.array([[256., 256.]]),
            domainDistance=np.zeros(len(vertices)), maximumDomainDistance=maximumDistance,
            minimumViewCosine=minimumCosine)


@pytest.mark.parametrize('invalidDistance', [np.nan, -np.inf, -.01])
def testVisibleBindingsRejectInvalidGeodesicDistances(invalidDistance):
    vertices, faces, cameras, _ = fixtureMesh()
    distance = np.zeros(len(vertices)); distance[0] = invalidDistance
    with pytest.raises(ValueError, match='nasal anatomical domain'):
        visibleNoseBindings(vertices, faces, cameras[0], (512, 512), np.array([[256., 256.]]),
            domainDistance=distance, maximumDomainDistance=.1)


def testUnreachableTriangleVertexCannotPassThroughZeroTimesInfinity():
    vertices, faces, cameras, _ = fixtureMesh()
    kwargs = dict(camera=cameras[0], imageShape=(512, 512), pixels=np.array([[256., 256.]]),
                  maximumDomainDistance=.1)
    ordinary = visibleNoseBindings(vertices, faces, domainDistance=np.zeros(len(vertices)), **kwargs)
    zeroColumn = int(np.abs(ordinary['barycentric'][0]).argmin())
    assert abs(ordinary['barycentric'][0, zeroColumn]) < 1e-12
    distance = np.zeros(len(vertices))
    distance[ordinary['vertexIndices'][0, zeroColumn]] = np.inf
    result = visibleNoseBindings(vertices, faces, domainDistance=distance, **kwargs)
    assert not result['acceptedMask'][0]
    assert np.isposinf(result['domainDistance'][0])
    assert result['reasons'][0] == ['outsideBroadNoseDomain']
