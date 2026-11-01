import copy

import numpy as np
import pytest

from faceAppearance.camera import projectPoints
from faceAppearance.eyeRefinement import (buildEyeDeformationBasis, refineEyeGeometry,
                                         _preparePointConstraints, _projectWithJacobian)


def makeGrid(xs=None, ys=None, dtype=np.float64):
    xs = np.linspace(-2, 2, 21) if xs is None else np.asarray(xs)
    ys = np.linspace(-2, 2, 21) if ys is None else np.asarray(ys)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).astype(dtype)
    faces = []
    for row in range(len(ys)-1):
        for column in range(len(xs)-1):
            first = row*len(xs)+column
            faces.extend([[first, first+1, first+len(xs)],
                          [first+1, first+len(xs)+1, first+len(xs)]])
    return vertices, np.asarray(faces)


def makeCameras():
    angle = .3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return [{"intrinsics": np.array([[100., 0, 80], [0, 100., 80], [0, 0, 1.]]),
             "rotation": rot, "translation": np.array([0., 0., 5.])} for rot in (np.eye(3), rotation)]


def pointConstraint(indices, targets, camera, viewIndex=0, **extra):
    return {"viewIndex": viewIndex, "name": "explicitEyePoints", "vertexIndices": np.asarray(indices)[:, None],
            "barycentric": np.ones((len(indices), 1)), "targetPixels": projectPoints(targets, **camera)[0], **extra}


def testAnalyticProjectionJacobianMatchesIndependentFiniteDifferences():
    camera = makeCameras()[1]
    points = np.array([[.1, -.2, .3], [-.5, .2, -.1]])
    pixels, depth, derivative = _projectWithJacobian(points, camera)
    np.testing.assert_allclose(pixels, projectPoints(points, **camera)[0], atol=1e-12)
    assert (depth > 0).all()
    numeric = np.zeros_like(derivative)
    for dimension in range(3):
        positive, negative = points.copy(), points.copy()
        positive[:, dimension] += 1e-6
        negative[:, dimension] -= 1e-6
        numeric[:, :, dimension] = (projectPoints(positive, **camera)[0]-projectPoints(negative, **camera)[0])/2e-6
    np.testing.assert_allclose(derivative, numeric, atol=1e-8)


def testCompactContinuousFieldMovesTinyEdgesTogetherAndPinsOutsideRadius():
    points = np.array([[0., 0, 0], [1e-8, 0, 0], [.999999, 0, 0], [1., 0, 0], [2., 0, 0]])
    field = buildEyeDeformationBasis(points, np.array([[0., 0, 0]]), supportRadius=1)
    difference = np.abs(field["basis"][0]-field["basis"][1]).max()
    assert difference < 1e-12
    assert field["basis"][2, 0] < 1e-20
    np.testing.assert_array_equal(field["basis"][3:], 0)
    np.testing.assert_array_equal(field["activeVertexMask"], [True, True, True, False, False])


def testFivePixelCornerRotationRecoversMostShapeWithoutRelaxingThirtyPercentGate():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    indices = np.array([10*21+5, 10*21+15, 8*21+10, 12*21+10])
    angle = np.deg2rad(15)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    target = vertices[indices]@rotation.T+np.array([.12, .04, 0])
    constraints = [pointConstraint(indices, target, camera, index) for index, camera in enumerate(cameras)]
    originals = copy.deepcopy((vertices, faces, cameras, constraints))
    result = refineEyeGeometry(vertices, faces, cameras, constraints, supportRadius=2.7,
                              maxDisplacement=.5, controlCount=4, iterations=60,
                              displacementWeight=1e-5, strainWeight=.001, controlWeight=1e-6)
    report = result["diagnostics"]
    assert report["initialMeanErrorPixels"] > 3
    assert report["finalMeanErrorPixels"] < .1
    assert report["safety"]["maximumRelativeEdgeChange"] <= .30
    assert report["safety"]["minimumNormalCosine"] > 0
    assert report["safety"]["minimumAreaRatio"] >= .25
    assert np.linalg.norm(result["vertices"][indices]-vertices[indices], axis=1).max() > .2
    np.testing.assert_array_equal(result["vertices"][~result["activeVertexMask"]], vertices[~result["activeVertexMask"]])
    np.testing.assert_array_equal(vertices, originals[0])
    np.testing.assert_array_equal(faces, originals[1])
    for current, previous in zip(cameras, originals[2]):
        for key in current:
            np.testing.assert_array_equal(current[key], previous[key])
    for current, previous in zip(constraints, originals[3]):
        np.testing.assert_array_equal(current["targetPixels"], previous["targetPixels"])


def testNearbyTinyEdgeDoesNotBlockCoherentTranslation():
    vertices, faces = makeGrid(xs=[-2, -1, 0, 1e-7, 1, 2], ys=[-2, -1, 0, 1, 2])
    cameras = makeCameras()[:1]
    ids = [2*6+2, 2*6+3]
    constraint = pointConstraint(ids, vertices[ids]+[.12, .05, 0], cameras[0])
    result = refineEyeGeometry(vertices, faces, cameras, [constraint], supportRadius=2.5,
                              maxDisplacement=.3, iterations=40, strainWeight=.001)
    assert result["diagnostics"]["status"] == "accepted"
    assert result["diagnostics"]["finalMeanErrorPixels"] < .03
    assert result["diagnostics"]["safety"]["maximumRelativeEdgeChange"] < .30
    np.testing.assert_allclose(np.diff(result["vertices"][ids], axis=0), [[1e-7, 0, 0]], atol=1e-9)


def testExtremeWrongTargetRejectsEntireFitAndReturnsOriginalMesh():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    ids = np.array([210, 220, 230])
    constraint = pointConstraint(ids, vertices[ids], cameras[0])
    constraint["targetPixels"][1] += [100, -80]
    result = refineEyeGeometry(vertices, faces, cameras, [constraint], iterations=50)
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert result["diagnostics"]["status"] == "unchanged"
    assert result["diagnostics"]["inputTargetGate"]["entireFitRejected"]
    assert result["history"] == []


def testExplicitSegmentPairsArePreferredAndMissingCurveSamplesAreNotBridged():
    camera = makeCameras()[0]
    points = np.array([[-1., 0, 0], [-.5, 0, 0], [.5, 0, 0], [1., 0, 0]])
    constraint = pointConstraint(np.arange(4), points, camera, orderedCurve=True,
                                 sampleIndices=np.array([0, 1, 3, 4]))
    prepared = _preparePointConstraints([constraint], 4, [camera])
    np.testing.assert_array_equal(prepared[0]["segmentPairs"], [[0, 1], [2, 3]])
    constraint["segmentPairs"] = np.array([[2, 3]])
    prepared = _preparePointConstraints([constraint], 4, [camera])
    np.testing.assert_array_equal(prepared[0]["segmentPairs"], [[2, 3]])


@pytest.mark.parametrize("explicitPairs", [False, True])
def testZeroWeightEndpointTargetCannotAffectCurveFit(explicitPairs):
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    ids = np.array([10*21+6, 10*21+8, 10*21+10, 10*21+12])
    target = vertices[ids]+[.03, .08, 0]
    constraint = pointConstraint(ids, target, cameras[0], orderedCurve=True,
                                 weights=np.array([1., 1., 0., 1.]))
    if explicitPairs:
        constraint["segmentPairs"] = np.array([[0, 1], [1, 2], [2, 3]])
    prepared = _preparePointConstraints([constraint], len(vertices), cameras)
    np.testing.assert_array_equal(prepared[0]["segmentPairs"], [[0, 1]])
    options = {"supportRadius": 2.7, "maxDisplacement": .5, "iterations": 40, "strainWeight": .001}
    baseline = refineEyeGeometry(vertices, faces, cameras, [constraint], **options)
    changed = copy.deepcopy(constraint)
    changed["targetPixels"][2] += [250., -180.]
    result = refineEyeGeometry(vertices, faces, cameras, [changed], **options)
    assert baseline["diagnostics"]["status"] == "accepted"
    np.testing.assert_array_equal(result["vertices"], baseline["vertices"])
    np.testing.assert_array_equal(result["controlOffsets"], baseline["controlOffsets"])
    assert not result["diagnostics"]["inputTargetGate"]["entireFitRejected"]
    assert result["diagnostics"]["finalGroups"][0]["curveSegmentCount"] == 1


def testCurveDirectionResidualAndReturnedGeometryMetricsDescribeRealProjection():
    vertices, faces = makeGrid(dtype=np.float32)
    cameras = makeCameras()[:1]
    ids = np.array([10*21+6, 10*21+8, 10*21+10, 10*21+12, 10*21+14])
    target = vertices[ids].astype(float)
    target[:, 1] += .2*target[:, 0]+.06
    constraint = pointConstraint(ids, target, cameras[0], orderedCurve=True,
                                 targetTangents=np.tile([1., .2], (len(ids), 1)))
    result = refineEyeGeometry(vertices, faces, cameras, [constraint], supportRadius=2.7,
                              maxDisplacement=.5, iterations=60, strainWeight=.001)
    report = result["diagnostics"]
    assert report["initialGroups"][0]["meanCurveDirectionErrorDegrees"] > 10
    assert report["finalGroups"][0]["meanCurveDirectionErrorDegrees"] < .5
    pixels = projectPoints(result["vertices"][ids].astype(float), **cameras[0])[0]
    actual = np.linalg.norm(pixels-constraint["targetPixels"], axis=1).mean()
    assert report["finalMeanErrorPixels"] == pytest.approx(actual, abs=1e-12)
    assert report["safety"]["maximumRelativeEdgeChange"] <= .30
    assert result["vertices"].dtype == np.float32


def testDirectionalPointWeightIsDistinctFromCurveDirectionConstraint():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    ids = np.array([10*21+10])
    constraint = pointConstraint(ids, vertices[ids]+[.2, 0, 0], cameras[0],
                                 targetTangents=np.array([[1., 0]]), tangentWeight=0)
    result = refineEyeGeometry(vertices, faces, cameras, [constraint], supportRadius=2)
    # Only tangent-position error exists and its weight is zero; there are no
    # segment pairs, so the solver must not claim a fitted curve orientation.
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert result["diagnostics"]["finalGroups"][0]["meanCurveDirectionErrorDegrees"] is None


def testNoConstraintsAndZeroIterationsReturnExactOriginal():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    result = refineEyeGeometry(vertices, faces, cameras, [])
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert not result["activeVertexMask"].any()
    constraint = pointConstraint([220], vertices[[220]]+[.1, 0, 0], cameras[0])
    result = refineEyeGeometry(vertices, faces, cameras, [constraint], iterations=0)
    np.testing.assert_array_equal(result["vertices"], vertices)


@pytest.mark.parametrize("options", [{"supportRadius": 0}, {"controlCount": True}, {"iterations": -1},
                                    {"maxRelativeEdgeChange": 1}, {"curveWeight": -1},
                                    {"maxTargetDistancePixels": 0}, {"initialDamping": 0}])
def testInvalidSettingsRejected(options):
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    constraint = pointConstraint([220], vertices[[220]], cameras[0])
    with pytest.raises(ValueError):
        refineEyeGeometry(vertices, faces, cameras, [constraint], **options)
