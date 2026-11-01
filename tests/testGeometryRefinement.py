import copy

import numpy as np
import pytest
import torch

from faceAppearance.camera import projectPoints
from faceAppearance.geometryRefinement import (buildSemanticBoundaryConstraints, projectTensorPoints,
                                              refineLocalGeometry, _sampleTensorImage,
                                              _perspectiveEdgeBarycentric)


@pytest.fixture(autouse=True)
def singleThreadedTorch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def makeGrid():
    vertices = np.array([[-1, -1, 0], [0, -1, 0], [1, -1, 0],
                         [-1, 0, 0], [0, 0, 0], [1, 0, 0],
                         [-1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
                      [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]])
    return vertices, faces


def makeCameras():
    intrinsics = np.array([[100, 0, 80], [0, 100, 80], [0, 0, 1]], dtype=float)
    angle = .5
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return [{"intrinsics": intrinsics.copy(), "rotation": value,
             "translation": np.array([0, 0, 5.])} for value in (np.eye(3), rotation)]


def pointConstraint(camera, target, viewIndex=0, vertexIndex=4):
    return {"viewIndex": viewIndex, "name": "fixedAnatomicalPoint",
            "vertexIndices": np.array([[vertexIndex]]), "barycentric": np.ones((1, 1)),
            "targetPixels": projectPoints(np.asarray(target).reshape(1, 3), **camera)[0]}


def testProjectionMatchesOpenCvAndFiniteDifferences():
    camera = makeCameras()[1]
    values = np.array([[.1, -.2, .3], [-.3, .2, -.1]])
    points = torch.tensor(values, requires_grad=True, dtype=torch.float64)
    pixels, depth = projectTensorPoints(points, **camera)
    expectedPixels, expectedDepth = projectPoints(values, **camera)
    np.testing.assert_allclose(pixels.detach(), expectedPixels, atol=1e-12)
    np.testing.assert_allclose(depth.detach(), expectedDepth, atol=1e-12)
    coefficients = torch.tensor([[.5, .3], [-.6, .8]], dtype=torch.float64)
    (pixels * coefficients).sum().backward()
    numeric = np.empty_like(values)
    for pointIndex in range(2):
        for axis in range(3):
            positive, negative = values.copy(), values.copy()
            positive[pointIndex, axis] += 1e-6
            negative[pointIndex, axis] -= 1e-6
            numeric[pointIndex, axis] = ((projectPoints(positive, **camera)[0]
                                         - projectPoints(negative, **camera)[0]) * coefficients.numpy()).sum() / 2e-6
    np.testing.assert_allclose(points.grad, numeric, atol=1e-7)


def testDistanceImageSamplingHasCorrectPixelGradient():
    yy, xx = np.mgrid[:30, :40]
    distance = torch.tensor(2 * xx + 3 * yy - 25, dtype=torch.float64)
    pixels = torch.tensor([[4.2, 7.3], [11.7, 9.2]], dtype=torch.float64, requires_grad=True)
    samples = _sampleTensorImage(distance, pixels)
    samples.sum().backward()
    np.testing.assert_allclose(samples.detach(), 2 * pixels.detach()[:, 0] + 3 * pixels.detach()[:, 1] - 25)
    np.testing.assert_allclose(pixels.grad, [[2, 3], [2, 3]], atol=1e-12)


def testCameraTensorsNeverReceiveGradients():
    camera = {key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
              for key, value in makeCameras()[0].items()}
    points = torch.tensor([[.1, .1, .1]], dtype=torch.float64, requires_grad=True)
    pixels, _ = projectTensorPoints(points, **camera)
    pixels.sum().backward()
    assert points.grad is not None
    assert all(value.grad is None for value in camera.values())


def testMultiviewPointRecoversDepthWithoutChangingTopologyCamerasOrPinnedVertices():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    originals = copy.deepcopy((vertices, faces, cameras))
    target = vertices[4] + [.08, .04, .2]
    constraints = [pointConstraint(camera, target, index) for index, camera in enumerate(cameras)]
    result = refineLocalGeometry(vertices, faces, cameras, constraints, maxDisplacement=.3,
                                 iterations=250, activeVertexMask=np.arange(9) == 4,
                                 displacementWeight=.0001, laplacianWeight=.0001, edgeWeight=.001)
    np.testing.assert_allclose(result["vertices"][4], target, atol=.002)
    np.testing.assert_array_equal(result["vertices"][np.arange(9) != 4], vertices[np.arange(9) != 4])
    np.testing.assert_array_equal(vertices, originals[0])
    np.testing.assert_array_equal(faces, originals[1])
    for current, original in zip(cameras, originals[2]):
        for key in current:
            np.testing.assert_array_equal(current[key], original[key])
    report = result["diagnostics"]
    assert report["status"] == "accepted"
    assert report["finalMeanErrorPixels"] < .02
    assert report["finalMeanErrorPixels"] < report["initialMeanErrorPixels"] * .01
    assert report["camerasFrozen"] and report["uvAndVertexOrderPreserved"]
    assert report["activeVertexCount"] == 1


def testReferenceProjectionCanStayFixedWhileSideViewRecoversDepth():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    target = vertices[4] + [0, 0, .15]
    constraints = [pointConstraint(camera, target, index) for index, camera in enumerate(cameras)]
    result = refineLocalGeometry(vertices, faces, cameras, constraints, maxDisplacement=.25,
                                 iterations=200, maxViewDegradation=.03, activeVertexMask=np.arange(9) == 4,
                                 displacementWeight=.0001, laplacianWeight=.0001, edgeWeight=.001)
    report = result["diagnostics"]
    assert report["initialViewErrors"][0] == 0
    assert report["finalViewErrors"][0] < .03
    assert report["finalViewErrors"][1] < .04
    assert abs(result["vertices"][4, 2] - .15) < .006


def testDisplacementBoundAndTriangleSafetyRejectImpossibleTarget():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    constraint = pointConstraint(cameras[0], [4, 3, 0])
    result = refineLocalGeometry(vertices, faces, cameras, [constraint], maxDisplacement=4,
                                 iterations=45, learningRate=.5, activeVertexMask=np.arange(9) == 4,
                                 displacementWeight=0, laplacianWeight=0, edgeWeight=0)
    report = result["diagnostics"]
    assert report["rejections"]["meshSafety"] > 0
    assert report["safety"]["minimumAreaRatio"] >= .25
    assert report["safety"]["minimumNormalCosine"] >= .1
    assert report["safety"]["maximumRelativeEdgeChange"] <= .35
    assert report["safety"]["maximumDisplacement"] <= 4 * (1 + 1e-7)


def testRadialBoundIsEnforcedForEveryAcceptedOutput():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    result = refineLocalGeometry(vertices, faces, cameras, [pointConstraint(cameras[0], [.8, .8, 0])],
                                 maxDisplacement=.04, iterations=80, learningRate=.3,
                                 displacementWeight=0, laplacianWeight=0, edgeWeight=0)
    assert result["diagnostics"]["status"] == "accepted"
    assert np.linalg.norm(result["vertices"] - vertices, axis=1).max() <= .04000001


def testViewDegradationGateCannotBeBypassedByLoweringReferenceLossWeight():
    vertices, faces = makeGrid()
    camera = makeCameras()[0]
    cameras = [copy.deepcopy(camera), copy.deepcopy(camera)]
    constraints = [pointConstraint(camera, [0, 0, 0], 0), pointConstraint(camera, [.2, 0, 0], 1)]
    result = refineLocalGeometry(vertices, faces, cameras, constraints, maxDisplacement=.3,
                                 iterations=30, referenceWeight=.01, maxViewDegradation=0,
                                 activeVertexMask=np.arange(9) == 4)
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert result["diagnostics"]["rejections"]["viewDegradation"] > 0
    assert result["diagnostics"]["status"] == "unchanged"


def testUnimprovableOrEmptyConstraintsReturnOriginalMesh():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    for constraints in ([], [pointConstraint(cameras[0], vertices[4])]):
        result = refineLocalGeometry(vertices, faces, cameras, constraints, iterations=15)
        np.testing.assert_array_equal(result["vertices"], vertices)
        assert result["diagnostics"]["status"] == "unchanged"
        assert result["diagnostics"]["bestIteration"] == 0


def testNoActiveVerticesAndZeroIterationsAreHonestNoops():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    constraint = pointConstraint(cameras[0], [.1, .1, 0])
    for options in ({"activeVertexMask": np.zeros(9, bool)}, {"iterations": 0}):
        result = refineLocalGeometry(vertices, faces, cameras, [constraint], **options)
        np.testing.assert_array_equal(result["vertices"], vertices)
        assert result["diagnostics"]["status"] == "unchanged"


def makeSemanticInputs():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    yy, xx = np.mgrid[:160, :160]
    labels = [np.where(xx < 80, 12, 1).astype(np.uint8), np.where(xx < 81, 12, 1).astype(np.uint8)]
    confidences = [np.ones_like(xx, dtype=float) for _ in cameras]
    visibility = np.ones((2, 9), bool)
    return vertices, faces, cameras, labels, confidences, visibility


def testSemanticBindingsAreFixedAndOnlyUseInitiallyVisibleConfidentSamples():
    inputs = makeSemanticInputs()
    vertices, faces, cameras, labels, confidences, visibility = inputs
    visibility[1, [0, 1, 2]] = False
    labelsBefore = [label.copy() for label in labels]
    built = buildSemanticBoundaryConstraints(*inputs, {"lip": {"labelIds": [12], "weight": 1.}})
    assert len(built["constraints"]) == 2
    first, second = built["constraints"]
    assert first["viewIndex"] == 0 and second["viewIndex"] == 1
    assert len(second["vertexIndices"]) < len(first["vertexIndices"])
    assert visibility[1, second["vertexIndices"]].all()
    np.testing.assert_allclose(first["barycentric"].sum(1), 1)
    before = copy.deepcopy(built["constraints"])
    refineLocalGeometry(vertices, faces, cameras, built["constraints"], iterations=20)
    for current, saved in zip(built["constraints"], before):
        for key in ("vertexIndices", "barycentric", "targetDistance", "weights"):
            np.testing.assert_array_equal(current[key], saved[key])
    for label, original in zip(labels, labelsBefore):
        np.testing.assert_array_equal(label, original)
    assert built["diagnostics"]["visibilityFrozen"]


def testMissingSemanticsAndLowConfidenceCreateNoFalseGeometryObservations():
    vertices, faces, cameras, labels, confidences, visibility = makeSemanticInputs()
    groups = {"missing": {"labelIds": [99], "weight": 1.}}
    assert buildSemanticBoundaryConstraints(vertices, faces, cameras, labels, confidences, visibility,
                                            groups)["constraints"] == []
    confidences[0][:] = .1
    groups = {"lip": {"labelIds": [12], "weight": 1.}}
    built = buildSemanticBoundaryConstraints(vertices, faces, cameras, labels, confidences, visibility, groups)
    assert built["constraints"] == []
    assert built["diagnostics"]["groups"]["lip"]["referenceSamples"] == 0


def testDistantSideContourIsExcludedInsteadOfForcingBadCorrespondence():
    vertices, faces, cameras, labels, confidences, visibility = makeSemanticInputs()
    _, xx = np.mgrid[:160, :160]
    labels[1] = np.where(xx < 120, 12, 1).astype(np.uint8)
    result = buildSemanticBoundaryConstraints(vertices, faces, cameras, labels, confidences, visibility,
                                             {"lip": {"labelIds": [12]}})
    assert {item["viewIndex"] for item in result["constraints"]} == {0}
    assert result["diagnostics"]["groups"]["lip"]["viewSamples"][1] == 0


def testPerspectiveCorrectBoundaryBindingRemovesScreenLinearInterpolationBias():
    camera = makeCameras()[0]
    points = np.array([[-1., .2, -2.], [2., .4, 2.]])
    pixels, depth = projectPoints(points, **camera)
    screenFraction = np.array([.37])
    barycentric = _perspectiveEdgeBarycentric(screenFraction, depth[None])
    boundPoint = (points[None] * barycentric[..., None]).sum(1)
    expected = pixels[0] * .63 + pixels[1] * .37
    np.testing.assert_allclose(projectPoints(boundPoint, **camera)[0][0], expected, atol=1e-12)
    wrongPoint = points[0] * .63 + points[1] * .37
    assert np.linalg.norm(projectPoints(wrongPoint[None], **camera)[0][0] - expected) > 3

    vertices, faces, cameras, labels, confidences, visibility = makeSemanticInputs()
    vertices[:, 2] = vertices[:, 0] * .8
    result = buildSemanticBoundaryConstraints(vertices, faces, cameras, labels, confidences, visibility,
                                             {"lip": {"labelIds": [12]}})
    initialError = result["diagnostics"]["groups"]["lip"]["initialReferenceBoundaryErrorPixels"]
    assert initialError["max"] < 1e-4


def testSmoothControlsFitAcrossTinyEdgesWithoutRelaxingTriangleSafety():
    xs = np.array([-1, -.4, 0, .0001, .4, 1])
    ys = np.array([-1, -.5, 0, .5, 1])
    vertices = np.array([[x, y, 0] for y in ys for x in xs], dtype=float)
    faces = []
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            first = row * len(xs) + column
            second, third = first + 1, first + len(xs)
            faces.extend([[first, second, third], [second, third + 1, third]])
    faces = np.array(faces)
    cameras = makeCameras()
    center = 2 * len(xs) + 2
    target = vertices[center] + [0, 0, .2]
    constraints = [pointConstraint(camera, target, index, center) for index, camera in enumerate(cameras)]
    options = {"supportRings": 8, "maxDisplacement": .25, "iterations": 200,
               "displacementWeight": .0001, "laplacianWeight": .0001,
               "edgeWeight": .001, "maxViewDegradation": .2}
    independent = refineLocalGeometry(vertices, faces, cameras, constraints, **options)
    smooth = refineLocalGeometry(vertices, faces, cameras, constraints, **options,
                                 deformationMode="smooth", smoothControlCount=16, smoothRadius=.5)
    np.testing.assert_allclose(smooth["vertices"][center], target, atol=.002)
    assert smooth["diagnostics"]["finalMeanErrorPixels"] < independent["diagnostics"]["finalMeanErrorPixels"] * .05
    assert independent["diagnostics"]["rejections"]["meshSafety"] > 0
    assert smooth["diagnostics"]["rejections"]["meshSafety"] == 0
    assert smooth["diagnostics"]["safety"]["maximumRelativeEdgeChange"] < .01
    assert smooth["diagnostics"]["referenceProjectionDrift"]["p95Pixels"] < .03


def testReferenceDriftReportsMotionAlongSemanticContourSeparatelyFromBoundaryLoss():
    vertices, faces = makeGrid()
    cameras = makeCameras()
    _, xx = np.mgrid[:160, :160]
    constraints = [{"viewIndex": 0, "vertexIndices": np.array([[4]]),
                    "barycentric": np.ones((1, 1)), "targetDistance": (xx - 80).astype(float)},
                   pointConstraint(cameras[1], [0, .1, 0], 1)]
    result = refineLocalGeometry(vertices, faces, cameras, constraints, maxDisplacement=.15,
                                 iterations=160, activeVertexMask=np.arange(9) == 4,
                                 displacementWeight=.0001, laplacianWeight=.0001, edgeWeight=.001)
    report = result["diagnostics"]
    assert report["finalViewErrors"][0] < .01
    assert report["referenceProjectionDrift"]["meanPixels"] > 1.9
    assert report["referenceProjectionDrift"]["sampleCount"] == 1


@pytest.mark.parametrize("mutation", ["badBarycentric", "floatingIndices", "bothTargets", "badWeights"])
def testInvalidConstraintsFailClearly(mutation):
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    constraint = pointConstraint(cameras[0], [.1, .1, 0])
    if mutation == "badBarycentric":
        constraint["barycentric"][:] = .5
    elif mutation == "floatingIndices":
        constraint["vertexIndices"] = np.array([[4.]])
    elif mutation == "bothTargets":
        constraint["targetDistance"] = np.zeros((20, 20))
    elif mutation == "badWeights":
        constraint["weights"] = [-1.]
    with pytest.raises(ValueError):
        refineLocalGeometry(vertices, faces, cameras, [constraint])


def testBehindCameraAndDegenerateInitialMeshFailClearly():
    vertices, faces = makeGrid()
    cameras = makeCameras()[:1]
    constraint = pointConstraint(cameras[0], [.1, .1, 0])
    cameras[0]["translation"][2] = -5
    with pytest.raises(ValueError, match="behind cameras"):
        refineLocalGeometry(vertices, faces, cameras, [constraint])
    vertices[1] = vertices[0]
    with pytest.raises(ValueError, match="degenerate"):
        refineLocalGeometry(vertices, faces, makeCameras(), [])


def testFloat32MeshKeepsDtypeAndUnmovedValues():
    vertices, faces = makeGrid()
    vertices = vertices.astype(np.float32)
    result = refineLocalGeometry(vertices, faces, makeCameras(), [], iterations=0)
    assert result["vertices"].dtype == np.float32
    np.testing.assert_array_equal(result["vertices"], vertices)
