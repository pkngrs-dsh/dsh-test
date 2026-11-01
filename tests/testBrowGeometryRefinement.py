"""Actual geometry, projection and safe-fallback contracts for brow fitting."""
import numpy as np
import pytest
import torch

from faceAppearance.browGeometryRefinement import refineBrowGeometry, _vertexNormals
from faceAppearance.camera import projectPoints


@pytest.fixture(autouse=True)
def singleThreadedTorch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def makeBump(dtype=np.float64):
    count = 17
    xx, yy = np.meshgrid(np.linspace(-1, 1, count), np.linspace(-1, 1, count))
    zz = .15 * np.exp(-((xx / .3) ** 2 + (yy / .3) ** 2))
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(dtype)
    faces = []
    for row in range(count - 1):
        for column in range(count - 1):
            first = row * count + column
            faces.extend([[first, first + 1, first + count], [first + 1, first + count + 1, first + count]])
    faces = np.asarray(faces, np.int32)
    region = np.maximum(1 - np.maximum(abs(vertices[:, 0]), abs(vertices[:, 1])) / .8, 0).astype(float) ** 2
    directions = np.tile([0., 0., 1.], (len(vertices), 1))
    target = vertices.astype(float).copy()
    target[:, 2] *= .4
    return vertices, faces, directions, region, target


def independentSafety(before, after, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    beforeLength = np.linalg.norm(before[edges[:, 1]] - before[edges[:, 0]], axis=1)
    afterLength = np.linalg.norm(after[edges[:, 1]] - after[edges[:, 0]], axis=1)
    beforeCross = np.cross(before[faces[:, 1]] - before[faces[:, 0]], before[faces[:, 2]] - before[faces[:, 0]])
    afterCross = np.cross(after[faces[:, 1]] - after[faces[:, 0]], after[faces[:, 2]] - after[faces[:, 0]])
    beforeArea, afterArea = np.linalg.norm(beforeCross, axis=1), np.linalg.norm(afterCross, axis=1)
    return {"maximumRelativeEdgeChange": float(np.abs(afterLength / beforeLength - 1).max()),
            "minimumAreaRatio": float((afterArea / beforeArea).min()),
            "minimumNormalCosine": float((np.sum(beforeCross * afterCross, axis=1) / (beforeArea * afterArea)).min()),
            "maximumDisplacementFraction": float(np.linalg.norm(after - before, axis=1).max() / np.ptp(before, axis=0).max())}


def testKnownNormalTargetsChangeRealGeometryTowardBumpAndKeepPins():
    vertices, faces, directions, region, target = makeBump()
    originals = [item.copy() for item in (vertices, faces, directions, region)]
    result = refineBrowGeometry(vertices, faces, directions, region, _vertexNormals(target, faces),
                               np.ones(len(vertices)), maxDisplacementFraction=.07, iterations=180)
    report = result["diagnostics"]
    assert report["status"] == "accepted"
    assert report["finalMetrics"]["normalMeanDegrees"] < report["initialMetrics"]["normalMeanDegrees"] * .25
    assert .035 < result["vertices"][:, 2].max() < .08
    np.testing.assert_array_equal(result["vertices"][region == 0], vertices[region == 0])
    np.testing.assert_array_equal(result["vertices"][:, :2], vertices[:, :2])
    for before, current in zip(originals, (vertices, faces, directions, region)):
        np.testing.assert_array_equal(before, current)
    assert report["controlCount"] == 32 and report["unsupportedActiveVertexCount"] == 0


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-10), (np.float32, 1e-4)])
def testFixedCameraRaysPreserveProjectionAndActualDtypeSafety(dtype, tolerance):
    vertices, faces, _, region, target = makeBump(dtype)
    angle = .21
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    camera = {"intrinsics": np.array([[740., 0., 320.], [0., 735., 240.], [0., 0., 1.]]),
              "rotation": rotation, "translation": np.array([.13, -.12, 4.])}
    center = -rotation.T @ camera["translation"]
    directions = vertices.astype(float) - center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    result = refineBrowGeometry(vertices, faces, directions, region, _vertexNormals(target, faces),
                               np.ones(len(vertices)), maxDisplacementFraction=.04)
    assert result["diagnostics"]["status"] == "accepted"
    before, _ = projectPoints(vertices.astype(float), **camera)
    after, depth = projectPoints(result["vertices"].astype(float), **camera)
    assert np.max(np.linalg.norm(after - before, axis=1)) < tolerance
    assert depth.min() > 0
    assert result["vertices"].dtype == dtype
    assert np.linalg.norm(result["displacement"], axis=1).max() > .02
    safety = independentSafety(vertices.astype(float), result["vertices"].astype(float), faces)
    for key, value in safety.items():
        assert result["diagnostics"]["safety"][key] == pytest.approx(value, abs=1e-12)
    assert safety["maximumDisplacementFraction"] <= .04


def testOptionalDepthUsesSignedRayDistanceAndImprovesTargetError():
    vertices, faces, directions, region, target = makeBump()
    targetDepth = target[:, 2] - vertices[:, 2]
    observed = (region > .05).astype(float)
    result = refineBrowGeometry(vertices, faces, directions, region, _vertexNormals(target, faces),
                               np.ones(len(vertices)), depthTargets=targetDepth, depthWeights=observed,
                               maxDisplacementFraction=.07, depthLossWeight=.03, iterations=180)
    report = result["diagnostics"]
    assert report["status"] == "accepted"
    assert report["finalMetrics"]["depthRmseCanonical"] < report["initialMetrics"]["depthRmseCanonical"] * .25
    assert result["scalarDisplacement"][len(vertices) // 2] < -.065
    np.testing.assert_allclose(result["scalarDisplacement"], result["vertices"][:, 2] - vertices[:, 2])


def testMissingTrustedObservationsSkipWithoutGuessingNormals():
    vertices, faces, directions, region, _ = makeBump()
    normals = np.full_like(vertices, np.nan)
    for options, reason in (({}, "noReliableTargets"), ({"regionWeights": np.zeros(len(vertices))}, "noActiveRegion")):
        values = {"regionWeights": region, **options}
        result = refineBrowGeometry(vertices, faces, directions, **values, normalTargets=normals,
                                   normalWeights=np.zeros(len(vertices)))
        np.testing.assert_array_equal(result["vertices"], vertices)
        assert result["diagnostics"]["reason"] == reason
        assert result["diagnostics"]["candidateChecks"] == 0
    normals[0] = np.nan
    weights = np.zeros(len(vertices)); weights[0] = 1
    with pytest.raises(ValueError, match="nonfinite trusted"):
        refineBrowGeometry(vertices, faces, directions, region, normals, weights)


def testWholeFieldBacktrackingNeverClipsIndividualVertices():
    vertices, faces, directions, region, target = makeBump()
    args = (vertices, faces, directions, region, _vertexNormals(target, faces), np.ones(len(vertices)))
    broad = refineBrowGeometry(*args, maxDisplacementFraction=.07)
    limited = refineBrowGeometry(*args, maxDisplacementFraction=.07, maxRelativeEdgeChange=.01)
    report = limited["diagnostics"]
    assert broad["diagnostics"]["acceptedScale"] == 1
    assert report["status"] == "accepted" and 0 < report["acceptedScale"] < 1
    np.testing.assert_allclose(limited["displacement"], broad["displacement"] * report["acceptedScale"], atol=1e-15)
    assert report["rejectionCounts"]["relativeEdgeChange"] > 0
    assert report["safety"]["maximumRelativeEdgeChange"] <= .01


def testExhaustedSafetyAndZeroOrTinyDisplacementReturnSafeOutputs():
    vertices, faces, directions, region, target = makeBump(np.float32)
    args = (vertices, faces, directions, region, _vertexNormals(target, faces), np.ones(len(vertices)))
    rejected = refineBrowGeometry(*args, maxDisplacementFraction=.07, maxRelativeEdgeChange=.001, maxBacktracking=0)
    assert rejected["diagnostics"]["reason"] == "noSafeImprovingStep"
    np.testing.assert_array_equal(rejected["vertices"], vertices)
    zero = refineBrowGeometry(*args, maxDisplacementFraction=0)
    assert zero["diagnostics"]["reason"] == "zeroDisplacementLimit"
    np.testing.assert_array_equal(zero["vertices"], vertices)
    tiny = refineBrowGeometry(*args, maxDisplacementFraction=1e-12)
    assert tiny["diagnostics"]["safety"]["maximumDisplacementFraction"] <= 1e-12


def testCoordinateScaleChangesPreserveNormalizedSolutionAndDepthUnits():
    vertices, faces, directions, region, target = makeBump()
    normals = _vertexNormals(target, faces)
    depth = target[:, 2] - vertices[:, 2]
    confidence = np.ones(len(vertices))
    first = refineBrowGeometry(vertices, faces, directions, region, normals, confidence,
                              depthTargets=depth, depthWeights=confidence, maxDisplacementFraction=.07)
    shift = np.array([12., -50., 30.])
    second = refineBrowGeometry(vertices * 100 + shift, faces, directions, region, normals, confidence,
                               depthTargets=depth * 100, depthWeights=confidence, maxDisplacementFraction=.07)
    np.testing.assert_allclose((second["vertices"] - shift) / 100, first["vertices"], atol=1e-11)


@pytest.mark.parametrize("problem", ["degenerate", "nonunitDirections", "zeroTrustedNormals", "missingDepth", "negativeConfidence"])
def testInvalidInputsFailClearly(problem):
    vertices, faces, directions, region, target = makeBump()
    normals, confidence = _vertexNormals(target, faces), np.ones(len(vertices))
    options = {}
    if problem == "degenerate": vertices[1] = vertices[0]
    if problem == "nonunitDirections": directions *= 2
    if problem == "zeroTrustedNormals": normals[50] = 0
    if problem == "missingDepth": options["depthWeights"] = np.ones(len(vertices))
    if problem == "negativeConfidence": confidence[0] = -1
    with pytest.raises(ValueError):
        refineBrowGeometry(vertices, faces, directions, region, normals, confidence, **options)
