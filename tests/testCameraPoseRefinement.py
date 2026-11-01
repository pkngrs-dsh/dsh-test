import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from faceAppearance.camera import projectPoints
from faceAppearance.cameraPoseRefinement import (
    solveBoundedCameraPose, validateCameraPoseCandidate,
)


def makeProblem():
    generator = np.random.default_rng(45)
    points = generator.uniform([-0.7, -0.9, -0.4], [0.7, 0.9, 0.4], (48, 3))
    intrinsics = np.array([[900., 0., 256.], [0., 880., 250.], [0., 0., 1.]])
    rotation = Rotation.from_euler("xyz", [4., -8., 2.], degrees=True).as_matrix()
    translation = np.array([0.1, -0.05, 4.])
    pivot = points.mean(0)
    # Build the target from an independent rigid transform around the centroid.
    trueDelta = Rotation.from_euler("xyz", [0.35, -0.55, 0.25], degrees=True).as_matrix()
    trueRotation = trueDelta @ rotation
    pivotInCamera = rotation @ pivot + translation
    trueTranslation = pivotInCamera - trueRotation @ pivot + [0.012, -0.009, 0.03]
    targets = projectPoints(points, intrinsics, trueRotation, trueTranslation)[0]
    training = np.arange(len(points)) % 2 == 0
    return {
        "objectPoints": points, "imagePoints": targets, "intrinsics": intrinsics,
        "rotation": rotation, "translation": translation, "trainingMask": training,
        "pivot": pivot,
    }, trueRotation, trueTranslation


def testRecoverSmallPoseOnHeldoutPointsWithFrozenInputs():
    problem, trueRotation, trueTranslation = makeProblem()
    snapshots = {name: value.copy() for name, value in problem.items()}
    candidate = solveBoundedCameraPose(**problem)
    heldout = ~problem["trainingMask"]
    group = heldout & (np.arange(len(heldout)) < 20)
    validation = validateCameraPoseCandidate(
        candidate, problem["imagePoints"], heldout, validationGroups={"nose": group},
    )
    assert candidate["solverConverged"]
    assert validation["accepted"]
    assert validation["heldout"]["meanAfter"] < 0.002
    np.testing.assert_allclose(candidate["rotation"], trueRotation, atol=1e-5)
    np.testing.assert_allclose(candidate["translation"], trueTranslation, atol=1e-5)
    np.testing.assert_array_equal(candidate["intrinsics"], problem["intrinsics"])
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(problem[name], snapshot)
    candidate["intrinsics"][0, 0] = 1
    np.testing.assert_array_equal(problem["intrinsics"], snapshots["intrinsics"])


def testHeldoutTargetsNeverInfluencePoseFit():
    problem, _, _ = makeProblem()
    first = solveBoundedCameraPose(**problem)
    problem["imagePoints"][~problem["trainingMask"]] += [31., -18.]
    second = solveBoundedCameraPose(**problem)
    np.testing.assert_array_equal(first["normalizedParameters"], second["normalizedParameters"])
    result = validateCameraPoseCandidate(second, problem["imagePoints"], ~problem["trainingMask"])
    assert not result["accepted"]
    assert "heldout:insufficientImprovement" in result["reasons"]


@pytest.mark.parametrize("field,replacement", [
    ("objectPoints", np.ones((48, 2))),
    ("imagePoints", np.full((48, 2), np.nan)),
    ("intrinsics", np.diag([np.inf, 1., 1.])),
    ("rotation", np.diag([1., 1., -1.])),
    ("translation", np.ones((1, 3))),
    ("pivot", np.array([0., np.nan, 0.])),
    ("trainingMask", np.ones(48, dtype=np.int32)),
    ("translationBounds", [0.1, -0.1, 0.2]),
])
def testRejectMalformedAndNonfiniteInput(field, replacement):
    problem, _, _ = makeProblem()
    problem[field] = replacement
    with pytest.raises(ValueError):
        solveBoundedCameraPose(**problem)


def testInsufficientTrainingReturnsUnchangedPoseAndRejects():
    problem, _, _ = makeProblem()
    problem["trainingMask"][:] = False
    problem["trainingMask"][:5] = True
    candidate = solveBoundedCameraPose(**problem)
    np.testing.assert_array_equal(candidate["rotation"], problem["rotation"])
    np.testing.assert_allclose(candidate["translation"], problem["translation"], atol=1e-15)
    result = validateCameraPoseCandidate(candidate, problem["imagePoints"], ~problem["trainingMask"])
    assert not result["accepted"]
    assert "insufficientTrainingConstraints" in result["reasons"]


def testRejectTrainingLeakageAndGroupOutsideHeldout():
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    with pytest.raises(ValueError, match="disjoint"):
        validateCameraPoseCandidate(candidate, problem["imagePoints"], np.ones(48, dtype=bool))
    with pytest.raises(ValueError, match="held-out subset"):
        validateCameraPoseCandidate(
            candidate, problem["imagePoints"], ~problem["trainingMask"],
            validationGroups={"bad": problem["trainingMask"]},
        )


@pytest.mark.parametrize("parameter,touches", [(0.995, False), (0.995001, True), (-1.0, True)])
def testBoundGateIsStrictAndIndependentOfImprovedHeldoutError(parameter, touches):
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    candidate["normalizedParameters"][0] = parameter
    result = validateCameraPoseCandidate(candidate, problem["imagePoints"], ~problem["trainingMask"])
    assert result["anyBoundTouched"] is touches
    assert result["accepted"] is not touches


def testSubgroupCanRejectOtherwiseImprovedHeldoutSet():
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    heldout = ~problem["trainingMask"]
    group = heldout & (np.arange(len(heldout)) < 12)
    # A worse semantic region must not hide behind the whole-face average.
    candidate["afterProjection"][group] = candidate["beforeProjection"][group]
    result = validateCameraPoseCandidate(
        candidate, problem["imagePoints"], heldout, validationGroups={"nose": group},
    )
    assert result["heldout"]["meanAfter"] < result["heldout"]["meanBefore"] * 0.8
    assert not result["accepted"]
    assert result["reasons"] == ["nose:insufficientImprovement"]


def testUnconvergedSolveIsRejectedDespiteImprovedHeldoutProjection():
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    candidate["solverConverged"] = False
    result = validateCameraPoseCandidate(candidate, problem["imagePoints"], ~problem["trainingMask"])
    assert result["heldout"]["meanAfter"] < 0.002
    assert not result["accepted"]
    assert result["reasons"] == ["solverNotConverged"]


@pytest.mark.parametrize("depthField,reason", [
    ("beforeDepth", "nonpositiveSourceDepth"),
    ("afterDepth", "nonpositiveCandidateDepth"),
])
@pytest.mark.parametrize("pointIndex", [0, 1])  # Training and held-out support.
@pytest.mark.parametrize("depth", [0., -0.1])
def testNonpositiveDepthIsRejectedEvenWhenProjectionLooksGood(depthField, reason, pointIndex, depth):
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    candidate[depthField][pointIndex] = depth
    result = validateCameraPoseCandidate(candidate, problem["imagePoints"], ~problem["trainingMask"])
    assert not result["accepted"]
    assert result["reasons"] == [reason]


def testUnusedPointBehindCameraDoesNotInvalidateIndependentSupport():
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    heldout = ~problem["trainingMask"]
    heldout[-1] = False
    candidate["beforeDepth"][-1] = candidate["afterDepth"][-1] = -1.
    result = validateCameraPoseCandidate(candidate, problem["imagePoints"], heldout)
    assert result["accepted"]


@pytest.mark.parametrize("depthField", ["beforeDepth", "afterDepth"])
@pytest.mark.parametrize("invalidDepth", [np.nan, np.inf])
def testNonfiniteDepthCannotEnterValidation(depthField, invalidDepth):
    problem, _, _ = makeProblem()
    candidate = solveBoundedCameraPose(**problem)
    candidate[depthField][0] = invalidDepth
    with pytest.raises(ValueError, match=depthField):
        validateCameraPoseCandidate(candidate, problem["imagePoints"], ~problem["trainingMask"])
