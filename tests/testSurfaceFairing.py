"""Independent behavior and safety contracts for direction-constrained fairing."""
import numpy as np
import pytest

from faceAppearance.camera import projectPoints
from faceAppearance.surfaceFairing import refineSurfaceAlongDirections


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
    weights = np.maximum(1 - np.maximum(abs(vertices[:, 0]), abs(vertices[:, 1])) / .8, 0).astype(float) ** 2
    directions = np.tile([0., 0., 1.], (len(vertices), 1))
    return vertices, faces, directions, weights


def independentSafety(before, after, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengthsBefore = np.linalg.norm(before[edges[:, 1]] - before[edges[:, 0]], axis=1)
    lengthsAfter = np.linalg.norm(after[edges[:, 1]] - after[edges[:, 0]], axis=1)
    crossBefore = np.cross(before[faces[:, 1]] - before[faces[:, 0]], before[faces[:, 2]] - before[faces[:, 0]])
    crossAfter = np.cross(after[faces[:, 1]] - after[faces[:, 0]], after[faces[:, 2]] - after[faces[:, 0]])
    areaBefore, areaAfter = np.linalg.norm(crossBefore, axis=1), np.linalg.norm(crossAfter, axis=1)
    return {"maximumRelativeEdgeChange": float(np.abs(lengthsAfter / lengthsBefore - 1).max()),
            "minimumAreaRatio": float((areaAfter / areaBefore).min()),
            "minimumNormalCosine": float((np.sum(crossBefore * crossAfter, axis=1) / (areaBefore * areaAfter)).min()),
            "maximumDisplacementFraction": float(np.linalg.norm(after - before, axis=1).max() / np.ptp(before, axis=0).max())}


def testLocalizedBumpShrinksWithExactlyFixedBoundaryAndUnmodifiedInputs():
    vertices, faces, directions, weights = makeBump()
    originals = [item.copy() for item in (vertices, faces, directions, weights)]
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights, maxDisplacementFraction=.15)
    assert result["diagnostics"]["status"] == "accepted"
    assert result["vertices"][:, 2].max() < vertices[:, 2].max() * .7
    assert result["diagnostics"]["finalEnergy"]["fairing"] < result["diagnostics"]["initialEnergy"]["fairing"] * .4
    np.testing.assert_array_equal(result["vertices"][weights == 0], vertices[weights == 0])
    np.testing.assert_array_equal(result["vertices"][:, :2], vertices[:, :2])
    for previous, actual in zip(originals, (vertices, faces, directions, weights)):
        np.testing.assert_array_equal(previous, actual)
    np.testing.assert_array_equal(result["displacement"], result["vertices"] - vertices)
    assert result["diagnostics"]["selfIntersectionChecked"] is False


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-10), (np.float32, 1e-4)])
def testPerspectiveRayDirectionsPreserveProjectionWithMeaningfulShapeChange(dtype, tolerance):
    vertices, faces, _, weights = makeBump(dtype)
    angle = .21
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    camera = {"intrinsics": np.array([[740., 0., 320.], [0., 735., 240.], [0., 0., 1.]]),
              "rotation": rotation, "translation": np.array([.13, -.12, 4.])}
    center = -rotation.T @ camera["translation"]
    directions = vertices.astype(float) - center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights, maxDisplacementFraction=.04)
    before, _ = projectPoints(vertices.astype(float), **camera)
    after, depth = projectPoints(result["vertices"].astype(float), **camera)
    assert result["diagnostics"]["status"] == "accepted"
    assert np.linalg.norm(result["displacement"], axis=1).max() > .025
    assert depth.min() > 0
    assert np.linalg.norm(after - before, axis=1).max() < tolerance
    assert result["vertices"].dtype == dtype
    actual = independentSafety(vertices.astype(float), result["vertices"].astype(float), faces)
    for name, value in actual.items():
        assert result["diagnostics"]["safety"][name] == pytest.approx(value, abs=1e-13)


def testAllZeroRegionReturnsFreshIdenticalMeshAndClearStatus():
    vertices, faces, directions, weights = makeBump()
    weights[:] = 0
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights, minNormalCosine=1.)
    np.testing.assert_array_equal(result["vertices"], vertices)
    np.testing.assert_array_equal(result["displacement"], 0.)
    assert result["diagnostics"]["reason"] == "noActiveRegion"
    assert result["diagnostics"]["candidateChecks"] == 0
    result["vertices"][:] = 7
    assert vertices[:, 2].max() == pytest.approx(.15)


def testWholeFieldBacktrackingSatisfiesTightEdgeGateWithoutPerVertexClipping():
    vertices, faces, directions, weights = makeBump()
    broad = refineSurfaceAlongDirections(vertices, faces, directions, weights, smoothWeight=100.,
                                         maxDisplacementFraction=.2)
    limited = refineSurfaceAlongDirections(vertices, faces, directions, weights, smoothWeight=100.,
                                           maxDisplacementFraction=.2, maxRelativeEdgeChange=.01)
    assert broad["diagnostics"]["acceptedScale"] == 1.
    assert limited["diagnostics"]["status"] == "accepted"
    assert 0 < limited["diagnostics"]["acceptedScale"] < 1
    np.testing.assert_allclose(limited["displacement"],
                               broad["displacement"] * limited["diagnostics"]["acceptedScale"], atol=1e-15)
    actual = independentSafety(vertices, limited["vertices"], faces)
    assert actual["maximumRelativeEdgeChange"] <= .01
    assert limited["diagnostics"]["rejectionCounts"]["relativeEdgeChange"] > 0


def testExhaustedSafetySearchReturnsUnchangedOriginal():
    vertices, faces, directions, weights = makeBump()
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights, smoothWeight=100.,
                                         maxDisplacementFraction=.2, maxRelativeEdgeChange=.001, maxBacktracking=0)
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert result["diagnostics"]["status"] == "unchanged"
    assert result["diagnostics"]["reason"] == "noSafeImprovingStep"
    assert result["diagnostics"]["acceptedScale"] == 0


def testFloat32SavedMeshReallySatisfiesDisplacementCap():
    vertices, faces, directions, weights = makeBump(np.float32)
    cap = .00317
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights, smoothWeight=100., maxDisplacementFraction=cap)
    actual = independentSafety(vertices.astype(float), result["vertices"].astype(float), faces)
    assert result["diagnostics"]["status"] == "accepted"
    assert actual["maximumDisplacementFraction"] <= cap
    assert actual["maximumDisplacementFraction"] > cap * .95
    assert result["diagnostics"]["safety"]["maximumDisplacementFraction"] == actual["maximumDisplacementFraction"]


def testCoordinateUnitAndTranslationChangesPreserveNormalizedSolution():
    vertices, faces, directions, weights = makeBump()
    base = refineSurfaceAlongDirections(vertices, faces, directions, weights, maxDisplacementFraction=.1)
    changed = refineSurfaceAlongDirections(vertices * 100 + [12., -50., 30.], faces, directions, weights,
                                          maxDisplacementFraction=.1)
    np.testing.assert_allclose(changed["displacement"], base["displacement"] * 100, atol=1e-12)
    assert changed["diagnostics"]["finalEnergy"]["total"] == pytest.approx(base["diagnostics"]["finalEnergy"]["total"], rel=1e-11)


def testTinyRegionWeightsDoNotBreakLinearSystemOrMovePinnedNeighbors():
    vertices, faces, directions, weights = makeBump()
    weights[weights > 0] *= 1e-200
    result = refineSurfaceAlongDirections(vertices, faces, directions, weights)
    assert np.isfinite(result["vertices"]).all()
    np.testing.assert_array_equal(result["vertices"], vertices)
    assert result["diagnostics"]["reason"] == "noEffectiveDisplacement"


@pytest.mark.parametrize("option,value", [("smoothWeight", -1), ("smoothWeight", np.inf),
    ("smoothWeight", True), ("smoothWeight", "10"), ("maxDisplacementFraction", -.01),
    ("maxRelativeEdgeChange", 1.), ("maxRelativeEdgeChange", 0.), ("minAreaRatio", 0.),
    ("minAreaRatio", 1.1), ("minNormalCosine", 0.), ("minNormalCosine", np.nan),
    ("maxBacktracking", -1), ("maxBacktracking", .5), ("maxBacktracking", True)])
def testInvalidParametersFailClearly(option, value):
    with pytest.raises(ValueError, match=option):
        refineSurfaceAlongDirections(*makeBump(), **{option: value})


@pytest.mark.parametrize("which,kind", [(0, "integer"), (0, "nan"), (0, "shape"),
    (1, "float"), (1, "range"), (1, "repeat"), (1, "degenerate"),
    (2, "shape"), (2, "nan"), (2, "notUnit"), (3, "shape"), (3, "nan"), (3, "negative"), (3, "large")])
def testInvalidInputArraysFailClearly(which, kind):
    arguments = list(makeBump())
    if kind == "integer":
        arguments[which] = arguments[which].astype(np.int64)
    elif kind == "float":
        arguments[which] = arguments[which].astype(float)
    elif kind == "shape":
        arguments[which] = arguments[which][:-1]
        if which == 0:
            arguments[which] = arguments[which][:, :2]
    elif kind == "range":
        arguments[1][0, 0] = len(arguments[0])
    elif kind == "repeat":
        arguments[1][0, 0] = arguments[1][0, 1]
    elif kind == "degenerate":
        arguments[0][arguments[1][0, 0]] = arguments[0][arguments[1][0, 1]]
    elif kind == "notUnit":
        arguments[2][0] *= 2
    elif kind == "negative":
        arguments[3][0] = -.1
    elif kind == "large":
        arguments[3][0] = 1.01
    else:
        arguments[which].flat[0] = np.nan
    with pytest.raises(ValueError):
        refineSurfaceAlongDirections(*arguments)
