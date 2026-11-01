"""Scientific contracts for the bounded mouth-relative proportion experiment."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('feedbackProportions',
    Path(__file__).resolve().parents[1] / 'tools/experimentFeedbackProportions.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def grid():
    y, x = np.mgrid[-.2:.21:.1, -.2:.21:.1]
    vertices = np.c_[x.ravel(), y.ravel(), (2 + .2 * x).ravel()]
    faces = []
    for row in range(4):
        for col in range(4):
            a = row * 5 + col
            faces.extend([[a, a + 1, a + 5], [a + 1, a + 6, a + 5]])
    camera = {'intrinsics': np.array([[500., 0, 256], [0, 500, 256], [0, 0, 1]]),
              'rotation': np.eye(3), 'translation': np.zeros(3)}
    active = np.zeros((5, 5), bool)
    active[1:4, 1:4] = True
    return vertices, np.asarray(faces), camera, active.ravel()


def solve(target, cap=.02):
    vertices, faces, camera, active = grid()
    ids, bary = np.array([[11, 12, 17]]), np.array([[.2, .5, .3]])
    proposed, diagnostics = MODULE.screenedProportionField(vertices, faces, camera, ids, bary,
        np.asarray([target], float), np.array([1.]), active, cap)
    return vertices, proposed, diagnostics, camera, ids, bary, active


def testExactPerspectiveConstraintPredictionPreservesFixedVerticesAndDepth():
    before, after, report, camera, ids, bary, active = solve([2., -1.])
    np.testing.assert_array_equal(before[~active], after[~active])
    np.testing.assert_array_equal(before[:, 2], after[:, 2])
    old, _ = MODULE.project((before[ids] * bary[..., None]).sum(1), camera)
    new, _ = MODULE.project((after[ids] * bary[..., None]).sum(1), camera)
    np.testing.assert_allclose(new - old, report['predictedPixelDeltas'], atol=1e-12)
    assert np.linalg.norm(new - old) > 0
    assert np.linalg.norm(new - old - [[2., -1.]]) < np.sqrt(5.)


def testLargeTargetUsesDeclaredGlobalBoundWithoutMovingFixedVertices():
    before, after, report, _, _, _, active = solve([100., -80.], cap=.003)
    assert report['boundScale'] < 1
    assert np.max(np.linalg.norm(after - before, axis=1)) <= .003 + 1e-14
    np.testing.assert_array_equal(before[~active], after[~active])


def testZeroTargetRetainsExactGeometry():
    before, after, _, _, _, _, _ = solve([0., 0.])
    np.testing.assert_array_equal(before, after)


def testUnavailableSupportRejectsInsteadOfMovingAnAnchor():
    v, f, c, active = grid()
    with pytest.raises(ValueError, match='nonempty active'):
        MODULE.screenedProportionField(v, f, c, np.array([[0, 1, 2]]), np.array([[1., 0., 0.]]),
            np.zeros((1, 2)), np.ones(1), np.zeros_like(active), .01)


def testMouthRelativeMetricDoesNotTreatGlobalImageTranslationAsChinLength():
    rng = np.random.default_rng(2)
    observed = rng.normal(size=(68, 2)) * 20
    shifted = observed + [7.5, -9.]
    ids, weights = np.arange(5, 12), np.ones(7) / 7
    result = MODULE.relativeMetric(shifted, observed, ids, weights)
    assert result['meanPixels'] < 1e-13
