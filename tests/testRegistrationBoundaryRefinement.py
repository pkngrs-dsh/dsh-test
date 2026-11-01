"""Independent numerical and semantic checks, with no renderer or model."""
import numpy as np
import pytest
from scipy import sparse

from faceAppearance.registrationBoundaryRefinement import buildEyeBoundaryTargets, solveBoundaryIncrement


def testBoundaryIncrementMatchesFullConstrainedQuadraticAndKeepsAnchor():
    rng = np.random.default_rng(12)
    design = rng.normal(size=(9, 5))
    operator = design.T @ design + np.eye(5)
    baseline = rng.normal(size=(5, 3))
    fixed = np.array([1], np.int64)
    ids = np.array([0, 3], np.int64)
    targets = baseline[ids] + [[.1, -.2, .15], [-.2, .05, .4]]
    before = baseline.copy()
    weight = 47.
    actual = solveBoundaryIncrement(baseline, sparse.csr_matrix(operator), fixed, ids, targets, weight)
    selector = np.eye(5)[ids]
    coefficient = weight / (3 * len(ids))
    fullOperator = operator + coefficient * selector.T @ selector
    rhs = operator @ baseline + coefficient * selector.T @ targets
    free = np.array([0, 2, 3, 4])
    expected = baseline.copy()
    expected[free] = np.linalg.solve(fullOperator[np.ix_(free, free)],
        rhs[free] - fullOperator[np.ix_(free, fixed)] @ baseline[fixed])
    np.testing.assert_allclose(actual, expected, atol=5e-15)
    np.testing.assert_array_equal(actual[fixed], baseline[fixed])
    np.testing.assert_array_equal(baseline, before)
    assert np.linalg.norm(actual[ids] - targets) < np.linalg.norm(baseline[ids] - targets)


def testZeroWeightAndMatchingTargetsDoNotMoveGeometry():
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    operator = sparse.eye(5)
    fixed, ids = np.array([1], np.int64), np.array([0, 3], np.int64)
    np.testing.assert_array_equal(solveBoundaryIncrement(vertices, operator, fixed, ids, vertices[ids] + 1, 0), vertices)
    np.testing.assert_array_equal(solveBoundaryIncrement(vertices, operator, fixed, ids, vertices[ids], 100), vertices)


def testBoundaryIncrementIsRigidTransformAndScaleEquivariant():
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    operator = sparse.eye(5)
    fixed, ids = np.array([1], np.int64), np.array([0, 3], np.int64)
    targets = vertices[ids] + [[.2, .1, -.3], [.4, -.1, .2]]
    result = solveBoundaryIncrement(vertices, operator, fixed, ids, targets, 100)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    apply = lambda points: points @ rotation * 2.7 + [3., -7., 2.]
    transformed = solveBoundaryIncrement(apply(vertices), operator, fixed, ids, apply(targets), 100)
    np.testing.assert_allclose(transformed, apply(result), atol=1e-13)


def testFrozenEyeArcsRetainCornerIdsAndSampleActualSourceDepth():
    # Explicit closed loops: left corner, lower middle, right corner, upper middle.
    one = np.array([[-1., 0., 0.], [0., -.5, .1], [1., 0., 0.], [0., .5, .2]])
    client = np.vstack([one + [-3., 0., 0.], one + [3., 0., 0.]])
    loops = [np.arange(4), np.arange(4, 8)]
    source = client * [1., 1.4, 2.]
    actual = source.copy(); actual[:, 2] += .37
    ids, targets, definitions = buildEyeBoundaryTargets(client, loops[::-1], source, loops, actual)
    np.testing.assert_array_equal(ids, np.arange(8))
    # Symmetric three-point arcs select exact source middle and corner vertices.
    np.testing.assert_allclose(targets, actual, atol=1e-14)
    assert definitions[0][0]['clientVertexIds'] == [0, 1, 2]
    assert definitions[0][1]['clientVertexIds'] == [0, 3, 2]
    # Anatomical IDs are fixed by references, not moved source extrema.
    actual[1, 0] = -9.
    _, _, movedDefinitions = buildEyeBoundaryTargets(client, loops, source, loops, actual)
    assert movedDefinitions == definitions


@pytest.mark.parametrize('damage', ['duplicate', 'overlap', 'range', 'nan', 'nonsymmetric', 'negativeWeight'])
def testMalformedBoundaryIncrementFails(damage):
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    operator = np.eye(5)
    fixed, ids, targets, weight = np.array([1]), np.array([0, 3]), vertices[[0, 3]].copy(), 100.
    if damage == 'duplicate': ids = np.array([0, 0])
    if damage == 'overlap': ids = np.array([0, 1])
    if damage == 'range': ids = np.array([0, 5])
    if damage == 'nan': targets[0, 0] = np.nan
    if damage == 'nonsymmetric': operator[0, 1] = 1.
    if damage == 'negativeWeight': weight = -1.
    with pytest.raises(ValueError):
        solveBoundaryIncrement(vertices, operator, fixed, ids, targets, weight)


def testEyeLoopAnnotationRequiredAndInvalidIndicesFail():
    vertices = np.zeros((8, 3))
    with pytest.raises(ValueError, match='Exactly two'):
        buildEyeBoundaryTargets(vertices, [np.arange(4)], vertices, [np.arange(4)], vertices)
    with pytest.raises(ValueError, match='distinct valid'):
        buildEyeBoundaryTargets(vertices, [np.array([0, 1, 2, 2]), np.arange(4, 8)],
                                vertices, [np.arange(4), np.arange(4, 8)], vertices)
