"""Independent constrained optima and invariance checks for global elimination."""
import numpy as np
import pytest
from scipy import sparse

from faceAppearance.guardedMeshQuadratic import solveGuardedMeshQuadratic


@pytest.fixture
def problem():
    rng = np.random.default_rng(81)
    design = rng.normal(size=(18, 12))
    matrix = design.T @ design + np.eye(12) * 2
    wanted = np.array([[1.2, -.1, .2], [.3, .2, -.4], [-.2, .3, .1], [.2, -.1, .5]])
    baseline = np.zeros((4, 3))
    baseline[3] = wanted[3]

    def constraints(vertices):
        return {'values': np.array([.3 - vertices[0, 0]]),
                'jacobian': sparse.csr_matrix(([-1.], ([0], [0])), shape=(1, 12))}

    return {'baselineVertices': baseline, 'operator': sparse.csr_matrix(matrix),
            'rhs': matrix @ wanted.ravel(), 'controlVertexIds': np.array([1, 0]),
            'fixedVertexIds': np.array([3]), 'constraintFunction': constraints, 'lengthScale': .7}


def denseEqualityOptimum(problem):
    # The desired unconstrained point violates x0<=.3. The only inequality is
    # active at the optimum; solve the full original dense KKT independently.
    h = problem['operator'].toarray()
    rows = np.eye(12)[[0, 9, 10, 11]]
    targets = np.r_[.3, problem['baselineVertices'][3]]
    kkt = np.block([[h, rows.T], [rows, np.zeros((4, 4))]])
    result = np.linalg.solve(kkt, np.r_[problem['rhs'], targets])
    assert result[12] > 0  # KKT multiplier of upper bound in x0-.3<=0 form.
    return result[:12].reshape(4, 3)


def testSchurSolutionMatchesFullDenseConstrainedKkt(problem):
    expected = denseEqualityOptimum(problem)
    original = {key: value.copy() for key, value in problem.items() if hasattr(value, 'copy')}
    solved = solveGuardedMeshQuadratic(**problem)
    np.testing.assert_allclose(solved['rawVertices'], expected, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(solved['rawVertices'][3], problem['baselineVertices'][3])
    report = solved['diagnostics']
    assert report['solverAccepted'] and report['constraintsFeasible']
    assert report['exteriorNormalEquationRelativeResidual'] < 1e-13
    assert report['eliminatedFreeVertexCount'] == 1
    assert report['initialExteriorAdjustmentMaximum'] > .01
    assert report['objectiveChangeFromConditionallyOptimalBaseline'] < 0
    assert not report['qualityAccepted']
    for key, before in original.items():
        actual = problem[key]
        np.testing.assert_array_equal(actual.toarray() if sparse.issparse(actual) else actual,
                                      before.toarray() if sparse.issparse(before) else before)


def testNoExteriorAndControlsMayIncludeFixedVertices(problem):
    problem['controlVertexIds'] = np.array([3, 2, 0, 1])
    result = solveGuardedMeshQuadratic(**problem)
    assert result['diagnostics']['solverAccepted']
    assert result['diagnostics']['eliminatedFreeVertexCount'] == 0
    np.testing.assert_allclose(result['rawVertices'], denseEqualityOptimum(problem), atol=2e-6)


def testControlOrderingAndFullVertexRenumberingDoNotChangeSolution(problem):
    baseline = solveGuardedMeshQuadratic(**problem)['rawVertices']
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    coordinates = (3 * permutation[:, None] + np.arange(3)).ravel()
    originalConstraint = problem['constraintFunction']

    def constraints(vertices):
        old = originalConstraint(vertices[inverse])
        return {'values': old['values'], 'jacobian': old['jacobian'][:, coordinates]}

    changed = {**problem, 'baselineVertices': problem['baselineVertices'][permutation],
        'operator': problem['operator'][coordinates][:, coordinates], 'rhs': problem['rhs'][coordinates],
        'controlVertexIds': inverse[problem['controlVertexIds']][::-1],
        'fixedVertexIds': inverse[problem['fixedVertexIds']], 'constraintFunction': constraints}
    second = solveGuardedMeshQuadratic(**changed)
    assert second['diagnostics']['solverAccepted']
    np.testing.assert_allclose(second['rawVertices'][inverse], baseline, atol=3e-6)


def testPhysicalUnitAndTranslationChangesPreserveTheConstrainedProblem(problem):
    first = solveGuardedMeshQuadratic(**problem)['rawVertices']
    factor, offset = 112.3, np.array([3., -9., 14.])
    newMatrix = problem['operator'] / factor**2
    translation = np.tile(offset, 4)
    originalConstraint = problem['constraintFunction']

    def constraints(vertices):
        original = originalConstraint((vertices-offset)/factor)
        return {'values': original['values'], 'jacobian': original['jacobian']/factor}

    transformed = {**problem, 'baselineVertices': problem['baselineVertices']*factor+offset,
        'operator': newMatrix, 'rhs': problem['rhs']/factor + newMatrix @ translation,
        'lengthScale': problem['lengthScale']*factor, 'constraintFunction': constraints}
    second = solveGuardedMeshQuadratic(**transformed)
    assert second['diagnostics']['solverAccepted']
    np.testing.assert_allclose((second['rawVertices']-offset)/factor, first, atol=3e-6)


def testNonlinearSphereConstraintAgreesWithAnalyticalOptimum():
    target = np.array([2., -1., .5])
    radius = .8

    def sphere(vertices):
        return {'values': np.array([1-np.sum(vertices**2)/radius**2]),
                'jacobian': sparse.csr_matrix((-2*vertices/radius**2).reshape(1, 3))}

    result = solveGuardedMeshQuadratic(np.zeros((1, 3)), sparse.eye(3, format='csr'), target,
        np.array([0]), np.array([], int), sphere, lengthScale=.4)
    assert result['diagnostics']['solverAccepted']
    np.testing.assert_allclose(result['rawVertices'][0], radius*target/np.linalg.norm(target), atol=1e-7)


def testIterationLimitIsRejectedWithoutFallbackOrAdoption(problem):
    result = solveGuardedMeshQuadratic(**problem, maxIterations=1)
    assert not result['diagnostics']['solverAccepted']
    assert not result['diagnostics']['optimizerSuccess']
    assert not result['diagnostics']['qualityAccepted']
    np.testing.assert_array_equal(result['rawVertices'][3], problem['baselineVertices'][3])


def testConstraintsCannotDependOnEliminatedVertices(problem):
    problem['constraintFunction'] = lambda vertices: {
        'values': np.array([3.-vertices[2, 0]]),
        'jacobian': sparse.csr_matrix(([-1.], ([0], [6])), shape=(1, 12))}
    with pytest.raises(ValueError, match='eliminated'):
        solveGuardedMeshQuadratic(**problem)


def testNonpositiveExteriorAndRetainedObjectivesAreRejected(problem):
    for badCoordinate in (0, 6):
        matrix = np.eye(12)
        matrix[badCoordinate, badCoordinate] = -1
        with pytest.raises(ValueError, match='positive definite'):
            solveGuardedMeshQuadratic(**{**problem, 'operator': sparse.csr_matrix(matrix)})


def testInfeasibleBaselineIsRejected(problem):
    problem['baselineVertices'][0, 0] = .4
    with pytest.raises(ValueError, match='Baseline is infeasible'):
        solveGuardedMeshQuadratic(**problem)


@pytest.mark.parametrize('key,value', [
    ('lengthScale', 0), ('lengthScale', True), ('lengthScale', '1'),
    ('maxIterations', 0), ('maxIterations', True),
    ('feasibilityTolerance', float('nan')), ('solverTolerance', -1),
    ('controlVertexIds', np.array([0, 0])), ('controlVertexIds', np.array([3])),
    ('fixedVertexIds', np.array([4])), ('rhs', np.ones(11)),
    ('operator', np.eye(12)), ('operator', sparse.csr_matrix(np.full((12, 12), np.inf))),
])
def testInvalidInputsFailClearly(problem, key, value):
    with pytest.raises(ValueError):
        solveGuardedMeshQuadratic(**{**problem, key: value})


def testAsymmetricObjectiveIsRejected(problem):
    matrix = problem['operator'].toarray()
    matrix[1, 0] += .1
    with pytest.raises(ValueError, match='symmetric'):
        solveGuardedMeshQuadratic(**{**problem, 'operator': sparse.csr_matrix(matrix)})
