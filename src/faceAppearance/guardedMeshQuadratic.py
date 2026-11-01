"""Constrained global quadratic fitting through exact exterior elimination.

This opt-in numerical module has no anatomical regions, asset IDs or acceptance
policy. A caller supplies differentiable inequalities on retained vertices.
Other free vertices remain part of the original objective and are recovered
by a sparse solve, rather than frozen or blended after optimization.
"""
from __future__ import annotations

from time import perf_counter
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _vertexIds(values, count, name):
    array = np.asarray(values)
    if (array.ndim != 1 or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array >= count)
            or len(np.unique(array)) != len(array)):
        raise ValueError(f'{name} must contain unique valid integer vertex IDs')
    return array.astype(np.int64, copy=True)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be finite and positive')
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} must be finite and positive') from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return result


def solveGuardedMeshQuadratic(baselineVertices, operator, rhs, controlVertexIds,
                              fixedVertexIds, constraintFunction, *, lengthScale,
                              maxIterations=200, feasibilityTolerance=1e-8,
                              solverTolerance=1e-10):
    """Minimize ``x.T @ operator @ x - 2*rhs.T @ x`` under supplied guards.

    ``x`` is vertex-major XYZ in one common coordinate frame. ``operator`` is
    a symmetric positive-definite sparse 3N square normal matrix; ``rhs`` has
    shape (3N,). All coordinates of fixed vertices retain their exact baseline
    values. Retained controls need not be in any particular order and may
    include fixed vertices. At least one control must be free.

    ``constraintFunction(vertices)`` returns a dictionary with finite 1D
    ``values`` (feasible when >=0) and a CSR/CSC/COO ``jacobian`` of shape
    (M,3N). Constraints must depend only on control/fixed vertices. They must
    be differentiable in the working neighborhood and consistently scaled to
    dimensionless quantities. Their declared sparse support is checked on
    every evaluation. Callback inputs are copies; inputs are never mutated.

    A Schur complement eliminates all other free coordinates. SLSQP optimizes
    only control displacements, in units of ``lengthScale``. The returned
    exterior is its exact conditional quadratic optimum. The starting exterior
    may differ from baseline if baseline is not conditionally optimal for this
    objective; that difference is reported, not silently called zero.

    The supplied baseline must satisfy all inequalities within the declared
    tolerance. Nonconvergence returns a rejected raw solution for diagnostics,
    never a different method or a partial interpolation. ``solverAccepted``
    requires optimizer success, feasible guards, exact fixed coordinates and
    an objective no worse than the feasible starting point. It does not check
    omitted geometry, collisions, data error, likeness or client suitability.
    """
    started = perf_counter()
    baseline = np.asarray(baselineVertices)
    if (baseline.dtype.kind != 'f' or baseline.ndim != 2 or baseline.shape[1] != 3
            or not len(baseline) or not np.isfinite(baseline).all()):
        raise ValueError('baselineVertices must be a finite floating Nx3 array')
    baseline = baseline.astype(np.float64, copy=True)
    count, dimension = len(baseline), baseline.size
    control = _vertexIds(controlVertexIds, count, 'controlVertexIds')
    fixed = _vertexIds(fixedVertexIds, count, 'fixedVertexIds')
    retained = control[~np.isin(control, fixed)]
    if not len(retained):
        raise ValueError('At least one control vertex must be free')
    # Dense Schur storage is intentional and bounded, independent of mesh size.
    if len(retained) > 512:
        raise ValueError('Retained control domain exceeds the 512-vertex dense Schur budget')
    lengthScale = _positive(lengthScale, 'lengthScale')
    feasibilityTolerance = _positive(feasibilityTolerance, 'feasibilityTolerance')
    solverTolerance = _positive(solverTolerance, 'solverTolerance')
    if type(maxIterations) is not int or maxIterations < 1:
        raise ValueError('maxIterations must be a positive integer')
    if not callable(constraintFunction):
        raise ValueError('constraintFunction must be callable')
    if (not sparse.issparse(operator) or operator.format not in ('csr', 'csc', 'coo')
            or operator.shape != (dimension, dimension) or operator.dtype.kind not in 'fiu'
            or operator.nnz > 20_000_000 or not np.isfinite(operator.data).all()):
        raise ValueError('operator must be a finite real sparse symmetric 3N square matrix')
    matrix = sparse.csc_matrix(operator, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError('operator duplicate coefficients overflowed')
    matrixScale = max(float(np.abs(matrix.data).max(initial=0)), np.finfo(float).tiny)
    asymmetry = matrix - matrix.T
    if np.abs(asymmetry.data).max(initial=0) > matrixScale * 1e-12:
        raise ValueError('operator must be symmetric')
    matrix = ((matrix + matrix.T) * .5).tocsc()
    target = np.asarray(rhs)
    if target.dtype.kind != 'f' or target.shape != (dimension,) or not np.isfinite(target).all():
        raise ValueError('rhs must be a finite floating 3N vector')
    target = target.astype(np.float64, copy=True)
    coordinateIds = lambda ids: (3 * ids[:, None] + np.arange(3)).ravel()
    fixedCoordinates, retainedCoordinates = coordinateIds(fixed), coordinateIds(retained)
    exteriorVertices = np.setdiff1d(np.arange(count), np.r_[fixed, retained])
    exteriorCoordinates = coordinateIds(exteriorVertices)
    if len(exteriorCoordinates) * len(retainedCoordinates) > 30_000_000:
        raise ValueError('Exterior-control elimination exceeds the dense workspace budget')
    adjusted = target - matrix[:, fixedCoordinates] @ baseline[fixed].ravel()
    hpp = matrix[retainedCoordinates][:, retainedCoordinates].toarray()
    bp = adjusted[retainedCoordinates]
    exteriorFactor, exteriorTarget, exteriorControl = None, None, None
    if len(exteriorCoordinates):
        hoo = matrix[exteriorCoordinates][:, exteriorCoordinates].tocsc()
        hop = matrix[exteriorCoordinates][:, retainedCoordinates].tocsc()
        try:
            exteriorFactor = splu(hoo, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0,
                                  options={'SymmetricMode': True})
        except RuntimeError as error:
            raise ValueError('Exterior normal matrix is singular') from error
        if (not np.array_equal(exteriorFactor.perm_r, exteriorFactor.perm_c)
                or exteriorFactor.U.diagonal().min() <= matrixScale * 1e-12):
            raise ValueError('Exterior normal matrix is not numerically positive definite')
        exteriorTarget = exteriorFactor.solve(adjusted[exteriorCoordinates])
        exteriorControl = exteriorFactor.solve(hop.toarray())
        reduced = hpp - hop.T @ exteriorControl
        reducedTarget = bp - hop.T @ exteriorTarget
    else:
        reduced, reducedTarget = hpp, bp
    reduced = (reduced + reduced.T) * .5
    if not np.isfinite(reduced).all() or not np.isfinite(reducedTarget).all():
        raise ValueError('Schur complement is nonfinite')
    try:
        cholesky(reduced, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError('Retained quadratic objective is not positive definite') from error
    initialControl = baseline[retained].ravel()
    initialGradient = reduced @ initialControl - reducedTarget
    objectiveScale = max(float(np.abs(np.diag(reduced)).max()) * lengthScale ** 2,
                         float(np.linalg.norm(initialGradient)) * lengthScale,
                         np.finfo(float).tiny)
    allowed = np.zeros(dimension, bool)
    allowed[coordinateIds(np.unique(np.r_[control, fixed]))] = True
    cachedY, cachedValues, cachedJacobian = None, None, None
    evaluationCount, constraintCount = 0, None

    def assemble(y, recoverExterior):
        vertices = baseline.copy()
        coordinates = initialControl + lengthScale * y
        vertices[retained] = coordinates.reshape(-1, 3)
        if recoverExterior and len(exteriorCoordinates):
            vertices[exteriorVertices] = (exteriorTarget - exteriorControl @ coordinates).reshape(-1, 3)
        return vertices

    def evaluate(y):
        nonlocal cachedY, cachedValues, cachedJacobian, evaluationCount, constraintCount
        if cachedY is not None and np.array_equal(y, cachedY):
            return cachedValues, cachedJacobian
        result = constraintFunction(assemble(y, False))
        if not isinstance(result, dict) or not {'values', 'jacobian'} <= set(result):
            raise ValueError('Constraint callback requires values and jacobian')
        values = np.asarray(result['values'])
        jac = result['jacobian']
        if (values.dtype.kind != 'f' or values.ndim != 1 or not len(values)
                or not np.isfinite(values).all() or not sparse.issparse(jac)
                or jac.format not in ('csr', 'csc', 'coo') or jac.shape != (len(values), dimension)
                or jac.dtype.kind not in 'fiu' or not np.isfinite(jac.data).all()):
            raise ValueError('Constraints require finite floating values and a finite real sparse Mx3N Jacobian')
        jac = sparse.csr_matrix(jac, dtype=np.float64, copy=True)
        jac.sum_duplicates()
        jac.eliminate_zeros()
        if not np.isfinite(jac.data).all() or np.any(~allowed[jac.indices]):
            raise ValueError('Constraint Jacobian depends on eliminated vertices or overflows')
        if constraintCount is not None and constraintCount != len(values):
            raise ValueError('Constraint count must remain fixed during optimization')
        constraintCount = len(values)
        cachedY = np.array(y, copy=True)
        cachedValues = np.array(values, dtype=np.float64, copy=True)
        cachedJacobian = jac[:, retainedCoordinates].toarray() * lengthScale
        evaluationCount += 1
        return cachedValues, cachedJacobian

    zero = np.zeros(len(retainedCoordinates))
    initialValues, _ = evaluate(zero)
    if initialValues.min() < -feasibilityTolerance:
        raise ValueError('Baseline is infeasible for the supplied shape constraints')
    initialRecovered = assemble(zero, True)

    def objective(y):
        delta = lengthScale * y
        return float((delta @ reduced @ delta + 2 * initialGradient @ delta) / objectiveScale)

    def gradient(y):
        return 2 * lengthScale * (reduced @ (lengthScale * y) + initialGradient) / objectiveScale

    optimization = minimize(objective, zero, jac=gradient, method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': lambda y: evaluate(y)[0],
                      'jac': lambda y: evaluate(y)[1]}],
        options={'maxiter': maxIterations, 'ftol': solverTolerance, 'disp': False})
    if not np.isfinite(optimization.x).all():
        raise ValueError('Constrained optimizer returned nonfinite coordinates')
    vertices = assemble(optimization.x, True)
    finalValues, _ = evaluate(optimization.x)
    # The callback must also agree on the actually recovered full geometry.
    recoveredValues = np.asarray(constraintFunction(vertices.copy())['values'], dtype=np.float64)
    if (recoveredValues.shape != finalValues.shape or not np.isfinite(recoveredValues).all()
            or not np.allclose(recoveredValues, finalValues, rtol=1e-12, atol=feasibilityTolerance * .01)):
        raise ValueError('Constraints changed after exterior recovery; callback violates retained-domain contract')
    if not np.array_equal(vertices[fixed], baseline[fixed]):
        raise ValueError('Hard-fixed coordinates changed')
    rawGradient = matrix @ vertices.ravel() - target
    exteriorResidual = rawGradient[exteriorCoordinates]
    exteriorResidualScale = max(float(np.linalg.norm(target[exteriorCoordinates])),
        float(np.linalg.norm(matrix[exteriorCoordinates] @ vertices.ravel())), np.finfo(float).tiny)
    relativeResidual = float(np.linalg.norm(exteriorResidual) / exteriorResidualScale)
    if not np.isfinite(relativeResidual) or relativeResidual > 1e-8:
        raise ValueError('Recovered exterior does not solve the original global normal equations')
    objectiveChange = objective(optimization.x) * objectiveScale
    feasible = bool(finalValues.min() >= -feasibilityTolerance)
    objectiveNotIncreased = bool(objectiveChange <= solverTolerance * objectiveScale)
    accepted = bool(optimization.success and feasible and objectiveNotIncreased)
    return {'rawVertices': vertices, 'diagnostics': {
        'method': 'globalQuadraticSchurEliminationWithRetainedShapeInequalities',
        'solver': 'scipySLSQP', 'solverAccepted': accepted, 'optimizerSuccess': bool(optimization.success),
        'optimizerStatus': int(optimization.status), 'optimizerMessage': str(optimization.message),
        'iterations': int(optimization.nit), 'constraintEvaluations': evaluationCount,
        'retainedFreeVertexCount': len(retained), 'eliminatedFreeVertexCount': len(exteriorVertices),
        'fixedVertexCount': len(fixed), 'hardStaticExactlyPreserved': True,
        'constraintCount': constraintCount, 'constraintMinimumBefore': float(initialValues.min()),
        'constraintMinimumAfter': float(finalValues.min()), 'constraintsFeasible': feasible,
        'objectiveChangeFromConditionallyOptimalBaseline': objectiveChange,
        'objectiveNotIncreased': objectiveNotIncreased, 'objectiveScale': objectiveScale,
        'exteriorNormalEquationRelativeResidual': relativeResidual,
        'initialExteriorAdjustmentMaximum': float(np.linalg.norm(initialRecovered - baseline, axis=1).max()),
        'maximumDisplacement': float(np.linalg.norm(vertices - baseline, axis=1).max()),
        'lengthScale': lengthScale, 'maxIterations': maxIterations,
        'feasibilityTolerance': feasibilityTolerance, 'solverTolerance': solverTolerance,
        'seconds': perf_counter() - started, 'qualityAccepted': False,
        'limitations': ['SLSQP finds a local constrained optimum; it is not a global nonlinear optimum certificate.',
            'Only supplied retained-domain inequalities are optimized; omitted exterior, collision and data-quality checks are caller responsibilities.',
            'Symmetric positive-pivot checks and Schur Cholesky are numerical checks, not exact-arithmetic definiteness certificates.',
            'No anatomy, texture, likeness, asset binding or downstream refinement is inferred.']}}
