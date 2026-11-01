"""Screened biharmonic displacement fields with exact fixed mesh vertices.

This module constructs a linear basis, not a mesh repair or collision test.
Only original triangle adjacency enters the bending operator. In particular,
geometrically close eyelids and disconnected accessories gain no new edges.
The original Euclidean Wendland driving fields can still correlate nearby
surfaces; the resulting finite element field is piecewise linear, not a claim
of globally C2 surface interpolation.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .eyeRefinement import buildEyeDeformationBasis


MAX_VERTICES = 20000
MAX_FACES = 40000
MAX_CONTROL_POINTS = 100000
MAX_CONTROLS = 128
MAX_DENSE_ELEMENTS = 4000000
MAX_OPERATOR_NNZ = 4000000
MAX_SPARSE_PRODUCT_TERMS = 20000000


def _csrEvidence(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.sort_indices()
    return dict(data=matrix.data.copy(), indices=matrix.indices.copy(),
                indptr=matrix.indptr.copy(), shape=np.asarray(matrix.shape, np.int64))


def _femOperators(vertices, faces):
    """Assemble exact linear-triangle stiffness and positive lumped area."""
    triangle = vertices[faces]
    edges = np.stack([triangle[:, 1]-triangle[:, 0],
                      triangle[:, 2]-triangle[:, 1],
                      triangle[:, 0]-triangle[:, 2]], axis=1)
    doubleArea = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
    squaredLengths = np.sum(edges*edges, axis=2)
    # A dimensionless shape test catches numerical degeneracy, not merely
    # small triangles. Thin but resolvable triangles are never regularized.
    relativeArea = doubleArea / np.maximum(squaredLengths.sum(1), np.finfo(float).tiny)
    if (not np.isfinite(doubleArea).all() or (doubleArea <= 0).any()
            or not np.isfinite(squaredLengths).all()
            or np.any(relativeArea <= 64*np.finfo(float).eps)):
        raise ValueError('Triangles are degenerate or numerically unresolved at the requested scale')
    cotangent = np.stack([
        np.sum((triangle[:, 1]-triangle[:, 0])*(triangle[:, 2]-triangle[:, 0]), axis=1),
        np.sum((triangle[:, 2]-triangle[:, 1])*(triangle[:, 0]-triangle[:, 1]), axis=1),
        np.sum((triangle[:, 0]-triangle[:, 2])*(triangle[:, 1]-triangle[:, 2]), axis=1),
    ], axis=1) / doubleArea[:, None]
    first = faces[:, [1, 2, 0]].ravel()
    second = faces[:, [2, 0, 1]].ravel()
    weights = .5*cotangent.ravel()
    stiffness = sparse.coo_matrix((np.r_[weights, weights, -weights, -weights],
        (np.r_[first, second, first, second], np.r_[first, second, second, first])),
        shape=(len(vertices), len(vertices))).tocsr()
    stiffness.sum_duplicates()
    stiffness.eliminate_zeros()
    mass = np.bincount(faces.ravel(), weights=np.repeat(doubleArea/6, 3), minlength=len(vertices))
    quality = 2*np.sqrt(3)*relativeArea
    return stiffness, mass, dict(minimumTriangleQuality=float(quality.min()),
        maximumAbsoluteCotangent=float(np.abs(cotangent).max()),
        negativeCotangentCount=int((cotangent < 0).sum()),
        thinTrianglePolicy='retain finite element weights without clipping; no quality certification')


def buildFixedBoundaryDeformationBasis(vertices, faces, controlPoints, *, fixedVertexIds,
                                      supportRadius, controlCount=32,
                                      smoothingRadiusFraction=.1):
    """Project ordinary Wendland drivers into an exact Dirichlet domain.

    In coordinates divided by ``supportRadius``, let K be triangle FEM
    stiffness, M lumped area, Q=K.T M^-1 K and U the non-fixed vertices.
    Solve ``(M_U + ell**4 Q_UU) B_U = M_U W_U``, with ``B_F=0`` and
    ``ell=smoothingRadiusFraction``. Q is formed before restriction: fixed
    rows also penalize a sharp boundary slope. No rest geometry is fitted,
    no topology is changed, and no post-solve displacement clipping is used.

    The SPD claim follows from positive free mass and positive-semidefinite
    Q, including for obtuse triangles. Symmetric diagonal equilibration
    changes no objective; there is no numerical ridge or cotangent clamp.
    All allocations have fixed work bounds, and invalid/over-budget inputs
    are rejected without subsampling. Returned evidence permits independent
    equation, fixed-boundary and normalization checks.
    """
    original = np.asarray(vertices, dtype=float)
    rawFaces = np.asarray(faces)
    points = np.asarray(controlPoints, dtype=float)
    rawFixed = np.asarray(fixedVertexIds)
    if (original.ndim != 2 or original.shape[1] != 3 or len(original) < 3
            or not np.isfinite(original).all() or rawFaces.ndim != 2
            or rawFaces.shape[1] != 3 or not len(rawFaces) or rawFaces.dtype.kind not in 'iu'
            or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
        raise ValueError('Expected finite vertices/control points and original integer triangle indices')
    if (rawFaces.min() < 0 or rawFaces.max() >= len(original)
            or np.any(np.diff(np.sort(rawFaces, axis=1), axis=1) == 0)):
        raise ValueError('Triangle indices are invalid or repeated')
    if rawFixed.ndim != 1:
        raise ValueError('fixedVertexIds must be a one-dimensional index vector')
    if rawFixed.size == 0:
        fixed = np.empty(0, np.int64)
    elif (rawFixed.ndim != 1 or rawFixed.dtype.kind not in 'iu'
          or rawFixed.min() < 0 or rawFixed.max() >= len(original)
          or len(np.unique(rawFixed)) != len(rawFixed)):
        raise ValueError('fixedVertexIds must be distinct in-range integer indices')
    else:
        fixed = np.sort(rawFixed.astype(np.int64))
    numericTypes = (int, float, np.integer, np.floating)
    if (isinstance(controlCount, (bool, np.bool_)) or not isinstance(controlCount, (int, np.integer))
            or controlCount < 1 or isinstance(supportRadius, (bool, np.bool_))
            or not isinstance(supportRadius, numericTypes) or not np.isfinite(supportRadius) or supportRadius <= 0
            or isinstance(smoothingRadiusFraction, (bool, np.bool_))
            or not isinstance(smoothingRadiusFraction, numericTypes) or not np.isfinite(smoothingRadiusFraction)
            or not 0 < smoothingRadiusFraction <= 1):
        raise ValueError('Expected positive finite radius, integer controls, and smoothing fraction in (0, 1]')
    if (len(original) > MAX_VERTICES or len(rawFaces) > MAX_FACES
            or len(points) > MAX_CONTROL_POINTS or controlCount > MAX_CONTROLS
            or len(original)*min(controlCount, len(points)) > MAX_DENSE_ELEMENTS):
        raise ValueError('Fixed-boundary field exceeds the declared work budget')
    faces = rawFaces.astype(np.int64, copy=True)
    radius = float(supportRadius)
    center = original.mean(0)
    normalized = (original-center)/radius
    normalizedPoints = (points-center)/radius
    if not np.isfinite(normalized).all() or not np.isfinite(normalizedPoints).all():
        raise ValueError('Coordinates cannot be resolved at the requested support scale')
    # Reuse precisely the existing point selection and ordinary drivers. The
    # operator alone is normalized; source point tie-breaking stays unchanged.
    drivers = buildEyeDeformationBasis(original, points, supportRadius=radius, controlCount=controlCount)
    drivingBasis = drivers['basis']*drivers['normalization']
    centers = drivers['controlPoints'].copy()
    stiffness, mass, triangleDiagnostics = _femOperators(normalized, faces)
    fixedMask = np.zeros(len(original), bool)
    fixedMask[fixed] = True
    free = np.flatnonzero(~fixedMask)
    if np.any(mass[free] <= 0):
        raise ValueError('A free vertex is isolated from the original triangle mesh')
    rowCounts = np.diff(stiffness.indptr).astype(np.int64)
    if stiffness.nnz > MAX_OPERATOR_NNZ or np.sum(rowCounts*rowCounts) > MAX_SPARSE_PRODUCT_TERMS:
        raise ValueError('Fixed-boundary sparse product exceeds the declared work budget')
    inverseMass = np.zeros_like(mass)
    np.divide(1., mass, out=inverseMass, where=mass > 0)
    bending = (stiffness.T @ sparse.diags(inverseMass) @ stiffness).tocsr()
    bending.eliminate_zeros()
    if bending.nnz > MAX_OPERATOR_NNZ or not np.isfinite(bending.data).all():
        raise ValueError('Fixed-boundary operator is nonfinite or exceeds the declared work budget')
    smoothing = float(smoothingRadiusFraction)
    matrix = (sparse.diags(mass[free]) + smoothing**4*bending[free][:, free]).tocsr()
    rhs = mass[free, None]*drivingBasis[free]
    rawBasis = np.zeros_like(drivingBasis)
    residual = 0.
    relativeResidual = 0.
    diagonalRatio = 1.
    if len(free) and len(centers):
        diagonal = matrix.diagonal()
        if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
            raise ValueError('Fixed-boundary SPD system has an invalid diagonal')
        diagonalRatio = float(diagonal.max()/diagonal.min())
        equilibration = 1./np.sqrt(diagonal)
        scaled = (sparse.diags(equilibration) @ matrix @ sparse.diags(equilibration)).tocsc()
        try:
            solved = splu(scaled).solve(equilibration[:, None]*rhs)
        except (RuntimeError, ValueError) as error:
            raise ValueError('Fixed-boundary factorization failed') from error
        rawBasis[free] = equilibration[:, None]*solved
        if not np.isfinite(rawBasis).all():
            raise ValueError('Fixed-boundary solve returned a nonfinite field')
        difference = matrix @ rawBasis[free]-rhs
        residual = float(np.max(np.abs(difference), initial=0.))
        matrixNorm = float(np.asarray(np.abs(matrix).sum(1)).max(initial=0.))
        denominator = matrixNorm*float(np.max(np.abs(rawBasis[free]), initial=0.)) + float(np.max(np.abs(rhs), initial=0.))
        relativeResidual = residual/max(denominator, np.finfo(float).tiny)
        if not np.isfinite(relativeResidual) or relativeResidual > 1e-10:
            raise ValueError('Fixed-boundary solve did not satisfy its equation')
    normalization = max(1., float(np.abs(rawBasis).sum(1).max(initial=0.)))
    basis = rawBasis/normalization
    active = np.any(basis != 0, axis=1)
    columnNorms = np.linalg.norm(basis, axis=0)
    singularValues = np.linalg.svd(basis[free], compute_uv=False) if len(free) and len(centers) else np.empty(0)
    rankTolerance = (max(basis.shape)*np.finfo(float).eps*singularValues[0]) if len(singularValues) else 0.
    rank = int((singularValues > rankTolerance).sum())
    rowNorms = np.linalg.norm(basis, axis=1)
    meaningful = rowNorms > rowNorms.max(initial=0.)*1e-8
    symmetry = matrix-matrix.T
    symmetryError = float(np.max(np.abs(symmetry.data), initial=0.))
    evidence = dict(referenceVertices=original.copy(), referenceFaces=faces.copy(), coordinateCenter=center.copy(),
        normalizedVertices=normalized.copy(), fixedVertexIds=fixed.copy(), freeVertexIds=free.copy(),
        inputControlPoints=points.copy(), controlPoints=centers.copy(), drivingBasis=drivingBasis.copy(), lumpedMass=mass.copy(),
        stiffness=_csrEvidence(stiffness), bending=_csrEvidence(bending), systemMatrix=_csrEvidence(matrix),
        rightHandSide=rhs.copy(), rawBasis=rawBasis.copy(), singularValues=singularValues.copy(),
        columnNorms=columnNorms.copy(), supportRadius=radius, smoothingRadiusFraction=smoothing,
        normalization=normalization)
    diagnostics = dict(method='screenedBiharmonicFixedBoundary',
        equation='(M_U + ell^4 * (K.T M^-1 K)_UU) B_U = M_U W_U; B_F = 0',
        status=('noControls' if not len(centers) else 'allVerticesFixed' if not len(free) else 'solved'),
        vertexCount=len(original), freeVertexCount=len(free), fixedVertexCount=len(fixed),
        controlCount=len(centers), requestedControlCount=int(controlCount), activeVertexCount=int(active.sum()),
        activeFreeFraction=float(active[free].mean()) if len(free) else 0.,
        meaningfulActiveVertexCount=int(meaningful.sum()), meaningfulRelativeThreshold=1e-8,
        columnNorms=columnNorms.copy(), numericalRank=rank, rankTolerance=float(rankTolerance),
        supportRadius=radius, smoothingRadiusFraction=smoothing,
        smoothingLength=radius*smoothing, normalization=normalization,
        fixedRowsExactlyZero=bool(np.array_equal(basis[fixed], np.zeros_like(basis[fixed]))),
        maximumAbsoluteRowSum=float(np.abs(basis).sum(1).max(initial=0.)),
        solveAbsoluteResidual=residual, solveRelativeResidual=relativeResidual,
        symmetryAbsoluteError=symmetryError, diagonalRatioBeforeEquilibration=diagonalRatio,
        positiveDefiniteByConstruction=True,
        positiveDefiniteReason='positive free lumped mass plus ell^4 times a PSD Gram operator',
        numericalRidge=0., cotangentClipping=False, postSolveClipping=False,
        operatorUsesOriginalTriangleAdjacencyOnly=True, triangleDiagnostics=triangleDiagnostics,
        limitations=['This basis does not certify collision freedom or triangle quality.',
                    'Euclidean driving fields may correlate opposite eyelids; no proximity edges are added.',
                    'The original full-assembly acceptance gates remain necessary.'])
    return dict(basis=basis, controlPoints=centers, activeVertexMask=active,
                referenceVertices=original.copy(), faces=faces.copy(),
                normalization=normalization, evidence=evidence, diagnostics=diagnostics)
