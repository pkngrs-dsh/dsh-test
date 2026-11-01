"""Opt-in semantic boundary increments around a converged registration.

This module loads no model or camera and never changes topology. Callers own
anatomical loop annotation and numerical/assembly acceptance. A boundary
constraint is not a proof that two meshes have identical anatomical semantics.
"""
from __future__ import annotations

from numbers import Real
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _splitEyeArcs(reference, loop):
    """Freeze corners at canonical X extrema; order lower/upper by canonical Y."""
    loop = np.asarray(loop, np.int64)
    first = int(np.argmin(reference[loop, 0]))
    ordered = np.roll(loop, -first)
    last = int(np.argmax(reference[ordered, 0]))
    if not 1 < last < len(ordered) - 1:
        raise ValueError('Eye corners must split into two arcs with interior samples')
    arcs = [ordered[:last + 1], np.r_[ordered[0], ordered[:last - 1:-1]]]
    arcs.sort(key=lambda ids: float(reference[ids[1:-1], 1].mean()))
    return arcs


def _arcFractions(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Coincident consecutive boundary samples')
    distances = np.r_[0., np.cumsum(lengths)]
    return distances / distances[-1]


def buildEyeBoundaryTargets(clientReference, clientLoops, sourceReference, sourceLoops, sourceVertices):
    """Pair audited eyes by canonical side, retaining 3D depth of each UFF arc.

    References share a canonical frame where X runs across the face and Y
    upwards. Only references select corner IDs and the upper/lower arc order.
    All loops must already have audited eye semantics and matching topology;
    index validation here cannot infer anatomical meaning.
    """
    clientReference, sourceReference, sourceVertices = (np.asarray(value, np.float64)
        for value in (clientReference, sourceReference, sourceVertices))
    if (any(value.ndim != 2 or value.shape[1:] != (3,) or not np.isfinite(value).all()
            for value in (clientReference, sourceReference, sourceVertices))
            or sourceVertices.shape != sourceReference.shape):
        raise ValueError('Expected finite client/source Nx3 geometry with fixed source vertex order')
    for loops, count in ((clientLoops, len(clientReference)), (sourceLoops, len(sourceReference))):
        if len(loops) != 2:
            raise ValueError('Exactly two explicitly annotated eye loops are required')
        for values in loops:
            ids = np.asarray(values)
            if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 4
                    or ids.min() < 0 or ids.max() >= count or len(np.unique(ids)) != len(ids)):
                raise ValueError('Eye loops require distinct valid integer vertex indices')
        if np.intersect1d(*loops).size:
            raise ValueError('Left and right eye loops must be disjoint')
    clientLoops = sorted(clientLoops, key=lambda ids: clientReference[ids, 0].mean())
    sourceLoops = sorted(sourceLoops, key=lambda ids: sourceReference[ids, 0].mean())
    targets, definitions = {}, []
    for clientIds, sourceIds in zip(clientLoops, sourceLoops):
        clientArcs = _splitEyeArcs(clientReference, clientIds)
        sourceArcs = _splitEyeArcs(sourceReference, sourceIds)
        eye = []
        for clientArc, sourceArc in zip(clientArcs, sourceArcs):
            fractions = _arcFractions(clientReference[clientArc])
            sourceFractions = _arcFractions(sourceVertices[sourceArc])
            points = np.column_stack([np.interp(fractions, sourceFractions, sourceVertices[sourceArc, axis])
                                      for axis in range(3)])
            for vertex, target in zip(clientArc, points):
                if int(vertex) in targets and not np.allclose(targets[int(vertex)], target, rtol=0, atol=1e-12):
                    raise ValueError('Eye arcs disagree at shared corner')
                targets[int(vertex)] = target
            eye.append(dict(clientVertexIds=clientArc.tolist(), sourceVertexIds=sourceArc.tolist(),
                            frozenClientArcFractions=fractions.tolist()))
        definitions.append(eye)
    ids = np.asarray(sorted(targets), np.int64)
    return ids, np.asarray([targets[int(i)] for i in ids]), definitions


def solveBoundaryIncrement(baseline, operator, fixedIds, boundaryIds, targets, weight):
    """Add mean-square vertex targets around a converged quadratic optimum.

    E(base+d)-E(base) = d.T H d on the free variables. Thus solving
    (H + c B.T B)d = c B.T(target-B base) preserves the baseline objective
    exactly without adding the boundary observations to Umeyama alignment.
    The caller must supply the true Hessian and converged base of that objective.
    """
    if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real) or not np.isfinite(weight) or weight < 0:
        raise ValueError('Boundary weight must be finite and nonnegative')
    baseline, targets = np.asarray(baseline), np.asarray(targets)
    fixedIds, boundaryIds = np.asarray(fixedIds), np.asarray(boundaryIds)
    if (boundaryIds.ndim != 1 or fixedIds.ndim != 1):
        raise ValueError('Constraints require one-dimensional integer indices')
    if (baseline.ndim != 2 or baseline.shape[1:] != (3,) or len(baseline) < 3 or baseline.dtype.kind != 'f'
            or not np.isfinite(baseline).all() or targets.shape != (len(boundaryIds), 3)
            or targets.dtype.kind != 'f' or not np.isfinite(targets).all()):
        raise ValueError('Expected finite floating-point baseline and boundary targets')
    for ids in (fixedIds, boundaryIds):
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= len(baseline))):
            raise ValueError('Constraints require distinct valid integer vertex indices')
    if not len(boundaryIds) or np.intersect1d(fixedIds, boundaryIds).size:
        raise ValueError('Nonempty boundary constraints must be disjoint from fixed anchors')
    operator = sparse.csr_matrix(operator, dtype=np.float64, copy=True)
    if operator.shape != (len(baseline), len(baseline)) or not np.isfinite(operator.data).all():
        raise ValueError('Expected a finite square normal operator matching the geometry')
    asymmetry = operator - operator.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12 * max(1., np.max(np.abs(operator.data))):
        raise ValueError('Normal operator must be symmetric')
    baseline, targets = baseline.astype(np.float64, copy=True), targets.astype(np.float64, copy=True)
    if weight == 0:
        return baseline.copy()
    n = len(baseline)
    selector = sparse.csr_matrix((np.ones(len(boundaryIds)),
        (np.arange(len(boundaryIds)), boundaryIds)), shape=(len(boundaryIds), n))
    coefficient = weight / (3 * len(boundaryIds))
    hessian = (operator + coefficient * selector.T @ selector).tocsc()
    rhs = coefficient * selector.T @ (targets - baseline[boundaryIds])
    free = np.setdiff1d(np.arange(n), fixedIds)
    result = baseline.copy()
    try:
        increment = splu(hessian[free][:, free]).solve(rhs[free])
    except RuntimeError as error:
        raise ValueError('Boundary increment has a singular normal operator') from error
    if not np.isfinite(increment).all():
        raise ValueError('Boundary increment is not finite')
    result[free] += increment
    return result
