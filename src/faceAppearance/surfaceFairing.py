"""Local mesh fairing constrained to one fixed direction per vertex.

This CPU module preserves topology and zero-weight vertices. Camera rays can
be supplied as directions to preserve a reference projection (up to output
rounding); it does not estimate anatomy or check visibility/self-intersection.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def _finiteArray(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fibu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64, copy=True)


def _scalar(value, name, *, lower=0., upper=None, lowerInclusive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real scalar")
    if np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if ((value < lower if lowerInclusive else value <= lower)
            or (upper is not None and value > upper)):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _prepareInputs(vertices, faces, directions, regionWeights):
    source = np.asarray(vertices)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("vertices must use float32 or float64")
    original = _finiteArray(source, "vertices")
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 3:
        raise ValueError("vertices must be a finite Vx3 array")
    rawFaces = np.asarray(faces)
    if (rawFaces.ndim != 2 or rawFaces.shape[1] != 3 or not len(rawFaces)
            or rawFaces.dtype.kind not in "iu" or rawFaces.min() < 0
            or rawFaces.max() >= len(original)):
        raise ValueError("faces must be valid integer Fx3 vertex indices")
    triangles = rawFaces.astype(np.int64, copy=True)
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("faces cannot repeat a vertex")
    extent = float(np.ptp(original, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Initial mesh must have a positive finite extent")
    normalized = (original - original[0]) / extent
    cross = np.cross(normalized[triangles[:, 1]] - normalized[triangles[:, 0]],
                     normalized[triangles[:, 2]] - normalized[triangles[:, 0]])
    doubleArea = np.linalg.norm(cross, axis=1)
    if not np.isfinite(normalized).all() or not np.isfinite(doubleArea).all() or np.any(doubleArea <= 0):
        raise ValueError("Initial mesh contains degenerate triangles")
    unitDirections = _finiteArray(directions, "directions")
    if unitDirections.shape != original.shape:
        raise ValueError("directions must have one unit 3D vector per vertex")
    lengths = np.linalg.norm(unitDirections, axis=1)
    if not np.allclose(lengths, 1., rtol=0, atol=1e-6):
        raise ValueError("directions must be unit length within 1e-6")
    unitDirections /= lengths[:, None]
    weights = _finiteArray(regionWeights, "regionWeights")
    if weights.shape != (len(original),) or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError("regionWeights must be a V-vector in [0, 1]")
    return original, triangles, unitDirections, weights, extent, normalized, source.dtype


def _buildLaplacian(vertexCount, faces):
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)
    rows, columns = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(vertexCount, vertexCount))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags((degree > 0).astype(float)) - sparse.diags(1 / np.maximum(degree, 1)) @ adjacency
    return laplacian.tocsr(), edges


def refineSurfaceAlongDirections(vertices, faces, directions, regionWeights, *, smoothWeight=10.,
                                 maxDisplacementFraction=.015, maxRelativeEdgeChange=.30,
                                 minAreaRatio=.35, minNormalCosine=.5, maxBacktracking=24):
    """Return safe fixed-topology fairing along supplied unit directions.

    With extent ``e=max(ptp(vertices))``, normalized positions ``p=V/e``, unit
    directions ``n``, weights ``w`` and scalar normalized displacement ``u``,
    solve the following quadratic over positive-weight vertices only::

        mean_active((u_i / w_i)**2)
        + smoothWeight * sum_i(q_i * ||L(p + n*u)_i||**2) / sum_i(q_i)

    ``L=I-D^-1*A`` is the uniform topology Laplacian; ``q_i`` is the maximum
    region weight at i and its one-ring neighbors. These are exactly the rows
    potentially changed by the field. This penalizes absolute surface
    differential coordinates, not displacement curvature. The substitution
    ``u=w*z`` avoids dividing by tiny weights in the linear system. Scale and
    row/count normalization make the terms independent of coordinate units
    and avoid trivially multiplying smoothness by the selected region size;
    they do not make this discretization invariant to remeshing.

    Zero weights are hard pins. A single multiplier backtracks the entire
    solved field; vertices are never independently clipped. Quality and
    objective reduction are checked on the actual dtype-cast output. Status
    is ``accepted`` or ``unchanged`` with a reason. ``displacement`` is the
    actual returned Vx3 delta in input units; ``scalarDisplacement`` is its
    projection onto the normalized supplied directions. Rounding can leave
    a tiny perpendicular delta, explicitly reported.

    Local relative area, normal, edge and displacement gates do not certify
    anatomical correctness, unchanged visibility, or no self-intersections.
    The caller must also check positive camera depth when using camera rays.
    """
    original, faces, directions, weights, extent, normalized, resultDtype = _prepareInputs(
        vertices, faces, directions, regionWeights)
    smoothWeight = _scalar(smoothWeight, "smoothWeight")
    maxDisplacementFraction = _scalar(maxDisplacementFraction, "maxDisplacementFraction")
    maxRelativeEdgeChange = _scalar(maxRelativeEdgeChange, "maxRelativeEdgeChange", upper=1., lowerInclusive=False)
    if maxRelativeEdgeChange >= 1:
        raise ValueError("maxRelativeEdgeChange must be less than one")
    minAreaRatio = _scalar(minAreaRatio, "minAreaRatio", upper=1., lowerInclusive=False)
    minNormalCosine = _scalar(minNormalCosine, "minNormalCosine", upper=1., lowerInclusive=False)
    if (isinstance(maxBacktracking, (bool, np.bool_)) or not isinstance(maxBacktracking, (int, np.integer))
            or maxBacktracking < 0):
        raise ValueError("maxBacktracking must be a nonnegative integer")
    active = weights > 0
    activeIds = np.flatnonzero(active)
    laplacian, edges = _buildLaplacian(len(original), faces)
    rowWeights = weights.copy()
    np.maximum.at(rowWeights, edges[:, 0], weights[edges[:, 1]])
    np.maximum.at(rowWeights, edges[:, 1], weights[edges[:, 0]])
    selectedRows = np.flatnonzero(rowWeights > 0)
    localLaplacian = laplacian[selectedRows]
    normalizedRowWeights = rowWeights[selectedRows] / rowWeights[selectedRows].sum() if len(selectedRows) else np.empty(0)
    initialEdgeLengths = np.linalg.norm(normalized[edges[:, 1]] - normalized[edges[:, 0]], axis=1)
    if np.any(initialEdgeLengths <= 0):
        raise ValueError("Initial mesh contains zero-length edges")
    initialCross = np.cross(normalized[faces[:, 1]] - normalized[faces[:, 0]],
                            normalized[faces[:, 2]] - normalized[faces[:, 0]])
    initialDoubleArea = np.linalg.norm(initialCross, axis=1)
    initialNormals = initialCross / initialDoubleArea[:, None]

    def evaluateEnergy(candidate):
        scaledDelta = (candidate - original) / extent
        data = float(np.square(scaledDelta[active] / weights[active, None]).sum() / len(activeIds)) if len(activeIds) else 0.
        lap = localLaplacian @ ((candidate - original[0]) / extent)
        fairing = float(normalizedRowWeights @ np.square(lap).sum(axis=1))
        return {"displacement": data, "fairing": fairing, "total": data + smoothWeight * fairing}

    def evaluateSafety(candidate):
        if np.array_equal(candidate, original):
            return {"maximumRelativeEdgeChange": 0., "minimumAreaRatio": 1., "maximumAreaRatio": 1.,
                    "minimumNormalCosine": 1., "invertedOrDegenerateFaceCount": 0,
                    "maximumDisplacement": 0., "maximumDisplacementFraction": 0.}, []
        points = (candidate - original[0]) / extent
        edgeLengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
        area = np.linalg.norm(cross, axis=1)
        normals = np.divide(cross, area[:, None], out=np.zeros_like(cross), where=area[:, None] > 0)
        cosine = np.clip(np.einsum("ij,ij->i", normals, initialNormals), -1., 1.)
        ratios = area / initialDoubleArea
        maximumDisplacement = float(np.linalg.norm(candidate - original, axis=1).max())
        metrics = {"maximumRelativeEdgeChange": float(np.abs(edgeLengths / initialEdgeLengths - 1).max()),
                   "minimumAreaRatio": float(ratios.min()), "maximumAreaRatio": float(ratios.max()),
                   "minimumNormalCosine": float(cosine.min()),
                   "invertedOrDegenerateFaceCount": int(np.count_nonzero(cosine <= 0)),
                   "maximumDisplacement": maximumDisplacement,
                   "maximumDisplacementFraction": maximumDisplacement / extent}
        reasons = []
        if not all(np.isfinite(value) for value in metrics.values()):
            reasons.append("nonfiniteGeometry")
        if metrics["maximumRelativeEdgeChange"] > maxRelativeEdgeChange:
            reasons.append("relativeEdgeChange")
        if metrics["minimumAreaRatio"] < minAreaRatio:
            reasons.append("areaRatio")
        if metrics["minimumNormalCosine"] < minNormalCosine:
            reasons.append("normalCosine")
        if metrics["maximumDisplacementFraction"] > maxDisplacementFraction:
            reasons.append("displacementLimit")
        return metrics, reasons

    initialEnergy = evaluateEnergy(original)
    diagnostics = {"status": "unchanged", "reason": "noActiveRegion", "acceptedScale": 0.,
                   "vertexCount": len(original), "faceCount": len(faces), "activeVertexCount": len(activeIds),
                   "fairingRowCount": len(selectedRows), "normalizationExtent": extent,
                   "smoothWeight": smoothWeight, "initialEnergy": initialEnergy, "rawEnergy": initialEnergy.copy(),
                   "candidateChecks": 0, "rejectionCounts": {}, "rawMaximumDisplacementFraction": 0.,
                   "safetyLimits": {"maxDisplacementFraction": maxDisplacementFraction,
                                    "maxRelativeEdgeChange": maxRelativeEdgeChange, "minAreaRatio": minAreaRatio,
                                    "minNormalCosine": minNormalCosine},
                   "topologyUnchanged": True, "selfIntersectionChecked": False,
                   "objective": "mean_active(||delta/extent/regionWeight||^2) + smoothWeight*weightedMean_rows(||L(V/extent)||^2)",
                   "rowWeights": "max regionWeight over vertex and its one-ring neighbors",
                   "backtracking": "one shared field multiplier; dtype-cast output rechecked",
                   "limits": ["Uniform graph fairing is not anatomical reconstruction or remeshing-invariant curvature.",
                              "No self-intersection, camera-depth, visibility or multi-view consistency test is performed."]}

    def finish(result, status, reason, scale=0.):
        result = np.asarray(result, dtype=resultDtype).copy()
        result[~active] = original[~active].astype(resultDtype)
        delta = result.astype(np.float64) - original
        scalarDelta = np.einsum("ij,ij->i", delta, directions)
        perpendicular = delta - scalarDelta[:, None] * directions
        safety, failures = evaluateSafety(result.astype(np.float64))
        if failures:
            raise RuntimeError("Internal error: returned fairing geometry violates its safety contract")
        diagnostics.update(status=status, reason=reason, acceptedScale=float(scale), safety=safety,
                           finalEnergy=evaluateEnergy(result.astype(np.float64)),
                           maximumOrthogonalDisplacement=float(np.linalg.norm(perpendicular, axis=1).max()),
                           zeroWeightVerticesExactlyUnchanged=bool(np.array_equal(result[~active], np.asarray(vertices)[~active])))
        return {"vertices": result, "displacement": delta, "scalarDisplacement": scalarDelta,
                "diagnostics": diagnostics}

    if not len(activeIds):
        return finish(original, "unchanged", "noActiveRegion")
    if smoothWeight == 0 or maxDisplacementFraction == 0:
        return finish(original, "unchanged", "zeroSmoothWeight" if smoothWeight == 0 else "zeroDisplacementLimit")
    selectedColumns = localLaplacian[:, activeIds]
    initialLaplacian = localLaplacian @ normalized
    operator = sparse.eye(len(activeIds), format="csc") / len(activeIds)
    rightHandSide = np.zeros(len(activeIds))
    for dimension in range(3):
        field = selectedColumns.multiply((weights[active] * directions[active, dimension])[None, :]).tocsr()
        operator = operator + smoothWeight * (field.T @ field.multiply(normalizedRowWeights[:, None]))
        rightHandSide -= smoothWeight * (field.T @ (normalizedRowWeights * initialLaplacian[:, dimension]))
    if not np.isfinite(operator.data).all() or not np.isfinite(rightHandSide).all():
        return finish(original, "unchanged", "nonfiniteLinearSystem")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            coefficients = spsolve(operator.tocsc(), rightHandSide)
    except (RuntimeError, MatrixRankWarning):
        return finish(original, "unchanged", "linearSolveFailed")
    if not np.isfinite(coefficients).all():
        return finish(original, "unchanged", "nonfiniteLinearSolution")
    rawScalar = np.zeros(len(original))
    rawScalar[active] = weights[active] * coefficients
    rawDelta = extent * rawScalar[:, None] * directions
    diagnostics["rawEnergy"] = evaluateEnergy(original + rawDelta)
    rawMaximumFraction = float(np.linalg.norm(rawDelta, axis=1).max() / extent)
    diagnostics["rawMaximumDisplacementFraction"] = rawMaximumFraction
    if rawMaximumFraction <= 1e-12:
        return finish(original, "unchanged", "noEffectiveDisplacement")
    initialScale = min(1., maxDisplacementFraction / rawMaximumFraction)
    if rawMaximumFraction >= maxDisplacementFraction:
        # Leave a small dtype-dependent margin at the displacement cap. The
        # actual cast still passes all gates below; no geometry is exempted.
        initialScale *= 1 - 8 * np.finfo(resultDtype).eps
    improvementFloor = 1e-12 * max(initialEnergy["total"], 1e-30)
    for iteration in range(maxBacktracking + 1):
        scale = initialScale * .5 ** iteration
        candidate = (original + scale * rawDelta).astype(resultDtype)
        candidate[~active] = original[~active].astype(resultDtype)
        checked = candidate.astype(np.float64)
        diagnostics["candidateChecks"] += 1
        _, failures = evaluateSafety(checked)
        energy = evaluateEnergy(checked)
        if not np.isfinite(energy["total"]) or initialEnergy["total"] - energy["total"] <= improvementFloor:
            failures.append("noObjectiveImprovement")
        if np.array_equal(candidate, np.asarray(vertices)):
            failures.append("noRepresentableChange")
        if not failures:
            reason = "acceptedFullStep" if scale == 1 else "acceptedBacktrackedStep"
            return finish(candidate, "accepted", reason, scale)
        for name in failures:
            diagnostics["rejectionCounts"][name] = diagnostics["rejectionCounts"].get(name, 0) + 1
    return finish(original, "unchanged", "noSafeImprovingStep")
