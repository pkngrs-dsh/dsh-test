"""Directly register a fixed client mesh using the old masked objective.

Each observation is bound to a CLIENT triangle and barycentric coordinates.
Client texture UVs are not used. Umeyama first aligns these client samples to
the valid reconstructed observations. The CPU solver then finds the converged
minimum of the old quadratic objective in that target coordinate frame::

    dataWeight * mean(concat(C @ V - Q, V[static] - alignedRest[static])**2)
    + smoothWeight * mean((L @ V)**2)
    + restWeight * mean((V - alignedRest)**2)

L is the unnormalized combinatorial Laplacian D-A, matching the upstream
largesteps.geometry.laplacian_uniform implementation (unique undirected edges):
https://github.com/rgl-epfl/large-steps-pytorch/blob/master/largesteps/geometry.py
The source vendor function is reconstruct_mesh_largesteps_certain_template_with_mask.

The default backend is NOT the old finite 600-step float32 AdamUniform
trajectory. It solves the same objective in float64, without the Large Steps
parameterization, whose invertible I+19L only affects the optimization path.
An optional float32 CPU algorithm replay uses that parameterization, the old
AdamUniform update and cosine schedule. Its SuperLU solve is not bitwise equal
to the original CUDA Cholesky backend. The default static anchors are SOFT;
the optional hard boundary is an explicitly different constraint. Smoothness
acts on absolute positions, not displacements. No current per-vertex guard,
rig inference, texture transfer or automatic quality acceptance is applied.
"""

from __future__ import annotations

from numbers import Real
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def _floats(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind != "f" or (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite floating-point array" + (f" of shape {shape}" if shape else ""))
    return np.array(array, dtype=np.float64, copy=True)


def _indices(value, name, shape, bound):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array >= bound)):
        raise ValueError(f"{name} must contain integer indices of shape {shape} in [0,{bound})")
    return np.array(array, dtype=np.int64, copy=True)


def _weight(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")
    return float(value)


def _laplacian(vertexCount, faces):
    # Upstream adds each directed neighbor exactly once, irrespective of how
    # many triangles share the edge, and sums one +1 diagonal per neighbor.
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                               faces[:, [2, 0]]]), axis=1), axis=0)
    rows, columns = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(vertexCount, vertexCount))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree) - adjacency).tocsr(), adjacency


def _umeyama(source, target):
    """Old unweighted row-vector fit, with explicit degeneracy checks."""
    sourceMean, targetMean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - sourceMean, target - targetMean
    covariance = x.T @ y / len(source)
    if not np.isfinite(covariance).all():
        raise ValueError("Similarity covariance is not finite")
    left, singular, right = np.linalg.svd(covariance)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-12:
        raise ValueError("Similarity needs at least three non-collinear paired observations")
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        singular[-1] *= -1
        left[:, -1] *= -1
    rotation = left @ right
    variance = np.var(source, axis=0).sum()
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Similarity source samples are collapsed")
    scale = float(singular.sum() / variance)
    translation = targetMean - sourceMean @ rotation * scale
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError("Similarity transform has invalid scale or translation")
    return {"scale": scale, "rotation": rotation, "translation": translation}


def _residualSummary(values):
    if not len(values):
        return {"count": 0, "rms": None, "median": None, "p95": None, "max": None}
    lengths = np.linalg.norm(values, axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("Registration residuals are not finite")
    return {"count": len(lengths), "rms": float(np.sqrt(np.mean(lengths**2))),
            "median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)),
            "max": float(lengths.max())}


def _objective(vertices, aligned, sampleMatrix, observations, staticIds, laplacian, coefficients,
               smoothTarget=None):
    observedResidual = sampleMatrix @ vertices - observations
    staticResidual = vertices[staticIds] - aligned[staticIds]
    laplacianValues = laplacian @ vertices
    smoothResidual = laplacianValues if smoothTarget is None else laplacianValues - smoothTarget
    restResidual = vertices - aligned
    observedSse = float(np.sum(observedResidual**2))
    staticSse = float(np.sum(staticResidual**2))
    laplacianSse = float(np.sum(laplacianValues**2))
    smoothResidualSse = float(np.sum(smoothResidual**2))
    restSse = float(np.sum(restResidual**2))
    data = coefficients["data"] * (observedSse + staticSse)
    smooth = coefficients["smooth"] * smoothResidualSse
    rest = coefficients["rest"] * restSse
    if not np.isfinite([observedSse, staticSse, laplacianSse, restSse, data, smooth, rest]).all():
        raise ValueError("Registration objective overflowed")
    return {"total": data + smooth + rest, "data": data, "smooth": smooth, "rest": rest,
            "dataObservedPart": coefficients["data"] * observedSse,
            "dataStaticAnchorPart": coefficients["data"] * staticSse,
            "observedSse": observedSse, "staticAnchorSse": staticSse,
            "absoluteLaplacianSse": laplacianSse, "smoothResidualSse": smoothResidualSse, "restSse": restSse,
            "observedResidual": _residualSummary(observedResidual),
            "staticAnchorResidual": _residualSummary(staticResidual),
            "restDisplacement": _residualSummary(restResidual)}


def _quadraticPrior(value, vertexCount):
    """Validate a caller-normalized sparse objective without densifying it."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"operator", "targets", "weight"}:
        raise ValueError("quadraticPrior requires exactly operator, targets and weight")
    raw = value["operator"]
    if (not sparse.issparse(raw) or raw.format not in ("csr", "csc", "coo")
            or len(raw.shape) != 2 or raw.shape[0] < 1 or raw.shape[1] != 3 * vertexCount
            or raw.dtype.kind not in "fiu"):
        raise ValueError("quadraticPrior operator must be a real CSR/CSC/COO sparse Mx(3N) array")
    # A.T@A can become dense even when A is sparse. Bound both storage and
    # per-row outer-product work BEFORE constructing that normal matrix.
    if raw.shape[0] > 1_000_000 or raw.nnz > 1_000_000:
        raise ValueError("quadraticPrior exceeds the sparse storage budget")
    if not np.isfinite(raw.data).all():
        raise ValueError("quadraticPrior operator must have finite coefficients")
    matrix = sparse.csr_matrix(raw, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError("quadraticPrior duplicate coefficients overflowed")
    rowCounts = np.diff(matrix.indptr).astype(np.float64)
    if np.dot(rowCounts, rowCounts) > 20_000_000:
        raise ValueError("quadraticPrior normal matrix would exceed the sparse product budget")
    targets = _floats(value["targets"], "quadraticPrior targets", (matrix.shape[0],))
    weight = _weight(value["weight"], "quadraticPrior weight")
    return {"operator": matrix, "targets": targets, "weight": weight}


def _addPriorObjective(objective, vertices, prior):
    residual = prior["operator"] @ vertices.ravel() - prior["targets"]
    with np.errstate(over="ignore", invalid="ignore"):
        sse = float(np.sum(residual**2))
        loss = prior["weight"] * sse
        total = objective["total"] + loss
    if not np.isfinite([sse, loss, total]).all():
        raise ValueError("Quadratic prior objective overflowed")
    objective.update(total=total, quadraticPrior=loss, quadraticPriorSse=sse,
        quadraticPriorResidual={"count": len(residual), "rms": float(np.sqrt(sse / len(residual))),
                                "maxAbs": float(np.abs(residual).max())})


def _legacyAdamCpu(aligned, design, targets, laplacian, coefficients, epochs):
    """Replay old AdamUniform semantics using analytic gradients on CPU.

    The second-moment denominator is one scalar maximum over ALL coordinates,
    as documented in upstream largesteps/optimize.py. M is symmetric, so the
    chain rule is dE/du = M^-1 dE/dV. This implementation adds no dependency and
    does not silently replace the scalar normalization with ordinary Adam.
    Like the vendor loop, it returns the V evaluated BEFORE the final update.
    """
    n = len(aligned)
    laplacian = laplacian.astype(np.float32)
    design = design.astype(np.float32)
    reference, targets = aligned.astype(np.float32), targets.astype(np.float32)
    parameterization = (sparse.eye(n, dtype=np.float32, format="csc")
                        + np.float32(19.) * laplacian).tocsc()
    try:
        inverse = splu(parameterization)
    except RuntimeError as error:
        raise ValueError("Large Steps parameterization cannot be factorized") from error
    u = np.asarray(parameterization @ reference, dtype=np.float32)
    firstMoment, secondMoment = np.zeros_like(u), np.zeros_like(u)
    lossHistory = []
    cd, cs, cr = (np.float32(coefficients[key]) for key in ("data", "smooth", "rest"))
    for step in range(1, epochs + 1):
        vertices = inverse.solve(u)
        observed = design @ vertices - targets
        smooth = laplacian @ vertices
        prior = vertices - reference
        loss = cd * np.sum(observed**2) + cs * np.sum(smooth**2) + cr * np.sum(prior**2)
        gradientV = (np.float32(2.) * cd * (design.T @ observed)
                     + np.float32(2.) * cs * (laplacian.T @ smooth)
                     + np.float32(2.) * cr * prior)
        gradient = inverse.solve(np.asarray(gradientV, dtype=np.float32), trans="T")
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("CPU Large Steps replay produced nonfinite loss or gradient")
        firstMoment *= np.float32(.9)
        firstMoment += np.float32(.1) * gradient
        secondMoment *= np.float32(.999)
        secondMoment += np.float32(.001) * gradient**2
        correctedFirst = firstMoment / np.float32(1 - .9**step)
        correctedSecond = secondMoment / np.float32(1 - .999**step)
        direction = correctedFirst / (np.float32(1e-8) + np.sqrt(correctedSecond).max())
        learningRate = .003 + (.01 - .003) * (1 + np.cos(np.pi * (step - 1) / epochs)) / 2
        u -= np.float32(learningRate) * direction
        lossHistory.append(float(loss))
    return vertices.astype(np.float64), {
        "parameterization": "I+19*(D-A)", "parameterizationSolve": "SciPySuperLUFloat32",
        "optimizer": "AdamUniformGlobalMaximumSecondMoment", "betas": [.9, .999], "epsilon": 1e-8,
        "learningRate": {"initial": .01, "minimum": .003, "schedule": "cosine", "lastUsed": learningRate},
        "lossHistoryFloat32": lossHistory, "returnedAfterParameterUpdates": epochs - 1,
        "returnConvention": "vendorReturnsVerticesEvaluatedBeforeFinalOptimizerStep",
        "precisionLimit": "Float32 CPU sparse solves and reductions differ from the original CUDA Cholesky execution.",
    }


def registerDirectTemplate(restVertices, faces, sampleFaceIds, sampleBarycentrics,
                           targetPoints, validMask, staticVertexIds, *, dataWeight=10000.,
                           smoothWeight=3000., restWeight=1000., solver="exact", epochs=600,
                           staticConstraint="soft", smoothnessMode="absolute",
                           differentialWeights=None, quadraticPrior=None):
    """Return the raw old-objective optimum, in the observations' coordinates.

    ``sampleFaceIds[k]`` and ``sampleBarycentrics[k]`` locate observation k on
    the CLIENT rest mesh; these are not indices of the reconstructed mesh.
    Only ``validMask`` observations enter alignment and optimization. All
    coordinates must be finite; excluded legacy extrapolation barycentrics may
    be outside the triangle, while included ones must be convex coordinates.

    ``staticVertexIds`` are unique vertex indices. They are concatenated with
    observations in one MSE, so their count affects the data-term denominator.
    The default ``staticConstraint='soft'`` reproduces the old objective.
    ``'hard'`` fixes those vertices to alignedRest, eliminates them from the
    system, and retains the OLD joint-data denominator for a controlled
    comparison; its static residuals are identically zero. Empty static indices
    should be an integer array.

    ``restWeight`` accepts any nonnegative value, including the experiment
    presets 1000 (old), 100 (weak) and 0 (none). ``solver='exact'`` ignores
    ``epochs`` except as reference metadata. ``solver='legacyAdamCpu'`` performs
    that many old-algorithm float32 steps with soft static anchors, using CPU
    SuperLU and an analytic chain-rule gradient. It needs no installed Large
    Steps package, but does not claim bitwise CUDA trajectory reproduction.

    Returned arrays: ``alignedRest``, ``rawVertices``; ``transform`` obeys
    alignedRest = restVertices @ rotation * scale + translation. Geometry
    safety/appearance/rig suitability are intentionally not acceptance gates.

    The default ``smoothnessMode='absolute'`` retains the legacy objective.
    Experimental ``'displacement'`` uses ``L@V - w[:,None]*(L@alignedRest)``
    instead of ``L@V``. ``differentialWeights`` is an optional finite [0,1]
    vector per Laplacian ROW, defaulting to ones; zero recovers absolute
    smoothing and one preserves the aligned template's differential coordinate.
    This is not ``L@(V-w*alignedRest)`` and does not rescale data observations.
    It retains a local template-curvature prior even with ``restWeight=0``;
    no anatomical fidelity or collision acceptance is implied. Only the exact
    solver supports this opt-in mode. Production defaults are unchanged.

    ``quadraticPrior`` optionally adds ``weight * ||A @ V.ravel() - t||^2``.
    Its exact keys are ``operator`` (real CSR/CSC/COO sparse Mx(3N)), ``targets``
    (finite floating M-vector) and ``weight`` (finite nonnegative scalar).
    Flattening is vertex-major XYZ. A and t must already describe the TARGET
    coordinate frame of alignedRest; no rest-to-target transform is applied to
    them. A caller needing alignedRest may first run the unmodified baseline
    with identical inputs to obtain its observation-only Umeyama transform.
    This prior never affects alignment or the legacy MSE denominators, and no
    row or target normalization is added. The caller owns physical units,
    confidence and normalization. Only ``solver='exact'`` accepts the option.

    Positive weights use a coupled XYZ sparse solve with exact hard-static
    elimination. None and a validated zero weight retain the original solver
    and diagnostics path exactly. Dense operators, excessive sparse products,
    and numerically singular coupled systems are rejected. The relative LU
    pivot threshold is a numerical solvability guard, not an exact rank proof.
    This is a generic quadratic term, not automatic anatomical or mesh-quality
    acceptance.
    """
    started = perf_counter()
    if solver not in ("exact", "legacyAdamCpu"):
        raise ValueError("solver must be 'exact' or 'legacyAdamCpu'")
    if staticConstraint not in ("soft", "hard"):
        raise ValueError("staticConstraint must be 'soft' or 'hard'")
    if smoothnessMode not in ("absolute", "displacement"):
        raise ValueError("smoothnessMode must be 'absolute' or 'displacement'")
    if smoothnessMode == "absolute" and differentialWeights is not None:
        raise ValueError("differentialWeights requires displacement smoothness")
    if smoothnessMode != "absolute" and solver != "exact":
        raise ValueError("Displacement smoothness requires the exact solver")
    if quadraticPrior is not None and solver != "exact":
        raise ValueError("quadraticPrior requires the exact solver")
    if solver == "legacyAdamCpu" and staticConstraint != "soft":
        raise ValueError("legacyAdamCpu reproduces only the old soft-anchor algorithm")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive reference trajectory length")
    vertices = _floats(restVertices, "restVertices")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError("restVertices must be an Nx3 array with at least three vertices")
    rawFaces = np.asarray(faces)
    if rawFaces.ndim != 2 or rawFaces.shape[1:] != (3,) or len(rawFaces) < 1:
        raise ValueError("faces must be a nonempty Fx3 triangle array")
    triangles = _indices(rawFaces, "faces", rawFaces.shape, len(vertices))
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("A triangle may not repeat a vertex")
    rawSamples = np.asarray(sampleFaceIds)
    if rawSamples.ndim != 1 or len(rawSamples) < 3:
        raise ValueError("At least three sampleFaceIds are required")
    sampleIds = _indices(rawSamples, "sampleFaceIds", rawSamples.shape, len(triangles))
    barycentrics = _floats(sampleBarycentrics, "sampleBarycentrics", (len(sampleIds), 3))
    observations = _floats(targetPoints, "targetPoints", (len(sampleIds), 3))
    valid = np.asarray(validMask)
    if valid.shape != sampleIds.shape or valid.dtype.kind != "b" or valid.sum() < 3:
        raise ValueError("validMask must be a boolean vector selecting at least three samples")
    valid = valid.copy()
    if (not np.allclose(barycentrics.sum(axis=1), 1., rtol=0, atol=1e-6)
            or np.any(barycentrics[valid] < 0) or np.any(barycentrics[valid] > 1)):
        raise ValueError("Barycentrics must sum to one; valid samples require convex triangle weights")
    rawStatic = np.asarray(staticVertexIds)
    if rawStatic.ndim != 1:
        raise ValueError("staticVertexIds must be a one-dimensional integer array")
    static = _indices(rawStatic, "staticVertexIds", rawStatic.shape, len(vertices))
    if len(np.unique(static)) != len(static):
        raise ValueError("staticVertexIds may not contain duplicates")
    dataWeight = _weight(dataWeight, "dataWeight", positive=True)
    smoothWeight = _weight(smoothWeight, "smoothWeight")
    restWeight = _weight(restWeight, "restWeight")

    sampleIds, barycentrics, observations = sampleIds[valid], barycentrics[valid], observations[valid]
    sampleCount, vertexCount, staticCount = len(sampleIds), len(vertices), len(static)
    prior = _quadraticPrior(quadraticPrior, vertexCount)
    priorEnabled = prior is not None and prior["weight"] > 0
    sampleMatrix = sparse.csr_matrix((barycentrics.ravel(),
        (np.repeat(np.arange(sampleCount), 3), triangles[sampleIds].ravel())),
        shape=(sampleCount, vertexCount))
    # Do not use uvVertices' stored mapping, weights or global alignment here.
    transform = _umeyama(sampleMatrix @ vertices, observations)
    aligned = vertices @ transform["rotation"] * transform["scale"] + transform["translation"]
    if not np.isfinite(aligned).all():
        raise ValueError("Aligned template is not finite")
    laplacian, adjacency = _laplacian(vertexCount, triangles)
    smoothTarget = None
    differential = None
    if smoothnessMode == "displacement":
        differential = (np.ones(vertexCount, np.float64) if differentialWeights is None else
                        _floats(differentialWeights, "differentialWeights", (vertexCount,)))
        if np.any((differential < 0) | (differential > 1)):
            raise ValueError("differentialWeights must lie in [0, 1]")
        smoothTarget = differential[:, None] * (laplacian @ aligned)
    anchors = sparse.csr_matrix((np.ones(staticCount), (np.arange(staticCount), static)),
                               shape=(staticCount, vertexCount))
    design = sparse.vstack((sampleMatrix, anchors), format="csr")
    targets = np.concatenate((observations, aligned[static]), axis=0)
    coefficients = {"data": dataWeight / (3 * (sampleCount + staticCount)),
                    "smooth": smoothWeight / (3 * vertexCount), "rest": restWeight / (3 * vertexCount)}
    if restWeight == 0 and smoothWeight > 0:
        componentCount, component = connected_components(adjacency, directed=False)
        anchored = np.zeros(componentCount, dtype=bool)
        sampledVertices = triangles[sampleIds][barycentrics > 0]
        anchored[component[np.r_[sampledVertices, static]]] = True
        if not anchored.all() and not priorEnabled:
            raise ValueError("Unobserved mesh components are unconstrained when restWeight is zero")
    operator = (coefficients["data"] * (design.T @ design)
                + coefficients["smooth"] * (laplacian.T @ laplacian)
                + coefficients["rest"] * sparse.eye(vertexCount, format="csr")).tocsc()
    rhs = coefficients["data"] * (design.T @ targets) + coefficients["rest"] * aligned
    if smoothTarget is not None:
        rhs = rhs + coefficients["smooth"] * (laplacian.T @ smoothTarget)
    if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
        raise ValueError("Registration system contains nonfinite coefficients")
    free = np.arange(vertexCount, dtype=np.int64)
    result = aligned.copy()
    if staticConstraint == "hard":
        free = np.setdiff1d(free, static, assume_unique=True)
    replay = None
    coupledOperator, coupledRhs, freeCoordinates = None, None, None
    if solver == "legacyAdamCpu":
        result, replay = _legacyAdamCpu(aligned, design, targets, laplacian, coefficients, epochs)
    elif priorEnabled:
        matrix, priorTargets, priorWeight = prior["operator"], prior["targets"], prior["weight"]
        with np.errstate(over="ignore", invalid="ignore"):
            coupledOperator = (sparse.kron(operator, sparse.eye(3), format="csc")
                               + priorWeight * (matrix.T @ matrix)).tocsc()
            coupledRhs = rhs.ravel() + priorWeight * (matrix.T @ priorTargets)
        if not np.isfinite(coupledOperator.data).all() or not np.isfinite(coupledRhs).all():
            raise ValueError("Quadratic prior system contains nonfinite coefficients")
        freeCoordinates = (3 * free[:, None] + np.arange(3)).ravel()
        if len(freeCoordinates):
            freeOperator = coupledOperator[freeCoordinates][:, freeCoordinates].tocsc()
            freeRhs = coupledRhs[freeCoordinates]
            if staticConstraint == "hard":
                staticCoordinates = (3 * static[:, None] + np.arange(3)).ravel()
                freeRhs = freeRhs - coupledOperator[freeCoordinates][:, staticCoordinates] @ aligned[static].ravel()
            try:
                factor = splu(freeOperator)
                # A tiny numerical pivot can otherwise hide a remaining rigid
                # translation nullspace in a relative-only prior. Reject that
                # numerically unconstrained solve instead of accepting a small
                # residual for an arbitrary member of a nonunique solution.
                pivots = np.abs(factor.U.diagonal())
                scale = max(float(np.abs(freeOperator.data).max(initial=0)), np.finfo(float).tiny)
                if pivots.min() <= scale * 1e-12:
                    raise ValueError("Quadratic prior leaves a singular or numerically unconstrained system")
                result[free] = factor.solve(freeRhs).reshape(-1, 3)
            except RuntimeError as error:
                raise ValueError("Quadratic prior leaves a singular unconstrained system") from error
    elif len(free):
        freeOperator = operator[free][:, free].tocsc()
        freeRhs = rhs[free]
        if staticConstraint == "hard":
            freeRhs = freeRhs - operator[free][:, static] @ aligned[static]
        try:
            factor = splu(freeOperator)
            result[free] = factor.solve(freeRhs)
        except RuntimeError as error:
            raise ValueError("Registration objective has a singular unconstrained system") from error
    if not np.isfinite(result).all():
        raise ValueError("Registration solver returned nonfinite vertices")
    if priorEnabled:
        residual = (coupledOperator @ result.ravel() - coupledRhs)[freeCoordinates]
        residualScale = np.linalg.norm(coupledRhs[freeCoordinates])
    else:
        residual = (operator @ result - rhs)[free]
        residualScale = np.linalg.norm(rhs[free])
    relativeResidual = float(np.linalg.norm(residual) / max(residualScale, np.finfo(float).tiny))
    if not np.isfinite(relativeResidual) or (solver == "exact" and relativeResidual > 1e-8):
        raise ValueError("Registration normal equations did not converge accurately")
    before = _objective(aligned, aligned, sampleMatrix, observations, static, laplacian, coefficients, smoothTarget)
    after = _objective(result, aligned, sampleMatrix, observations, static, laplacian, coefficients, smoothTarget)
    if priorEnabled:
        _addPriorObjective(before, aligned, prior)
        _addPriorObjective(after, result, prior)
    if solver == "exact" and after["total"] > before["total"] + max(1., before["total"]) * 1e-8:
        raise ValueError("Registration solution increased the objective")
    diagnostics = {
        "method": ("directClientSurfaceSamplesLegacyMaskedObjective" if smoothnessMode == "absolute" else
                   "directClientSurfaceSamplesDifferentialRegularization"),
        "solver": "cpuFloat64SparseQuadraticOptimum" if solver == "exact" else "cpuFloat32LargeStepsAdamUniformReplay",
        "convergedObjectiveSolution": solver == "exact",
        "legacyFiniteStepTrajectoryMatched": False, "referenceEpochs": epochs,
        "legacyFiniteStepAlgorithmReplayed": solver == "legacyAdamCpu",
        "iterationsPerformed": 0 if solver == "exact" else epochs,
        "laplacian": "unnormalizedCombinatorialDegreeMinusAdjacency",
        "smoothnessOn": ("absoluteVertexPositions" if smoothnessMode == "absolute" else
                         "weightedAlignedRestDifferentialCoordinates"),
        "staticConstraint": ("softJointDataMseAnchors" if staticConstraint == "soft"
                             else "hardFixedAlignedRestWithOldJointDataDenominator"),
        "fixedVertexCount": vertexCount - len(free), "freeVertexCount": len(free),
        "alignment": "unweightedUmeyamaFromValidClientSurfaceSamples",
        "coordinateFrame": "targetPointsFrameAndUnits", "guardApplied": False,
        "vertexOrderPreserved": True, "sampleCount": len(valid), "validSampleCount": sampleCount,
        "staticAnchorCount": staticCount, "vertexCount": vertexCount,
        "weights": {"data": dataWeight, "smooth": smoothWeight, "rest": restWeight},
        "mseDenominators": {"jointData": 3 * (sampleCount + staticCount), "smooth": 3 * vertexCount,
                            "rest": 3 * vertexCount},
        "effectiveCoefficients": coefficients, "objectiveBefore": before, "objectiveAfter": after,
        "normalEquationRelativeResidual": relativeResidual,
        "normalEquationGradientMaxAbs": float(2 * np.abs(residual).max()) if residual.size else 0.,
        "normalEquationResidualDomain": "allVertices" if staticConstraint == "soft" else "freeVerticesOnly",
        "seconds": perf_counter() - started,
        "limitations": [("Converged float64 objective solution, not the old 600-step float32 AdamUniform trajectory."
                          if solver == "exact" else "Old finite-step algorithm replayed on CPU; not bitwise original CUDA execution."),
                        "No triangle inversion, collision, anatomical correspondence or rig validation is performed.",
                        "Output follows target units/frame; caller must preserve and apply the inverse similarity when needed."],
    }
    if differential is not None:
        diagnostics["differentialRegularization"] = {
            "reference": "alignedRestTemplate", "weightsApplyTo": "LaplacianRows",
            "minimumWeight": float(differential.min()), "maximumWeight": float(differential.max()),
            "meanWeight": float(differential.mean()), "nonzeroRows": int(np.count_nonzero(differential)),
            "templateCurvaturePriorPresent": True,
            "equation": "L@V - differentialWeights[:,None]*(L@alignedRest)"}
        diagnostics["limitations"].append(
            "Differential regularization preserves template local curvature, not measured source anatomy; opt-in experiment only.")
    if priorEnabled:
        diagnostics["quadraticPrior"] = {
            "baseMethod": diagnostics["method"], "weight": prior["weight"],
            "rowCount": prior["operator"].shape[0], "columnCount": prior["operator"].shape[1],
            "nonzeroCoefficients": prior["operator"].nnz, "flatteningOrder": "vertexMajorXYZ",
            "coordinateFrame": "alreadyAlignedTargetPointsFrameAndUnits",
            "objective": "weight * sum((operator @ vertices.ravel() - targets)**2)",
            "rowNormalization": "callerSuppliedUnchanged", "alignmentChangedByPrior": False,
            "normalEquationDiagnosticsIncludePrior": True,
            "pivotCheck": "numericalSolvabilityGuardNotExactRankProof",
            "minimumPivotRelativeToSystemCoefficientLimit": 1e-12}
        diagnostics["method"] = "directClientSurfaceSamplesWithQuadraticPrior"
        diagnostics["limitations"].append(
            "Caller-supplied quadratic targets are priors, not measured anatomy; their target coordinate frame and units must be correct.")
    if replay is not None:
        diagnostics["algorithmReplay"] = replay
    return {"alignedRest": aligned, "transform": transform, "rawVertices": result, "diagnostics": diagnostics}
