"""Low-dimensional, ray-constrained geometry fitting to external surface normals.

Targets must already be in the mesh's canonical coordinate frame. This module
does not infer normal orientation, register cameras or establish visibility.
"""
from __future__ import annotations

import numpy as np

from .surfaceFairing import _buildLaplacian, _prepareInputs, _scalar


def _vertexNormals(vertices, faces):
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)


def _compactBasis(points, weights, controlCount, radius):
    activeIds = np.flatnonzero(weights > 0)
    if not len(activeIds):
        return np.zeros((len(points), 0)), np.empty(0, dtype=np.int64)
    activePoints = points[activeIds]
    selected = [int(np.argmax(weights[activeIds]))]
    distances = np.sum((activePoints - activePoints[selected[0]]) ** 2, axis=1)
    for _ in range(1, min(controlCount, len(activeIds))):
        index = int(np.argmax(distances))
        if distances[index] <= 1e-24:
            break
        selected.append(index)
        distances = np.minimum(distances, np.sum((activePoints - activePoints[index]) ** 2, axis=1))
    controlIds = activeIds[np.asarray(selected)]
    distance = np.linalg.norm(points[:, None] - points[controlIds][None], axis=2) / radius
    clipped = np.clip(distance, 0, 1)
    basis = (1 - clipped) ** 4 * (4 * clipped + 1)
    totals = basis.sum(1, keepdims=True)
    basis = np.divide(basis, totals, out=np.zeros_like(basis), where=totals > 1e-14)
    basis *= weights[:, None]
    return basis, controlIds


def _targets(values, weights, count, name, *, vector=False):
    expectedShape = (count, 3) if vector else (count,)
    if weights is None:
        weights = np.zeros(count)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name}Weights must be finite nonnegative V values")
    if values is None:
        if weights.any():
            raise ValueError(f"Positive {name}Weights require targets")
        values = np.zeros(expectedShape)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != expectedShape or not np.isfinite(values[weights > 0]).all():
        raise ValueError(f"{name}Targets have invalid shape or nonfinite trusted values")
    result = np.zeros(expectedShape)
    result[weights > 0] = values[weights > 0]
    if vector:
        lengths = np.linalg.norm(result, axis=1)
        if np.any(lengths[weights > 0] < 1e-8):
            raise ValueError("Trusted normalTargets cannot be zero vectors")
        result = np.divide(result, lengths[:, None], out=np.zeros_like(result), where=lengths[:, None] > 0)
    return result, weights.copy()


def refineBrowGeometry(vertices, faces, directions, regionWeights, normalTargets, normalWeights, *,
                       depthTargets=None, depthWeights=None, normalLossWeight=1., depthLossWeight=1.,
                       depthScaleFraction=.01, controlCount=32, rbfRadiusFraction=.15,
                       iterations=160, learningRate=.05, displacementWeight=.01, strainWeight=.05,
                       boundaryWeight=.02, maxDisplacementFraction=.015, maxRelativeEdgeChange=.30,
                       minAreaRatio=.35, minNormalCosine=.5, maxBacktracking=24):
    """Move actual mesh vertices along fixed unit rays, using CPU float64 Adam.

    ``normalTargets`` (Vx3) are external, consistently oriented canonical
    normals; ``normalWeights`` (V) contain nonnegative confidence. Zero-weight
    targets may contain NaNs and are ignored. ``depthTargets`` (V) are signed
    distances FROM INITIAL VERTICES ALONG ``directions``, in canonical mesh
    units, not absolute camera Z. Depth is optional and needs explicit weights.

    A normalized compact Wendland-C2 RBF field with fixed control locations is
    multiplied by regionWeights. A zero region weight is a hard positional pin.
    Coefficients are bounded with tanh; no vertex is independently clipped.
    The objective fits area-weighted mesh vertex normals using 1-cosine, and
    signed depth residuals scaled by extent*depthScaleFraction, with normalized
    displacement, relative edge strain and pinned-boundary strain penalties.
    Final acceptance backtracks one multiplier for the entire optimized field
    and tests the actual dtype-cast output against the same local safety gates
    as surfaceFairing. Rays preserve their camera projection up to rounding;
    positive camera depth/visibility and self-intersections require caller tests.
    """
    original, faces, directions, region, extent, normalized, resultDtype = _prepareInputs(
        vertices, faces, directions, regionWeights)
    for name, value in (("iterations", iterations), ("controlCount", controlCount),
                        ("maxBacktracking", maxBacktracking)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (1 if name == "controlCount" else 0):
            raise ValueError(f"{name} must be an appropriate nonnegative integer")
    scalarOptions = {"normalLossWeight": normalLossWeight, "depthLossWeight": depthLossWeight,
                     "displacementWeight": displacementWeight, "strainWeight": strainWeight,
                     "boundaryWeight": boundaryWeight, "maxDisplacementFraction": maxDisplacementFraction}
    scalarOptions = {name: _scalar(value, name) for name, value in scalarOptions.items()}
    normalLossWeight, depthLossWeight = scalarOptions["normalLossWeight"], scalarOptions["depthLossWeight"]
    displacementWeight, strainWeight = scalarOptions["displacementWeight"], scalarOptions["strainWeight"]
    boundaryWeight, maxDisplacementFraction = scalarOptions["boundaryWeight"], scalarOptions["maxDisplacementFraction"]
    depthScaleFraction = _scalar(depthScaleFraction, "depthScaleFraction", lowerInclusive=False)
    rbfRadiusFraction = _scalar(rbfRadiusFraction, "rbfRadiusFraction", lowerInclusive=False)
    learningRate = _scalar(learningRate, "learningRate", lowerInclusive=False)
    maxRelativeEdgeChange = _scalar(maxRelativeEdgeChange, "maxRelativeEdgeChange", upper=1., lowerInclusive=False)
    if maxRelativeEdgeChange >= 1:
        raise ValueError("maxRelativeEdgeChange must be less than one")
    minAreaRatio = _scalar(minAreaRatio, "minAreaRatio", upper=1., lowerInclusive=False)
    minNormalCosine = _scalar(minNormalCosine, "minNormalCosine", upper=1., lowerInclusive=False)
    targets, confidence = _targets(normalTargets, normalWeights, len(original), "normal", vector=True)
    targetDepth, depthConfidence = _targets(depthTargets, depthWeights, len(original), "depth")
    active = region > 0
    _, edges = _buildLaplacian(len(original), faces)
    affected = region.copy()
    np.maximum.at(affected, edges[:, 0], region[edges[:, 1]])
    np.maximum.at(affected, edges[:, 1], region[edges[:, 0]])
    confidence *= affected
    confidence *= np.linalg.norm(_vertexNormals(normalized, faces), axis=1) > 0
    depthConfidence *= active
    localEdges = active[edges].any(1)
    boundaryEdges = active[edges].any(1) & ~active[edges].all(1)
    initialEdges = np.linalg.norm(normalized[edges[:, 1]] - normalized[edges[:, 0]], axis=1)
    initialCross = np.cross(normalized[faces[:, 1]] - normalized[faces[:, 0]],
                            normalized[faces[:, 2]] - normalized[faces[:, 0]])
    initialAreas = np.linalg.norm(initialCross, axis=1)
    if np.any(initialEdges <= 0):
        raise ValueError("Initial mesh contains zero-length edges")
    basis, controlIds = _compactBasis(normalized, region, controlCount, rbfRadiusFraction)
    history = []

    def metrics(candidate):
        delta = candidate - original
        signed = np.sum(delta * directions, axis=1)
        normals = _vertexNormals((candidate - original[0]) / extent, faces)
        angles = np.degrees(np.arccos(np.clip(np.sum(normals * targets, axis=1), -1, 1)))
        normalMean = float(np.sum(confidence * angles) / confidence.sum()) if confidence.sum() else None
        depthRmse = float(np.sqrt(np.sum(depthConfidence * (signed - targetDepth) ** 2) / depthConfidence.sum())) if depthConfidence.sum() else None
        return {"normalMeanDegrees": normalMean,
                "normalP95Degrees": float(np.percentile(angles[confidence > 0], 95)) if confidence.any() else None,
                "depthRmseCanonical": depthRmse,
                "maximumOrthogonalDisplacement": float(np.linalg.norm(delta - signed[:, None] * directions, axis=1).max())}

    def safety(candidate):
        if np.array_equal(candidate, original):
            return {"maximumRelativeEdgeChange": 0., "minimumAreaRatio": 1., "maximumAreaRatio": 1.,
                    "minimumNormalCosine": 1., "invertedOrDegenerateFaceCount": 0,
                    "maximumDisplacement": 0., "maximumDisplacementFraction": 0.}, []
        points = (candidate - original[0]) / extent
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
        areas = np.linalg.norm(cross, axis=1)
        cosine = np.divide(np.sum(cross * initialCross, axis=1), areas * initialAreas,
                           out=np.zeros(len(faces)), where=areas > 0)
        maximumShift = float(np.linalg.norm(candidate - original, axis=1).max())
        report = {"maximumRelativeEdgeChange": float(np.abs(lengths / initialEdges - 1).max()),
                  "minimumAreaRatio": float((areas / initialAreas).min()),
                  "maximumAreaRatio": float((areas / initialAreas).max()),
                  "minimumNormalCosine": float(np.clip(cosine, -1, 1).min()),
                  "invertedOrDegenerateFaceCount": int(np.count_nonzero(cosine <= 0)),
                  "maximumDisplacement": maximumShift, "maximumDisplacementFraction": maximumShift / extent}
        failures = []
        for failed, name in ((report["maximumRelativeEdgeChange"] > maxRelativeEdgeChange, "relativeEdgeChange"),
                             (report["minimumAreaRatio"] < minAreaRatio, "areaRatio"),
                             (report["minimumNormalCosine"] < minNormalCosine, "normalCosine"),
                             (report["maximumDisplacementFraction"] > maxDisplacementFraction, "displacementLimit"),
                             (not all(np.isfinite(value) for value in report.values()), "nonfiniteGeometry")):
            if failed:
                failures.append(name)
        return report, failures

    diagnostics = {"status": "unchanged", "reason": "notRun", "normalizationExtent": extent,
                   "device": "cpu", "optimizationDtype": "float64",
                   "vertexCount": len(original), "faceCount": len(faces), "activeVertexCount": int(active.sum()),
                   "normalTargetCount": int(np.count_nonzero(confidence)), "depthTargetCount": int(np.count_nonzero(depthConfidence)),
                   "controlCount": len(controlIds), "controlVertexIds": controlIds.tolist(),
                   "unsupportedActiveVertexCount": int(np.count_nonzero(active & (basis.sum(1) == 0))),
                   "rbfRadiusFraction": rbfRadiusFraction, "iterationsRequested": int(iterations),
                   "iterationsCompleted": 0, "bestIteration": 0, "acceptedScale": 0., "candidateChecks": 0,
                   "initialEnergy": None, "rawEnergy": None, "finalEnergy": None,
                   "rejectionCounts": {}, "initialMetrics": metrics(original), "depthUnits": "signed canonical distance along original unit directions",
                   "safetyLimits": {"maxDisplacementFraction": maxDisplacementFraction,
                                    "maxRelativeEdgeChange": maxRelativeEdgeChange, "minAreaRatio": minAreaRatio,
                                    "minNormalCosine": minNormalCosine},
                   "objectiveWeights": {**scalarOptions, "depthScaleFraction": depthScaleFraction},
                   "topologyUnchanged": True, "selfIntersectionChecked": False,
                   "backtracking": "one multiplier for complete displacement field; dtype-cast output rechecked",
                   "limits": ["Normal targets are external predictions, not measured anatomical ground truth.",
                              "No camera-depth, changing visibility, global self-intersection or held-out-view test is performed."]}

    def finish(result, reason, scale=0.):
        result = np.asarray(result, dtype=resultDtype).copy()
        result[~active] = np.asarray(vertices)[~active]
        delta = result.astype(np.float64) - original
        report, failures = safety(result.astype(np.float64))
        if failures:
            raise RuntimeError("Returned brow geometry violates its safety contract")
        diagnostics.update(status="accepted" if scale else "unchanged", reason=reason, acceptedScale=float(scale),
                           safety=report, finalMetrics=metrics(result.astype(np.float64)),
                           zeroWeightVerticesExactlyUnchanged=bool(np.array_equal(result[~active], np.asarray(vertices)[~active])))
        return {"vertices": result, "displacement": delta, "scalarDisplacement": np.sum(delta * directions, axis=1),
                "diagnostics": diagnostics, "history": history}

    if not active.any():
        return finish(original, "noActiveRegion")
    if not ((confidence.any() and normalLossWeight > 0) or (depthConfidence.any() and depthLossWeight > 0)):
        return finish(original, "noReliableTargets")
    if not iterations or not maxDisplacementFraction:
        return finish(original, "zeroIterations" if not iterations else "zeroDisplacementLimit")
    import torch

    tensor = lambda value, dtype=torch.float64: torch.as_tensor(value, dtype=dtype, device="cpu")
    base, ray, rbf = tensor(normalized), tensor(directions), tensor(basis)
    triangles, edgeIds = tensor(faces, torch.int64), tensor(edges, torch.int64)
    normalsTarget, normalWeight = tensor(targets), tensor(confidence)
    depthTarget, depthWeight = tensor(targetDepth / extent), tensor(depthConfidence)
    restLengths = tensor(initialEdges)
    local, boundary = tensor(localEdges, torch.bool), tensor(boundaryEdges, torch.bool)
    activeTensor = tensor(active, torch.bool)

    def objective(candidate):
        cross = torch.cross(candidate[triangles[:, 1]] - candidate[triangles[:, 0]],
                            candidate[triangles[:, 2]] - candidate[triangles[:, 0]], dim=1)
        normals = torch.zeros_like(candidate)
        for corner in range(3):
            normals.index_add_(0, triangles[:, corner], cross)
        normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-15)
        normalLoss = (normalWeight * (1 - (normals * normalsTarget).sum(1).clamp(-1, 1))).sum() / normalWeight.sum().clamp_min(1e-30)
        delta = candidate - base
        signed = (delta * ray).sum(1)
        depthLoss = (depthWeight * ((signed - depthTarget) / depthScaleFraction).square()).sum() / depthWeight.sum().clamp_min(1e-30)
        displacement = delta[activeTensor].square().sum(1).mean()
        lengths = torch.linalg.vector_norm(candidate[edgeIds[:, 1]] - candidate[edgeIds[:, 0]], dim=1)
        strain = (lengths[local] / restLengths[local] - 1).square().mean() if localEdges.any() else delta.sum() * 0
        fieldGradient = (delta[edgeIds[:, 1]] - delta[edgeIds[:, 0]]) / restLengths[:, None]
        boundaryLoss = fieldGradient[boundary].square().sum(1).mean() if boundaryEdges.any() else delta.sum() * 0
        total = (normalLossWeight * normalLoss + depthLossWeight * depthLoss + displacementWeight * displacement
                 + strainWeight * strain + boundaryWeight * boundaryLoss)
        return total, {"normal": normalLoss, "depth": depthLoss, "displacement": displacement,
                       "strain": strain, "boundary": boundaryLoss, "total": total}

    def recordedEnergy(candidate):
        with torch.no_grad():
            _, values = objective(tensor((candidate - original[0]) / extent))
        return {name: float(value) for name, value in values.items()}

    parameters = torch.zeros(len(controlIds), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=learningRate)
    initialEnergy = recordedEnergy(original)
    bestLoss, bestField = initialEnergy["total"], np.zeros(len(original))
    diagnostics["initialEnergy"] = initialEnergy
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        field = maxDisplacementFraction * (rbf @ torch.tanh(parameters))
        loss, _ = objective(base + field[:, None] * ray)
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
            break
        optimizer.step()
        with torch.no_grad():
            field = maxDisplacementFraction * (rbf @ torch.tanh(parameters))
            loss, _ = objective(base + field[:, None] * ray)
            value = float(loss)
            if np.isfinite(value) and value < bestLoss:
                bestLoss, bestField = value, field.numpy().copy()
                diagnostics["bestIteration"] = iteration
            diagnostics["iterationsCompleted"] = iteration
            if iteration == 1 or iteration % 20 == 0 or iteration == iterations:
                history.append({"iteration": iteration, "objective": value, "bestObjective": bestLoss})
    rawDelta = extent * bestField[:, None] * directions
    diagnostics["rawMaximumDisplacementFraction"] = float(np.linalg.norm(rawDelta, axis=1).max() / extent)
    diagnostics["rawEnergy"] = recordedEnergy(original + rawDelta)
    diagnostics["rawMetrics"] = metrics(original + rawDelta)
    if not np.any(rawDelta):
        diagnostics["finalEnergy"] = initialEnergy.copy()
        return finish(original, "noImprovingOptimizationStep")
    improvementFloor = 1e-12 * max(initialEnergy["total"], 1e-30)
    for backtrack in range(maxBacktracking + 1):
        scale = .5 ** backtrack
        candidate = (original + scale * rawDelta).astype(resultDtype)
        candidate[~active] = np.asarray(vertices)[~active]
        checked = candidate.astype(np.float64)
        _, failures = safety(checked)
        energy = recordedEnergy(checked)
        diagnostics["candidateChecks"] += 1
        if not np.isfinite(energy["total"]) or initialEnergy["total"] - energy["total"] <= improvementFloor:
            failures.append("noObjectiveImprovement")
        if np.array_equal(candidate, np.asarray(vertices)):
            failures.append("noRepresentableChange")
        if not failures:
            diagnostics["finalEnergy"] = energy
            return finish(candidate, "acceptedFullStep" if backtrack == 0 else "acceptedBacktrackedStep", scale)
        for reason in failures:
            diagnostics["rejectionCounts"][reason] = diagnostics["rejectionCounts"].get(reason, 0) + 1
    diagnostics["finalEnergy"] = initialEnergy.copy()
    return finish(original, "noSafeImprovingStep")
