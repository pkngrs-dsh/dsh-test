"""Fixed-camera eye fitting with a continuous compact spatial deformation field.

Inputs are frozen anatomical point/curve correspondences, not segmentations to
be rebound during fitting. The CPU solver changes only vertex positions.
Triangle quality checks do not prove absence of global self-intersection.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from .geometryRefinement import _meshEdges, _validateCameras, _validateMesh


def buildEyeDeformationBasis(vertices, controlPoints, *, supportRadius, controlCount=24):
    """Build compact C2 Wendland fields without graph-ring boundary clipping.

    Every field is ``(1-r)^4 (4r+1)`` for ``r<1`` and zero elsewhere. A single
    global normalization retains smoothness; there is no nearest-pinned-vertex
    envelope. Nearby vertices therefore receive nearby displacements even if
    triangulation contains exceptionally short edges. Radius is in mesh units.
    """
    vertices, points = np.asarray(vertices, float), np.asarray(controlPoints, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.isfinite(supportRadius) or supportRadius <= 0
            or not isinstance(controlCount, (int, np.integer)) or isinstance(controlCount, (bool, np.bool_))
            or controlCount < 1):
        raise ValueError("Expected finite Vx3/control-point arrays and positive radius/control count")
    points = np.unique(points, axis=0)
    if not len(points):
        return {"basis": np.zeros((len(vertices), 0)), "controlPoints": points,
                "activeVertexMask": np.zeros(len(vertices), bool), "normalization": 1.}
    chosen = [int(np.argmax(np.linalg.norm(points-points.mean(0), axis=1)))]
    distances = np.sum((points-points[chosen[0]])**2, axis=1)
    for _ in range(1, min(controlCount, len(points))):
        chosen.append(int(distances.argmax()))
        distances = np.minimum(distances, np.sum((points-points[chosen[-1]])**2, axis=1))
    centers = points[chosen]
    radius = np.linalg.norm(vertices[:, None]-centers[None], axis=-1)/supportRadius
    inside = radius < 1
    compact = np.where(inside, (1-np.minimum(radius, 1))**4*(4*np.minimum(radius, 1)+1), 0.)
    normalization = max(float(compact.sum(1).max(initial=0)), 1.)
    return {"basis": compact/normalization, "controlPoints": centers,
            "activeVertexMask": inside.any(1), "normalization": normalization}


def _preparePointConstraints(constraints, vertexCount, cameras):
    prepared = []
    for constraint in constraints:
        view = constraint["viewIndex"]
        indices = np.asarray(constraint["vertexIndices"])
        barycentric = np.asarray(constraint["barycentric"], float)
        target = np.asarray(constraint["targetPixels"], float)
        weights = np.asarray(constraint.get("weights", np.ones(len(indices))), float)
        if (not isinstance(view, (int, np.integer)) or isinstance(view, (bool, np.bool_))
                or not 0 <= view < len(cameras) or indices.ndim != 2 or not len(indices)
                or indices.shape[1] not in (1, 2, 3) or indices.dtype.kind not in "iu"
                or indices.min() < 0 or indices.max() >= vertexCount
                or barycentric.shape != indices.shape or not np.isfinite(barycentric).all()
                or np.any(barycentric < 0) or not np.allclose(barycentric.sum(1), 1, atol=1e-7)
                or target.shape != (len(indices), 2) or not np.isfinite(target).all()
                or weights.shape != (len(indices),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0 or "targetDistance" in constraint):
            raise ValueError("Eye constraints require valid fixed surface indices/barycentrics and targetPixels")
        tangentWeight = float(constraint.get("tangentWeight", 1.))
        if not np.isfinite(tangentWeight) or not 0 <= tangentWeight <= 1:
            raise ValueError("tangentWeight must be in [0, 1]")
        transform = np.tile(np.eye(2), (len(indices), 1, 1))
        if "targetTangents" in constraint:
            tangent = np.asarray(constraint["targetTangents"], float)
            if (tangent.shape != target.shape or not np.isfinite(tangent).all()
                    or np.any(np.linalg.norm(tangent, axis=1) < 1e-12)):
                raise ValueError("targetTangents must contain one finite nonzero 2D vector per point")
            tangent = tangent/np.linalg.norm(tangent, axis=1)[:, None]
            # Unit weight is exactly the ordinary XY objective and Jacobian.
            if tangentWeight != 1:
                transform[:, 0] = np.column_stack([-tangent[:, 1], tangent[:, 0]])
                transform[:, 1] = np.sqrt(tangentWeight)*tangent
        elif tangentWeight != 1:
            raise ValueError("Directional point weighting requires targetTangents")
        if "segmentPairs" in constraint:
            pairs = np.asarray(constraint["segmentPairs"])
        elif constraint.get("orderedCurve", False):
            sampleIndices = np.asarray(constraint.get("sampleIndices", np.arange(len(indices))))
            if sampleIndices.shape != (len(indices),) or sampleIndices.dtype.kind not in "iu":
                raise ValueError("sampleIndices must be integer M-vector")
            first = np.flatnonzero(np.diff(sampleIndices) == 1)
            pairs = np.column_stack([first, first+1])
        else:
            pairs = np.empty((0, 2), np.int64)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
                or (len(pairs) and (pairs.min() < 0 or pairs.max() >= len(indices)
                                   or np.any(pairs[:, 0] == pairs[:, 1])))):
            raise ValueError("segmentPairs must be valid distinct-point Ex2 local indices")
        if len(pairs):
            # A disabled point must stay disabled in the curve term as well;
            # averaging endpoint weights would otherwise reactivate it.
            pairs = pairs[(weights[pairs[:, 0]] > 0) & (weights[pairs[:, 1]] > 0)]
            targetVectors = target[pairs[:, 1]]-target[pairs[:, 0]]
            pairs = pairs[np.linalg.norm(targetVectors, axis=1) > 1e-8]
        prepared.append({"viewIndex": int(view), "name": str(constraint.get("name", "eyePoints")),
                         "vertexIndices": indices.copy(), "barycentric": barycentric.copy(),
                         "targetPixels": target.copy(), "weights": weights.copy(),
                         "transform": transform, "segmentPairs": pairs.copy()})
    return prepared


def _projectWithJacobian(points, camera):
    cameraPoints = points@camera["rotation"].T+camera["translation"]
    homogeneous = cameraPoints@camera["intrinsics"].T
    denominator = homogeneous[:, 2:]
    safeDenominator = np.maximum(denominator, 1e-12)
    pixels = homogeneous[:, :2]/safeDenominator
    derivative = ((camera["intrinsics"][:2][None]*safeDenominator[:, :, None]
                   - homogeneous[:, :2, None]*camera["intrinsics"][2])
                  / safeDenominator[:, :, None]**2)@camera["rotation"]
    return pixels, cameraPoints[:, 2], derivative


def refineEyeGeometry(vertices, faces, cameras, constraints, *, supportRadius=None,
                      controlCount=24, maxDisplacement=None, referenceViewIndex=0,
                      referenceWeight=4., iterations=100, pixelScale=4.,
                      displacementWeight=.002, strainWeight=.03, controlWeight=.0001,
                      curveWeight=.15, maxViewDegradation=.5, maxRelativeEdgeChange=.30,
                      minAreaRatio=.25, minNormalCosine=.1, minImprovementPixels=.001,
                      initialDamping=.001, maxBacktracking=14, maxTargetDistancePixels=40.,
                      deformationFrame="world3d", additionalResidual=None,
                      additionalSafety=None, deformationBasis=None,
                      captureSolverEvidence=False):
    """Fit ordered eyelid points/corners with a low-dimensional smooth field.

    Constraint format matches ``refineLocalGeometry``'s targetPixels variant.
    Optional targetTangents/tangentWeight only weight point residual directions.
    Optional segmentPairs (preferred) or orderedCurve/sampleIndices add a
    separate adjacent projected-vector residual, matching target direction AND
    length. These target curves are caller-supplied observations/proxies, not
    ground truth. Missing samples must not be bridged by segmentPairs.

    Damped Gauss-Newton and a safety backtracking search retain area, normal,
    relative-edge, displacement and per-view error gates. Rejected steps change
    damping rather than permanently shrinking a global learning rate. Cameras,
    topology, vertex order and UVs are fixed; visibility/correspondences are
    frozen. Self-intersection between separate triangles is NOT tested.
    ``referenceImagePlane`` limits displacements to the reference camera XY
    plane, preserving each vertex's camera depth (up to output rounding).
    Optional callbacks operate on read-only full vertex arrays. A residual
    callback returns normalized ``residual`` and a sparse vertex-coordinate
    ``jacobian`` when requested; its least-squares energy is added directly.
    A safety callback returns a boolean ``accepted`` field and is checked at
    every trial step and after output rounding. The initial mesh must satisfy it.
    ``deformationBasis`` optionally supplies a field constructed on these exact
    input vertices/faces. All geometric gates and the camera frame still apply.
    Evidence capture records coefficients and trial gates without changing the
    solve; trial vertices can be reconstructed from the saved basis and axes.
    """
    if type(captureSolverEvidence) is not bool:
        raise ValueError("captureSolverEvidence must be boolean")
    for callback in (additionalResidual, additionalSafety):
        if callback is not None and not callable(callback):
            raise ValueError("Additional solver hooks must be callable or None")
    originalDtype = np.asarray(vertices).dtype
    original, faces = _validateMesh(vertices, faces)
    cameras = _validateCameras(cameras)
    extent = float(np.ptp(original, axis=0).max())
    supportRadius = .12*extent if supportRadius is None else float(supportRadius)
    maxDisplacement = .025*extent if maxDisplacement is None else float(maxDisplacement)
    if (not isinstance(iterations, (int, np.integer)) or isinstance(iterations, (bool, np.bool_)) or iterations < 0
            or not isinstance(maxBacktracking, (int, np.integer)) or isinstance(maxBacktracking, (bool, np.bool_))
            or maxBacktracking < 1 or not isinstance(referenceViewIndex, (int, np.integer))
            or isinstance(referenceViewIndex, (bool, np.bool_)) or not 0 <= referenceViewIndex < len(cameras)):
        raise ValueError("Invalid eye fitting iteration/reference/backtracking settings")
    if deformationFrame not in ("world3d", "referenceImagePlane"):
        raise ValueError("deformationFrame must be world3d or referenceImagePlane")
    axes = (np.eye(3) if deformationFrame == "world3d"
            else cameras[referenceViewIndex]["rotation"].T[:, :2])
    degreesOfFreedom = axes.shape[1]
    scalars = [supportRadius, maxDisplacement, referenceWeight, pixelScale, displacementWeight,
               strainWeight, controlWeight, curveWeight, maxViewDegradation, maxRelativeEdgeChange,
               minAreaRatio, minNormalCosine, minImprovementPixels, initialDamping, maxTargetDistancePixels]
    if (not np.isfinite(scalars).all() or min(supportRadius, maxDisplacement, referenceWeight, pixelScale, initialDamping) <= 0
            or min(displacementWeight, strainWeight, controlWeight, curveWeight, maxViewDegradation, minImprovementPixels) < 0
            or maxTargetDistancePixels <= 0 or not 0 < maxRelativeEdgeChange < 1
            or not 0 < minAreaRatio <= 1 or not 0 <= minNormalCosine < 1):
        raise ValueError("Invalid eye fitting regularization/safety parameters")
    prepared = _preparePointConstraints(constraints, len(original), cameras)
    points = [(original[item["vertexIndices"]]*item["barycentric"][..., None]).sum(1) for item in prepared]
    if deformationBasis is None:
        field = buildEyeDeformationBasis(original, np.concatenate(points) if points else np.empty((0, 3)),
                                        supportRadius=supportRadius, controlCount=controlCount)
    else:
        if (not isinstance(controlCount, (int, np.integer)) or isinstance(controlCount, (bool, np.bool_))
                or not 1 <= controlCount <= 256):
            raise ValueError("controlCount must be an integer between one and 256")
        if not isinstance(deformationBasis, dict):
            raise ValueError("deformationBasis must be a validated field mapping")
        field = deepcopy(deformationBasis)
        required = {"basis", "controlPoints", "activeVertexMask", "normalization", "referenceVertices", "faces"}
        if not required.issubset(field):
            raise ValueError("deformationBasis is missing geometry and field identity")
        matrix, centers, mask = (np.asarray(field[key]) for key in ("basis", "controlPoints", "activeVertexMask"))
        if (not np.array_equal(field["referenceVertices"], original) or not np.array_equal(field["faces"], faces)
                or matrix.ndim != 2 or matrix.shape[0] != len(original) or matrix.shape[1] > controlCount
                or matrix.size > 10000000 or matrix.dtype.kind not in "fi"
                or not np.isfinite(matrix).all() or centers.shape != (matrix.shape[1], 3)
                or not np.isfinite(centers).all() or mask.dtype.kind != "b" or mask.shape != (len(original),)
                or not np.array_equal(mask, np.any(matrix != 0, axis=1))
                or not np.isscalar(field["normalization"]) or isinstance(field["normalization"], (bool, np.bool_))
                or not isinstance(field["normalization"], (float, int, np.floating, np.integer))
                or not np.isfinite(field["normalization"]) or field["normalization"] <= 0
                or np.max(np.sum(np.abs(matrix), axis=1), initial=0) > 1.+1e-10):
            raise ValueError("Invalid or stale deformation field; expected finite bounded basis on exact input mesh")
        field["basis"], field["controlPoints"], field["activeVertexMask"] = matrix.astype(float), centers.astype(float), mask.copy()
    basis, active = field["basis"], field["activeVertexMask"]
    parameterCount = basis.shape[1]*degreesOfFreedom
    hookCounts = {"residualEvaluations": 0, "safetyEvaluations": 0, "safetyRejections": 0}
    if additionalResidual is not None:
        from scipy import sparse
        vertexParameterJacobian = maxDisplacement*sparse.kron(basis, axes, format="csr")
    residualSize = None

    def readonly(candidate):
        # Copy isolation also prevents callbacks from altering solver state by
        # changing writeability on their own ndarray.
        frozen = np.array(candidate, dtype=float, copy=True)
        frozen.setflags(write=False)
        return frozen

    def extraSafety(candidate):
        hookCounts["safetyEvaluations"] += 1
        report = additionalSafety(readonly(candidate))
        if not isinstance(report, dict) or not isinstance(report.get("accepted"), (bool, np.bool_)):
            raise ValueError("additionalSafety must return an explicit boolean accepted field")
        if not report["accepted"]:
            hookCounts["safetyRejections"] += 1
        return bool(report["accepted"]), report
    edges = _meshEdges(faces)
    restEdgeVectors = original[edges[:, 1]]-original[edges[:, 0]]
    restLengths = np.linalg.norm(restEdgeVectors, axis=1)
    restCross = np.cross(original[faces[:, 1]]-original[faces[:, 0]], original[faces[:, 2]]-original[faces[:, 0]])
    restArea = np.linalg.norm(restCross, axis=1)
    localEdges = active[edges].any(1)
    observedViews = sorted({item["viewIndex"] for item in prepared})
    viewTotals = {view: sum(item["weights"].sum() for item in prepared if item["viewIndex"] == view)
                  for view in observedViews}
    viewMultipliers = {view: referenceWeight if view == referenceViewIndex else 1. for view in observedViews}
    normalizer = max(sum(viewMultipliers.values()), 1.)
    for item, point in zip(prepared, points):
        item["originalPoints"] = point
        item["basis"] = (basis[item["vertexIndices"]]*item["barycentric"][..., None]).sum(1)
        item["normalizedWeights"] = item["weights"]*viewMultipliers[item["viewIndex"]]/(viewTotals[item["viewIndex"]]*normalizer)
    regularizer = controlWeight/max(basis.shape[1], 1)*np.eye(parameterCount)
    if active.any():
        small = displacementWeight/max(int(active.sum()), 1)*(basis[active].T@basis[active])
        if localEdges.any():
            difference = ((basis[edges[localEdges, 1]]-basis[edges[localEdges, 0]])
                          * (maxDisplacement/restLengths[localEdges])[:, None])
            small += strainWeight/int(localEdges.sum())*(difference.T@difference)
        regularizer += np.kron(small, np.eye(degreesOfFreedom))

    def evaluate(coefficients, withJacobian=False, actualVertices=None):
        nonlocal residualSize
        flat = coefficients.ravel()
        objective = .5*float(flat@regularizer@flat)
        regularizationEnergy, additionalEnergy = objective, 0.
        normal = regularizer.copy() if withJacobian else None
        gradient = regularizer@flat if withJacobian else None
        if additionalResidual is not None:
            candidate = (original+maxDisplacement*((basis@coefficients)@axes.T)
                         if actualVertices is None else actualVertices)
            extra = additionalResidual(readonly(candidate), withJacobian)
            hookCounts["residualEvaluations"] += 1
            if not isinstance(extra, dict) or "residual" not in extra:
                raise ValueError("additionalResidual must return residual and optional sparse jacobian")
            residual = np.asarray(extra["residual"], dtype=float)
            if (residual.ndim != 1 or len(residual) > 100000 or not np.isfinite(residual).all()
                    or (residualSize is not None and len(residual) != residualSize)):
                raise ValueError("Additional residual must have a fixed finite bounded vector shape")
            residualSize = len(residual)
            energy = .5*float(residual@residual)
            if not np.isfinite(energy):
                raise ValueError("Additional residual energy must be finite")
            objective += energy
            additionalEnergy = energy
            if withJacobian:
                jacobian = extra.get("jacobian")
                if (not sparse.issparse(jacobian) or jacobian.shape != (len(residual), 3*len(original))
                        or jacobian.nnz > 10000000 or len(residual)*parameterCount > 10000000
                        or not np.isfinite(jacobian.data).all()):
                    raise ValueError("Additional jacobian must be finite sparse residualCount x 3*vertexCount")
                chained = (jacobian.tocsr()@vertexParameterJacobian).toarray()
                if not np.isfinite(chained).all():
                    raise ValueError("Additional coefficient jacobian must be finite")
                normal += chained.T@chained
                gradient += chained.T@residual
        meanError, valid = 0., True
        viewNumerators = {view: 0. for view in observedViews}
        groups = []
        def addResidual(residual, jacobian, sampleWeights, multiplier=1.):
            nonlocal objective, normal, gradient
            magnitude = np.linalg.norm(residual, axis=1)/pixelScale
            robust = np.where(magnitude <= 1, .5*magnitude**2, magnitude-.5)
            objective += multiplier*float(sampleWeights@robust)
            if withJacobian:
                irls = np.sqrt(multiplier*sampleWeights/np.maximum(magnitude, 1.))/pixelScale
                weighted = (residual*irls[:, None]).reshape(-1)
                matrix = (jacobian*irls[:, None, None]).reshape(-1, parameterCount)
                normal += matrix.T@matrix
                gradient += matrix.T@weighted
        for item in prepared:
            point = (item["originalPoints"]+maxDisplacement*((item["basis"]@coefficients)@axes.T)
                     if actualVertices is None else
                     (actualVertices[item["vertexIndices"]]*item["barycentric"][..., None]).sum(1))
            pixels, depth, derivative = _projectWithJacobian(point, cameras[item["viewIndex"]])
            valid &= bool(np.isfinite(pixels).all() and (depth > 1e-6).all())
            error = pixels-item["targetPixels"]
            rawNorm = np.linalg.norm(error, axis=1)
            weightedError = float(item["weights"]@rawNorm)
            viewNumerators[item["viewIndex"]] += weightedError
            meanError += float(item["normalizedWeights"]@rawNorm)
            derivative = derivative@axes
            jacobian = ((derivative[:, :, None, :]*item["basis"][:, None, :, None])*maxDisplacement).reshape(len(point), 2, parameterCount)
            directional = np.einsum("nij,nj->ni", item["transform"], error)
            directionalJacobian = np.einsum("nij,njk->nik", item["transform"], jacobian)
            addResidual(directional, directionalJacobian, item["normalizedWeights"])
            pairs = item["segmentPairs"]
            angles = np.empty(0)
            if len(pairs):
                first, second = pairs.T
                vectors = pixels[second]-pixels[first]
                targetVectors = item["targetPixels"][second]-item["targetPixels"][first]
                pairWeights = .5*(item["normalizedWeights"][first]+item["normalizedWeights"][second])
                if curveWeight:
                    addResidual(vectors-targetVectors, jacobian[second]-jacobian[first], pairWeights, curveWeight)
                lengths = np.linalg.norm(vectors, axis=1)*np.linalg.norm(targetVectors, axis=1)
                cosine = np.sum(vectors*targetVectors, axis=1)/np.maximum(lengths, 1e-15)
                angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            groups.append({"name": item["name"], "viewIndex": item["viewIndex"], "sampleCount": len(point),
                           "meanErrorPixels": weightedError/item["weights"].sum(),
                           "curveSegmentCount": len(pairs),
                           "meanCurveDirectionErrorDegrees": float(angles.mean()) if len(angles) else None})
        viewErrors = {view: viewNumerators[view]/viewTotals[view] for view in observedViews}
        evaluated = (objective, meanError, viewErrors, groups, valid, normal, gradient)
        if captureSolverEvidence:
            evaluated += ({"regularization": regularizationEnergy, "additionalResidual": additionalEnergy,
                           "photoAndCurve": objective-regularizationEnergy-additionalEnergy},)
        return evaluated

    def safety(candidate):
        cross = np.cross(candidate[faces[:, 1]]-candidate[faces[:, 0]], candidate[faces[:, 2]]-candidate[faces[:, 0]])
        area = np.linalg.norm(cross, axis=1)
        cosine = np.sum(cross*restCross, axis=1)/np.maximum(area*restArea, 1e-25)
        ratios = np.linalg.norm(candidate[edges[:, 1]]-candidate[edges[:, 0]], axis=1)/restLengths
        change = np.abs(ratios-1)
        maximumEdge = int(change.argmax())
        report = {"minimumAreaRatio": float((area/restArea).min()), "minimumNormalCosine": float(cosine.min()),
                  "maximumRelativeEdgeChange": float(change.max()),
                  "maximumDisplacement": float(np.linalg.norm(candidate-original, axis=1).max()),
                  "worstEdgeVertexIndices": edges[maximumEdge].tolist(), "worstEdgeRestLength": float(restLengths[maximumEdge]),
                  "selfIntersectionChecked": False}
        safe = (np.isfinite(candidate).all() and report["minimumAreaRatio"] >= minAreaRatio
                and report["minimumNormalCosine"] >= minNormalCosine
                and report["maximumRelativeEdgeChange"] <= maxRelativeEdgeChange
                and report["maximumDisplacement"] <= maxDisplacement*(1+1e-8))
        if captureSolverEvidence:
            report["geometricSafetyAccepted"] = bool(safe)
        if additionalSafety is not None:
            extraAccepted, extraReport = extraSafety(candidate)
            report["additionalSafety"] = extraReport
            safe = safe and extraAccepted
        return bool(safe), report

    if additionalSafety is not None and not extraSafety(original)[0]:
        raise ValueError("Initial mesh violates additionalSafety; no fit was attempted")
    coefficients = np.zeros((basis.shape[1], degreesOfFreedom))
    initial = evaluate(coefficients)
    if not initial[4]:
        raise ValueError("Initial eye constraints project behind a camera")
    rejectedTargetCount = sum(int(((np.linalg.norm(_projectWithJacobian(item["originalPoints"], cameras[item["viewIndex"]])[0]
                                                  -item["targetPixels"], axis=1) > maxTargetDistancePixels)
                                  & (item["weights"] > 0)).sum()) for item in prepared)
    bestCoefficients, bestObjective, bestIteration = coefficients.copy(), initial[0], 0
    history, damping = [], initialDamping
    trialEvidence = []
    rejections = {"meshSafety": 0, "projection": 0, "viewDegradation": 0, "objective": 0}
    completed = 0
    for iteration in range(1, iterations+1) if parameterCount and not rejectedTargetCount else ():
        completed = iteration
        current = evaluate(coefficients, True)
        diagonal = np.maximum(np.diag(current[5]), 1e-8)
        try:
            step = np.linalg.solve(current[5]+damping*np.diag(diagonal), -current[6]).reshape(-1, degreesOfFreedom)
        except np.linalg.LinAlgError:
            damping *= 10
            continue
        if not np.isfinite(step).all() or np.linalg.norm(step) < 1e-10:
            break
        accepted = False
        for backtrack in range(maxBacktracking):
            scale = .5**backtrack
            proposed = coefficients+scale*step
            candidate = original+maxDisplacement*((basis@proposed)@axes.T)
            safe, quality = safety(candidate)
            trial = None
            if captureSolverEvidence:
                trial = {"iteration": iteration, "backtrack": backtrack, "stepScale": scale,
                    "beforeCoefficients": coefficients.copy(), "trialCoefficients": proposed.copy(),
                    "coefficientStep": step.copy(), "dampingBeforeTrial": damping,
                    "currentObjective": current[0], "currentObjectiveComponents": deepcopy(current[7]),
                    "safety": deepcopy(quality), "safetyAccepted": bool(safe),
                    "evaluation": None, "decision": "notEvaluated"}
                trialEvidence.append(trial)
            if not safe:
                rejections["meshSafety"] += 1
                if trial is not None: trial["decision"] = "safetyRejected"
                continue
            evaluated = evaluate(proposed)
            if trial is not None:
                trial["evaluation"] = {"objective": evaluated[0], "meanErrorPixels": evaluated[1],
                    "viewErrors": deepcopy(evaluated[2]), "projectionValid": bool(evaluated[4]),
                    "objectiveComponents": deepcopy(evaluated[7])}
            if not evaluated[4]:
                rejections["projection"] += 1
                if trial is not None: trial["decision"] = "projectionRejected"
                continue
            if any(evaluated[2][view] > initial[2][view]+maxViewDegradation for view in observedViews):
                rejections["viewDegradation"] += 1
                if trial is not None: trial["decision"] = "viewDegradationRejected"
                continue
            if not np.isfinite(evaluated[0]) or evaluated[0] >= current[0]-1e-13:
                rejections["objective"] += 1
                if trial is not None: trial["decision"] = "objectiveRejected"
                continue
            coefficients = proposed
            if trial is not None: trial["decision"] = "acceptedStep"
            accepted = True
            damping = max(damping*.5, 1e-8)
            if evaluated[0] < bestObjective and evaluated[1] < initial[1]-minImprovementPixels:
                bestCoefficients, bestObjective, bestIteration = proposed.copy(), evaluated[0], iteration
            history.append({"iteration": iteration, "objective": evaluated[0], "meanErrorPixels": evaluated[1],
                            "viewErrors": evaluated[2], "stepScale": scale, "damping": damping,
                            "maximumRelativeEdgeChange": quality["maximumRelativeEdgeChange"], "accepted": True})
            break
        if not accepted:
            damping *= 10
            history.append({"iteration": iteration, "objective": current[0], "meanErrorPixels": current[1],
                            "damping": damping, "accepted": False})
            if damping > 1e10:
                break
    resultDtype = originalDtype if np.issubdtype(originalDtype, np.floating) else np.float64
    result = (original+maxDisplacement*((basis@bestCoefficients)@axes.T)).astype(resultDtype)
    result[~active] = original[~active].astype(resultDtype)
    safe, quality = safety(result.astype(float))
    castBacktracks, castScale = 0, 1.
    unroundedBest = bestCoefficients.copy()
    castTrials = []
    def recordCast():
        if captureSolverEvidence:
            castTrials.append(dict(scale=castScale, coefficients=bestCoefficients.copy(),
                safetyAccepted=bool(safe), safety=deepcopy(quality), outputDtype=str(result.dtype)))
    recordCast()
    while not safe and castBacktracks < 16:
        castScale = max(0., 1.-1e-4*2**castBacktracks)
        castBacktracks += 1
        bestCoefficients = unroundedBest*castScale
        result = (original+maxDisplacement*((basis@bestCoefficients)@axes.T)).astype(resultDtype)
        result[~active] = original[~active].astype(resultDtype)
        safe, quality = safety(result.astype(float))
        recordCast()
    final = evaluate(bestCoefficients, actualVertices=result.astype(float))
    castRejected = not safe or castScale == 0 or (bestIteration > 0 and (final[1] >= initial[1]-minImprovementPixels
                    or any(final[2][view] > initial[2][view]+maxViewDegradation for view in observedViews)))
    if castRejected or castScale == 0:
        result = original.astype(resultDtype)
        bestCoefficients[:] = 0
        bestObjective, bestIteration = initial[0], 0
        _, quality = safety(result.astype(float))
        final = evaluate(bestCoefficients, actualVertices=result.astype(float))
    else:
        bestObjective = final[0]
    output = {"vertices": result, "activeVertexMask": active, "controlPoints": field["controlPoints"],
            "controlOffsets": maxDisplacement*(bestCoefficients@axes.T), "history": history,
            "diagnostics": {"status": "accepted" if bestIteration else "unchanged", "bestIteration": bestIteration,
                "iterationsRequested": int(iterations), "iterationsCompleted": completed,
                "initialMeanErrorPixels": initial[1], "finalMeanErrorPixels": final[1],
                "initialViewErrors": initial[2], "finalViewErrors": final[2],
                "initialGroups": initial[3], "finalGroups": final[3], "activeVertexCount": int(active.sum()),
                "constraintCount": len(prepared), "supportRadius": supportRadius, "controlCount": basis.shape[1],
                "parameterization": "globally normalized compact C2 Wendland displacement fields; no graph-ring clipping",
                "deformationFrame": deformationFrame, "degreesOfFreedomPerControl": degreesOfFreedom,
                "maximumReferenceDepthChange": float(np.abs((result.astype(float)-original)
                    @cameras[referenceViewIndex]["rotation"][2]).max()),
                "maxDisplacement": maxDisplacement, "maxRelativeEdgeChange": maxRelativeEdgeChange,
                "minAreaRatio": minAreaRatio, "minNormalCosine": minNormalCosine,
                "maxViewDegradation": maxViewDegradation, "referenceViewIndex": int(referenceViewIndex),
                "referenceWeight": referenceWeight, "pixelScale": pixelScale,
                "regularization": {"displacementWeight": displacementWeight, "strainWeight": strainWeight,
                                   "controlWeight": controlWeight, "curveWeight": curveWeight},
                "minimumRestEdgeLength": float(restLengths.min()), "medianRestEdgeLength": float(np.median(restLengths)),
                "curveWeight": curveWeight, "curveLoss": "adjacent projected difference vectors versus target difference vectors",
                "bestRegularizedObjective": bestObjective, "safety": quality, "rejections": rejections,
                "castSafetyRejected": bool(castRejected), "castSafetyBacktrackingSteps": castBacktracks,
                "castSafetyScale": castScale, "device": "cpu", "camerasFrozen": True,
                "inputTargetGate": {"maximumInitialDistancePixels": maxTargetDistancePixels,
                                    "rejectedTargetCount": rejectedTargetCount,
                                    "entireFitRejected": bool(rejectedTargetCount)},
                "visibilityFrozen": True, "uvAndVertexOrderPreserved": True,
                "topologySha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                "limits": ["Point and curve targets are frozen input correspondences or proxies, not ground truth.",
                           "Local area/normal/edge gates do not certify absence of self-intersection.",
                           "No global or local nonadjacent triangle intersection test is performed.",
                           "Occlusion and image support must be re-evaluated after geometry changes."]}}
    if additionalResidual is not None or additionalSafety is not None:
        output["diagnostics"]["additionalHooks"] = dict(hookCounts,
            residualCount=residualSize, residualEnabled=additionalResidual is not None,
            safetyEnabled=additionalSafety is not None)
    if deformationBasis is not None:
        output["diagnostics"]["parameterization"] = "fixed-boundary screened biharmonic displacement basis"
        output["diagnostics"]["deformationBasis"] = deepcopy(field.get("diagnostics", {}))
    if captureSolverEvidence:
        output["solverEvidence"] = dict(referenceVertices=original.copy(), faces=faces.copy(),
            basis=basis.copy(), controlPoints=field["controlPoints"].copy(), activeVertexMask=active.copy(),
            normalization=field["normalization"], axes=axes.copy(), maxDisplacement=maxDisplacement,
            regularizer=regularizer.copy(), preparedConstraints=deepcopy(prepared),
            initialObjective=initial[0], initialObjectiveComponents=deepcopy(initial[7]),
            trials=trialEvidence, finalCoefficients=bestCoefficients.copy(), outputVertices=result.copy(),
            unroundedBestCoefficients=unroundedBest, castTrials=castTrials,
            castRejected=bool(castRejected), finalEvaluation=dict(objective=final[0], meanErrorPixels=final[1],
                viewErrors=deepcopy(final[2]), projectionValid=bool(final[4]), objectiveComponents=deepcopy(final[7])),
            fieldConstruction=deepcopy(field.get("evidence")), outputDtype=str(result.dtype),
            scope="Trial vertices = reference + maxDisplacement * (basis @ coefficients) @ axes.T; no collision or raster gate is implied by a GN trial.")
    return output
