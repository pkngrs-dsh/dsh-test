"""Local front-photo landmark correction of an already reconstructed mesh.

The prior is the input reconstructed shape, never a separate template shape.
Camera, topology, semantic bindings and UVs stay fixed. This is a conservative
adapter around the shared compact-field solver; it does not detect triangle
self-intersections and does not certify an output for production use.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .eyeRefinement import (
    _preparePointConstraints,
    _projectWithJacobian,
    buildEyeDeformationBasis,
    refineEyeGeometry,
)
from .geometryRefinement import _meshEdges, _validateCameras, _validateMesh


@lru_cache(maxsize=8)
def _cachedFrontEdges(faceShape, faceBytes):
    """Cache only immutable topology, not mutable caller arrays or candidates."""
    faces = np.frombuffer(faceBytes, dtype=np.int64).reshape(faceShape)
    edges = _meshEdges(faces)
    edges.setflags(write=False)
    return edges


def checkFrontGeometry(reference, candidate, faces, *, fixedVertexIds=(), camera,
                       maxDisplacement, maxRelativeEdgeChange=.25,
                       minAreaRatio=.4, minNormalCosine=.3, depthTolerance=None):
    """Recheck a complete candidate against the ORIGINAL pre-correction mesh.

    Use after every collision-driven interpolation and every local feature
    stage. Safe endpoints alone do not prove their intermediate mesh is safe,
    and stage-relative limits alone do not bound cumulative deformation.
    ``reference`` must therefore stay fixed across stages. Fixed vertices are
    compared exactly in this common canonical frame. Depth tolerance defaults
    to an output-dtype rounding allowance; callers may request a stricter
    explicit tolerance. Triangle intersections are deliberately not checked.

    Returns ``{"accepted": bool, "metrics": dict, "reasons": list[str]}``.
    Invalid inputs raise ValueError; a finite-shape candidate with NaNs/Infs is
    rejected with a report so an interpolation search can continue safely.
    """
    original, checkedFaces = _validateMesh(reference, faces)
    candidateInput = np.asarray(candidate)
    proposed = np.asarray(candidate, dtype=np.float64)
    if proposed.shape != original.shape:
        raise ValueError("candidate must have the same Vx3 shape as reference")
    checkedCamera = _validateCameras([camera])[0]
    rawFixed = np.asarray(fixedVertexIds)
    if rawFixed.size == 0:
        fixed = np.empty(0, np.int64)
    elif (rawFixed.ndim != 1 or rawFixed.dtype.kind not in "iu"
          or rawFixed.min() < 0 or rawFixed.max() >= len(original)):
        raise ValueError("fixedVertexIds must be a valid integer vertex-index vector")
    else:
        fixed = np.unique(rawFixed.astype(np.int64))
    if (not np.isfinite([maxDisplacement, maxRelativeEdgeChange,
                         minAreaRatio, minNormalCosine]).all()
            or maxDisplacement <= 0 or not 0 < maxRelativeEdgeChange < 1
            or not 0 < minAreaRatio <= 1 or not 0 <= minNormalCosine < 1):
        raise ValueError("Invalid cumulative front geometry safety limits")
    extent = float(np.ptp(original, axis=0).max())
    if depthTolerance is None:
        epsilon = (np.finfo(candidateInput.dtype).eps
                   if np.issubdtype(candidateInput.dtype, np.floating)
                   else np.finfo(np.float64).eps)
        coordinateScale = max(float(np.abs(original).max()), extent, 1e-12)
        depthTolerance = max(extent*1e-10, 16.*epsilon*coordinateScale)
    depthTolerance = float(depthTolerance)
    if not np.isfinite(depthTolerance) or depthTolerance < 0:
        raise ValueError("depthTolerance must be finite and nonnegative")
    metrics = {
        "reference": "original input reconstructed geometry",
        "vertexCount": len(original), "faceCount": len(checkedFaces),
        "fixedVertexCount": len(fixed),
        "fixedVerticesExactlyUnchanged": bool(np.array_equal(proposed[fixed], original[fixed])),
        "nonFiniteCoordinateCount": int((~np.isfinite(proposed)).sum()),
        "maximumDisplacement": None, "maximumRelativeEdgeChange": None,
        "minimumAreaRatio": None, "minimumNormalCosine": None,
        "maximumReferenceDepthChange": None, "maximumFixedVertexDisplacement": None,
        "limits": {"maxDisplacement": float(maxDisplacement),
                   "maxRelativeEdgeChange": float(maxRelativeEdgeChange),
                   "minAreaRatio": float(minAreaRatio),
                   "minNormalCosine": float(minNormalCosine),
                   "depthTolerance": depthTolerance},
        "selfIntersectionChecked": False,
    }
    if metrics["nonFiniteCoordinateCount"]:
        return {"accepted": False, "metrics": metrics, "reasons": ["nonFiniteCandidate"]}
    edges = _cachedFrontEdges(checkedFaces.shape, checkedFaces.tobytes())
    originalEdges = original[edges[:, 1]]-original[edges[:, 0]]
    proposedEdges = proposed[edges[:, 1]]-proposed[edges[:, 0]]
    edgeLengths = np.linalg.norm(originalEdges, axis=1)
    change = np.abs(np.linalg.norm(proposedEdges, axis=1)/edgeLengths-1.)
    originalCross = np.cross(original[checkedFaces[:, 1]]-original[checkedFaces[:, 0]],
                             original[checkedFaces[:, 2]]-original[checkedFaces[:, 0]])
    proposedCross = np.cross(proposed[checkedFaces[:, 1]]-proposed[checkedFaces[:, 0]],
                             proposed[checkedFaces[:, 2]]-proposed[checkedFaces[:, 0]])
    originalArea = np.linalg.norm(originalCross, axis=1)
    proposedArea = np.linalg.norm(proposedCross, axis=1)
    areaRatio = proposedArea/originalArea
    cosine = (np.sum(originalCross*proposedCross, axis=1)
              /np.maximum(originalArea*proposedArea, 1e-25))
    delta = proposed-original
    displacement = np.linalg.norm(delta, axis=1)
    depthChange = np.abs(delta@checkedCamera["rotation"][2])
    worstEdge = int(change.argmax())
    metrics.update({
        "maximumDisplacement": float(displacement.max()),
        "maximumDisplacementVertexIndex": int(displacement.argmax()),
        "maximumRelativeEdgeChange": float(change.max()),
        "worstEdgeVertexIndices": edges[worstEdge].tolist(),
        "worstEdgeReferenceLength": float(edgeLengths[worstEdge]),
        "minimumAreaRatio": float(areaRatio.min()),
        "minimumAreaRatioFaceIndex": int(areaRatio.argmin()),
        "minimumNormalCosine": float(cosine.min()),
        "minimumNormalCosineFaceIndex": int(cosine.argmin()),
        "maximumReferenceDepthChange": float(depthChange.max()),
        "maximumFixedVertexDisplacement": float(displacement[fixed].max(initial=0)),
    })
    reasons = []
    for rejected, reason in (
        (metrics["maximumDisplacement"] > maxDisplacement*(1.+1e-8), "maximumDisplacementExceeded"),
        (metrics["maximumRelativeEdgeChange"] > maxRelativeEdgeChange, "relativeEdgeChangeExceeded"),
        (metrics["minimumAreaRatio"] < minAreaRatio, "triangleAreaTooSmall"),
        (metrics["minimumNormalCosine"] < minNormalCosine, "triangleNormalChangedTooFar"),
        (metrics["maximumReferenceDepthChange"] > depthTolerance, "referenceDepthChanged"),
        (not metrics["fixedVerticesExactlyUnchanged"], "fixedVertexChanged"),
    ):
        if rejected:
            reasons.append(reason)
    return {"accepted": not reasons, "metrics": metrics, "reasons": reasons}


def refineFrontLandmarks(vertices, faces, camera, constraints, *, fixedVertexIds=(),
                         supportRadius=None, controlCount=24, maxDisplacement=None,
                         minConfidence=.5, minWeight=.05, maxTargetDistancePixels=40.,
                         deformationMode="compact", smoothingRadiusFraction=.1,
                         **solverOptions):
    """Fit reliable front landmarks by moving vertices only in camera XY.

    ``camera`` uses the OpenCV ``intrinsics``, ``rotation``, ``translation``
    convention. Each constraint dictionary supplies ``vertexIndices`` (Mx1,
    Mx2 or Mx3), ``barycentric`` of the same shape, ``targetPixels`` (Mx2), and
    optional nonnegative ``weights`` (M). Optional ``confidence`` is a separate
    M-vector in [0, 1]; weights need not be probabilities. ``viewIndex`` may be
    omitted, or must be zero. Optional point tangents and explicit segment pairs
    follow ``refineEyeGeometry``. A rejected point also removes its incident
    curve segments, without bridging gaps. Group/semantic minimum sample counts
    and the choice of visible front landmarks belong to the caller.

    Low confidence, low weight, nonpositive-depth and distant targets are removed
    individually before fitting; a bad target never rejects all other targets.
    The remaining confidence values multiply their observation weights.

    The solver reuses compact C2 fields and robust damped Gauss-Newton. To impose
    hard fixed vertices without clipping this smooth field, its global support
    radius is capped at the distance from the selected field centers to the
    nearest fixed vertex. Thus every basis function is exactly zero at every
    fixed vertex. This conservative cap can reduce the useful fit area and is
    reported. A fixed vertex coinciding with a center leaves the mesh unchanged.
    Opt-in ``fixedBoundaryBiharmonic`` instead smooths the displacement basis
    over the original triangle topology with fixed rows eliminated. It retains
    the requested support radius and the same control-center selection; the
    uniform smoothing length is a fraction of that radius. It smooths the
    displacement, not the reference face, and does not add camera-depth freedom.

    Displacement, strain, area, normal and edge gates are measured relative to
    this input reconstruction. No separate template attraction is introduced.
    Depth is preserved up to output floating-point rounding. ``solverOptions``
    forwards regularization/iteration/quality settings to the shared solver;
    camera-frame and target rejection policies cannot be overridden.
    """
    if deformationMode not in ("compact", "fixedBoundaryBiharmonic"):
        raise ValueError("Unknown front deformationMode")
    if (not isinstance(smoothingRadiusFraction, (int, float, np.integer, np.floating))
            or isinstance(smoothingRadiusFraction, (bool, np.bool_))
            or not np.isfinite(smoothingRadiusFraction) or not 0 < smoothingRadiusFraction <= 1):
        raise ValueError("smoothingRadiusFraction must be positive and at most one")
    inputVertices = np.asarray(vertices)
    original, checkedFaces = _validateMesh(vertices, faces)
    checkedCamera = _validateCameras([camera])[0]
    rawFixed = np.asarray(fixedVertexIds)
    if rawFixed.size == 0:
        fixed = np.empty(0, np.int64)
    elif (rawFixed.ndim != 1 or rawFixed.dtype.kind not in "iu"
          or rawFixed.min() < 0 or rawFixed.max() >= len(original)):
        raise ValueError("fixedVertexIds must be a valid integer vertex-index vector")
    else:
        fixed = np.unique(rawFixed.astype(np.int64))
    if (not np.isfinite([minConfidence, minWeight, maxTargetDistancePixels]).all()
            or not 0 <= minConfidence <= 1 or minWeight < 0
            or maxTargetDistancePixels <= 0):
        raise ValueError("Invalid front landmark confidence, weight or distance threshold")
    reserved = {"deformationFrame", "referenceViewIndex", "referenceWeight", "cameras", "deformationBasis"}
    if reserved.intersection(solverOptions):
        raise ValueError("Front landmark camera frame and reference settings are fixed")
    extent = float(np.ptp(original, axis=0).max())
    requestedRadius = .12*extent if supportRadius is None else float(supportRadius)
    displacementLimit = .025*extent if maxDisplacement is None else float(maxDisplacement)
    if (not np.isfinite([requestedRadius, displacementLimit]).all()
            or requestedRadius <= 0 or displacementLimit <= 0):
        raise ValueError("supportRadius and maxDisplacement must be positive and finite")

    filtered, inputReports = [], []
    for groupIndex, supplied in enumerate(constraints):
        item = dict(supplied)
        item.setdefault("viewIndex", 0)
        item.setdefault("name", "frontLandmarks")
        # Validate even an all-disabled group, which the shared solver otherwise
        # refuses before our per-point filtering can remove it.
        indices = np.asarray(item["vertexIndices"])
        count = len(indices)
        weights = np.asarray(item.get("weights", np.ones(count)), float)
        confidence = np.asarray(item.get("confidence", np.ones(count)), float)
        if (weights.shape != (count,) or not np.isfinite(weights).all()
                or np.any(weights < 0)):
            raise ValueError("Front landmark weights must be a finite nonnegative M-vector")
        if (confidence.shape != (count,) or not np.isfinite(confidence).all()
                or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError("Front landmark confidence must be a finite M-vector in [0, 1]")
        validating = dict(item)
        validating["weights"] = weights if weights.sum() > 0 else np.ones(count)
        prepared = _preparePointConstraints([validating], len(original), [checkedCamera])[0]
        points = (original[prepared["vertexIndices"]]
                  * prepared["barycentric"][..., None]).sum(1)
        pixels, depth, _ = _projectWithJacobian(points, checkedCamera)
        distance = np.linalg.norm(pixels-prepared["targetPixels"], axis=1)
        reasons = np.full(count, "kept", dtype=object)
        for rejected, name in (
            ((weights <= 0) | (weights < minWeight), "lowWeight"),
            ((confidence <= 0) | (confidence < minConfidence), "lowConfidence"),
            ((depth <= 1e-6) | ~np.isfinite(pixels).all(1), "invalidProjection"),
            (~np.isfinite(distance) | (distance > maxTargetDistancePixels), "targetTooDistant"),
        ):
            reasons[rejected & (reasons == "kept")] = name
        keep = reasons == "kept"
        report = {"groupIndex": groupIndex, "name": prepared["name"],
                  "inputCount": count, "keptCount": int(keep.sum()),
                  "keptIndices": np.flatnonzero(keep).tolist(),
                  "rejectedIndices": np.flatnonzero(~keep).tolist(),
                  "reasons": reasons.tolist(),
                  "initialDistancePixels": [float(value) if np.isfinite(value) else None
                                            for value in distance]}
        inputReports.append(report)
        if not keep.any():
            continue
        retained = {"viewIndex": 0, "name": prepared["name"],
                    "vertexIndices": prepared["vertexIndices"][keep].copy(),
                    "barycentric": prepared["barycentric"][keep].copy(),
                    "targetPixels": prepared["targetPixels"][keep].copy(),
                    "weights": (weights[keep]*confidence[keep]).copy(),
                    "tangentWeight": float(item.get("tangentWeight", 1.))}
        if "targetTangents" in item:
            retained["targetTangents"] = np.asarray(item["targetTangents"], float)[keep].copy()
        pairs = prepared["segmentPairs"]
        pairs = pairs[keep[pairs].all(1)]
        remap = np.full(count, -1, np.int64)
        remap[keep] = np.arange(keep.sum())
        retained["segmentPairs"] = remap[pairs]
        filtered.append(retained)

    points = [(original[item["vertexIndices"]]*item["barycentric"][..., None]).sum(1)
              for item in filtered]
    field = buildEyeDeformationBasis(original,
                                    np.concatenate(points) if points else np.empty((0, 3)),
                                    supportRadius=requestedRadius, controlCount=controlCount)
    effectiveRadius = requestedRadius
    nearestFixedDistance = None
    blockedByFixed = False
    if len(fixed) and len(field["controlPoints"]):
        nearestFixedDistance = float(np.linalg.norm(
            field["controlPoints"][:, None]-original[fixed][None], axis=2).min())
        # A small inward margin avoids a platform-dependent value just below 1
        # in the r/supportRadius comparison, without changing field smoothness.
        if deformationMode == "compact":
            effectiveRadius = min(requestedRadius, nearestFixedDistance*(1.-1e-12))
            blockedByFixed = effectiveRadius <= extent*1e-12
    if blockedByFixed:
        fittingConstraints = []
        solverRadius = requestedRadius
    else:
        fittingConstraints = filtered
        solverRadius = effectiveRadius

    if deformationMode == "fixedBoundaryBiharmonic":
        from .fixedBoundaryDeformation import buildFixedBoundaryDeformationBasis
        field = buildFixedBoundaryDeformationBasis(original, checkedFaces,
            np.concatenate(points) if points else np.empty((0, 3)), fixedVertexIds=fixed,
            supportRadius=requestedRadius, controlCount=controlCount,
            smoothingRadiusFraction=smoothingRadiusFraction)
        solverOptions["deformationBasis"] = field

    result = refineEyeGeometry(
        vertices, checkedFaces, [checkedCamera], fittingConstraints,
        supportRadius=solverRadius, controlCount=controlCount,
        maxDisplacement=displacementLimit, referenceViewIndex=0, referenceWeight=1.,
        maxTargetDistancePixels=maxTargetDistancePixels,
        deformationFrame="referenceImagePlane", **solverOptions)
    resultDtype = result["vertices"].dtype
    originalOutput = original.astype(resultDtype)
    if len(fixed) and not np.array_equal(result["vertices"][fixed], originalOutput[fixed]):
        # A fixed-interface violation is a programming/numerical error, not an
        # invitation to clip the solved field after its safety checks.
        raise RuntimeError("Front landmark field changed a hard fixed vertex")
    result["originalVertices"] = originalOutput.copy()
    result["keptConstraints"] = filtered
    diagnostics = result["diagnostics"]
    diagnostics["method"] = "frontLandmarkRefinement"
    diagnostics["priorGeometry"] = "input reconstructed mesh; no client-template rest-shape attraction"
    diagnostics["inputVertexDtype"] = str(inputVertices.dtype)
    diagnostics["inputTargetGate"] = {
        "maximumInitialDistancePixels": float(maxTargetDistancePixels),
        "minimumConfidence": float(minConfidence), "minimumWeight": float(minWeight),
        "inputTargetCount": sum(item["inputCount"] for item in inputReports),
        "keptTargetCount": sum(item["keptCount"] for item in inputReports),
        "rejectedTargetCount": sum(item["inputCount"]-item["keptCount"] for item in inputReports),
        "entireFitRejected": False, "policy": "reject individual observations and their incident segments",
        "groups": inputReports,
    }
    diagnostics["fixedBoundary"] = {
        "fixedVertexCount": len(fixed), "exactlyPreserved": True,
        "policy": "compact support excludes fixed vertices; no post-solve clipping",
        "requestedSupportRadius": requestedRadius, "effectiveSupportRadius": effectiveRadius,
        "nearestControlToFixedDistance": nearestFixedDistance,
        "supportRadiusReduced": effectiveRadius < requestedRadius,
        "blockedByCoincidentFixedVertex": blockedByFixed,
    }
    if deformationMode == "fixedBoundaryBiharmonic":
        diagnostics["fixedBoundary"]["policy"] = "Dirichlet coefficients eliminated on fixed vertices; full-mesh biharmonic response, no post-solve clipping"
        diagnostics["fixedBoundary"]["deformationMode"] = deformationMode
        diagnostics["fixedBoundary"]["smoothingRadiusFraction"] = float(smoothingRadiusFraction)
    diagnostics["selfIntersectionChecked"] = False
    diagnostics["productionQualityApproved"] = False
    if blockedByFixed:
        diagnostics["status"] = "unchanged_fixed_boundary_blocks_support"
    elif not filtered:
        diagnostics["status"] = "unchanged_no_reliable_targets"
    diagnostics["limits"].append(
        ("Hard boundary preservation can shrink the common compact support radius and limit fitting. "
         if deformationMode == "compact" else
         "The mesh displacement field is continuous but does not certify surface regularity or anatomical correctness. ")
        + "The caller must check nonadjacent triangle intersections and refreshed image/texture support.")
    if "solverEvidence" in result:
        result["solverEvidence"].update(deformationMode=deformationMode, fixedVertexIds=fixed.copy(),
            requestedSupportRadius=requestedRadius, effectiveSupportRadius=effectiveRadius,
            inputControlPoints=np.concatenate(points) if points else np.empty((0, 3)),
            keptConstraints=filtered, inputReports=inputReports)
    return result
