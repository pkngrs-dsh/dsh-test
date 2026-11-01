"""Experimental, fixed-camera chin correction on an existing client surface.

The caller supplies candidate correspondences and accepted photo observations.
Static jaw bindings are not a detector of the dynamic visible face contour.
Only the reference camera XY plane is deformable; topology and depth stay fixed.
"""
from __future__ import annotations

import numpy as np

from .clientGeometrySafety import intersectionPairs
from .eyeRefinement import _preparePointConstraints, _projectWithJacobian
from .frontLandmarkRefinement import checkFrontGeometry, refineFrontLandmarks
from .geometryRefinement import _validateCameras, _validateMesh
from .surfaceVisibility import ProjectedSurfaceVisibility


def _ids(value, count, name):
    value = np.asarray(value)
    if not value.size:
        return np.empty(0, np.int64)
    if value.ndim != 1 or value.dtype.kind not in "iu" or value.min() < 0 or value.max() >= count:
        raise ValueError(f"{name} must be a valid vertex-index vector")
    return np.unique(value.astype(np.int64))


def _sample(vertices, item):
    return (vertices[item["vertexIndices"]]*item["barycentric"][..., None]).sum(1)


def _errors(vertices, item, camera):
    pixels, depth, _ = _projectWithJacobian(_sample(vertices, item), camera)
    delta = pixels-item["targetPixels"]
    distances = np.linalg.norm(delta, axis=-1)
    weights = np.asarray(item["weights"], np.float64)
    projectionValid = (depth > 1e-6) & np.isfinite(pixels).all(1)
    valid = projectionValid & (weights > 0)
    normalMean = None
    if "targetTangents" in item and valid.any():
        tangent = np.array(item["targetTangents"], float, copy=True)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        normalMean = float(np.average(np.abs(np.sum(delta[valid]*normal[valid], axis=1)), weights=weights[valid]))
    return {"count": int(valid.sum()), "invalidProjectionCount": int(((weights > 0) & ~projectionValid).sum()),
            "meanErrorPixels": float(np.average(distances[valid], weights=weights[valid])) if valid.any() else None,
            "maximumErrorPixels": float(distances[valid].max()) if valid.any() else None,
            "normalMeanErrorPixels": normalMean,
            "projectedPixels": [[float(x), float(y)] if good else None for (x, y), good in zip(pixels, valid)]}


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def refineClientChin(vertices, faces, cameras, jawConstraint, *, referenceViewIndex=0,
                     fixedVertexIds=(), protectedVertexIds=(), protectedBindings=(),
                     supportRadius=None, maxDisplacement=None, minConfidence=None,
                     minimumPoints=5, maxTargetDistancePixels=25.,
                     candidateScales=(1., .5, .25, .125), crossViewConstraints=(),
                     controlCount=12, iterations=80, minImprovementPixels=.05,
                     maxCrossViewErrorIncreasePixels=1., maxRelativeEdgeChange=.25,
                     minAreaRatio=.4, minNormalCosine=.3):
    """Fit supplied jaw candidates with compact, smooth camera-XY displacement.

    ``jawConstraint`` has ``vertexIndices``, ``barycentric``, ``targetPixels``,
    mandatory ``confidence`` and optional weights, landmarkIds, targetTangents,
    tangentWeight and segmentPairs. Correspondences remain frozen throughout.
    Normal-direction fitting is available via targetTangents/tangentWeight;
    there is no implicit contour reattachment. ``minConfidence`` must be set
    explicitly by the caller for this detector; scores are not probabilities.

    All support vertices of ``protectedBindings`` are hard fixed, as are the
    supplied static and protected vertex IDs. The shared solver may shrink
    compact support to respect these fixed vertices; inadequate support is
    reported and never relabeled as a successful chin correction.

    Optional crossViewConstraints use the same binding format with viewIndex.
    They are diagnostic unless ``enforce=True`` is explicitly supplied. Every
    camera additionally reports projection movement, not an inferred photo
    fitting error. Static jaw residuals remain correspondence proxies.
    """
    original, triangles = _validateMesh(vertices, faces)
    rawCameras = list(cameras)
    checkedCameras = _validateCameras(rawCameras)
    if (isinstance(referenceViewIndex, (bool, np.bool_)) or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < len(checkedCameras)):
        raise ValueError("referenceViewIndex must address an accepted fixed camera")
    referenceViewIndex = int(referenceViewIndex)
    if rawCameras[referenceViewIndex].get("valid", True) is not True:
        raise ValueError("The reference camera was not accepted")
    if (minConfidence is None or not np.isfinite(minConfidence) or not 0 <= minConfidence <= 1
            or isinstance(minimumPoints, (bool, np.bool_)) or not isinstance(minimumPoints, (int, np.integer))
            or minimumPoints < 5 or not np.isfinite([maxTargetDistancePixels, minImprovementPixels,
                                                   maxCrossViewErrorIncreasePixels]).all()
            or maxTargetDistancePixels <= 0 or minImprovementPixels <= 0 or maxCrossViewErrorIncreasePixels < 0):
        raise ValueError("Set the detector-specific minConfidence explicitly and valid observation limits")
    scales = np.asarray(candidateScales, np.float64)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or (scales <= 0).any() or (scales > 1).any():
        raise ValueError("candidateScales must be nonempty positive fractions <=1")
    scales = np.unique(scales)[::-1]
    fixed = _ids(fixedVertexIds, len(original), "fixedVertexIds")
    protected = _ids(protectedVertexIds, len(original), "protectedVertexIds")
    for binding in protectedBindings:
        indices = np.asarray(binding["vertexIndices"])
        if indices.ndim != 2 or indices.shape[1] not in (1, 2, 3):
            raise ValueError("Protected bindings need Mx1/2/3 supporting vertex indices")
        protected = np.union1d(protected, _ids(indices.ravel(), len(original), "protected binding vertices"))
    hardFixed = np.union1d(fixed, protected).astype(np.int64)
    extent = float(np.ptp(original, axis=0).max())
    radius = .14*extent if supportRadius is None else float(supportRadius)
    maximum = .02*extent if maxDisplacement is None else float(maxDisplacement)
    if not np.isfinite([radius, maximum]).all() or radius <= 0 or maximum <= 0:
        raise ValueError("supportRadius and maxDisplacement must be positive finite mesh distances")
    camera = checkedCameras[referenceViewIndex]
    item = dict(jawConstraint)
    item["viewIndex"] = 0
    item.setdefault("name", "frontChinCandidate")
    if "confidence" not in item:
        raise ValueError("jawConstraint requires explicit detector confidence scores")
    if "targetTangents" in item:
        item.setdefault("tangentWeight", .05)
    prepared = _preparePointConstraints([item], len(original), [camera])[0]
    confidence = np.asarray(item["confidence"], float)
    if confidence.shape != (len(prepared["vertexIndices"]),) or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("confidence must be a finite detector-score vector in [0,1]")
    projected, depths, _ = _projectWithJacobian(_sample(original, prepared), camera)
    distances = np.linalg.norm(projected-prepared["targetPixels"], axis=1)
    kept = ((confidence >= minConfidence) & (confidence > 0) & (prepared["weights"] > 0)
            & (depths > 1e-6) & np.isfinite(projected).all(1) & (distances <= maxTargetDistancePixels))
    # Set rejected observations to zero without bridging their curve segments.
    prepared["weights"] *= confidence*kept
    if "targetTangents" in item:
        prepared["targetTangents"] = np.asarray(item["targetTangents"], float).copy()
    metricItem = prepared
    crossItems = []
    for supplied in crossViewConstraints:
        cross = dict(supplied)
        validated = _preparePointConstraints([cross], len(original), checkedCameras)[0]
        if "targetTangents" in cross:
            validated["targetTangents"] = np.asarray(cross["targetTangents"], float).copy()
        if "confidence" in cross:
            score = np.asarray(cross["confidence"], float)
            threshold = cross.get("minConfidence")
            if (threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1
                    or score.shape != validated["weights"].shape or not np.isfinite(score).all()
                    or (score < 0).any() or (score > 1).any()):
                raise ValueError("Cross-view confidence requires explicit valid detector threshold")
            validated["weights"] *= score*(score >= threshold)
        validated["enforce"] = bool(cross.get("enforce", False))
        validated["correspondencePolicy"] = str(cross.get("correspondencePolicy", "fixedBindingProxy"))
        crossItems.append(validated)
    beforeError = _errors(original, metricItem, camera)
    baselineCross = [_errors(original, cross, checkedCameras[cross["viewIndex"]]) for cross in crossItems]
    safetyOptions = dict(fixedVertexIds=hardFixed, camera=camera, maxDisplacement=maximum,
                         maxRelativeEdgeChange=maxRelativeEdgeChange, minAreaRatio=minAreaRatio,
                         minNormalCosine=minNormalCosine)
    initialCheck = checkFrontGeometry(original, original, triangles, **safetyOptions)
    diagnostics = {"method": "localFrontChinCandidate", "status": "unchanged",
        "correspondencePolicy": str(item.get("correspondencePolicy", "fixedJawCandidate")),
        "correspondenceValidatedAsDynamicContour": False,
        "confidenceMeaning": str(item.get("confidenceMeaning", "caller-supplied detector scores, not calibrated probabilities")),
        "confidenceThreshold": float(minConfidence), "inputPointCount": len(kept), "keptPointCount": int(kept.sum()),
        "keptInputIndices": np.flatnonzero(kept).tolist(), "rejectedInputIndices": np.flatnonzero(~kept).tolist(),
        "landmarkIds": item.get("landmarkIds"), "initialTargetDistancesPixels": distances,
        "referenceViewIndex": referenceViewIndex, "camerasFrozen": True,
        "staticVertexCount": len(fixed), "protectedVertexCount": len(protected), "hardFixedVertexCount": len(hardFixed),
        "requestedSupportRadius": radius, "maximumDisplacement": maximum,
        "beforeReference": beforeError, "initialGeometryCheck": initialCheck, "attempts": [],
        "selfIntersectionChecked": False, "productionQualityApproved": False,
        "limitations": [
            "Fixed template jaw bindings are candidate locations, not the dynamic photographed silhouette.",
            "Reference camera XY correction preserves depth and cannot resolve chin protrusion from a frontal photo.",
            "Cameras and target bindings stay frozen; all visibility and photo baking support must be recomputed after acceptance.",
            "Cross-view projected movement without reliable observations is not photo fitting accuracy.",
        ]}
    current = original.copy()
    baselinePairs = None
    if kept.sum() < minimumPoints:
        diagnostics["status"] = "unchanged_insufficient_reliable_points"
    else:
        proposal = refineFrontLandmarks(original, triangles, camera, [item], fixedVertexIds=hardFixed,
            supportRadius=radius, maxDisplacement=maximum, controlCount=controlCount,
            minConfidence=float(minConfidence), minWeight=0., maxTargetDistancePixels=maxTargetDistancePixels,
            iterations=iterations, curveWeight=0., maxRelativeEdgeChange=maxRelativeEdgeChange,
            minAreaRatio=minAreaRatio, minNormalCosine=minNormalCosine)
        diagnostics["solver"] = proposal["diagnostics"]
        fixedReport = proposal["diagnostics"]["fixedBoundary"]
        diagnostics["supportLimitedByFixedVertices"] = bool(fixedReport["supportRadiusReduced"])
        diagnostics["effectiveSupportRadius"] = float(fixedReport["effectiveSupportRadius"])
        diagnostics["supportRadiusFraction"] = float(fixedReport["effectiveSupportRadius"]/radius)
        diagnostics["supportSeverelyLimited"] = bool(fixedReport["effectiveSupportRadius"] < .25*radius)
        baselinePairs = intersectionPairs(original, triangles)
        diagnostics["selfIntersectionChecked"] = True
        for scale in scales:
            candidate = original+float(scale)*(proposal["vertices"]-original)
            geometryCheck = checkFrontGeometry(original, candidate, triangles, **safetyOptions)
            error = _errors(candidate, metricItem, camera)
            errorKey = "normalMeanErrorPixels" if "targetTangents" in metricItem else "meanErrorPixels"
            improvement = beforeError[errorKey]-error[errorKey] if error[errorKey] is not None else -np.inf
            attempt = {"scale": float(scale), "geometry": geometryCheck, "reference": error,
                       "referenceImprovementPixels": float(improvement), "objectiveMetric": errorKey}
            if not geometryCheck["accepted"] or improvement < minImprovementPixels or error["invalidProjectionCount"] > beforeError["invalidProjectionCount"]:
                attempt["accepted"] = False; diagnostics["attempts"].append(attempt); continue
            pairs = intersectionPairs(candidate, triangles)
            added = sorted(pairs-baselinePairs)
            crossChecks, crossSafe = [], True
            for cross, previous in zip(crossItems, baselineCross):
                after = _errors(candidate, cross, checkedCameras[cross["viewIndex"]])
                delta = (after["meanErrorPixels"]-previous["meanErrorPixels"]
                         if min(previous["count"], after["count"]) > 0 else None)
                accepted = not cross["enforce"] or (delta is not None and delta <= maxCrossViewErrorIncreasePixels
                                                     and after["count"] == previous["count"])
                crossSafe &= accepted
                crossChecks.append({"viewIndex": cross["viewIndex"], "name": cross["name"], "before": previous,
                    "after": after, "meanErrorChangePixels": delta, "enforced": cross["enforce"],
                    "accepted": accepted, "correspondencePolicy": cross["correspondencePolicy"]})
            attempt.update(newIntersectionPairs=len(added), newFacePairs=added, crossViewConstraints=crossChecks,
                           accepted=not added and crossSafe)
            diagnostics["attempts"].append(attempt)
            if attempt["accepted"]:
                current = candidate; diagnostics["status"] = "accepted"; diagnostics["acceptedScale"] = float(scale); break
        if diagnostics["status"] != "accepted":
            diagnostics["status"] = ("unchanged_fixed_support_blocks_fit" if diagnostics["supportSeverelyLimited"]
                                     else "unchanged_no_safe_improvement")
    finalCheck = checkFrontGeometry(original, current, triangles, **safetyOptions)
    if not finalCheck["accepted"]:
        raise RuntimeError("Final chin candidate failed cumulative geometry checks")
    if baselinePairs is not None:
        finalPairs = intersectionPairs(current, triangles)
        if finalPairs-baselinePairs:
            raise RuntimeError("Final chin candidate introduced triangle intersections")
        diagnostics.update(baselineIntersectionPairs=len(baselinePairs), finalIntersectionPairs=len(finalPairs), newIntersectionPairs=0)
    projections = []
    for viewIndex, checked in enumerate(checkedCameras):
        p0, z0, _ = _projectWithJacobian(_sample(original, metricItem), checked)
        p1, z1, _ = _projectWithJacobian(_sample(current, metricItem), checked)
        valid = kept & (z0 > 1e-6) & (z1 > 1e-6) & np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
        moved = np.linalg.norm(p1-p0, axis=1)
        projections.append({"viewIndex": viewIndex, "acceptedCamera": bool(rawCameras[viewIndex].get("valid", True)),
            "count": int(valid.sum()), "meanJawProjectionMovementPixels": float(moved[valid].mean()) if valid.any() else None,
            "maximumJawProjectionMovementPixels": float(moved[valid].max()) if valid.any() else None,
            "beforePixels": p0, "afterPixels": p1,
            "meaning": "fixed candidate binding movement; not a dynamic silhouette error"})
    diagnostics.update(finalGeometryCheck=finalCheck, afterReference=_errors(current, metricItem, camera),
        projectionDiagnostics=projections, changedVertexCount=int(np.any(current != original, axis=1).sum()),
        fixedVerticesExactlyUnchanged=bool(np.array_equal(current[hardFixed], original[hardFixed])))
    return {"vertices": current, "originalVertices": original.copy(), "displacement": current-original,
            "activeVertexMask": np.any(current != original, axis=1), "diagnostics": _json(diagnostics)}


def visibleLowerJawConstraints(vertices, faces, camera, targetPixels, confidence, landmarkIds,
                               minimumYPixels, *, imageShape=(512, 512), maxTargetDistancePixels=35.,
                               targetTangents=None, minimumVisibleFraction=.5, maximumCandidatesPerPoint=12,
                               maximumChainDetourRatio=3., maximumChainSlackPixels=12.,
                               maximumTangentErrorPixels=6.):
    """Freeze reliable reference-view *visible edge* candidates for lower-jaw fitting.

    Find edges shared by a front- and back-facing triangle. Candidate points
    intersect each photograph target's normal line with a projected edge;
    closest segment points are the fallback when that line misses the segment.
    Screen-edge fractions are converted to perspective-correct 3D barycentrics.
    Continuous two-sided ray depth qualifies every candidate, and interior edge
    samples reject edges visible only at an incidental endpoint.
    Endpoint fallbacks must stay within maximumTangentErrorPixels of the
    target normal. Correspondence selection penalizes tangential mismatch
    independently of the small tangentWeight used by the deformation solver;
    an unrelated endpoint must not win by sliding sideways along the contour.

    All selected edges must lie in one lower-face transition component. A
    shortest-path check bounds mesh-contour detours between successive points,
    and image-X order must agree with target order. Short hidden connectors in
    the same transition component may join folded contour segments; they are
    connectivity only, never fitted observations. Their count is reported.
    This explicitly excludes disconnected lateral-neck chains, but remains a
    geometric candidate association rather than certified facial anatomy.

    Reinvoke after deformation for dynamic edge diagnostics; the returned fit
    correspondence itself must remain frozen within one optimization solve.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra

    original, triangles = _validateMesh(vertices, faces)
    checked = _validateCameras([camera])[0]
    target = np.asarray(targetPixels, np.float64)
    scores = np.asarray(confidence, np.float64)
    ids = np.asarray(landmarkIds)
    if (target.ndim != 2 or target.shape[1] != 2 or len(target) < 5 or not np.isfinite(target).all()
            or scores.shape != (len(target),) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or ids.shape != (len(target),) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)):
        raise ValueError("Expected at least five ordered lower-jaw targets with explicit scores and unique landmark IDs")
    if (not np.isfinite([minimumYPixels, maxTargetDistancePixels, minimumVisibleFraction,
                         maximumChainDetourRatio, maximumChainSlackPixels, maximumTangentErrorPixels]).all()
            or maxTargetDistancePixels <= 0 or not 0 < minimumVisibleFraction <= 1
            or maximumChainDetourRatio < 1 or maximumChainSlackPixels < 0
            or maximumTangentErrorPixels < 0
            or not isinstance(maximumCandidatesPerPoint, int) or maximumCandidatesPerPoint < 1):
        raise ValueError("Invalid visible-contour search limits")
    if len(imageShape) != 2 or min(imageShape) < 2:
        raise ValueError("imageShape must be a positive photograph H,W pair")
    height, width = map(int, imageShape)
    if (target < 0).any() or (target >= [width, height]).any():
        raise ValueError("Jaw targets must remain within the working photograph")
    xDifference = np.diff(target[:, 0])
    orderSign = 1 if np.all(xDifference > .25) else (-1 if np.all(xDifference < -.25) else 0)
    if not orderSign:
        raise ValueError("Central lower-jaw targets must have a consistent image-X order")
    tangents = np.gradient(target, axis=0) if targetTangents is None else np.asarray(targetTangents, np.float64)
    if tangents.shape != target.shape or not np.isfinite(tangents).all() or (np.linalg.norm(tangents, axis=1) < 1e-8).any():
        raise ValueError("targetTangents must contain finite nonzero vectors")
    tangents = tangents/np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    cameraVertices = original@checked["rotation"].T+checked["translation"]
    homogeneous = cameraVertices@checked["intrinsics"].T
    pixels = homogeneous[:, :2]/np.maximum(homogeneous[:, 2:], 1e-12)
    facePoints = cameraVertices[triangles]
    faceNormal = np.cross(facePoints[:, 1]-facePoints[:, 0], facePoints[:, 2]-facePoints[:, 0])
    facing = np.sum(faceNormal*(-facePoints.mean(1)), axis=1) > 0
    rawEdges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    faceOwners = np.tile(np.arange(len(triangles)), 3)
    edges, inverse, counts = np.unique(rawEdges, axis=0, return_inverse=True, return_counts=True)
    frontCounts = np.bincount(inverse, weights=facing[faceOwners], minlength=len(edges))
    transition = (counts == 2) & (frontCounts == 1)
    transitionCount = int(transition.sum())
    edgeIds = np.flatnonzero(transition)
    edges = edges[edgeIds]
    endpoints = pixels[edges]
    margin = float(maxTargetDistancePixels)
    bounds = [max(0., float(target[:, 0].min())-margin), float(minimumYPixels),
              min(width-1., float(target[:, 0].max())+margin),
              min(height-1., float(target[:, 1].max())+margin)]
    eligible = ((cameraVertices[edges, 2] > 1e-6).all(1)
                & (endpoints[..., 0].max(1) >= bounds[0]) & (endpoints[..., 0].min(1) <= bounds[2])
                & (endpoints[..., 1].max(1) >= bounds[1]) & (endpoints[..., 1].min(1) <= bounds[3]))
    edges, edgeIds, endpoints = edges[eligible], edgeIds[eligible], endpoints[eligible]
    lengths = np.linalg.norm(endpoints[:, 1]-endpoints[:, 0], axis=1)
    keep = lengths > 1e-6
    edges, edgeIds, endpoints, lengths = edges[keep], edgeIds[keep], endpoints[keep], lengths[keep]
    if not len(edges):
        raise ValueError("No lower-front facing-transition edges were found")
    nodes, localEdges = np.unique(edges, return_inverse=True)
    localEdges = localEdges.reshape(-1, 2)
    graph = coo_matrix((np.concatenate([lengths, lengths]),
        (np.concatenate([localEdges[:, 0], localEdges[:, 1]]),
         np.concatenate([localEdges[:, 1], localEdges[:, 0]]))), shape=(len(nodes), len(nodes))).tocsr()
    componentCount, components = connected_components(graph, directed=False)
    edgeComponents = components[localEdges[:, 0]]
    pathDistances = dijkstra(graph, directed=False)
    visibility = ProjectedSurfaceVisibility(cameraVertices, triangles, checked["intrinsics"], (height, width))
    tolerance = float(np.ptp(original, axis=0).max())*1e-6
    fractions = np.linspace(.05, .95, 11)
    edgePoints = (cameraVertices[edges[:, 0], None]*(1-fractions[None, :, None])
                  + cameraVertices[edges[:, 1], None]*fractions[None, :, None])
    depth = visibility.closestDepth(edgePoints.reshape(-1, 3))["depth"].reshape(len(edges), -1)
    visibleSamples = np.isfinite(depth) & (np.abs(depth-edgePoints[..., 2]) <= tolerance)
    visibleFraction = visibleSamples.mean(1)
    visibleEdges = visibleFraction >= minimumVisibleFraction
    pools = [[] for _ in target]
    rejectedTangentialCandidates = np.zeros(len(target), np.int64)
    for pointIndex, (point, tangent, normal) in enumerate(zip(target, tangents, normals)):
        for edgeIndex in np.flatnonzero(visibleEdges):
            first, last = endpoints[edgeIndex]
            vector = last-first
            denominator = float(vector@tangent)
            crossing = float((point-first)@tangent/denominator) if abs(denominator) > 1e-8 else np.nan
            normalIntersection = bool(np.isfinite(crossing) and 0 <= crossing <= 1)
            fraction = crossing if normalIntersection else float(np.clip((point-first)@vector/(vector@vector), 0, 1))
            location = first+fraction*vector
            if not (bounds[0] <= location[0] <= bounds[2] and bounds[1] <= location[1] <= bounds[3]):
                continue
            offset = location-point
            distance = float(np.linalg.norm(offset))
            if distance > margin:
                continue
            perspective = np.array([1-fraction, fraction])/cameraVertices[edges[edgeIndex], 2]
            perspective /= perspective.sum()
            sample = perspective@cameraVertices[edges[edgeIndex]]
            hit = visibility.closestDepth(sample[None])["depth"][0]
            if not np.isfinite(hit) or abs(float(hit)-sample[2]) > tolerance:
                continue
            normalError, tangentError = float(offset@normal), float(offset@tangent)
            if abs(tangentError) > maximumTangentErrorPixels:
                rejectedTangentialCandidates[pointIndex] += 1
                continue
            pools[pointIndex].append({"edgeIndex": int(edgeIndex), "component": int(edgeComponents[edgeIndex]),
                "pixel": location, "screenFraction": float(fraction), "barycentric": perspective,
                "cameraDepth": float(sample[2]), "normalErrorPixels": normalError,
                "tangentErrorPixels": tangentError, "distancePixels": distance,
                "normalIntersection": normalIntersection,
                "cost": normalError**2+4.*tangentError**2})
    available = [set(row["component"] for row in pool) for pool in pools]
    commonComponents = set.intersection(*available) if all(available) else set()
    if not commonComponents:
        raise ValueError("No single visible lower-jaw transition chain supports all supplied targets within the distance limit")

    def contourDistance(first, second):
        a, b = first["edgeIndex"], second["edgeIndex"]
        if a == b:
            return abs(first["screenFraction"]-second["screenFraction"])*lengths[a]
        costA = np.array([first["screenFraction"], 1-first["screenFraction"]])*lengths[a]
        costB = np.array([second["screenFraction"], 1-second["screenFraction"]])*lengths[b]
        return float((costA[:, None]+pathDistances[np.ix_(localEdges[a], localEdges[b])]+costB[None]).min())

    solutions = []
    for component in sorted(commonComponents):
        selectedPools = [sorted((row for row in pool if row["component"] == component), key=lambda row: row["cost"])
                         [:maximumCandidatesPerPoint] for pool in pools]
        scoresDP = np.asarray([row["cost"] for row in selectedPools[0]])
        backtracks = []
        for pointIndex in range(1, len(target)):
            previous, following = selectedPools[pointIndex-1], selectedPools[pointIndex]
            nextScores = np.full(len(following), np.inf)
            links = np.full(len(following), -1, np.int64)
            for nextIndex, after in enumerate(following):
                for oldIndex, before in enumerate(previous):
                    if not np.isfinite(scoresDP[oldIndex]) or orderSign*(after["pixel"][0]-before["pixel"][0]) <= .25:
                        continue
                    direct = float(np.linalg.norm(after["pixel"]-before["pixel"]))
                    path = contourDistance(before, after)
                    if path > maximumChainDetourRatio*direct+maximumChainSlackPixels:
                        continue
                    cost = scoresDP[oldIndex]+after["cost"]+.02*max(path-direct, 0.)**2
                    if cost < nextScores[nextIndex]:
                        nextScores[nextIndex], links[nextIndex] = cost, oldIndex
            scoresDP = nextScores; backtracks.append(links)
        if not np.isfinite(scoresDP).any():
            continue
        last = int(np.argmin(scoresDP))
        ordered = [last]
        for links in backtracks[::-1]:
            ordered.append(int(links[ordered[-1]]))
        ordered.reverse()
        solution = [pool[index] for pool, index in zip(selectedPools, ordered)]
        solutions.append((float(scoresDP[last]), float(np.mean([row["cameraDepth"] for row in solution])), component, solution))
    if not solutions:
        raise ValueError("Visible edge candidates fail ordered-chain continuity or detour checks")
    _, _, chosenComponent, solution = min(solutions, key=lambda value: (value[0], value[1]))
    chosenEdges = np.asarray([row["edgeIndex"] for row in solution], np.int64)
    binding = {"name": "visibleLowerJawEdges", "viewIndex": 0,
        "vertexIndices": edges[chosenEdges].copy(), "barycentric": np.asarray([row["barycentric"] for row in solution]),
        "targetPixels": target.copy(), "confidence": scores.copy(), "weights": np.ones(len(target)),
        "targetTangents": tangents.copy(), "tangentWeight": .05, "landmarkIds": ids.astype(np.int64).copy(),
        "segmentPairs": np.column_stack([np.arange(len(target)-1), np.arange(1, len(target))]),
        "correspondencePolicy": "frozenVisibleLowerJawEdgeCandidate",
        "confidenceMeaning": "caller supplied detector scores, not calibrated probabilities"}
    chainDistances = [contourDistance(a, b) for a, b in zip(solution[:-1], solution[1:])]
    diagnostics = {"method": "facingTransitionVisibleLowerJawEdgeAssociation", "status": "candidates_found",
        "targetCount": len(target), "landmarkIds": ids, "frontBackTransitionEdgeCount": transitionCount,
        "lowerRoiEdgeCount": len(edges), "interiorVisibleEdgeCount": int(visibleEdges.sum()),
        "regionBoundsXyxy": bounds, "targetDistanceLimitPixels": margin,
        "maximumTangentErrorPixels": float(maximumTangentErrorPixels),
        "correspondenceTangentSquaredPenalty": 4.,
        "candidateCounts": [len(pool) for pool in pools],
        "tangentRejectedCandidateCounts": rejectedTangentialCandidates,
        "componentCount": int(componentCount), "selectedComponent": int(chosenComponent),
        "componentHiddenConnectorEdgeCount": int(((edgeComponents == chosenComponent) & ~visibleEdges).sum()),
        "allCandidatesInOneTransitionComponent": True, "imageXOrderVerified": True,
        "boundedConsecutiveContourDistancesPixels": chainDistances,
        "maximumChainDetourRatio": float(maximumChainDetourRatio), "maximumChainSlackPixels": float(maximumChainSlackPixels),
        "minimumInteriorVisibleFraction": float(minimumVisibleFraction), "depthTolerance": tolerance,
        "normalMeanAbsoluteErrorPixels": float(np.mean([abs(row["normalErrorPixels"]) for row in solution])),
        "meanEuclideanErrorPixels": float(np.mean([row["distancePixels"] for row in solution])),
        "candidatePixels": np.asarray([row["pixel"] for row in solution]),
        "selectedEdges": [{"vertexIndices": edges[row["edgeIndex"]], "edgeIndex": int(edgeIds[row["edgeIndex"]]),
            "pixel": row["pixel"], "barycentric": row["barycentric"], "cameraDepth": row["cameraDepth"],
            "normalErrorPixels": row["normalErrorPixels"], "tangentErrorPixels": row["tangentErrorPixels"],
            "normalIntersection": row["normalIntersection"], "visibleFraction": float(visibleFraction[row["edgeIndex"]])}
            for row in solution],
        "limitations": ["Candidate edge identity is specific to this geometry and camera; recompute it for after-shape contour evaluation.",
            "Each fitted sample is ray-visible; hidden edges may only connect folded segments topologically.",
            "Screen-X order and short contour paths are geometric checks, not proof of anatomical jaw correspondence.",
            "Non-manifold and mesh-boundary edges are omitted; only two-face front/back transitions are considered."]}
    return {"constraint": binding, "diagnostics": _json(diagnostics)}
