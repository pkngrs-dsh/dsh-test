"""Conservative fixed-topology geometry fitting to frozen photo constraints.

This module differentiates OpenCV point projection, not image rasterization or
photometric shading. All semantic surface bindings and visibility are frozen
before fitting. Photo semantic boundaries are observations, not anatomical
ground truth; in particular a parser's nose region is not a 3D silhouette.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


def projectTensorPoints(points, intrinsics, rotation, translation):
    """Differentiable integer-centered OpenCV projection; return pixels, depth.

    Constants inherit the input tensor's dtype/device. Cameras are never model
    parameters. Points with nonpositive depth must be rejected by the caller.
    """
    import torch

    constant = lambda value: torch.as_tensor(value, dtype=points.dtype, device=points.device).detach()
    cameraPoints = points @ constant(rotation).T + constant(translation).reshape(1, 3)
    projected = cameraPoints @ constant(intrinsics).T
    return projected[:, :2] / projected[:, 2:3].clamp_min(1e-9), cameraPoints[:, 2]


def _meshEdges(faces):
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _validateMesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    rawFaces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite Vx3 array")
    if rawFaces.ndim != 2 or rawFaces.shape[1] != 3 or not len(rawFaces):
        raise ValueError("faces must be a nonempty Fx3 array")
    if not np.issubdtype(rawFaces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    faces = rawFaces.astype(np.int64, copy=False)
    if faces.min() < 0 or faces.max() >= len(vertices) or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("faces contain invalid or repeated vertex indices")
    doubleArea = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                        vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    if doubleArea.min() <= 1e-14 or np.ptp(vertices, axis=0).max() <= 1e-12:
        raise ValueError("Initial mesh contains degenerate triangles")
    return vertices, faces


def _validateCameras(cameras):
    if not len(cameras):
        raise ValueError("At least one frozen camera is required")
    result = []
    for camera in cameras:
        intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        translation = np.asarray(camera["translation"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or rotation.shape != (3, 3) or translation.size != 3:
            raise ValueError("Cameras require 3x3 intrinsics/rotation and 3-vector translation")
        if not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation)):
            raise ValueError("Cameras must be finite")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or np.linalg.det(rotation) < .999 or abs(np.linalg.det(intrinsic)) < 1e-12
                or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-7)):
            raise ValueError("Expected a proper OpenCV pinhole camera")
        result.append({"intrinsics": intrinsic.copy(), "rotation": rotation.copy(),
                       "translation": translation.reshape(3).copy()})
    return result


def _sampleImage(image, pixels):
    return map_coordinates(np.asarray(image, dtype=np.float64), pixels[:, ::-1].T,
                           order=1, mode="nearest")


def _insideImage(pixels, shape, margin=1.):
    height, width = shape
    return ((pixels[:, 0] >= margin) & (pixels[:, 0] <= width - 1 - margin)
            & (pixels[:, 1] >= margin) & (pixels[:, 1] <= height - 1 - margin))


def _perspectiveEdgeBarycentric(screenFraction, endpointDepths):
    """Convert linear screen-edge interpolation into 3D edge barycentrics."""
    weights = np.stack([1 - screenFraction, screenFraction], axis=1) / endpointDepths
    return weights / weights.sum(1, keepdims=True)


def buildSemanticBoundaryConstraints(vertices, faces, cameras, labelMaps, confidenceMaps,
                                     vertexVisibility, semanticGroups, *, referenceViewIndex=0,
                                     minConfidence=.5, maxTargetDistance=16., maxSamplesPerGroup=200):
    """Bind reference semantic contours to fixed mesh edges, then other photos.

    ``semanticGroups`` maps a name to ``{"labelIds": [int], "weight": float}``.
    Cameras, maps and visibility use identical view order and original photo
    pixel coordinates. ``vertexVisibility`` is a caller-supplied NxV boolean
    depth-tested mask computed on the initial mesh; no visibility is inferred
    from semantics. A sample is kept only when both supporting vertices are
    visible. Each returned constraint stores immutable surface indices and
    barycentric weights, a target signed-distance image, and sample weights.

    A reference contour is attached where its signed distance crosses an
    initial mesh edge. This anchors its initial front projection, rather than
    claiming the mesh's anatomical eye/lip boundary has been independently
    measured. Other views provide the fitting signal. Large mismatches are
    rejected as unreliable initial correspondence, not silently fitted.
    """
    from .camera import projectPoints

    vertices, faces = _validateMesh(vertices, faces)
    cameras = _validateCameras(cameras)
    viewCount = len(cameras)
    visibility = np.asarray(vertexVisibility)
    if visibility.shape != (viewCount, len(vertices)) or visibility.dtype != np.bool_:
        raise ValueError("vertexVisibility must be a boolean NxV array")
    if len(labelMaps) != viewCount or len(confidenceMaps) != viewCount:
        raise ValueError("One label/confidence map is required per camera")
    if (not 0 <= referenceViewIndex < viewCount or not 0 <= minConfidence <= 1
            or not np.isfinite(maxTargetDistance) or maxTargetDistance <= 0
            or not isinstance(maxSamplesPerGroup, int) or maxSamplesPerGroup < 1):
        raise ValueError("Invalid semantic binding configuration")
    labels, confidences, projectedVertices = [], [], []
    for labelMap, confidenceMap, camera in zip(labelMaps, confidenceMaps, cameras):
        labelMap = np.asarray(labelMap)
        confidenceMap = np.asarray(confidenceMap, dtype=np.float64)
        if (labelMap.ndim != 2 or min(labelMap.shape) < 3 or not np.issubdtype(labelMap.dtype, np.integer)
                or confidenceMap.shape != labelMap.shape or not np.isfinite(confidenceMap).all()
                or np.any((confidenceMap < 0) | (confidenceMap > 1))):
            raise ValueError("Expected integer HW labels and matching finite [0,1] confidence")
        labels.append(labelMap.copy())
        confidences.append(confidenceMap.copy())
        projectedVertices.append(projectPoints(vertices, **camera))

    edges = _meshEdges(faces)
    referencePixels, referenceDepth = projectedVertices[referenceViewIndex]
    referenceValid = (visibility[referenceViewIndex] & (referenceDepth > 0)
                      & _insideImage(referencePixels, labels[referenceViewIndex].shape)
                      & (_sampleImage(confidences[referenceViewIndex], referencePixels) >= minConfidence))
    constraints, groupReports = [], {}
    for name, group in semanticGroups.items():
        labelIds, groupWeight = group["labelIds"], float(group.get("weight", 1.))
        if not labelIds or not np.isfinite(groupWeight) or groupWeight <= 0:
            raise ValueError("Semantic groups need labelIds and a positive finite weight")
        distanceMaps = []
        for labelMap in labels:
            region = np.isin(labelMap, labelIds)
            # Missing/all-frame regions have no meaningful contour.
            distanceMaps.append((distance_transform_edt(~region) - distance_transform_edt(region))
                                if region.any() and not region.all() else None)
        report = {"referenceSamples": 0, "viewSamples": [0] * viewCount,
                  "weight": groupWeight, "labelIds": [int(value) for value in labelIds]}
        groupReports[name] = report
        if distanceMaps[referenceViewIndex] is None:
            continue
        vertexDistance = _sampleImage(distanceMaps[referenceViewIndex], referencePixels)
        crossings = referenceValid[edges].all(1) & (vertexDistance[edges[:, 0]] * vertexDistance[edges[:, 1]] < 0)
        selectedEdges = edges[crossings]
        if not len(selectedEdges):
            continue
        if len(selectedEdges) > maxSamplesPerGroup:
            selectedEdges = selectedEdges[np.linspace(0, len(selectedEdges) - 1, maxSamplesPerGroup).astype(int)]
        endDistances = vertexDistance[selectedEdges]
        edgePixels = referencePixels[selectedEdges]
        # Signed distances need not vary linearly along a long image-space edge.
        # Bisect the fixed image field; each bracket still changes sign, without
        # ever relabeling the mesh during geometry fitting.
        lower, upper = np.zeros(len(selectedEdges)), np.ones(len(selectedEdges))
        lowerDistance = endDistances[:, 0].copy()
        for _ in range(24):
            middle = (lower + upper) * .5
            samplePixels = edgePixels[:, 0] * (1 - middle[:, None]) + edgePixels[:, 1] * middle[:, None]
            middleDistance = _sampleImage(distanceMaps[referenceViewIndex], samplePixels)
            lowerHalf = lowerDistance * middleDistance <= 0
            upper = np.where(lowerHalf, middle, upper)
            lower = np.where(lowerHalf, lower, middle)
            lowerDistance = np.where(lowerHalf, lowerDistance, middleDistance)
        fraction = (lower + upper) * .5
        barycentric = _perspectiveEdgeBarycentric(fraction, referenceDepth[selectedEdges])
        points = (vertices[selectedEdges] * barycentric[..., None]).sum(1)
        sourcePixels, _ = projectPoints(points, **cameras[referenceViewIndex])
        sourceConfidence = _sampleImage(confidences[referenceViewIndex], sourcePixels)
        report["referenceSamples"] = len(points)
        sourceResidual = np.abs(_sampleImage(distanceMaps[referenceViewIndex], sourcePixels))
        report["initialReferenceBoundaryErrorPixels"] = {
            "mean": float(sourceResidual.mean()), "p95": float(np.percentile(sourceResidual, 95)),
            "max": float(sourceResidual.max())}
        for viewIndex, camera in enumerate(cameras):
            if distanceMaps[viewIndex] is None:
                continue
            pixels, depth = projectPoints(points, **camera)
            targetConfidence = _sampleImage(confidences[viewIndex], pixels)
            residual = _sampleImage(distanceMaps[viewIndex], pixels)
            valid = (visibility[viewIndex, selectedEdges].all(1) & (depth > 0)
                     & _insideImage(pixels, labels[viewIndex].shape)
                     & (sourceConfidence >= minConfidence) & (targetConfidence >= minConfidence)
                     & (np.abs(residual) <= maxTargetDistance))
            report["viewSamples"][viewIndex] = int(valid.sum())
            if not valid.any():
                continue
            constraints.append({"viewIndex": viewIndex, "name": str(name),
                                "vertexIndices": selectedEdges[valid].copy(),
                                "barycentric": barycentric[valid].copy(),
                                "weights": groupWeight * sourceConfidence[valid] * targetConfidence[valid],
                                "targetDistance": distanceMaps[viewIndex].astype(np.float32)})
    return {"constraints": constraints,
            "diagnostics": {"referenceViewIndex": int(referenceViewIndex), "groups": groupReports,
                            "constraintCount": len(constraints), "visibilityFrozen": True,
                            "binding": "bisected reference semantic zero crossing with perspective-correct fixed edge barycentrics",
                            "limitation": "Semantic boundary correspondence, not independent anatomical landmarks"}}


def _sampleTensorImage(image, pixels):
    import torch.nn.functional as functional

    height, width = image.shape
    normalized = pixels / pixels.new_tensor([width - 1, height - 1]) * 2 - 1
    return functional.grid_sample(image[None, None], normalized[None, None],
                                  align_corners=True, mode="bilinear", padding_mode="border").reshape(-1)


def _prepareConstraints(constraints, vertexCount, cameras, device, dtype):
    import torch

    prepared = []
    tensor = lambda value, kind=dtype: torch.as_tensor(value, dtype=kind, device=device)
    for constraint in constraints:
        viewIndex = constraint["viewIndex"]
        indices = np.asarray(constraint["vertexIndices"])
        barycentric = np.asarray(constraint["barycentric"], dtype=np.float64)
        weights = np.asarray(constraint.get("weights", np.ones(len(indices))), dtype=np.float64)
        if (not isinstance(viewIndex, (int, np.integer)) or not 0 <= viewIndex < len(cameras)
                or indices.ndim != 2 or not len(indices) or indices.shape[1] not in (1, 2, 3)
                or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= vertexCount
                or barycentric.shape != indices.shape or not np.isfinite(barycentric).all()
                or np.any(barycentric < 0) or not np.allclose(barycentric.sum(1), 1, atol=1e-6)
                or weights.shape != (len(indices),) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Invalid frozen surface constraint indices/weights/barycentric coordinates")
        item = {"viewIndex": int(viewIndex), "name": str(constraint.get("name", "landmarks")),
                "vertexIndices": tensor(indices, torch.int64), "barycentric": tensor(barycentric),
                "weights": tensor(weights)}
        hasDistance, hasPixels = "targetDistance" in constraint, "targetPixels" in constraint
        if hasDistance == hasPixels:
            raise ValueError("Each constraint needs exactly one of targetDistance or targetPixels")
        if hasDistance:
            target = np.asarray(constraint["targetDistance"])
            if target.ndim != 2 or min(target.shape) < 2 or not np.isfinite(target).all():
                raise ValueError("targetDistance must be a finite HW signed-distance image")
            item["targetDistance"] = tensor(target)
        else:
            target = np.asarray(constraint["targetPixels"])
            if target.shape != (len(indices), 2) or not np.isfinite(target).all():
                raise ValueError("targetPixels must be a finite Mx2 array")
            item["targetPixels"] = tensor(target)
        prepared.append(item)
    return prepared


def _buildSmoothDeformationBasis(vertices, activeMask, controlCount, radius):
    """Normalized Gaussian controls, tapered continuously toward pinned vertices.

    Euclidean RBFs make nearby vertices co-move even when triangulation contains
    extremely short edges. This is a fixed low-dimensional displacement field,
    not a remeshing operation. The smooth boundary envelope keeps pinned
    vertices fixed without introducing a hard displacement step at the support.
    """
    points = vertices[activeMask]
    if not len(points):
        return np.zeros((0, 0)), {"controlCount": 0, "radius": float(radius)}
    count = min(controlCount, len(points))
    indices = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    distanceSquared = np.sum((points - points[indices[0]]) ** 2, axis=1)
    for _ in range(1, count):
        indices.append(int(np.argmax(distanceSquared)))
        distanceSquared = np.minimum(distanceSquared, np.sum((points - points[indices[-1]]) ** 2, axis=1))
    distances = np.sum((points[:, None] - points[np.asarray(indices)][None]) ** 2, axis=-1)
    # Subtract the nearest distance before exponentiation to preserve numerical
    # normalization when disconnected active regions are far apart.
    weights = np.exp(-(distances - distances.min(1, keepdims=True)) / (2 * radius ** 2))
    weights /= weights.sum(1, keepdims=True)
    pinned = vertices[~activeMask]
    if len(pinned):
        boundaryDistance = cKDTree(pinned).query(points)[0]
        position = np.clip(boundaryDistance / radius, 0, 1)
        envelope = position * position * (3 - 2 * position)
        weights *= envelope[:, None]
    else:
        envelope = np.ones(len(points))
    return weights, {"controlCount": count, "radius": float(radius),
                     "minimumEnvelope": float(envelope.min()), "maximumEnvelope": float(envelope.max()),
                     "parameterization": "fixed Gaussian RBF controls with smooth pinned-boundary envelope"}


def refineLocalGeometry(vertices, faces, cameras, constraints, *, maxDisplacement=None,
                        supportRings=3, referenceViewIndex=0, referenceWeight=4.,
                        maxViewDegradation=1., iterations=160, learningRate=.03,
                        pixelScale=4., displacementWeight=.01, laplacianWeight=.05,
                        edgeWeight=.1, minAreaRatio=.25, minNormalCosine=.1,
                        maxRelativeEdgeChange=.35, minImprovementPixels=.001,
                        activeVertexMask=None, device="cpu", deformationMode="vertex",
                        smoothControlCount=64, smoothRadius=None, referenceAnchorWeight=0.):
    """Fit bounded vertex offsets; preserve topology, cameras and best safe state.

    Constraint dictionaries contain ``viewIndex``, ``vertexIndices`` (Mx1/2/3),
    ``barycentric`` (same shape), positive ``weights`` (M), and exactly one of
    ``targetDistance`` (HW signed pixel distance) or ``targetPixels`` (Mx2).
    Optional ``name`` identifies each group in diagnostics. No target, binding,
    camera, or visibility updates occur during optimization.

    Losses use mean robust pixel residual per view, with a higher reference
    weight, plus displacement/Laplacian/relative-edge regularization. Offsets
    are radially bounded in original mesh units (default .005 mesh extent).
    Candidate checkpoints must reduce aggregate pixel error and regularized
    objective, stay within every view's absolute pixel degradation allowance,
    keep positive observed depth, and pass local triangle/edge checks. These
    checks prevent triangle collapse/orientation reversal relative to the
    initial mesh; they do not establish absence of global self-intersection.
    Vertices beyond the constraint support rings are pinned exactly.
    ``deformationMode='smooth'`` uses spatial RBF controls to avoid independent
    neighboring offsets fighting across extremely short mesh edges. Its safety
    gates are identical. ``referenceAnchorWeight`` optionally penalizes full
    2D drift of initially bound reference points, separately from distance to a
    semantic contour; neither the anchor nor its target changes during fitting.
    """
    import torch

    originalDtype = np.asarray(vertices).dtype
    vertices, faces = _validateMesh(vertices, faces)
    cameras = _validateCameras(cameras)
    if (not isinstance(iterations, int) or iterations < 0 or not isinstance(supportRings, int)
            or supportRings < 0 or not 0 <= referenceViewIndex < len(cameras)):
        raise ValueError("Invalid iteration/support/reference configuration")
    parameters = [referenceWeight, maxViewDegradation, learningRate, pixelScale, displacementWeight,
                  laplacianWeight, edgeWeight, minAreaRatio, minNormalCosine,
                  maxRelativeEdgeChange, minImprovementPixels, referenceAnchorWeight]
    if (not np.isfinite(parameters).all() or min(referenceWeight, learningRate, pixelScale) <= 0
            or min(maxViewDegradation, displacementWeight, laplacianWeight, edgeWeight,
                   minImprovementPixels, referenceAnchorWeight) < 0
            or not 0 < minAreaRatio <= 1 or not 0 <= minNormalCosine < 1
            or not 0 < maxRelativeEdgeChange < 1):
        raise ValueError("Invalid refinement safety/regularization parameters")
    if maxDisplacement is None:
        maxDisplacement = .005 * np.ptp(vertices, axis=0).max()
    if not np.isfinite(maxDisplacement) or maxDisplacement <= 0:
        raise ValueError("maxDisplacement must be positive and finite")
    if deformationMode not in ("vertex", "smooth") or not isinstance(smoothControlCount, int) or smoothControlCount < 1:
        raise ValueError("Expected deformationMode vertex/smooth and positive smoothControlCount")
    if smoothRadius is None:
        smoothRadius = .05 * np.ptp(vertices, axis=0).max()
    if not np.isfinite(smoothRadius) or smoothRadius <= 0:
        raise ValueError("smoothRadius must be positive and finite")
    dtype = torch.float64
    tensor = lambda value, kind=dtype: torch.as_tensor(value, dtype=kind, device=device)
    original = tensor(vertices.copy())
    triangleIndices = tensor(faces.copy(), torch.int64)
    prepared = _prepareConstraints(constraints, len(vertices), cameras, device, dtype)
    cameraTensors = [{key: tensor(value) for key, value in camera.items()} for camera in cameras]
    edgeIndices = _meshEdges(faces)
    edges = tensor(edgeIndices, torch.int64)
    activeMask = np.zeros(len(vertices), dtype=bool)
    for constraint in constraints:
        activeMask[np.asarray(constraint["vertexIndices"]).reshape(-1)] = True
    for _ in range(supportRings):
        activeMask[edgeIndices[activeMask[edgeIndices].any(1)].reshape(-1)] = True
    if activeVertexMask is not None:
        suppliedMask = np.asarray(activeVertexMask)
        if suppliedMask.shape != activeMask.shape or suppliedMask.dtype != np.bool_:
            raise ValueError("activeVertexMask must be a boolean V-vector")
        activeMask &= suppliedMask
    active = tensor(activeMask, torch.bool)
    activeIndices = tensor(np.flatnonzero(activeMask), torch.int64)
    restEdges = torch.linalg.vector_norm(original[edges[:, 1]] - original[edges[:, 0]], dim=1)
    restCross = torch.cross(original[triangleIndices[:, 1]] - original[triangleIndices[:, 0]],
                            original[triangleIndices[:, 2]] - original[triangleIndices[:, 0]], dim=1)
    restArea = torch.linalg.vector_norm(restCross, dim=1)
    degrees = torch.zeros(len(vertices), dtype=dtype, device=device)
    degrees.index_add_(0, edges[:, 0], torch.ones(len(edges), dtype=dtype, device=device))
    degrees.index_add_(0, edges[:, 1], torch.ones(len(edges), dtype=dtype, device=device))
    observedViews = sorted({item["viewIndex"] for item in prepared})
    history = []
    referenceBindings = [item for item in prepared if item["viewIndex"] == referenceViewIndex]
    referenceInitialPixels = []
    for item in referenceBindings:
        points = (original[item["vertexIndices"]] * item["barycentric"][..., None]).sum(1)
        pixels, _ = projectTensorPoints(points, **cameraTensors[referenceViewIndex])
        referenceInitialPixels.append(pixels.detach())
    deformationReport = {"mode": deformationMode, "controlCount": len(activeIndices)}
    basis = None
    if deformationMode == "smooth":
        basisValues, basisReport = _buildSmoothDeformationBasis(vertices, activeMask, smoothControlCount, smoothRadius)
        basis = tensor(basisValues)
        deformationReport.update(basisReport)

    def referenceDrift(candidate):
        values = []
        for item, initialPixels in zip(referenceBindings, referenceInitialPixels):
            points = (candidate[item["vertexIndices"]] * item["barycentric"][..., None]).sum(1)
            pixels, _ = projectTensorPoints(points, **cameraTensors[referenceViewIndex])
            values.append(torch.linalg.vector_norm(pixels - initialPixels, dim=-1))
        return torch.cat(values) if values else candidate.new_zeros((0,))

    def evaluate(candidate):
        viewNumerators = {index: candidate.new_zeros(()) for index in observedViews}
        viewErrors = {index: candidate.new_zeros(()) for index in observedViews}
        viewDenominators = {index: candidate.new_zeros(()) for index in observedViews}
        groupErrors, validProjection = [], True
        for item in prepared:
            points = (candidate[item["vertexIndices"]] * item["barycentric"][..., None]).sum(1)
            pixels, depth = projectTensorPoints(points, **cameraTensors[item["viewIndex"]])
            validProjection &= bool((depth > 1e-6).all().detach())
            if "targetDistance" in item:
                residual = _sampleTensorImage(item["targetDistance"], pixels).abs()
                height, width = item["targetDistance"].shape
                validProjection &= bool(((pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
                                         & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)).all().detach())
            else:
                residual = torch.linalg.vector_norm(pixels - item["targetPixels"], dim=-1)
            scaled = residual / pixelScale
            robust = torch.where(scaled <= 1, .5 * scaled.square(), scaled - .5)
            viewIndex, weights = item["viewIndex"], item["weights"]
            viewNumerators[viewIndex] = viewNumerators[viewIndex] + (robust * weights).sum()
            viewErrors[viewIndex] = viewErrors[viewIndex] + (residual * weights).sum()
            viewDenominators[viewIndex] = viewDenominators[viewIndex] + weights.sum()
            groupErrors.append({"viewIndex": viewIndex, "name": item["name"],
                                "sampleCount": len(weights),
                                "meanErrorPixels": float(((residual * weights).sum() / weights.sum()).detach())})
        totalLoss, pixelError, normalization = candidate.new_zeros(()), candidate.new_zeros(()), 0.
        errors = {}
        for viewIndex in observedViews:
            weight = referenceWeight if viewIndex == referenceViewIndex else 1.
            totalLoss = totalLoss + weight * viewNumerators[viewIndex] / viewDenominators[viewIndex]
            value = viewErrors[viewIndex] / viewDenominators[viewIndex]
            pixelError = pixelError + weight * value
            errors[viewIndex] = float(value.detach())
            normalization += weight
        return (totalLoss / max(normalization, 1.), float((pixelError / max(normalization, 1.)).detach()),
                errors, groupErrors, validProjection)

    def regularization(candidate):
        delta = (candidate - original) / maxDisplacement
        neighboring = torch.zeros_like(delta)
        neighboring.index_add_(0, edges[:, 0], delta[edges[:, 1]])
        neighboring.index_add_(0, edges[:, 1], delta[edges[:, 0]])
        laplacian = delta - neighboring / degrees.clamp_min(1)[:, None]
        lengths = torch.linalg.vector_norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], dim=1)
        if len(activeIndices):
            displacement = delta[activeIndices].square().sum(1).mean()
            smoothness = laplacian[activeIndices].square().sum(1).mean()
        else:
            displacement, smoothness = delta.sum() * 0, delta.sum() * 0
        # Include the support's pinned border so concentrated tearing is penalized.
        localEdges = active[edges].any(1)
        distortion = ((lengths[localEdges] / restEdges[localEdges] - 1).square().mean()
                      if bool(localEdges.any()) else delta.sum() * 0)
        anchor = referenceDrift(candidate).square().mean() / pixelScale ** 2 if referenceBindings else delta.sum() * 0
        return (displacementWeight * displacement + laplacianWeight * smoothness + edgeWeight * distortion
                + referenceAnchorWeight * anchor)

    def safety(candidate):
        cross = torch.cross(candidate[triangleIndices[:, 1]] - candidate[triangleIndices[:, 0]],
                            candidate[triangleIndices[:, 2]] - candidate[triangleIndices[:, 0]], dim=1)
        area = torch.linalg.vector_norm(cross, dim=1)
        cosine = (cross * restCross).sum(1) / (area * restArea).clamp_min(1e-25)
        ratios = torch.linalg.vector_norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], dim=1) / restEdges
        movement = torch.linalg.vector_norm(candidate - original, dim=1)
        summary = {"minimumAreaRatio": float((area / restArea).min()),
                   "minimumNormalCosine": float(cosine.min()),
                   "maximumRelativeEdgeChange": float((ratios - 1).abs().max()),
                   "maximumDisplacement": float(movement.max())}
        safe = (bool(torch.isfinite(candidate).all()) and summary["minimumAreaRatio"] >= minAreaRatio
                and summary["minimumNormalCosine"] >= minNormalCosine
                and summary["maximumRelativeEdgeChange"] <= maxRelativeEdgeChange
                and summary["maximumDisplacement"] <= maxDisplacement * (1 + 1e-7))
        return safe, summary

    with torch.no_grad():
        initialLoss, initialError, initialViewErrors, initialGroups, initialValid = evaluate(original)
    if not initialValid:
        raise ValueError("Initial constraints project behind cameras or outside target distance images")
    bestVertices = original.clone()
    bestObjective, bestIteration, completedIterations = float(initialLoss), 0, 0
    rejectionCounts = {"meshSafety": 0, "projection": 0, "viewDegradation": 0, "nonfinite": 0}
    if prepared and len(activeIndices) and iterations:
        # Only supported vertices are parameters; pinned vertices remain bit-exact.
        parameterCount = len(activeIndices) if basis is None else basis.shape[1]
        offsets = torch.zeros((parameterCount, 3), dtype=dtype, device=device, requires_grad=True)
        bestOffsets = offsets.detach().clone()
        def displacedVertices():
            delta = offsets if basis is None else basis @ offsets
            return original.index_add(0, activeIndices, delta * maxDisplacement)

        optimizer = torch.optim.Adam([offsets], lr=learningRate)
        for iteration in range(1, iterations + 1):
            completedIterations = iteration
            optimizer.zero_grad(set_to_none=True)
            candidate = displacedVertices()
            dataLoss, _, _, _, _ = evaluate(candidate)
            objective = dataLoss + regularization(candidate)
            if not bool(torch.isfinite(objective)):
                rejectionCounts["nonfinite"] += 1
                break
            objective.backward()
            if offsets.grad is None or not bool(torch.isfinite(offsets.grad).all()):
                rejectionCounts["nonfinite"] += 1
                break
            optimizer.step()
            with torch.no_grad():
                offsets.div_(torch.linalg.vector_norm(offsets, dim=1).clamp_min(1)[:, None])
                candidate = displacedVertices()
                currentLoss, error, viewErrors, _, projectionValid = evaluate(candidate)
                currentObjective = float(currentLoss + regularization(candidate))
                safe, safetyReport = safety(candidate)
                viewsSafe = all(viewErrors[index] <= initialViewErrors[index] + maxViewDegradation
                                for index in observedViews)
                if not safe:
                    rejectionCounts["meshSafety"] += 1
                if not projectionValid:
                    rejectionCounts["projection"] += 1
                if not viewsSafe:
                    rejectionCounts["viewDegradation"] += 1
                accepted = (safe and projectionValid and viewsSafe and np.isfinite(currentObjective)
                            and currentObjective < bestObjective and error < initialError - minImprovementPixels)
                if accepted:
                    bestVertices, bestObjective = candidate.clone(), currentObjective
                    bestIteration = iteration
                    bestOffsets = offsets.clone()
                if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
                    history.append({"iteration": iteration, "objective": currentObjective,
                                    "meanErrorPixels": error, "viewErrors": viewErrors,
                                    "safe": bool(safe and projectionValid and viewsSafe), "accepted": bool(accepted)})
                if not safe or not projectionValid:
                    # Return to the last accepted state and damp the optimizer.
                    # A rejected excursion must not seed progressively worse steps.
                    offsets.copy_(bestOffsets)
                    optimizer.state.clear()
                    for group in optimizer.param_groups:
                        group["lr"] *= .5
    with torch.no_grad():
        _, finalError, finalViewErrors, finalGroups, _ = evaluate(bestVertices)
        _, safetyReport = safety(bestVertices)
        referenceDrifts = referenceDrift(bestVertices).cpu().numpy()
        driftReport = {"sampleCount": len(referenceDrifts),
                       "meanPixels": float(referenceDrifts.mean()) if len(referenceDrifts) else None,
                       "p95Pixels": float(np.percentile(referenceDrifts, 95)) if len(referenceDrifts) else None,
                       "maxPixels": float(referenceDrifts.max()) if len(referenceDrifts) else None,
                       "meaning": "full XY displacement of fixed reference surface bindings from their initial projections"}
    status = "accepted" if bestIteration else "unchanged"
    resultDtype = originalDtype if np.issubdtype(originalDtype, np.floating) else np.float64
    result = bestVertices.detach().cpu().numpy().astype(resultDtype)
    # Exact original values at all pinned vertices; no faces/UV arrays are modified.
    result[~activeMask] = vertices[~activeMask].astype(resultDtype)
    return {"vertices": result,
            "diagnostics": {"status": status, "bestIteration": bestIteration,
                            "iterationsRequested": iterations, "iterationsCompleted": completedIterations,
                            "bestRegularizedObjective": bestObjective, "device": str(device),
                            "activeVertexCount": int(activeMask.sum()), "constraintCount": len(prepared),
                            "initialMeanErrorPixels": initialError, "finalMeanErrorPixels": finalError,
                            "initialViewErrors": initialViewErrors, "finalViewErrors": finalViewErrors,
                            "initialGroups": initialGroups, "finalGroups": finalGroups,
                            "maxDisplacement": float(maxDisplacement), "maxViewDegradation": float(maxViewDegradation),
                            "referenceViewIndex": int(referenceViewIndex), "referenceWeight": float(referenceWeight),
                            "referenceProjectionDrift": driftReport, "deformation": deformationReport,
                            "regularization": {"displacementWeight": float(displacementWeight),
                                               "laplacianWeight": float(laplacianWeight),
                                               "edgeWeight": float(edgeWeight),
                                               "referenceAnchorWeight": float(referenceAnchorWeight)},
                            "safety": safetyReport, "rejections": rejectionCounts,
                            "topologySha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                            "vertexCount": len(vertices), "faceCount": len(faces), "camerasFrozen": True,
                            "visibilityFrozen": True, "uvAndVertexOrderPreserved": True,
                            "limitation": "Frozen semantic constraints; no photometric renderer or global self-intersection test"},
            "history": history}
