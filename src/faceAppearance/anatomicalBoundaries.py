"""Reusable template boundary indices and conservative photo constraints.

Topology establishes that an edge borders a mesh hole; it does not establish
an anatomical name. Names are explicit, reviewable annotations. Asset matching
checks vertex numbering and oriented face order before reusing any indices.
No face parser, rasterizer or trainable model is loaded by this module.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates


def _validateMesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.dtype.kind not in "iu"
            or faces.min() < 0 or faces.max() >= len(vertices)
            or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
        raise ValueError("Expected finite Vx3 vertices and valid integer Fx3 triangle indices")
    return vertices, faces.astype(np.int64, copy=False)


def meshTopologyHash(vertexCount, faces):
    """Hash numbered, ordered triangles; vertex coordinates may change safely."""
    faces = np.asarray(faces)
    if (not isinstance(vertexCount, (int, np.integer)) or isinstance(vertexCount, bool)
            or vertexCount < 3 or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.dtype.kind not in "iu"
            or faces.min() < 0 or faces.max() >= vertexCount):
        raise ValueError("Invalid topology")
    digest = hashlib.sha256()
    digest.update(b"numberedOrientedTriangleTopologyV1\0")
    digest.update(np.asarray([vertexCount, len(faces)], dtype="<i8").tobytes())
    digest.update(np.asarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def extractMeshBoundaryLoops(vertices, faces):
    """Extract deterministic ordered closed loops; reject ambiguous topology.

    Loops start at their smallest numbered vertex, take the smaller neighbor
    first, then sort by decreasing size and first vertex. Winding is therefore
    deterministic but is not claimed to be an anatomical or surface orientation.
    Non-manifold edges and boundary branches fail rather than producing a
    guessed contour. A closed manifold mesh legitimately returns no loops.
    """
    vertices, faces = _validateMesh(vertices, faces)
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Duplicate triangles make boundary interpretation ambiguous")
    allEdges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                      faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(allEdges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Non-manifold edge belongs to more than two triangles")
    boundaryEdges = edges[counts == 1]
    adjacency = {}
    for first, second in boundaryEdges.tolist():
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Boundary branches do not form unambiguous closed loops")
    remaining, loops = set(adjacency), []
    while remaining:
        start = min(remaining)
        ordered, previous, current = [start], start, min(adjacency[start])
        remaining.remove(start)
        while current != start:
            if current not in remaining:
                raise ValueError("Boundary loop traverses an already visited vertex")
            ordered.append(current)
            remaining.remove(current)
            previous, current = current, next(vertex for vertex in adjacency[current] if vertex != previous)
        loops.append(np.asarray(ordered, dtype=np.int64))
    loops.sort(key=lambda loop: (-len(loop), int(loop[0])))
    return {"loops": loops, "topologySha256": meshTopologyHash(len(vertices), faces),
            "diagnostics": {"vertexCount": len(vertices), "triangleCount": len(faces),
                            "boundaryEdgeCount": len(boundaryEdges), "loopCount": len(loops),
                            "loopVertexCounts": [len(loop) for loop in loops],
                            "nonManifoldEdgeCount": 0, "closedBoundaryLoops": True}}


def createBoundaryAsset(vertices, faces, annotations, *, source="", reviewStatus="unreviewed"):
    """Create an editable JSON-compatible asset; annotation keys are loop indices.

    Annotations contain ``name`` and may contain ``description``. A caller may
    annotate only some loops. These names are supplied by the caller, never
    inferred from vertex count or photographic semantics by this function.
    """
    extracted = extractMeshBoundaryLoops(vertices, faces)
    names, records = set(), []
    for loopIndex, annotation in sorted(annotations.items()):
        name = annotation.get("name")
        if (not isinstance(loopIndex, (int, np.integer)) or not 0 <= loopIndex < len(extracted["loops"])
                or not isinstance(name, str) or not name.strip() or name in names):
            raise ValueError("Annotations need distinct nonempty names and existing loop indices")
        names.add(name)
        records.append({"name": name, "loopIndex": int(loopIndex),
                        "vertexIndices": extracted["loops"][loopIndex].tolist(),
                        "description": str(annotation.get("description", ""))})
    return {"schemaVersion": 1, "topologySha256": extracted["topologySha256"],
            "templateGeometrySha256": hashlib.sha256(np.asarray(vertices, dtype="<f8").tobytes()).hexdigest(),
            "vertexCount": len(vertices), "source": str(source), "reviewStatus": str(reviewStatus),
            "indexConvention": "zeroBasedOriginalVertexOrder", "boundaries": records,
            "diagnostics": extracted["diagnostics"]}


def validateBoundaryAsset(vertices, faces, asset):
    """Fail closed on wrong topology, invented paths, partial or duplicate loops."""
    extracted = extractMeshBoundaryLoops(vertices, faces)
    if (asset.get("schemaVersion") != 1 or asset.get("vertexCount") != len(vertices)
            or asset.get("indexConvention") != "zeroBasedOriginalVertexOrder"
            or asset.get("topologySha256") != extracted["topologySha256"]):
        raise ValueError("Boundary asset does not match numbered mesh topology")
    names, loopIds, result = set(), set(), {}
    for record in asset.get("boundaries", []):
        name, index = record.get("name"), record.get("loopIndex")
        values = np.asarray(record.get("vertexIndices"))
        if (not isinstance(name, str) or not name.strip() or name in names
                or not isinstance(index, int) or not 0 <= index < len(extracted["loops"])
                or index in loopIds or values.dtype.kind not in "iu"
                or not np.array_equal(values, extracted["loops"][index])):
            raise ValueError("Asset annotations must reference distinct exact ordered topology loops")
        names.add(name)
        loopIds.add(index)
        result[name] = values.astype(np.int64)
    return result


def surfacePathBindings(vertices, faces, asset):
    """Validate reviewed mesh-edge feature paths separately from true holes.

    A visible lip seam can lie on the surface ahead of an occluded oral hole.
    Such a feature must be explicitly annotated, not invented from topology.
    The asset contains topologySha256 and surfacePaths: each record has name,
    zero-based vertexIndices and explicit closed boolean. This function makes
    one-vertex bindings suitable for independent photo contour supervision.
    """
    vertices, faces = _validateMesh(vertices, faces)
    if asset.get("topologySha256") != meshTopologyHash(len(vertices), faces):
        raise ValueError("Surface feature asset does not match numbered mesh topology")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edgeSet = set(map(tuple, edges.tolist()))
    result = {}
    for record in asset.get("surfacePaths", []):
        name, closed = record.get("name"), record.get("closed")
        indices = np.asarray(record.get("vertexIndices"))
        if (not isinstance(name, str) or not name.strip() or name in result or not isinstance(closed, bool)
                or indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < (3 if closed else 2)
                or indices.min() < 0 or indices.max() >= len(vertices) or len(np.unique(indices)) != len(indices)):
            raise ValueError("Invalid explicitly annotated surface path")
        pairs = list(zip(indices[:-1], indices[1:]))
        if closed:
            pairs.append((indices[-1], indices[0]))
        if any(tuple(sorted(pair)) not in edgeSet for pair in pairs):
            raise ValueError("Surface feature path must follow existing connected mesh edges")
        result[name] = {"vertexIndices": indices[:, None].copy(), "barycentric": np.ones((len(indices), 1))}
    return result


def boundaryVisibilityDiagnostics(vertices, faces, cameras, imageShapes, boundaryLoops, *,
                                   minimumCosine=.05, depthToleranceFraction=.005):
    """Separate hole-edge visibility gates and inspect incident face centroids.

    A boundary vertex may round into a hole and have no raster support. This
    diagnostic distinguishes that case from a supported depth mismatch and
    backfacing geometry; incident triangle centroids provide an interior check.
    It does not relax occlusion or create geometry constraints.
    """
    from .camera import projectPoints
    from .texture import rasterizeDepth, vertexNormals

    vertices, faces = _validateMesh(vertices, faces)
    if (len(cameras) != len(imageShapes) or not 0 <= minimumCosine < 1
            or not np.isfinite(depthToleranceFraction) or depthToleranceFraction <= 0):
        raise ValueError("Invalid visibility diagnostic configuration")
    normals = vertexNormals(vertices, faces)
    faceNormals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                           vertices[faces[:, 2]]-vertices[faces[:, 0]])
    faceNormals /= np.maximum(np.linalg.norm(faceNormals, axis=1, keepdims=True), 1e-15)
    faceCenters = vertices[faces].mean(1)
    tolerance = max(float(np.ptp(vertices, axis=0).max())*depthToleranceFraction, 1e-8)
    groups = []
    for viewIndex, (camera, imageShape) in enumerate(zip(cameras, imageShapes)):
        height, width = imageShape
        rotation, translation = np.asarray(camera["rotation"]), np.asarray(camera["translation"])
        cameraVertices = vertices @ rotation.T + translation
        depthMap = rasterizeDepth(cameraVertices, faces, camera["intrinsics"], height, width)

        def inspect(points, pointNormals):
            cameraPoints = points @ rotation.T + translation
            directions = -cameraPoints / np.maximum(np.linalg.norm(cameraPoints, axis=1, keepdims=True), 1e-15)
            cosine = np.sum((pointNormals @ rotation.T)*directions, axis=1)
            pixels, depth = projectPoints(points, camera["intrinsics"], rotation, translation)
            finite = np.isfinite(pixels).all(1)
            inside = (finite & (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width-1)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height-1))
            residual = np.full(len(points), np.inf)
            indices = np.flatnonzero(inside)
            rounded = np.floor(pixels[indices]+.5).astype(int)
            residual[indices] = abs(depthMap[rounded[:, 1], rounded[:, 0]] - depth[indices])
            rasterSupport = np.isfinite(residual)
            return {"count": len(points), "insidePhotoPositiveDepth": int(inside.sum()),
                    "frontFacingCount": int((inside & (cosine > minimumCosine)).sum()),
                    "finiteRasterSupportCount": int(rasterSupport.sum()),
                    "depthTolerancePassedCount": int((residual <= tolerance).sum()),
                    "visibleCount": int((inside & (cosine > minimumCosine) & (residual <= tolerance)).sum()),
                    "normalCosineMinMedianMax": np.percentile(cosine, [0, 50, 100]).tolist(),
                    "supportedAbsoluteDepthResidualMinMedianMax": np.percentile(residual[rasterSupport], [0, 50, 100]).tolist()
                        if rasterSupport.any() else None}

        for name, loop in boundaryLoops.items():
            indices = np.asarray(loop)
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or indices.min() < 0 or indices.max() >= len(vertices)):
                raise ValueError("Invalid boundary indices for visibility diagnostic")
            incidentFaces = np.flatnonzero(np.isin(faces, indices).any(1))
            groups.append({"viewIndex": viewIndex, "name": str(name),
                           "boundaryVertices": inspect(vertices[indices], normals[indices]),
                           "incidentFaceCentroids": inspect(faceCenters[incidentFaces], faceNormals[incidentFaces])})
    return {"depthTolerance": tolerance, "depthToleranceFraction": depthToleranceFraction,
            "minimumCosine": minimumCosine, "groups": groups,
            "meaning": "Gate counts independent; a failed depth test with finite support differs from sampling a hole"}


def lipInterfaceMask(labelMap, confidence, *, upperLabel=12, lowerLabel=13, minConfidence=.55):
    """Return directly touching upper/lower lip pixels, never the outer lip rim.

    Four-neighbor contact avoids bridging a mouth opening, skin, or unknown
    region. A pixel and its opposite-class neighbor must both pass confidence.
    This parser interface is an uncertain observation, not a measured lip seam.
    """
    labels, confidence = np.asarray(labelMap), np.asarray(confidence)
    if (labels.ndim != 2 or labels.dtype.kind not in "iu" or confidence.shape != labels.shape
            or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1))
            or not np.isfinite(minConfidence) or not 0 <= minConfidence <= 1 or upperLabel == lowerLabel):
        raise ValueError("Expected integer HW labels and finite [0,1] confidence")
    upper = (labels == upperLabel) & (confidence >= minConfidence)
    lower = (labels == lowerLabel) & (confidence >= minConfidence)
    return (upper & binary_dilation(lower)) | (lower & binary_dilation(upper))


def rasterizeBoundaryAnnotations(imageShape, annotations):
    """Convert explicit photo polylines into targets without guessing anatomy.

    Each record needs unique ``name``, ``points`` (XY original-photo pixels),
    explicit boolean ``closed``, and optional ``confidence`` in (0,1]. Closed
    nostril/eye rims and open mouth centerlines can use this common contract.
    Rasterization has one-pixel precision; source polyline coordinates should
    remain in the annotation JSON for review and subsequent subpixel fitting.
    """
    import cv2

    if (len(imageShape) != 2 or any(not isinstance(value, (int, np.integer)) or value < 3 for value in imageShape)):
        raise ValueError("Expected positive integer image height and width")
    height, width = imageShape
    result = {}
    for record in annotations:
        name, closed = record.get("name"), record.get("closed")
        points = np.asarray(record.get("points"), float)
        confidence = record.get("confidence", 1.)
        if (not isinstance(name, str) or not name.strip() or name in result or not isinstance(closed, bool)
                or points.ndim != 2 or points.shape[1] != 2 or len(points) < (3 if closed else 2)
                or not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points[:, 0] > width-1) or np.any(points[:, 1] > height-1)
                or not isinstance(confidence, (float, int)) or isinstance(confidence, bool)
                or not np.isfinite(confidence) or not 0 < confidence <= 1):
            raise ValueError("Invalid explicit photo contour annotation")
        pixels = np.floor(points+.5).astype(np.int32)
        if len(np.unique(pixels, axis=0)) < (3 if closed else 2):
            raise ValueError("Contour collapses after one-pixel rasterization")
        canvas = np.zeros((height, width), np.uint8)
        cv2.polylines(canvas, [pixels], closed, 1, thickness=1, lineType=cv2.LINE_8)
        mask = canvas.astype(bool)
        result[name] = {"contourMask": mask, "confidence": mask.astype(np.float32)*confidence}
    return result


def pairBoundaryRims(projectedVertices, loop, *, sampleCount=24, endpointMargin=.1):
    """Bind a thin hole's centerline to fixed opposite-rim vertex pairs.

    This is a one-time binding in the reference photo: the same vertex pairs
    are reused in every view. Fit their midpoints, not both rims to the same
    line, so fitting a closed-mouth seam does not explicitly collapse a hole.
    Principal-axis bins pick actual vertices, not a guessed anatomical path.
    Callers must verify the supplied loop is a thin mouth aperture. Returned
    pairs are features spanning the aperture, not points on the mesh surface.
    """
    pixels, loop = np.asarray(projectedVertices, float), np.asarray(loop)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all()
            or loop.ndim != 1 or len(loop) < 6 or loop.dtype.kind not in "iu"
            or loop.min() < 0 or loop.max() >= len(pixels) or len(np.unique(loop)) != len(loop)
            or not isinstance(sampleCount, int) or sampleCount < 3
            or not np.isfinite(endpointMargin) or not 0 < endpointMargin < .4):
        raise ValueError("Invalid projected loop or centerline sampling parameters")
    points = pixels[loop]
    centered = points - points.mean(0)
    _, singularValues, axes = np.linalg.svd(centered, full_matrices=False)
    if singularValues[0] < 1e-6 or singularValues[1] / singularValues[0] > .35:
        raise ValueError("Centerline pairing requires a thin, nondegenerate aperture")
    principal = axes[0] * (1 if axes[0, 0] >= 0 else -1)
    along = centered @ principal
    first, last = int(np.argmin(along)), int(np.argmax(along))
    forward = np.arange(first, first + len(loop)) % len(loop)
    forward = forward[:int(np.flatnonzero(forward == last)[0]) + 1]
    backward = np.arange(first, first - len(loop), -1) % len(loop)
    backward = backward[:int(np.flatnonzero(backward == last)[0]) + 1]
    span = float(np.ptp(along))
    targets = np.linspace(along.min() + endpointMargin * span, along.max() - endpointMargin * span, sampleCount)
    pairs = []
    for target in targets:
        a = forward[np.argmin(abs(along[forward] - target))]
        b = backward[np.argmin(abs(along[backward] - target))]
        if a != b and abs(along[a] - along[b]) <= .08 * span:
            pair = tuple(sorted((int(loop[a]), int(loop[b]))))
            if pair not in pairs:
                pairs.append(pair)
    if len(pairs) < 3:
        raise ValueError("Aperture cannot provide enough stable opposite-rim pairs")
    return {"vertexIndices": np.asarray(pairs, np.int64),
            "barycentric": np.full((len(pairs), 2), .5),
            "diagnostics": {"sampleCount": len(pairs), "referenceAxis": principal.tolist(),
                            "minorMajorRatio": float(singularValues[1] / singularValues[0]),
                            "binding": "fixed opposite topology rim pairs; centerline spans aperture"}}


def buildAnatomicalConstraints(vertices, cameras, bindings, targetContours, vertexVisibility, *,
                               maxTargetDistance=24., minimumSamples=4):
    """Gate fixed anatomical bindings against explicit per-view contour targets.

    ``bindings`` maps names to vertexIndices and barycentric arrays (Mx1/2/3).
    ``targetContours`` is a list (camera order) of dictionaries keyed by binding
    name, each holding HW ``contourMask`` and HW ``confidence``. Missing targets
    yield no constraint. Nearest contour confidence and maximum target distance
    are gates; no semantic name or 3D visibility is guessed here. A point needs
    all supporting vertices visible in the supplied frozen depth-tested mask.
    """
    from .camera import projectPoints

    vertices = np.asarray(vertices)
    visibility = np.asarray(vertexVisibility)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or len(cameras) != len(targetContours) or not len(cameras)
            or visibility.shape != (len(cameras), len(vertices)) or visibility.dtype != np.bool_
            or not np.isfinite(maxTargetDistance) or maxTargetDistance <= 0
            or not isinstance(minimumSamples, int) or minimumSamples < 1):
        raise ValueError("Invalid mesh, target or visibility dimensions")
    constraints, reports = [], []
    for name, binding in bindings.items():
        indices, barycentric = np.asarray(binding["vertexIndices"]), np.asarray(binding["barycentric"], float)
        if (indices.ndim != 2 or indices.shape[1] not in (1, 2, 3) or indices.dtype.kind not in "iu"
                or not len(indices) or indices.min() < 0 or indices.max() >= len(vertices)
                or barycentric.shape != indices.shape or not np.isfinite(barycentric).all()
                or np.any(barycentric < 0) or not np.allclose(barycentric.sum(1), 1)):
            raise ValueError("Invalid anatomical binding")
        points = (vertices[indices] * barycentric[..., None]).sum(1)
        for viewIndex, (camera, targets) in enumerate(zip(cameras, targetContours)):
            report = {"name": str(name), "viewIndex": viewIndex, "bindingCount": len(indices), "sampleCount": 0}
            reports.append(report)
            if name not in targets:
                report["reason"] = "noIndependentPhotoTarget"
                continue
            target = targets[name]
            mask, confidence = np.asarray(target["contourMask"]), np.asarray(target["confidence"], float)
            if (mask.ndim != 2 or min(mask.shape) < 3 or mask.dtype != np.bool_
                    or confidence.shape != mask.shape or not np.isfinite(confidence).all()
                    or np.any((confidence < 0) | (confidence > 1))):
                raise ValueError("Expected boolean HW contour and [0,1] confidence")
            mask = mask & (confidence > 0)
            if not mask.any():
                report["reason"] = "emptyOrUncertainPhotoTarget"
                continue
            distance, nearest = distance_transform_edt(~mask, return_indices=True)
            nearestConfidence = confidence[nearest[0], nearest[1]]
            pixels, depth = projectPoints(points, camera["intrinsics"], camera["rotation"], camera["translation"])
            height, width = mask.shape
            inside = ((depth > 0) & (pixels[:, 0] >= 1) & (pixels[:, 0] <= width-2)
                      & (pixels[:, 1] >= 1) & (pixels[:, 1] <= height-2) & np.isfinite(pixels).all(1))
            finitePixels = np.where(np.isfinite(pixels), pixels, 0)
            residual = map_coordinates(distance, finitePixels[:, ::-1].T, order=1, mode="nearest")
            weights = map_coordinates(nearestConfidence, finitePixels[:, ::-1].T, order=1, mode="nearest")
            valid = inside & visibility[viewIndex, indices].all(1) & (residual <= maxTargetDistance) & (weights > 0)
            report.update({"sampleCount": int(valid.sum()), "targetPixelCount": int(mask.sum())})
            if valid.sum() < minimumSamples:
                report["reason"] = "insufficientVisibleNearbySamples"
                continue
            report.update({"reason": "accepted", "initialMeanDistancePixels": float(residual[valid].mean()),
                           "initialP95DistancePixels": float(np.percentile(residual[valid], 95))})
            constraints.append({"name": str(name), "viewIndex": viewIndex,
                                "vertexIndices": indices[valid].copy(), "barycentric": barycentric[valid].copy(),
                                "weights": weights[valid], "targetDistance": distance.astype(np.float32)})
    return {"constraints": constraints, "diagnostics": {"groups": reports,
            "constraintCount": len(constraints), "visibilityFrozen": True,
            "targets": "independent photo contours; anatomy names and bindings supplied explicitly"}}
