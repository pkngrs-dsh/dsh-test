"""Conservative, observation-scaled chin shortening on a client surface.

This adapter is deliberately allowed to return the input unchanged.  Photo
landmarks are detector observations, and even ray-visible mesh edges are only
candidates for the anatomical jaw.  No subject paths, fixed camera index,
working-image coordinates, or canonical-unit depth cutoffs are used here.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .clientChinRefinement import visibleLowerJawConstraints, _ids, _json
from .clientGeometrySafety import intersectionPairs
from .frontLandmarkRefinement import checkFrontGeometry
from .geometryRefinement import _validateCameras, _validateMesh


# Pixel and length thresholds below are ratios of each view's observed
# eye-midpoint-to-chin distance, or its corresponding reference-camera length.
_LIMITS = {
    "minimumRawFanPeak": .25,
    "maximumCropVariationFaceFraction": .022,
    "minimumReferenceSymmetry": .52,
    "maximumReferencePoseRadians": .55,
    "maximumBindingMedianErrorFaceFraction": .045,
    "maximumBindingP90ErrorFaceFraction": .08,
    "maximumTargetDistanceFaceFraction": .17,
    "maximumTangentErrorFaceFraction": .026,
    "maximumChainSlackFaceFraction": .05,
    "minimumExcessLengthFaceFraction": .012,
    "minimumPositiveResidualFraction": .8,
    "minimumImprovementFaceFraction": .007,
    "acceptableResidualFaceFraction": .035,
    "maximumCrossViewIncreaseFaceFraction": .006,
    "maximumCrossViewPointIncreaseFaceFraction": .016,
    "maximumAmplitudeFaceFraction": .10,
    "maximumDisplacementFaceFraction": .10,
    "maximumRelativeEdgeChange": .25,
    "minimumAreaRatio": .4,
    "minimumNormalCosine": .3,
    "maximumNormalRotationRadians": .25,
    "maximumDihedralChangeRadians": .18,
    "maximumDihedralChangeP95Radians": .055,
    "maximumLayerMovementDifferenceFaceFraction": .010,
    "maximumRelativeLayerMovementDifference": .32,
}


def _project(vertices, camera):
    p = np.asarray(vertices) @ camera["rotation"].T + camera["translation"]
    h = p @ camera["intrinsics"].T
    return h[:, :2] / h[:, 2:], p[:, 2]


def _transform(points, matrix):
    h = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return h[:, :2] / h[:, 2:]


def _smooth(value):
    t = np.clip(value, 0., 1.)
    return t*t*(3.-2.*t)


def _image_shape(observations, raw_camera, index, count):
    for name in ("workingImageShape", "imageShape", "imageShapes"):
        if name in observations:
            shape = np.asarray(observations[name])
            if shape.shape == (count, 2):
                shape = shape[index]
            if shape.shape == (2,) and np.isfinite(shape).all() and (shape >= 2).all():
                return tuple(map(int, shape)), "observation_"+name
    for name in ("imageShape", "imageSize"):
        if name in raw_camera:
            shape = np.asarray(raw_camera[name])
            if shape.shape == (2,) and np.isfinite(shape).all() and (shape >= 2).all():
                if name == "imageSize":
                    shape = shape[::-1]  # Camera imageSize convention is W,H.
                return tuple(map(int, shape)), "camera_"+name
    k = np.asarray(raw_camera["intrinsics"])
    shape = (int(np.ceil(2*k[1, 2])), int(np.ceil(2*k[0, 2])))
    if min(shape) < 2:
        raise ValueError("No usable image dimensions or centered principal point")
    return shape, "inferred_from_centered_principal_point"


def _view_frame(camera, landmarks, shape):
    """Rotate camera about its optical axis, preserving metric camera depth."""
    k = camera["intrinsics"]
    eyes = np.asarray([landmarks[36:42].mean(0), landmarks[42:48].mean(0)])
    eye_rays = _transform(eyes, np.linalg.inv(k))
    horizontal = eye_rays[1]-eye_rays[0]
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.array([-horizontal[1], horizontal[0]])
    q = np.eye(3)
    q[0, :2], q[1, :2] = horizontal, vertical
    focal = float(np.sqrt(k[0, 0]*k[1, 1]))
    new_k = np.diag([focal, focal, 1.])
    h = new_k @ q @ np.linalg.inv(k)
    height, width = shape
    corners = _transform(np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]), h)
    offset = -np.floor(corners.min(0)) + 1.
    new_k[:2, 2] = offset
    h = new_k @ q @ np.linalg.inv(k)
    bounds = _transform(np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]), h)
    new_shape = tuple((np.ceil(bounds.max(0))+2).astype(int)[::-1])
    frame = {"intrinsics": new_k, "rotation": q @ camera["rotation"],
             "translation": q @ camera["translation"]}
    return frame, _transform(landmarks, h), h, new_shape


def _binding(vertices, supplied):
    if supplied is None:
        return None
    ids = np.asarray(supplied["ibugIds"], int)
    indices = np.asarray(supplied["vertexIndices"])
    weights = np.asarray(supplied.get("barycentricWeights", supplied.get("barycentric")), float)
    if (ids.ndim != 1 or indices.ndim != 2 or indices.shape != weights.shape or len(indices) != len(ids)
            or indices.dtype.kind not in "iu" or indices.min() < 0 or indices.max() >= len(vertices)
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.allclose(weights.sum(1), 1.) or len(np.unique(ids)) != len(ids)):
        raise ValueError("Invalid semantic eye/mouth landmarkBinding")
    keep = (ids >= 36) & (ids <= 67)
    return {"ids": ids[keep], "indices": indices[keep],
            "points": (vertices[indices[keep]]*weights[keep, :, None]).sum(1)}


def _pose(binding, camera):
    if binding is None:
        return None
    ids, points = binding["ids"], binding["points"]
    groups = [points[(ids >= a) & (ids <= b)] for a, b in [(36, 41), (42, 47), (48, 59)]]
    if any(len(group) < 2 for group in groups):
        return None
    left, right, mouth = [group.mean(0) for group in groups]
    x = right-left
    x /= np.linalg.norm(x)
    y = mouth-(left+right)/2
    y -= (y@x)*x
    y /= np.linalg.norm(y)
    n = np.cross(x, y)
    view = camera["rotation"][2]
    angle = float(np.arccos(np.clip(abs(n@view), 0., 1.)))
    return {"totalAngleRadians": angle,
            "yawProxyRadians": float(np.arctan2(abs(x@view), abs(n@view))),
            "pitchProxyRadians": float(np.arctan2(abs(y@view), abs(n@view))),
            "meaning": "eye/mouth surface-plane pose proxy, not calibrated head pose"}


def _prepare_views(vertices, raw_cameras, cameras, observations, binding):
    points = np.asarray(observations["faceLandmarksWorking"], float)
    peaks = np.asarray(observations["faceRawPeakScores"], float)
    mask = np.asarray(observations["landmarkCandidateValidMask"], bool)
    variation = np.asarray(observations["maximumCropVariationPixels"], float)
    n = len(cameras)
    if points.shape != (n, 68, 2) or peaks.shape != (n, 68) or mask.shape != (n, 68) or variation.shape != (n, 68):
        raise ValueError("Expected full-face observations for every saved camera")
    views, reports = [], []
    # A whole jaw acceptedViewMask may exclude a view for remote lateral points;
    # the central support's per-landmark validity is the applicable gate here.
    required = np.r_[6:11, 36:48, 48, 54, 57]
    for index, (camera, raw, p, score, good, crop) in enumerate(zip(cameras, raw_cameras, points, peaks, mask, variation)):
        row = {"viewIndex": index, "status": "rejected"}
        reports.append(row)
        try:
            if not raw.get("valid", True):
                raise ValueError("camera_not_accepted")
            if (not np.isfinite(p[required]).all() or not np.isfinite(score[required]).all()
                    or not np.isfinite(crop[required]).all() or not good[required].all()
                    or (score[required] < _LIMITS["minimumRawFanPeak"]).any()):
                raise ValueError("unreliable_eye_mouth_or_central_jaw_observations")
            shape, shape_source = _image_shape(observations, raw, index, n)
            if ((p[required] < 0).any() or (p[required] >= [shape[1], shape[0]]).any()):
                raise ValueError("required_landmarks_outside_original_working_image")
            frame, local, homography, frame_shape = _view_frame(camera, p, shape)
            eye_left, eye_right = local[36:42].mean(0), local[42:48].mean(0)
            eye_mid = (eye_left+eye_right)/2
            scale = float(np.linalg.norm(local[8]-eye_mid))
            interocular = float(np.linalg.norm(eye_right-eye_left))
            if not np.isfinite(scale) or scale <= 0 or not .24 < interocular/scale < .85:
                raise ValueError("implausible_eye_to_chin_proportions")
            # Crop variability is transformed conservatively by the largest
            # singular value of this affine, depth-preserving image rotation.
            crop_factor = float(np.linalg.svd(homography[:2, :2], compute_uv=False)[0])
            if (crop[required]*crop_factor > _LIMITS["maximumCropVariationFaceFraction"]*scale).any():
                raise ValueError("unstable_central_jaw_or_feature_crop")
            if not np.all(np.diff(local[6:11, 0]) > .01*scale):
                raise ValueError("central_jaw_not_monotone_in_eye_aligned_frame")
            lower_mouth = float(local[[48, 54, 57], 1].max())
            gap = float(local[8, 1]-lower_mouth)
            if not .14 < gap/scale < .58 or lower_mouth <= eye_mid[1]+.25*scale:
                raise ValueError("implausible_lower_mouth_to_chin_gap")
            if (good[62] and good[66] and min(score[62], score[66]) >= _LIMITS["minimumRawFanPeak"]
                    and np.linalg.norm(local[66]-local[62]) > .10*scale):
                raise ValueError("open_mouth_expression_not_supported")
            mouth_center = local[[48, 54]].mean(0)
            asymmetry = (abs(float(mouth_center[0]-eye_mid[0]))/interocular
                         + abs(float(local[8, 0]-eye_mid[0]))/interocular)
            half_widths = np.abs(local[[6, 10], 0]-local[8, 0])
            asymmetry += .5*abs(float(half_widths[0]-half_widths[1]))/max(float(half_widths.sum()), 1e-12)
            symmetry = float(np.exp(-2.*asymmetry))
            pose = _pose(binding, camera)
            registration = None
            if binding is not None and len(binding["ids"]):
                available = good[binding["ids"]] & (score[binding["ids"]] >= _LIMITS["minimumRawFanPeak"])
                # Match the face-scale coordinate system even for anisotropic
                # focal lengths or skew in the original working-image camera.
                projected, depth = _project(binding["points"], frame)
                residuals = np.linalg.norm(projected-local[binding["ids"]], axis=1)/scale
                available &= (depth > 0) & np.isfinite(residuals)
                if available.sum() < 8:
                    raise ValueError("insufficient_registered_eye_mouth_landmarks")
                registration = {"count": int(available.sum()), "medianFaceFraction": float(np.median(residuals[available])),
                                "p90FaceFraction": float(np.quantile(residuals[available], .9))}
            reliable = float(np.median(score[required]))
            pose_factor = float(np.cos(pose["totalAngleRadians"])**2) if pose is not None else 1.
            selection_score = reliable*symmetry*pose_factor
            reference_ok = symmetry >= _LIMITS["minimumReferenceSymmetry"]
            if pose is not None:
                reference_ok &= pose["totalAngleRadians"] <= _LIMITS["maximumReferencePoseRadians"]
            if registration is not None:
                reference_ok &= (registration["medianFaceFraction"] <= _LIMITS["maximumBindingMedianErrorFaceFraction"]
                                 and registration["p90FaceFraction"] <= _LIMITS["maximumBindingP90ErrorFaceFraction"])
            xs = np.r_[np.linspace(local[7, 0], local[8, 0], 4), np.linspace(local[8, 0], local[9, 0], 4)[1:]]
            curve = PchipInterpolator(local[6:11, 0], local[6:11, 1])
            targets = np.column_stack([xs, curve(xs)])
            tangents = np.column_stack([np.ones(len(xs)), curve.derivative()(xs)])
            args = {"targetPixels": targets, "confidence": np.full(7, float(score[6:11].min())),
                    "landmarkIds": np.arange(700, 707), "minimumYPixels": lower_mouth+.02*scale,
                    "imageShape": frame_shape, "targetTangents": tangents,
                    "maxTargetDistancePixels": _LIMITS["maximumTargetDistanceFaceFraction"]*scale,
                    "maximumTangentErrorPixels": _LIMITS["maximumTangentErrorFaceFraction"]*scale,
                    "maximumChainSlackPixels": _LIMITS["maximumChainSlackFaceFraction"]*scale}
            row.update(status="observation_accepted", faceScalePixels=scale, observedSymmetry=symmetry,
                       referenceSelectionScore=selection_score, referenceEligible=bool(reference_ok),
                       pose=pose, bindingRegistration=registration, imageShape=shape, imageShapeSource=shape_source,
                       rawPeakMedian=reliable, centralIndependentLandmarkIds=[7, 8, 9], splineSupportIds=[6, 7, 8, 9, 10],
                       interpolatedConstraintCount=7, mouthToChinFaceFraction=gap/scale,
                       localImageHomography=homography)
            views.append({"index": index, "camera": frame, "points": local, "scale": scale,
                          "homography": homography, "args": args, "report": row})
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            row["reason"] = str(error)
    return views, reports


def _topology(vertices, faces):
    raw_edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, inverse, counts = np.unique(raw_edges, axis=0, return_inverse=True, return_counts=True)
    owners = np.tile(np.arange(len(faces)), 3)
    ordering = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = counts == 2
    face_pairs = np.column_stack([owners[ordering[starts[paired]]], owners[ordering[starts[paired]+1]]])
    paired_edges = edges[paired]
    lengths = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
    weights = 1./lengths
    adjacency = coo_matrix((np.r_[weights, weights], (np.r_[edges[:, 0], edges[:, 1]],
                              np.r_[edges[:, 1], edges[:, 0]])), shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    if (degree <= 0).any():
        raise ValueError("mesh_contains_isolated_vertices")
    laplacian = diags(np.ones(len(vertices)))-diags(1/degree)@adjacency
    return laplacian, face_pairs, paired_edges


def _normals(vertices, faces):
    p = vertices[faces]
    normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    return normals/np.linalg.norm(normals, axis=1, keepdims=True)


def _dihedral(vertices, normals, pairs, edges):
    direction = vertices[edges[:, 1]]-vertices[edges[:, 0]]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    a, b = normals[pairs[:, 0]], normals[pairs[:, 1]]
    return np.arctan2(np.sum(direction*np.cross(a, b), axis=1), np.sum(a*b, axis=1))


def _contour(vertices, faces, view):
    result = visibleLowerJawConstraints(vertices, faces, view["camera"], **view["args"])
    pixels = np.asarray(result["diagnostics"]["candidatePixels"])
    tangent = np.asarray(view["args"]["targetTangents"])
    tangent = tangent/np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    delta = pixels-view["args"]["targetPixels"]
    result["signedResiduals"] = np.sum(delta*normal, axis=1)
    result["verticalResiduals"] = delta[:, 1]
    return result


def _plot_contours(views, after_by_view=None):
    rows = []
    for view in views:
        if "before" not in view:
            continue
        after_item = view["before"] if after_by_view is None else after_by_view[view["index"]]
        inverse = np.linalg.inv(view["homography"])
        before = np.asarray(view["before"]["diagnostics"]["candidatePixels"])
        after = np.asarray(after_item["diagnostics"]["candidatePixels"])
        rows.append({"viewIndex": view["index"], "faceScalePixels": view["scale"],
            "targetPixels": _transform(view["args"]["targetPixels"], inverse),
            "beforePixels": _transform(before, inverse), "afterPixels": _transform(after, inverse),
            "beforeSignedResidualPixels": view["before"]["signedResiduals"],
            "afterSignedResidualPixels": after_item["signedResiduals"],
            "meaning": "independently re-extracted visible contours with fixed observed target set"})
    return rows


def refineAdaptiveChin(vertices, faces, cameras, observations, *, fixedVertexIds=(),
                       protectedVertexIds=(), landmarkBinding=None):
    """Return an accepted adaptive shortening or an explicitly reasoned no-op.

    ``observations`` requires Nx68x2 ``faceLandmarksWorking`` and Nx68 arrays
    ``faceRawPeakScores``, ``landmarkCandidateValidMask``, and
    ``maximumCropVariationPixels``. Optional ``workingImageShape`` is H,W or
    Nx2. Raw FAN peaks are detector scores, never calibrated probabilities.
    Optional ``landmarkBinding`` supplies ``ibugIds``, MxK ``vertexIndices``
    and ``barycentricWeights`` (or ``barycentric``); its eye/mouth supports are
    fixed automatically. Existing mouthState dictionaries have this layout.

    Only three central jaw observations (7,8,9) determine each seven-sample
    target, with 6 and 10 supporting PCHIP tangents. Every candidate re-extracts
    a visible contour in every baseline-comparable view. A reference view is
    selected before measuring shape residuals, based on reliable observations,
    photo symmetry, and optional bound eye/mouth pose and registration.
    """
    unchanged = np.array(vertices, copy=True)
    result = {"vertices": unchanged, "originalVertices": unchanged.copy(),
              "displacement": np.zeros_like(unchanged),
              "activeVertexMask": np.zeros(len(unchanged), bool), "contours": []}
    diagnostics = {"method": "adaptiveBroadChinShortening", "status": "unchanged",
                   "thresholds": dict(_LIMITS), "candidates": [], "productionQualityApproved": False,
                   "confidenceMeaning": "raw FAN peaks are not calibrated probabilities",
                   "limitations": ["Visible transition edges remain geometric candidates for anatomical chin contour.",
                       "Camera depth is frozen; this cannot fix chin protrusion or camera calibration.",
                       "Central contour has three independent photo anchors, not seven independent landmarks.",
                       "Missing or ambiguous views are reported; acceptance only covers comparable observed views."]}
    result["diagnostics"] = diagnostics
    try:
        original, triangles = _validateMesh(vertices, faces)
        raw_cameras = list(cameras)
        checked = _validateCameras(raw_cameras)
        fixed = _ids(fixedVertexIds, len(original), "fixedVertexIds")
        protected = _ids(protectedVertexIds, len(original), "protectedVertexIds")
        binding = _binding(original, landmarkBinding)
        if binding is not None:
            protected = np.union1d(protected, binding["indices"].ravel()).astype(np.int64)
        views, view_reports = _prepare_views(original, raw_cameras, checked, observations, binding)
        diagnostics["views"] = view_reports
        eligible = [v for v in views if v["report"]["referenceEligible"]]
        if not eligible:
            diagnostics.update(status="unchanged_no_reliable_frontal_reference", reason="No view passed observation, pose, symmetry and registration gates")
            result["diagnostics"] = _json(diagnostics)
            return result
        # Deterministic score tie break is only an index; residual is not used.
        reference = max(eligible, key=lambda v: (v["report"]["referenceSelectionScore"], -v["index"]))
        diagnostics.update(referenceViewIndex=reference["index"],
                           referenceSelectionPolicy="photo reliability/symmetry and optional semantic surface pose; independent of chin residual",
                           bindingSupplied=binding is not None)
        comparable = []
        for view in views:
            try:
                view["before"] = _contour(original, triangles, view)
                view["report"]["baselineContour"] = view["before"]["diagnostics"]
                view["report"]["contourStatus"] = "comparable"
                comparable.append(view)
            except ValueError as error:
                view["report"].update(contourStatus="unavailable", contourReason=str(error))
        result["contours"] = _plot_contours(comparable)
        if "before" not in reference:
            diagnostics.update(status="unchanged_reference_contour_ambiguous", reason=reference["report"].get("contourReason"))
            result["diagnostics"] = _json(diagnostics)
            return result
        if len(checked) > 1 and len(comparable) < 2:
            diagnostics.update(status="unchanged_insufficient_multiview_contours", reason="At least one additional reliable visible contour is required when multiple photos are supplied")
            result["diagnostics"] = _json(diagnostics)
            return result
        scale, camera = reference["scale"], reference["camera"]
        residuals = reference["before"]["signedResiduals"]
        excess = float(np.median(reference["before"]["verticalResiduals"]))
        positive = float((residuals > _LIMITS["minimumExcessLengthFaceFraction"]*scale).mean())
        diagnostics.update(measuredExcessPixels=excess, measuredExcessFaceFraction=excess/scale,
                           positiveResidualFraction=positive, comparableViewIndices=[v["index"] for v in comparable],
                           displayMetrics={"before": float(np.mean(np.abs(residuals))),
                                           "after": float(np.mean(np.abs(residuals))),
                                           "unit": "reference virtual-image pixels, normal contour error"})
        if excess <= _LIMITS["minimumExcessLengthFaceFraction"]*scale or positive < _LIMITS["minimumPositiveResidualFraction"]:
            diagnostics.update(status="unchanged_no_consistent_excess_length", reason="Central signed contour residual does not reliably support shortening")
            result["diagnostics"] = _json(diagnostics)
            return result
        pixels, depth = _project(original, camera)
        contour_depth = float(np.median([edge["cameraDepth"] for edge in reference["before"]["diagnostics"]["selectedEdges"]]))
        world_scale = scale*contour_depth/camera["intrinsics"][1, 1]
        local = reference["points"]
        mouth_y = float(local[[48, 54, 57], 1].max())
        if binding is not None:
            mouth_points = binding["points"][(binding["ids"] >= 48) & (binding["ids"] <= 67)]
            if len(mouth_points):
                mouth_y = max(mouth_y, float(_project(mouth_points, camera)[0][:, 1].max()))
        mouth_y += .01*scale
        chin_y, chin_x = local[8, 1], local[8, 0]
        if chin_y-mouth_y <= .1*scale:
            diagnostics.update(status="unchanged_insufficient_free_lower_face", reason="Protected mouth leaves insufficient lower-face deformation height")
            result["diagnostics"] = _json(diagnostics)
            return result
        extra_fixed = np.flatnonzero((pixels[:, 1] <= mouth_y) | (depth >= contour_depth+.95*world_scale) | (depth <= 0))
        hard_fixed = np.union1d(np.union1d(fixed, protected), extra_fixed).astype(np.int64)
        free = np.setdiff1d(np.arange(len(original)), hard_fixed)
        field = _smooth((pixels[:, 1]-mouth_y)/(chin_y-mouth_y))
        field *= 1-_smooth((pixels[:, 1]-chin_y-.14*scale)/(.45*scale))
        field *= 1-_smooth((np.abs(pixels[:, 0]-chin_x)-.27*scale)/(.42*scale))
        field *= 1-_smooth((depth-contour_depth-.4*world_scale)/(.55*world_scale))
        field[hard_fixed] = 0.
        if not len(free) or field.max(initial=0) <= 0:
            diagnostics.update(status="unchanged_no_free_lower_face_vertices", reason="The derived lower-face region is entirely fixed")
            result["diagnostics"] = _json(diagnostics)
            return result
        laplacian, face_pairs, paired_edges = _topology(original, triangles)
        bending = laplacian.T@laplacian
        original_normals = _normals(original, triangles)
        original_dihedral = _dihedral(original, original_normals, face_pairs, paired_edges)
        baseline_pairs = intersectionPairs(original, triangles)
        # Before-defined, spatially overlapping front chin depth layers. Neither
        # target mesh nor a previous subject-specific candidate defines these.
        region = ((np.abs(pixels[:, 0]-chin_x) < .36*scale) & (pixels[:, 1] > mouth_y)
                  & (pixels[:, 1] < chin_y+.16*scale) & (depth > contour_depth-.2*world_scale)
                  & (depth < contour_depth+.4*world_scale))
        region_ids = np.flatnonzero(region)
        nearby = cKDTree(pixels[region_ids]).query_pairs(.04*scale, output_type="ndarray")
        layer_pairs = region_ids[nearby] if len(nearby) else np.empty((0, 2), int)
        if len(layer_pairs):
            differences = np.abs(depth[layer_pairs[:, 0]]-depth[layer_pairs[:, 1]])
            layer_pairs = layer_pairs[(differences > .03*world_scale) & (differences < .4*world_scale)]
        options = dict(fixedVertexIds=hard_fixed, camera=camera,
                       maxDisplacement=_LIMITS["maximumDisplacementFaceFraction"]*world_scale,
                       maxRelativeEdgeChange=_LIMITS["maximumRelativeEdgeChange"],
                       minAreaRatio=_LIMITS["minimumAreaRatio"], minNormalCosine=_LIMITS["minimumNormalCosine"])
        diagnostics.update(faceScalePixels=scale, faceScaleCanonical=world_scale, hardFixedVertexCount=len(hard_fixed),
                           staticVertexCount=len(fixed), protectedVertexCount=len(protected),
                           deformationRegion={"mouthBoundaryPixels": mouth_y, "observedChinPixels": [chin_x, chin_y],
                               "originalContourDepth": contour_depth, "depthFadeStart": contour_depth+.4*world_scale,
                               "depthFixedBoundary": contour_depth+.95*world_scale, "coherencePairCount": len(layer_pairs)},
                           baselineIntersectionPairs=len(baseline_pairs), selfIntersectionChecked=True,
                           candidatePolicy="amplitude fractions of measured excess; moderate residual target with strain and smoothness margin",
                           noOpCandidate={"amplitudeFraction": 0., "retainedIfNoSafeImprovement": True})
        accepted = []
        for smoothing in (4., 16., 64.):
            system = diags(np.ones(len(original)))+smoothing*bending
            broad = np.zeros(len(original))
            broad[free] = spsolve(system[free][:, free], field[free])
            if not np.isfinite(broad).all():
                continue
            for fraction in (.35, .5, .7, .9):
                amplitude = min(excess*fraction, _LIMITS["maximumAmplitudeFaceFraction"]*scale)
                delta_y = -amplitude*broad*depth/camera["intrinsics"][1, 1]
                proposed = original+delta_y[:, None]*camera["rotation"][1]
                proposed[hard_fixed] = original[hard_fixed]
                geometry = checkFrontGeometry(original, proposed, triangles, **options)
                row = {"smoothing": smoothing, "amplitudeFraction": fraction, "amplitudePixels": amplitude,
                       "status": "rejected_geometry", "geometry": geometry}
                diagnostics["candidates"].append(row)
                if not geometry["accepted"]:
                    continue
                normals = _normals(proposed, triangles)
                normal_rotation = np.arccos(np.clip(np.sum(original_normals*normals, axis=1), -1, 1))
                angle = _dihedral(proposed, normals, face_pairs, paired_edges)-original_dihedral
                angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
                moved_vertices = np.linalg.norm(proposed-original, axis=1) > world_scale*1e-10
                active_faces = moved_vertices[triangles].any(1)
                active_edges = active_faces[face_pairs].any(1)
                movement = -amplitude*broad
                layer_gap = float(np.abs(movement[layer_pairs[:, 0]]-movement[layer_pairs[:, 1]]).max(initial=0))
                crease = {"maximumNormalRotationRadians": float(normal_rotation.max()),
                          "maximumDihedralChangeRadians": float(angle.max(initial=0)),
                          "globalP95DihedralChangeRadians": float(np.quantile(angle, .95)) if len(angle) else 0.,
                          "activeP95DihedralChangeRadians": float(np.quantile(angle[active_edges], .95)) if active_edges.any() else 0.,
                          "activeDihedralEdgeCount": int(active_edges.sum()),
                          "activeDihedralScope": "edges adjacent to a face with at least one changed vertex; original-relative angles",
                          "p95AcceptanceScope": "global mesh; active-region percentile additionally reported; maximum-angle gate covers every edge",
                          "maximumLayerMovementDifferencePixels": layer_gap}
                crease["accepted"] = bool(crease["maximumNormalRotationRadians"] <= _LIMITS["maximumNormalRotationRadians"]
                    and crease["maximumDihedralChangeRadians"] <= _LIMITS["maximumDihedralChangeRadians"]
                    and crease["globalP95DihedralChangeRadians"] <= _LIMITS["maximumDihedralChangeP95Radians"]
                    and layer_gap <= max(_LIMITS["maximumLayerMovementDifferenceFaceFraction"]*scale,
                                         _LIMITS["maximumRelativeLayerMovementDifference"]*amplitude))
                row["smoothness"] = crease
                if not crease["accepted"]:
                    row["status"] = "rejected_smoothness"
                    continue
                after_views, cross_rows = {}, []
                cross_safe = True
                for view in comparable:
                    cross = {"viewIndex": view["index"], "targetSetUnchanged": True}
                    cross_rows.append(cross)
                    try:
                        after = _contour(proposed, triangles, view)
                        after_views[view["index"]] = after
                        prior_abs = np.abs(view["before"]["signedResiduals"])
                        after_abs = np.abs(after["signedResiduals"])
                        increase = float(after_abs.mean()-prior_abs.mean())/view["scale"]
                        worst = float((after_abs-prior_abs).max())/view["scale"]
                        safe = (increase <= _LIMITS["maximumCrossViewIncreaseFaceFraction"]
                                and worst <= _LIMITS["maximumCrossViewPointIncreaseFaceFraction"])
                        cross.update(accepted=bool(safe), beforeErrorPixels=float(prior_abs.mean()),
                                     afterErrorPixels=float(after_abs.mean()), meanIncreaseFaceFraction=increase,
                                     maximumPointIncreaseFaceFraction=worst,
                                     candidatePixels=after["diagnostics"]["candidatePixels"])
                        cross_safe &= safe
                    except ValueError as error:
                        cross.update(accepted=False, reason=str(error))
                        cross_safe = False
                row["crossViewContours"] = cross_rows
                if not cross_safe:
                    row["status"] = "rejected_multiview_contour"
                    continue
                after_ref = after_views[reference["index"]]
                before_error = float(np.mean(np.abs(residuals)))
                error = float(np.mean(np.abs(after_ref["signedResiduals"])))
                improvement = before_error-error
                row.update(beforeErrorPixels=before_error, afterErrorPixels=error, improvementPixels=improvement)
                if improvement < _LIMITS["minimumImprovementFaceFraction"]*scale or improvement < .15*before_error:
                    row["status"] = "rejected_insufficient_improvement"
                    continue
                pairs = intersectionPairs(proposed, triangles)
                added = sorted(pairs-baseline_pairs)
                row.update(newIntersectionPairs=len(added), newFacePairs=added, finalIntersectionPairs=len(pairs))
                if added:
                    row["status"] = "rejected_intersections"
                    continue
                strain = geometry["metrics"]["maximumRelativeEdgeChange"]
                cost = (max(error/scale-_LIMITS["acceptableResidualFaceFraction"], 0.)**2
                        + .004*strain**2 + .2*(layer_gap/scale)**2 + .0001*fraction**2)
                row.update(status="accepted", selectionCost=float(cost))
                accepted.append((cost, row, proposed, after_views))
        if accepted:
            _, selected, final, final_contours = min(accepted, key=lambda item: item[0])
            diagnostics.update(status="accepted", selectedCandidate=selected,
                               changedVertexCount=int(np.any(final != original, axis=1).sum()),
                               fixedVerticesExactlyUnchanged=bool(np.array_equal(final[hard_fixed], original[hard_fixed])),
                               eyesMouthSupportExactlyUnchanged=bool(np.array_equal(final[protected], original[protected])),
                               finalGeometryCheck=selected["geometry"],
                               finalIntersectionPairs=selected["finalIntersectionPairs"], newIntersectionPairs=0)
            diagnostics["displayMetrics"]["after"] = selected["afterErrorPixels"]
            result.update(vertices=final, displacement=final-original,
                          activeVertexMask=np.any(final != original, axis=1))
        else:
            final_contours = {view["index"]: view["before"] for view in comparable}
            diagnostics.update(status="unchanged_no_safe_improvement", reason="Every candidate failed an original-relative geometry, smoothness, multiview, intersection or improvement gate")
        result["allHardFixedVertexIds"] = hard_fixed
        result["contours"] = _plot_contours(comparable, final_contours)
    except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        diagnostics.update(status="unchanged_invalid_or_unsupported_input", reason=str(error))
    result["diagnostics"] = _json(diagnostics)
    return result
