"""Conservative view-dependent FAN jaw correspondences on client geometry.

The same semantic ID is compared in a photograph and three independently
rendered appearances. A ray hit is not an anatomical label: rest-topology jaw
support and view checks are required before it can become a constraint. This
module does not modify geometry or use photo-derived textures.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .clientLandmarks import (IBUG68_MEDIAPIPE_IDS, ibugClientLandmarks,
                              loadClientLandmarks, sampleClientLandmarks)
from .surfaceVisibility import ProjectedSurfaceVisibility


POLICY = {
    "semanticIds": list(range(5, 12)),
    "primaryAppearance": "template",
    "requiredAppearances": ["template", "gray", "grayAlternate"],
    "maximumAppearanceDriftFaceFraction": .020,
    "maximumGrayLightingDriftFaceFraction": .020,
    "maximumTargetDistanceFaceFraction": .12,
    "maximumJawGeodesicFaceFraction": .25,
    "maximumPosteriorOffsetFromJawFaceFraction": .20,
    "maximumBelowChinFaceFraction": .14,
    "minimumBelowLowerLipFaceFraction": .015,
    "maximumLateralMarginFaceFraction": .12,
    "maximumSecondarySurfaceDistanceFaceFraction": .22,
    "minimumViewNormalCosine": .015,
    "minimumPlausibleAppearanceRayCount": 2,
    "minimumConstraintsPerView": 3,
    "minimumRawPeak": .25,
    "minimumCropVariantPeak": .20,
    "maximumCropVariationFaceFraction": .015,
}


def _project(points, camera):
    p = points @ np.asarray(camera["rotation"]).T + np.asarray(camera["translation"])
    h = p @ np.asarray(camera["intrinsics"]).T
    return h[:, :2] / h[:, 2:], p


def lowerJawDomain(templateVertices, faces, *, fixedVertexIds=(), protectedVertexIds=()):
    """Make a broad semantic gate in undeformed template coordinates.

    Legacy MP anchors are used only to define a coarse region. They do not
    prescribe correspondence locations or force the final mesh to this shape.
    The mask is not a hard deformation boundary; a solver may need a skirt.
    """
    v, f = np.asarray(templateVertices, float), np.asarray(faces, np.int64)
    binding = loadClientLandmarks(f)
    mp = sampleClientLandmarks(v, binding)
    ibug = np.asarray([mp[list(ids)].mean(0) for ids in IBUG68_MEDIAPIPE_IDS])
    eyes = np.asarray([ibug[36:42].mean(0), ibug[42:48].mean(0)])
    center = eyes.mean(0)
    x = eyes[1]-eyes[0]
    x /= np.linalg.norm(x)
    y = ibug[48:60].mean(0)-center
    y -= (y@x)*x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z *= np.sign((ibug[30]-center)@z)
    axes = np.stack([x, y, z], axis=1)
    scale = float(np.linalg.norm(ibug[8]-center))
    frame = (v-center)@axes/scale
    landmarks = (ibug-center)@axes/scale
    ids = np.asarray(POLICY["semanticIds"])
    mp_ids = np.array([IBUG68_MEDIAPIPE_IDS[i][0] for i in ids])
    seeds = np.unique(binding["vertexIndices"][mp_ids])
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(v[edges[:, 0]]-v[edges[:, 1]], axis=1)/scale
    graph = coo_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]],
                       np.r_[edges[:, 1], edges[:, 0]])), shape=(len(v), len(v))).tocsr()
    distances = dijkstra(graph, directed=False, indices=seeds, min_only=True)
    bounds = {
        "minimumY": float(landmarks[57, 1]+POLICY["minimumBelowLowerLipFaceFraction"]),
        "maximumY": float(landmarks[8, 1]+POLICY["maximumBelowChinFaceFraction"]),
        "minimumX": float(landmarks[5, 0]-POLICY["maximumLateralMarginFaceFraction"]),
        "maximumX": float(landmarks[11, 0]+POLICY["maximumLateralMarginFaceFraction"]),
    }
    # A central posterior bound must track the jaw's curved depth. A single
    # lateral-jaw z cutoff would incorrectly admit central neck triangles.
    order = np.argsort(landmarks[ids, 0])
    jaw_x, jaw_z = landmarks[ids[order], 0], landmarks[ids[order], 2]
    posterior = np.interp(frame[:, 0], jaw_x, jaw_z)-POLICY["maximumPosteriorOffsetFromJawFaceFraction"]
    mask = ((distances <= POLICY["maximumJawGeodesicFaceFraction"])
            & (frame[:, 0] >= bounds["minimumX"]) & (frame[:, 0] <= bounds["maximumX"])
            & (frame[:, 1] >= bounds["minimumY"]) & (frame[:, 1] <= bounds["maximumY"])
            & (frame[:, 2] >= posterior))
    mask[np.union1d(fixedVertexIds, protectedVertexIds).astype(int)] = False
    return {"vertexMask": mask, "geodesicDistanceFaceFraction": distances,
            "jawSeedVertexIds": seeds, "fixedVertexIds": np.asarray(fixedVertexIds, int),
            "protectedVertexIds": np.asarray(protectedVertexIds, int),
            "restCoordinates": frame, "restIbugCoordinates": landmarks,
            "restEyeCenter": center, "restAxes": axes, "restFaceScale": scale,
            "bounds": bounds, "jawDepthX": jaw_x, "jawDepthZ": jaw_z,
            "landmarkBinding": ibugClientLandmarks(binding),
            "meaning": "Approximate broad anatomical support; not manual jaw labels or deformation support"}


def _ray_bindings(vertices, faces, camera, shape, points, domain):
    """Exact nearest skin intersections, then separate anatomical gates."""
    _, cv = _project(vertices, camera)
    k = np.asarray(camera["intrinsics"], float)
    rays = np.c_[points, np.ones(len(points))]@np.linalg.inv(k).T
    visibility = ProjectedSurfaceVisibility(cv, faces, k, tuple(map(int, shape)))
    hits = visibility.closestDepth(rays)
    face_ids = hits["faceIds"]
    hit = face_ids >= 0
    weights = np.full((len(points), 3), np.nan)
    canonical = np.full_like(weights, np.nan)
    view_cosine = np.full(len(points), np.nan)
    rest = np.full_like(weights, np.nan)
    geodesic = np.full(len(points), np.inf)
    reason = [[] for _ in points]
    for i in range(len(points)):
        if not hit[i]:
            reason[i].append("no_skin_ray_hit")
            continue
        ids = faces[face_ids[i]]
        tri = cv[ids]
        point = rays[i]*hits["depth"][i]/rays[i, 2]
        basis = np.stack([tri[1]-tri[0], tri[2]-tri[0]], axis=1)
        uv = np.linalg.lstsq(basis, point-tri[0], rcond=None)[0]
        w = np.r_[1.-uv.sum(), uv]
        weights[i] = w
        canonical[i] = w@vertices[ids]
        rest[i] = w@domain["restCoordinates"][ids]
        geodesic[i] = w@domain["geodesicDistanceFaceFraction"][ids]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        view_cosine[i] = float(normal@(-point)/(np.linalg.norm(normal)*np.linalg.norm(point)))
        if not np.isfinite(w).all() or w.min() < -1e-6 or abs(w.sum()-1.) > 1e-6:
            reason[i].append("invalid_ray_barycentric")
        if geodesic[i] > POLICY["maximumJawGeodesicFaceFraction"]:
            reason[i].append("outside_jaw_geodesic_band")
        b = domain["bounds"]
        if not b["minimumX"] <= rest[i, 0] <= b["maximumX"]:
            reason[i].append("outside_lower_jaw_width")
        if not b["minimumY"] <= rest[i, 1] <= b["maximumY"]:
            reason[i].append("outside_lower_lip_to_chin_band")
        posterior = np.interp(rest[i, 0], domain["jawDepthX"], domain["jawDepthZ"])-POLICY["maximumPosteriorOffsetFromJawFaceFraction"]
        if rest[i, 2] < posterior:
            reason[i].append("posterior_surface_or_neck")
        if view_cosine[i] < POLICY["minimumViewNormalCosine"]:
            reason[i].append("grazing_or_back_facing_triangle")
        if np.isin(ids, np.union1d(domain["fixedVertexIds"], domain["protectedVertexIds"])).any():
            reason[i].append("protected_triangle")
    accepted = np.array([not reasons for reasons in reason], bool)
    return {"faceIds": face_ids, "vertexIndices": faces[np.maximum(face_ids, 0)],
            "barycentricWeights": weights, "canonicalPoints": canonical,
            "restCoordinates": rest, "geodesicDistanceFaceFraction": geodesic,
            "viewNormalCosine": view_cosine, "acceptedMask": accepted, "reasons": reason}


def prepareRenderedJawCorrespondence(vertices, faces, templateVertices, cameras,
        photoObservations, renderViews, *, imageShapes, fixedVertexIds=(), protectedVertexIds=(), domain=None):
    """Return filtered constraints and exhaustive per-point rejection reports.

    ``renderViews`` is a list of ``{viewIndex, variants: {name: detection}}``.
    Detection dicts need ``points[68,2], valid[68], spread[68], peaks[3,68]``.
    They may contain old diagnostic ray arrays, but rays are always recomputed
    on supplied vertices, allowing exactly the same gates after re-rendering.
    ``photoObservations`` uses the observeClientJawBatch NPZ field names.
    Views retain distinct surface bindings; no semantic ID is merged in 3D.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if domain is None:
        domain = lowerJawDomain(templateVertices, faces, fixedVertexIds=fixedVertexIds,
                               protectedVertexIds=protectedVertexIds)
    ids = np.asarray(POLICY["semanticIds"], int)
    views, diagnostics = [], []
    for item in renderViews:
        index, variants = int(item["viewIndex"]), item["variants"]
        camera, shape = cameras[index], imageShapes[index]
        scale = float(photoObservations["faceScaleWorkingPixels"][index])
        target = np.asarray(photoObservations["faceLandmarksWorking"])[index, ids]
        photo_valid = np.asarray(photoObservations["landmarkCandidateValidMask"])[index, ids]
        photo_peaks = np.asarray(photoObservations["faceRawPeakScores"])[index, ids]
        photo_spread = np.asarray(photoObservations["maximumCropVariationPixels"])[index, ids]
        primary = variants["template"]
        source = np.asarray(primary["points"])[ids]
        ray_data = {name: _ray_bindings(vertices, faces, camera, shape, np.asarray(variants[name]["points"])[ids], domain)
                    for name in POLICY["requiredAppearances"]}
        drift = np.maximum.reduce([np.linalg.norm(np.asarray(variants[name]["points"])[ids]-source, axis=1)
                                   for name in ("gray", "grayAlternate")])/scale
        light = np.linalg.norm(np.asarray(variants["gray"]["points"])[ids]-np.asarray(variants["grayAlternate"]["points"])[ids], axis=1)/scale
        residual = np.linalg.norm(target-source, axis=1)/scale
        primary_rays = ray_data["template"]
        reports, accepted, confidence = [], [], []
        for row, semantic_id in enumerate(ids):
            reasons = []
            if not camera.get("valid", True): reasons.append("camera_rejected")
            if not photo_valid[row] or not np.isfinite(target[row]).all() or photo_peaks[row] < POLICY["minimumRawPeak"]:
                reasons.append("photo_semantic_invalid")
            # The original photograph observer used a slightly looser crop
            # threshold; fitting applies the same stricter .015 threshold.
            if photo_spread[row]/scale > POLICY["maximumCropVariationFaceFraction"]:
                reasons.append("photo_crop_unstable")
            for name in POLICY["requiredAppearances"]:
                d = variants[name]
                peaks = np.asarray(d["peaks"])[:, semantic_id]
                if (not bool(d["valid"][semantic_id]) or not np.isfinite(peaks).all()
                        or peaks[0] < POLICY["minimumRawPeak"] or peaks.min() < POLICY["minimumCropVariantPeak"]
                        or d["spread"][semantic_id]/scale > POLICY["maximumCropVariationFaceFraction"]):
                    reasons.append(name+"_detection_invalid_or_unstable")
            if drift[row] > POLICY["maximumAppearanceDriftFaceFraction"]: reasons.append("appearance_drift")
            if light[row] > POLICY["maximumGrayLightingDriftFaceFraction"]: reasons.append("lighting_drift")
            if residual[row] > POLICY["maximumTargetDistanceFaceFraction"]: reasons.append("excessive_photo_render_gap")
            reasons.extend(primary_rays["reasons"][row])
            plausible = sum(bool(data["acceptedMask"][row]) for data in ray_data.values())
            if plausible < POLICY["minimumPlausibleAppearanceRayCount"]:
                reasons.append("insufficient_anatomical_ray_consensus")
            secondary_distance = []
            for name in ("gray", "grayAlternate"):
                data = ray_data[name]
                if primary_rays["acceptedMask"][row] and data["acceptedMask"][row]:
                    secondary_distance.append(float(np.linalg.norm(data["restCoordinates"][row]-primary_rays["restCoordinates"][row])))
            if secondary_distance and max(secondary_distance) > POLICY["maximumSecondarySurfaceDistanceFaceFraction"]:
                reasons.append("appearance_surface_disagreement")
            score = min(float(photo_peaks[row]), *(float(np.asarray(variants[name]["peaks"])[:, semantic_id].min()) for name in POLICY["requiredAppearances"]))
            score *= np.exp(-.5*(drift[row]/POLICY["maximumAppearanceDriftFaceFraction"])**2)
            accepted.append(not reasons)
            confidence.append(score)
            reports.append({"semanticId": int(semantic_id), "accepted": not reasons, "reasons": reasons,
                "sourcePixel": source[row], "targetPixel": target[row], "confidence": score,
                "appearanceDriftFaceFraction": drift[row], "lightingDriftFaceFraction": light[row],
                "photoRenderDistanceFaceFraction": residual[row], "plausibleAppearanceRayCount": plausible,
                "primaryFaceId": int(primary_rays["faceIds"][row]),
                "primaryRestCoordinates": primary_rays["restCoordinates"][row],
                "primaryGeodesicFaceFraction": primary_rays["geodesicDistanceFaceFraction"][row],
                "primaryViewNormalCosine": primary_rays["viewNormalCosine"][row],
                "appearanceRayReasons": {name: data["reasons"][row] for name, data in ray_data.items()}})
        accepted = np.asarray(accepted, bool)
        eligible = bool(accepted.sum() >= POLICY["minimumConstraintsPerView"] and accepted[(ids >= 7)&(ids <= 9)].any())
        view = {"viewIndex": index, "semanticIds": ids[accepted], "ids": ids[accepted],
            "vertexIndices": primary_rays["vertexIndices"][accepted],
            "barycentricWeights": primary_rays["barycentricWeights"][accepted],
            "faceIds": primary_rays["faceIds"][accepted], "targetPixels": target[accepted],
            "sourcePixels": source[accepted], "confidence": np.asarray(confidence)[accepted],
            "canonicalPoints": primary_rays["canonicalPoints"][accepted], "acceptedMask": np.ones(accepted.sum(), bool),
            "camera": {key: np.asarray(camera[key]) for key in ("rotation", "translation", "intrinsics")},
            "imageShape": np.asarray(shape, int), "faceScalePixels": scale, "eligible": eligible,
            "allSemanticIds": ids, "allAcceptedMask": accepted}
        views.append(view)
        diagnostics.append({"viewIndex": index, "eligible": eligible, "acceptedCount": int(accepted.sum()),
                            "acceptedSemanticIds": ids[accepted], "landmarks": reports})
    return {"views": views, "domain": domain, "policy": dict(POLICY), "diagnostics": diagnostics,
            "limitations": ["Accepted correspondences are automatic observations, not anatomical ground truth.",
                "The rest-template jaw domain is an approximate exclusion gate, not a validated anatomical annotation.",
                "Distinct views can bind the same semantic ID to different surface positions.",
                "Agreement across rendered appearances does not establish absolute detector accuracy."]}
