"""Freeze skin bindings beside the template's visible eye material.

This is a template-front orthographic material boundary, not an anatomical
annotation. The camera looks along -Z; X is image-right and Y is image-up.
Only original NICP skin may supply a binding. Other opaque parts still occlude.
"""

from __future__ import annotations

import numpy as np


def _raster(vertices, faces, faceIds, pixelVertices, resolution):
    depth = np.full((resolution, resolution), -np.inf, np.float64)
    hits = np.full((resolution, resolution), -1, np.int64)
    weights = np.zeros((resolution, resolution, 3), np.float64)
    for faceId in faceIds:
        indices = faces[faceId]
        tri = pixelVertices[indices]
        first, second, third = tri
        denominator = ((second[1]-third[1])*(first[0]-third[0])
                       + (third[0]-second[0])*(first[1]-third[1]))
        if abs(denominator) < 1e-12:
            continue
        lo = np.maximum(np.ceil(tri.min(0)-1e-8).astype(np.int64), 0)
        hi = np.minimum(np.floor(tri.max(0)+1e-8).astype(np.int64), resolution-1)
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        w0 = ((second[1]-third[1])*(xx-third[0])
              + (third[0]-second[0])*(yy-third[1]))/denominator
        w1 = ((third[1]-first[1])*(xx-third[0])
              + (first[0]-third[0])*(yy-third[1]))/denominator
        w2 = 1-w0-w1
        contained = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        bary = np.stack([w0, w1, w2], axis=-1)
        z = bary @ vertices[indices, 2]
        area = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        take = contained & (z > depth[area])
        depth[area][take] = z[take]
        hits[area][take] = faceId
        # Clamp only the containment tolerance, retaining exact convex weights.
        selected = np.maximum(bary[take], 0)
        selected /= selected.sum(1, keepdims=True)
        weights[area][take] = selected
    return hits, weights


def buildVisibleEyeRimBinding(arrays, eyeComponent, *, resolution=192):
    """Return about 64 ordered, uniformly spaced visible-skin surface samples.

    ``arrays`` is a client template NPZ mapping and ``eyeComponent`` its original
    material-2 component metadata. All returned 3D coordinates use the original
    template's units (client centimeters). ``vertexIndices`` and ``faceIds`` use
    NICP order, never full-mesh order. ``restPoints`` equals the barycentric sum.

    Each sample is the nearest 8-neighbor visible skin pixel outside an eye
    contour sample. Its offset is at most sqrt(2) pixels, recorded explicitly.
    Arc positions whose neighboring pixels belong to accessories/background are
    omitted. Skin behind the original eye cap or more than two radii from its
    center is also excluded: a gap beside the cap may reveal the rear skull.
    The function raises if the eye is absent, clipped or fewer than 16
    distinct skin samples survive; callers should retain their prior placement.
    No learned inference, external asset path, or view-specific run is used.
    """
    import cv2

    if (not isinstance(resolution, (int, np.integer)) or isinstance(resolution, bool)
            or not 64 <= resolution <= 1024):
        raise ValueError("Eye-rim resolution must be an integer between 64 and 1024")
    vertices = np.asarray(arrays["vertices"], np.float64)
    faces = np.asarray(arrays["faces"])
    materials = np.asarray(arrays["materialIds"])
    nicpFaces = np.asarray(arrays["nicpFaces"])
    nicpToFull = np.asarray(arrays["nicpToFullIds"])
    eyeIds = np.asarray(eyeComponent["vertexIndices"])
    eyeFaces = np.asarray(eyeComponent["faceIndices"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
            or materials.shape != (len(faces),) or materials.dtype.kind not in "iu"
            or nicpFaces.ndim != 2 or nicpFaces.shape[1] != 3 or nicpFaces.dtype.kind not in "iu"
            or nicpToFull.ndim != 1 or nicpToFull.dtype.kind not in "iu"
            or not len(nicpToFull) or nicpToFull.min() < 0 or nicpToFull.max() >= len(vertices)
            or len(np.unique(nicpToFull)) != len(nicpToFull)
            or not len(nicpFaces) or nicpFaces.min() < 0 or nicpFaces.max() >= len(nicpToFull)
            or eyeIds.ndim != 1 or eyeIds.dtype.kind not in "iu" or len(eyeIds) < 4
            or eyeIds.min() < 0 or eyeIds.max() >= len(vertices)
            or eyeFaces.ndim != 1 or eyeFaces.dtype.kind not in "iu" or not len(eyeFaces)
            or eyeFaces.min() < 0 or eyeFaces.max() >= len(faces)
            or np.any(materials[eyeFaces] != 2)
            or not np.isin(faces[eyeFaces], eyeIds).all()):
        raise ValueError("Invalid original client template or material-2 eye component")

    # Fit the original sphere cap, not its forward-biased vertex centroid.
    eye = vertices[eyeIds]
    mean = eye.mean(0)
    centered = eye-mean
    design = np.c_[2*centered, np.ones(len(eye))]
    fitted, _, rank, _ = np.linalg.lstsq(design, (centered*centered).sum(1), rcond=None)
    radiusSquared = fitted[3]+fitted[:3] @ fitted[:3]
    if rank != 4 or not np.isfinite(radiusSquared) or radiusSquared <= 0:
        raise ValueError("Eye component does not provide a finite sphere-cap fit")
    center, radius = fitted[:3]+mean, float(np.sqrt(radiusSquared))
    residual = np.abs(np.linalg.norm(eye-center, axis=1)-radius)

    # A square crop with substantial margin prevents treating viewport edges as
    # eyelids. Pixel centers lie at integer coordinates.
    halfExtent = max(radius*1.25, float(np.abs(eye[:, :2]-center[:2]).max())*1.15)
    unitsPerPixel = 2*halfExtent/(resolution-1)
    pixels = (vertices[:, :2]-center[:2])/unitsPerPixel
    pixels[:, 1] *= -1
    pixels += (resolution-1)/2
    opaqueIds = np.flatnonzero(np.isin(materials, [0, 1, 2]))
    hits, baryImage = _raster(vertices, faces, opaqueIds, pixels, resolution)

    # Match full triangles back to NICP order. Reorder barycentrics if needed.
    inverse = np.full(len(vertices), -1, np.int64)
    inverse[nicpToFull] = np.arange(len(nicpToFull))
    localFaces = inverse[faces]
    faceLookup = {tuple(sorted(map(int, row))): i for i, row in enumerate(nicpFaces)}
    fullToNicpFace = np.full(len(faces), -1, np.int64)
    permutations = np.zeros((len(faces), 3), np.int64)
    for fullId in np.flatnonzero((materials == 0) & (localFaces >= 0).all(1)):
        local = localFaces[fullId]
        nicpId = faceLookup.get(tuple(sorted(map(int, local))))
        if nicpId is not None:
            fullToNicpFace[fullId] = nicpId
            permutations[fullId] = [int(np.flatnonzero(local == i)[0]) for i in nicpFaces[nicpId]]
    validHit = hits >= 0
    eyeMask = validHit & np.isin(hits, eyeFaces)
    skinMask = validHit & (fullToNicpFace[np.maximum(hits, 0)] >= 0)
    skinPixels = np.column_stack(np.nonzero(skinMask))
    skinHits = hits[skinPixels[:, 0], skinPixels[:, 1]]
    skinPoints = (vertices[faces[skinHits]]
                  * baryImage[skinPixels[:, 0], skinPixels[:, 1], :, None]).sum(1)
    socketLocal = ((skinPoints[:, 2] >= eye[:, 2].min()-unitsPerPixel)
                   & (np.linalg.norm(skinPoints-center, axis=1) <= 2*radius))
    eligibleSkinMask = np.zeros_like(skinMask)
    eligibleSkinMask[skinPixels[socketLocal, 0], skinPixels[socketLocal, 1]] = True
    if not eyeMask.any() or eyeMask[[0, -1], :].any() or eyeMask[:, [0, -1]].any():
        raise ValueError("Template eye is invisible or clipped in the frontal crop")
    contours, _ = cv2.findContours(eyeMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=lambda value: abs(cv2.contourArea(value)))[:, 0].astype(np.float64)
    if len(contour) < 16:
        raise ValueError("Visible eye contour is too small for a stable binding")
    segments = np.linalg.norm(np.roll(contour, -1, axis=0)-contour, axis=1)
    arc = np.r_[0., np.cumsum(segments)]
    desiredArc = np.arange(64)*arc[-1]/64
    selected = np.searchsorted(arc, desiredArc, side="right")-1
    offsets = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)
                        if x or y], np.int64)
    samplePixels, arcFractions, contourSamples, seen = [], [], [], set()
    eligibleCount = 0
    for position, contourId in zip(desiredArc, selected):
        fraction = (position-arc[contourId])/segments[contourId]
        point = contour[contourId]*(1-fraction)+contour[(contourId+1) % len(contour)]*fraction
        # Include neighbors of both segment ends so diagonal raster steps do
        # not introduce a preferred side or bias the sampling density.
        candidates = np.unique(np.r_[contour[contourId]+offsets,
                                     contour[(contourId+1) % len(contour)]+offsets].astype(np.int64), axis=0)
        inside = ((candidates >= 0) & (candidates < resolution)).all(1)
        candidates = candidates[inside]
        candidates = candidates[eligibleSkinMask[candidates[:, 1], candidates[:, 0]]]
        distance = np.linalg.norm(candidates-point, axis=1)
        candidates, distance = candidates[distance <= np.sqrt(2)+1e-8], distance[distance <= np.sqrt(2)+1e-8]
        if not len(candidates):
            continue
        eligibleCount += 1
        chosen = candidates[int(distance.argmin())]
        key = tuple(map(int, chosen))
        if key in seen:
            continue
        seen.add(key)
        samplePixels.append(chosen)
        contourSamples.append(point)
        arcFractions.append(position/arc[-1])
    if len(samplePixels) < 16:
        raise ValueError("Fewer than 16 distinct visible skin samples beside the eye")
    samplePixels = np.asarray(samplePixels, np.int64)
    contourSamples = np.asarray(contourSamples, np.float64)
    fullIds = hits[samplePixels[:, 1], samplePixels[:, 0]]
    faceIds = fullToNicpFace[fullIds]
    weights = baryImage[samplePixels[:, 1], samplePixels[:, 0]]
    weights = np.take_along_axis(weights, permutations[fullIds], axis=1)
    vertexIndices = nicpFaces[faceIds].astype(np.int64)
    restPoints = (vertices[nicpToFull][vertexIndices]*weights[..., None]).sum(1)
    return {
        "vertexIndices": vertexIndices, "barycentricWeights": weights,
        "faceIds": faceIds, "fullFaceIds": fullIds, "restPoints": restPoints,
        "samplePixels": samplePixels, "contourPixels": contour,
        "contourSamplePixels": contourSamples, "arcFractions": np.asarray(arcFractions),
        "sphereCenter": center, "sphereRadius": radius,
        "eyeMask": eyeMask, "skinMask": skinMask,
        "eligibleSkinMask": eligibleSkinMask, "faceIdImage": hits,
        "diagnostics": {
            "method": "templateFrontalOrthographicVisibleMaterialBoundary",
            "cameraDirection": [0., 0., -1.], "twoSidedOcclusion": True,
            "resolution": int(resolution), "unitsPerPixel": float(unitsPerPixel),
            "cropCenterXY": center[:2].tolist(), "cropHalfExtent": float(halfExtent),
            "eyeVisiblePixels": int(eyeMask.sum()), "visibleContourCount": len(contours),
            "excludedNonlocalSkinPixels": int(np.count_nonzero(~socketLocal)),
            "minimumSocketSkinZ": float(eye[:, 2].min()-unitsPerPixel),
            "maximumSocketSkinRadius": 2*radius,
            "selectedContourPixels": len(contour), "requestedSamples": 64,
            "eligibleArcSamples": eligibleCount, "sampleCount": len(samplePixels),
            "arcCoverageFraction": eligibleCount/64.,
            "maximumContourOffsetPixels": float(np.linalg.norm(samplePixels-contourSamples, axis=1).max()),
            "sphereFitMaximumResidual": float(residual.max()),
            "sphereFitMedianResidual": float(np.median(residual)),
            "excludedMaterials": [3], "bindingUnits": "originalTemplateUnits",
            "scope": "Largest visible eye component; adjacent skin pixel centers, not anatomical ground truth",
        },
    }
