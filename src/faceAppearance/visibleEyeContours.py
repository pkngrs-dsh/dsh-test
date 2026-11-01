"""Local experiment: photo-camera visible eye boundaries and frozen skin bindings.

The boundary comes from the first visible material of the complete assembly,
including cutout accessories. It is not an anatomical detector. The photograph
supplies upper/lower-lid semantics and eye association, while raster visibility
supplies the model's actual aperture. All cameras and vertices must already use
the same UFF canonical space. No geometry or production pipeline is modified.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .camera import projectPoints


def loadLegacyEyeBindingInputs(archivePath, nicpToFullIds):
    """Load the real ZIP without changing vertex, material, UV, or alpha identity.

    Return keyword arguments for :func:`bindVisibleEyeContours`. ZIP vertices
    are meters; the exact recorded similarity operates on client centimeters.
    Alpha arrays use the same top-left-origin UV convention as the ZIP viewer.
    """
    from PIL import Image
    from .legacyPreviewAsset import _load

    source = _load(Path(archivePath))
    transform = source.manifest['assemblyDiagnostics']['templateToUff']
    vertices = (np.asarray(source.mesh['vertices'], float)*100
                @ np.asarray(transform['rotation'], float)*float(transform['scale'])
                + np.asarray(transform['translation'], float))
    materials = source.materials['materials']
    alpha = {}
    for index, material in enumerate(materials):
        if material['alphaMode'] == 'MASK':
            image = Image.open(io.BytesIO(source.payloads[material['baseColorTexture']['path']]))
            alpha[index] = np.asarray(image.convert('RGBA'), np.float64)[..., 3]/255
    return {'vertices': vertices, 'faces': source.mesh['faces'],
            'materialIds': source.mesh['materialIds'], 'nicpToFullIds': np.asarray(nicpToFullIds),
            'uvs': source.mesh['uvs'], 'uvFaces': source.mesh['uvFaces'],
            'materials': materials, 'alphaTextures': alpha}


def _project(vertices, camera):
    return projectPoints(vertices, np.asarray(camera['intrinsics'], float),
                         np.asarray(camera['rotation'], float), np.asarray(camera['translation'], float))


def _alphaSample(alpha, uv):
    """Bilinear border sampling equivalent to grid_sample align_corners=False."""
    height, width = alpha.shape
    xy = np.clip(uv, 0, 1)*[width, height]-.5
    lo = np.floor(xy).astype(np.int64)
    fraction = xy-lo
    hi = lo+1
    lo = np.clip(lo, [0, 0], [width-1, height-1])
    hi = np.clip(hi, [0, 0], [width-1, height-1])
    x, y = fraction[:, 0], fraction[:, 1]
    return ((1-x)*(1-y)*alpha[lo[:, 1], lo[:, 0]]
            + x*(1-y)*alpha[lo[:, 1], hi[:, 0]]
            + (1-x)*y*alpha[hi[:, 1], lo[:, 0]]
            + x*y*alpha[hi[:, 1], hi[:, 0]])


def _raster(vertices, faces, materialIds, camera, roi, uvs, uvFaces, materials, alphaTextures):
    """Integer-centered ROI raster with exact perspective-correct barycentrics.

    Rasterize only overlapping, front-facing triangles. Per-fragment alpha is
    tested before choosing the nearest depth, so transparent cards reveal every
    deeper surface without a depth-layer limit. All original face IDs survive.
    """
    height, width = roi['shape']
    origin, scale = np.asarray(roi['origin'], float), roi['scale']
    pixels, depth = _project(vertices, camera)
    pixels = (pixels-origin)*scale
    cv = vertices @ np.asarray(camera['rotation'], float).T + np.asarray(camera['translation'], float)
    triangle = cv[faces]
    normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    facing = (normal*triangle.mean(1)).sum(1) < -1e-14
    projected = pixels[faces]
    lo, hi = projected.min(1), projected.max(1)
    overlap = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] <= width-1) & (lo[:, 1] <= height-1)
    candidates = np.flatnonzero(facing & overlap & (depth[faces] > 1e-8).all(1))
    zbuffer = np.full((height, width), np.inf, np.float64)
    hits = np.full((height, width), -1, np.int64)
    weights = np.zeros((height, width, 3), np.float64)
    rejectedAlpha, alphaCandidates = 0, 0
    for faceId in candidates:
        ids = faces[faceId]
        first, second, third = pixels[ids]
        denominator = ((second[1]-third[1])*(first[0]-third[0])
                       + (third[0]-second[0])*(first[1]-third[1]))
        if abs(denominator) < 1e-12:
            continue
        lower = np.maximum(np.ceil(projected[faceId].min(0)-1e-8).astype(int), 0)
        upper = np.minimum(np.floor(projected[faceId].max(0)+1e-8).astype(int), [width-1, height-1])
        if (lower > upper).any():
            continue
        yy, xx = np.mgrid[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
        w0 = ((second[1]-third[1])*(xx-third[0]) + (third[0]-second[0])*(yy-third[1]))/denominator
        w1 = ((third[1]-first[1])*(xx-third[0]) + (first[0]-third[0])*(yy-third[1]))/denominator
        linear = np.stack([w0, w1, 1-w0-w1], -1)
        inside = (linear >= -1e-8).all(-1)
        inverse = np.maximum(linear, 0)/depth[ids]
        total = inverse.sum(-1)
        bary = inverse/np.maximum(total[..., None], 1e-30)
        z = 1/np.maximum(total, 1e-30)
        materialId = int(materialIds[faceId])
        if materials is not None and materials[materialId]['alphaMode'] == 'MASK':
            uv = bary[inside] @ uvs[uvFaces[faceId]]
            alpha = _alphaSample(alphaTextures[materialId], uv)
            alpha *= float(materials[materialId].get('baseColorFactor', [1, 1, 1, 1])[3])
            good = alpha >= float(materials[materialId].get('alphaCutoff', .5))
            alphaCandidates += len(alpha)
            rejectedAlpha += int((~good).sum())
            inside[inside] = good
        region = np.s_[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
        take = inside & (z < zbuffer[region])
        zbuffer[region][take] = z[take]
        hits[region][take] = faceId
        weights[region][take] = bary[take]
    return hits, weights, zbuffer, {
        'inputFaces': len(faces), 'overlappingFrontFaces': len(candidates),
        'alphaTestedFragments': alphaCandidates, 'alphaRejectedFragments': rejectedAlpha,
        'perspectiveCorrect': True, 'backfaceCulling': True,
        'pixelConvention': 'ROI integer centers map to origin + pixel / scale',
        'alphaSampling': 'bilinear border, align_corners=False; test before depth selection'}


def _components(faces, materialIds, eyeIds):
    selected = np.flatnonzero(np.isin(materialIds, eyeIds))
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for face in faces[selected]:
        leader = root(int(face[0]))
        for vertex in face[1:]:
            parent[root(int(vertex))] = leader
    grouped = {}
    for faceId in selected:
        grouped.setdefault(root(int(faces[faceId, 0])), []).append(int(faceId))
    return [np.asarray(value, np.int64) for value in grouped.values()]


def _sphere(vertices):
    mean = vertices.mean(0)
    centered = vertices-mean
    answer, _, rank, _ = np.linalg.lstsq(np.c_[2*centered, np.ones(len(vertices))],
                                        np.sum(centered**2, axis=1), rcond=None)
    radius2 = answer[3]+answer[:3]@answer[:3]
    if rank != 4 or radius2 <= 0 or not np.isfinite(radius2):
        raise ValueError('Eye component does not have a stable sphere fit')
    center, radius = mean+answer[:3], float(np.sqrt(radius2))
    return center, radius, float(np.max(np.abs(np.linalg.norm(vertices-center, axis=1)-radius)))


def _resample(points, count):
    points = np.asarray(points, float)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] <= 1e-10:
        raise ValueError('Degenerate eye curve')
    distance = np.linspace(0, arc[-1], count)
    return np.c_[np.interp(distance, arc, points[:, 0]), np.interp(distance, arc, points[:, 1])]


def _emptyPart(count, targets):
    return {'vertexIndices': np.full((count, 3), -1, np.int64),
            'barycentricWeights': np.zeros((count, 3)), 'fullFaceIds': np.full(count, -1, np.int64),
            'canonicalPoints': np.zeros((count, 3)), 'sourcePixels': np.zeros((count, 2)),
            'boundPixels': np.zeros((count, 2)), 'pixelOffsets': np.zeros((count, 2)),
            'targetPixels': targets, 'acceptedMask': np.zeros(count, bool),
            'reasons': [['eye_rejected'] for _ in range(count)]}


def bindVisibleEyeContours(vertices, faces, materialIds, nicpToFullIds, camera, photoEyes, *,
        imageShape, uvs=None, uvFaces=None, materials=None, alphaTextures=None,
        skinMaterialIds=(0,), eyeMaterialIds=(2,), pointsPerLid=12,
        minimumEyeRasterWidth=256, minimumLidCoverage=.75, maximumRoiSize=768,
        apertureAccessoryMode='off', apertureAccessoryFaceIds=None,
        _measurementOnly=False):
    """Freeze photo-aligned, visible skin bindings around each actual eye aperture.

    ``photoEyes`` is ``{eyeName: {'upper': Nx2, 'lower': Nx2}}`` in the saved
    camera's pixels. Curves must trace a lid continuously; either endpoint
    orientation is accepted and normalized to image-left -> image-right.
    Eye names retain the caller's subject-side semantics. Upper and lower each
    return ``pointsPerLid`` samples including both corners; duplicated corner
    observations should be downweighted by the fitting caller.

    ``vertices`` is the FULL assembled mesh in UFF world coordinates. Returned
    ``vertexIndices`` address NICP skin order, with barycentrics in that exact
    triangle corner order. Fit projected barycentric points to
    ``targetPixels - pixelOffsets``: offsets retain the baseline aperture-to-
    adjacent-skin displacement. This is a local approximation; independently
    re-extract candidate contours for acceptance. Do not update these frozen
    constraints during the solve.

    Incomplete/occluded lids produce status ``rejected`` and no accepted samples.
    A complete eye needs >= minimumLidCoverage on EACH lid, supported corners,
    and >= .80 adjacency coverage of the whole visible material contour.
    Missing alpha data is an error, never silently treated as opaque. The
    measured boundary always includes all visible local skin material, including
    static head accessories. Fitting additionally requires an adjacent NICP
    triangle. A candidate's measurable contour need not be newly bindable.

    ``apertureAccessoryMode='innerCornerHeadAccessories'`` explicitly includes
    the caller-audited inner-corner accessory faces in the eye aperture. Supply
    their original full-mesh face IDs as ``apertureAccessoryFaceIds``. This is
    required to compare the whole eyelid opening rather than just visible eye
    material. Face/component topology, non-NICP material-0 identity, same-side
    proximity, nasal placement, and connected visible aperture are validated;
    no accessory is inferred solely from its size or material. Default is off.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt

    v, f, m, mapping = np.asarray(vertices, float), np.asarray(faces), np.asarray(materialIds), np.asarray(nicpToFullIds)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu' or not len(f)
            or f.min() < 0 or f.max() >= len(v) or m.shape != (len(f),) or m.dtype.kind not in 'iu'
            or mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or not len(mapping)
            or mapping.min() < 0 or mapping.max() >= len(v) or len(np.unique(mapping)) != len(mapping)):
        raise ValueError('Invalid full assembly or NICP index mapping')
    if (len(imageShape) != 2 or min(imageShape) < 2 or not 6 <= pointsPerLid <= 64
            or not 64 <= minimumEyeRasterWidth <= maximumRoiSize or not .5 <= minimumLidCoverage <= 1):
        raise ValueError('Invalid image shape or contour sampling policy')
    if camera.get('valid', True) is not True:
        raise ValueError('Cannot bind to a rejected photo camera')
    k, r, t = (np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation'))
    if (k.shape != (3, 3) or r.shape != (3, 3) or t.size != 3
            or not np.isfinite(k).all() or not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('Expected a finite OpenCV pinhole camera')
    camera = {**camera, 'intrinsics': k, 'rotation': r, 'translation': t.reshape(3)}
    if materials is None and np.any(~np.isin(m, np.r_[skinMaterialIds, eyeMaterialIds])):
        raise ValueError('Full assembly requires explicit materials and alpha for other parts')
    if materials is not None:
        if m.min() < 0 or m.max() >= len(materials):
            raise ValueError('Material IDs outside material list')
        for index, material in enumerate(materials):
            if material.get('alphaMode') not in ('OPAQUE', 'MASK'):
                raise ValueError('Only OPAQUE and MASK are supported')
            if material['alphaMode'] == 'MASK':
                if uvs is None or uvFaces is None or alphaTextures is None or index not in alphaTextures:
                    raise ValueError('Cutout materials require original UVs and alpha textures')
                alpha = np.asarray(alphaTextures[index])
                if alpha.ndim != 2 or not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
                    raise ValueError('Alpha textures must be finite HxW arrays in [0,1]')
    if uvs is not None:
        uvs, uvFaces = np.asarray(uvs, float), np.asarray(uvFaces)
        if (uvs.ndim != 2 or uvs.shape[1] != 2 or not np.isfinite(uvs).all()
                or uvFaces.shape != f.shape or uvFaces.dtype.kind not in 'iu'
                or uvFaces.min() < 0 or uvFaces.max() >= len(uvs)):
            raise ValueError('Invalid original UV arrays')
    inverse = np.full(len(v), -1, np.int64)
    inverse[mapping] = np.arange(len(mapping))
    skinFaces = np.isin(m, skinMaterialIds) & (inverse[f] >= 0).all(1)
    components = _components(f, m, eyeMaterialIds)
    if not components or not isinstance(photoEyes, dict) or not photoEyes:
        raise ValueError('Expected eye material components and photo eye curves')
    fullPixels, depths = _project(v, camera)
    componentCenters = []
    for component in components:
        ids = np.unique(f[component])
        if (depths[ids] <= 1e-8).any():
            componentCenters.append(np.full(2, np.inf))
        else:
            componentCenters.append(fullPixels[ids].mean(0))
    if apertureAccessoryMode not in ('off', 'innerCornerHeadAccessories'):
        raise ValueError('Unknown eye aperture accessory policy')
    accessoryGroups = []
    if apertureAccessoryMode == 'innerCornerHeadAccessories':
        accessoryIds = np.asarray(apertureAccessoryFaceIds)
        if (accessoryIds.ndim != 1 or accessoryIds.dtype.kind not in 'iu' or not len(accessoryIds)
                or accessoryIds.min() < 0 or accessoryIds.max() >= len(f)
                or len(np.unique(accessoryIds)) != len(accessoryIds)
                or not np.isin(m[accessoryIds], skinMaterialIds).all()
                or (inverse[f[accessoryIds]] >= 0).any()):
            raise ValueError('Explicit aperture accessories must be original, entirely non-NICP skin faces')
        syntheticMaterials = np.isin(np.arange(len(f)), accessoryIds).astype(np.int64)
        for group in _components(f, syntheticMaterials, (1,)):
            mean = fullPixels[np.unique(f[group])].mean(0)
            associated = int(np.linalg.norm(np.asarray(componentCenters)-mean, axis=1).argmin())
            accessoryGroups.append((associated, group, mean))
        if len(accessoryGroups) != len(components) or len(set(item[0] for item in accessoryGroups)) != len(components):
            raise ValueError('Expected exactly one explicit inner-corner accessory per eye component')
    elif apertureAccessoryFaceIds is not None:
        raise ValueError('Explicit accessory faces require the innerCornerHeadAccessories policy')
    usedComponents, resultEyes = set(), {}
    for name, eye in photoEyes.items():
        targets = {}
        for lid in ('upper', 'lower'):
            curve = np.asarray(eye[lid], float)
            if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 4 or not np.isfinite(curve).all():
                raise ValueError('Each eye needs finite continuous upper/lower curves with >=4 points')
            if curve[0, 0] > curve[-1, 0]:
                curve = curve[::-1]
            targets[lid] = curve.copy() if len(curve) == pointsPerLid else _resample(curve, pointsPerLid)
        photo = np.concatenate(list(targets.values()))
        eyeWidth = float(np.ptp(photo[:, 0]))
        if eyeWidth < 4:
            raise ValueError('Photograph eye width is too small for meaningful shape fitting')
        output = {lid: _emptyPart(pointsPerLid, targets[lid]) for lid in ('upper', 'lower')}
        output.update(status='rejected', reasons=[], diagnostics={})
        resultEyes[name] = output
        componentId = int(np.linalg.norm(np.asarray(componentCenters)-photo.mean(0), axis=1).argmin())
        if componentId in usedComponents:
            output['reasons'].append('duplicate_eye_component_assignment')
            continue
        usedComponents.add(componentId)
        component, componentVertices = components[componentId], np.unique(f[components[componentId]])
        centerError = float(np.linalg.norm(componentCenters[componentId]-photo.mean(0))/eyeWidth)
        if not np.isfinite(centerError) or centerError > .8:
            output['reasons'].append('eye_component_too_far_from_photo')
            continue
        try:
            center, radius, sphereResidual = _sphere(v[componentVertices])
        except ValueError as error:
            output['reasons'].append(str(error))
            continue
        accessory = next((item for item in accessoryGroups if item[0] == componentId), None)
        apertureFaces = component
        if accessory is not None:
            _, accessoryFaces, accessoryCenter = accessory
            others = np.asarray([c for i, c in enumerate(componentCenters) if i != componentId])
            inward = others.mean(0)-np.asarray(componentCenters[componentId])
            accessoryDelta = accessoryCenter-np.asarray(componentCenters[componentId])
            accessoryVertices = np.unique(f[accessoryFaces])
            if (len(others) != 1 or accessoryDelta@inward <= 0
                    or np.linalg.norm(accessoryCenter-photo.mean(0)) > .8*eyeWidth
                    or np.linalg.norm(v[accessoryVertices]-center, axis=1).max() > 2*radius):
                output['reasons'].append('explicit_accessory_not_local_inner_corner')
                continue
            apertureFaces = np.r_[component, accessoryFaces]
        desired = np.r_[photo, fullPixels[np.unique(f[apertureFaces])]]
        low, high = desired.min(0), desired.max(0)
        margin = max(4., eyeWidth*.30)
        origin = np.maximum(np.floor(low-margin), [0, 0])
        end = np.minimum(np.ceil(high+margin), [imageShape[1]-1, imageShape[0]-1])
        scale = max(1., minimumEyeRasterWidth/eyeWidth)
        scale = min(scale, (maximumRoiSize-1)/max(end-origin))
        shape = tuple((np.ceil((end-origin)*scale).astype(int)+1)[::-1])
        roi = {'origin': origin.tolist(), 'scale': float(scale), 'shape': shape,
               'sourceImageShape': list(imageShape), 'toSourcePixels': 'roiXY / scale + origin'}
        hits, bary, zbuffer, rasterStats = _raster(v, f, m, camera, roi, uvs, uvFaces, materials, alphaTextures)
        hit = hits >= 0
        safeHits = np.maximum(hits, 0)
        eyeMask = hit & np.isin(hits, apertureFaces)
        # Static material-0 canthus accessories are visible skin too. They must
        # delimit measurement even though their vertices cannot supply an NICP
        # deformation binding. Conflating these two masks would falsely reject
        # a correctly measured aperture when its corner touches an accessory.
        skinMask = hit & np.isin(m[safeHits], skinMaterialIds) & ~eyeMask
        bindableSkinMask = hit & skinFaces[safeHits]
        eligible = skinMask.copy()
        yy, xx = np.nonzero(skinMask)
        skinPoints = (v[f[hits[yy, xx]]]*bary[yy, xx, :, None]).sum(1)
        cvEye = v[componentVertices] @ r.T+t.reshape(3)
        cvSkin = skinPoints @ r.T+t.reshape(3)
        worldPixel = radius/max(1., eyeWidth*scale)
        local = ((np.linalg.norm(skinPoints-center, axis=1) <= 2*radius)
                 & (cvSkin[:, 2] <= cvEye[:, 2].max()+2*worldPixel))
        eligible[yy[~local], xx[~local]] = False
        eligibleBindable = eligible & bindableSkinMask
        output['roi'], output['roiMasks'] = roi, {
            'eye': eyeMask, 'skin': skinMask, 'eligibleSkin': eligible,
            'bindableSkin': bindableSkinMask, 'eligibleBindableSkin': eligibleBindable,
            'faceIds': hits, 'materialIds': np.where(hit, m[safeHits], -1)}
        diagnostics = output['diagnostics']
        diagnostics.update(rasterStats, eyeComponentId=componentId, eyeComponentFaces=len(component),
            apertureAccessoryMode=apertureAccessoryMode,
            apertureAccessoryFaceIds=[] if accessory is None else accessory[1].tolist(),
            eyeVisiblePixels=int(eyeMask.sum()), photoEyeWidthPixels=eyeWidth,
            eyeComponentCenterDistanceWidths=centerError,
            sphereCenter=center.tolist(), sphereRadius=radius, sphereMaximumResidual=sphereResidual,
            excludedNonlocalSkinPixels=int((~local).sum()), rasterPixelSizeInPhoto=1/scale)
        if not eyeMask.any() or not eligible.any():
            output['reasons'].append('eye_or_adjacent_skin_invisible')
            continue
        if eyeMask[[0, -1], :].any() or eyeMask[:, [0, -1]].any():
            output['reasons'].append('eye_clipped_by_roi')
            continue
        contours, hierarchy = cv2.findContours(eyeMask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        external = [c[:, 0].astype(float) for i, c in enumerate(contours) if hierarchy[0, i, 3] == -1]
        if not external:
            output['reasons'].append('no_complete_eye_contour')
            continue
        contour = max(external, key=lambda c: abs(cv2.contourArea(c.astype(np.float32))))
        majorArea = abs(cv2.contourArea(contour.astype(np.float32)))
        otherArea = sum(abs(cv2.contourArea(c.astype(np.float32))) for c in external)-majorArea
        diagnostics.update(visibleContourCount=len(external), disconnectedAreaFraction=otherArea/max(1., majorArea))
        if len(contour) < 32 or majorArea < 64 or otherArea > .03*majorArea:
            output['reasons'].append('eye_fragmented_or_too_small')
            continue
        distances, nearest = distance_transform_edt(~eligible, return_indices=True)
        if eligibleBindable.any():
            bindingDistances, bindingNearest = distance_transform_edt(~eligibleBindable, return_indices=True)
        else:
            bindingDistances, bindingNearest = np.full_like(distances, np.inf), nearest
        contourInt = contour.astype(int)
        adjacency = distances[contourInt[:, 1], contourInt[:, 0]] <= 2.1
        diagnostics['wholeContourSkinAdjacencyFraction'] = float(adjacency.mean())
        # Canthus axis follows the photo to preserve a rolled head, with a
        # downward normal. It only labels/splits the model's measured contour.
        first = (targets['upper'][0]+targets['lower'][0])/2
        last = (targets['upper'][-1]+targets['lower'][-1])/2
        axis = last-first
        axis /= np.linalg.norm(axis)
        down = np.array([-axis[1], axis[0]])
        along = contour@axis
        left, right = int(along.argmin()), int(along.argmax())
        cyclic = np.r_[contour, contour]
        if left <= right:
            arcA, arcB = contour[left:right+1], cyclic[right:left+len(contour)+1][::-1]
        else:
            arcA, arcB = cyclic[left:right+len(contour)+1], contour[right:left+1][::-1]
        arcs = sorted([arcA, arcB], key=lambda a: float((a@down).mean()))
        diagnostics['contourPixels'] = len(contour)
        output['contourPixels'] = contour/scale+origin
        output['canthiPixels'] = contour[[left, right]]/scale+origin
        if adjacency.mean() < .80:
            output['reasons'].append('material_boundary_occluded_or_missing_skin')
        for lid, arc in zip(('upper', 'lower'), arcs):
            part = output[lid]
            samples = _resample(arc, pointsPerLid)
            sampleInt = np.rint(samples).astype(int)
            near = nearest[:, sampleInt[:, 1], sampleInt[:, 0]].T[:, ::-1]
            gap = np.linalg.norm(near-samples, axis=1)
            curveValid = gap <= 2.6
            boundNear = bindingNearest[:, sampleInt[:, 1], sampleInt[:, 0]].T[:, ::-1]
            bindingGap = np.linalg.norm(boundNear-samples, axis=1)
            faceIds = hits[boundNear[:, 1], boundNear[:, 0]]
            valid = (curveValid & (bindingGap <= 2.6) & skinFaces[faceIds]
                     & np.isfinite(bindingDistances[sampleInt[:, 1], sampleInt[:, 0]]))
            weights = bary[boundNear[:, 1], boundNear[:, 0]]
            points = (v[f[faceIds]]*weights[..., None]).sum(1)
            projectedBound, _ = _project(points, camera)
            # The midpoint of the inside-eye sample and its adjacent skin pixel
            # approximates the material transition with <=1 raster-pixel error.
            boundary = (samples+near)/2/scale+origin
            offsets = boundary-projectedBound
            part.update(vertexIndices=inverse[f[faceIds]], barycentricWeights=weights,
                        fullFaceIds=faceIds, canonicalPoints=points, sourcePixels=boundary,
                        boundPixels=projectedBound, pixelOffsets=offsets, acceptedMask=valid,
                        curveValidMask=curveValid, bindingValidMask=valid.copy(),
                        reasons=[[] if good else ['no_adjacent_visible_local_skin'] for good in valid])
            diagnostics[lid+'CoverageFraction'] = float(curveValid.mean())
            diagnostics[lid+'BindingCoverageFraction'] = float(valid.mean())
            diagnostics[lid+'MaximumBindingOffsetPixels'] = float(np.linalg.norm(offsets, axis=1).max())
            if curveValid.mean() < minimumLidCoverage:
                output['reasons'].append(lid+'_incomplete_visible_skin_support')
            if not curveValid[[0, -1]].all():
                output['reasons'].append(lid+'_canthus_not_visible_skin_supported')
            if not _measurementOnly:
                if valid.mean() < minimumLidCoverage:
                    output['reasons'].append(lid+'_incomplete_bindable_skin_support')
                if not valid[[0, -1]].all():
                    output['reasons'].append(lid+'_canthus_not_bindable')
        if not output['reasons']:
            output['status'] = 'accepted'
        else:
            for lid in ('upper', 'lower'):
                output[lid]['acceptedMask'][:] = False
                for reasons in output[lid]['reasons']:
                    reasons.extend(output['reasons'])
    return {'eyes': resultEyes, 'diagnostics': {
        'method': 'fullAssemblyPerspectiveVisibleEyeMaterialBoundary',
        'scope': 'local experimental frontal eye aperture; not anatomical ground truth',
        'acceptedEyes': sum(eye['status'] == 'accepted' for eye in resultEyes.values()),
        'requestedEyes': len(resultEyes), 'pointsPerLid': pointsPerLid,
        'bindingVertexOrder': 'NICP', 'bindingSpace': 'UFF canonical world',
        'curveOrder': 'imageLeftEndpointToImageRightEndpoint',
        'measurementOnly': bool(_measurementOnly),
        'apertureAccessoryMode': apertureAccessoryMode,
        'skinMaterialIds': list(skinMaterialIds), 'eyeMaterialIds': list(eyeMaterialIds),
        'fitTarget': 'targetPixels - pixelOffsets', 'frozenBaselineBindingRequired': True}}


def extractVisibleEyeContours(*args, **kwargs):
    """Re-extract a candidate's visible curves for independent contour scoring.

    Same signature/result as ``bindVisibleEyeContours``. Status evaluates the
    actual eye/local-skin boundary, including static skin accessories. The
    returned ``acceptedMask``/``bindingValidMask`` still identify only usable
    NICP bindings and may be false even for a measurable accepted eye. Never
    replace frozen baseline fitting constraints with these candidate bindings.
    Photo curves, camera, alpha and component association stay fixed.
    """
    return bindVisibleEyeContours(*args, **kwargs, _measurementOnly=True)


def evaluateApertureMask(observedEyesResult, photoEyes, *, originalPixelScale=1.):
    """Compare complete visible aperture masks without inventing a closed rim.

    In particular, a fragmented baseline can be measured faithfully. Every
    foreground component and every boundary is retained. Photo upper/lower
    observations form a filled polygon, not manually annotated truth. Input
    curves and camera use working pixels; ``originalPixelScale`` is working
    pixels per original-image pixel (isotropic only). Status from extraction
    remains separate from the numerical distances and cannot be overridden by
    a good score. Callers must require accepted candidate visibility and must
    check that candidate fragmentation does not increase.
    """
    import cv2
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not np.isfinite(originalPixelScale) or originalPixelScale <= 0:
        raise ValueError('Original pixel scale must be finite and positive')
    values = {}
    for name, eye in observedEyesResult['eyes'].items():
        if 'roiMasks' not in eye or 'roi' not in eye:
            raise ValueError(f'{name}: no visible aperture mask available for measurement')
        roi = eye['roi']
        model = np.asarray(eye['roiMasks']['eye'], bool)
        targetParts = []
        for lid in ('upper', 'lower'):
            points = np.asarray(photoEyes[name][lid], float)
            if points[0, 0] > points[-1, 0]:
                points = points[::-1]
            targetParts.append(points)
        polygon = (np.r_[targetParts[0], targetParts[1][::-1]]-np.asarray(roi['origin']))*roi['scale']
        if (not np.isfinite(polygon).all() or np.any(polygon < 0)
                or np.any(polygon[:, 0] > model.shape[1]-1) or np.any(polygon[:, 1] > model.shape[0]-1)):
            raise ValueError(f'{name}: photo aperture is clipped by the measured ROI')
        target = np.zeros(model.shape, np.uint8)
        # Subpixel fixed-point polygon coordinates avoid rounding the detected
        # lid to an integer ROI pixel before scan conversion.
        cv2.fillPoly(target, [np.rint(polygon*256).astype(np.int32)], 1, shift=8)
        target = target.astype(bool)
        if not model.any() or not target.any():
            raise ValueError(f'{name}: empty photo or model aperture mask')
        modelBoundary = model & ~binary_erosion(model, structure=np.ones((3, 3)), border_value=0)
        targetBoundary = target & ~binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
        unit = 1/(roi['scale']*originalPixelScale)
        modelToTarget = distance_transform_edt(~targetBoundary)[modelBoundary]*unit
        targetToModel = distance_transform_edt(~modelBoundary)[targetBoundary]*unit
        both = np.r_[modelToTarget, targetToModel]
        components, _, stats, _ = cv2.connectedComponentsWithStats(model.astype(np.uint8), 8)
        componentAreas = sorted((int(v) for v in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
        intersection, union = int((model & target).sum()), int((model | target).sum())
        values[name] = {
            'extractionStatus': eye['status'], 'extractionReasons': eye['reasons'],
            'intersectionOverUnion': intersection/union,
            'symmetricBoundaryMeanOriginalPixels': .5*(float(modelToTarget.mean())+float(targetToModel.mean())),
            'symmetricBoundaryP95OriginalPixels': float(np.percentile(both, 95)),
            'symmetricBoundaryMaximumOriginalPixels': float(both.max()),
            'modelToPhotoBoundaryMeanOriginalPixels': float(modelToTarget.mean()),
            'photoToModelBoundaryMeanOriginalPixels': float(targetToModel.mean()),
            'visibleComponentCount': int(components-1), 'visibleComponentRasterAreas': componentAreas,
            'modelRasterArea': int(model.sum()), 'photoRasterArea': int(target.sum()),
            'secondaryComponentAreaFraction': sum(componentAreas[1:])/sum(componentAreas),
            'roiRasterPixelSizeOriginalPixels': unit,
            'apertureAccessoryMode': eye['diagnostics'].get('apertureAccessoryMode', 'off'),
            'scope': 'Every visible full-aperture fragment against fixed dense automatic photo polygon; not manual ground truth'}
    return {'eyes': values,
            'meanIntersectionOverUnion': float(np.mean([e['intersectionOverUnion'] for e in values.values()])),
            'meanSymmetricBoundaryDistanceOriginalPixels': float(np.mean([
                e['symmetricBoundaryMeanOriginalPixels'] for e in values.values()])),
            'originalPixelScale': float(originalPixelScale),
            'allCandidateExtractionsAccepted': all(e['extractionStatus'] == 'accepted' for e in values.values()),
            'scope': 'Independent filled-aperture mask comparison; no new landmark binding and no contour completion'}
