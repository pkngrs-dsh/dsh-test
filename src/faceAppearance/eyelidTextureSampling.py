"""Local photo resampling around fixed client eyelids, with explicit provenance.

This experimental module never moves vertices, UVs, or accessories. Eyelid
curve correspondence relocates photo queries close to the edge; its influence
vanishes smoothly outside a finite band. Relocated or exterior-extended color
is an estimate, not a direct observation of the destination surface. All four
bilinear source pixels must be valid and outside BOTH photographic eye
openings. Missing support leaves the baseline atlas unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import samplePixels, trianglePixels, validateTriangles


def _smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def _rgb(value, name):
    value = np.asarray(value)
    if (value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 2
            or value.dtype.kind != 'f' or not np.isfinite(value).all()
            or (value < 0).any() or (value > 1).any()):
        raise ValueError(f'{name} must be finite floating HxWx3 linear RGB in [0,1]')
    return value


def _geometry(value):
    vertices, faces = validateTriangles(value['vertices'], value['faces'], 3)
    uvs, uvFaces = validateTriangles(value['uvs'], value['uvFaces'], 2)
    if (faces.shape != uvFaces.shape or (uvs < 0).any() or (uvs > 1).any()
            or value.get('uvConvention', 'topLeftPixelEdges') != 'topLeftPixelEdges'):
        raise ValueError('Geometry needs original matching per-corner topLeftPixelEdges UVs in [0,1]')
    extent = float(np.ptp(vertices, axis=0).max())
    if extent <= 0:
        raise ValueError('Geometry must have positive extent')
    return vertices, faces, uvs, uvFaces, extent


def _curves(value, imageShape, name):
    curves = {}
    for eyeName in ('imageLeft', 'imageRight'):
        eye = value[eyeName]
        curves[eyeName] = {}
        for part in ('upper', 'lower'):
            curve = np.asarray(eye[part], np.float64)
            if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3
                    or not np.isfinite(curve).all() or (curve < 0).any()
                    or (curve > np.array(imageShape[::-1]) - 1).any()
                    or (np.linalg.norm(np.diff(curve, axis=0), axis=1) < 1e-7).any()
                    or curve[0, 0] >= curve[-1, 0]):
                raise ValueError(f'{name}.{eyeName}.{part} must be an ordered nondegenerate left-to-right Nx2 photo curve')
            curves[eyeName][part] = curve
        upper, lower = curves[eyeName]['upper'], curves[eyeName]['lower']
        polygon = np.concatenate([upper, lower[::-1]])
        area = abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                          - polygon[:, 1] * np.roll(polygon[:, 0], -1))) * .5
        if area < 1:
            raise ValueError(f'{name}.{eyeName} eye opening is degenerate')
    return curves


def _lookup(vertices, faces, uvs, uvFaces, atlasShape, extent, pixelOffset=(0., 0.)):
    """Detect conflicting surface samples, including coincident UV islands."""
    height, width = atlasShape
    faceIds = np.full(atlasShape, -1, np.int32)
    weights = np.zeros((*atlasShape, 3), np.float64)
    alias = np.zeros(atlasShape, bool)
    pixelUvs = uvs * [width, height] - .5 - np.asarray(pixelOffset)
    tolerance = extent * 1e-7
    for faceIndex, uvFace in enumerate(uvFaces):
        raster = trianglePixels(pixelUvs[uvFace], height, width)
        if raster is None:
            continue
        yy, xx, bary = raster
        previous = faceIds[yy, xx]
        overlaps = previous >= 0
        if overlaps.any():
            newPoints = bary[overlaps] @ vertices[faces[faceIndex]]
            priorPoints = np.sum(vertices[faces[previous[overlaps]]]
                                 * weights[yy[overlaps], xx[overlaps], :, None], axis=1)
            conflicting = np.linalg.norm(newPoints - priorPoints, axis=1) > tolerance
            alias[yy[overlaps][conflicting], xx[overlaps][conflicting]] = True
        fresh = ~overlaps
        faceIds[yy[fresh], xx[fresh]] = faceIndex
        weights[yy[fresh], xx[fresh]] = bary[fresh]
    return faceIds, weights, alias


def _nearestSegments(query, start, end):
    vectors = end - start
    square = np.sum(vectors * vectors, axis=1)
    fraction = np.clip(np.sum((query[:, None] - start) * vectors, axis=2)
                       / np.maximum(square, 1e-16), 0., 1.)
    nearest = start + fraction[..., None] * vectors
    squaredDistance = np.sum((query[:, None] - nearest) ** 2, axis=2)
    segment = squaredDistance.argmin(axis=1)
    row = np.arange(len(query))
    return np.sqrt(squaredDistance[row, segment]), segment, fraction[row, segment]


def _outsidePolygon(query, polygon, margin):
    # Exact continuous segment distance plus an even-odd containment test.
    # Duplicate canthus endpoints have zero-length segments, which are harmless.
    first, last = polygon, np.roll(polygon, -1, axis=0)
    distance, _, _ = _nearestSegments(query, first, last)
    y = query[:, None, 1]
    crosses = (first[:, 1] > y) != (last[:, 1] > y)
    denominator = last[:, 1] - first[:, 1]
    xIntersection = first[:, 0] + (y - first[:, 1]) * (last[:, 0] - first[:, 0]) / np.where(abs(denominator) > 1e-15, denominator, 1.)
    inside = np.logical_xor.reduce(crosses & (query[:, None, 0] < xIntersection), axis=1)
    return ~inside & (distance >= margin)


def _sourceSafety(validMask, photoEyes, margin):
    height, width = validMask.shape
    safe = validMask >= .999
    for eye in photoEyes.values():
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        minimum = np.maximum(np.floor(polygon.min(0) - margin - 1).astype(int), 0)
        maximum = np.minimum(np.ceil(polygon.max(0) + margin + 1).astype(int), [width - 1, height - 1])
        yy, xx = np.mgrid[minimum[1]:maximum[1]+1, minimum[0]:maximum[0]+1]
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        safe[yy, xx] &= _outsidePolygon(xy, polygon, margin).reshape(yy.shape)
    # Every location in this cell has exclusively safe bilinear support.
    safeCells = safe[:-1, :-1] & safe[1:, :-1] & safe[:-1, 1:] & safe[1:, 1:]
    return safe, safeCells


def _querySupport(query, safeCells):
    height, width = safeCells.shape
    floors = np.floor(query).astype(np.int64)
    inside = (floors >= 0).all(1) & (floors < [width, height]).all(1)
    accepted = np.zeros(len(query), bool)
    accepted[inside] = safeCells[floors[inside, 1], floors[inside, 0]]
    return accepted


def _reprojectEyelidTexture(geometry, camera, sourceLinear, sourceValidMask,
                          baselineLinear, photoEyes, modelEyes, *,
                          writableMask=None, visibleScene=None, corePixels=3.,
                          outerPixels=14., sourceSafetyPixels=.75,
                          maxExtensionPixels=4., _lookupOffset=(0., 0.)):
    """Resample a finite eyelid band without modifying unrelated atlas texels.

    Geometry and optional visibleScene contain canonical vertices/faces. Camera
    is OpenCV world-to-camera (rotation, translation, intrinsics). Curves are
    dictionaries imageLeft/imageRight -> upper/lower -> Nx2 working pixels,
    ordered image-left to image-right; corresponding curves must share indices.
    writableMask must exclude UVs also used by any omitted accessory or chart.
    visibleScene should contain all opaque occluders; supplied triangles are
    conservatively treated as opaque on both sides.

    RGB inputs are already linear and color-matched by the caller. The output
    linearRGB preserves baseline dtype and values exactly wherever alpha==0.
    projectedXY records actual geometry projection; sourceXY records accepted
    final sampling. NaN means no applicable projection/sample. observedMask is
    direct projection only; estimatedMask comprises curve warp and/or exterior
    extension. Neither mask is a learned skin probability.
    """
    vertices, faces, uvs, uvFaces, extent = _geometry(geometry)
    source, baseline = _rgb(sourceLinear, 'sourceLinear'), _rgb(baselineLinear, 'baselineLinear')
    imageShape, atlasShape = source.shape[:2], baseline.shape[:2]
    validMask = np.asarray(sourceValidMask)
    if (validMask.shape != imageShape or validMask.dtype.kind not in 'bifu'
            or not np.isfinite(validMask).all() or (validMask < 0).any() or (validMask > 1).any()):
        raise ValueError('sourceValidMask must match source image and lie in [0,1]')
    if writableMask is None:
        writable = np.ones(atlasShape, bool)
    else:
        writable = np.asarray(writableMask)
        if writable.shape != atlasShape or writable.dtype != np.bool_:
            raise ValueError('writableMask must be a boolean atlas mask')
    if (not np.isfinite([corePixels, outerPixels, sourceSafetyPixels, maxExtensionPixels]).all()
            or corePixels < 0 or outerPixels <= corePixels
            or sourceSafetyPixels < 0 or maxExtensionPixels < 0):
        raise ValueError('Invalid finite eyelid band, source margin or extension limit')
    K, R, t = (np.asarray(camera[key], np.float64) for key in ('intrinsics', 'rotation', 'translation'))
    if (K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,)
            or not all(np.isfinite(v).all() for v in (K, R, t))
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
        raise ValueError('Camera needs finite OpenCV intrinsics, proper rotation and length-three translation')
    photoEyes = _curves(photoEyes, imageShape, 'photoEyes')
    modelEyes = _curves(modelEyes, imageShape, 'modelEyes')
    modelStart, modelEnd, photoStart, photoEnd = [], [], [], []
    for eyeName in ('imageLeft', 'imageRight'):
        for part in ('upper', 'lower'):
            photo, model = photoEyes[eyeName][part], modelEyes[eyeName][part]
            if photo.shape != model.shape:
                raise ValueError(f'{eyeName}.{part} model/photo curves need matching semantic point indices')
            modelStart.extend(model[:-1]); modelEnd.extend(model[1:])
            photoStart.extend(photo[:-1]); photoEnd.extend(photo[1:])
    modelStart, modelEnd, photoStart, photoEnd = map(np.asarray, (modelStart, modelEnd, photoStart, photoEnd))
    cameraVertices = vertices @ R.T + t
    if visibleScene is None:
        sceneVertices, sceneFaces = vertices, faces
    else:
        sceneVertices, sceneFaces = validateTriangles(visibleScene['vertices'], visibleScene['faces'], 3)
    visibility = ProjectedSurfaceVisibility(sceneVertices @ R.T + t, sceneFaces, K, imageShape)
    faceIds, barycentrics, alias = _lookup(vertices, faces, uvs, uvFaces, atlasShape, extent, _lookupOffset)
    yy, xx = np.nonzero((faceIds >= 0) & writable & ~alias)
    ids = faceIds[yy, xx]
    points = np.sum(cameraVertices[faces[ids]] * barycentrics[yy, xx, :, None], axis=1)
    homogeneous = points @ K.T
    xy = np.divide(homogeneous[:, :2], homogeneous[:, 2:], out=np.full((len(points), 2), -1.), where=homogeneous[:, 2:] > 1e-8)
    bounds = (points[:, 2] > 1e-8) & (xy >= 0).all(1) & (xy < np.array(imageShape[::-1]) - 1).all(1)
    # Bound large atlases before the segment query; no full-head dense NxS array.
    allEndpoints = np.concatenate([modelStart, modelEnd])
    minimum = allEndpoints.min(0) - outerPixels
    maximum = allEndpoints.max(0) + outerPixels
    localIndices = np.flatnonzero(bounds & (xy >= minimum).all(1) & (xy <= maximum).all(1))
    distance, segment, fraction = _nearestSegments(xy[localIndices], modelStart, modelEnd)
    inBand = distance < outerPixels
    localIndices, distance, segment, fraction = (v[inBand] for v in (localIndices, distance, segment, fraction))
    triangles = cameraVertices[faces[ids[localIndices]]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    facing = np.sum(normals * -points[localIndices], axis=1) > 0
    depthTolerance = extent * 1e-6
    hit = visibility.closestDepth(points[localIndices])
    visible = np.isfinite(hit['depth']) & (abs(hit['depth'] - points[localIndices, 2]) <= depthTolerance)
    eligible = facing & visible
    eligibleIndices = localIndices[eligible]
    distance, segment, fraction = distance[eligible], segment[eligible], fraction[eligible]
    actual = xy[eligibleIndices]
    amount = 1. - _smooth((distance - corePixels) / (outerPixels - corePixels))
    correspondingModel = modelStart[segment] + fraction[:, None] * (modelEnd[segment] - modelStart[segment])
    correspondingPhoto = photoStart[segment] + fraction[:, None] * (photoEnd[segment] - photoStart[segment])
    displacement = (correspondingPhoto - correspondingModel) * amount[:, None]
    query = actual + displacement
    warped = np.linalg.norm(displacement, axis=1) > 1e-6
    safePixels, safeCells = _sourceSafety(validMask, photoEyes, sourceSafetyPixels)
    supported = _querySupport(query, safeCells)
    extended = np.zeros(len(query), bool)
    extensionDistance = np.zeros(len(query), np.float64)
    if maxExtensionPixels > 0 and safeCells.any() and (~supported).any():
        # Nearest *safe bilinear cell center*, bounded in distance. This is
        # conservative color continuation, explicitly tagged as estimated.
        _, nearest = distance_transform_edt(~safeCells, return_indices=True)
        rejected = np.flatnonzero(~supported)
        cells = np.floor(query[rejected]).astype(np.int64)
        clipped = np.clip(cells, [0, 0], np.array(safeCells.shape[::-1]) - 1)
        donor = np.column_stack([nearest[1, clipped[:, 1], clipped[:, 0]], nearest[0, clipped[:, 1], clipped[:, 0]]]).astype(np.float64) + .5
        extension = np.linalg.norm(donor - query[rejected], axis=1)
        recovered = (extension <= maxExtensionPixels) & _querySupport(donor, safeCells)
        accepted = rejected[recovered]
        query[accepted] = donor[recovered]
        extended[accepted] = True
        supported[accepted] = True
        extensionDistance[accepted] = extension[recovered]
    accepted = np.flatnonzero(supported & (amount > 1e-7))
    chosen = eligibleIndices[accepted]
    index = (yy[chosen], xx[chosen])
    alpha = np.zeros(atlasShape, np.float32)
    alpha[index] = amount[accepted]
    linear = baseline.copy()
    sampled = samplePixels(source, query[accepted])
    blended = baseline[index].astype(np.float64) * (1. - amount[accepted, None]) + sampled.astype(np.float64) * amount[accepted, None]
    linear[index] = np.clip(blended, 0., 1.).astype(baseline.dtype)
    projectedXY = np.full((*atlasShape, 2), np.nan, np.float32)
    projectionMask = np.zeros(atlasShape, bool)
    projectionMask[yy[localIndices], xx[localIndices]] = True
    projectedXY[projectionMask] = xy[localIndices]
    # Preserve the coordinates actually sampled: float32 rounding near an
    # integer can change floor(query) and therefore its four source pixels.
    sourceXY = np.full((*atlasShape, 2), np.nan, np.float64)
    sourceXY[index] = query[accepted]
    warpMask, extensionMask = np.zeros(atlasShape, bool), np.zeros(atlasShape, bool)
    warpMask[index], extensionMask[index] = warped[accepted], extended[accepted]
    estimated = warpMask | extensionMask
    observed = (alpha > 0) & ~estimated
    changed = np.any(linear != baseline, axis=2)
    if (changed & ~writable).any() or (changed & alias).any() or (changed & ~(alpha > 0)).any():
        raise RuntimeError('Eyelid texture sampling escaped its explicitly writable support')
    return {'linearRGB': linear, 'alpha': alpha, 'observedMask': observed,
            'estimatedMask': estimated, 'warpMask': warpMask, 'extensionMask': extensionMask,
            'projectedXY': projectedXY, 'sourceXY': sourceXY,
            'projectionMask': projectionMask, 'aliasMask': alias, 'surfaceFaceIds': faceIds,
            'sourceSafePixelMask': safePixels, 'sourceSafeCellMask': safeCells,
            'diagnostics': {
                'method': 'fixedGeometryLocalEyelidCurveWarpWithSafeBilinearSourceSupport',
                'coreWorkingPixels': float(corePixels), 'outerWorkingPixels': float(outerPixels),
                'sourceSafetyWorkingPixels': float(sourceSafetyPixels),
                'maximumExtensionWorkingPixels': float(maxExtensionPixels),
                'inputUvTexels': int((faceIds >= 0).sum()), 'uvAliasExcludedTexels': int(alias.sum()),
                'localProjectedTexels': int(len(localIndices)),
                'backFacingLocalTexels': int((~facing).sum()),
                'occludedLocalTexels': int((~visible).sum()),
                'visibleFrontFacingTexels': int(len(eligibleIndices)),
                'unsafeSourceRejectedTexels': int((~supported).sum()),
                'sourceSampledTexels': int(len(accepted)), 'changedTexels': int(changed.sum()),
                'directObservationTexels': int(observed.sum()), 'estimatedTexels': int(estimated.sum()),
                'curveWarpTexels': int(warpMask.sum()), 'exteriorExtendedTexels': int(extensionMask.sum()),
                'maximumAppliedWarpWorkingPixels': float(np.linalg.norm(displacement[accepted], axis=1).max()) if len(accepted) else 0.,
                'maximumAppliedExtensionWorkingPixels': float(extensionDistance[accepted].max()) if len(accepted) else 0.,
                'completeSceneProvided': visibleScene is not None,
                'allSourceBilinearSupportOutsidePhotoEyes': True,
                'sourceColorGainAppliedInternally': False,
                'noSmoothingOrGeneratedSkin': True,
                'unmodifiedOutsideAlphaExactly': bool(np.array_equal(linear[alpha == 0], baseline[alpha == 0])),
                'visibility': visibility.diagnostics,
                'depthTolerance': float(depthTolerance),
                'limitations': ['Dense photographic curves are not manual ground truth',
                                'Warped or exterior-extended color is estimated, not a surface observation',
                                'Caller must exclude accessory/shared atlas charts in writableMask',
                                'Provided scene triangles are conservatively opaque; alpha handling belongs to caller']}}



def reprojectEyelidTexture(geometry, camera, sourceLinear, sourceValidMask,
                          baselineLinear, photoEyes, modelEyes, *,
                          writableMask=None, visibleScene=None, corePixels=3.,
                          outerPixels=14., sourceSafetyPixels=.75,
                          maxExtensionPixels=4., filterFootprint=False):
    """Run local fixed-geometry eye sampling, optionally with filter support.

    Input/output conventions are documented by _reprojectEyelidTexture. The
    optional 4x4 tent footprint changes only writable texels with alpha == 0.
    Each accepted subposition passes the SAME projection, occlusion, curve
    warp and photographic eye-exclusion gates as a center. Contributor faces
    must connect through shared (geometry vertex, UV vertex), including the
    center owner when one exists. Filter padding is never an observed center.
    Its sparse filterSample* arrays record every accepted contributing query.
    """
    if not isinstance(filterFootprint, (bool, np.bool_)):
        raise ValueError('filterFootprint must be boolean')
    options = dict(writableMask=writableMask, visibleScene=visibleScene,
                   corePixels=corePixels, outerPixels=outerPixels,
                   sourceSafetyPixels=sourceSafetyPixels,
                   maxExtensionPixels=maxExtensionPixels)
    result = _reprojectEyelidTexture(geometry, camera, sourceLinear, sourceValidMask,
                                    baselineLinear, photoEyes, modelEyes, **options)
    shape = result['alpha'].shape
    padding = np.zeros(shape, bool)
    filterAlpha = np.zeros(shape, np.float32)
    kernelFraction = np.zeros(shape, np.float32)
    result.update(filterPadding=padding, filterAlpha=filterAlpha,
                  filterSupportedKernelFraction=kernelFraction)
    if not filterFootprint:
        result['diagnostics']['filterFootprint'] = {'enabled': False}
        return result
    writable = np.ones(shape, bool) if writableMask is None else np.asarray(writableMask)
    candidate = writable & (result['alpha'] == 0) & ~result['aliasMask']
    # Restrict the work to the one-texel filter footprint of the actual local
    # projection, including outside the rasterized chart. Full lookup still
    # checks aliases, and every subposition still receives exact visibility.
    from scipy.ndimage import binary_dilation
    candidate &= binary_dilation(result['projectionMask'], iterations=2)
    options['writableMask'] = candidate
    corners = [set(zip(face.tolist(), uvFace.tolist()))
               for face, uvFace in zip(np.asarray(geometry['faces']), np.asarray(geometry['uvFaces']))]
    owners = {}
    rgbSum = np.zeros_like(result['linearRGB'], dtype=np.float64)
    weightSum = np.zeros(shape, np.float64)
    alphaSum = np.zeros(shape, np.float64)
    rows = []
    offsets = (-.75, -.25, .25, .75)
    totalWeight = sum((1 - abs(dx)) * (1 - abs(dy)) for dy in offsets for dx in offsets)
    for dy in offsets:
        for dx in offsets:
            sampled = _reprojectEyelidTexture(geometry, camera, sourceLinear, sourceValidMask,
                        baselineLinear, photoEyes, modelEyes, _lookupOffset=(dx, dy), **options)
            support = sampled['alpha'] > 0
            if not support.any():
                continue
            yy, xx = np.nonzero(support)
            weight = (1 - abs(dx)) * (1 - abs(dy))
            rgbSum[support] += weight * sampled['linearRGB'][support]
            weightSum[support] += weight
            alphaSum[support] += weight * sampled['alpha'][support]
            for y, x, faceId in zip(yy, xx, sampled['surfaceFaceIds'][support]):
                owners.setdefault((int(y), int(x)), set()).add(int(faceId))
            rows.append({'atlasXY': np.column_stack([xx, yy]),
                         'offsetXY': np.tile([dx, dy], (len(xx), 1)),
                         'projectedXY': sampled['projectedXY'][support],
                         'sourceXY': sampled['sourceXY'][support],
                         'weight': np.full(len(xx), weight),
                         'alpha': sampled['alpha'][support],
                         'faceIds': sampled['surfaceFaceIds'][support],
                         'warped': sampled['warpMask'][support],
                         'extended': sampled['extensionMask'][support]})
    padding[:] = weightSum > 0
    disconnected = 0
    for pixel, contributors in owners.items():
        remaining = set(contributors)
        centerFace = int(result['surfaceFaceIds'][pixel])
        if centerFace >= 0:
            remaining.add(centerFace)
        connected = {remaining.pop()}
        while remaining:
            neighbors = {face for face in remaining
                         if any(corners[face] & corners[other] for other in connected)}
            if not neighbors:
                break
            connected.update(neighbors)
            remaining.difference_update(neighbors)
        if remaining:
            padding[pixel] = False
            disconnected += 1
    if padding.any():
        result['linearRGB'][padding] = np.clip(rgbSum[padding] / weightSum[padding, None], 0, 1).astype(result['linearRGB'].dtype)
        filterAlpha[padding] = alphaSum[padding] / weightSum[padding]
        kernelFraction[padding] = weightSum[padding] / totalWeight
    for name, example in (('atlasXY', np.empty((0, 2), np.int64)),
                          ('offsetXY', np.empty((0, 2))),
                          ('projectedXY', np.empty((0, 2))),
                          ('sourceXY', np.empty((0, 2))),
                          ('weight', np.empty(0)), ('alpha', np.empty(0)),
                          ('faceIds', np.empty(0, np.int32)),
                          ('warped', np.empty(0, bool)), ('extended', np.empty(0, bool))):
        retained = [row[name][padding[row['atlasXY'][:, 1], row['atlasXY'][:, 0]]] for row in rows]
        result['filterSample' + name[0].upper() + name[1:]] = np.concatenate(retained) if retained else example
    baseline = np.asarray(baselineLinear)
    unchanged = ~(padding | (result['alpha'] > 0))
    if not np.array_equal(result['linearRGB'][unchanged], baseline[unchanged]):
        raise RuntimeError('Filter support modified a texel outside its recorded support')
    if (padding & (~writable | (result['alpha'] > 0))).any():
        raise RuntimeError('Filter support escaped writable unobserved-center domain')
    diagnostics = result['diagnostics']
    diagnostics['unmodifiedOutsideAlphaExactly'] = bool(np.array_equal(result['linearRGB'][result['alpha'] == 0], baseline[result['alpha'] == 0]))
    diagnostics['unmodifiedOutsideAlphaAndFilterExactly'] = True
    diagnostics['changedTexels'] = int(np.any(result['linearRGB'] != baseline, axis=2).sum())
    diagnostics['filterFootprint'] = {
        'enabled': True, 'filterPaddingTexels': int(padding.sum()),
        'kernel': 'bilinear tent, radius one atlas texel, 4x4 subpositions',
        'disconnectedUvGeometryFootprintsRejected': int(disconnected),
        'newObservedCenterTexels': 0,
        'observedAndEstimatedCenterMasksUnchanged': True,
        'source': 'same exact visibility and safe photographic bilinear support as centers',
        'adjacency': 'connected shared geometry-UV corner pairs including center owner',
        'meaning': 'filter support only, not new observed centers or geometry repair'}
    return result
