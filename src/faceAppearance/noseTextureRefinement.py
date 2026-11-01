"""Conservative local nose sampling on a fixed accepted surface.

This is direct perspective reprojection, without a landmark warp or additional
view blending. Original photographic validity, successful SH correction and
all four source bilinear contributors are required. The returned support is a
sampling diagnostic, not new anatomical coverage or proof of true albedo.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.ndimage import distance_transform_edt

from .eyelidTextureSampling import _geometry, _lookup, _rgb
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import samplePixels, validateTriangles


def _booleanMask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean mask matching its image')
    return value


def _cameraMatrices(camera):
    K, R, t = (np.asarray(camera[key], np.float64) for key in ('intrinsics', 'rotation', 'translation'))
    if (camera.get('valid', True) is not True or K.shape != (3, 3) or R.shape != (3, 3)
            or t.shape != (3,) or not all(np.isfinite(value).all() for value in (K, R, t))
            or not np.allclose(K[2], [0., 0., 1.], rtol=0, atol=1e-12)
            or abs(np.linalg.det(K)) <= 1e-12
            or not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(R), 1., rtol=0, atol=1e-8)):
        raise ValueError('Expected a valid finite OpenCV camera with proper rotation')
    return K, R, t


def evaluateNoseViewQuality(geometry, cameras, sourceSupportMasks, noseUvMask, *,
        referenceViewIndex, visibleScene, minimumFacingCosine=.5, fullFacingCosine=.65,
        minimumRelativeFacing=.85, fullRelativeFacing=.95):
    """Condition a front contribution on its actual surface viewing direction.

    The comparison uses all supplied cameras, but only views with positive
    depth, complete-scene visibility and four valid source footprint pixels
    are available. No color is sampled or synthesized here. Quality is zero
    at/below either minimum and increases smoothly to one at both full values.
    A UV texel that is well seen from below need not be suitable for front RGB.
    """
    limits = np.array([minimumFacingCosine, fullFacingCosine, minimumRelativeFacing, fullRelativeFacing])
    if (not np.isfinite(limits).all() or not 0 <= minimumFacingCosine < fullFacingCosine <= 1
            or not 0 <= minimumRelativeFacing < fullRelativeFacing <= 1):
        raise ValueError('Quality thresholds must be finite increasing cosine/ratio pairs in [0,1]')
    domain = np.asarray(noseUvMask)
    if domain.ndim != 2 or domain.dtype != np.bool_:
        raise ValueError('noseUvMask must be a two-dimensional boolean mask')
    if len(cameras) != len(sourceSupportMasks) or not cameras:
        raise ValueError('Every camera needs an original-valid and SH-corrected source support mask')
    indices = [camera.get('viewIndex') for camera in cameras]
    if (any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0 for index in indices)
            or len(set(indices)) != len(indices) or referenceViewIndex not in indices):
        raise ValueError('Expected unique nonnegative camera indices including the reference view')
    vertices, faces, uvs, uvFaces, extent = _geometry(geometry)
    sceneVertices, sceneFaces = validateTriangles(visibleScene['vertices'], visibleScene['faces'], 3)
    faceIds, bary, aliases = _lookup(vertices, faces, uvs, uvFaces, domain.shape, extent)
    yy, xx = np.nonzero(domain & (faceIds >= 0) & ~aliases)
    ids = faceIds[yy, xx]
    triangles = vertices[faces[ids]]
    points = np.sum(triangles*bary[yy, xx, :, None], axis=1)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    cosines = np.zeros((len(cameras), len(points)), np.float64)
    availability = np.zeros(cosines.shape, bool)
    for index, (camera, support) in enumerate(zip(cameras, sourceSupportMasks)):
        support = np.asarray(support)
        if support.ndim != 2 or support.dtype != np.bool_:
            raise ValueError('Each source support mask must be two-dimensional and boolean')
        K, R, t = _cameraMatrices(camera)
        cameraPoints = points @ R.T+t
        toCamera = -t @ R-points
        toCamera /= np.maximum(np.linalg.norm(toCamera, axis=1, keepdims=True), 1e-15)
        cosines[index] = np.sum(normals*toCamera, axis=1)
        projected = cameraPoints @ K.T
        xy = np.divide(projected[:, :2], projected[:, 2:], out=np.full((len(points), 2), -1.), where=projected[:, 2:] > 1e-8)
        bounded = ((cameraPoints[:, 2] > 1e-8) & (xy >= 0).all(1)
                   & (xy < np.array(support.shape[::-1])-1).all(1) & (cosines[index] > 0))
        selected = np.flatnonzero(bounded)
        floors = np.floor(xy[selected]).astype(np.int64)
        footprint = np.stack([floors, floors+[1, 0], floors+[0, 1], floors+[1, 1]], axis=1)
        selected = selected[support[footprint[..., 1], footprint[..., 0]].all(1)]
        visibility = ProjectedSurfaceVisibility(sceneVertices @ R.T+t, sceneFaces, K, support.shape)
        if len(selected):
            closest = visibility.closestDepth(cameraPoints[selected])['depth']
            availability[index, selected] = np.isfinite(closest) & (abs(closest-cameraPoints[selected, 2]) <= extent*1e-6)
    ref = indices.index(referenceViewIndex)
    availableCosines = np.where(availability, cosines, 0.)
    best = availableCosines.max(axis=0)
    relative = np.divide(cosines[ref], best, out=np.zeros_like(best), where=best > 0)
    fraction = np.clip((cosines[ref]-minimumFacingCosine)/(fullFacingCosine-minimumFacingCosine), 0, 1)
    relativeFraction = np.clip((relative-minimumRelativeFacing)/(fullRelativeFacing-minimumRelativeFacing), 0, 1)
    quality = fraction*fraction*(3-2*fraction)*relativeFraction*relativeFraction*(3-2*relativeFraction)
    quality[~availability[ref]] = 0
    qualityMap = np.zeros(domain.shape, np.float64); qualityMap[yy, xx] = quality
    cosinesMap = np.zeros((len(cameras), *domain.shape), np.float64); cosinesMap[:, yy, xx] = cosines
    availableMap = np.zeros(cosinesMap.shape, bool); availableMap[:, yy, xx] = availability
    bestMap = np.zeros(domain.shape, np.float64); bestMap[yy, xx] = best
    relativeMap = np.zeros(domain.shape, np.float64); relativeMap[yy, xx] = relative
    bestIndexMap = np.full(domain.shape, -1, np.int32)
    bestIndexMap[yy[best > 0], xx[best > 0]] = np.array(indices)[np.argmax(availableCosines[:, best > 0], axis=0)]
    return {'sourceQuality': qualityMap, 'facingCosines': cosinesMap, 'viewAvailableMask': availableMap,
        'bestFacingCosine': bestMap, 'relativeFacing': relativeMap, 'bestViewIndex': bestIndexMap,
        'diagnostics': {'method': 'perspectiveSurfaceIncidenceRelativeToAvailableViews', 'cameraViewIndices': indices,
            'referenceViewIndex': int(referenceViewIndex), 'minimumFacingCosine': float(minimumFacingCosine),
            'fullFacingCosine': float(fullFacingCosine), 'minimumRelativeFacing': float(minimumRelativeFacing),
            'fullRelativeFacing': float(fullRelativeFacing), 'nonzeroQualityTexels': int((quality > 0).sum()),
            'referenceAvailableTexels': int(availability[ref].sum()), 'normalPolicy': 'original triangle normal at each actual UV surface point',
            'comparisonSupport': 'positive-depth front-facing complete-scene visible, four original-mask and SH-success pixels',
            'doesNotEstimate': 'photometric confidence, semantic alignment, projected texel footprint, or true albedo'}}


def refineNoseTexture(geometry, camera, sourceLinear, sourceValidMask,
        sourceCorrectionMask, baselineLinear, noseUvMask, protectedUvMask, *,
        visibleScene, featherTexels=6., minimumFacingCosine=.05, sourceQuality=None):
    """Return a local linear-RGB candidate and full sampling provenance.

    Geometry uses original per-corner top-left pixel-edge UVs and canonical
    vertices. Camera is OpenCV world-to-camera in that same canonical frame.
    ``visibleScene`` supplies complete opaque occluders; caller must handle
    alpha-cutout materials rather than incorrectly treating their cards as
    opaque. Source RGB must already be linear and corrected using the declared
    lighting/normal policy. Missing support leaves the baseline exactly intact.

    Feathering reduces contribution at the final accepted support boundary,
    including internal quality, visibility and validity holes. It cannot turn
    failed source support into an observation or expand the supported set.
    There is deliberately no output ``coverageMask`` and no semantic relocation.
    """
    started = perf_counter()
    vertices, faces, uvs, uvFaces, extent = _geometry(geometry)
    source = _rgb(sourceLinear, 'sourceLinear')
    baseline = _rgb(baselineLinear, 'baselineLinear')
    imageShape, shape = source.shape[:2], baseline.shape[:2]
    valid = _booleanMask(sourceValidMask, imageShape, 'sourceValidMask')
    correction = _booleanMask(sourceCorrectionMask, imageShape, 'sourceCorrectionMask')
    domain = _booleanMask(noseUvMask, shape, 'noseUvMask')
    protected = _booleanMask(protectedUvMask, shape, 'protectedUvMask')
    if (not np.isfinite([featherTexels, minimumFacingCosine]).all()
            or featherTexels <= 0 or not 0 <= minimumFacingCosine < 1):
        raise ValueError('Expected positive finite featherTexels and facing cosine in [0,1)')
    quality = np.ones(shape, np.float64) if sourceQuality is None else np.asarray(sourceQuality)
    if (quality.shape != shape or quality.dtype.kind not in 'fiu' or not np.isfinite(quality).all()
            or (quality < 0).any() or (quality > 1).any()):
        raise ValueError('sourceQuality must be finite per-texel weights in [0,1]')
    K, R, t = _cameraMatrices(camera)
    viewIndex = camera.get('viewIndex', 0)
    if isinstance(viewIndex, (bool, np.bool_)) or not isinstance(viewIndex, (int, np.integer)) or viewIndex < 0:
        raise ValueError('Camera viewIndex must be a nonnegative integer')
    if not isinstance(visibleScene, dict):
        raise ValueError('visibleScene must explicitly supply the accepted opaque scene')
    sceneVertices, sceneFaces = validateTriangles(visibleScene['vertices'], visibleScene['faces'], 3)
    faceIds, bary, aliases = _lookup(vertices, faces, uvs, uvFaces, shape, extent)
    writable = domain & ~protected & (faceIds >= 0) & ~aliases
    yy, xx = np.nonzero(writable)
    ids = faceIds[yy, xx]
    cv = vertices @ R.T + t
    points = np.sum(cv[faces[ids]] * bary[yy, xx, :, None], axis=1)
    h = points @ K.T
    xy = np.divide(h[:, :2], h[:, 2:], out=np.full((len(points), 2), -1.), where=h[:, 2:] > 1e-8)
    bounded = ((points[:, 2] > 1e-8) & (xy >= 0).all(1)
               & (xy < np.array(imageShape[::-1])-1).all(1))
    triangles = cv[faces[ids]]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    view = -points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-15)
    facing = np.sum(normals*view, axis=1) > minimumFacingCosine
    visibility = ProjectedSurfaceVisibility(sceneVertices @ R.T+t, sceneFaces, K, imageShape)
    visible = np.zeros(len(points), bool)
    selected = np.flatnonzero(bounded & facing)
    tolerance = extent * 1e-6
    closest = np.full(len(points), np.inf)
    if len(selected):
        closest[selected] = visibility.closestDepth(points[selected])['depth']
        visible[selected] = np.isfinite(closest[selected]) & (abs(closest[selected]-points[selected, 2]) <= tolerance)
    validSupport = np.zeros(len(points), bool)
    correctionSupport = np.zeros(len(points), bool)
    selected = np.flatnonzero(bounded)
    floors = np.floor(xy[selected]).astype(np.int64)
    support = np.stack([floors, floors+[1, 0], floors+[0, 1], floors+[1, 1]], axis=1)
    validSupport[selected] = valid[support[..., 1], support[..., 0]].all(1)
    correctionSupport[selected] = correction[support[..., 1], support[..., 0]].all(1)
    accepted = bounded & facing & visible & validSupport & correctionSupport & (quality[yy, xx] > 0)
    ay, ax = yy[accepted], xx[accepted]
    acceptedMask = np.zeros(shape, bool)
    acceptedMask[ay, ax] = True
    # Feather the final support, not the wider requested domain: rejected
    # internal holes and abrupt incidence cutoffs must fade to the baseline.
    # Padding also gives atlas-border support a finite boundary distance.
    distance = distance_transform_edt(np.pad(acceptedMask, 1))[1:-1, 1:-1]
    fraction = np.clip(distance / featherTexels, 0., 1.)
    feather = fraction * fraction * (3. - 2. * fraction)
    alpha = np.zeros(shape, np.float32)
    alpha[ay, ax] = feather[ay, ax]*quality[ay, ax]
    result = baseline.copy()
    colors = samplePixels(source, xy[accepted])
    result[ay, ax] = (baseline[ay, ax] * (1-alpha[ay, ax, None]) + colors * alpha[ay, ax, None]).astype(baseline.dtype)
    projectedXY = np.full((*shape, 2), np.nan, np.float64)
    projectedXY[yy, xx] = xy
    sourceXY = np.full((*shape, 2), np.nan, np.float64)
    sourceXY[ay, ax] = xy[accepted]
    sourceIndex = np.full(shape, -1, np.int32)
    sourceIndex[ay, ax] = viewIndex
    direct = alpha > 0
    visibleMap = np.zeros(shape, bool); visibleMap[yy, xx] = visible
    photoSupportMap = np.zeros(shape, bool); photoSupportMap[yy, xx] = validSupport
    correctionSupportMap = np.zeros(shape, bool); correctionSupportMap[yy, xx] = correctionSupport
    # Stable reason IDs allow inspecting every destination without rerunning.
    reason = np.zeros(shape, np.uint8)
    reason[domain & protected] = 1
    reason[domain & ~protected & (faceIds < 0)] = 2
    reason[domain & ~protected & aliases] = 3
    localReason = np.full(len(points), 9, np.uint8)
    localReason[quality[yy, xx] <= 0] = 10
    localReason[~correctionSupport] = 8
    localReason[~validSupport] = 7
    localReason[~visible] = 6
    localReason[~facing] = 5
    localReason[~bounded] = 4
    reason[yy, xx] = localReason
    changed = np.any(result != baseline, axis=-1)
    if (changed & ~direct).any() or (direct & protected).any() or not np.array_equal(result[~direct], baseline[~direct]):
        raise RuntimeError('Nose sampling escaped its protected local support')
    diagnostics = {'method': 'fixedSurfacePerspectiveFrontNoseReprojection',
        'referenceViewIndex': int(viewIndex), 'seconds': perf_counter()-started,
        'featherTexels': float(featherTexels), 'minimumFacingCosine': float(minimumFacingCosine),
        'featherDomain': 'finalAcceptedSupportIncludingInternalRejectedHoles',
        'acceptedSupportBoundaryTexels': int((acceptedMask & (distance < featherTexels)).sum()),
        'acceptedFullFeatherTexels': int((acceptedMask & (distance >= featherTexels)).sum()),
        'maximumAcceptedBoundaryDistance': float(distance.max()),
        'totalNewSourceAlpha': float(alpha.sum()),
        'depthToleranceCanonical': float(tolerance), 'requestedDomainTexels': int(domain.sum()),
        'protectedRequestedTexels': int((domain & protected).sum()), 'uvAliasRejectedTexels': int((domain & aliases).sum()),
        'directReprojectionSupportedTexels': int(direct.sum()), 'changedLinearTexels': int(changed.sum()),
        'reasonCounts': {str(value): int((reason == value).sum()) for value in range(11)},
        'reasonLabels': ['outsideDomain', 'protected', 'noUvSurface', 'uvAlias', 'outsidePhoto',
                         'backFacingOrGrazing', 'occluded', 'invalidPhotoFootprint', 'uncorrectedShFootprint', 'supported', 'insufficientSourceQuality'],
        'sourceQualityApplied': sourceQuality is not None,
        'allFourSourcePixelsValidAndShCorrected': True, 'semanticWarpApplied': False,
        'otherViewSamplingApplied': False, 'outsideModifiedValuesExactlyIdentical': True,
        'supportMeaning': 'Accepted direct photographic contribution to this fixed surface; not new anatomical coverage or physical albedo truth',
        'visibility': visibility.diagnostics,
        'limitations': ['Camera and geometry errors can still misplace directly sampled features.',
            'SH success does not remove all specular highlights, cast shadows or self-occlusion.',
            'Retained baseline contributions preserve their previous provenance; alpha is only the new source contribution.']}
    return {'linearRGB': result, 'alpha': alpha, 'sourceXY': sourceXY, 'projectedXY': projectedXY,
        'sourceIndex': sourceIndex, 'directReprojectionSupportMask': direct,
        'visibleMask': visibleMap, 'photoFootprintSupportMask': photoSupportMap,
        'shFootprintSupportMask': correctionSupportMap, 'unsupportedMask': domain & ~protected & ~direct,
        'surfaceFaceIds': faceIds, 'uvAliasMask': aliases, 'reasonCode': reason,
        'acceptedSupportBoundaryDistance': distance,
        'diagnostics': diagnostics}


def encodeNoseTextureEdit(originalPixels, linearRGB, alpha, protectedUvMask):
    """Quantize only accepted RGB texels; preserve original alpha/other pixels."""
    original = np.asarray(originalPixels)
    colors = _rgb(linearRGB, 'linearRGB')
    weights = np.asarray(alpha)
    if (original.dtype != np.uint8 or original.ndim != 3 or original.shape[-1] not in (3, 4)
            or original.shape[:2] != colors.shape[:2] or weights.shape != colors.shape[:2]
            or weights.dtype.kind not in 'fiu' or not np.isfinite(weights).all()
            or (weights < 0).any() or (weights > 1).any()):
        raise ValueError('Expected matching uint8 RGB/RGBA pixels, linear RGB and alpha in [0,1]')
    protected = _booleanMask(protectedUvMask, colors.shape[:2], 'protectedUvMask')
    modified = weights > 0
    if (modified & protected).any():
        raise ValueError('Nose edit overlaps protected UV pixels')
    values = colors[modified]
    srgb = np.where(values <= .0031308, 12.92*values, 1.055*values**(1/2.4)-.055)
    output = original.copy()
    output[modified, :3] = np.rint(np.clip(srgb, 0, 1)*255).astype(np.uint8)
    return output
