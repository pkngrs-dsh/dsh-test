"""Bounded frontal eyelid resampling with a sparse UV lookup.

This local experiment uses existing geometry and photo contours; it performs no
model inference and never creates an observed sample from a warped/extended
query. The caller owns accessory UV exclusions and color-space conversion.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

from .eyelidTextureSampling import (
    _rgb, _geometry, _curves, _nearestSegments, _sourceSafety,
    _querySupport, _smooth,
)
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import trianglePixels, samplePixels, validateTriangles


def _localLookup(vertices, faces, uvs, uvFaces, shape, extent, projected, boxes, frontFacing):
    """Raster only the UV envelope of photo-local triangles, checking aliases.

    Every other triangle whose UV bounds touch the envelope participates in the
    lookup, including surfaces outside the photo ROI. Thus another chart cannot
    hide a conflicting coincident UV surface just by missing the photo ROI.
    """
    ph = projected[faces]
    lo, hi = ph.min(1), ph.max(1)
    selected = np.zeros(len(faces), bool)
    for minimum, maximum in boxes:
        selected |= frontFacing & (hi >= minimum).all(1) & (lo <= maximum).all(1)
    height, width = shape
    pixelUvs = uvs * [width, height] - .5
    uvTriangles = pixelUvs[uvFaces]
    uvLo, uvHi = uvTriangles.min(1), uvTriangles.max(1)
    candidate = np.zeros(len(faces), bool)
    # Separate eye envelopes keep the nose/forehead charts out of the raster.
    for minimum, maximum in boxes:
        local = selected & (hi >= minimum).all(1) & (lo <= maximum).all(1)
        if local.any():
            lower, upper = uvLo[local].min(0)-2., uvHi[local].max(0)+2.
            candidate |= (uvHi >= lower).all(1) & (uvLo <= upper).all(1)
    faceIds = np.full(shape, -1, np.int32)
    bary = np.zeros((*shape, 3), np.float32)
    alias = np.zeros(shape, bool)
    tolerance = extent * 1e-7
    for faceId in np.flatnonzero(candidate):
        raster = trianglePixels(uvTriangles[faceId], height, width)
        if raster is None:
            continue
        yy, xx, weights = raster
        prior = faceIds[yy, xx]
        overlap = prior >= 0
        if overlap.any():
            old = np.sum(vertices[faces[prior[overlap]]] * bary[yy[overlap], xx[overlap], :, None], axis=1)
            new = weights[overlap] @ vertices[faces[faceId]]
            conflict = np.linalg.norm(new-old, axis=1) > tolerance
            alias[yy[overlap][conflict], xx[overlap][conflict]] = True
        fresh = ~overlap
        faceIds[yy[fresh], xx[fresh]] = faceId
        bary[yy[fresh], xx[fresh]] = weights[fresh]
    return faceIds, bary, alias, int(candidate.sum())


def reprojectFastFrontEyeTexture(geometry, camera, sourceLinear, sourceValidMask,
        baselineLinear, photoEyes, modelEyes, *, writableMask=None,
        visibleScene=None, upperBandFraction=.42, lowerBandFraction=.28,
        coreBandFraction=.10, maximumWarpFraction=.10,
        sourceSafetyFraction=.0025, maximumExtensionFraction=.035,
        filterFootprint=True, directProjectionFallback=False,
        directProjectionDomainMask=None):
    """Sample frontal lid skin in working-image pixel coordinates.

    Both eyes have ``upper``/``lower`` left-to-right Nx2 curves with matching
    semantic indices. Band sizes and all geometric limits are fractions of each
    photo eye's canthus width. A sample whose semantic correspondence
    exceeds ``maximumWarpFraction`` is rejected instead of clamped, with soft
    confidence tapering before that limit. Full scene
    geometry supplies opaque occlusion; MASK materials must be handled by caller.

    This is estimated appearance where ``warpMask`` or ``extensionMask`` is true.
    ``filterPadding`` is one-texel UV filter continuation, also estimated. Empty
    support leaves baseline byte-equivalent (as floating RGB). Diagnostic timing
    includes validation, sparse rasterization and exact local visibility.

    ``directProjectionFallback`` optionally retains visible skin with excessive
    contour residual when its *unmoved* photographic projection has safe source
    support. It cannot extend a query across the photographic eye opening.
    ``directProjectionDomainMask`` can restrict that fallback to atlas texels
    lacking useful bake coverage. Fallback is observed photographic color,
    blended with baseline using the same spatial band as ordinary sampling;
    ``alpha`` is its new observation contribution, not final observation purity.
    """
    start = perf_counter()
    vertices, faces, uvs, uvFaces, extent = _geometry(geometry)
    source, baseline = _rgb(sourceLinear, 'sourceLinear'), _rgb(baselineLinear, 'baselineLinear')
    imageShape, shape = source.shape[:2], baseline.shape[:2]
    valid = np.asarray(sourceValidMask)
    if valid.shape != imageShape or not np.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any():
        raise ValueError('sourceValidMask must match source and lie in [0,1]')
    writable = np.ones(shape, bool) if writableMask is None else np.asarray(writableMask)
    if writable.shape != shape or writable.dtype != np.bool_:
        raise ValueError('writableMask must be a boolean atlas mask')
    limits = [upperBandFraction, lowerBandFraction, coreBandFraction, maximumWarpFraction,
              sourceSafetyFraction, maximumExtensionFraction]
    if not np.isfinite(limits).all() or min(limits) < 0 or min(upperBandFraction, lowerBandFraction) <= coreBandFraction or maximumWarpFraction <= 0:
        raise ValueError('Invalid width-normalized eye sampling limits')
    if not isinstance(filterFootprint, (bool, np.bool_)):
        raise ValueError('filterFootprint must be boolean')
    if not isinstance(directProjectionFallback, (bool, np.bool_)):
        raise ValueError('directProjectionFallback must be boolean')
    directDomain = np.ones(shape, bool) if directProjectionDomainMask is None else np.asarray(directProjectionDomainMask)
    if directDomain.shape != shape or directDomain.dtype != np.bool_:
        raise ValueError('directProjectionDomainMask must be a boolean atlas mask')
    K, R, t = (np.asarray(camera[k], np.float64) for k in ('intrinsics','rotation','translation'))
    if (K.shape != (3,3) or R.shape != (3,3) or t.shape != (3,) or
            not all(np.isfinite(x).all() for x in (K,R,t)) or
            not np.allclose(R@R.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5)):
        raise ValueError('Camera must be finite OpenCV world-to-camera')
    photoEyes = _curves(photoEyes,imageShape,'photoEyes')
    modelEyes = _curves(modelEyes,imageShape,'modelEyes')
    starts, ends, photoStarts, photoEnds = [], [], [], []
    band, core, warpLimit, extensionLimit, segmentGood, eyeIds = [], [], [], [], [], []
    boxes, eyeWidths, contourReport = [], {}, {}
    for ei, eye in enumerate(('imageLeft','imageRight')):
        width = float(np.linalg.norm(photoEyes[eye]['upper'][-1]-photoEyes[eye]['upper'][0]))
        if width < 2:
            raise ValueError('Photographic eye width is too small')
        eyeWidths[eye] = width
        cloud = np.concatenate(list(modelEyes[eye].values()))
        maximumBand = width * max(upperBandFraction, lowerBandFraction)
        boxes.append((cloud.min(0)-maximumBand, cloud.max(0)+maximumBand))
        contourReport[eye] = {}
        for part, outer in (('upper',upperBandFraction),('lower',lowerBandFraction)):
            p,m = photoEyes[eye][part],modelEyes[eye][part]
            if p.shape != m.shape:
                raise ValueError('Model/photo curve semantic indices must correspond')
            error = np.linalg.norm(p-m,axis=1)
            good = np.maximum(error[:-1],error[1:]) <= maximumWarpFraction*width
            count = len(m)-1
            starts.extend(m[:-1]); ends.extend(m[1:])
            photoStarts.extend(p[:-1]); photoEnds.extend(p[1:])
            band.extend([outer*width]*count); core.extend([coreBandFraction*width]*count)
            warpLimit.extend([maximumWarpFraction*width]*count)
            extensionLimit.extend([maximumExtensionFraction*width]*count)
            segmentGood.extend(good); eyeIds.extend([ei]*count)
            contourReport[eye][part] = {'maximumResidualPixels':float(error.max()),
                'maximumResidualEyeWidths':float(error.max()/width),
                'segmentsWithEndpointOverLimit':int((~good).sum()),'segments':count}
    starts,ends,photoStarts,photoEnds = map(np.asarray,(starts,ends,photoStarts,photoEnds))
    band,core,warpLimit,extensionLimit = map(np.asarray,(band,core,warpLimit,extensionLimit))
    segmentGood,eyeIds = np.asarray(segmentGood),np.asarray(eyeIds)
    cv = vertices@R.T+t
    homogeneous = cv@K.T
    projected = np.divide(homogeneous[:,:2], homogeneous[:,2:],out=np.full((len(cv),2),-1e10),where=homogeneous[:,2:]>1e-8)
    ct = cv[faces]
    faceFront = (np.sum(np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]) * -ct.mean(1),axis=1)>0) & (ct[:,:,2]>1e-8).all(1)
    faceIds,bary,alias,rasterFaceCount = _localLookup(vertices,faces,uvs,uvFaces,shape,extent,projected,boxes,faceFront)
    yy,xx = np.nonzero((faceIds>=0)&writable&~alias)
    ids = faceIds[yy,xx]
    points = np.sum(cv[faces[ids]]*bary[yy,xx,:,None],axis=1)
    h = points@K.T
    xy = np.divide(h[:,:2],h[:,2:],out=np.full((len(points),2),-1.),where=h[:,2:]>1e-8)
    possible = (points[:,2]>1e-8)&(xy>=0).all(1)&(xy<np.array(imageShape[::-1])-1).all(1)
    inBox = np.zeros(len(points),bool)
    for minimum,maximum in boxes:
        inBox |= (xy>=minimum).all(1)&(xy<=maximum).all(1)
    local = np.flatnonzero(possible&inBox)
    distance,segment,fraction = _nearestSegments(xy[local],starts,ends)
    correspondingModel = starts[segment]+fraction[:,None]*(ends[segment]-starts[segment])
    correspondingPhoto = photoStarts[segment]+fraction[:,None]*(photoEnds[segment]-photoStarts[segment])
    residual = np.linalg.norm(correspondingPhoto-correspondingModel,axis=1)
    qualifiedResidual = residual < warpLimit[segment]
    eligibleBand = (distance<band[segment]) & (qualifiedResidual | directProjectionFallback)
    rejectedWarp = int(((distance<band[segment]) & ~qualifiedResidual).sum())
    local,distance,segment,fraction = [a[eligibleBand] for a in (local,distance,segment,fraction)]
    if visibleScene is None:
        sv,sf = vertices,faces
    else:
        sv,sf = validateTriangles(visibleScene['vertices'],visibleScene['faces'],3)
    visibility = ProjectedSurfaceVisibility(sv@R.T+t,sf,K,imageShape)
    triangles = cv[faces[ids[local]]]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    facing = np.sum(normals*-points[local],axis=1)>0
    depth = visibility.closestDepth(points[local])['depth']
    visible = np.isfinite(depth)&(abs(depth-points[local,2])<=extent*1e-6)
    take = facing&visible
    local,distance,segment,fraction = [a[take] for a in (local,distance,segment,fraction)]
    amount = 1-_smooth((distance-core[segment])/(band[segment]-core[segment]))
    spatialAmount = amount.copy()
    cm = starts[segment]+fraction[:,None]*(ends[segment]-starts[segment])
    cp = photoStarts[segment]+fraction[:,None]*(photoEnds[segment]-photoStarts[segment])
    confidence = _smooth((warpLimit[segment]-np.linalg.norm(cp-cm,axis=1))/(warpLimit[segment]*.35))
    amount *= confidence
    displacement = (cp-cm)*amount[:,None]
    actual,query = xy[local],xy[local]+displacement
    warped = np.linalg.norm(displacement,axis=1)>1e-6
    # Correspondence quality controls only the semantic warp. A rejected
    # correspondence may still have a valid unwarped photographic observation.
    # Keep this path separate so source safety can never turn into an extension
    # across the opening or conceal a geometric mismatch.
    directCandidate = (np.linalg.norm(cp-cm,axis=1) >= warpLimit[segment]) & directProjectionFallback
    domainAllowed = directDomain[yy[local],xx[local]]
    directFallback = directCandidate & domainAllowed
    amount[directFallback] = spatialAmount[directFallback]
    query[directFallback] = actual[directFallback]
    displacement[directFallback] = 0.
    warped[directFallback] = False
    # The largest eye scale is conservative for both eyes; no sample footprint
    # may touch either photo opening. Margin remains small enough to retain the
    # dark external eyelash line when it is genuinely outside the opening.
    safety = max(.10,sourceSafetyFraction*max(eyeWidths.values()))
    safePixels,safeCells = _sourceSafety(valid,photoEyes,safety)
    supported = _querySupport(query,safeCells)
    unsafeDirectCount = int((directFallback & ~supported).sum())
    extended = np.zeros(len(query),bool)
    extensionDistance = np.zeros(len(query))
    extendable = ~supported & ~directCandidate
    if maximumExtensionFraction>0 and safeCells.any() and extendable.any():
        _,nearest = distance_transform_edt(~safeCells,return_indices=True)
        rejected = np.flatnonzero(extendable)
        cell = np.clip(np.floor(query[rejected]).astype(int),[0,0],np.array(safeCells.shape[::-1])-1)
        donor = np.column_stack([nearest[1,cell[:,1],cell[:,0]],nearest[0,cell[:,1],cell[:,0]]])+.5
        delta = np.linalg.norm(donor-query[rejected],axis=1)
        ok = (delta<=extensionLimit[segment[rejected]])&_querySupport(donor,safeCells)
        recovered = rejected[ok]
        query[recovered] = donor[ok]
        supported[recovered] = extended[recovered] = True
        extensionDistance[recovered] = delta[ok]
    accepted = np.flatnonzero(supported&(amount>1e-7))
    chosen = local[accepted]
    index = yy[chosen],xx[chosen]
    alpha = np.zeros(shape,np.float32); alpha[index]=amount[accepted]
    linear = baseline.copy()
    colors = samplePixels(source,query[accepted])
    linear[index] = (baseline[index]*(1-amount[accepted,None])+colors*amount[accepted,None]).astype(baseline.dtype)
    projectedXY,sourceXY = (np.full((*shape,2),np.nan,np.float32) for _ in range(2))
    projectedXY[yy[local],xx[local]] = actual
    sourceXY[index] = query[accepted]
    warpMask,extensionMask = (np.zeros(shape,bool) for _ in range(2))
    warpMask[index],extensionMask[index] = warped[accepted],extended[accepted]
    directFallbackMask = np.zeros(shape,bool)
    directFallbackMask[index] = directFallback[accepted]
    projection = np.zeros(shape,bool); projection[yy[local],xx[local]]=True
    padding = np.zeros(shape,bool)
    filterAlpha = np.zeros(shape,np.float32)
    # One-texel UV continuation for the GPU's bilinear filter. Only unowned
    # chart-exterior texels qualify; existing skin and unrelated chart owners
    # are never overwritten from a neighbor. These are estimates, not rays.
    if filterFootprint and (alpha>0).any():
        paddedCandidates = binary_dilation(alpha>0)&(faceIds<0)&writable&~alias
        if paddedCandidates.any():
            _,nearest = distance_transform_edt(alpha==0,return_indices=True)
            py,px = np.nonzero(paddedCandidates)
            ny,nx = nearest[:,py,px]
            padding[py,px]=True
            linear[py,px]=linear[ny,nx]
            filterAlpha[py,px]=alpha[ny,nx]
            sourceXY[py,px]=sourceXY[ny,nx]
    estimated = warpMask|extensionMask|padding
    observed = (alpha>0)&~estimated
    modified = (alpha>0)|padding
    changed = np.any(linear!=baseline,axis=2)
    if (changed&~writable).any() or (changed&alias).any() or (changed&~modified).any():
        raise RuntimeError('Fast eyelid resampling escaped qualified UV support')
    diagnostics = {
        'method':'sparseUvWidthNormalizedBoundedFrontEyeSampling',
        'seconds':perf_counter()-start,'eyeWidthsWorkingPixels':eyeWidths,
        'upperBandEyeWidths':upperBandFraction,'lowerBandEyeWidths':lowerBandFraction,
        'coreBandEyeWidths':coreBandFraction,'maximumWarpEyeWidths':maximumWarpFraction,
        'maximumExtensionEyeWidths':maximumExtensionFraction,'sourceSafetyWorkingPixels':safety,
        'contourResiduals':contourReport,'geometryResidualRejectedTexels':rejectedWarp,
        'directProjectionFallbackEnabled':bool(directProjectionFallback),
        'directProjectionDomainProvided':directProjectionDomainMask is not None,
        'directProjectionFallbackCandidateTexels':int(directCandidate.sum()),
        'directProjectionFallbackDomainExcludedTexels':int((directCandidate & ~domainAllowed).sum()),
        'directProjectionFallbackAcceptedTexels':int(directFallbackMask.sum()),
        'directProjectionFallbackUnsafeSourceRejectedTexels':unsafeDirectCount,
        'directProjectionFallbackObservationContribution':float(alpha[directFallbackMask].sum()),
        'alphaMeaning':'New source contribution; retained baseline may include template or older observations',
        'inputFaces':len(faces),'uvRasterizedFaces':rasterFaceCount,
        'uvAliasExcludedTexels':int(alias.sum()),'visibleBandTexels':len(local),
        'backFacingLocalTexels':int((~facing).sum()),'occludedLocalTexels':int((~visible).sum()),
        'unsafeSourceRejectedTexels':int((~supported).sum()),'changedTexels':int(changed.sum()),
        'directObservationTexels':int(observed.sum()),'estimatedTexels':int(estimated.sum()),
        'curveWarpTexels':int(warpMask.sum()),'exteriorExtendedTexels':int(extensionMask.sum()),
        'filterPaddingTexels':int(padding.sum()),'completeSceneProvided':visibleScene is not None,
        'maximumAppliedWarpWorkingPixels':float(np.linalg.norm(displacement[accepted],axis=1).max(initial=0)),
        'maximumAppliedExtensionWorkingPixels':float(extensionDistance[accepted].max(initial=0)),
        'allSourceBilinearSupportOutsidePhotoEyes':True,'noInternalColorGain':True,
        'unmodifiedOutsideEditExactly':bool(np.array_equal(linear[~modified],baseline[~modified])),
        'visibility':visibility.diagnostics,
        'limitations':['Automatic photo contours are not manual ground truth',
            'Local eye appearance only; no replacement for full geometry fit or all-view bake',
            'Caller must exclude shared/accessory UVs and apply alpha material semantics',
            'Warp, bounded exterior continuation and filter padding are estimated appearance']}
    return {'linearRGB':linear,'alpha':alpha,'observedMask':observed,'estimatedMask':estimated,
        'warpMask':warpMask,'extensionMask':extensionMask,'projectedXY':projectedXY,'sourceXY':sourceXY,
        'directProjectionFallbackMask':directFallbackMask,
        'projectionMask':projection,'aliasMask':alias,'surfaceFaceIds':faceIds,
        'sourceSafePixelMask':safePixels,'sourceSafeCellMask':safeCells,
        'filterPadding':padding,'filterAlpha':filterAlpha,'diagnostics':diagnostics}


# Useful when switching the local experiment runner between implementations.
reprojectEyelidTexture = reprojectFastFrontEyeTexture
