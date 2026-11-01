"""Dynamic photo and visible-surface lip curves for conservative client fitting.

All search regions come from the current observations. Curves are appearance
and occlusion proxies, not anatomical annotations or independent validation.
"""
from __future__ import annotations

import numpy as np

from .camera import projectPoints
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import samplePixels


class UnreliableLipObservation(ValueError):
    """A valid input cannot support this optional closed-lip correction."""


def _path(scores, allowed, y, xStep, scale):
    n, m = scores.shape
    cost = np.where(allowed[0], -scores[0], np.inf)
    previous = np.full((n, m), -1, np.int32)
    offsets = np.arange(-max(1, int(round(.75*xStep/(y[1]-y[0])))),
                       max(1, int(round(.75*xStep/(y[1]-y[0]))))+1)
    for column in range(1, n):
        choices = np.full((len(offsets), m), np.inf)
        for k, delta in enumerate(offsets):
            here = np.arange(max(0, -delta), min(m, m-delta))
            choices[k, here] = cost[here+delta]+.08*((y[1]-y[0])*delta/scale)**2
        best = choices.argmin(0)
        previous[column] = np.arange(m)+offsets[best]
        cost = np.where(allowed[column], choices[best, np.arange(m)]-scores[column], np.inf)
    if not np.isfinite(cost).any():
        raise UnreliableLipObservation('noContinuousPhotoLipPath')
    indices = np.empty(n, np.int32)
    indices[-1] = cost.argmin()
    for column in range(n-1, 0, -1):
        indices[column-1] = previous[column, indices[column]]
    return indices


def _scores(image, x, y, allowed, scale):
    intensity = np.asarray(image, np.float32)/255 @ np.array([.2126, .7152, .0722])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    points = np.stack([xx, yy], axis=-1)
    center = samplePixels(intensity, points.reshape(-1, 2)).reshape(xx.shape)
    above = samplePixels(intensity, (points+[0, -2*scale]).reshape(-1, 2)).reshape(xx.shape)
    below = samplePixels(intensity, (points+[0, 2*scale]).reshape(-1, 2)).reshape(xx.shape)
    ridge = (above+below)*.5-center
    lower = np.min(np.where(allowed, center, np.inf), axis=1, keepdims=True)
    upper = np.max(np.where(allowed, center, -np.inf), axis=1, keepdims=True)
    darkness = np.clip((upper-center)/np.maximum(upper-lower, .01), 0, 1)
    contrast = max(float(np.percentile(np.maximum(ridge[allowed], 0), 90)), .005)
    scores = .8*np.clip(ridge/contrast, -2, 2)+.2*darkness
    return np.where(allowed, scores, -np.inf), ridge, contrast


def mouthObservationQuality(landmarks, confidence, imageShape):
    """Assess a complete ordered 68-point current-frame observation."""
    points, scores = np.asarray(landmarks, float), np.asarray(confidence, float)
    reasons = []
    if points.shape != (68, 2) or scores.shape != (68,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
        return {'reliable': False, 'reasons': ['invalidFaceObservation']}
    height, imageWidth = imageShape
    mouth = points[48:68]
    width = float(points[64, 0]-points[60, 0])
    if width < max(16., imageWidth*.035) or width > imageWidth*.5:
        reasons.append('mouthSpanUnreliable')
    if (mouth < 1).any() or (mouth >= [imageWidth-2, height-2]).any():
        reasons.append('mouthOutOfImage')
    if (scores[48:68] < .4).any():
        reasons.append('lowMouthHeatmapPeak')
    arcs = [[60,61,62,63,64], [60,67,66,65,64], list(range(48,55)), [48,59,58,57,56,55,54]]
    if any((np.diff(points[ids, 0]) <= 0).any() for ids in arcs):
        reasons.append('nonmonotonicLipArc')
    eyeLeft, eyeRight = points[36:42].mean(0), points[42:48].mean(0)
    eyeDistance = float(np.linalg.norm(eyeRight-eyeLeft))
    if eyeDistance < 10 or eyeRight[0]-eyeLeft[0] < .9*eyeDistance:
        reasons.append('referenceRollOrEyeOrderUnreliable')
    if abs(points[30,0]-(eyeLeft[0]+eyeRight[0])/2) > .25*max(eyeDistance, 1):
        reasons.append('referenceMayNotBeFrontal')
    opening = np.linalg.norm(points[[67,66,65]]-points[[61,62,63]], axis=1)
    openingRatio = float(np.median(opening)/max(width, 1.))
    if openingRatio > .16:
        reasons.append('openMouthOutsideClosedLipPolicy')
    return {'reliable': not reasons, 'reasons': reasons, 'mouthWidthPixels': width,
            'imageScale': max(width/80., .2), 'openingToWidthRatio': openingRatio,
            'minimumMouthHeatmapPeak': float(scores[48:68].min()),
            'qualification': 'heuristic current-frame checks; not validated across all subjects'}


def tracePhotoLip(originalWorkingRgb, delightSrgb, landmarks, confidence, *, validMask=None):
    raw, delight = np.asarray(originalWorkingRgb), np.asarray(delightSrgb)
    if raw.ndim != 3 or raw.shape[-1] != 3 or raw.dtype != np.uint8 or delight.shape != raw.shape or delight.dtype != np.uint8:
        raise ValueError('Lip tracing expects matching current working RGB uint8 photographs')
    quality = mouthObservationQuality(landmarks, confidence, raw.shape[:2])
    if not quality['reliable']:
        raise UnreliableLipObservation(','.join(quality['reasons']))
    points = np.asarray(landmarks, float)
    upper, lower = points[[60,61,62,63,64]], points[[60,67,66,65,64]]
    scale = quality['imageScale']; step = max(.5, scale); yStep = max(.125, scale*.25)
    x = np.arange(np.ceil((upper[0,0]-2*scale)/step)*step, upper[-1,0]+2*scale+step*.1, step)
    y = np.arange(np.floor((min(upper[:,1].min(), lower[:,1].min())-4*scale)/yStep)*yStep,
                  max(upper[:,1].max(), lower[:,1].max())+4*scale+yStep*.1, yStep)
    if len(x) < 20 or len(y) < 4 or len(x)*len(y) > 250000:
        raise UnreliableLipObservation('photoTraceSearchSizeInvalid')
    upperY = np.interp(x, upper[:,0], upper[:,1]); lowerY = np.interp(x, lower[:,0], lower[:,1])
    outerTop, outerBottom = points[48:55], points[[48,59,58,57,56,55,54]]
    lo = np.maximum(np.minimum(upperY,lowerY)-3*scale, np.interp(x,outerTop[:,0],outerTop[:,1])+scale)
    hi = np.minimum(np.maximum(upperY,lowerY)+2*scale, np.interp(x,outerBottom[:,0],outerBottom[:,1])-scale)
    allowed = (y[None,:]>=lo[:,None]) & (y[None,:]<=hi[:,None])
    if not allowed.any(1).all():
        raise UnreliableLipObservation('closedOrAmbiguousPhotoCorridor')
    rawScores, rawRidge, rawScale = _scores(raw, x, y, allowed, scale)
    delightScores, delightRidge, delightScale = _scores(delight, x, y, allowed, scale)
    rawIndex = _path(rawScores, allowed, y, step, scale)
    delightIndex = _path(delightScores, allowed, y, step, scale)
    combined = (rawScores+delightScores)/2
    selected = _path(combined, allowed, y, step, scale); curve = y[selected]; rows = np.arange(len(x))
    ridge = np.minimum(rawRidge[rows,selected]/rawScale, delightRidge[rows,selected]/delightScale)
    disagreement = abs(y[rawIndex]-y[delightIndex])
    endpoint = np.clip(np.minimum(x-upper[0,0],upper[-1,0]-x)/(6*scale), 0, 1)
    confidence = np.clip(ridge,0,1)*np.exp(-disagreement/max(scale,.5))*endpoint
    trusted = (confidence>=.35) & (endpoint>=.99)
    if validMask is not None:
        valid = np.asarray(validMask)
        if valid.shape != raw.shape[:2] or not np.isfinite(valid).all():
            raise ValueError('Current OpenDelight validity shape mismatch')
        trusted &= samplePixels(valid.astype(float),np.column_stack([x,curve])) >= .999
    if trusted.sum()<max(12,int(.35*len(x))) or np.ptp(x[trusted])<.4*quality['mouthWidthPixels']:
        raise UnreliableLipObservation('insufficientReliablePhotoLipCurve')
    if np.quantile(disagreement[trusted],.95)>5*scale:
        raise UnreliableLipObservation('rawAndDelightLipCurvesDisagree')
    return {'x':x,'y':curve,'xy':np.column_stack([x,curve]),'confidence':confidence,'trustedMask':trusted,
            'rawPhotoY':y[rawIndex],'openDelightY':y[delightIndex],'rawVsDelightDisagreementPixels':disagreement,
            'searchLowerY':lo,'searchUpperY':hi,'imageScale':scale,'quality':quality,
            'meaning':'automatic raw-photo/OpenDelight dark-ridge proxy, not manually labeled lip anatomy'}


def deriveCreaseSearch(photoPoints, projectedBindingPoints, imageShape):
    """Fix a per-job search band before fitting; reuse it for every candidate."""
    photo, projected = np.asarray(photoPoints,float), np.asarray(projectedBindingPoints,float)
    width = float(photo[64,0]-photo[60,0]); scale = width/80.
    inner = np.concatenate([photo[60:68], projected[60:68]])
    height, imageWidth = imageShape
    xRange = [max(0.,float(inner[:,0].min()-2*scale)), min(imageWidth-1.,float(inner[:,0].max()+2*scale))]
    yRange = [max(0.,float(inner[:,1].min()-10*scale)), min(height-1.,float(inner[:,1].max()+10*scale))]
    if xRange[1]-xRange[0] > width*1.75 or yRange[1]-yRange[0] > width*.75:
        raise UnreliableLipObservation('geometryAndPhotoMouthTooFarApart')
    return {'xRange':xRange,'yRange':yRange,'imageShape':tuple(imageShape),'sampleStepPixels':max(.125,.25*scale)}


def traceVisibleCrease(vertices, faces, camera, search):
    """NumPy continuous depth envelope, with original face/barycentric samples."""
    vertices, faces = np.asarray(vertices,float), np.asarray(faces,np.int64)
    K,R,t = (np.asarray(camera[key],float) for key in ('intrinsics','rotation','translation'))
    cv = vertices@R.T+t
    step = float(search['sampleStepPixels'])
    x = np.arange(search['xRange'][0],search['xRange'][1]+step*.1,step)
    y = np.arange(search['yRange'][0],search['yRange'][1]+step*.1,step)
    if len(x)<4 or len(y)<4 or len(x)*len(y)>250000:
        raise UnreliableLipObservation('geometricCreaseSearchTooLarge')
    xx,yy = np.meshgrid(x,y)
    rays = np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])@np.linalg.inv(K).T
    query = ProjectedSurfaceVisibility(cv,faces,K,tuple(search['imageShape'])).closestDepth(rays)
    depth = query['depth'].reshape(xx.shape); faceImage = query['faceIds'].reshape(xx.shape)
    rows = np.argmax(np.where(np.isfinite(depth),depth,-np.inf),axis=0); columns=np.arange(len(x))
    faceIds = faceImage[rows,columns]; valid=(faceIds>=0)&(rows>0)&(rows<len(y)-1)
    pixels = np.column_stack([x,y[rows]])
    projected = projectPoints(vertices,K,R,t)[0]
    triangles = projected[faces[np.maximum(faceIds,0)]]; a,b,c = triangles[:,0],triangles[:,1],triangles[:,2]
    den = (b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    valid &= abs(den)>1e-12; den=np.where(abs(den)>1e-12,den,1.)
    wa=((b[:,1]-c[:,1])*(pixels[:,0]-c[:,0])+(c[:,0]-b[:,0])*(pixels[:,1]-c[:,1]))/den
    wb=((c[:,1]-a[:,1])*(pixels[:,0]-c[:,0])+(a[:,0]-c[:,0])*(pixels[:,1]-c[:,1]))/den
    weights=np.column_stack([wa,wb,1-wa-wb])/cv[faces[np.maximum(faceIds,0)],2]
    weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    peak=depth[rows,columns]; fraction=(y[rows]-y[0])/(y[-1]-y[0])
    baseline=depth[0]*(1-fraction)+depth[-1]*fraction
    prominence=np.where(np.isfinite(peak)&np.isfinite(baseline),peak-baseline,0.)
    extent=float(np.ptp(vertices,axis=0).max())
    valid &= prominence>extent*1e-5
    return {'pixels':pixels,'faceIds':faceIds,'vertexIndices':faces[np.maximum(faceIds,0)],
            'barycentricWeights':weights,'points':(vertices[faces[np.maximum(faceIds,0)]]*weights[...,None]).sum(1),
            'cameraDepth':peak,'validMask':valid,'depthProminence':prominence,
            'meaning':'current-camera visible depth envelope; can switch nonadjacent surfaces'}


def creaseError(trace, photo):
    curve=trace['pixels'][trace['validMask']]
    if len(curve)<12:
        raise UnreliableLipObservation('insufficientVisibleGeometricCrease')
    keep=photo['trustedMask']&(photo['x']>=curve[:,0].min())&(photo['x']<=curve[:,0].max())
    if keep.sum()<12:
        raise UnreliableLipObservation('insufficientCommonCreaseSupport')
    residual=np.interp(photo['x'][keep],curve[:,0],curve[:,1])-photo['y'][keep]
    return {'trustedPointCount':int(keep.sum()),'meanAbsoluteYErrorPixels':float(abs(residual).mean()),
            'medianSignedYErrorPixels':float(np.median(residual)),'maxAbsoluteYErrorPixels':float(abs(residual).max())}
