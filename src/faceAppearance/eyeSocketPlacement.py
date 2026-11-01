"""Place original eye caps from frozen visible socket bindings.

Client coordinates are centimeters, Y up and Z forward. This adapter does not
infer gaze: it preserves the original iris direction, UVs and vertex order.
Only the center and a bounded uniform scale change. Template-visible skin/eye
interfaces provide surface anchors, rather than all-sphere nearest neighbors.
"""
from __future__ import annotations

import numpy as np

from .eyeSocketVisibility import buildVisibleEyeRimBinding


def _sphere(vertices):
    points=np.asarray(vertices,np.float64)
    origin=points.mean(0)
    centered=points-origin
    solution=np.linalg.lstsq(np.c_[2*centered,np.ones(len(points))],np.sum(centered**2,axis=1),rcond=None)[0]
    center=origin+solution[:3]
    distances=np.linalg.norm(points-center,axis=1)
    radius=float(np.median(distances))
    if radius<=0 or np.max(abs(distances-radius))>.05*radius:
        raise ValueError('Eye socket placement requires an approximately spherical original eye cap')
    return center,radius


def buildEyeSocketBindings(arrays,components):
    """Prepare two immutable eye bindings once from a verified client bundle."""
    result={}
    for component in components:
        label=component.get('label')
        if label not in ('leftEye','rightEye'):continue
        if label in result or component.get('materialId')!=2:
            raise ValueError('Expected one separately indexed component for each original eye')
        ids=np.asarray(component['vertexIndices'],np.int64)
        vertices=np.asarray(arrays['vertices'][ids],np.float64).copy()
        center,radius=_sphere(vertices)
        rim=buildVisibleEyeRimBinding(arrays,component)
        rest=np.asarray(rim['restPoints'],np.float64)
        if len(rest)<24 or rest.shape[1:]!=(3,) or not np.isfinite(rest).all():
            raise ValueError('Too few visible skin-eye boundary anchors for socket placement')
        design=np.c_[rest[:,:2]-rest[:,:2].mean(0),np.ones(len(rest))]
        if np.linalg.matrix_rank(design)!=3:
            raise ValueError('Visible eye boundary must span both screen axes')
        result[label]={'componentVertexIds':ids.copy(),'templateEyeVertices':vertices,
            'sphereCenter':center,'sphereRadius':radius,'restPoints':rest,
            'vertexIndices':np.asarray(rim['vertexIndices'],np.int64).copy(),
            'barycentricWeights':np.asarray(rim['barycentricWeights'],np.float64).copy(),
            'skinVertexCount':len(arrays['nicpToFullIds']),
            'bindingDiagnostics':rim['diagnostics'],
            'source':'templateVisibleSkinEyeInterface','version':'visibleSocketTranslationV1'}
        for value in result[label].values():
            if isinstance(value,np.ndarray):value.setflags(write=False)
    if result and set(result)!= {'leftEye','rightEye'}:
        raise ValueError('Eye socket placement requires both original eye components')
    return result


def _summary(values):
    return {'min':float(np.min(values)),'median':float(np.median(values)),
            'max':float(np.max(values)),'rms':float(np.sqrt(np.mean(np.asarray(values)**2)))}


def placeEyeInSocket(binding,deformedSkin):
    """Return original eye-cap vertices after translation and ≤5% shrinking.

    The frontal center follows the affine displacement of the frozen visible
    rim. Depth preserves each anchor's template radial offset, using a median
    to avoid a corner or isolated folded triangle driving the sphere forward.
    Scale may shrink from 1 to .95 with socket width; it never enlarges an iris.
    All displacement bounds are relative to this subject's deformed eye rim.
    """
    skin=np.asarray(deformedSkin,np.float64)
    if skin.shape!=(binding['skinVertexCount'],3) or not np.isfinite(skin).all():
        raise ValueError('Socket placement requires finite ordered client skin vertices')
    rest=binding['restPoints'];center=binding['sphereCenter'];radius=binding['sphereRadius']
    target=(skin[binding['vertexIndices']]*binding['barycentricWeights'][...,None]).sum(1)
    restMid=rest.mean(0);targetMid=target.mean(0)
    restSpan=np.ptp(rest[:,:2],axis=0);targetSpan=np.ptp(target[:,:2],axis=0)
    if np.any(restSpan<=0) or np.any(targetSpan<restSpan*.15) or np.any(targetSpan>restSpan*3.):
        raise ValueError('Deformed eye socket collapsed or exceeded supported proportions')
    requestedScale=float(targetSpan[0]/restSpan[0])
    scale=float(np.clip(requestedScale,.95,1.))
    design=np.c_[rest[:,:2]-restMid[:2],np.ones(len(rest))]
    affine=np.linalg.lstsq(design,target[:,:2],rcond=None)[0]
    requestedCenterXY=np.r_[center[:2]-restMid[:2],1.]@affine
    proportional=targetMid[:2]+(center[:2]-restMid[:2])*np.clip(targetSpan/restSpan,.5,1.5)
    # A tightly bounded correction prevents an ill-conditioned narrow opening
    # from extrapolating the iris center far from the observed socket.
    centerXY=np.clip(requestedCenterXY,proportional-.2*radius,proportional+.2*radius)
    baselineCenter=targetMid+(center-restMid)*scale
    scaledDistances=np.linalg.norm(rest-center,axis=1)*scale
    squared=scaledDistances**2-np.sum((target[:,:2]-centerXY)**2,axis=1)
    qualified=(squared>0)&((rest[:,2]-center[2])>.1*radius)
    if np.count_nonzero(qualified)<max(16,int(.6*len(rest))):
        raise ValueError('Eye socket anchors do not support a stable front-facing sphere depth')
    inferredZ=target[qualified,2]-np.sqrt(squared[qualified])
    requestedZ=float(np.median(inferredZ))
    centerZ=float(np.clip(requestedZ,baselineCenter[2]-.35*radius,baselineCenter[2]+.35*radius))
    placedCenter=np.r_[centerXY,centerZ]
    translation=placedCenter-scale*center
    vertices=binding['templateEyeVertices']*scale+translation
    matrix=np.eye(4);matrix[:3,:3]*=scale;matrix[3,:3]=translation
    originalGap=np.linalg.norm(rest-center,axis=1)-radius
    placedGap=np.linalg.norm(target-placedCenter,axis=1)-radius*scale
    residual=placedGap-originalGap*scale
    diagnostics={'method':binding['version'],'bindingSource':binding['source'],
        'binding':binding['bindingDiagnostics'],'anchorCount':len(rest),
        'templateSphereCenterCm':center,'placedSphereCenterCm':placedCenter,
        'templateSphereRadiusCm':radius,'placedSphereRadiusCm':radius*scale,
        'uniformScale':scale,'requestedWidthScale':requestedScale,'scaleRange':[.95,1.],
        'irisDirection':'originalTemplateUnchanged','rotationDegrees':0.,
        'sphereCenterTranslationCm':placedCenter-center,'frontalAffine':affine,
        'requestedCenterXYCm':requestedCenterXY,'requestedDepthCm':requestedZ,
        'depthQualifiedAnchorCount':int(qualified.sum()),
        'centerBoundsApplied':bool(not np.allclose(centerXY,requestedCenterXY,rtol=0,atol=1e-10) or abs(centerZ-requestedZ)>1e-10),
        'templateRadialGapCm':_summary(originalGap),'placedRadialGapCm':_summary(placedGap),
        'radialGapChangeCm':_summary(residual),
        'limitations':['Frontal template-visible bindings follow the skin but are not patient-specific anatomy.',
            'No gaze inference, nonrigid eyeball deformation, pupil texture change or collision-free certification.']}
    return {'vertices':vertices,'transformRow4x4':matrix,'translationCm':translation,
            'uniformScale':scale,'diagnostics':diagnostics}
