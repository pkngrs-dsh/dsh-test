"""Direct client registration and current-photo geometric refinement.

No experiment cache, sample identity or tools module is used. Missing or
unreliable observations skip optional refinements with explicit diagnostics;
direct-registration, template-hash and topology failures propagate to callers.
Existing mouth intersections are measured, not represented as repaired.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from .adapters.eyelidLandmarks import EyelidLandmarksAdapter
from .camera import projectPoints
from .clientGeometrySafety import intersectionPairs
from .clientLandmarks import loadClientLandmarks, ibugClientLandmarks
from .clientLipTrace import (UnreliableLipObservation, mouthObservationQuality,
    tracePhotoLip, deriveCreaseSearch, traceVisibleCrease, creaseError)
from .frontLandmarkRefinement import refineFrontLandmarks, checkFrontGeometry
from .geometryRefinement import _validateCameras
from .headAssembly import decodeColor
from .imageColor import readSrgbImage
from .semanticEyeSupport import (buildSemanticEyeSupport, normalizeSemanticEyeSupportPolicy,
    semanticEyeFitMetrics)
from .surfaceVisibility import ProjectedSurfaceVisibility


ALGORITHM_VERSION = 'direct-staged-landmarks-affine-lip-v1'


def _json(value):
    if isinstance(value, np.ndarray): return _json(value.tolist())
    if isinstance(value, np.generic): return _json(value.item())
    if isinstance(value, dict): return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


@lru_cache(maxsize=2)
def _adapter(projectRoot):
    return EyelidLandmarksAdapter(Path(projectRoot)).loadModels()


def _infer(detector, image, anchors):
    # Invalid current-frame crop/landmark geometry is an observation failure;
    # the input-image identity checks happen separately and still propagate.
    try:
        return detector.infer(image,faceAnchors=anchors)
    except ValueError as exc:
        raise UnreliableLipObservation('landmarkInferenceRejected: '+str(exc)) from exc


def _project(vertices, camera):
    return projectPoints(vertices,camera['intrinsics'],camera['rotation'],camera['translation'])[0]


def _sample(vertices, binding):
    return (vertices[binding['vertexIndices']]*binding['barycentricWeights'][...,None]).sum(1)


def _visible(vertices, faces, positions, camera, imageShape):
    cv=vertices@camera['rotation'].T+camera['translation']
    query=positions@camera['rotation'].T+camera['translation']
    checked=ProjectedSurfaceVisibility(cv,faces,camera['intrinsics'],imageShape).closestDepth(query)
    tolerance=float(np.ptp(vertices,axis=0).max())*.002
    return checked['insidePhoto'] & np.isfinite(checked['depth']) & (abs(checked['depth']-query[:,2])<=tolerance)


def _observations(projectRoot, photoPaths, workingPhotoPaths, inputs, cameras):
    """Infer ordered STFR face points on current original images when present."""
    results=[]; records=[]
    try:
        detector=_adapter(str(Path(projectRoot).resolve()))
    except (FileNotFoundError, OSError, ValueError, cv2.error) as exc:
        return [None]*len(inputs), [{'viewIndex':i,'status':'unavailable',
            'reason':'landmarkBackendUnavailable','detail':str(exc)} for i in range(len(inputs))]
    for index,item in enumerate(inputs):
        record={'viewIndex':index,'status':'unavailable'}
        try:
            anchors=np.asarray(item.get('faceDetection',{}).get('landmarks',[]),float)
            if anchors.shape not in ((4,2),(5,2)) or not np.isfinite(anchors).all():
                raise UnreliableLipObservation('missingCurrentFaceAnchors')
            matrix=np.asarray(item['inputToWorking'],float); size=tuple(int(v) for v in item['workingSize'])
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or min(size)<4:
                raise ValueError('Invalid current input coordinate transform')
            originalPath=Path(photoPaths[index]) if photoPaths[index] is not None else None
            if originalPath is not None and originalPath.is_file():
                sourceBytes=originalPath.read_bytes(); sourceHash=hashlib.sha256(sourceBytes).hexdigest()
                if item.get('sha256') and sourceHash!=item['sha256']:
                    raise ValueError('Original photograph hash differs from reconstruction input')
                original, colorManagement=readSrgbImage(io.BytesIO(sourceBytes))
                image=np.asarray(original)
                record['colorManagement']=colorManagement
                if item.get('orientedSize') and list(image.shape[1::-1])!=list(item['orientedSize']):
                    raise ValueError('Original photograph dimensions differ from reconstruction input')
                inferred=_infer(detector,image,anchors)
                hom=np.column_stack([inferred['faceLandmarks'],np.ones(68)])@matrix.T
                points=hom[:,:2]/hom[:,2:]
                working=cv2.warpPerspective(image,matrix,size)
                record.update(source='originalPhotograph',sourceSha256=sourceHash)
            elif workingPhotoPaths is not None and workingPhotoPaths[index] is not None:
                workingImage, colorManagement=readSrgbImage(workingPhotoPaths[index])
                working=np.asarray(workingImage)
                record['colorManagement']=colorManagement
                if tuple(working.shape[1::-1])!=size:
                    raise ValueError('Working photograph dimensions differ from reconstruction input')
                hom=np.column_stack([anchors,np.ones(len(anchors))])@matrix.T
                inferred=_infer(detector,working,hom[:,:2]/hom[:,2:])
                points=np.asarray(inferred['faceLandmarks'],float)
                record.update(source='providedWorkingPhotograph',
                    sourceSha256=hashlib.sha256(Path(workingPhotoPaths[index]).read_bytes()).hexdigest(),
                    limitation='Detection on a working crop can differ from original-image inference')
            else:
                raise UnreliableLipObservation('currentPhotographUnavailable')
            scores=np.asarray(inferred['faceRawPeakScores'],float)
            if points.shape!=(68,2) or scores.shape!=(68,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise UnreliableLipObservation('nonfiniteLandmarkObservation')
            if cameras[index] is None:
                raise UnreliableLipObservation('cameraRejected')
            record.update(status='detected',model=inferred['metadata'],
                          mouthQuality=mouthObservationQuality(points,scores,working.shape[:2]))
            results.append({'pixels':points,'confidence':scores,'workingRgb':working})
        except (UnreliableLipObservation, FileNotFoundError, OSError, cv2.error) as exc:
            record.update(reason=type(exc).__name__,detail=str(exc));results.append(None)
        records.append(record)
    return results,records


def _errors(vertices,binding,cameras,observations,masks):
    points=_sample(vertices,binding); result=[]
    for view,(camera,observation,mask) in enumerate(zip(cameras,observations,masks)):
        groups={}
        distances=(np.linalg.norm(_project(points,camera)-observation['pixels'][binding['ibugIds']],axis=1)
                   if camera is not None and observation is not None else np.full(len(mask),np.nan))
        for name,region in [('eyes',binding['ibugIds']<48),('mouth',binding['ibugIds']>=48)]:
            keep=mask&region
            groups[name]={'count':int(keep.sum()),'meanErrorPixels':float(distances[keep].mean()) if keep.any() else None,
                          'maxErrorPixels':float(distances[keep].max()) if keep.any() else None}
        result.append({'viewIndex':view,'groups':groups})
    return result


def _sideDelta(candidate,previous,referenceView,group,minimum=4):
    changes=[]
    for view in range(len(candidate)):
        first,second=candidate[view]['groups'][group],previous[view]['groups'][group]
        if view!=referenceView and min(first['count'],second['count'])>=minimum:
            changes.append({'viewIndex':view,'changePixels':first['meanErrorPixels']-second['meanErrorPixels']})
    return changes


def _constraint(name,selected,binding,observation):
    ids=binding['ibugIds'][selected];pairs=[]
    if name=='mouth':
        for a,b in [(i,i+1) for i in range(48,67) if i!=59]+[(59,48),(67,60)]:
            if a in ids and b in ids:
                pairs.append([int(np.flatnonzero(ids==a)[0]),int(np.flatnonzero(ids==b)[0])])
    return {'name':name,'vertexIndices':binding['vertexIndices'][selected],
        'barycentric':binding['barycentricWeights'][selected],'targetPixels':observation['pixels'][ids],
        'weights':np.where(np.isin(ids,[36,39,42,45,48,54]),4.,1.),
        'confidence':np.clip(observation['confidence'][ids],0,1),
        **({'segmentPairs':np.asarray(pairs,np.int64).reshape(-1,2)} if pairs else {})}


def _stagedLandmarks(before,faces,fixed,binding,cameras,observations,masks,referenceView,frontQuality,*,
                     semanticEyeRefinement=True,semanticEyeSupportPolicy=None,captureSemanticEyeEvidence=False):
    if not isinstance(captureSemanticEyeEvidence,bool):raise ValueError('captureSemanticEyeEvidence must be boolean')
    eyePolicy=(None if semanticEyeSupportPolicy is None else normalizeSemanticEyeSupportPolicy(semanticEyeSupportPolicy))
    if eyePolicy is not None and not semanticEyeRefinement:
        raise ValueError('Local semantic eye support requires the existing six-point eye stages')
    camera=cameras[referenceView];extent=float(np.ptp(before,axis=0).max());current=before.copy()
    baselinePairs=intersectionPairs(before,faces);currentPairs=baselinePairs
    beforeErrors=_errors(before,binding,cameras,observations,masks);stages=[]
    observation=observations[referenceView]
    imageScale=min(observation['workingRgb'].shape[:2])/512.
    fit=masks[referenceView] & (np.linalg.norm(_project(_sample(before,binding),camera)-observation['pixels'][binding['ibugIds']],axis=1)<=40*imageScale)
    referenceFrontal=not any(reason in frontQuality['reasons'] for reason in ('referenceMayNotBeFrontal','referenceRollOrEyeOrderUnreliable'))
    for name,ids,minCount in [('mouth',range(48,68),6),('imageLeftEye',range(36,42),4),('imageRightEye',range(42,48),4)]:
        if name!='mouth' and not semanticEyeRefinement:
            stages.append({'name':name,'status':'replacedByVisibleEyeRefinement','scale':0.})
            continue
        selected=fit & np.isin(binding['ibugIds'],list(ids))
        if not referenceFrontal or selected.sum()<minCount or (name=='mouth' and not frontQuality['reliable']):
            stages.append({'name':name,'status':'skippedUnreliableObservation','scale':0.,'selectedPoints':int(selected.sum())});continue
        beforeStage=current.copy();previousErrors=_errors(current,binding,cameras,observations,masks)
        stageFixed=fixed;stageDetails={};localSupport=None
        constraint=_constraint(name,selected,binding,observation)
        captureEye=name!='mouth' and (eyePolicy is not None or captureSemanticEyeEvidence)
        if captureEye:
            stageDetails={'stageEvidence':{
                'frozenSupportReferenceVertices':before.copy(),'before':beforeStage.copy(),
                'acceptedAfter':beforeStage.copy(),'selectedBindingMask':selected.copy(),
                'ibugIds':binding['ibugIds'][selected].copy(),
                'vertexIndices':constraint['vertexIndices'].copy(),
                'barycentricWeights':constraint['barycentric'].copy(),
                'targetPixels':constraint['targetPixels'].copy(),'weights':constraint['weights'].copy(),
                'confidence':constraint['confidence'].copy()}}
            eyeBefore=semanticEyeFitMetrics(beforeStage,binding,observation,camera,selected)
            stageDetails['stageEvidence']['beforeFit']=eyeBefore
            stageDetails['stageEvidence']['supportMode']='frozenLocalTopology' if eyePolicy is not None else 'legacyGlobalCompactField'
        if name!='mouth' and eyePolicy is not None:
            # Freeze both support domains against the original staged input,
            # not a preceding eye proposal or the already moved current skin.
            localSupport=buildSemanticEyeSupport(before,faces,binding,observation,camera,
                eyeIbugIds=list(ids),eligibleMask=selected,fixedVertexIds=fixed,policy=eyePolicy)
            stageDetails['localSupport']=localSupport
            if localSupport['diagnostics']['status']!='supported':
                stageDetails['stageEvidence']['afterFit']=eyeBefore
                stages.append({'name':name,'status':'skippedInsufficientLocalEyeSupport','scale':0.,**stageDetails})
                continue
            stageFixed=localSupport['fixedVertexIds']
        try:
            proposed=refineFrontLandmarks(current,faces,camera,[constraint],
                fixedVertexIds=stageFixed,minConfidence=.4 if name=='mouth' else .5,
                supportRadius=extent*.14,controlCount=32 if name=='mouth' else 12,
                maxDisplacement=extent*.03,maxTargetDistancePixels=40*imageScale,
                maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3,iterations=80)
        except (ValueError,np.linalg.LinAlgError) as exc:
            if captureEye:stageDetails['stageEvidence']['afterFit']=eyeBefore
            stages.append({'name':name,'status':'solverRejected','scale':0.,'detail':str(exc),**stageDetails});continue
        if captureEye:stageDetails['stageEvidence']['proposal']=proposed['vertices'].copy()
        attempts=[];amount=0.
        for alpha in ((1.,.5,.25,.125,.0625) if name=='mouth' else (1.,.5,.25,.125,.0625,.03125,.015625)):
            candidate=beforeStage+alpha*(proposed['vertices']-beforeStage)
            local=checkFrontGeometry(before,candidate,faces,camera=camera,fixedVertexIds=fixed,
                maxDisplacement=extent*.03,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
            attempt={'scale':alpha,'geometry':local}
            if not local['accepted']:
                attempts.append(attempt);continue
            if localSupport is not None:
                stageCheck=checkFrontGeometry(beforeStage,candidate,faces,camera=camera,fixedVertexIds=stageFixed,
                    maxDisplacement=extent*.03,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
                eyeAfter=semanticEyeFitMetrics(candidate,binding,observation,camera,selected)
                tolerance=.05*imageScale
                eyeSafe=(eyeAfter['meanErrorPixels']<=eyeBefore['meanErrorPixels']+tolerance
                    and eyeAfter['maximumErrorPixels']<=eyeBefore['maximumErrorPixels']+tolerance)
                attempt.update(localSupportGeometry=stageCheck,localEyeAlignment={
                    'accepted':bool(eyeSafe),'before':eyeBefore,'after':eyeAfter,
                    'tolerancePixels':tolerance,'scope':'same eligible original six-point targets; both mean and maximum may not regress'})
                if not stageCheck['accepted'] or not eyeSafe:
                    attempts.append(attempt);continue
            pairs=intersectionPairs(candidate,faces);added=pairs-currentPairs
            candidateErrors=_errors(candidate,binding,cameras,observations,masks)
            group='mouth' if name=='mouth' else 'eyes'
            sideDeltas=_sideDelta(candidateErrors,beforeErrors,referenceView,group)
            sideSafe=all(row['changePixels']<=imageScale for row in sideDeltas)
            frontDeltas=[candidateErrors[referenceView]['groups'][region]['meanErrorPixels']-previousErrors[referenceView]['groups'][region]['meanErrorPixels']
                         for region in ('mouth','eyes') if previousErrors[referenceView]['groups'][region]['count']>0]
            frontSafe=not frontDeltas or max(frontDeltas)<=.05*imageScale
            attempt.update(newIntersectionPairs=len(added),newFacePairs=sorted(added),sideMeanErrorChangesPixels=sideDeltas,
                           sideGate=sideSafe,previousStageGate=frontSafe,frontMeanChanges=frontDeltas)
            attempts.append(attempt)
            if not added and sideSafe and frontSafe:
                current=candidate;currentPairs=pairs;amount=alpha;break
        if captureEye:
            stageDetails['stageEvidence']['acceptedAfter']=current.copy()
            stageDetails['stageEvidence']['afterFit']=semanticEyeFitMetrics(current,binding,observation,camera,selected)
        stages.append({'name':name,'status':'accepted' if amount else 'unchangedAfterGates','scale':amount,
                       'solver':proposed['diagnostics'],'attempts':attempts,**stageDetails})
    finalCheck=checkFrontGeometry(before,current,faces,camera=camera,fixedVertexIds=fixed,
        maxDisplacement=extent*.03,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
    finalPairs=intersectionPairs(current,faces)
    if not finalCheck['accepted'] or finalPairs-baselinePairs:
        raise RuntimeError('Final staged client candidate violated cumulative geometry gates')
    return current,fit,{'stages':stages,'before':beforeErrors,'after':_errors(current,binding,cameras,observations,masks),
        'beforeIntersectionPairs':len(baselinePairs),'afterIntersectionPairs':len(finalPairs),'newIntersectionPairs':0,
        'finalGeometryCheck':finalCheck,'fixedVerticesExactlyUnchanged':bool(np.array_equal(current[fixed],before[fixed]))}


def _smooth(value):
    x=np.clip(value,0,1);return x*x*(3-2*x)


def _curveSpline(xy):
    order=np.argsort(xy[:,0]);points=np.asarray(xy,float)[order]
    if len(points)<8 or (np.diff(points[:,0])<=0).any():
        raise UnreliableLipObservation('invalidOrderedVisibleCrease')
    return UnivariateSpline(points[:,0],points[:,1],s=.04*len(points),k=3,ext=3)


def _lateralViewPolicy(cameras,referenceView):
    """Select lateral evidence from relative orientation, retaining all errors.

    Yaw/pitch describe the third column of view-R @ reference-R.T. This is
    a camera-relative gate, not a classification of the capture interaction.
    The accepted experiment gated lateral views; strongly pitched views are
    recorded separately rather than silently reducing its intended correction.
    """
    reference=cameras[referenceView];records=[]
    for view,camera in enumerate(cameras):
        if view==referenceView:continue
        if camera is None:
            records.append({'viewIndex':view,'usedForGate':False,'reason':'cameraUnavailable'});continue
        relative=camera['rotation']@reference['rotation'].T
        yaw=float(np.degrees(np.arctan2(relative[0,2],relative[2,2])))
        pitch=float(np.degrees(np.arctan2(-relative[1,2],np.hypot(relative[0,2],relative[2,2]))))
        lateral=abs(pitch)<=10. and abs(yaw)>=max(5.,1.5*abs(pitch))
        records.append({'viewIndex':view,'relativeYawDegrees':yaw,'relativePitchDegrees':pitch,
            'usedForGate':bool(lateral),'reason':'lateralView' if lateral else 'nonLateralOrStrongPitch',
            'orientationConvention':'third column of viewRotation @ referenceRotation.T'})
    return records


def _affineField(vertices,faces,camera,sourceTrace,photoTrace,fixed,amount):
    K,R,t=(camera[key] for key in ('intrinsics','rotation','translation'))
    cv=vertices@R.T+t;xy=_project(vertices,camera)
    curve=sourceTrace['pixels'][sourceTrace['validMask']]
    target=photoTrace['xy'];confidence=photoTrace['confidence'];scale=photoTrace['imageScale']
    # Match the accepted experiment: confidence-weight every positive target
    # sample. The trusted mask qualifies/evaluates the trace, but is not an
    # extra hard truncation of the affine least-squares fitting population.
    source=_curveSpline(curve)
    keep=(confidence>0)&np.isfinite(confidence)
    if keep.sum()<12:raise UnreliableLipObservation('insufficientAffineCurveSupport')
    centerX=float(np.average(target[keep,0],weights=confidence[keep]));lo=max(curve[0,0],target[0,0]);hi=min(curve[-1,0],target[-1,0])
    if hi-lo<.4*photoTrace['quality']['mouthWidthPixels']:
        raise UnreliableLipObservation('insufficientCommonAffineCurveSpan')
    difference=target[keep,1]-source(target[keep,0])
    design=np.column_stack([target[keep,0]-centerX,np.ones(keep.sum())]);weight=np.sqrt(confidence[keep])
    coefficients=np.linalg.lstsq(design*weight[:,None],difference*weight,rcond=None)[0]
    centerY=float(np.median(curve[:,1]));width=80*scale
    # Ratios retain the calibrated support shape while translating/scaling it
    # with this photograph's mouth. No absolute reference-image ROI is used.
    x0,x1=lo-15*scale,hi+15*scale;y0,y1=centerY-12*scale,centerY+18*scale
    fadeX,fadeTop,fadeBottom=20*scale,40*scale,60*scale
    xWeight=_smooth((xy[:,0]-(x0-fadeX))/fadeX)*_smooth(((x1+fadeX)-xy[:,0])/fadeX)
    yWeight=_smooth((xy[:,1]-(y0-fadeTop))/fadeTop)*_smooth(((y1+fadeBottom)-xy[:,1])/fadeBottom)
    sourceY=source(xy[:,0])
    lip=(xy[:,0]>=lo)&(xy[:,0]<=hi)&(abs(xy[:,1]-sourceY)<10*scale)
    if lip.sum()<8:raise UnreliableLipObservation('insufficientMouthLayerVertices')
    frontDepth=float(np.quantile(cv[lip,2],.2));extent=float(np.ptp(vertices,axis=0).max())
    zWeight=1-_smooth((cv[:,2]-(frontDepth+.12*extent))/(.12*extent))
    shift=(xy[:,0]-centerX)*coefficients[0]+coefficients[1]
    displacement=amount*shift*xWeight*yWeight*zWeight
    if np.max(abs(displacement))>20*scale:
        raise UnreliableLipObservation('affineShiftExceedsObservationScale')
    if np.any(abs(displacement[fixed])>1e-12):
        raise UnreliableLipObservation('affineSupportTouchesFixedInterface')
    displacement[fixed]=0.
    cameraDelta=(np.column_stack([np.zeros(len(vertices)),displacement,np.zeros(len(vertices))])@np.linalg.inv(K).T)*cv[:,2,None]
    candidate=vertices+cameraDelta@R;candidate[displacement==0]=vertices[displacement==0]
    gx,gy=np.meshgrid(np.linspace(x0-fadeX,x1+fadeX,201),np.linspace(y0-fadeTop,y1+fadeBottom,201))
    fieldShift=((gx-centerX)*coefficients[0]+coefficients[1])*_smooth((gx-(x0-fadeX))/fadeX)*_smooth(((x1+fadeX)-gx)/fadeX)
    mapped=gy+amount*fieldShift*_smooth((gy-(y0-fadeTop))/fadeTop)*_smooth(((y1+fadeBottom)-gy)/fadeBottom)
    jacobian=np.gradient(mapped,gy[:,0],axis=0)
    return candidate,{'amount':amount,'affineCenterX':centerX,'affineYShiftCoefficients':coefficients.tolist(),
        'affinePlateau':{'x':[x0,x1],'y':[y0,y1],'xFadePixels':fadeX,'topFadePixels':fadeTop,'bottomFadePixels':fadeBottom},
        'minimumSampledVerticalJacobian':float(jacobian.min()),'maximumImageDisplacementPixels':float(abs(displacement).max()),
        'changedVertexCount':int(np.count_nonzero(displacement)),'depthGatePlateauEndsAt':frontDepth+.12*extent,
        'referenceDepthFixed':True,'horizontalShiftPixels':0,'supportDerivedFromCurrentMouth':True,
        'affineFitPointCount':int(keep.sum()),'affineFitPolicy':'all positive-confidence photo-curve samples; smoothed cubic source spline; weighted least squares'}


def _refineCrease(current,faces,fixed,binding,cameras,observations,masks,referenceView,delight,output,*,cumulativeReference):
    observation=observations[referenceView];camera=cameras[referenceView]
    if delight is None or 'albedoImages' not in delight:
        raise UnreliableLipObservation('currentOpenDelightImagesUnavailable')
    native=np.asarray(delight['albedoImages'][referenceView])
    linear=decodeColor(native,'gamma2.2')
    value=np.clip(linear,0,1);srgb=np.where(value<=.0031308,12.92*value,1.055*value**(1/2.4)-.055)
    image=np.rint(np.clip(srgb,0,1)*255).astype(np.uint8)
    validity=delight['validMasks'][referenceView] if 'validMasks' in delight else None
    photo=tracePhotoLip(observation['workingRgb'],image,observation['pixels'],observation['confidence'],validMask=validity)
    projected=np.full((68,2),np.nan);projected[binding['ibugIds']]=_project(_sample(current,binding),camera)
    search=deriveCreaseSearch(observation['pixels'],projected,observation['workingRgb'].shape[:2])
    sourceTrace=traceVisibleCrease(current,faces,camera,search);beforeError=creaseError(sourceTrace,photo)
    pairsBefore=intersectionPairs(current,faces);extent=float(np.ptp(current,axis=0).max())
    beforeErrors=_errors(current,binding,cameras,observations,masks);attempts=[];chosen=current.copy();afterError=beforeError;chosenTrace=sourceTrace;amountChosen=0.
    viewPolicy=_lateralViewPolicy(cameras,referenceView)
    lateralIndices={row['viewIndex'] for row in viewPolicy if row['usedForGate']}
    scale=photo['imageScale']
    for amount in (1.,.75,.5,.25,.125,.0625):
        try:candidate,field=_affineField(current,faces,camera,sourceTrace,photo,fixed,amount)
        except UnreliableLipObservation as exc:
            attempts.append({'amount':amount,'accepted':False,'reason':str(exc)});continue
        local=checkFrontGeometry(current,candidate,faces,camera=camera,fixedVertexIds=fixed,
            maxDisplacement=extent*.04,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        cumulative=checkFrontGeometry(cumulativeReference,candidate,faces,camera=camera,fixedVertexIds=fixed,
            maxDisplacement=float(np.ptp(cumulativeReference,axis=0).max())*.07,
            maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        record={'amount':amount,'field':field,'geometry':local,'cumulativeGeometry':cumulative,'accepted':False}
        if not local['accepted'] or not cumulative['accepted'] or field['minimumSampledVerticalJacobian']<=.25:
            attempts.append(record);continue
        pairs=intersectionPairs(candidate,faces);added=pairs-pairsBefore
        try:
            newTrace=traceVisibleCrease(candidate,faces,camera,search);error=creaseError(newTrace,photo)
        except UnreliableLipObservation as exc:
            record['reason']=str(exc);attempts.append(record);continue
        errors=_errors(candidate,binding,cameras,observations,masks)
        sideDeltas=_sideDelta(errors,beforeErrors,referenceView,'mouth',minimum=6)
        gateDeltas=[row for row in sideDeltas if row['viewIndex'] in lateralIndices]
        sideSafe=all(row['changePixels']<=max(1.,scale) for row in gateDeltas)
        accepted=not added and sideSafe and error['meanAbsoluteYErrorPixels']<beforeError['meanAbsoluteYErrorPixels']-.025*scale
        record.update(accepted=accepted,newIntersectionPairs=len(added),newFacePairs=sorted(added),
            removedIntersectionPairs=len(pairsBefore-pairs),creaseProxy=error,sideMouthMeanErrorChangesPixels=sideDeltas,
            lateralMouthGateChangesPixels=gateDeltas,ungatedViewChangesPixels=[row for row in sideDeltas if row['viewIndex'] not in lateralIndices],sideGate=sideSafe)
        attempts.append(record)
        if accepted:
            chosen=candidate;afterError=error;chosenTrace=newTrace;amountChosen=amount;break
    np.savez_compressed(output/'photoLipTrace.npz',**{k:v for k,v in photo.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(output/'geometryLipTrace.npz',beforePixels=sourceTrace['pixels'],beforeValid=sourceTrace['validMask'],
        afterPixels=chosenTrace['pixels'],afterValid=chosenTrace['validMask'])
    finalPairs=intersectionPairs(chosen,faces)
    if finalPairs-pairsBefore:raise RuntimeError('Final lip candidate introduced an intersection')
    return chosen,{'status':'accepted' if amountChosen else 'unchangedAfterGates','chosenAmount':amountChosen,
        'beforeCreaseProxy':beforeError,'afterCreaseProxy':afterError,'attempts':attempts,'search':search,
        'photoQuality':photo['quality'],'beforeIntersectionPairs':len(pairsBefore),'afterIntersectionPairs':len(finalPairs),
        'newIntersectionPairs':0,'beforeViewErrors':beforeErrors,'afterViewErrors':_errors(chosen,binding,cameras,observations,masks),
        'viewGatePolicy':viewPolicy,'lateralGateCriterion':'abs(relativePitch)<=10 degrees and abs(relativeYaw)>=max(5 degrees,1.5*abs(pitch)); at least 6 usable mouth points',
        'ungatedViewsStillReported':True,'curveMeaning':photo['meaning'],'depthPreserved':True}


def refineClientGeometry(assets, geometry, cameras, inputs, *, photoPaths, delight,
                         projectRoot, outputDirectory, workingPhotoPaths=None, referenceViewIndex=0):
    """Return a refined ordered client skin and complete per-job diagnostics.

    ``assets`` must already be a verified directSurface AssemblyAssets with
    restWeight=0 and hard static constraints. ``geometry`` is native UFF data.
    ``cameras`` and ``inputs`` retain the same view order as photoPaths and
    delight albedoImages/validMasks. Optional workingPhotoPaths is used only
    when an original image is unavailable, with that difference recorded.

    Returns vertices (2795 canonical), templateVertices (2795 template cm),
    directTemplateVertices, faces, templateToUff, diagnostics and mouthState.
    ``mouthObservationReliable`` must be honored before texture mouth override.
    An explicit ``referenceViewIndex=None`` retains direct registration and
    records unavailable reference constraints; it never substitutes view zero.
    """
    start=perf_counter();output=Path(outputDirectory);output.mkdir(parents=True,exist_ok=True)
    if (assets.geometryMethod!='directSurface' or assets.directArrays is None
            or abs(float(assets.geometryTransfer.get('restWeight',1000)))>1e-12
            or assets.geometryTransfer.get('staticConstraint')!='hard'):
        raise ValueError('Client refinement requires verified directSurface with zero shape prior and hard fixed vertices')
    count=len(inputs)
    if not count or len(cameras)!=count or len(photoPaths)!=count or (workingPhotoPaths is not None and len(workingPhotoPaths)!=count):
        raise ValueError('Current photographs, input transforms and cameras must retain identical view order')
    if referenceViewIndex is not None and (isinstance(referenceViewIndex,bool)
            or not isinstance(referenceViewIndex,int) or not 0<=referenceViewIndex<count):
        raise ValueError('Invalid reference view index')
    # Mandatory registration errors deliberately propagate: no legacy or raw
    # UFF geometry can silently replace a failed requested client registration.
    assembled,_,directDiagnostics=assets._deform(geometry)
    ids=assets.arrays['nicpToFullIds'];faces=assets.arrays['nicpFaces'];fixed=np.flatnonzero(assets.arrays['staticMask'])
    directTemplate=np.asarray(assembled['vertices'][ids],float).copy();transform=directDiagnostics['templateToUff']
    rotation=np.asarray(transform['rotation'],float);translation=np.asarray(transform['translation'],float);scale=float(transform['scale'])
    if directTemplate.shape!=(2795,3) or not np.isfinite(directTemplate).all() or not np.isfinite(scale) or scale<=0:
        raise ValueError('Direct registration returned invalid canonical client geometry')
    before=(directTemplate@rotation)*scale+translation
    return _refineRegisteredGeometry(assets,before,directTemplate,transform,directDiagnostics,
        cameras,inputs,photoPaths=photoPaths,delight=delight,projectRoot=projectRoot,
        outputDirectory=output,workingPhotoPaths=workingPhotoPaths,
        referenceViewIndex=referenceViewIndex,start=start)


def observeClientGeometryPhotos(projectRoot, photoPaths, inputs, cameras, *, workingPhotoPaths=None):
    """Infer reusable photo-only observations in the current working frame.

    The returned observations contain no surface bindings, visibility, or
    displacement. Those are recomputed for each registered mesh. Original
    photograph hashes are checked by the existing inference path; a caller
    persisting this bundle must also pin its arrays, reports and input frames.
    """
    count=len(inputs)
    if (not count or len(cameras)!=count or len(photoPaths)!=count
            or (workingPhotoPaths is not None and len(workingPhotoPaths)!=count)):
        raise ValueError('Photographs, input frames and cameras must retain identical view order')
    checked=[]
    for camera in cameras:
        try:
            value=_validateCameras([camera])[0]
            if not bool(camera.get('valid',True)):raise ValueError('Rejected camera')
            checked.append(value)
        except (ValueError,KeyError,TypeError):
            checked.append(None)
    observations,reports=_observations(projectRoot,photoPaths,workingPhotoPaths,inputs,checked)
    return {'observations':observations,'reports':reports}


def _cachedGeometryObservations(cachedObservations, inputs, checkedCameras):
    """Check numerical/frame schema, leaving persisted source hashes to callers."""
    if not isinstance(cachedObservations,dict) or set(cachedObservations)!={'observations','reports'}:
        raise ValueError('Cached observations require observations and reports')
    observations=cachedObservations['observations'];reports=cachedObservations['reports']
    if len(observations)!=len(inputs) or len(reports)!=len(inputs):
        raise ValueError('Cached observation count differs from current inputs')
    result=[]
    for index,(observation,record,item,camera) in enumerate(zip(observations,reports,inputs,checkedCameras)):
        if not isinstance(record,dict) or record.get('viewIndex')!=index:
            raise ValueError('Cached observation reports must retain ordered view indices')
        if observation is None:
            result.append(None);continue
        if not isinstance(observation,dict) or not {'pixels','confidence','workingRgb'}<=set(observation):
            raise ValueError('Cached observation lacks pixels, confidence or workingRgb')
        pixels=np.asarray(observation['pixels']);scores=np.asarray(observation['confidence'])
        rgb=np.asarray(observation['workingRgb']);size=np.asarray(item.get('workingSize',[]))
        if (pixels.shape!=(68,2) or scores.shape!=(68,) or pixels.dtype.kind not in 'fiu'
                or scores.dtype.kind not in 'fiu' or not np.isfinite(pixels).all()
                or not np.isfinite(scores).all()):
            raise ValueError('Cached landmarks and confidence must be finite 68-point observations')
        if (size.shape!=(2,) or size.dtype.kind not in 'iu' or np.any(size<4)
                or rgb.shape!=(int(size[1]),int(size[0]),3) or rgb.dtype!=np.uint8):
            raise ValueError('Cached workingRgb must be uint8 RGB in the current workingSize frame')
        if camera is None:
            raise ValueError('Cached accepted observation has no valid current camera')
        # Copies prevent either branch from changing the other branch's input.
        result.append({'pixels':pixels.copy(),'confidence':scores.copy(),'workingRgb':rgb.copy()})
    return result,deepcopy(reports)


def refineRegisteredClientGeometry(assets, vertices, faces, templateToUff, cameras, inputs, *,
                                  photoPaths, delight, projectRoot, outputDirectory,
                                  workingPhotoPaths=None, referenceViewIndex=0,
                                  cachedObservations=None, registrationDiagnostics=None,
                                  semanticEyeRefinement=True,semanticEyeSupportPolicy=None,captureSemanticEyeEvidence=False):
    """Refine an explicitly supplied initial canonical client skin.

    This opt-in entry never registers again or copies a historical deformation.
    It shares the production eye/mouth/lip stages and their safety gates, all
    relative to this exact input mesh. Cameras, visibility and lip bindings are
    evaluated afresh. Static vertices are checked against the transformed rest
    template within floating-point tolerance and then retained exactly, without
    a template/canonical round trip. The caller validates the initial solver's
    acceptance, template/source hashes and cached-observation provenance.

    ``cachedObservations`` is the photo-only bundle returned by
    ``observeClientGeometryPhotos``. Persisted bundles need source/frame hash
    validation by the caller; this entry validates their numerical schema.

    Set ``semanticEyeRefinement=False`` only when replacing the six-point eye
    stages with a separately validated visible-aperture module after assembly.
    Mouth and lip stages, fixed vertices and cumulative safety gates remain.
    ``semanticEyeSupportPolicy`` instead keeps those six-point eye stages and
    optionally hard-pins skin outside a frozen local topology band. None keeps
    the original support and behavior. It cannot be combined with disabling
    the semantic eyes, and never changes accessory placement.
    ``captureSemanticEyeEvidence=True`` records each eligible eye stage's
    exact states and original point targets for paired diagnostic comparisons;
    it does not change fitting or acceptance. The default adds no evidence.
    """
    start=perf_counter()
    if not isinstance(semanticEyeRefinement,bool):
        raise ValueError('semanticEyeRefinement must be boolean')
    if not isinstance(captureSemanticEyeEvidence,bool):raise ValueError('captureSemanticEyeEvidence must be boolean')
    if semanticEyeSupportPolicy is not None:
        semanticEyeSupportPolicy=normalizeSemanticEyeSupportPolicy(semanticEyeSupportPolicy)
        if not semanticEyeRefinement:
            raise ValueError('Local semantic eye support requires the existing six-point eye stages')
    if (assets.geometryMethod!='directSurface' or assets.directArrays is None
            or abs(float(assets.geometryTransfer.get('restWeight',1000)))>1e-12
            or assets.geometryTransfer.get('staticConstraint')!='hard'):
        raise ValueError('Client refinement requires verified directSurface with zero shape prior and hard fixed vertices')
    count=len(inputs)
    if (not count or len(cameras)!=count or len(photoPaths)!=count
            or (workingPhotoPaths is not None and len(workingPhotoPaths)!=count)):
        raise ValueError('Current photographs, input transforms and cameras must retain identical view order')
    if referenceViewIndex is not None and (isinstance(referenceViewIndex,bool)
            or not isinstance(referenceViewIndex,int) or not 0<=referenceViewIndex<count):
        raise ValueError('Invalid reference view index')
    arrays=assets.arrays;ids=np.asarray(arrays['nicpToFullIds']);expectedFaces=np.asarray(arrays['nicpFaces'])
    suppliedFaces=np.asarray(faces);before=np.asarray(vertices)
    if (suppliedFaces.dtype.kind not in 'iu' or not np.array_equal(suppliedFaces,expectedFaces)):
        raise ValueError('Registered skin must retain exact client face order and winding')
    if (before.shape!=(len(ids),3) or len(ids)!=2795 or before.dtype.kind!='f'
            or not np.isfinite(before).all()):
        raise ValueError('Registered skin must be finite floating-point canonical client vertices')
    before=before.astype(np.float64,copy=True)
    transform=deepcopy(templateToUff)
    rotation=np.asarray(transform['rotation'],float);translation=np.asarray(transform['translation'],float)
    scale=float(transform['scale'])
    if (rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all() or not np.isfinite(scale) or scale<=0
            or not np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-8)
            or np.linalg.det(rotation)<=0):
        raise ValueError('Registered skin requires a proper template-to-UFF similarity')
    static=np.asarray(arrays['staticMask'])
    if static.shape!=(len(ids),) or static.dtype.kind!='b':
        raise ValueError('Client static mask must be boolean per ordered vertex')
    rest=np.asarray(arrays['vertices'],float)[ids]
    alignedRest=(rest@rotation)*scale+translation
    fixed=np.flatnonzero(static)
    tolerance=max(float(np.ptp(alignedRest,axis=0).max())*1e-12,
                  float(np.abs(alignedRest).max())*np.finfo(float).eps*64,np.finfo(float).tiny)
    if not np.allclose(before[fixed],alignedRest[fixed],rtol=0,atol=tolerance):
        raise ValueError('Registered static vertices differ from the transformed rest template')
    directTemplate=((before-translation)@rotation.T)/scale
    directTemplate[fixed]=rest[fixed]
    directDiagnostics=deepcopy(registrationDiagnostics) if registrationDiagnostics is not None else {}
    if not isinstance(directDiagnostics,dict):
        raise ValueError('Registration diagnostics must be a mapping')
    if 'templateToUff' in directDiagnostics:
        previous=directDiagnostics['templateToUff']
        if any(not np.array_equal(np.asarray(previous[key]),np.asarray(transform[key]))
               for key in ('rotation','translation','scale')):
            raise ValueError('Registration diagnostics disagree with templateToUff')
    directDiagnostics.update(templateToUff=transform,initialRegistrationProvided=True)
    output=Path(outputDirectory);output.mkdir(parents=True,exist_ok=True)
    return _refineRegisteredGeometry(assets,before,directTemplate,transform,directDiagnostics,
        cameras,inputs,photoPaths=photoPaths,delight=delight,projectRoot=projectRoot,
        outputDirectory=output,workingPhotoPaths=workingPhotoPaths,
        referenceViewIndex=referenceViewIndex,start=start,cachedObservations=cachedObservations,
        semanticEyeRefinement=semanticEyeRefinement,semanticEyeSupportPolicy=semanticEyeSupportPolicy,
        captureSemanticEyeEvidence=captureSemanticEyeEvidence)


def _refineRegisteredGeometry(assets,before,directTemplate,transform,directDiagnostics,
                              cameras,inputs,*,photoPaths,delight,projectRoot,outputDirectory,
                              workingPhotoPaths,referenceViewIndex,start,cachedObservations=None,
                              semanticEyeRefinement=True,semanticEyeSupportPolicy=None,captureSemanticEyeEvidence=False):
    """Common production refinement, starting from exact canonical vertices."""
    output=Path(outputDirectory)
    faces=assets.arrays['nicpFaces'];fixed=np.flatnonzero(assets.arrays['staticMask'])
    rotation=np.asarray(transform['rotation'],float);translation=np.asarray(transform['translation'],float)
    scale=float(transform['scale'])
    fullBinding=loadClientLandmarks(faces,path=Path(projectRoot)/'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json')
    if not fullBinding['isOriginalBinding']:
        raise ValueError('Client refinement requires the audited semantic binding hash')
    binding=ibugClientLandmarks(fullBinding,groups=('eyes','mouth'))
    checked=[];cameraReports=[]
    for view,camera in enumerate(cameras):
        try:
            value=_validateCameras([camera])[0]
            if not bool(camera.get('valid',True)):raise ValueError('Camera was not accepted by reconstruction')
            checked.append(value);cameraReports.append({'viewIndex':view,'valid':True})
        except (ValueError,KeyError,TypeError) as exc:
            checked.append(None);cameraReports.append({'viewIndex':view,'valid':False,'reason':str(exc)})
    observations,observationReports=(_observations(projectRoot,photoPaths,workingPhotoPaths,inputs,checked)
        if cachedObservations is None else _cachedGeometryObservations(cachedObservations,inputs,checked))
    masks=[]
    for camera,observation in zip(checked,observations):
        if camera is None or observation is None:masks.append(np.zeros(len(binding['ibugIds']),bool));continue
        points=observation['pixels'][binding['ibugIds']];scores=observation['confidence'][binding['ibugIds']]
        height,width=observation['workingRgb'].shape[:2]
        keep=_visible(before,faces,_sample(before,binding),camera,(height,width))&binding['validMask']
        keep&=scores>=np.where(binding['ibugIds']>=48,.4,.5)
        keep&=(points>=0).all(1)&(points<[width,height]).all(1);masks.append(keep)
    reference=observations[referenceViewIndex] if referenceViewIndex is not None else None
    current=before.copy();fit=np.zeros(len(binding['ibugIds']),bool)
    quality=mouthObservationQuality(reference['pixels'],reference['confidence'],reference['workingRgb'].shape[:2]) if reference is not None else {'reliable':False,'reasons':['referenceObservationUnavailable']}
    stages={'status':'skipped','reason':'referenceObservationUnavailable'};lip={'status':'skipped','reason':'referenceObservationUnavailable'}
    if reference is not None and checked[referenceViewIndex] is not None:
        stageOptions={} if semanticEyeRefinement else {'semanticEyeRefinement':False}
        if semanticEyeSupportPolicy is not None:stageOptions['semanticEyeSupportPolicy']=semanticEyeSupportPolicy
        if captureSemanticEyeEvidence:stageOptions['captureSemanticEyeEvidence']=True
        current,fit,stages=_stagedLandmarks(before,faces,fixed,binding,checked,observations,masks,referenceViewIndex,quality,**stageOptions)
        stages['status']='completedWithPerStageGates'
        try:current,lip=_refineCrease(current,faces,fixed,binding,checked,observations,masks,referenceViewIndex,delight,output,cumulativeReference=before)
        except UnreliableLipObservation as exc:lip={'status':'skippedUnreliableObservation','reason':str(exc)}
    finalTemplate=((current-translation)@rotation.T)/scale
    if referenceViewIndex is None:
        finalTemplate=directTemplate.copy()
    finalTemplate[fixed]=directTemplate[fixed]
    # Eliminate inverse-transform noise by reporting the exact final template
    # values while retaining the checked canonical geometry for texture baking.
    if not np.array_equal(current[fixed],before[fixed]):raise RuntimeError('Refinement changed the fixed client interface')
    finalPairs=intersectionPairs(current,faces);baselinePairs=intersectionPairs(before,faces)
    if finalPairs-baselinePairs:raise RuntimeError('Combined client refinement introduced an intersection')
    combinedCheck=(checkFrontGeometry(before,current,faces,camera=checked[referenceViewIndex],fixedVertexIds=fixed,
        maxDisplacement=float(np.ptp(before,axis=0).max())*.07,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        if referenceViewIndex is not None and checked[referenceViewIndex] is not None else None)
    if combinedCheck is not None and not combinedCheck['accepted']:
        raise RuntimeError('Combined client refinement violated cumulative geometry gates')
    frontPoints=reference['pixels'][binding['ibugIds']] if reference is not None else np.full((len(binding['ibugIds']),2),np.nan)
    frontScores=reference['confidence'][binding['ibugIds']] if reference is not None else np.zeros(len(binding['ibugIds']))
    mouthState={'ibugIds':binding['ibugIds'].copy(),'vertexIndices':binding['vertexIndices'].copy(),
        'barycentricWeights':binding['barycentricWeights'].copy(),'frontTargetPixels':frontPoints.copy(),
        'frontConfidence':frontScores.copy(),'frontEligibleFitMask':fit,
        'frontEvaluationMask':masks[referenceViewIndex] if referenceViewIndex is not None else np.zeros(len(binding['ibugIds']),bool),
        'beforeCanonical':before,'afterCanonical':current,'referenceViewIndex':referenceViewIndex,
        'outerLipPixels':reference['pixels'][48:60].copy() if reference is not None else np.full((12,2),np.nan),
        'outerLipConfidence':reference['confidence'][48:60].copy() if reference is not None else np.zeros(12),
        'faceLandmarksWorking':reference['pixels'].copy() if reference is not None else np.full((68,2),np.nan),
        'mouthObservationReliable':bool(quality['reliable']),'mouthObservationRejectionReasons':quality['reasons']}
    diagnostics={'algorithmVersion':ALGORITHM_VERSION,'status':'modified' if np.max(abs(current-before))>1e-12 else 'directResultRetained',
        'directRegistration':directDiagnostics,'referenceViewIndex':referenceViewIndex,'camerasRefitted':False,
        'cameraStatus':cameraReports,'observations':observationReports,'stagedLandmarks':stages,'lipCrease':lip,
        'beforeViewErrors':_errors(before,binding,checked,observations,masks),'afterViewErrors':_errors(current,binding,checked,observations,masks),
        'fixedVerticesExactlyUnchanged':True,'maxDisplacementCm':float(np.linalg.norm(finalTemplate-directTemplate,axis=1).max()),
        'beforeIntersectionPairs':len(baselinePairs),'afterIntersectionPairs':len(finalPairs),'newIntersectionPairs':0,
        'combinedGeometryCheck':combinedCheck,
        'mouthObservationQuality':quality,'seconds':perf_counter()-start,
        'limitations':['Current-photo automatic landmarks and dark curves are proxies, not anatomical ground truth.',
            'The latest lip adjustment was developed on one male sample; current-frame gates are not validation of all inputs.',
            'Existing nonadjacent intersections remain; shared-vertex triangle contacts are excluded from the regression gate.',
            'Unknown or unreliable observations retain direct geometry or the last accepted stage and report why.',
            'Reference camera depth, original triangle order and client UVs are preserved.']}
    np.savez_compressed(output/'clientGeometry.npz',vertices=current,templateVertices=finalTemplate,directTemplateVertices=directTemplate,faces=faces,
        templateToUffRotation=rotation,templateToUffTranslation=translation,templateToUffScale=np.asarray(scale))
    np.savez_compressed(output/'mouthState.npz',**{k:v for k,v in mouthState.items() if isinstance(v,np.ndarray)},
        referenceViewIndex=np.asarray(referenceViewIndex if referenceViewIndex is not None else -1),
        referenceViewAvailable=np.asarray(referenceViewIndex is not None),mouthObservationReliable=np.asarray(quality['reliable']))
    np.savez_compressed(output/'landmarkObservations.npz',faceLandmarksWorking=np.stack([x['pixels'] if x is not None else np.full((68,2),np.nan) for x in observations]),
        faceRawPeakScores=np.stack([x['confidence'] if x is not None else np.zeros(68) for x in observations]),
        acceptedViewMask=np.array([x is not None for x in observations]))
    diagnostics=_json(diagnostics)
    (output/'geometryRefinement.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'vertices':current,'templateVertices':finalTemplate,'directTemplateVertices':directTemplate,'faces':faces.copy(),
        'templateToUff':transform,'diagnostics':diagnostics,'mouthState':mouthState}
