"""Read saved aperture masks; no rasterization, fitting or candidate mutation.

This separates the eye stage's frozen pre-eye assembly from the factorial
baseline, which still uses the old six-point stages. A missing baseline mask
is reported as missing, never silently replaced with the pre-eye aperture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/featureGeometry20261024'
OUT=ROOT/'reports/featureGeometry20261024/eyeAudit'
NAMES=('imageLeft','imageRight')
SIZE_KEYS=('widthPixels','maximumOpeningHeightPixels','areaPixelsSquared')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    def numpyValue(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(f'Unsupported JSON value: {type(item).__name__}')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=numpyValue))


def loadTree(value,arrays):
    if isinstance(value,dict):
        if 'arrayKey' in value:
            array=arrays[value['arrayKey']]
            if list(array.shape)!=value['shape'] or str(array.dtype)!=value['dtype']:
                raise ValueError('Evidence shape/dtype differs from its JSON index')
            return array
        return {key:loadTree(item,arrays) for key,item in value.items()}
    if isinstance(value,list):return [loadTree(item,arrays) for item in value]
    return value


def polygonMetrics(polygon,axis):
    """Continuous photo polygon span and maximum section normal to its axis."""
    points=np.asarray(polygon,float)@np.column_stack([axis,[-axis[1],axis[0]]])
    width=float(np.ptp(points[:,0]));other=np.roll(points,-1,axis=0);heights=[]
    for x in np.linspace(points[:,0].min()+width*1e-7,points[:,0].max()-width*1e-7,513):
        cross=((points[:,0]<=x)&(other[:,0]>x))|((other[:,0]<=x)&(points[:,0]>x))
        y=points[cross,1]+(x-points[cross,0])*(other[cross,1]-points[cross,1])/(other[cross,0]-points[cross,0])
        y.sort();heights.append(float(np.diff(y.reshape(-1,2),axis=1).sum()) if len(y)%2==0 else 0.)
    return dict(widthPixels=width,maximumOpeningHeightPixels=max(heights),
        areaPixelsSquared=float(abs(np.sum(points[:,0]*other[:,1]-points[:,1]*other[:,0]))*.5))


def maskMetrics(eye,photo,originalPixelScale):
    if 'roiMasks' not in eye or 'roi' not in eye:
        return {'status':'missingVisibleMask','extractionStatus':eye.get('status'),'extractionReasons':eye.get('reasons',[])}
    roi=eye['roi'];model=np.asarray(eye['roiMasks']['eye'],bool)
    scale=float(roi['scale']);origin=np.asarray(roi['origin'])
    first=(photo['upper'][0]+photo['lower'][0])*.5
    last=(photo['upper'][-1]+photo['lower'][-1])*.5
    axis=(last-first)/np.linalg.norm(last-first)
    polygon=np.r_[photo['upper'],photo['lower'][::-1]]
    photoSize=polygonMetrics(polygon,axis)
    target=np.zeros(model.shape,np.uint8)
    cv2.fillPoly(target,[np.rint((polygon-origin)*scale*256).astype(np.int32)],1,shift=8)
    target=target.astype(bool)
    if not model.any() or not target.any():
        return {'status':'emptyMask','extractionStatus':eye.get('status'),'extractionReasons':eye.get('reasons',[])}
    contours,_=cv2.findContours(model.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=[row[:,0]/scale+origin for row in contours if len(row)>=3]
    pixels=np.column_stack(np.nonzero(model))[:,::-1]/scale+origin
    modelSize=dict(widthPixels=float(np.ptp(pixels@axis)),
        maximumOpeningHeightPixels=max(polygonMetrics(row,axis)['maximumOpeningHeightPixels'] for row in contours),
        areaPixelsSquared=float(model.sum()/scale**2))
    modelEdge=model & ~binary_erosion(model,structure=np.ones((3,3)),border_value=0)
    targetEdge=target & ~binary_erosion(target,structure=np.ones((3,3)),border_value=0)
    unit=1/(scale*originalPixelScale)
    forward=distance_transform_edt(~targetEdge)[modelEdge]*unit
    backward=distance_transform_edt(~modelEdge)[targetEdge]*unit
    count,_,stats,_=cv2.connectedComponentsWithStats(model.astype(np.uint8),8)
    components=sorted(map(int,stats[1:,cv2.CC_STAT_AREA]),reverse=True)
    return dict(status='measuredFromSavedFullApertureMask',extractionStatus=eye.get('status'),
        extractionReasons=eye.get('reasons',[]),modelSizeWorkingPixels=modelSize,
        photoSizeWorkingPixels=photoSize,modelToPhotoRatios={k:modelSize[k]/photoSize[k] for k in SIZE_KEYS},
        intersectionOverUnion=float((model&target).sum()/(model|target).sum()),
        symmetricBoundaryMeanOriginalPixels=.5*float(forward.mean()+backward.mean()),
        symmetricBoundaryP95OriginalPixels=float(np.percentile(np.r_[forward,backward],95)),
        visibleComponentCount=int(count-1),secondaryComponentAreaFraction=sum(components[1:])/sum(components),
        modelRasterArea=int(model.sum()),photoRasterArea=int(target.sum()),
        outsidePhotoAreaWorkingPixelsSquared=float((model&~target).sum()/scale**2),
        rasterPixelSizeWorkingPixels=1/scale,originalPixelScale=float(originalPixelScale),
        maskSha256=hashlib.sha256(model.tobytes()).hexdigest(),roi=roi,
        apertureAccessoryMode=eye['diagnostics'].get('apertureAccessoryMode','off'),
        scope='Every saved full visible aperture fragment including explicit canthi; automatic photo polygon is a proxy, not anatomical ground truth')


def compare(before,after):
    result={}
    for name in NAMES:
        a,b=before[name],after[name]
        if a['status']!='measuredFromSavedFullApertureMask' or b['status']!=a['status']:
            result[name]={'status':'unavailable'};continue
        result[name]=dict(status='comparableFrozenPhotoAndCamera',
            sizeRatioDifference={k:b['modelToPhotoRatios'][k]-a['modelToPhotoRatios'][k] for k in SIZE_KEYS},
            intersectionOverUnionDifference=b['intersectionOverUnion']-a['intersectionOverUnion'],
            boundaryMeanOriginalPixelsDifference=b['symmetricBoundaryMeanOriginalPixels']-a['symmetricBoundaryMeanOriginalPixels'],
            boundaryP95OriginalPixelsDifference=b['symmetricBoundaryP95OriginalPixels']-a['symmetricBoundaryP95OriginalPixels'],
            componentCountDifference=b['visibleComponentCount']-a['visibleComponentCount'])
    return result


def summarizeCase(caseId):
    caseRoot=RUN/caseId
    if not (caseRoot/'inputContext.json').exists():return dict(caseId=caseId,status='pendingCase')
    context=read(caseRoot/'inputContext.json');reference=context['referenceSelection']['referenceViewIndex']
    densePath=ROOT/'runs/shapePipeline20261024'/caseId/'observations/observations.npz'
    denseReportPath=densePath.with_suffix('.json');denseReport=read(denseReportPath)
    if sha(densePath)!=denseReport['observationSha256'] or reference!=denseReport['referenceViewIndex']:
        raise ValueError('Dense photo identity differs from the shared reference')
    with np.load(densePath,allow_pickle=False) as arrays:
        curves=arrays['eyelidCurvesWorking'];accepted=arrays['acceptedEyeMask']
    if not accepted.all():raise ValueError('Rejected shared photo eyes')
    photo={name:{part:curves[i,j] for j,part in enumerate(('lower','upper'))} for i,name in enumerate(NAMES)}
    reconstruction=Path(context['entry']['reconstruction']);inputsPath=reconstruction/'inputs.json'
    cameraPath=reconstruction/'cameras.json';photoPath=Path(context['entry']['workingPhotos'])/f'input{reference:02d}.png'
    pins={str(path):sha(path) for path in (caseRoot/'inputContext.json',densePath,denseReportPath,inputsPath,cameraPath,photoPath)}
    if pins[str(photoPath)]!=denseReport['sourcePhotoSha256'] or pins[str(cameraPath)]!=denseReport['sourceHashes'][str(cameraPath)]:
        raise ValueError('Frozen camera or working RGB identity changed')
    originalPixelScale=float(np.asarray(read(inputsPath)[reference]['inputToWorking'])[0,0])
    branches={}
    for branch in ('visibleEyes','combined'):
        directory=caseRoot/branch;reportPath=directory/'report.json'
        if not reportPath.exists():branches[branch]={'status':'pendingBranchReport'};continue
        report=read(reportPath);pins[str(reportPath)]=sha(reportPath)
        if not (directory/'eyeEvidence.json').exists():
            branches[branch]=dict(status='noEyeEvidence',branchStatus=report['status']);continue
        for filename in ('eyeEvidence.json','eyeEvidence.npz','fullBeforeEyes.npz','fullGeometry.npz'):
            path=directory/filename;pins[str(path)]=sha(path)
            if pins[str(path)]!=report['outputHashes'][filename]:raise ValueError(f'Frozen branch evidence changed: {path}')
        index=read(directory/'eyeEvidence.json')
        with np.load(directory/'eyeEvidence.npz',allow_pickle=False) as arrays:
            before=loadTree(index['beforeContours'],arrays)
            after=loadTree(index['afterContours'],arrays)
            solver=loadTree(index.get('solverContours',index['afterContours']),arrays)
        before={name:maskMetrics(before['eyes'][name],photo[name],originalPixelScale) for name in NAMES}
        after={name:maskMetrics(after['eyes'][name],photo[name],originalPixelScale) for name in NAMES}
        solver={name:maskMetrics(solver['eyes'][name],photo[name],originalPixelScale) for name in NAMES}
        with np.load(directory/'fullBeforeEyes.npz',allow_pickle=False) as arrays:original=arrays['vertices']
        with np.load(directory/'fullGeometry.npz',allow_pickle=False) as arrays:final=arrays['vertices']
        proposals=[]
        for row in index['candidateProposals']:
            attempt=row.get('solverAttempt',{})
            proposals.append(dict(round=row['round'],scale=row['scale'],strictAccepted=row['strict']['accepted'],
                strictReasons=row['strict']['reasons'],strictFullAssembly=row['strict'].get('fullAssembly'),
                originalGeometryAccepted=row['internalGeometry']['accepted'],
                apertureAccepted=attempt.get('apertureGate'),apertureMetrics=attempt.get('apertureMetrics'),
                eyeContactAccepted=attempt.get('eyeContact',{}).get('accepted'),selectedByFastRound=row.get('selectedByFastRound',False),
                rawFullApertureRemeasured='apertureMetrics' in attempt))
        branches[branch]=dict(status='savedEvidenceSummarized',eyeEffectAccepted=report['eyeEffectAccepted'],
            geometrySafetyAccepted=report['geometrySafetyAccepted'],finalCandidateAccepted=report['finalCandidateAccepted'],
            eyeStatus=report['eyeRefinement']['status'],noOp=bool(np.array_equal(original,final)),
            changedVertexCount=int(np.any(original!=final,axis=1).sum()),
            maximumDisplacementCanonical=float(np.linalg.norm(final-original,axis=1).max()),
            beforeLocalEyeStage=before,afterSelectedEyeStage=after,solverFinalIterate=solver,
            localStageEffect=compare(before,after),proposals=proposals,
            fullPairGateAgainstOriginal=report['newFullPairsAgainstOriginalPureDirect'],
            fullPairGateAgainstCurrent=report['newFullPairsAgainstCurrentBounded'],
            finalStrict=report['eyeRefinement'].get('strict'),
            preEyeScope='Own branch after shared mouth/crease/jaw/assembly with old six-point eye stages disabled; not the factorial baseline')
    baselinePath=OUT/caseId/'factorialBaselineMasks.json'
    baseline=read(baselinePath) if baselinePath.exists() else {
        'status':'notMeasuredInSavedEyeEvidence',
        'reason':'The baseline branch retains old six-point stages and has no saved complete-aperture mask. It cannot be substituted with visibleEyes preEye; local ROI rasterization waits for a CPU slot.'}
    for path,expected in baseline.get('sourceHashes',{}).items():
        if sha(path)!=expected:raise ValueError(f'Control measurement source changed: {path}')
    if baselinePath.exists():pins[str(baselinePath)]=sha(baselinePath)
    for branch,control in (('visibleEyes','baseline'),('combined','sourcePrior')):
        row=branches.get(branch,{})
        measuredControl=baseline.get('branches',{}).get(control,{})
        if row.get('status')=='savedEvidenceSummarized' and 'eyes' in measuredControl:
            row['factorialControlBranch']=control
            row['factorialControlToFinalEffect']=compare(measuredControl['eyes'],row['afterSelectedEyeStage'])
    result=dict(caseId=caseId,referenceViewIndex=reference,status='cachedEvidenceSummary',branches=branches,
        factorialBaseline=baseline,sourceHashes=pins,noNewRasterization=True,noFitting=True,noNewAcceptanceThresholds=True,
        limitations=['These are automatically observed photo contours, not manually verified anatomy.',
            'All full aperture fragments are retained; extraction status and numerical discrepancy are separate.',
            'No-op means eye module returned its own pre-eye input. Disabling old eye stages can still change the factorial branch.',
            'Unmeasured rejected raw proposals are retained in source evidence; they are not assigned fabricated photo scores.',
            'Appearance has not been rebaked, and whole-pipeline acceptance remains independent.'])
    write(OUT/caseId/'report.json',result)
    return result


def measureControls(caseId):
    """Optional local ROI raster only; run after an explicit free CPU slot."""
    completedPath=OUT/caseId/'factorialBaselineMasks.json'
    if completedPath.exists():
        completed=read(completedPath)
        if (completed.get('status')=='measuredFinalFactorialControls'
                and all(row.get('status')=='measuredExactFinalCanonicalArrays' for row in completed.get('branches',{}).values())
                and set(completed.get('branches',{}))=={'baseline','sourcePrior'}):
            for path,expected in completed['sourceHashes'].items():
                if sha(path)!=expected:raise ValueError('Previously measured control identity changed')
            return
    sys.path.insert(0,str(ROOT/'src'))
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, extractVisibleEyeContours
    context=read(RUN/caseId/'inputContext.json')
    reference=context['referenceSelection']['referenceViewIndex']
    reconstruction=Path(context['entry']['reconstruction'])
    camera=read(reconstruction/'cameras.json')[reference]
    inputs=read(reconstruction/'inputs.json');shape=tuple(inputs[reference]['workingSize'][::-1])
    pixelScale=float(np.asarray(inputs[reference]['inputToWorking'])[0,0])
    densePath=ROOT/'runs/shapePipeline20261024'/caseId/'observations/observations.npz'
    denseReport=read(densePath.with_suffix('.json'))
    if sha(densePath)!=denseReport['observationSha256'] or reference!=denseReport['referenceViewIndex']:
        raise ValueError('Control photo observation identity differs')
    with np.load(densePath,allow_pickle=False) as arrays:
        curves=arrays['eyelidCurvesWorking']
        if not arrays['acceptedEyeMask'].all():raise ValueError('Rejected shared photo observations')
    photo={name:{part:curves[i,j] for j,part in enumerate(('lower','upper'))} for i,name in enumerate(NAMES)}
    template=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
    with np.load(template/'template.npz',allow_pickle=False) as arrays:mapping=arrays['nicpToFullIds']
    components=read(template/'components.json')
    corners=[row for row in components if row['name'] in ('male00SG.component001','male00SG.component002')]
    if len(corners)!=2 or any(row['materialId']!=0 or len(row['faceIndices'])!=16 for row in corners):
        raise ValueError('Audited canthus template identities differ')
    cornerIds=np.concatenate([row['faceIndices'] for row in corners])
    output=dict(status='measuredFinalFactorialControls',branches={},sourceHashes={})
    for branch in ('baseline','sourcePrior'):
        directory=RUN/caseId/branch;reportPath=directory/'report.json'
        if not reportPath.exists():output['branches'][branch]={'status':'pending'};continue
        report=read(reportPath)
        if report['status']!='geometryDiagnosticComplete':
            output['branches'][branch]={'status':report['status']};continue
        archive=Path(report['asset']['archivePath']);fullPath=directory/'fullGeometry.npz'
        for path in (reportPath,archive,fullPath):output['sourceHashes'][str(path)]=sha(path)
        if sha(fullPath)!=report['outputHashes']['fullGeometry.npz']:raise ValueError('Final control geometry changed')
        scene=loadLegacyEyeBindingInputs(archive,mapping)
        with np.load(fullPath,allow_pickle=False) as arrays:
            np.testing.assert_array_equal(scene['faces'],arrays['faces'])
            scene['vertices']=arrays['vertices'].copy()
        measured=extractVisibleEyeContours(**scene,camera=camera,photoEyes=photo,imageShape=shape,
            apertureAccessoryMode='innerCornerHeadAccessories',apertureAccessoryFaceIds=cornerIds,
            minimumEyeRasterWidth=256,maximumRoiSize=768)
        output['branches'][branch]=dict(status='measuredExactFinalCanonicalArrays',
            geometrySafetyAccepted=report['geometrySafetyAccepted'],
            eyes={name:maskMetrics(measured['eyes'][name],photo[name],pixelScale) for name in NAMES})
        masks={}
        for name,eye in measured['eyes'].items():
            if 'roiMasks' not in eye:continue
            masks[name+'/fullAperture']=eye['roiMasks']['eye']
            masks[name+'/origin']=np.asarray(eye['roi']['origin'])
            masks[name+'/scale']=np.asarray(eye['roi']['scale'])
        outputDir=OUT/caseId;outputDir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(outputDir/(branch+'-controlMasks.npz'),**masks)
    for path in (densePath,densePath.with_suffix('.json'),reconstruction/'cameras.json',reconstruction/'inputs.json',
                 template/'template.npz',template/'components.json',ROOT/'src/faceAppearance/visibleEyeContours.py'):
        output['sourceHashes'][str(path)]=sha(path)
    output.update(noGeometryFitting=True,originalCamerasPreserved=True,localRoiOnly=True,
        fullAssemblyIncluded=True,allVisibleFragmentsIncluded=True,
        note='Exact saved final canonical arrays with original materials/UV/alpha. Baseline still has old six-point stages; sourcePrior is a distinct factorial branch.')
    write(OUT/caseId/'factorialBaselineMasks.json',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',action='append')
    parser.add_argument('--measure-controls',action='store_true',help='Requires a free CPU slot: measure final baseline/sourcePrior local ROIs')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    config=read(RUN/'experimentPlan.json');cases=args.case or config['cases']
    if args.measure_controls:
        for caseId in cases:measureControls(caseId)
    results=[summarizeCase(caseId) for caseId in cases]
    summary=dict(status='partialOrCompleteSavedEyeEvidenceSummary',cases=results,
        planSha256=sha(RUN/'experimentPlan.json'),scriptSha256=sha(__file__),
        scope='Saved eye masks plus separately identified optional local ROI control measurement; no inference, fitting, client asset output or safety gate changes')
    complete=[branch for case in results for branch in case.get('branches',{}).values() if branch.get('status')=='savedEvidenceSummarized']
    summary['completedEyeBranchCounts']=dict(branches=len(complete),localEyeEffectAccepted=sum(bool(row['eyeEffectAccepted']) for row in complete),
        localEyeNoOp=sum(row['noOp'] for row in complete),wholeGeometrySafetyAccepted=sum(row['geometrySafetyAccepted'] for row in complete),
        fullSelectedEyeExtractionsAccepted=sum(eye['extractionStatus']=='accepted' for row in complete for eye in row['afterSelectedEyeStage'].values()))
    comparisons={}
    for branch in ('visibleEyes','combined'):
        rows=[]
        for case in results:
            value=case.get('branches',{}).get(branch,{})
            rows.extend(row for row in value.get('factorialControlToFinalEffect',{}).values() if row.get('status')=='comparableFrozenPhotoAndCamera')
        if rows:
            iou=[row['intersectionOverUnionDifference'] for row in rows]
            error=[row['boundaryMeanOriginalPixelsDifference'] for row in rows]
            comparisons[branch]=dict(measuredEyes=len(rows),iouImproved=sum(value>0 for value in iou),iouWorsened=sum(value<0 for value in iou),
                boundaryMeanImproved=sum(value<0 for value in error),boundaryMeanWorsened=sum(value>0 for value in error),
                iouDifferenceRange=[min(iou),max(iou)],boundaryMeanOriginalPixelsDifferenceRange=[min(error),max(error)],
                interpretation='Descriptive comparison only, not a new acceptance threshold. Includes replacement of old six-point stages and the resulting reassembly; the local eye solver effect is separately reported.')
    summary['factorialComparisonCounts']=comparisons
    summary['sourceAndEvidenceHashesVerified']=True
    summary['hashVerificationScope']='Current dense-observation/reference-camera/reference-photo identities and saved completed eye evidence hashes checked against branch report; final control geometry hashes checked against branch reports. This is not the separate complete pipeline/implementation audit.'
    write(OUT/'summary.json',summary)
    for row in results:
        print(json.dumps({'caseId':row['caseId'],'branches':{key:{name:value.get(name) for name in ('status','eyeStatus','eyeEffectAccepted','noOp')}
            for key,value in row.get('branches',{}).items()}}),flush=True)


if __name__=='__main__':main()
