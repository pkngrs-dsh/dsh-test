"""Read-only semantic-eye locality and complete-aperture diagnostics.

Default mode reads saved states only. --measure-apertures additionally performs
local ROI measurement using frozen cameras, and must wait for a free CPU slot.
Neither path refits geometry, changes acceptance thresholds or exports assets.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from summarizeFeatureVisibleEyes import maskMetrics,compare,write

RUN=ROOT/'runs/protectedFeature20261024'
OUT=ROOT/'reports/protectedFeature20261024/eyeAudit'
TEMPLATE=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
BRANCHES=('baseline','sourceProtected','localEyes','combined')
PAIRS={'localEyes':'baseline','combined':'sourceProtected'}
EYE_STAGES=('imageLeftEye','imageRightEye')
EYE_NAMES=('imageLeft','imageRight')
PLAN_SHA='6119af96fe2b269e15f33a45b89514887378ed0377fc4e3995a587975696f4ce'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as data:return {key:data[key].copy() for key in data.files}


def fitMetrics(vertices,evidence,camera):
    indices=np.asarray(evidence['vertexIndices'],np.int64)
    weights=np.asarray(evidence['barycentricWeights'],float)
    samples=(np.asarray(vertices)[indices]*weights[...,None]).sum(1)
    cameraPoints=samples@np.asarray(camera['rotation']).T+np.asarray(camera['translation'])
    homogeneous=cameraPoints@np.asarray(camera['intrinsics']).T
    if not np.isfinite(homogeneous).all() or not (cameraPoints[:,2]>0).all():raise ValueError('Invalid semantic projection')
    pixels=homogeneous[:,:2]/homogeneous[:,2,None]
    errors=np.linalg.norm(pixels-np.asarray(evidence['targetPixels']),axis=1)
    return dict(ibugIds=evidence['ibugIds'],pointCount=len(errors),errorsWorkingPixels=errors,
        meanWorkingPixels=float(errors.mean()),maximumWorkingPixels=float(errors.max()))


def displacementMetrics(before,after,mask,unit):
    difference=np.asarray(after)-np.asarray(before)
    lengths=np.linalg.norm(difference[mask],axis=1)/unit
    return dict(vertexCount=int(mask.sum()),changedVertexCount=int(np.any(difference[mask]!=0,axis=1).sum()),
        maximumMm=float(lengths.max(initial=0)),meanMm=float(lengths.mean()) if len(lengths) else 0.,
        exactUnchanged=bool(not np.any(difference[mask]!=0)))


def basisDiagnostics(stage,evidence,staticIds,unit):
    solver=stage.get('solver')
    if solver is None:return {'status':'solverUnavailable'}
    fixed=(np.asarray(stage['localSupport']['fixedVertexIds'],int) if 'localSupport' in stage else staticIds)
    vertices=np.asarray(evidence['before']);indices=np.asarray(evidence['vertexIndices'],int)
    points=(vertices[indices]*np.asarray(evidence['barycentricWeights'])[...,None]).sum(1)
    groups=solver['inputTargetGate']['groups']
    if len(groups)!=1:raise ValueError('Expected the original single eye constraint')
    controls=np.unique(points[groups[0]['keptIndices']],axis=0)
    if len(controls)>12:raise ValueError('Unexpected semantic eye control selection')
    distances=np.linalg.norm(controls[:,None]-vertices[fixed][None],axis=2)
    nearest=float(distances.min()) if distances.size else None
    boundary=solver['fixedBoundary'];requested=boundary['requestedSupportRadius']
    expected=min(requested,nearest*(1-1e-12)) if nearest is not None else requested
    reproduced=bool(np.isclose(expected,boundary['effectiveSupportRadius'],atol=1e-14,rtol=1e-12)
        and (nearest is None or np.isclose(nearest,boundary['nearestControlToFixedDistance'],atol=1e-14,rtol=1e-12)))
    return dict(status='reconstructedCommonRadiusFromOriginalControlsAndFixedVertices',reproduced=reproduced,
        requestedRadiusMm=requested/unit,effectiveRadiusMm=boundary['effectiveSupportRadius']/unit,
        effectiveOverRequestedRadius=boundary['effectiveSupportRadius']/requested,
        nearestControlToFixedDistanceMm=None if nearest is None else nearest/unit,
        fixedVertexCount=len(fixed),activeVertexCount=solver['activeVertexCount'],controlCount=len(controls),
        safety=solver['safety'],rejections=solver['rejections'],
        iterationsCompleted=solver['iterationsCompleted'],bestIteration=solver['bestIteration'],
        bestRegularizedObjective=solver['bestRegularizedObjective'],
        solverInitialWeightedMeanErrorPixels=solver['initialMeanErrorPixels'],
        solverFinalWeightedMeanErrorPixels=solver['finalMeanErrorPixels'],
        scope='The nearest distance over every retained control/fixed pair caps one common compact-field radius. Solver error is weighted; independent corresponding-stage mean is unweighted. Objective is optimization evidence, not anatomical ground truth.')


def caseInputs(caseId):
    directory=RUN/caseId
    if not (directory/'inputContext.json').exists():return None
    context=read(directory/'inputContext.json');entry=context['entry'];reference=context['referenceSelection']['referenceViewIndex']
    reconstruction=Path(entry['reconstruction']);cameraPath=reconstruction/'cameras.json';inputPath=reconstruction/'inputs.json'
    photoPath=Path(entry['workingPhotos'])/f'input{reference:02d}.png'
    densePath=ROOT/'runs/shapePipeline20261024'/caseId/'observations/observations.npz';denseReportPath=densePath.with_suffix('.json')
    denseReport=read(denseReportPath);dense=load(densePath)
    if sha(densePath)!=denseReport['observationSha256'] or reference!=denseReport['referenceViewIndex'] or not dense['acceptedEyeMask'].all():
        raise ValueError('Current dense photo identity/quality differs')
    if sha(photoPath)!=denseReport['sourcePhotoSha256'] or sha(cameraPath)!=denseReport['sourceHashes'][str(cameraPath)]:
        raise ValueError('Reference photo or camera changed')
    original=load(entry['preparedGeometry']);camera=read(cameraPath)[reference];inputs=read(inputPath)
    curves=dense['eyelidCurvesWorking']
    photo={name:{part:curves[i,j] for j,part in enumerate(('lower','upper'))} for i,name in enumerate(EYE_NAMES)}
    pins={str(path):sha(path) for path in (directory/'inputContext.json',cameraPath,inputPath,photoPath,densePath,denseReportPath,Path(entry['preparedGeometry']))}
    for path,actual in pins.items():
        if path==str(directory/'inputContext.json'):continue
        if context['sourceHashes'].get(path)!=actual:raise ValueError(f'Frozen context identity differs: {path}')
    return dict(context=context,reference=reference,camera=camera,inputs=inputs,photo=photo,original=original,
        sourceHashes=pins,shape=tuple(inputs[reference]['workingSize'][::-1]),
        originalPixelScale=float(np.asarray(inputs[reference]['inputToWorking'])[0,0]))


def auditCase(caseId,measureApertures=False):
    state=caseInputs(caseId)
    if state is None:return dict(caseId=caseId,status='pending')
    directory=RUN/caseId;unit=float(state['original']['unit']);skinIds=state['original']['skinIds'];records={};full={};pins=state['sourceHashes']
    for branch in BRANCHES:
        reportPath=directory/branch/'report.json'
        if not reportPath.exists():continue
        report=read(reportPath);pins[str(reportPath)]=sha(reportPath)
        if report['status']!='geometryDiagnosticComplete':records[branch]=report;continue
        for filename in ('fullGeometry.npz','states.npz','refinementEvidence.json','refinementEvidence.npz'):
            path=directory/branch/filename
            if sha(path)!=report['outputHashes'][filename]:raise ValueError(f'Frozen completed output changed: {path}')
            pins[str(path)]=sha(path)
        full[branch]=load(directory/branch/'fullGeometry.npz')
        records[branch]=report
    if 'baseline' not in full:return dict(caseId=caseId,status='waitingBaseline')
    nonSkin=np.ones(len(full['baseline']['vertices']),bool);nonSkin[skinIds]=False
    result=dict(caseId=caseId,referenceViewIndex=state['reference'],branches={},comparisons={},sourceHashes=pins)
    for branch,report in records.items():
        if branch not in full:result['branches'][branch]={'status':report['status']};continue
        if not np.array_equal(full[branch]['faces'],full['baseline']['faces']):raise ValueError('Full original face identity changed')
        stages={row['name']:row for row in report['geometryRefinement']['stagedLandmarks']['stages']}
        rows={}
        for name in EYE_STAGES:
            stage=stages[name]
            if 'stageEvidence' not in stage:rows[name]={'status':stage['status'],'measurement':'missingStageEvidence'};continue
            evidence=stage['stageEvidence'];before=np.asarray(evidence['before']);after=np.asarray(evidence['acceptedAfter'])
            row=dict(status=stage['status'],selectedScale=stage['scale'],
                before=fitMetrics(before,evidence,state['camera']),after=fitMetrics(after,evidence,state['camera']),
                allStageMotion=displacementMetrics(before,after,np.ones(len(before),bool),unit),
                basis=basisDiagnostics(stage,evidence,state['original']['staticIds'],unit),
                attempts=[{key:attempt[key] for key in ('scale','newIntersectionPairs','sideGate','previousStageGate') if key in attempt}
                    | {key:attempt[key].get('accepted') for key in ('geometry','localSupportGeometry','localEyeAlignment') if key in attempt}
                    for attempt in stage.get('attempts',[])])
            if 'localSupport' in stage:
                support=stage['localSupport'];fixed=np.asarray(support['fixedMask'],bool)
                row.update(outsideLocalBandMotion=displacementMetrics(before,after,fixed,unit),
                    support=support['diagnostics'],fixedVertexIds=np.flatnonzero(fixed),seedVertexIds=support['seedVertexIds'])
            rows[name]=row
        result['branches'][branch]=dict(status='savedStateAuditComplete',stages=rows,
            allNonSkinMotion=displacementMetrics(full['baseline']['vertices'],full[branch]['vertices'],nonSkin,unit),
            reportedGeometrySafetyAccepted=report['geometrySafetyAccepted'],
            reportedNoNewRegressionAgainstPairedBaseline=report['noNewRegressionAgainstPairedBaseline'],
            finalCandidateAccepted=report['finalCandidateAccepted'],
            finalAdaptiveAssemblyValidated=report['finalAdaptiveAssemblyValidated'],
            newFullPairsAgainstPairedBaseline=report['newFullPairsAgainstPairedBaseline'])
    plan=read(RUN/'experimentPlan.json');tolerance=plan['localEyes']['pairedMeanErrorToleranceImageScale']*min(state['shape'])/512.
    for localName,globalName in PAIRS.items():
        if localName not in full or globalName not in full:continue
        localStages={row['name']:row for row in records[localName]['geometryRefinement']['stagedLandmarks']['stages']}
        globalStages={row['name']:row for row in records[globalName]['geometryRefinement']['stagedLandmarks']['stages']}
        paired={}
        for name in EYE_STAGES:
            local,globalStage=localStages[name],globalStages[name]
            if 'stageEvidence' not in local or 'stageEvidence' not in globalStage or 'localSupport' not in local:
                paired[name]={'status':'missingComparableEvidence'};continue
            le,ge=local['stageEvidence'],globalStage['stageEvidence']
            same=all(np.array_equal(np.asarray(le[key]),np.asarray(ge[key])) for key in
                ('ibugIds','vertexIndices','barycentricWeights','targetPixels','weights','confidence'))
            lm=fitMetrics(le['acceptedAfter'],le,state['camera']);gm=fitMetrics(ge['acceptedAfter'],le,state['camera'])
            fixed=np.asarray(local['localSupport']['fixedMask'],bool)
            difference=lm['meanWorkingPixels']-gm['meanWorkingPixels']
            paired[name]=dict(status='measuredCorrespondingEyeStageEnds',targetsAndBindingsExactlyEqual=same,
                local=lm,globalOnSameTargets=gm,meanChangeWorkingPixels=difference,
                frozenPairedToleranceWorkingPixels=tolerance,pairedAlignmentAccepted=bool(same and difference<=tolerance),
                localOutsideBandMotion=displacementMetrics(le['before'],le['acceptedAfter'],fixed,unit),
                globalOutsideSameBandMotion=displacementMetrics(ge['before'],ge['acceptedAfter'],fixed,unit),
                scope='Per-eye stage only. Outside the topology band is a distant-region proxy, not a manually labeled brow/cheek anatomy mask.')
        result['comparisons'][localName]=dict(globalBranch=globalName,eyes=paired)
        if all('localSupport' in localStages[name] and 'stageEvidence' in localStages[name] for name in EYE_STAGES):
            outside=np.logical_and.reduce([np.asarray(localStages[name]['localSupport']['fixedMask'],bool) for name in EYE_STAGES])
            reference=np.asarray(localStages[EYE_STAGES[0]]['stageEvidence']['frozenSupportReferenceVertices'])
            result['comparisons'][localName]['finalOutsideCombinedEyeBand']=dict(
                localFromStagedInput=displacementMetrics(reference,full[localName]['vertices'][skinIds],outside,unit),
                globalFromSameStagedInput=displacementMetrics(reference,full[globalName]['vertices'][skinIds],outside,unit),
                localVersusGlobalFinal=displacementMetrics(full[globalName]['vertices'][skinIds],full[localName]['vertices'][skinIds],outside,unit),
                scope='Includes common mouth, crease and jaw processing after the staged input. This is not isolated eye-stage movement; the exact per-stage measurement is reported separately.')
    aperturePath=OUT/caseId/'apertures.json'
    if measureApertures:
        measureCaseApertures(caseId,state,records,full,aperturePath)
    if aperturePath.exists():
        aperture=read(aperturePath)
        for path,expected in aperture['sourceHashes'].items():
            if sha(path)!=expected:raise ValueError('Frozen aperture measurement source changed')
        for branch,value in aperture['branches'].items():
            masks=load(aperturePath.parent/(branch+'-apertureMasks.npz'))
            for name,metric in value['eyes'].items():
                if metric['status']!='measuredFromSavedFullApertureMask':continue
                mask=np.asarray(masks[name+'/fullAperture'],bool)
                if hashlib.sha256(mask.tobytes()).hexdigest()!=metric['maskSha256']:
                    raise ValueError('Saved full aperture mask changed')
                if not np.array_equal(masks[name+'/origin'],metric['roi']['origin']) or float(masks[name+'/scale'])!=metric['roi']['scale']:
                    raise ValueError('Saved aperture frame changed')
        result['apertures']=aperture
        result['apertureMasksAndFramesIndependentlyVerified']=True
        for local,globalBranch in PAIRS.items():
            if local in aperture['branches'] and globalBranch in aperture['branches']:
                result['comparisons'].setdefault(local,{'globalBranch':globalBranch})['apertureDiagnosticDifference']=compare(
                    aperture['branches'][globalBranch]['eyes'],aperture['branches'][local]['eyes'])
    else:result['apertures']={'status':'notMeasured; waitingForFreeCpuSlot'}
    result.update(status='completedSavedStateAudit',sourceAndEvidenceHashesVerified=True,
        noNewAcceptanceThresholds=True,noGeometryFitting=True,
        limitations=['Six-point observations and dense contours are automatic proxies, not manually verified anatomy.',
            'Semantic alignment uses the corresponding eye-stage endpoints, not the later whole-face endpoint.',
            'Aperture metrics, when present, describe the exact saved final assembly; they do not replace the frozen semantic or collision gates.',
            'Fixed baseline accessories isolate coupling. Fresh adaptive assembly has not been certified.',
            'The separate independent validator owns complete mesh/collision acceptance.'])
    write(OUT/caseId/'report.json',result)
    return result


def measureCaseApertures(caseId,state,records,full,path):
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs,extractVisibleEyeContours
    prior=read(path) if path.exists() else {'branches':{},'sourceHashes':{}}
    for source,expected in prior['sourceHashes'].items():
        if sha(source)!=expected:raise ValueError('Existing measured aperture source changed')
    components=read(TEMPLATE/'components.json')
    corners=[row for row in components if row['name'] in ('male00SG.component001','male00SG.component002')]
    if len(corners)!=2 or any(row['materialId']!=0 or len(row['faceIndices'])!=16 for row in corners):raise ValueError('Explicit canthus identities changed')
    cornerIds=np.concatenate([row['faceIndices'] for row in corners])
    archive=Path(records['baseline']['asset']['archivePath'])
    archiveKey=str(archive.relative_to(RUN/caseId/'baseline'))
    if sha(archive)!=records['baseline']['outputHashes'][archiveKey]:raise ValueError('Baseline assembly archive identity changed')
    scene=loadLegacyEyeBindingInputs(archive,state['original']['skinIds'])
    for branch,mesh in full.items():
        if branch in prior['branches']:continue
        scene['vertices']=mesh['vertices'];np.testing.assert_array_equal(scene['faces'],mesh['faces'])
        measured=extractVisibleEyeContours(**scene,camera=state['camera'],photoEyes=state['photo'],imageShape=state['shape'],
            apertureAccessoryMode='innerCornerHeadAccessories',apertureAccessoryFaceIds=cornerIds,
            minimumEyeRasterWidth=256,maximumRoiSize=768)
        prior['branches'][branch]=dict(eyes={name:maskMetrics(measured['eyes'][name],state['photo'][name],state['originalPixelScale']) for name in EYE_NAMES})
        masks={}
        for name,eye in measured['eyes'].items():
            if 'roiMasks' not in eye:continue
            masks[name+'/fullAperture']=eye['roiMasks']['eye'];masks[name+'/origin']=np.asarray(eye['roi']['origin']);masks[name+'/scale']=np.asarray(eye['roi']['scale'])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.parent/(branch+'-apertureMasks.npz'),**masks)
        fullPath=RUN/caseId/branch/'fullGeometry.npz';prior['sourceHashes'][str(fullPath)]=sha(fullPath)
    for source in (archive,TEMPLATE/'components.json',ROOT/'src/faceAppearance/visibleEyeContours.py'):
        prior['sourceHashes'][str(source)]=sha(source)
    prior['sourceHashes'].update(state['sourceHashes']);prior.update(status='measuredFrozenFinalApertures',diagnosticOnly=True,
        completeApertureIncludesExplicitCanthi=True,allVisibleFragmentsRetained=True,geometryOrCamerasRefitted=False)
    write(path,prior)


def aggregate(results):
    rows=[];apertureRows=[];apertureObservations=[]
    for result in results:
        for branch,value in result.get('apertures',{}).get('branches',{}).items():
            for eye,measurement in value['eyes'].items():
                apertureObservations.append(dict(caseId=result['caseId'],branch=branch,eye=eye,
                    status=measurement['status'],extractionStatus=measurement.get('extractionStatus'),
                    extractionReasons=measurement.get('extractionReasons',[])))
        for local,pair in result.get('comparisons',{}).items():
            for eye,value in pair.get('eyes',{}).items():
                if value['status']!='measuredCorrespondingEyeStageEnds':continue
                rows.append(dict(caseId=result['caseId'],localBranch=local,pairedGlobal=pair['globalBranch'],eye=eye,
                    localMeanPixels=value['local']['meanWorkingPixels'],globalMeanPixels=value['globalOnSameTargets']['meanWorkingPixels'],
                    meanChangePixels=value['meanChangeWorkingPixels'],pairedAlignmentAccepted=value['pairedAlignmentAccepted'],
                    targetsAndBindingsExactlyEqual=value['targetsAndBindingsExactlyEqual'],
                    localOutsideBandExact=value['localOutsideBandMotion']['exactUnchanged'],
                    globalOutsideBandMaximumMm=value['globalOutsideSameBandMotion']['maximumMm']))
            for eye,value in pair.get('apertureDiagnosticDifference',{}).items():
                if value['status']!='comparableFrozenPhotoAndCamera':continue
                apertureRows.append(dict(caseId=result['caseId'],localBranch=local,eye=eye,**value))
    completed=[branch for result in results for branch in result.get('branches',{}).values() if branch.get('status')=='savedStateAuditComplete']
    return dict(measuredSixPointEyeStagePairs=len(rows),pairedAlignmentPassed=sum(row['pairedAlignmentAccepted'] for row in rows),
        exactOriginalTargets=sum(row['targetsAndBindingsExactlyEqual'] for row in rows),
        exactLocalOutsideBand=sum(row['localOutsideBandExact'] for row in rows),
        completedBranches=len(completed),exactCommonAttachments=sum(row['allNonSkinMotion']['exactUnchanged'] for row in completed),
        sixPointRows=rows,apertureRows=apertureRows,apertureObservations=apertureObservations,
        acceptedApertureExtractionCount=sum(row['extractionStatus']=='accepted' for row in apertureObservations),
        rejectedOrMissingApertureExtractionCount=sum(row['extractionStatus']!='accepted' for row in apertureObservations),
        apertureMetricScope='All full-visible masks remain diagnostic, including those whose visible-contour support rejects; quality statuses and reasons are retained, with no selection by metric outcome.',
        apertureIouImproved=sum(row['intersectionOverUnionDifference']>0 for row in apertureRows),
        apertureIouWorsened=sum(row['intersectionOverUnionDifference']<0 for row in apertureRows),
        apertureIouEqual=sum(row['intersectionOverUnionDifference']==0 for row in apertureRows),
        apertureBoundaryMeanImproved=sum(row['boundaryMeanOriginalPixelsDifference']<0 for row in apertureRows),
        apertureBoundaryMeanWorsened=sum(row['boundaryMeanOriginalPixelsDifference']>0 for row in apertureRows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',action='append')
    parser.add_argument('--measure-apertures',action='store_true',help='Local ROI rasterization; wait for an explicit free CPU slot')
    args=parser.parse_args();cv2.setNumThreads(1)
    if sha(RUN/'experimentPlan.json')!=PLAN_SHA:raise ValueError('Prospectively frozen plan changed')
    cases=args.case or read(RUN/'experimentPlan.json')['cases']
    results=[auditCase(case,args.measure_apertures) for case in cases]
    summary=dict(cases=results,aggregate=aggregate(results),planSha256=sha(RUN/'experimentPlan.json'),scriptSha256=sha(__file__),
        scope='Read-only saved-state and optional local ROI audit; no source/runner/candidate mutation')
    write(OUT/'summary.json',summary)
    for result in results:
        print(json.dumps({'caseId':result['caseId'],'status':result['status'],'comparisons':{
            branch:{name:{key:row.get(key) for key in ('meanChangeWorkingPixels','pairedAlignmentAccepted')} for name,row in value.get('eyes',{}).items()}
            for branch,value in result.get('comparisons',{}).items()}}),flush=True)


if __name__=='__main__':main()
