"""Frozen seven-case geometry factorial; failures are retained as diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from PIL import Image
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.clientReferenceView import selectClientReference
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyGeometryVariant import exportLegacyGeometryVariant
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs
from runShapePipelineExperiment import finalGeometry, saveArrays
from experimentShapePreservation import (buildShapeObjective,buildGuardSpec,normalizedShapePolicy,
    shapeSafetyMetrics,baselinePreflight)
RUN=ROOT/'runs/protectedFeature20261024'
OLD=ROOT/'runs/shapePipeline20261024'
PRIOR=ROOT/'runs/shapePreservation20261024'
PLAN_SHA='6119af96fe2b269e15f33a45b89514887378ed0377fc4e3995a587975696f4ce'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(hashes):
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError(f'Frozen input changed: {path}')
def saveEvidence(path,value):
    """Store nested dense arrays losslessly, with a reviewable JSON index."""
    saved={}
    def convert(item,key='root'):
        if isinstance(item,np.ndarray):
            saved[key]=item
            return {'arrayKey':key,'shape':list(item.shape),'dtype':str(item.dtype)}
        if sparse.issparse(item):return {'sparseStoredSeparately':True,'shape':list(item.shape)}
        if isinstance(item,dict):return {str(k):convert(v,key+'/'+str(k)) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [convert(v,key+'/'+str(i)) for i,v in enumerate(item)]
        return item
    metadata=convert(value);np.savez_compressed(path.with_suffix('.npz'),**saved)
    writeJson(path.with_suffix('.json'),metadata)

def cavityBenefit(objective,vertices):
    raw=objective['raw'];unit=float(raw['unit']);beforeSquared=afterSquared=mass=0.
    for name,operator in objective['depthOperators'].items():
        before=operator@raw['pureDirect'].ravel()/unit;after=operator@vertices.ravel()/unit
        rest=operator@raw['alignedRest'].ravel()/unit
        weight=raw[name+'_priorWeights']*(rest>0)
        beforeSquared+=float(np.sum(weight*np.maximum(rest-before,0)**2))
        afterSquared+=float(np.sum(weight*np.maximum(rest-after,0)**2));mass+=float(weight.sum())
    defined=mass>0 and beforeSquared>0
    improvement=1-np.sqrt(afterSquared/beforeSquared) if defined else None
    return {'defined':defined,'improvementFraction':improvement,
        'accepted':bool(defined and improvement>=.15-1e-8)}
def plan():
    if sha(RUN/'experimentPlan.json')!=PLAN_SHA:raise ValueError('Prospective plan changed')
    return read(RUN/'experimentPlan.json')
def historicalHashes(hashes):
    """Validate historical code at its immutable copy, current data in place."""
    frozen=read(ROOT/'reports/shapePipeline20261024/frozenImplementation/index.json')['files']
    result={}
    for path,expected in hashes.items():
        if sha(path)==expected:result[path]=expected;continue
        item=frozen.get(path)
        if item is None or item['sha256']!=expected or sha(item['snapshotPath'])!=expected:
            raise ValueError(f'Historical data/code identity lost: {path}')
        result[item['snapshotPath']]=expected
    return result

def freeze():
    plan();destination=RUN/'implementation';destination.mkdir(exist_ok=False)
    paths=sorted((ROOT/'src/faceAppearance').rglob('*.py'))+[ROOT/'tools'/name for name in (
        'runProtectedFeatureExperiment.py','runShapePipelineExperiment.py','experimentShapePreservation.py',
        'experimentEarlyCavityRegistration.py','experimentNoseSurfaceRegistration.py')]
    hashes={str(p):sha(p) for p in paths};copies={}
    for p in paths:
        q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        copies[str(p)]={'snapshotPath':str(q),'sha256':hashes[str(p)]}
    writeJson(RUN/'implementationHashes.json',{'sourceHashes':hashes,'snapshots':copies,'planSha256':PLAN_SHA})
    print(json.dumps({'status':'implementationFrozen','files':len(hashes)}),flush=True)

def contextFor(caseId):
    config=plan();implementation=read(RUN/'implementationHashes.json');verify(implementation['sourceHashes'])
    split=read(PRIOR/'split.json')
    if sha(PRIOR/'split.json')!=config['parentSplitSha256']:raise ValueError('Split changed')
    entry=next(e for e in split['cases'] if e['caseId']==caseId)
    prep=read(entry['preparationReport']);parent=read(PRIOR/caseId/'boundedWeight001/report.json')
    hashes=historicalHashes({**split['commonSourceHashes'],**prep['sourceHashes'],**parent['sourceHashes']})
    hashes.update(implementation['sourceHashes'])
    paths=[Path(entry['preparedGeometry']),Path(entry['preparationReport']),PRIOR/caseId/'boundedWeight001/geometry.npz',
        RUN/'experimentPlan.json',RUN/'implementationHashes.json',Path(config['initialPolicy']),Path(config['initialSolverSettings'])]
    if (sha(entry['preparedGeometry'])!=entry['preparedGeometrySha256'] or
        sha(entry['preparationReport'])!=entry['preparationReportSha256'] or
        sha(paths[2])!=parent['geometrySha256'] or not parent['acceptedForInitialGeometryDiagnostic']):raise ValueError('Initial source identity/acceptance mismatch')
    raw=arrays(entry['preparedGeometry']);bounded=arrays(paths[2]);reconstruction=Path(entry['reconstruction'])
    for k in ('pureDirect','faces','alignedRest'):np.testing.assert_array_equal(raw[k],bounded[k])
    inputs=read(reconstruction/'inputs.json');cameras=read(reconstruction/'cameras.json')
    fan=arrays(reconstruction/'directClient/renderedJaw/photoObservations.npz')
    fan['diagnostics']=read(reconstruction/'directClient/renderedJaw/photoObservations.json')
    selection=selectClientReference(cameras,inputs,fan);reference=selection['referenceViewIndex']
    paths.extend(reconstruction/p for p in ('inputs.json','cameras.json','geometry.npz','delight.npz',
        'directClient/renderedJaw/photoObservations.npz','directClient/renderedJaw/photoObservations.json'))
    photos=[Path(entry['workingPhotos'])/f'input{i:02d}.png' for i in range(len(inputs))]
    paths.extend(photos)
    old=OLD/caseId
    cache=arrays(old/'geometry/photoObservations.npz');records=read(old/'geometry/photoObservations.json')['records']
    shared={'reports':records,'observations':[]}
    for i,path in enumerate(photos):
        rgb=np.asarray(Image.open(path).convert('RGB'))
        if list(rgb.shape[1::-1])!=inputs[i]['workingSize']:raise ValueError('Photo frame mismatch')
        record=records[i]
        if (record['viewIndex']!=i or record.get('source')!='providedWorkingPhotograph'
                or record.get('sourceSha256')!=sha(path)
                or bool(cache['accepted'][i])!=(record['status']=='detected')):
            raise ValueError('Cached 68-point photo provenance mismatch')
        if record.get('model'):
            model=record['model']
            if sha(model['weightPath'])!=model['weightSha256']:raise ValueError('Photo model identity changed')
            hashes[model['weightPath']]=model['weightSha256']
        shared['observations'].append({'pixels':cache['points'][i],'confidence':cache['confidence'][i],
            'workingRgb':rgb} if cache['accepted'][i] else None)
    denseReport=read(old/'observations/observations.json');dense=arrays(old/'observations/observations.npz')
    hashes.update(historicalHashes(denseReport['sourceHashes']))
    if (denseReport['referenceViewIndex']!=reference or denseReport['geometryUsedForInference'] is not False or
        denseReport['inferenceFrame']!='savedWorkingRgbPng' or denseReport['sourcePhotoSha256']!=sha(photos[reference]) or
        denseReport['observationSha256']!=sha(old/'observations/observations.npz') or
        denseReport['sourceFanSha256']!=sha(reconstruction/'directClient/renderedJaw/photoObservations.npz')):raise ValueError('Dense photo identity mismatch')
    variants=dense['cropVariantCurvesWorking'];variation=np.linalg.norm(variants-variants[:1],axis=-1).max(0)
    np.testing.assert_array_equal(variants[0],dense['eyelidCurvesWorking'])
    np.testing.assert_array_equal(variation,dense['maximumVariationWorkingPixels'])
    accepted=dense['pointValidMask'].all(axis=(1,2)) & (variation.max(axis=(1,2))<=3.)
    accepted &= np.asarray([item['valid'] for item in denseReport['quality']])
    np.testing.assert_array_equal(accepted,dense['acceptedEyeMask'])
    if variants.shape!=(5,2,2,12,2) or not accepted.all() or denseReport['status']!='accepted':raise ValueError('Dense eyes rejected')
    if denseReport['policy']['anchorOffsetsWorkingPixels']!=[[0,0],[2,0],[-2,0],[0,2],[0,-2]]:raise ValueError('Dense crop policy mismatch')
    paths.extend(old/p for p in ('geometry/photoObservations.npz','geometry/photoObservations.json',
        'observations/observations.npz','observations/observations.json'))
    for branch in ('pureDirect','bounded'):
        report=read(old/'geometry'/branch/'report.json');hashes.update(historicalHashes(report['sourceHashes']))
        for name,expected in report['outputHashes'].items():
            path=old/'geometry'/branch/name
            if sha(path)!=expected:raise ValueError('Old branch output changed')
            paths.append(path)
    hashes.update({str(p):sha(p) for p in paths});verify(hashes)
    root=RUN/caseId;root.mkdir(exist_ok=True)
    writeJson(root/'inputContext.json',{'entry':entry,'planSha256':PLAN_SHA,'referenceSelection':selection,
        'oldRoot':str(old),'sourceHashes':hashes,'observationReplay':'saved photo-only caches shared by every branch'})
    return dict(entry=entry,raw=raw,bounded=bounded,cameras=cameras,inputs=inputs,fan=fan,photos=photos,
        delight=arrays(reconstruction/'delight.npz'),source=arrays(reconstruction/'geometry.npz'),shared=shared,
        dense=dense,reference=reference,selection=selection,hashes=hashes,root=root)

def sourceInitial(context,assets):
    from faceAppearance.sourceShapePreservation import buildSourceShapePreservationPrior
    from faceAppearance.guardedMeshQuadratic import solveGuardedMeshQuadratic
    from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
    from faceAppearance.supportedShapeConstraints import buildSupportedShapeConstraintSpec,evaluateSupportedShapeConstraints
    directory=context['root']/'initial/sourceProtected';directory.mkdir(parents=True,exist_ok=False)
    raw=context['raw'];mapping=assets.directArrays;source=context['source'];config=plan()['sourceShape']
    np.testing.assert_array_equal(source['faces'],assets.arrays['sourceFaces'])
    q=np.einsum('ij,ijk->ik',mapping['sourceBarycentricWeights'],source['vertices'].astype(float)[source['faces'][mapping['sourceTriangleIds']]])
    np.testing.assert_allclose(q,raw['targetPoints'],rtol=0,atol=1e-14)
    length=config['lengthScaleMm']*float(raw['unit']);weight=10000.*length**2
    built=buildSourceShapePreservationPrior(raw['faces'],raw['sampleFaceIds'],raw['sampleBarycentric'],raw['targetPoints'],
        source['vertices'].astype(float),source['faces'],mapping['sourceTriangleIds'],mapping['sourceBarycentricWeights'],
        clientVertexCount=len(raw['pureDirect']),validMask=raw['validMask'],coverageMask=raw['validMask'],lengthScale=length,
        neighborhoodRadius=config['neighborhoodRadiusMm']*float(raw['unit']),maxNeighbors=config['maxNeighbors'],
        minNeighbors=config['minNeighbors'],weight=weight)
    prior=built['quadraticPrior'];a=prior['operator'];target=prior['targets'];weight=prior['weight']
    sparse.save_npz(directory/'sourceOperator.npz',a);np.savez_compressed(directory/'sourceTargets.npz',targets=target)
    sparse.save_npz(directory/'sampleMatrix.npz',built['sampleMatrix']);sparse.save_npz(directory/'differentialOperator.npz',built['differentialOperator'])
    saveEvidence(directory/'sourceEvidence',built);writeJson(directory/'sourcePrior.json',built['diagnostics'])
    objective=buildShapeObjective(raw,.01);policy=normalizedShapePolicy(read(PRIOR/'policy.json'));guard=buildGuardSpec(objective,policy)
    baseline=context['bounded']['vertices']
    support=buildSupportedShapeConstraintSpec(baseline,raw['faces'],built['sampleMatrix'][raw['validMask']],raw['staticIds'],
        delegatedVertexIds=guard['controlVertexIds'],protectedVertexIds=raw['eyeMouthIds'],lengthScale=float(raw['unit']),
        safetyReferenceVertices=raw['pureDirect'])
    saveEvidence(directory/'supportSpec',support)
    controls=np.union1d(guard['controlVertexIds'],support['controlVertexIds']);fixed=support['hardFixedVertexIds']
    np.savez_compressed(directory/'solveDomain.npz',controlVertexIds=controls,fixedVertexIds=fixed,startVertices=baseline)
    def constraints(vertices):
        left=evaluateMeshShapeConstraints(vertices,guard);right=evaluateSupportedShapeConstraints(vertices,support)
        return {'values':np.r_[left['values'],right['values']],'jacobian':sparse.vstack([left['jacobian'],right['jacobian']],format='csr')}
    h=(objective['H']+weight*(a.T@a)).tocsc();b=objective['b']+weight*(a.T@target)
    sparse.save_npz(directory/'normalMatrixH.npz',h);np.savez_compressed(directory/'normalRhs.npz',b=b)
    _,skinPairs,fullPairs=baselinePreflight(objective,policy)
    settings=read(plan()['initialSolverSettings'])['commonToAllCases'];settings={k:v for k,v in settings.items() if k!='weight'}
    start=perf_counter()
    result=solveGuardedMeshQuadratic(baseline,h,b,controls,fixed,constraints,lengthScale=float(raw['unit']),**settings)
    vertices=result['rawVertices'];np.savez_compressed(directory/'geometry.npz',vertices=vertices,faces=raw['faces'],delta=vertices-raw['pureDirect'])
    gate=shapeSafetyMetrics(objective['raw'],vertices,objective['depthOperators'],policy,skinPairs,fullPairs)
    fixedExact=bool(np.array_equal(vertices[fixed],baseline[fixed]))
    writeJson(directory/'report.json',{'caseId':context['entry']['caseId'],'solver':result['diagnostics'],'gate':gate,
        'accepted':bool(result['diagnostics']['solverAccepted'] and gate['accepted'] and fixedExact),'weight':weight,'seconds':perf_counter()-start,
        'support':support['diagnostics'],'hardFixedExactlyPreserved':fixedExact,
        'sourcePrior':built['diagnostics'],'sourceHashes':context['hashes'],'rawRejectedRetained':True,
        'differentialObjectiveBefore':float(np.square(a@baseline.ravel()-target).sum()),
        'differentialObjectiveAfter':float(np.square(a@vertices.ravel()-target).sum()),
        'differentialMetricIsOptimizationObjectiveNotGroundTruth':True,
        'outputHashes':{str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    print(json.dumps({'case':context['entry']['caseId'],'stage':'sourceInitial','accepted':gate['accepted'],'reasons':gate['reasons']}),flush=True)
    return vertices


def pairedEyeAudit(local,globalResult,camera,imageScale):
    from faceAppearance.camera import projectPoints
    localStages={x['name']:x for x in local['diagnostics']['stagedLandmarks']['stages']}
    globalStages={x['name']:x for x in globalResult['diagnostics']['stagedLandmarks']['stages']}
    result={}
    for name in ('imageLeftEye','imageRightEye'):
        ls=localStages[name];gs=globalStages[name]
        if 'stageEvidence' not in ls or 'stageEvidence' not in gs:
            result[name]={'accepted':False,'reason':'missingComparableStageEvidence'};continue
        left={k:np.asarray(v) for k,v in ls['stageEvidence'].items() if k in ('ibugIds','vertexIndices','barycentricWeights','targetPixels','acceptedAfter')}
        right={k:np.asarray(v) for k,v in gs['stageEvidence'].items() if k in ('ibugIds','vertexIndices','barycentricWeights','targetPixels','acceptedAfter')}
        if any(not np.array_equal(left[k],right[k]) for k in ('ibugIds','vertexIndices','barycentricWeights','targetPixels')):
            result[name]={'accepted':False,'reason':'pairedStageTargetsChanged'};continue
        def measure(e):
            points=np.einsum('ij,ijk->ik',left['barycentricWeights'],e['acceptedAfter'][left['vertexIndices']])
            pixels,depth=projectPoints(points,np.asarray(camera['intrinsics']),np.asarray(camera['rotation']),np.asarray(camera['translation']))
            if not np.isfinite(pixels).all() or not (depth>0).all():raise ValueError('Invalid paired eye projection')
            errors=np.linalg.norm(pixels-left['targetPixels'],axis=1)
            return {'meanPixels':float(errors.mean()),'maximumPixels':float(errors.max()),'errorsPixels':errors,'pointCount':len(errors)}
        lm,gm=measure(left),measure(right);tolerance=plan()['localEyes']['pairedMeanErrorToleranceImageScale']*imageScale
        change=lm['meanPixels']-gm['meanPixels']
        result[name]={'accepted':bool(change<=tolerance),'meanTolerancePixels':tolerance,'meanChangePixels':change,'local':lm,'global':gm}
    return {'accepted':all(x['accepted'] for x in result.values()),'eyes':result,'metric':'same six-point mean at corresponding eye-stage end; not anatomical ground truth'}


def runCase(caseId):
    from faceAppearance.clientGeometryRefinement import refineRegisteredClientGeometry
    from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
    from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
    from faceAppearance.geometryRefinement import _validateCameras
    context=contextFor(caseId);assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    raw=context['raw'];transform={k:raw[k].tolist() for k in ('rotation','translation','scale')}
    try:source=sourceInitial(context,assets)
    except Exception as error:
        writeJson(context['root']/'initial/sourceFailure.json',{'caseId':caseId,'status':'sourceInitialException','errorType':type(error).__name__,'error':str(error),'accepted':False,'noFallback':True})
        source=None;print(json.dumps({'case':caseId,'stage':'sourceInitialException','error':str(error)}),flush=True)
    reference=context['reference'];camera=_validateCameras([context['cameras'][reference]])[0]
    objective=buildShapeObjective(raw,.01);guard=buildGuardSpec(objective,normalizedShapePolicy(read(PRIOR/'policy.json')))
    pure=arrays(OLD/caseId/'geometry/pureDirect/fullGeometry.npz');oldBounded=arrays(OLD/caseId/'geometry/bounded/fullGeometry.npz')
    purePairs=intersectionPairs(pure['vertices'],pure['faces']);currentPairs=intersectionPairs(oldBounded['vertices'],oldBounded['faces'])
    summaries={};refinedBranches={};commonFull=None;commonScene=None;commonArchive=None;commonPairs=None
    for branch,options in plan()['branches'].items():
        directory=context['root']/branch;directory.mkdir(exist_ok=False);start=perf_counter()
        if options['sourcePrior'] and source is None:
            writeJson(directory/'report.json',{'caseId':caseId,'branch':branch,'status':'blockedByInitialException','geometrySafetyAccepted':False,'finalCandidateAccepted':False})
            summaries[branch]={'report':str(directory/'report.json'),'geometrySafetyAccepted':False,'status':'blockedByInitialException'};continue
        initial=source if options['sourcePrior'] else context['bounded']['vertices']
        np.savez_compressed(directory/'initial.npz',vertices=initial,faces=raw['faces'])
        refined=refineRegisteredClientGeometry(assets,initial,raw['faces'],transform,context['cameras'],context['inputs'],
            photoPaths=[None]*len(context['inputs']),delight=context['delight'],projectRoot=ROOT,
            outputDirectory=directory/'refinement',workingPhotoPaths=context['photos'],referenceViewIndex=reference,
            cachedObservations=context['shared'],captureSemanticEyeEvidence=True,
            semanticEyeSupportPolicy={'radiusEyeWidthFraction':.5} if options['localEyes'] else None,
            registrationDiagnostics={'method':'protectedFeatureFactorial','branch':branch,'templateToUff':transform})
        saveEvidence(directory/'refinementEvidence',refined['diagnostics']);refinedBranches[branch]=refined
        preJaw=refined['vertices']
        jaw=refineRenderedClientJaw(assets,preJaw,raw['faces'],transform,context['cameras'],context['inputs'],
            [None]*len(context['inputs']),projectRoot=ROOT,outputDirectory=directory/'renderedJaw',
            referenceViewIndex=reference,photoObservations=context['fan'])
        skin=jaw['vertices'];geometry=finalGeometry(assets,skin)
        if branch=='baseline':
            template=assets.materials[0]['baseColor'][...,:3]
            assembly=assets.assembleFinalClientSurface(geometry,transform,template,{'coverageMask':np.zeros(template.shape[:2],bool)},
                accessoryReferenceCanonical=preJaw,diagnostics={'method':'protectedFeatureFactorial','branch':branch,'diagnosticOnly':True})
            exported=exportLegacyAsset(directory/'commonAssembly',assembly,gender='male');commonArchive=exported['archivePath']
            commonScene=loadLegacyEyeBindingInputs(commonArchive,assets.arrays['nicpToFullIds'])
            commonFull=assembly['geometry']['vertices']@np.asarray(transform['rotation'])*transform['scale']+transform['translation']
            commonFull[raw['skinIds']]=skin;commonPairs=intersectionPairs(commonFull,commonScene['faces'])
        full=commonFull.copy();full[raw['skinIds']]=skin;nonSkin=np.setdiff1d(np.arange(len(full)),raw['skinIds'])
        np.testing.assert_array_equal(full[nonSkin],commonFull[nonSkin])
        saveArrays(directory/'finalGeometry.npz',geometry);np.savez_compressed(directory/'fullGeometry.npz',vertices=full,faces=commonScene['faces'])
        np.savez_compressed(directory/'states.npz',initial=initial,preJaw=preJaw,final=skin,fixedVertexIds=raw['staticIds'],**{k:np.asarray(v) for k,v in transform.items()})
        saveArrays(directory/'mouthState.npz',{**refined['mouthState'],'afterCanonical':skin.copy()})
        asset=exportLegacyGeometryVariant(commonArchive,full,directory/'asset',diagnostics={'method':'protectedFeatureFactorial','branch':branch,'diagnosticOnly':True,'allNonSkinFixedToBaseline':True})
        check=checkFrontGeometry(initial,skin,raw['faces'],camera=camera,fixedVertexIds=raw['staticIds'],
            maxDisplacement=float(np.ptp(initial,axis=0).max())*.07,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        newSkin=sorted(intersectionPairs(skin,raw['faces'])-intersectionPairs(initial,raw['faces']))
        local=evaluateMeshShapeConstraints(skin,guard,withJacobian=False)['values'];benefit=cavityBenefit(objective,skin)
        fullPairs=intersectionPairs(full,commonScene['faces']);newPure=sorted(fullPairs-purePairs);newCurrent=sorted(fullPairs-currentPairs);newPaired=sorted(fullPairs-commonPairs)
        replay=None
        if branch=='baseline':
            replay=float(np.abs(skin-arrays(OLD/caseId/'geometry/bounded/finalGeometry.npz')['vertices']).max())
            if replay>1e-12:raise ValueError(f'Old bounded skin replay failed: {replay}')
        paired=None
        if options['localEyes']:
            partner=plan()['localEyes']['pairs'][branch]
            imageScale=min(context['shared']['observations'][reference]['workingRgb'].shape[:2])/512.
            paired=pairedEyeAudit(refined,refinedBranches[partner],camera,imageScale)
        initialPassed=not options['sourcePrior'] or read(context['root']/'initial/sourceProtected/report.json')['accepted']
        downstream=bool(check['accepted'] and not newSkin and (local>=-1e-8).all() and benefit['accepted'])
        accepted=bool(initialPassed and downstream and not newPure and not newCurrent and not newPaired and (paired is None or paired['accepted']))
        report={'caseId':caseId,'branch':branch,'options':options,'status':'geometryDiagnosticComplete','seconds':perf_counter()-start,
            'referenceViewIndex':reference,'initialAccepted':initialPassed,'cumulativeDownstreamGate':check,'newSkinPairsAgainstBranchInitial':newSkin,
            'finalCavityGuard':{'accepted':bool((local>=-1e-8).all()),'minimumConstraint':float(local.min()),'violatedCount':int((local< -1e-8).sum())},
            'finalCavityBenefit':benefit,'pairedSemanticAlignment':paired,'allNonSkinExactlyFixed':True,
            'newFullPairsAgainstOriginalPureDirect':newPure,'newFullPairsAgainstCurrentBounded':newCurrent,'newFullPairsAgainstPairedBaseline':newPaired,
            'originalBaselineReplayMaximumError':replay,'geometryRefinement':refined['diagnostics'],'renderedJaw':jaw['diagnostics'],
            'templateToUff':transform,'asset':asset,'geometrySafetyAccepted':accepted,'noNewRegressionAgainstPairedBaseline':bool(downstream and not newPaired),
            'finalCandidateAccepted':False,'finalAdaptiveAssemblyValidated':False,'sourceHashes':context['hashes'],'textureRebaked':False,
            'outputHashes':{str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}}
        verify(context['hashes']);writeJson(directory/'report.json',report)
        summaries[branch]={'report':str(directory/'report.json'),'geometrySafetyAccepted':accepted}
        print(json.dumps({'case':caseId,'branch':branch,'accepted':accepted,'newPairs':len(newPure),'newPaired':len(newPaired),'seconds':report['seconds']}),flush=True)
    writeJson(context['root']/'report.json',{'caseId':caseId,'branches':summaries,'status':'complete','planSha256':PLAN_SHA})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--case');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.case:runCase(args.case)
    else:parser.error('Choose --freeze or --case')
