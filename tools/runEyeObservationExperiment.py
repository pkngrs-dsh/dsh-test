"""Prospective unified-eye-observation ablation on the same completed baseline.

No previous runner or artifact is overwritten. The original-observation arm must replay
the preceding raw proposal chain, not merely return the same fallback mesh.
Only the reference-eye observation definitions change in the candidate arm.
Original observation gates remain separately evaluated and cannot be relabeled
as passed merely because the unified-target policy passes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from copy import deepcopy
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.legacyGeometryVariant import exportLegacyGeometryVariant
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, _sphere
from faceAppearance.visibleEyeContours import evaluateApertureMask
from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
from faceAppearance.surfaceContactConstraints import buildSurfaceContactSpec, evaluateSurfaceContactSafety
from runProtectedFeatureExperiment import saveEvidence, cavityBenefit
from experimentShapePreservation import buildShapeObjective, buildGuardSpec, normalizedShapePolicy
RUN=ROOT/'runs/eyeObservation20261024'
PARENT=ROOT/'runs/protectedFeature20261024'
REPLAY=ROOT/'runs/boundaryField20261024'
OLD=ROOT/'runs/shapePipeline20261024'
PRIOR=ROOT/'runs/shapePreservation20261024'
BRANCHES=('baseline','boundaryReplay','unifiedObservationContact')
PARENT_SHA='6119af96fe2b269e15f33a45b89514887378ed0377fc4e3995a587975696f4ce'
REPLAY_PLAN_SHA='afd8ba41054a7583852bdabb71d42c949268d884cfa25cc8cad88d80cf0381d1'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
def verify(hashes):
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError(f'Frozen input changed: {path}')
def readEvidence(stem):
    """Read only the existing JSON-indexed NPZ format; never deserialize pickle."""
    stem=Path(stem);metadata=read(stem.with_suffix('.json'));saved=arrays(stem.with_suffix('.npz'));used=set()
    def restore(value):
        if isinstance(value,dict) and 'arrayKey' in value:
            key=value['arrayKey'];item=saved[key]
            if list(item.shape)!=value['shape'] or str(item.dtype)!=value['dtype']:
                raise ValueError(f'Evidence array identity mismatch: {key}')
            used.add(key);return item
        if isinstance(value,dict):return {key:restore(item) for key,item in value.items()}
        if isinstance(value,list):return [restore(item) for item in value]
        return value
    value=restore(metadata)
    if used!=set(saved):raise ValueError('Unreferenced evidence arrays')
    return value

def sameEvidence(first,second):
    """Compare numerical evidence without treating JSON key conversion as a change."""
    if isinstance(first,dict) or isinstance(second,dict):
        if not isinstance(first,dict) or not isinstance(second,dict):return False
        a={str(key):value for key,value in first.items()};b={str(key):value for key,value in second.items()}
        return len(a)==len(first) and len(b)==len(second) and a.keys()==b.keys() and all(sameEvidence(a[k],b[k]) for k in a)
    if isinstance(first,np.ndarray) or isinstance(second,np.ndarray):
        a,b=np.asarray(first),np.asarray(second)
        return a.shape==b.shape and np.array_equal(a,b)
    if isinstance(first,(list,tuple)) or isinstance(second,(list,tuple)):
        return isinstance(first,(list,tuple)) and isinstance(second,(list,tuple)) and len(first)==len(second) and all(sameEvidence(a,b) for a,b in zip(first,second))
    return first==second
def plan():
    manifest=read(RUN/'implementationHashes.json')
    if sha(RUN/'experimentPlan.json')!=manifest['planSha256']:raise ValueError('Plan changed')
    return read(RUN/'experimentPlan.json')

def resolveHistorical(hashes):
    snapshots={}
    for manifestPath in (PARENT/'implementationHashes.json',ROOT/'runs/residualContact20261024/implementationHashes.json',REPLAY/'implementationHashes.json',ROOT/'reports/shapePipeline20261024/frozenImplementation/index.json'):
        manifest=read(manifestPath)
        for source,item in manifest.get('snapshots',manifest.get('files',{})).items():
            snapshots[(source,item['sha256'])]=item['snapshotPath']
    resolved={}
    for path,expected in hashes.items():
        target=path if Path(path).is_file() and sha(path)==expected else snapshots.get((path,expected))
        if target is None or sha(target)!=expected:raise ValueError(f'Historical identity lost: {path}')
        resolved[target]=expected
    return resolved

def freeze():
    if sha(PARENT/'experimentPlan.json')!=PARENT_SHA:raise ValueError('Parent plan changed')
    if sha(REPLAY/'experimentPlan.json')!=REPLAY_PLAN_SHA:raise ValueError('Legacy replay plan changed')
    if read(REPLAY/'implementationHashes.json')['planSha256']!=REPLAY_PLAN_SHA:raise ValueError('Replay implementation/plan mismatch')
    cases=read(PARENT/'experimentPlan.json')['cases']
    if read(REPLAY/'experimentPlan.json')['cases']!=cases:raise ValueError('Historical seven-case split changed')
    historical=[PARENT/'experimentPlan.json',PARENT/'implementationHashes.json',REPLAY/'experimentPlan.json',
        REPLAY/'implementationHashes.json',PRIOR/'split.json']
    for case in cases:
        historical.extend([PARENT/case/'baseline/report.json',REPLAY/case/'report.json',REPLAY/case/'boundaryFieldContact/report.json'])
    historicalHashes={str(path):sha(path) for path in historical}
    RUN.mkdir(parents=True,exist_ok=False)
    config=dict(schemaVersion='unifiedEyeObservationAblationV1',status='frozenBeforeRealCandidateSolves',
        frozenAt=datetime.now(timezone.utc).isoformat(),parentPlan=str(PARENT/'experimentPlan.json'),parentPlanSha256=PARENT_SHA,
        legacyReplayPlan=str(REPLAY/'experimentPlan.json'),legacyReplayPlanSha256=REPLAY_PLAN_SHA,historicalHashes=historicalHashes,
        parentSplit=str(PRIOR/'split.json'),parentSplitSha256=sha(PRIOR/'split.json'),
        cases=cases,branches=list(BRANCHES),
        start='Exact completed protectedFeature baseline fullGeometry; no coarse fit, jaw fit or reassembly',
        observationPolicy=dict(source='Exact original frozen dense working-photo eyelid curves; not manual ground truth',
            table='One image-left/right ordered iBUG36:48 table from original reference 68-point observations',
            internalPoints='Keep original coordinate along the dense-canthus axis; interpolate the corresponding upper/lower piecewise-linear curve at that coordinate',
            corners='Mean of upper/lower corresponding endpoints; corners may change both axis coordinates',
            failure='Reject ambiguous, reversed, nonmonotone or out-of-domain observations; no sorting, extrapolation or endpoint clamping',
            consumers=['coarse guard with original retained IDs and bindings','reference front-eye observation guard with original eligible mask'],
            unchanged=['dense fitting targets','complete-aperture evaluation targets','all side views','mouth','weights','confidence','eligibility','geometry','camera','support','contact spec'],
            originalPolicy='Original coarse and reference-68 gates are still evaluated separately; legacy absolute eligibility requires both',
            anatomicalAccuracyEstablished=False,originalResolutionDiagnosticAffectsGeometry=False),
        residualPolicy=dict(geodesicBandEyeWidthFraction=.30,rounds=2,iterationsPerRound=20),
        fieldPolicy=dict(legacyMode='fixedBoundaryBiharmonic',candidateMode='fixedBoundaryBiharmonic',smoothingRadiusFraction=.1,
            controlCount=32,captureSolverEvidence=True,deformationFrame='referenceImagePlane',degreesOfFreedomPerControl=2,
            supportRadiusExtentFraction=.12,maxDisplacementExtentFraction=.03,
            requestedCenters='Same deterministic farthest-point selection from each round original kept point bindings; no extra centers',
            support='Same completed-baseline frozen movable/fixed masks; FEM rebuilt from current local mesh each round',
            equation='Coordinates normalized by requested supportRadius; (M_U + ell^4 (K.T M^-1 K)_UU) B_U = M_U W_U, B_F=0, ell=.1',
            stiffness='Original-triangle FEM cotangent stiffness; positive lumped triangle area; no proximity edges/cotangent clamp/ridge',
            normalization='Both arms retain the identical fixed-boundary field with max(1,max absolute row sum)',
            regularization=dict(displacementWeight=.002,strainWeight=.03,controlWeight=.0001,curveWeight=0.,pixelScale=4.),
            solver=dict(initialDamping=.001,maxBacktracking=14,outerScales=[1.,.5,.25],sameTwoRoundsTwentyIterations=True),
            coefficientPenaltyCaveat='Same controlWeight is retained; coefficient penalties are basis-dependent. Physical strain/displacement formulas and all gates are unchanged.',
            capture='Full B, axes, normalization, coefficients, evaluated trials, field construction, and current reference mesh retained per round'),
        coarseGate=dict(meanToleranceImageScale=.05,maximumToleranceImageScale=.05,anchor='completed final baseline'),
        contactPolicy=dict(lengthScaleMm=1.,searchRadiusMm=2.,activationDistanceMm=1.,minimumGapMm=0.,weight=1.,tolerance=1e-8),
        orientation=dict(semanticMaterialId=2,normalSign=1,maximumSphereResidualRatio=.02,minimumRadialFaceCosine=.95,
            rule='Explicit template winding independently verified against fitted sphere center; same check on fixed assembled eye; open surface, no inside/outside certificate'),
        fixed='All non-skin including canthi, plus static skin and outside baseline-frozen local support; real lips fixed',
        omittedConstraints='Only boundary samples with all three vertices fixed; complete masks and full collision evaluation retain all surfaces',
        safety='Unchanged cumulative original bounds and cavity benefit, complete mask benefit/fragmentation, original pureDirect/currentBounded/paired completed baseline full collision anchors',
        render=dict(allFourSavedCameras=True,resolution=512,modes=['clay','baseColor','lit'],commonTemplateTextures=True,totalImages=420),
        noPerCaseParameters=True,noParameterSelection=True,retainRejectedCandidates=True,productionDefaultsChanged=False,
        realLipOptimization=False,finalAdaptiveAssemblyValidated=False,textureRebaked=False,
        contactGate='Identical baseline-frozen contact spec and residual/hard gate inside GN on BOTH fitting arms',
        legacyReplayGate='Exact vertices/solver/raw arrays and ordered round/scale proposal vertices against preceding boundaryFieldContact; saved replayMatch then raise on mismatch',
        finalEffectPolicy=dict(anchor='Exact completed baseline full visible aperture masks',enforcedOn='unifiedObservationContact',
            legacyReplayOutputUnchanged=True,saveRawRefinementBeforeEffectGate=True,requireSameRoi=True,requireFinalCompleteExtraction=True,
            globalBoundaryImprovement=.03,perEyeMeanTolerance=.15,perEyeP95Tolerance=.30,
            pixelToleranceConvention='Values divided by originalPixelScale',minimumIoUChange=-.005,
            maximumComponentCountIncrease=0,maximumSecondaryAreaFractionIncrease=.005,
            rejection='Return exact completed baseline; retain raw refinement and solver/proposal evidence',
            meaning='Final-versus-baseline whole-aperture effect only; geometric safety and inherited failure remain separate'),
        legacyPhotoGate='Each outer proposal: front mouth/eyes mean <= completed baseline + .05*imageScale; side eyes >=4 points mean <= original bounded initial + 1*imageScale; unchanged visibility and eligibility',
        inheritedFailurePolicy='Report incrementalSafe separately from absoluteEligible; no-new-pairs never erases inherited failures',
        historicalHoldouts='Previously inspected regression cases, not independent unseen generalization evidence')
    writeJson(RUN/'experimentPlan.json',config)
    paths=sorted((ROOT/'src/faceAppearance').rglob('*.py'))+[ROOT/'tools'/name for name in (
        'runEyeObservationExperiment.py','renderEyeObservationExperiment.py')]
    paths += [ROOT/'tools'/name for name in ('runProtectedFeatureExperiment.py','runShapePipelineExperiment.py','experimentShapePreservation.py','experimentEarlyCavityRegistration.py','experimentNoseSurfaceRegistration.py','validateProtectedFeatureExperiment.py','validateFeatureGeometryExperiment.py')]
    paths += [ROOT/'tools'/name for name in ('renderClientAppearanceEvidence.py','renderClientAssembly.py',
        'renderFeedbackAppearanceComparison.py','renderNoseRepairComparison.py','renderShapePreservationComparison.py')]
    destination=RUN/'implementation';copies={}
    for path in sorted(set(paths)):
        target=destination/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        copies[str(path)]=dict(snapshotPath=str(target),sha256=sha(path))
    writeJson(RUN/'implementationHashes.json',dict(sourceHashes={p:v['sha256'] for p,v in copies.items()},snapshots=copies,planSha256=sha(RUN/'experimentPlan.json')))
    print(json.dumps(dict(status='frozen',files=len(copies),planSha256=sha(RUN/'experimentPlan.json'))),flush=True)

def contextFor(caseId):
    config=plan()
    if caseId not in config['cases']:raise ValueError('Case outside frozen split')
    verify(config['historicalHashes'])
    implementation=read(RUN/'implementationHashes.json');verify(implementation['sourceHashes'])
    context=read(PARENT/caseId/'inputContext.json');entry=context['entry'];baselineDir=PARENT/caseId/'baseline'
    report=read(baselineDir/'report.json');hashes=resolveHistorical(context['sourceHashes'])
    hashes.update(resolveHistorical(report['sourceHashes']))
    for name,expected in report['outputHashes'].items():
        p=baselineDir/name
        if sha(p)!=expected:raise ValueError('Completed baseline output changed')
        hashes[str(p)]=expected
    replayRoot=REPLAY/caseId
    replayContext=read(replayRoot/'inputContext.json')
    replayComplete=read(replayRoot/'report.json')
    replayReport=read(replayRoot/'boundaryFieldContact/report.json')
    if (not sameEvidence(replayContext['entry'],entry)
            or not sameEvidence(replayContext['referenceSelection'],context['referenceSelection'])
            or replayContext['completedBaselineRoot']!=str(baselineDir)
            or replayContext['planSha256']!=REPLAY_PLAN_SHA):
        raise ValueError('Legacy replay context is not the same completed baseline/photo split')
    hashes.update(resolveHistorical(replayContext['sourceHashes']))
    hashes.update(resolveHistorical(replayReport['sourceHashes']))
    verify(replayComplete['preparationHashes']);hashes.update(replayComplete['preparationHashes'])
    for name,expected in replayReport['outputHashes'].items():
        path=replayRoot/'boundaryFieldContact'/name
        if sha(path)!=expected:raise ValueError('Legacy replay output changed')
        hashes[str(path)]=expected
    hashes.update(config['historicalHashes'])
    hashes.update({str(p):sha(p) for p in (PARENT/caseId/'inputContext.json',baselineDir/'report.json',RUN/'experimentPlan.json',RUN/'implementationHashes.json',PRIOR/'policy.json')})
    hashes.update(implementation['sourceHashes'])
    raw=arrays(entry['preparedGeometry']);full=arrays(baselineDir/'fullGeometry.npz');states=arrays(baselineDir/'states.npz')
    np.testing.assert_array_equal(full['vertices'][raw['skinIds']],states['final'])
    cameras=read(Path(entry['reconstruction'])/'cameras.json');inputs=read(Path(entry['reconstruction'])/'inputs.json')
    reference=context['referenceSelection']['referenceViewIndex'];dense=arrays(OLD/caseId/'observations/observations.npz')
    photo={name:{part:dense['eyelidCurvesWorking'][i,j] for j,part in enumerate(('lower','upper'))} for i,name in enumerate(('imageLeft','imageRight'))}
    constraints=[]
    for stage in report['geometryRefinement']['stagedLandmarks']['stages']:
        if stage['name'] not in ('imageLeftEye','imageRightEye'):continue
        evidence=stage['stageEvidence'];keep=np.asarray(stage['solver']['inputTargetGate']['groups'][0]['keptIndices'],int)
        constraints.append(dict(name=stage['name'],vertexIndices=raw['skinIds'][np.asarray(evidence['vertexIndices'],int)[keep]],
            barycentric=np.asarray(evidence['barycentricWeights'])[keep],**{key:np.asarray(evidence[key])[keep] for key in ('targetPixels','ibugIds','weights','confidence')}))
    root=RUN/caseId;root.mkdir(exist_ok=False)
    verify(hashes)
    writeJson(root/'inputContext.json',dict(entry=entry,referenceSelection=context['referenceSelection'],sourceHashes=hashes,
        parentReport=str(baselineDir/'report.json'),completedBaselineRoot=str(baselineDir),oldPipelineRoot=str(OLD/caseId),
        legacyReplayRoot=str(replayRoot),legacyReplayReport=str(replayRoot/'boundaryFieldContact/report.json'),
        planSha256=implementation['planSha256']))
    return dict(root=root,entry=entry,raw=raw,full=full,states=states,baselineReport=report,baselineDir=baselineDir,
        camera=cameras[reference],cameras=cameras,inputs=inputs,reference=reference,photoEyes=photo,coarseEyeConstraints=constraints,
        imageShape=tuple(inputs[reference]['workingSize'][::-1]),originalPixelScale=float(np.asarray(inputs[reference]['inputToWorking'])[0,0]),
        hashes=hashes,legacyReplayRoot=replayRoot)

def legacyReplayMatch(result,previous,previousFull):
    """Exact numeric replay; timing/new instrumentation are intentionally absent."""
    checks={};details={}
    for name in ('vertices','solverVertices','rawCandidateVertices'):
        before,after=np.asarray(previous[name]),np.asarray(result[name])
        checks[name]=before.shape==after.shape and np.array_equal(before,after)
        details[name]=dict(shape=list(after.shape),exact=bool(checks[name]),
            maximumAbsoluteDifference=float(np.max(np.abs(after-before),initial=0)) if before.shape==after.shape else None)
    checks['historicalFullGeometry']=np.array_equal(result['vertices'],previousFull)
    first,second=previous['candidateProposals'],result['candidateProposals']
    checks['proposalCount']=len(first)==len(second);proposals=[]
    for index,(before,after) in enumerate(zip(first,second)):
        sameShape=np.shape(before['vertices'])==np.shape(after['vertices'])
        local=dict(round=before['round']==after['round'],scale=before['scale']==after['scale'],
            vertices=sameShape and np.array_equal(before['vertices'],after['vertices']),
            selectedByFastRound=before['selectedByFastRound']==after['selectedByFastRound'],
            strictAccepted=before['strict']['accepted']==after['strict']['accepted'])
        proposals.append(dict(index=index,round=after['round'],scale=after['scale'],checks=local,
            exact=bool(all(local.values())),maximumAbsoluteDifference=float(np.max(np.abs(after['vertices']-before['vertices']),initial=0)) if sameShape else None))
    checks['orderedProposalChain']=checks['proposalCount'] and all(item['exact'] for item in proposals)
    for key in ('status','accepted','geometryChanged','fastSolverInvoked','contactOptimizationEnabled'):
        checks['diagnostic'+key]=previous['diagnostics'].get(key)==result['diagnostics'].get(key)
    oldStages=previous['diagnostics'].get('fast',{}).get('stages',[])
    newStages=result['diagnostics'].get('fast',{}).get('stages',[])
    checks['stageCount']=len(oldStages)==len(newStages);stages=[]
    for before,after in zip(oldStages,newStages):
        local={key:sameEvidence(before.get(key),after.get(key)) for key in ('round','status','selectedScale','binding')}
        oldSolver,newSolver=before.get('solver',{}),after.get('solver',{})
        for key in ('inputTargetGate','fixedBoundary','initialMeanErrorPixels','finalMeanErrorPixels',
                    'initialGroups','finalGroups','bestIteration','iterationsCompleted','controlCount',
                    'activeVertexCount','maximumReferenceDepthChange','supportRadius','maxDisplacement'):
            local['solver'+key]=sameEvidence(oldSolver.get(key),newSolver.get(key))
        stages.append(dict(round=after['round'],checks=local,exact=bool(all(local.values()))))
    checks['stageBindingsAndKeySolverDiagnostics']=checks['stageCount'] and all(item['exact'] for item in stages)
    return dict(accepted=bool(all(checks.values())),checks=checks,arrays=details,proposals=proposals,stages=stages,
        tolerance=0.,comparison='Exact full/solver/raw/proposal vertices and kept/binding/solver key state; timing and added capture fields omitted')

def finalBaselineEffectGate(prepared,result,policy):
    """Existing complete-mask metrics, now anchored at the common FINAL baseline.

    Per-round aperture gates remain unchanged inside the solver. This separate
    final check prevents several individually tolerated regressions accumulating
    across rounds. Texture-only contour proxies are never used here.
    """
    before=prepared['beforeContours'];after=result['afterContours'];scale=prepared['originalPixelScale']
    checks={};beforeMetrics=afterMetrics=None;reason=None
    try:
        beforeMetrics=evaluateApertureMask(before,prepared['photoEyes'],originalPixelScale=scale)
        afterMetrics=evaluateApertureMask(after,prepared['photoEyes'],originalPixelScale=scale)
        eyes=('imageLeft','imageRight')
        checks['sameTwoEyes']=set(before['eyes'])==set(after['eyes'])==set(eyes)
        for name in eyes:
            a,b=before['eyes'][name],after['eyes'][name]
            checks[name+'SameRoi']=sameEvidence(a['roi'],b['roi']) and np.shape(a['roiMasks']['eye'])==np.shape(b['roiMasks']['eye'])
            checks[name+'CompleteFinalExtraction']=b['status']=='accepted'
        oldMean=float(beforeMetrics['meanSymmetricBoundaryDistanceOriginalPixels'])
        newMean=float(afterMetrics['meanSymmetricBoundaryDistanceOriginalPixels'])
        checks['globalBoundaryBenefit']=bool(np.isfinite([oldMean,newMean]).all() and newMean<oldMean-policy['globalBoundaryImprovement']/scale)
        for name in eyes:
            a,b=beforeMetrics['eyes'][name],afterMetrics['eyes'][name]
            for metric,limit,direction in (
                ('symmetricBoundaryMeanOriginalPixels',policy['perEyeMeanTolerance']/scale,1),
                ('symmetricBoundaryP95OriginalPixels',policy['perEyeP95Tolerance']/scale,1),
                ('intersectionOverUnion',-policy['minimumIoUChange'],-1),
                ('visibleComponentCount',policy['maximumComponentCountIncrease'],1),
                ('secondaryComponentAreaFraction',policy['maximumSecondaryAreaFractionIncrease'],1)):
                left,right=float(a[metric]),float(b[metric])
                checks[name+metric]=bool(np.isfinite([left,right]).all() and direction*(right-left)<=limit)
    except (KeyError,TypeError,ValueError) as error:
        checks['completeActualApertureMetricsAvailable']=False
        reason=f'{type(error).__name__}: {error}'
    return dict(accepted=bool(checks and all(checks.values())),checks=checks,
        baseline=beforeMetrics,candidate=afterMetrics,reason=reason,originalPixelScale=float(scale),
        anchor='Exact completed baseline full visible aperture masks',perRoundGatesUnchanged=True,
        candidateChanged=not np.array_equal(result['vertices'],prepared['scene']['vertices']),
        photoTruthEstablished=False,textureContourProxyUsed=False)

def orientationEvidence(vertices,faces,faceIds,config):
    triangles=faces[faceIds];center,radius,residual=_sphere(vertices[np.unique(triangles)])
    normal=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
    radial=vertices[triangles].mean(1)-center
    cos=np.sum(normal*radial,axis=1)/(np.linalg.norm(normal,axis=1)*np.linalg.norm(radial,axis=1))
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    accepted=bool(residual/radius<=config['maximumSphereResidualRatio'] and np.min(cos)>=config['minimumRadialFaceCosine'] and (counts<=2).all())
    if not accepted:raise ValueError('Frozen template radial winding rule is ambiguous')
    return dict(accepted=accepted,center=center,radius=radius,maximumSphereResidual=residual,residualRatio=residual/radius,
        minimumRadialFaceCosine=float(cos.min()),boundaryEdgeCount=int((counts==1).sum()),closed=bool((counts==2).all()),normalSign=1)

def runCase(caseId):
    from faceAppearance.clientResidualEyeRefinement import prepareClientResidualEyes,refineClientResidualEyes,evaluateResidualCoarseAlignment
    context=contextFor(caseId);root=context['root'];raw=context['raw'];skinIds=raw['skinIds'];config=plan()
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    scene=loadLegacyEyeBindingInputs(context['baselineReport']['asset']['archivePath'],skinIds)
    np.testing.assert_allclose(scene['vertices'],context['full']['vertices'],atol=1e-12,rtol=0)
    archiveCanonical=scene['vertices'].copy()
    scene['vertices']=context['full']['vertices'].copy()
    np.testing.assert_array_equal(scene['faces'],context['full']['faces'])
    cornerFaces=np.concatenate([c['faceIndices'] for c in assets.components if c['name'] in ('male00SG.component001','male00SG.component002')])
    prepared=prepareClientResidualEyes(scene,context['camera'],context['photoEyes'],coarseEyeConstraints=context['coarseEyeConstraints'],
        imageShape=context['imageShape'],cornerFaceIds=cornerFaces,fixedVertexIds=skinIds[raw['staticIds']],
        originalPixelScale=context['originalPixelScale'],policy=config['residualPolicy'])
    oldPreparation=readEvidence(context['legacyReplayRoot']/'preparation')
    preparationChecks={key:sameEvidence(prepared[key],oldPreparation[key]) for key in
        ('scene','camera','photoEyes','imageShape','cornerFaceIds','originalPixelScale','policy','support',
         'coarseEyeConstraints','beforeCoarseAlignment','beforeContours','beforeApertureMetrics','baselineVerticesSha256')}
    writeJson(root/'preparationReplayMatch.json',dict(accepted=bool(all(preparationChecks.values())),checks=preparationChecks))
    if not all(preparationChecks.values()):raise ValueError('Common baseline preparation drifted from legacy experiment')
    saveEvidence(root/'preparation',prepared)
    groups=[];orientation=[];sources=np.flatnonzero(prepared['support']['movableMask'])
    for component in assets.components:
        if component['materialId']!=2:continue
        faceIds=np.asarray(component['faceIndices'],int)
        template=orientationEvidence(assets.arrays['vertices'],assets.arrays['faces'],faceIds,config['orientation'])
        assembled=orientationEvidence(scene['vertices'],scene['faces'],faceIds,config['orientation'])
        provenance='legacyMaleDirectServiceV1 '+component['name']+'; template and fixed assembled eye radial winding verified by frozen .02/.95 rule; open sphere patch; no skin-gap sign inference'
        if len(sources):groups.append(dict(name=component['name'],sourceVertexIds=sources,targetFaceIds=faceIds,normalSign=1,orientationProvenance=provenance))
        orientation.append(dict(name=component['name'],faceIds=faceIds,template=template,assembled=assembled,provenance=provenance))
    writeJson(root/'orientation.json',orientation)
    unit=float(raw['unit']);contact=config['contactPolicy']
    spec=buildSurfaceContactSpec(scene['vertices'],scene['faces'],groups,lengthScale=unit*contact['lengthScaleMm'],
        searchRadius=unit*contact['searchRadiusMm'],activationDistance=unit*contact['activationDistanceMm'],minimumGap=unit*contact['minimumGapMm'],weight=contact['weight'])
    oldSpec=readEvidence(context['legacyReplayRoot']/'contactSpec')
    contactChecks={key:sameEvidence(value,oldSpec[key]) for key,value in spec.items() if key!='gapOperator'}
    oldOperator=sparse.load_npz(context['legacyReplayRoot']/'contactOperator.npz')
    contactChecks['gapOperatorExact']=oldOperator.shape==spec['gapOperator'].shape and (oldOperator!=spec['gapOperator']).nnz==0
    writeJson(root/'contactReplayMatch.json',dict(accepted=bool(all(contactChecks.values())),checks=contactChecks))
    if not all(contactChecks.values()):raise ValueError('Frozen contact spec drifted from legacy experiment')
    saveEvidence(root/'contactSpec',spec);sparse.save_npz(root/'contactOperator.npz',spec['gapOperator'])
    from faceAppearance.clientLandmarks import loadClientLandmarks,ibugClientLandmarks
    from validateProtectedFeatureExperiment import originalObservationMasks,originalObservationErrors
    binding=ibugClientLandmarks(loadClientLandmarks(raw['faces'],path=ROOT/'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'),groups=('eyes','mouth'))
    observation=arrays(OLD/caseId/'geometry/photoObservations.npz')
    masks=originalObservationMasks(context['states']['initial'],raw['faces'],binding,context['cameras'],context['inputs'],observation)
    beforeErrors=originalObservationErrors(context['states']['initial'],binding,context['cameras'],observation,masks)
    baselineErrors=originalObservationErrors(context['states']['final'],binding,context['cameras'],observation,masks)
    saveEvidence(root/'legacyObservationState',dict(binding=binding,observations=observation,masks=masks,
        initialErrors=beforeErrors,completedBaselineErrors=baselineErrors))
    from faceAppearance.eyeObservationAlignment import alignEyeLandmarkTargets, alignCoarseEyeTargets
    reference=context['reference']
    originalPrepared=prepared
    unchangedPreparationKeys=('scene','camera','photoEyes','imageShape','cornerFaceIds','originalPixelScale','policy',
        'support','beforeContours','beforeApertureMetrics','baselineVerticesSha256')
    # Invalid observations reject only the alternative arm. Preserve all seven
    # cases and the original-policy replay without silently repairing targets.
    alignmentError=None;table=None;aligned=None
    unifiedObservation=observation;unifiedBaselineErrors=baselineErrors;unifiedPrepared=originalPrepared
    try:
        table=alignEyeLandmarkTargets(context['photoEyes'],observation['points'][reference,36:48])
        aligned=alignCoarseEyeTargets(context['photoEyes'],context['coarseEyeConstraints'],mappedTable=table)
    except ValueError as error:
        alignmentError=str(error);table=None;aligned=None
    alignmentValid=alignmentError is None
    if alignmentValid:
        unifiedObservation={key:value.copy() for key,value in observation.items()}
        # Both consumers use the exact single float64 table. Other values are
        # losslessly promoted; masks, confidence and bindings remain original.
        unifiedObservation['points']=observation['points'].astype(np.float64,copy=True)
        unifiedObservation['points'][reference,36:48]=table['targetPixels']
        unifiedBaselineErrors=originalObservationErrors(context['states']['final'],binding,context['cameras'],unifiedObservation,masks)
        unifiedPrepared=prepareClientResidualEyes(scene,context['camera'],context['photoEyes'],coarseEyeConstraints=aligned['coarseGroups'],
            imageShape=context['imageShape'],cornerFaceIds=cornerFaces,fixedVertexIds=skinIds[raw['staticIds']],
            originalPixelScale=context['originalPixelScale'],policy=config['residualPolicy'])
        if not all(sameEvidence(originalPrepared[key],unifiedPrepared[key]) for key in unchangedPreparationKeys):
            raise ValueError('Observation unification changed geometry, support or complete-mask targets')
        saveEvidence(root/'unifiedPreparation',unifiedPrepared)
    saveEvidence(root/'observationAlignment',dict(valid=alignmentValid,error=alignmentError,table=table,coarseAlignment=aligned,
        originalObservations=observation,unifiedObservations=unifiedObservation,masks=masks,
        originalBaselineErrors=baselineErrors,unifiedBaselineErrors=unifiedBaselineErrors,
        referenceViewIndex=reference,sourceDensePath=str(OLD/caseId/'observations/observations.npz'),
        sourceDenseSha256=sha(OLD/caseId/'observations/observations.npz'),
        unchangedPreparationKeys=unchangedPreparationKeys,anatomicalAccuracyEstablished=False))

    def observationGate(full,observations,baselineScores):
        errors=originalObservationErrors(full[skinIds],binding,context['cameras'],observations,masks)
        reference=context['reference'];imageScale=min(context['imageShape'])/512.
        front={region:errors[reference][region]['meanPixels']-baselineScores[reference][region]['meanPixels']
            for region in ('mouth','eyes') if baselineScores[reference][region]['count']>0}
        side=[dict(viewIndex=i,changePixels=value['eyes']['meanPixels']-beforeErrors[i]['eyes']['meanPixels'])
            for i,value in enumerate(errors) if i!=reference and value['eyes']['count']>=4]
        legacy=dict(accepted=all(v<=.05*imageScale for v in front.values()) and all(v['changePixels']<=imageScale for v in side),
            frontMeanChanges=front,sideMeanChangesAgainstOriginalInitial=side,frontTolerance=.05*imageScale,sideTolerance=imageScale)
        return legacy

    def candidateGate(full):
        active=observationGate(full,activeObservation,activeBaselineErrors)
        contactGate=evaluateSurfaceContactSafety(full,spec,tolerance=contact['tolerance'])
        result=dict(accepted=bool(active['accepted'] and contactGate['accepted']),legacyGate=active,contactGate=contactGate)
        # The replay arm must preserve even its historical rejection traces.
        # Alternative-target diagnostics never override the active decision.
        if branch=='unifiedObservationContact':
            result['originalPolicy']=dict(coarseAlignment=evaluateResidualCoarseAlignment(originalPrepared,full),
                legacyGate=observationGate(full,observation,baselineErrors))
            result['observationTableSha256']=table['tableSha256'] if table else None
        return result
    objective=buildShapeObjective(raw,.01);guard=buildGuardSpec(objective,normalizedShapePolicy(read(PRIOR/'policy.json')))
    initial=context['states']['initial'];initialPairs=intersectionPairs(initial,raw['faces'])
    anchors={}
    for name,branch in (('OriginalPureDirect','pureDirect'),('CurrentBounded','bounded')):
        old=arrays(OLD/caseId/'geometry'/branch/'fullGeometry.npz')
        anchors[name]=intersectionPairs(old['vertices'],old['faces'])
    anchors['PairedBaseline']=intersectionPairs(scene['vertices'],scene['faces'])
    summaries={}
    for branch in BRANCHES:
        start=perf_counter();directory=root/branch;directory.mkdir(exist_ok=False)
        isUnified=branch=='unifiedObservationContact'
        prepared=unifiedPrepared if isUnified else originalPrepared
        activeObservation=unifiedObservation if isUnified else observation
        activeBaselineErrors=unifiedBaselineErrors if isUnified else baselineErrors
        mode='none'
        if branch=='baseline' or isUnified and not alignmentValid:
            rawResult=dict(vertices=scene['vertices'].copy(),rawCandidateVertices=scene['vertices'].copy(),solverVertices=scene['vertices'].copy(),candidateProposals=[],
                beforeContours=prepared['beforeContours'],afterContours=prepared['beforeContours'],support=prepared['support'],
                solverEvidence=[],diagnostics=dict(status='skippedInvalidUnifiedObservation' if isUnified else 'unchangedCompletedCoarseBaseline',
                    candidateAccepted=False,accepted=False,contactOptimizationEnabled=False,observationAlignmentError=alignmentError if isUnified else None))
        else:
            mode=config['fieldPolicy']['legacyMode' if branch=='boundaryReplay' else 'candidateMode']
            rawResult=refineClientResidualEyes(prepared,contactSpec=spec,candidateGate=candidateGate,
                deformationMode=mode,smoothingRadiusFraction=config['fieldPolicy']['smoothingRadiusFraction'],
                captureSolverEvidence=config['fieldPolicy']['captureSolverEvidence'])
        # Preserve the complete original refinement decision, every proposal,
        # and GN evidence before the new final-versus-baseline effect policy.
        saveEvidence(directory/'rawRefinement',rawResult)
        saveEvidence(directory/'solverEvidence',rawResult.get('solverEvidence',[]))
        replayMatch=None
        if branch=='boundaryReplay':
            oldDirectory=context['legacyReplayRoot']/'boundaryFieldContact'
            previousRaw=readEvidence(oldDirectory/'rawRefinement')
            replayMatch=legacyReplayMatch(rawResult,previousRaw,previousRaw['vertices'])
            writeJson(directory/'replayMatch.json',replayMatch)
            if not replayMatch['accepted']:raise ValueError('Original-policy replay differs; no observation comparison is valid')
        effect=finalBaselineEffectGate(prepared,rawResult,config['finalEffectPolicy'])
        effect['enforced']=branch==config['finalEffectPolicy']['enforcedOn']
        effect['isBaselineReference']=branch=='baseline'
        effect['rawRefinementAccepted']=bool(rawResult['diagnostics'].get('accepted',False))
        effect['fallbackToExactBaseline']=bool(effect['enforced'] and alignmentValid and not effect['accepted'])
        writeJson(directory/'finalEffectGate.json',effect)
        result=dict(rawResult)
        result['diagnostics']=deepcopy(rawResult['diagnostics'])
        if effect['fallbackToExactBaseline']:
            result.update(vertices=scene['vertices'].copy(),afterContours=prepared['beforeContours'],textureContours=None)
            result['diagnostics'].update(status='rejectedFinalBaselineEffect',accepted=False,geometryChanged=False,
                rawRefinementStatus=rawResult['diagnostics']['status'],finalEffectFallback=True)
        result['diagnostics']['finalBaselineEffect']=effect
        full=result['vertices'];skin=full[skinIds]
        saveEvidence(directory/'evidence',result)
        np.savez_compressed(directory/'fullGeometry.npz',vertices=full,faces=scene['faces'])
        np.savez_compressed(directory/'states.npz',initial=initial,baseline=context['states']['final'],completedCoarseBaseline=context['states']['final'],final=skin,
            fixedVertexIds=raw['staticIds'],**{k:raw[k] for k in ('rotation','translation','scale')})
        # Numerical fitting retains the exact saved baseline. Export uses the
        # original ZIP's own round-trip for fixed rows so its OBJ bytes do not
        # change merely because the historical NPZ differs at ~1e-14.
        changed=np.any(full!=scene['vertices'],axis=1)
        if (changed & ~prepared['support']['movableMask']).any():raise ValueError('Fixed coordinate changed')
        exportCanonical=archiveCanonical.copy();exportCanonical[changed]=full[changed]
        np.testing.assert_allclose(exportCanonical,full,rtol=0,atol=1e-12)
        np.savez_compressed(directory/'exportIdentity.npz',changedVertexMask=changed,canonicalVertices=exportCanonical)
        asset=exportLegacyGeometryVariant(context['baselineReport']['asset']['archivePath'],exportCanonical,directory/'asset',
            diagnostics=dict(method='unifiedEyeObservationDiagnostic',branch=branch,diagnosticOnly=True,allNonSkinFixedToBaseline=True))
        if asset['changedVertexCount']!=int(changed.sum()):raise ValueError('Export changed additional rows')
        cumulative=checkFrontGeometry(initial,skin,raw['faces'],camera=prepared['camera'],fixedVertexIds=raw['staticIds'],
            maxDisplacement=float(np.ptp(initial,axis=0).max())*.07,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        skinNew=sorted(intersectionPairs(skin,raw['faces'])-initialPairs)
        local=evaluateMeshShapeConstraints(skin,guard,withJacobian=False)['values'];benefit=cavityBenefit(objective,skin)
        current=intersectionPairs(full,scene['faces']);new={name:sorted(current-pairs) for name,pairs in anchors.items()}
        coarse=evaluateResidualCoarseAlignment(prepared,full)
        originalCoarse=evaluateResidualCoarseAlignment(originalPrepared,full)
        fixed=bool(np.array_equal(full[~prepared['support']['movableMask']],scene['vertices'][~prepared['support']['movableMask']]))
        downstream=bool(cumulative['accepted'] and not skinNew and (local>=-1e-8).all() and benefit['accepted'])
        contactSafety=evaluateSurfaceContactSafety(full,spec,tolerance=contact['tolerance'])
        external=candidateGate(full)
        originalLegacy=observationGate(full,observation,baselineErrors)
        incremental=bool(fixed and coarse['accepted'] and not new['PairedBaseline'] and contactSafety['accepted'])
        originalPolicyAccepted=bool(originalCoarse['accepted'] and originalLegacy['accepted'])
        activePolicyAccepted=None if isUnified and not alignmentValid else bool(coarse['accepted'] and external['legacyGate']['accepted'])
        absolute=bool((not isUnified or alignmentValid) and context['baselineReport']['initialAccepted'] and incremental and downstream and originalPolicyAccepted and activePolicyAccepted
            and not new['OriginalPureDirect'] and not new['CurrentBounded'])
        geometryChanged=not np.array_equal(full,scene['vertices'])
        improvement=bool(branch!='baseline' and geometryChanged and effect['accepted'] and incremental and result['diagnostics'].get('accepted',False))
        report=dict(caseId=caseId,branch=branch,status='geometryDiagnosticComplete',seconds=perf_counter()-start,referenceViewIndex=context['reference'],
            initialAccepted=context['baselineReport']['initialAccepted'],cumulativeDownstreamGate=cumulative,newSkinPairsAgainstBranchInitial=skinNew,
            finalCavityGuard=dict(accepted=bool((local>=-1e-8).all()),minimumConstraint=float(local.min()),violatedCount=int((local< -1e-8).sum())),
            finalCavityBenefit=benefit,pairedSemanticAlignment=coarse,contactSafety=contactSafety,
            legacyObservationGate=originalLegacy,activeLegacyObservationGate=external['legacyGate'],
            originalCoarseAlignment=originalCoarse,activeCoarseAlignment=coarse,
            originalObservationPolicyAccepted=originalPolicyAccepted,activeObservationPolicyAccepted=activePolicyAccepted,
            originalCoarseAndReferenceEyeGatesRetainedAsAudit=True,observationPolicyChangedForDiagnostic=bool(isUnified and alignmentValid),
            observationAlignmentValid=alignmentValid,observationAlignmentError=alignmentError,
            observationTableSha256=table['tableSha256'] if table else None,legacyAbsoluteEligible=absolute,allOutsideSupportExactlyFixed=fixed,
            asset=asset,geometrySafetyAccepted=absolute,incrementalSafe=incremental,absoluteEligible=absolute,
            residualCandidateAccepted=result['diagnostics'].get('accepted',False),
            rawRefinementAccepted=bool(rawResult['diagnostics'].get('accepted',False)),
            finalEffectGate=effect,improvementAccepted=improvement,geometryChanged=geometryChanged,
            changingAbsoluteEligible=bool(improvement and absolute),
            replayMatch=replayMatch,fieldMode='none' if branch=='baseline' else mode,
            rawRefinementEvidence=str(directory/'rawRefinement.json'),solverEvidence=str(directory/'solverEvidence.json'),
            finalCandidateAccepted=False,finalAdaptiveAssemblyValidated=False,textureRebaked=False,
            diagnostics=result['diagnostics'],sourceHashes=context['hashes'],
            **{'newFullPairsAgainst'+name:value for name,value in new.items()})
        report['outputHashes']={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        writeJson(directory/'report.json',report);summaries[branch]={k:report[k] for k in ('incrementalSafe','absoluteEligible','residualCandidateAccepted','improvementAccepted','changingAbsoluteEligible')}
        print(json.dumps(dict(case=caseId,branch=branch,status=result['diagnostics']['status'],**summaries[branch],seconds=report['seconds'])),flush=True)
    verify(context['hashes'])
    writeJson(root/'report.json',dict(caseId=caseId,status='complete',branches=summaries,planSha256=sha(RUN/'experimentPlan.json'),
        preparationHashes={str(p):sha(p) for p in root.iterdir() if p.is_file()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--case');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.case:runCase(args.case)
    else:parser.error('Choose --freeze or --case')
