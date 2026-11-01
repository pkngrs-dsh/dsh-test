"""Prospective seven-case residual-eye comparison after completed coarse fitting."""
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
from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
from faceAppearance.surfaceContactConstraints import buildSurfaceContactSpec, evaluateSurfaceContactSafety
from runProtectedFeatureExperiment import saveEvidence, cavityBenefit
from experimentShapePreservation import buildShapeObjective, buildGuardSpec, normalizedShapePolicy
RUN=ROOT/'runs/residualContact20261024'
PARENT=ROOT/'runs/protectedFeature20261024'
OLD=ROOT/'runs/shapePipeline20261024'
PRIOR=ROOT/'runs/shapePreservation20261024'
BRANCHES=('baseline','residualOnly','residualWithContact')
PARENT_SHA='6119af96fe2b269e15f33a45b89514887378ed0377fc4e3995a587975696f4ce'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
def verify(hashes):
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError(f'Frozen input changed: {path}')
def plan():
    manifest=read(RUN/'implementationHashes.json')
    if sha(RUN/'experimentPlan.json')!=manifest['planSha256']:raise ValueError('Plan changed')
    return read(RUN/'experimentPlan.json')

def resolveHistorical(hashes):
    snapshots={}
    for manifestPath in (PARENT/'implementationHashes.json',ROOT/'reports/shapePipeline20261024/frozenImplementation/index.json'):
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
    RUN.mkdir(parents=True,exist_ok=False)
    config=dict(schemaVersion='completedCoarseResidualContactV1',status='frozenBeforeRealCandidateSolves',
        frozenAt=datetime.now(timezone.utc).isoformat(),parentPlan=str(PARENT/'experimentPlan.json'),parentPlanSha256=PARENT_SHA,
        parentSplit=str(PRIOR/'split.json'),parentSplitSha256=sha(PRIOR/'split.json'),
        cases=read(PARENT/'experimentPlan.json')['cases'],branches=list(BRANCHES),
        start='Exact completed protectedFeature baseline fullGeometry; no coarse fit, jaw fit or reassembly',
        residualPolicy=dict(geodesicBandEyeWidthFraction=.30,rounds=2,iterationsPerRound=20),
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
        contactGate='Same final contact envelope on both residual arms; contact arm additionally applies residual and hard gate inside GN',
        legacyPhotoGate='Each outer proposal: front mouth/eyes mean <= completed baseline + .05*imageScale; side eyes >=4 points mean <= original bounded initial + 1*imageScale; unchanged visibility and eligibility',
        inheritedFailurePolicy='Report incrementalSafe separately from absoluteEligible; no-new-pairs never erases inherited failures',
        historicalHoldouts='Previously inspected regression cases, not independent unseen generalization evidence')
    writeJson(RUN/'experimentPlan.json',config)
    paths=sorted((ROOT/'src/faceAppearance').rglob('*.py'))+[ROOT/'tools'/name for name in (
        'runResidualContactExperiment.py','renderResidualContactExperiment.py')]
    paths += [ROOT/'tools'/name for name in ('runProtectedFeatureExperiment.py','runShapePipelineExperiment.py','experimentShapePreservation.py','experimentEarlyCavityRegistration.py','experimentNoseSurfaceRegistration.py','validateProtectedFeatureExperiment.py','validateFeatureGeometryExperiment.py')]
    destination=RUN/'implementation';copies={}
    for path in sorted(set(paths)):
        target=destination/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        copies[str(path)]=dict(snapshotPath=str(target),sha256=sha(path))
    writeJson(RUN/'implementationHashes.json',dict(sourceHashes={p:v['sha256'] for p,v in copies.items()},snapshots=copies,planSha256=sha(RUN/'experimentPlan.json')))
    print(json.dumps(dict(status='frozen',files=len(copies),planSha256=sha(RUN/'experimentPlan.json'))),flush=True)

def contextFor(caseId):
    config=plan()
    if caseId not in config['cases']:raise ValueError('Case outside frozen split')
    implementation=read(RUN/'implementationHashes.json');verify(implementation['sourceHashes'])
    context=read(PARENT/caseId/'inputContext.json');entry=context['entry'];baselineDir=PARENT/caseId/'baseline'
    report=read(baselineDir/'report.json');hashes=resolveHistorical(context['sourceHashes'])
    hashes.update(resolveHistorical(report['sourceHashes']))
    for name,expected in report['outputHashes'].items():
        p=baselineDir/name
        if sha(p)!=expected:raise ValueError('Completed baseline output changed')
        hashes[str(p)]=expected
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
        parentReport=str(baselineDir/'report.json'),completedBaselineRoot=str(baselineDir),oldPipelineRoot=str(OLD/caseId),planSha256=implementation['planSha256']))
    return dict(root=root,entry=entry,raw=raw,full=full,states=states,baselineReport=report,baselineDir=baselineDir,
        camera=cameras[reference],cameras=cameras,inputs=inputs,reference=reference,photoEyes=photo,coarseEyeConstraints=constraints,
        imageShape=tuple(inputs[reference]['workingSize'][::-1]),originalPixelScale=float(np.asarray(inputs[reference]['inputToWorking'])[0,0]),hashes=hashes)

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
    def candidateGate(full):
        errors=originalObservationErrors(full[skinIds],binding,context['cameras'],observation,masks)
        reference=context['reference'];imageScale=min(context['imageShape'])/512.
        front={region:errors[reference][region]['meanPixels']-baselineErrors[reference][region]['meanPixels']
            for region in ('mouth','eyes') if baselineErrors[reference][region]['count']>0}
        side=[dict(viewIndex=i,changePixels=value['eyes']['meanPixels']-beforeErrors[i]['eyes']['meanPixels'])
            for i,value in enumerate(errors) if i!=reference and value['eyes']['count']>=4]
        legacy=dict(accepted=all(v<=.05*imageScale for v in front.values()) and all(v['changePixels']<=imageScale for v in side),
            frontMeanChanges=front,sideMeanChangesAgainstOriginalInitial=side,frontTolerance=.05*imageScale,sideTolerance=imageScale)
        contactGate=evaluateSurfaceContactSafety(full,spec,tolerance=contact['tolerance'])
        return dict(accepted=bool(legacy['accepted'] and contactGate['accepted']),legacyGate=legacy,contactGate=contactGate)
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
        if branch=='baseline':
            result=dict(vertices=scene['vertices'].copy(),rawCandidateVertices=scene['vertices'].copy(),solverVertices=scene['vertices'].copy(),candidateProposals=[],
                beforeContours=prepared['beforeContours'],afterContours=prepared['beforeContours'],support=prepared['support'],
                diagnostics=dict(status='unchangedCompletedCoarseBaseline',candidateAccepted=False,contactOptimizationEnabled=False))
        else:result=refineClientResidualEyes(prepared,contactSpec=spec if branch=='residualWithContact' else None,candidateGate=candidateGate)
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
            diagnostics=dict(method='completedCoarseResidualContact',branch=branch,diagnosticOnly=True,allNonSkinFixedToBaseline=True))
        if asset['changedVertexCount']!=int(changed.sum()):raise ValueError('Export changed additional rows')
        cumulative=checkFrontGeometry(initial,skin,raw['faces'],camera=prepared['camera'],fixedVertexIds=raw['staticIds'],
            maxDisplacement=float(np.ptp(initial,axis=0).max())*.07,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        skinNew=sorted(intersectionPairs(skin,raw['faces'])-initialPairs)
        local=evaluateMeshShapeConstraints(skin,guard,withJacobian=False)['values'];benefit=cavityBenefit(objective,skin)
        current=intersectionPairs(full,scene['faces']);new={name:sorted(current-pairs) for name,pairs in anchors.items()}
        coarse=evaluateResidualCoarseAlignment(prepared,full)
        fixed=bool(np.array_equal(full[~prepared['support']['movableMask']],scene['vertices'][~prepared['support']['movableMask']]))
        downstream=bool(cumulative['accepted'] and not skinNew and (local>=-1e-8).all() and benefit['accepted'])
        contactSafety=evaluateSurfaceContactSafety(full,spec,tolerance=contact['tolerance'])
        external=candidateGate(full)
        incremental=bool(fixed and coarse['accepted'] and not new['PairedBaseline'] and contactSafety['accepted'])
        absolute=bool(context['baselineReport']['initialAccepted'] and incremental and downstream and external['legacyGate']['accepted'] and not new['OriginalPureDirect'] and not new['CurrentBounded'])
        report=dict(caseId=caseId,branch=branch,status='geometryDiagnosticComplete',seconds=perf_counter()-start,referenceViewIndex=context['reference'],
            initialAccepted=context['baselineReport']['initialAccepted'],cumulativeDownstreamGate=cumulative,newSkinPairsAgainstBranchInitial=skinNew,
            finalCavityGuard=dict(accepted=bool((local>=-1e-8).all()),minimumConstraint=float(local.min()),violatedCount=int((local< -1e-8).sum())),
            finalCavityBenefit=benefit,pairedSemanticAlignment=coarse,contactSafety=contactSafety,legacyObservationGate=external['legacyGate'],allOutsideSupportExactlyFixed=fixed,
            asset=asset,geometrySafetyAccepted=absolute,incrementalSafe=incremental,absoluteEligible=absolute,
            residualCandidateAccepted=result['diagnostics'].get('accepted',False),
            finalCandidateAccepted=False,finalAdaptiveAssemblyValidated=False,textureRebaked=False,
            diagnostics=result['diagnostics'],sourceHashes=context['hashes'],
            **{'newFullPairsAgainst'+name:value for name,value in new.items()})
        report['outputHashes']={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        writeJson(directory/'report.json',report);summaries[branch]={k:report[k] for k in ('incrementalSafe','absoluteEligible','residualCandidateAccepted')}
        print(json.dumps(dict(case=caseId,branch=branch,status=result['diagnostics']['status'],**summaries[branch],seconds=report['seconds'])),flush=True)
    verify(context['hashes'])
    writeJson(root/'report.json',dict(caseId=caseId,status='complete',branches=summaries,planSha256=sha(RUN/'experimentPlan.json'),
        preparationHashes={str(p):sha(p) for p in root.iterdir() if p.is_file()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--case');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.case:runCase(args.case)
    else:parser.error('Choose --freeze or --case')
