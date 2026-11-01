"""Independent prospective checks for a single frozen dense-eye observation table.

Only reference-eye photo targets may change. Original iBUG eligibility, weights,
bindings, cameras and all non-eye observations stay immutable. Unified targets
are diagnostic proxies; original-policy failures remain explicitly disqualifying
for the original complete acceptance, even when active-policy residuals improve.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]

EYE_IDS = np.arange(36, 48)
EYES = ('imageLeft', 'imageRight')
AXIS_TOLERANCE = 64*np.finfo(np.float64).eps


def independentUnifiedTargets(photoEyes, originalTargets, *, ibugIds=EYE_IDS):
    """Intersect each original eye-axis coordinate with the original lid chain.

    The independent implementation solves a two-dimensional segment/axis
    intersection, retaining segment witnesses. No model coordinate, visibility,
    candidate result, sorting repair, clamp or extrapolation is accepted.
    """
    targets=np.asarray(originalTargets,dtype=float);ids=np.asarray(ibugIds)
    if (ids.shape!=(12,) or ids.dtype.kind not in 'iu' or set(ids)!=set(EYE_IDS)
            or targets.shape!=(12,2) or not np.isfinite(targets).all() or set(photoEyes)!=set(EYES)):
        raise ValueError('Expected the unique original twelve iBUG eye targets and both dense eye curves')
    result=targets.copy();byId={int(k):i for i,k in enumerate(ids)};evidence={}
    for name,start in zip(EYES,(36,42)):
        upper=np.asarray(photoEyes[name]['upper'],float);lower=np.asarray(photoEyes[name]['lower'],float)
        if any(p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=4096 or not np.isfinite(p).all() for p in (upper,lower)):
            raise ValueError('Original dense lids must contain 3 to 4096 finite ordered pixels')
        a0=.5*upper[0]+.5*lower[0];a1=.5*upper[-1]+.5*lower[-1];axis=a1-a0;width=np.linalg.norm(axis)
        if not np.isfinite(width) or width<=0:raise ValueError('Eye axis has zero or invalid width')
        tangent=axis/width;normal=np.array([-tangent[1],tangent[0]])
        def coordinates(points):
            offset=(points-a0)/width;return np.c_[offset@tangent,width*(offset@normal)]
        lids={lid:coordinates(points) for lid,points in (('upper',upper),('lower',lower))}
        if any(not (np.diff(value[:,0])>AXIS_TOLERANCE).all() for value in lids.values()):
            raise ValueError('Dense lid chain folds or has repeated eye-axis coordinates')
        old=coordinates(targets[[byId[k] for k in range(start,start+6)]])
        if any((np.diff(old[order,0])<=AXIS_TOLERANCE).any() for order in ([0,1,2,3],[0,5,4,3])):
            raise ValueError('Original semantic corners or lid points have reversed eye-axis order')
        def hit(lid,s):
            points={'upper':upper,'lower':lower}[lid];coordinates_=lids[lid]
            if s<coordinates_[0,0] or s>coordinates_[-1,0]:raise ValueError('Target eye-axis coordinate outside dense lid extent')
            # Intersect t·(p−a0)=s*width with the ORIGINAL directed segment.
            segment=min(int(np.searchsorted(coordinates_[:,0],s,side='right'))-1,len(points)-2)
            segment=max(segment,0)
            delta=points[segment+1]-points[segment]
            weight=(s*width-float((points[segment]-a0)@tangent))/float(delta@tangent)
            point=points[segment]+weight*delta
            return point,segment,float(weight)
        low=max(v[0,0] for v in lids.values());high=min(v[-1,0] for v in lids.values())
        if high-low<=AXIS_TOLERANCE:raise ValueError('Dense lids have no common eye-axis interval')
        knots=np.unique(np.concatenate([np.array([low,high])]+[v[(v[:,0]>low)&(v[:,0]<high),0] for v in lids.values()]))
        separations=[]
        for s in knots:
            up,_,_=hit('upper',s);down,_,_=hit('lower',s);separations.append(float((down-up)@normal))
        separations=np.asarray(separations)
        endpoint=(np.abs(knots)<=AXIS_TOLERANCE)|(np.abs(knots-1)<=AXIS_TOLERANCE)
        normalizedSeparation=separations/width
        if (normalizedSeparation < -AXIS_TOLERANCE).any() or (normalizedSeparation[~endpoint] <= AXIS_TOLERANCE).any() or not (normalizedSeparation>AXIS_TOLERANCE).any():
            raise ValueError('Original upper and lower dense lids cross or close within their common interval')
        points=[]
        for local,ibug in enumerate(range(start,start+6)):
            index=byId[ibug];corner=local in (0,3);s=float(old[local,0]);h=float(old[local,1])
            if corner:
                aligned=a0 if local==0 else a1;segment=None;weight=None;lid='corner'
            else:
                if not 0<s<1:raise ValueError('Internal semantic point lies outside the open eye-axis interval')
                lid='upper' if local in (1,2) else 'lower';aligned,segment,weight=hit(lid,s)
            result[index]=aligned
            points.append(dict(ibugId=ibug,originalTarget=targets[index].copy(),alignedTarget=aligned.copy(),
                originalAxisCoordinate=s,originalNormalCoordinate=h,lid=lid,segmentIndex=segment,segmentWeight=weight,
                cornerAxisException=corner,axisChange=float((aligned-targets[index])@tangent)/width))
        evidence[name]=dict(axisStart=a0,axisEnd=a1,axisUnit=tangent,normalUnit=normal,width=float(width),
            projectedLids=lids,overlapKnots=knots,lidSeparations=separations,points=points)
    return dict(ibugIds=ids.copy(),originalTargetPixels=targets.copy(),targetPixels=result,evidence=evidence,
        geometryIndependent=True,anatomicalTruthEstablished=False)


def observationReplacementAudit(originalObservations, alignedObservations, referenceView, independentTable):
    """Only the unique reference-view twelve-point table may be replaced."""
    if originalObservations.keys()!=alignedObservations.keys():raise ValueError('Observation metadata keys changed')
    original=np.asarray(originalObservations['points']);aligned=np.asarray(alignedObservations['points'])
    if original.ndim!=3 or original.shape[1:]!=(68,2) or aligned.shape!=original.shape or not 0<=referenceView<len(original):
        raise ValueError('Expected the same multiview 68-point observations')
    expected=original.astype(np.float64,copy=True);expected[referenceView,independentTable['ibugIds']]=independentTable['targetPixels']
    checks=dict(referenceTargetTable=np.array_equal(aligned,expected),losslessFloat64=aligned.dtype==np.dtype('float64'),
        allOtherMetadataExact=all(np.array_equal(originalObservations[k],alignedObservations[k]) for k in originalObservations if k!='points'),
        originalTargetTableSource=np.array_equal(original[referenceView,independentTable['ibugIds']],independentTable['originalTargetPixels']))
    # Equality is appropriate for table assignment after casting to the original
    # observation dtype. Independent axis arithmetic is audited before this step.
    return dict(validationPassed=bool(all(checks.values())),checks=checks,
        changedScalarCount=int(np.count_nonzero(original!=aligned)),
        referenceView=int(referenceView),onlyPhotoEyeTargetsMayChange=True)


def coarseTableReuseAudit(originalGroups, alignedGroups, table):
    if len(originalGroups)!=2 or len(alignedGroups)!=2:raise ValueError('Exactly two original coarse eye groups are required')
    byId={int(k):i for i,k in enumerate(table['ibugIds'])};checks={}
    for old,new in zip(originalGroups,alignedGroups):
        name=old['name'];checks[name+'SameKeys']=old.keys()==new.keys()
        checks[name+'EverythingExceptTargetExact']=all(np.array_equal(old[k],new.get(k)) for k in old if k!='targetPixels')
        expected=np.asarray([table['targetPixels'][byId[int(k)]] for k in old['ibugIds']])
        checks[name+'UniqueTableTargets']=np.array_equal(new['targetPixels'],expected)
    return dict(validationPassed=bool(all(checks.values())),checks=checks)


def classifyObservationCandidate(*,producerAccepted,changed,activeObservationAccepted,physicalIncrementalSafe,
                                  legacyOriginalAccepted,legacyAbsoluteEligible,finalBaselineEffectAccepted,evidenceValidated):
    """Never let the new photo proxy relabel an old acceptance failure as passed."""
    values=(producerAccepted,changed,activeObservationAccepted,physicalIncrementalSafe,legacyOriginalAccepted,
        legacyAbsoluteEligible,finalBaselineEffectAccepted,evidenceValidated)
    if any(type(v) not in (bool,np.bool_) for v in values):raise ValueError('All classification predicates must be explicit booleans')
    improved=bool(producerAccepted and changed and activeObservationAccepted and physicalIncrementalSafe and finalBaselineEffectAccepted and evidenceValidated)
    return dict(diagnosticImprovementAccepted=improved,activeObservationAccepted=bool(activeObservationAccepted),
        originalObservationPolicyAccepted=bool(legacyOriginalAccepted),legacyAbsoluteEligible=bool(legacyAbsoluteEligible),
        improvementPreservesOriginalPolicy=bool(improved and legacyOriginalAccepted),
        changingAbsoluteEligible=bool(improved and legacyAbsoluteEligible and legacyOriginalAccepted),clientReady=False,
        unifiedObservationIsGroundTruth=False)


def alignmentTableAudit(photoEyes,originalTargets,storedTable):
    """Authenticate every target against an independent segment intersection."""
    import hashlib
    independent=independentUnifiedTargets(photoEyes,originalTargets)
    saved=storedTable;e=saved['evidence'];scale=max(1.,float(np.max(np.abs(originalTargets))),
        *[float(np.max(np.abs(v))) for eye in photoEyes.values() for v in eye.values()])
    tolerance=128*np.finfo(float).eps*scale
    checks=dict(schema=saved['schemaVersion']=='denseEyeAxisTargetsV1',
        exactSourceTargets=np.array_equal(originalTargets,saved['originalTargetPixels']),
        exactIds=np.array_equal(saved['ibugIds'],EYE_IDS),
        independentTargets=np.allclose(saved['targetPixels'],independent['targetPixels'],atol=tolerance,rtol=0),
        evidenceTargets=np.array_equal(e['mappedTargets'],saved['targetPixels']),
        evidenceOriginalTargets=np.array_equal(e['originalTargets'],originalTargets),
        toleranceIdentity=e['monotonicityToleranceInEyeWidths']==AXIS_TOLERANCE and e['upperLowerSeparationToleranceInEyeWidths']==AXIS_TOLERANCE,
        noModelOrCandidate=e['geometryUsed'] is False and e['candidateProjectionUsed'] is False,
        noReorderingOrCaseTuning=e['sourceSampleOrderChanged'] is False and e['parametersSelectedFromCases'] is False)
    digest=hashlib.sha256(b'denseEyeAxisTargetsV1')
    for name in EYES:
        for lid in ('upper','lower'):
            value=np.ascontiguousarray(photoEyes[name][lid],dtype=np.float64)
            checks[name+lid+'ExactSource']=np.array_equal(value,e['sourcePhotoEyes'][name][lid])
            digest.update((name+'/'+lid+str(value.shape)).encode());digest.update(value.tobytes())
        axis=e['axes'][name];expected=independent['evidence'][name]
        for source,key in (('origin','axisStart'),('end','axisEnd'),('widthPixels','width'),('tangent','axisUnit'),('normal','normalUnit'),('overlapKnots','overlapKnots')):
            checks[name+source]=np.allclose(axis[source],expected[key],atol=tolerance,rtol=0)
        checks[name+'Separation']=np.allclose(axis['separationAtKnots'],expected['lidSeparations']/expected['width'],atol=128*np.finfo(float).eps,rtol=0)
        for lid in ('upper','lower'):
            checks[name+lid+'Axis']=np.allclose(axis['curves'][lid]['along'],expected['projectedLids'][lid][:,0],atol=128*np.finfo(float).eps,rtol=0)
            checks[name+lid+'Normal']=np.allclose(axis['curves'][lid]['normal'],expected['projectedLids'][lid][:,1]/expected['width'],atol=128*np.finfo(float).eps,rtol=0)
        for point in expected['points']:
            index=point['ibugId']-36;row=e['points'][index]
            checks['point'+str(index)+'Identity']=row['ibugId']==point['ibugId'] and row['eye']==name and row['canthusAlongException']==point['cornerAxisException']
            checks['point'+str(index)+'OriginalCoordinates']=abs(row['originalAlong']-point['originalAxisCoordinate'])<=128*np.finfo(float).eps and abs(row['originalNormal']-point['originalNormalCoordinate']/expected['width'])<=128*np.finfo(float).eps
            if point['cornerAxisException']:
                endpoint=0 if index%6==0 else 1
                checks['point'+str(index)+'Corner']=row['role']=='canthus' and row['sourceEndpoint']==endpoint and row['targetAlong']==endpoint and row['targetNormal']==0
                checks['point'+str(index)+'ExactMidpoint']=np.array_equal(saved['targetPixels'][index],expected['axisStart' if endpoint==0 else 'axisEnd'])
            else:
                checks['point'+str(index)+'Segment']=row['role']==point['lid'] and np.array_equal(row['sourceSegment'],[point['segmentIndex'],point['segmentIndex']+1]) and abs(row['segmentFraction']-point['segmentWeight'])<=1e-10
                checks['point'+str(index)+'AxisPreserved']=row['targetAlong']==row['originalAlong'] and row['alongDeltaPixels']==0 and abs(point['axisChange'])<=128*np.finfo(float).eps
                normalValue=float((saved['targetPixels'][index]-expected['axisStart'])@expected['normalUnit'])/expected['width']
                checks['point'+str(index)+'MappedNormal']=abs(row['targetNormal']-normalValue)<=128*np.finfo(float).eps
    for value in (np.asarray(originalTargets,float),saved['targetPixels']):
        value=np.ascontiguousarray(value,dtype=np.float64);digest.update(str(value.shape).encode());digest.update(value.tobytes())
    checks['sourceAndTargetIdentityDigest']=saved['tableSha256']==digest.hexdigest()
    return dict(validationPassed=bool(all(checks.values())),checks=checks,
        maximumTargetDifferencePixels=float(np.max(np.abs(saved['targetPixels']-independent['targetPixels']))),
        arithmeticComparisonTolerancePixels=tolerance,independentTable=independent,
        independentMethod='Original directed segment intersected at the original photo eye-axis coordinate; no producer mapping call.',
        anatomicalTruthEstablished=False)


def originalPolicyEvidenceAudit(rawEvidence,baseline,originalConstraints,camera,imageScale,originalLegacyGate):
    """Report original guards without pretending they selected new GN steps."""
    from validateResidualContactExperiment import coarseEyePreservationAudit
    from validateBoundaryFieldExperiment import reconstructFieldTrial
    rounds=[]
    for record in rawEvidence.get('solverEvidence',[]):
        e=record['evidence'];reference=record['fullReferenceVertices'];ids=record['fullVertexIds'];trials=[]
        for trial in e['trials']:
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],e['maxDisplacement'],trial['trialCoefficients'])
            full=reference.copy();full[ids]=local
            trials.append(dict(iteration=trial['iteration'],backtrack=trial['backtrack'],activeDecision=trial['decision'],
                originalCoarse=coarseEyePreservationAudit(baseline,full,originalConstraints,camera,imageScale=imageScale),
                originalCoarseParticipatedInActiveDecision=False,reference68ParticipatedInGn=False))
        casts=[]
        for trial in e['castTrials']:
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],e['maxDisplacement'],trial['coefficients']).astype(trial['outputDtype'])
            local[~e['activeVertexMask']]=e['referenceVertices'][~e['activeVertexMask']].astype(local.dtype)
            full=reference.copy();full[ids]=local
            casts.append(dict(scale=trial['scale'],originalCoarse=coarseEyePreservationAudit(baseline,full,originalConstraints,camera,imageScale=imageScale),
                originalCoarseParticipatedInActiveDecision=False,reference68ParticipatedInGn=False))
        rounds.append(dict(round=record['round'],trials=trials,castTrials=casts))
    outer=[]
    for proposal in rawEvidence['candidateProposals']:
        coarse=coarseEyePreservationAudit(baseline,proposal['vertices'],originalConstraints,camera,imageScale=imageScale)
        legacy=originalLegacyGate(proposal['vertices']);saved=proposal['strict']['externalCandidateGate']['originalPolicy']
        checks=dict(coarseDecision=coarse['accepted']==saved['coarseAlignment']['accepted'],originalLegacyDecision=legacy['accepted']==saved['legacyGate']['accepted'])
        for name,value in coarse['eyes'].items():
            for ours,theirs in (('baseline','before'),('candidate','after')):
                a=value[ours];b=saved['coarseAlignment']['eyes'][name][theirs]
                checks[name+ours+'Mean']=abs(a['meanPixels']-b['meanErrorPixels'])<=1e-10
                checks[name+ours+'Maximum']=abs(a['maximumPixels']-b['maximumErrorPixels'])<=1e-10
        for row in legacy['front']:
            checks['front'+row['group']]=abs(row['changePixels']-saved['legacyGate']['frontMeanChanges'][row['group']])<=1e-10
        sides=saved['legacyGate']['sideMeanChangesAgainstOriginalInitial']
        checks['sideCount']=len(sides)==len(legacy['side'])
        for old,new in zip(legacy['side'],sides):
            checks['side'+str(old['viewIndex'])]=old['viewIndex']==new['viewIndex'] and abs(old['changeAgainstOriginalInitialPixels']-new['changePixels'])<=1e-10
        outer.append(dict(round=proposal['round'],scale=proposal['scale'],validationPassed=bool(all(checks.values())),checks=checks,
            originalCoarse=coarse,originalLegacy=legacy,originalPolicyParticipatedInActiveDecision=False))
    return dict(validationPassed=all(row['validationPassed'] for row in outer),rounds=rounds,outerProposals=outer,
        scope='Counterfactual original coarse guard for GN/cast, and original coarse/front68/side/mouth for outer proposals; original front68 is not a GN gate.')


import argparse
import json
import shutil
import traceback
from scipy import sparse
from validateBoundaryFieldExperiment import (auditSolverFields,classifyBoundaryCandidate,replayGeometryEvidenceAudit,
    sameEvidence,fullApertureRasterAudit)
from validateResidualContactExperiment import (
    COARSE_EYES,coarseEyePreservationAudit,frozenAssemblyAudit,legacyResidualObservationAudit,
    historicalCollisionAudit,frozenContactGapAudit,contactSpecAudit,independentEyeSupport,
    radialWindingAudit,exportIdentityAudit,auditResidualEvidence,_scoreFieldsAgree)
from validateFeatureGeometryExperiment import (readJson,readArrays,readEvidence,sha256,fileIdentityAudit,
    loadOriginalAudit,downstreamGeometryAudit,initialGeometryAudit,_contourMasksAgree)
from validateProtectedFeatureExperiment import originalObservationMasks,originalObservationErrors
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
PLAN_SHA = 'fffd0854fbbbaf4e0cc3cd18fee75c4cd8eba22a9cd75fc1569ccdcdc5605c95'
BRANCHES = ('baseline','boundaryReplay','unifiedObservationContact')
def auditCase(case, runRoot, outputRoot):
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.legacyPreviewAsset import _load
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, extractVisibleEyeContours
    from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
    import hashlib
    root = runRoot / case; planPath = runRoot / 'experimentPlan.json'; config = readJson(planPath)
    if sha256(planPath) != PLAN_SHA or case not in config['cases']: raise ValueError('Case or prospective plan identity changed')
    context = readJson(root / 'inputContext.json'); complete = readJson(root / 'report.json')
    originalAudit = loadOriginalAudit(); system = originalAudit.priorAudit.independentSystem(case); raw = system['a']
    parentRoot, oldRoot = Path(context['completedBaselineRoot']), Path(context['oldPipelineRoot'])
    parentReport = readJson(parentRoot / 'report.json'); parentFull = readArrays(parentRoot / 'fullGeometry.npz')
    parentStates = readArrays(parentRoot / 'states.npz'); initial = parentStates['initial']; base = parentFull['vertices']
    skinIds = raw['skinIds']; reference = context['referenceSelection']['referenceViewIndex']
    cameras = readJson(Path(context['entry']['reconstruction']) / 'cameras.json'); camera = cameras[reference]
    inputs = readJson(Path(context['entry']['reconstruction']) / 'inputs.json'); imageShape = tuple(inputs[reference]['workingSize'][::-1])
    pixelScale = float(np.asarray(inputs[reference]['inputToWorking'])[0, 0]); imageScale = min(imageShape) / 512.
    photo = readArrays(oldRoot / 'observations/observations.npz')['eyelidCurvesWorking']
    photoEyes = {name: {part: photo[i, j] for j, part in enumerate(('lower', 'upper'))} for i, name in enumerate(EYES)}
    prepared = readEvidence(root / 'preparation'); spec = readEvidence(root / 'contactSpec'); legacy = readEvidence(root / 'legacyObservationState')
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    sourceAsset = _load(parentReport['asset']['archivePath'])
    scene = loadLegacyEyeBindingInputs(parentReport['asset']['archivePath'], skinIds); scene['vertices'] = base
    corners = np.concatenate([c['faceIndices'] for c in assets.components if c['name'] in ('male00SG.component001', 'male00SG.component002')])
    caseChecks = dict(planIdentity=context['planSha256'] == PLAN_SHA and complete['planSha256'] == PLAN_SHA,
        exactCompletedSkin=np.array_equal(base[skinIds], parentStates['final']),
        originalInitial=np.array_equal(initial, readArrays(ROOT / 'runs/shapePreservation20261024' / case / 'boundedWeight001/geometry.npz')['vertices']),
        exactScene=sameEvidence(scene, prepared['scene']), exactCamera=sameEvidence({k: camera[k] for k in ('intrinsics', 'rotation', 'translation')}, prepared['camera']),
        originalPhotoEyes=sameEvidence(photoEyes, prepared['photoEyes']), imageShape=tuple(prepared['imageShape']) == imageShape,
        originalPixelScale=prepared['originalPixelScale'] == pixelScale, policy=prepared['policy'] == config['residualPolicy'],
        cornerFaceIds=np.array_equal(prepared['cornerFaceIds'], corners),
        baselineVerticesHash=prepared['baselineVerticesSha256'] == hashlib.sha256(base.tobytes()).hexdigest())
    legacyRoot = Path(context['legacyReplayRoot'])
    previousPrepared = readEvidence(legacyRoot / 'preparation')
    previousSpec = readEvidence(legacyRoot / 'contactSpec')
    caseChecks['legacyPreparationExact'] = all(sameEvidence(prepared[k],previousPrepared[k]) for k in ('scene','camera','photoEyes','imageShape','cornerFaceIds','originalPixelScale','policy','support','coarseEyeConstraints','beforeCoarseAlignment','beforeContours','beforeApertureMetrics','baselineVerticesSha256'))
    caseChecks['legacyContactExact'] = sameEvidence(spec, previousSpec)
    previousOperator = sparse.load_npz(legacyRoot / 'contactOperator.npz'); currentOperator = sparse.load_npz(root / 'contactOperator.npz')
    caseChecks['legacyContactOperatorExact'] = previousOperator.shape == currentOperator.shape and (previousOperator != currentOperator).nnz == 0
    caseChecks['casePreparationHashes'] = fileIdentityAudit(complete['preparationHashes'])['accepted']
    caseChecks['parentOutputHashes'] = fileIdentityAudit({str(parentRoot / k): v for k, v in parentReport['outputHashes'].items()})['accepted']
    identities = fileIdentityAudit(context['sourceHashes']); caseChecks['inputHashes'] = identities['accepted']
    constraints = []
    for stage in parentReport['geometryRefinement']['stagedLandmarks']['stages']:
        if stage['name'] not in COARSE_EYES: continue
        e = stage['stageEvidence']; keep = np.asarray(stage['solver']['inputTargetGate']['groups'][0]['keptIndices'], int)
        constraints.append(dict(name=stage['name'], vertexIndices=skinIds[np.asarray(e['vertexIndices'], int)[keep]],
            barycentric=np.asarray(e['barycentricWeights'])[keep], **{k: np.asarray(e[k])[keep] for k in ('targetPixels', 'ibugIds', 'weights', 'confidence')}))
    caseChecks['exactOriginalSolverKeptCoarseConstraints'] = sameEvidence(constraints, prepared['coarseEyeConstraints'])
    binding = ibugClientLandmarks(loadClientLandmarks(raw['faces'], path=ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'), groups=('eyes', 'mouth'))
    observations = readArrays(oldRoot / 'geometry/photoObservations.npz')
    masks = originalObservationMasks(initial, raw['faces'], binding, cameras, inputs, observations)
    caseChecks['legacyInputIdentity'] = sameEvidence(legacy['binding'], binding) and sameEvidence(legacy['observations'], observations) and sameEvidence(legacy['masks'], masks)
    caseChecks['legacyInitialScores'] = sameEvidence(legacy['initialErrors'], originalObservationErrors(initial, binding, cameras, observations, masks))
    caseChecks['legacyCompletedScores'] = sameEvidence(legacy['completedBaselineErrors'], originalObservationErrors(base[skinIds], binding, cameras, observations, masks))
    measurementCache = {}; pairCache = {}
    def measure(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in measurementCache:
            measurementCache[key] = extractVisibleEyeContours(**{**scene, 'vertices': vertices}, camera=camera,
                photoEyes=photoEyes, imageShape=imageShape, apertureAccessoryMode='innerCornerHeadAccessories',
                apertureAccessoryFaceIds=corners, minimumEyeRasterWidth=256, maximumRoiSize=768)
        return measurementCache[key]
    def pairs(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in pairCache: pairCache[key] = intersectionPairs(vertices, scene['faces'])
        return pairCache[key]
    baselineMeasurement = measure(base)
    caseChecks['completeBaselineRaster'] = _contourMasksAgree(baselineMeasurement, prepared['beforeContours'])
    support = independentEyeSupport(scene, camera, photoEyes, baselineMeasurement, skinIds[raw['staticIds']], config['residualPolicy']['geodesicBandEyeWidthFraction'])
    caseChecks['frozenSupportMask'] = np.array_equal(support['movableMask'], prepared['support']['movableMask'])
    caseChecks['frozenFixedIds'] = np.array_equal(support['fixedVertexIds'], prepared['support']['fixedVertexIds'])
    for name in EYES:
        for key, value in support['eyes'][name].items():
            caseChecks[name + 'Support' + key] = sameEvidence(value, prepared['support']['eyes'][name].get(key))
    orientation = []; groups = []; savedOrientation = readJson(root / 'orientation.json')
    for component in assets.components:
        if component['materialId'] != config['orientation']['semanticMaterialId']: continue
        ids = np.asarray(component['faceIndices'], int)
        template = radialWindingAudit(assets.arrays['vertices'], assets.arrays['faces'], ids, config['orientation'])
        assembled = radialWindingAudit(base, scene['faces'], ids, config['orientation'])
        provenance = 'legacyMaleDirectServiceV1 ' + component['name'] + '; template and fixed assembled eye radial winding verified by frozen .02/.95 rule; open sphere patch; no skin-gap sign inference'
        if support['movableMask'].any(): groups.append(dict(name=component['name'], sourceVertexIds=np.flatnonzero(support['movableMask']), targetFaceIds=ids, normalSign=1, orientationProvenance=provenance))
        saved = next(row for row in savedOrientation if row['name'] == component['name'])
        valid = template['accepted'] and assembled['accepted'] and saved['provenance'] == provenance and np.array_equal(ids, saved['faceIds'])
        for key in ('template', 'assembled'):
            value = template if key == 'template' else assembled
            valid &= saved[key]['accepted'] == value['accepted'] and saved[key]['normalSign'] == 1
            for field in ('radius', 'residualRatio', 'minimumRadialFaceCosine'):
                valid &= abs(saved[key][field] - value[field]) <= 1e-10
        orientation.append(dict(name=component['name'], template=template, assembled=assembled, validationPassed=bool(valid)))
    caseChecks['radialWindingProvenance'] = all(row['validationPassed'] for row in orientation)
    contactPolicy = {key: float(raw['unit']) * config['contactPolicy'][key + 'Mm'] for key in ('lengthScale', 'searchRadius', 'activationDistance', 'minimumGap')}
    contactPolicy['weight'] = config['contactPolicy']['weight']
    contactProof = contactSpecAudit(spec, base, scene['faces'], groups, contactPolicy, savedOperator=sparse.load_npz(root / 'contactOperator.npz'))
    operator = contactProof.pop('reconstructedOperator'); caseChecks['contactSpec'] = contactProof['validationPassed']
    def contactGap(vertices):
        return frozenContactGapAudit(base, vertices, spec['sourceVertexIds'], spec['targetVertexIds'], spec['targetBarycentric'], spec['normals'],
            spec['minimumGap'], lengthScale=spec['lengthScale'], tolerance=config['contactPolicy']['tolerance'])
    def legacyGate(vertices):
        return legacyResidualObservationAudit(initial, base[skinIds], vertices[skinIds], raw['faces'], binding, cameras, inputs, observations, reference, masks=masks)
    pairAnchors = {'completedCoarseBaseline': pairs(base)}
    for name, arm in (('originalPureDirect', 'pureDirect'), ('currentBounded', 'bounded')):
        anchor = readArrays(oldRoot / 'geometry' / arm / 'fullGeometry.npz')
        caseChecks[name + 'FullTopology'] = np.array_equal(anchor['faces'], scene['faces'])
        pairAnchors[name] = pairs(anchor['vertices'])
    initialGate = initialGeometryAudit(system, initial)
    caseChecks['inheritedInitialDecision'] = initialGate['accepted'] == parentReport['initialAccepted']
    alignment=readEvidence(root/'observationAlignment')
    alignmentError=None; independentTable=None
    try: independentTable=independentUnifiedTargets(photoEyes,observations['points'][reference,36:48])
    except ValueError as error: alignmentError=str(error)
    alignmentValid=alignmentError is None
    alignmentChecks=dict(validDecision=alignmentValid==alignment['valid'],
        originalObservations=sameEvidence(observations,alignment['originalObservations']),masks=sameEvidence(masks,alignment['masks']),
        originalBaselineScores=sameEvidence(alignment['originalBaselineErrors'],originalObservationErrors(base[skinIds],binding,cameras,observations,masks)),
        referenceView=alignment['referenceViewIndex']==reference,
        densePath=Path(alignment['sourceDensePath'])==oldRoot/'observations/observations.npz',
        denseHash=alignment['sourceDenseSha256']==sha256(oldRoot/'observations/observations.npz'),
        noTruthClaim=alignment['anatomicalAccuracyEstablished'] is False)
    if alignmentValid:
        tableProof=alignmentTableAudit(photoEyes,observations['points'][reference,36:48],alignment['table'])
        alignmentChecks['independentTargetMapping']=tableProof['validationPassed']
        replacement=observationReplacementAudit(observations,alignment['unifiedObservations'],reference,alignment['table'])
        alignmentChecks['onlyReferenceEyeTargetsChanged']=replacement['validationPassed']
        alignmentChecks['sameTableCoarseWrapper']=sameEvidence(alignment['coarseAlignment']['mappedTable'],alignment['table'])
        alignedConstraints=alignment['coarseAlignment']['coarseGroups']
        reuse=coarseTableReuseAudit(constraints,alignedConstraints,alignment['table']);alignmentChecks['sameTableBothConsumers']=reuse['validationPassed']
        unifiedPrepared=readEvidence(root/'unifiedPreparation')
        unchanged=('scene','camera','photoEyes','imageShape','cornerFaceIds','originalPixelScale','policy','support','beforeContours','beforeApertureMetrics','baselineVerticesSha256')
        alignmentChecks['unchangedPreparationIdentity']=tuple(alignment['unchangedPreparationKeys'])==unchanged and all(sameEvidence(prepared[k],unifiedPrepared[k]) for k in unchanged)
        alignmentChecks['newCoarseConstraints']=sameEvidence(unifiedPrepared['coarseEyeConstraints'],alignedConstraints)
        # Scores must be recomputed on the SAME completed geometry under each target table.
        expectedBefore=coarseEyePreservationAudit(base,base,alignedConstraints,camera,imageScale=imageScale)
        for name, row in expectedBefore['eyes'].items():
            saved=unifiedPrepared['beforeCoarseAlignment'][name]
            alignmentChecks[name+'CompletedAnchor']=abs(saved['meanErrorPixels']-row['baseline']['meanPixels'])<=1e-10 and abs(saved['maximumErrorPixels']-row['baseline']['maximumPixels'])<=1e-10
        unifiedObservations=alignment['unifiedObservations']
        alignmentChecks['unifiedBaselineScores']=sameEvidence(alignment['unifiedBaselineErrors'],originalObservationErrors(base[skinIds],binding,cameras,unifiedObservations,masks))
        alignmentChecks['noReportedInputError']=alignment['error'] is None
    else:
        tableProof=dict(validationPassed=True,valid=False,independentError=alignmentError,producerError=alignment['error'])
        alignmentChecks['explicitInputRejection']=isinstance(alignment['error'],str) and bool(alignment['error']) and alignment['table'] is None and alignment['coarseAlignment'] is None
        alignmentChecks['noFabricatedPreparation']=not (root/'unifiedPreparation.json').exists() and not (root/'unifiedPreparation.npz').exists()
        alignmentChecks['invalidRetainsOriginalObservations']=sameEvidence(alignment['unifiedObservations'],observations)
        unifiedPrepared=prepared;alignedConstraints=constraints;unifiedObservations=observations
    caseChecks['observationAlignment']=all(alignmentChecks.values())
    originalPrepared=prepared;originalConstraints=constraints
    originalLegacyGate=legacyGate
    branches = {}
    for branch in BRANCHES:
        isUnified=branch=='unifiedObservationContact'; invalidUnified=isUnified and not alignmentValid
        prepared=unifiedPrepared if isUnified else originalPrepared
        constraints=alignedConstraints if isUnified else originalConstraints
        activeObservations=unifiedObservations if isUnified else observations
        def activeLegacyGate(vertices):
            return legacyResidualObservationAudit(initial,base[skinIds],vertices[skinIds],raw['faces'],binding,cameras,inputs,activeObservations,reference,masks=masks)
        directory = root / branch; report = readJson(directory / 'report.json')
        full = readArrays(directory / 'fullGeometry.npz'); states = readArrays(directory / 'states.npz'); evidence = readEvidence(directory / 'evidence')
        rawEvidence = readEvidence(directory / 'rawRefinement')
        final = full['vertices']; outputs = fileIdentityAudit({str(directory / k): v for k, v in report['outputHashes'].items()})
        checks = dict(caseAndBranch=report['caseId'] == case and report['branch'] == branch,
            allOutputHashes=outputs['accepted'], sameInputHashes=report['sourceHashes'] == context['sourceHashes'],
            referenceView=report['referenceViewIndex'] == reference, fullTopology=np.array_equal(full['faces'], scene['faces']),
            initialState=np.array_equal(states['initial'], initial), completedState=np.array_equal(states['completedCoarseBaseline'], base[skinIds]),
            baselineState=np.array_equal(states['baseline'], base[skinIds]), finalState=np.array_equal(states['final'], final[skinIds]),
            evidenceFinal=np.array_equal(evidence['vertices'], final), sameFrozenSupport=sameEvidence(evidence['support'], prepared['support']),
            noClientPromotion=report['finalCandidateAccepted'] is False and report['finalAdaptiveAssemblyValidated'] is False and report['textureRebaked'] is False)
        fixed = frozenAssemblyAudit(scene, {**scene, 'vertices': final}, support['movableMask'], staticFullIds=skinIds[raw['staticIds']])
        checks['fixedAssembly'] = fixed['accepted']
        mouthFullIds = skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]
        checks['realMouthExactlyFixed'] = np.array_equal(final[mouthFullIds], base[mouthFullIds])
        archivePath = Path(report['asset']['archivePath']); archive = _load(archivePath)
        export = exportIdentityAudit(sourceAsset, archive, readArrays(directory / 'exportIdentity.npz'), base, final, support['movableMask'], raw, report['asset'])
        checks['exportIdentity'] = export['validationPassed']; checks['archiveHash'] = sha256(archivePath) == report['asset']['sha256']
        downstream = downstreamGeometryAudit(system, initial, final[skinIds], camera)
        coarse = coarseEyePreservationAudit(base, final, constraints, camera, imageScale=imageScale)
        contacts = contactGap(final); legacyReport = originalLegacyGate(final); activeLegacyReport=activeLegacyGate(final)
        originalCoarse=coarseEyePreservationAudit(base,final,originalConstraints,camera,imageScale=imageScale)
        collision = historicalCollisionAudit(pairAnchors['completedCoarseBaseline'], pairs(final), pairAnchors, scene['materialIds'])
        for name, suffix in (('originalPureDirect', 'OriginalPureDirect'), ('currentBounded', 'CurrentBounded'), ('completedCoarseBaseline', 'PairedBaseline')):
            checks[name + 'NewPairs'] = collision['anchors'][name]['newPairs'] == [tuple(v) for v in report['newFullPairsAgainst' + suffix]]
        checks['skinNewPairs'] = downstream['cumulativeDownstream']['collisions']['newPairs'] == [tuple(v) for v in report['newSkinPairsAgainstBranchInitial']]
        checks['downstreamDecision'] = downstream['cumulativeDownstream']['accepted'] == bool(report['cumulativeDownstreamGate']['accepted'] and not report['newSkinPairsAgainstBranchInitial'])
        checks['cavityDecision'] = downstream['originalCavity']['accepted'] == bool(report['finalCavityGuard']['accepted'] and report['finalCavityBenefit']['accepted'])
        checks['coarseDecision'] = coarse['accepted'] == report['pairedSemanticAlignment']['accepted']
        checks['contactDecision'] = contacts['accepted'] == report['contactSafety']['accepted']
        checks['legacyDecision'] = legacyReport['accepted'] == report['legacyObservationGate']['accepted']
        incremental = bool(fixed['accepted'] and coarse['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs'])
        originalPolicyAccepted=bool(originalCoarse['accepted'] and legacyReport['accepted'])
        activePolicyAccepted=None if invalidUnified else bool(coarse['accepted'] and activeLegacyReport['accepted'])
        absolute = bool(not invalidUnified and initialGate['accepted'] and incremental and downstream['accepted'] and originalPolicyAccepted and activePolicyAccepted and collision['allHistoricalAnchorsAccepted'])
        checks['oldPolicyDecision']=originalPolicyAccepted==report['originalObservationPolicyAccepted']
        checks['activePolicyDecision']=activePolicyAccepted==report['activeObservationPolicyAccepted']
        checks['originalCoarseDecision']=originalCoarse['accepted']==report['originalCoarseAlignment']['accepted']
        checks['activeCoarseDecision']=coarse['accepted']==report['activeCoarseAlignment']['accepted']
        checks['activeLegacyDecision']=activeLegacyReport['accepted']==report['activeLegacyObservationGate']['accepted']
        checks['legacyAbsoluteDecision']=absolute==report['legacyAbsoluteEligible']
        checks['alignmentValidity']=report['observationAlignmentValid']==alignmentValid and report['observationPolicyChangedForDiagnostic']==bool(isUnified and alignmentValid)
        checks['tableIdentity']=report['observationTableSha256']==(alignment['table']['tableSha256'] if alignmentValid else None)
        checks['invalidAlignmentErrorRecorded']=report['observationAlignmentError']==alignment['error']
        checks['incrementalDecision'] = incremental == report['incrementalSafe']
        checks['absoluteDecision'] = absolute == report['absoluteEligible'] == report['geometrySafetyAccepted']
        effect = fullApertureRasterAudit(baselineMeasurement, measure(final), photoEyes, originalPixelScale=pixelScale)
        eyeProof = auditResidualEvidence('baseline' if branch == 'baseline' or invalidUnified else 'residualWithContact', prepared, rawEvidence, base, rawEvidence['vertices'], scene, support, constraints, camera, imageScale,
            baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap, activeLegacyGate, spec, operator)
        checks['residualEvidence'] = eyeProof['validationPassed']
        checks['rawProducerDecision'] = eyeProof['residualCandidateAccepted'] == report['rawRefinementAccepted']
        effectRaw = fullApertureRasterAudit(baselineMeasurement, measure(rawEvidence['vertices']), photoEyes, originalPixelScale=pixelScale)
        savedEffect = readJson(directory / 'finalEffectGate.json')
        checks['finalEffectReportIdentity'] = sameEvidence(savedEffect, report['finalEffectGate'])
        checks['finalEffectDecision'] = effectRaw['accepted'] == savedEffect['accepted']
        checks['finalEffectRawMetrics'] = (not effectRaw['effectDefined'] or _scoreFieldsAgree(effectRaw['metrics']['baseline'], savedEffect['baseline']) and _scoreFieldsAgree(effectRaw['metrics']['candidate'], savedEffect['candidate']))
        fallback = isUnified and alignmentValid and not effectRaw['accepted']
        checks['fallbackPolicy'] = fallback == savedEffect['fallbackToExactBaseline']
        checks['finalOutputPolicy'] = np.array_equal(final, base if fallback else rawEvidence['vertices'])
        checks['effectOnlyFallbackLeavesRawAndSolver'] = all(np.array_equal(evidence[k],rawEvidence[k]) for k in ('rawCandidateVertices','solverVertices')) and sameEvidence(evidence['candidateProposals'],rawEvidence['candidateProposals'])
        producerAccepted = bool(eyeProof['residualCandidateAccepted'] and not fallback)
        checks['outputProducerDecision'] = producerAccepted == report['residualCandidateAccepted']
        changed = not np.array_equal(final,base)
        checks['changedDecision'] = changed == report['geometryChanged']
        fieldProof = auditSolverFields(branch,prepared,rawEvidence,camera,base,scene,support,constraints,imageScale,spec,operator,measure,config) if branch != 'baseline' and not invalidUnified else dict(validationPassed=not rawEvidence['solverEvidence'],checks={'noSolver':not rawEvidence['solverEvidence']},rounds=[])
        checks['solverFieldsAndTrials'] = fieldProof['validationPassed']
        checks['solverSidecarIdentity'] = sameEvidence(readEvidence(directory / 'solverEvidence'),rawEvidence['solverEvidence'])
        replay = None
        if branch == 'boundaryReplay':
            oldEvidence = readEvidence(Path(context['legacyReplayRoot']) / 'boundaryFieldContact/rawRefinement')
            replay = replayGeometryEvidenceAudit(oldEvidence,rawEvidence)
            replaySaved = readJson(directory / 'replayMatch.json')
            checks['legacyReplayExact'] = replay['validationPassed'] and replaySaved['accepted']
            checks['legacyReplayReportIdentity'] = sameEvidence(replaySaved,report['replayMatch'])
        originalPolicyProof=originalPolicyEvidenceAudit(rawEvidence,base,originalConstraints,camera,imageScale,originalLegacyGate) if isUnified and alignmentValid else dict(validationPassed=True,rounds=[],outerProposals=[],notApplicable=True)
        checks['originalCounterfactualAndOuterEvidence']=originalPolicyProof['validationPassed']
        if invalidUnified:
            checks['invalidObservationSkip']=rawEvidence['diagnostics']['status']=='skippedInvalidUnifiedObservation' and evidence['diagnostics']['status']=='skippedInvalidUnifiedObservation' and not rawEvidence['candidateProposals'] and not rawEvidence['solverEvidence'] and np.array_equal(rawEvidence['vertices'],base) and np.array_equal(final,base)
            checks['noInventedUnifiedGateEvaluation']=report['activeObservationPolicyAccepted'] is None
        classification = classifyBoundaryCandidate(producerAccepted=producerAccepted,geometryChanged=changed,
            incrementalSafe=incremental,absoluteEligible=absolute,finalBaselineEffect=effectRaw,evidenceValidated=bool(all(checks.values())))
        policyClassification=classifyObservationCandidate(producerAccepted=producerAccepted,changed=changed,
            activeObservationAccepted=bool(activePolicyAccepted),physicalIncrementalSafe=bool(fixed['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs']),
            legacyOriginalAccepted=originalPolicyAccepted,legacyAbsoluteEligible=absolute,finalBaselineEffectAccepted=bool(effectRaw['accepted'] and effectRaw['effectDefined']),evidenceValidated=bool(all(checks.values())))
        classification.update(policyClassification)
        classification['activeObservationPolicyAccepted']=activePolicyAccepted
        classification['activeObservationPolicyEvaluated']=not invalidUnified
        checks['diagnosticImprovementDecision']=policyClassification['diagnosticImprovementAccepted']==report['improvementAccepted']
        checks['improvementDecision'] = classification['improvementAccepted'] == report['improvementAccepted']
        checks['changingAbsoluteDecision'] = classification['changingAbsoluteEligible'] == report['changingAbsoluteEligible']
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks,
            incrementalSafe=incremental, absoluteEligible=absolute, producerAccepted=producerAccepted,
            residualCandidateAccepted=producerAccepted, rawRefinementAccepted=eyeProof['residualCandidateAccepted'],
            requestedEffectAccepted=classification['requestedEffectAccepted'], improvementAccepted=classification['improvementAccepted'],
            changingAbsoluteEligible=classification['changingAbsoluteEligible'], evidenceValidated=bool(all(checks.values())),
            classification=classification, finalEffectGate=effectRaw, finalOutputEffect=effect,
            originalObservationPolicyAccepted=originalPolicyAccepted,activeObservationPolicyAccepted=activePolicyAccepted,legacyAbsoluteEligible=absolute,
            observationAlignmentValid=alignmentValid,originalPolicyEvidence=originalPolicyProof,originalCoarseAlignment=originalCoarse,activeCoarseAlignment=coarse,activeLegacyPhotoGate=activeLegacyReport,
            gnEvidence=fieldProof, replayEvidence=replay,
            unchangedRealMouth=np.array_equal(final[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]], base[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]]),
            fixedAssembly=fixed, exportIdentity=export, downstream=downstream,
            contactEnvelope=contacts, legacyPhotoGate=legacyReport, fullCollisionAnchors=collision,
            fullApertureEffect=effect, residualEvidence=eyeProof, producerOutputs=outputs,
            reportSha256=sha256(directory / 'report.json'), archiveSha256=sha256(archivePath), clientReady=False)
    caseChecks['inputHashesAfterAudit'] = fileIdentityAudit(context['sourceHashes'])['accepted']
    result = dict(caseId=case, validationPassed=bool(all(caseChecks.values()) and all(v['validationPassed'] for v in branches.values())),
        checks=caseChecks, initialGate=initialGate, contactSpecification=contactProof, orientation=orientation,
        observationAlignment=dict(validationPassed=bool(all(alignmentChecks.values())),valid=alignmentValid,checks=alignmentChecks,tableProof=tableProof),
        originalInputIdentities=identities, branches=branches, planSha256=PLAN_SHA, validatorSha256=sha256(__file__),
        candidateSolvesPerformed=False, clientReady=False,
        scope='Same completed coarse baseline and fixed attachments. Contact proxy and automatic photo curves are not anatomical ground truth; inherited failures stay failures.')
    outputRoot.mkdir(parents=True, exist_ok=True); writeJson(outputRoot / f'{case}.json', result)
    return result



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs='+',required=True)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/eyeObservation20261024')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/eyeObservation20261024/independentAudit')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'tests/testEyeObservationValidation.py',ROOT/'tools/validateResidualContactExperiment.py',ROOT/'tools/validateBoundaryFieldExperiment.py',
        ROOT/'tools/validateFeatureGeometryExperiment.py',ROOT/'tools/validateProtectedFeatureExperiment.py']
    snapshots={}
    for path in files:
        target=args.output/'implementation'/sha256(path)/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(path,target)
        snapshots[str(path)]=dict(snapshotPath=str(target),sha256=sha256(path))
    writeJson(args.output/'auditImplementation.json',dict(planSha256=PLAN_SHA,validatorSha256=sha256(__file__),snapshots=snapshots,
        hashes={str(p):sha256(p) for p in files},candidateSolvesPerformed=False,
        scope='Independent observation mapping, original and active policy, basis/IRLS/GN/cast/contact/mask/full scene audit. No fitting or target inference; old visibility/intersection predicates reused.'))
    results=[]; failures=[]
    for case in args.case:
        try:
            result=auditCase(case,args.run_root,args.output);results.append(result)
            print(json.dumps(dict(caseId=case,validationPassed=result['validationPassed'],improvements=[k for k,v in result['branches'].items() if v['improvementAccepted']])),flush=True)
        except Exception as error:
            failure=dict(caseId=case,validationPassed=False,toolException=True,exceptionType=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            failures.append(failure);writeJson(args.output/(case+'.toolException.json'),failure);print(json.dumps(failure),flush=True)
    counts={branch:{key:sum(row['branches'][branch][key] for row in results) for key in
        ('validationPassed','incrementalSafe','absoluteEligible','rawRefinementAccepted','requestedEffectAccepted','improvementAccepted','changingAbsoluteEligible')} for branch in BRANCHES}
    summary=dict(validationPassed=not failures and all(r['validationPassed'] for r in results),caseCount=len(results),counts=counts,
        toolExceptionCount=len(failures),failures=failures,cases=[dict(caseId=r['caseId'],validationPassed=r['validationPassed'],
            report=str(args.output/(r['caseId']+'.json')),reportSha256=sha256(args.output/(r['caseId']+'.json'))) for r in results],
        validatorSha256=sha256(__file__),planSha256=PLAN_SHA,clientReady=False,expectedRejectionsAreToolFailures=False)
    writeJson(args.output/'validationSummary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    if not summary['validationPassed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
