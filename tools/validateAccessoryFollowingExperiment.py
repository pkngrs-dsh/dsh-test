"""Independent audit of template-bound eye accessory following.

The preceding frozen eye observations, skin solver and gate stages are retained.
Only six explicit template parts may receive deterministic total skin displacement.
Full-scene geometry and collision rules retain every surface. Qualification under
this declared accessory policy stays distinct from the older all-accessories-fixed
policy. This tool reconstructs evidence; it never fits or changes a candidate.
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

# This is a template-level allowlist, never inferred from a patient's result.
ACCESSORY_COMPONENTS = {
    'male00SG.component001': ('headAccessory', 'rightEye', 'canthus', 0, 12, 16),
    'male00SG.component002': ('headAccessory', 'leftEye', 'canthus', 0, 12, 16),
    'male00SG3.component000': ('lowerEyelash', 'leftEye', 'lower', 3, 54, 80),
    'male00SG3.component001': ('upperEyelash', 'leftEye', 'upper', 3, 66, 100),
    'male00SG3.component002': ('lowerEyelash', 'rightEye', 'lower', 3, 54, 80),
    'male00SG3.component003': ('upperEyelash', 'rightEye', 'upper', 3, 66, 100),
}


def templateAccessoryWhitelist(templateArrays, components):
    """Authenticate the six original parts without selecting from a candidate."""
    vertices = np.asarray(templateArrays['vertices'])
    faces = np.asarray(templateArrays['faces'])
    skin = np.asarray(templateArrays['nicpToFullIds'])
    byName = {row['name']: row for row in components}
    if len(byName) != len(components) or not set(ACCESSORY_COMPONENTS) <= byName.keys():
        raise ValueError('Original unique six-component template metadata is required')
    selected = []; reports = []
    for name, (label, eye, lid, material, vertexCount, faceCount) in ACCESSORY_COMPONENTS.items():
        component = byName[name]
        ids = np.asarray(component['vertexIndices'])
        faceIds = np.asarray(component['faceIndices'])
        if (ids.dtype.kind not in 'iu' or faceIds.dtype.kind not in 'iu'
                or ids.shape != (vertexCount,) or faceIds.shape != (faceCount,)
                or len(np.unique(ids)) != len(ids) or len(np.unique(faceIds)) != len(faceIds)
                or np.any(ids < 0) or np.any(ids >= len(vertices))
                or np.any(faceIds < 0) or np.any(faceIds >= len(faces))):
            raise ValueError('Original accessory index identity or count changed: ' + name)
        direction = 1 if eye == 'leftEye' else -1
        checks = dict(label=component['label'] == label, material=component['materialId'] == material,
            faceMaterial=np.all(templateArrays['materialIds'][faceIds] == material),
            componentFacesExactlyUseVertices=np.array_equal(np.unique(faces[faceIds]), np.sort(ids)),
            noSkinVertexOverlap=not np.isin(ids, skin).any(),
            originalTemplateSide=np.all(vertices[ids, 0] * direction > 0))
        if not all(checks.values()): raise ValueError('Accessory semantics changed: ' + name)
        selected.extend(ids.tolist())
        reports.append(dict(name=name, eye=eye, lid=lid, vertexIds=ids.copy(), faceIds=faceIds.copy(), checks=checks))
    if len(set(selected)) != len(selected): raise ValueError('Whitelisted accessories share vertices')
    return dict(vertexIds=np.sort(selected), components=reports,
        scope='Fixed original template identifiers; x-positive is the template left-eye label.')


def whitelistAssemblyAudit(baselineScene, candidateScene, movableSkinMask,
                           followerVertexIds, *, staticFullIds=()):
    """Permit only the frozen skin domain and explicit six accessory parts."""
    from validateFeatureGeometryExperiment import sceneIdentityAudit
    before = np.asarray(baselineScene['vertices']); after = np.asarray(candidateScene['vertices'])
    skinIds = np.asarray(baselineScene['nicpToFullIds'])
    movable = np.asarray(movableSkinMask); followers = np.asarray(followerVertexIds)
    if (movable.shape != (len(before),) or movable.dtype.kind != 'b'
            or followers.ndim != 1 or followers.dtype.kind not in 'iu'
            or len(np.unique(followers)) != len(followers)
            or np.any(followers < 0) or np.any(followers >= len(before))):
        raise ValueError('Expected original boolean skin support and unique accessory IDs')
    skin = np.zeros(len(before), bool); skin[skinIds] = True
    if (movable & ~skin).any() or skin[followers].any():
        raise ValueError('Follower and independent skin domains must be disjoint')
    allowed = movable.copy(); allowed[followers] = True
    fixedNonSkin = ~skin; fixedNonSkin = fixedNonSkin.copy(); fixedNonSkin[followers] = False
    report = sceneIdentityAudit(baselineScene, candidateScene,
        fixedVertexIds=np.flatnonzero(~allowed), protectedVertexIds=np.asarray(staticFullIds, dtype=int))
    report['checks']['allOtherAccessoriesExactlyFixed'] = np.array_equal(before[fixedNonSkin], after[fixedNonSkin])
    report['accepted'] = bool(all(report['checks'].values()))
    report.update(followerVertexIds=followers.copy(), effectiveFixedVertexIds=np.flatnonzero(~allowed),
        allNonSkinExactlyFixed=bool(np.array_equal(before[~skin], after[~skin])),
        allNonSkinFixedIsRequired=False, movableMask=allowed,
        scope='Only the explicit canthus and eyelash allowlist may follow skin; eyes, mouth parts, brows and remote skin stay exact.')
    return report


def independentAccessoryPositions(baselineVertices, candidateVertices, followerVertexIds,
                                 sourceVertexIndices, barycentricWeights):
    """Apply total skin displacement from the fixed baseline, with no new DOFs."""
    baseline = np.asarray(baselineVertices); candidate = np.asarray(candidateVertices)
    followers = np.asarray(followerVertexIds); sources = np.asarray(sourceVertexIndices)
    weights = np.asarray(barycentricWeights)
    if (baseline.ndim != 2 or baseline.shape[1] != 3 or candidate.shape != baseline.shape
            or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
            or followers.ndim != 1 or followers.dtype.kind not in 'iu'
            or sources.shape != (len(followers), 3) or sources.dtype.kind not in 'iu'
            or weights.shape != sources.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights > 1)
            or not np.allclose(weights.sum(1), 1, atol=1e-12, rtol=0)
            or np.any(sources < 0) or np.any(sources >= len(baseline))
            or np.any(followers < 0) or np.any(followers >= len(baseline))
            or len(np.unique(followers)) != len(followers)
            or np.isin(sources, followers).any()):
        raise ValueError('Expected independent finite baseline, source triangles and convex follower weights')
    delta = candidate[sources] - baseline[sources]
    # Sum each corner explicitly, independently of the producer sparse operator.
    displacement = np.zeros((len(followers), 3))
    for corner in range(3): displacement += weights[:, corner, None] * delta[:, corner]
    result = baseline[followers].copy()
    changed=np.any(displacement!=0,axis=1)
    result[changed] += displacement[changed]
    return result


def independentlyMapFollowing(spec, candidate):
    result = np.asarray(candidate).copy()
    for part in spec['binding']['components']:
        result[part['targetVertexIds']] = independentAccessoryPositions(spec['baselineVertices'], candidate,
            part['targetVertexIds'], part['sourceVertexIndices'], part['barycentricWeights'])
    return result


def mappedCoordinatesAgree(expected, actual):
    """Independent sums may differ by rounding; fixed rows are checked exactly."""
    expected = np.asarray(expected); actual = np.asarray(actual)
    tolerance = 64*np.finfo(float).eps*max(1., float(np.abs(expected).max(initial=0)))
    return bool(expected.shape == actual.shape and np.allclose(expected, actual, atol=tolerance, rtol=0))


def followingRelationAudit(spec, candidate):
    expected = independentlyMapFollowing(spec, candidate)
    followers = spec['followerVertexIds']; fixed = spec['effectiveFixedVertexIds']
    checks = dict(followerPositionsFromTotalSkin=mappedCoordinatesAgree(expected[followers], candidate[followers]),
        allOtherFixedExact=np.array_equal(candidate[fixed], spec['baselineVertices'][fixed]),
        originalSkinValuesKeptExact=np.array_equal(expected[spec['skinVertexIds']], candidate[spec['skinVertexIds']]))
    return dict(validationPassed=bool(all(checks.values())), checks=checks,
        maximumFollowerDifference=float(np.abs(expected[followers]-candidate[followers]).max(initial=0)),
        independentAccessoryDegreesOfFreedom=0, reference='Frozen completed full assembly, never the previous round')


def evidenceIdentity(value):
    """Independent serializer for the published binary evidence digest contract."""
    import hashlib
    import json
    digest = hashlib.sha256()
    def consume(item):
        if isinstance(item, np.ndarray):
            digest.update(b'array\0'+str(item.dtype).encode()+repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item): digest.update(str(key).encode()+b'\0'); consume(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b'list\0')
            for member in item: consume(member)
        elif isinstance(item, np.generic): consume(item.item())
        else: digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode()+b'\0')
    consume(value)
    return digest.hexdigest()


def accessoryBindingAudit(binding, arrays, components, policy):
    """Rebuild side/lid domains and nearest surface rows without its builder."""
    import hashlib
    from faceAppearance.eyeSocketVisibility import buildVisibleEyeRimBinding
    from validateResidualContactExperiment import _closestFeatureIndependent
    whitelist = templateAccessoryWhitelist(arrays, components)
    vertices, faces = arrays['vertices'], arrays['faces']
    byName = {part['name']: part for part in components}
    eyes = sorted((part for part in components if part['label'] in ('leftEye', 'rightEye')),
        key=lambda part: vertices[part['vertexIndices'], 0].mean())
    centerX = float(np.mean([vertices[part['vertexIndices'], 0].mean() for part in eyes]))
    digest = hashlib.sha256(b'numberedOrientedTriangleTopologyV1\0')
    digest.update(np.asarray([len(vertices), len(faces)], dtype='<i8').tobytes())
    digest.update(np.asarray(faces, dtype='<i8').tobytes())
    checks = dict(schema=binding['schemaVersion']=='templateEyeAccessoryBindingV1',
        vertexCount=binding['vertexCount']==len(vertices),
        coordinateFrame=binding['coordinateFrame']=='originalClientTemplateCentimetersYUpZForward',
        originalVertices=np.array_equal(binding['templateVertices'], vertices),
        originalFaces=np.array_equal(binding['faces'], faces),
        originalSkin=np.array_equal(binding['skinVertexIds'], arrays['nicpToFullIds']),
        originalWhitelist=np.array_equal(binding['followerVertexIds'], whitelist['vertexIds']),
        originalComponentNames=[p['componentName'] for p in binding['components']]==sorted(ACCESSORY_COMPONENTS),
        commonPolicy=binding['policy']==policy, sideDivision=abs(binding['sideDivisionX']-centerX)<=1e-12,
        arrayHashes=binding['templateArraySha256']=={k:evidenceIdentity(np.asarray(v)) for k,v in arrays.items()},
        componentHash=binding['templateComponentsSha256']==evidenceIdentity(components),
        topologyHash=binding['topologySha256']==digest.hexdigest(),
        bindingDigest=binding['bindingSha256']==evidenceIdentity({k:v for k,v in binding.items() if k!='bindingSha256'}),
        geometryIndependent=binding['diagnostics']['candidateDependent'] is False,
        photoIndependent=binding['diagnostics']['photoDependent'] is False,
        noNewParameters=binding['diagnostics']['additionalDegreesOfFreedom']==0)
    domains = {}; domainReports = []
    for index, eye in enumerate(eyes):
        saved = binding['eyeDomains'][index]
        rim = buildVisibleEyeRimBinding(arrays, eye, resolution=policy['rimResolution'])
        columns = np.rint(rim['contourSamplePixels'][:,0]).astype(int)
        endpoints = [np.flatnonzero(rim['eyeMask'][:,x])[[0,-1]] for x in columns]
        endpoints = np.asarray(endpoints)
        upperDistance = np.abs(rim['contourSamplePixels'][:,1]-endpoints[:,0])
        lowerDistance = np.abs(rim['contourSamplePixels'][:,1]-endpoints[:,1])
        upper, lower = upperDistance<lowerDistance, lowerDistance<upperDistance
        x = rim['restPoints'][:,0]; width = float(x.max()-x.min()); sign = -1 if index==0 else 1
        inward = x.max()-x if index==0 else x-x.min()
        inner = inward <= policy['innerCornerWidthFraction']*width
        allowed = dict(upper=np.unique(rim['fullFaceIds'][upper]), lower=np.unique(rim['fullFaceIds'][lower]),
            innerCorner=np.unique(rim['fullFaceIds'][inner & (upper|lower)]))
        valid = dict(originalRim=sameEvidence(saved['visibleRimEvidence'],rim),
            component=saved['componentName']==eye['name'] and saved['name']==eye['label'],
            sideSign=saved['sideSign']==sign, center=np.array_equal(saved['center'],vertices[eye['vertexIndices']].mean(0)),
            width=saved['visibleEyeWidth']==width, bounds=np.array_equal(saved['visibleRimXBounds'],[x.min(),x.max()]),
            upperLowerDisjoint=not np.intersect1d(allowed['upper'],allowed['lower']).size,
            sourceDomains=sameEvidence(allowed,saved['sourceFaceDomains']))
        expected = dict(columnIndices=columns, upperMaskRows=endpoints[:,0], lowerMaskRows=endpoints[:,1],
            upperDistances=upperDistance, lowerDistances=lowerDistance, upperSampleMask=upper,
            lowerSampleMask=lower, innerCornerSampleMask=inner, innerCornerDistances=inward)
        valid.update({key:np.array_equal(value,saved[key]) for key,value in expected.items()})
        for role, ids in allowed.items():
            valid[role+'OnlySameSideSkin'] = bool(len(ids) and np.isin(faces[ids],arrays['nicpToFullIds']).all()
                and ((vertices[faces[ids],0]-centerX)*sign>0).all())
        domains[eye['label']] = dict(allowed=allowed, width=width, center=vertices[eye['vertexIndices']].mean(0), sign=sign)
        domainReports.append(dict(eye=eye['label'], validationPassed=bool(all(valid.values())), checks=valid))
    componentReports = []
    for part in binding['components']:
        name=part['componentName']; original=byName[name]
        label,eye,lid,material,_,_=ACCESSORY_COMPONENTS[name]
        role='innerCorner' if lid=='canthus' else lid
        domain=domains[eye]; allowed=domain['allowed'][role]; triangles=vertices[faces[allowed]]
        valid=dict(semanticRole=part['eyeName']==eye and part['role']==role and part['label']==label,
            material=part['materialId']==material,
            exactComponentVertices=np.array_equal(part['targetVertexIds'],original['vertexIndices']),
            exactComponentFaces=np.array_equal(part['targetFaceIds'],original['faceIndices']),
            exactAllowedFaces=np.array_equal(part['allowedFaceIds'],allowed),
            sourceFacesInDomain=np.isin(part['sourceFaceIds'],allowed).all(),
            originalSourceTriangles=np.array_equal(part['sourceVertexIndices'],faces[part['sourceFaceIds']]),
            noCrossEye=((vertices[part['targetVertexIds'],0]-centerX)*domain['sign']>0).all())
        if role=='innerCorner': valid['innerHalfOfEye']=((vertices[part['targetVertexIds'],0]-domain['center'][0])*domain['sign']<0).all()
        rows=[]; tolerance=policy['ambiguityToleranceEyeWidthFraction']*domain['width']
        for index,target in enumerate(part['targetVertexIds']):
            bary,distances,_ = _closestFeatureIndependent(vertices[target],triangles)
            closest=float(distances.min()); tied=np.flatnonzero(distances<=closest+tolerance)
            chosen=int(tied[0]); source=faces[allowed[chosen]]; weights=bary[chosen]
            ids=np.unique(faces[allowed[tied]])
            projectedRows=np.zeros((len(tied),len(ids)))
            for row,face in enumerate(tied): projectedRows[row,np.searchsorted(ids,faces[allowed[face]])]=bary[face]
            equivalents=bool(np.max(np.abs(projectedRows-projectedRows[:1]))<=policy['ambiguityToleranceEyeWidthFraction'])
            expectedPoint=(vertices[source]*weights[:,None]).sum(0)
            rowChecks=dict(chosenFace=int(part['sourceFaceIds'][index])==int(allowed[chosen]),
                tiedFaceSet=np.array_equal(part['equivalentTieFaceIds'][index],allowed[tied]),
                tiedRowsEquivalent=equivalents, barycentric=np.allclose(part['barycentricWeights'][index],weights,atol=1e-10,rtol=0),
                closestPoint=np.allclose(part['closestRestPoints'][index],expectedPoint,atol=1e-10,rtol=0),
                distance=abs(float(part['bindingDistances'][index])-closest)<=1e-10,
                boundedDistance=closest<=policy['maximumBindingDistanceEyeWidthFraction']*domain['width'])
            rows.append(dict(targetVertexId=int(target),validationPassed=bool(all(rowChecks.values())),checks=rowChecks))
        valid['allSurfaceRows']=all(row['validationPassed'] for row in rows)
        componentReports.append(dict(componentName=name,validationPassed=bool(all(valid.values())),checks=valid,rows=rows))
    checks['eyeDomains']=all(row['validationPassed'] for row in domainReports)
    checks['allComponents']=all(row['validationPassed'] for row in componentReports)
    return dict(validationPassed=bool(all(checks.values())),checks=checks,eyeDomains=domainReports,components=componentReports,
        sharedPrimitive='Read-only original-template material visibility raster and rim samples',
        independentMethod='Rebuild same-side lid domains; dominant-axis projection plus all closed edges; compare sparse global interpolation rows for ties',
        anatomicalTruthEstablished=False)


def followingSpecificationAudit(spec, binding, base, scene, support, savedOperator):
    from scipy import sparse
    count=len(base); skin=scene['nicpToFullIds']; followers=binding['followerVertexIds']
    matrix=sparse.lil_matrix((count,count)); matrix[skin,skin]=1
    for part in binding['components']:
        for target,sources,weights in zip(part['targetVertexIds'],part['sourceVertexIndices'],part['barycentricWeights']):
            for source,weight in zip(sources,weights): matrix[target,source]=weight
    matrix=matrix.tocsr(); matrix.eliminate_zeros()
    checks=dict(schema=spec['schemaVersion']=='completedEyeAccessoryFollowingV1',
        exactBinding=sameEvidence(spec['binding'],binding), baseline=np.array_equal(spec['baselineVertices'],base),
        faces=np.array_equal(spec['faces'],scene['faces']),skin=np.array_equal(spec['skinVertexIds'],skin),
        fixed=np.array_equal(spec['originalFixedVertexIds'],support['fixedVertexIds']),
        movableSkin=np.array_equal(spec['movableSkinVertexIds'],np.flatnonzero(support['movableMask'])),
        followers=np.array_equal(spec['followerVertexIds'],followers),
        effectiveFixed=np.array_equal(spec['effectiveFixedVertexIds'],np.setdiff1d(support['fixedVertexIds'],followers)),
        digest=spec['specSha256']==evidenceIdentity({k:v for k,v in spec.items() if k!='specSha256'}),
        baselineIdentity=np.array_equal(independentlyMapFollowing(spec,base),base),
        scalarOperator=savedOperator.shape==matrix.shape and (savedOperator!=matrix).nnz==0,
        zeroFollowerInputColumns=matrix[:,followers].nnz==0,
        noIndependentParameters=spec['diagnostics']['independentAccessoryParameters']==0)
    return dict(validationPassed=bool(all(checks.values())),checks=checks,reconstructedOperator=matrix)


def skippedObservationReplayAudit(oldEvidence, newEvidence, baseline, support, beforeContours, error):
    """Authenticate the actual skip schema without inventing solver outputs.

    An invalid dense observation never enters a fit. Its legacy replay has two
    baseline contour records and no solver contour, contact solve or proposal.
    Every field that actually exists must match the immutable previous result.
    """
    required={'vertices','rawCandidateVertices','solverVertices','candidateProposals','beforeContours',
        'afterContours','support','solverEvidence','diagnostics'}
    expectedDiagnostics=dict(status='skippedInvalidUnifiedObservation',candidateAccepted=False,
        accepted=False,contactOptimizationEnabled=False,observationAlignmentError=error)
    checks=dict(explicitInputFailure=isinstance(error,str) and bool(error),
        originalSkipSchema=set(oldEvidence)==required,newSkipSchema=set(newEvidence)==required,
        allActualFieldsExact=sameEvidence(oldEvidence,newEvidence))
    for prefix,evidence in (('original',oldEvidence),('replay',newEvidence)):
        checks[prefix+'Diagnostics']=sameEvidence(evidence.get('diagnostics'),expectedDiagnostics)
        for key in ('vertices','rawCandidateVertices','solverVertices'):
            checks[prefix+key]=key in evidence and np.array_equal(evidence[key],baseline)
        checks[prefix+'NoProposals']=evidence.get('candidateProposals')==[]
        checks[prefix+'NoSolverCaptures']=evidence.get('solverEvidence')==[]
        checks[prefix+'FrozenSupport']=sameEvidence(evidence.get('support'),support)
        checks[prefix+'BeforeContours']=sameEvidence(evidence.get('beforeContours'),beforeContours)
        checks[prefix+'AfterContours']=sameEvidence(evidence.get('afterContours'),beforeContours)
    return dict(validationPassed=bool(all(checks.values())),checks=checks,solverWasSkipped=True,
        skippedBecause=error,scope='Exact original skip fields; no fabricated solverContours or fit traces.')


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


def originalPolicyEvidenceAudit(rawEvidence,baseline,originalConstraints,camera,imageScale,originalLegacyGate,accessorySpec=None):
    """Report original guards without pretending they selected new GN steps."""
    from validateResidualContactExperiment import coarseEyePreservationAudit
    from validateBoundaryFieldExperiment import reconstructFieldTrial
    rounds=[]
    for record in rawEvidence.get('solverEvidence',[]):
        e=record['evidence'];reference=record['fullReferenceVertices'];ids=record['fullVertexIds'];trials=[]
        for trial in e['trials']:
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],e['maxDisplacement'],trial['trialCoefficients'])
            full=reference.copy();full[ids]=local
            if accessorySpec is not None: full=independentlyMapFollowing(accessorySpec,full)
            trials.append(dict(iteration=trial['iteration'],backtrack=trial['backtrack'],activeDecision=trial['decision'],
                originalCoarse=coarseEyePreservationAudit(baseline,full,originalConstraints,camera,imageScale=imageScale),
                originalCoarseParticipatedInActiveDecision=False,reference68ParticipatedInGn=False))
        casts=[]
        for trial in e['castTrials']:
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],e['maxDisplacement'],trial['coefficients']).astype(trial['outputDtype'])
            local[~e['activeVertexMask']]=e['referenceVertices'][~e['activeVertexMask']].astype(local.dtype)
            full=reference.copy();full[ids]=local
            if accessorySpec is not None: full=independentlyMapFollowing(accessorySpec,full)
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
    sameEvidence,fullApertureRasterAudit,independentFrontFiltering,fieldRepresentationAudit,
    savedBoundaryFieldAudit,independentCompactField,independentFieldRegularizer,reconstructFieldTrial,
    independentSolverObjective,localTrialSafety,trialFieldEvidenceAudit,independentGaussNewtonSystem,_close)
from validateResidualContactExperiment import (
    COARSE_EYES,coarseEyePreservationAudit,frozenAssemblyAudit,legacyResidualObservationAudit,
    historicalCollisionAudit,frozenContactGapAudit,contactSpecAudit,independentEyeSupport,
    radialWindingAudit,exportIdentityAudit,auditResidualEvidence,_scoreFieldsAgree,
    fullLocalGeometryAudit,residualAcceptanceFromRounds)
from validateFeatureGeometryExperiment import (readJson,readArrays,readEvidence,sha256,fileIdentityAudit,
    loadOriginalAudit,downstreamGeometryAudit,initialGeometryAudit,_contourMasksAgree,intersectionSetAudit)
from validateProtectedFeatureExperiment import originalObservationMasks,originalObservationErrors
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
PLAN_SHA = 'c0c10f64d74ae7b45b49300ccb4ec3f13e8f35ed1b73279f6da158bc746b2b06'
BRANCHES = ('baseline','fixedAccessoriesReplay','followingAccessories')
def auditFollowingSolverFields(branch, prepared, rawEvidence, camera, baseline, scene, support, constraints, imageScale, spec, operator, measure, config, accessorySpec=None):
    """Independently reconstruct fields, inner trial gates and outer field steps."""
    from faceAppearance.fastFrontEyeGeometry import _constraints
    records = rawEvidence.get('solverEvidence', []); stages = rawEvidence['diagnostics'].get('fast', {}).get('stages', [])
    allChecks = dict(everySolverCaptured=[r['round'] for r in records] == [s['round'] for s in stages if 'solver' in s])
    mode = config['fieldPolicy']['legacyMode' if branch == 'legacyResidualReplay' else 'candidateMode']
    ids = np.unique(np.r_[scene['nicpToFullIds'], scene['faces'][prepared['cornerFaceIds']].ravel()])
    inverse = np.full(len(baseline), -1, int); inverse[ids] = np.arange(len(ids))
    faceIds = np.flatnonzero((inverse[scene['faces']] >= 0).all(1)); faces = inverse[scene['faces'][faceIds]]
    localFixed = inverse[support['fixedVertexIds']]; localFixed = localFixed[localFixed >= 0]
    fixedFull = np.ones(len(baseline), bool); fixedFull[ids] = False; fixedFull[support['fixedVertexIds']] = True
    extent = float(np.ptp(baseline[scene['nicpToFullIds']], axis=0).max()); limit, radius = extent*.03, extent*.12
    current = baseline.copy(); reports = []
    for record in records:
        e = record['evidence']; roundId = record['round']; stage = next(row for row in stages if row['round'] == roundId)
        checks = dict(fullVertexIds=np.array_equal(record['fullVertexIds'], ids), fullFaceIds=np.array_equal(record['fullFaceIds'], faceIds),
            fullFixed=np.array_equal(record['fullFixedVertexIds'], np.flatnonzero(fixedFull)), originalRoundReference=np.array_equal(record['fullReferenceVertices'], current),
            localOriginal=np.array_equal(e['referenceVertices'], current[ids]), localFaces=np.array_equal(e['faces'], faces),
            localFixed=np.array_equal(e['fixedVertexIds'], localFixed), mode=e['deformationMode'] == mode,
            radius=e['requestedSupportRadius'] == radius, maxDisplacement=e['maxDisplacement'] == limit)
        if accessorySpec is not None:
            checks['physicalFixedEvidence']=np.array_equal(record['physicalFixedVertexIds'],accessorySpec['effectiveFixedVertexIds'])
            checks['sameFollowingSpec']=record['accessoryFollowingSpecSha256']==accessorySpec['specSha256']
            checks['sameFollowers']=np.array_equal(record['followerVertexIds'],accessorySpec['followerVertexIds'])
            checks['globalRestNotIncremental']=record['fullAssemblyFrom']=='global completed baseline plus mapped total skin displacement'
            followerColumns=(accessorySpec['followerVertexIds'][:,None]*3+np.arange(3)).ravel()
            checks['contactJacobianUnchangedByFollowerChain']=operator[:,followerColumns].nnz==0
        else:
            checks['noFollowingRecordOnReplay']=all(key not in record for key in ('physicalFixedVertexIds','accessoryFollowingSpecSha256','followerVertexIds'))
        originalConstraints, _ = _constraints(current, scene['faces'], scene['materialIds'], camera, prepared['photoEyes'], measure(current), ids,
            prepared['cornerFaceIds'], scene['uvs'], scene['uvFaces'], scene['materials'], scene['alphaTextures'], fixedVertexIds=support['fixedVertexIds'])
        kept, inputReports, preparedRows, points = independentFrontFiltering(originalConstraints, current[ids], camera)
        checks['originalFilteredConstraints'] = sameEvidence(kept, e['keptConstraints'])
        checks['originalEligibility'] = all(sameEvidence({k:v for k,v in a.items() if k != 'initialDistancePixels'}, {k:v for k,v in b.items() if k != 'initialDistancePixels'}) and _close(a['initialDistancePixels'], b['initialDistancePixels']) for a,b in zip(inputReports, e['inputReports'])) and len(inputReports) == len(e['inputReports'])
        checks['inputControlPoints'] = np.array_equal(points, e['inputControlPoints'])
        checks['preparedConstraintCount'] = len(preparedRows) == len(e['preparedConstraints'])
        for index, (a,b) in enumerate(zip(preparedRows, e['preparedConstraints'])):
            checks['prepared'+str(index)] = all(sameEvidence(a[k], b[k]) for k in ('viewIndex','name','vertexIndices','barycentric','targetPixels','weights','segmentPairs','transform'))
        field = fieldRepresentationAudit(e['referenceVertices'], e['basis'], e['axes'], fixedVertexIds=localFixed, maxDisplacement=limit,
            cameraRotation=np.asarray(camera['rotation']), reportedActiveMask=e['activeVertexMask'])
        checks['fieldRepresentation'] = field['validationPassed']
        if mode == 'fixedBoundaryBiharmonic':
            construction = savedBoundaryFieldAudit(e['fieldConstruction'], e['basis'], requestedControlCount=32,
                supportRadius=radius, smoothingRadiusFraction=.1, expectedFixedVertexIds=localFixed)
            checks['fieldConstruction'] = construction['validationPassed']
            checks['fieldActualConstructionIdentity'] = np.array_equal(e['referenceVertices'], e['fieldConstruction']['referenceVertices']) and np.array_equal(faces, e['fieldConstruction']['referenceFaces']) and np.array_equal(points, e['fieldConstruction']['inputControlPoints'])
            checks['normalizationIdentity'] = e['normalization'] == e['fieldConstruction']['normalization']
            checks['unshrunkRequestedRadius'] = e['effectiveSupportRadius'] == radius
        else:
            _, centers, _ = independentCompactField(current[ids], points, radius)
            nearest = np.linalg.norm(centers[:, None]-current[ids][localFixed][None], axis=2).min() if len(centers) and len(localFixed) else radius/(1-1e-12)
            effective = min(radius, float(nearest)*(1-1e-12)); blocked = effective <= float(np.ptp(current[ids],axis=0).max())*1e-12
            expectedB, expectedCenters, norm = independentCompactField(current[ids], np.empty((0,3)) if blocked else points, radius if blocked else effective)
            checks['compactFormula'] = np.array_equal(e['basis'], expectedB) and np.array_equal(e['controlPoints'], expectedCenters) and e['normalization'] == norm
            checks['oldRadiusCap'] = e['effectiveSupportRadius'] == effective
            checks['noNewFieldOnReplay'] = e['fieldConstruction'] is None
            construction = dict(validationPassed=checks['compactFormula'] and checks['oldRadiusCap'], compactRadius=effective, requestedRadius=radius, blocked=bool(blocked))
        reg = independentFieldRegularizer(e['referenceVertices'], faces, e['basis'], e['activeVertexMask'], maxDisplacement=limit, displacementWeight=.002, strainWeight=.03, controlWeight=.0001)
        checks['unchangedPhysicalAndCoefficientRegularizer'] = _close(reg, e['regularizer'], atol=1e-12, rtol=1e-10)
        def fullFrom(local):
            result = current.copy(); result[ids] = local
            return independentlyMapFollowing(accessorySpec, result) if accessorySpec is not None else result
        def energy(local):
            gaps = operator @ fullFrom(local).ravel()
            residual = np.sqrt(spec['weight']/len(gaps))*np.minimum((gaps-spec['minimumGap'])/spec['lengthScale'],0) if len(gaps) else np.empty(0)
            return .5*float(residual@residual)
        def evaluate(coeff, actual=None):
            local = reconstructFieldTrial(e['referenceVertices'], e['basis'], e['axes'], limit, coeff) if actual is None else actual
            return independentSolverObjective(e, coeff, camera, regularizer=reg, pixelScale=4., curveWeight=0., additionalEnergy=energy(local), actualVertices=actual)
        def safety(local, saved):
            mesh = localTrialSafety(e['referenceVertices'],local,faces,limit); full = fullFrom(local)
            coarse = coarseEyePreservationAudit(baseline, full, constraints, camera, imageScale=imageScale)
            contact = frozenContactGapAudit(baseline,full,spec['sourceVertexIds'],spec['targetVertexIds'],spec['targetBarycentric'],spec['normals'],spec['minimumGap'],lengthScale=spec['lengthScale'],tolerance=1e-8)
            safe = mesh['accepted'] and coarse['accepted'] and contact['accepted']
            detail = dict(mesh=mesh, coarse=coarse, contact=contact)
            consistent = saved['geometricSafetyAccepted'] == mesh['accepted'] and _close(mesh['metrics'], {k:saved[k] for k in mesh['metrics']})
            extra = saved['additionalSafety']; consistent &= extra['accepted'] == (coarse['accepted'] and contact['accepted']) and extra['coarseAlignment']['accepted'] == coarse['accepted'] and extra['contact']['accepted'] == contact['accepted']
            if accessorySpec is not None:
                import hashlib
                snapshot=extra.get('accessoryFollowingSnapshot',{})
                changedCount=int(np.any(full[accessorySpec['followerVertexIds']]!=baseline[accessorySpec['followerVertexIds']],axis=1).sum())
                witness=dict(specSha256=accessorySpec['specSha256'],fullVerticesSha256=hashlib.sha256(full.tobytes()).hexdigest(),
                    changedFollowerVertexCount=changedCount,effectiveFixedExact=bool(np.array_equal(full[accessorySpec['effectiveFixedVertexIds']],baseline[accessorySpec['effectiveFixedVertexIds']])))
                snapshotExact=snapshot==witness
                consistent &= snapshotExact
                detail['followingSnapshot']=dict(validationPassed=snapshotExact,independent=witness,actualCallback=snapshot)
                detail['followingRelationship']=followingRelationAudit(accessorySpec,full)
            else: consistent &= 'accessoryFollowingSnapshot' not in extra
            return bool(safe), bool(consistent), detail
        coefficients = np.zeros_like(e['finalCoefficients']); initial = evaluate(coefficients); best = coefficients.copy(); bestObjective = initial['objective']; bestIteration = 0
        checks['initialObjective'] = _close(initial['objective'], e['initialObjective']) and _close(initial['objectiveComponents'], e['initialObjectiveComponents']) and initial['preparedBindingsValidated']
        trials = []; lastIteration = 0; lastBacktrack = -1; lastAccepted = False; previousDamping = .001
        for trial in e['trials']:
            proposed = trial['trialCoefficients']; raw = reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],limit,proposed)
            proof = trialFieldEvidenceAudit(e['referenceVertices'],e['basis'],e['axes'],maxDisplacement=limit,
                coefficientsBefore=trial['beforeCoefficients'],coefficientStep=trial['coefficientStep'],stepScale=trial['stepScale'],trialCoefficients=proposed,rawVertices=raw,fixedVertexIds=localFixed)
            pc = dict(coefficientArithmetic=proof['validationPassed'], selectedCoefficientChain=np.array_equal(coefficients,trial['beforeCoefficients']),
                boundedBacktrack=trial['stepScale']==.5**trial['backtrack'] and 0<=trial['backtrack']<14,
                boundedIteration=1<=trial['iteration']<=20,
                trialOrder=(trial['iteration']>lastIteration and trial['backtrack']==0) or trial['iteration']==lastIteration and trial['backtrack']==lastBacktrack+1 and not lastAccepted)
            if trial['iteration']==lastIteration: pc['sameDampingWithinIteration'] = trial['dampingBeforeTrial'] == previousDamping
            elif lastIteration == 0 and trial['iteration']==1: pc['initialDamping'] = trial['dampingBeforeTrial'] == .001
            elif trial['iteration']==lastIteration+1: pc['nextDamping'] = trial['dampingBeforeTrial'] == (max(previousDamping*.5,1e-8) if lastAccepted else previousDamping*10)
            now=evaluate(coefficients);pc['currentObjective']=_close(now['objective'],trial['currentObjective']) and _close(now['objectiveComponents'],trial['currentObjectiveComponents'])
            H,g=independentGaussNewtonSystem(e,coefficients,camera,reg,fullReference=current,fullVertexIds=ids,contactOperator=operator,contactSpec=spec)
            lhs=H+trial['dampingBeforeTrial']*np.diag(np.maximum(np.diag(H),1e-8)); step=trial['coefficientStep'].ravel()
            denominator=max(float(np.linalg.norm(lhs,np.inf)*np.linalg.norm(step,np.inf)),float(np.linalg.norm(g,np.inf)),np.finfo(float).tiny)
            stepError=float(np.linalg.norm(lhs@step+g,np.inf)/denominator);pc['gaussNewtonStepEquation']=stepError<=1e-9
            safe,consistent,detail=safety(raw,trial['safety']);pc['safetyEvidence']=consistent and safe==trial['safetyAccepted']
            result=evaluate(proposed) if safe else None
            if not safe: decision='safetyRejected'
            elif not result['projectionValid']: decision='projectionRejected'
            elif any(v>initial['viewErrors'][k]+.5 for k,v in result['viewErrors'].items()):decision='viewDegradationRejected'
            elif not np.isfinite(result['objective']) or result['objective']>=now['objective']-1e-13:decision='objectiveRejected'
            else:decision='acceptedStep'
            pc['decision']=decision==trial['decision'];pc['explicitSkippedEvaluation']=trial['evaluation'] is None if not safe else trial['evaluation'] is not None
            if safe and trial['evaluation'] is not None:
                pc['evaluatedObjective']=all(_close(result[k],trial['evaluation'][k]) for k in ('objective','objectiveComponents','meanErrorPixels','viewErrors','projectionValid')) and result['preparedBindingsValidated']
            if decision=='acceptedStep':
                coefficients=proposed.copy()
                if result['objective']<bestObjective and result['meanErrorPixels']<initial['meanErrorPixels']-.001:
                    best=proposed.copy();bestObjective=result['objective'];bestIteration=trial['iteration']
            trials.append(dict(iteration=trial['iteration'],backtrack=trial['backtrack'],scale=trial['stepScale'],decision=decision,
                validationPassed=bool(all(pc.values())),checks=pc,normalEquationRelativeResidual=stepError,independentSafety=detail,evaluation=result,
                rawCoordinatesReconstructedFromSavedCoefficients=True,fullSceneCollisionEvaluated=False))
            lastIteration,lastBacktrack,lastAccepted,previousDamping=trial['iteration'],trial['backtrack'],decision=='acceptedStep',trial['dampingBeforeTrial']
        checks['unroundedBestSelection']=np.array_equal(best,e['unroundedBestCoefficients'])
        casts=[]; finalBeforeFallback=None; castSafe=False; castScale=None; finalCoeff=None
        for index,cast in enumerate(e['castTrials']):
            expectedScale=1. if index==0 else max(0.,1.-1e-4*2**(index-1)); cc=best*cast['scale']
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],limit,cc).astype(np.dtype(cast['outputDtype']))
            # Original solver explicitly copies inactive values after dtype cast.
            local[~e['activeVertexMask']]=e['referenceVertices'][~e['activeVertexMask']].astype(local.dtype)
            castSafe,consistent,detail=safety(local.astype(float),cast['safety'])
            pc=dict(scale=cast['scale']==expectedScale,coefficients=np.array_equal(cc,cast['coefficients']),safety=consistent and castSafe==cast['safetyAccepted'],
                noTrialAfterSafe=index==0 or not casts[-1]['safe'],outputDtype=cast['outputDtype']==e['outputDtype'])
            casts.append(dict(index=index,scale=cast['scale'],safe=castSafe,validationPassed=bool(all(pc.values())),checks=pc,independentSafety=detail))
            finalBeforeFallback=local;castScale=cast['scale'];finalCoeff=cc
        checks['castTracePresent']=bool(casts) and len(casts)<=17
        checks['castTraceTerminatedBySafetyOrBudget']=bool(casts) and (casts[-1]['safe'] or len(casts)==17)
        evaluated=evaluate(finalCoeff,finalBeforeFallback.astype(float))
        rejected=not castSafe or castScale==0 or (bestIteration>0 and (evaluated['meanErrorPixels']>=initial['meanErrorPixels']-.001 or any(v>initial['viewErrors'][k]+.5 for k,v in evaluated['viewErrors'].items())))
        output=e['referenceVertices'].astype(np.dtype(e['outputDtype'])) if rejected else finalBeforeFallback
        finalCoeff=np.zeros_like(finalCoeff) if rejected else finalCoeff
        evaluated=evaluate(finalCoeff,output.astype(float))
        checks['castRejectionDecision']=rejected==e['castRejected'];checks['finalCoefficients']=np.array_equal(finalCoeff,e['finalCoefficients'])
        checks['finalOutputFromFieldAndCast']=np.array_equal(output,e['outputVertices'])
        checks['finalEvaluation']=all(_close(evaluated[k],e['finalEvaluation'][k]) for k in ('objective','objectiveComponents','meanErrorPixels','viewErrors','projectionValid'))
        outer=[]
        for proposal in (row for row in rawEvidence['candidateProposals'] if row['round']==roundId):
            expected=current.copy();expected[ids]=current[ids]+proposal['scale']*(output-current[ids]);expected[fixedFull]=baseline[fixedFull]
            if accessorySpec is not None: expected=independentlyMapFollowing(accessorySpec,expected)
            outer.append(dict(scale=proposal['scale'],coordinatesExactlyFromSolverField=mappedCoordinatesAgree(expected,proposal['vertices']) if accessorySpec is not None else np.array_equal(expected,proposal['vertices']),selected=proposal['selectedByFastRound']))
        checks['outerProposalsFromActualField']=all(row['coordinatesExactlyFromSolverField'] for row in outer)
        checks['allGnTrials']=all(row['validationPassed'] for row in trials);checks['allCastTrials']=all(row['validationPassed'] for row in casts)
        reports.append(dict(round=roundId,validationPassed=bool(all(checks.values())),checks=checks,fieldRepresentation=field,fieldConstruction=construction,
            trials=trials,castTrials=casts,outerFieldProposals=outer,finalEvaluation=evaluated))
        selected=[row for row in rawEvidence['candidateProposals'] if row['round']==roundId and row['selectedByFastRound']]
        if selected: current=selected[-1]['vertices'].copy()
    allChecks['allRounds']=all(row['validationPassed'] for row in reports)
    return dict(validationPassed=bool(all(allChecks.values())),checks=allChecks,rounds=reports,scope='Independent FEM/compact basis, IRLS equation, GN decision trace and cast; complete-scene collisions independently checked for every OUTER proposal.')


def auditFollowingResidualEvidence(branch, prepared, evidence, baseline, final, scene, support, constraints, camera,
                          imageScale, baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap,
                          legacyGate, spec, operator, accessorySpec=None):
    """Retain every rejected proposal and reproduce selection without fitting."""
    from faceAppearance.fastFrontEyeGeometry import _constraints, _eyeContactSafety
    diagnostics = evidence['diagnostics']; proposals = evidence['candidateProposals']
    checks = dict(beforeFullRaster=_contourMasksAgree(baselineMeasurement, evidence['beforeContours']),
        afterFullRaster=_contourMasksAgree(measure(final), evidence['afterContours']),
        rawAndSolverRetained=all(k in evidence for k in ('rawCandidateVertices', 'solverVertices')))
    if branch == 'baseline':
        checks.update(unchangedBaseline=np.array_equal(final, baseline), noCandidateProposals=not proposals,
            rawExactlyBaseline=np.array_equal(evidence['rawCandidateVertices'], baseline),
            solverExactlyBaseline=np.array_equal(evidence['solverVertices'], baseline))
        return dict(validationPassed=bool(all(checks.values())), checks=checks, residualCandidateAccepted=False, proposals=[])
    withContact = branch == 'residualWithContact'
    checks.update(contactFactor=diagnostics['contactOptimizationEnabled'] == withContact,
        frozenOriginalCoarse=sameEvidence(evidence['coarseEyeConstraints'], constraints),
        neverRerunCoarse=all(diagnostics[k] is False for k in ('coarseStagesRerun', 'coarseStagesSkipped', 'coarseStagesRestricted')),
        outerCandidateGateUsed=diagnostics['externalCandidateGateUsed'] is True,
        rawIsLastProposal=np.array_equal(evidence['rawCandidateVertices'], proposals[-1]['vertices'] if proposals else evidence['solverVertices']))
    if withContact:
        checks['sameFrozenContactSpec'] = sameEvidence(evidence['contactSpec'], spec)
    else: checks['noContactOptimizationSpec'] = evidence['contactSpec'] is None
    baselinePairs = pairs(baseline); strictCache = {}
    def strict(vertices):
        import hashlib
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in strictCache:
            physicalFixed = support['fixedVertexIds'] if accessorySpec is None else accessorySpec['effectiveFixedVertexIds']
            geometry = fullLocalGeometryAudit(baseline, vertices, scene, physicalFixed, camera)
            fullPairs = intersectionSetAudit(baselinePairs, pairs(vertices), scene['materialIds'])
            coarse = coarseEyePreservationAudit(baseline, vertices, constraints, camera, imageScale=imageScale)
            contact = contactGap(vertices); legacy = legacyGate(vertices)
            strictCache[key] = dict(accepted=bool(geometry['accepted'] and fullPairs['accepted'] and coarse['accepted'] and contact['accepted'] and legacy['accepted']),
                fullGeometry=geometry, fullPairs=fullPairs, coarse=coarse, contact=contact, legacy=legacy)
        return strictCache[key]
    reports = []; current = baseline.copy(); selected = []
    stages = diagnostics.get('fast', {}).get('stages', [])
    checks['everyAttemptRetained'] = len(proposals) == sum(len(s.get('attempts', [])) for s in stages) == diagnostics['proposalCount']
    checks['boundedRoundAndIterationSettings'] = (not diagnostics['fastSolverInvoked'] or
        diagnostics['fast']['roundsRequested'] == prepared['policy']['rounds'] and diagnostics['fast']['maximumSolverIterations'] == prepared['policy']['rounds'] * prepared['policy']['iterationsPerRound'])
    movableIds = np.unique(np.r_[scene['nicpToFullIds'], scene['faces'][prepared['cornerFaceIds']].ravel()])
    localInverse=np.full(len(baseline),-1,int);localInverse[movableIds]=np.arange(len(movableIds))
    subsetFaces=localInverse[scene['faces']];subsetFaces=subsetFaces[(subsetFaces>=0).all(1)]
    subsetScene=dict(faces=subsetFaces,nicpToFullIds=localInverse[scene['nicpToFullIds']])
    physicalFixed=support['fixedVertexIds'] if accessorySpec is None else accessorySpec['effectiveFixedVertexIds']
    subsetFixed=localInverse[physicalFixed];subsetFixed=subsetFixed[subsetFixed>=0]
    epsilon = max(np.linalg.norm(np.ptp(baseline, axis=0)) * 1e-9, 1e-12)
    for stage in stages:
        currentMeasurement = measure(current)
        # This is a read-only visibility binding primitive, not an optimization.
        _, bindingStats = _constraints(current, scene['faces'], scene['materialIds'], camera, photoEyes, currentMeasurement,
            movableIds, prepared['cornerFaceIds'], scene['uvs'], scene['uvFaces'], scene['materials'], scene['alphaTextures'],
            fixedVertexIds=support['fixedVertexIds'])
        checks['round' + str(stage['round']) + 'OnlyFullyFixedBindingOmissions'] = sameEvidence(bindingStats, stage['binding'])
        stageProposals = [p for p in proposals if p['round'] == stage['round']]
        checks['round' + str(stage['round']) + 'AttemptOrder'] = [p['scale'] for p in stageProposals] == [a['scale'] for a in stage['attempts']]
        predictedSelected = None
        for proposal, attempt in zip(stageProposals, stage['attempts']):
            vertices = proposal['vertices']; gate = strict(vertices); stored = proposal['strict']; pc = {}
            if accessorySpec is not None: pc['frozenFollowerMapping']=followingRelationAudit(accessorySpec,vertices)['validationPassed']
            subsetGeometry=fullLocalGeometryAudit(baseline[movableIds],vertices[movableIds],subsetScene,subsetFixed,camera)
            pc['physicalSubsetGeometryDecision']=subsetGeometry['accepted']==proposal['internalGeometry']['accepted']
            pc['physicalSubsetGeometryMetrics']=all(_close(subsetGeometry['metrics'][name],proposal['internalGeometry']['metrics'][name])
                for name in ('maximumDisplacement','maximumRelativeEdgeChange','minimumAreaRatio','minimumNormalCosine','maximumReferenceDepthChange'))
            pc['physicalSubsetFixedCount']=len(subsetFixed)==proposal['internalGeometry']['metrics']['fixedVertexCount']
            pc['strictDecision'] = gate['accepted'] == stored['accepted']
            pc['geometryDecision'] = gate['fullGeometry']['accepted'] == stored['geometry']['accepted']
            pc['coarseDecision'] = gate['coarse']['accepted'] == stored['coarseAlignment']['accepted']
            pc['sharedContactDecision'] = gate['contact']['accepted'] == stored['externalCandidateGate']['contactGate']['accepted']
            pc['legacyDecision'] = gate['legacy']['accepted'] == stored['externalCandidateGate']['legacyGate']['accepted']
            pc['externalDecision'] = bool(gate['legacy']['accepted'] and gate['contact']['accepted']) == stored['externalCandidateGate']['accepted']
            if 'newPairs' in stored.get('fullAssembly', {}):
                pc['fullUnexcludedPairSet'] = gate['fullPairs']['newPairs'] == [tuple(v) for v in stored['fullAssembly']['newPairs']]
            else: pc['collisionSkipOnlyOnGeometryFailure'] = not gate['fullGeometry']['accepted']
            pc['solverAttemptExact'] = sameEvidence(proposal['solverAttempt'], attempt)
            pc['internalGeometrySaved'] = sameEvidence(proposal['internalGeometry'], attempt['geometry'])
            pc['externalStrictSaved'] = sameEvidence(stored, attempt['externalCandidateGate'])
            if withContact:
                pc['insideContactGate'] = gate['contact']['accepted'] == stored['contact']['accepted']
                actual = operator @ vertices.ravel(); snapshot = proposal['contactEvaluation']
                pc['contactGapEvidence'] = np.allclose(actual, snapshot['gaps'], atol=1e-12, rtol=0)
                expectedResidual = np.sqrt(spec['weight'] / len(actual)) * np.minimum((actual - spec['minimumGap']) / spec['lengthScale'], 0) if len(actual) else np.empty(0)
                pc['contactEnergy'] = abs(.5 * float(expectedResidual @ expectedResidual) - snapshot['energy']) <= 1e-10
            else: pc['noInsideContactOptimization'] = stored['contact'] is None and proposal['contactEvaluation'] is None
            aperture = None; shouldSelect = False
            if attempt['geometry']['accepted'] and gate['accepted']:
                measured = measure(vertices)
                aperture = fullApertureRasterAudit(currentMeasurement, measured, photoEyes, originalPixelScale=pixelScale)
                score = aperture.get('metrics', {}).get('candidate')
                pc['actualCompleteMaskScores'] = _scoreFieldsAgree(score, attempt.get('apertureMetrics'))
                maskBenefit = aperture.get('originalApertureGate', {}).get('accepted', False)
                pc['perRoundApertureGate'] = maskBenefit == attempt.get('apertureGate')
                if maskBenefit:
                    sphere = _eyeContactSafety(baseline, vertices, scene['faces'], scene['materialIds'], movableIds, epsilon)
                    pc['legacyEyeSphereGate'] = sphere['accepted'] == attempt.get('eyeContact', {}).get('accepted')
                    shouldSelect = bool(sphere['accepted'] and gate['fullPairs']['accepted'])
                    if shouldSelect: pc['fastSubsetCollisionNotSubstitute'] = attempt.get('intersection', {}).get('newPairCount') == 0
            else: pc['notMeasuredAfterHardGateFailure'] = 'apertureMetrics' not in attempt
            pc['selectedByFastRound'] = bool(proposal['selectedByFastRound']) == shouldSelect
            if shouldSelect:
                predictedSelected = vertices.copy(); selected.append((stage['round'], proposal['scale']))
                pc['selectedScaleMatches'] = stage['status'] == 'accepted' and stage['selectedScale'] == proposal['scale']
            reports.append(dict(round=proposal['round'], scale=proposal['scale'], selected=shouldSelect,
                validationPassed=bool(all(pc.values())), checks=pc, independentStrict=gate,
                completeAperture=aperture, producerStrict=stored, physicalSubsetGeometry=subsetGeometry))
        if predictedSelected is not None: current = predictedSelected
        else: checks['round' + str(stage['round']) + 'NoFalseSelection'] = stage['status'] != 'accepted'
    solver = evidence['solverVertices']; checks['solverEqualsSelectedProposalChain'] = np.array_equal(solver, current)
    checks['solverFullRaster'] = _contourMasksAgree(measure(solver), evidence['solverContours'])
    solverGate = strict(solver)
    solverEffect = fullApertureRasterAudit(baselineMeasurement, measure(solver), photoEyes, originalPixelScale=pixelScale)
    actualComplete = all(measure(solver)['eyes'][name]['status'] == 'accepted' for name in EYES)
    roiSame = all(solverEffect['checks'].get(name + 'SameRaster', False) for name in EYES)
    producerHasFast = diagnostics['fastSolverInvoked']
    if producerHasFast:
        predictedStrict = bool(solverGate['accepted'] and actualComplete and roiSame)
        checks['finalStrictDecision'] = predictedStrict == diagnostics['strict']['accepted']
        checks['fastChangeStatus'] = (not np.array_equal(solver, baseline)) == (diagnostics['fast']['status'] == 'acceptedLocalExperiment')
        accepted = residualAcceptanceFromRounds(predictedStrict, not np.array_equal(solver, baseline),
            diagnostics['fast']['status'], [p['completeAperture']['originalApertureGate']['accepted'] for p in reports if p['selected']])
    else:
        status = diagnostics['status']
        skipValid = (status == 'skippedInsufficientVisibleSupport' and not support['supported'] or
            status == 'skippedUnmeasurableCompleteAperture' and prepared['beforeApertureMetrics'] is None or
            status == 'skipNoActiveContactRows' and withContact and len(spec['sourceVertexIds']) == 0)
        checks['skipHasActualInputReason'] = skipValid; accepted = False
    checks['wrapperAcceptanceReproduced'] = accepted == diagnostics['accepted']
    checks['returnedSelectedOrExactBaseline'] = np.array_equal(final, solver if accepted else baseline)
    checks['acceptedRequiresActualPerRoundMaskBenefit'] = not accepted or all(
        p['completeAperture']['originalApertureGate']['accepted'] for p in reports if p['selected'])
    if withContact and producerHasFast and any(s.get('solver') for s in stages):
        calls = diagnostics['contactEvaluationCalls']
        checks['contactInsideOptimizationEvidence'] = calls['residual'] > 0 and calls['jacobian'] > 0 and calls['safety'] > 0
    elif not withContact:
        checks['noContactResidualCalls'] = diagnostics['contactEvaluationCalls']['residual'] == diagnostics['contactEvaluationCalls']['jacobian'] == 0
    checks['allRawProposalEvidenceValidated'] = all(p['validationPassed'] for p in reports)
    return dict(validationPassed=bool(all(checks.values())), checks=checks, residualCandidateAccepted=accepted,
        solverEffect=solverEffect, finalVersusOriginalEffectAccepted=solverEffect['accepted'],
        finalVersusOriginalEffectIsSeparateFromProducerDecision=True, finalStrict=solverGate, proposals=reports,
        contactOptimizationEnabled=withContact, actualCompleteContourStatus={name: measure(solver)['eyes'][name]['status'] for name in EYES},
        sharedPrimitives=['complete triangle/alpha visibility raster', 'triangle intersection predicate', 'fast boundary binding and supplementary original eye-sphere gate'],
        independentReconstruction=['support geodesics', 'coarse/legacy error and masks', 'complete mask score and decision', 'contact nearest feature/operator/gap', 'all full-scene pair differences and decision chain'])


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
    templateBinding = readEvidence(runRoot/'templateAccessoryBinding')
    templateProof = accessoryBindingAudit(templateBinding, assets.arrays, assets.components, config['accessoryFollowing']['policy'])
    followingSpec = readEvidence(root/'accessoryFollowingSpec')
    followingProof = followingSpecificationAudit(followingSpec, templateBinding, base, scene, support,
        sparse.load_npz(root/'accessoryFollowingOperator.npz'))
    followingOperator = followingProof.pop('reconstructedOperator')
    caseChecks['templateBinding'] = templateProof['validationPassed']
    caseChecks['followingSpec'] = followingProof['validationPassed']
    caseChecks['commonTemplateBinding'] = config['accessoryFollowing']['bindingSha256']==templateBinding['bindingSha256']
    caseChecks['templateInputHashes'] = fileIdentityAudit(config['accessoryFollowing']['inputHashes'])['accepted']
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
    followerColumns=(followingSpec['followerVertexIds'][:,None]*3+np.arange(3)).ravel()
    caseChecks['contactIndependentOfFollowers'] = operator[:,followerColumns].nnz==0
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
    caseChecks['exactPreviousObservationAlignment'] = sameEvidence(alignment,readEvidence(legacyRoot/'observationAlignment'))
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
        isUnified=branch!='baseline'; invalidUnified=isUnified and not alignmentValid
        isFollowing=branch=='followingAccessories'; activeFollowing=followingSpec if isFollowing else None
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
        if isFollowing and not invalidUnified:
            rawDiagnostics=rawEvidence['diagnostics'];nonSkin=np.setdiff1d(np.arange(len(base)),skinIds)
            checks['solverSpecEvidence']=sameEvidence(rawEvidence['accessoryFollowingSpec'],followingSpec)
            checks['rawFollowingEnabled']=rawDiagnostics['accessoryFollowingEnabled'] is True and rawDiagnostics['accessoryDegreesOfFreedom']==0
            checks['rawFixedFlagHonest']=rawDiagnostics['fixedDomainExactlyPreserved']==np.array_equal(rawEvidence['vertices'][support['fixedVertexIds']],base[support['fixedVertexIds']])
            checks['rawNonSkinFlagHonest']=rawDiagnostics['allNonSkinExactlyFixed']==np.array_equal(rawEvidence['vertices'][nonSkin],base[nonSkin])
            checks['rawEffectiveFixedFlagHonest']=rawDiagnostics['effectiveFixedDomainExactlyPreserved']==np.array_equal(rawEvidence['vertices'][followingSpec['effectiveFixedVertexIds']],base[followingSpec['effectiveFixedVertexIds']])
            checks['rawExactOriginalSupport']=rawDiagnostics['frozenSolverSupportPreserved'] is True
        else: checks['noFollowingSolverForReplayOrInvalidObservations']='accessoryFollowingSpec' not in rawEvidence
        fixed = (whitelistAssemblyAudit(scene, {**scene, 'vertices': final}, support['movableMask'], followingSpec['followerVertexIds'], staticFullIds=skinIds[raw['staticIds']])
            if isFollowing else frozenAssemblyAudit(scene, {**scene, 'vertices': final}, support['movableMask'], staticFullIds=skinIds[raw['staticIds']]))
        checks['fixedAssembly'] = fixed['accepted']
        allowed=support['movableMask'].copy()
        if isFollowing: allowed[followingSpec['followerVertexIds']]=True
        savedAllowed=readArrays(directory/'allowedMovement.npz')
        checks['allowedMovementMask']=np.array_equal(savedAllowed['movableMask'],allowed)
        checks['allowedFollowers']=np.array_equal(savedAllowed['followerVertexIds'],followingSpec['followerVertexIds'] if isFollowing else np.empty(0,int))
        followerProof=followingRelationAudit(followingSpec,final) if isFollowing else dict(validationPassed=True,notApplicable=True)
        checks['followingRelationship']=followerProof['validationPassed'] and report['followerRelationExactlySatisfied'] is True
        checks['followingEnabled']=report['accessoryFollowingEnabled']==isFollowing
        checks['followingSolverEnabled']=report['accessoryFollowingAppliedToSolver']==bool(isFollowing and alignmentValid)
        checks['allNonSkinPolicyHonest']=report['originalAllNonSkinFixedPolicyActive']==(not isFollowing)
        checks['noAddedAccessoryDegreesOfFreedom']=report['accessoryDegreesOfFreedom']==0
        checks['followingSpecIdentity']=report['accessoryFollowingSpecSha256']==(followingSpec['specSha256'] if isFollowing else None)
        checks['changedFollowerCount']=report['changedFollowerVertexCount']==np.any(final!=base,axis=1)[followingSpec['followerVertexIds']].sum()
        checks['originalFixedDomainTruthful']=report['allOutsideSupportExactlyFixed']==np.array_equal(final[~support['movableMask']],base[~support['movableMask']])
        checks['effectiveFixedDomainTruthful']=report['allOutsideAllowedMovementExactlyFixed']==np.array_equal(final[~allowed],base[~allowed])
        mouthFullIds = skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]
        checks['realMouthExactlyFixed'] = np.array_equal(final[mouthFullIds], base[mouthFullIds])
        archivePath = Path(report['asset']['archivePath']); archive = _load(archivePath)
        export = exportIdentityAudit(sourceAsset, archive, readArrays(directory / 'exportIdentity.npz'), base, final, allowed, raw, report['asset'])
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
        incremental = bool(fixed['accepted'] and followerProof['validationPassed'] and coarse['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs'])
        originalPolicyAccepted=bool(originalCoarse['accepted'] and legacyReport['accepted'])
        activePolicyAccepted=None if invalidUnified else bool(coarse['accepted'] and activeLegacyReport['accepted'])
        absolute = bool(not invalidUnified and initialGate['accepted'] and incremental and downstream['accepted'] and originalPolicyAccepted and activePolicyAccepted and collision['allHistoricalAnchorsAccepted'])
        nonSkinIds=np.setdiff1d(np.arange(len(base)),skinIds)
        legacyAbsolute=bool(absolute and np.array_equal(final[nonSkinIds],base[nonSkinIds]))
        checks['oldPolicyDecision']=originalPolicyAccepted==report['originalObservationPolicyAccepted']
        checks['activePolicyDecision']=activePolicyAccepted==report['activeObservationPolicyAccepted']
        checks['originalCoarseDecision']=originalCoarse['accepted']==report['originalCoarseAlignment']['accepted']
        checks['activeCoarseDecision']=coarse['accepted']==report['activeCoarseAlignment']['accepted']
        checks['activeLegacyDecision']=activeLegacyReport['accepted']==report['activeLegacyObservationGate']['accepted']
        checks['legacyAbsoluteDecision']=legacyAbsolute==report['legacyAbsoluteEligible']
        checks['absolutePolicyNamed']=report['absoluteEligibilityPolicy']=='currentExplicitAccessoryPolicyWithOriginalObservationAudit'
        checks['alignmentValidity']=report['observationAlignmentValid']==alignmentValid and report['observationPolicyChangedForDiagnostic']==bool(isUnified and alignmentValid)
        checks['tableIdentity']=report['observationTableSha256']==(alignment['table']['tableSha256'] if alignmentValid else None)
        checks['invalidAlignmentErrorRecorded']=report['observationAlignmentError']==alignment['error']
        checks['incrementalDecision'] = incremental == report['incrementalSafe']
        checks['absoluteDecision'] = absolute == report['absoluteEligible'] == report['geometrySafetyAccepted']
        effect = fullApertureRasterAudit(baselineMeasurement, measure(final), photoEyes, originalPixelScale=pixelScale)
        eyeProof = auditFollowingResidualEvidence('baseline' if branch == 'baseline' or invalidUnified else 'residualWithContact', prepared, rawEvidence, base, rawEvidence['vertices'], scene, support, constraints, camera, imageScale,
            baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap, activeLegacyGate, spec, operator,activeFollowing)
        checks['residualEvidence'] = eyeProof['validationPassed']
        checks['rawProducerDecision'] = eyeProof['residualCandidateAccepted'] == report['rawRefinementAccepted']
        effectRaw = fullApertureRasterAudit(baselineMeasurement, measure(rawEvidence['vertices']), photoEyes, originalPixelScale=pixelScale)
        savedEffect = readJson(directory / 'finalEffectGate.json')
        checks['finalEffectReportIdentity'] = sameEvidence(savedEffect, report['finalEffectGate'])
        checks['finalEffectDecision'] = effectRaw['accepted'] == savedEffect['accepted']
        checks['finalEffectRawMetrics'] = (not effectRaw['effectDefined'] or _scoreFieldsAgree(effectRaw['metrics']['baseline'], savedEffect['baseline']) and _scoreFieldsAgree(effectRaw['metrics']['candidate'], savedEffect['candidate']))
        fallback = branch=='followingAccessories' and alignmentValid and not effectRaw['accepted']
        checks['fallbackPolicy'] = fallback == savedEffect['fallbackToExactBaseline']
        checks['finalOutputPolicy'] = np.array_equal(final, base if fallback else rawEvidence['vertices'])
        checks['effectOnlyFallbackLeavesRawAndSolver'] = all(np.array_equal(evidence[k],rawEvidence[k]) for k in ('rawCandidateVertices','solverVertices')) and sameEvidence(evidence['candidateProposals'],rawEvidence['candidateProposals'])
        producerAccepted = bool(eyeProof['residualCandidateAccepted'] and not fallback)
        checks['outputProducerDecision'] = producerAccepted == report['residualCandidateAccepted']
        changed = not np.array_equal(final,base)
        checks['changedDecision'] = changed == report['geometryChanged']
        fieldProof = auditFollowingSolverFields(branch,prepared,rawEvidence,camera,base,scene,support,constraints,imageScale,spec,operator,measure,config,activeFollowing) if branch != 'baseline' and not invalidUnified else dict(validationPassed=not rawEvidence['solverEvidence'],checks={'noSolver':not rawEvidence['solverEvidence']},rounds=[])
        checks['solverFieldsAndTrials'] = fieldProof['validationPassed']
        checks['solverSidecarIdentity'] = sameEvidence(readEvidence(directory / 'solverEvidence'),rawEvidence['solverEvidence'])
        replay = None
        if branch == 'fixedAccessoriesReplay':
            oldEvidence = readEvidence(Path(context['legacyReplayRoot']) / 'unifiedObservationContact/rawRefinement')
            replay = (skippedObservationReplayAudit(oldEvidence,rawEvidence,base,prepared['support'],prepared['beforeContours'],alignment['error'])
                if invalidUnified else replayGeometryEvidenceAudit(oldEvidence,rawEvidence))
            replaySaved = readJson(directory / 'replayMatch.json')
            checks['legacyReplayExact'] = replay['validationPassed'] and replaySaved['accepted']
            checks['legacyReplayReportIdentity'] = sameEvidence(replaySaved,report['replayMatch'])
        originalPolicyProof=originalPolicyEvidenceAudit(rawEvidence,base,originalConstraints,camera,imageScale,originalLegacyGate,activeFollowing) if isUnified and alignmentValid else dict(validationPassed=True,rounds=[],outerProposals=[],notApplicable=True)
        checks['originalCounterfactualAndOuterEvidence']=originalPolicyProof['validationPassed']
        if invalidUnified:
            checks['invalidObservationSkip']=rawEvidence['diagnostics']['status']=='skippedInvalidUnifiedObservation' and evidence['diagnostics']['status']=='skippedInvalidUnifiedObservation' and not rawEvidence['candidateProposals'] and not rawEvidence['solverEvidence'] and np.array_equal(rawEvidence['vertices'],base) and np.array_equal(final,base)
            checks['noInventedUnifiedGateEvaluation']=report['activeObservationPolicyAccepted'] is None
        classification = classifyBoundaryCandidate(producerAccepted=producerAccepted,geometryChanged=changed,
            incrementalSafe=incremental,absoluteEligible=absolute,finalBaselineEffect=effectRaw,evidenceValidated=bool(all(checks.values())))
        policyClassification=classifyObservationCandidate(producerAccepted=producerAccepted,changed=changed,
            activeObservationAccepted=bool(activePolicyAccepted),physicalIncrementalSafe=bool(fixed['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs']),
            legacyOriginalAccepted=originalPolicyAccepted,legacyAbsoluteEligible=legacyAbsolute,finalBaselineEffectAccepted=bool(effectRaw['accepted'] and effectRaw['effectDefined']),evidenceValidated=bool(all(checks.values())))
        policyClassification['changingLegacyAbsoluteEligible']=policyClassification.pop('changingAbsoluteEligible')
        classification.update(policyClassification)
        classification['absoluteEligibilityPolicy']='currentExplicitAccessoryPolicyWithOriginalObservationAudit'
        classification['activeObservationPolicyAccepted']=activePolicyAccepted
        classification['activeObservationPolicyEvaluated']=not invalidUnified
        checks['diagnosticImprovementDecision']=policyClassification['diagnosticImprovementAccepted']==report['improvementAccepted']
        checks['improvementDecision'] = classification['improvementAccepted'] == report['improvementAccepted']
        checks['changingAbsoluteDecision'] = classification['changingAbsoluteEligible'] == report['changingAbsoluteEligible']
        checks['changingLegacyAbsoluteDecision'] = classification['changingLegacyAbsoluteEligible'] == bool(report['improvementAccepted'] and report['legacyAbsoluteEligible'])
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks,
            incrementalSafe=incremental, absoluteEligible=absolute, producerAccepted=producerAccepted,
            residualCandidateAccepted=producerAccepted, rawRefinementAccepted=eyeProof['residualCandidateAccepted'],
            requestedEffectAccepted=classification['requestedEffectAccepted'], improvementAccepted=classification['improvementAccepted'],
            changingAbsoluteEligible=classification['changingAbsoluteEligible'], evidenceValidated=bool(all(checks.values())),
            classification=classification, finalEffectGate=effectRaw, finalOutputEffect=effect,
            originalObservationPolicyAccepted=originalPolicyAccepted,activeObservationPolicyAccepted=activePolicyAccepted,legacyAbsoluteEligible=legacyAbsolute,
            changingLegacyAbsoluteEligible=classification['changingLegacyAbsoluteEligible'],absoluteEligibilityPolicy=classification['absoluteEligibilityPolicy'],
            observationAlignmentValid=alignmentValid,originalPolicyEvidence=originalPolicyProof,originalCoarseAlignment=originalCoarse,activeCoarseAlignment=coarse,activeLegacyPhotoGate=activeLegacyReport,
            gnEvidence=fieldProof, replayEvidence=replay,
            unchangedRealMouth=np.array_equal(final[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]], base[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]]),
            fixedAssembly=fixed, followerRelationship=followerProof, exportIdentity=export, downstream=downstream,
            contactEnvelope=contacts, legacyPhotoGate=legacyReport, fullCollisionAnchors=collision,
            fullApertureEffect=effect, residualEvidence=eyeProof, producerOutputs=outputs,
            reportSha256=sha256(directory / 'report.json'), archiveSha256=sha256(archivePath), clientReady=False)
    caseChecks['inputHashesAfterAudit'] = fileIdentityAudit(context['sourceHashes'])['accepted']
    result = dict(caseId=case, validationPassed=bool(all(caseChecks.values()) and all(v['validationPassed'] for v in branches.values())),
        checks=caseChecks, initialGate=initialGate, contactSpecification=contactProof, orientation=orientation,
        accessoryTemplateBinding=templateProof,accessoryFollowingSpecification=followingProof,
        observationAlignment=dict(validationPassed=bool(all(alignmentChecks.values())),valid=alignmentValid,checks=alignmentChecks,tableProof=tableProof),
        originalInputIdentities=identities, branches=branches, planSha256=PLAN_SHA, validatorSha256=sha256(__file__),
        candidateSolvesPerformed=False, clientReady=False,
        scope='Same completed coarse baseline and eye observations. Six whitelisted attachments follow skin with zero independent parameters; other accessories stay fixed. Every full-scene surface remains in the outer collision gate. Contact and photo curves are proxies; inherited failures stay failures.')
    outputRoot.mkdir(parents=True, exist_ok=True); writeJson(outputRoot / f'{case}.json', result)
    return result



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs='+',required=True)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/accessoryFollowing20261024')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/accessoryFollowing20261024/independentAudit')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'tests/testAccessoryFollowingValidation.py',ROOT/'tools/validateResidualContactExperiment.py',ROOT/'tools/validateBoundaryFieldExperiment.py',
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
