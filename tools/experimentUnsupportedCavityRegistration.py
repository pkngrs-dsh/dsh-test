"""One opt-in local registration solve with an unsupported-cavity depth prior.

The original pureDirect quadratic objective is reproduced before adding the
prior. No asset is exported, no renderer is run, and production defaults remain
unchanged. Failed candidates and all predetermined safety trials are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate, _laplacian
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.geometryRefinement import _meshEdges
from faceAppearance.legacyGeometryVariant import _similarity
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.unsupportedCavityPrior import buildUnsupportedCavityPrior
from experimentNoseSurfaceRegistration import arrays, digest, summary

POLICY = {'method': 'localOriginalRegistrationPlusUnsupportedRelativeCavityDepth',
    'depthPriorWeight': .25, 'targetMissingDepthFraction': .35, 'maximumTargetDepthIncrementMm': 2.,
    'depthScaleMm': 2., 'supportThreshold': .15, 'freeRegionRings': 2,
    'sampleSupportWeight': '1 for valid original objective observations; 0 for excluded observations',
    'priorNormalization': 'all audited interior vertices across both cavities, not sum of confidence weights',
    'maximumDisplacementMm': 2., 'safetyAmounts': [1., .5, .25],
    'maximumRelativeEdgeChange': .25, 'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'solverCount': 1, 'productionDefaultsChanged': False,
    'outsideRegion': 'Exactly fixed at pureDirect; identical local delta transferred onto accepted eyelid baseline',
    'protected': 'Static interface, eye/mouth landmark support, mesh boundaries and audited columella guard',
    'acceptance': 'Geometry diagnostic only; no photo likeness, export, render or production acceptance'}
REGIONS = ('frontImageLeft', 'frontImageRight')


def run(case, chain, baselinePath, audit, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh registration experiment output directory')
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output/'policy.json', POLICY)
    asset = ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
    templates = arrays(asset/'template.npz'); mapping = arrays(asset/'directRegistration.npz')
    manifest = json.loads((asset/'manifest.json').read_text())
    regions = arrays(audit/'regions.npz'); auditReport = json.loads((audit/'report.json').read_text())
    if digest(audit/'regions.npz') != auditReport['outputHashes']['regions.npz']:
        raise ValueError('Audited cavity region identity changed')
    for name, expected in auditReport['sourceHashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Audited source changed: {name}')
    source = arrays(case/'reconstruction/geometry.npz')
    faces, rest = templates['nicpFaces'], templates['vertices'][templates['nicpToFullIds']]
    np.testing.assert_array_equal(source['faces'], templates['sourceFaces'])
    np.testing.assert_array_equal(faces, regions['skinFaces'])
    valid, static = mapping['sampleWeights'] > 0, np.flatnonzero(templates['staticMask'])
    targets = sampleSurface(source['vertices'].astype(float), source['faces'],
        {'triangleIds': mapping['sourceTriangleIds'], 'barycentricWeights': mapping['sourceBarycentricWeights']})
    options = {key: manifest['geometryTransfer'][key] for key in
        ('dataWeight', 'smoothWeight', 'restWeight', 'staticConstraint', 'solver')}
    if options['solver'] != 'exact' or options['staticConstraint'] != 'hard':
        raise ValueError('This local replay requires the existing exact, hard-static pureDirect objective')
    baseline = registerDirectTemplate(rest, faces, mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'],
        targets, valid, static, **options)
    pureDirect, aligned = baseline['rawVertices'], baseline['alignedRest']
    transform = baseline['transform']; unit = float(transform['scale'])/10.
    replayDifference = float(np.abs(pureDirect-regions['skinVerticesPureDirect']).max())
    if replayDifference > 1e-9:
        raise ValueError(f'Pure direct replay does not match stored registration: {replayDifference}')
    np.testing.assert_allclose(aligned, regions['skinVerticesRestAligned'], rtol=0, atol=1e-10)
    geometry, frozenTransform = arrays(chain/'finalGeometry.npz'), arrays(chain/'templateToUff.npz')
    before = geometry['vertices']
    np.testing.assert_array_equal(before, regions['skinVerticesCurrent'])
    np.testing.assert_array_equal(faces, geometry['faces'])
    for name in ('rotation', 'translation', 'scale'):
        np.testing.assert_allclose(transform[name], frozenTransform[name], rtol=0, atol=1e-10)
    archive = _load(baselinePath)
    rotation, translation, scale = _similarity(archive.manifest)
    fullBefore = archive.mesh['vertices']*100 @ rotation*scale+translation
    skinIds = templates['nicpToFullIds']
    np.testing.assert_allclose(fullBefore[skinIds], before, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(archive.mesh['faces'], templates['faces'])
    cameras = json.loads((case/'reconstruction/cameras.json').read_text())
    camera = {key: np.asarray(cameras[3][key]) for key in ('intrinsics', 'rotation', 'translation')}
    paths = [asset/'template.npz', asset/'directRegistration.npz', asset/'manifest.json', audit/'regions.npz',
        audit/'report.json', baselinePath, case/'reconstruction/geometry.npz', case/'reconstruction/cameras.json',
        chain/'finalGeometry.npz', chain/'templateToUff.npz', Path(__file__),
        ROOT/'src/faceAppearance/unsupportedCavityPrior.py', ROOT/'src/faceAppearance/directTemplateRegistration.py']
    inputHashes = {str(path): digest(path) for path in paths}

    priors, increments = {}, []
    for name in REGIONS:
        prior = buildUnsupportedCavityPrior(aligned, faces, regions[name+'_rimVertexIdsOrdered'],
            regions[name+'_interiorVertexIds'], faces[mapping['sampleNicpFaceIds']], mapping['sampleNicpWeights'],
            valid.astype(float), inwardDirection=-regions[name+'_restPlaneBasis'][:, 2],
            depthScale=POLICY['depthScaleMm']*unit, supportThreshold=POLICY['supportThreshold'])
        initial = prior['depthOperator']@pureDirect.ravel()
        increment = POLICY['targetMissingDepthFraction']*np.maximum(prior['restRelativeDepths']-initial, 0.)
        increment[prior['priorWeights'] == 0] = 0.
        prior['baselineDepths'], prior['requestedIncrement'] = initial, increment
        priors[name] = prior; increments.append(increment)
    maximumTarget = float(np.concatenate(increments).max())
    targetScale = min(1., POLICY['maximumTargetDepthIncrementMm']*unit/max(maximumTarget, 1e-30))
    totalInterior = sum(len(prior['priorWeights']) for prior in priors.values())
    weightedMatrices, weightedTargets = [], []
    for name, prior in priors.items():
        prior['targetDepths'] = prior['baselineDepths']+targetScale*prior['requestedIncrement']
        factor = np.sqrt(len(prior['priorWeights'])/totalInterior)
        weightedMatrices.append(prior['weightedOperator']*factor)
        weightedTargets.append(prior['rowScales']*prior['targetDepths']*factor)
        sparse.save_npz(output/f'{name}DepthOperator.npz', prior['depthOperator'])
        np.savez_compressed(output/f'{name}Prior.npz', vertexSupport=prior['vertexSupport'],
            priorWeights=prior['priorWeights'], interiorSupportRatio=prior['interiorSupportRatio'],
            baselineDepths=prior['baselineDepths'], restRelativeDepths=prior['restRelativeDepths'],
            targetDepths=prior['targetDepths'], rowScales=prior['rowScales'],
            interiorVertexIds=prior['interiorVertexIds'], rimVertexIds=prior['rimVertexIds'])
    priorMatrix, priorTarget = sparse.vstack(weightedMatrices, format='csr'), np.concatenate(weightedTargets)
    edges = _meshEdges(faces)
    active = np.zeros(len(rest), bool)
    active[np.concatenate([regions[name+'_vertexIds'] for name in REGIONS])] = True
    for _ in range(POLICY['freeRegionRings']):
        active[edges[active[edges].any(1)].ravel()] = True
    binding = loadClientLandmarks(faces)
    protected = np.unique(np.r_[static, regions['columellaGuardVertexIds'],
        selectClientLandmarks(binding, groups=('eyes', 'mouth'))['vertexIndices'].ravel(),
        np.concatenate(extractMeshBoundaryLoops(before, faces)['loops'])])
    active[protected] = False
    fixed, free = np.flatnonzero(~active), np.flatnonzero(active)
    if not len(free):
        raise ValueError('No free local registration support')
    includedFaces, includedBary = faces[mapping['sampleNicpFaceIds'][valid]], mapping['sampleNicpWeights'][valid]
    sample = sparse.csr_matrix((includedBary.ravel(),
        (np.repeat(np.arange(valid.sum()), 3), includedFaces.ravel())), shape=(int(valid.sum()), len(rest)))
    anchors = sparse.csr_matrix((np.ones(len(static)), (np.arange(len(static)), static)), shape=(len(static), len(rest)))
    design = sparse.vstack([sample, anchors], format='csr')
    observedTargets = np.concatenate([targets[valid], aligned[static]])
    laplacian, _ = _laplacian(len(rest), faces)
    cd = options['dataWeight']/(3*(valid.sum()+len(static)))
    cs, cr = options['smoothWeight']/(3*len(rest)), options['restWeight']/(3*len(rest))
    originalOperator = cd*(design.T@design)+cs*(laplacian.T@laplacian)+cr*sparse.eye(len(rest))
    originalRhs = cd*(design.T@observedTargets)+cr*aligned
    residual = originalOperator@pureDirect-originalRhs
    globalFree = np.setdiff1d(np.arange(len(rest)), static)
    baselineEquationError = float(np.linalg.norm(residual[globalFree])/max(np.linalg.norm(originalRhs[globalFree]), 1e-30))
    if baselineEquationError > 1e-9:
        raise ValueError('Reconstructed original objective does not have pureDirect as its optimum')
    freeCoordinates = (3*free[:, None]+np.arange(3)).ravel()
    localPrior = priorMatrix[:, freeCoordinates]
    localOriginal = sparse.kron(originalOperator.tocsr()[free][:, free], sparse.eye(3), format='csc')
    weight = POLICY['depthPriorWeight']
    operator = (localOriginal+weight*(localPrior.T@localPrior)).tocsc()
    rhs = weight*(localPrior.T@(priorTarget-priorMatrix@pureDirect.ravel()))-residual[free].ravel()
    delta = np.zeros_like(pureDirect)
    delta[free] = splu(operator).solve(rhs).reshape(-1, 3)
    if not np.isfinite(delta).all():
        raise ValueError('Local registration solution is nonfinite')
    solveResidual = float(np.linalg.norm(operator@delta[free].ravel()-rhs)/max(np.linalg.norm(rhs), 1e-30))
    if solveResidual > 1e-9:
        raise ValueError('Local registration normal equations did not converge')
    rawRegistration = pureDirect.copy(); rawRegistration[free] += delta[free]
    proposal = before.copy(); proposal[free] += delta[free]

    def objectives(vertices):
        old = cd*np.sum((design@vertices-observedTargets)**2)+cs*np.sum((laplacian@vertices)**2)+cr*np.sum((vertices-aligned)**2)
        cavity = weight*np.sum((priorMatrix@vertices.ravel()-priorTarget)**2)
        return {'original': float(old), 'cavityPrior': float(cavity), 'combined': float(old+cavity)}
    baseObjective, rawObjective = objectives(pureDirect), objectives(rawRegistration)
    if rawObjective['combined'] > baseObjective['combined']+1e-10:
        raise ValueError('Single solve increased the augmented objective')
    writeJson(output/'solveCheckpoint.json', {'status': 'oneSolveComplete', 'policy': POLICY,
        'pureDirectReplayMaximumError': replayDifference, 'originalEquationRelativeResidual': baselineEquationError,
        'localEquationRelativeResidual': solveResidual, 'freeVertexCount': len(free),
        'rawMaximumDisplacementMm': float(np.linalg.norm(delta, axis=1).max()/unit),
        'baseObjective': baseObjective, 'rawObjective': rawObjective})
    print(json.dumps({'stage': 'oneSolveComplete', 'freeVertices': len(free),
        'rawMaximumDisplacementMm': float(np.linalg.norm(delta, axis=1).max()/unit)}), flush=True)
    # Always retain the unaccepted raw field before potentially expensive gates.
    np.savez_compressed(output/'rawGeometry.npz', pureDirect=pureDirect, alignedRest=aligned, before=before,
        rawRegistration=rawRegistration, proposal=proposal, delta=delta, activeMask=active,
        fixedVertexIds=fixed, protectedVertexIds=protected, faces=faces)
    basePairs, beforePairs = intersectionPairs(pureDirect, faces), intersectionPairs(before, faces)
    fullPairs = intersectionPairs(fullBefore, archive.mesh['faces'])
    cap = POLICY['maximumDisplacementMm']*unit
    existingDepth = float(np.abs((before-pureDirect)@camera['rotation'][2]).max())
    attempts, candidate, amount = [], before.copy(), 0.
    common = dict(camera=camera, maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
        minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'])
    for fraction in POLICY['safetyAmounts']:
        trialDirect, trialAccepted = pureDirect.copy(), before.copy()
        trialDirect[free] += fraction*delta[free]; trialAccepted[free] += fraction*delta[free]
        directGate = checkFrontGeometry(pureDirect, trialDirect, faces, fixedVertexIds=fixed,
            maxDisplacement=cap, depthTolerance=cap, **common)
        acceptedGate = checkFrontGeometry(before, trialAccepted, faces, fixedVertexIds=fixed,
            maxDisplacement=cap, depthTolerance=cap, **common)
        cumulativeGate = checkFrontGeometry(pureDirect, trialAccepted, faces, fixedVertexIds=static,
            maxDisplacement=.07*np.ptp(pureDirect, axis=0).max(), depthTolerance=existingDepth+cap, **common)
        trialFull = fullBefore.copy(); trialFull[skinIds[free]] = trialAccepted[free]
        directPairs, acceptedPairs = intersectionPairs(trialDirect, faces), intersectionPairs(trialAccepted, faces)
        added = {'rawRegistration': len(directPairs-basePairs), 'acceptedSkin': len(acceptedPairs-beforePairs),
            'cumulativeSkin': len(acceptedPairs-basePairs),
            'assembled': len(intersectionPairs(trialFull, archive.mesh['faces'])-fullPairs)}
        passed = bool(directGate['accepted'] and acceptedGate['accepted'] and cumulativeGate['accepted'] and not any(added.values()))
        attempts.append({'amount': fraction, 'registration': directGate, 'acceptedBaseline': acceptedGate,
            'cumulative': cumulativeGate, 'addedIntersectionCounts': added, 'geometryPassed': passed})
        print(json.dumps({'stage': 'geometryGate', 'amount': fraction, 'geometryPassed': passed,
            'registrationReasons': directGate['reasons'], 'acceptedReasons': acceptedGate['reasons'],
            'cumulativeReasons': cumulativeGate['reasons'], 'addedIntersections': added}), flush=True)
        if passed:
            candidate, amount = trialAccepted, fraction
            break
    depthMetrics = {}
    for name, prior in priors.items():
        depthMetrics[name] = {'prior': prior['diagnostics'], 'priorWeights': summary(prior['priorWeights']),
            'supportRatio': summary(prior['interiorSupportRatio']),
            'restDepthMm': summary(prior['restRelativeDepths']/unit),
            'pureDirectDepthMm': summary(prior['baselineDepths']/unit),
            'targetDepthMm': summary(prior['targetDepths']/unit),
            'rawRegisteredDepthMm': summary((prior['depthOperator']@rawRegistration.ravel())/unit),
            'acceptedBeforeDepthMm': summary((prior['depthOperator']@before.ravel())/unit),
            'transferredProposalDepthMm': summary((prior['depthOperator']@proposal.ravel())/unit),
            'candidateDepthMm': summary((prior['depthOperator']@candidate.ravel())/unit)}
    np.savez_compressed(output/'geometry.npz', pureDirect=pureDirect, alignedRest=aligned, before=before,
        rawRegistration=rawRegistration, proposal=proposal, candidate=candidate, after=before,
        delta=delta, activeMask=active, fixedVertexIds=fixed, protectedVertexIds=protected, faces=faces)
    report = {'status': 'geometryDiagnosticPendingPhotoReview' if amount else 'rejectedByOriginalGeometrySafety',
        'finalCandidateAccepted': False, 'exported': False, 'rendered': False, 'caseId': case.name,
        'policy': POLICY, 'sourceHashes': inputHashes, 'registrationOptions': options,
        'pureDirectReplayMaximumError': replayDifference, 'originalEquationRelativeResidual': baselineEquationError,
        'localEquationRelativeResidual': solveResidual, 'freeVertexCount': len(free),
        'protectedVertexCount': len(protected), 'targetIncrementUniformScale': targetScale,
        'rawMaximumDisplacementMm': float(np.linalg.norm(delta, axis=1).max()/unit),
        'selectedSafetyAmount': amount, 'maximumCandidateDisplacementMm': float(np.linalg.norm(candidate-before, axis=1).max()/unit),
        'baseObjective': baseObjective, 'rawObjective': rawObjective,
        'outsideRegionExactlyFixed': bool(np.array_equal(proposal[fixed], before[fixed])),
        'depthMetrics': depthMetrics, 'safetyAttempts': attempts,
        'limitations': ['Template cavity prior preserves relative depth, not photo-measured anatomy.',
            'Depths use the fixed aligned-template inward direction; no local axis refit or rotation inference.',
            'Source observation support measures quadratic influence, not independent photographic evidence.',
            'Pure registration delta is transferred onto accepted geometry for a controlled diagnostic; downstream stages are not rerun.',
            'Geometry tests do not establish improved photo likeness. No asset is exported or adopted.']}
    for path, expected in inputHashes.items():
        if digest(path) != expected:
            raise ValueError(f'Experiment input changed during solve: {path}')
    writeJson(output/'report.json', report)
    print(json.dumps({'stage': report['status'], 'selectedSafetyAmount': amount}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.case.resolve(), args.chain.resolve(), args.baseline.resolve(), args.audit.resolve(), args.output.resolve())
