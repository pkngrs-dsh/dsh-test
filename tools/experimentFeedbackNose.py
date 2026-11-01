"""One bounded nasal repair using saved cameras and current visible surfaces."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.geometryRefinement import _meshEdges
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.noseGeometryRefinement import (eyeMouthAnchors, noseAnchorFractions,
    refineNoseGeometry, visibleNoseBindings)
from faceAppearance.renderedJawRuntime import APPEARANCES, buildScene, renderDetect

POLICY = {'method': 'visibleMultiviewNoseScreenedFieldV1', 'referenceViewIndex': 3,
    'fitSemanticIds': [27, 28, 30, 31, 35], 'heldOutSemanticIds': [29, 32, 33, 34],
    'semanticIds': list(range(27, 36)), 'alarWeight': .25, 'referenceWeight': 4.,
    'maximumAppearanceDriftPixels': 2.5, 'minimumViewCosine': .08,
    'maximumTargetDistanceFaceFraction': .04, 'minimumRidgeSamplesPerFitView': 2,
    'minimumReliableFitViews': 2, 'supportGeodesicFaceFraction': .28,
    'bindingGeodesicFaceFraction': .12, 'maximumDisplacementMm': 2.,
    'smoothnessWeight': 10., 'strainWeight': .1, 'screenWeight': .002,
    'geometryAmounts': [1., .5, .25], 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'minimumFrontRelativeImprovementPixels': .35, 'minimumFrontRelativeImprovementFraction': .15,
    'maximumHeldOutMeanIncreasePixels': .25, 'minimumHeldOutConsensusImprovementPixels': .15,
    'maximumOtherViewMeanIncreasePixels': .5, 'maximumOtherViewPointIncreasePixels': 1.5,
    'minimumValidationSamplesPerView': 4, 'minimumValidationRidgeSamplesPerView': 2,
    'validation': 'Frozen before-fit reliable mask; missing view support prevents full acceptance; no after-anchor refit',
    'target': 'Photo nose relative to its eye-mouth line versus current rendered nose relative to its own line',
    'correspondence': 'Independent nearest visible surface ray for each view; no fixed MP nasal positions or cross-view alar identity',
    'search': 'One linearized 3D solve, only three safety-only scales, one live validation',
    'texture': 'Frozen eyelidCandidate ZIP, separately exported/rendered by caller'}


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(points, photo, ids, fractions, frozenAnchorPoints):
    relative = points[ids] - eyeMouthAnchors(frozenAnchorPoints, fractions)
    target = photo[ids] - eyeMouthAnchors(photo, fractions)
    errors = np.linalg.norm(relative - target, axis=1)
    return {'relativeMeanPixels': float(errors.mean()), 'relativePointErrorsPixels': errors,
        'absoluteMeanPixels': float(np.linalg.norm(points[ids] - photo[ids], axis=1).mean()),
        'absolutePointErrorsPixels': np.linalg.norm(points[ids] - photo[ids], axis=1)}


def run(case, chain, frozenZip, output, skipLive=False):
    if output.exists() and any(output.iterdir()): raise FileExistsError('Output must be new')
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / 'policy.json', POLICY)
    geometry = arrays(chain / 'finalGeometry.npz')
    before, faces = geometry['vertices'], geometry['faces']
    if not np.array_equal(before, arrays(chain / 'renderedJaw/sourceState.npz')['vertices']):
        raise ValueError('Cached render geometry differs from current baseline')
    cameras = json.loads((case / 'reconstruction/cameras.json').read_text())
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    camera = {key: np.asarray(cameras[3][key]) for key in ('rotation', 'translation', 'intrinsics')}
    transform = arrays(chain / 'templateToUff.npz')
    transform['scale'] = float(transform['scale'])
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    original = arrays(case / 'reconstruction/directClient/refinement/clientGeometry.npz')
    pureDirect = original['directTemplateVertices'] @ transform['rotation'] * transform['scale'] + transform['translation']
    photoData = arrays(chain / 'renderedJaw/photoObservations.npz')
    photoData['diagnostics'] = json.loads((chain / 'renderedJaw/photoObservations.json').read_text())
    baseline = [{name: arrays(chain / f'renderedJaw/baseline/view{view}-{name}.npz') for name in APPEARANCES}
                for view in range(len(inputs))]
    fullBinding = loadClientLandmarks(faces)
    protectedBinding = selectClientLandmarks(fullBinding, groups=('eyes', 'mouth'))
    loops = extractMeshBoundaryLoops(before, faces)['loops']
    protected = np.unique(np.r_[protectedBinding['vertexIndices'].ravel(), np.concatenate(loops)])
    static = np.flatnonzero(assets.arrays['staticMask'])
    fixed = np.union1d(static, protected)
    rest = assets.arrays['vertices'][assets.arrays['nicpToFullIds']]
    seeds = np.unique(selectClientLandmarks(fullBinding, groups='nose')['vertexIndices'])
    edges = _meshEdges(faces)
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    graph = sparse.coo_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]],
        np.r_[edges[:, 1], edges[:, 0]])), shape=(len(rest), len(rest))).tocsr()
    distance = dijkstra(graph, directed=False, indices=seeds, min_only=True)
    faceScale = float(np.ptp(rest, axis=0).max())
    domainDistance = distance / faceScale
    active = domainDistance <= POLICY['supportGeodesicFaceFraction']
    active[fixed] = False
    ids = np.asarray(POLICY['semanticIds'])
    constraints, observationReports, validationMasks = [], [], []
    for view, variants in enumerate(baseline):
        photo = photoData['faceLandmarksWorking'][view]
        fractions = noseAnchorFractions(photo, ids)
        consensus = np.mean([variants[name]['points'] for name in APPEARANCES], axis=0)
        scatter = np.max([np.linalg.norm(variants[name]['points'][ids] - consensus[ids], axis=1) for name in APPEARANCES], axis=0)
        delta = (photo[ids] - eyeMouthAnchors(photo, fractions)) - (consensus[ids] - eyeMouthAnchors(consensus, fractions))
        rays = visibleNoseBindings(before, faces, cameras[view], tuple(inputs[view]['workingSize'][::-1]), consensus[ids],
            domainDistance=domainDistance, maximumDomainDistance=POLICY['bindingGeodesicFaceFraction'],
            fixedVertexIds=fixed, minimumViewCosine=POLICY['minimumViewCosine'])
        reasons = [list(item) for item in rays['reasons']]
        for row, semanticId in enumerate(ids):
            if not photoData['landmarkCandidateValidMask'][view, semanticId]: reasons[row].append('invalidPhotoObservation')
            if not all(variants[name]['valid'][semanticId] for name in APPEARANCES): reasons[row].append('invalidRenderedObservation')
            if scatter[row] > POLICY['maximumAppearanceDriftPixels']: reasons[row].append('appearanceDisagreement')
            if np.linalg.norm(delta[row]) > POLICY['maximumTargetDistanceFaceFraction'] * photoData['faceScaleWorkingPixels'][view]:
                reasons[row].append('excessiveRelativeTarget')
            anchorIds = [36, 39, 42, 45, 48, 54]
            if (not photoData['landmarkCandidateValidMask'][view, anchorIds].all()
                    or not all(variants[name]['valid'][anchorIds].all() for name in APPEARANCES)):
                reasons[row].append('invalidAnchorObservation')
        valid = np.asarray([not item for item in reasons])
        validationMasks.append(valid)
        fit = valid & np.isin(ids, POLICY['fitSemanticIds'])
        eligible = int((fit & (ids <= 30)).sum()) >= POLICY['minimumRidgeSamplesPerFitView']
        peak = np.minimum.reduce([photoData['faceRawPeakScores'][view, ids]] +
            [variants[name]['peaks'][:, ids].min(0) for name in APPEARANCES])
        weight = peak / (1 + (scatter / 1.5)**2) * np.where(ids >= 31, POLICY['alarWeight'], 1.)
        weight *= POLICY['referenceWeight'] if view == 3 else 1.
        if eligible:
            constraints.append({'viewIndex': view, 'name': 'visibleNose', 'semanticIds': ids[fit],
                'vertexIndices': rays['vertexIndices'][fit], 'barycentric': rays['barycentric'][fit],
                'targetPixels': consensus[ids][fit] + delta[fit], 'weights': weight[fit]})
        observationReports.append({'viewIndex': view, 'eligibleForFit': bool(eligible), 'semanticIds': ids,
            'reasons': reasons, 'validMask': valid, 'fitMask': fit if eligible else np.zeros(len(ids), bool),
            'appearanceScatterPixels': scatter, 'consensusPixels': consensus[ids], 'targetDeltaPixels': delta,
            'anchorFractions': fractions, 'rays': rays, 'confidenceWeights': weight})
    writeJson(output / 'observations.json', observationReports)
    if len(constraints) < POLICY['minimumReliableFitViews'] or not any(item['viewIndex'] == 3 for item in constraints):
        writeJson(output / 'report.json', {'status': 'insufficientReliableMultiviewObservations', 'policy': POLICY})
        raise ValueError('Insufficient reliable multiview nose surface observations')
    cap = POLICY['maximumDisplacementMm'] / 10 * transform['scale']
    proposal, solve = refineNoseGeometry(before, faces, cameras, constraints, activeMask=active,
        maxDisplacement=cap, smoothnessWeight=POLICY['smoothnessWeight'], strainWeight=POLICY['strainWeight'],
        screenWeight=POLICY['screenWeight'])
    currentPairs, originalPairs = intersectionPairs(before, faces), intersectionPairs(pureDirect, faces)
    extent = float(np.ptp(pureDirect, axis=0).max())
    existingDepth = float(np.abs((before - pureDirect) @ camera['rotation'][2]).max())
    candidate, amount, attempts = before.copy(), 0., []
    for step in POLICY['geometryAmounts']:
        trial = before + step * (proposal - before)
        current = checkFrontGeometry(before, trial, faces, camera=camera, fixedVertexIds=fixed,
            maxDisplacement=cap, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3, depthTolerance=cap)
        cumulative = checkFrontGeometry(pureDirect, trial, faces, camera=camera, fixedVertexIds=static,
            maxDisplacement=extent * .07, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3,
            depthTolerance=existingDepth + cap)
        pairs = intersectionPairs(trial, faces)
        current['newIntersectionPairs'] = len(pairs - currentPairs)
        cumulative['newIntersectionPairsAgainstOriginalDirect'] = len(pairs - originalPairs)
        passed = current['accepted'] and cumulative['accepted'] and not (pairs - currentPairs) and not (pairs - originalPairs)
        attempts.append({'amount': step, 'currentBaseline': current, 'originalDirect': cumulative, 'accepted': bool(passed)})
        if passed:
            candidate, amount = trial, step
            break
    result = {'before': before, 'proposal': proposal, 'candidate': candidate, 'after': before,
        'faces': faces, 'originalDirect': pureDirect, 'staticVertexIds': static, 'fixedVertexIds': fixed,
        'protectedVertexIds': protected, 'activeMask': active}
    np.savez_compressed(output / 'geometry.npz', **result)
    report = {'status': 'geometryReadyPendingLiveCheck' if amount else 'rejectedByGeometrySafety',
        'case': case.name, 'referenceViewIndex': 3, 'policy': POLICY, 'selectedGeometryAmount': amount,
        'solve': solve, 'geometryAttempts': attempts, 'finalCandidateAccepted': False,
        'fixedVerticesExactlyUnchanged': bool(np.array_equal(candidate[fixed], before[fixed])),
        'cameraChanged': False, 'textureChanged': False, 'frozenTextureZip': str(frozenZip),
        'constraints': constraints, 'validationMasks': validationMasks,
        'sourceHashes': {str(path): fileHash(path) for path in [chain/'finalGeometry.npz', chain/'templateToUff.npz',
            chain/'renderedJaw/sourceState.npz', chain/'renderedJaw/photoObservations.npz',
            case/'reconstruction/cameras.json', frozenZip, Path(__file__), ROOT/'src/faceAppearance/noseGeometryRefinement.py']},
        'limitations': ['FAN nose semantics and saved cameras are estimates, not anatomical ground truth.',
            'Independent visible view bindings cannot establish exact cross-view anatomical correspondence.',
            'Live controls use template/gray appearances; frozen actual eyelid ZIP rendering is separate.',
            'No view with insufficient reliable observations is declared verified; after stays baseline until complete acceptance.']}
    writeJson(output / 'report.json', report)
    print(json.dumps({'stage': report['status'], 'amount': amount, 'freeVertices': solve['freeVertexCount'], 'output': str(output)}), flush=True)
    if not amount or skipLive: return
    scene = buildScene(assets, before, transform)
    afterViews, timing = renderDetect(scene, candidate, cameras, inputs, photoData, output/'live', projectRoot=ROOT)
    checks, accepted, validationComplete = [], True, True
    for view, variants in enumerate(baseline):
        row = {'viewIndex': view, 'appearances': []}
        valid = validationMasks[view]
        enough = valid.sum() >= POLICY['minimumValidationSamplesPerView'] and (valid & (ids <= 30)).sum() >= POLICY['minimumValidationRidgeSamplesPerView']
        if view == 3: enough = enough and valid.all()
        row['validationComplete'] = bool(enough)
        validationComplete = validationComplete and enough
        used = np.flatnonzero(valid)
        held = used[np.isin(ids[used], POLICY['heldOutSemanticIds'])]
        heldGains = []
        photo = photoData['faceLandmarksWorking'][view]
        fractions = observationReports[view]['anchorFractions']
        for name in APPEARANCES:
            old, new = variants[name], afterViews[view]['variants'][name]
            if not len(used):
                row['appearances'].append({'variant': name, 'status': 'unverifiableNoFrozenReliableSamples'})
                accepted = False
                continue
            first = metrics(old['points'], photo, ids[used], fractions[used], old['points'])
            last = metrics(new['points'], photo, ids[used], fractions[used], old['points'])
            gain = first['relativeMeanPixels'] - last['relativeMeanPixels']
            worst = float(np.max(last['relativePointErrorsPixels'] - first['relativePointErrorsPixels']))
            reliable = bool(new['valid'][ids[used]].all())
            required = max(.35, .15 * first['relativeMeanPixels']) if view == 3 else -.5
            passed = reliable and gain >= required and (view == 3 or worst <= 1.5)
            heldCheck = None
            if len(held):
                heldBefore = metrics(old['points'], photo, ids[held], fractions[held], old['points'])
                heldAfter = metrics(new['points'], photo, ids[held], fractions[held], old['points'])
                heldGain = heldBefore['relativeMeanPixels'] - heldAfter['relativeMeanPixels']
                heldGains.append(heldGain)
                heldCheck = {'semanticIds': ids[held], 'before': heldBefore, 'after': heldAfter, 'gainPixels': heldGain}
                if view == 3: passed = passed and heldGain >= -.25
            row['appearances'].append({'variant': name, 'before': first, 'after': last, 'gainPixels': gain,
                'requiredGainPixels': required, 'maximumPointIncreasePixels': worst, 'heldOut': heldCheck,
                'reliableAfter': reliable, 'accepted': bool(passed)})
            accepted = accepted and passed
        if view == 3:
            row['heldOutConsensusGainPixels'] = float(np.mean(heldGains)) if heldGains else None
            accepted = accepted and len(heldGains) == 3 and np.mean(heldGains) >= .15
        checks.append(row)
    accepted = bool(accepted and validationComplete)
    report.update(status='accepted' if accepted else 'notAcceptedByLiveNoseValidation',
        finalCandidateAccepted=accepted, validationComplete=bool(validationComplete), liveChecks=checks, liveTiming=timing)
    result['after'] = candidate if accepted else before
    np.savez_compressed(output/'geometry.npz', **result)
    writeJson(output/'report.json', report)
    print(json.dumps({'stage': report['status'], 'validationComplete': bool(validationComplete)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--frozen-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-live', action='store_true')
    options = parser.parse_args()
    run(options.case.resolve(), options.chain.resolve(), options.frozen_zip.resolve(), options.output.resolve(), options.skip_live)
