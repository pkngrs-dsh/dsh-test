"""One bounded, mouth-relative lower-face proportion experiment on saved data.

Frozen cameras, photograph observations and texture. One screened biharmonic
solve followed by [1, .5, .25] geometry backtracking. No production defaults,
registration, appearance baking or artifact export are changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.renderedJawCorrespondence import lowerJawDomain, _ray_bindings
from faceAppearance.renderedJawRuntime import buildScene, renderDetect, APPEARANCES

POLICY = {'method': 'mouthRelativeConsensusLowerFaceBiharmonicV1',
    'referenceViewIndex': 3, 'semanticIds': list(range(5, 12)),
    'mouthAnchorIds': [48, 54], 'geometryAmounts': [1., .5, .25],
    'smoothnessWeight': 10., 'strainWeight': .1, 'screenWeight': .002,
    'maximumDisplacementMm': 2.5, 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'minimumLiveImprovementPixels': 1., 'minimumRelativeLiveImprovement': .15,
    'maximumSideMeanIncreasePixels': 1., 'maximumSidePointIncreasePixels': 3.,
    'target': 'Photographic jaw relative to photo mouth midpoint, versus mean of three rendered jaw/mouth observations',
    'constraints': 'Current geometry ray bindings at render consensus; never fixed template jaw positions',
    'parameterSearch': 'One linear solve; at most three safety-only step sizes; one live validation',
    'texture': 'Unchanged eyelidCandidate ZIP is independently rendered/exported by caller'}


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(vertices, camera):
    cv = vertices @ np.asarray(camera['rotation']).T + camera['translation']
    hom = cv @ np.asarray(camera['intrinsics']).T
    return hom[:, :2] / hom[:, 2:], cv


def screenedProportionField(vertices, faces, camera, vertexIndices, barycentric,
                           pixelDeltas, weights, activeMask, maxDisplacement):
    """Solve a smooth pixel displacement, with exact Dirichlet-fixed vertices.

    Barycentric projection uses vertex-depth/sample-depth weights, so this
    linear constraint is exact while camera depth stays fixed. The regularizer
    contains a graph biharmonic term, short-edge strain, and screened magnitude.
    """
    original = np.asarray(vertices, float)
    active = np.asarray(activeMask, bool)
    if active.shape != (len(original),) or not active.any():
        raise ValueError('Need a nonempty active vertex mask')
    ids, bary = np.asarray(vertexIndices, int), np.asarray(barycentric, float)
    target, weights = np.asarray(pixelDeltas, float), np.asarray(weights, float)
    if (ids.ndim != 2 or ids.shape[1] != 3 or ids.shape != bary.shape
            or target.shape != (len(ids), 2) or weights.shape != (len(ids),)
            or not np.isfinite(np.r_[bary.ravel(), target.ravel(), weights]).all()
            or (weights <= 0).any() or (bary < -1e-6).any()
            or not np.allclose(bary.sum(1), 1) or maxDisplacement <= 0):
        raise ValueError('Invalid fixed surface constraints')
    _, cv = project(original, camera)
    if (cv[:, 2] <= 0).any():
        raise ValueError('Reference camera requires positive depth')
    sampleDepth = (cv[ids, 2] * bary).sum(1)
    values = bary * cv[ids, 2] / sampleDepth[:, None]
    constraint = sparse.coo_matrix((values.ravel(),
        (np.repeat(np.arange(len(ids)), 3), ids.ravel())), shape=(len(ids), len(original))).tocsr()
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(original), len(original))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = sparse.eye(len(original)) - sparse.diags(1 / np.maximum(degree, 1)) @ adjacency
    lengths = np.linalg.norm(original[edges[:, 1]] - original[edges[:, 0]], axis=1)
    shortEdgeWeight = np.median(lengths) / np.maximum(lengths, 1e-12)
    gradient = sparse.coo_matrix((np.c_[-shortEdgeWeight, shortEdgeWeight].ravel(),
        (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), len(original))).tocsr()
    free = np.flatnonzero(active)
    b, l, g = constraint[:, free], laplacian[:, free], gradient[:, free]
    w = sparse.diags(weights / weights.sum())
    hessian = (b.T @ w @ b + POLICY['smoothnessWeight'] / len(original) * (l.T @ l)
        + POLICY['strainWeight'] / len(edges) * (g.T @ g)
        + POLICY['screenWeight'] / len(original) * sparse.eye(len(free)))
    offset = np.zeros((len(original), 2))
    offset[free] = spsolve(hessian.tocsc(), b.T @ w @ target)
    deltaCamera = (np.c_[offset, np.zeros(len(original))] @ np.linalg.inv(camera['intrinsics']).T) * cv[:, 2, None]
    delta = deltaCamera @ camera['rotation']
    boundScale = min(1., maxDisplacement / max(float(np.linalg.norm(delta, axis=1).max()), 1e-20))
    delta *= boundScale
    proposal = original.copy()
    proposal[free] += delta[free]
    return proposal, {'freeVertexCount': len(free), 'boundScale': boundScale,
        'maximumDisplacementCanonical': float(np.linalg.norm(delta, axis=1).max()),
        'predictedPixelDeltas': constraint @ (offset * boundScale),
        'targetPixelDeltas': target, 'fixedVerticesExactlyUnchanged': bool(np.array_equal(proposal[~active], original[~active]))}


def relativeMetric(points, target, ids, weights):
    mouth = points[[48, 54]].mean(0)
    expected = target[ids] - target[[48, 54]].mean(0)
    residual = (points[ids] - mouth) - expected
    errors = np.linalg.norm(residual, axis=1)
    return {'meanPixels': float(weights @ errors), 'pointErrorsPixels': errors.tolist(),
        'mouthCenterPixels': mouth.tolist(), 'jaw6to10WidthPixels': float(np.linalg.norm(points[6] - points[10])),
        'jaw5to11WidthPixels': float(np.linalg.norm(points[5] - points[11])),
        'mouthToChinPixelsY': float(points[8, 1] - mouth[1])}


def run(case, chain, frozenZip, output, validateLive=True):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Experiment output must be new')
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / 'policy.json', POLICY)
    geo = arrays(chain / 'finalGeometry.npz')
    before, faces = geo['vertices'], geo['faces']
    jawSource = arrays(chain / 'renderedJaw/sourceState.npz')
    if not np.array_equal(before, jawSource['vertices']):
        raise ValueError('Cached render observations are not from the exact current geometry')
    cameras = json.loads((case / 'reconstruction/cameras.json').read_text())
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    ref = POLICY['referenceViewIndex']
    camera = {key: np.asarray(cameras[ref][key], float) for key in ('intrinsics', 'rotation', 'translation')}
    transform = arrays(chain / 'templateToUff.npz')
    transform['scale'] = float(transform['scale'])
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    original = arrays(case / 'reconstruction/directClient/refinement/clientGeometry.npz')
    originalDirect = original['directTemplateVertices'] @ transform['rotation'] * transform['scale'] + transform['translation']
    observations = arrays(chain / 'renderedJaw/photoObservations.npz')
    observations['diagnostics'] = json.loads((chain / 'renderedJaw/photoObservations.json').read_text())
    baseline = [{'viewIndex': view, 'variants': {name: arrays(chain / f'renderedJaw/baseline/view{view}-{name}.npz')
        for name in APPEARANCES}} for view in range(len(inputs))]
    full = loadClientLandmarks(faces)
    feature = selectClientLandmarks(full, groups=('eyes', 'nose', 'mouth'))
    loops = extractMeshBoundaryLoops(before, faces)['loops']
    protected = np.unique(np.r_[feature['vertexIndices'].ravel(), np.concatenate(loops)])
    static = np.flatnonzero(assets.arrays['staticMask'])
    fixed = np.union1d(static, protected)
    templateVertices = assets.arrays['vertices'][assets.arrays['nicpToFullIds']]
    domain = lowerJawDomain(templateVertices, faces, fixedVertexIds=static, protectedVertexIds=protected)
    consensus = np.mean([baseline[ref]['variants'][name]['points'] for name in APPEARANCES], axis=0)
    ids = np.asarray(POLICY['semanticIds'])
    rays = _ray_bindings(before, faces, cameras[ref], inputs[ref]['workingSize'][::-1], consensus[ids], domain)
    eligible = rays['acceptedMask'].copy() & observations['landmarkCandidateValidMask'][ref, ids]
    for name in APPEARANCES:
        eligible &= baseline[ref]['variants'][name]['valid'][ids]
    if not eligible.all():
        writeJson(output / 'rejectedObservations.json', {'ids': ids, 'eligible': eligible, 'rayReasons': rays['reasons']})
        raise ValueError('Frozen seven-point consensus includes unreliable/anatomically invalid observations')
    photo = observations['faceLandmarksWorking'][ref]
    for source in [photo] + [baseline[ref]['variants'][name]['points'] for name in APPEARANCES]:
        if not np.isfinite(source[np.r_[ids, 48, 54]]).all():
            raise ValueError('Missing current mouth-relative target')
    delta = (photo[ids] - photo[[48, 54]].mean(0)) - (consensus[ids] - consensus[[48, 54]].mean(0))
    peaks = np.minimum.reduce([observations['faceRawPeakScores'][ref, ids]] +
        [baseline[ref]['variants'][name]['peaks'][:, ids].min(0) for name in APPEARANCES])
    scatter = np.sqrt(np.mean([np.linalg.norm(baseline[ref]['variants'][name]['points'][ids] - consensus[ids], axis=1)**2
                              for name in APPEARANCES], axis=0))
    weights = peaks / (1 + (scatter / 2.)**2)
    weights /= weights.sum()
    rest = domain['restCoordinates']
    posterior = np.interp(rest[:, 0], domain['jawDepthX'], domain['jawDepthZ']) - .15
    lower = domain['restIbugCoordinates'][33, 1]
    upper = domain['restIbugCoordinates'][8, 1] + .12
    active = (rest[:, 1] >= lower) & (rest[:, 1] <= upper) & (rest[:, 2] >= posterior)
    active[fixed] = False
    cap = POLICY['maximumDisplacementMm'] / 10. * transform['scale']
    proposal, solve = screenedProportionField(before, faces, camera, rays['vertexIndices'],
        rays['barycentricWeights'], delta, weights, active, cap)
    originalPairs, currentPairs = intersectionPairs(originalDirect, faces), intersectionPairs(before, faces)
    extent = float(np.ptp(originalDirect, axis=0).max())
    depthBudget = float(np.abs((before - originalDirect) @ camera['rotation'][2]).max()) + extent * 1e-9
    attempts, candidate, chosen = [], before.copy(), 0.
    for amount in POLICY['geometryAmounts']:
        trial = before + amount * (proposal - before)
        currentGate = checkFrontGeometry(before, trial, faces, camera=camera, fixedVertexIds=fixed,
            maxDisplacement=cap, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
        cumulativeGate = checkFrontGeometry(originalDirect, trial, faces, camera=camera, fixedVertexIds=static,
            maxDisplacement=extent * .07, maxRelativeEdgeChange=.25, minAreaRatio=.4,
            minNormalCosine=.3, depthTolerance=depthBudget)
        pairs = intersectionPairs(trial, faces)
        currentGate['newIntersectionPairs'] = len(pairs - currentPairs)
        cumulativeGate['newIntersectionPairsAgainstOriginalDirect'] = len(pairs - originalPairs)
        passed = currentGate['accepted'] and cumulativeGate['accepted'] and not (pairs - currentPairs) and not (pairs - originalPairs)
        attempts.append({'amount': amount, 'currentBaseline': currentGate, 'originalDirect': cumulativeGate, 'accepted': bool(passed)})
        if passed:
            candidate, chosen = trial, amount
            break
    np.savez_compressed(output / 'geometry.npz', before=before, proposal=proposal, candidate=candidate,
        after=before, faces=faces, originalDirect=originalDirect, staticVertexIds=static,
        protectedVertexIds=protected, fixedVertexIds=fixed, activeMask=active)
    report = {'status': 'geometryReadyPendingLiveCheck' if chosen else 'rejectedByGeometrySafety',
        'case': case.name, 'referenceViewIndex': ref, 'policy': POLICY, 'selectedGeometryAmount': chosen,
        'solve': solve, 'geometryAttempts': attempts, 'target': {'ids': ids, 'pixelDeltas': delta,
            'weights': weights, 'appearanceScatterPixels': scatter, 'consensusPixels': consensus[ids],
            'rayVertexIndices': rays['vertexIndices'], 'rayBarycentric': rays['barycentricWeights']},
        'sourceHashes': {str(path): fileHash(path) for path in [chain/'finalGeometry.npz', chain/'templateToUff.npz',
            chain/'renderedJaw/sourceState.npz', chain/'renderedJaw/photoObservations.npz', frozenZip, Path(__file__)]},
        'frozenTextureZip': str(frozenZip), 'textureChanged': False, 'cameraChanged': False,
        'frontDepthPreserved': True, 'fixedVerticesExactlyUnchanged': bool(np.array_equal(candidate[fixed], before[fixed])),
        'baselineMeasurements': {name: relativeMetric(baseline[ref]['variants'][name]['points'], photo, ids, weights) for name in APPEARANCES},
        'limitations': ['Automatic FAN proportions are proxies, not anatomical ground truth.',
            'Live detection uses cached template and gray material controls; final frozen eyelid texture is a separate visual check.',
            'Only lower-face width/height relative to mouth is fitted; global camera/face offset and nose depth are untouched.',
            'Rejected candidate remains available for diagnosis; geometry.npz after is baseline unless live gates pass.']}
    writeJson(output / 'report.json', report)
    print(json.dumps({'stage': report['status'], 'selectedGeometryAmount': chosen,
        'output': str(output), 'freeVertexCount': solve['freeVertexCount']}), flush=True)
    if not chosen or not validateLive:
        return
    scene = buildScene(assets, before, transform)
    afterViews, timing = renderDetect(scene, candidate, cameras, inputs, observations,
        output / 'live', projectRoot=ROOT)
    checks, accepted = [], True
    for view in range(len(inputs)):
        target = observations['faceLandmarksWorking'][view]
        row = {'viewIndex': view, 'appearances': []}
        for name in APPEARANCES:
            old, new = baseline[view]['variants'][name], afterViews[view]['variants'][name]
            valid = observations['landmarkCandidateValidMask'][view, ids] & old['valid'][ids]
            usedIds = ids if view == ref else ids[valid]
            usedWeights = weights if view == ref else np.ones(len(usedIds)) / max(len(usedIds), 1)
            if not len(usedIds):
                row['appearances'].append({'variant': name, 'status': 'noBaselinePoints'}); continue
            first, last = relativeMetric(old['points'], target, usedIds, usedWeights), relativeMetric(new['points'], target, usedIds, usedWeights)
            gain = first['meanPixels'] - last['meanPixels']
            needed = max(POLICY['minimumLiveImprovementPixels'], POLICY['minimumRelativeLiveImprovement'] * first['meanPixels']) if view == ref else None
            worst = float(np.max(np.asarray(last['pointErrorsPixels']) - first['pointErrorsPixels']))
            reliable = bool(new['valid'][np.r_[usedIds, 48, 54]].all())
            passed = reliable and (gain >= needed if view == ref else gain >= -1. and worst <= 3.)
            row['appearances'].append({'variant': name, 'before': first, 'after': last, 'improvementPixels': gain,
                'requiredImprovementPixels': needed, 'maximumPointIncreasePixels': worst, 'reliable': reliable, 'accepted': bool(passed)})
            accepted = accepted and passed
        checks.append(row)
    report.update(status='accepted' if accepted else 'rejectedByLiveProportionCheck', liveChecks=checks, liveTiming=timing,
                  finalCandidateAccepted=bool(accepted))
    with np.load(output/'geometry.npz', allow_pickle=False) as stored:
        result = {key: stored[key].copy() for key in stored.files}
    result['after'] = candidate if accepted else before
    np.savez_compressed(output/'geometry.npz', **result)
    writeJson(output/'report.json', report)
    print(json.dumps({'stage': report['status']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--frozen-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-live', action='store_true')
    args = parser.parse_args()
    run(args.case.resolve(), args.chain.resolve(), args.frozen_zip.resolve(), args.output.resolve(), not args.skip_live)
