"""Restore supported outer-nose surface detail lost during client registration.

Original UFF is a reconstruction reference, not anatomical ground truth.
Correspondences are frozen nearest surfaces, gated by normals, reciprocity and
source boundary distance. Nose holes are excluded: closed client patches are
not counterparts of open UFF nostrils. No topology, camera or texture changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from trimesh.triangles import closest_point

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.correspondence import SurfaceIndex
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.geometryRefinement import _meshEdges
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyGeometryVariant import exportLegacyGeometryVariant, _similarity
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.localSurfaceRegistration import solveLocalSurfaceIncrement
from faceAppearance.renderedJawRuntime import APPEARANCES, buildScene, renderDetect
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility


POLICY = {
    'method': 'frozenReciprocalOuterNoseSurfaceRbfIncrement',
    'supportRestExtentFraction': .04, 'sourceBoundaryExcludedRings': 2,
    'maximumCorrespondenceDistanceMm': 2., 'minimumNormalDot': .7,
    'maximumReciprocalDistanceMm': .5, 'maximumDisplacementMm': 2.,
    'minimumSourceVisibleViews': 2, 'minimumViewNormalCosine': .1,
    'minimumFaceMaskSupport': .5, 'visibilityDepthExtentFraction': 1e-6,
    'controlCount': 32, 'controlRadiusMm': 8.,
    'normalWeight': 1., 'tangentWeight': .05,
    'displacementWeight': .01, 'strainWeight': .1,
    'safetyAmounts': [1., .5, .25], 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'heldout': 'Every fourth eligible original client face; never used in solve',
    'minimumHeldoutPlaneImprovementFraction': .15,
    'maximumHeldoutDistanceP95IncreaseMm': .1,
    'referenceViewIndex': 3, 'maximumPhotoMeanErrorIncreasePixels': .5,
    'maximumPhotoPointErrorIncreasePixels': 1.5,
    'adoption': 'Surface metrics alone cannot accept a likeness improvement; manual multiview review remains required',
}


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0, 'mean': None, 'median': None, 'p95': None, 'maximum': None}
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def normals(vertices, faces):
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    return cross / np.linalg.norm(cross, axis=1, keepdims=True), np.linalg.norm(cross, axis=1) / 2


def exactQuery(index, points):
    # SurfaceIndex keeps compact float32 output for stored maps. Re-evaluate
    # the selected exact triangle in float64 for the small registration solve.
    ids = index.query(points)['triangleIds']
    closest = closest_point(index.triangles[ids], np.asarray(points, float))
    return ids, closest, np.linalg.norm(closest - points, axis=1)


def sourceVisibility(source, cameras, inputs, points, targetNormals):
    """Count original-photo support using continuous source depth and four mask taps."""
    vertices, faces = source['vertices'].astype(float), source['faces']
    masks = np.asarray(source['faceMasks'])
    if (masks.ndim != 3 or len(masks) != len(cameras) or len(inputs) != len(cameras)
            or len(cameras) < POLICY['minimumSourceVisibleViews'] or not np.isfinite(masks).all()):
        raise ValueError('Expected finite face masks for every source camera')
    tolerance = np.ptp(vertices, axis=0).max() * POLICY['visibilityDepthExtentFraction']
    visibility = np.zeros((len(cameras), len(points)), bool)
    for view, (camera, metadata, mask) in enumerate(zip(cameras, inputs, masks)):
        if not camera.get('valid') or camera.get('viewIndex') != view:
            raise ValueError('Expected valid ordered source cameras')
        rotation, translation, intrinsics = (np.asarray(camera[key], float)
            for key in ('rotation', 'translation', 'intrinsics'))
        height, width = mask.shape
        if [width, height] != metadata['workingSize']:
            raise ValueError('Photo-mask shape differs from saved camera image')
        cameraPoints = points @ rotation.T + translation
        depths = ProjectedSurfaceVisibility(vertices @ rotation.T + translation, faces,
            intrinsics, (height, width)).closestDepth(cameraPoints)
        projected = cameraPoints @ intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:]
        taps = np.floor(pixels).astype(int)
        inside = (depths['insidePhoto'] & (taps >= 0).all(1)
            & (taps[:, 0]+1 < width) & (taps[:, 1]+1 < height))
        supported = np.zeros(len(points), bool)
        x, y = taps[inside].T
        supported[inside] = np.minimum.reduce([mask[y, x], mask[y, x+1],
            mask[y+1, x], mask[y+1, x+1]]) >= POLICY['minimumFaceMaskSupport']
        viewDirection = -cameraPoints / np.linalg.norm(cameraPoints, axis=1, keepdims=True)
        cosine = np.sum((targetNormals @ rotation.T) * viewDirection, axis=1)
        visibility[view] = (inside & supported & np.isfinite(depths['depth'])
            & (np.abs(depths['depth']-cameraPoints[:, 2]) <= tolerance)
            & (cosine >= POLICY['minimumViewNormalCosine']))
    return visibility


def verifyInputs(identityPath, expectedIdentityHash=None):
    if expectedIdentityHash is not None and digest(identityPath) != expectedIdentityHash:
        raise ValueError('Frozen input identity manifest changed')
    identity = json.loads(identityPath.read_text())
    for path, expected in identity['files'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen experiment input changed: {path}')
    return identity


def correspondence(points, queryNormals, sourceIndex, currentIndex, excludedSourceFaces, unit):
    ids, targets, distances = exactQuery(sourceIndex, points)
    targetNormals = sourceIndex.faceNormals[ids]
    dots = np.sum(queryNormals * targetNormals, axis=1)
    _, reciprocalPoints, _ = exactQuery(currentIndex, targets)
    reciprocalDistance = np.linalg.norm(reciprocalPoints - points, axis=1)
    gates = {
        'awayFromSourceBoundary': ~excludedSourceFaces[ids],
        'nearbySurface': distances <= POLICY['maximumCorrespondenceDistanceMm'] * unit,
        'compatibleNormals': dots >= POLICY['minimumNormalDot'],
        'reciprocalSurface': reciprocalDistance <= POLICY['maximumReciprocalDistanceMm'] * unit,
    }
    accepted = np.logical_and.reduce(list(gates.values()))
    return {'targetFaceIds': ids, 'targetPoints': targets, 'targetNormals': targetNormals,
            'distance': distances, 'normalDot': dots, 'reciprocalDistance': reciprocalDistance,
            'acceptedMask': accepted, 'gates': gates}


def surfaceMetrics(vertices, sampleFaces, bary, targets, targetNormals, sourceIndex, unit):
    points = (vertices[sampleFaces] * bary[..., None]).sum(1)
    residual = points - targets
    _, _, nearestDistance = exactQuery(sourceIndex, points)
    normalResidual = np.sum(residual * targetNormals, axis=1) / unit
    return {'planeRmsMm': float(np.sqrt(np.mean(normalResidual**2))),
            'frozenTargetDistanceMm': summary(np.linalg.norm(residual, axis=1) / unit),
            'nearestSurfaceDistanceMm': summary(nearestDistance / unit)}


def run(case, chain, frozenZip, output, identityPath, skipLive=False):
    identityHash = digest(identityPath)
    identity = verifyInputs(identityPath)
    for required in [frozenZip, case/'reconstruction/geometry.npz', case/'reconstruction/cameras.json',
        case/'reconstruction/inputs.json', chain/'finalGeometry.npz', chain/'templateToUff.npz']:
        if str(required) not in identity['files']:
            raise ValueError(f'Missing frozen input identity: {required}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new surface registration experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / 'policy.json', POLICY)
    geometry = arrays(chain / 'finalGeometry.npz')
    before, faces = geometry['vertices'], geometry['faces']
    source = arrays(case / 'reconstruction/geometry.npz')
    sourceVertices, sourceFaces = source['vertices'].astype(float), source['faces']
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    transform = arrays(chain / 'templateToUff.npz')
    unit = float(transform['scale']) / 10.  # canonical length per nominal template mm
    rest = assets.arrays['vertices'][assets.arrays['nicpToFullIds']]
    cameras = json.loads((case / 'reconstruction/cameras.json').read_text())
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    camera = {key: np.asarray(cameras[3][key]) for key in ('intrinsics', 'rotation', 'translation')}
    archive = _load(frozenZip)
    archiveRotation, archiveTranslation, archiveScale = _similarity(archive.manifest)
    for actual, expected in [(archiveRotation, transform['rotation']),
        (archiveTranslation, transform['translation']), (archiveScale, transform['scale'])]:
        np.testing.assert_array_equal(actual, expected)
    fullBaseline = archive.mesh['vertices'] * 100 @ transform['rotation'] * float(transform['scale']) + transform['translation']
    skinIds = assets.arrays['nicpToFullIds']
    inverseSkin = np.full(len(fullBaseline), -1, int)
    inverseSkin[skinIds] = np.arange(len(skinIds))
    skinFaceMask = (inverseSkin[archive.mesh['faces']] >= 0).all(1)
    np.testing.assert_array_equal(inverseSkin[archive.mesh['faces'][skinFaceMask]], faces)
    np.testing.assert_array_equal(archive.mesh['faces'], assets.arrays['faces'])
    np.testing.assert_array_equal(archive.mesh['uvs'], geometry['uvs'])
    np.testing.assert_array_equal(archive.mesh['uvFaces'][skinFaceMask], geometry['uvFaces'])
    np.testing.assert_array_equal(archive.mesh['materialIds'], assets.arrays['materialIds'])
    np.testing.assert_allclose(fullBaseline[skinIds], before, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(sourceFaces, assets.arrays['sourceFaces'])
    sourceIndex, currentIndex = SurfaceIndex(sourceVertices, sourceFaces), SurfaceIndex(before, faces)
    sourceBoundary = np.zeros(len(sourceVertices), bool)
    sourceLoops = extractMeshBoundaryLoops(sourceVertices, sourceFaces)['loops']
    sourceBoundary[np.concatenate(sourceLoops)] = True
    sourceEdges = _meshEdges(sourceFaces)
    for _ in range(POLICY['sourceBoundaryExcludedRings']):
        sourceBoundary[sourceEdges[sourceBoundary[sourceEdges].any(1)].ravel()] = True
    excludedSourceFaces = sourceBoundary[sourceFaces].any(1)
    binding = loadClientLandmarks(faces)
    seeds = np.unique(selectClientLandmarks(binding, groups='nose')['vertexIndices'])
    protected = np.unique(np.r_[
        selectClientLandmarks(binding, groups=('eyes', 'mouth'))['vertexIndices'].ravel(),
        np.concatenate(extractMeshBoundaryLoops(before, faces)['loops'])])
    static = np.flatnonzero(assets.arrays['staticMask'])
    edges = _meshEdges(faces)
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    graph = sparse.coo_matrix((np.r_[lengths, lengths],
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(rest), len(rest))).tocsr()
    distances = dijkstra(graph, directed=False, indices=seeds, min_only=True)
    active = distances <= POLICY['supportRestExtentFraction'] * np.ptp(rest, axis=0).max()
    active[np.union1d(protected, static)] = False
    # Do not deform client points whose nearest UFF support is a cut boundary.
    # This freezes the unsupported caps rather than pulling them onto hole rims.
    activeIds = np.flatnonzero(active)
    nearestIds, _, _ = exactQuery(sourceIndex, before[activeIds])
    active[activeIds[excludedSourceFaces[nearestIds]]] = False
    fixed = np.flatnonzero(~active)
    faceNormals, faceAreas = normals(before, faces)
    eligibleFaces = np.flatnonzero(active[faces].all(1))
    sampleFaces = faces[eligibleFaces]
    bary = np.full(sampleFaces.shape, 1/3.)
    points = (before[sampleFaces] * bary[..., None]).sum(1)
    matching = correspondence(points, faceNormals[eligibleFaces], sourceIndex, currentIndex, excludedSourceFaces, unit)
    visibility = sourceVisibility(source, cameras, inputs, matching['targetPoints'], matching['targetNormals'])
    matching['sourceVisibleViews'] = visibility.sum(0)
    matching['gates']['multiplePhotoViews'] = matching['sourceVisibleViews'] >= POLICY['minimumSourceVisibleViews']
    matching['acceptedMask'] &= matching['gates']['multiplePhotoViews']
    eligible = matching['acceptedMask']
    heldout = eligible & (eligibleFaces % 4 == 0)
    training = eligible & ~heldout
    if training.sum() < 32 or heldout.sum() < 16:
        raise ValueError('Insufficient supported outer-nose samples for independent geometric validation')
    weights = faceAreas[eligibleFaces] * matching['normalDot'].clip(0, 1)**2
    proposal, solve = solveLocalSurfaceIncrement(before, faces, sampleFaces[training], bary[training],
        matching['targetPoints'][training], matching['targetNormals'][training], weights[training],
        activeMask=active, maxDisplacement=POLICY['maximumDisplacementMm']*unit,
        controlCount=POLICY['controlCount'], controlRadius=POLICY['controlRadiusMm']*unit,
        normalWeight=POLICY['normalWeight'], tangentWeight=POLICY['tangentWeight'],
        displacementWeight=POLICY['displacementWeight'], strainWeight=POLICY['strainWeight'])
    original = arrays(case / 'reconstruction/directClient/refinement/clientGeometry.npz')
    for name, key in [('rotation', 'templateToUffRotation'), ('translation', 'templateToUffTranslation'),
                      ('scale', 'templateToUffScale')]:
        np.testing.assert_allclose(transform[name], original[key], rtol=0, atol=1e-12)
    pureDirect = original['directTemplateVertices'] @ transform['rotation'] * float(transform['scale']) + transform['translation']
    currentPairs, originalPairs = intersectionPairs(before, faces), intersectionPairs(pureDirect, faces)
    fullPairs = intersectionPairs(fullBaseline, archive.mesh['faces'])
    cap = POLICY['maximumDisplacementMm'] * unit
    existingDepth = np.abs((before - pureDirect) @ camera['rotation'][2]).max()
    attempts, amount, candidate = [], 0., before.copy()
    for step in POLICY['safetyAmounts']:
        trial = before + step * (proposal - before)
        current = checkFrontGeometry(before, trial, faces, fixedVertexIds=fixed, camera=camera,
            maxDisplacement=cap, maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
            minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'], depthTolerance=cap)
        cumulative = checkFrontGeometry(pureDirect, trial, faces, fixedVertexIds=static, camera=camera,
            maxDisplacement=.07*np.ptp(pureDirect, axis=0).max(), maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
            minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'], depthTolerance=float(existingDepth + cap))
        pairs = intersectionPairs(trial, faces)
        addedCurrent, addedOriginal = len(pairs-currentPairs), len(pairs-originalPairs)
        accepted = current['accepted'] and cumulative['accepted'] and not addedCurrent and not addedOriginal
        addedFull = None
        if accepted:
            trialFull = fullBaseline.copy()
            changed = np.any(trial != before, axis=1)
            trialFull[skinIds[changed]] = trial[changed]
            addedFull = len(intersectionPairs(trialFull, archive.mesh['faces']) - fullPairs)
            accepted = not addedFull
        attempts.append({'amount': step, 'current': current, 'originalDirect': cumulative,
            'newCurrentIntersections': addedCurrent, 'newOriginalIntersections': addedOriginal,
            'newAssembledIntersections': addedFull, 'accepted': bool(accepted)})
        if accepted:
            candidate, amount = trial, step
            break
    checks = {}
    for name, mask in [('training', training), ('heldout', heldout)]:
        arguments = (sampleFaces[mask], bary[mask], matching['targetPoints'][mask], matching['targetNormals'][mask], sourceIndex, unit)
        checks[name] = {'before': surfaceMetrics(before, *arguments), 'candidate': surfaceMetrics(candidate, *arguments)}
    first, last = checks['heldout']['before'], checks['heldout']['candidate']
    surfacePassed = (amount > 0 and last['planeRmsMm'] <= (1-POLICY['minimumHeldoutPlaneImprovementFraction'])*first['planeRmsMm']
        and last['nearestSurfaceDistanceMm']['p95'] <= first['nearestSurfaceDistanceMm']['p95'] + POLICY['maximumHeldoutDistanceP95IncreaseMm'])
    np.savez_compressed(output / 'geometry.npz', before=before, proposal=proposal, candidate=candidate, after=before,
        faces=faces, activeMask=active, fixedVertexIds=fixed, protectedVertexIds=protected,
        staticVertexIds=static, originalDirect=pureDirect, sourceBoundaryExcludedMask=sourceBoundary)
    np.savez_compressed(output / 'correspondence.npz', sampleFaceIds=eligibleFaces, sampleVertexIndices=sampleFaces,
        sampleBarycentric=bary, trainingMask=training, heldoutMask=heldout, weights=weights,
        **{key: value for key, value in matching.items() if isinstance(value, np.ndarray)})
    report = {'status': 'geometryReadyPendingPhotoReview' if amount else 'rejectedByGeometrySafety',
        'case': case.name, 'policy': POLICY, 'selectedSafetyAmount': amount, 'solve': solve, 'safetyAttempts': attempts,
        'activeVertexCount': int(active.sum()), 'candidateSampleCount': len(points),
        'trainingCount': int(training.sum()), 'heldoutCount': int(heldout.sum()),
        'inputIdentity': {'path': str(identityPath), 'sha256': identityHash, 'files': identity['files']},
        'sourceVisibleViewHistogram': {str(count): int((visibility.sum(0)==count).sum()) for count in range(len(cameras)+1)},
        'correspondenceGateCounts': {key: int(value.sum()) for key, value in matching['gates'].items()},
        'surfaceMetrics': checks, 'surfaceValidationPassed': bool(surfacePassed), 'finalCandidateAccepted': False,
        'inactiveVerticesExactlyUnchanged': bool(np.array_equal(candidate[fixed], before[fixed])),
        'maximumDisplacementMm': float(np.linalg.norm(candidate-before, axis=1).max()/unit),
        'topologyUvAndBindingChanged': False, 'cameraChanged': False, 'textureChanged': False,
        'sourceHashes': {str(path): digest(path) for path in [chain/'finalGeometry.npz', chain/'templateToUff.npz',
            case/'reconstruction/geometry.npz', case/'reconstruction/cameras.json', frozenZip, Path(__file__),
            ROOT/'src/faceAppearance/localSurfaceRegistration.py']},
        'limitations': ['Closest surface is not a semantic or true-anatomy correspondence.',
            'UFF nostril boundary bands are excluded; this does not create nostril cavities in the fixed client topology.',
            'Heldout geometric fidelity to UFF does not prove photographic likeness.',
            'Surface triangle area weights reduce sampling-density bias but are not measurement uncertainty estimates.']}
    verifyInputs(identityPath, identityHash)
    if amount:
        fullCandidate = fullBaseline.copy()
        changed = np.any(candidate != before, axis=1)
        fullCandidate[skinIds[changed]] = candidate[changed]
        report['variant'] = exportLegacyGeometryVariant(frozenZip, fullCandidate, output/'diagnosticAsset', diagnostics={
            'diagnosticOnly': True, 'method': POLICY['method'], 'geometrySafetyPassed': True,
            'surfaceValidationPassed': bool(surfacePassed), 'liveAcceptancePassed': False})
    writeJson(output/'report.json', report)
    print(json.dumps({'stage': report['status'], 'amount': amount, 'activeVertices': int(active.sum()),
        'heldoutPlaneRmsMm': [first['planeRmsMm'], last['planeRmsMm']], 'output': str(output)}), flush=True)
    if not amount or skipLive:
        return report
    photoData = arrays(chain/'renderedJaw/photoObservations.npz')
    np.testing.assert_array_equal(before, arrays(chain/'renderedJaw/sourceState.npz')['vertices'])
    photoData['diagnostics'] = json.loads((chain/'renderedJaw/photoObservations.json').read_text())
    scene = buildScene(assets, before, transform)
    afterViews, timing = renderDetect(scene, candidate, cameras, inputs, photoData, output/'live', projectRoot=ROOT)
    photoChecks = []
    for view in range(len(cameras)):
        items = []
        for name in APPEARANCES:
            old = arrays(chain/f'renderedJaw/baseline/view{view}-{name}.npz')
            new = afterViews[view]['variants'][name]
            ids = np.arange(27, 36)
            reliable = old['valid'][ids] & photoData['landmarkCandidateValidMask'][view, ids]
            selected = ids[reliable]
            if len(selected):
                beforeError = np.linalg.norm(old['points'][selected]-photoData['faceLandmarksWorking'][view, selected], axis=1)
                afterError = np.linalg.norm(new['points'][selected]-photoData['faceLandmarksWorking'][view, selected], axis=1)
                passed = bool(new['valid'][selected].all()
                    and (afterError-beforeError).mean() <= POLICY['maximumPhotoMeanErrorIncreasePixels']
                    and (afterError-beforeError).max() <= POLICY['maximumPhotoPointErrorIncreasePixels'])
                items.append({'appearance': name, 'ids': selected, 'before': summary(beforeError),
                    'candidate': summary(afterError), 'meanImprovementPixels': float((beforeError-afterError).mean()),
                    'maximumPointDegradationPixels': float((afterError-beforeError).max()), 'nonDegradationPassed': passed})
            else:
                items.append({'appearance': name, 'ids': [], 'nonDegradationPassed': None})
        photoChecks.append({'viewIndex': view, 'checks': items})
    report.update(status='completeDiagnosticPendingManualReview', photoChecks=photoChecks, liveTiming=timing)
    verifyInputs(identityPath, identityHash)
    writeJson(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'chain', 'baseline', 'output', 'identity'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--skip-live', action='store_true')
    args = parser.parse_args()
    run(args.case.resolve(), args.chain.resolve(), args.baseline.resolve(), args.output.resolve(), args.identity.resolve(), args.skip_live)
