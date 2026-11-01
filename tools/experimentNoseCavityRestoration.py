"""Restore a bounded amount of existing template cavity depth, locally.

Only the audited interior vertices may move. Rim, outer nose, columella,
binding, cameras and texture are fixed. Photometric contours are validation
evidence, never assumed to be a fully identified anatomical rim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.cavityPrior import buildCavityDepthPrior
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyGeometryVariant import _similarity, exportLegacyGeometryVariant
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.localSurfaceRegistration import solveLocalSurfaceIncrement
from faceAppearance.renderedJawRuntime import APPEARANCES, buildScene, renderDetect
from experimentNoseSurfaceRegistration import arrays, digest, summary, verifyInputs

POLICY = {'method': 'fixedRimMissingTemplateCavityDepthRbf', 'restorationFraction': .35,
    'maximumDisplacementMm': 2., 'controlCount': 12, 'controlRadiusMm': 4.,
    'normalWeight': 1., 'tangentWeight': .05, 'displacementWeight': .01, 'strainWeight': .1,
    'maximumRimPlaneRmsOverRadius': .3, 'minimumCurrentRestNormalDot': .7,
    'safetyAmounts': [1., .5, .25], 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'maximumPhotoMeanIncreasePixels': .5, 'maximumPhotoPointIncreasePixels': 1.5,
    'source': 'Accepted eyelid baseline; previous experimental surface geometry is not accumulated',
    'boundary': 'All 29 audited rim and all exterior skin/attachments remain exactly fixed',
    'photometricContourRole': 'Independent partial weak edge validation; no forced correspondence to initialization rim',
    'acceptance': 'Diagnostic candidate requires original four-camera gray and frozen-texture review; template depth is not subject truth'}
REGIONS = ('frontImageLeft', 'frontImageRight')


def run(case, chain, baseline, audit, identityPath, output):
    identity = verifyInputs(identityPath)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh cavity experiment directory')
    regionPath, regionReportPath = audit/'regions.npz', audit/'report.json'
    regions, regionReport = arrays(regionPath), json.loads(regionReportPath.read_text())
    if digest(regionPath) != regionReport['outputHashes']['regions.npz']:
        raise ValueError('Audited cavity regions changed')
    for name, expected in regionReport['sourceHashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Audited source changed: {name}')
    frozenPaths = [identityPath, regionPath, regionReportPath, baseline,
        ROOT/'src/faceAppearance/cavityPrior.py', ROOT/'src/faceAppearance/localSurfaceRegistration.py', Path(__file__)]
    frozenHashes = {str(path): digest(path) for path in frozenPaths}
    geometry = arrays(chain/'finalGeometry.npz')
    before, faces, rest = geometry['vertices'], geometry['faces'], regions['skinVerticesRestAligned']
    np.testing.assert_array_equal(before, regions['skinVerticesCurrent'])
    np.testing.assert_array_equal(faces, regions['skinFaces'])
    np.testing.assert_array_equal(before, arrays(chain/'renderedJaw/sourceState.npz')['vertices'])
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    transform = arrays(chain/'templateToUff.npz')
    unit = float(transform['scale'])/10.
    np.testing.assert_allclose(1/unit, regions['templateMillimetersPerCanonical'], rtol=0, atol=1e-10)
    archive = _load(baseline)
    rotation, translation, scale = _similarity(archive.manifest)
    for first, last in [(rotation, transform['rotation']), (translation, transform['translation']), (scale, transform['scale'])]:
        np.testing.assert_array_equal(first, last)
    skinIds = assets.arrays['nicpToFullIds']
    fullBefore = archive.mesh['vertices']*100 @ rotation*scale+translation
    np.testing.assert_allclose(fullBefore[skinIds], before, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(archive.mesh['faces'], assets.arrays['faces'])
    inverse = np.full(len(fullBefore), -1, int); inverse[skinIds] = np.arange(len(skinIds))
    skinFaces = (inverse[archive.mesh['faces']] >= 0).all(1)
    np.testing.assert_array_equal(inverse[archive.mesh['faces'][skinFaces]], faces)
    np.testing.assert_array_equal(archive.mesh['uvs'], geometry['uvs'])
    np.testing.assert_array_equal(archive.mesh['uvFaces'][skinFaces], geometry['uvFaces'])
    cameras = json.loads((case/'reconstruction/cameras.json').read_text())
    inputs = json.loads((case/'reconstruction/inputs.json').read_text())
    camera = {key: np.asarray(cameras[3][key]) for key in ('intrinsics', 'rotation', 'translation')}
    cap = POLICY['maximumDisplacementMm']*unit
    active = np.zeros(len(before), bool)
    priors, rimIds = {}, []
    for name in REGIONS:
        rim, interior = regions[name+'_rimVertexIdsOrdered'], regions[name+'_interiorVertexIds']
        if active[interior].any():
            raise ValueError('Cavity interiors overlap')
        active[interior] = True; rimIds.extend(rim)
        prior = buildCavityDepthPrior(before, rest, rim, interior,
            patchFaces=faces[regions[name+'_faceIds']], inwardDirection=-regions[name+'_currentPlaneBasis'][:, 2],
            maxAddedDepth=cap, restorationFraction=POLICY['restorationFraction'])
        diagnostics = prior['diagnostics']
        if (max(diagnostics['currentRimPlaneRmsOverRadius'], diagnostics['restRimPlaneRmsOverRadius'])
                > POLICY['maximumRimPlaneRmsOverRadius']
                or diagnostics['inwardNormalDot'] < POLICY['minimumCurrentRestNormalDot']):
            raise ValueError('Unstable or inconsistent cavity boundary frames')
        priors[name] = prior
    protected = np.unique(np.r_[rimIds, regions['columellaGuardVertexIds'], np.flatnonzero(assets.arrays['staticMask'])])
    if active[protected].any():
        raise ValueError('Cavity interior overlaps a protected region')
    fixed = np.flatnonzero(~active)
    joined = {key: np.concatenate([priors[name][key] for name in REGIONS])
        for key in ('vertexIndices', 'barycentric', 'targetPoints', 'targetNormals', 'weights')}
    proposal, solve = solveLocalSurfaceIncrement(before, faces, joined['vertexIndices'], joined['barycentric'],
        joined['targetPoints'], joined['targetNormals'], joined['weights'], activeMask=active,
        maxDisplacement=cap, controlCount=POLICY['controlCount'], controlRadius=POLICY['controlRadiusMm']*unit,
        normalWeight=POLICY['normalWeight'], tangentWeight=POLICY['tangentWeight'],
        displacementWeight=POLICY['displacementWeight'], strainWeight=POLICY['strainWeight'])
    pureDirect = regions['skinVerticesPureDirect']
    original = arrays(case/'reconstruction/directClient/refinement/clientGeometry.npz')
    np.testing.assert_allclose(pureDirect, original['directTemplateVertices']@rotation*scale+translation, rtol=0, atol=1e-12)
    pairsBefore, pairsOriginal, pairsFull = intersectionPairs(before, faces), intersectionPairs(pureDirect, faces), intersectionPairs(fullBefore, archive.mesh['faces'])
    existingDepth = float(np.abs((before-pureDirect)@camera['rotation'][2]).max())
    candidate, amount, attempts = before.copy(), 0., []
    for step in POLICY['safetyAmounts']:
        trial = before+step*(proposal-before)
        common = dict(camera=camera, maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
            minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'])
        local = checkFrontGeometry(before, trial, faces, fixedVertexIds=fixed,
            maxDisplacement=cap, depthTolerance=cap, **common)
        cumulative = checkFrontGeometry(pureDirect, trial, faces,
            fixedVertexIds=np.flatnonzero(assets.arrays['staticMask']),
            maxDisplacement=.07*np.ptp(pureDirect, axis=0).max(), depthTolerance=existingDepth+cap, **common)
        trialPairs = intersectionPairs(trial, faces)
        full = fullBefore.copy(); changed = np.any(trial != before, axis=1); full[skinIds[changed]] = trial[changed]
        added = {'currentSkin': len(trialPairs-pairsBefore), 'originalSkin': len(trialPairs-pairsOriginal),
                 'assembled': len(intersectionPairs(full, archive.mesh['faces'])-pairsFull)}
        passed = bool(local['accepted'] and cumulative['accepted'] and not any(added.values()))
        attempts.append({'amount': step, 'current': local, 'originalDirect': cumulative,
            'newIntersectionCounts': added, 'accepted': passed})
        if passed:
            candidate, amount = trial, step
            break
    depthMetrics = {}
    for name in REGIONS:
        current = buildCavityDepthPrior(candidate, rest, regions[name+'_rimVertexIdsOrdered'],
            regions[name+'_interiorVertexIds'], patchFaces=faces[regions[name+'_faceIds']],
            inwardDirection=-regions[name+'_currentPlaneBasis'][:, 2], maxAddedDepth=cap,
            restorationFraction=POLICY['restorationFraction'])
        depthMetrics[name] = {'beforeCorrectedDepthMm': summary(priors[name]['beforeDepth']/unit),
            'candidateCorrectedDepthMm': summary(current['beforeDepth']/unit),
            'scaledRestCorrectedDepthMm': summary(priors[name]['scaledRestDepth']/unit),
            'meanActualDepthIncreaseMm': float((current['beforeDepth']-priors[name]['beforeDepth']).mean()/unit)}
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output/'policy.json', POLICY)
    np.savez_compressed(output/'geometry.npz', before=before, proposal=proposal, candidate=candidate,
        after=before, faces=faces, activeMask=active, fixedVertexIds=fixed, rimVertexIds=np.asarray(rimIds),
        columellaGuardVertexIds=regions['columellaGuardVertexIds'])
    np.savez_compressed(output/'depthAnchors.npz', **joined)
    report = {'status': 'geometryReady' if amount else 'rejectedByGeometrySafety', 'case': case.name,
        'policy': POLICY, 'sourceHashes': frozenHashes, 'frozenInputIdentity': identity,
        'selectedSafetyAmount': amount, 'activeVertexCount': int(active.sum()), 'fixedVertexCount': len(fixed),
        'maximumDisplacementMm': float(np.linalg.norm(candidate-before, axis=1).max()/unit),
        'rimAndExteriorExactlyFixed': bool(np.array_equal(before[fixed], candidate[fixed])),
        'solve': solve, 'depthPrior': {name: priors[name]['diagnostics'] for name in REGIONS},
        'depthMetrics': depthMetrics, 'safetyAttempts': attempts, 'finalCandidateAccepted': False,
        'cameraChanged': False, 'textureChanged': False, 'topologyUvBindingChanged': False,
        'limitations': ['Audited patches are existing closed cavities, but their outer loops are not reviewed true nostril rims.',
            'Template depth is an explicit prior; the subject cavity is not reconstructed from photographs.',
            'Only cavity depth is repaired in this ablation; original nose/nostril position and appearance errors remain.']}
    for path, expected in frozenHashes.items():
        if digest(path) != expected: raise ValueError('Experiment source changed')
    verifyInputs(identityPath)
    if amount:
        full = fullBefore.copy(); changed = np.any(candidate != before, axis=1); full[skinIds[changed]] = candidate[changed]
        report['variant'] = exportLegacyGeometryVariant(baseline, full, output/'diagnosticAsset', diagnostics={
            'diagnosticOnly': True, 'method': POLICY['method'], 'geometrySafetyPassed': True, 'liveAcceptancePassed': False})
    writeJson(output/'report.json', report)
    print(json.dumps({'stage': report['status'], 'amount': amount, 'maximumDisplacementMm': report['maximumDisplacementMm'],
        'depthMetrics': depthMetrics}), flush=True)
    if not amount: return report
    photo = arrays(chain/'renderedJaw/photoObservations.npz')
    photo['diagnostics'] = json.loads((chain/'renderedJaw/photoObservations.json').read_text())
    views, timing = renderDetect(buildScene(assets, before, transform), candidate, cameras, inputs, photo, output/'live', projectRoot=ROOT)
    checks = []
    for view in range(len(cameras)):
        items = []
        for appearance in APPEARANCES:
            old = arrays(chain/f'renderedJaw/baseline/view{view}-{appearance}.npz'); new = views[view]['variants'][appearance]
            ids = np.arange(27, 36)
            ids = ids[old['valid'][ids] & photo['landmarkCandidateValidMask'][view, ids]]
            if not len(ids):
                items.append({'appearance': appearance, 'nonDegradationPassed': None}); continue
            target = photo['faceLandmarksWorking'][view, ids]
            beforeError, afterError = np.linalg.norm(old['points'][ids]-target, axis=1), np.linalg.norm(new['points'][ids]-target, axis=1)
            items.append({'appearance': appearance, 'ids': ids, 'meanImprovementPixels': float((beforeError-afterError).mean()),
                'maximumPointDegradationPixels': float((afterError-beforeError).max()),
                'nonDegradationPassed': bool(new['valid'][ids].all()
                    and (afterError-beforeError).mean() <= POLICY['maximumPhotoMeanIncreasePixels']
                    and (afterError-beforeError).max() <= POLICY['maximumPhotoPointIncreasePixels'])})
        checks.append({'viewIndex': view, 'checks': items})
    report.update(status='completeDiagnosticPendingPhotoContourReview', photoChecks=checks, liveTiming=timing)
    verifyInputs(identityPath)
    writeJson(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('case', 'chain', 'baseline', 'audit', 'identity', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    options = parser.parse_args()
    run(*(getattr(options, key).resolve() for key in ('case', 'chain', 'baseline', 'audit', 'identity', 'output')))
