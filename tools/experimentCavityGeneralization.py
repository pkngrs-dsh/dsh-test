"""Run a frozen, template-only nasal-cavity ablation on independent subjects.

Each subject supplies its own delivered geometry, original direct registration,
similarity and cameras. Only the 69 existing cavity interior vertices may move.
The old AI example contributes audited topology IDs, never subject coordinates,
photo observations or fitted deformation. Passing geometric checks produces a
diagnostic archive; it does not establish a photographic likeness improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.cavityPrior import buildCavityDepthPrior
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.legacyGeometryVariant import _similarity, exportLegacyGeometryVariant
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.localSurfaceRegistration import solveLocalSurfaceIncrement
from experimentNoseSurfaceRegistration import arrays, digest, summary

REGIONS = ('frontImageLeft', 'frontImageRight')
POLICY = {
    'method': 'independentSubjectFixedRimMissingTemplateCavityDepthRbf',
    'restorationFraction': .35, 'maximumDisplacementMm': 2.,
    'controlCount': 12, 'controlRadiusMm': 4., 'normalWeight': 1.,
    'tangentWeight': .05, 'displacementWeight': .01, 'strainWeight': .1,
    'maximumRimPlaneRmsOverRadius': .3, 'minimumCurrentRestNormalDot': .7,
    'safetyAmounts': [1., .5, .25], 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'cumulativeDisplacementExtentFraction': .07,
    'baselineArchiveCanonicalAgreementTolerance': 1e-12,
    'referenceViewIndex': 0,
    'source': 'Each subject original production ZIP and cached float64 canonical geometry',
    'boundary': 'Only 69 cavity-interior vertices; 29 patch-rim and all other vertices fixed',
    'acceptance': 'Geometry gates only; independent original-camera photo review required',
}


def verifyHashes(hashes):
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def candidateFullGeometry(fullBefore, skinIds, before, candidate):
    """Avoid rewriting protected OBJ vertices after a numerical round trip."""
    changed = np.any(candidate != before, axis=1)
    full = fullBefore.copy()
    full[skinIds[changed]] = candidate[changed]
    return full


def runCase(auditDirectory, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh cavity generalization directory')
    geometryPath, auditPath = auditDirectory / 'geometry.npz', auditDirectory / 'report.json'
    audit = json.loads(auditPath.read_text())
    geometry = arrays(geometryPath)
    case = ROOT / 'reports/feedbackAudit20261024/cases' / audit['caseId']
    baseline = case / 'asset/legacyAsset.zip'
    cachedGeometryPath = case / 'reconstruction/directClient/geometry.npz'
    camerasPath = case / 'reconstruction/cameras.json'
    frozenHashes = {str(path.resolve()): digest(path) for path in [
        geometryPath, auditPath, baseline, cachedGeometryPath, camerasPath,
        ROOT / 'src/faceAppearance/cavityPrior.py',
        ROOT / 'src/faceAppearance/localSurfaceRegistration.py', Path(__file__),
    ]}
    # The audit hashes every source used for its pure-direct replay and frame.
    for path, expected in audit['sourceHashes'].items():
        frozenHashes[str((ROOT / path).resolve())] = expected
    verifyHashes(frozenHashes)
    archive = _load(baseline)
    rotation, translation, scale = _similarity(archive.manifest)
    for actual, expected in ((rotation, geometry['rotation']),
                             (translation, geometry['translation']),
                             (scale, geometry['scale'])):
        np.testing.assert_array_equal(actual, expected)
    fullBefore = archive.mesh['vertices'] * 100 @ rotation * scale + translation
    np.testing.assert_array_equal(fullBefore, geometry['fullDelivered'])
    skinIds, faces = geometry['nicpToFullIds'], geometry['faces']
    cachedGeometry = arrays(cachedGeometryPath)
    before, rest, pureDirect = cachedGeometry['vertices'], geometry['alignedRest'], geometry['pureDirect']
    np.testing.assert_array_equal(cachedGeometry['faces'], faces)
    np.testing.assert_array_equal(geometry['delivered'], fullBefore[skinIds])
    archiveRoundTripError = float(np.abs(before - geometry['delivered']).max())
    if archiveRoundTripError > POLICY['baselineArchiveCanonicalAgreementTolerance']:
        raise ValueError('Cached canonical final geometry differs materially from delivered ZIP')
    inverse = np.full(len(fullBefore), -1, int)
    inverse[skinIds] = np.arange(len(skinIds))
    skinFaceMask = (inverse[archive.mesh['faces']] >= 0).all(1)
    np.testing.assert_array_equal(inverse[archive.mesh['faces'][skinFaceMask]], faces)
    np.testing.assert_array_equal(archive.mesh['uvs'], cachedGeometry['uvs'])
    np.testing.assert_array_equal(archive.mesh['uvFaces'][skinFaceMask], cachedGeometry['uvFaces'])
    unit = scale / 10.
    np.testing.assert_allclose(1 / unit, geometry['millimetersPerCanonicalUnit'], rtol=0, atol=1e-10)
    cameras = json.loads(camerasPath.read_text())
    camera = {key: np.asarray(cameras[POLICY['referenceViewIndex']][key])
              for key in ('intrinsics', 'rotation', 'translation')}
    cap = POLICY['maximumDisplacementMm'] * unit
    active = np.zeros(len(before), bool)
    priors, rimIds = {}, []
    for name in REGIONS:
        rim, interior = geometry[name + '_rimVertexIdsOrdered'], geometry[name + '_interiorVertexIds']
        if active[interior].any():
            raise ValueError('Cavity interior regions overlap')
        active[interior] = True
        rimIds.extend(rim)
        prior = buildCavityDepthPrior(before, rest, rim, interior,
            patchFaces=faces[geometry[name + '_faceIds']],
            inwardDirection=geometry[name + '_inwardDirection'],
            maxAddedDepth=cap, restorationFraction=POLICY['restorationFraction'])
        metrics = prior['diagnostics']
        if (max(metrics['currentRimPlaneRmsOverRadius'], metrics['restRimPlaneRmsOverRadius'])
                > POLICY['maximumRimPlaneRmsOverRadius']
                or metrics['inwardNormalDot'] < POLICY['minimumCurrentRestNormalDot']):
            raise ValueError('Unstable or inconsistent cavity boundary frames')
        priors[name] = prior
    if active.sum() != 69 or len(rimIds) != 29:
        raise ValueError('Unexpected audited cavity topology')
    protected = np.unique(np.r_[rimIds, geometry['columellaGuardVertexIds'], geometry['staticVertexIds']])
    if active[protected].any():
        raise ValueError('Active interior overlaps a protected region')
    fixed = np.flatnonzero(~active)
    joined = {key: np.concatenate([priors[name][key] for name in REGIONS])
        for key in ('vertexIndices', 'barycentric', 'targetPoints', 'targetNormals', 'weights')}
    proposal, solve = solveLocalSurfaceIncrement(before, faces, joined['vertexIndices'],
        joined['barycentric'], joined['targetPoints'], joined['targetNormals'], joined['weights'],
        activeMask=active, maxDisplacement=cap, controlCount=POLICY['controlCount'],
        controlRadius=POLICY['controlRadiusMm'] * unit, normalWeight=POLICY['normalWeight'],
        tangentWeight=POLICY['tangentWeight'], displacementWeight=POLICY['displacementWeight'],
        strainWeight=POLICY['strainWeight'])
    common = dict(camera=camera, maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
        minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'])
    existingDepth = float(np.abs((before - pureDirect) @ camera['rotation'][2]).max())
    cumulativeLimits = dict(fixedVertexIds=geometry['staticVertexIds'],
        maxDisplacement=POLICY['cumulativeDisplacementExtentFraction'] * np.ptp(pureDirect, axis=0).max(),
        depthTolerance=existingDepth + cap)
    baselineSafety = checkFrontGeometry(pureDirect, before, faces, **cumulativeLimits, **common)
    pairsBefore, pairsOriginal = intersectionPairs(before, faces), intersectionPairs(pureDirect, faces)
    pairsFull = intersectionPairs(fullBefore, archive.mesh['faces'])
    baselineNewPairs = pairsBefore - pairsOriginal
    baselinePassed = bool(baselineSafety['accepted'] and not baselineNewPairs)
    candidate, amount, attempts = before.copy(), 0., []
    trialArrays = {}
    for trialIndex, step in enumerate(POLICY['safetyAmounts']):
        trial = before + step * (proposal - before)
        trialArrays[f'trial{trialIndex}'] = trial
        local = checkFrontGeometry(before, trial, faces, fixedVertexIds=fixed,
            maxDisplacement=cap, depthTolerance=cap, **common)
        cumulative = checkFrontGeometry(pureDirect, trial, faces, **cumulativeLimits, **common)
        # A rejected deformation needs no costly full-assembly intersection check.
        # Passing candidates always undergo both skin and assembled checks.
        added = None
        if local['accepted'] and cumulative['accepted']:
            trialPairs = intersectionPairs(trial, faces)
            fullTrial = candidateFullGeometry(fullBefore, skinIds, before, trial)
            added = {'currentSkin': len(trialPairs - pairsBefore),
                'originalSkin': len(trialPairs - pairsOriginal),
                'assembled': len(intersectionPairs(fullTrial, archive.mesh['faces']) - pairsFull)}
        passed = bool(baselinePassed and local['accepted'] and cumulative['accepted']
            and added is not None and not any(added.values()))
        attempts.append({'amount': step, 'current': local, 'originalDirect': cumulative,
            'newIntersectionCounts': added, 'intersectionCheckSkippedBecauseGeometryRejected': added is None,
            'baselineAlreadyFailed': not baselinePassed, 'accepted': passed})
        if passed:
            candidate, amount = trial, step
            break
    depthMetrics = {}
    for name in REGIONS:
        current = buildCavityDepthPrior(candidate, rest, geometry[name + '_rimVertexIdsOrdered'],
            geometry[name + '_interiorVertexIds'], patchFaces=faces[geometry[name + '_faceIds']],
            inwardDirection=geometry[name + '_inwardDirection'], maxAddedDepth=cap,
            restorationFraction=POLICY['restorationFraction'])
        actual = (candidate - before)[geometry[name + '_interiorVertexIds']]
        inward = np.asarray(priors[name]['diagnostics']['currentInwardNormal'])
        depthChange = actual @ inward / unit
        depthMetrics[name] = {
            'beforeCorrectedDepthMm': summary(priors[name]['beforeDepth'] / unit),
            'candidateCorrectedDepthMm': summary(current['beforeDepth'] / unit),
            'scaledRestCorrectedDepthMm': summary(priors[name]['scaledRestDepth'] / unit),
            'meanActualDepthIncreaseMm': float(depthChange.mean()),
            'minimumActualInwardDisplacementMm': float(depthChange.min()),
            'maximumTangentialDisplacementMm': float(np.linalg.norm(actual - np.outer(actual @ inward, inward), axis=1).max() / unit),
        }
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / 'policy.json', POLICY)
    np.savez_compressed(output / 'geometry.npz', before=before, proposal=proposal,
        candidate=candidate, after=before, pureDirect=pureDirect, rest=rest, faces=faces,
        activeMask=active, fixedVertexIds=fixed, rimVertexIds=np.asarray(rimIds),
        columellaGuardVertexIds=geometry['columellaGuardVertexIds'], **trialArrays)
    np.savez_compressed(output / 'depthAnchors.npz', **joined)
    report = {
        'status': 'geometryReadyDiagnosticOnly' if amount else 'rejectedByGeometrySafety',
        'caseId': case.name, 'sourceArchive': str(baseline), 'sourceHashes': frozenHashes,
        'policy': POLICY, 'device': 'cpu', 'threads': 1, 'diagnosticOnly': True,
        'finalCandidateAccepted': False, 'defaultPipelineChanged': False,
        'baselineNumericalRepresentation': {
            'source': str(cachedGeometryPath), 'maximumZipCanonicalRoundTripError': archiveRoundTripError,
            'policy': 'Use saved float64 canonical vertices for exact static checks; export only changed rows into original ZIP canonical array',
            'normalization': 'No coordinates rounded or snapped; canonical cache avoids OBJ meter serialization round trip'},
        'baselineOriginalDirectSafety': baselineSafety,
        'baselineNewSkinIntersectionCount': len(baselineNewPairs),
        'baselinePassedCumulativeGates': baselinePassed,
        'existingIntersectionCounts': {'currentSkin': len(pairsBefore), 'originalSkin': len(pairsOriginal), 'assembled': len(pairsFull)},
        'selectedSafetyAmount': amount, 'activeVertexCount': int(active.sum()),
        'fixedVertexCount': len(fixed), 'maximumDisplacementMm': float(np.linalg.norm(candidate - before, axis=1).max() / unit),
        'proposalMaximumDisplacementMm': float(np.linalg.norm(proposal - before, axis=1).max() / unit),
        'rimAndExteriorExactlyFixed': bool(np.array_equal(before[fixed], candidate[fixed])),
        'solve': solve, 'depthPrior': {name: priors[name]['diagnostics'] for name in REGIONS},
        'depthMetrics': depthMetrics, 'safetyAttempts': attempts,
        'cameraChanged': False, 'textureChanged': False, 'topologyUvBindingChanged': False,
        'limitations': [
            'Closed-cavity template regions are not reviewed photographic nostril aperture curves.',
            'Template depth is a shape prior, not subject-specific photo measured anatomy.',
            'No AI-case camera, correspondence or fitted deformation is reused.',
            'Nasal rim position, visible aperture, skin appearance and global face likeness are not fitted here.',
            'No automatic quality acceptance; fixed-camera photographs and rendering must be reviewed.'],
    }
    verifyHashes(frozenHashes)
    if amount:
        full = candidateFullGeometry(fullBefore, skinIds, before, candidate)
        report['variant'] = exportLegacyGeometryVariant(baseline, full, output / 'diagnosticAsset', diagnostics={
            'diagnosticOnly': True, 'method': POLICY['method'], 'geometrySafetyPassed': True,
            'liveAcceptancePassed': False, 'finalCandidateAccepted': False})
    writeJson(output / 'report.json', report)
    print(json.dumps({'case': case.name, 'status': report['status'], 'amount': amount,
        'baselinePassed': baselinePassed, 'maximumDisplacementMm': report['maximumDisplacementMm'],
        'baselineReasons': baselineSafety['reasons'], 'depthMetrics': depthMetrics}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['110bac9f', '06732111', '10840519'])
    options = parser.parse_args()
    reports = []
    for caseId in options.cases:
        reports.append(runCase(options.audit_root.resolve() / caseId,
            options.output_root.resolve() / caseId / 'cavityOnly'))
    writeJson(options.output_root.resolve() / 'batchReport.json', {
        'status': 'completeLocalDiagnosticBatch', 'policy': POLICY,
        'finalCandidateAccepted': False, 'cases': [
            {key: report[key] for key in ('caseId', 'status', 'selectedSafetyAmount',
                'maximumDisplacementMm', 'baselinePassedCumulativeGates', 'depthMetrics')}
            for report in reports]})
