"""One frozen, view-specific nasal contour experiment on an audited local asset.

Bindings must be independently qualified before this tool runs. A fitted image
edge is not automatically an anatomical rim, and the exported ZIP is diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.cavityPrior import buildCavityDepthPrior
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.directTemplateRegistration import _laplacian
from faceAppearance.eyeRefinement import _projectWithJacobian
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyGeometryVariant import _similarity, exportLegacyGeometryVariant
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.localContourRefinement import refineLocalContours
from experimentNoseSurfaceRegistration import arrays, digest, verifyInputs

POLICY = {'maximumDisplacementMm': 1.5, 'controlCount': 20, 'controlRadiusMm': 5.,
    'activeGraphRings': 2, 'displacementWeight': .01, 'strainWeight': .1,
    'safetyAmounts': [1., .5, .25], 'maximumRelativeEdgeChange': .25,
    'minimumAreaRatio': .4, 'minimumNormalCosine': .3,
    'maximumMeanCavityDepthLossMm': .1, 'maximumPointCavityDepthLossMm': .25,
    'minimumHeldOutBandImprovementPixels': .1,
    'maximumHeldOutPointDegradationPixels': 1.,
    'cameraAndAppearanceFrozen': True, 'qualityAccepted': False}


def measure(vertices, groups, cameras):
    results = []
    for group in groups:
        ids, bary = np.asarray(group['vertexIndices']), np.asarray(group['barycentric'])
        points = (vertices[ids]*bary[..., None]).sum(1)
        pixels, depths, _ = _projectWithJacobian(points, cameras[group['viewIndex']])
        normal = np.asarray(group['normalPixels'], float)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        residual = np.einsum('ij,ij->i', pixels-np.asarray(group['targetPixels']), normal)
        uncertainty = np.asarray(group['uncertaintyPixels'], float)
        if uncertainty.shape != residual.shape or not np.isfinite(uncertainty).all() or (uncertainty <= 0).any():
            raise ValueError('Every contour sample requires a positive measured uncertainty band')
        excess = np.maximum(np.abs(residual)-uncertainty, 0.)
        weights = np.asarray(group['weights'], float); weights /= weights.sum()
        results.append({'name': group['name'], 'viewIndex': group['viewIndex'],
            'rmsPixels': float(np.sqrt(np.sum(weights*residual**2))),
            'bandExcessRmsPixels': float(np.sqrt(np.sum(weights*excess**2))),
            'bandExcessPixels': excess, 'uncertaintyPixels': uncertainty,
            'normalResidualPixels': residual, 'pixels': pixels,
            'allPositiveDepth': bool((depths > 0).all())})
    return results


def frozenBandTargets(vertices, groups, cameras):
    """Reach the nearest frozen uncertainty-band boundary, never its center."""
    prepared = copy.deepcopy(groups)
    for group, metrics in zip(prepared, measure(vertices, groups, cameras)):
        normals = np.asarray(group['normalPixels'], float)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        residual = metrics['normalResidualPixels']
        correction = -np.sign(residual)*metrics['bandExcessPixels']
        group['targetPixels'] = metrics['pixels']+correction[:, None]*normals
    return prepared


def run(bindingsPath, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh experiment directory')
    initialBindingsHash = digest(bindingsPath)
    bindings = json.loads(bindingsPath.read_text())
    if digest(bindingsPath) != initialBindingsHash:
        raise ValueError('Bindings changed while being read')
    if bindings.get('status') != 'qualifiedForSingleLocalExperiment':
        raise ValueError('Bindings have not passed the separate semantic/visibility audit')
    for name, expected in bindings['sourceHashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Frozen binding source changed: {name}')
    train, heldOut = bindings['train'], bindings['heldOut']
    if not train or not heldOut:
        raise ValueError('Independent training and held-out groups are required')
    observationIds = lambda groups: {(g['viewIndex'], int(i)) for g in groups for i in g['photoObservationIds']}
    if observationIds(train) & observationIds(heldOut):
        raise ValueError('A photo observation was reused for training and validation')
    identityPath = ROOT/'reports/noseRegistration20261024/inputIdentity.json'
    identity = verifyInputs(identityPath)
    case = ROOT/'reports/feedbackAudit20261024/cases/28696732-c03c-4855-bde2-fe06f8ccaee7'
    chain = ROOT/'reports/feedbackAudit20261024/referenceGeometryReplay/ai286BoundarySafeChain'
    regionPath = ROOT/'reports/noseCavity20261024/topologyAudit/regions.npz'
    parent = ROOT/'runs/noseCavity20261024/28696732/cavityOnly/diagnosticAsset/legacyAsset.zip'
    geometryPath = ROOT/'runs/noseCavity20261024/28696732/cavityOnly/geometry.npz'
    frozenPaths = [bindingsPath, regionPath, parent, geometryPath, Path(__file__),
        ROOT/'src/faceAppearance/localContourRefinement.py', ROOT/'src/faceAppearance/cavityPrior.py',
        ROOT/'src/faceAppearance/clientGeometrySafety.py', ROOT/'src/faceAppearance/frontLandmarkRefinement.py']
    frozenHashes = {str(p.resolve()): digest(p) for p in frozenPaths}
    frozenHashes.update({str((ROOT/p).resolve()): sha for p, sha in bindings['sourceHashes'].items()})
    if frozenHashes[str(bindingsPath)] != initialBindingsHash:
        raise ValueError('Bindings changed before the experiment')
    for name, expected in frozenHashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'New experiment input changed: {name}')
    regions = arrays(regionPath)
    if digest(parent) != '96a14a523496517300d9b565487c5f6bf89ce2b04cbde4ffdc0eb171ea908020':
        raise ValueError('Unexpected shallow-cavity source asset')
    archive = _load(parent)
    rotation, translation, scale = _similarity(archive.manifest)
    unit = scale/10.
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    skinIds = assets.arrays['nicpToFullIds']
    fullBefore = archive.mesh['vertices']*100 @ rotation*scale+translation
    serializedBefore = fullBefore[skinIds]
    frozen = arrays(geometryPath)
    np.testing.assert_allclose(serializedBefore, frozen['candidate'], rtol=0, atol=1e-12)
    # Use the frozen canonical state for exact fixed-vertex comparisons. The
    # inverse OBJ similarity round trip differs only at floating-point epsilon.
    before = frozen['candidate'].copy()
    faces = regions['skinFaces']
    cameras = json.loads((case/'reconstruction/cameras.json').read_text())
    cameras = [{k: np.asarray(item[k]) for k in ('intrinsics', 'rotation', 'translation')} for item in cameras]
    allowedSide = bindings['anatomicalRegion']
    if allowedSide not in ('frontImageLeft', 'frontImageRight'):
        raise ValueError('This ablation permits only one independently qualified cavity')
    opposite = 'frontImageRight' if allowedSide == 'frontImageLeft' else 'frontImageLeft'
    seeds = regions[allowedSide+'_vertexIds']
    _, adjacency = _laplacian(len(before), faces)
    distances = dijkstra(adjacency, directed=False, indices=seeds, unweighted=True, min_only=True)
    active = distances <= POLICY['activeGraphRings']
    protected = np.unique(np.r_[regions[opposite+'_vertexIds'], regions['columellaGuardVertexIds'],
                                  np.flatnonzero(assets.arrays['staticMask'])])
    active[protected] = False
    fixed = np.flatnonzero(~active)
    cap = POLICY['maximumDisplacementMm']*unit
    effectiveTrain = frozenBandTargets(before, train, cameras)
    proposal, solve = refineLocalContours(before, faces, cameras, effectiveTrain, activeMask=active,
        maxDisplacement=cap, controlCount=POLICY['controlCount'],
        controlRadius=POLICY['controlRadiusMm']*unit,
        displacementWeight=POLICY['displacementWeight'], strainWeight=POLICY['strainWeight'])

    def depth(vertices):
        prior = buildCavityDepthPrior(vertices, regions['skinVerticesRestAligned'],
            regions[allowedSide+'_rimVertexIdsOrdered'], regions[allowedSide+'_interiorVertexIds'],
            patchFaces=faces[regions[allowedSide+'_faceIds']],
            inwardDirection=-regions[allowedSide+'_currentPlaneBasis'][:, 2],
            maxAddedDepth=cap)
        return prior['beforeDepth']/unit

    pureDirect = regions['skinVerticesPureDirect']
    pairsBefore, pairsDirect = intersectionPairs(before, faces), intersectionPairs(pureDirect, faces)
    pairsFull = intersectionPairs(fullBefore, archive.mesh['faces'])
    existingDepth = float(np.abs((before-pureDirect) @ cameras[3]['rotation'][2]).max())
    initialDepth = depth(before)
    candidate, amount, attempts = before.copy(), 0., []
    for step in POLICY['safetyAmounts']:
        trial = before+step*(proposal-before)
        common = dict(camera=cameras[3], maxRelativeEdgeChange=POLICY['maximumRelativeEdgeChange'],
            minAreaRatio=POLICY['minimumAreaRatio'], minNormalCosine=POLICY['minimumNormalCosine'])
        local = checkFrontGeometry(before, trial, faces, fixedVertexIds=fixed,
            maxDisplacement=cap, depthTolerance=cap, **common)
        cumulative = checkFrontGeometry(pureDirect, trial, faces,
            fixedVertexIds=np.flatnonzero(assets.arrays['staticMask']),
            maxDisplacement=.07*np.ptp(pureDirect, axis=0).max(), depthTolerance=existingDepth+cap, **common)
        trialPairs = intersectionPairs(trial, faces)
        full = fullBefore.copy(); changed = np.any(trial != before, axis=1); full[skinIds[changed]] = trial[changed]
        added = {'currentSkin': len(trialPairs-pairsBefore), 'originalSkin': len(trialPairs-pairsDirect),
                 'assembled': len(intersectionPairs(full, archive.mesh['faces'])-pairsFull)}
        depthDifference = initialDepth-depth(trial)
        depthLoss, worstDepthLoss = float(depthDifference.mean()), float(depthDifference.max())
        passed = bool(local['accepted'] and cumulative['accepted'] and not any(added.values())
                      and depthLoss <= POLICY['maximumMeanCavityDepthLossMm']
                      and worstDepthLoss <= POLICY['maximumPointCavityDepthLossMm'])
        attempts.append({'amount': step, 'current': local, 'originalDirect': cumulative,
            'newIntersectionCounts': added, 'meanCavityDepthLossMm': depthLoss,
            'maximumPointCavityDepthLossMm': worstDepthLoss, 'geometryAccepted': passed})
        if passed:
            candidate, amount = trial, step
            break
    beforeHeld, afterHeld = measure(before, heldOut, cameras), measure(candidate, heldOut, cameras)
    checks = []
    for old, new in zip(beforeHeld, afterHeld):
        improvement = old['rmsPixels']-new['rmsPixels']
        bandImprovement = old['bandExcessRmsPixels']-new['bandExcessRmsPixels']
        worst = float((np.abs(new['normalResidualPixels'])-np.abs(old['normalResidualPixels'])).max())
        checks.append({'name': old['name'], 'before': old, 'candidate': new,
            'rmsImprovementPixels': improvement, 'maximumPointDegradationPixels': worst,
            'bandExcessRmsImprovementPixels': bandImprovement,
            'heldOutGatePassed': bool(new['allPositiveDepth']
                and bandImprovement >= min(POLICY['minimumHeldOutBandImprovementPixels'], old['bandExcessRmsPixels'])-1e-10
                and worst <= POLICY['maximumHeldOutPointDegradationPixels'])})
    output.mkdir(parents=True)
    report = {'status': 'geometryDiagnosticReady' if amount else 'rejectedByGeometrySafety',
        'qualityAccepted': False, 'finalAdoptionEligible': False,
        'heldOutPassed': bool(amount and all(c['heldOutGatePassed'] for c in checks)),
        'policy': POLICY, 'sourceHashes': frozenHashes,
        'frozenInputIdentity': identity, 'selectedSafetyAmount': amount,
        'maximumDisplacementMm': float(np.linalg.norm(candidate-before, axis=1).max()/unit),
        'sourceObjRoundTripMaximumCanonicalError': float(np.abs(serializedBefore-before).max()),
        'activeVertexCount': int(active.sum()), 'fixedVertexCount': len(fixed),
        'solve': solve, 'safetyAttempts': attempts, 'heldOutChecks': checks,
        'effectiveTrainingTargets': effectiveTrain,
        'uncertaintyTreatment': 'One initial frozen projection to nearest normal-distance band edge; no center fitting or rebinding',
        'trainBefore': measure(before, train, cameras), 'trainCandidate': measure(candidate, train, cameras),
        'fixedVerticesExact': bool(np.array_equal(before[fixed], candidate[fixed])),
        'limitations': ['Bindings retain audit uncertainty; fit improvement is not anatomical ground truth.',
            'One side only. The opposite unqualified nostril is exactly fixed.',
            'Original appearance, cameras and topology are unchanged. Independent renders and visibility review are required.']}
    np.savez_compressed(output/'geometry.npz', before=before, proposal=proposal, candidate=candidate,
        after=before, faces=faces, activeMask=active, fixedVertexIds=fixed,
        initialCavityDepthMm=initialDepth, candidateCavityDepthMm=depth(candidate))
    if amount:
        full = fullBefore.copy(); changed = np.any(candidate != before, axis=1); full[skinIds[changed]] = candidate[changed]
        report['variant'] = exportLegacyGeometryVariant(parent, full, output/'diagnosticAsset', diagnostics={
            'diagnosticOnly': True, 'method': 'frozenViewSpecificNostrilContourNormals',
            'geometrySafetyPassed': True, 'qualityAccepted': False})
    verifyInputs(identityPath)
    for name, expected in frozenHashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Experiment input changed during execution: {name}')
    writeJson(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'amount': amount,
        'maximumDisplacementMm': report['maximumDisplacementMm'],
        'heldOutImprovements': [r['rmsImprovementPixels'] for r in checks]}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    run(options.bindings.resolve(), options.output.resolve())
