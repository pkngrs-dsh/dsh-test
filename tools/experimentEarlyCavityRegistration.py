"""Opt-in initial GLOBAL direct-registration cavity-prior experiment.

Preparation replays the unchanged objective and freezes each subject's inputs.
Solving adds a relative template-depth prior to the complete original global
system. No local displacement is transferred onto an existing reconstruction.
The two diagnostic archives share existing color and accessories and compare
pure registration only; downstream geometric refinement is not rerun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate
from faceAppearance.geometryRefinement import _meshEdges
from faceAppearance.legacyGeometryVariant import exportLegacyGeometryVariant, _similarity
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.unsupportedCavityPrior import buildUnsupportedCavityPrior
from experimentNoseSurfaceRegistration import arrays, digest, summary

REGIONS = ('frontImageLeft', 'frontImageRight')
PRIOR_POLICY = {'targetDepthFraction': .35, 'depthScaleMm': 2., 'supportThreshold': .15,
    'weights': [.01, .03, .1, .25],
    'normalization': 'sum weighted squared depth residual over all 69 cavity interior vertices; confidence amplitude retained',
    'target': '0.35 times aligned-rest harmonic-relative depth; never collapsed baseline plus a missing-depth increment',
    'scope': 'Complete global initial direct objective; original hard-static vertices only'}


def normalizedPolicy(frozenPolicy):
    """Map the independent frozen specification without changing any threshold."""
    solve = frozenPolicy['solvePolicy']
    expected = (PRIOR_POLICY['targetDepthFraction'], PRIOR_POLICY['depthScaleMm'],
        PRIOR_POLICY['supportThreshold'], PRIOR_POLICY['weights'])
    actual = (solve['targetDepthFractionOfAlignedRest'], solve['depthScaleMm'],
        solve['supportThreshold'], solve['priorWeightsToTry'])
    if actual != expected:
        raise ValueError('Independent policy differs from the prepared relative prior')
    outside, cavity, numeric = frozenPolicy['outside'], frozenPolicy['cavity'], frozenPolicy['numericTolerance']
    if abs(outside['minimumEdgeRatioToBaseline'] + outside['maximumEdgeRatioToBaseline'] - 2.) > 1e-15:
        raise ValueError('Runner expects the frozen symmetric outside edge ratio interval')
    return {'prior': PRIOR_POLICY, 'independentFrozenPolicy': frozenPolicy,
        'exterior': {'maximumDisplacementMm': outside['maximumNonInteriorDisplacementMm'],
            'eyeMouthMaximumDisplacementMm': outside['maximumEyeMouthSupportDisplacementMm'],
            'maximumRelativeEdgeChange': outside['maximumEdgeRatioToBaseline'] - 1.,
            'minimumAreaRatio': outside['minimumAreaRatioToBaseline'],
            'maximumAreaRatio': outside['maximumAreaRatioToBaseline'],
            'minimumNormalCosine': outside['minimumNormalCosineToBaseline']},
        'cavity': {'maximumDisplacementMm': cavity['maximumInteriorDisplacementMm'],
            'minimumRestEdgeRatio': cavity['edgeReferenceLowerRatio'],
            'maximumRestEdgeRatio': cavity['edgeReferenceUpperRatio'],
            'minimumRestAreaRatio': cavity['areaReferenceLowerRatio'],
            'maximumRestAreaRatio': cavity['areaReferenceUpperRatio'],
            'minimumRestQualityRatio': cavity['minimumQualityFractionOfRest'],
            'minimumBaselineNormalCosine': cavity['minimumNormalCosineToBaseline'],
            'positiveTargetMinimumRestDepthMm': 1.},
        'maximumDataMseRatio': frozenPolicy['dataAndBenefit']['maximumValidSampleDataMseRatioToBaseline'],
        'minimumWeightedMissingDepthRmsImprovementFraction': frozenPolicy['dataAndBenefit']['minimumWeightedUnsupportedMissingDepthRmsImprovementFraction'],
        'numericalTolerance': {'lengthAndDepthMm': numeric['lengthOrDepthMm'],
            'areaMmSquared': numeric['areaMm2'], 'normalCosine': numeric['normalDot'],
            'dimensionless': numeric['dimensionless'], 'dataMseRatio': numeric['dataMseRatio'],
            'missingDepthImprovementFraction': numeric['missingDepthImprovementFraction']}}


def verifyHashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Frozen early-registration input changed: {path}')


def sampled(vertices, faces, sampleFaceIds, barycentric):
    return (vertices[faces[sampleFaceIds]] * barycentric[..., None]).sum(1)


def triangleMeasures(vertices, faces):
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    length = np.linalg.norm(cross, axis=1)
    return length * .5, cross / np.maximum(length[:, None], 1e-30)


def ratioMetrics(reference, candidate, faces, edges, faceMask, edgeMask):
    referenceArea, referenceNormal = triangleMeasures(reference, faces)
    candidateArea, candidateNormal = triangleMeasures(candidate, faces)
    referenceEdge = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    candidateEdge = np.linalg.norm(candidate[edges[:, 0]] - candidate[edges[:, 1]], axis=1)
    areaRatio = candidateArea[faceMask] / np.maximum(referenceArea[faceMask], 1e-30)
    edgeRatio = candidateEdge[edgeMask] / np.maximum(referenceEdge[edgeMask], 1e-30)
    cosine = np.sum(referenceNormal[faceMask] * candidateNormal[faceMask], axis=1)
    return {'minimumEdgeRatio': float(edgeRatio.min()), 'maximumEdgeRatio': float(edgeRatio.max()),
        'maximumAbsoluteRelativeEdgeChange': float(np.abs(edgeRatio - 1).max()),
        'minimumAreaRatio': float(areaRatio.min()), 'maximumAreaRatio': float(areaRatio.max()),
        'minimumNormalCosine': float(cosine.min()), 'nonpositiveNormalCosineCount': int((cosine <= 0).sum()),
        'edgeCount': int(edgeMask.sum()), 'faceCount': int(faceMask.sum())}


def prepareCase(shortId, output):
    prepared = output / 'preparation'
    if prepared.exists() and any(prepared.iterdir()):
        raise FileExistsError('Preparation already exists; reuse it with --solve-prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    assetDirectory = ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1'
    auditDirectory = ROOT / 'reports/noseGeneralization20261024/geometryAudit' / shortId
    audit = json.loads((auditDirectory / 'report.json').read_text())
    regions = arrays(auditDirectory / 'geometry.npz')
    case = ROOT / 'reports/feedbackAudit20261024/cases' / audit['caseId']
    appearance = ROOT / 'runs/feedbackFix20261024' / shortId / 'realEyelidCandidate'
    sourceArchive = appearance / 'asset/legacyAsset.zip'
    appearanceReport = json.loads((appearance / 'report.json').read_text())
    if appearanceReport['status'] != 'complete' or appearanceReport['sourceTaskId'] != case.name:
        raise ValueError('Expected completed subject-specific eyelid appearance parent')
    if appearanceReport['variant']['sha256'] != digest(sourceArchive):
        raise ValueError('Appearance parent archive identity differs')
    templates, mapping = arrays(assetDirectory / 'template.npz'), arrays(assetDirectory / 'directRegistration.npz')
    manifest = json.loads((assetDirectory / 'manifest.json').read_text())
    source = arrays(case / 'reconstruction/geometry.npz')
    np.testing.assert_array_equal(source['faces'], templates['sourceFaces'])
    faces = templates['nicpFaces']
    np.testing.assert_array_equal(faces, regions['faces'])
    rest = templates['vertices'][templates['nicpToFullIds']]
    valid = mapping['sampleWeights'] > 0
    static = np.flatnonzero(templates['staticMask'])
    targets = sampleSurface(source['vertices'].astype(float), source['faces'],
        {'triangleIds': mapping['sourceTriangleIds'], 'barycentricWeights': mapping['sourceBarycentricWeights']})
    options = {key: manifest['geometryTransfer'][key] for key in
        ('dataWeight', 'smoothWeight', 'restWeight', 'staticConstraint', 'solver')}
    expectedOptions = {'dataWeight': 10000., 'smoothWeight': 3000., 'restWeight': 0.,
                       'staticConstraint': 'hard', 'solver': 'exact'}
    if options != expectedOptions:
        raise ValueError(f'Unexpected original registration objective: {options}')
    baseline = registerDirectTemplate(rest, faces, mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'],
        targets, valid, static, **options)
    pureDirect, aligned = baseline['rawVertices'], baseline['alignedRest']
    replayError = float(np.abs(pureDirect - regions['pureDirect']).max())
    alignedError = float(np.abs(aligned - regions['alignedRest']).max())
    if max(replayError, alignedError) > 1e-12:
        raise ValueError('Initial direct objective replay differs from independently audited cache')
    sourceAsset = _load(sourceArchive)
    rotation, translation, scale = _similarity(sourceAsset.manifest)
    for actual, expected in ((rotation, baseline['transform']['rotation']),
                             (translation, baseline['transform']['translation']),
                             (scale, baseline['transform']['scale'])):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    fullAppearance = sourceAsset.mesh['vertices'] * 100 @ rotation * scale + translation
    np.testing.assert_array_equal(sourceAsset.mesh['faces'], templates['faces'])
    unit = float(scale) / 10.
    interiorMask = np.zeros(len(rest), bool)
    patchMask = np.zeros(len(rest), bool)
    priors, matrices, weightedTargets, priorArrays = {}, [], [], {}
    totalInterior = sum(len(regions[name + '_interiorVertexIds']) for name in REGIONS)
    for name in REGIONS:
        rim, interior = regions[name + '_rimVertexIdsOrdered'], regions[name + '_interiorVertexIds']
        interiorMask[interior] = True
        patchMask[np.r_[rim, interior]] = True
        prior = buildUnsupportedCavityPrior(aligned, faces, rim, interior,
            faces[mapping['sampleNicpFaceIds']], mapping['sampleNicpWeights'], valid.astype(float),
            inwardDirection=regions[name + '_inwardDirection'],
            depthScale=PRIOR_POLICY['depthScaleMm'] * unit,
            supportThreshold=PRIOR_POLICY['supportThreshold'],
            targetDepthFraction=PRIOR_POLICY['targetDepthFraction'])
        fraction = np.sqrt(len(interior) / totalInterior)
        matrices.append(prior['weightedOperator'] * fraction)
        weightedTargets.append(prior['weightedTargets'] * fraction)
        sparse.save_npz(prepared / f'{name}DepthOperator.npz', prior['depthOperator'])
        priors[name] = prior
        for key in ('vertexSupport', 'priorWeights', 'interiorSupportRatio', 'restRelativeDepths',
                    'targetDepths', 'rowScales', 'interiorVertexIds', 'rimVertexIds', 'inwardDirection'):
            priorArrays[name + '_' + key] = prior[key]
        priorArrays[name + '_baselineDepths'] = prior['depthOperator'] @ pureDirect.ravel()
    priorOperator = sparse.vstack(matrices, format='csr')
    sparse.save_npz(prepared / 'weightedPriorOperator.npz', priorOperator)
    priorTarget = np.concatenate(weightedTargets)
    priorArrays['weightedTargets'] = priorTarget
    binding = loadClientLandmarks(faces)
    eyeMouthIds = np.unique(selectClientLandmarks(binding, groups=('eyes', 'mouth'))['vertexIndices'])
    edges = _meshEdges(faces)
    interiorEdgeMask = interiorMask[edges].any(1)
    patchFaceMask = interiorMask[faces].any(1)
    auditedPatchFaces = np.sort(np.r_[regions[REGIONS[0] + '_faceIds'], regions[REGIONS[1] + '_faceIds']])
    if not np.isin(np.flatnonzero(patchFaceMask), auditedPatchFaces).all():
        raise ValueError('An interior incident face is outside the audited cavity patches')
    raw = {'rawRest': rest, 'pureDirect': pureDirect, 'alignedRest': aligned, 'targetPoints': targets,
        'validMask': valid, 'sampleFaceIds': mapping['sampleNicpFaceIds'],
        'sampleBarycentric': mapping['sampleNicpWeights'], 'faces': faces, 'fullFaces': sourceAsset.mesh['faces'],
        'fullAppearance': fullAppearance, 'skinIds': templates['nicpToFullIds'], 'staticIds': static,
        'eyeMouthIds': eyeMouthIds, 'interiorMask': interiorMask, 'patchMask': patchMask,
        'edges': edges, 'interiorEdgeMask': interiorEdgeMask, 'patchFaceMask': patchFaceMask,
        'rotation': rotation, 'translation': translation, 'scale': scale, 'unit': unit,
        **priorArrays}
    np.savez_compressed(prepared / 'geometry.npz', **raw)
    baselineDataResidual = sampled(pureDirect, faces, raw['sampleFaceIds'], raw['sampleBarycentric'])[valid] - targets[valid]
    inputPaths = [sourceArchive, appearance / 'report.json', auditDirectory / 'geometry.npz',
        auditDirectory / 'report.json', assetDirectory / 'template.npz', assetDirectory / 'directRegistration.npz',
        assetDirectory / 'manifest.json', case / 'reconstruction/geometry.npz',
        case / 'reconstruction/cameras.json', case / 'reconstruction/inputs.json',
        ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json',
        ROOT / 'src/faceAppearance/unsupportedCavityPrior.py']
    sourceHashes = {str(path): digest(path) for path in inputPaths}
    for name in ('directTemplateRegistration.py', 'unsupportedCavityPrior.py'):
        shutil.copyfile(ROOT / 'src/faceAppearance' / name, prepared / name)
    report = {'status': 'originalDirectReplayAndRelativePriorPrepared', 'caseId': case.name,
        'sourceArchive': str(sourceArchive), 'sourceHashes': sourceHashes,
        'registrationOptions': options, 'priorPolicy': PRIOR_POLICY,
        'pureDirectReplayMaximumError': replayError, 'alignedRestMaximumError': alignedError,
        'baselineDiagnostics': baseline['diagnostics'], 'validSampleCount': int(valid.sum()),
        'staticVertexCount': len(static), 'globallyFreeVertexCount': len(rest) - len(static),
        'eyeMouthProtectedSupportCount': len(eyeMouthIds), 'priorInteriorCount': int(interiorMask.sum()),
        'interiorIncidentFaceCount': int(patchFaceMask.sum()),
        'auditedPatchRimOnlyFacesKeptInStrictExternalGate': np.setdiff1d(auditedPatchFaces, np.flatnonzero(patchFaceMask)),
        'baselineDataMseCanonicalSquared': float(np.mean(baselineDataResidual ** 2)),
        'baselineDataRmsMm': float(np.sqrt(np.mean(np.sum(baselineDataResidual ** 2, axis=1))) / unit),
        'baselineCavityVersusAlignedRest': ratioMetrics(aligned, pureDirect, faces, edges, patchFaceMask, interiorEdgeMask),
        'priorDiagnostics': {name: prior['diagnostics'] for name, prior in priors.items()},
        'depthMetrics': {name: {
            'restDepthMm': summary(prior['restRelativeDepths'] / unit),
            'baselineDepthMm': summary((prior['depthOperator'] @ pureDirect.ravel()) / unit),
            'targetDepthMm': summary(prior['targetDepths'] / unit),
            'supportRatio': summary(prior['interiorSupportRatio']),
            'priorWeights': summary(prior['priorWeights'])} for name, prior in priors.items()},
        'solverNotYetRunWithPrior': True, 'finalCandidateAccepted': False,
        'limitations': ['Template relative depth is a prior; the subject cavity is not photo measured.',
            'All nonstatic skin vertices remain globally free; this is not a local 69-vertex patch.']}
    report['preparedHashes'] = {path.name: digest(path) for path in prepared.iterdir() if path.is_file()}
    verifyHashes(sourceHashes)
    writeJson(prepared / 'report.json', report)
    print(json.dumps({'case': shortId, 'stage': report['status'], 'replayError': replayError,
        'cavityRestMetrics': report['baselineCavityVersusAlignedRest']}), flush=True)
    return report


def assembleInitial(fullAppearance, skinIds, staticIds, vertices):
    """Keep source OBJ static and non-skin vertices byte exact on export."""
    full = fullAppearance.copy()
    free = np.setdiff1d(np.arange(len(skinIds)), staticIds)
    full[skinIds[free]] = vertices[free]
    return full


def safetyMetrics(raw, candidate, depthOperators, policy, baselineSkinPairs, baselineFullPairs):
    before, rest, faces, edges = raw['pureDirect'], raw['alignedRest'], raw['faces'], raw['edges']
    unit = float(raw['unit'])
    inner, cavityFaces, cavityEdges = raw['interiorMask'], raw['patchFaceMask'], raw['interiorEdgeMask']
    delta = np.linalg.norm(candidate - before, axis=1) / unit
    tolerance = policy['numericalTolerance']
    lengthTolerance = tolerance['lengthAndDepthMm'] * unit
    areaTolerance = tolerance['areaMmSquared'] * unit ** 2
    dimensionTolerance = tolerance['dimensionless']
    oldArea, oldNormal = triangleMeasures(before, faces)
    restArea, restNormal = triangleMeasures(rest, faces)
    newArea, newNormal = triangleMeasures(candidate, faces)
    oldLength = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    restLength = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    newLength = np.linalg.norm(candidate[edges[:, 0]] - candidate[edges[:, 1]], axis=1)
    oldRestDot = np.sum(oldNormal * restNormal, axis=1)
    newRestDot = np.sum(newNormal * restNormal, axis=1)
    newOldDot = np.sum(newNormal * oldNormal, axis=1)
    edgeLower = np.minimum(oldLength, policy['cavity']['minimumRestEdgeRatio'] * restLength)
    edgeUpper = np.maximum(oldLength, policy['cavity']['maximumRestEdgeRatio'] * restLength)
    areaLower = np.minimum(oldArea, policy['cavity']['minimumRestAreaRatio'] * restArea)
    areaUpper = np.maximum(oldArea, policy['cavity']['maximumRestAreaRatio'] * restArea)

    def quality(vertices, area):
        triangle = vertices[faces]
        denominator = sum(np.sum((triangle[:, j] - triangle[:, i]) ** 2, axis=1)
                          for i, j in ((0, 1), (1, 2), (2, 0)))
        return 4 * np.sqrt(3) * area / np.maximum(denominator, 1e-30)

    oldQuality, restQuality, newQuality = quality(before, oldArea), quality(rest, restArea), quality(candidate, newArea)
    qualityLower = np.minimum(oldQuality, policy['cavity']['minimumRestQualityRatio'] * restQuality)
    gateMasks = {
        'cavityEdgeBelowGrandfatherBound': cavityEdges & (newLength < edgeLower - lengthTolerance),
        'cavityEdgeAboveGrandfatherBound': cavityEdges & (newLength > edgeUpper + lengthTolerance),
        'cavityAreaBelowGrandfatherBound': cavityFaces & (newArea < areaLower - areaTolerance),
        'cavityAreaAboveGrandfatherBound': cavityFaces & (newArea > areaUpper + areaTolerance),
        'cavityQualityBelowGrandfatherBound': cavityFaces & (newQuality < qualityLower - dimensionTolerance),
        'cavityNormalRotatedTooFarFromBaseline': cavityFaces & (newOldDot < policy['cavity']['minimumBaselineNormalCosine'] - tolerance['normalCosine']),
        'cavityRestNormalNewOrWorsenedReversal': cavityFaces & (newRestDot < np.minimum(oldRestDot, 0.) - tolerance['normalCosine']),
    }
    external = ratioMetrics(before, candidate, faces, edges, ~cavityFaces, ~cavityEdges)
    dataResidual = sampled(candidate, faces, raw['sampleFaceIds'], raw['sampleBarycentric'])[raw['validMask']] - raw['targetPoints'][raw['validMask']]
    oldResidual = sampled(before, faces, raw['sampleFaceIds'], raw['sampleBarycentric'])[raw['validMask']] - raw['targetPoints'][raw['validMask']]
    dataMse, oldDataMse = float(np.mean(dataResidual ** 2)), float(np.mean(oldResidual ** 2))
    dataRatio = dataMse / max(oldDataMse, 1e-30)
    checks = {
        'staticVerticesExactlyUnchanged': bool(np.array_equal(candidate[raw['staticIds']], before[raw['staticIds']])),
        'exteriorDisplacementBound': bool(delta[~inner].max() <= policy['exterior']['maximumDisplacementMm'] + tolerance['lengthAndDepthMm']),
        'eyeMouthSupportDisplacementBound': bool(delta[raw['eyeMouthIds']].max() <= policy['exterior']['eyeMouthMaximumDisplacementMm'] + tolerance['lengthAndDepthMm']),
        'cavityDisplacementBound': bool(delta[inner].max() <= policy['cavity']['maximumDisplacementMm'] + tolerance['lengthAndDepthMm']),
        'externalEdgeChangeBound': external['maximumAbsoluteRelativeEdgeChange'] <= policy['exterior']['maximumRelativeEdgeChange'] + dimensionTolerance,
        'externalMinimumAreaRatio': external['minimumAreaRatio'] >= policy['exterior']['minimumAreaRatio'] - dimensionTolerance,
        'externalMaximumAreaRatio': external['maximumAreaRatio'] <= policy['exterior']['maximumAreaRatio'] + dimensionTolerance,
        'externalNormalCosineBound': external['minimumNormalCosine'] >= policy['exterior']['minimumNormalCosine'] - tolerance['normalCosine'],
        'dataMseDeteriorationBound': dataRatio <= policy['maximumDataMseRatio'] + tolerance['dataMseRatio'],
    }
    failures = {name: np.flatnonzero(mask).tolist() for name, mask in gateMasks.items()}
    checks.update({name: not bool(mask.any()) for name, mask in gateMasks.items()})
    depthMetrics = {}
    weightedBefore, weightedAfter, totalWeights = 0., 0., 0.
    for name in REGIONS:
        operator = depthOperators[name]
        d0, dn = operator @ before.ravel(), operator @ candidate.ravel()
        dr = raw[name + '_restRelativeDepths']
        weights = raw[name + '_priorWeights']
        lower, upper = np.minimum(d0, 0.), np.maximum(d0, dr)
        targetPoints = (weights > 0) & (dr / unit >= policy['cavity']['positiveTargetMinimumRestDepthMm'])
        below = dn < lower - lengthTolerance
        above = dn > upper + lengthTolerance
        positiveBelow = targetPoints & (dn < -lengthTolerance)
        positiveAbove = targetPoints & (dn > dr + lengthTolerance)
        reasons = {'belowOriginalOrZeroDepth': np.flatnonzero(below).tolist(),
            'aboveOriginalOrTemplateDepth': np.flatnonzero(above).tolist(),
            'unsupportedPositiveTargetBelowZero': np.flatnonzero(positiveBelow).tolist(),
            'unsupportedPositiveTargetAboveTemplate': np.flatnonzero(positiveAbove).tolist()}
        checks[name + 'DepthIntervals'] = not any(reasons.values())
        depthMetrics[name] = {'beforeMm': summary(d0 / unit), 'candidateMm': summary(dn / unit),
            'alignedRestMm': summary(dr / unit), 'targetPriorMm': summary(raw[name + '_targetDepths'] / unit),
            'violationsInteriorLocalIndices': reasons, 'interiorVertexIds': raw[name + '_interiorVertexIds'],
            'unsupportedPositiveTargetCount': int(targetPoints.sum())}
        weightedBefore += float(np.sum(weights * np.maximum(dr - d0, 0.) ** 2))
        weightedAfter += float(np.sum(weights * np.maximum(dr - dn, 0.) ** 2))
        totalWeights += float(weights.sum())
    missingBefore = np.sqrt(weightedBefore / max(totalWeights, 1e-30)) / unit
    missingAfter = np.sqrt(weightedAfter / max(totalWeights, 1e-30)) / unit
    improvement = 1. - missingAfter / max(missingBefore, 1e-30)
    checks['minimumWeightedMissingDepthImprovement'] = improvement >= policy['minimumWeightedMissingDepthRmsImprovementFraction'] - dimensionTolerance
    skinPairs = intersectionPairs(candidate, faces)
    full = assembleInitial(raw['fullAppearance'], raw['skinIds'], raw['staticIds'], candidate)
    fullPairs = intersectionPairs(full, raw['fullFaces'])
    newSkinPairs, newFullPairs = skinPairs - baselineSkinPairs, fullPairs - baselineFullPairs
    checks['noNewSkinIntersectionPairs'] = not newSkinPairs
    checks['noNewAssembledIntersectionPairs'] = not newFullPairs
    absoluteRestViolations = {
        'edgeBelowQuarterRest': int((cavityEdges & (newLength < policy['cavity']['minimumRestEdgeRatio'] * restLength - lengthTolerance)).sum()),
        'edgeAboveDoubleRest': int((cavityEdges & (newLength > policy['cavity']['maximumRestEdgeRatio'] * restLength + lengthTolerance)).sum()),
        'areaBelowTenthRest': int((cavityFaces & (newArea < policy['cavity']['minimumRestAreaRatio'] * restArea - areaTolerance)).sum()),
        'areaAboveTripleRest': int((cavityFaces & (newArea > policy['cavity']['maximumRestAreaRatio'] * restArea + areaTolerance)).sum()),
        'qualityBelowFortyPercentRest': int((cavityFaces & (newQuality < policy['cavity']['minimumRestQualityRatio'] * restQuality - dimensionTolerance)).sum()),
        'nonpositiveRestNormal': int((cavityFaces & (newRestDot <= 0)).sum()),
    }
    def absoluteRecords(old, new, domain, lower, upper, elementKind):
        oldDeficit, newDeficit = np.maximum(lower - old, 0.), np.maximum(lower - new, 0.)
        oldExcess, newExcess = np.maximum(old - upper, 0.), np.maximum(new - upper, 0.)
        oldSeverity, newSeverity = oldDeficit + oldExcess, newDeficit + newExcess
        indices = np.flatnonzero(domain & ((oldSeverity > 0) | (newSeverity > 0)))
        records = []
        for index in indices:
            change = newSeverity[index] - oldSeverity[index]
            records.append({'elementIndex': int(index),
                'vertexIds': edges[index] if elementKind == 'edge' else faces[index],
                'before': float(old[index]), 'candidate': float(new[index]),
                'beforeDeficit': float(oldDeficit[index]), 'candidateDeficit': float(newDeficit[index]),
                'beforeExcess': float(oldExcess[index]), 'candidateExcess': float(newExcess[index]),
                'change': 'worsened' if change > dimensionTolerance else 'improved' if change < -dimensionTolerance else 'unchanged'})
        return {'elementKind': elementKind, 'lower': lower, 'upper': None if np.isinf(upper) else upper,
            'beforeViolationCount': int((domain & (oldSeverity > 0)).sum()),
            'candidateViolationCount': int((domain & (newSeverity > 0)).sum()),
            'beforeMaximumSeverity': float(oldSeverity[domain].max()),
            'candidateMaximumSeverity': float(newSeverity[domain].max()),
            'beforeSummedSeverity': float(oldSeverity[domain].sum()),
            'candidateSummedSeverity': float(newSeverity[domain].sum()), 'records': records}
    absoluteDetails = {
        'restEdgeRatio': absoluteRecords(oldLength / restLength, newLength / restLength, cavityEdges,
            policy['cavity']['minimumRestEdgeRatio'], policy['cavity']['maximumRestEdgeRatio'], 'edge'),
        'restAreaRatio': absoluteRecords(oldArea / restArea, newArea / restArea, cavityFaces,
            policy['cavity']['minimumRestAreaRatio'], policy['cavity']['maximumRestAreaRatio'], 'face'),
        'restTriangleQualityRatio': absoluteRecords(oldQuality / restQuality, newQuality / restQuality, cavityFaces,
            policy['cavity']['minimumRestQualityRatio'], np.inf, 'face'),
        'restNormalDot': absoluteRecords(oldRestDot, newRestDot, cavityFaces, 0., np.inf, 'face'),
    }
    return {'accepted': bool(all(checks.values())), 'checks': checks,
        'reasons': [name for name, accepted in checks.items() if not accepted],
        'failedElementIndices': failures, 'exteriorMetrics': external,
        'cavityVsRestMetrics': ratioMetrics(rest, candidate, faces, edges, cavityFaces, cavityEdges),
        'remainingAbsoluteRestThresholdViolations': absoluteRestViolations,
        'absoluteRestQualityDetails': absoluteDetails,
        'depthMetrics': depthMetrics, 'weightedMissingDepthRmsMm': {'before': missingBefore,
            'candidate': missingAfter, 'improvementFraction': improvement},
        'sampleDataMseRatio': dataRatio, 'sampleDataMseCanonicalSquared': dataMse,
        'sampleDataRmsMm': float(np.sqrt(np.mean(np.sum(dataResidual ** 2, axis=1))) / unit),
        'maximumDisplacementMm': float(delta.max()), 'maximumCavityDisplacementMm': float(delta[inner].max()),
        'maximumExteriorDisplacementMm': float(delta[~inner].max()),
        'maximumEyeMouthSupportDisplacementMm': float(delta[raw['eyeMouthIds']].max()),
        'globalPropagationCounts': {f'moreThan{threshold:g}Mm': int((delta > threshold).sum())
                                    for threshold in (1e-6, .001, .01, .1, .5, 1.)},
        'changedNonstaticVertexCountExact': int(np.any(candidate != before, axis=1).sum()),
        'newSkinIntersectionPairs': sorted(newSkinPairs), 'newAssembledIntersectionPairs': sorted(newFullPairs),
        'skinIntersectionCount': len(skinPairs), 'assembledIntersectionCount': len(fullPairs),
        'gateMeaning': 'No new or worsened geometric degeneration within the frozen envelope; not template-quality or anatomical recovery certification'}


def solvePreparedCase(shortId, output, policyPath):
    prepared = output / 'preparation'
    reportPath = prepared / 'report.json'
    original = json.loads(reportPath.read_text())
    verifyHashes(original['sourceHashes'])
    for name, expected in original['preparedHashes'].items():
        if digest(prepared / name) != expected:
            raise ValueError('Prepared original registration state changed')
    if original['priorPolicy'] != PRIOR_POLICY:
        raise ValueError('Prepared prior definition differs from current frozen experiment')
    policy = normalizedPolicy(json.loads(policyPath.read_text()))
    if (output / 'report.json').exists() or (output / 'candidates').exists():
        raise FileExistsError('Prior solve outputs already exist; preserve earlier experiment')
    writeJson(output / 'policy.json', policy)
    implementation = output / 'implementation'
    implementation.mkdir()
    implementationFiles = [Path(__file__), ROOT / 'src/faceAppearance/directTemplateRegistration.py',
        ROOT / 'src/faceAppearance/unsupportedCavityPrior.py', ROOT / 'src/faceAppearance/clientGeometrySafety.py']
    implementationHashes = {str(path): digest(path) for path in implementationFiles}
    for path in implementationFiles:
        shutil.copyfile(path, implementation / path.name)
    frozen = {**original['sourceHashes'], **implementationHashes, str(policyPath): digest(policyPath), str(reportPath): digest(reportPath)}
    raw = arrays(prepared / 'geometry.npz')
    depthOperators = {name: sparse.load_npz(prepared / f'{name}DepthOperator.npz') for name in REGIONS}
    weightedOperator = sparse.load_npz(prepared / 'weightedPriorOperator.npz')
    weightedTargets = raw['weightedTargets']
    pureDirect = raw['pureDirect']
    fullBaseline = assembleInitial(raw['fullAppearance'], raw['skinIds'], raw['staticIds'], pureDirect)
    baselineSkinPairs = intersectionPairs(pureDirect, raw['faces'])
    baselineFullPairs = intersectionPairs(fullBaseline, raw['fullFaces'])
    appearanceFullPairs = intersectionPairs(raw['fullAppearance'], raw['fullFaces'])
    stageCollisionDifference = {
        'scope': 'Initial registration baseline versus current eyelid-refined complete asset; diagnostic only, not a changed prior gate',
        'initialFullIntersectionCount': len(baselineFullPairs),
        'currentEyelidFullIntersectionCount': len(appearanceFullPairs),
        'onlyInInitialRegistrationPairs': sorted(baselineFullPairs - appearanceFullPairs),
        'onlyInCurrentEyelidAssetPairs': sorted(appearanceFullPairs - baselineFullPairs)}
    writeJson(output / 'initialVersusCurrentAssetIntersections.json', stageCollisionDifference)
    baselineGate = safetyMetrics(raw, pureDirect, depthOperators, policy, baselineSkinPairs, baselineFullPairs)
    writeJson(output / 'baselineMetrics.json', baselineGate)
    reports, selected, selectedWeight = [], None, None
    started = perf_counter()
    for weight in PRIOR_POLICY['weights']:
        candidateDirectory = output / 'candidates' / f'weight{weight:g}'
        candidateDirectory.mkdir(parents=True)
        result = registerDirectTemplate(raw['rawRest'], raw['faces'], raw['sampleFaceIds'],
            raw['sampleBarycentric'], raw['targetPoints'], raw['validMask'], raw['staticIds'],
            **original['registrationOptions'], quadraticPrior={
                'operator': weightedOperator, 'targets': weightedTargets, 'weight': weight})
        candidate = result['rawVertices']
        np.testing.assert_array_equal(result['alignedRest'], raw['alignedRest'])
        np.testing.assert_array_equal(candidate[raw['staticIds']], pureDirect[raw['staticIds']])
        np.savez_compressed(candidateDirectory / 'geometry.npz', vertices=candidate, pureDirect=pureDirect,
            alignedRest=raw['alignedRest'], faces=raw['faces'], delta=candidate - pureDirect)
        gate = safetyMetrics(raw, candidate, depthOperators, policy, baselineSkinPairs, baselineFullPairs)
        loss = float(weight * np.sum((weightedOperator @ candidate.ravel() - weightedTargets) ** 2))
        item = {'weight': weight, 'geometryGate': gate, 'registrationDiagnostics': result['diagnostics'],
            'depthPriorLoss': loss, 'geometrySha256': digest(candidateDirectory / 'geometry.npz'),
            'status': 'passedFrozenGeometryAndMinimumEffectGates' if gate['accepted'] else 'rejected',
            'finalCandidateAccepted': False}
        writeJson(candidateDirectory / 'report.json', item)
        reports.append(item)
        if gate['accepted'] and selected is None:
            selected, selectedWeight = candidate, weight
        print(json.dumps({'case': shortId, 'weight': weight, 'accepted': gate['accepted'],
            'reasons': gate['reasons'], 'maximumDisplacementMm': gate['maximumDisplacementMm'],
            'missingDepthImprovement': gate['weightedMissingDepthRmsMm']['improvementFraction'],
            'dataMseRatio': gate['sampleDataMseRatio']}), flush=True)
        verifyHashes(frozen)
    final = {'status': 'selectedDiagnosticPendingPhotographicReview' if selected is not None else 'allFrozenStrengthsRejected',
        'caseId': original['caseId'], 'sourceArchive': original['sourceArchive'], 'sourceHashes': frozen,
        'policy': policy, 'selectedWeight': selectedWeight, 'finalCandidateAccepted': False,
        'defaultPipelineChanged': False, 'diagnosticOnly': True, 'seconds': perf_counter() - started,
        'initialPureDirectBaseline': True, 'initialGlobalPriorSolve': True, 'downstreamGeometryRerun': False,
        'textureRebakedForInitialGeometry': False, 'localDeltaTransferredToDeliveredAsset': False,
        'baselineIntersectionCounts': {'skin': len(baselineSkinPairs), 'assembled': len(baselineFullPairs)},
        'initialVersusCurrentAssetIntersections': stageCollisionDifference,
        'selection': 'Lowest predetermined strength passing every frozen safety gate and minimum weighted missing-depth improvement; all four raw solves retained',
        'candidateReports': reports, 'limitations': [
            'This compares initial registration stages, not fully refined deliverable geometry.',
            'Previous eye and mouth geometric postprocessing has deliberately not been rerun.',
            'Both branches reuse the identical latest eyelid color and attachments; color is not rebaked to the changed initial geometry.',
            'Unsupported template depth is a shape prior, not photo-measured subject anatomy.',
            'Grandfather envelopes prohibit new degradation but may retain preexisting collapsed or reversed elements.',
            'Global free-variable propagation is measured and is not claimed to affect only 69 vertices.']}
    if selected is not None:
        source = Path(original['sourceArchive'])
        final['baselineArchive'] = exportLegacyGeometryVariant(source, fullBaseline, output / 'baselineAsset', diagnostics={
            'diagnosticOnly': True, 'stage': 'initialRegistrationBaseline', 'downstreamGeometryRerun': False,
            'appearanceRebaked': False, 'finalCandidateAccepted': False})
        fullSelected = assembleInitial(raw['fullAppearance'], raw['skinIds'], raw['staticIds'], selected)
        final['candidateArchive'] = exportLegacyGeometryVariant(source, fullSelected, output / 'diagnosticAsset', diagnostics={
            'diagnosticOnly': True, 'stage': 'initialRegistrationPrior', 'selectedWeight': selectedWeight,
            'downstreamGeometryRerun': False, 'appearanceRebaked': False, 'geometrySafetyPassed': True,
            'finalCandidateAccepted': False})
        final['staticSourceObjPreservationMaximumCanonicalError'] = float(np.abs(
            fullBaseline[raw['skinIds'][raw['staticIds']]] - pureDirect[raw['staticIds']]).max())
        if final['staticSourceObjPreservationMaximumCanonicalError'] > 1e-12:
            raise ValueError('Source OBJ static frame differs materially from the exact solver static frame')
        np.savez_compressed(output / 'geometry.npz', before=pureDirect, candidate=selected, after=pureDirect,
            faces=raw['faces'], alignedRest=raw['alignedRest'], delta=selected - pureDirect,
            staticVertexIds=raw['staticIds'], interiorMask=raw['interiorMask'], eyeMouthVertexIds=raw['eyeMouthIds'])
    verifyHashes(frozen)
    writeJson(output / 'report.json', final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/earlyCavity20261024')
    parser.add_argument('--cases', nargs='+', default=['110bac9f', '06732111', '10840519'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--solve-prepared', action='store_true')
    parser.add_argument('--policy', type=Path)
    options = parser.parse_args()
    if options.prepare_only == options.solve_prepared:
        parser.error('Choose exactly one of --prepare-only or --solve-prepared')
    if options.solve_prepared:
        if options.policy is None:
            parser.error('--solve-prepared requires a previously frozen --policy')
        reports = [solvePreparedCase(shortId, options.output_root.resolve() / shortId, options.policy.resolve())
                   for shortId in options.cases]
        writeJson(options.output_root.resolve() / 'batchReport.json', {
            'status': 'completeFrozenInitialGlobalRegistrationExperiment', 'policySha256': digest(options.policy),
            'finalCandidateAccepted': False, 'cases': [{key: report[key] for key in
                ('caseId', 'status', 'selectedWeight', 'seconds')} for report in reports]})
        return
    reports = [prepareCase(shortId, options.output_root.resolve() / shortId) for shortId in options.cases]
    writeJson(options.output_root.resolve() / 'preparationReport.json', {
        'status': 'threeIndependentOriginalDirectReplaysPrepared', 'priorPolicy': PRIOR_POLICY,
        'cases': [{key: report[key] for key in ('caseId', 'pureDirectReplayMaximumError',
            'baselineDataRmsMm', 'baselineCavityVersusAlignedRest')} for report in reports]})


if __name__ == '__main__':
    main()
