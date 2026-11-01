"""Read-only gates for the four-arm feature geometry experiment.

This file does not solve, repair, select or export a candidate. The old physical
limits are retained. Triangle intersections use the repository's shared numeric
primitive; recomputation is independent of producer reports, not an independent
collision algorithm. Layout-specific loading is intentionally kept separate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.artifacts import writeJson

BRANCHES = ('baseline', 'sourcePrior', 'visibleEyes', 'combined')
ORIGINAL_AUDIT = ROOT / 'reports/shapePipeline20261024/independentAudit/validatePipeline.py'
_originalAudit = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loadOriginalAudit():
    """Reuse the already independent physical formulas, without running its CLI."""
    global _originalAudit
    if _originalAudit is None:
        spec = importlib.util.spec_from_file_location('featureOriginalPipelineAudit', ORIGINAL_AUDIT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _originalAudit = module
    return _originalAudit


def _vertices(value, name):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite N by 3 array')
    return result


def _ids(value, count, name):
    result = np.asarray(value)
    if not result.size:
        return np.empty(0, np.int64)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or result.min() < 0
            or result.max() >= count or len(np.unique(result)) != len(result)):
        raise ValueError(f'{name} must be distinct in-range integer IDs')
    return result.astype(np.int64, copy=False)


def _pairs(value):
    output = set()
    for pair in value:
        if len(pair) != 2 or any(type(item) not in (int, np.int32, np.int64) for item in pair):
            raise ValueError('Intersection pairs must contain two integer face IDs')
        first, second = sorted(map(int, pair))
        if first < 0 or first == second:
            raise ValueError('Intersection pairs must contain two distinct nonnegative IDs')
        output.add((first, second))
    return output


def intersectionSetAudit(referencePairs, candidatePairs, materialIds=None):
    """Compare stable face-ID sets; a lower total count does not establish safety."""
    before, after = _pairs(referencePairs), _pairs(candidatePairs)
    added, removed = sorted(after - before), sorted(before - after)
    byMaterial = {}
    if materialIds is not None:
        materialIds = np.asarray(materialIds)
        for pair in added:
            key = '/'.join(map(str, sorted(materialIds[list(pair)].tolist())))
            byMaterial[key] = byMaterial.get(key, 0) + 1
    return dict(accepted=not added, baselinePairs=sorted(before), candidatePairs=sorted(after),
        baselinePairCount=len(before), candidatePairCount=len(after),
        newPairs=added, removedPairs=removed, inheritedPairs=sorted(before & after),
        newPairsByMaterialPair=byMaterial, excludedMaterialIds=[],
        scope='All full faces including eye balls and alpha cards; shared-index neighbors alone are excluded by the shared primitive.')


def strictIntersectionAudit(reference, candidate, faces, materialIds=None):
    before, after = _vertices(reference, 'reference'), _vertices(candidate, 'candidate')
    faces = np.asarray(faces)
    if (before.shape != after.shape or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or not len(faces)
            or faces.min() < 0 or faces.max() >= len(before)):
        raise ValueError('Strict collision comparison requires identical full topology')
    if materialIds is not None and np.asarray(materialIds).shape != (len(faces),):
        raise ValueError('Expected one material ID for every full face')
    result = intersectionSetAudit(intersectionPairs(before, faces), intersectionPairs(after, faces), materialIds)
    result.update(baselineEpsilon=max(float(np.linalg.norm(np.ptp(before, axis=0))) * 1e-9, 1e-12),
                  candidateEpsilon=max(float(np.linalg.norm(np.ptp(after, axis=0))) * 1e-9, 1e-12),
                  primitive='faceAppearance.clientGeometrySafety.intersectionPairs')
    return result


def sceneIdentityAudit(reference, candidate, *, fixedVertexIds=(), protectedVertexIds=()):
    """Check in-memory identities exactly; OBJ roundtrip has its own separate gate."""
    before, after = np.asarray(reference['vertices']), np.asarray(candidate['vertices'])
    sameShape = before.shape == after.shape and before.ndim == 2 and before.shape[1] == 3
    fixed = _ids(fixedVertexIds, len(before), 'fixedVertexIds')
    protected = _ids(protectedVertexIds, len(before), 'protectedVertexIds')
    checks = dict(vertexCountAndCoordinatesShape=sameShape,
        finite=bool(np.isfinite(after).all()),
        fixedExactlyUnchanged=bool(sameShape and np.array_equal(before[fixed], after[fixed])),
        protectedExactlyUnchanged=bool(sameShape and np.array_equal(before[protected], after[protected])))
    for key in ('faces', 'uvs', 'uvFaces', 'materialIds', 'nicpToFullIds'):
        checks[key + 'ExactlyPreserved'] = key in reference and key in candidate and np.array_equal(reference[key], candidate[key])
    changed = np.flatnonzero(np.any(before != after, axis=1)) if sameShape else None
    return dict(accepted=bool(all(checks.values())), checks=checks, changedVertexIds=changed)


def originalDataFitAudit(sampleOperator, targetPoints, baseline, candidate):
    """Use the original C/Q observations, not source-prior differential targets."""
    operator = sparse.csr_matrix(sampleOperator)
    target = _vertices(targetPoints, 'targetPoints')
    before, after = _vertices(baseline, 'baseline'), _vertices(candidate, 'candidate')
    if before.shape != after.shape or operator.shape != (len(target), len(before)) or not len(target):
        raise ValueError('Original sample dimensions must agree')
    if not np.isfinite(operator.data).all():
        raise ValueError('Original sample operator must be finite')
    baseResidual, residual = operator @ before - target, operator @ after - target
    baseMse, nextMse = float(np.mean(baseResidual ** 2)), float(np.mean(residual ** 2))
    # Existing real cases have a positive denominator. Zero is reported without
    # inventing a tolerance or presenting an undefined improvement as success.
    ratio = nextMse / baseMse if baseMse > 0 else (1. if nextMse == 0 else None)
    accepted = nextMse == 0 if baseMse == 0 else ratio <= 1.02 + 1e-10
    return dict(accepted=bool(accepted), baselineMse=baseMse, candidateMse=nextMse,
        mseRatio=ratio, ratioDefined=baseMse > 0, maximumMseRatio=1.02,
        ratioNumericalTolerance=1e-10, baselineResidual=baseResidual,
        candidateResidual=residual, sampleCount=len(target),
        interpretation='Original observation fit is an independent objective diagnostic, not independently measured anatomical truth.')


def sourceCorrespondenceAudit(sourceVertices, sourceFaces, sampleFaceIds, sampleBarycentric,
                              targetPoints, validMask, *, identityTolerance):
    """Verify fixed input mesh barycentrics produce Q in the same canonical frame.

    The caller must supply the tolerance frozen in the input contract. This
    function never rebinds samples, optimizes alignment or consults a candidate.
    """
    vertices = _vertices(sourceVertices, 'sourceVertices')
    faces, faceIds = np.asarray(sourceFaces), np.asarray(sampleFaceIds)
    bary, targets, valid = np.asarray(sampleBarycentric), _vertices(targetPoints, 'targetPoints'), np.asarray(validMask)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)
            or faceIds.shape != (len(targets),) or faceIds.dtype.kind not in 'iu'
            or bary.shape != (len(targets), 3) or valid.shape != (len(targets),)
            or valid.dtype != np.bool_ or not valid.any() or not np.isfinite(bary).all()
            or not np.isfinite(identityTolerance) or identityTolerance < 0):
        raise ValueError('Invalid original source correspondence contract')
    chosen = faceIds[valid]
    if chosen.min() < 0 or chosen.max() >= len(faces):
        raise ValueError('Valid source samples must address original source faces')
    reconstructed = np.einsum('mi,mij->mj', bary[valid], vertices[faces[chosen]])
    error = np.max(np.abs(reconstructed - targets[valid]), axis=1)
    # Source bindings retain their original stored floating-point partition.
    # The source module permits 1e-6; only client C has the stricter 1e-12 rule.
    checks = dict(barycentricPartition=bool(np.all(np.abs(bary[valid].sum(1) - 1) <= 1e-6)),
        barycentricInTriangle=bool(np.all((bary[valid] >= -1e-12) & (bary[valid] <= 1 + 1e-12))),
        sourceTargetsAgree=bool(error.max() <= identityTolerance))
    return dict(accepted=bool(all(checks.values())), checks=checks,
        sampleCount=int(valid.sum()), maximumCanonicalTargetError=float(error.max()),
        identityTolerance=float(identityTolerance), targetErrors=error,
        sourceFrame='original source mesh expressed in the fixed prepared canonical frame',
        candidateDependentRebinding=False)


def differentialFitAudit(differenceOperator, sampleOperator, targetPoints, baseline, candidate,
                         *, reliabilityWeights, reliabilityProvenance):
    """Report fixed D(CV-Q) without treating a regularizer gain as ground truth."""
    difference, sample = sparse.csr_matrix(differenceOperator), sparse.csr_matrix(sampleOperator)
    target, before, after = np.asarray(targetPoints), np.asarray(baseline), np.asarray(candidate)
    weights = np.asarray(reliabilityWeights)
    if (difference.shape[1] != sample.shape[0] or target.shape != (sample.shape[0], 3)
            or before.shape != after.shape or before.shape != (sample.shape[1], 3)
            or weights.shape != (difference.shape[0],) or (weights < 0).any()
            or not np.isfinite(weights).all() or not reliabilityProvenance):
        raise ValueError('Expected fixed differential rows and explicit reliability provenance')
    if any(not np.isfinite(value).all() for value in (difference.data, sample.data, target, before, after)):
        raise ValueError('Differential inputs must be finite')
    old = difference @ (sample @ before - target)
    new = difference @ (sample @ after - target)
    mass = float(weights.sum())
    oldSquared, newSquared = float(np.sum(weights[:, None] * old ** 2)), float(np.sum(weights[:, None] * new ** 2))
    effectDefined = mass > 0 and oldSquared > 0
    return dict(effectDefined=effectDefined,
        baselineWeightedRms=np.sqrt(oldSquared / (3 * mass)) if mass > 0 else None,
        candidateWeightedRms=np.sqrt(newSquared / (3 * mass)) if mass > 0 else None,
        improvementFraction=1 - np.sqrt(newSquared / oldSquared) if effectDefined else None,
        baselineDifferentialResidual=old, candidateDifferentialResidual=new,
        reliabilityProvenance=reliabilityProvenance,
        diagnosticOnly=True, anatomicalTruthEstablished=False)


def initialGeometryAudit(system, vertices):
    """Original initial-registration gates including full accessories and C fit."""
    audit = loadOriginalAudit()
    raw, edges = system['a'], system['edges']
    before, after = raw['pureDirect'], _vertices(vertices, 'vertices')
    faces, interior, unit = raw['faces'], raw['interiorMask'], float(raw['unit'])
    if before.shape != after.shape:
        raise ValueError('Expected original initial skin vertex count')
    oldArea, oldNormal, _ = audit.priorAudit.faceStats(before, faces)
    area, normal, _ = audit.priorAudit.faceStats(after, faces)
    oldLength = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    length = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    exteriorEdges, exteriorFaces = ~interior[edges].any(1), ~interior[faces].any(1)
    displacement = np.linalg.norm(after - before, axis=1) / unit
    normalDot = np.einsum('ij,ij->i', normal, oldNormal)
    checks = dict(staticExactlyUnchanged=np.array_equal(after[raw['staticIds']], before[raw['staticIds']]),
        exteriorDisplacement=displacement[~interior].max() <= .5 + 1e-7,
        eyeMouthDisplacement=displacement[raw['eyeMouthIds']].max() <= .1 + 1e-7,
        exteriorEdgeRatio=np.abs(length[exteriorEdges] / oldLength[exteriorEdges] - 1).max() <= .1 + 1e-8,
        exteriorAreaMinimum=(area[exteriorFaces] / oldArea[exteriorFaces]).min() >= .8 - 1e-8,
        exteriorAreaMaximum=(area[exteriorFaces] / oldArea[exteriorFaces]).max() <= 1.25 + 1e-8,
        exteriorNormal=normalDot[exteriorFaces].min() >= .95 - 1e-6)
    cavity = audit.cavityMetrics(system, after)
    fit = originalDataFitAudit(system['C'], raw['targetPoints'][raw['validMask']], before, after)
    skin = strictIntersectionAudit(before, after, faces)
    full = strictIntersectionAudit(audit.priorAudit.assemble(raw, before), audit.priorAudit.assemble(raw, after), raw['fullFaces'])
    accepted = all(checks.values()) and cavity['accepted'] and fit['accepted'] and skin['accepted'] and full['accepted']
    return dict(accepted=bool(accepted), checks=checks, cavity=cavity, originalDataFit=fit,
        skinIntersections=skin, fullIntersections=full,
        scope='Original initial-stage envelope; do not apply whole-exterior or data-fit limits to intentional downstream refinement.')


def downstreamGeometryAudit(system, initial, final, camera):
    """Preserve original cumulative downstream and original-rest cavity anchors."""
    audit = loadOriginalAudit()
    raw = system['a']
    downstream = audit.downstreamMetrics(initial, final, raw['faces'], system['edges'], raw['staticIds'], camera)
    cavity = audit.cavityMetrics(system, final)
    return dict(accepted=bool(downstream['accepted'] and cavity['accepted']),
        cumulativeDownstream=downstream, originalCavity=cavity,
        cavityAnchor='Original prepared pureDirect and alignedRest, never reset to a candidate.')


def apertureMetricAudit(before, after, *, originalPixelScale=1.):
    """Existing complete-aperture benefit/no-regression limits, without proxy curves."""
    if not np.isfinite(originalPixelScale) or originalPixelScale <= 0:
        raise ValueError('originalPixelScale must be positive')
    checks = {}
    try:
        oldMean = float(before['meanSymmetricBoundaryDistanceOriginalPixels'])
        newMean = float(after['meanSymmetricBoundaryDistanceOriginalPixels'])
        checks['globalBoundaryBenefit'] = np.isfinite([oldMean, newMean]).all() and newMean < oldMean - .03 / originalPixelScale
        checks['sameTwoEyes'] = set(before['eyes']) == set(after['eyes']) == {'imageLeft', 'imageRight'}
        for name in ('imageLeft', 'imageRight'):
            old, new = before['eyes'][name], after['eyes'][name]
            for key, limit, direction in (
                ('symmetricBoundaryMeanOriginalPixels', .15 / originalPixelScale, 1),
                ('symmetricBoundaryP95OriginalPixels', .30 / originalPixelScale, 1),
                ('intersectionOverUnion', .005, -1),
                ('visibleComponentCount', 0., 1),
                ('secondaryComponentAreaFraction', .005, 1)):
                left, right = float(old[key]), float(new[key])
                checks[name + '_' + key] = bool(np.isfinite([left, right]).all() and direction * (right - left) <= limit)
    except (KeyError, TypeError, ValueError):
        checks['completeActualApertureMetricsAvailable'] = False
    return dict(accepted=bool(all(checks.values())), checks=checks,
        source='Complete visible aperture masks; texture proxy acceptance is not used.',
        originalPixelScale=float(originalPixelScale))


def readJson(path):
    return json.loads(Path(path).read_text())


def readArrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def readEvidence(stem):
    """Load the runner's JSON-indexed, non-pickle recursive array evidence."""
    stem = Path(stem)
    metadata, values = readJson(stem.with_suffix('.json')), readArrays(stem.with_suffix('.npz'))
    consumed = set()
    def restore(item):
        if isinstance(item, dict) and 'arrayKey' in item:
            key = item['arrayKey']
            array = values[key]
            if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype']:
                raise ValueError(f'Evidence array metadata mismatch: {key}')
            consumed.add(key)
            return array
        if isinstance(item, dict):
            return {key: restore(value) for key, value in item.items()}
        if isinstance(item, list):
            return [restore(value) for value in item]
        return item
    result = restore(metadata)
    if consumed != set(values):
        raise ValueError('Evidence archive has unreferenced array payloads')
    return result


def fileIdentityAudit(hashes):
    failed = [str(path) for path, expected in hashes.items()
              if not Path(path).is_file() or sha256(path) != expected]
    return dict(accepted=not failed, failures=failed, count=len(hashes))


def _sparseMaximumDifference(first, second):
    if first.shape != second.shape:
        return None
    difference = (first - second).tocsr()
    return float(np.max(np.abs(difference.data), initial=0))


def auditSourcePrior(system, context, assets, runRoot, config):
    """Reconstruct fixed source D/C/Q and normal equations without any fitting."""
    from scipy.sparse.csgraph import dijkstra
    raw, case = system['a'], context['entry']['caseId']
    directory = runRoot / case / 'initial/sourcePrior'
    report, evidence = readJson(directory / 'report.json'), readEvidence(directory / 'sourceEvidence')
    source = readArrays(Path(context['entry']['reconstruction']) / 'geometry.npz')
    mapping = assets.directArrays
    q = raw['targetPoints']; valid = raw['validMask']; n = len(raw['pureDirect']); m = len(q)
    clientCorners = raw['faces'][raw['sampleFaceIds']]
    sourceCorners = source['faces'][mapping['sourceTriangleIds']]
    clientWeights, sourceWeights = raw['sampleBarycentric'], mapping['sourceBarycentricWeights']
    provenance = sourceCorrespondenceAudit(source['vertices'].astype(float), source['faces'],
        mapping['sourceTriangleIds'], sourceWeights, q, valid, identityTolerance=1e-14)
    sample = sparse.csr_matrix((clientWeights.ravel(), (np.repeat(np.arange(m), 3), clientCorners.ravel())), shape=(m, n))
    groups = {}
    for row in np.flatnonzero(valid):
        key = tuple(tuple(sorted((int(i), float(w)) for i, w in zip(ids, weights) if w != 0))
                    for ids, weights in ((clientCorners[row], clientWeights[row]), (sourceCorners[row], sourceWeights[row])))
        groups.setdefault(key, []).append(int(row))
    representatives = np.asarray([rows[0] for rows in groups.values()], np.int64)
    inputToUnique = np.full(m, -1, np.int64)
    for uniqueIndex, rows in enumerate(groups.values()):
        inputToUnique[rows] = uniqueIndex
    sourceVertices = source['vertices'].astype(float)
    faces = source['faces']
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    edgeLengths = np.linalg.norm(sourceVertices[edges[:, 0]] - sourceVertices[edges[:, 1]], axis=1)
    attached = sourceCorners[representatives]
    attachmentLengths = np.linalg.norm(q[representatives, None] - sourceVertices[attached], axis=2).ravel()
    sampleNodes = np.repeat(np.arange(len(representatives)) + len(sourceVertices), 3)
    rows = np.r_[edges[:, 0], edges[:, 1], sampleNodes, attached.ravel()]
    columns = np.r_[edges[:, 1], edges[:, 0], attached.ravel(), sampleNodes]
    graph = sparse.csr_matrix((np.r_[edgeLengths, edgeLengths, attachmentLengths, attachmentLengths], (rows, columns)),
        shape=(len(sourceVertices) + len(representatives),) * 2)
    policy = config['sourceShape']; unit = float(raw['unit'])
    radius, length = policy['neighborhoodRadiusMm'] * unit, policy['lengthScaleMm'] * unit
    centers, neighborsSaved, distancesSaved, rowIds, columnIds, coefficients = [], [], [], [], [], []
    tolerance = radius * 1e-10
    for index, center in enumerate(representatives):
        distance = dijkstra(graph, directed=True, indices=int(len(sourceVertices) + index), limit=radius + tolerance)[len(sourceVertices):]
        selected = np.flatnonzero(np.isfinite(distance) & (distance <= radius + tolerance) & (np.arange(len(representatives)) != index))
        if len(selected) > policy['maxNeighbors']:
            cutoff = np.partition(distance[selected], policy['maxNeighbors'] - 1)[policy['maxNeighbors'] - 1]
            inclusive = selected[distance[selected] <= cutoff + tolerance]
            selected = inclusive if len(inclusive) <= policy['maxNeighbors'] else selected[distance[selected] < cutoff - tolerance]
        if len(selected) < policy['minNeighbors']:
            continue
        weights = np.exp(-.5 * (distance[selected] / radius) ** 2); weights /= weights.sum()
        rowIds.extend([len(centers)] * (len(selected) + 1))
        columnIds.extend([int(center), *representatives[selected].tolist()]); coefficients.extend([1., *(-weights).tolist()])
        centers.append(int(center)); neighborsSaved.append(representatives[selected]); distancesSaved.append(distance[selected])
    difference = sparse.csr_matrix((coefficients, (rowIds, columnIds)), shape=(len(centers), m))
    rowScales = np.ones(len(centers)) / np.sqrt(3 * len(representatives)) / length
    weighted = sparse.kron(sparse.diags(rowScales) @ difference @ sample, sparse.eye(3), format='csr')
    target = ((difference @ q) * rowScales[:, None]).ravel()
    active = len(centers) > 0 and weighted.nnz > 0
    effectiveWeight = 10000 * length ** 2 if active else 0.
    if not active:
        weighted, target = sparse.csr_matrix((1, 3 * n)), np.zeros(1)
    savedSample = sparse.load_npz(directory / 'sampleMatrix.npz')
    savedDifference = sparse.load_npz(directory / 'differentialOperator.npz')
    savedOperator = sparse.load_npz(directory / 'sourceOperator.npz')
    differences = dict(sample=_sparseMaximumDifference(sample, savedSample),
        differential=_sparseMaximumDifference(difference, savedDifference), operator=_sparseMaximumDifference(weighted, savedOperator))
    h = sparse.kron(system['scalarH'], sparse.eye(3), format='csr') + .01 * (system['P'].T @ system['P']) + effectiveWeight * (weighted.T @ weighted)
    rhs = system['rhs'].ravel() + .01 * (system['P'].T @ system['target']) + effectiveWeight * (weighted.T @ target)
    differences['normalMatrix'] = _sparseMaximumDifference(h, sparse.load_npz(directory / 'normalMatrixH.npz'))
    differences['normalRhs'] = float(np.abs(rhs - readArrays(directory / 'normalRhs.npz')['b']).max(initial=0))
    differences['targets'] = float(np.abs(target - readArrays(directory / 'sourceTargets.npz')['targets']).max(initial=0))
    checks = dict(sourceProvenance=provenance['accepted'],
        originalSourceTopology=np.array_equal(source['faces'], assets.arrays['sourceFaces']),
        originalUniqueRepresentatives=np.array_equal(representatives, evidence['uniqueRepresentativeSampleIds']),
        originalUniqueMapping=np.array_equal(inputToUnique, evidence['inputToUniqueSample']),
        originalCenters=np.array_equal(centers, evidence['centerSampleIds']),
        originalNeighbors=len(neighborsSaved) == len(evidence['neighborSampleIds']) and all(np.array_equal(a, b) for a, b in zip(neighborsSaved, evidence['neighborSampleIds'])),
        originalGeodesicDistances=len(distancesSaved) == len(evidence['neighborDistances']) and all(np.allclose(a, b, atol=1e-14, rtol=1e-12) for a, b in zip(distancesSaved, evidence['neighborDistances'])),
        uniformReliability=np.array_equal(evidence['rowReliability'], np.ones(len(centers))),
        rowScales=np.allclose(evidence['rowScales'], rowScales, atol=1e-14, rtol=1e-12),
        fixedEffectiveWeight=report['weight'] == effectiveWeight,
        originalActiveStatus=report['sourcePrior']['active'] == active,
        noCandidateRebinding=report['sourcePrior']['candidateRebinding'] is False,
        noConfidenceClaim=report['sourcePrior']['reliabilityProvided'] is False)
    checks.update({key + 'Reconstructed': value is not None and value < 1e-9 for key, value in differences.items()})
    vertices = readArrays(directory / 'geometry.npz')['vertices']
    gradient = (h @ vertices.ravel() - rhs).reshape(-1, 3)
    uncontrolled = np.setdiff1d(system['free'], system['controls'])
    relativeResidual = float(np.linalg.norm(gradient[uncontrolled]) / max(np.linalg.norm(rhs.reshape(-1, 3)[uncontrolled]), 1e-30))
    if report['solver']['solverAccepted']:
        checks['acceptedSolverFreeNormalResidual'] = relativeResidual < 1e-10
    fit = differentialFitAudit(difference, sample, q, raw['pureDirect'], vertices,
        reliabilityWeights=np.ones(len(centers)), reliabilityProvenance=report['sourcePrior']['reliabilityProvenance'])
    hashes = fileIdentityAudit({str(directory / key): value for key, value in report.get('outputHashes', {}).items()})
    checks['producerOutputIdentitiesPresentAndCorrect'] = bool(report.get('outputHashes')) and hashes['accepted']
    return dict(validationPassed=bool(all(checks.values())), checks=checks, matrixMaximumErrors=differences,
        originalCorrespondence=provenance, differentialFit=fit, relativeFreeNormalResidual=relativeResidual,
        effectAvailable=bool(active), initialReport=str(directory / 'report.json'), producerOutputHashes=hashes)


def _contourMasksAgree(first, second):
    if set(first.get('eyes', {})) != set(second.get('eyes', {})):
        return False
    for name, eye in first.get('eyes', {}).items():
        other = second['eyes'][name]
        if eye['status'] != other['status']:
            return False
        a, b = eye.get('roiMasks', {}), other.get('roiMasks', {})
        if set(a) != set(b) or any(not np.array_equal(a[key], b[key], equal_nan=True) for key in a):
            return False
    return True


def auditEyeEvidence(scene, before, final, camera, photoEyes, evidence, assets, imageShape, pixelScale, static):
    from faceAppearance.visibleEyeContours import extractVisibleEyeContours, evaluateApertureMask
    from faceAppearance.clientVisibleEyeRefinement import buildClientVisibleEyeSupport
    oldAudit = loadOriginalAudit()
    cornerIds = np.concatenate([c['faceIndices'] for c in assets.components
                               if c['name'] in ('male00SG.component001', 'male00SG.component002')])
    scene = {**scene, 'vertices': before}
    def measure(vertices):
        return extractVisibleEyeContours(**{**scene, 'vertices': vertices}, camera=camera,
            photoEyes=photoEyes, imageShape=imageShape, apertureAccessoryMode='innerCornerHeadAccessories',
            apertureAccessoryFaceIds=cornerIds, minimumEyeRasterWidth=256, maximumRoiSize=768)
    baseline, measured = measure(before), measure(final)
    support = buildClientVisibleEyeSupport(scene, camera, photoEyes, baseline, cornerFaceIds=cornerIds,
        fixedVertexIds=static, policy={'geodesicBandEyeWidthFraction': .30, 'rounds': 2, 'iterationsPerRound': 20})
    checks = dict(resultExactlySaved=np.array_equal(final, evidence['vertices']),
        supportMaskReproduced=np.array_equal(support['movableMask'], evidence['support']['movableMask']),
        supportFixedIdsReproduced=np.array_equal(support['fixedVertexIds'], evidence['support']['fixedVertexIds']),
        outsidePreEyeSupportExactlyFixed=np.array_equal(before[support['fixedMask']], final[support['fixedMask']]),
        beforeActualApertureReproduced=_contourMasksAgree(baseline, evidence['beforeContours']),
        afterActualApertureReproduced=_contourMasksAgree(measured, evidence['afterContours']))
    protectedAccessories = np.setdiff1d(np.arange(len(before)), np.r_[scene['nicpToFullIds'], scene['faces'][cornerIds].ravel()])
    checks['allOtherAccessoriesExactlyFixed'] = np.array_equal(before[protectedAccessories], final[protectedAccessories])
    try:
        beforeScore = evaluateApertureMask(baseline, photoEyes, originalPixelScale=pixelScale)
        finalScore = evaluateApertureMask(measured, photoEyes, originalPixelScale=pixelScale)
        effect = apertureMetricAudit(beforeScore, finalScore, originalPixelScale=pixelScale)
        effect['actualCompleteContoursAccepted'] = finalScore['allCandidateExtractionsAccepted']
        effect['accepted'] = bool(effect['accepted'] and effect['actualCompleteContoursAccepted'])
    except ValueError as error:
        beforeScore = finalScore = None
        effect = dict(accepted=False, status='unavailableActualAperture', reason=str(error))
    pairGate = strictIntersectionAudit(before, final, scene['faces'], scene['materialIds'])
    edges = np.unique(np.sort(np.r_[scene['faces'][:, [0, 1]], scene['faces'][:, [1, 2]], scene['faces'][:, [2, 0]]], axis=1), axis=0)
    limit = float(np.ptp(before[scene['nicpToFullIds']], axis=0).max()) * .03
    attempts = []
    for proposal in evidence.get('candidateProposals', []):
        candidate = proposal['vertices']
        gate = oldAudit.downstreamMetrics(before, candidate, scene['faces'], edges, support['fixedVertexIds'], camera)
        gate['checks']['maximumDisplacement'] = gate['metrics']['maximumDisplacement'] <= limit * (1 + 1e-8)
        gate['accepted'] = bool(all(gate['checks'].values()))
        actualPairs = gate['collisions']['newPairs']
        expectedPairs = proposal['strict'].get('fullAssembly', {}).get('newPairs', [])
        agrees = gate['accepted'] == proposal['strict']['accepted'] and actualPairs == [tuple(pair) for pair in expectedPairs]
        checks[f"round{proposal['round']}Scale{proposal['scale']}StrictGateReproduced"] = agrees
        attempts.append(dict(round=proposal['round'], scale=proposal['scale'], selected=proposal['selectedByFastRound'],
            independentStrict=gate, producerStrict=proposal['strict'], validationPassed=agrees))
    fastStages = evidence['diagnostics'].get('fast', {}).get('stages', [])
    checks['everyAttemptRetained'] = len(attempts) == sum(len(stage.get('attempts', [])) for stage in fastStages)
    producerAccepted = evidence['diagnostics']['accepted']
    unchanged = np.array_equal(final, before)
    if producerAccepted:
        checks['acceptedResultHasActualBenefitAndStrictSafety'] = effect['accepted'] and pairGate['accepted'] and not unchanged
    else:
        checks['rejectedResultReturnsExactInput'] = unchanged
    return dict(validationPassed=bool(all(checks.values())), checks=checks, effectAccepted=bool(effect['accepted'] and producerAccepted),
        actualApertureEffect=effect, beforeActualApertureMetrics=beforeScore, finalActualApertureMetrics=finalScore,
        fullIntersections=pairGate, proposals=attempts,
        sharedPrimitives=['extractVisibleEyeContours raster visibility', 'buildClientVisibleEyeSupport domain reconstruction', 'triangle intersection primitive'],
        rawRejectedRetained='rawCandidateVertices' in evidence and 'solverVertices' in evidence)


def auditCase(case, runRoot, outputRoot):
    """Validate saved artifacts only. Missing files are failures, never fitted here."""
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.legacyPreviewAsset import _load
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs
    oldAudit = loadOriginalAudit()
    config = readJson(runRoot / 'experimentPlan.json')
    context = readJson(runRoot / case / 'inputContext.json')
    system = oldAudit.priorAudit.independentSystem(case); raw = system['a']
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    entry = context['entry']; oldRoot = Path(context['oldRoot'])
    reference = context['referenceSelection']['referenceViewIndex']
    cameras = readJson(Path(entry['reconstruction']) / 'cameras.json')
    camera = cameras[reference]
    inputs = readJson(Path(entry['reconstruction']) / 'inputs.json')
    boundedInitial = readArrays(ROOT / 'runs/shapePreservation20261024' / case / 'boundedWeight001/geometry.npz')['vertices']
    initial = {'bounded': initialGeometryAudit(system, boundedInitial)}
    sourceFailurePath = runRoot / case / 'initial/sourceFailure.json'
    if sourceFailurePath.is_file():
        sourceFailure = readJson(sourceFailurePath)
        preservedFailure = sourceFailure.get('accepted') is False and sourceFailure.get('noFallback') is True
        sourceInitial = None
        sourceProof = dict(validationPassed=preservedFailure, effectAvailable=False,
            status='unavailableSourceInitial', retainedFailure=sourceFailure, failureSha256=sha256(sourceFailurePath))
        initial['sourcePrior'] = dict(accepted=False, solverAccepted=False, producerAcceptedAgrees=preservedFailure,
            status='unavailableSourceInitial')
    else:
        sourceInitial = readArrays(runRoot / case / 'initial/sourcePrior/geometry.npz')['vertices']
        sourceReport = readJson(runRoot / case / 'initial/sourcePrior/report.json')
        initial['sourcePrior'] = initialGeometryAudit(system, sourceInitial)
        sourceProof = auditSourcePrior(system, context, assets, runRoot, config)
        initial['sourcePrior']['solverAccepted'] = sourceReport['solver']['solverAccepted']
        initial['sourcePrior']['accepted'] &= sourceReport['solver']['solverAccepted']
        initial['sourcePrior']['producerAcceptedAgrees'] = initial['sourcePrior']['accepted'] == sourceReport['accepted']
    dense = readArrays(oldRoot / 'observations/observations.npz')['eyelidCurvesWorking']
    photoEyes = {name: {part: dense[i, j] for j, part in enumerate(('lower', 'upper'))}
                for i, name in enumerate(('imageLeft', 'imageRight'))}
    fullFaceLookup = {tuple(face): index for index, face in enumerate(assets.arrays['faces'])}
    expectedUvFaces = assets.arrays['uvFaces'][np.asarray([fullFaceLookup[tuple(face)] for face in raw['skinIds'][raw['faces']]])]
    anchors = {name: readArrays(path / 'fullGeometry.npz') for name, path in (
        ('originalPureDirect', oldRoot / 'geometry/pureDirect'),
        ('currentBounded', oldRoot / 'geometry/bounded'),
        ('replayedBaseline', runRoot / case / 'baseline'))}
    pairAnchors = {name: intersectionPairs(value['vertices'], value['faces']) for name, value in anchors.items()}
    branches = {}
    for branch, options in config['branches'].items():
        directory = runRoot / case / branch
        report = readJson(directory / 'report.json')
        if options['sourcePrior'] and sourceInitial is None:
            preserved = report.get('status') == 'blockedByInitialException' and report.get('geometrySafetyAccepted') is False
            branches[branch] = dict(validationPassed=preserved, status='blockedByInitialException',
                geometrySafetyAccepted=False, requestedEffectsAccepted=False, geometryExperimentAccepted=False,
                report=str(directory / 'report.json'), reportSha256=sha256(directory / 'report.json'), clientReady=False)
            continue
        states, final = readArrays(directory / 'states.npz'), readArrays(directory / 'finalGeometry.npz')
        full, beforeFull = readArrays(directory / 'fullGeometry.npz'), readArrays(directory / 'fullBeforeEyes.npz')
        beforeSkin, storedInitial = readArrays(directory / 'skinBeforeEyes.npz'), readArrays(directory / 'initial.npz')
        expectedInitial = sourceInitial if options['sourcePrior'] else boundedInitial
        expectedInitialGate = initial['sourcePrior' if options['sourcePrior'] else 'bounded']
        checks = dict(planIdentity=context['planSha256'] == sha256(runRoot / 'experimentPlan.json'),
            caseAndBranch=report['caseId'] == case and report['branch'] == branch,
            factorsAsFrozen=report['options'] == options,
            sameInputs=report['sourceHashes'] == context['sourceHashes'],
            sameReference=report['referenceViewIndex'] == reference,
            initialExactlyExpected=np.array_equal(storedInitial['vertices'], expectedInitial) and np.array_equal(states['initial'], expectedInitial),
            finalExactlyState=np.array_equal(final['vertices'], states['final']),
            fullSkinExactlyFinal=np.array_equal(full['vertices'][raw['skinIds']], final['vertices']),
            fullBeforeSkinExactlyBeforeSkin=np.array_equal(beforeFull['vertices'][raw['skinIds']], beforeSkin['vertices']),
            skinFacesOriginal=np.array_equal(final['faces'], raw['faces']),
            fullFacesOriginal=np.array_equal(full['faces'], raw['fullFaces']),
            uvsOriginal=np.array_equal(final['uvs'], assets.arrays['uvs']),
            skinUvFacesOriginal=np.array_equal(final['uvFaces'], expectedUvFaces),
            finalNotClientPromoted=report['finalCandidateAccepted'] is False,
            noAppearanceReplay=report['textureRebaked'] is False,
            initialDecisionReproduced=expectedInitialGate['accepted'] == report['initialAccepted'])
        artifactHashes = fileIdentityAudit({str(directory / key): value for key, value in report['outputHashes'].items()})
        checks['producerOutputHashes'] = artifactHashes['accepted']
        assetPath = Path(report['asset']['archivePath']); asset = _load(assetPath)
        beforeAsset = _load(directory / 'beforeEyesAsset/legacyAsset.zip')
        canonical = asset.mesh['vertices'] * 100 @ raw['rotation'] * raw['scale'] + raw['translation']
        checks['archiveRoundtrip'] = float(np.max(np.abs(canonical - full['vertices']))) < 1e-12
        checks['archiveIdentity'] = sha256(assetPath) == report['asset']['sha256']
        checks['exactTextureAndOtherPayloadIdentity'] = all(payload == asset.payloads[name] for name, payload in beforeAsset.payloads.items() if name != 'face_nicp.obj')
        checks['bindingMetadataExactlyPreserved'] = beforeAsset.manifest['binding'] == asset.manifest['binding']
        checks['originalOrderBindingFlags'] = all(asset.manifest['binding'][key] for key in ('vertexOrderPreserved', 'faceOrderPreserved', 'uvOrderPreserved', 'uvFaceOrderPreserved', 'materialFaceOrderPreserved')) and not asset.manifest['binding']['rigGenerated']
        for key in ('faces', 'uvFaces', 'materialIds'):
            checks['archive' + key + 'Original'] = np.array_equal(asset.mesh[key], assets.arrays[key])
        checks['archiveUvOriginal'] = np.max(np.abs(asset.mesh['uvs'] - assets.arrays['uvs'])) < 1e-15
        checks['archiveStaticOriginal'] = np.max(np.abs(asset.mesh['vertices'][raw['skinIds'][raw['staticIds']]] * 100 - assets.arrays['vertices'][raw['skinIds'][raw['staticIds']]])) < 1e-12
        downstream = downstreamGeometryAudit(system, expectedInitial, final['vertices'], camera)
        candidatePairs = intersectionPairs(full['vertices'], full['faces'])
        collisions = {name: intersectionSetAudit(pairs, candidatePairs, assets.arrays['materialIds']) for name, pairs in pairAnchors.items()}
        checks['originalFullPairDecisionReproduced'] = collisions['originalPureDirect']['newPairs'] == [tuple(pair) for pair in report['newFullPairsAgainstOriginalPureDirect']]
        checks['currentFullPairDecisionReproduced'] = collisions['currentBounded']['newPairs'] == [tuple(pair) for pair in report['newFullPairsAgainstCurrentBounded']]
        eye = None
        if options['visibleEyes']:
            scene = loadLegacyEyeBindingInputs(directory / 'beforeEyesAsset/legacyAsset.zip', raw['skinIds'])
            evidence = readEvidence(directory / 'eyeEvidence')
            eye = auditEyeEvidence(scene, beforeFull['vertices'], full['vertices'], camera, photoEyes, evidence, assets,
                tuple(inputs[reference]['workingSize'][::-1]), float(np.asarray(inputs[reference]['inputToWorking'])[0, 0]), raw['skinIds'][raw['staticIds']])
            checks['eyeEvidenceValidated'] = eye['validationPassed']
            checks['eyeEffectDecisionReproduced'] = report['eyeEffectAccepted'] == eye['effectAccepted']
        else:
            checks['noEyeStageGeometryChange'] = np.array_equal(beforeFull['vertices'], full['vertices'])
        stages = report['geometryRefinement']['stagedLandmarks'].get('stages', [])
        eyeStages = [value for value in stages if value['name'] in ('imageLeftEye', 'imageRightEye')]
        checks['legacyEyeReplacementAsFrozen'] = (all(value['status'] == 'replacedByVisibleEyeRefinement' for value in eyeStages)
            if options['visibleEyes'] else all(value['status'] != 'replacedByVisibleEyeRefinement' for value in eyeStages))
        safety = bool(expectedInitialGate['accepted'] and downstream['accepted'] and all(value['accepted'] for value in collisions.values()))
        producerSafetyAgreement = safety == report['geometrySafetyAccepted']
        # Physical gates have their original tolerances; the producer also
        # reports normalized solver constraints. A disagreement is explicit.
        checks['producerSafetyDecisionReproduced'] = producerSafetyAgreement
        effect = (not options['sourcePrior'] or sourceProof['effectAvailable']) and (not options['visibleEyes'] or eye['effectAccepted'])
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks,
            geometrySafetyAccepted=safety, requestedEffectsAccepted=bool(effect),
            geometryExperimentAccepted=bool(safety and effect and all(checks.values()) and sourceProof['validationPassed']),
            downstream=downstream, fullCollisionAnchors=collisions, eye=eye,
            producerOutputIdentities=artifactHashes, report=str(directory / 'report.json'),
            reportSha256=sha256(directory / 'report.json'), assetSha256=sha256(assetPath), clientReady=False)
    originalFull = anchors['currentBounded']['vertices']; baselineFull = anchors['replayedBaseline']['vertices']
    controlReplayError = float(np.max(np.abs(originalFull - baselineFull)))
    inputHashes = fileIdentityAudit(context['sourceHashes'])
    complete = bool(inputHashes['accepted'] and sourceProof['validationPassed']
        and initial['sourcePrior']['producerAcceptedAgrees'] and controlReplayError <= 1e-12
        and all(value['validationPassed'] for value in branches.values()))
    result = dict(caseId=case, validationPassed=complete, originalInputIdentities=inputHashes,
        replayedFullBaselineMaximumError=controlReplayError, sourcePriorEvidence=sourceProof,
        initial=initial, branches=branches, planSha256=sha256(runRoot / 'experimentPlan.json'),
        validatorSha256=sha256(__file__), originalAuditSha256=sha256(ORIGINAL_AUDIT),
        clientReady=False, noCandidateSolvesPerformed=True)
    outputRoot.mkdir(parents=True, exist_ok=True)
    writeJson(outputRoot / f'{case}.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/featureGeometry20261024')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/featureGeometry20261024/independentAudit')
    args = parser.parse_args()
    results = [auditCase(case, args.run_root, args.output) for case in args.case]
    summary = dict(validationPassed=all(value['validationPassed'] for value in results),
        cases=[dict(caseId=value['caseId'], validationPassed=value['validationPassed'],
            acceptedBranches=[name for name, branch in value['branches'].items() if branch['geometryExperimentAccepted']]) for value in results],
        clientReady=False, validatorSha256=sha256(__file__))
    writeJson(args.output / 'validationSummary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
