"""Read-only acceptance for supported-source and local semantic-eye factors.

No fitting or candidate selection is performed here. The original physical
gates and complete-face collision primitive are reused from the prior audit.
Additional checks establish that this round actually keeps common accessory
positions and derives protection from input support rather than case IDs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from validateFeatureGeometryExperiment import (
    downstreamGeometryAudit, fileIdentityAudit, initialGeometryAudit,
    intersectionSetAudit, loadOriginalAudit, readArrays, readEvidence,
    readJson, sceneIdentityAudit, sha256, strictIntersectionAudit,
    _sparseMaximumDifference,
)
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.camera import projectPoints
from faceAppearance.artifacts import writeJson
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry

BRANCHES = ('baseline', 'sourceProtected', 'localEyes', 'combined')
GLOBAL_EYE_PAIRS = {'localEyes': 'baseline', 'combined': 'sourceProtected'}


def _vertices(value, name):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite N by 3 coordinates')
    return result


def _ids(value, count, name):
    result = np.asarray(value)
    if not result.size:
        return np.empty(0, np.int64)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or result.min() < 0
            or result.max() >= count or len(np.unique(result)) != len(result)):
        raise ValueError(f'{name} must be distinct in-range integer IDs')
    return result.astype(np.int64, copy=False)


def _mask(ids, count):
    result = np.zeros(count, bool)
    result[ids] = True
    return result


def structuralSupportDomain(sampleOperator, faces, *, originalStaticIds=(), delegatedCavityIds=(), eyeMouthIds=()):
    """Reconstruct support and one-ring protection from original C and topology.

    A nonzero C column establishes structural support only, regardless of its
    magnitude. No squared-support threshold, confidence estimate, case name or
    candidate coordinate enters this rule. Stored duplicates are summed before
    explicit zero coefficients are removed.
    """
    sample = sparse.csr_matrix(sampleOperator, dtype=np.float64, copy=True)
    sample.sum_duplicates(); sample.eliminate_zeros()
    count = sample.shape[1]; faces = np.asarray(faces)
    if (not count or not np.isfinite(sample.data).all()
            or (sample.data < 0).any() or np.max(np.abs(np.asarray(sample.sum(1)).ravel() - 1), initial=0) > 1e-12
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min() < 0 or faces.max() >= count):
        raise ValueError('Expected original finite convex affine C and valid skin topology')
    fixed = _mask(_ids(originalStaticIds, count, 'originalStaticIds'), count)
    delegated = _mask(_ids(delegatedCavityIds, count, 'delegatedCavityIds'), count)
    eyeMouth = _mask(_ids(eyeMouthIds, count, 'eyeMouthIds'), count)
    supported = np.diff(sample.tocsc().indptr) > 0
    unsupportedFixed = ~supported & ~fixed & ~delegated
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    adjacent = np.zeros(count, bool)
    adjacent[edges[unsupportedFixed[edges].any(1)].ravel()] = True
    transition = adjacent & supported & ~fixed & ~delegated
    # The cavity's existing controls and bounds remain separately delegated.
    localControls = (transition | eyeMouth) & ~fixed & ~unsupportedFixed
    controls = localControls | (delegated & ~fixed)
    graph = sparse.csr_matrix((np.ones(2 * len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(count, count))
    unobservedIds = np.flatnonzero(unsupportedFixed)
    componentCount, labels = connected_components(graph[unobservedIds][:, unobservedIds], directed=False)
    return dict(supportedMask=supported, unsupportedFixedMask=unsupportedFixed,
        transitionMask=transition, eyeMouthMask=eyeMouth, delegatedCavityMask=delegated,
        originalStaticMask=fixed, hardFixedMask=fixed | unsupportedFixed,
        hardFixedVertexIds=np.flatnonzero(fixed | unsupportedFixed),
        unsupportedFixedVertexIds=unobservedIds, transitionVertexIds=np.flatnonzero(transition),
        localControlVertexIds=np.flatnonzero(localControls), controlVertexIds=np.flatnonzero(controls),
        unsupportedComponentCount=int(componentCount), unsupportedComponentLabels=labels,
        incidentProtectedEdgeMask=unsupportedFixed[edges].any(1),
        incidentProtectedFaceMask=unsupportedFixed[faces].any(1), edges=edges,
        guardedEdgeMask=unsupportedFixed[edges].any(1) & ~delegated[edges].any(1),
        guardedFaceMask=unsupportedFixed[faces].any(1) & ~delegated[faces].any(1),
        rule='C nonzero columns after sparse canonicalization; unsupported nonstatic noncavity fixed; one skin-topology ring.',
        confidenceEstimated=False, candidateDependent=False)


def hardSupportProtectionAudit(baselineBounded, candidate, domain):
    before, after = _vertices(baselineBounded, 'baselineBounded'), _vertices(candidate, 'candidate')
    if before.shape != after.shape or domain['hardFixedMask'].shape != (len(before),):
        raise ValueError('Hard support protection requires common canonical vertex identities')
    change = np.any(before != after, axis=1)
    checks = dict(originalStaticExactlyFixed=not (change & domain['originalStaticMask']).any(),
        unsupportedExactlyAtBoundedBaseline=not (change & domain['unsupportedFixedMask']).any(),
        hardFixedExactlyPreserved=not (change & domain['hardFixedMask']).any())
    return dict(accepted=bool(all(checks.values())), checks=checks,
        unexpectedChangedVertexIds=np.flatnonzero(change & domain['hardFixedMask']),
        supportedChangedVertexIds=np.flatnonzero(change & domain['supportedMask']),
        delegatedCavityChangedVertexIds=np.flatnonzero(change & domain['delegatedCavityMask']),
        anchor='Exact bounded initial canonical coordinates; original pureDirect/rest still anchor all physical gates.')


def fixedAttachmentAudit(baselineScene, candidateScene, candidateSkin, *, staticSkinIds=()):
    """Protect every non-skin ID, including material-zero canthus accessories."""
    before, after = _vertices(baselineScene['vertices'], 'baselineFull'), _vertices(candidateScene['vertices'], 'candidateFull')
    skinIds = _ids(baselineScene['nicpToFullIds'], len(before), 'nicpToFullIds')
    skin = _vertices(candidateSkin, 'candidateSkin')
    if skin.shape != (len(skinIds), 3) or before.shape != after.shape:
        raise ValueError('Full scene and skin canonical arrays must share the original mapping')
    nonSkin = np.flatnonzero(~_mask(skinIds, len(before)))
    fixed = skinIds[_ids(staticSkinIds, len(skinIds), 'staticSkinIds')]
    identity = sceneIdentityAudit(baselineScene, candidateScene, fixedVertexIds=fixed, protectedVertexIds=nonSkin)
    identity['checks']['fullSkinExactlyMatchesCandidate'] = np.array_equal(after[skinIds], skin)
    expected = before.copy(); expected[skinIds] = skin
    identity['checks']['exactlyCommonFullWithSkinReplaced'] = np.array_equal(after, expected)
    identity['accepted'] = bool(all(identity['checks'].values()))
    identity.update(nonSkinVertexIds=nonSkin,
        allNonSkinFixed=True, materialBasedExclusionUsed=False,
        interpretation='Fixed-baseline accessory control, not proof that fresh candidate assembly is safe.')
    return identity


def threeAnchorFullAudit(candidateVertices, faces, materialIds, anchors):
    required = {'originalPureDirect', 'currentBounded', 'pairedBaseline'}
    if set(anchors) != required:
        raise ValueError('All three original full-scene collision anchors are required')
    candidate = _vertices(candidateVertices, 'candidateVertices')
    faces = np.asarray(faces)
    nextPairs = intersectionPairs(candidate, faces)
    reports = {}
    for name, value in anchors.items():
        if not np.array_equal(value['faces'], faces) or np.asarray(value['vertices']).shape != candidate.shape:
            raise ValueError('Anchor face and vertex identities must be unchanged')
        reports[name] = intersectionSetAudit(intersectionPairs(value['vertices'], faces), nextPairs, materialIds)
    return dict(accepted=all(value['accepted'] for value in reports.values()), anchors=reports,
        noRegressionAgainstPairedBaseline=reports['pairedBaseline']['accepted'],
        historicalAnchorRequirementsRetained=True)


def localEyeStageIdentityAudit(beforeStage, afterStage, supportMask, *, originalStaticIds=()):
    """A bounded semantic stage may not move any coordinate outside its support."""
    before, after = _vertices(beforeStage, 'beforeStage'), _vertices(afterStage, 'afterStage')
    support = np.asarray(supportMask)
    if before.shape != after.shape or support.shape != (len(before),) or support.dtype != np.bool_:
        raise ValueError('Expected common coordinates and a frozen boolean eye support')
    static = _ids(originalStaticIds, len(before), 'originalStaticIds')
    change = np.any(before != after, axis=1)
    checks = dict(outsideSupportExactlyFixed=np.array_equal(before[~support], after[~support]),
        originalStaticExactlyFixed=np.array_equal(before[static], after[static]))
    return dict(accepted=bool(all(checks.values())), checks=checks,
        changedVertexIds=np.flatnonzero(change), changedOutsideSupportIds=np.flatnonzero(change & ~support),
        geometryChanged=bool(change.any()), anchor='This stage input, not the original pre-mouth mesh.')


def semanticTargetErrors(vertices, bindingVertexIds, barycentric, targetPixels, camera):
    """Measure the same original six-point constraints on any paired geometry."""
    vertices = _vertices(vertices, 'vertices'); ids = np.asarray(bindingVertexIds)
    bary, target = np.asarray(barycentric), np.asarray(targetPixels)
    if (ids.ndim != 2 or ids.shape[1] != 3 or ids.dtype.kind not in 'iu' or not len(ids)
            or ids.min() < 0 or ids.max() >= len(vertices) or bary.shape != ids.shape
            or target.shape != (len(ids), 2) or not np.isfinite(bary).all() or not np.isfinite(target).all()
            or (bary < -1e-7).any() or (bary > 1 + 1e-7).any()
            or np.max(np.abs(bary.sum(1) - 1)) > 1e-7):
        raise ValueError('Expected fixed original semantic binding and photo targets')
    points = np.einsum('ij,ijk->ik', bary, vertices[ids])
    pixels, depth = projectPoints(points, np.asarray(camera['intrinsics']), np.asarray(camera['rotation']), np.asarray(camera['translation']))
    if not np.isfinite(pixels).all() or not (depth > 0).all():
        raise ValueError('Semantic constraints must project in front of the fixed camera')
    errors = np.linalg.norm(pixels - target, axis=1)
    return dict(meanPixels=float(errors.mean()), maximumPixels=float(errors.max()),
        errorsPixels=errors, projectedPixels=pixels, targetPixels=target, pointCount=len(errors))


def pairedSemanticAlignmentAudit(localMetrics, globalMetrics, *, tolerancePixels=None):
    """Keep coarse-fit evidence distinct from locality and anatomical likeness.

    A hard paired threshold must come from the prospectively frozen plan. With
    no threshold this function reports measurements without inventing a gate.
    """
    if localMetrics['pointCount'] != globalMetrics['pointCount'] or not np.array_equal(localMetrics['targetPixels'], globalMetrics['targetPixels']):
        raise ValueError('Paired comparison requires identical selected photo constraints')
    meanChange = localMetrics['meanPixels'] - globalMetrics['meanPixels']
    maxChange = localMetrics['maximumPixels'] - globalMetrics['maximumPixels']
    result = dict(meanChangePixels=meanChange, maximumChangePixels=maxChange,
        sameTargets=True, diagnosticOnly=tolerancePixels is None, anatomicalTruthEstablished=False)
    if tolerancePixels is not None:
        if not np.isfinite(tolerancePixels) or tolerancePixels < 0:
            raise ValueError('The frozen paired tolerance must be finite and nonnegative')
        result.update(accepted=bool(meanChange <= tolerancePixels), meanTolerancePixels=float(tolerancePixels))
    return result


PLAN_SHA = '6119af96fe2b269e15f33a45b89514887378ed0377fc4e3995a587975696f4ce'
PREVIOUS_AUDIT_SHA = '52b3e2fbaf04412dddd8b272867c7c6c36b8df22c737b134b9dbd8584d6fb5ef'


def supportSpecAudit(system, baseline, candidate, spec, solveDomain):
    """Rebuild the protected domain and bound values without the producer builder."""
    raw = system['a']; unit = float(raw['unit']); faces = raw['faces']
    domain = structuralSupportDomain(system['C'], faces, originalStaticIds=raw['staticIds'],
        delegatedCavityIds=system['controls'], eyeMouthIds=raw['eyeMouthIds'])
    expected = dict(supportedMask=domain['supportedMask'], unsupportedFreeMask=domain['unsupportedFixedMask'],
        originalStaticVertexIds=raw['staticIds'], delegatedVertexIds=system['controls'], protectedVertexIds=raw['eyeMouthIds'],
        hardFixedVertexIds=domain['hardFixedVertexIds'], transitionVertexIds=domain['transitionVertexIds'],
        controlVertexIds=domain['localControlVertexIds'], edgeVertexIds=domain['edges'][domain['guardedEdgeMask']],
        faceVertexIds=faces[domain['guardedFaceMask']], faceIds=np.flatnonzero(domain['guardedFaceMask']),
        referenceVertices=baseline, safetyReferenceVertices=raw['pureDirect'], faces=faces)
    checks = {key + 'Reconstructed': np.array_equal(spec[key], value) for key, value in expected.items()}
    components = [domain['unsupportedFixedVertexIds'][domain['unsupportedComponentLabels'] == i]
                  for i in range(domain['unsupportedComponentCount'])]
    checks['componentsReconstructed'] = (len(components) == len(spec['unsupportedComponentVertexIds'])
        and all(np.array_equal(a, b) for a, b in zip(components, spec['unsupportedComponentVertexIds'])))
    checks.update(controlUnionReconstructed=np.array_equal(solveDomain['controlVertexIds'], domain['controlVertexIds']),
        hardEliminationReconstructed=np.array_equal(solveDomain['fixedVertexIds'], domain['hardFixedVertexIds']),
        initialBoundedNotPureDirect=np.array_equal(solveDomain['startVertices'], baseline),
        physicalUnitUnchanged=spec['lengthScale'] == unit,
        noConfidenceOrRenormalization=spec['diagnostics']['supportThreshold'] == 0
            and spec['diagnostics']['sampleMatrixRenormalized'] is False
            and spec['diagnostics']['candidateDependentSelection'] is False)
    checks['commonPolicyExactlyFrozen'] = spec['policy'] == dict(maximumTransitionDisplacement=.5,
        maximumProtectedDisplacement=.1, minimumEdgeRatio=.9, maximumEdgeRatio=1.1,
        minimumAreaRatio=.8, maximumAreaRatio=1.25, minimumNormalCosine=.95)
    edges, triangles = expected['edgeVertexIds'], expected['faceVertexIds']
    oldLength = np.linalg.norm(raw['pureDirect'][edges[:, 1]] - raw['pureDirect'][edges[:, 0]], axis=1)
    faceStats = loadOriginalAudit().priorAudit.faceStats
    oldArea, oldNormal, _ = faceStats(raw['pureDirect'], triangles)
    limits = np.where(domain['eyeMouthMask'][expected['controlVertexIds']], .1, .5) * unit
    numeric = dict(edgeLower=.9 * oldLength, edgeUpper=1.1 * oldLength,
        areaLower=.8 * oldArea, areaUpper=1.25 * oldArea, safetyReferenceNormals=oldNormal,
        displacementLimits=limits)
    for key, value in numeric.items():
        checks[key + 'OriginalSafetyFrame'] = np.allclose(spec[key], value, atol=1e-14, rtol=1e-12)
    fixed = hardSupportProtectionAudit(baseline, candidate, domain)
    lengths = np.linalg.norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1) / unit
    area, normal, _ = faceStats(candidate / unit, triangles)
    shift = (candidate[expected['controlVertexIds']] - raw['pureDirect'][expected['controlVertexIds']]) / unit
    values = np.r_[lengths - .9 * oldLength / unit, 1.1 * oldLength / unit - lengths,
        area - .8 * oldArea / unit**2, 1.25 * oldArea / unit**2 - area,
        np.einsum('ij,ij->i', normal, oldNormal) - .95,
        (limits / unit)**2 - np.einsum('ij,ij->i', shift, shift)]
    return dict(validationPassed=bool(all(checks.values())), checks=checks, hardProtection=fixed,
        constraintAccepted=bool((values >= -1e-8).all()), minimumConstraint=float(values.min(initial=0)),
        violatedConstraintIds=np.flatnonzero(values < -1e-8),
        counts=dict(newHardFixed=int(domain['unsupportedFixedMask'].sum()), controls=len(domain['controlVertexIds']),
            transitions=len(domain['transitionVertexIds']), guardedEdges=len(edges), guardedFaces=len(triangles)),
        fixedVertexIds=domain['hardFixedVertexIds'], controlVertexIds=domain['controlVertexIds'])


def sourceObjectiveAudit(system, context, directory, candidate, support):
    """Reuse independently proven unchanged D; reconstruct new full H/b and gradient.

    The previous source graph was independently reconstructed for all seven
    inputs. This round may change constraints only. Every old proof, artifact,
    input, C/D/A/target and source-evidence byte is checked before reusing it.
    """
    case = context['entry']['caseId']; oldRoot = ROOT / 'runs/featureGeometry20261024' / case
    oldDirectory = oldRoot / 'initial/sourcePrior'
    summaryPath = ROOT / 'reports/featureGeometry20261024/independentAudit/auditSummary.json'
    summary = readJson(summaryPath)
    proofPath = summaryPath.parent / f'{case}.json'; proof = readJson(proofPath)
    report = readJson(directory / 'report.json'); oldReport = readJson(oldDirectory / 'report.json')
    oldContext = readJson(oldRoot / 'inputContext.json')
    raw = system['a']; unit = float(raw['unit'])
    checks = dict(previousAuditFrozen=sha256(summaryPath) == PREVIOUS_AUDIT_SHA,
        previousCaseProofFrozen=sha256(proofPath) == summary['hashes'][str(proofPath)],
        previousObjectiveIndependentlyValidated=proof['sourcePriorEvidence']['validationPassed'],
        samePreparedInput=context['entry']['preparedGeometrySha256'] == oldContext['entry']['preparedGeometrySha256'],
        sameSourceInput=context['sourceHashes'][str(Path(context['entry']['reconstruction']) / 'geometry.npz')]
            == oldContext['sourceHashes'][str(Path(context['entry']['reconstruction']) / 'geometry.npz')])
    oldHashes = fileIdentityAudit({str(oldDirectory / key): value for key, value in oldReport['outputHashes'].items()})
    checks['oldObjectiveArtifactHashes'] = oldHashes['accepted']
    checks['oldIndependentValidatorIdentity'] = sha256(ROOT / 'tools/validateFeatureGeometryExperiment.py') == proof['validatorSha256']
    errors = {}
    for name in ('sampleMatrix', 'differentialOperator', 'sourceOperator'):
        errors[name] = _sparseMaximumDifference(sparse.load_npz(directory / f'{name}.npz'), sparse.load_npz(oldDirectory / f'{name}.npz'))
        checks[name + 'ExactlySameValidatedObjective'] = errors[name] == 0.
    for name in ('sourceEvidence.json', 'sourceEvidence.npz', 'sourceTargets.npz', 'sourcePrior.json'):
        # NPZ containers can carry creation timestamps; compare payloads, not ZIP bytes.
        if name.endswith('.npz'):
            first, second = readArrays(directory / name), readArrays(oldDirectory / name)
            checks[name + 'SamePayload'] = first.keys() == second.keys() and all(np.array_equal(first[k], second[k]) for k in first)
        else:
            checks[name + 'SamePayload'] = readJson(directory / name) == readJson(oldDirectory / name)
    a = sparse.load_npz(directory / 'sourceOperator.npz'); target = readArrays(directory / 'sourceTargets.npz')['targets']
    weight = oldReport['weight']; checks['sourceWeightFrozen'] = report['weight'] == weight
    h = sparse.kron(system['scalarH'], sparse.eye(3), format='csr') + .01 * (system['P'].T @ system['P']) + weight * (a.T @ a)
    rhs = system['rhs'].ravel() + .01 * (system['P'].T @ system['target']) + weight * (a.T @ target)
    errors['normalMatrix'] = _sparseMaximumDifference(h, sparse.load_npz(directory / 'normalMatrixH.npz'))
    errors['normalRhs'] = float(np.abs(rhs - readArrays(directory / 'normalRhs.npz')['b']).max(initial=0))
    checks['fullNormalMatrixReconstructed'] = errors['normalMatrix'] < 1e-9
    checks['fullNormalRhsReconstructed'] = errors['normalRhs'] < 1e-9
    free = np.setdiff1d(np.arange(len(candidate)), np.union1d(support['fixedVertexIds'], support['controlVertexIds']))
    gradient = (h @ candidate.ravel() - rhs).reshape(-1, 3)
    relativeGradient = float(np.linalg.norm(gradient[free]) / max(np.linalg.norm(rhs.reshape(-1, 3)[free]), 1e-30))
    checks['exactSchurRecoveredFreeGradient'] = not report['solver']['solverAccepted'] or relativeGradient < 1e-10
    solver = report['solver']
    checks['commonSolverSettings'] = (solver['lengthScale'] == unit and solver['maxIterations'] == 200
        and solver['feasibilityTolerance'] == 1e-8 and solver['solverTolerance'] == 1e-10)
    checks['solverDomainCountsReconstructed'] = (solver['retainedFreeVertexCount'] == len(support['controlVertexIds'])
        and solver['fixedVertexCount'] == len(support['fixedVertexIds']) and solver['eliminatedFreeVertexCount'] == len(free))
    checks['solverAcceptanceBooleanReproduced'] = solver['solverAccepted'] == bool(solver['optimizerSuccess']
        and solver['constraintsFeasible'] and solver['objectiveNotIncreased'])
    hashes = fileIdentityAudit({str(directory / key): value for key, value in report['outputHashes'].items()})
    checks['newOutputHashes'] = hashes['accepted']
    return dict(validationPassed=bool(all(checks.values())), checks=checks, matrixMaximumErrors=errors,
        relativeFreeNormalResidual=relativeGradient, effectiveWeight=weight,
        effectAvailable=bool(report['sourcePrior']['active']), sourcePriorChanged=False,
        sourceDifferenceMeaning='Original fixed UFF optimization objective, not independently calibrated confidence or anatomical truth.',
        proofReuse=dict(path=str(proofPath), sha256=sha256(proofPath), frozenSummarySha256=PREVIOUS_AUDIT_SHA),
        sourceHashes=hashes)


def eyeSupportAudit(evidence, support, faces, camera, photo, confidence, imageShape, static, fraction):
    """Independently rebuild the fixed physical eye-width skin graph domain."""
    before = np.asarray(evidence['frozenSupportReferenceVertices']); ids = np.asarray(evidence['vertexIndices'])
    bary = np.asarray(evidence['barycentricWeights']); eyeIds = np.asarray(evidence['ibugIds'])
    canthi = np.array([36, 39] if eyeIds.min() < 42 else [42, 45])
    seeds = np.unique(ids); samples = (before[ids] * bary[..., None]).sum(1)
    depths = samples @ np.asarray(camera['rotation'])[2] + np.asarray(camera['translation'])[2]
    reasons = []
    if len(eyeIds) < 4: reasons.append('insufficientEligibleEyeTargets')
    if (confidence[canthi] < .5).any(): reasons.append('unreliablePhotoCanthusWidth')
    if (photo[canthi] < 0).any() or (photo[canthi] >= np.array(imageShape[::-1])).any(): reasons.append('photoCanthusOutsideWorkingImage')
    if not (depths > 0).all(): reasons.append('semanticSeedsBehindCamera')
    radius = width = depth = None; distance = np.full(len(before), np.inf); movable = np.zeros(len(before), bool)
    if not reasons:
        rays = np.linalg.solve(np.asarray(camera['intrinsics']), np.c_[photo[canthi], np.ones(2)].T).T
        rays /= rays[:, 2, None]; depth = float(np.median(depths)); width = float(np.linalg.norm(rays[1] - rays[0]) * depth)
        if not np.isfinite(width) or width <= 0: reasons.append('degeneratePhotoCanthusWidth')
        else:
            radius = fraction * width
            edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
            lengths = np.linalg.norm(before[edges[:, 1]] - before[edges[:, 0]], axis=1)
            graph = sparse.csr_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(before),) * 2)
            distance = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=radius * (1 + 1e-12))
            movable = distance <= radius * (1 + 1e-12); movable[static] = False
            if not movable.any(): reasons.append('allLocalSupportHardFixed')
    if reasons: movable[:] = False
    measured = np.isfinite(distance); storedDistance = np.where(measured, distance, 0.)
    checks = dict(seedsOriginal=np.array_equal(support['seedVertexIds'], seeds),
        selectedBindingsOriginal=np.array_equal(support['selectedBindingMask'], evidence['selectedBindingMask']),
        photoCanthiOriginal=np.array_equal(support['photoCanthusPixels'], photo[canthi]),
        movableReconstructed=np.array_equal(support['movableMask'], movable),
        fixedComplementExact=np.array_equal(support['fixedMask'], ~movable),
        fixedIdsExact=np.array_equal(support['fixedVertexIds'], np.flatnonzero(~movable)),
        originalStaticExact=np.array_equal(support['originalFixedVertexIds'], static),
        truncatedDistanceMask=np.array_equal(support['geodesicDistanceMeasuredMask'], measured),
        geodesicDistances=np.allclose(support['geodesicDistance'], storedDistance, atol=1e-14, rtol=1e-12),
        reasonsReproduced=support['diagnostics']['reasons'] == reasons,
        statusReproduced=support['diagnostics']['status'] == ('supported' if not reasons else 'unsupported'),
        commonPolicy=support['diagnostics']['policy'] == {'radiusEyeWidthFraction': fraction})
    for key, value in (('localCameraDepth', depth), ('photoEyeWidthCanonical', width), ('geodesicRadiusCanonical', radius)):
        checks[key] = support['diagnostics'][key] == value
    return dict(validationPassed=bool(all(checks.values())), checks=checks, movableMask=movable,
        unsupportedReasons=reasons, movableVertexCount=int(movable.sum()), radiusCanonical=radius)


def _fitAgrees(saved, metrics):
    return (saved['count'] == metrics['pointCount']
        and np.allclose(saved['errorsPixels'], metrics['errorsPixels'], atol=1e-11, rtol=1e-12)
        and abs(saved['meanErrorPixels'] - metrics['meanPixels']) < 1e-11
        and abs(saved['maximumErrorPixels'] - metrics['maximumPixels']) < 1e-11)


def originalObservationMasks(initial, faces, binding, cameras, inputs, observations):
    """Reapply the original visibility primitive and fixed observation eligibility."""
    from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
    points = (initial[binding['vertexIndices']] * binding['barycentricWeights'][..., None]).sum(1)
    masks = []
    for view, camera in enumerate(cameras):
        if not camera.get('valid', True) or not observations['accepted'][view]:
            masks.append(np.zeros(len(binding['ibugIds']), bool)); continue
        rotation = np.asarray(camera['rotation']); translation = np.asarray(camera['translation'])
        cv = initial @ rotation.T + translation; query = points @ rotation.T + translation
        size = inputs[view]['workingSize']
        checked = ProjectedSurfaceVisibility(cv, faces, np.asarray(camera['intrinsics']), tuple(size[::-1])).closestDepth(query)
        keep = checked['insidePhoto'] & np.isfinite(checked['depth'])
        keep &= abs(checked['depth'] - query[:, 2]) <= float(np.ptp(initial, axis=0).max()) * .002
        target = observations['points'][view][binding['ibugIds']]
        score = observations['confidence'][view][binding['ibugIds']]
        keep &= binding['validMask'] & (score >= np.where(binding['ibugIds'] >= 48, .4, .5))
        keep &= (target >= 0).all(1) & (target < size).all(1)
        masks.append(keep)
    return masks


def originalObservationErrors(vertices, binding, cameras, observations, masks):
    samples = (vertices[binding['vertexIndices']] * binding['barycentricWeights'][..., None]).sum(1)
    output = []
    for view, (camera, mask) in enumerate(zip(cameras, masks)):
        groups = {}
        if mask.any():
            pixels = projectPoints(samples, np.asarray(camera['intrinsics']), np.asarray(camera['rotation']), np.asarray(camera['translation']))[0]
            error = np.linalg.norm(pixels - observations['points'][view][binding['ibugIds']], axis=1)
        for name, condition in (('eyes', binding['ibugIds'] < 48), ('mouth', binding['ibugIds'] >= 48)):
            selected = mask & condition
            groups[name] = dict(count=int(selected.sum()), meanPixels=float(error[selected].mean()) if selected.any() else None)
        output.append(groups)
    return output


def auditSemanticStages(directory, context, raw, cameras, inputs, options, config):
    """Recompute stage proposals, locality and same-target error gates, no solver."""
    from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
    from faceAppearance.clientLipTrace import mouthObservationQuality
    binding = ibugClientLandmarks(loadClientLandmarks(raw['faces'], path=ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'), groups=('eyes', 'mouth'))
    diagnostics = readEvidence(directory / 'refinementEvidence')
    initial = readArrays(directory / 'initial.npz')['vertices']
    mouth = readArrays(directory / 'mouthState.npz')
    observation = readArrays(Path(context['oldRoot']) / 'geometry/photoObservations.npz')
    ref = context['referenceSelection']['referenceViewIndex']; camera = cameras[ref]
    photo, confidence = observation['points'][ref], observation['confidence'][ref]
    shape = tuple(inputs[ref]['workingSize'][::-1]); imageScale = min(shape) / 512.
    masks = originalObservationMasks(initial, raw['faces'], binding, cameras, inputs, observation)
    projected = projectPoints((initial[binding['vertexIndices']] * binding['barycentricWeights'][..., None]).sum(1),
        np.asarray(camera['intrinsics']), np.asarray(camera['rotation']), np.asarray(camera['translation']))[0]
    eligible = masks[ref] & (np.linalg.norm(projected - photo[binding['ibugIds']], axis=1) <= 40 * imageScale)
    eligibilityAgrees = np.array_equal(eligible, mouth['frontEligibleFitMask'])
    quality = mouthObservationQuality(photo, confidence, shape)
    referenceFrontal = not any(reason in quality['reasons'] for reason in ('referenceMayNotBeFrontal', 'referenceRollOrEyeOrderUnreliable'))
    baselineErrors = originalObservationErrors(initial, binding, cameras, observation, masks)
    stageMap = {value['name']: value for value in diagnostics['stagedLandmarks'].get('stages', [])}
    result = {}; previousAfter = None
    for name, group in (('imageLeftEye', np.arange(36, 42)), ('imageRightEye', np.arange(42, 48))):
        stage = stageMap[name]; checks = dict(originalStageRetained=stage['status'] != 'replacedByVisibleEyeRefinement',
            originalEligibilityReconstructed=eligibilityAgrees)
        if 'stageEvidence' not in stage:
            count = int((eligible & np.isin(binding['ibugIds'], group)).sum())
            validSkip = (stage['status'] == 'skippedUnreliableObservation' and (not referenceFrontal or count < 4)
                and stage['scale'] == 0. and stage.get('selectedPoints') == count)
            result[name] = dict(validationPassed=bool(validSkip and all(checks.values())), status=stage['status'],
                comparable=False, geometryChanged=False, checks=checks)
            continue
        e = stage['stageEvidence']; before, after = np.asarray(e['before']), np.asarray(e['acceptedAfter'])
        selected = eligible & np.isin(binding['ibugIds'], group)
        expected = dict(selectedBindingMask=selected, ibugIds=binding['ibugIds'][selected],
            vertexIndices=binding['vertexIndices'][selected], barycentricWeights=binding['barycentricWeights'][selected],
            targetPixels=photo[binding['ibugIds'][selected]], confidence=np.clip(confidence[binding['ibugIds'][selected]], 0, 1),
            weights=np.where(np.isin(binding['ibugIds'][selected], [36, 39, 42, 45, 48, 54]), 4., 1.),
            frozenSupportReferenceVertices=initial)
        checks.update({key + 'Original': np.array_equal(e[key], value) for key, value in expected.items()})
        if previousAfter is not None: checks['previousEyeStageExactlyPreserved'] = np.array_equal(before, previousAfter)
        beforeMetrics = semanticTargetErrors(before, expected['vertexIndices'], expected['barycentricWeights'], expected['targetPixels'], camera)
        afterMetrics = semanticTargetErrors(after, expected['vertexIndices'], expected['barycentricWeights'], expected['targetPixels'], camera)
        checks['beforeErrorReproduced'] = _fitAgrees(e['beforeFit'], beforeMetrics)
        checks['afterErrorReproduced'] = _fitAgrees(e['afterFit'], afterMetrics)
        supportProof = None
        if options['localEyes']:
            supportProof = eyeSupportAudit(e, stage['localSupport'], raw['faces'], camera, photo, confidence, shape, raw['staticIds'], config['localEyes']['radiusEyeWidthFraction'])
            checks['supportValidated'] = supportProof['validationPassed']
            locality = localEyeStageIdentityAudit(before, after, supportProof['movableMask'], originalStaticIds=raw['staticIds'])
            checks['outsideStageSupportExactlyFixed'] = locality['accepted']
        else:
            checks['legacyGlobalSupportMode'] = e['supportMode'] == 'legacyGlobalCompactField' and 'localSupport' not in stage
        extent = float(np.ptp(initial, axis=0).max()); attempts = []; expectedAfter = before.copy()
        previousErrors = originalObservationErrors(before, binding, cameras, observation, masks)
        proposal = np.asarray(e['proposal']) if 'proposal' in e else None
        if proposal is not None:
            scales = [1., .5, .25, .125, .0625, .03125, .015625]
            checks['originalBacktrackingSequence'] = [item['scale'] for item in stage['attempts']] == scales[:len(stage['attempts'])]
            beforePairs = intersectionPairs(before, raw['faces'])
            for item in stage['attempts']:
                candidate = before + item['scale'] * (proposal - before)
                cumulative = checkFrontGeometry(initial, candidate, raw['faces'], camera=camera, fixedVertexIds=raw['staticIds'], maxDisplacement=extent*.03, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
                ac = dict(originalCumulativeGate=cumulative['accepted'] == item['geometry']['accepted'])
                localPassed = True
                if options['localEyes'] and cumulative['accepted']:
                    fixed = np.flatnonzero(~supportProof['movableMask'])
                    local = checkFrontGeometry(before, candidate, raw['faces'], camera=camera, fixedVertexIds=fixed, maxDisplacement=extent*.03, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
                    fit = semanticTargetErrors(candidate, expected['vertexIndices'], expected['barycentricWeights'], expected['targetPixels'], camera)
                    fitPassed = fit['meanPixels'] <= beforeMetrics['meanPixels'] + .05*imageScale and fit['maximumPixels'] <= beforeMetrics['maximumPixels'] + .05*imageScale
                    ac['localPhysicalGate'] = local['accepted'] == item['localSupportGeometry']['accepted']
                    ac['ownEyeMeanAndMaximumGate'] = fitPassed == item['localEyeAlignment']['accepted']
                    localPassed = local['accepted'] and fitPassed
                collisionPassed = True
                if 'newFacePairs' in item:
                    added = sorted(intersectionPairs(candidate, raw['faces']) - beforePairs)
                    ac['stageSkinPairSet'] = added == [tuple(pair) for pair in item['newFacePairs']]
                    collisionPassed = not added
                    errors = originalObservationErrors(candidate, binding, cameras, observation, masks)
                    side = [dict(viewIndex=i, changePixels=value['eyes']['meanPixels'] - baselineErrors[i]['eyes']['meanPixels'])
                        for i, value in enumerate(errors) if i != ref and value['eyes']['count'] >= 4]
                    front = [errors[ref][region]['meanPixels'] - previousErrors[ref][region]['meanPixels']
                        for region in ('mouth', 'eyes') if previousErrors[ref][region]['count'] > 0]
                    ac['originalSideMeanGate'] = all(row['changePixels'] <= imageScale for row in side) == item['sideGate']
                    ac['previousStageFrontGate'] = (not front or max(front) <= .05 * imageScale) == item['previousStageGate']
                    ac['sideResidualsReproduced'] = len(side) == len(item['sideMeanErrorChangesPixels']) and all(
                        a['viewIndex'] == b['viewIndex'] and abs(a['changePixels'] - b['changePixels']) < 1e-10
                        for a, b in zip(side, item['sideMeanErrorChangesPixels']))
                    ac['frontResidualsReproduced'] = np.allclose(front, item['frontMeanChanges'], atol=1e-10, rtol=1e-12)
                selectedAttempt = bool(cumulative['accepted'] and localPassed and collisionPassed and item.get('sideGate', False) and item.get('previousStageGate', False))
                ac['selectedAttemptMatchesScale'] = (item['scale'] == stage['scale']) == selectedAttempt
                if selectedAttempt: expectedAfter = candidate
                attempts.append(dict(scale=item['scale'], validationPassed=bool(all(ac.values())), checks=ac, selected=selectedAttempt))
        checks['acceptedStageReconstructed'] = np.array_equal(expectedAfter, after)
        checks['attemptsValidated'] = all(item['validationPassed'] for item in attempts)
        checks['stageStatusConsistent'] = (stage['status'] == 'accepted') == bool(stage['scale'])
        previousAfter = after
        result[name] = dict(validationPassed=bool(all(checks.values())), checks=checks, status=stage['status'],
            comparable=True, geometryChanged=not np.array_equal(before, after), support=supportProof,
            afterMetrics=afterMetrics, attempts=attempts, stageEvidence=e)
    return dict(validationPassed=all(value['validationPassed'] for value in result.values()), eyes=result,
        legacySideAndFrontObservationGates='Original frozen visibility primitive, eligibility, all-view means and front/side decisions independently replayed from original photos and bindings.')


def auditCase(case, runRoot, outputRoot):
    """Read saved four-arm results; expected rejection is a validated outcome."""
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.legacyPreviewAsset import _load
    oldAudit = loadOriginalAudit(); config = readJson(runRoot / 'experimentPlan.json')
    if sha256(runRoot / 'experimentPlan.json') != PLAN_SHA: raise ValueError('Frozen prospective plan changed')
    if case not in config['cases']: raise ValueError('Case outside the frozen seven-group regression split')
    context = readJson(runRoot / case / 'inputContext.json')
    system = oldAudit.priorAudit.independentSystem(case); raw = system['a']; oldRoot = Path(context['oldRoot'])
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    cameras = readJson(Path(context['entry']['reconstruction']) / 'cameras.json')
    inputs = readJson(Path(context['entry']['reconstruction']) / 'inputs.json')
    reference = context['referenceSelection']['referenceViewIndex']; camera = cameras[reference]
    bounded = readArrays(ROOT / 'runs/shapePreservation20261024' / case / 'boundedWeight001/geometry.npz')['vertices']
    initialReports = {'bounded': initialGeometryAudit(system, bounded)}
    sourceDirectory = runRoot / case / 'initial/sourceProtected'
    failurePath = runRoot / case / 'initial/sourceFailure.json'
    if failurePath.is_file():
        failure = readJson(failurePath); preserved = failure.get('accepted') is False and failure.get('noFallback') is True
        source = None; sourceProof = dict(validationPassed=preserved, effectAvailable=False, failure=failure,
            failureSha256=sha256(failurePath), status='unavailableSourceInitial')
        initialReports['sourceProtected'] = dict(accepted=False, producerAcceptedAgrees=preserved, status='unavailableSourceInitial')
    else:
        source = readArrays(sourceDirectory / 'geometry.npz')['vertices']; sourceReport = readJson(sourceDirectory / 'report.json')
        support = supportSpecAudit(system, bounded, source, readEvidence(sourceDirectory / 'supportSpec'), readArrays(sourceDirectory / 'solveDomain.npz'))
        sourceProof = sourceObjectiveAudit(system, context, sourceDirectory, source, support)
        sourceProof['support'] = support
        sourceProof['validationPassed'] &= support['validationPassed']
        initialReports['sourceProtected'] = initialGeometryAudit(system, source)
        initialReports['sourceProtected']['solverAccepted'] = sourceReport['solver']['solverAccepted']
        initialReports['sourceProtected']['accepted'] &= bool(sourceReport['solver']['solverAccepted'] and support['hardProtection']['accepted'])
        initialReports['sourceProtected']['producerAcceptedAgrees'] = initialReports['sourceProtected']['accepted'] == sourceReport['accepted']
        sourceProof['checks']['acceptedSolverSupportConstraints'] = not sourceReport['solver']['solverAccepted'] or support['constraintAccepted']
        sourceProof['validationPassed'] &= sourceProof['checks']['acceptedSolverSupportConstraints']
    anchors = {name: readArrays(path / 'fullGeometry.npz') for name, path in (
        ('originalPureDirect', oldRoot / 'geometry/pureDirect'), ('currentBounded', oldRoot / 'geometry/bounded'),
        ('pairedBaseline', runRoot / case / 'baseline'))}
    pairAnchors = {name: intersectionPairs(value['vertices'], value['faces']) for name, value in anchors.items()}
    sharedAsset = _load(runRoot / case / 'baseline/commonAssembly/legacyAsset.zip')
    sceneIdentity = dict(faces=assets.arrays['faces'], uvs=assets.arrays['uvs'], uvFaces=assets.arrays['uvFaces'],
        materialIds=assets.arrays['materialIds'], nicpToFullIds=raw['skinIds'])
    commonScene = dict(vertices=anchors['pairedBaseline']['vertices'], **sceneIdentity)
    fullFaceLookup = {tuple(face): i for i, face in enumerate(assets.arrays['faces'])}
    uvFaces = assets.arrays['uvFaces'][np.array([fullFaceLookup[tuple(face)] for face in raw['skinIds'][raw['faces']]])]
    branches = {}; stagesByBranch = {}
    for branch, options in config['branches'].items():
        directory = runRoot / case / branch; report = readJson(directory / 'report.json')
        if options['sourcePrior'] and source is None:
            preserved = report.get('status') == 'blockedByInitialException' and report.get('geometrySafetyAccepted') is False
            branches[branch] = dict(validationPassed=preserved, geometrySafetyAccepted=False, requestedEffectsAccepted=False,
                geometryExperimentAccepted=False, status='blockedByInitialException', reportSha256=sha256(directory / 'report.json'), clientReady=False)
            continue
        states = readArrays(directory / 'states.npz'); final = readArrays(directory / 'finalGeometry.npz')
        full = readArrays(directory / 'fullGeometry.npz'); storedInitial = readArrays(directory / 'initial.npz')
        expectedInitial = source if options['sourcePrior'] else bounded
        initialGate = initialReports['sourceProtected' if options['sourcePrior'] else 'bounded']
        checks = dict(planIdentity=context['planSha256'] == PLAN_SHA, caseAndBranch=report['caseId'] == case and report['branch'] == branch,
            frozenFactors=report['options'] == options, inputHashesIdentical=report['sourceHashes'] == context['sourceHashes'],
            referenceUnchanged=report['referenceViewIndex'] == reference,
            initialExactlyExpected=np.array_equal(storedInitial['vertices'], expectedInitial) and np.array_equal(states['initial'], expectedInitial),
            finalExactlyState=np.array_equal(final['vertices'], states['final']), skinFacesOriginal=np.array_equal(final['faces'], raw['faces']),
            fullFacesOriginal=np.array_equal(full['faces'], raw['fullFaces']), uvsOriginal=np.array_equal(final['uvs'], assets.arrays['uvs']),
            skinUvFacesOriginal=np.array_equal(final['uvFaces'], uvFaces),
            initialDecisionReproduced=initialGate['accepted'] == report['initialAccepted'],
            notClientPromoted=report['finalCandidateAccepted'] is False and report['finalAdaptiveAssemblyValidated'] is False,
            appearanceNotRebaked=report['textureRebaked'] is False)
        outputs = fileIdentityAudit({str(directory / key): value for key, value in report['outputHashes'].items()})
        checks['producerOutputHashes'] = outputs['accepted']
        attachments = fixedAttachmentAudit(commonScene, dict(vertices=full['vertices'], **sceneIdentity), final['vertices'], staticSkinIds=raw['staticIds'])
        checks['allNonSkinExactlyCommonBaseline'] = attachments['accepted'] and report['allNonSkinExactlyFixed']
        archivePath = Path(report['asset']['archivePath']); archive = _load(archivePath)
        canonical = archive.mesh['vertices'] * 100 @ raw['rotation'] * raw['scale'] + raw['translation']
        checks['archiveRoundtrip'] = np.max(np.abs(canonical - full['vertices'])) < 1e-12
        checks['archiveHash'] = sha256(archivePath) == report['asset']['sha256']
        checks['sharedTextureAndPayload'] = all(archive.payloads[key] == value for key, value in sharedAsset.payloads.items() if key != 'face_nicp.obj')
        checks['sharedBindingMetadata'] = archive.manifest['binding'] == sharedAsset.manifest['binding']
        flags = ('vertexOrderPreserved', 'faceOrderPreserved', 'uvOrderPreserved', 'uvFaceOrderPreserved', 'materialFaceOrderPreserved')
        checks['originalBindingFlags'] = all(archive.manifest['binding'][key] for key in flags) and not archive.manifest['binding']['rigGenerated']
        for key in ('faces', 'uvFaces', 'materialIds'): checks['archive' + key + 'Original'] = np.array_equal(archive.mesh[key], assets.arrays[key])
        checks['archiveUvOriginal'] = np.max(np.abs(archive.mesh['uvs'] - assets.arrays['uvs'])) < 1e-15
        checks['archiveStaticOriginal'] = np.max(np.abs(archive.mesh['vertices'][raw['skinIds'][raw['staticIds']]] * 100 - assets.arrays['vertices'][raw['skinIds'][raw['staticIds']]])) < 1e-12
        downstream = downstreamGeometryAudit(system, expectedInitial, final['vertices'], camera)
        nextPairs = intersectionPairs(full['vertices'], full['faces'])
        collisions = {name: intersectionSetAudit(pairs, nextPairs, assets.arrays['materialIds']) for name, pairs in pairAnchors.items()}
        for name, key in (('originalPureDirect', 'newFullPairsAgainstOriginalPureDirect'), ('currentBounded', 'newFullPairsAgainstCurrentBounded'), ('pairedBaseline', 'newFullPairsAgainstPairedBaseline')):
            checks[name + 'FullPairSetReproduced'] = collisions[name]['newPairs'] == [tuple(pair) for pair in report[key]]
        stageAudit = auditSemanticStages(directory, context, raw, cameras, inputs, options, config)
        stagesByBranch[branch] = stageAudit; checks['semanticStageEvidenceValidated'] = stageAudit['validationPassed']
        paired = None
        if options['localEyes']:
            paired = {}; partner = stagesByBranch[GLOBAL_EYE_PAIRS[branch]]
            for name, value in stageAudit['eyes'].items():
                other = partner['eyes'][name]
                if not value['comparable'] or not other['comparable']:
                    paired[name] = dict(accepted=False, reason='missingComparableStageEvidence'); continue
                le, ge = value['stageEvidence'], other['stageEvidence']
                same = all(np.array_equal(le[key], ge[key]) for key in ('ibugIds', 'vertexIndices', 'barycentricWeights', 'targetPixels'))
                if not same:
                    paired[name] = dict(accepted=False, reason='pairedStageTargetsChanged'); continue
                paired[name] = pairedSemanticAlignmentAudit(value['afterMetrics'], other['afterMetrics'],
                    tolerancePixels=config['localEyes']['pairedMeanErrorToleranceImageScale'] * min(inputs[reference]['workingSize']) / 512.)
            pairAccepted = all(value['accepted'] for value in paired.values())
            checks['pairedAlignmentDecisionReproduced'] = pairAccepted == report['pairedSemanticAlignment']['accepted']
            for name, value in paired.items():
                saved = report['pairedSemanticAlignment']['eyes'][name]
                checks[name + 'PairedDecisionReproduced'] = saved['accepted'] == value['accepted']
                if 'meanChangePixels' in value:
                    checks[name + 'PairedMeanReproduced'] = abs(saved['meanChangePixels'] - value['meanChangePixels']) < 1e-10
        else: pairAccepted = True
        safety = bool(initialGate['accepted'] and downstream['accepted'] and all(value['accepted'] for value in collisions.values()) and pairAccepted)
        checks['producerSafetyDecisionReproduced'] = safety == report['geometrySafetyAccepted']
        noNew = bool(downstream['accepted'] and collisions['pairedBaseline']['accepted'])
        checks['pairedNoRegressionDecisionReproduced'] = noNew == report['noNewRegressionAgainstPairedBaseline']
        effect = bool((not options['sourcePrior'] or sourceProof['effectAvailable']) and pairAccepted)
        # Dense aperture is deliberately NOT a replacement for the prospectively
        # fixed semantic-pair gate; it remains a separate diagnostic renderer.
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks, geometrySafetyAccepted=safety,
            requestedEffectsAccepted=effect, geometryExperimentAccepted=bool(safety and effect and all(checks.values()) and sourceProof['validationPassed']),
            noNewRegressionAgainstPairedBaseline=noNew, attachments=attachments, downstream=downstream,
            fullCollisionAnchors=collisions, semanticStages=stageAudit, pairedSemanticAlignment=paired,
            producerOutputs=outputs, reportSha256=sha256(directory / 'report.json'), assetSha256=sha256(archivePath), clientReady=False)
    # Evidence needed for later paired arms must remain available until their
    # measurement completes, then omit dense duplicate arrays from final JSON.
    for stageAudit in stagesByBranch.values():
        for value in stageAudit['eyes'].values(): value.pop('stageEvidence', None)
    baselineError = float(np.max(np.abs(anchors['currentBounded']['vertices'] - anchors['pairedBaseline']['vertices'])))
    identities = fileIdentityAudit(context['sourceHashes'])
    complete = bool(identities['accepted'] and sourceProof['validationPassed'] and initialReports['sourceProtected']['producerAcceptedAgrees']
        and baselineError < 1e-12 and all(value['validationPassed'] for value in branches.values()))
    result = dict(caseId=case, validationPassed=complete, initial=initialReports, sourceEvidence=sourceProof,
        originalInputIdentities=identities, baselineFullReplayMaximumError=baselineError, branches=branches,
        planSha256=PLAN_SHA, validatorSha256=sha256(__file__), clientReady=False, noCandidateSolvesPerformed=True,
        interpretation='Same baseline attachments isolate skin changes; passing these controls does not validate adaptive final assembly or anatomical likeness.')
    outputRoot.mkdir(parents=True, exist_ok=True); writeJson(outputRoot / f'{case}.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/protectedFeature20261024')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/protectedFeature20261024/independentAudit')
    args = parser.parse_args(); results = []; failures = []
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.case:
        try:
            result = auditCase(case, args.run_root, args.output); results.append(result)
            print(json.dumps(dict(caseId=case, validationPassed=result['validationPassed'],
                geometryAccepted=[k for k, v in result['branches'].items() if v['geometrySafetyAccepted']])), flush=True)
        except Exception as error:
            failure = dict(caseId=case, validationPassed=False, toolException=True, exceptionType=type(error).__name__, message=str(error))
            import traceback
            failure['traceback'] = traceback.format_exc(); failures.append(failure)
            writeJson(args.output / f'{case}.toolException.json', failure); print(json.dumps(failure), flush=True)
    counts = {branch: dict(evidenceValidated=sum(r['branches'][branch]['validationPassed'] for r in results),
        geometrySafetyAccepted=sum(r['branches'][branch]['geometrySafetyAccepted'] for r in results),
        requestedEffectsAccepted=sum(r['branches'][branch]['requestedEffectsAccepted'] for r in results),
        geometryExperimentAccepted=sum(r['branches'][branch]['geometryExperimentAccepted'] for r in results)) for branch in BRANCHES}
    summary = dict(validationPassed=not failures and all(r['validationPassed'] for r in results),
        caseCount=len(results), counts=counts, toolExceptionCount=len(failures), failures=failures,
        cases=[dict(caseId=r['caseId'], validationPassed=r['validationPassed']) for r in results],
        validatorSha256=sha256(__file__), planSha256=PLAN_SHA, clientReady=False)
    writeJson(args.output / 'validationSummary.json', summary); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
