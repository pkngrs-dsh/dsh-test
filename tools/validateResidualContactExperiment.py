"""Independent read-only gates for coarse-preserving visible-eye residuals.

This validator never fits, reassembles, rebinds photo targets, or edits a saved
candidate. Contact-plane proxies remain separate from complete-mesh collision
checks and actual full-aperture measurements.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from validateFeatureGeometryExperiment import apertureMetricAudit, intersectionSetAudit
from validateProtectedFeatureExperiment import (
    semanticTargetErrors, sceneIdentityAudit, originalObservationMasks, originalObservationErrors,
)

BRANCHES = ('baseline', 'residualOnly', 'residualWithContact')
EYES = ('imageLeft', 'imageRight')
COARSE_EYES = ('imageLeftEye', 'imageRightEye')


def coarseEyePreservationAudit(baselineVertices, candidateVertices, constraints, camera, *, imageScale):
    """Preserve completed coarse alignment, measured at its FINAL baseline state.

    Constraints must already have been independently identified as the original
    semantic binding and photo observations. Every proposal compares to this
    same final baseline, never a preceding residual iteration or an old stage's
    pre-jaw geometry. Both mean and maximum errors retain the original .05 px
    working-scale tolerance.
    """
    if isinstance(constraints, (list, tuple)):
        if len(constraints) != 2: raise ValueError('Exactly two original semantic eye groups are required')
        constraints = {entry['name']: entry for entry in constraints}
    if not np.isfinite(imageScale) or imageScale <= 0 or set(constraints) != set(COARSE_EYES):
        raise ValueError('Both original eyes and a positive working-image scale are required')
    reports = {}
    for name in COARSE_EYES:
        entry = constraints[name]
        indices, weights, targets = (np.asarray(entry[key]) for key in ('vertexIndices', 'barycentric', 'targetPixels'))
        if len(indices) < 4: raise ValueError('At least four original eligible points are required per eye')
        before = semanticTargetErrors(baselineVertices, indices, weights, targets, camera)
        after = semanticTargetErrors(candidateVertices, indices, weights, targets, camera)
        tolerance = .05 * imageScale
        checks = dict(meanNotWorse=after['meanPixels'] <= before['meanPixels'] + tolerance,
            maximumNotWorse=after['maximumPixels'] <= before['maximumPixels'] + tolerance)
        reports[name] = dict(accepted=bool(all(checks.values())), checks=checks, baseline=before, candidate=after,
            tolerancePixels=tolerance, meanChangePixels=after['meanPixels'] - before['meanPixels'],
            maximumChangePixels=after['maximumPixels'] - before['maximumPixels'])
    return dict(accepted=all(value['accepted'] for value in reports.values()), eyes=reports,
        anchor='Exact final coarse-aligned baseline, shared by every residual round and branch.',
        anatomicalTruthEstablished=False)


def frozenAssemblyAudit(baselineScene, candidateScene, movableMask, *, staticFullIds=()):
    """Pin every accessory, including material-zero canthus pieces, and remote skin."""
    before, after = np.asarray(baselineScene['vertices']), np.asarray(candidateScene['vertices'])
    skinIds = np.asarray(baselineScene['nicpToFullIds']); movable = np.asarray(movableMask)
    if (movable.shape != (len(before),) or movable.dtype.kind != 'b' or before.shape != after.shape
            or skinIds.dtype.kind not in 'iu' or len(np.unique(skinIds)) != len(skinIds)):
        raise ValueError('Expected a frozen boolean full-mesh domain and original unique skin mapping')
    skin = np.zeros(len(before), bool); skin[skinIds] = True
    if (movable & ~skin).any(): raise ValueError('No accessory may enter the residual movable domain')
    identity = sceneIdentityAudit(baselineScene, candidateScene, fixedVertexIds=np.flatnonzero(~movable),
        protectedVertexIds=np.asarray(staticFullIds, dtype=np.int64))
    identity['checks']['allNonSkinExactlyFixed'] = np.array_equal(before[~skin], after[~skin])
    identity['accepted'] = bool(all(identity['checks'].values()))
    identity['nonSkinIds'] = np.flatnonzero(~skin)
    identity['scope'] = 'Existing complete baseline assembly; no fresh assembly, eye relocation or canthus deformation.'
    return identity


def legacyResidualObservationAudit(originalInitial, completedBaseline, candidate, faces, binding,
                                   cameras, inputs, observations, referenceView, *, masks=None):
    """Keep original side-view budget and the new stage's front-view budget separate."""
    if masks is None:
        masks = originalObservationMasks(originalInitial, faces, binding, cameras, inputs, observations)
    original = originalObservationErrors(originalInitial, binding, cameras, observations, masks)
    baseline = originalObservationErrors(completedBaseline, binding, cameras, observations, masks)
    after = originalObservationErrors(candidate, binding, cameras, observations, masks)
    scale = min(inputs[referenceView]['workingSize']) / 512.
    front = []
    for group in ('mouth', 'eyes'):
        if baseline[referenceView][group]['count']:
            change = after[referenceView][group]['meanPixels'] - baseline[referenceView][group]['meanPixels']
            front.append(dict(group=group, changePixels=change, accepted=bool(change <= .05 * scale)))
    side = []
    for view, groups in enumerate(after):
        if view == referenceView or groups['eyes']['count'] < 4: continue
        difference = groups['eyes']['meanPixels'] - original[view]['eyes']['meanPixels']
        inherited = baseline[view]['eyes']['meanPixels'] - original[view]['eyes']['meanPixels']
        side.append(dict(viewIndex=view, count=groups['eyes']['count'], changeAgainstOriginalInitialPixels=difference,
            changeAgainstCompletedBaselinePixels=groups['eyes']['meanPixels'] - baseline[view]['eyes']['meanPixels'],
            baselineAlreadyExceedsOriginalBudget=bool(inherited > scale), accepted=bool(difference <= scale)))
    return dict(accepted=all(row['accepted'] for row in front + side), front=front, side=side,
        frontTolerancePixels=.05 * scale, originalSideTolerancePixels=scale,
        initialAnchorNotReset=True, sideCompletedBaselineChangeIsDiagnosticOnly=True)


def historicalCollisionAudit(baselinePairs, candidatePairs, anchors, materialIds=None):
    """Separate inherited historical failure from newly introduced pair identities."""
    if set(anchors) != {'originalPureDirect', 'currentBounded', 'completedCoarseBaseline'}:
        raise ValueError('All three original complete-mesh anchors are mandatory')
    base, candidate = set(map(tuple, baselinePairs)), set(map(tuple, candidatePairs))
    if set(map(tuple, anchors['completedCoarseBaseline'])) != base:
        raise ValueError('The local anchor must be the exact completed coarse baseline')
    reports = {}
    for name, pairs in anchors.items():
        old = set(map(tuple, pairs)); result = intersectionSetAudit(old, candidate, materialIds)
        inherited = base - old
        result.update(baselineAlreadyFailed=bool(inherited), baselineHistoricalNewPairs=sorted(inherited),
            inheritedPairsStillPresent=sorted(inherited & candidate), repairedHistoricalPairs=sorted(inherited - candidate),
            newlyIntroducedSinceCompletedBaseline=sorted(candidate - base))
        reports[name] = result
    return dict(incrementalNoNewPairs=not (candidate - base),
        allHistoricalAnchorsAccepted=all(value['accepted'] for value in reports.values()), anchors=reports,
        scope='Complete original full-face IDs, with eyes, oral parts, alpha cards and canthi included.')


def fullApertureRasterAudit(baselineMeasurement, candidateMeasurement, photoEyes, *, originalPixelScale):
    """Recompute mask scores on the same full visible aperture raster, without fitting.

    A fragmented baseline mask may supply a meaningful error. Empty/missing
    masks cannot define a benefit; final extraction must accept both complete
    eyes. Texture-only contour proxies and completed synthetic curves are not
    interchangeable with these masks.
    """
    import cv2
    if not np.isfinite(originalPixelScale) or originalPixelScale <= 0 or set(photoEyes) != set(EYES):
        raise ValueError('Expected both photo eyes and a positive original-pixel scale')
    checks = {}; metrics = {'baseline': {}, 'candidate': {}}; failures = []
    for name in EYES:
        old = baselineMeasurement.get('eyes', {}).get(name, {}); new = candidateMeasurement.get('eyes', {}).get(name, {})
        if any('roi' not in eye or 'roiMasks' not in eye for eye in (old, new)):
            failures.append(name + ':missingCompleteApertureMask'); continue
        beforeMask = np.asarray(old['roiMasks']['eye']); afterMask = np.asarray(new['roiMasks']['eye'])
        if beforeMask.ndim != 2 or afterMask.ndim != 2 or beforeMask.dtype.kind != 'b' or afterMask.dtype.kind != 'b':
            raise ValueError('Complete aperture masks must be boolean H by W arrays')
        same = beforeMask.shape == afterMask.shape and all(np.array_equal(old['roi'][key], new['roi'][key]) for key in ('origin', 'scale', 'shape'))
        checks[name + 'SameRaster'] = same
        if not same:
            failures.append(name + ':changedRasterIdentity'); continue
        roi = old['roi']; scale = float(roi['scale']); origin = np.asarray(roi['origin'])
        if not np.isfinite(scale) or scale <= 0 or origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError('ROI origin and scale must be finite and valid')
        parts = []
        for lid in ('upper', 'lower'):
            curve = np.asarray(photoEyes[name][lid], float)
            if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 4 or not np.isfinite(curve).all():
                raise ValueError('Original photo eyes need finite dense curves')
            parts.append(curve[::-1] if curve[0, 0] > curve[-1, 0] else curve)
        polygon = (np.r_[parts[0], parts[1][::-1]] - origin) * scale
        if (polygon < 0).any() or (polygon > np.asarray(beforeMask.shape[::-1]) - 1).any():
            failures.append(name + ':clippedOriginalPhotoPolygon'); continue
        photo = np.zeros(beforeMask.shape, np.uint8)
        cv2.fillPoly(photo, [np.rint(polygon * 256).astype(np.int32)], 1, shift=8); photo = photo.astype(bool)
        if not photo.any() or not beforeMask.any() or not afterMask.any():
            failures.append(name + ':undefinedEmptyMaskBenefit'); continue
        targetBoundary = photo & ~binary_erosion(photo, structure=np.ones((3, 3)), border_value=0)
        for which, mask, measurement in (('baseline', beforeMask, old), ('candidate', afterMask, new)):
            boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
            unit = 1 / (scale * originalPixelScale)
            forward = distance_transform_edt(~targetBoundary)[boundary] * unit
            backward = distance_transform_edt(~boundary)[targetBoundary] * unit
            componentCount, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            areas = sorted((int(area) for area in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
            metrics[which][name] = dict(symmetricBoundaryMeanOriginalPixels=.5*(float(forward.mean())+float(backward.mean())),
                symmetricBoundaryP95OriginalPixels=float(np.percentile(np.r_[forward, backward], 95)),
                intersectionOverUnion=float((mask & photo).sum() / (mask | photo).sum()),
                visibleComponentCount=int(componentCount - 1), secondaryComponentAreaFraction=sum(areas[1:])/sum(areas),
                extractionStatus=measurement['status'], extractionReasons=measurement['reasons'],
                modelRasterArea=int(mask.sum()), photoRasterArea=int(photo.sum()))
        checks[name + 'FinalCompleteExtractionAccepted'] = new['status'] == 'accepted'
    if failures:
        return dict(accepted=False, effectDefined=False, checks=checks, reasons=failures, metrics=metrics)
    scores = {key: dict(eyes=value, meanSymmetricBoundaryDistanceOriginalPixels=float(np.mean([
        value[name]['symmetricBoundaryMeanOriginalPixels'] for name in EYES]))) for key, value in metrics.items()}
    benefit = apertureMetricAudit(scores['baseline'], scores['candidate'], originalPixelScale=originalPixelScale)
    return dict(accepted=bool(all(checks.values()) and benefit['accepted']), effectDefined=True,
        checks=checks, reasons=[], metrics=scores, originalApertureGate=benefit,
        baselineContourRejectionDoesNotEraseValidMask=True)


def frozenContactGapAudit(baselineVertices, candidateVertices, sourceVertexIds, targetVertexIds,
                         targetBarycentric, referenceNormals, minimumGap, *, lengthScale, tolerance):
    """Evaluate a fixed oriented point/triangle proxy with both surfaces movable.

    Closest-triangle choice, barycentrics and orientation provenance must be
    authenticated separately. Negative baseline gaps are inherited, never
    relabeled as valid anatomical contact. This is not a signed mesh distance.
    """
    baseline = np.asarray(baselineVertices); candidate = np.asarray(candidateVertices)
    source = np.asarray(sourceVertexIds); target = np.asarray(targetVertexIds)
    bary = np.asarray(targetBarycentric); normals = np.asarray(referenceNormals)
    m = len(source)
    if (baseline.ndim != 2 or baseline.shape[1] != 3 or candidate.shape != baseline.shape
            or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
            or source.shape != (m,) or source.dtype.kind not in 'iu'
            or target.shape != (m, 3) or target.dtype.kind not in 'iu'
            or bary.shape != (m, 3) or normals.shape != (m, 3)
            or not np.isfinite(bary).all() or not np.isfinite(normals).all()
            or np.any(source < 0) or np.any(source >= len(baseline)) or np.any(target < 0) or np.any(target >= len(baseline))
            or np.any(bary < -1e-12) or np.max(np.abs(bary.sum(1) - 1), initial=0) > 1e-12
            or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12, rtol=0)
            or not np.isfinite(lengthScale) or lengthScale <= 0 or not np.isfinite(tolerance) or tolerance < 0
            or not np.isfinite(minimumGap)):
        raise ValueError('Expected fixed finite contact bindings, unit reference normals and positive physical scale')
    def gaps(vertices):
        points = (vertices[target] * bary[..., None]).sum(1)
        return np.einsum('ij,ij->i', vertices[source] - points, normals)
    before, after = gaps(baseline), gaps(candidate)
    lower = np.minimum(before, minimumGap)
    violation = (after - lower) / lengthScale < -tolerance
    return dict(accepted=bool(not violation.any()), baselineGap=before, candidateGap=after, lowerBound=lower,
        violatedRowIds=np.flatnonzero(violation), inheritedNegativeRowIds=np.flatnonzero(before < 0),
        remainingNegativeRowIds=np.flatnonzero(after < 0), rowCount=m, emptyContactDomain=m == 0,
        scope='Frozen oriented contact-plane proxy, with both source and target positions updated; full collision checks remain required.')


def _closestFeatureIndependent(point, triangles):
    """Enumerate orthogonal plane projections and closed segments.

    Barycentrics use a dominant-axis 2D projection, independently of the
    producer's 3D cross-product formula. Every target triangle is considered.
    """
    triangles = np.asarray(triangles, float); a = triangles[:, 0]
    u, v = triangles[:, 1] - a, triangles[:, 2] - a
    normal = np.cross(u, v); norm = np.linalg.norm(normal, axis=1)
    if np.any(norm <= 0) or not np.isfinite(norm).all(): raise ValueError('Contact targets must have nondegenerate finite triangles')
    normal /= norm[:, None]
    projected = point - np.einsum('ij,ij->i', point - a, normal)[:, None] * normal
    bary = np.zeros((len(triangles), 3))
    for dropped in range(3):
        selected = np.argmax(np.abs(normal), axis=1) == dropped
        axes = [index for index in range(3) if index != dropped]
        x, y = u[selected][:, axes], v[selected][:, axes]; relative = (projected - a)[selected][:, axes]
        det = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        beta = (relative[:, 0] * y[:, 1] - relative[:, 1] * y[:, 0]) / det
        gamma = (x[:, 0] * relative[:, 1] - x[:, 1] * relative[:, 0]) / det
        bary[selected] = np.c_[1 - beta - gamma, beta, gamma]
    distances = [np.where((bary >= 0).all(1), np.linalg.norm(projected - point, axis=1), np.inf)]
    candidates = [bary]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        segment = triangles[:, second] - triangles[:, first]
        fraction = np.clip(np.einsum('ij,ij->i', point - triangles[:, first], segment) / np.einsum('ij,ij->i', segment, segment), 0, 1)
        feature = np.zeros((len(triangles), 3)); feature[:, first] = 1 - fraction; feature[:, second] = fraction
        candidates.append(feature)
        distances.append(np.linalg.norm(triangles[:, first] + fraction[:, None] * segment - point, axis=1))
    distances = np.asarray(distances); best = distances.argmin(0); ids = np.arange(len(triangles))
    return np.asarray(candidates)[best, ids], distances[best, ids], normal


def contactSpecAudit(spec, baselineVertices, faces, expectedGroups, policy, *, savedOperator):
    """Authenticate the frozen oriented proxy without invoking its builder.

    The caller independently obtains expected semantic groups and orientation
    provenance from the frozen template plan. A numerically tied closest face
    may have a different valid minimizer; it must still be globally closest,
    with its own real winding normal and barycentric feature retained.
    """
    from scipy import sparse
    baseline = np.asarray(baselineVertices); faces = np.asarray(faces)
    scale = float(policy['lengthScale'])
    tolerance = max(scale * 1e-10, np.finfo(float).eps * max(float(np.abs(baseline).max()), scale) * 256)
    checks = dict(exactBaseline=np.array_equal(spec['baselineVertices'], baseline),
        exactFaces=np.array_equal(spec['faces'], faces), groupCount=len(spec['contactGroups']) == len(expectedGroups),
        completeSelectionEvidence=len(spec['selectionEvidence']) == len(expectedGroups),
        noSignChosenFromBaselineGap=spec['diagnostics']['orientationFromBaselineGap'] is False,
        proxyNotCollisionCertificate=spec['diagnostics']['isCollisionCertificate'] is False,
        normalsAndBarycentricFrozen=spec['diagnostics']['normalAndBarycentricFrozen'] is True)
    for key in ('lengthScale', 'searchRadius', 'activationDistance', 'minimumGap', 'weight'):
        checks[key + 'Frozen'] = spec[key] == policy[key]
    if not checks['groupCount'] or not checks['completeSelectionEvidence']:
        return dict(validationPassed=False, checks=checks, reasons=['missingOrAdditionalContactGroups'])
    sources, selectedFaces, selectedBary, selectedNormals, groupIds = [], [], [], [], []
    selections = []; spans = {}
    for groupIndex, (group, saved, expected) in enumerate(zip(spec['contactGroups'], spec['selectionEvidence'], expectedGroups)):
        sourceIds, faceIds = np.sort(expected['sourceVertexIds']), np.sort(expected['targetFaceIds'])
        gc = dict(name=group['name'] == expected['name'], sourceIds=np.array_equal(group['sourceVertexIds'], sourceIds),
            targetFaceIds=np.array_equal(group['targetFaceIds'], faceIds),
            explicitTemplateOrientation=group['normalSign'] == expected['normalSign'] and group['orientationProvenance'] == expected['orientationProvenance'],
            selectionSources=np.array_equal(saved['sourceVertexIds'], sourceIds))
        start = len(sources); selectedTargets = faces[faceIds]; perSource = []
        for index, sourceId in enumerate(sourceIds):
            keep = ~np.any(selectedTargets == sourceId, axis=1); eligibleIds = faceIds[keep]
            sc = dict(incidentExclusionCount=saved['sourceIncidentExcludedCounts'][index] == int((~keep).sum()),
                hasCandidateMask=bool(saved['hasCandidateMask'][index]) == bool(len(eligibleIds)))
            if not len(eligibleIds):
                sc['noInventedTarget'] = saved['nearestTargetFaceIds'][index] == -1 and not saved['activeMask'][index]
                perSource.append(dict(sourceVertexId=int(sourceId), checks=sc, validationPassed=bool(all(sc.values())))); continue
            independentBary, distances, rawNormals = _closestFeatureIndependent(baseline[sourceId], baseline[faces[eligibleIds]])
            chosen = int(saved['nearestTargetFaceIds'][index]); locations = np.flatnonzero(eligibleIds == chosen)
            sc['chosenActualEligibleFace'] = len(locations) == 1
            if not len(locations):
                perSource.append(dict(sourceVertexId=int(sourceId), checks=sc, validationPassed=False)); continue
            location = int(locations[0]); bary = np.asarray(saved['nearestBarycentric'][index])
            normal = expected['normalSign'] * rawNormals[location]
            point = bary @ baseline[faces[chosen]]; distance = float(np.linalg.norm(baseline[sourceId] - point))
            gap = float(normal @ (baseline[sourceId] - point))
            sc['validBarycentricFeature'] = (bary >= -1e-12).all() and abs(float(bary.sum()) - 1) <= 1e-12
            sc['globalClosestFeature'] = abs(distance - float(distances.min())) <= tolerance
            sc['closestOnChosenTriangle'] = abs(distance - float(distances[location])) <= tolerance
            sc['normalFromFrozenWinding'] = np.allclose(saved['nearestNormals'][index], normal, atol=1e-12, rtol=1e-12)
            sc['baselineDistance'] = abs(saved['baselineDistances'][index] - distance) <= tolerance
            sc['baselineSignedGap'] = abs(saved['baselineSignedGaps'][index] - gap) <= tolerance
            within = distance <= policy['searchRadius']; active = within and gap <= policy['activationDistance']
            sc['searchRadiusSelection'] = bool(saved['withinSearchRadiusMask'][index]) == within
            sc['activationSelection'] = bool(saved['activeMask'][index]) == active
            if active:
                sources.append(int(sourceId)); selectedFaces.append(chosen); selectedBary.append(bary); selectedNormals.append(normal); groupIds.append(groupIndex)
            perSource.append(dict(sourceVertexId=int(sourceId), checks=sc, validationPassed=bool(all(sc.values()))))
        spans[group['name']] = dict(start=start, stop=len(sources))
        selections.append(dict(name=group['name'], checks=gc, validationPassed=bool(all(gc.values()) and all(row['validationPassed'] for row in perSource)), sourceRows=perSource))
    sourceArray = np.asarray(sources, np.int64); faceArray = np.asarray(selectedFaces, np.int64)
    baryArray = np.asarray(selectedBary).reshape(-1, 3); normalArray = np.asarray(selectedNormals).reshape(-1, 3)
    expectedArrays = dict(sourceVertexIds=sourceArray, targetFaceIds=faceArray, targetVertexIds=faces[faceArray],
        rowGroupIds=np.asarray(groupIds, np.int64))
    for key, value in expectedArrays.items(): checks[key + 'Reconstructed'] = np.array_equal(spec[key], value)
    for key, value in (('targetBarycentric', baryArray), ('normals', normalArray)):
        checks[key + 'Reconstructed'] = np.asarray(spec[key]).shape == value.shape and np.allclose(spec[key], value, atol=1e-12, rtol=1e-12)
    checks['groupSpansReconstructed'] = spec['groups'] == spans
    rows, columns, coefficients = [], [], []
    for row, (sourceId, faceId, bary, normal) in enumerate(zip(sourceArray, faceArray, baryArray, normalArray)):
        for vertexId, weight in zip(np.r_[sourceId, faces[faceId]], np.r_[1., -bary]):
            rows.extend([row] * 3); columns.extend((3 * vertexId + np.arange(3)).tolist()); coefficients.extend((weight * normal).tolist())
    operator = sparse.csr_matrix((coefficients, (rows, columns)), shape=(len(sourceArray), baseline.size))
    difference = (operator - savedOperator).tocsr() if operator.shape == savedOperator.shape else None
    checks['fullBothSurfaceOperator'] = difference is not None and np.max(np.abs(difference.data), initial=0) <= 1e-12
    gaps = operator @ baseline.ravel(); lower = np.minimum(gaps, policy['minimumGap'])
    checks['baselineGapsReconstructed'] = np.asarray(spec['baselineGaps']).shape == gaps.shape and np.allclose(spec['baselineGaps'], gaps, atol=tolerance, rtol=0)
    checks['grandfatherBoundsReconstructed'] = np.asarray(spec['lowerGaps']).shape == lower.shape and np.allclose(spec['lowerGaps'], lower, atol=tolerance, rtol=0)
    checks['activeCount'] = spec['diagnostics']['activeRowCount'] == len(sourceArray)
    return dict(validationPassed=bool(all(checks.values()) and all(row['validationPassed'] for row in selections)),
        checks=checks, groups=selections, reconstructedOperator=operator, numericalGapTolerance=tolerance,
        scope='Every frozen nearest feature enumerated independently; template semantic/sign provenance supplied separately, never inferred from candidate gaps.')


# The following CLI consumes immutable artifacts only. Raster visibility and
# triangle intersection predicates are shared primitives; domain construction,
# contact operators, decision aggregation and mask metrics are reconstructed.
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from validateFeatureGeometryExperiment import (
    readJson, readArrays, readEvidence, sha256, fileIdentityAudit, loadOriginalAudit,
    downstreamGeometryAudit, initialGeometryAudit, _contourMasksAgree,
)
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs

PLAN_SHA = '11d68f860b22b612ccdfde8e5a1d75d0f4355901c406ed236c80a647f4105aa0'


def sameEvidence(first, second):
    if isinstance(first, dict) or isinstance(second, dict):
        if not isinstance(first, dict) or not isinstance(second, dict): return False
        # saveEvidence serializes mapping keys as JSON strings, including the
        # original integer alpha-texture material IDs. No values are coerced.
        left = {str(k): v for k, v in first.items()}; right = {str(k): v for k, v in second.items()}
        return (len(left) == len(first) and len(right) == len(second) and set(left) == set(right)
                and all(sameEvidence(left[key], right[key]) for key in left))
    if isinstance(first, (list, tuple, np.ndarray)) or isinstance(second, (list, tuple, np.ndarray)):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            a, b = np.asarray(first), np.asarray(second)
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return len(first) == len(second) and all(sameEvidence(a, b) for a, b in zip(first, second))
    return first == second


def independentEyeSupport(scene, camera, photoEyes, measurement, staticFullIds, fraction):
    """Derive visible seeds and truncated geodesics from the frozen baseline."""
    vertices, faces = scene['vertices'], scene['faces']; count = len(vertices)
    skin = np.zeros(count, bool); skin[scene['nicpToFullIds']] = True
    skinFaces = skin[faces].all(1)
    tri = faces[skinFaces]
    edges = np.unique(np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    graph = sparse.csr_matrix((np.r_[lengths, lengths],
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(count, count))
    depth = vertices @ np.asarray(camera['rotation'])[2] + camera['translation'][2]
    movable = np.zeros(count, bool); reports = {}
    for name in EYES:
        masks = measurement.get('eyes', {}).get(name, {}).get('roiMasks')
        if masks is None:
            reports[name] = dict(status='noVisibleRoi', seedVertexIds=np.empty(0, np.int64)); continue
        aperture, bindable, hits = (np.asarray(masks[k]) for k in ('eye', 'eligibleBindableSkin', 'faceIds'))
        if not bindable.any():
            reports[name] = dict(status='noBindableVisibleSkin', seedVertexIds=np.empty(0, np.int64)); continue
        boundary = aperture & ~binary_erosion(aperture, structure=np.ones((3, 3)), border_value=0)
        distance, nearest = distance_transform_edt(~bindable, return_indices=True)
        yy, xx = np.nonzero(boundary & (distance <= 2.6))
        faceIds = hits[nearest[0, yy, xx], nearest[1, yy, xx]]
        faceIds = faceIds[(faceIds >= 0) & (faceIds < len(faces))]
        faceIds = faceIds[skinFaces[faceIds]]; seeds = np.unique(faces[faceIds])
        if len(seeds) < 3 or not (depth[seeds] > 0).all():
            reports[name] = dict(status='insufficientVisibleSkinSeeds', seedVertexIds=seeds); continue
        upper, lower = (np.asarray(photoEyes[name][key], np.float64) for key in ('upper', 'lower'))
        endpoints = (np.asarray([upper[0], upper[-1]]) + np.asarray([lower[0], lower[-1]])) / 2
        rays = np.linalg.solve(camera['intrinsics'], np.c_[endpoints, np.ones(2)].T).T
        rays /= rays[:, 2, None]; localDepth = float(np.median(depth[seeds]))
        width = np.linalg.norm(rays[1] - rays[0]) * localDepth; radius = width * fraction
        distances = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=radius)
        selected = skin & (distances <= radius); movable |= selected
        reports[name] = dict(status='supported', seedVertexIds=seeds, skinFaceIds=np.unique(faceIds),
            distance=distances, movableSkinVertexIds=np.flatnonzero(selected), localDepth=localDepth,
            eyeWidth=width, geodesicRadius=radius)
    supported = all(row['status'] == 'supported' for row in reports.values())
    if not supported: movable[:] = False
    # Unlike the old visible-eye experiment, every non-skin ID is hard fixed.
    movable[~skin] = False; movable[np.asarray(staticFullIds, int)] = False
    return dict(movableMask=movable, fixedMask=~movable, fixedVertexIds=np.flatnonzero(~movable), eyes=reports,
        supported=supported)


def radialWindingAudit(vertices, faces, faceIds, config):
    """Verify an open spherical patch's explicit winding, not signed volume."""
    triangles = np.asarray(faces)[faceIds]; points = np.asarray(vertices)[np.unique(triangles)]
    coefficients = np.linalg.lstsq(np.c_[2 * points, np.ones(len(points))], np.sum(points ** 2, axis=1), rcond=None)[0]
    center = coefficients[:3]; distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.sqrt(coefficients[3] + center @ center))
    residual = float(np.max(np.abs(distances - radius)))
    tri = vertices[triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); radial = tri.mean(1) - center
    cosine = np.sum(normal * radial, axis=1) / (np.linalg.norm(normal, axis=1) * np.linalg.norm(radial, axis=1))
    edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    checks = dict(sphereResidual=residual / radius <= config['maximumSphereResidualRatio'],
        radialFaceWinding=cosine.min() >= config['minimumRadialFaceCosine'], noNonmanifoldEdges=bool((counts <= 2).all()))
    return dict(accepted=bool(all(checks.values())), checks=checks, center=center, radius=radius,
        residualRatio=residual / radius, minimumRadialFaceCosine=float(cosine.min()),
        boundaryEdgeCount=int((counts == 1).sum()), closed=bool((counts == 2).all()), normalSign=1,
        insideOutsideCertificate=False)


def fullLocalGeometryAudit(before, after, scene, fixedIds, camera):
    """Independent original full-scene 3%/25%/.4/.3/depth envelope; no exclusions."""
    old = loadOriginalAudit(); faces = scene['faces']
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    la = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    lb = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    aa, na, _ = old.priorAudit.faceStats(before, faces); ab, nb, _ = old.priorAudit.faceStats(after, faces)
    extent = float(np.ptp(before[scene['nicpToFullIds']], axis=0).max())
    fullExtent = float(np.ptp(before, axis=0).max())
    tolerance = max(fullExtent * 1e-10, 16 * np.finfo(after.dtype).eps * max(float(np.abs(before).max()), fullExtent, 1e-12))
    delta = after - before
    metrics = dict(maximumDisplacement=float(np.linalg.norm(delta, axis=1).max()),
        maximumRelativeEdgeChange=float(np.max(np.abs(lb / la - 1))), minimumAreaRatio=float(np.min(ab / aa)),
        minimumNormalCosine=float(np.min(np.sum(na * nb, axis=1))),
        maximumReferenceDepthChange=float(np.max(np.abs(delta @ np.asarray(camera['rotation'])[2]))), referenceDepthTolerance=tolerance)
    checks = dict(finite=np.isfinite(after).all(), exactlyFixed=np.array_equal(before[fixedIds], after[fixedIds]),
        maximumDisplacement=metrics['maximumDisplacement'] <= extent * .03 * (1 + 1e-8),
        maximumRelativeEdgeChange=metrics['maximumRelativeEdgeChange'] <= .25,
        minimumAreaRatio=metrics['minimumAreaRatio'] >= .4, minimumNormalCosine=metrics['minimumNormalCosine'] >= .3,
        referenceDepth=metrics['maximumReferenceDepthChange'] <= tolerance)
    return dict(accepted=bool(all(checks.values())), checks=checks, metrics=metrics)


def exportIdentityAudit(parentAsset, archive, exportIdentity, baseline, final, movable, raw, report):
    changed = np.any(final != baseline, axis=1)
    sourceCanonical = parentAsset.mesh['vertices'] * 100 @ raw['rotation'] * raw['scale'] + raw['translation']
    expected = sourceCanonical.copy(); expected[changed] = final[changed]
    exported = archive.mesh['vertices'] * 100 @ raw['rotation'] * raw['scale'] + raw['translation']
    checks = dict(actualChangedMask=np.array_equal(changed, exportIdentity['changedVertexMask']),
        onlyMovableRowsChanged=not (changed & ~movable).any(),
        exportUsesOriginalArchiveForFixedRows=np.array_equal(exportIdentity['canonicalVertices'], expected),
        baselineRoundtrip=float(np.max(np.abs(sourceCanonical - baseline))) < 1e-12,
        candidateRoundtrip=float(np.max(np.abs(exported - final))) < 1e-12,
        unchangedParsedObjRowsExact=np.array_equal(archive.mesh['vertices'][~changed], parentAsset.mesh['vertices'][~changed]),
        nonGeometryPayloadExact=all(archive.payloads[k] == v for k, v in parentAsset.payloads.items() if k != 'face_nicp.obj'),
        bindingMetadataExact=archive.manifest['binding'] == parentAsset.manifest['binding'],
        materialMetadataExact=archive.manifest['materials'] == parentAsset.manifest['materials'],
        changedCount=report['changedVertexCount'] == int(changed.sum()))
    for key in ('faces', 'uvs', 'uvFaces', 'materialIds'):
        checks[key + 'Exact'] = np.array_equal(archive.mesh[key], parentAsset.mesh[key])
    # Preserve all OBJ non-vertex lines and complete original fixed vertex lines.
    sourceLines = parentAsset.payloads['face_nicp.obj'].decode('utf-8-sig').splitlines(keepends=True)
    candidateLines = archive.payloads['face_nicp.obj'].decode('utf-8-sig').splitlines(keepends=True)
    checks['sameObjLineCount'] = len(sourceLines) == len(candidateLines)
    lineIdentity = len(sourceLines) == len(candidateLines); vertexIndex = 0
    for left, right in zip(sourceLines, candidateLines):
        tokens = left.split('#', 1)[0].split()
        isVertex = bool(tokens and tokens[0] == 'v')
        if not isVertex or not changed[vertexIndex]: lineIdentity &= left == right
        if isVertex: vertexIndex += 1
    checks['fixedAndNonVertexObjLinesExact'] = bool(lineIdentity and vertexIndex == len(changed))
    return dict(validationPassed=bool(all(checks.values())), checks=checks, changedVertexIds=np.flatnonzero(changed),
        canonicalSerializationTolerance=1e-12, exactnessAppliesTo='Optimization NPZ and original fixed OBJ parsed coordinates/lines independently.')


def auditCase(case, runRoot, outputRoot):
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.legacyPreviewAsset import _load
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, extractVisibleEyeContours
    from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
    import hashlib
    root = runRoot / case; planPath = runRoot / 'experimentPlan.json'; config = readJson(planPath)
    if sha256(planPath) != PLAN_SHA or case not in config['cases']: raise ValueError('Case or prospective plan identity changed')
    context = readJson(root / 'inputContext.json'); complete = readJson(root / 'report.json')
    originalAudit = loadOriginalAudit(); system = originalAudit.priorAudit.independentSystem(case); raw = system['a']
    parentRoot, oldRoot = Path(context['completedBaselineRoot']), Path(context['oldPipelineRoot'])
    parentReport = readJson(parentRoot / 'report.json'); parentFull = readArrays(parentRoot / 'fullGeometry.npz')
    parentStates = readArrays(parentRoot / 'states.npz'); initial = parentStates['initial']; base = parentFull['vertices']
    skinIds = raw['skinIds']; reference = context['referenceSelection']['referenceViewIndex']
    cameras = readJson(Path(context['entry']['reconstruction']) / 'cameras.json'); camera = cameras[reference]
    inputs = readJson(Path(context['entry']['reconstruction']) / 'inputs.json'); imageShape = tuple(inputs[reference]['workingSize'][::-1])
    pixelScale = float(np.asarray(inputs[reference]['inputToWorking'])[0, 0]); imageScale = min(imageShape) / 512.
    photo = readArrays(oldRoot / 'observations/observations.npz')['eyelidCurvesWorking']
    photoEyes = {name: {part: photo[i, j] for j, part in enumerate(('lower', 'upper'))} for i, name in enumerate(EYES)}
    prepared = readEvidence(root / 'preparation'); spec = readEvidence(root / 'contactSpec'); legacy = readEvidence(root / 'legacyObservationState')
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    sourceAsset = _load(parentReport['asset']['archivePath'])
    scene = loadLegacyEyeBindingInputs(parentReport['asset']['archivePath'], skinIds); scene['vertices'] = base
    corners = np.concatenate([c['faceIndices'] for c in assets.components if c['name'] in ('male00SG.component001', 'male00SG.component002')])
    caseChecks = dict(planIdentity=context['planSha256'] == PLAN_SHA and complete['planSha256'] == PLAN_SHA,
        exactCompletedSkin=np.array_equal(base[skinIds], parentStates['final']),
        originalInitial=np.array_equal(initial, readArrays(ROOT / 'runs/shapePreservation20261024' / case / 'boundedWeight001/geometry.npz')['vertices']),
        exactScene=sameEvidence(scene, prepared['scene']), exactCamera=sameEvidence({k: camera[k] for k in ('intrinsics', 'rotation', 'translation')}, prepared['camera']),
        originalPhotoEyes=sameEvidence(photoEyes, prepared['photoEyes']), imageShape=tuple(prepared['imageShape']) == imageShape,
        originalPixelScale=prepared['originalPixelScale'] == pixelScale, policy=prepared['policy'] == config['residualPolicy'],
        cornerFaceIds=np.array_equal(prepared['cornerFaceIds'], corners),
        baselineVerticesHash=prepared['baselineVerticesSha256'] == hashlib.sha256(base.tobytes()).hexdigest())
    caseChecks['casePreparationHashes'] = fileIdentityAudit(complete['preparationHashes'])['accepted']
    caseChecks['parentOutputHashes'] = fileIdentityAudit({str(parentRoot / k): v for k, v in parentReport['outputHashes'].items()})['accepted']
    identities = fileIdentityAudit(context['sourceHashes']); caseChecks['inputHashes'] = identities['accepted']
    constraints = []
    for stage in parentReport['geometryRefinement']['stagedLandmarks']['stages']:
        if stage['name'] not in COARSE_EYES: continue
        e = stage['stageEvidence']; keep = np.asarray(stage['solver']['inputTargetGate']['groups'][0]['keptIndices'], int)
        constraints.append(dict(name=stage['name'], vertexIndices=skinIds[np.asarray(e['vertexIndices'], int)[keep]],
            barycentric=np.asarray(e['barycentricWeights'])[keep], **{k: np.asarray(e[k])[keep] for k in ('targetPixels', 'ibugIds', 'weights', 'confidence')}))
    caseChecks['exactOriginalSolverKeptCoarseConstraints'] = sameEvidence(constraints, prepared['coarseEyeConstraints'])
    binding = ibugClientLandmarks(loadClientLandmarks(raw['faces'], path=ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'), groups=('eyes', 'mouth'))
    observations = readArrays(oldRoot / 'geometry/photoObservations.npz')
    masks = originalObservationMasks(initial, raw['faces'], binding, cameras, inputs, observations)
    caseChecks['legacyInputIdentity'] = sameEvidence(legacy['binding'], binding) and sameEvidence(legacy['observations'], observations) and sameEvidence(legacy['masks'], masks)
    caseChecks['legacyInitialScores'] = sameEvidence(legacy['initialErrors'], originalObservationErrors(initial, binding, cameras, observations, masks))
    caseChecks['legacyCompletedScores'] = sameEvidence(legacy['completedBaselineErrors'], originalObservationErrors(base[skinIds], binding, cameras, observations, masks))
    measurementCache = {}; pairCache = {}
    def measure(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in measurementCache:
            measurementCache[key] = extractVisibleEyeContours(**{**scene, 'vertices': vertices}, camera=camera,
                photoEyes=photoEyes, imageShape=imageShape, apertureAccessoryMode='innerCornerHeadAccessories',
                apertureAccessoryFaceIds=corners, minimumEyeRasterWidth=256, maximumRoiSize=768)
        return measurementCache[key]
    def pairs(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in pairCache: pairCache[key] = intersectionPairs(vertices, scene['faces'])
        return pairCache[key]
    baselineMeasurement = measure(base)
    caseChecks['completeBaselineRaster'] = _contourMasksAgree(baselineMeasurement, prepared['beforeContours'])
    support = independentEyeSupport(scene, camera, photoEyes, baselineMeasurement, skinIds[raw['staticIds']], config['residualPolicy']['geodesicBandEyeWidthFraction'])
    caseChecks['frozenSupportMask'] = np.array_equal(support['movableMask'], prepared['support']['movableMask'])
    caseChecks['frozenFixedIds'] = np.array_equal(support['fixedVertexIds'], prepared['support']['fixedVertexIds'])
    for name in EYES:
        for key, value in support['eyes'][name].items():
            caseChecks[name + 'Support' + key] = sameEvidence(value, prepared['support']['eyes'][name].get(key))
    orientation = []; groups = []; savedOrientation = readJson(root / 'orientation.json')
    for component in assets.components:
        if component['materialId'] != config['orientation']['semanticMaterialId']: continue
        ids = np.asarray(component['faceIndices'], int)
        template = radialWindingAudit(assets.arrays['vertices'], assets.arrays['faces'], ids, config['orientation'])
        assembled = radialWindingAudit(base, scene['faces'], ids, config['orientation'])
        provenance = 'legacyMaleDirectServiceV1 ' + component['name'] + '; template and fixed assembled eye radial winding verified by frozen .02/.95 rule; open sphere patch; no skin-gap sign inference'
        if support['movableMask'].any(): groups.append(dict(name=component['name'], sourceVertexIds=np.flatnonzero(support['movableMask']), targetFaceIds=ids, normalSign=1, orientationProvenance=provenance))
        saved = next(row for row in savedOrientation if row['name'] == component['name'])
        valid = template['accepted'] and assembled['accepted'] and saved['provenance'] == provenance and np.array_equal(ids, saved['faceIds'])
        for key in ('template', 'assembled'):
            value = template if key == 'template' else assembled
            valid &= saved[key]['accepted'] == value['accepted'] and saved[key]['normalSign'] == 1
            for field in ('radius', 'residualRatio', 'minimumRadialFaceCosine'):
                valid &= abs(saved[key][field] - value[field]) <= 1e-10
        orientation.append(dict(name=component['name'], template=template, assembled=assembled, validationPassed=bool(valid)))
    caseChecks['radialWindingProvenance'] = all(row['validationPassed'] for row in orientation)
    contactPolicy = {key: float(raw['unit']) * config['contactPolicy'][key + 'Mm'] for key in ('lengthScale', 'searchRadius', 'activationDistance', 'minimumGap')}
    contactPolicy['weight'] = config['contactPolicy']['weight']
    contactProof = contactSpecAudit(spec, base, scene['faces'], groups, contactPolicy, savedOperator=sparse.load_npz(root / 'contactOperator.npz'))
    operator = contactProof.pop('reconstructedOperator'); caseChecks['contactSpec'] = contactProof['validationPassed']
    def contactGap(vertices):
        return frozenContactGapAudit(base, vertices, spec['sourceVertexIds'], spec['targetVertexIds'], spec['targetBarycentric'], spec['normals'],
            spec['minimumGap'], lengthScale=spec['lengthScale'], tolerance=config['contactPolicy']['tolerance'])
    def legacyGate(vertices):
        return legacyResidualObservationAudit(initial, base[skinIds], vertices[skinIds], raw['faces'], binding, cameras, inputs, observations, reference, masks=masks)
    pairAnchors = {'completedCoarseBaseline': pairs(base)}
    for name, arm in (('originalPureDirect', 'pureDirect'), ('currentBounded', 'bounded')):
        anchor = readArrays(oldRoot / 'geometry' / arm / 'fullGeometry.npz')
        caseChecks[name + 'FullTopology'] = np.array_equal(anchor['faces'], scene['faces'])
        pairAnchors[name] = pairs(anchor['vertices'])
    initialGate = initialGeometryAudit(system, initial)
    caseChecks['inheritedInitialDecision'] = initialGate['accepted'] == parentReport['initialAccepted']
    branches = {}
    for branch in BRANCHES:
        directory = root / branch; report = readJson(directory / 'report.json')
        full = readArrays(directory / 'fullGeometry.npz'); states = readArrays(directory / 'states.npz'); evidence = readEvidence(directory / 'evidence')
        final = full['vertices']; outputs = fileIdentityAudit({str(directory / k): v for k, v in report['outputHashes'].items()})
        checks = dict(caseAndBranch=report['caseId'] == case and report['branch'] == branch,
            allOutputHashes=outputs['accepted'], sameInputHashes=report['sourceHashes'] == context['sourceHashes'],
            referenceView=report['referenceViewIndex'] == reference, fullTopology=np.array_equal(full['faces'], scene['faces']),
            initialState=np.array_equal(states['initial'], initial), completedState=np.array_equal(states['completedCoarseBaseline'], base[skinIds]),
            baselineState=np.array_equal(states['baseline'], base[skinIds]), finalState=np.array_equal(states['final'], final[skinIds]),
            evidenceFinal=np.array_equal(evidence['vertices'], final), sameFrozenSupport=sameEvidence(evidence['support'], prepared['support']),
            noClientPromotion=report['finalCandidateAccepted'] is False and report['finalAdaptiveAssemblyValidated'] is False and report['textureRebaked'] is False)
        fixed = frozenAssemblyAudit(scene, {**scene, 'vertices': final}, support['movableMask'], staticFullIds=skinIds[raw['staticIds']])
        checks['fixedAssembly'] = fixed['accepted']
        mouthFullIds = skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]
        checks['realMouthExactlyFixed'] = np.array_equal(final[mouthFullIds], base[mouthFullIds])
        archivePath = Path(report['asset']['archivePath']); archive = _load(archivePath)
        export = exportIdentityAudit(sourceAsset, archive, readArrays(directory / 'exportIdentity.npz'), base, final, support['movableMask'], raw, report['asset'])
        checks['exportIdentity'] = export['validationPassed']; checks['archiveHash'] = sha256(archivePath) == report['asset']['sha256']
        downstream = downstreamGeometryAudit(system, initial, final[skinIds], camera)
        coarse = coarseEyePreservationAudit(base, final, constraints, camera, imageScale=imageScale)
        contacts = contactGap(final); legacyReport = legacyGate(final)
        collision = historicalCollisionAudit(pairAnchors['completedCoarseBaseline'], pairs(final), pairAnchors, scene['materialIds'])
        for name, suffix in (('originalPureDirect', 'OriginalPureDirect'), ('currentBounded', 'CurrentBounded'), ('completedCoarseBaseline', 'PairedBaseline')):
            checks[name + 'NewPairs'] = collision['anchors'][name]['newPairs'] == [tuple(v) for v in report['newFullPairsAgainst' + suffix]]
        checks['skinNewPairs'] = downstream['cumulativeDownstream']['collisions']['newPairs'] == [tuple(v) for v in report['newSkinPairsAgainstBranchInitial']]
        checks['downstreamDecision'] = downstream['cumulativeDownstream']['accepted'] == bool(report['cumulativeDownstreamGate']['accepted'] and not report['newSkinPairsAgainstBranchInitial'])
        checks['cavityDecision'] = downstream['originalCavity']['accepted'] == bool(report['finalCavityGuard']['accepted'] and report['finalCavityBenefit']['accepted'])
        checks['coarseDecision'] = coarse['accepted'] == report['pairedSemanticAlignment']['accepted']
        checks['contactDecision'] = contacts['accepted'] == report['contactSafety']['accepted']
        checks['legacyDecision'] = legacyReport['accepted'] == report['legacyObservationGate']['accepted']
        incremental = bool(fixed['accepted'] and coarse['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs'])
        absolute = bool(initialGate['accepted'] and incremental and downstream['accepted'] and legacyReport['accepted'] and collision['allHistoricalAnchorsAccepted'])
        checks['incrementalDecision'] = incremental == report['incrementalSafe']
        checks['absoluteDecision'] = absolute == report['absoluteEligible'] == report['geometrySafetyAccepted']
        effect = fullApertureRasterAudit(baselineMeasurement, measure(final), photoEyes, originalPixelScale=pixelScale)
        eyeProof = auditResidualEvidence(branch, prepared, evidence, base, final, scene, support, constraints, camera, imageScale,
            baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap, legacyGate, spec, operator)
        checks['residualEvidence'] = eyeProof['validationPassed']
        checks['residualDecision'] = eyeProof['residualCandidateAccepted'] == report['residualCandidateAccepted']
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks,
            incrementalSafe=incremental, absoluteEligible=absolute, residualCandidateAccepted=eyeProof['residualCandidateAccepted'],
            requestedEffectAccepted=bool(effect['accepted'] and eyeProof['residualCandidateAccepted']),
            geometryExperimentAccepted=bool(absolute and eyeProof['residualCandidateAccepted'] and all(checks.values())),
            unchangedRealMouth=np.array_equal(final[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]], base[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]]),
            fixedAssembly=fixed, exportIdentity=export, downstream=downstream, originalCoarseAlignment=coarse,
            contactEnvelope=contacts, legacyPhotoGate=legacyReport, fullCollisionAnchors=collision,
            fullApertureEffect=effect, residualEvidence=eyeProof, producerOutputs=outputs,
            reportSha256=sha256(directory / 'report.json'), archiveSha256=sha256(archivePath), clientReady=False)
    caseChecks['inputHashesAfterAudit'] = fileIdentityAudit(context['sourceHashes'])['accepted']
    result = dict(caseId=case, validationPassed=bool(all(caseChecks.values()) and all(v['validationPassed'] for v in branches.values())),
        checks=caseChecks, initialGate=initialGate, contactSpecification=contactProof, orientation=orientation,
        originalInputIdentities=identities, branches=branches, planSha256=PLAN_SHA, validatorSha256=sha256(__file__),
        candidateSolvesPerformed=False, clientReady=False,
        scope='Same completed coarse baseline and fixed attachments. Contact proxy and automatic photo curves are not anatomical ground truth; inherited failures stay failures.')
    outputRoot.mkdir(parents=True, exist_ok=True); writeJson(outputRoot / f'{case}.json', result)
    return result


def _scoreFieldsAgree(independent, saved):
    if independent is None: return saved is None
    if saved is None: return False
    if abs(independent['meanSymmetricBoundaryDistanceOriginalPixels'] - saved['meanSymmetricBoundaryDistanceOriginalPixels']) > 1e-10: return False
    for name in EYES:
        for key in ('symmetricBoundaryMeanOriginalPixels', 'symmetricBoundaryP95OriginalPixels', 'intersectionOverUnion',
                    'visibleComponentCount', 'secondaryComponentAreaFraction', 'modelRasterArea', 'photoRasterArea'):
            if abs(independent['eyes'][name][key] - saved['eyes'][name][key]) > 1e-10: return False
    return True


def residualAcceptanceFromRounds(strictAccepted, changed, fastStatus, selectedRoundGates):
    """Reproduce the existing per-round fitter, without inventing a new gate.

    Final-versus-original aperture regression remains separately reported as a
    quality diagnostic. It must not change whether the producer was faithfully
    reproduced: its per-eye tolerances apply to each selected round's input.
    """
    return bool(strictAccepted and changed and fastStatus == 'acceptedLocalExperiment'
                and selectedRoundGates and all(selectedRoundGates))


def auditResidualEvidence(branch, prepared, evidence, baseline, final, scene, support, constraints, camera,
                          imageScale, baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap,
                          legacyGate, spec, operator):
    """Retain every rejected proposal and reproduce selection without fitting."""
    from faceAppearance.fastFrontEyeGeometry import _constraints, _eyeContactSafety
    diagnostics = evidence['diagnostics']; proposals = evidence['candidateProposals']
    checks = dict(beforeFullRaster=_contourMasksAgree(baselineMeasurement, evidence['beforeContours']),
        afterFullRaster=_contourMasksAgree(measure(final), evidence['afterContours']),
        rawAndSolverRetained=all(k in evidence for k in ('rawCandidateVertices', 'solverVertices')))
    if branch == 'baseline':
        checks.update(unchangedBaseline=np.array_equal(final, baseline), noCandidateProposals=not proposals,
            rawExactlyBaseline=np.array_equal(evidence['rawCandidateVertices'], baseline),
            solverExactlyBaseline=np.array_equal(evidence['solverVertices'], baseline))
        return dict(validationPassed=bool(all(checks.values())), checks=checks, residualCandidateAccepted=False, proposals=[])
    withContact = branch == 'residualWithContact'
    checks.update(contactFactor=diagnostics['contactOptimizationEnabled'] == withContact,
        frozenOriginalCoarse=sameEvidence(evidence['coarseEyeConstraints'], constraints),
        neverRerunCoarse=all(diagnostics[k] is False for k in ('coarseStagesRerun', 'coarseStagesSkipped', 'coarseStagesRestricted')),
        outerCandidateGateUsed=diagnostics['externalCandidateGateUsed'] is True,
        rawIsLastProposal=np.array_equal(evidence['rawCandidateVertices'], proposals[-1]['vertices'] if proposals else evidence['solverVertices']))
    if withContact:
        checks['sameFrozenContactSpec'] = sameEvidence(evidence['contactSpec'], spec)
    else: checks['noContactOptimizationSpec'] = evidence['contactSpec'] is None
    baselinePairs = pairs(baseline); strictCache = {}
    def strict(vertices):
        import hashlib
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in strictCache:
            geometry = fullLocalGeometryAudit(baseline, vertices, scene, support['fixedVertexIds'], camera)
            fullPairs = intersectionSetAudit(baselinePairs, pairs(vertices), scene['materialIds'])
            coarse = coarseEyePreservationAudit(baseline, vertices, constraints, camera, imageScale=imageScale)
            contact = contactGap(vertices); legacy = legacyGate(vertices)
            strictCache[key] = dict(accepted=bool(geometry['accepted'] and fullPairs['accepted'] and coarse['accepted'] and contact['accepted'] and legacy['accepted']),
                fullGeometry=geometry, fullPairs=fullPairs, coarse=coarse, contact=contact, legacy=legacy)
        return strictCache[key]
    reports = []; current = baseline.copy(); selected = []
    stages = diagnostics.get('fast', {}).get('stages', [])
    checks['everyAttemptRetained'] = len(proposals) == sum(len(s.get('attempts', [])) for s in stages) == diagnostics['proposalCount']
    checks['boundedRoundAndIterationSettings'] = (not diagnostics['fastSolverInvoked'] or
        diagnostics['fast']['roundsRequested'] == prepared['policy']['rounds'] and diagnostics['fast']['maximumSolverIterations'] == prepared['policy']['rounds'] * prepared['policy']['iterationsPerRound'])
    movableIds = np.unique(np.r_[scene['nicpToFullIds'], scene['faces'][prepared['cornerFaceIds']].ravel()])
    epsilon = max(np.linalg.norm(np.ptp(baseline, axis=0)) * 1e-9, 1e-12)
    for stage in stages:
        currentMeasurement = measure(current)
        # This is a read-only visibility binding primitive, not an optimization.
        _, bindingStats = _constraints(current, scene['faces'], scene['materialIds'], camera, photoEyes, currentMeasurement,
            movableIds, prepared['cornerFaceIds'], scene['uvs'], scene['uvFaces'], scene['materials'], scene['alphaTextures'],
            fixedVertexIds=support['fixedVertexIds'])
        checks['round' + str(stage['round']) + 'OnlyFullyFixedBindingOmissions'] = sameEvidence(bindingStats, stage['binding'])
        stageProposals = [p for p in proposals if p['round'] == stage['round']]
        checks['round' + str(stage['round']) + 'AttemptOrder'] = [p['scale'] for p in stageProposals] == [a['scale'] for a in stage['attempts']]
        predictedSelected = None
        for proposal, attempt in zip(stageProposals, stage['attempts']):
            vertices = proposal['vertices']; gate = strict(vertices); stored = proposal['strict']; pc = {}
            pc['strictDecision'] = gate['accepted'] == stored['accepted']
            pc['geometryDecision'] = gate['fullGeometry']['accepted'] == stored['geometry']['accepted']
            pc['coarseDecision'] = gate['coarse']['accepted'] == stored['coarseAlignment']['accepted']
            pc['sharedContactDecision'] = gate['contact']['accepted'] == stored['externalCandidateGate']['contactGate']['accepted']
            pc['legacyDecision'] = gate['legacy']['accepted'] == stored['externalCandidateGate']['legacyGate']['accepted']
            pc['externalDecision'] = bool(gate['legacy']['accepted'] and gate['contact']['accepted']) == stored['externalCandidateGate']['accepted']
            if 'newPairs' in stored.get('fullAssembly', {}):
                pc['fullUnexcludedPairSet'] = gate['fullPairs']['newPairs'] == [tuple(v) for v in stored['fullAssembly']['newPairs']]
            else: pc['collisionSkipOnlyOnGeometryFailure'] = not gate['fullGeometry']['accepted']
            pc['solverAttemptExact'] = sameEvidence(proposal['solverAttempt'], attempt)
            pc['internalGeometrySaved'] = sameEvidence(proposal['internalGeometry'], attempt['geometry'])
            pc['externalStrictSaved'] = sameEvidence(stored, attempt['externalCandidateGate'])
            if withContact:
                pc['insideContactGate'] = gate['contact']['accepted'] == stored['contact']['accepted']
                actual = operator @ vertices.ravel(); snapshot = proposal['contactEvaluation']
                pc['contactGapEvidence'] = np.allclose(actual, snapshot['gaps'], atol=1e-12, rtol=0)
                expectedResidual = np.sqrt(spec['weight'] / len(actual)) * np.minimum((actual - spec['minimumGap']) / spec['lengthScale'], 0) if len(actual) else np.empty(0)
                pc['contactEnergy'] = abs(.5 * float(expectedResidual @ expectedResidual) - snapshot['energy']) <= 1e-10
            else: pc['noInsideContactOptimization'] = stored['contact'] is None and proposal['contactEvaluation'] is None
            aperture = None; shouldSelect = False
            if attempt['geometry']['accepted'] and gate['accepted']:
                measured = measure(vertices)
                aperture = fullApertureRasterAudit(currentMeasurement, measured, photoEyes, originalPixelScale=pixelScale)
                score = aperture.get('metrics', {}).get('candidate')
                pc['actualCompleteMaskScores'] = _scoreFieldsAgree(score, attempt.get('apertureMetrics'))
                maskBenefit = aperture.get('originalApertureGate', {}).get('accepted', False)
                pc['perRoundApertureGate'] = maskBenefit == attempt.get('apertureGate')
                if maskBenefit:
                    sphere = _eyeContactSafety(baseline, vertices, scene['faces'], scene['materialIds'], movableIds, epsilon)
                    pc['legacyEyeSphereGate'] = sphere['accepted'] == attempt.get('eyeContact', {}).get('accepted')
                    shouldSelect = bool(sphere['accepted'] and gate['fullPairs']['accepted'])
                    if shouldSelect: pc['fastSubsetCollisionNotSubstitute'] = attempt.get('intersection', {}).get('newPairCount') == 0
            else: pc['notMeasuredAfterHardGateFailure'] = 'apertureMetrics' not in attempt
            pc['selectedByFastRound'] = bool(proposal['selectedByFastRound']) == shouldSelect
            if shouldSelect:
                predictedSelected = vertices.copy(); selected.append((stage['round'], proposal['scale']))
                pc['selectedScaleMatches'] = stage['status'] == 'accepted' and stage['selectedScale'] == proposal['scale']
            reports.append(dict(round=proposal['round'], scale=proposal['scale'], selected=shouldSelect,
                validationPassed=bool(all(pc.values())), checks=pc, independentStrict=gate,
                completeAperture=aperture, producerStrict=stored))
        if predictedSelected is not None: current = predictedSelected
        else: checks['round' + str(stage['round']) + 'NoFalseSelection'] = stage['status'] != 'accepted'
    solver = evidence['solverVertices']; checks['solverEqualsSelectedProposalChain'] = np.array_equal(solver, current)
    checks['solverFullRaster'] = _contourMasksAgree(measure(solver), evidence['solverContours'])
    solverGate = strict(solver)
    solverEffect = fullApertureRasterAudit(baselineMeasurement, measure(solver), photoEyes, originalPixelScale=pixelScale)
    actualComplete = all(measure(solver)['eyes'][name]['status'] == 'accepted' for name in EYES)
    roiSame = all(solverEffect['checks'].get(name + 'SameRaster', False) for name in EYES)
    producerHasFast = diagnostics['fastSolverInvoked']
    if producerHasFast:
        predictedStrict = bool(solverGate['accepted'] and actualComplete and roiSame)
        checks['finalStrictDecision'] = predictedStrict == diagnostics['strict']['accepted']
        checks['fastChangeStatus'] = (not np.array_equal(solver, baseline)) == (diagnostics['fast']['status'] == 'acceptedLocalExperiment')
        accepted = residualAcceptanceFromRounds(predictedStrict, not np.array_equal(solver, baseline),
            diagnostics['fast']['status'], [p['completeAperture']['originalApertureGate']['accepted'] for p in reports if p['selected']])
    else:
        status = diagnostics['status']
        skipValid = (status == 'skippedInsufficientVisibleSupport' and not support['supported'] or
            status == 'skippedUnmeasurableCompleteAperture' and prepared['beforeApertureMetrics'] is None or
            status == 'skipNoActiveContactRows' and withContact and len(spec['sourceVertexIds']) == 0)
        checks['skipHasActualInputReason'] = skipValid; accepted = False
    checks['wrapperAcceptanceReproduced'] = accepted == diagnostics['accepted']
    checks['returnedSelectedOrExactBaseline'] = np.array_equal(final, solver if accepted else baseline)
    checks['acceptedRequiresActualPerRoundMaskBenefit'] = not accepted or all(
        p['completeAperture']['originalApertureGate']['accepted'] for p in reports if p['selected'])
    if withContact and producerHasFast and any(s.get('solver') for s in stages):
        calls = diagnostics['contactEvaluationCalls']
        checks['contactInsideOptimizationEvidence'] = calls['residual'] > 0 and calls['jacobian'] > 0 and calls['safety'] > 0
    elif not withContact:
        checks['noContactResidualCalls'] = diagnostics['contactEvaluationCalls']['residual'] == diagnostics['contactEvaluationCalls']['jacobian'] == 0
    checks['allRawProposalEvidenceValidated'] = all(p['validationPassed'] for p in reports)
    return dict(validationPassed=bool(all(checks.values())), checks=checks, residualCandidateAccepted=accepted,
        solverEffect=solverEffect, finalVersusOriginalEffectAccepted=solverEffect['accepted'],
        finalVersusOriginalEffectIsSeparateFromProducerDecision=True, finalStrict=solverGate, proposals=reports,
        contactOptimizationEnabled=withContact, actualCompleteContourStatus={name: measure(solver)['eyes'][name]['status'] for name in EYES},
        sharedPrimitives=['complete triangle/alpha visibility raster', 'triangle intersection predicate', 'fast boundary binding and supplementary original eye-sphere gate'],
        independentReconstruction=['support geodesics', 'coarse/legacy error and masks', 'complete mask score and decision', 'contact nearest feature/operator/gap', 'all full-scene pair differences and decision chain'])


def main():
    import argparse
    import json
    import traceback
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/residualContact20261024')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/residualContact20261024/independentAudit')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for case in args.case:
        try:
            result = auditCase(case, args.run_root, args.output); results.append(result)
            print(json.dumps(dict(caseId=case, validationPassed=result['validationPassed'],
                residualAccepted=[k for k, v in result['branches'].items() if v['residualCandidateAccepted']])), flush=True)
        except Exception as error:
            failure = dict(caseId=case, validationPassed=False, toolException=True, exceptionType=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            failures.append(failure); writeJson(args.output / f'{case}.toolException.json', failure)
            print(json.dumps(failure), flush=True)
    counts = {branch: {key: sum(row['branches'][branch][key] for row in results) for key in
        ('validationPassed', 'incrementalSafe', 'absoluteEligible', 'residualCandidateAccepted', 'requestedEffectAccepted', 'geometryExperimentAccepted')} for branch in BRANCHES}
    summary = dict(validationPassed=not failures and all(row['validationPassed'] for row in results), caseCount=len(results), counts=counts,
        toolExceptionCount=len(failures), failures=failures, cases=[dict(caseId=row['caseId'], validationPassed=row['validationPassed'], report=str(args.output / (row['caseId'] + '.json')),
            reportSha256=sha256(args.output / (row['caseId'] + '.json'))) for row in results], validatorSha256=sha256(__file__),
        planSha256=PLAN_SHA, clientReady=False, expectedRejectionsAreToolFailures=False)
    writeJson(args.output / 'validationSummary.json', summary); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
