"""Bounded, frozen oriented point/triangle contact proxies.

These local halfspaces are optimization aids, not collision certificates.
Orientation comes from explicitly identified template winding; it is never
inferred by making a baseline point's signed gap positive. Both sides may move.
The original full-scene triangle intersection gate remains mandatory.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


_MAX_VERTICES = 100000
_MAX_FACES = 200000
_MAX_SAMPLES = 4096
_MAX_TARGET_FACES = 8192
_MAX_CANDIDATE_PAIRS = 2000000


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f'{name} must be a finite scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite real scalar') from error
    if not np.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _vertices(value, expectedShape=None):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 3 or not 3 <= len(result) <= _MAX_VERTICES
            or not np.isfinite(result).all() or (expectedShape is not None and result.shape != expectedShape)):
        raise ValueError('vertices must be finite Nx3 coordinates matching the frozen scene')
    return result


def _ids(value, name, count):
    result = np.asarray(value)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or not len(result)
            or np.any(result < 0) or np.any(result >= count) or len(np.unique(result)) != len(result)):
        raise ValueError(f'{name} must contain nonempty unique valid integer IDs')
    return np.sort(result.astype(np.int64, copy=True))


def _closestTriangleFeatures(point, triangles):
    """Exact minimum over the plane interior and all three closed segments.

    All supplied faces are enumerated, so no centroid-neighbor heuristic can
    miss a long triangle. Callers validate nondegenerate target triangles first.
    Work is O(number of targets), with no source-by-target dense allocation.
    """
    a = triangles[:, 0]
    u, v = triangles[:, 1]-a, triangles[:, 2]-a
    cross = np.cross(u, v)
    crossSquared = np.sum(cross*cross, axis=1)
    relative = point-a
    # Cross products avoid subtraction of nearly equal Gram determinants.
    beta = np.sum(np.cross(relative, v)*cross, axis=1)/crossSquared
    gamma = np.sum(np.cross(u, relative)*cross, axis=1)/crossSquared
    planeBary = np.column_stack((1-beta-gamma, beta, gamma))
    planePoints = np.einsum('fi,fij->fj', planeBary, triangles)
    planeSquared = np.sum((point-planePoints)**2, axis=1)
    planeSquared[np.any(planeBary < 0, axis=1)] = np.inf
    candidates = [planeBary]
    squared = [planeSquared]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, second]-triangles[:, first]
        fraction = np.clip(np.sum((point-triangles[:, first])*edge, axis=1)/np.sum(edge*edge, axis=1), 0., 1.)
        bary = np.zeros((len(triangles), 3))
        bary[:, first], bary[:, second] = 1-fraction, fraction
        projected = triangles[:, first]+fraction[:, None]*edge
        candidates.append(bary)
        squared.append(np.sum((point-projected)**2, axis=1))
    distances = np.asarray(squared)
    selected = np.argmin(distances, axis=0)
    rows = np.arange(len(triangles))
    chosenBary, chosenSquared = np.asarray(candidates)[selected, rows], distances[selected, rows]
    if not np.isfinite(chosenBary).all() or not np.isfinite(chosenSquared).all():
        raise ValueError('Closest-feature arithmetic overflowed; rescale scene coordinates')
    return chosenBary, chosenSquared


def buildSurfaceContactSpec(
    baselineVertices, faces, contactGroups, *, lengthScale, searchRadius,
    activationDistance, minimumGap=0., weight=1.,
):
    """Build a data-only frozen halfspace specification for a complete scene.

    Each group contains ``name``, ``sourceVertexIds``, ``targetFaceIds``,
    ``normalSign`` (+1/-1), and nonempty ``orientationProvenance``. IDs refer to
    the SAME full-scene vertex/face order. The caller selects template semantic
    domains (e.g. skin around one eye and that opaque eyeball), establishes the
    outward winding independently, and freezes all intended fixed vertices.
    This function does not infer semantic pairing, orientation or confidence.
    Alpha cards are not suitable oriented solid obstacles.

    Distances and lengthScale use the same coordinate units. All target faces
    in a group are searched exactly. Source-incident faces are excluded, and
    nearest features farther than searchRadius are not replaced by another
    component. An in-radius feature activates when signed gap <=
    activationDistance, including negative gaps. All selections and skip masks
    are saved. A negative signed gap is an oriented proxy violation, not proof
    of volumetric inside/outside on an open surface or at an edge feature.

    For a frozen target barycentric point b and outward normal n,
    ``gap = n . (Vsource - sum(b * Vtarget))``. The hard lower bound is
    ``min(baselineGap, minimumGap)``: inherited negative gaps are explicitly
    recorded and cannot worsen. This is grandfathering, not new clearance or
    acceptance of existing intersections. Both moving surfaces participate in
    the Jacobian. A hinge residual towards minimumGap encourages recovery.

    Residuals include sqrt(weight / activeRowCount), so 0.5*||residual||^2 is
    half the weighted mean squared normalized clearance deficit. Hard rows are
    independent of weight. Missing/inactive groups produce zero rows, not a
    fabricated constraint. Geometry trust regions and the original strict full
    triangle-pair difference gate MUST be retained by the caller: frozen normal
    and feature barycentrics do not track later sliding or face rotation, and
    vertex samples cannot detect every edge/face intersection.
    """
    baseline = _vertices(baselineVertices).copy()
    triangles = np.asarray(faces)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or not 1 <= len(triangles) <= _MAX_FACES or np.any(triangles < 0)
            or np.any(triangles >= len(baseline))
            or np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0)):
        raise ValueError('faces must contain valid distinct integer triangle corners within the scene budget')
    triangles = triangles.astype(np.int64, copy=True)
    scale = _scalar(lengthScale, 'lengthScale', True)
    radius = _scalar(searchRadius, 'searchRadius', True)
    activation = _scalar(activationDistance, 'activationDistance')
    clearance = _scalar(minimumGap, 'minimumGap')
    energyWeight = _scalar(weight, 'weight')
    if clearance > activation or activation > radius:
        raise ValueError('Require 0 <= minimumGap <= activationDistance <= searchRadius')
    if not isinstance(contactGroups, (list, tuple)):
        raise ValueError('contactGroups must be a sequence of explicit oriented template domains')
    groups, names = [], set()
    sampleCount = targetCount = pairCount = 0
    for group in contactGroups:
        required = {'name', 'sourceVertexIds', 'targetFaceIds', 'normalSign', 'orientationProvenance'}
        if not isinstance(group, dict) or set(group) != required:
            raise ValueError('Each contact group must specify exactly name, sourceVertexIds, targetFaceIds, normalSign, orientationProvenance')
        name, provenance = group['name'], group['orientationProvenance']
        sign = group['normalSign']
        if (not isinstance(name, str) or not name.strip() or name in names
                or not isinstance(provenance, str) or not provenance.strip()
                or isinstance(sign, (bool, np.bool_)) or np.ndim(sign) != 0
                or np.asarray(sign).dtype.kind not in 'iuf' or sign not in (-1, 1)):
            raise ValueError('Groups need unique names, explicit +/-1 winding signs and orientation provenance')
        source = _ids(group['sourceVertexIds'], 'sourceVertexIds', len(baseline))
        target = _ids(group['targetFaceIds'], 'targetFaceIds', len(triangles))
        sampleCount += len(source); targetCount += len(target); pairCount += len(source)*len(target)
        if sampleCount > _MAX_SAMPLES or targetCount > _MAX_TARGET_FACES or pairCount > _MAX_CANDIDATE_PAIRS:
            raise ValueError('Contact sample/target/pair budget exceeded; refusing silent truncation')
        names.add(name)
        groups.append({'name': name, 'sourceVertexIds': source, 'targetFaceIds': target,
            'normalSign': int(sign), 'orientationProvenance': provenance})

    sourceRows, faceRows, baryRows, normalRows, groupRows = [], [], [], [], []
    groupSpans, selections = {}, []
    for groupIndex, group in enumerate(groups):
        start = len(sourceRows)
        targetIds = group['targetFaceIds']
        targetFaces = triangles[targetIds]
        targetPoints = baseline[targetFaces]
        cross = np.cross(targetPoints[:, 1]-targetPoints[:, 0], targetPoints[:, 2]-targetPoints[:, 0])
        norms = np.linalg.norm(cross, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('Contact target triangles must have finite nonzero area')
        normals = group['normalSign']*cross/norms[:, None]
        count = len(group['sourceVertexIds'])
        selectedFaces = np.full(count, -1, np.int64)
        selectedBary = np.zeros((count, 3))
        selectedNormals = np.zeros((count, 3))
        distances = np.zeros(count)
        signedGaps = np.zeros(count)
        hasCandidate = np.zeros(count, bool)
        inRadius = np.zeros(count, bool)
        active = np.zeros(count, bool)
        tied = np.zeros(count, bool)
        incidentCounts = np.zeros(count, np.int64)
        for index, sourceId in enumerate(group['sourceVertexIds']):
            eligible = np.flatnonzero(~np.any(targetFaces == sourceId, axis=1))
            incidentCounts[index] = len(targetFaces)-len(eligible)
            if not len(eligible):
                continue
            bary, distanceSquared = _closestTriangleFeatures(baseline[sourceId], targetPoints[eligible])
            closest = int(np.argmin(distanceSquared))
            targetIndex = eligible[closest]
            point = bary[closest] @ targetPoints[targetIndex]
            distance = float(np.sqrt(distanceSquared[closest]))
            gap = float(normals[targetIndex] @ (baseline[sourceId]-point))
            hasCandidate[index] = True
            selectedFaces[index], selectedBary[index] = targetIds[targetIndex], bary[closest]
            selectedNormals[index], distances[index], signedGaps[index] = normals[targetIndex], distance, gap
            tied[index] = np.count_nonzero(np.isclose(distanceSquared, distanceSquared[closest], rtol=1e-12, atol=1e-24*scale*scale)) > 1
            inRadius[index] = distance <= radius
            active[index] = inRadius[index] and gap <= activation
            if active[index]:
                sourceRows.append(sourceId); faceRows.append(targetIds[targetIndex])
                baryRows.append(bary[closest]); normalRows.append(normals[targetIndex]); groupRows.append(groupIndex)
        groupSpans[group['name']] = {'start': start, 'stop': len(sourceRows)}
        selections.append({'name': group['name'], 'sourceVertexIds': group['sourceVertexIds'].copy(),
            'nearestTargetFaceIds': selectedFaces, 'nearestBarycentric': selectedBary,
            'nearestNormals': selectedNormals, 'baselineDistances': distances,
            'baselineSignedGaps': signedGaps, 'hasCandidateMask': hasCandidate,
            'withinSearchRadiusMask': inRadius, 'activeMask': active,
            'nearestTieMask': tied, 'sourceIncidentExcludedCounts': incidentCounts})

    sourceRows = np.asarray(sourceRows, np.int64)
    faceRows = np.asarray(faceRows, np.int64)
    baryRows = np.asarray(baryRows, float).reshape(-1, 3)
    normalRows = np.asarray(normalRows, float).reshape(-1, 3)
    targetVertices = triangles[faceRows]
    rowCount = len(sourceRows)
    corners = np.column_stack((sourceRows, targetVertices))
    cornerWeights = np.column_stack((np.ones(rowCount), -baryRows))
    columns = 3*corners[:, :, None]+np.arange(3)[None, None, :]
    rows = np.broadcast_to(np.arange(rowCount)[:, None, None], columns.shape)
    coefficients = cornerWeights[:, :, None]*normalRows[:, None, :]
    operator = sparse.csr_matrix((coefficients.ravel(), (rows.ravel(), columns.ravel())), shape=(rowCount, baseline.size))
    operator.sum_duplicates(); operator.eliminate_zeros()
    baselineGaps = np.asarray(operator @ baseline.ravel())
    if not np.isfinite(baselineGaps).all():
        raise ValueError('Contact gap computation overflowed')
    inherited = baselineGaps < 0
    lower = np.minimum(baselineGaps, clearance)
    diagnostics = {'candidateSourceCount': sampleCount, 'enumeratedPairBudget': pairCount,
        'activeRowCount': rowCount, 'inheritedNegativeGapCount': int(inherited.sum()),
        'inheritedClearanceDeficitCount': int(np.count_nonzero(baselineGaps < clearance)),
        'baselineCandidateNegativeGapCount': sum(int((x['hasCandidateMask'] & (x['baselineSignedGaps'] < 0)).sum()) for x in selections),
        'outsideSearchNegativeGapCount': sum(int((x['hasCandidateMask'] & ~x['withinSearchRadiusMask'] & (x['baselineSignedGaps'] < 0)).sum()) for x in selections),
        'activeEdgeOrVertexFeatureCount': int(np.any(baryRows <= 0, axis=1).sum()),
        'noEligibleTargetCount': sum(int((~x['hasCandidateMask']).sum()) for x in selections),
        'outsideSearchRadiusCount': sum(int((x['hasCandidateMask'] & ~x['withinSearchRadiusMask']).sum()) for x in selections),
        'insideSearchButInactiveCount': sum(int((x['withinSearchRadiusMask'] & ~x['activeMask']).sum()) for x in selections),
        'nearestTieCount': sum(int(x['nearestTieMask'].sum()) for x in selections),
        'orientationFromBaselineGap': False, 'isCollisionCertificate': False,
        'normalAndBarycentricFrozen': True, 'allTargetFacesEnumerated': True,
        'limitation': 'Oriented local point/triangle proxy; no continuous contact, inside/outside or full triangle nonintersection guarantee. Retain trust regions and strict full-scene intersection gate.'}
    return {'schemaVersion': 1, 'vertexCount': len(baseline), 'baselineVertices': baseline,
        'faces': triangles, 'contactGroups': groups, 'lengthScale': scale,
        'searchRadius': radius, 'activationDistance': activation, 'minimumGap': clearance,
        'weight': energyWeight, 'sourceVertexIds': sourceRows, 'targetFaceIds': faceRows,
        'targetVertexIds': targetVertices, 'targetBarycentric': baryRows,
        'normals': normalRows, 'rowGroupIds': np.asarray(groupRows, np.int64),
        'gapOperator': operator, 'baselineGaps': baselineGaps, 'lowerGaps': lower,
        'inheritedPenetrationMask': inherited, 'groups': groupSpans,
        'selectionEvidence': selections, 'diagnostics': diagnostics}


def evaluateSurfaceContacts(vertices, contactSpec, withJacobian=True):
    """Evaluate frozen hard rows >=0 and weighted hinge residual/energy.

    Jacobians include both source and target corners. The hinge derivative at
    exactly zero is chosen as zero; squared energy is differentiable there.
    This does not refit correspondences or normals to the proposed geometry.
    """
    current = _vertices(vertices, contactSpec['baselineVertices'].shape)
    operator = contactSpec['gapOperator']
    scale = contactSpec['lengthScale']
    gaps = np.asarray(operator @ current.ravel())
    values = (gaps-contactSpec['lowerGaps'])/scale
    deficits = np.minimum((gaps-contactSpec['minimumGap'])/scale, 0.)
    factor = np.sqrt(contactSpec['weight']/max(len(gaps), 1))
    residual = factor*deficits
    with np.errstate(over='ignore', invalid='ignore'):
        energy = .5*float(residual @ residual)
    if not np.isfinite(values).all() or not np.isfinite(residual).all() or not np.isfinite(energy):
        raise ValueError('Contact evaluation must remain finite')
    result = {'values': values, 'gaps': gaps, 'residual': residual,
        'energy': energy, 'groups': contactSpec['groups']}
    if withJacobian:
        jacobian = operator/scale
        residualJacobian = sparse.diags(factor*(deficits < 0)) @ jacobian
        result.update({'jacobian': jacobian.tocsr(), 'residualJacobian': residualJacobian.tocsr(),
            'energyGradient': np.asarray(residualJacobian.T @ residual).reshape(current.shape)})
        if (not np.isfinite(result['jacobian'].data).all()
                or not np.isfinite(result['residualJacobian'].data).all()
                or not np.isfinite(result['energyGradient']).all()):
            raise ValueError('Contact derivatives must remain finite; rescale coordinates or weight')
    return result


def evaluateSurfaceContactResidual(vertices, contactSpec, withJacobian=True):
    """Bridge to additionalResidual; normalization/weight are already included."""
    evaluated = evaluateSurfaceContacts(vertices, contactSpec, withJacobian)
    result = {'residual': evaluated['residual']}
    if withJacobian:
        result['jacobian'] = evaluated['residualJacobian']
    return result


def evaluateSurfaceContactSafety(vertices, contactSpec, *, tolerance=1e-8):
    """Only the frozen proxy's no-worsening gate; never full collision acceptance."""
    checkedTolerance = _scalar(tolerance, 'tolerance')
    evaluated = evaluateSurfaceContacts(vertices, contactSpec, False)
    values, gaps = evaluated['values'], evaluated['gaps']
    violating = np.flatnonzero(values < -checkedTolerance)
    return {'accepted': not len(violating), 'scope': 'frozen oriented contact proxy only',
        'activeRowCount': len(values), 'violatingRowIds': violating.tolist(),
        'minimumValue': float(np.min(values)) if len(values) else None,
        'baselineNegativeGapCount': int(contactSpec['inheritedPenetrationMask'].sum()),
        'currentNegativeGapCount': int(np.count_nonzero(gaps < 0)),
        'noActiveContacts': not len(values), 'tolerance': checkedTolerance,
        'strictFullIntersectionGateRequired': True, 'isCollisionCertificate': False}
