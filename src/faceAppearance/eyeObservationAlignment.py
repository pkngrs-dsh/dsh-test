"""Unify sparse eye observations with frozen dense photo eyelid curves.

This is an explicit observation-definition ablation, not a geometry fitter or
a claim that dense detections are anatomical truth. One complete 12-point
iBUG table is computed from original photo points, then reused by every gate.
Interior points preserve their position along the dense canthus axis; corner
points are an explicit exception and move to dense endpoint midpoints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np


SCHEMA_VERSION = 'denseEyeAxisTargetsV1'
AXIS_TOLERANCE = 64*np.finfo(np.float64).eps
MAX_CURVE_POINTS = 4096
EYE_SPECS = {
    'imageLeft': dict(group='imageLeftEye', left=36, right=39, upper=(37, 38), lower=(41, 40)),
    'imageRight': dict(group='imageRightEye', left=42, right=45, upper=(43, 44), lower=(47, 46)),
}


def _finitePoints(value, name, count=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be finite real Nx2 photo points')
    if count is not None and len(raw) != count:
        raise ValueError(f'{name} must contain exactly {count} points')
    return raw.astype(np.float64, copy=True)


def _curves(photoEyes):
    if not isinstance(photoEyes, dict) or set(photoEyes) != set(EYE_SPECS):
        raise ValueError('Expected exactly the original imageLeft and imageRight dense photo eyes')
    result = {}
    for name in EYE_SPECS:
        eye = photoEyes[name]
        if not isinstance(eye, dict) or set(eye) != {'upper', 'lower'}:
            raise ValueError('Each dense eye must provide upper and lower curves')
        result[name] = {}
        for lid in ('upper', 'lower'):
            curve = _finitePoints(eye[lid], f'{name}.{lid}')
            if not 3 <= len(curve) <= MAX_CURVE_POINTS:
                raise ValueError('Each dense curve requires 3 to 4096 original ordered samples')
            result[name][lid] = curve
    return result


def _axis(eye, name):
    upper, lower = eye['upper'], eye['lower']
    first = .5*upper[0]+.5*lower[0]
    last = .5*upper[-1]+.5*lower[-1]
    width = float(np.linalg.norm(last-first))
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f'{name}: dense canthus axis is degenerate')
    tangent = (last-first)/width
    normal = np.array([-tangent[1], tangent[0]])
    curves = {}
    for lid in ('upper', 'lower'):
        normalized = (eye[lid]-first)/width
        along, height = normalized @ tangent, normalized @ normal
        if not np.isfinite(along).all() or not np.isfinite(height).all():
            raise ValueError(f'{name}: photo points cannot be resolved at the dense eye scale')
        if np.any(np.diff(along) <= AXIS_TOLERANCE):
            raise ValueError(f'{name}.{lid}: dense axis coordinates must be strictly increasing; no sorting or repair')
        curves[lid] = dict(along=along, normal=height)
    start = max(curves[lid]['along'][0] for lid in curves)
    stop = min(curves[lid]['along'][-1] for lid in curves)
    if stop-start <= AXIS_TOLERANCE:
        raise ValueError(f'{name}: upper and lower curves have no common axis interval')
    knots = np.unique(np.r_[start, stop,
        curves['upper']['along'][(curves['upper']['along'] > start) & (curves['upper']['along'] < stop)],
        curves['lower']['along'][(curves['lower']['along'] > start) & (curves['lower']['along'] < stop)]])
    separation = (np.interp(knots, curves['lower']['along'], curves['lower']['normal'])
                  - np.interp(knots, curves['upper']['along'], curves['upper']['normal']))
    # The difference is linear between the union of both sets of knots, so
    # checking these values excludes a hidden crossing between samples.
    endpoint = (np.abs(knots) <= AXIS_TOLERANCE) | (np.abs(knots-1) <= AXIS_TOLERANCE)
    if np.any(separation < -AXIS_TOLERANCE) or np.any(separation[~endpoint] <= AXIS_TOLERANCE):
        raise ValueError(f'{name}: reversed upper/lower semantics, intersecting lids or an interior touch')
    if not np.any(separation > AXIS_TOLERANCE):
        raise ValueError(f'{name}: dense eye opening is degenerate')
    return dict(origin=first, end=last, widthPixels=width, tangent=tangent, normal=normal,
        curves=curves, overlapKnots=knots, separationAtKnots=separation,
        endpointGapPixels=np.array([np.linalg.norm(upper[0]-lower[0]), np.linalg.norm(upper[-1]-lower[-1])]))


def _identity(source, original, targets):
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    for name in EYE_SPECS:
        for lid in ('upper', 'lower'):
            value = np.ascontiguousarray(source[name][lid], dtype=np.float64)
            digest.update((name+'/'+lid+str(value.shape)).encode()); digest.update(value.tobytes())
    for value in (original, targets):
        value = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def alignEyeLandmarkTargets(photoEyes, eyeLandmarks):
    """Build the sole mapped table for original photo iBUG points 36..47.

    ``eyeLandmarks`` must be the untouched photo's 12 ordered eye points.
    No reconstructed geometry or projected candidate positions are accepted.
    Dense sample order is semantic and never sorted/reversed automatically.
    Translation, positive scale and proper 2D rotation commute with this map;
    mirrored data require correctly relabeled iBUG/dense observations first.

    Strict axis monotonicity and lid separation use 64 float64 eps in units of
    eye width. Interior target-domain checks are exact and never clamp or
    extrapolate. At corners only, both along-axis and normal coordinates may
    change to the midpoint of the corresponding upper/lower dense endpoints.
    """
    source = _curves(photoEyes)
    original = _finitePoints(eyeLandmarks, 'eyeLandmarks', 12)
    targets = original.copy(); axes = {}; pointEvidence = [None]*12
    for name, spec in EYE_SPECS.items():
        axis = _axis(source[name], name); axes[name] = axis
        first, width, tangent, normal = (axis[key] for key in ('origin', 'widthPixels', 'tangent', 'normal'))
        relative = (original-first)/width
        along, height = relative @ tangent, relative @ normal
        if not np.isfinite(along).all() or not np.isfinite(height).all():
            raise ValueError(f'{name}: original photo points cannot be resolved at the dense eye scale')
        for lid in ('upper', 'lower'):
            ordered = (spec['left'], *spec[lid], spec['right'])
            if np.any(np.diff(along[np.asarray(ordered)-36]) <= AXIS_TOLERANCE):
                raise ValueError(f'{name}.{lid}: original iBUG point order conflicts with the dense canthus axis')
            curve = axis['curves'][lid]
            for ibugId in spec[lid]:
                row = ibugId-36; s = float(along[row])
                if not 0 < s < 1 or s < curve['along'][0] or s > curve['along'][-1]:
                    raise ValueError(f'iBUG {ibugId}: original axis coordinate lies outside the dense curve; no clamping')
                segment = min(int(np.searchsorted(curve['along'], s, side='right'))-1, len(curve['along'])-2)
                # The exact-domain checks above guarantee a real segment;
                # this endpoint convention chooses the last segment at its end.
                if segment < 0: segment = 0
                weight = (s-curve['along'][segment])/(curve['along'][segment+1]-curve['along'][segment])
                targetHeight = float((1-weight)*curve['normal'][segment]+weight*curve['normal'][segment+1])
                targets[row] = original[row]+width*(targetHeight-height[row])*normal
                pointEvidence[row] = dict(ibugId=ibugId, eye=name, role=lid,
                    originalAlong=s, targetAlong=s, originalNormal=float(height[row]), targetNormal=targetHeight,
                    alongDeltaPixels=0., normalDeltaPixels=float(width*(targetHeight-height[row])),
                    sourceSegment=np.array([segment, segment+1], np.int64), segmentFraction=float(weight),
                    canthusAlongException=False)
        for ibugId, endpointIndex in ((spec['left'], 0), (spec['right'], 1)):
            row = ibugId-36
            targets[row] = axis['origin'] if endpointIndex == 0 else axis['end']
            pointEvidence[row] = dict(ibugId=ibugId, eye=name, role='canthus',
                originalAlong=float(along[row]), targetAlong=float(endpointIndex),
                originalNormal=float(height[row]), targetNormal=0.,
                alongDeltaPixels=float(width*(endpointIndex-along[row])), normalDeltaPixels=float(-width*height[row]),
                sourceEndpoint=endpointIndex, endpointPolicy='midpoint of matching dense upper/lower endpoints',
                canthusAlongException=True)
    if not np.isfinite(targets).all(): raise ValueError('Mapped eye observations became nonfinite')
    evidence = dict(source='Frozen dense photo curves and original photo iBUG points only; not anatomical truth',
        sourcePhotoEyes=deepcopy(source), originalTargets=original.copy(), mappedTargets=targets.copy(),
        axes=axes, points=pointEvidence, monotonicityToleranceInEyeWidths=AXIS_TOLERANCE,
        upperLowerSeparationToleranceInEyeWidths=AXIS_TOLERANCE,
        targetDomainPolicy='Strict original axis coordinate inside (0,1) and its actual lid interval; no clamp/extrapolation',
        cornersAreAlongAxisExceptions=True, geometryUsed=False, candidateProjectionUsed=False,
        sourceSampleOrderChanged=False, parametersSelectedFromCases=False)
    return dict(schemaVersion=SCHEMA_VERSION, ibugIds=np.arange(36, 48, dtype=np.int64),
        originalTargetPixels=original.copy(), targetPixels=targets, evidence=evidence,
        tableSha256=_identity(source, original, targets))


def _same(first, second):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict) or isinstance(second, dict):
        return isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys() and all(_same(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        return isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)) and len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    return first == second


def _coarseGroups(coarseGroups):
    if not isinstance(coarseGroups, (tuple, list)) or len(coarseGroups) != 2:
        raise ValueError('Original coarse groups are required for both eyes')
    groups = deepcopy(coarseGroups); seen = set()
    names = {spec['group']: spec for spec in EYE_SPECS.values()}
    for group in groups:
        if not isinstance(group, dict) or group.get('name') not in names or group['name'] in seen:
            raise ValueError('Coarse groups must have distinct imageLeftEye/imageRightEye names')
        if not {'ibugIds', 'targetPixels'} <= group.keys():
            raise ValueError('Coarse groups must retain original iBUG IDs and photo targets')
        seen.add(group['name']); spec = names[group['name']]
        ids = np.asarray(group['ibugIds']); targets = _finitePoints(group['targetPixels'], 'coarse targetPixels')
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= len(ids) <= 6
                or len(np.unique(ids)) != len(ids) or not np.isin(ids, np.arange(spec['left'], spec['left']+6)).all()
                or len(targets) != len(ids)):
            raise ValueError('Coarse iBUG IDs must be distinct original observations of the named eye')
        for key in ('weights', 'confidence'):
            if key not in group: continue
            value = np.asarray(group[key])
            if (value.dtype.kind not in 'fiu' or value.shape != (len(ids),) or not np.isfinite(value).all()
                    or (value < 0).any() or key == 'weights' and (value == 0).any()
                    or key == 'confidence' and (value > 1).any()):
                raise ValueError('Original weights/confidence are invalid')
        if ('vertexIndices' in group) != ('barycentric' in group):
            raise ValueError('Triangle indices and barycentric weights must be supplied together')
        if 'vertexIndices' in group:
            indices, bary = np.asarray(group['vertexIndices']), np.asarray(group['barycentric'])
            if (indices.shape != (len(ids), 3) or indices.dtype.kind not in 'iu' or (indices < 0).any()
                    or bary.shape != indices.shape or bary.dtype.kind not in 'fiu' or not np.isfinite(bary).all()
                    or (bary < -1e-7).any() or (bary > 1+1e-7).any() or not np.allclose(bary.sum(1), 1., atol=1e-7, rtol=0)):
                raise ValueError('Original triangle bindings are invalid')
    return groups


def alignCoarseEyeTargets(photoEyes, coarseGroups, *, mappedTable=None, eyeLandmarks=None):
    """Index the same authoritative 12-point table without rebinding geometry.

    A subset of eligible coarse points requires either ``mappedTable`` from
    :func:`alignEyeLandmarkTargets` or the complete original ``eyeLandmarks``.
    Omitting both is supported only when the two groups contain all 12 points.
    A supplied table is authenticated by rebuilding from its original photo
    points, never from coarse subsets or any candidate geometry.
    """
    groups = _coarseGroups(coarseGroups)
    if mappedTable is not None and eyeLandmarks is not None:
        raise ValueError('Supply either the unique mapped table or the complete original photo eye points')
    if mappedTable is not None:
        if not isinstance(mappedTable, dict) or 'originalTargetPixels' not in mappedTable:
            raise ValueError('Expected the complete authenticated eye-observation table')
        table = alignEyeLandmarkTargets(photoEyes, mappedTable['originalTargetPixels'])
        if not _same(table, mappedTable): raise ValueError('Mapped table is stale or differs from its original photo sources')
    else:
        if eyeLandmarks is None:
            original = np.empty((12, 2), float); available = np.zeros(12, bool)
            for group in groups:
                rows = np.asarray(group['ibugIds'], int)-36
                original[rows] = group['targetPixels']; available[rows] = True
            if not available.all():
                raise ValueError('Coarse subsets require the unique full 12-point table or original eyeLandmarks')
            eyeLandmarks = original
        table = alignEyeLandmarkTargets(photoEyes, eyeLandmarks)
    references = []
    for group in groups:
        rows = np.asarray(group['ibugIds'], int)-36
        if not np.array_equal(group['targetPixels'], table['originalTargetPixels'][rows]):
            raise ValueError('Coarse targets do not reference the same original 12 photo observations')
        originalTargets = np.asarray(group['targetPixels']).copy()
        group['targetPixels'] = table['targetPixels'][rows].copy()
        references.append(dict(name=group['name'], ibugIds=np.asarray(group['ibugIds']).copy(),
            tableRows=rows.copy(), originalTargets=originalTargets, mappedTargets=group['targetPixels'].copy()))
    return dict(coarseGroups=groups, mappedTable=deepcopy(table),
        evidence=dict(tableSha256=table['tableSha256'], groups=references,
            fieldsChanged=['targetPixels'], originalBindingsWeightsConfidenceAndIdsPreserved=True,
            observationSelectionChanged=False, geometryUsed=False))
