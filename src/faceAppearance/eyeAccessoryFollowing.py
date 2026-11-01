"""Frozen skin displacement bindings for explicitly labelled eye accessories.

Bindings are made once in the unchanged client template. Visible material-2
rim samples define separate upper/lower source face domains; no subject photo,
optimized vertex or candidate score participates. A subject's completed
assembly remains the rest state: accessories receive only the barycentric
skin displacement, with no independent parameters and no re-placement.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
from scipy import sparse

from .anatomicalBoundaries import meshTopologyHash
from .eyeSocketVisibility import buildVisibleEyeRimBinding


DEFAULT_POLICY = dict(rimResolution=192, innerCornerWidthFraction=.30,
    maximumBindingDistanceEyeWidthFraction=.60, ambiguityToleranceEyeWidthFraction=1e-10)


def _identity(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(b'array\0'+str(item.dtype).encode()+repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()+b'\0'); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b'list\0')
            for entry in item: visit(entry)
        elif isinstance(item, np.generic): visit(item.item())
        else: digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode()+b'\0')
    visit(value)
    return digest.hexdigest()


def _indices(values, count, name, *, nonempty=True):
    result = np.asarray(values)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or (nonempty and not len(result))
            or (len(result) and (result.min() < 0 or result.max() >= count))
            or len(np.unique(result)) != len(result)):
        raise ValueError(f'{name} must contain distinct valid integer indices')
    return result.astype(np.int64, copy=True)


def _policy(policy):
    result = DEFAULT_POLICY.copy()
    if policy is not None:
        if not isinstance(policy, dict) or set(policy)-set(result):
            raise ValueError('Unknown eye accessory binding policy')
        result.update(policy)
    if type(result['rimResolution']) is not int or not 64 <= result['rimResolution'] <= 1024:
        raise ValueError('Invalid template rim resolution')
    for key in tuple(result)[1:]:
        value = result[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or not 0 < value < 1):
            raise ValueError('Accessory policy fractions must be finite and in (0, 1)')
        result[key] = float(value)
    if result['innerCornerWidthFraction'] >= .5:
        raise ValueError('Inner corner source domain must exclude the outer half')
    return result


def _closestOnTriangles(point, triangles):
    """Closed triangle projection, including all edge and vertex regions."""
    first = triangles[:, 0]
    u, v = triangles[:, 1]-first, triangles[:, 2]-first
    uu, uv, vv = (u*u).sum(1), (u*v).sum(1), (v*v).sum(1)
    determinant = uu*vv-uv*uv
    if np.any(determinant <= np.maximum(uu*vv, 1e-300)*1e-14):
        raise ValueError('Degenerate source triangle in accessory binding domain')
    w = point-first
    wu, wv = (w*u).sum(1), (w*v).sum(1)
    b1 = (wu*vv-wv*uv)/determinant
    b2 = (wv*uu-wu*uv)/determinant
    plane = np.stack([1-b1-b2, b1, b2], 1)
    candidates = [plane]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, b]-triangles[:, a]
        parameter = np.clip(((point-triangles[:, a])*edge).sum(1)/(edge*edge).sum(1), 0., 1.)
        bary = np.zeros_like(plane); bary[:, a] = 1-parameter; bary[:, b] = parameter
        candidates.append(bary)
    weights = np.stack(candidates, 1)
    closest = (triangles[:, None]*weights[..., None]).sum(2)
    squared = ((closest-point)**2).sum(2)
    squared[:, 0] = np.where((plane >= 0).all(1), squared[:, 0], np.inf)
    chosen = squared.argmin(1)
    return weights[np.arange(len(triangles)), chosen], closest[np.arange(len(triangles)), chosen], np.sqrt(squared[np.arange(len(triangles)), chosen])


def _equivalentRows(firstIds, firstWeights, secondIds, secondWeights, tolerance):
    ids = np.union1d(firstIds, secondIds)
    first = np.zeros(len(ids)); second = first.copy()
    first[np.searchsorted(ids, firstIds)] = firstWeights
    second[np.searchsorted(ids, secondIds)] = secondWeights
    return bool(np.max(np.abs(first-second)) <= tolerance)


def _componentIds(component, vertices, faces, materials):
    ids = _indices(component['vertexIndices'], len(vertices), 'component vertices')
    faceIds = _indices(component['faceIndices'], len(faces), 'component faces')
    if (not np.array_equal(np.unique(faces[faceIds]), np.sort(ids))
            or not np.all(materials[faceIds] == component['materialId'])):
        raise ValueError('Component metadata must exactly cover its numbered faces and material')
    return ids, faceIds


def buildEyeAccessoryBinding(templateArrays, components, *, policy=None):
    """Bind two inner canthi and four labelled eyelash sheets to template skin.

    The template contract is client centimeters, Y up, Z front. The existing
    ``headAccessory`` label alone is insufficient: each such component must be
    on the inner half of exactly one original spherical eye and close to that
    eye's visible inner-corner skin. Two upper and two lower eyelash components
    are required. Eyebrows, eye caps, teeth and every other component are fixed.
    """
    effective = _policy(policy)
    vertices = np.asarray(templateArrays['vertices'], np.float64)
    faces = np.asarray(templateArrays['faces'])
    materials = np.asarray(templateArrays['materialIds'])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
            or materials.shape != (len(faces),) or materials.dtype.kind not in 'iu'):
        raise ValueError('Invalid original client template geometry')
    skinIds = _indices(templateArrays['nicpToFullIds'], len(vertices), 'template skin vertices')
    skinMask = np.zeros(len(vertices), bool); skinMask[skinIds] = True
    components = deepcopy(list(components))
    if len({part['name'] for part in components}) != len(components):
        raise ValueError('Component names must be unique')
    eyeParts = [part for part in components if part.get('label') in ('leftEye', 'rightEye')]
    selected = [part for part in components if part.get('label') in ('headAccessory', 'upperEyelash', 'lowerEyelash')]
    if (len(eyeParts) != 2 or {part['label'] for part in eyeParts} != {'leftEye', 'rightEye'}
            or len(selected) != 6 or any(sum(part['label'] == label for part in selected) != 2
            for label in ('headAccessory', 'upperEyelash', 'lowerEyelash'))):
        raise ValueError('Expected exactly two eyes, two inner head accessories and four eyelash sheets')
    eyeParts.sort(key=lambda part: vertices[part['vertexIndices'], 0].mean())
    eyeCenters = np.asarray([vertices[part['vertexIndices']].mean(0) for part in eyeParts])
    centerX = float(eyeCenters[:, 0].mean())
    if not eyeCenters[0, 0] < centerX < eyeCenters[1, 0]:
        raise ValueError('Template eye sides are ambiguous')
    eyes = []
    for eyeIndex, part in enumerate(eyeParts):
        _componentIds(part, vertices, faces, materials)
        rim = buildVisibleEyeRimBinding(templateArrays, part, resolution=effective['rimResolution'])
        mask = rim['eyeMask']; columns = np.rint(rim['contourSamplePixels'][:, 0]).astype(np.int64)
        if not mask[:, columns].any(0).all():
            raise ValueError('Visible contour sample does not address an occupied eye-mask column')
        upperY = mask[:, columns].argmax(0)
        lowerY = mask.shape[0]-1-mask[::-1, columns].argmax(0)
        contourY = rim['contourSamplePixels'][:, 1]
        upperDistance = np.abs(contourY-upperY); lowerDistance = np.abs(contourY-lowerY)
        upper = upperDistance < lowerDistance; lower = lowerDistance < upperDistance
        upperFaces = np.unique(rim['fullFaceIds'][upper]); lowerFaces = np.unique(rim['fullFaceIds'][lower])
        if not len(upperFaces) or not len(lowerFaces) or len(np.intersect1d(upperFaces, lowerFaces)):
            raise ValueError('Upper/lower template source face domains overlap or are empty')
        allFaces = np.union1d(upperFaces, lowerFaces)
        sourceX = vertices[faces[allFaces], 0]
        sign = -1 if eyeIndex == 0 else 1
        if not (skinMask[faces[allFaces]].all() and ((sourceX-centerX)*sign > 0).all()):
            raise ValueError('Visible eye rim contains a non-skin or cross-side source triangle')
        restX = rim['restPoints'][:, 0]; minimum, maximum = float(restX.min()), float(restX.max())
        width = maximum-minimum
        if width <= 0: raise ValueError('Degenerate visible template eye width')
        innerDistance = maximum-restX if eyeIndex == 0 else restX-minimum
        inner = innerDistance <= effective['innerCornerWidthFraction']*width
        innerFaces = np.unique(rim['fullFaceIds'][inner & (upper | lower)])
        if not len(innerFaces): raise ValueError('Inner-corner source face domain is empty')
        eyes.append(dict(name=part['label'], componentName=part['name'], sideSign=sign,
            center=eyeCenters[eyeIndex], visibleEyeWidth=width, visibleRimXBounds=np.array([minimum, maximum]),
            sourceFaceDomains=dict(upper=upperFaces, lower=lowerFaces, innerCorner=innerFaces),
            visibleRimEvidence=deepcopy(rim), columnIndices=columns, upperMaskRows=upperY, lowerMaskRows=lowerY,
            upperDistances=upperDistance, lowerDistances=lowerDistance, upperSampleMask=upper,
            lowerSampleMask=lower, innerCornerSampleMask=inner, innerCornerDistances=innerDistance))
    records, seen, assignmentCounts = [], set(), {}
    for part in sorted(selected, key=lambda row: row['name']):
        ids, faceIds = _componentIds(part, vertices, faces, materials)
        if any(int(index) in seen for index in ids) or skinMask[ids].any():
            raise ValueError('Accessory components overlap each other or original skin')
        seen.update(map(int, ids))
        positions = vertices[ids]
        if (positions[:, 0] < centerX).all(): eyeIndex = 0
        elif (positions[:, 0] > centerX).all(): eyeIndex = 1
        else: raise ValueError('Accessory straddles the two eye sides')
        eye = eyes[eyeIndex]; label = part['label']
        expectedMaterial = 0 if label == 'headAccessory' else 3
        if part['materialId'] != expectedMaterial: raise ValueError('Accessory material does not match its declared role')
        role = {'headAccessory': 'innerCorner', 'upperEyelash': 'upper', 'lowerEyelash': 'lower'}[label]
        key = (eyeIndex, role); assignmentCounts[key] = assignmentCounts.get(key, 0)+1
        if assignmentCounts[key] != 1: raise ValueError('Duplicate accessory role on one eye side')
        if role == 'innerCorner' and not ((positions[:, 0]-eye['center'][0])*eye['sideSign'] < 0).all():
            raise ValueError('Head accessory does not lie wholly on the inner side of its eye')
        domain = eye['sourceFaceDomains'][role]
        triangles = vertices[faces[domain]]
        weights, chosenIds, closestPoints, distances, tiedIds = [], [], [], [], []
        tolerance = effective['ambiguityToleranceEyeWidthFraction']*eye['visibleEyeWidth']
        for point in positions:
            bary, closest, distance = _closestOnTriangles(point, triangles)
            nearest = float(distance.min()); tied = np.flatnonzero(distance <= nearest+tolerance)
            chosen = int(tied[0])
            for other in tied[1:]:
                if not _equivalentRows(faces[domain[chosen]], bary[chosen], faces[domain[other]], bary[other],
                        effective['ambiguityToleranceEyeWidthFraction']):
                    raise ValueError('Accessory has ambiguous equally near source skin triangles')
            if nearest > effective['maximumBindingDistanceEyeWidthFraction']*eye['visibleEyeWidth']:
                raise ValueError('Accessory source binding exceeds the template eye-width distance bound')
            weights.append(bary[chosen]); chosenIds.append(domain[chosen]); closestPoints.append(closest[chosen])
            distances.append(nearest); tiedIds.append(domain[tied].copy())
        chosenIds = np.asarray(chosenIds, np.int64)
        records.append(dict(componentName=part['name'], label=label, materialId=part['materialId'],
            eyeName=eye['name'], eyeIndex=eyeIndex, role=role, targetVertexIds=ids, targetFaceIds=faceIds,
            allowedFaceIds=domain.copy(), sourceFaceIds=chosenIds, sourceVertexIndices=faces[chosenIds].copy(),
            barycentricWeights=np.asarray(weights), closestRestPoints=np.asarray(closestPoints),
            bindingDistances=np.asarray(distances), equivalentTieFaceIds=tiedIds))
    if set(assignmentCounts) != {(eye, role) for eye in range(2) for role in ('innerCorner', 'upper', 'lower')}:
        raise ValueError('Not every eye has exactly one component for each following role')
    templateHashes = {key: _identity(np.asarray(templateArrays[key])) for key in templateArrays}
    result = dict(schemaVersion='templateEyeAccessoryBindingV1', vertexCount=len(vertices),
        templateVertices=vertices.copy(), faces=faces.astype(np.int64, copy=True),
        skinVertexIds=skinIds, components=records, eyeDomains=eyes, sideDivisionX=centerX,
        followerVertexIds=np.asarray(sorted(seen), np.int64), policy=effective,
        templateArraySha256=templateHashes, templateComponentsSha256=_identity(components),
        topologySha256=meshTopologyHash(len(vertices), faces),
        coordinateFrame='originalClientTemplateCentimetersYUpZForward',
        diagnostics=dict(candidateDependent=False, photoDependent=False, additionalDegreesOfFreedom=0,
            targetRoles=['innerCorner', 'upperEyelash', 'lowerEyelash'], otherAccessoriesFixed=True,
            source='frozen template visible material-2 rim adjacent skin triangles; anatomical proxy'))
    result['bindingSha256'] = _identity(result)
    return result


def _validateBinding(binding):
    if (binding.get('schemaVersion') != 'templateEyeAccessoryBindingV1'
            or binding.get('bindingSha256') != _identity({key: value for key, value in binding.items() if key != 'bindingSha256'})):
        raise ValueError('Eye accessory template binding changed')
    count = binding['vertexCount']; skin = _indices(binding['skinVertexIds'], count, 'binding skin')
    followers = _indices(binding['followerVertexIds'], count, 'binding followers')
    if len(np.intersect1d(skin, followers)): raise ValueError('Followers must not be skin')
    rows = []
    for part in binding['components']:
        ids = _indices(part['targetVertexIds'], count, 'following component')
        sources = np.asarray(part['sourceVertexIndices']); weights = np.asarray(part['barycentricWeights'])
        if (sources.shape != (len(ids), 3) or sources.dtype.kind not in 'iu'
                or not np.isin(sources, skin).all() or weights.shape != sources.shape
                or not np.isfinite(weights).all() or (weights < 0).any()
                or not np.allclose(weights.sum(1), 1., atol=1e-12, rtol=0)
                or not np.array_equal(binding['faces'][part['sourceFaceIds']], sources)
                or not np.isin(part['sourceFaceIds'], part['allowedFaceIds']).all()):
            raise ValueError('Invalid numbered skin triangle barycentric following rows')
        rows.extend(ids.tolist())
    if len(set(rows)) != len(rows) or not np.array_equal(np.sort(rows), followers):
        raise ValueError('Following component rows do not cover the exact whitelist')


def prepareEyeAccessoryFollowing(binding, baselineVertices, faces, skinVertexIds,
                                  originalFixedVertexIds, *, movableSkinVertexIds=None):
    """Freeze a template binding against one completed full-assembly baseline."""
    _validateBinding(binding)
    baseline = np.asarray(baselineVertices, np.float64)
    count = binding['vertexCount']; faces = np.asarray(faces)
    skin = _indices(skinVertexIds, count, 'skinVertexIds')
    fixed = _indices(originalFixedVertexIds, count, 'originalFixedVertexIds', nonempty=False)
    if (baseline.shape != (count, 3) or not np.isfinite(baseline).all()
            or faces.dtype.kind not in 'iu' or not np.array_equal(faces, binding['faces'])
            or not np.array_equal(skin, binding['skinVertexIds'])):
        raise ValueError('Completed assembly must preserve the original numbered template topology and skin order')
    nonSkin = np.setdiff1d(np.arange(count), skin)
    if not np.isin(nonSkin, fixed).all(): raise ValueError('Original residual support must fix every non-skin vertex')
    movable = np.setdiff1d(skin, fixed) if movableSkinVertexIds is None else _indices(
        movableSkinVertexIds, count, 'movableSkinVertexIds', nonempty=False)
    if (not np.isin(movable, skin).all() or len(np.intersect1d(movable, fixed))
            or not np.array_equal(np.sort(movable), np.setdiff1d(skin, fixed))):
        raise ValueError('Following cannot change or expand the original movable skin support')
    result = dict(schemaVersion='completedEyeAccessoryFollowingV1', binding=deepcopy(binding),
        baselineVertices=baseline.copy(), faces=faces.astype(np.int64, copy=True), skinVertexIds=skin,
        movableSkinVertexIds=movable, originalFixedVertexIds=fixed,
        followerVertexIds=binding['followerVertexIds'].copy(),
        effectiveFixedVertexIds=np.setdiff1d(fixed, binding['followerVertexIds']),
        diagnostics=dict(independentAccessoryParameters=0, skinSupportUnchanged=True,
            baselinePlacementUnchanged=True, mapping='A0 + B * (skinCandidate - skinBaseline)',
            ignoreCandidateAccessoryCoordinates=True, productionDefaultsChanged=False))
    result['specSha256'] = _identity(result)
    return result


def validateEyeAccessoryFollowingSpec(spec, *, baselineVertices=None, faces=None,
                                      skinVertexIds=None, originalFixedVertexIds=None):
    """Check immutable identity at solver entry; application itself is cheap."""
    if (spec.get('schemaVersion') != 'completedEyeAccessoryFollowingV1'
            or spec.get('specSha256') != _identity({key: value for key, value in spec.items() if key != 'specSha256'})):
        raise ValueError('Eye accessory following specification changed')
    _validateBinding(spec['binding'])
    rebuilt = prepareEyeAccessoryFollowing(spec['binding'], spec['baselineVertices'], spec['faces'],
        spec['skinVertexIds'], spec['originalFixedVertexIds'], movableSkinVertexIds=spec['movableSkinVertexIds'])
    if rebuilt['specSha256'] != spec['specSha256']:
        raise ValueError('Inconsistent eye accessory following specification')
    for name, value in dict(baselineVertices=baselineVertices, faces=faces,
            skinVertexIds=skinVertexIds, originalFixedVertexIds=originalFixedVertexIds).items():
        if value is not None and not np.array_equal(np.asarray(value), spec[name]):
            raise ValueError(f'Eye accessory following {name} does not match solver inputs')
    return True


def eyeAccessoryDisplacementOperator(spec):
    """Return scalar CSR F; vertex-major XYZ Jacobian is kron(F, eye(3)).

    Skin rows are identity. Every follower row contains only its three frozen
    skin weights; its own input column is zero. All other accessory rows are
    zero. Call ``validateEyeAccessoryFollowingSpec`` once before using a spec.
    """
    skin = spec['skinVertexIds']; rows = [skin]; columns = [skin]; values = [np.ones(len(skin))]
    for part in spec['binding']['components']:
        rows.append(np.repeat(part['targetVertexIds'], 3)); columns.append(part['sourceVertexIndices'].ravel())
        values.append(part['barycentricWeights'].ravel())
    count = len(spec['baselineVertices'])
    operator = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(count, count))
    operator.eliminate_zeros(); operator.sort_indices()
    return operator


def applyEyeAccessoryFollowing(spec, fullVertices):
    """Apply total skin displacement to the completed baseline, without drift.

    Entry validation belongs to the caller; this function deliberately avoids
    hashing or rebuilding template domains during each solver trial. Even if
    the input already contains following accessories, their coordinates are
    ignored. Changes to fixed skin or any other accessory fail explicitly.
    Zero skin displacement returns bit-exact baseline coordinates.
    """
    candidate = np.asarray(fullVertices, np.float64); baseline = spec['baselineVertices']
    if candidate.shape != baseline.shape or not np.isfinite(candidate).all():
        raise ValueError('Expected finite full-scene candidate vertices')
    fixed = spec['effectiveFixedVertexIds']
    if not np.array_equal(candidate[fixed], baseline[fixed]):
        raise ValueError('Candidate changed an original fixed vertex outside the accessory whitelist')
    result = candidate.copy(); skin = spec['skinVertexIds']
    result[spec['followerVertexIds']] = baseline[spec['followerVertexIds']]
    for part in spec['binding']['components']:
        sources = part['sourceVertexIndices']
        delta = ((candidate[sources]-baseline[sources])*part['barycentricWeights'][..., None]).sum(1)
        ids = part['targetVertexIds']; changed = np.any(delta != 0, axis=1)
        result[ids[changed]] = baseline[ids[changed]]+delta[changed]
    return result
