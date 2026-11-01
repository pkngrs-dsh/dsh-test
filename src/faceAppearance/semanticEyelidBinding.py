"""Fixed legacy eyelid semantic candidates, independent of visible eye masks.

These are 32 existing MediaPipe surface samples, not an anatomical annotation
or a continuous surface path. Upper and lower sample sequences share their
original corner rows and run from the subject's inner to outer corner. No
eyeball, photo, rasterization, nearest-surface search or fitting is involved.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .clientLandmarks import loadClientLandmarks


SCHEMA_VERSION = 'fixedLegacySemanticEyelidCandidateV1'
REVIEW_STATUS = 'unreviewedLegacySemanticCandidate'
EYE_PATHS = {
    'subjectRight': {
        'upper': (133, 173, 157, 158, 159, 160, 161, 246, 33),
        'lower': (133, 155, 154, 153, 145, 144, 163, 7, 33),
    },
    'subjectLeft': {
        'upper': (362, 398, 384, 385, 386, 387, 388, 466, 263),
        'lower': (362, 382, 381, 380, 374, 373, 390, 249, 263),
    },
}
_ARRAY_FIELDS = ('mpIds', 'faceIds', 'vertexIndices', 'barycentricWeights')
_ROLES = ('upper', 'lower', 'innerCorner', 'outerCorner')


def _identity(binding):
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    for key in ('sourceSha256', 'nicpFacesHash', 'reviewStatus'):
        digest.update(str(binding[key]).encode())
    for eyeName in EYE_PATHS:
        for role in _ROLES:
            digest.update((eyeName+'/'+role).encode())
            for key in _ARRAY_FIELDS:
                dtype = '<f8' if key == 'barycentricWeights' else '<i8'
                value = np.ascontiguousarray(binding['eyes'][eyeName][role][key], dtype=dtype)
                digest.update((key+str(value.shape)).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        for item in value.values():
            _freeze(item)
    elif isinstance(value, np.ndarray):
        value.setflags(write=False)
    return value


def buildSemanticEyelidBinding(nicpFaces, landmarkBinding=None):
    """Select unchanged original 468-table rows in explicit semantic order.

    Numbered NICP topology and bundled source identity are checked by the
    existing loader. An optional loaded table must exactly reproduce that
    source; this function does not accept an unreviewed replacement mapping.
    The returned ``eyes`` contain two nine-sample sequences per eye, sharing
    both endpoints, hence 32 unique samples overall. Connecting their sampled
    positions with display lines does not establish a path on the mesh.
    """
    original = loadClientLandmarks(nicpFaces)
    if landmarkBinding is not None:
        if not isinstance(landmarkBinding, dict):
            raise ValueError('Expected the original loaded client landmark table')
        for key in ('sourceSha256', 'nicpFacesHash', 'nicpVertexCount'):
            if landmarkBinding.get(key) != original[key]:
                raise ValueError('Supplied landmark provenance differs from the bundled source')
        for key in ('landmarkIds', 'mediapipeIds', 'faceIds', 'vertexIndices',
                    'barycentricWeights', 'validMask'):
            if key not in landmarkBinding or not np.array_equal(landmarkBinding[key], original[key]):
                raise ValueError('Supplied landmark rows differ from the bundled source')

    def select(ids):
        ids = np.asarray(ids, np.int64)
        if not original['validMask'][ids].all():
            raise ValueError('A required semantic eyelid sample has invalid barycentric weights')
        return dict(mpIds=ids.copy(), faceIds=original['faceIds'][ids].copy(),
            vertexIndices=original['vertexIndices'][ids].copy(),
            barycentricWeights=original['barycentricWeights'][ids].copy())

    eyes = {}
    for eyeName, paths in EYE_PATHS.items():
        eyes[eyeName] = {role: select(ids) for role, ids in paths.items()}
        eyes[eyeName]['innerCorner'] = select([paths['upper'][0]])
        eyes[eyeName]['outerCorner'] = select([paths['upper'][-1]])
    result = dict(schemaVersion=SCHEMA_VERSION, reviewStatus=REVIEW_STATUS,
        sourcePath=original['sourcePath'], sourceSha256=original['sourceSha256'],
        nicpFacesHash=original['nicpFacesHash'], nicpVertexCount=original['nicpVertexCount'],
        uniqueSampleCount=32, eyes=eyes, sequenceDirection='subjectInnerToOuter',
        coordinateConvention='Caller NICP geometry coordinates; subject sides, no image mirroring',
        surfacePathCreated=False, anatomicalAccuracyValidated=False,
        method='Exact selection of existing legacy MediaPipe triangle and barycentric rows',
        limitations=[
            'Numerically valid legacy semantic candidates, not anatomical ground truth.',
            'No continuous surface path is supplied between the fixed sample locations.',
            'Image eye identity and mirror conventions must be supplied by the caller.',
            'Projection reports no visibility or eye-contact classification.',
        ])
    result['bindingSha256'] = _identity(result)
    return _freeze(result)


def _validateBinding(binding):
    if (not isinstance(binding, dict) or binding.get('schemaVersion') != SCHEMA_VERSION
            or binding.get('reviewStatus') != REVIEW_STATUS
            or binding.get('nicpVertexCount') != 2795
            or binding.get('uniqueSampleCount') != 32
            or binding.get('sequenceDirection') != 'subjectInnerToOuter'
            or binding.get('surfacePathCreated') is not False
            or binding.get('anatomicalAccuracyValidated') is not False
            or not isinstance(binding.get('eyes'), dict)
            or set(binding['eyes']) != set(EYE_PATHS)):
        raise ValueError('Expected an unmodified fixed legacy semantic eyelid candidate')
    for key in ('sourceSha256', 'nicpFacesHash', 'bindingSha256'):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing or malformed semantic binding identity')
    for eyeName, paths in EYE_PATHS.items():
        eye = binding['eyes'][eyeName]
        if not isinstance(eye, dict) or set(eye) != set(_ROLES):
            raise ValueError('Each semantic eye requires upper/lower and both corners')
        expected = dict(paths, innerCorner=(paths['upper'][0],), outerCorner=(paths['upper'][-1],))
        for role, ids in expected.items():
            row = eye[role]
            if not isinstance(row, dict) or set(row) != set(_ARRAY_FIELDS):
                raise ValueError('Invalid semantic surface sample record')
            mpIds, faceIds, indices, weights = (np.asarray(row[key]) for key in _ARRAY_FIELDS)
            if (mpIds.dtype.kind not in 'iu' or not np.array_equal(mpIds, ids)
                    or faceIds.dtype.kind not in 'iu' or faceIds.shape != (len(ids),)
                    or (faceIds < 0).any() or (faceIds >= 5532).any()
                    or indices.dtype.kind not in 'iu' or indices.shape != (len(ids), 3)
                    or (indices < 0).any() or (indices >= 2795).any()
                    or weights.dtype.kind not in 'fiu' or weights.shape != indices.shape
                    or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any()
                    or not np.allclose(weights.sum(1), 1., atol=1e-14, rtol=0)):
                raise ValueError('Semantic sample order or surface binding is invalid')
        for role, index in (('innerCorner', 0), ('outerCorner', -1)):
            for key in _ARRAY_FIELDS:
                corner = np.asarray(eye[role][key])[0]
                if any(not np.array_equal(corner, np.asarray(eye[lid][key])[index]) for lid in ('upper', 'lower')):
                    raise ValueError('Upper and lower eyelids must share the exact original corner rows')
    if _identity(binding) != binding['bindingSha256']:
        raise ValueError('Semantic binding content does not match its frozen identity')


def sampleSemanticEyelids(nicpVertices, binding):
    """Return fixed surface sample positions for each subject eye and role.

    ``upper`` and ``lower`` are (9, 3) arrays; ``innerCorner`` and
    ``outerCorner`` are (3,) arrays. The supplied geometry is not modified.
    Consecutive points are not interpolated or projected back to the surface.
    """
    _validateBinding(binding)
    raw = np.asarray(nicpVertices)
    if (raw.dtype.kind not in 'fiu' or raw.shape != (binding['nicpVertexCount'], 3)
            or not np.isfinite(raw).all()):
        raise ValueError('Expected finite original-order NICP vertices with shape (2795, 3)')
    vertices = raw.astype(np.float64, copy=False)
    result = {}
    for eyeName, eye in binding['eyes'].items():
        result[eyeName] = {}
        for role in _ROLES:
            row = eye[role]
            points = (vertices[row['vertexIndices']]*np.asarray(row['barycentricWeights'])[..., None]).sum(1)
            result[eyeName][role] = points[0] if role.endswith('Corner') else points
    return result


def projectSemanticEyelids(nicpVertices, binding, camera):
    """Project fixed samples with an unchanged supplied camera, without fitting.

    Each eye/role returns ``pixels``, ``depth`` and ``positiveDepthMask``.
    Behind-camera samples have NaN pixels, not a fabricated visible position.
    Positive depth does not establish visibility through the full assembly.
    """
    if not isinstance(camera, dict) or not {'intrinsics', 'rotation', 'translation'} <= camera.keys():
        raise ValueError('Expected explicit camera intrinsics, rotation and translation')
    matrices = {}
    for key, shape in (('intrinsics', (3, 3)), ('rotation', (3, 3)), ('translation', (3,))):
        value = np.asarray(camera[key])
        if value.dtype.kind not in 'fiu' or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Camera arrays must be finite with explicit 3x3/3 shapes')
        matrices[key] = value.astype(np.float64, copy=False)
    if (not np.allclose(matrices['intrinsics'][2], [0., 0., 1.], atol=1e-12, rtol=0)
            or matrices['intrinsics'][0, 0] <= 0 or matrices['intrinsics'][1, 1] <= 0
            or not np.allclose(matrices['rotation'] @ matrices['rotation'].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrices['rotation']), 1., atol=1e-6, rtol=0)):
        raise ValueError('Expected pinhole intrinsics and a proper camera rotation')
    samples = sampleSemanticEyelids(nicpVertices, binding)
    result = {}
    for eyeName, eye in samples.items():
        result[eyeName] = {}
        for role, points in eye.items():
            cameraPoints = points @ matrices['rotation'].T + matrices['translation']
            homogeneous = cameraPoints @ matrices['intrinsics'].T
            positive = cameraPoints[..., 2] > 0
            pixels = np.full(points.shape[:-1]+(2,), np.nan, np.float64)
            np.divide(homogeneous[..., :2], homogeneous[..., 2, None],
                out=pixels, where=np.asarray(positive)[..., None])
            result[eyeName][role] = dict(pixels=pixels, depth=cameraPoints[..., 2],
                positiveDepthMask=positive)
    return result
