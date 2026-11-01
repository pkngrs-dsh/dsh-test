"""Conservative frontal eyelid preferences for a frozen COLMAP mesh.

This changes view-selection scores only. It never warps a photograph, changes
geometry, or turns an invalid view into a valid observation. The caller must
apply the same sparse prior to color, support, and provenance-marker bakes.
Face indices always address the supplied NICP skin mesh in its original order;
eye, corner, lash, and other accessory triangles must not be in that mesh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .eyelidTextureSampling import _curves, _nearestSegments, _outsidePolygon, _sourceSafety
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import trianglePixels, validateTriangles


def _smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def _regionWeights(query, eyes, *, coreWidthRatio, upperWidthRatio, lowerWidthRatio):
    """A broad exterior lid band whose size follows the projected eye width."""
    weight = np.zeros(len(query), np.float64)
    for eye in eyes.values():
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        width = float(np.ptp(polygon[:, 0]))
        exterior = _outsidePolygon(query, polygon, 0.)
        eyeWeight = np.zeros(len(query), np.float64)
        for part, ratio in (('upper', upperWidthRatio), ('lower', lowerWidthRatio)):
            curve = eye[part]
            distance, segment, fraction = _nearestSegments(query, curve[:-1], curve[1:])
            amount = 1. - _smooth((distance / width - coreWidthRatio)
                                  / (ratio - coreWidthRatio))
            # Keep upper/lower widths on their respective side of the opening.
            midY = curve[segment, 1] + fraction * (curve[segment + 1, 1] - curve[segment, 1])
            side = query[:, 1] <= midY if part == 'upper' else query[:, 1] >= midY
            eyeWeight = np.maximum(eyeWeight, amount * side)
        weight = np.maximum(weight, eyeWeight * exterior)
    return weight


def _triangleCells(triangle, imageShape):
    """All closed unit bilinear cells intersecting the projected triangle.

    A separating-axis test includes cells touched only by a sliver or edge.
    Checking all four corners of every retained cell conservatively guarantees
    that every continuous query in the face has valid, exterior-eye bilinear
    support; a centroid-only test would miss eye-white leaks near the vertices.
    """
    height, width = imageShape
    if (not np.isfinite(triangle).all() or (triangle < 0).any()
            or (triangle >= [width - 1, height - 1]).any()):
        return None
    minimum = np.floor(triangle.min(0)).astype(np.int64)
    maximum = np.floor(triangle.max(0)).astype(np.int64)
    yy, xx = np.mgrid[minimum[1]:maximum[1]+1, minimum[0]:maximum[0]+1]
    centers = np.column_stack([xx.ravel(), yy.ravel()]).astype(float) + .5
    edges = np.roll(triangle, -1, axis=0) - triangle
    axes = np.column_stack([-edges[:, 1], edges[:, 0]])
    triangleProjection = triangle @ axes.T
    cellProjection = centers @ axes.T
    radius = .5 * np.abs(axes).sum(1)
    retained = ((cellProjection + radius >= triangleProjection.min(0) - 1e-10)
                & (cellProjection - radius <= triangleProjection.max(0) + 1e-10)).all(1)
    return yy.ravel()[retained], xx.ravel()[retained]


def buildEyelidViewPrior(geometry, camera, sourceValidMask, photoEyes, modelEyes, *,
                       visibleScene=None, strength=2., coreWidthRatio=.10,
                       upperWidthRatio=.45, lowerWidthRatio=.30,
                       sourceSafetyPixels=.75, minCosNormalAngle=.10, frontImageId=1):
    """Return per-face frontal selection preferences plus diagnostic maps.

    ``geometry`` is the frozen canonical NICP mesh (vertices/faces); ``camera``
    is OpenCV world-to-camera. Both eye dictionaries use imageLeft/imageRight,
    upper/lower left-to-right Nx2 curves in working-image pixels. ``photoEyes``
    must be independently accepted frontal photo observations. ``modelEyes``
    describes the actual visible opening of this exact frozen model.

    ``visibleScene`` should include every opaque accessory in canonical space.
    MASK cards can be excluded consistently with the existing local texture
    experiments; COLMAP's own eligibility tests still run after this prior.

    Every non-unit score multiplier passes these independent gates: positive
    depth and facing; all bilinear cells touched by the whole projected face
    are valid and outside both photographic eye openings; continuous depth
    queries on seven interior samples and all covered photo pixel centers
    agree with the complete opaque scene. No sampling coordinate is moved.

    The returned ``eyelidWeight`` is an unqualified semantic band weight;
    ``priorWeight`` is that weight after gates, in [0,1]. ``frontEligible``
    is local to nonzero-band candidates, not a full-head visibility mask.
    ``scoreMultiplier = 1 + strength * priorWeight``. Source-view eligibility
    must never be relaxed by the consumer, including during label smoothing.
    """
    vertices, faces = validateTriangles(geometry['vertices'], geometry['faces'], 3)
    if (not isinstance(frontImageId, (int, np.integer)) or isinstance(frontImageId, bool)
            or frontImageId < 1):
        raise ValueError('frontImageId must be a positive COLMAP image ID')
    valid = np.asarray(sourceValidMask)
    if (valid.ndim != 2 or min(valid.shape) < 2 or valid.dtype.kind not in 'bifu'
            or not np.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any()):
        raise ValueError('sourceValidMask must be finite HxW in [0,1]')
    params = np.asarray([strength, coreWidthRatio, upperWidthRatio, lowerWidthRatio,
                         sourceSafetyPixels, minCosNormalAngle], float)
    if (not np.isfinite(params).all() or strength < 0 or coreWidthRatio < 0
            or min(upperWidthRatio, lowerWidthRatio) <= coreWidthRatio
            or max(upperWidthRatio, lowerWidthRatio) > 1 or sourceSafetyPixels < 0
            or not 0 <= minCosNormalAngle < 1):
        raise ValueError('Invalid finite strength, eye-width band ratios or safety gates')
    K, R, t = (np.asarray(camera[key], np.float64)
               for key in ('intrinsics', 'rotation', 'translation'))
    if (K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,)
            or not all(np.isfinite(v).all() for v in (K, R, t))
            or not np.allclose(K[2], [0, 0, 1], atol=1e-12)
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
        raise ValueError('Expected finite OpenCV intrinsics and proper world-to-camera pose')
    imageShape = valid.shape
    photoEyes = _curves(photoEyes, imageShape, 'photoEyes')
    modelEyes = _curves(modelEyes, imageShape, 'modelEyes')
    extent = float(np.ptp(vertices, axis=0).max())
    if extent <= 0:
        raise ValueError('Geometry must have positive extent')
    cv = vertices @ R.T + t
    h = cv @ K.T
    xy = np.divide(h[:, :2], h[:, 2:], out=np.full((len(cv), 2), -1.), where=h[:, 2:] > 1e-8)
    triangles, projected = cv[faces], xy[faces]
    centers = triangles.mean(1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    cosine = np.divide(np.sum(normals * -centers, 1),
                       np.linalg.norm(normals, axis=1) * np.linalg.norm(centers, axis=1),
                       out=np.zeros(len(faces)), where=np.linalg.norm(normals, axis=1) > 1e-14)
    # These same barycentrics are evaluated on the projected triangle for the
    # semantic score, and on the 3D triangle for independent depth checks.
    samples = np.asarray([[1/3, 1/3, 1/3], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8],
                          [.45, .45, .1], [.45, .1, .45], [.1, .45, .45]])
    regionParams = dict(coreWidthRatio=coreWidthRatio, upperWidthRatio=upperWidthRatio,
                        lowerWidthRatio=lowerWidthRatio)
    semanticSamples = np.einsum('ki,fij->fkj', samples, projected)
    weights = _regionWeights(semanticSamples.reshape(-1, 2), modelEyes, **regionParams)
    weight = weights.reshape(len(faces), -1).mean(1)
    weight[~(triangles[:, :, 2] > 1e-8).all(1)] = 0
    candidates = weight > 1e-7
    safePixels, safeCells = _sourceSafety(valid, photoEyes, sourceSafetyPixels)
    sourceSupported = np.zeros(len(faces), bool)
    sourceCellCount = np.zeros(len(faces), np.int32)
    visibilityAccepted = np.zeros(len(faces), bool)
    visibilityQueries = np.zeros(len(faces), np.int32)
    eligibility = np.zeros(len(faces), bool)
    if visibleScene is None:
        sceneVertices, sceneFaces = vertices, faces
    else:
        sceneVertices, sceneFaces = validateTriangles(visibleScene['vertices'], visibleScene['faces'], 3)
    index = ProjectedSurfaceVisibility(sceneVertices @ R.T + t, sceneFaces, K, imageShape)
    depthTolerance = extent * 1e-6
    for faceIndex in np.flatnonzero(candidates & (cosine >= minCosNormalAngle)):
        cells = _triangleCells(projected[faceIndex], imageShape)
        if cells is None:
            continue
        cy, cx = cells
        sourceCellCount[faceIndex] = len(cx)
        supported = bool(len(cx) and safeCells[cy, cx].all())
        sourceSupported[faceIndex] = supported
        if not supported:
            continue
        queries = samples @ triangles[faceIndex]
        raster = trianglePixels(projected[faceIndex], *imageShape)
        if raster is not None:
            # Convert screen-affine barycentrics to the real perspective surface.
            bary = raster[2] / triangles[faceIndex, :, 2]
            bary /= bary.sum(1, keepdims=True)
            queries = np.concatenate([queries, bary @ triangles[faceIndex]])
        hit = index.closestDepth(queries)
        visible = hit['insidePhoto'] & np.isfinite(hit['depth'])
        visible &= np.abs(hit['depth'] - queries[:, 2]) <= depthTolerance
        visibilityQueries[faceIndex] = len(queries)
        visibilityAccepted[faceIndex] = bool(visible.all())
        eligibility[faceIndex] = bool(visible.all())
    priorWeight = weight * eligibility
    multiplier = 1. + strength * priorWeight
    # Photo-space diagnostics: continuous region independently of mesh, followed
    # by rasterized eligible mesh support. These are never input color masks.
    regionMap = np.zeros(imageShape, np.float32)
    supportedMap = np.zeros(imageShape, np.float32)
    allPoints = np.concatenate([e[p] for e in modelEyes.values() for p in ('upper', 'lower')])
    maxWidth = max(float(np.ptp(np.concatenate(list(e.values()))[:, 0])) for e in modelEyes.values())
    margin = maxWidth * max(upperWidthRatio, lowerWidthRatio)
    lo = np.maximum(np.floor(allPoints.min(0) - margin).astype(int), 0)
    hi = np.minimum(np.ceil(allPoints.max(0) + margin).astype(int), np.array(imageShape[::-1]) - 1)
    yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    regionMap[yy, xx] = _regionWeights(np.column_stack([xx.ravel(), yy.ravel()]), modelEyes,
                                      **regionParams).reshape(yy.shape)
    for faceIndex in np.flatnonzero(eligibility):
        raster = trianglePixels(projected[faceIndex], *imageShape)
        if raster is not None:
            yy, xx = raster[:2]
            supportedMap[yy, xx] = np.maximum(supportedMap[yy, xx], priorWeight[faceIndex])
    reasonBits = np.zeros(len(faces), np.uint8)
    reasonBits[candidates & (cosine < minCosNormalAngle)] |= 1
    reasonBits[candidates & (cosine >= minCosNormalAngle) & ~sourceSupported] |= 2
    reasonBits[candidates & sourceSupported & ~visibilityAccepted] |= 4
    diagnostics = {
        'method': 'frontalEyelidPerFaceScorePriorWithoutPhotoWarp',
        'faceCount': len(faces), 'candidateFaces': int(candidates.sum()),
        'eligibleFaces': int(eligibility.sum()),
        'sourceSupportedCandidateFaces': int(sourceSupported.sum()),
        'rejectedFacingFaces': int(np.count_nonzero(reasonBits & 1)),
        'rejectedSourceSupportFaces': int(np.count_nonzero(reasonBits & 2)),
        'rejectedVisibilityFaces': int(np.count_nonzero(reasonBits & 4)),
        'maximumMultiplier': float(multiplier.max()), 'strength': float(strength),
        'frontImageId': int(frontImageId),
        **{name: float(value) for name, value in regionParams.items()},
        'sourceSafetyPixels': float(sourceSafetyPixels), 'minCosNormalAngle': float(minCosNormalAngle),
        'depthTolerance': depthTolerance, 'sourceQueriesWarped': False,
        'allTouchedBilinearCellsChecked': True, 'sourceValidityThreshold': .999,
        'visibilitySampling': 'seven interior 3D points plus every covered photo pixel center',
        'reasonBits': {'1': 'facingRejected', '2': 'wholeFaceSourceSupportRejected', '4': 'depthRejected'},
        'note': 'A prior is not permission to bypass COLMAP candidate visibility or alter observation support.',
    }
    return {'eyelidWeight': weight.astype(np.float32), 'frontEligible': eligibility,
            'priorWeight': priorWeight.astype(np.float32), 'scoreMultiplier': multiplier.astype(np.float32),
            'projectedVertices': xy, 'normalCosine': cosine.astype(np.float32),
            'sourceSupported': sourceSupported, 'sourceCellCount': sourceCellCount,
            'visibilityAccepted': visibilityAccepted, 'visibilityQueryCount': visibilityQueries,
            'reasonBits': reasonBits, 'eyelidWeightMap': regionMap,
            'eligibleFaceWeightMap': supportedMap, 'sourceSafePixelMask': safePixels,
            'sourceSafeCellMask': safeCells, 'diagnostics': diagnostics}


def writeEyelidViewPrior(path, result, *, numImages, frontImageId=1):
    """Write zero-based faceIndex, COLMAP imageId, multiplier, regionWeight.

    Missing pairs have multiplier 1 and region weight 0. File rows are sorted
    by face index and contain only positive qualified preferences; the same
    file can be consumed by all three independent texture passes.
    """
    if (not isinstance(frontImageId, (int, np.integer)) or isinstance(frontImageId, bool)
            or frontImageId < 1):
        raise ValueError('frontImageId must be a positive COLMAP image ID')
    if (not isinstance(numImages, (int, np.integer)) or isinstance(numImages, bool)
            or numImages < 1):
        raise ValueError('numImages must be the positive count of registered COLMAP images')
    weights = np.asarray(result['priorWeight'])
    multipliers = np.asarray(result['scoreMultiplier'])
    if (weights.ndim != 1 or multipliers.shape != weights.shape
            or not np.isfinite(weights).all() or not np.isfinite(multipliers).all()
            or (weights < 0).any() or (weights > 1).any() or (multipliers < 1).any()):
        raise ValueError('Invalid per-face weights or score multipliers')
    ids = np.flatnonzero(weights > 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(f'COLMAP_EYELID_PRIOR_V1 {len(weights)} {int(numImages)}\n')
        for faceIndex in ids:
            handle.write(f'{faceIndex} {int(frontImageId)} {multipliers[faceIndex]:.9g} {weights[faceIndex]:.9g}\n')
    return {'path': str(path), 'rowCount': len(ids), 'frontImageId': int(frontImageId),
            'numImages': int(numImages),
            'faceIndexConvention': 'zeroBasedOriginalGeometryFaces'}
