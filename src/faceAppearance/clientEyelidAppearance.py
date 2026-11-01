"""Safe local eyelid color sampling for an already assembled client surface.

This module accepts linear RGB and current-photo contours. It performs no
model inference, color encoding, file IO, geometry fitting or asset export.
Visible contours are measured on the caller's actual full assembly. A rejected
measurement preserves the baseline exactly and reports why it was skipped.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .camera import projectPoints
from .colmapTexture import _destinationUvLookup
from .eyelidTextureSampling import reprojectEyelidTexture, _outsidePolygon, _rgb, _curves, _geometry
from .texture import interpolateSurface, validateTriangles
from .visibleEyeContours import extractVisibleEyeContours


DEFAULT_POLICY = {
    'coreWorkingPixels': 12., 'outerWorkingPixels': 24.,
    'sourceSafetyWorkingPixels': .75, 'maximumExtensionWorkingPixels': 4.,
    'photoBoundsPaddingWorkingPixels': 14., 'destinationBoxFeatherWorkingPixels': 4.,
    'accessoryUvPaddingPixels': 2, 'pointsPerLid': 24, 'filterFootprint': False,
}


def _policy(policy):
    result = DEFAULT_POLICY.copy()
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(result):
            raise ValueError('Unknown eyelid appearance policy entries')
        result.update(policy)
    for key in ('accessoryUvPaddingPixels', 'pointsPerLid'):
        if type(result[key]) is not int or result[key] < (1 if key == 'accessoryUvPaddingPixels' else 3):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('coreWorkingPixels', 'outerWorkingPixels', 'sourceSafetyWorkingPixels',
                'maximumExtensionWorkingPixels', 'photoBoundsPaddingWorkingPixels', 'destinationBoxFeatherWorkingPixels'):
        value = result[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
        result[key] = float(value)
    if result['outerWorkingPixels'] <= result['coreWorkingPixels'] or result['destinationBoxFeatherWorkingPixels'] <= 0:
        raise ValueError('Eyelid outer band must exceed core and feather must be positive')
    if result['filterFootprint'] is not False:
        raise ValueError('Client eyelid appearance requires filterFootprint=False')
    return result


def _validateSceneIdentity(geometry, scene):
    if geometry.get('uvConvention') != 'topLeftPixelEdges':
        raise ValueError('Client eyelid appearance requires explicit topLeftPixelEdges UV coordinates')
    vertices, faces, uvs, uvFaces, _ = _geometry(geometry)
    fullVertices, fullFaces = validateTriangles(scene['vertices'], scene['faces'], 3)
    mapping, materials = np.asarray(scene['nicpToFullIds']), np.asarray(scene['materialIds'])
    if (mapping.shape != (len(vertices),) or mapping.dtype.kind not in 'iu'
            or len(np.unique(mapping)) != len(mapping) or (mapping < 0).any() or (mapping >= len(fullVertices)).any()
            or materials.shape != (len(fullFaces),) or materials.dtype.kind not in 'iu'):
        raise ValueError('Scene needs original skin mapping and full-face material IDs')
    if not np.allclose(fullVertices[mapping], vertices, rtol=0, atol=1e-12):
        raise ValueError('Assembled scene skin differs from the supplied final geometry')
    faceLookup = {tuple(face): index for index, face in enumerate(fullFaces)}
    try:
        fullFaceIds = np.asarray([faceLookup[tuple(face)] for face in mapping[faces]], dtype=np.int64)
    except KeyError as error:
        raise ValueError('Skin face order/corners differ from the full assembly') from error
    if (not np.array_equal(uvs, scene['uvs']) or not np.array_equal(uvFaces, np.asarray(scene['uvFaces'])[fullFaceIds])
            or not np.all(materials[fullFaceIds] == 0)):
        raise ValueError('Skin UV/corner/material identity differs from the full assembly')
    if materials.min() < 0 or materials.max() >= len(scene['materials']):
        raise ValueError('Scene material IDs do not address the supplied materials')
    return mapping


def _localDomain(geometry, camera, lookup, sourceMask, photoEyes, rawEyes, protected, policy):
    skin = lookup['validMask']
    surface = interpolateSurface(geometry['vertices'], geometry['faces'], lookup)
    projected, depth = projectPoints(surface[skin], np.asarray(camera['intrinsics']),
        np.asarray(camera['rotation']), np.asarray(camera['translation']))
    rows, columns = np.indices(sourceMask.shape)
    sourceBand = np.zeros(sourceMask.shape, bool)
    destinationBand = np.zeros(len(projected), bool)
    featherValues = np.zeros(len(projected), np.float64)
    boxes = []
    for eye in photoEyes.values():
        points = np.concatenate([eye['upper'], eye['lower']])
        minimum = points.min(0) - policy['photoBoundsPaddingWorkingPixels']
        maximum = points.max(0) + policy['photoBoundsPaddingWorkingPixels']
        boxes.append(np.r_[minimum, maximum])
        sourceBand |= ((columns >= minimum[0]) & (columns <= maximum[0]) &
                       (rows >= minimum[1]) & (rows <= maximum[1]))
        destinationBand |= ((projected >= minimum).all(1) & (projected <= maximum).all(1) & (depth > 0))
        distance = np.minimum(projected - minimum, maximum - projected).min(1)
        fraction = np.clip(distance / policy['destinationBoxFeatherWorkingPixels'], 0, 1)
        featherValues = np.maximum(featherValues, fraction * fraction * (3 - 2 * fraction))
    local = np.zeros(skin.shape, bool); local[skin] = destinationBand
    feather = np.zeros(skin.shape, np.float32); feather[skin] = featherValues
    safe = sourceMask.copy() & sourceBand
    sourceRows, sourceColumns = np.nonzero(safe)
    coordinates = np.column_stack([sourceColumns, sourceRows])
    outside = np.ones(len(coordinates), bool)
    for eye in rawEyes.values():
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        outside &= _outsidePolygon(coordinates, polygon, policy['sourceSafetyWorkingPixels'])
    safe[sourceRows, sourceColumns] &= outside
    return skin & local & ~protected, feather, safe, np.asarray(boxes)


def validateEyelidSourceFootprints(sourceXY, queryMask, sourceValidMask, photoEyes, originalPhotoEyes, *, margin=.75):
    """Independently check all four bilinear contributors of every edited texel."""
    coordinates, mask, validMask = np.asarray(sourceXY), np.asarray(queryMask), np.asarray(sourceValidMask)
    if (coordinates.ndim != 3 or coordinates.shape[-1] != 2 or mask.shape != coordinates.shape[:2]
            or mask.dtype != np.bool_ or validMask.ndim != 2 or validMask.dtype != np.bool_
            or min(validMask.shape) < 2 or not np.isfinite(margin) or margin < 0):
        raise ValueError('Source footprint validation requires matching XY, boolean masks and a nonnegative margin')
    query = coordinates[mask]
    if not np.isfinite(query).all():
        raise ValueError('Modified eyelid texel has a nonfinite source query')
    floors = np.floor(query).astype(np.int64)
    support = np.stack([floors, floors + [0, 1], floors + [1, 0], floors + [1, 1]], axis=1)
    inBounds = ((support >= 0) & (support < np.asarray(sourceValidMask.shape[::-1]))).all(-1)
    valid = np.zeros(inBounds.shape, bool)
    valid[inBounds] = sourceValidMask[support[inBounds, 1], support[inBounds, 0]]
    outside = np.ones(inBounds.shape, bool)
    for eye in list(photoEyes.values()) + list(originalPhotoEyes.values()):
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        outside &= _outsidePolygon(support.reshape(-1, 2), polygon, margin).reshape(inBounds.shape)
    invalid = ~(inBounds & valid & outside).all(1)
    return {'sourceQueriesIncludingFilterPadding': int(queryMask.sum()),
        'allFourSourcePixelsInBounds': bool(inBounds.all()), 'allFourSourcePixelsValid': bool(valid.all()),
        'allFourSourcePixelsOutsideBothPhotoOpeningsWithMargin': bool(outside.all()),
        'originalDenseAndResampledContoursBothChecked': True,
        'invalidSourceSupportQueryCount': int(invalid.sum())}


def refineClientEyelidAppearance(geometry, scene, camera, sourceLinear, sourceValidMask,
                                 baselineLinear, originalPhotoEyes, *, policy=None):
    """Return ``linearRGB``, provenance ``maps`` and diagnostics without mutation.

    Source RGB is already linear and color-scaled by the caller. Source masks
    are boolean original-photo support, not inferred SH correction success.
    Geometry and scene contain the SAME final canonical skin with original UVs;
    scene has the fields returned by ``loadLegacyEyeBindingInputs``. Baseline is
    a square linear RGB atlas. Alpha/material payloads are never accepted or
    rewritten. The caller must have accepted its current-photo detector gates.

    Rejected visible measurements return exact baseline color. Unsafe four-tap
    support raises an error: a caller must not export a partially checked edit.
    """
    effective = _policy(policy)
    source, baseline = _rgb(sourceLinear, 'sourceLinear'), _rgb(baselineLinear, 'baselineLinear')
    if baseline.shape[0] != baseline.shape[1]:
        raise ValueError('Client head atlas must be square')
    valid = np.asarray(sourceValidMask)
    if valid.shape != source.shape[:2] or valid.dtype != np.bool_:
        raise ValueError('sourceValidMask must be a boolean source-image mask')
    rawEyes = _curves(originalPhotoEyes, source.shape[:2], 'originalPhotoEyes')
    mapping = _validateSceneIdentity(geometry, scene)
    measured = extractVisibleEyeContours(**scene, camera=camera, photoEyes=rawEyes,
        imageShape=source.shape[:2], pointsPerLid=effective['pointsPerLid'])
    states = {name: measured['eyes'][name]['status'] for name in rawEyes}
    diagnostics = {'method': 'safeCurrentAssemblyEyelidColorSamplingV1', 'policy': effective,
        'geometryChanged': False, 'uvChanged': False, 'cameraChanged': False,
        'appearanceMeasurementStatus': states, 'visibleMeasurement': measured,
        'geometryBindingClaimed': False,
        'limitations': ['Relocated and exterior-extended color is estimated appearance.',
            'Eye aperture, eyeball placement, gaze and lid geometry are unchanged.']}
    if any(status != 'accepted' for status in states.values()):
        diagnostics.update(status='skippedUnreliableVisibleContours',
            reasons={name: measured['eyes'][name].get('reasons', []) for name in rawEyes}, changedTexels=0)
        return {'linearRGB': baseline.copy(), 'maps': {'alpha': np.zeros(baseline.shape[:2], np.float32),
            'modifiedMask': np.zeros(baseline.shape[:2], bool)}, 'diagnostics': diagnostics}
    modelEyes = {name: {part: np.asarray(measured['eyes'][name][part]['sourcePixels']) for part in ('upper', 'lower')} for name in rawEyes}
    photoEyes = {name: {part: np.asarray(measured['eyes'][name][part]['targetPixels']) for part in ('upper', 'lower')} for name in rawEyes}
    size = baseline.shape[0]
    member = np.zeros(len(scene['vertices']), bool); member[mapping] = True
    accessoryFaces = (np.asarray(scene['materialIds']) == 0) & ~member[np.asarray(scene['faces'])].all(1)
    protected = np.zeros((size, size), bool)
    if accessoryFaces.any():
        accessory = {'vertices': scene['vertices'], 'faces': np.asarray(scene['faces'])[accessoryFaces],
            'uvs': scene['uvs'], 'uvFaces': np.asarray(scene['uvFaces'])[accessoryFaces], 'uvConvention': 'topLeftPixelEdges'}
        protected = binary_dilation(_destinationUvLookup(accessory, size)[0]['validMask'],
            iterations=effective['accessoryUvPaddingPixels'])
    lookup = _destinationUvLookup(geometry, size)[0]
    writable, feather, safe, boxes = _localDomain(geometry, camera, lookup, valid,
        photoEyes, rawEyes, protected, effective)
    opaqueMaterials = np.asarray([material.get('alphaMode', 'OPAQUE') != 'MASK' for material in scene['materials']])
    visible = {'vertices': scene['vertices'], 'faces': np.asarray(scene['faces'])[opaqueMaterials[np.asarray(scene['materialIds'])]]}
    result = reprojectEyelidTexture(geometry, camera, source, safe, baseline, photoEyes, modelEyes,
        writableMask=writable, visibleScene=visible, corePixels=effective['coreWorkingPixels'],
        outerPixels=effective['outerWorkingPixels'], sourceSafetyPixels=effective['sourceSafetyWorkingPixels'],
        maxExtensionPixels=effective['maximumExtensionWorkingPixels'], filterFootprint=False)
    linear = baseline.copy()
    alpha = result['alpha'] * feather
    changedDomain = (alpha > 0) | result['filterPadding']
    linear[changedDomain] = (baseline[changedDomain] +
        (result['linearRGB'][changedDomain] - baseline[changedDomain]) * feather[changedDomain, None])
    checks = validateEyelidSourceFootprints(result['sourceXY'], changedDomain, safe,
        photoEyes, rawEyes, margin=effective['sourceSafetyWorkingPixels'])
    if checks['invalidSourceSupportQueryCount']:
        raise ValueError('Independent eyelid four-contributor source support validation failed')
    if (changedDomain & ~writable).any() or not np.array_equal(linear[~changedDomain], baseline[~changedDomain]) or not np.array_equal(linear[protected], baseline[protected]):
        raise ValueError('Eyelid color edit escaped its permitted domain')
    maps = {key: value.copy() for key, value in result.items() if isinstance(value, np.ndarray) and key != 'linearRGB'}
    maps['alpha'] = alpha
    for key in ('observedMask', 'estimatedMask', 'warpMask', 'extensionMask'):
        maps[key] &= alpha > 0
    maps.update(writableMask=writable, protectedAccessoryUvs=protected, sourceValidMask=safe,
        originalSourceValidMask=valid.copy(), destinationFeather=feather, skinDomainMask=lookup['validMask'],
        modifiedMask=changedDomain,
        rawPhotoCurves=np.asarray([[rawEyes[name][part] for part in ('lower', 'upper')] for name in rawEyes]),
        resampledPhotoCurves=np.asarray([[photoEyes[name][part] for part in ('lower', 'upper')] for name in rawEyes]),
        modelCurves=np.asarray([[modelEyes[name][part] for part in ('lower', 'upper')] for name in rawEyes]))
    changed = np.any(linear != baseline, axis=-1)
    diagnostics.update(status='applied' if changed.any() else 'noSafeVisibleColorChange',
        strictSourceSupportChecks=checks, sourceAndDestinationBoxesWorking=boxes,
        changedTexels=int(changed.sum()), maximumAbsoluteLinearChange=float(np.abs(linear - baseline).max()),
        sourceColorGain=[1., 1., 1.], alphaPayloadChanged=False,
        textureDiagnostics=result['diagnostics'])
    return {'linearRGB': linear, 'maps': maps, 'diagnostics': diagnostics}
