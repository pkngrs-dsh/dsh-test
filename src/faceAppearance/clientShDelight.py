"""CPU SH delighting on a frozen client mesh and calibrated photo views.

Inputs and outputs are linear RGB. The explicit reference-view RGB gauge
preserves a chosen photograph's color scale; it is not physical calibration.
Visibility uses the final mesh, independent of the historical dense UV map.
"""
from __future__ import annotations

import numpy as np

from .colmapTexture import _destinationUvLookup
from .shDelight import shBasis, solveShDelight
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import interpolateSurface, samplePixels, trianglePixels, vertexNormals

def unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)



def collectClientPhotoObservations(geometry, cameras, raw, predicted, masks, resolution):
    """Shared true-perspective observations in native client UV, with exact depth."""
    lookup, sampling = _destinationUvLookup(geometry, resolution)
    atlas = lookup['validMask']
    normals = unit(interpolateSurface(vertexNormals(geometry['vertices'], geometry['faces']), geometry['faces'], lookup))
    # Retain float64 positions for the tight continuous-depth visibility gate.
    ids = geometry['faces'][lookup['faceIds'][atlas]]
    surface = (geometry['vertices'][ids] * lookup['barycentrics'][atlas, :, None]).sum(1)
    flatNormals = normals[atlas]
    count, height, width = masks.shape
    observations = np.zeros((count, resolution, resolution, 3), np.float32)
    od = np.zeros_like(observations)
    weights = np.zeros(observations.shape[:3], np.float32)
    pixels = np.full((*weights.shape, 2), -1, np.float32)
    yy, xx = np.nonzero(atlas)
    tolerance = float(np.ptp(geometry['vertices'], axis=0).max()) * 1e-6
    records = []
    for i, camera in enumerate(cameras):
        if camera.get('valid') is not True or camera.get('viewIndex') != i:
            raise ValueError('Expected ordered accepted saved cameras')
        R, t, K = [np.asarray(camera[k], np.float64) for k in ('rotation', 'translation', 'intrinsics')]
        cameraVertices = geometry['vertices'] @ R.T + t
        points = surface @ R.T + t
        projection = points @ K.T
        xy = projection[:, :2] / np.maximum(projection[:, 2:], 1e-12)
        cosine = np.sum((flatNormals @ R.T) * unit(-points), axis=-1)
        eligible = ((points[:, 2] > 1e-8) & (cosine > .1) & (xy >= 0).all(-1)
                    & (xy < [width - 1, height - 1]).all(-1))
        candidates = np.flatnonzero(eligible)
        gate = ProjectedSurfaceVisibility(cameraVertices, geometry['faces'], K, (height, width))
        hit = gate.closestDepth(points[candidates])
        visible = np.isfinite(hit['depth']) & (abs(hit['depth'] - points[candidates, 2]) <= tolerance)
        supported = samplePixels(masks[i].astype(np.float32), xy[candidates]) >= .999
        accepted = candidates[visible & supported]
        y, x = yy[accepted], xx[accepted]
        observations[i, y, x] = samplePixels(raw[i], xy[accepted])
        od[i, y, x] = samplePixels(predicted[i], xy[accepted])
        quality = 1 / (1 + (float(camera.get('medianErrorPixels', 0)) / 4) ** 2)
        weights[i, y, x] = cosine[accepted] ** 4 * quality
        pixels[i, y, x] = xy[accepted]
        records.append({'viewIndex': i, 'observedTexels': len(accepted), 'candidateTexels': len(candidates)})
    return {'raw': observations, 'openDelight': od, 'weights': weights, 'normals': normals,
            'pixels': pixels, 'diagnostics': {'views': records, 'uvSampling': sampling,
                'depthTolerance': tolerance, 'visibility': 'continuous two-sided closest ray depth',
                'normalFrame': 'final client geometry in common UFF object frame',
                'imageSampling': 'same saved 512 crop as OpenDelight, no dense-UV correction'}}


def renderClientPhotoNormals(geometry, camera, shape):
    """Perspective-correct canonical vertex normals at integer photo centers."""
    height, width = shape
    R, t, K = [np.asarray(camera[k], np.float64) for k in ('rotation', 'translation', 'intrinsics')]
    vertices = geometry['vertices'] @ R.T + t
    normals = vertexNormals(geometry['vertices'], geometry['faces'])
    projected = vertices @ K.T
    pixels = projected[:, :2] / projected[:, 2:]
    depth = np.full(shape, np.inf, np.float64)
    image = np.zeros((*shape, 3), np.float64)
    facing = np.zeros(shape, bool)
    for face in geometry['faces']:
        if (vertices[face, 2] <= 1e-8).any():
            continue
        raster = trianglePixels(pixels[face], height, width)
        if raster is None:
            continue
        y, x, bary = raster
        inverse = bary / vertices[face, 2]
        z = 1 / inverse.sum(-1)
        keep = z < depth[y, x]
        y, x, z, inverse = y[keep], x[keep], z[keep], inverse[keep]
        perspective = inverse * z[:, None]
        normal = unit(perspective @ normals[face])
        point = perspective @ vertices[face]
        image[y, x] = normal
        depth[y, x] = z
        facing[y, x] = np.sum((normal @ R.T) * unit(-point), axis=-1) > 0
    return image.astype(np.float32), np.isfinite(depth) & facing


def applyPhotoShLighting(rawLinear, normals, normalMask, coefficients, *, fitted=True,
                         minLighting=.25, maxLighting=4., photoMask=None):
    """Divide by bounded SH; retain source pixels where correction is unsafe.

    Returning the exact correction mask prevents treating retained photographic
    shading as successfully estimated albedo. No clipping of unsafe colors is
    used as a substitute for a valid estimate. Optional photoMask scopes the
    diagnostics only; it does not change the correction or retained raw pixels.
    """
    raw = np.asarray(rawLinear)
    normals = np.asarray(normals)
    valid = np.asarray(normalMask)
    coefficients = np.asarray(coefficients)
    if (raw.ndim != 3 or raw.shape[-1] != 3 or normals.shape != raw.shape
            or valid.shape != raw.shape[:2] or valid.dtype != np.bool_
            or coefficients.shape != (9, 3)):
        raise ValueError('Expected RGB/normal HxWx3, boolean HxW support and 9x3 SH')
    if (not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any()
            or not np.isfinite(normals).all() or not np.isfinite(coefficients).all()
            or not 0 < minLighting < maxLighting):
        raise ValueError('SH application requires finite linear RGB and bounded positive lighting')
    support = np.ones(valid.shape, bool) if photoMask is None else np.asarray(photoMask)
    if support.shape != valid.shape or support.dtype != np.bool_:
        raise ValueError('photoMask must be boolean HxW support aligned with the RGB image')
    lighting = np.einsum('hwj,jc->hwc', shBasis(normals), coefficients)
    lightSafe = ((lighting >= minLighting).all(-1)
                 & (lighting <= maxLighting).all(-1))
    candidate = np.divide(raw, lighting, out=np.zeros_like(raw, dtype=np.float64),
                          where=lightSafe[..., None])
    rangeSafe = (candidate >= 0).all(-1) & (candidate <= 1).all(-1)
    normalSafe = valid & (np.linalg.norm(normals, axis=-1) > 1e-8)
    correctedMask = normalSafe & lightSafe & rangeSafe & bool(fitted)
    corrected = raw.copy()
    corrected[correctedMask] = candidate[correctedMask]
    reasons = {'missingNormalPixels': ~normalSafe,
               'unsafeLightingPixels': normalSafe & ~lightSafe,
               'outOfRangePixels': normalSafe & lightSafe & ~rangeSafe,
               'unfittedPixels': normalSafe & lightSafe & rangeSafe & ~np.bool_(fitted)}

    def counts(domain):
        pixels = int(domain.sum())
        fallback = int((domain & ~correctedMask).sum())
        return {'pixels': pixels, 'correctedPixels': pixels - fallback,
                'rawFallbackPixels': fallback,
                'rawFallbackFraction': float(fallback / max(pixels, 1)),
                **{name: int((domain & mask).sum()) for name, mask in reasons.items()}}

    fullCounts, photoCounts = counts(np.ones(valid.shape, bool)), counts(support)
    return {'linearRGB': corrected, 'correctionMask': correctedMask,
            'lighting': lighting.astype(np.float32),
            'diagnostics': {'fitSucceeded': bool(fitted),
                'reasonCountsDomain': 'fullImage' if photoMask is None else 'validPhotoMask',
                **{name: photoCounts[name] for name in reasons},
                'fullImage': fullCounts, 'validPhoto': photoCounts if photoMask is not None else None}}


def delightClientPhotos(geometry, cameras, rawLinear, validMasks, *,
                         referenceViewIndex=None, fitResolution=256, iterations=8):
    """Fit shared low-frequency reflectance and per-view lighting, then apply it.

    The supplied reference fixes only the ambiguous RGB scale; every accepted
    view still constrains the solve. Invalid correction pixels retain original
    colors and are recorded separately. Camera fitting, geometry edits and UV
    baking remain separate modules.
    """
    raw = np.asarray(rawLinear, np.float32)
    masks = np.asarray(validMasks)
    if (raw.ndim != 4 or raw.shape[-1] != 3 or masks.shape != raw.shape[:3]
            or masks.dtype != np.bool_ or len(cameras) != len(raw)
            or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any()):
        raise ValueError('Expected aligned finite linear RGB photos, boolean masks and ordered cameras')
    if isinstance(fitResolution, bool) or not isinstance(fitResolution, int) or fitResolution < 32:
        raise ValueError('fitResolution must be an integer >= 32')
    observations = collectClientPhotoObservations(geometry, cameras, raw, raw, masks, fitResolution)
    observations.pop('openDelight')
    fit = solveShDelight(observations['raw'], observations['weights'], observations['normals'],
                        mode='shOnly', iterations=iterations, gaugeReferenceView=referenceViewIndex)
    corrected, correctionMasks, reports = [], [], []
    for index, camera in enumerate(cameras):
        normals, valid = renderClientPhotoNormals(geometry, camera, masks.shape[1:])
        result = applyPhotoShLighting(raw[index], normals, valid, fit['coefficients'][index],
                                      fitted=fit['diagnostics']['views'][index]['status'] == 'fitted',
                                      photoMask=masks[index])
        corrected.append(result['linearRGB'])
        correctionMasks.append(result['correctionMask'])
        supported = masks[index]
        count = int(supported.sum())
        fallback = supported & ~result['correctionMask']
        reports.append({'viewIndex': index, **result['diagnostics'],
                        'validPhotoPixels': count, 'correctedValidPhotoPixels': int((supported & result['correctionMask']).sum()),
                        'fallbackValidPhotoPixels': int(fallback.sum()),
                        'rawFallbackFraction': float(fallback.sum() / max(count, 1))})
    totalPixels = sum(item['validPhotoPixels'] for item in reports)
    totalFallback = sum(item['fallbackValidPhotoPixels'] for item in reports)
    applicationSummary = {'validPhotoPixels': totalPixels,
        'correctedValidPhotoPixels': totalPixels - totalFallback,
        'fallbackValidPhotoPixels': totalFallback,
        'rawFallbackFraction': float(totalFallback / max(totalPixels, 1)),
        'domain': 'Sum of valid photo pixels over all views, not whole images or final UV texels',
        'coverageMeaning': 'validMasks retain photograph support, including raw fallback; correctionMasks identify successful SH application'}
    return {'linearImages': np.asarray(corrected, np.float32),
            'validMasks': masks.copy(), 'correctionMasks': np.asarray(correctionMasks),
            'coefficients': fit['coefficients'], 'observations': observations, 'fit': fit,
            'diagnostics': {'method': 'clientSurfaceShPhotoDelightV1',
                'referenceViewIndex': referenceViewIndex, 'fitResolution': fitResolution,
                'fit': fit['diagnostics'], 'application': reports, 'applicationSummary': applicationSummary,
                'limitations': ['RGB gauge inherits photographic exposure and white balance.',
                    'Uncorrected pixels retain photographed lighting; inspect correctionMasks.',
                    'SH does not model cast shadows, specular highlights or subsurface scattering.']}}
