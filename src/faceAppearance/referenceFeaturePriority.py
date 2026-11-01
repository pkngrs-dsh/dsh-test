"""Optional local preference for already observed reference-photo features.

This module changes source ownership only. It neither moves sampling positions,
creates coverage, estimates illumination, nor inpaints unobserved texels. Feature
curves and samplingPixels must use the same original-photo XY coordinate frame.
"""
from __future__ import annotations

import copy

import numpy as np


def _curveDistances(points, curves, maximumDistance):
    """Exact distance to each polyline segment; separate curves never connect."""
    distanceSquared = np.full(len(points), np.inf)
    for curve in curves:
        minimum, maximum = curve.min(0)-maximumDistance, curve.max(0)+maximumDistance
        eligible = np.flatnonzero(((points >= minimum) & (points <= maximum)).all(1))
        if not len(eligible):
            continue
        query = points[eligible]
        best = distanceSquared[eligible].copy()
        for first, last in zip(curve[:-1], curve[1:]):
            direction = last-first
            lengthSquared = float(np.dot(direction, direction))
            if lengthSquared <= 1e-20:
                continue
            fraction = np.clip(((query-first)@direction)/lengthSquared, 0, 1)
            residual = query-(first+fraction[:, None]*direction)
            best = np.minimum(best, np.sum(residual*residual, axis=1))
        distanceSquared[eligible] = best
    return np.sqrt(distanceSquared)


def applyReferenceFeaturePriority(fusion, referenceObservation, referenceWeights, samplingPixels,
                                  featureCurves, *, referenceViewIndex=0, coreRadiusPixels=3.,
                                  falloffRadiusPixels=8.):
    """Raise reference ownership near explicit photo feature curves.

    The core copies a qualified reference observation exactly (in the fusion's
    floating dtype). A cosine transition blends with the existing fused color.
    For local weight w and old reference alpha a, the actual new contribution is
    1-(1-a)*(1-w), not max(a,w). Existing fully owned reference colors and the
    geometric referenceCoreMask remain unchanged. Only already-covered texels
    with positive referenceWeights can change; invalid/outside colors are bit
    identical. The original source-index contract is -1 missing, -2 mixed, and
    nonnegative for a single known view. Matching albedo cache values, if present,
    use the project's explicit gamma-2.2 convention.
    """
    required = ('linearAlbedo', 'coverageMask', 'referenceAlpha', 'sourceIndex')
    if any(key not in fusion for key in required):
        raise ValueError('Fusion needs linearAlbedo, coverageMask, referenceAlpha and sourceIndex')
    colors = np.asarray(fusion['linearAlbedo'])
    reference = np.asarray(referenceObservation)
    weights, pixels = np.asarray(referenceWeights), np.asarray(samplingPixels)
    coverage, alpha, source = (np.asarray(fusion[key]) for key in ('coverageMask', 'referenceAlpha', 'sourceIndex'))
    if (colors.ndim != 3 or colors.shape[-1] != 3 or min(colors.shape[:2]) < 1
            or colors.dtype.kind != 'f' or reference.shape != colors.shape or reference.dtype.kind != 'f'
            or not np.isfinite(colors).all() or not np.isfinite(reference).all()
            or np.any((colors < 0) | (colors > 1)) or np.any((reference < 0) | (reference > 1))
            or coverage.shape != colors.shape[:2] or coverage.dtype != np.bool_
            or weights.shape != coverage.shape or weights.dtype.kind not in 'fiu'
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or pixels.shape != (*coverage.shape, 2) or pixels.dtype.kind not in 'fiu'
            or alpha.shape != coverage.shape or alpha.dtype.kind != 'f' or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1)) or source.shape != coverage.shape or source.dtype.kind != 'i'
            or np.any(source < -2) or np.any(source[~coverage] != -1) or np.any(source[coverage] == -1)):
        raise ValueError('Invalid linear colors, coverage, observation weights, coordinates, alpha or known source map')
    if (not isinstance(referenceViewIndex, (int, np.integer)) or isinstance(referenceViewIndex, (bool, np.bool_))
            or referenceViewIndex < 0 or referenceViewIndex > np.iinfo(source.dtype).max
            or not np.isfinite([coreRadiusPixels, falloffRadiusPixels]).all()
            or not 0 <= coreRadiusPixels < falloffRadiusPixels):
        raise ValueError('Expected a valid reference index and 0 <= core radius < falloff radius')
    eligible = coverage & (weights > 0)
    if (not np.isfinite(pixels[eligible]).all() or np.any(pixels[eligible] < 0)):
        raise ValueError('Qualified reference observations require finite nonnegative original-photo pixels')
    trusted = np.asarray(fusion.get('referenceCoreMask', np.zeros_like(coverage)))
    protection = np.asarray(fusion.get('protectionMask', np.zeros_like(coverage)))
    if (trusted.shape != coverage.shape or protection.shape != coverage.shape
            or trusted.dtype != np.bool_ or protection.dtype != np.bool_):
        raise ValueError('Reference-core/protection masks must be matching boolean arrays')
    confidence = None
    if 'confidence' in fusion:
        confidence = np.asarray(fusion['confidence'])
        if (confidence.shape != coverage.shape or confidence.dtype.kind != 'f'
                or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError('Fusion confidence must be a finite floating array in [0,1]')
    if 'albedo' in fusion:
        encoded = np.asarray(fusion['albedo'])
        if encoded.shape != colors.shape or encoded.dtype.kind != 'f' or not np.isfinite(encoded).all():
            raise ValueError('Optional gamma-2.2 albedo cache must match fusion color shape')
    if 'auxiliarySourceIndex' in fusion:
        auxiliarySource = np.asarray(fusion['auxiliarySourceIndex'])
        if (auxiliarySource.shape != coverage.shape or auxiliarySource.dtype.kind != 'i'
                or np.any(auxiliarySource < -1)):
            raise ValueError('Optional auxiliary source map must contain known indices or -1')
    reference = reference.astype(colors.dtype, copy=False)
    fullyOwned = eligible & (alpha == 1)
    if (np.any(trusted & eligible & (alpha != 1))
            or np.any(source[fullyOwned] != referenceViewIndex)
            or not np.array_equal(colors[fullyOwned], reference[fullyOwned])):
        raise ValueError('Existing reference ownership must match the supplied reference observation/view')
    curves = []
    for values in featureCurves:
        curve = np.asarray(values, dtype=np.float64)
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2 or not np.isfinite(curve).all()
                or not np.any(np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-10)):
            raise ValueError('Each feature curve must be a finite nondegenerate Nx2 polyline')
        curves.append(curve)
    locality = np.zeros(coverage.shape, dtype=np.float32)
    distances = _curveDistances(pixels[eligible].astype(np.float64), curves, falloffRadiusPixels)
    localValues = np.zeros(len(distances), dtype=np.float64)
    coreValues = distances <= coreRadiusPixels
    transition = (distances > coreRadiusPixels) & (distances < falloffRadiusPixels)
    localValues[coreValues] = 1
    localValues[transition] = .5*(1+np.cos(np.pi*(distances[transition]-coreRadiusPixels)
                                         /(falloffRadiusPixels-coreRadiusPixels)))
    locality[eligible] = np.clip(localValues, 0, 1)
    # Keep the exact geometric core separate from float32 rounding at the
    # start of the falloff, so a transition never invents full ownership.
    priorityCore = np.zeros_like(coverage)
    priorityCore[eligible] = coreValues
    roundedTransition = (locality == 1) & ~priorityCore
    locality[roundedTransition] = np.nextafter(np.float32(1), np.float32(0))
    priorityMask = locality > 0
    changedOwnership = priorityMask & ~fullyOwned & ~trusted
    strictCore = priorityCore & changedOwnership
    result = {key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
              for key, value in fusion.items()}
    amount = locality[changedOwnership].astype(np.float64)
    result['linearAlbedo'][changedOwnership] = (
        (1-amount[:, None])*colors[changedOwnership]+amount[:, None]*reference[changedOwnership])
    result['linearAlbedo'][strictCore] = reference[strictCore]
    result['referenceAlpha'][changedOwnership] = 1-(1-alpha[changedOwnership])*(1-amount)
    result['referenceAlpha'][strictCore] = 1
    roundedMixed = changedOwnership & ~strictCore & (result['referenceAlpha'] == 1)
    result['referenceAlpha'][roundedMixed] = np.nextafter(alpha.dtype.type(1), alpha.dtype.type(0))
    result['sourceIndex'][changedOwnership] = -2
    result['sourceIndex'][changedOwnership & (result['referenceAlpha'] == 1)] = referenceViewIndex
    if 'auxiliarySourceIndex' in result:
        result['auxiliarySourceIndex'][changedOwnership & (result['referenceAlpha'] == 1)] = -1
    if confidence is not None:
        result['confidence'][changedOwnership] = ((1-amount)*confidence[changedOwnership]
                                                  + amount*np.clip(weights[changedOwnership], 0, 1))
        result['confidence'][strictCore] = np.clip(weights[strictCore], 0, 1)
    if 'albedo' in result:
        result['albedo'][changedOwnership] = np.clip(result['linearAlbedo'][changedOwnership], 0, 1)**(1/2.2)
    if 'blendAllowedMask' in result:
        allowed = result['blendAllowedMask']
        if allowed.shape != coverage.shape or allowed.dtype != np.bool_:
            raise ValueError('blendAllowedMask must be a matching boolean array')
        allowed[changedOwnership & (result['referenceAlpha'] < 1)] = True
    result.update({'protectionMask': protection.copy() | priorityCore, 'priorityMask': priorityMask,
                   'priorityWeight': locality, 'priorityCoreMask': priorityCore})
    diagnostics = dict(result.get('diagnostics', {}))
    diagnostics['referenceFeaturePriority'] = {
        'enabled': True, 'referenceViewIndex': int(referenceViewIndex), 'curveCount': len(curves),
        'coreRadiusOriginalPixels': float(coreRadiusPixels), 'falloffRadiusOriginalPixels': float(falloffRadiusPixels),
        'distanceMethod': 'exactDistanceToSeparatePhotoPolylineSegments', 'falloff': 'raisedCosine',
        'eligibleReferenceTexels': int(eligible.sum()), 'priorityTexels': int(priorityMask.sum()),
        'strictCoreTexels': int(priorityCore.sum()), 'ownershipChangedTexels': int(changedOwnership.sum()),
        'colorChangedTexels': int(np.any(result['linearAlbedo'] != colors, axis=-1).sum()),
        'coverageExpanded': False, 'geometricReferenceCoreExpanded': False,
        'originalTrustedCorePreserved': bool(np.array_equal(result['linearAlbedo'][trusted], colors[trusted])),
        'strictCoreMaximumReferenceError': float(abs(result['linearAlbedo'][priorityCore]-reference[priorityCore]).max())
             if priorityCore.any() else 0.,
        'effectiveAlpha': '1-(1-oldReferenceAlpha)*(1-priorityWeight)',
        'limits': ['Only already-observed reference colors are reused; no missing texels are filled',
                   'Photo-curve proximity is a source preference, not a new visibility or material-confidence estimate',
                   'Reference likeness can improve at the expense of other viewing angles']}
    diagnostics.update({'referenceOwnedTexels': int((coverage & (result['referenceAlpha'] == 1)).sum()),
                        'blendedTexels': int((coverage & (result['referenceAlpha'] > 0) & (result['referenceAlpha'] < 1)).sum()),
                        'auxiliaryOnlyTexels': int((coverage & (result['referenceAlpha'] == 0)).sum())})
    result['diagnostics'] = diagnostics
    return result
