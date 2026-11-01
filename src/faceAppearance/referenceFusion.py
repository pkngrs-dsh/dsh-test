"""Reference-owned linear-RGB fusion with auxiliary views used for completion.

This module selects existing observations; it neither estimates illumination nor
repairs geometric correspondences. A fully trusted reference texel is copied
exactly, regardless of disagreement or confidence in the auxiliary views.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _smoothStep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    return values * values * (3 - 2 * values)


def _matchAuxiliaryGains(values: np.ndarray, weights: np.ndarray,
                        viewingCosines: np.ndarray, referenceViewIndex: int,
                        referenceFullCosine: float, enabled: bool) -> tuple[np.ndarray, list[dict]]:
    """Estimate bounded RGB gains from reliable, nonsaturated overlap only."""
    gains = np.ones((len(values), 3), np.float32)
    diagnostics = []
    reference = values[referenceViewIndex]
    referenceReliable = ((weights[referenceViewIndex] > 0)
                         & (viewingCosines[referenceViewIndex] >= referenceFullCosine))
    minimumSamples = 32
    for viewIndex in range(len(values)):
        sampleCounts = [0, 0, 0]
        if enabled and viewIndex != referenceViewIndex:
            reliable = (referenceReliable & (weights[viewIndex] > 0)
                        & (viewingCosines[viewIndex] >= referenceFullCosine))
            for channel in range(3):
                source = values[viewIndex, ..., channel]
                target = reference[..., channel]
                valid = (reliable & (source > 0.015) & (source < 0.985)
                         & (target > 0.015) & (target < 0.985))
                sampleCounts[channel] = int(valid.sum())
                if sampleCounts[channel] >= minimumSamples:
                    # A median in log space resists mismatched local features;
                    # a single global gain cannot correct spatial misalignment.
                    logRatios = np.log(target[valid].astype(np.float64)) - np.log(source[valid])
                    gains[viewIndex, channel] = np.clip(np.exp(np.median(logRatios)), 2 / 3, 1.5)
        diagnostics.append({"viewIndex": viewIndex, "gainsRgb": gains[viewIndex].tolist(),
                            "overlapSamplesRgb": sampleCounts,
                            "calibratedChannels": [count >= minimumSamples for count in sampleCounts]})
    return gains, diagnostics


def fuseReferenceObservations(observations: np.ndarray, weights: np.ndarray,
                              viewingCosines: np.ndarray, *, referenceViewIndex: int = 0,
                              referenceBlendWidth: float = 12.0,
                              referenceMinCosine: float = 0.35,
                              referenceFullCosine: float = 0.65,
                              matchAuxiliaryColors: bool = True) -> dict:
    """Fuse NHW3 linear colors while preserving an explicit reference core.

    ``weights > 0`` defines qualified support; callers must already reject
    occlusion, invalid exposure and incorrect sampling positions. Reference
    alpha combines a normal-angle ramp with a UV-pixel distance ramp from that
    support boundary. The outermost valid pixel has zero boundary alpha when
    feathering is enabled. Atlas boundaries count as unsupported neighbors.

    Auxiliary views are selected by maximum weight, never averaged together.
    Optional per-channel gains use at least 32 reliable overlap samples and are
    clamped to [2/3, 1.5]. These gains are empirical color matching, not physical
    exposure calibration. Only the selected auxiliary colors are adjusted.

    ``sourceIndex`` is -1 for missing support, -2 for reference/auxiliary blends,
    or the selected view index for unmixed colors. ``referenceCoreMask`` marks
    the full geometric trust region. Alpha also becomes one outside that core
    wherever the reference is the only available source, preserving coverage
    without claiming additional trust. Confidence is a clipped, alpha-weighted
    source weight, not a calibrated probability or correspondence guarantee.
    """
    values = np.asarray(observations, np.float32)
    weightArray = np.asarray(weights, np.float32)
    cosines = np.asarray(viewingCosines, np.float32)
    if (values.ndim != 4 or values.shape[-1] != 3 or min(values.shape[:3], default=0) < 1
            or weightArray.shape != values.shape[:3] or cosines.shape != weightArray.shape
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.isfinite(weightArray).all() or np.any(weightArray < 0)
            or not np.isfinite(cosines).all() or np.any(cosines < 0) or np.any(cosines > 1)):
        raise ValueError("Expected finite NHW3 linear colors in [0, 1], nonnegative NHW weights and NHW cosines in [0, 1]")
    if (isinstance(referenceViewIndex, (bool, np.bool_))
            or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < len(values)):
        raise ValueError("referenceViewIndex must identify an available view")
    parameters = np.asarray([referenceBlendWidth, referenceMinCosine, referenceFullCosine], np.float64)
    if (not np.isfinite(parameters).all() or referenceBlendWidth < 0
            or not 0 <= referenceMinCosine < referenceFullCosine <= 1):
        raise ValueError("Expected nonnegative finite blend width and 0 <= minimum cosine < full cosine <= 1")
    if not isinstance(matchAuxiliaryColors, (bool, np.bool_)):
        raise ValueError("matchAuxiliaryColors must be boolean")

    referenceValid = weightArray[referenceViewIndex] > 0
    paddedSupport = np.pad(referenceValid, 1, mode="constant", constant_values=False)
    boundaryDistance = distance_transform_edt(paddedSupport)[1:-1, 1:-1] - 1
    if referenceBlendWidth > 0:
        boundaryAlpha = _smoothStep(boundaryDistance / referenceBlendWidth)
    else:
        boundaryAlpha = referenceValid.astype(np.float32)
    angleAlpha = _smoothStep((cosines[referenceViewIndex] - referenceMinCosine)
                             / (referenceFullCosine - referenceMinCosine))
    referenceAlpha = (boundaryAlpha * angleAlpha).astype(np.float32)
    referenceAlpha[~referenceValid] = 0
    referenceCoreMask = referenceValid & (referenceAlpha == 1)

    auxiliaryWeights = weightArray.copy()
    auxiliaryWeights[referenceViewIndex] = 0
    auxiliarySource = np.argmax(auxiliaryWeights, axis=0).astype(np.int32)
    auxiliaryWeight = np.take_along_axis(auxiliaryWeights, auxiliarySource[None], axis=0)[0]
    auxiliaryValid = auxiliaryWeight > 0
    auxiliarySource[~auxiliaryValid] = -1
    referenceOnly = referenceValid & ~auxiliaryValid
    referenceAlpha[referenceOnly] = 1
    coverage = referenceValid | auxiliaryValid

    gains, gainDiagnostics = _matchAuxiliaryGains(values, weightArray, cosines, referenceViewIndex,
                                                 referenceFullCosine, matchAuxiliaryColors)
    yy, xx = np.indices(coverage.shape)
    selectedAuxiliary = values[np.maximum(auxiliarySource, 0), yy, xx].copy()
    selectedAuxiliary *= gains[np.maximum(auxiliarySource, 0)]
    selectedAuxiliary[~auxiliaryValid] = 0
    clippedAuxiliary = auxiliaryValid & (referenceAlpha < 1) & np.any(selectedAuxiliary > 1, axis=-1)
    np.clip(selectedAuxiliary, 0, 1, out=selectedAuxiliary)
    linearAlbedo = (referenceAlpha[..., None] * values[referenceViewIndex]
                    + (1 - referenceAlpha[..., None]) * selectedAuxiliary)
    linearAlbedo[~coverage] = 0

    mixed = coverage & (referenceAlpha > 0) & (referenceAlpha < 1)
    sourceIndex = auxiliarySource.copy()
    sourceIndex[referenceAlpha == 1] = referenceViewIndex
    sourceIndex[mixed] = -2
    sourceIndex[~coverage] = -1
    confidence = (referenceAlpha * np.clip(weightArray[referenceViewIndex], 0, 1)
                  + (1 - referenceAlpha) * np.clip(auxiliaryWeight, 0, 1))
    confidence[~coverage] = 0
    return {"linearAlbedo": linearAlbedo, "coverageMask": coverage,
            "confidence": confidence.astype(np.float32), "sourceIndex": sourceIndex,
            "referenceAlpha": referenceAlpha, "referenceCoreMask": referenceCoreMask,
            "auxiliarySourceIndex": auxiliarySource,
            "diagnostics": {"fusion": "referenceFirstBestAuxiliary", "referenceViewIndex": int(referenceViewIndex),
                            "referenceBlendWidthUvPixels": float(referenceBlendWidth),
                            "referenceMinCosine": float(referenceMinCosine),
                            "referenceFullCosine": float(referenceFullCosine),
                            "observedTexels": int(coverage.sum()),
                            "referenceSupportedTexels": int(referenceValid.sum()),
                            "referenceCoreTexels": int(referenceCoreMask.sum()),
                            "referenceOnlyFallbackTexels": int(referenceOnly.sum()),
                            "referenceOwnedTexels": int((referenceAlpha == 1).sum()),
                            "blendedTexels": int(mixed.sum()),
                            "auxiliaryOnlyTexels": int((coverage & (referenceAlpha == 0)).sum()),
                            "auxiliaryGainClippedTexels": int(clippedAuxiliary.sum()),
                            "sourceIndexSemantics": "-1 uncovered; -2 reference/auxiliary mixture; otherwise unmixed view index",
                            "confidenceSemantics": "clipped alpha-weighted source weight; not a calibrated probability",
                            "colorMatching": {"enabled": bool(matchAuxiliaryColors), "minimumSamplesPerChannel": 32,
                                              "gainLimits": [2 / 3, 1.5], "views": gainDiagnostics},
                            "limitations": ["Reference ownership does not correct geometric correspondences or illumination.",
                                            "Best auxiliary source switches can form seams; no auxiliary feature alignment is performed.",
                                            "Global RGB gain matching cannot correct local lighting or white-balance nonlinearities."]}}
