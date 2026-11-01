"""Experimental CPU multi-view SH delighting in a shared object-space UV atlas.

All RGB arrays are linear-light values. No transfer function is guessed here.
SH/albedo decomposition is underdetermined; the explicit gauge and optional
low-frequency prior are regularizers, not a claim of intrinsic ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def shBasis(normals: np.ndarray) -> np.ndarray:
    """Real orthonormal SH through degree 2, ordered 1,y,z,x,xy,yz,z2,xz,x2-y2.

    The nine columns already include their standard normalization constants.
    RGB coefficients model irradiance directly; no extra Lambertian convolution
    constants are applied. Axes are the caller's common object-space axes.
    """
    normals = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-8)
    x, y, z = np.moveaxis(unit, -1, 0)
    return np.stack([np.full_like(x, 0.28209479177387814),
                     0.4886025119029199 * y, 0.4886025119029199 * z,
                     0.4886025119029199 * x, 1.0925484305920792 * x * y,
                     1.0925484305920792 * y * z, 0.31539156525252005 * (3 * z * z - 1),
                     1.0925484305920792 * x * z, 0.5462742152960396 * (x * x - y * y)], axis=-1)


def _weightedMedian(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sortedWeights = weights[order]
    index = np.searchsorted(np.cumsum(sortedWeights), sortedWeights.sum() / 2)
    return float(values[order[min(index, len(order) - 1)]])


def _blur(image: np.ndarray, weights: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = gaussian_filter(weights.astype(np.float32), sigma=sigma, mode="constant")
    numerator = gaussian_filter((image * weights[..., None]).astype(np.float32),
                                sigma=(sigma, sigma, 0), mode="constant")
    blurred = np.divide(numerator, denominator[..., None], out=np.zeros_like(numerator),
                        where=denominator[..., None] > 1e-10)
    return blurred, denominator


def _huberWeights(residual: np.ndarray, floor: float = 0.005) -> np.ndarray:
    center = np.median(residual)
    scale = max(floor, float(1.4826 * np.median(np.abs(residual - center))))
    return np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))


def _fitLighting(basis: np.ndarray, albedo: np.ndarray, image: np.ndarray,
                  weights: np.ndarray, initial: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    active = weights > 0
    basis, albedo, image, weights = basis[active], albedo[active], image[active], weights[active]
    weights = weights / weights.sum()
    coefficients = initial.copy()
    degreePenalty = np.array([0.01, 1, 1, 1, 4, 4, 4, 4, 4], dtype=np.float64)
    neutral = np.zeros(9)
    neutral[0] = 1 / 0.28209479177387814
    condition = 0.0
    for channel in range(3):
        design = basis * albedo[:, channel, None]
        for _ in range(3):
            residual = design @ coefficients[:, channel] - image[:, channel]
            robust = weights * _huberWeights(residual)
            gram = design.T @ (robust[:, None] * design)
            strength = ridge * max(float(np.trace(gram) / 9), 1e-8)
            regularizer = strength * degreePenalty
            system = gram + np.diag(regularizer)
            target = design.T @ (robust * image[:, channel]) + regularizer * neutral
            coefficients[:, channel] = np.linalg.solve(system, target)
            condition = max(condition, float(np.linalg.cond(system)))
    return coefficients, condition


def _validateInputs(observations, weights, normals, priorAlbedo, mode, iterations):
    observations = np.asarray(observations)
    weights, normals = np.asarray(weights), np.asarray(normals)
    if (observations.ndim != 4 or observations.shape[-1] != 3 or observations.shape[0] < 1
            or min(observations.shape[1:3]) < 4):
        raise ValueError("observations must have shape (N,H,W,3), N>=1 and H,W>=4")
    if weights.shape != observations.shape[:3] or normals.shape != observations.shape[1:]:
        raise ValueError("weights (N,H,W) and normals (H,W,3) must match observations")
    for value, name in ((observations, "observations"), (weights, "weights"), (normals, "normals")):
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric values")
    if observations.min() < 0 or observations.max() > 1:
        raise ValueError("observations must be linear RGB in [0,1]")
    if weights.min() < 0:
        raise ValueError("weights must be nonnegative")
    if mode not in ("shOnly", "shPrior"):
        raise ValueError("mode must be shOnly or shPrior")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 50:
        raise ValueError("iterations must be an integer in [1,50]")
    if priorAlbedo is not None:
        priorAlbedo = np.asarray(priorAlbedo)
        if (priorAlbedo.shape != observations.shape[1:] or priorAlbedo.dtype.kind not in "fiu"
                or not np.isfinite(priorAlbedo).all() or priorAlbedo.min() < 0 or priorAlbedo.max() > 1):
            raise ValueError("priorAlbedo must be finite approximate linear RGB (H,W,3) in [0,1]")
    elif mode == "shPrior":
        raise ValueError("shPrior requires priorAlbedo")
    return (observations.astype(np.float32, copy=False), weights.astype(np.float32, copy=False),
            normals.astype(np.float64, copy=False), priorAlbedo)


def solveShDelight(observations: np.ndarray, weights: np.ndarray, normals: np.ndarray, *,
                   priorAlbedo: np.ndarray | None = None, mode: str = "shOnly",
                   iterations: int = 8, gaugeReferenceView: int | None = None) -> dict:
    """Alternate shared low-frequency albedo and per-view RGB SH, then divide RGB.

    A fixed weighted mean of low-frequency albedo fixes the three RGB scale
    gauges. shOnly derives that mean from robust observed RGB statistics;
    shPrior derives it from the supplied approximate-linear OpenDelight prior.
    This convention cannot determine absolute reflectance or true illuminant
    color. shOnly completely ignores prior colors even when a prior is passed.
    With gaugeReferenceView, shOnly takes its RGB gauge from that one view;
    other views still constrain the lighting fit. This is not calibration.

    Final division uses original, unblurred observations. Invalid/unsafe pixels
    receive zero corrected RGB and zero weight, never a clamped substitute.
    Coefficients have shape (N,9,3); shading and corrected RGB are (N,H,W,3).
    Missing normal support and views with <12 low-frequency samples are omitted.
    """
    observations, weights, normals, priorAlbedo = _validateInputs(
        observations, weights, normals, priorAlbedo, mode, iterations)
    viewCount, height, width, _ = observations.shape
    if gaugeReferenceView is not None:
        if (isinstance(gaugeReferenceView, (bool, np.bool_))
                or not isinstance(gaugeReferenceView, (int, np.integer))
                or not 0 <= gaugeReferenceView < viewCount or mode != "shOnly"):
            raise ValueError("gaugeReferenceView must be a valid view index in shOnly mode")
    darkThreshold, saturationThreshold = 0.015, 0.985
    minShading, maxShading, maxGain = 0.25, 4.0, 4.0
    ridge, priorStrength, sigma = 0.02, (0.25 if mode == "shPrior" else 0.0), max(0.8, min(height, width) / 64)
    normalValid = np.linalg.norm(normals, axis=-1) > 1e-8
    luminance = observations @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dark = luminance < darkThreshold
    saturated = observations.max(axis=-1) >= saturationThreshold
    inputSupport = weights > 0
    eligible = inputSupport & normalValid[None] & ~dark & ~saturated
    if not eligible.any():
        raise ValueError("No usable observations remain after normal, dark and saturation checks")
    # Scale weights globally for stable arithmetic without changing their ratios.
    normalizedWeights = weights / max(float(weights.max()), 1e-12)
    fitWeights = normalizedWeights * eligible
    basis = shBasis(normals).astype(np.float32)
    ys = np.rint(np.linspace(0, height - 1, min(height, 96))).astype(int)
    xs = np.rint(np.linspace(0, width - 1, min(width, 96))).astype(int)
    lowImages, lowBases, lowWeights = [], [], []
    for viewIndex in range(viewCount):
        lowImage, lowWeight = _blur(observations[viewIndex], fitWeights[viewIndex], sigma)
        lowBasis, _ = _blur(basis, fitWeights[viewIndex], sigma)
        lowImages.append(lowImage[ys[:, None], xs])
        lowBases.append(lowBasis[ys[:, None], xs])
        lowWeights.append(lowWeight[ys[:, None], xs] * eligible[viewIndex, ys[:, None], xs])
    lowImages, lowBases, lowWeights = (np.asarray(value, dtype=np.float64)
                                      for value in (lowImages, lowBases, lowWeights))
    lowShape = lowWeights.shape[1:]
    usableViews = (lowWeights > 0).sum(axis=(1, 2)) >= 12
    lowWeights[~usableViews] = 0
    gaugeWeights = lowWeights.sum(axis=0)
    if not usableViews.any() or not (gaugeWeights > 0).any():
        raise ValueError("No view has at least 12 usable low-frequency SH samples")
    anchorSamples = lowImages[lowWeights > 0]
    anchorWeights = lowWeights[lowWeights > 0]
    if gaugeReferenceView is not None:
        if not usableViews[gaugeReferenceView]:
            raise ValueError("The gauge reference view has insufficient usable observations")
        referenceSupport = lowWeights[gaugeReferenceView] > 0
        anchorSamples = lowImages[gaugeReferenceView][referenceSupport]
        anchorWeights = lowWeights[gaugeReferenceView][referenceSupport]
    anchor = np.array([_weightedMedian(anchorSamples[:, channel], anchorWeights) for channel in range(3)])
    lowPrior = np.zeros((*lowShape, 3), dtype=np.float64)
    priorSupport = np.zeros(lowShape, dtype=bool)
    if mode == "shPrior":
        # Black holes in an unfilled prior atlas carry no albedo constraint.
        priorMask = (priorAlbedo.mean(axis=-1) > 0.005) & normalValid & eligible.any(axis=0)
        priorBlur, priorWeight = _blur(priorAlbedo, priorMask.astype(np.float32), sigma)
        lowPrior = priorBlur[ys[:, None], xs].astype(np.float64)
        priorSupport = (priorWeight[ys[:, None], xs] > 0.1) & (gaugeWeights > 0)
        if not priorSupport.any():
            raise ValueError("shPrior has no usable nonblack prior overlapping observed normals")
        anchor = np.average(lowPrior[priorSupport], axis=0, weights=gaugeWeights[priorSupport])
    anchor = np.clip(anchor, 0.02, 0.9)
    albedo = np.broadcast_to(anchor, (*lowShape, 3)).copy()
    if mode == "shPrior":
        albedo[priorSupport] = lowPrior[priorSupport]
    coefficients = np.zeros((viewCount, 9, 3), dtype=np.float64)
    coefficients[:, 0] = 1 / 0.28209479177387814
    trace = []
    for iterationIndex in range(iterations):
        maxCondition = 0.0
        for viewIndex in np.flatnonzero(usableViews):
            coefficients[viewIndex], condition = _fitLighting(
                lowBases[viewIndex], albedo, lowImages[viewIndex], lowWeights[viewIndex],
                coefficients[viewIndex], ridge)
            maxCondition = max(maxCondition, condition)
        lighting = np.einsum("nhwj,njc->nhwc", lowBases, coefficients)
        safeLow = ((lighting >= minShading).all(axis=-1) & (lighting <= maxShading).all(axis=-1))
        residual = lighting * albedo[None] - lowImages
        robustWeights = np.zeros_like(lowImages)
        for viewIndex in np.flatnonzero(usableViews):
            supported = (lowWeights[viewIndex] > 0) & safeLow[viewIndex]
            for channel in range(3):
                if supported.any():
                    robustWeights[viewIndex, ..., channel][supported] = _huberWeights(
                        residual[viewIndex, ..., channel][supported])
        activeWeights = lowWeights[..., None] * safeLow[..., None] * robustWeights
        numerator = (activeWeights * lighting * lowImages).sum(axis=0)
        denominator = (activeWeights * lighting ** 2).sum(axis=0)
        priorPenalty = priorStrength * gaugeWeights * priorSupport
        numerator += priorPenalty[..., None] * lowPrior
        denominator += priorPenalty[..., None]
        updated = np.divide(numerator, denominator, out=albedo.copy(), where=denominator > 1e-10)
        # This bounds a latent fit variable only. Unsafe corrected observations
        # are rejected separately rather than made valid by this regularizer.
        updated = np.clip(updated, 0.005, 1.0)
        currentMean = np.average(updated.reshape(-1, 3), axis=0, weights=gaugeWeights.ravel())
        gaugeScale = anchor / np.maximum(currentMean, 1e-8)
        updated *= gaugeScale
        coefficients /= gaugeScale[None, None, :]
        lighting /= gaugeScale[None, None, None, :]
        active = lowWeights > 0
        finalResidual = lighting * updated[None] - lowImages
        trace.append({"iteration": iterationIndex + 1,
                      "weightedRmseLinear": float(np.sqrt(np.sum(lowWeights[..., None] * finalResidual ** 2)
                                                           / (3 * lowWeights.sum()))),
                      "medianAbsoluteResidualLinear": float(np.median(np.abs(finalResidual[active]))),
                      "safeLowFrequencyFraction": float(np.sum(lowWeights * safeLow) / lowWeights.sum()),
                      "albedoMeanRgb": np.average(updated.reshape(-1, 3), axis=0,
                                                   weights=gaugeWeights.ravel()).tolist(),
                      "maxRegularizedSystemCondition": maxCondition,
                      "meanAlbedoChange": float(np.mean(np.abs(updated - albedo)))})
        albedo = updated
    shading = np.einsum("hwj,njc->nhwc", basis, coefficients).astype(np.float32)
    lightingSafe = (np.isfinite(shading).all(axis=-1) & (shading >= minShading).all(axis=-1)
                    & (shading <= maxShading).all(axis=-1))
    safe = eligible & lightingSafe & usableViews[:, None, None]
    corrected = np.divide(observations, shading, out=np.zeros_like(observations),
                          where=safe[..., None])
    rangeSafe = np.isfinite(corrected).all(axis=-1) & (corrected >= 0).all(axis=-1) & (corrected <= 1).all(axis=-1)
    safe &= rangeSafe
    corrected[~safe] = 0
    validWeights = weights * safe
    shared = np.zeros((height, width, 3), dtype=np.float32)
    # Robust view consensus preserves spatial texture: it never blurs corrected
    # RGB. Tiling limits temporary storage for high-resolution atlases.
    flatCorrected = corrected.reshape(viewCount, -1, 3)
    flatWeights = validWeights.reshape(viewCount, -1)
    flatShared = shared.reshape(-1, 3)
    for start in range(0, height * width, 32768):
        end = min(start + 32768, height * width)
        colors, localWeights = flatCorrected[:, start:end], flatWeights[:, start:end]
        order = np.argsort(colors, axis=0)
        sortedWeights = np.take_along_axis(np.broadcast_to(localWeights[..., None], colors.shape), order, axis=0)
        cumulative = np.cumsum(sortedWeights, axis=0)
        medianIndices = np.argmax(cumulative >= cumulative[-1:] / 2, axis=0)
        sortedColors = np.take_along_axis(colors, order, axis=0)
        median = np.take_along_axis(sortedColors, medianIndices[None], axis=0)[0]
        disagreement = np.linalg.norm(colors - median[None], axis=-1)
        consensus = np.minimum(1.0, 0.04 / np.maximum(disagreement, 1e-12))
        localWeights *= consensus
        total = localWeights.sum(axis=0)
        flatShared[start:end] = np.divide((colors * localWeights[..., None]).sum(axis=0), total[:, None],
                                           out=np.zeros_like(median), where=total[:, None] > 0)
    inputCount, eligibleCount, safeCount = int(inputSupport.sum()), int(eligible.sum()), int(safe.sum())
    viewDiagnostics = []
    for viewIndex in range(viewCount):
        viewSupport = inputSupport[viewIndex]
        viewDiagnostics.append({"viewIndex": viewIndex,
                                "status": "fitted" if usableViews[viewIndex] else "insufficientFitSupport",
                                "inputPixelCount": int(viewSupport.sum()),
                                "eligiblePixelCount": int(eligible[viewIndex].sum()),
                                "safePixelCount": int(safe[viewIndex].sum()),
                                "darkRejectedPixelCount": int((viewSupport & dark[viewIndex]).sum()),
                                "saturatedRejectedPixelCount": int((viewSupport & saturated[viewIndex]).sum()),
                                "unsafeShadingPixelCount": int((eligible[viewIndex] & ~lightingSafe[viewIndex]).sum()),
                                "outOfRangeCorrectionPixelCount": int((eligible[viewIndex] & lightingSafe[viewIndex] & ~rangeSafe[viewIndex]).sum())})
    diagnostics = {
        "algorithm": "alternatingSharedLowFrequencyAlbedoAndPerViewRgbSH2_IRLS_Ridge",
        "mode": mode, "status": "experimental" if safeCount else "noSafeCorrection",
        "notGroundTruth": True, "colorEncoding": "linearRGB",
        "sourceNormalFrame": "commonObjectSpace", "shBasisOrder": ["Y00", "Y1y", "Y1z", "Y1x", "Y2xy", "Y2yz", "Y2zz", "Y2xz", "Y2xxMinusYy"],
        "ambiguity": "Albedo and low-frequency lighting are not uniquely identifiable; the gauge fixes scale/color by convention, not physical calibration. Cast shadows, specularity and visibility-dependent illumination exceed this SH model.",
        "gauge": "fixedWeightedMeanLowFrequencyAlbedoRgb",
        "gaugeAnchorRgb": anchor.tolist(),
        "gaugeAnchorSource": ("referenceViewRobustObservedRgb" if gaugeReferenceView is not None
                              else "robustObservedRgb" if mode == "shOnly" else "approximateLinearOpenDelightPrior"),
        "gaugeReferenceView": gaugeReferenceView,
        "priorUsed": mode == "shPrior", "priorProvidedButIgnored": mode == "shOnly" and priorAlbedo is not None,
        "priorEncodingAssumption": "Caller supplied approximate linear OpenDelight colors; no source transfer function is certified" if mode == "shPrior" else None,
        "priorStrength": priorStrength, "ridgeStrength": ridge,
        "ridgeDegreePenalty": [0.01, 1, 1, 1, 4, 4, 4, 4, 4], "lightingIrlsSteps": 3,
        "gaussianSigmaPixels": float(sigma), "fitGridSize": list(lowShape),
        "lowFrequencyApproximation": "Mask-normalized Gaussian means of RGB and SH basis; products use shared low-frequency albedo on a <=96-pixel grid",
        "darkLuminanceThreshold": darkThreshold, "saturationChannelThreshold": saturationThreshold,
        "minimumShading": minShading, "maximumShading": maxShading, "maximumCorrectionGain": maxGain,
        "unsafeCorrectionPolicy": "zeroWeightAndZeroRgb; neverClampAsObservation",
        "inputPixelCount": inputCount, "eligiblePixelCount": eligibleCount, "safePixelCount": safeCount,
        "safeCoverageOfInput": safeCount / inputCount if inputCount else 0.0,
        "sharedObservedPixelCount": int((validWeights.sum(axis=0) > 0).sum()),
        "consensusWeightMeaning": "Input weights times safety gate times cross-view RGB Huber-like agreement; not calibrated probabilities",
        "trace": trace, "views": viewDiagnostics,
    }
    return {"correctedObservations": corrected, "validWeights": validWeights.astype(np.float32),
            "sharedAlbedo": shared, "shading": shading, "coefficients": coefficients.astype(np.float32),
            "diagnostics": diagnostics}
