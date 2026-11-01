"""Fit low-dimensional diffuse photo illumination with frozen albedo/geometry.

The caller supplies linear RGB and object-space normals. RGB SH coefficients
absorb exposure and white balance together; these effects are not separately
identifiable. This diagnostic cannot represent specularity or hard shadows.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .shDelight import shBasis


def _validatePhotoInputs(albedoLinear, observedLinear, normals, validMask,
                         initialCoefficients, maxSamples, ridge):
    albedo, observed, normals = (np.asarray(value) for value in
                                (albedoLinear, observedLinear, normals))
    if albedo.ndim != 3 or albedo.shape[-1] != 3 or min(albedo.shape[:2]) < 4:
        raise ValueError("albedoLinear must have shape (H,W,3), with H,W >= 4")
    if observed.shape != albedo.shape or normals.shape != albedo.shape:
        raise ValueError("observedLinear and normals must match albedoLinear")
    for value, name in ((albedo, "albedoLinear"), (observed, "observedLinear"),
                        (normals, "normals")):
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric values")
    if min(albedo.min(), observed.min()) < 0 or max(albedo.max(), observed.max()) > 1:
        raise ValueError("albedoLinear and observedLinear must be linear RGB in [0,1]")
    mask = np.asarray(validMask)
    if (mask.shape != albedo.shape[:2] or mask.dtype.kind not in "bfiu"
            or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all()):
        raise ValueError("validMask must be a boolean or zero/one array matching (H,W)")
    if (isinstance(maxSamples, (bool, np.bool_)) or not isinstance(maxSamples, (int, np.integer))
            or maxSamples < 12):
        raise ValueError("maxSamples must be an integer >= 12")
    if isinstance(ridge, (bool, np.bool_)) or not np.isscalar(ridge) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative number")
    if initialCoefficients is not None:
        initialCoefficients = np.asarray(initialCoefficients)
        if (initialCoefficients.shape != (9, 3) or initialCoefficients.dtype.kind not in "fiu"
                or not np.isfinite(initialCoefficients).all()):
            raise ValueError("initialCoefficients must be finite with shape (9,3)")
        initialCoefficients = initialCoefficients.astype(np.float64, copy=True)
    return (albedo.astype(np.float64, copy=False), observed.astype(np.float64, copy=False),
            normals.astype(np.float64, copy=False), mask.astype(bool), initialCoefficients)


def _positiveLighting(basis, coefficients, neutral, mask, lower, upper):
    """Keep a single SH function: blend its coefficients, never clip its pixels."""
    predicted = np.einsum("hwj,jc->hwc", basis, coefficients)
    supported = predicted[mask]
    unsafe = ((supported < lower) | (supported > upper)).any(axis=-1)
    alpha = 1.0
    below, above = supported < lower, supported > upper
    if below.any():
        alpha = min(alpha, float(np.min((1 - lower) / (1 - supported[below]))))
    if above.any():
        alpha = min(alpha, float(np.min((upper - 1) / (supported[above] - 1))))
    # Stay inside the bounds despite floating-point roundoff at the boundary.
    if alpha < 1:
        alpha *= 1 - 1e-8
    return neutral + alpha * (coefficients - neutral), alpha, int(unsafe.sum())


def fitPhotoShLighting(albedoLinear: np.ndarray, observedLinear: np.ndarray,
                       normals: np.ndarray, validMask: np.ndarray, *,
                       initialCoefficients: np.ndarray | None = None,
                       maxSamples: int = 20000, ridge: float = 0.02) -> dict:
    """Fit 27 RGB SH coefficients, leaving albedo and geometry exactly fixed.

    A masked Gaussian filters the *design* (albedo times each SH basis) and
    observation before fitting. This keeps their linear relationship exact;
    multiplying separately blurred albedo and normals would not. Deterministic
    uniformly spaced supported-pixel samples and Huber IRLS limit memory and
    suppress local photo outliers. The degree-weighted ridge is centered on a
    supplied initialization, or neutral unit irradiance when absent.

    Illumination outside [0.05,8] on the declared geometric support is recorded
    and pulled toward neutral by one coefficient-space blend. It remains SH,
    without per-pixel clipping or extra shading freedom. ``renderedLinear`` is
    the full, unclipped albedo-times-shading array; use ``validMask`` to evaluate
    it. ``fitMask`` additionally excludes dark/saturated observations and
    uninformative black albedo. These exclusions must not silently change a
    caller's common evaluation mask. Residuals here are in-sample diagnostics,
    not ground-truth reflectance or held-out reconstruction accuracy.
    """
    albedo, observed, normals, declaredMask, initial = _validatePhotoInputs(
        albedoLinear, observedLinear, normals, validMask, initialCoefficients, maxSamples, ridge)
    normalValid = np.linalg.norm(normals, axis=-1) > 1e-8
    evaluationMask = declaredMask & normalValid
    luminance = observed @ np.array([0.2126, 0.7152, 0.0722])
    dark, saturated = luminance < 0.015, observed.max(axis=-1) >= 0.985
    blackAlbedo = albedo.max(axis=-1) < 0.005
    fitMask = evaluationMask & ~dark & ~saturated & ~blackAlbedo
    activeIndices = np.flatnonzero(fitMask)
    if len(activeIndices) < 12:
        raise ValueError("At least 12 usable pixels are required for photo SH lighting")
    selectedIndices = activeIndices[np.rint(np.linspace(
        0, len(activeIndices) - 1, min(len(activeIndices), int(maxSamples)))).astype(int)]
    height, width = declaredMask.shape
    sigma = max(0.8, min(height, width) / 128)
    support = gaussian_filter(fitMask.astype(np.float64), sigma=sigma, mode="constant")
    basis = shBasis(normals)
    neutral = np.zeros((9, 3), dtype=np.float64)
    neutral[0] = 1 / 0.28209479177387814
    lower, upper = 0.05, 8.0
    prior, initialBlend, initialUnsafe = _positiveLighting(
        basis, neutral if initial is None else initial, neutral, evaluationMask, lower, upper)
    coefficients = prior.copy()
    degreePenalty = np.array([0.01, 1, 1, 1, 4, 4, 4, 4, 4], dtype=np.float64)
    sampleSupport = support.ravel()[selectedIndices]
    baseWeights = sampleSupport / sampleSupport.sum()
    conditionNumbers, downweightedFractions = [], []
    for channel in range(3):
        designImage = basis * (albedo[..., channel] * fitMask)[..., None]
        designImage = gaussian_filter(designImage, sigma=(sigma, sigma, 0), mode="constant")
        design = designImage.reshape(-1, 9)[selectedIndices] / sampleSupport[:, None]
        filteredPhoto = gaussian_filter(observed[..., channel] * fitMask,
                                        sigma=sigma, mode="constant")
        target = filteredPhoto.ravel()[selectedIndices] / sampleSupport
        robust = np.ones(len(selectedIndices))
        for _ in range(6):
            weights = baseWeights * robust
            gram = design.T @ (weights[:, None] * design)
            strength = float(ridge) * max(float(np.trace(gram) / 9), 1e-10)
            regularizer = strength * degreePenalty
            system = gram + np.diag(regularizer)
            right = design.T @ (weights * target) + regularizer * prior[:, channel]
            coefficients[:, channel] = np.linalg.lstsq(system, right, rcond=1e-12)[0]
            residual = design @ coefficients[:, channel] - target
            centered = residual - np.median(residual)
            scale = max(0.003, float(1.4826 * np.median(np.abs(centered))))
            robust = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        condition = float(np.linalg.cond(system))
        conditionNumbers.append(condition if np.isfinite(condition) else None)
        downweightedFractions.append(float((robust < 0.5).mean()))
    coefficients, blend, unsafeBefore = _positiveLighting(
        basis, coefficients, neutral, evaluationMask, lower, upper)
    shading = np.einsum("hwj,jc->hwc", basis, coefficients)
    finite = np.isfinite(shading).all(axis=-1)
    safe = finite & (shading >= lower).all(axis=-1) & (shading <= upper).all(axis=-1)
    valid = evaluationMask & safe
    rendered = albedo * shading
    residual = rendered - observed
    diagnostics = {
        "algorithm": "fixedAlbedoMaskedLowFrequencyRgbSH2_HuberIRLS_Ridge",
        "status": "experimental", "colorEncoding": "linearRGB", "normalFrame": "commonObjectSpace",
        "shConvention": "shDelight.shBasis; orthonormal real degree 0-2; direct irradiance",
        "photometricParameterCount": 27, "albedoAndGeometryFrozen": True,
        "metricScope": "inSampleLightingFit", "notGroundTruth": True,
        "limitations": "Diffuse SH only: no specularity, cast shadows or independent exposure/white-balance identification.",
        "declaredPixelCount": int(declaredMask.sum()), "evaluationPixelCount": int(evaluationMask.sum()),
        "fitPixelCount": int(fitMask.sum()), "sampleCount": len(selectedIndices),
        "sampling": "deterministic equally spaced quantiles of supported pixels in spatial scan order",
        "lowFrequencySigmaPixels": sigma, "ridgeStrength": float(ridge), "irlsIterations": 6,
        "initialCoefficientsProvided": initial is not None,
        "initialUnsafeShadingPixelCount": initialUnsafe, "initialCoefficientBlendWithNeutral": initialBlend,
        "unsafeFittedShadingPixelCountBeforeBlend": unsafeBefore, "coefficientBlendWithNeutral": blend,
        "safeShadingRange": [lower, upper], "unsafeFinalShadingPixelCount": int((evaluationMask & ~safe).sum()),
        "invalidNormalPixelCount": int((declaredMask & ~normalValid).sum()),
        "darkExcludedFitPixelCount": int((evaluationMask & dark).sum()),
        "saturatedExcludedFitPixelCount": int((evaluationMask & saturated).sum()),
        "blackAlbedoExcludedFitPixelCount": int((evaluationMask & blackAlbedo).sum()),
        "regularizedSystemConditionPerChannel": conditionNumbers,
        "heavilyDownweightedSampleFractionPerChannel": downweightedFractions,
        "evaluationRmseLinear": float(np.sqrt(np.mean(residual[evaluationMask] ** 2))),
        "fitRmseLinear": float(np.sqrt(np.mean(residual[fitMask] ** 2))),
        "evaluationRenderedRange": [float(rendered[evaluationMask].min()), float(rendered[evaluationMask].max())],
        "evaluationShadingRange": [float(shading[evaluationMask].min()), float(shading[evaluationMask].max())],
        "renderedOutOfUnitRangePixelCount": int((evaluationMask & ((rendered < 0).any(axis=-1)
                                                                          | (rendered > 1).any(axis=-1))).sum()),
        "nonfiniteRenderedPixelCount": int((evaluationMask & ~np.isfinite(rendered).all(axis=-1)).sum()),
    }
    return {"coefficients": coefficients, "shading": shading, "renderedLinear": rendered,
            "validMask": valid, "fitMask": fitMask, "diagnostics": diagnostics}
