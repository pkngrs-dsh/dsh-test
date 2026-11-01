"""Local experiment: adapt template skin color to an unchanged SH face atlas.

Only the supplied fallback atlas is returned. No observation coverage, source
selection, geometry, or photographic pixel is rewritten here. A smooth RGB
multiplier retains the template's fine texture; it is not inferred reflectance.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from .skinExtrapolation import smoothstep


def _rgb(value, name, shape=None):
    value = np.asarray(value, np.float64)
    if (value.ndim != 3 or value.shape[-1] != 3 or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()):
        raise ValueError(f"{name} must be finite linear RGB in [0,1], with matching atlas dimensions")
    return value


def _mask(provenance, key, shape, *, optional=False):
    if key not in provenance and optional:
        return np.zeros(shape, bool)
    value = np.asarray(provenance[key])
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{key} must be a boolean atlas mask")
    return value


def _alpha(provenance, key, shape):
    value = np.asarray(provenance[key], np.float64)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{key} must be a finite atlas weight in [0,1]")
    return value


def matchTemplateToSh(templateLinear, shLinear, odLinear, provenance):
    """Return an adapted, already-gained fallback for the unchanged SH bake.

    ``templateLinear`` must be the exact fallback used by the original SH
    branch, including its existing global gain. Pass this returned atlas to
    the baker with ``fallbackGain=None`` to avoid applying that gain twice.

    ``provenance`` contains the SH bake maps and the additional boolean map
    ``openDelightBoundarySafeSkinDonor``. Optional
    ``openDelightDirectPhotoAlpha`` further restricts common donors. Colors
    are sampled only on shared, pure-photo, unmodified safe interior skin.

    Global per-channel gains are limited to [0.65,1.75]. A mask-normalized
    Gaussian of log residuals adds a weak local multiplier in [0.8,1.25],
    fading back to the global gain 128/1024 atlas widths from a donor. The
    complete multiplier is bounded to [0.5,2]. Local data are never shared
    across connected UV islands or disjoint permitted skin regions.
    """
    template = _rgb(templateLinear, "templateLinear")
    sh = _rgb(shLinear, "shLinear", template.shape)
    od = _rgb(odLinear, "odLinear", template.shape)
    shape = template.shape[:2]
    if shape[0] != shape[1] or shape[0] < 32:
        raise ValueError("Expected a square client atlas of at least 32 pixels")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must contain the existing SH coverage and donor maps")
    scale = shape[0] / 1024.
    domain = _mask(provenance, "atlasMask", shape)
    observed = _mask(provenance, "coverageMask", shape)
    protected = _mask(provenance, "boundaryProtectedFeatures", shape)
    donors = (_mask(provenance, "boundarySafeSkinDonor", shape)
              & _mask(provenance, "openDelightBoundarySafeSkinDonor", shape)
              & observed & domain & ~protected
              & (_alpha(provenance, "boundaryDirectPhotoAlpha", shape) >= .999))
    for key in ("boundaryModified", "filterPadding", "gutter", "interiorSkinHoleFilled"):
        donors &= ~_mask(provenance, key, shape, optional=True)
    if "openDelightDirectPhotoAlpha" in provenance:
        donors &= _alpha(provenance, "openDelightDirectPhotoAlpha", shape) >= .999
    donors &= ((template.min(-1) > .03) & (template.max(-1) < .95)
               & (sh.min(-1) > .015) & (sh.max(-1) < .985)
               & (od.min(-1) > .015) & (od.max(-1) < .985))
    if donors.sum() < 128:
        raise ValueError("At least 128 common pure-photo skin donors are required for template matching")

    epsilon = 1e-4
    logDifference = np.log(np.maximum(sh, epsilon)) - np.log(np.maximum(template, epsilon))
    samples = logDifference[donors]
    median = np.median(samples, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(samples-median), axis=0), .035)
    robust = np.max(np.abs(samples-median)/spread, axis=-1) <= 3.5
    qualified = np.zeros(shape, bool)
    qualified[donors] = robust
    if qualified.sum() < 128:
        raise ValueError("Insufficient robust skin donors after log-color outlier rejection")
    rawGain = np.exp(np.median(logDifference[qualified], axis=0))
    gain = np.clip(rawGain, .65, 1.75)
    globalLog = np.log(gain)
    residual = logDifference-globalLog

    skinGate = np.zeros(shape, np.float32)
    localLog = np.zeros(template.shape, np.float64)
    localConfidence = np.zeros(shape, np.float32)
    charts, chartCount = label(domain)
    regionReports = []
    luminanceVector = np.array([.2126, .7152, .0722])
    # The unadapted template establishes barriers so a successful color match
    # cannot turn original scalp/hair or dark cavities into new skin targets.
    referenceRgb = template[qualified]
    referenceChroma = referenceRgb / np.maximum(referenceRgb.sum(-1, keepdims=True), epsilon)
    chromaCenter = np.median(referenceChroma, axis=0)
    chromaSpread = np.maximum(1.4826*np.median(np.abs(referenceChroma-chromaCenter), axis=0), .018)
    referenceLuminance = float(np.median(referenceRgb @ luminanceVector))
    sigma = max(32*scale, 1.)
    nearDistance, farDistance = 24*scale, 128*scale
    for chartId in range(1, chartCount+1):
        chart = charts == chartId
        normalization = gaussian_filter(chart.astype(np.float64), max(3*scale, .5))
        lowTemplate = gaussian_filter(template*chart[..., None], (max(3*scale, .5), max(3*scale, .5), 0))
        lowTemplate /= np.maximum(normalization[..., None], 1e-12)
        chroma = lowTemplate / np.maximum(lowTemplate.sum(-1, keepdims=True), epsilon)
        chromaDistance = np.max(np.abs(chroma-chromaCenter)/chromaSpread, axis=-1)
        luminance = lowTemplate @ luminanceVector
        relativeLuminance = luminance / max(referenceLuminance, epsilon)
        gate = (1-smoothstep((chromaDistance-2.)/1.5))
        gate *= smoothstep((relativeLuminance-.20)/.30)
        gate *= 1-smoothstep((relativeLuminance-1.70)/.70)
        gate *= smoothstep((luminance-.035)/.045)
        # Features prohibit donors, not fallback skin targets: uncovered skin
        # beside a brow still needs the same complexion. Dark eyes, cavities
        # and hair remain excluded by the original-template color gate above.
        gate *= chart
        skinGate[chart] = gate[chart]
        regions, regionCount = label(chart & (gate > .02))
        donorCounts = np.bincount(regions[qualified], minlength=regionCount+1)
        for regionId in np.flatnonzero(donorCounts >= 32):
            if regionId == 0:
                continue
            region = regions == regionId
            regionDonors = qualified & region
            donorCount = int(regionDonors.sum())
            if donorCount < 32:
                continue
            normalization = gaussian_filter(regionDonors.astype(np.float64), sigma)
            local = gaussian_filter(residual*regionDonors[..., None], (sigma, sigma, 0))
            local /= np.maximum(normalization[..., None], 1e-12)
            local = np.clip(local, np.log(.8), np.log(1.25))
            distance = distance_transform_edt(~regionDonors)
            confidence = 1-smoothstep((distance-nearDistance)/(farDistance-nearDistance))
            confidence *= region & (normalization > 1e-6)
            localLog[region] = (local*confidence[..., None])[region]
            localConfidence[region] = confidence[region]
            regionReports.append({"chart": int(chartId), "region": int(regionId), "donorTexels": donorCount,
                                  "localAdjustmentTexels": int((confidence > 0).sum())})

    multiplier = np.clip(np.exp(globalLog+localLog), .5, 2.)
    # Geometric interpolation of the multiplier preserves relative template
    # detail. The gate changes only its strength, never substitutes flat RGB.
    effectiveMultiplier = np.exp(np.log(multiplier)*skinGate[..., None])
    # Carry the boundary multiplier into a small unused UV gutter so bilinear
    # texture filtering does not immediately sample the old template color.
    # Each padding pixel has one nearest atlas owner; it never supplies donors
    # and never changes either atlas coverage or source observation coverage.
    outsideDistance, nearest = distance_transform_edt(~domain, return_indices=True)
    paddingRadius = max(4*scale, 2.)
    padding = (~domain & (outsideDistance <= paddingRadius)
               & (skinGate[nearest[0], nearest[1]] > 0))
    paddingWeight = 1-smoothstep((outsideDistance-1.)/(paddingRadius-1.))
    nearestMultiplier = effectiveMultiplier[nearest[0], nearest[1]]
    effectiveMultiplier[padding] = np.exp(np.log(nearestMultiplier[padding])*paddingWeight[padding, None])
    unbounded = template*effectiveMultiplier
    clipped = (unbounded > 1).any(-1) & ((skinGate > 0) | padding)
    adapted = np.clip(unbounded, 0, 1).astype(np.float32)
    unchanged = (skinGate == 0) & ~padding
    adapted[unchanged] = np.asarray(templateLinear, np.float32)[unchanged]
    before = np.abs(sh[qualified]-template[qualified])
    after = np.abs(sh[qualified]-adapted[qualified])
    maps = {
        "templateMatchSafeDonor": qualified,
        "templateMatchSkinWeight": skinGate,
        "templateMatchLocalConfidence": localConfidence,
        "templateMatchFilterPadding": padding,
        "templateMatchModified": np.any(adapted != np.asarray(templateLinear, np.float32), axis=-1),
        "templateMatchClipped": clipped,
    }
    diagnostics = {
        "method": "commonSafeSkinGlobalAndLocalLogTemplateColor", "experimental": True,
        "inputTemplateAlreadyGained": True, "bakerFallbackGainRequired": "None or unit RGB gain",
        "commonDonorTexelsBeforeRobustGate": int(donors.sum()), "safeDonorTexels": int(qualified.sum()),
        "rawGlobalRgbGain": rawGain.tolist(), "globalRgbGain": gain.tolist(),
        "globalGainBounds": [.65, 1.75], "localGainBounds": [.8, 1.25], "combinedGainBounds": [.5, 2.],
        "localGaussianSigmaPixels": sigma, "localFullStrengthDistancePixels": nearDistance,
        "localZeroStrengthDistancePixels": farDistance,
        "filterPaddingRadiusPixels": paddingRadius, "filterPaddingTexels": int(padding.sum()),
        "templateBarrierSource": "unadapted template low-frequency chroma/luminance",
        "featureProtectionPolicy": "exclude photographic features from donors; allow skin-colored fallback within feature exclusions",
        "uvChartCount": int(chartCount),
        "localRegions": regionReports, "modifiedTexels": int(maps["templateMatchModified"].sum()),
        "modifiedUnobservedTexels": int((maps["templateMatchModified"] & ~observed).sum()),
        "clippedTexels": int(clipped.sum()),
        "featureExcludedFallbackTexelsAdapted": int((maps["templateMatchModified"] & protected).sum()),
        "featureExcludedUnobservedFallbackTexelsAdapted": int((maps["templateMatchModified"] & protected & ~observed).sum()),
        "outsideAtlasExceptPaddingExactlyUnchanged": bool(np.array_equal(
            adapted[~(domain | padding)], np.asarray(templateLinear, np.float32)[~(domain | padding)])),
        "coverageUnchanged": True, "photographicAtlasModified": False,
        "donorFitMeanAbsoluteLinearRgbBefore": float(before.mean()),
        "donorFitMeanAbsoluteLinearRgbAfter": float(after.mean()),
        "donorFitP95MeanAbsoluteLinearRgbAfter": float(np.percentile(after.mean(-1), 95)),
        "diagnosticMeaning": "fit-region template color mismatch, not held-out reflectance or seam accuracy",
        "limitations": [
            "Target skin gating is a color heuristic, not positive semantic segmentation; feature exclusions constrain only donors.",
            "Unobserved ear/neck skin uses global color adaptation; its true reflectance is not observed.",
            "Local UV smoothing stays within connected permitted raster regions, not surface-geodesic distance.",
            "Smooth multiplication retains template detail including any original template shading.",
            "Head/torso texture compatibility beyond this head asset is not evaluated.",
        ],
    }
    return {"linearRGB": adapted, "maps": maps, "diagnostics": diagnostics,
            "debug": {"effectiveRgbGain": effectiveMultiplier.astype(np.float32)}}


def matchTemplateDetail(templateLinear, shLinear, provenance, skinWeight, *, sigmaPixels=2.):
    """Reduce excess template detail while retaining its adapted low-frequency color.

    This consumes the matched *fallback*, not the already composited face.
    Common, pure-photo donors estimate the ratio of SH/template log-luminance
    high-pass MAD. The ratio is clipped to [0.2,1]: no detail is amplified and
    at least 20 percent of the original template high-frequency signal remains.
    A chart-normalized log-RGB split then attenuates detail through the supplied
    soft skin weight. Hair, dark cavities and zero-weight targets stay unchanged.

    Returned maps are HW booleans or weights in [0,1]. Observation/source maps
    are never modified. The caller must apply the fallback delta only through
    the existing template-contribution weight in an already composited atlas.
    """
    template = _rgb(templateLinear, "templateLinear")
    sh = _rgb(shLinear, "shLinear", template.shape)
    shape = template.shape[:2]
    if (shape[0] != shape[1] or shape[0] < 32 or not isinstance(provenance, dict)
            or isinstance(sigmaPixels, (bool, np.bool_)) or not np.isfinite(sigmaPixels)
            or not 0 < sigmaPixels <= 16):
        raise ValueError("Expected a square atlas and finite sigmaPixels in (0,16]")
    sigma = float(sigmaPixels)
    skin = np.asarray(skinWeight, np.float64)
    if skin.shape != shape or not np.isfinite(skin).all() or (skin < 0).any() or (skin > 1).any():
        raise ValueError("skinWeight must be a finite HW weight in [0,1]")
    domain = _mask(provenance, "atlasMask", shape)
    observed = _mask(provenance, "coverageMask", shape)
    protected = _mask(provenance, "boundaryProtectedFeatures", shape)
    support = domain & observed & ~protected
    for key in ("boundaryModified", "filterPadding", "gutter", "interiorSkinHoleFilled"):
        support &= ~_mask(provenance, key, shape, optional=True)
    donors = (support & _mask(provenance, "boundarySafeSkinDonor", shape)
              & _mask(provenance, "openDelightBoundarySafeSkinDonor", shape)
              & (_alpha(provenance, "boundaryDirectPhotoAlpha", shape) >= .999)
              & (skin >= .5))
    if "openDelightDirectPhotoAlpha" in provenance:
        donors &= _alpha(provenance, "openDelightDirectPhotoAlpha", shape) >= .999
    # Avoid counting SH/template transitions or feature edges as photograph
    # texture. Padding defines an outside even for charts touching the image.
    interior = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    donors &= interior > 3*sigma
    lumaWeights = np.array([.2126, .7152, .0722])
    templateLuma, shLuma = template @ lumaWeights, sh @ lumaWeights
    donors &= ((templateLuma > .03) & (templateLuma < .95)
               & (shLuma > .03) & (shLuma < .95))
    if donors.sum() < 128:
        raise ValueError("At least 128 shared interior skin donors are required to match template detail")
    epsilon = 1e-4
    templateLogLuma = np.log(np.maximum(templateLuma, epsilon))
    shLogLuma = np.log(np.maximum(shLuma, epsilon))
    templateHigh = np.zeros(shape, np.float64)
    shHigh = np.zeros(shape, np.float64)
    charts, chartCount = label(domain)
    for chartId in range(1, chartCount+1):
        chart = charts == chartId
        fitSupport = chart & support
        if not (donors & chart).any():
            continue
        denominator = gaussian_filter(fitSupport.astype(np.float64), sigma)
        for logImage, high in ((templateLogLuma, templateHigh), (shLogLuma, shHigh)):
            low = gaussian_filter(logImage*fitSupport, sigma)/np.maximum(denominator, 1e-12)
            high[chart] = (logImage-low)[chart]
    def mad(values):
        return float(1.4826*np.median(np.abs(values-np.median(values))))
    templateMad, shMad = mad(templateHigh[donors]), mad(shHigh[donors])
    rawRatio = shMad/templateMad if templateMad > 1e-8 else 1.
    retention = float(np.clip(rawRatio, .2, 1.))

    logTemplate = np.log(np.maximum(template, epsilon))
    adjusted = template.copy()
    detailWeight = (skin*(1-retention)*domain).astype(np.float32)
    regionCountTotal = 0
    # Split even one UV chart into disjoint permitted skin regions. Normalized
    # convolution never includes a separate island, hair or zero-weight cavity.
    for chartId in range(1, chartCount+1):
        regions, count = label((charts == chartId) & (skin > 1e-6))
        regionCountTotal += int(count)
        for regionId in range(1, count+1):
            region = regions == regionId
            weights = skin*region
            denominator = gaussian_filter(weights, sigma)
            low = gaussian_filter(logTemplate*weights[..., None], (sigma, sigma, 0))
            low /= np.maximum(denominator[..., None], 1e-12)
            target = logTemplate + detailWeight[..., None]*(low-logTemplate)
            adjusted[region] = np.exp(target[region])
    scale = shape[0]/1024.
    paddingRadius = max(4*scale, 2.)
    distance, nearest = distance_transform_edt(~domain, return_indices=True)
    padding = (~domain & (distance <= paddingRadius)
               & (detailWeight[nearest[0], nearest[1]] > 0))
    # Extend a multiplicative delta into native texture padding. This remains
    # filtering support, never new observation coverage or a smoothing donor.
    deltaLog = np.log(np.maximum(adjusted, epsilon))-logTemplate
    paddingWeight = 1-smoothstep((distance-1.)/(paddingRadius-1.))
    adjusted[padding] = (template[padding]*np.exp(deltaLog[nearest[0][padding], nearest[1][padding]]
                                                * paddingWeight[padding, None]))
    clipped = ((adjusted < 0) | (adjusted > 1)).any(-1)
    result = np.clip(adjusted, 0, 1).astype(np.float32)
    unchanged = (detailWeight == 0) & ~padding
    result[unchanged] = np.asarray(templateLinear, np.float32)[unchanged]
    maps = {"templateDetailSafeDonor": donors,
            "templateDetailReductionWeight": detailWeight,
            "templateDetailFilterPadding": padding,
            "templateDetailModified": np.any(result != np.asarray(templateLinear, np.float32), axis=-1),
            "templateDetailClipped": clipped}
    diagnostics = {
        "method": "sharedSkinLogLuminanceMadTemplateDetail", "experimental": True,
        "donorTexels": int(donors.sum()), "sigmaPixels": sigma,
        "featureAndBoundaryBufferPixels": 3*sigma,
        "templateLogLuminanceResidualMad": templateMad, "shLogLuminanceResidualMad": shMad,
        "rawShToTemplateDetailRatio": float(rawRatio), "retainedTemplateDetailFraction": retention,
        "retentionBounds": [.2, 1.], "detailAmplificationAllowed": False,
        "targetFilter": "skin-weighted normalized log RGB Gaussian within each connected UV/skin region",
        "lowFrequencyPolicy": "retain template Gaussian log-RGB component; attenuate only residual",
        "uvChartCount": int(chartCount), "skinRegionCount": int(regionCountTotal),
        "filterPaddingRadiusPixels": float(paddingRadius), "filterPaddingTexels": int(padding.sum()),
        "modifiedTexels": int(maps["templateDetailModified"].sum()), "clippedTexels": int(clipped.sum()),
        "zeroSkinWeightExceptPaddingExactlyUnchanged": bool(np.array_equal(
            result[unchanged], np.asarray(templateLinear, np.float32)[unchanged])),
        "coverageUnchanged": True, "photographicAtlasModified": False,
        "diagnosticMeaning": "shared donor high-frequency strength ratio, not image quality, accuracy or identity",
        "limitations": [
            "SH observations include sampling and prior image smoothing; this ratio is not a biological skin-detail measurement.",
            "At least 20 percent of template detail remains even when the measured ratio is lower.",
            "The target skin weight remains heuristic; original template shadows broader than sigma are preserved.",
            "Color is preserved in the low-frequency log-RGB component, not as exact arithmetic neighborhood means.",
        ],
    }
    return {"linearRGB": result, "maps": maps, "diagnostics": diagnostics}
