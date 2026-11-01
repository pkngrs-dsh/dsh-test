"""Extend low-frequency skin across an observed atlas silhouette before blending.

Distances are measured in client-atlas pixels. Estimated color is kept separate
from measured coverage; eye/mouth holes and disjoint UV islands are not donors
or dilation targets. Optional semantic masks are supplied in this same atlas.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_fill_holes, binary_propagation,
                           distance_transform_edt, gaussian_filter, label)


def smoothstep(value):
    x = np.clip(value, 0, 1).astype(np.float32)
    return x*x*(3-2*x)


def insideDistance(mask):
    # Padding defines an exterior even when an island touches the image frame.
    return distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean atlas mask")
    return value


def _skinColorGate(image, reference, *, luminanceRange=(.45, 1.9)):
    """Relative chroma/luminance gate, not a fixed skin-color or race classifier."""
    samples = image[reference]
    if not len(samples):
        return np.zeros(reference.shape, bool)
    lum = image @ np.array([.2126, .7152, .0722], np.float32)
    anchor = float(np.median(lum[reference]))
    chroma = image / np.maximum(image.sum(axis=-1, keepdims=True), 1e-6)
    median = np.median(chroma[reference], axis=0)
    spread = np.maximum(1.4826*np.median(np.abs(chroma[reference]-median), axis=0), .018)
    chromaGate = np.max(np.abs(chroma-median)/spread, axis=-1) < 3.0
    return (chromaGate & (lum >= max(anchor*luminanceRange[0], .004))
            & (lum <= min(anchor*luminanceRange[1], .98)))


def blendSkinExterior(projected, observed, template, domain, *, safeSkinMask=None,
                      protectedMask=None, confidence=None, seamFeatherPixels=24.,
                      insetPixels=24., extensionPixels=48., donorBandPixels=36.,
                      colorSmoothPixels=3., barrierFeatherPixels=4.):
    """Return linear composite, contribution maps, raw SDF and diagnostics.

    Outside weights fade to the matched template over extensionPixels. Inside
    the silhouette, only the outer inset band is repaired from safe donors.
    The legacy blend is retained at internal holes, protected features and
    wherever no nearby reliable donor is available. Tiny observed fragments
    near the exterior are repaired with the same estimated skin as their
    surroundings; they never become donors or change measured coverage.
    """
    projected, template = np.asarray(projected), np.asarray(template)
    if (projected.ndim != 3 or projected.shape[-1] != 3 or template.shape != projected.shape
            or not np.isfinite(projected).all() or not np.isfinite(template).all()
            or projected.min() < 0 or projected.max() > 1 or template.min() < 0 or template.max() > 1):
        raise ValueError("Expected finite linear RGB atlases in [0,1]")
    shape = projected.shape[:2]
    observed, domain = _mask(observed, shape, "observed"), _mask(domain, shape, "domain")
    semanticProvided = safeSkinMask is not None
    safe = observed.copy() if safeSkinMask is None else _mask(safeSkinMask, shape, "safeSkinMask") & observed
    protected = np.zeros(shape, bool) if protectedMask is None else _mask(protectedMask, shape, "protectedMask")
    quality = np.ones(shape, np.float32) if confidence is None else np.asarray(confidence)
    if quality.shape != shape or not np.isfinite(quality).all() or np.any(quality < 0):
        raise ValueError("Fusion confidence must be finite nonnegative atlas weights")
    parameters = (seamFeatherPixels, insetPixels, extensionPixels, donorBandPixels,
                  colorSmoothPixels, barrierFeatherPixels)
    if any(not np.isfinite(v) or not 0 < v <= 256 for v in parameters):
        raise ValueError("Skin extrapolation distances must be finite in (0,256]")
    alpha = np.minimum(insideDistance(observed)/seamFeatherPixels, 1).astype(np.float32)
    estimated = np.zeros(shape, np.float32)
    repair = np.zeros(shape, np.float32)
    donorMap = np.zeros(shape, bool)
    exteriorMap = np.zeros(shape, bool)
    signedDistance = np.zeros(shape, np.float32)
    extended = np.zeros_like(projected, dtype=np.float32)
    color = projected.astype(np.float32).copy()
    charts, chartCount = label(domain)
    records = []
    for chartId in range(1, chartCount+1):
        chart = charts == chartId
        support = observed & chart
        components, count = label(support)
        sizes = np.bincount(components.ravel())
        eligible = np.flatnonzero(sizes >= 128)
        eligible = eligible[eligible != 0]
        if not len(eligible):
            continue
        # Tiny mapping fragments do not seed a silhouette or supply colors.
        main = np.isin(components, eligible)
        # The SDF describes only the observed exterior silhouette. Intersecting
        # this filled shape with chart would punch intrinsic UV holes back into
        # it and incorrectly classify their rims as exterior repair boundaries.
        # Actual donors/recipients are constrained to chart separately below.
        silhouette = binary_fill_holes(main)
        interior = insideDistance(silhouette)
        exterior = distance_transform_edt(~silhouette)
        sdf = interior-exterior
        signedDistance[chart] = sdf[chart]
        band = main & (interior >= insetPixels) & (interior <= insetPixels+donorBandPixels)
        reference = band & safe & ~protected
        qualityValues = quality[reference & (quality > 0)]
        if len(qualityValues) < 32:
            records.append({"chart": chartId, "status": "noReliableSkinDonors"})
            continue
        # Confidence is a relative fusion score, never interpreted as probability.
        threshold = float(np.quantile(qualityValues, .2))
        donors = reference & (quality >= threshold)
        donors &= _skinColorGate(projected, donors)
        if donors.sum() < 32:
            records.append({"chart": chartId, "status": "insufficientSkinAfterColorGate"})
            continue
        # A separate template-color gate prevents expansion into scalp/hair or
        # dark cavities. Classify low-frequency color: a template freckle/pore
        # must not punch a bright hole into the raised source alpha. Normalize
        # within this chart to avoid mixing padding or another UV island.
        templateWeight = gaussian_filter(chart.astype(np.float32), colorSmoothPixels)
        lowTemplate = gaussian_filter(template*chart[..., None], (colorSmoothPixels, colorSmoothPixels, 0))
        lowTemplate /= np.maximum(templateWeight[..., None], 1e-12)
        targetSkin = _skinColorGate(lowTemplate, donors, luminanceRange=(.48, 1.9)) & chart
        # Internal holes remain excluded even though they were filled for SDF.
        allowed = (targetSkin & ((~silhouette) | main) & ~protected
                   & (sdf >= -extensionPixels))
        regions, regionCount = label(allowed)
        donorCounts = np.bincount(regions[donors], minlength=regionCount+1)
        regionRecords = []
        chartDonors, chartExterior = 0, 0
        for regionId in np.flatnonzero(donorCounts >= 32):
            if regionId == 0:
                continue
            region = regions == regionId
            regionDonors = donors & region
            # Seed only from real donors. An observed fragment across a
            # protected/template barrier must not make that region reachable.
            reachable = binary_propagation(regionDonors, mask=region)
            targetGate = smoothstep(insideDistance(region)/barrierFeatherPixels)
            targetGate *= reachable
            distance, nearest = distance_transform_edt(~regionDonors, return_indices=True)
            reachLimit = extensionPixels+insetPixels+donorBandPixels
            donorGate = 1-smoothstep((distance-.65*reachLimit)/(.35*reachLimit))
            targetGate *= donorGate
            # Both normalized convolutions are evaluated independently for
            # each connected allowed region. Neither color nor normalization
            # weights can come from another region across a protected barrier.
            denominator = gaussian_filter(regionDonors.astype(np.float32), colorSmoothPixels)
            low = gaussian_filter(projected*regionDonors[..., None], (colorSmoothPixels, colorSmoothPixels, 0))
            low /= np.maximum(denominator[..., None], 1e-12)
            skin = low[nearest[0], nearest[1]]
            smoothDomain = reachable.astype(np.float32)
            smoothDenominator = gaussian_filter(smoothDomain, colorSmoothPixels)
            skin = gaussian_filter(skin*smoothDomain[..., None], (colorSmoothPixels, colorSmoothPixels, 0))
            skin /= np.maximum(smoothDenominator[..., None], 1e-12)
            skin = np.clip(skin, 0, 1)
            outerBand = main & (interior < insetPixels) & reachable
            weight = (1-smoothstep(interior/insetPixels))*targetGate*outerBand
            color[outerBand] = (projected*(1-weight[..., None])+skin*weight[..., None])[outerBand]
            # Keep the full feather-width alpha band: stopping at inset
            # would create an alpha discontinuity when seamFeather is larger.
            alphaBand = main & (interior < max(insetPixels, seamFeatherPixels)) & reachable
            alpha[alphaBand] += (1-alpha[alphaBand])*targetGate[alphaBand]
            outside = chart & ~silhouette & (exterior <= extensionPixels) & reachable
            outerAlpha = (1-smoothstep(exterior/extensionPixels))*targetGate
            alpha[outside] = outerAlpha[outside]
            color[outside] = skin[outside]
            repair[outerBand] = weight[outerBand]
            # Disconnected one-pixel mapping fragments must not retain low
            # legacy alpha and leave template-colored dots in the extension.
            repair[outside & observed] = 1
            estimated[outerBand] = alpha[outerBand]*weight[outerBand]
            estimated[outside] = alpha[outside]
            donorMap |= regionDonors
            exteriorMap |= outside & ~observed & (alpha > 0)
            extended[reachable] = skin[reachable]
            donorCount = int(regionDonors.sum())
            exteriorCount = int(np.count_nonzero(outside & ~observed & (alpha > 0)))
            chartDonors += donorCount
            chartExterior += exteriorCount
            regionRecords.append({"region": int(regionId), "donorTexels": donorCount,
                                  "exteriorTexels": exteriorCount})
        records.append({"chart": chartId, "status": "extended" if regionRecords else "noConnectedSkinDonors",
                        "donorTexels": chartDonors, "exteriorTexels": chartExterior,
                        "allowedRegionCount": int(regionCount), "processedRegions": regionRecords,
                        "relativeConfidenceThreshold": threshold})
    composite = color*alpha[..., None]+template*(1-alpha[..., None])
    return {"color": composite.astype(np.float32), "alpha": alpha,
            "maps": {"safeSkinDonor": donorMap, "skinExtrapolated": exteriorMap,
                     "skinRepairWeight": repair, "estimatedSkinBlendAlpha": estimated,
                     "directPhotoBlendAlpha": np.maximum(alpha-estimated, 0),
                     "skinSdfPreview": np.where(domain, np.clip(.5+signedDistance/(2*extensionPixels), 0, 1), 0).astype(np.float32)},
            "debug": {"signedDistancePixels": signedDistance, "extendedSkinLinear": extended,
                      "projectedLinear": projected, "protectedFeatures": protected, "safeSkinInput": safe},
            "diagnostics": {"method": "sdfSkinExterior", "semanticSkinMaskProvided": semanticProvided,
                            "skinDonorPolicy": "semanticSkinAndRelativeColor" if semanticProvided else "relativeColorHeuristic",
                            "insetPixels": insetPixels, "extensionPixels": extensionPixels,
                            "donorBandPixels": donorBandPixels, "colorSmoothPixels": colorSmoothPixels,
                            "barrierFeatherPixels": barrierFeatherPixels,
                            "templateBarrierPolicy": "relativeColorOnLowFrequencyTemplate",
                            "safeDonorTexels": int(donorMap.sum()), "extrapolatedTexels": int(exteriorMap.sum()),
                            "repairedObservedTexels": int(np.count_nonzero(repair)), "charts": records,
                            "coverageExpanded": False, "internalHolesExpanded": False,
                            "estimatedSkinMeaning": "Low-frequency color extrapolation, not new photograph observations"}}
