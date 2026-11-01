"""Experimental frequency-separated, feature-coherent UV observation fusion.

Low frequencies retain semantic fusion's soft transitions. High frequencies
come from one observation at each texel. A connected feature region receives
one source only when that source reliably observes the *entire* region.
This is an engineering experiment, not geometric or optical-flow alignment.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from faceAppearance.semanticFusion import fuseSemanticObservations, protectedFaceLabelIds


def _validateChartLabels(uvChartLabels: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if uvChartLabels is None:
        return np.zeros(shape, np.int32)
    charts = np.asarray(uvChartLabels)
    if (charts.shape != shape or not np.issubdtype(charts.dtype, np.integer)
            or np.issubdtype(charts.dtype, np.bool_) or np.any(charts < -1)):
        raise ValueError("uvChartLabels must be integer HW chart IDs >= -1; -1 is an isolated unknown chart")
    return charts


def decomposeObservationFrequency(observations: np.ndarray, validMask: np.ndarray, *,
                                  lowFrequencySigma: float = 3.0,
                                  uvChartLabels: np.ndarray | None = None) -> dict:
    """Split valid linear RGB into low and signed high frequencies on the CPU.

    Repeated normalized four-neighbor averaging approximates a Gaussian with
    the requested interior sigma. A fractional final step interpolates with
    the previous state, keeping smoothing continuous as sigma approaches zero.
    Every averaging edge requires two valid
    texels in the same known UV chart. Thus no step traverses an invalid texel
    or chart boundary. Unlike a masked Euclidean Gaussian, separate islands
    cannot exchange color through a narrow gap. Diffusion can still travel
    around a hole along a valid path, which is intentional surface smoothing.
    Unknown chart IDs (-1) are isolated, retaining their original low color.
    Supply UV chart IDs when distinct atlas charts touch in the image.
    """
    values = np.asarray(observations, np.float32)
    valid = np.asarray(validMask)
    if (values.ndim != 4 or values.shape[-1] != 3 or min(values.shape[:3], default=0) < 1
            or valid.shape != values.shape[:3] or valid.dtype != np.bool_
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Expected finite NHW3 colors in [0, 1] and boolean NHW validMask")
    if (isinstance(lowFrequencySigma, (bool, np.bool_)) or not np.isfinite(lowFrequencySigma)
            or not 0 <= lowFrequencySigma <= 16):
        raise ValueError("lowFrequencySigma must be finite in [0, 16]")
    charts = _validateChartLabels(uvChartLabels, values.shape[1:3])
    horizontalCharts = (charts[:, :-1] == charts[:, 1:]) & (charts[:, :-1] >= 0)
    verticalCharts = (charts[:-1] == charts[1:]) & (charts[:-1] >= 0)
    effectiveSteps = float(4 * lowFrequencySigma ** 2)
    fullSteps = int(np.floor(effectiveSteps))
    finalStepWeight = effectiveSteps - fullSteps
    steps = fullSteps + int(finalStepWeight > 0)
    lowFrequency = np.zeros_like(values)
    for viewIndex in range(len(values)):
        support = valid[viewIndex]
        horizontal = support[:, :-1] & support[:, 1:] & horizontalCharts
        vertical = support[:-1] & support[1:] & verticalCharts
        denominator = np.full(support.shape, 0.5, np.float32)
        denominator[:, :-1] += 0.125 * horizontal
        denominator[:, 1:] += 0.125 * horizontal
        denominator[:-1] += 0.125 * vertical
        denominator[1:] += 0.125 * vertical
        current = np.where(support[..., None], values[viewIndex], 0)
        for stepIndex in range(steps):
            updated = 0.5 * current
            updated[:, :-1] += 0.125 * current[:, 1:] * horizontal[..., None]
            updated[:, 1:] += 0.125 * current[:, :-1] * horizontal[..., None]
            updated[:-1] += 0.125 * current[1:] * vertical[..., None]
            updated[1:] += 0.125 * current[:-1] * vertical[..., None]
            nextState = updated / denominator[..., None]
            # Leave full steps exactly as before, including floating-point
            # operation order. Only a fractional last step needs interpolation.
            current = (nextState if stepIndex < fullSteps
                       else current + finalStepWeight * (nextState - current))
        lowFrequency[viewIndex] = current
    highFrequency = np.where(valid[..., None], values - lowFrequency, 0)
    return {"lowFrequency": lowFrequency, "highFrequency": highFrequency,
            "diagnostics": {"algorithm": "normalizedFourNeighborDiffusion",
                            "requestedInteriorSigmaUvPixels": float(lowFrequencySigma),
                            "diffusionSteps": steps, "effectiveDiffusionSteps": effectiveSteps,
                            "fullDiffusionSteps": fullSteps, "finalStepWeight": finalStepWeight,
                            "chartLabelsProvided": uvChartLabels is not None,
                            "invalidSupportAndChartEdgesBlocked": True}}


def _dilateOnSurface(region: np.ndarray, support: np.ndarray, charts: np.ndarray, steps: int) -> np.ndarray:
    horizontal = support[:, :-1] & support[:, 1:] & (charts[:, :-1] == charts[:, 1:]) & (charts[:, :-1] >= 0)
    vertical = support[:-1] & support[1:] & (charts[:-1] == charts[1:]) & (charts[:-1] >= 0)
    result = region.copy()
    for _ in range(steps):
        expanded = result.copy()
        expanded[:, :-1] |= result[:, 1:] & horizontal
        expanded[:, 1:] |= result[:, :-1] & horizontal
        expanded[:-1] |= result[1:] & vertical
        expanded[1:] |= result[:-1] & vertical
        result = expanded
    return result


def _selectFeatureOwners(weights: np.ndarray, cosines: np.ndarray, labels: np.ndarray,
                         confidence: np.ndarray, coverage: np.ndarray, charts: np.ndarray,
                         immutableReference: np.ndarray, *, referenceViewIndex: int,
                         protectedLabelIds: tuple[int, ...], featureConfidence: float,
                         featureMinimumCosine: float, featureSeedConfidence: float,
                         featureSeedCosine: float, featureHaloWidth: int) -> dict:
    # Parsing uncertainty is expected on a feature/skin boundary. It should
    # prevent that pixel from *seeding* a feature, but does not imply geometric
    # invisibility of an otherwise valid source. Requiring confident parsing
    # at every halo texel would reject even a fully observed eyebrow.
    reliable = (weights > 0) & (cosines >= featureMinimumCosine)
    feature = reliable & (confidence >= featureConfidence) & np.isin(labels, protectedLabelIds)
    seed = (feature & (confidence >= featureSeedConfidence) & (cosines >= featureSeedCosine))
    featureGroups = labels.copy()
    # Upper/lower lips and the mouth opening are one anatomical region for
    # conflict checks. Skin around a feature remains allowed; another known
    # feature (e.g. eye inside an eyebrow region) is never a whole-region fill.
    featureGroups[np.isin(labels, (11, 12, 13))] = 12
    conflictIds = tuple(set(protectedLabelIds) | ({11} if 12 in protectedLabelIds or 13 in protectedLabelIds else set()))
    knownFeature = (confidence >= featureConfidence) & np.isin(labels, conflictIds)
    # The union includes displaced features from auxiliary views. Ownership
    # over this whole union can suppress their duplicate contour instead of
    # preserving one shifted fragment from each photograph.
    union = _dilateOnSurface(feature.any(axis=0), coverage, charts, featureHaloWidth)
    owner = np.full(coverage.shape, -1, np.int32)
    regionIndex = np.full(coverage.shape, -1, np.int32)
    records = []
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
    for chartId in np.unique(charts[union & (charts >= 0)]):
        components, componentCount = label(union & (charts == chartId), structure)
        for componentId in range(1, componentCount + 1):
            region = components == componentId
            number = len(records)
            regionIndex[region] = number
            eligible = np.all(reliable[:, region], axis=1) & np.any(seed[:, region], axis=1)
            regionGroups = np.unique(featureGroups[:, region][feature[:, region]])
            incompatible = (knownFeature[:, region]
                            & ~np.isin(featureGroups[:, region], regionGroups))
            semanticConflict = np.any(incompatible, axis=1)
            # Do not treat a accidentally connected eye/nose/brow union as a
            # single feature. It needs correspondence correction first.
            if len(regionGroups) != 1:
                semanticConflict[:] = True
            eligible &= ~semanticConflict
            if immutableReference[region].any():
                eligible[np.arange(len(weights)) != referenceViewIndex] = False
            candidateIndices = np.flatnonzero(eligible)
            selected = -1
            if eligible[referenceViewIndex]:
                selected = referenceViewIndex
            elif len(candidateIndices):
                scores = np.mean(weights[candidateIndices][:, region].astype(np.float64)
                                 * cosines[candidateIndices][:, region]
                                 * confidence[candidateIndices][:, region], axis=1)
                selected = int(candidateIndices[scores.argmax()])
            if selected >= 0:
                owner[region] = selected
            records.append({"regionIndex": number, "chartId": int(chartId),
                            "texels": int(region.sum()), "sourceIndex": selected,
                            "eligibleSources": candidateIndices.tolist(),
                            "featureGroups": regionGroups.tolist(),
                            "semanticConflictSources": np.flatnonzero(semanticConflict).tolist(),
                            "hasImmutableReference": bool(immutableReference[region].any()),
                            "reason": "wholeRegionReliableSource" if selected >= 0
                            else "noSingleReliableSourceForWholeRegion"})
    return {"sourceIndex": owner, "regionIndex": regionIndex, "unionMask": union,
            "diagnostics": {"regions": records, "regionCount": len(records),
                            "ownedRegionCount": sum(item["sourceIndex"] >= 0 for item in records),
                            "unownedRegionCount": sum(item["sourceIndex"] < 0 for item in records),
                            "ownedTexels": int((owner >= 0).sum()),
                            "unknownChartFeatureTexels": int((union & (charts < 0)).sum()),
                            "haloWidthUvPixels": int(featureHaloWidth),
                            "wholeRegionGate": "positiveObservationWeightAndMinimumViewingCosine",
                            "semanticConfidenceGate": "featureUnionAndStrongSourceSeedsOnly"}}


def _edgeJump(colors: np.ndarray, sources: np.ndarray, coverage: np.ndarray, charts: np.ndarray) -> dict:
    jumps = []
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        valid = (coverage[first] & coverage[second] & (sources[first] != sources[second])
                 & (charts[first] == charts[second]) & (charts[first] >= 0))
        jumps.append(np.mean(np.abs(colors[first] - colors[second]), axis=-1)[valid])
    values = np.concatenate(jumps)
    return {"sourceSwitchEdges": len(values), "meanLinearRgbJump": float(values.mean()) if len(values) else 0.0,
            "p95LinearRgbJump": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            "maximumLinearRgbJump": float(values.max()) if len(values) else 0.0}


def fuseFrequencyObservations(observations: np.ndarray, weights: np.ndarray,
                              viewingCosines: np.ndarray, semanticLabels: np.ndarray,
                              semanticConfidence: np.ndarray, *, referenceViewIndex: int = 0,
                              lowFrequencySigma: float = 3.0, uvChartLabels: np.ndarray | None = None,
                              protectFeatureRegions: bool = True,
                              featureConfidence: float = 0.65, featureMinimumCosine: float = 0.35,
                              featureSeedConfidence: float = 0.8, featureSeedCosine: float = 0.65,
                              featureHaloWidth: int = 3, referenceBlendWidth: float = 12.0,
                              referenceMinCosine: float = 0.35, referenceFullCosine: float = 0.65,
                              matchAuxiliaryColors: bool = True,
                              protectedLabelIds: tuple[int, ...] = protectedFaceLabelIds,
                              protectionConfidence: float = 0.65, protectionMinCosine: float = 0.35,
                              protectionSeedConfidence: float = 0.8, protectionSeedCosine: float = 0.65,
                              protectionHaloWidth: int = 2, sourceSmoothness: float = 0.3,
                              sourceColorWeight: float = 1.0, sourceIterations: int = 6) -> dict:
    """Fuse linear NHW3 observations without averaging high-frequency detail.

    The semantic baseline remains unchanged. Its RGB gains and continuous
    auxiliary source map are reused. Low frequencies mix with its soft alpha;
    the corresponding high-frequency source uses alpha >= 0.5, or whole-region
    feature ownership where safe. Whole-region visibility requires positive
    qualified observation weights and minimum viewing cosine everywhere;
    parsing confidence gates feature union/seed detection, not halo visibility.
    Whole reference regions and its original
    trusted core are bit-exact copies. A whole auxiliary feature takes its
    selected source's complete gain-matched color to avoid low-frequency ghosts.

    ``sourceIndex=-2`` means low-frequency mixing; ``highFrequencySourceIndex``
    always identifies one observed source, or -1 without support. Signed
    residual recombination can exceed [0, 1]; return and diagnose unclipped RGB
    before clipping the exported albedo. Source-edge jumps are descriptive,
    not a quality score: genuine image edges may also occur at source seams.
    """
    if (not isinstance(protectFeatureRegions, (bool, np.bool_))
            or isinstance(featureHaloWidth, (bool, np.bool_))
            or not isinstance(featureHaloWidth, (int, np.integer)) or featureHaloWidth < 0):
        raise ValueError("Expected boolean protectFeatureRegions and nonnegative integer featureHaloWidth")
    thresholds = [featureConfidence, featureMinimumCosine, featureSeedConfidence, featureSeedCosine]
    if (not np.isfinite(thresholds).all() or not 0 < featureConfidence <= featureSeedConfidence <= 1
            or not 0 <= featureMinimumCosine <= featureSeedCosine <= 1):
        raise ValueError("Expected valid feature confidence and cosine thresholds, seed >= region thresholds")
    baseline = fuseSemanticObservations(
        observations, weights, viewingCosines, semanticLabels, semanticConfidence,
        referenceViewIndex=referenceViewIndex, referenceBlendWidth=referenceBlendWidth,
        referenceMinCosine=referenceMinCosine, referenceFullCosine=referenceFullCosine,
        matchAuxiliaryColors=matchAuxiliaryColors, protectedLabelIds=protectedLabelIds,
        protectionConfidence=protectionConfidence, protectionMinCosine=protectionMinCosine,
        protectionSeedConfidence=protectionSeedConfidence, protectionSeedCosine=protectionSeedCosine,
        protectionHaloWidth=protectionHaloWidth, sourceSmoothness=sourceSmoothness,
        sourceColorWeight=sourceColorWeight, sourceIterations=sourceIterations,
        narrowBoundaryBlend=False)
    values = np.asarray(observations, np.float32)
    weightArray = np.asarray(weights, np.float32)
    cosines = np.asarray(viewingCosines, np.float32)
    labels = np.asarray(semanticLabels)
    semanticWeights = np.asarray(semanticConfidence, np.float32)
    charts = _validateChartLabels(uvChartLabels, values.shape[1:3])
    coverage = baseline["coverageMask"]
    gains = np.asarray([item["gainsRgb"] for item in baseline["diagnostics"]["colorMatching"]["views"]], np.float32)
    matched = np.clip(values * gains[:, None, None], 0, 1)
    decomposition = decomposeObservationFrequency(matched, weightArray > 0,
                                                 lowFrequencySigma=lowFrequencySigma, uvChartLabels=uvChartLabels)
    low, high = decomposition["lowFrequency"], decomposition["highFrequency"]
    alpha = baseline["referenceAlpha"].copy()
    auxiliary = baseline["auxiliarySourceIndex"].copy()
    immutableReference = alpha == 1
    owners = _selectFeatureOwners(
        weightArray, cosines, labels, semanticWeights, coverage, charts, immutableReference,
        referenceViewIndex=referenceViewIndex, protectedLabelIds=protectedLabelIds,
        featureConfidence=featureConfidence, featureMinimumCosine=featureMinimumCosine,
        featureSeedConfidence=featureSeedConfidence, featureSeedCosine=featureSeedCosine,
        featureHaloWidth=featureHaloWidth) if protectFeatureRegions else None
    featureSource = owners["sourceIndex"] if owners is not None else np.full(coverage.shape, -1, np.int32)
    featureOwned = featureSource >= 0
    featureReference = featureSource == referenceViewIndex
    featureAuxiliary = featureOwned & ~featureReference
    alpha[featureReference] = 1
    alpha[featureAuxiliary] = 0
    auxiliary[featureAuxiliary] = featureSource[featureAuxiliary]
    yy, xx = np.indices(coverage.shape)
    safeAuxiliary = np.maximum(auxiliary, 0)
    highSource = np.where(alpha >= 0.5, referenceViewIndex, auxiliary).astype(np.int32)
    highSource[~coverage] = -1
    safeHighSource = np.maximum(highSource, 0)
    lowAlbedo = alpha[..., None] * low[referenceViewIndex] + (1 - alpha[..., None]) * low[safeAuxiliary, yy, xx]
    highAlbedo = high[safeHighSource, yy, xx].copy()
    unclipped = lowAlbedo + highAlbedo
    # Reconstruct full-source colors by exact copy, avoiding subtract/add
    # roundoff and guaranteeing the public reference preservation contract.
    fullSource = coverage & ((alpha == 0) | (alpha == 1))
    unclipped[fullSource] = matched[safeHighSource, yy, xx][fullSource]
    for array in (lowAlbedo, highAlbedo, unclipped):
        array[~coverage] = 0
    clipping = coverage & np.any((unclipped < 0) | (unclipped > 1), axis=-1)
    linearAlbedo = np.clip(unclipped, 0, 1)
    sourceIndex = highSource.copy()
    mixed = coverage & (alpha > 0) & (alpha < 1)
    sourceIndex[mixed] = -2
    selectedWeights = weightArray[safeHighSource, yy, xx]
    outputConfidence = (alpha * np.clip(weightArray[referenceViewIndex], 0, 1)
                        + (1 - alpha) * np.clip(weightArray[safeAuxiliary, yy, xx], 0, 1))
    outputConfidence[fullSource] = np.clip(selectedWeights[fullSource], 0, 1)
    outputConfidence[~coverage] = 0
    core = baseline["referenceCoreMask"] | baseline["protectionMask"]
    diagnostics = {"fusion": "semanticFrequencySeparatedWholeFeatureSources",
                   "baseline": baseline["diagnostics"], "decomposition": decomposition["diagnostics"],
                   "featureOwnership": owners["diagnostics"] if owners is not None else {"enabled": False},
                   "highFrequencyMixedTexels": 0, "lowFrequencyMixedTexels": int(mixed.sum()),
                   "referenceExactTexels": int((coverage & (alpha == 1)).sum()),
                   "referenceCoreAndProtectionMaximumError": float(np.max(np.abs(linearAlbedo[core] - values[referenceViewIndex][core]))) if core.any() else 0.0,
                   "recombinationClippedTexels": int(clipping.sum()),
                   "recombinationClippedFraction": float(clipping.sum() / max(int(coverage.sum()), 1)),
                   "unclippedMinimum": float(unclipped[coverage].min()) if coverage.any() else 0.0,
                   "unclippedMaximum": float(unclipped[coverage].max()) if coverage.any() else 0.0,
                   "outputEdgeJump": _edgeJump(linearAlbedo, highSource, coverage, charts),
                   "baselineAtSameEdges": _edgeJump(baseline["linearAlbedo"], highSource, coverage, charts),
                   "limitations": ["This changes color ownership, not geometric correspondence.",
                                   "A whole feature is selected only with complete reliable single-view support.",
                                   "Residual low-frequency duplicates and high-frequency source seams can remain.",
                                   "UV chart IDs are needed to distinguish touching atlas charts.",
                                   "Recombined colors may clip; unclipped outputs are retained for evaluation."]}
    return {"linearAlbedo": linearAlbedo, "unclippedLinearAlbedo": unclipped,
            "lowFrequencyAlbedo": lowAlbedo, "highFrequencyResidual": highAlbedo,
            "coverageMask": coverage.copy(), "confidence": outputConfidence.astype(np.float32),
            "sourceIndex": sourceIndex, "highFrequencySourceIndex": highSource,
            "auxiliarySourceIndex": auxiliary, "referenceAlpha": alpha,
            "baselineReferenceAlpha": baseline["referenceAlpha"].copy(),
            "referenceCoreMask": baseline["referenceCoreMask"].copy(),
            "protectionMask": baseline["protectionMask"].copy(),
            "featureOwnershipMask": featureOwned, "featureSourceIndex": featureSource,
            "featureRegionIndex": owners["regionIndex"] if owners is not None else np.full(coverage.shape, -1, np.int32),
            "clippingMask": clipping, "diagnostics": diagnostics}
