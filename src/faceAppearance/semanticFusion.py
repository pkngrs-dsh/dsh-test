"""Semantic feature protection and spatially coherent observation selection.

Inputs use the project's 19-class face-parsing schema (CelebAMask-HQ order).
This module never predicts new colors, expands visibility, or moves geometry.
Its CPU-only optimization is a deterministic, four-neighbor Potts ICM; it is
not a globally optimal graph cut and does not align inconsistent photographs.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from faceAppearance.referenceFusion import fuseReferenceObservations


protectedFaceLabelIds = (2, 3, 4, 5, 10, 12, 13)


def _validateSemantics(semanticLabels: np.ndarray, semanticConfidence: np.ndarray,
                       shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(semanticLabels)
    confidence = np.asarray(semanticConfidence, np.float32)
    if (labels.shape != shape or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_) or np.any(labels < 0) or np.any(labels > 18)
            or confidence.shape != shape or not np.isfinite(confidence).all()
            or np.any(confidence < 0) or np.any(confidence > 1)):
        raise ValueError("Expected integer NHW semantic labels in [0, 18] and finite NHW confidence in [0, 1]")
    return labels.astype(np.int16, copy=False), confidence


def _validateReference(referenceViewIndex: int, viewCount: int) -> None:
    if (isinstance(referenceViewIndex, (bool, np.bool_))
            or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < viewCount):
        raise ValueError("referenceViewIndex must identify an available view")


def _validateInteger(value: int, name: str) -> None:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer")


def buildSemanticProtection(semanticLabels: np.ndarray, semanticConfidence: np.ndarray,
                            weights: np.ndarray, viewingCosines: np.ndarray, *,
                            referenceViewIndex: int = 0,
                            protectedLabelIds: tuple[int, ...] = protectedFaceLabelIds,
                            confidenceThreshold: float = 0.65, minimumCosine: float = 0.35,
                            seedConfidenceThreshold: float = 0.8, seedMinimumCosine: float = 0.65,
                            haloWidth: int = 2) -> dict:
    """Protect reliable reference feature observations without crossing semantics.

    Every reference texel in a protected class passing the seed thresholds is
    a seed. A bounded four-connected halo can include lower-confidence edge
    texels, but only inside the *same* label and qualified reference support.
    A semantic prediction over a mesh hole therefore never creates support.
    Unknown predictions should be represented by confidence zero.
    """
    weightArray = np.asarray(weights, np.float32)
    cosines = np.asarray(viewingCosines, np.float32)
    if (weightArray.ndim != 3 or min(weightArray.shape, default=0) < 1
            or cosines.shape != weightArray.shape or not np.isfinite(weightArray).all()
            or np.any(weightArray < 0) or not np.isfinite(cosines).all()
            or np.any(cosines < 0) or np.any(cosines > 1)):
        raise ValueError("Expected finite nonnegative NHW weights and NHW cosines in [0, 1]")
    labels, confidence = _validateSemantics(semanticLabels, semanticConfidence, weightArray.shape)
    _validateReference(referenceViewIndex, len(weightArray))
    _validateInteger(haloWidth, "haloWidth")
    thresholds = np.asarray([confidenceThreshold, minimumCosine, seedConfidenceThreshold, seedMinimumCosine])
    if (not np.isfinite(thresholds).all() or not 0 < confidenceThreshold <= seedConfidenceThreshold <= 1
            or not 0 <= minimumCosine <= seedMinimumCosine <= 1):
        raise ValueError("Expected 0 < confidence <= seed confidence <= 1 and 0 <= cosine <= seed cosine <= 1")
    if (not isinstance(protectedLabelIds, (tuple, list)) or not protectedLabelIds
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                   or not 1 <= value <= 18 for value in protectedLabelIds)
            or len(set(protectedLabelIds)) != len(protectedLabelIds)):
        raise ValueError("protectedLabelIds must contain distinct non-background label IDs in [1, 18]")

    referenceLabels = labels[referenceViewIndex]
    referenceConfidence = confidence[referenceViewIndex]
    referenceCosines = cosines[referenceViewIndex]
    referenceValid = weightArray[referenceViewIndex] > 0
    allowed = (referenceValid & (referenceConfidence >= confidenceThreshold)
               & (referenceCosines >= minimumCosine))
    seeds = (allowed & np.isin(referenceLabels, protectedLabelIds)
             & (referenceConfidence >= seedConfidenceThreshold)
             & (referenceCosines >= seedMinimumCosine))
    protectionMask = seeds.copy()
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
    for labelId in protectedLabelIds:
        labelSeeds = seeds & (referenceLabels == labelId)
        if not labelSeeds.any() or haloWidth == 0:
            continue
        # Mask each step, rather than dilating first and intersecting afterward:
        # a halo must not jump over an invalid texel or another semantic class.
        region = labelSeeds
        sameClassAllowed = allowed & (referenceLabels == labelId)
        for _ in range(haloWidth):
            region = binary_dilation(region, structure=structure) & sameClassAllowed
        protectionMask |= region
    return {"protectionMask": protectionMask, "seedMask": seeds,
            "diagnostics": {"protectedLabelIds": [int(value) for value in protectedLabelIds],
                            "referenceViewIndex": int(referenceViewIndex),
                            "seedConfidenceThreshold": float(seedConfidenceThreshold),
                            "seedMinimumCosine": float(seedMinimumCosine),
                            "confidenceThreshold": float(confidenceThreshold),
                            "minimumCosine": float(minimumCosine), "haloWidthUvPixels": int(haloWidth),
                            "seedTexels": int(seeds.sum()), "protectedTexels": int(protectionMask.sum()),
                            "haloTexels": int((protectionMask & ~seeds).sum()),
                            "protectedTexelsByLabel": {str(value): int((protectionMask & (referenceLabels == value)).sum())
                                                       for value in protectedLabelIds}}}


def _sourceContinuity(sourceIndex: np.ndarray) -> dict:
    valid = sourceIndex >= 0
    horizontalValid = valid[:, :-1] & valid[:, 1:]
    verticalValid = valid[:-1] & valid[1:]
    edgeCount = int(horizontalValid.sum() + verticalValid.sum())
    switchCount = int(((sourceIndex[:, :-1] != sourceIndex[:, 1:]) & horizontalValid).sum()
                      + ((sourceIndex[:-1] != sourceIndex[1:]) & verticalValid).sum())
    return {"adjacentValidEdges": edgeCount, "sourceSwitchEdges": switchCount,
            "sourceSwitchFraction": float(switchCount / edgeCount) if edgeCount else 0.0}


def selectContinuousSources(observations: np.ndarray, weights: np.ndarray,
                            semanticLabels: np.ndarray, semanticConfidence: np.ndarray, *,
                            referenceViewIndex: int = 0, excludeReference: bool = True,
                            smoothness: float = 0.3, colorWeight: float = 1.0,
                            iterations: int = 6, confidenceThreshold: float = 0.65) -> dict:
    """Select actual observed colors with a spatially regularized source map.

    Unary cost combines relative observation reliability with color distance
    to the reference where it exists, or the weighted auxiliary color mean.
    Reliable semantic agreement is preferred whenever an agreeing candidate
    exists. With no agreeing candidate, retain valid coverage and report the
    fallback; never fabricate a hole from a segmentation disagreement.

    Red-black ICM minimizes a fixed four-neighbor energy with edge-aware Potts
    penalties. Each accepted step is energy nonincreasing, but local optima
    remain possible. Semantic/color edges weaken spatial regularization.
    """
    values = np.asarray(observations, np.float32)
    weightArray = np.asarray(weights, np.float32)
    if (values.ndim != 4 or values.shape[-1] != 3 or min(values.shape[:3], default=0) < 1
            or weightArray.shape != values.shape[:3] or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1) or not np.isfinite(weightArray).all()
            or np.any(weightArray < 0)):
        raise ValueError("Expected finite NHW3 linear colors in [0, 1] and nonnegative NHW weights")
    labels, confidence = _validateSemantics(semanticLabels, semanticConfidence, weightArray.shape)
    _validateReference(referenceViewIndex, len(values))
    _validateInteger(iterations, "iterations")
    if (not isinstance(excludeReference, (bool, np.bool_))
            or not np.isfinite([smoothness, colorWeight, confidenceThreshold]).all()
            or smoothness < 0 or colorWeight < 0 or not 0 < confidenceThreshold <= 1):
        raise ValueError("Expected boolean excludeReference, nonnegative finite costs and confidence in (0, 1]")
    candidateWeights = weightArray.copy()
    if excludeReference:
        candidateWeights[referenceViewIndex] = 0
    candidateValid = candidateWeights > 0
    coverage = candidateValid.any(axis=0)
    yy, xx = np.indices(coverage.shape)

    semanticReliable = (confidence >= confidenceThreshold) & (weightArray > 0)
    semanticScores = np.where(semanticReliable, weightArray * confidence, 0)
    semanticSource = semanticScores.argmax(axis=0)
    targetLabels = labels[semanticSource, yy, xx].copy()
    targetKnown = semanticScores.max(axis=0) > 0
    referenceKnown = semanticReliable[referenceViewIndex]
    targetLabels[referenceKnown] = labels[referenceViewIndex][referenceKnown]
    targetKnown |= referenceKnown
    compatibleCandidates = (candidateValid & (labels == targetLabels[None])
                            & (confidence >= confidenceThreshold) & targetKnown[None])
    compatibleExists = compatibleCandidates.any(axis=0)
    eligible = candidateValid & (~compatibleExists[None] | compatibleCandidates)

    # Normalize in float64 so unusually large yet finite source weights cannot
    # overflow when summing, then return the usual float32 calculation arrays.
    maximumWeight = candidateWeights.max(axis=0).astype(np.float64)
    relativeWeights = (candidateWeights / np.maximum(maximumWeight, np.finfo(np.float32).tiny)).astype(np.float32)
    weightSum = relativeWeights.sum(axis=0)
    targetColor = (values * relativeWeights[..., None]).sum(axis=0) / np.maximum(weightSum[..., None], 1e-12)
    referenceValid = weightArray[referenceViewIndex] > 0
    targetColor[referenceValid] = values[referenceViewIndex][referenceValid]
    unary = (-np.log(np.maximum(relativeWeights, 1e-12))
             + colorWeight * np.mean(np.abs(values - targetColor[None]), axis=-1) / 0.1)
    unary[~eligible] = np.inf
    sourceIndex = unary.argmin(axis=0).astype(np.int32)
    sourceIndex[~coverage] = -1
    initialSource = sourceIndex.copy()

    def edgeCosts(first: tuple, second: tuple) -> np.ndarray:
        colorGradient = np.mean(np.abs(targetColor[first] - targetColor[second]), axis=-1)
        cost = smoothness * (0.25 + 0.75 * np.exp(-colorGradient / 0.1))
        differentSemantics = targetKnown[first] & targetKnown[second] & (targetLabels[first] != targetLabels[second])
        cost[differentSemantics] *= 0.2
        cost[~(coverage[first] & coverage[second])] = 0
        return cost

    horizontalCost = edgeCosts((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    verticalCost = edgeCosts((slice(None, -1), slice(None)), (slice(1, None), slice(None)))

    def totalEnergy(source: np.ndarray) -> float:
        dataEnergy = unary[np.maximum(source, 0), yy, xx][coverage].sum(dtype=np.float64)
        return float(dataEnergy
                     + (horizontalCost * (source[:, :-1] != source[:, 1:])).sum(dtype=np.float64)
                     + (verticalCost * (source[:-1] != source[1:])).sum(dtype=np.float64))

    energyHistory = [totalEnergy(sourceIndex)]
    parity = (yy + xx) % 2
    completedIterations = 0
    labelIndices = np.arange(len(values))[:, None, None]
    if smoothness > 0:
        for _ in range(iterations):
            changedTexels = 0
            for color in (0, 1):
                costs = unary.copy()
                costs[:, :, 1:] += horizontalCost[None] * (labelIndices != sourceIndex[None, :, :-1])
                costs[:, :, :-1] += horizontalCost[None] * (labelIndices != sourceIndex[None, :, 1:])
                costs[:, 1:] += verticalCost[None] * (labelIndices != sourceIndex[None, :-1])
                costs[:, :-1] += verticalCost[None] * (labelIndices != sourceIndex[None, 1:])
                bestSource = costs.argmin(axis=0)
                currentCost = costs[np.maximum(sourceIndex, 0), yy, xx]
                bestCost = costs[bestSource, yy, xx]
                update = coverage & (parity == color) & (bestCost < currentCost - 1e-7)
                changedTexels += int(update.sum())
                sourceIndex[update] = bestSource[update]
            completedIterations += 1
            energyHistory.append(totalEnergy(sourceIndex))
            if changedTexels == 0:
                break
    return {"sourceIndex": sourceIndex, "coverageMask": coverage,
            "targetSemanticLabels": targetLabels, "targetSemanticKnown": targetKnown,
            "diagnostics": {"algorithm": "fourNeighborRedBlackIcm", "smoothness": float(smoothness),
                            "colorWeight": float(colorWeight), "colorDistanceScale": 0.1,
                            "iterationsRequested": int(iterations), "iterationsCompleted": completedIterations,
                            "energyHistory": energyHistory, "changedTexels": int((sourceIndex != initialSource).sum()),
                            "semanticAgreementFallbackTexels": int((coverage & targetKnown & ~compatibleExists).sum()),
                            "initialContinuity": _sourceContinuity(initialSource),
                            "finalContinuity": _sourceContinuity(sourceIndex)}}


def fuseSemanticObservations(observations: np.ndarray, weights: np.ndarray,
                             viewingCosines: np.ndarray, semanticLabels: np.ndarray | None = None,
                             semanticConfidence: np.ndarray | None = None, *, enabled: bool = True,
                             referenceViewIndex: int = 0, referenceBlendWidth: float = 12.0,
                             referenceMinCosine: float = 0.35, referenceFullCosine: float = 0.65,
                             matchAuxiliaryColors: bool = True,
                             protectedLabelIds: tuple[int, ...] = protectedFaceLabelIds,
                             protectionConfidence: float = 0.65, protectionMinCosine: float = 0.35,
                             protectionSeedConfidence: float = 0.8, protectionSeedCosine: float = 0.65,
                             protectionHaloWidth: int = 2, narrowBoundaryBlend: bool = False,
                             sourceSmoothness: float = 0.3,
                             sourceColorWeight: float = 1.0, sourceIterations: int = 6) -> dict:
    """Extend reference fusion with exact feature ownership and coherent fill.

    ``enabled=False`` delegates directly to the original reference fusion,
    including its diagnostics. Enabled fusion preserves its angle/boundary
    alpha outside protected features, but never mixes confident disagreeing
    semantics: a reliable reference wins there, otherwise the auxiliary wins.
    Low-confidence or unknown semantics never delete support. By default,
    the original soft angle ramp and support-boundary feather are preserved
    outside semantic protection/disagreement overrides. Experimental
    ``narrowBoundaryBlend=True`` restricts intermediate alpha to within
    ``referenceBlendWidth`` UV pixels of an ownership boundary; this can
    introduce visible seams when a broad angle ramp is snapped to one source.
    The original fully trusted reference core remains exact. Auxiliary views
    are never mixed with one another.
    """
    if not isinstance(enabled, (bool, np.bool_)) or not isinstance(narrowBoundaryBlend, (bool, np.bool_)):
        raise ValueError("enabled and narrowBoundaryBlend must be boolean")
    baseline = fuseReferenceObservations(observations, weights, viewingCosines,
                                         referenceViewIndex=referenceViewIndex,
                                         referenceBlendWidth=referenceBlendWidth,
                                         referenceMinCosine=referenceMinCosine,
                                         referenceFullCosine=referenceFullCosine,
                                         matchAuxiliaryColors=matchAuxiliaryColors)
    if not enabled:
        return baseline
    if semanticLabels is None or semanticConfidence is None:
        raise ValueError("Enabled semantic fusion requires semanticLabels and semanticConfidence")
    values = np.asarray(observations, np.float32)
    weightArray = np.asarray(weights, np.float32)
    cosines = np.asarray(viewingCosines, np.float32)
    labels, semanticWeights = _validateSemantics(semanticLabels, semanticConfidence, weightArray.shape)
    protection = buildSemanticProtection(labels, semanticWeights, weightArray, cosines,
                                          referenceViewIndex=referenceViewIndex,
                                          protectedLabelIds=protectedLabelIds,
                                          confidenceThreshold=protectionConfidence,
                                          minimumCosine=protectionMinCosine,
                                          seedConfidenceThreshold=protectionSeedConfidence,
                                          seedMinimumCosine=protectionSeedCosine,
                                          haloWidth=protectionHaloWidth)
    gains = np.array([entry["gainsRgb"] for entry in baseline["diagnostics"]["colorMatching"]["views"]], np.float32)
    matchedValues = np.clip(values * gains[:, None, None], 0, 1)
    selectionWeights = weightArray.copy()
    selectionWeights[referenceViewIndex, cosines[referenceViewIndex] < protectionMinCosine] = 0
    sourceSelection = selectContinuousSources(matchedValues, selectionWeights, labels, semanticWeights,
                                              referenceViewIndex=referenceViewIndex,
                                              smoothness=sourceSmoothness, colorWeight=sourceColorWeight,
                                              iterations=sourceIterations, confidenceThreshold=protectionConfidence)
    auxiliarySource = sourceSelection["sourceIndex"]
    auxiliaryValid = auxiliarySource >= 0
    yy, xx = np.indices(auxiliarySource.shape)
    safeSource = np.maximum(auxiliarySource, 0)
    selectedAuxiliary = matchedValues[safeSource, yy, xx].copy()
    selectedAuxiliary[~auxiliaryValid] = 0
    auxiliaryWeight = weightArray[safeSource, yy, xx].copy()
    auxiliaryWeight[~auxiliaryValid] = 0
    referenceValid = weightArray[referenceViewIndex] > 0
    referenceAlpha = baseline["referenceAlpha"].copy()
    protectionMask = protection["protectionMask"]
    referenceAlpha[protectionMask] = 1

    referenceKnown = semanticWeights[referenceViewIndex] >= protectionConfidence
    auxiliaryKnown = semanticWeights[safeSource, yy, xx] >= protectionConfidence
    semanticDisagreement = (referenceValid & auxiliaryValid & referenceKnown & auxiliaryKnown
                            & (labels[referenceViewIndex] != labels[safeSource, yy, xx]))
    referenceReliable = referenceValid & (cosines[referenceViewIndex] >= protectionMinCosine)
    # Even one sharp semantic edge must not become a cross-class RGB mixture.
    referenceAlpha[semanticDisagreement & referenceReliable] = 1
    referenceAlpha[semanticDisagreement & ~referenceReliable] = 0
    referenceAlpha[referenceValid & ~auxiliaryValid] = 1
    referenceAlpha[~referenceValid] = 0
    coverage = referenceValid | auxiliaryValid
    blendAllowedMask = coverage.copy()
    snappedTexels = 0
    if narrowBoundaryBlend:
        ownedByReference = referenceAlpha >= 0.5
        sourceBoundary = np.zeros_like(coverage)
        horizontalSwitch = (coverage[:, :-1] & coverage[:, 1:]
                            & (ownedByReference[:, :-1] != ownedByReference[:, 1:]))
        verticalSwitch = (coverage[:-1] & coverage[1:]
                          & (ownedByReference[:-1] != ownedByReference[1:]))
        sourceBoundary[:, :-1] |= horizontalSwitch
        sourceBoundary[:, 1:] |= horizontalSwitch
        sourceBoundary[:-1] |= verticalSwitch
        sourceBoundary[1:] |= verticalSwitch
        if referenceBlendWidth > 0 and sourceBoundary.any():
            blendAllowedMask = coverage & (distance_transform_edt(~sourceBoundary) < referenceBlendWidth)
        else:
            blendAllowedMask = np.zeros_like(coverage)
        snapMask = coverage & ~blendAllowedMask & (referenceAlpha > 0) & (referenceAlpha < 1)
        snappedTexels = int(snapMask.sum())
        referenceAlpha[snapMask] = ownedByReference[snapMask]
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
    unclippedSelected = values[safeSource, yy, xx] * gains[safeSource]
    clippedAuxiliary = auxiliaryValid & (referenceAlpha < 1) & np.any(unclippedSelected > 1, axis=-1)
    visibleAuxiliarySources = auxiliarySource.copy()
    visibleAuxiliarySources[referenceAlpha == 1] = -1
    diagnostics = dict(baseline["diagnostics"])
    diagnostics.update({"fusion": "semanticReferenceContinuousAuxiliary", "semanticProtection": protection["diagnostics"],
                        "sourceSelection": sourceSelection["diagnostics"],
                        "referenceOwnedTexels": int((referenceAlpha == 1).sum()),
                        "blendedTexels": int(mixed.sum()),
                        "auxiliaryOnlyTexels": int((coverage & (referenceAlpha == 0)).sum()),
                        "auxiliaryGainClippedTexels": int(clippedAuxiliary.sum()),
                        "visibleAuxiliaryContinuity": _sourceContinuity(visibleAuxiliarySources),
                        "boundaryBlending": {"enabled": bool(narrowBoundaryBlend),
                                             "widthUvPixels": float(referenceBlendWidth),
                                             "snappedOutsideBandTexels": snappedTexels},
                        "semanticDisagreementTexels": int(semanticDisagreement.sum()),
                        "semanticReferenceChosenTexels": int((semanticDisagreement & referenceReliable).sum()),
                        "semanticAuxiliaryChosenTexels": int((semanticDisagreement & ~referenceReliable).sum()),
                        "protectedReferenceMaximumError": float(np.max(np.abs(linearAlbedo[protectionMask]
                                                                        - values[referenceViewIndex][protectionMask])))
                        if protectionMask.any() else 0.0,
                        "limitations": ["Semantic protection cannot repair geometry or correspondence errors.",
                                        "ICM is a local energy minimizer; source seams can remain.",
                                        "Unknown semantic predictions do not constrain source selection.",
                                        "Global RGB gain matching does not remove local illumination."]})
    return {"linearAlbedo": linearAlbedo, "coverageMask": coverage, "confidence": confidence.astype(np.float32),
            "sourceIndex": sourceIndex, "referenceAlpha": referenceAlpha,
            "referenceCoreMask": baseline["referenceCoreMask"], "auxiliarySourceIndex": auxiliarySource,
            "protectionMask": protectionMask, "protectionSeedMask": protection["seedMask"],
            "semanticCompatibleMask": coverage & ~semanticDisagreement,
            "blendAllowedMask": blendAllowedMask, "diagnostics": diagnostics}
