"""Bounded auxiliary-view alignment after projection into a shared UV atlas.

The reference observation and the atlas/mesh never move. ``samplingFlow[y,x]``
is a *backward sampling map*: output[y,x] reads source[y+dy,x+dx]. It is not a
geometric displacement, and fitting error here is not independent validation.
Unknown semantics, holes and rejected estimates retain their original samples.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


class UvFlowAdapter(Protocol):
    """Replaceable estimator; images are float32 HW structure values in [0, 1]."""

    def estimateFlow(self, referenceStructure: np.ndarray,
                     sourceStructure: np.ndarray) -> np.ndarray:
        """Return HW2 displacements locating each reference point in source."""
        ...


class OpenCvUvFlowAdapter:
    """CPU DIS optical flow isolated from support and semantic policy.

    OpenCV is an existing project dependency. A fresh estimator per call avoids
    retained temporal state; this adapter never changes OpenCV's global threads.
    https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html
    """

    def estimateFlow(self, referenceStructure: np.ndarray,
                     sourceStructure: np.ndarray) -> np.ndarray:
        first = np.asarray(referenceStructure, np.float32)
        second = np.asarray(sourceStructure, np.float32)
        if (first.ndim != 2 or second.shape != first.shape or min(first.shape) < 16
                or not np.isfinite(first).all() or not np.isfinite(second).all()
                or np.any(first < 0) or np.any(first > 1)
                or np.any(second < 0) or np.any(second > 1)):
            raise ValueError("Expected same-size finite HW structures in [0, 1], at least 16 pixels")
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setFinestScale(0)
        estimator.setUseMeanNormalization(True)
        estimator.setUseSpatialPropagation(True)
        return estimator.calc(np.ascontiguousarray(np.rint(first * 255), dtype=np.uint8),
                              np.ascontiguousarray(np.rint(second * 255), dtype=np.uint8), None)


def sampleUvArray(values: np.ndarray, samplingCoordinates: np.ndarray, *,
                  nearest: bool = False) -> np.ndarray:
    """Sample HW or HWC data using pixel-center (x,y) coordinates, zero outside.

    This primitive does not invent a validity policy. Alignment separately
    verifies every nonzero bilinear footprint before using sampled observations.
    Callers can use it for matching photo coordinates/cosines with the same map.
    """
    values = np.asarray(values)
    coordinates = np.asarray(samplingCoordinates, np.float32)
    if (values.ndim not in (2, 3) or coordinates.shape != values.shape[:2] + (2,)
            or not np.isfinite(coordinates).all()):
        raise ValueError("Expected HW/HWC values and finite HW2 sampling coordinates")
    return cv2.remap(values.astype(np.float32), coordinates[..., 0], coordinates[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _normalizedStructure(colors: np.ndarray, support: np.ndarray) -> np.ndarray:
    luminance = colors @ np.array([.2126, .7152, .0722], np.float32)
    logLuminance = np.log(np.maximum(luminance, .01))
    denominator = gaussian_filter(support.astype(np.float32), 8, mode="constant")
    localMean = gaussian_filter(logLuminance * support, 8, mode="constant") / np.maximum(denominator, 1e-6)
    residual = logLuminance - localMean
    variance = gaussian_filter(residual ** 2 * support, 8, mode="constant") / np.maximum(denominator, 1e-6)
    normalized = np.clip(residual / np.sqrt(variance + .025 ** 2), -3, 3)
    return np.where(support, .5 + normalized / 6, .5).astype(np.float32)


def _smoothFlow(flow: np.ndarray, eligibility: np.ndarray,
                labels: np.ndarray, sigma: float) -> np.ndarray:
    result = np.zeros_like(flow)
    # Independent semantic masks prevent smoothing brow motion into eyelids.
    for labelId in np.unique(labels[eligibility]):
        mask = eligibility & (labels == labelId)
        if sigma == 0:
            result[mask] = flow[mask]
            continue
        denominator = gaussian_filter(mask.astype(np.float32), sigma, mode="constant")
        for axis in range(2):
            numerator = gaussian_filter(flow[..., axis] * mask, sigma, mode="constant")
            result[..., axis][mask] = (numerator / np.maximum(denominator, 1e-6))[mask]
    return result


def _safeFootprint(mask: np.ndarray, labels: np.ndarray, targetLabels: np.ndarray,
                   coordinates: np.ndarray) -> np.ndarray:
    """Every geometrically contributing corner must be valid and same-class."""
    height, width = mask.shape
    x, y = coordinates[..., 0], coordinates[..., 1]
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    good = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        # Exact integer samples do not use the upper interpolation neighbor.
        needed = ((dx == 0) | (x > x0 + 1e-6)) & ((dy == 0) | (y > y0 + 1e-6))
        sx, sy = np.clip(x0 + dx, 0, width - 1), np.clip(y0 + dy, 0, height - 1)
        good &= ~needed | (mask[sy, sx] & (labels[sy, sx] == targetLabels))
    return good


def _pathIsSafe(flow: np.ndarray, grid: np.ndarray, sourceSupport: np.ndarray,
                sourceLabels: np.ndarray, referenceSupport: np.ndarray,
                referenceLabels: np.ndarray, maxDisplacement: float) -> np.ndarray:
    safe = sourceSupport & referenceSupport & (sourceLabels == referenceLabels)
    # Half-pixel steps plus complete bilinear footprints cannot jump a one-texel
    # hole or semantic strip even when both endpoints themselves are valid.
    for fraction in np.linspace(0, 1, max(2, int(np.ceil(maxDisplacement * 2)) + 1)):
        coordinates = grid + flow * fraction
        safe &= _safeFootprint(sourceSupport, sourceLabels, referenceLabels, coordinates)
        safe &= _safeFootprint(referenceSupport, referenceLabels, referenceLabels, coordinates)
    return safe


def _jacobianDeterminant(flow: np.ndarray) -> np.ndarray:
    dxDy, dxDx = np.gradient(flow[..., 0])
    dyDy, dyDx = np.gradient(flow[..., 1])
    return (1 + dxDx) * (1 + dyDy) - dxDy * dyDx


def _taperFlow(flow: np.ndarray, mask: np.ndarray, width: float) -> np.ndarray:
    # Padding treats image boundaries as invalid too, including all-true masks.
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    fraction = np.clip(np.maximum(distance - 1, 0) / width, 0, 1)
    fraction = fraction * fraction * (3 - 2 * fraction)
    return (flow * fraction[..., None]).astype(np.float32)


def alignUvObservations(observations: np.ndarray, weights: np.ndarray,
                        semanticLabels: np.ndarray, semanticConfidence: np.ndarray, *,
                        referenceViewIndex: int = 0, flowAdapter: UvFlowAdapter | None = None,
                        maxDisplacement: float = 6, confidenceThreshold: float = .65,
                        consistencyThreshold: float = 1.5, smoothingSigma: float = 2,
                        taperWidth: float = 12, minimumJacobian: float = .35,
                        maximumJacobian: float = 2.5, minimumOverlap: int = 128,
                        structureTolerance: float = .005) -> dict:
    """Conservatively align auxiliary UV observations to a fixed reference.

    Colors are SH-corrected linear RGB. Motion is estimated from local normalized
    log luminance to reduce smooth lighting/color differences. Semantics must be
    confident and agree along the entire path in both observations. Estimates
    exceeding the bound, inconsistent forward/backward flow, poor structure fit,
    and final fold/area-change violations fall back to original observations.

    Structural fit is a training signal, not proof of true correspondence. This
    intentionally cannot repair different expressions or move semantic borders.
    Weights never increase, support never expands, and reference arrays are exact.
    """
    colors = np.asarray(observations, np.float32)
    weightArray = np.asarray(weights, np.float32)
    labels = np.asarray(semanticLabels)
    confidence = np.asarray(semanticConfidence, np.float32)
    shape = colors.shape[:3]
    if (colors.ndim != 4 or colors.shape[-1] != 3 or len(shape) != 3 or shape[0] < 1
            or min(shape[1:]) < 2 or weightArray.shape != shape or labels.shape != shape
            or confidence.shape != shape or labels.dtype.kind not in "iu"
            or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1)
            or not np.isfinite(weightArray).all() or np.any(weightArray < 0)
            or not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1)
            or np.any(labels < 0) or np.any(labels > 18)):
        raise ValueError("Expected finite NHW3 linear colors in [0,1], nonnegative weights and standard semantics")
    if (isinstance(referenceViewIndex, (bool, np.bool_))
            or not isinstance(referenceViewIndex, (int, np.integer))
            or not 0 <= referenceViewIndex < shape[0]):
        raise ValueError("referenceViewIndex must identify an available view")
    scalarParameters = [maxDisplacement, confidenceThreshold, consistencyThreshold, smoothingSigma,
                        taperWidth, minimumJacobian, maximumJacobian, structureTolerance]
    if (not np.isfinite(scalarParameters).all() or maxDisplacement <= 0
            or not 0 < confidenceThreshold <= 1 or consistencyThreshold < 0
            or smoothingSigma < 0 or taperWidth < maxDisplacement * 2
            or not 0 < minimumJacobian <= 1 <= maximumJacobian or structureTolerance < 0
            or isinstance(minimumOverlap, (bool, np.bool_))
            or not isinstance(minimumOverlap, (int, np.integer)) or minimumOverlap < 1):
        raise ValueError("Invalid alignment thresholds; taperWidth must be at least twice maxDisplacement")
    estimator = flowAdapter if flowAdapter is not None else OpenCvUvFlowAdapter()
    aligned = colors.copy()
    alignedWeights = weightArray.copy()
    alignedLabels = labels.copy()
    alignedConfidence = confidence.copy()
    samplingFlow = np.zeros(shape + (2,), np.float32)
    acceptedMask = np.zeros(shape, bool)
    yy, xx = np.indices(shape[1:], dtype=np.float32)
    grid = np.stack([xx, yy], axis=-1)
    support = (weightArray > 0) & (confidence >= confidenceThreshold) & (labels > 0)
    structures = np.stack([_normalizedStructure(color, mask) for color, mask in zip(colors, support)])
    referenceLabels = labels[referenceViewIndex]
    referenceSupport = support[referenceViewIndex]
    referenceStructure = structures[referenceViewIndex]
    viewDiagnostics = []
    for viewIndex in range(shape[0]):
        stats = {"viewIndex": viewIndex, "status": "referenceUnchanged" if viewIndex == referenceViewIndex else "pending",
                 "acceptedTexels": 0}
        viewDiagnostics.append(stats)
        if viewIndex == referenceViewIndex:
            continue
        eligible = (referenceSupport & support[viewIndex]
                    & (referenceLabels == labels[viewIndex]))
        stats["eligibleOverlapTexels"] = int(eligible.sum())
        if eligible.sum() < minimumOverlap:
            stats["status"] = "insufficientSemanticOverlap"
            continue
        sourceStructure = structures[viewIndex]
        try:
            forward = np.asarray(estimator.estimateFlow(referenceStructure, sourceStructure), np.float32)
            backward = np.asarray(estimator.estimateFlow(sourceStructure, referenceStructure), np.float32)
            if (forward.shape != grid.shape or backward.shape != grid.shape
                    or not np.isfinite(forward).all() or not np.isfinite(backward).all()):
                raise ValueError("Flow adapter returned nonfinite values or a non-HW2 shape")
        except (ValueError, RuntimeError, cv2.error) as error:
            stats.update(status="estimatorFailed", failureReason=str(error))
            continue
        bounded = (np.linalg.norm(forward, axis=-1) <= maxDisplacement)
        backwardBounded = np.linalg.norm(backward, axis=-1) <= maxDisplacement
        stats["rejectedExcessDisplacementTexels"] = int((eligible & ~bounded).sum())
        stats["rejectedExcessBackwardDisplacementTexels"] = int((eligible & ~backwardBounded).sum())
        candidateMask = eligible & bounded & backwardBounded
        forward = _smoothFlow(forward, candidateMask, referenceLabels, smoothingSigma)
        backward = _smoothFlow(backward, candidateMask, referenceLabels, smoothingSigma)
        candidateFlow = _taperFlow(forward, candidateMask, taperWidth)
        candidateBackward = _taperFlow(backward, candidateMask, taperWidth)
        pathSafe = _pathIsSafe(candidateFlow, grid, support[viewIndex], labels[viewIndex],
                               referenceSupport, referenceLabels, maxDisplacement)
        consistent = ((np.linalg.norm(candidateFlow + sampleUvArray(candidateBackward, grid + candidateFlow), axis=-1) <= consistencyThreshold)
                      & _safeFootprint(eligible & backwardBounded, referenceLabels, referenceLabels, grid + candidateFlow))
        beforeError = np.abs(referenceStructure - sourceStructure)
        afterError = np.abs(referenceStructure - sampleUvArray(sourceStructure, grid + candidateFlow))
        # Compare smoothed residuals to suppress isolated noise-driven acceptance.
        improved = (gaussian_filter(afterError, 2) <= gaussian_filter(beforeError, 2) + structureTolerance)
        stats["rejectedUnsafePathTexels"] = int((candidateMask & ~pathSafe).sum())
        stats["rejectedInconsistentTexels"] = int((candidateMask & ~consistent).sum())
        stats["rejectedStructureTexels"] = int((candidateMask & ~improved).sum())
        candidateMask &= pathSafe & consistent & improved
        # Rejecting flow is itself a deformation. Re-taper and verify the actual
        # exported field, not merely the unconstrained optical-flow proposal.
        finalFlow = _taperFlow(forward, candidateMask, taperWidth)
        for _ in range(6):
            finalFlow[np.linalg.norm(finalFlow, axis=-1) <= 1e-4] = 0
            jacobian = _jacobianDeterminant(finalFlow)
            finalBackward = _taperFlow(backward, candidateMask, taperWidth)
            finalConsistency = np.linalg.norm(finalFlow + sampleUvArray(finalBackward, grid + finalFlow), axis=-1)
            safe = (_pathIsSafe(finalFlow, grid, support[viewIndex], labels[viewIndex],
                                referenceSupport, referenceLabels, maxDisplacement)
                    & (jacobian >= minimumJacobian) & (jacobian <= maximumJacobian)
                    & (finalConsistency <= consistencyThreshold)
                    & _safeFootprint(eligible & backwardBounded, referenceLabels, referenceLabels, grid + finalFlow))
            unsafe = candidateMask & ~safe
            if not unsafe.any():
                break
            candidateMask &= safe
            finalFlow = _taperFlow(forward, candidateMask, taperWidth)
        finalFlow[np.linalg.norm(finalFlow, axis=-1) <= 1e-4] = 0
        jacobian = _jacobianDeterminant(finalFlow)
        finalBackward = _taperFlow(backward, candidateMask, taperWidth)
        finalConsistency = np.linalg.norm(finalFlow + sampleUvArray(finalBackward, grid + finalFlow), axis=-1)
        if (np.any((jacobian < minimumJacobian) | (jacobian > maximumJacobian))
                or np.any(candidateMask & (finalConsistency > consistencyThreshold))
                or np.any(candidateMask & ~_safeFootprint(eligible & backwardBounded, referenceLabels,
                                                          referenceLabels, grid + finalFlow))
                or np.any(candidateMask & ~_pathIsSafe(finalFlow, grid, support[viewIndex], labels[viewIndex],
                                                       referenceSupport, referenceLabels, maxDisplacement))):
            stats["status"] = "unsafeFinalFieldFallback"
            continue
        coordinates = grid + finalFlow
        finalAfterError = np.abs(referenceStructure - sampleUvArray(sourceStructure, coordinates))
        # Global rollback uses a fixed overlap, never a post-fit shrunken mask.
        if finalAfterError[eligible].mean() > beforeError[eligible].mean() + 1e-6:
            stats.update(status="structureRegressionFallback",
                         structureBefore=float(beforeError[eligible].mean()),
                         proposedStructureAfter=float(finalAfterError[eligible].mean()))
            continue
        accepted = candidateMask & (np.linalg.norm(finalFlow, axis=-1) > 1e-4)
        samplingFlow[viewIndex] = finalFlow
        acceptedMask[viewIndex] = accepted
        warped = sampleUvArray(colors[viewIndex], coordinates)
        aligned[viewIndex][accepted] = warped[accepted]
        alignedWeights[viewIndex][accepted] = np.minimum(weightArray[viewIndex], sampleUvArray(weightArray[viewIndex], coordinates))[accepted]
        alignedLabels[viewIndex][accepted] = sampleUvArray(labels[viewIndex], coordinates, nearest=True)[accepted].astype(labels.dtype)
        alignedConfidence[viewIndex][accepted] = np.minimum(confidence[viewIndex], sampleUvArray(confidence[viewIndex], coordinates, nearest=True))[accepted]
        stats.update(status="aligned" if accepted.any() else "noAcceptedMotion",
                     acceptedTexels=int(accepted.sum()),
                     maximumAppliedDisplacement=float(np.linalg.norm(finalFlow, axis=-1).max()),
                     maximumFinalForwardBackwardError=float(finalConsistency[accepted].max()) if accepted.any() else 0.,
                     minimumFinalJacobian=float(jacobian.min()), maximumFinalJacobian=float(jacobian.max()),
                     structureBefore=float(beforeError[eligible].mean()),
                     structureAfter=float(finalAfterError[eligible].mean()))
    return {"alignedObservations": aligned, "alignedWeights": alignedWeights,
            "alignedSemanticLabels": alignedLabels, "alignedSemanticConfidence": alignedConfidence,
            "samplingFlow": samplingFlow, "samplingCoordinates": grid[None] + samplingFlow,
            "acceptedMask": acceptedMask,
            "diagnostics": {"referenceViewIndex": int(referenceViewIndex),
                            "estimator": type(estimator).__name__,
                            "flowConvention": "output[y,x] = source[y+dy,x+dx]",
                            "maxDisplacementUvPixels": float(maxDisplacement),
                            "taperWidthUvPixels": float(taperWidth),
                            "confidenceThreshold": float(confidenceThreshold),
                            "consistencyThreshold": float(consistencyThreshold),
                            "structureMetric": "absolute local-normalized log-luminance difference; training fit only",
                            "rejectedPolicy": "preserve original observation and weight",
                            "supportExpandedTexels": int(((alignedWeights > 0) & (weightArray <= 0)).sum()),
                            "referenceMaximumError": float(np.abs(aligned[referenceViewIndex] - colors[referenceViewIndex]).max()),
                            "views": viewDiagnostics}}
