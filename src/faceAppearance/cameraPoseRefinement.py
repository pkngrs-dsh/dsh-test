"""Small bounded camera-extrinsic refinement with fixed 3D correspondences.

Rotation is applied about a fixed object-space pivot, followed by translation in
camera coordinates. Intrinsics and object points are never optimized. Held-out
points and semantic groups are evaluated separately from the fitting objective.
The caller remains responsible for the quality of the supplied correspondences.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import projectPoints


def _finiteArray(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _mask(value, count, name):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask with shape ({count},)")
    return array


def _positiveScalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def solveBoundedCameraPose(
    objectPoints, imagePoints, intrinsics, rotation, translation, trainingMask,
    *, pivot, rotationVectorComponentDegrees=2.0,
    translationBounds=(0.05, 0.05, 0.12), regularizerWeight=0.2,
    softL1Pixels=2.0, minimumTrainingCount=6, maximumEvaluations=200,
    solverTolerance=1e-11,
):
    """Fit six normalized extrinsic parameters, each bounded to [-1, 1].

    ``translationBounds`` uses the same length unit as ``objectPoints``. The
    rotation limit applies to each rotation-vector component, not its norm.
    Only ``trainingMask`` points enter the residual. A candidate is returned even
    when it touches a bound; call :func:`validateCameraPoseCandidate` before use.
    """
    points = np.asarray(objectPoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("objectPoints must have shape (N, 3)")
    count = len(points)
    points = _finiteArray(points, (count, 3), "objectPoints")
    targets = _finiteArray(imagePoints, (count, 2), "imagePoints")
    k = _finiteArray(intrinsics, (3, 3), "intrinsics")
    r = _finiteArray(rotation, (3, 3), "rotation")
    t = _finiteArray(translation, (3,), "translation")
    fixedPivot = _finiteArray(pivot, (3,), "pivot")
    bounds = _finiteArray(translationBounds, (3,), "translationBounds")
    training = _mask(trainingMask, count, "trainingMask")
    if np.any(bounds <= 0):
        raise ValueError("translationBounds must be positive")
    if abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("intrinsics must be invertible")
    if (not np.allclose(r @ r.T, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-7, rtol=0)):
        raise ValueError("rotation must be a proper orthonormal matrix")
    for value, name in ((rotationVectorComponentDegrees, "rotationVectorComponentDegrees"),
                        (softL1Pixels, "softL1Pixels"), (solverTolerance, "solverTolerance")):
        _positiveScalar(value, name)
    if not np.isfinite(regularizerWeight) or regularizerWeight < 0:
        raise ValueError("regularizerWeight must be finite and nonnegative")
    for value, name in ((minimumTrainingCount, "minimumTrainingCount"),
                        (maximumEvaluations, "maximumEvaluations")):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    rotationBound = np.deg2rad(rotationVectorComponentDegrees)
    pivotCv = fixedPivot @ r.T + t

    def pose(parameters):
        delta = Rotation.from_rotvec(parameters[:3] * rotationBound).as_matrix()
        newRotation = delta @ r
        newTranslation = delta @ (t - pivotCv) + pivotCv + parameters[3:] * bounds
        return newRotation, newTranslation

    def residual(parameters):
        newRotation, newTranslation = pose(parameters)
        projected = projectPoints(points[training], k, newRotation, newTranslation)[0]
        return np.r_[(projected - targets[training]).ravel(), parameters * regularizerWeight]

    fit = None
    if training.sum() >= minimumTrainingCount:
        fit = least_squares(
            residual, np.zeros(6), bounds=(-np.ones(6), np.ones(6)),
            loss="soft_l1", f_scale=softL1Pixels, max_nfev=maximumEvaluations,
            xtol=solverTolerance, ftol=solverTolerance, gtol=solverTolerance,
        )
    parameters = np.zeros(6) if fit is None else fit.x
    newRotation, newTranslation = pose(parameters)
    before, beforeDepth = projectPoints(points, k, r, t)
    after, afterDepth = projectPoints(points, k, newRotation, newTranslation)
    return {
        "status": "fitted" if fit is not None else "insufficientTrainingConstraints",
        "solverConverged": bool(fit is not None and fit.success),
        "solverEvaluations": 0 if fit is None else int(fit.nfev),
        "intrinsics": k.copy(), "rotation": newRotation, "translation": newTranslation,
        "normalizedParameters": parameters.copy(), "trainingMask": training.copy(),
        "beforeProjection": before, "afterProjection": after,
        "beforeDepth": beforeDepth, "afterDepth": afterDepth,
        "rotationDeltaDegrees": float(np.degrees(np.linalg.norm(parameters[:3] * rotationBound))),
        "translationAtPivot": parameters[3:] * bounds,
    }


def _errorMetrics(before, after, targets, mask):
    count = int(mask.sum())
    if not count:
        return {"count": 0, "meanBefore": None, "meanAfter": None}
    return {
        "count": count,
        "meanBefore": float(np.linalg.norm(before[mask] - targets[mask], axis=1).mean()),
        "meanAfter": float(np.linalg.norm(after[mask] - targets[mask], axis=1).mean()),
    }


def validateCameraPoseCandidate(
    candidate, imagePoints, heldoutMask, *, validationGroups=None,
    minimumHeldoutCount=8, minimumGroupCount=5, maximumErrorRatio=0.8,
    normalizedBoundThreshold=0.995,
):
    """Require improvement on disjoint held-out points and each named subgroup.

    Every validation group must be a subset of ``heldoutMask``. A converged
    solve and positive source/candidate camera depth are required for every
    training or held-out point. The error thresholds require 20% lower mean
    Euclidean pixel error on all held-out points and on each subgroup, without
    touching bounds. This is a numerical diagnostic gate, not proof of
    anatomical correctness.
    """
    targets = np.asarray(imagePoints, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("imagePoints must have shape (N, 2)")
    count = len(targets)
    targets = _finiteArray(targets, (count, 2), "imagePoints")
    before = _finiteArray(candidate["beforeProjection"], (count, 2), "beforeProjection")
    after = _finiteArray(candidate["afterProjection"], (count, 2), "afterProjection")
    beforeDepth = _finiteArray(candidate["beforeDepth"], (count,), "beforeDepth")
    afterDepth = _finiteArray(candidate["afterDepth"], (count,), "afterDepth")
    parameters = _finiteArray(candidate["normalizedParameters"], (6,), "normalizedParameters")
    training = _mask(candidate["trainingMask"], count, "trainingMask")
    heldout = _mask(heldoutMask, count, "heldoutMask")
    if np.any(training & heldout):
        raise ValueError("Training and held-out masks must be disjoint")
    for value, name in ((minimumHeldoutCount, "minimumHeldoutCount"),
                        (minimumGroupCount, "minimumGroupCount")):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(maximumErrorRatio) or not 0 < maximumErrorRatio <= 1:
        raise ValueError("maximumErrorRatio must be in (0, 1]")
    if not np.isfinite(normalizedBoundThreshold) or not 0 < normalizedBoundThreshold <= 1:
        raise ValueError("normalizedBoundThreshold must be in (0, 1]")
    groups = {}
    for name, mask in (validationGroups or {}).items():
        mask = _mask(mask, count, f"validationGroups[{name}]")
        if np.any(mask & ~heldout):
            raise ValueError(f"Validation group {name} must be a held-out subset")
        groups[name] = _errorMetrics(before, after, targets, mask)

    heldoutMetrics = _errorMetrics(before, after, targets, heldout)
    touchesBound = bool(np.max(np.abs(parameters)) > normalizedBoundThreshold)
    reasons = []
    if candidate["status"] != "fitted":
        reasons.append("insufficientTrainingConstraints")
    if not candidate["solverConverged"]:
        reasons.append("solverNotConverged")
    required = training | heldout
    if np.any(beforeDepth[required] <= 0):
        reasons.append("nonpositiveSourceDepth")
    if np.any(afterDepth[required] <= 0):
        reasons.append("nonpositiveCandidateDepth")
    if touchesBound:
        reasons.append("parameterBoundTouched")
    for name, metrics, minimum in [
        ("heldout", heldoutMetrics, minimumHeldoutCount),
        *[(name, metrics, minimumGroupCount) for name, metrics in groups.items()],
    ]:
        if metrics["count"] < minimum:
            reasons.append(f"{name}:insufficientPoints")
        elif metrics["meanAfter"] > metrics["meanBefore"] * maximumErrorRatio:
            reasons.append(f"{name}:insufficientImprovement")
    return {
        "accepted": not reasons, "reasons": reasons, "anyBoundTouched": touchesBound,
        "heldout": heldoutMetrics, "groups": groups,
    }
