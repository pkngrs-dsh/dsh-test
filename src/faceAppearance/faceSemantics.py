"""Model-independent face semantics in original image pixel coordinates.

IDs retain the old project's 19-class BiSeNet order, but callers use names.
Left/right preserve the source dataset's labels: they are never inferred from
image X coordinates or swapped for a mirrored photograph. They must not be
treated as verified anatomical sides by a consumer. Pairwise unions (brows,
eyes, ears) are safe for view-independent protection masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Iterable

import numpy as np


semanticLabelNames = (
    "background", "skin", "leftBrow", "rightBrow", "leftEye", "rightEye",
    "glasses", "leftEar", "rightEar", "earring", "nose", "innerMouth",
    "upperLip", "lowerLip", "neck", "necklace", "cloth", "hair", "hat",
)
semanticLabelIds = {name: index for index, name in enumerate(semanticLabelNames)}
sourceLabelMapping = {
    "background": "background", "face": "skin", "lb": "leftBrow",
    "rb": "rightBrow", "le": "leftEye", "re": "rightEye", "eyeg": "glasses",
    "lr": "leftEar", "rr": "rightEar", "earr": "earring", "nose": "nose",
    "imouth": "innerMouth", "ulip": "upperLip", "llip": "lowerLip",
    "neck": "neck", "neck_l": "necklace", "cloth": "cloth", "hair": "hair",
    "hat": "hat",
}


def normalizeParserLabels(labelMap: np.ndarray, sourceLabelNames: Iterable[str]) -> np.ndarray:
    """Remap FACER labels explicitly; an unknown upstream schema fails closed."""
    labels = np.asarray(labelMap)
    names = tuple(sourceLabelNames)
    if (labels.ndim != 2 or min(labels.shape) < 1
            or labels.dtype.kind not in "iu" or labels.dtype.kind == "b"):
        raise ValueError("labelMap must be a nonempty HW integer array")
    if not names or len(set(names)) != len(names):
        raise ValueError("sourceLabelNames must be nonempty and unique")
    unknownNames = set(names) - sourceLabelMapping.keys()
    if unknownNames:
        raise ValueError(f"Unknown parser labels: {sorted(unknownNames)}")
    if labels.min() < 0 or labels.max() >= len(names):
        raise ValueError("labelMap contains an ID outside sourceLabelNames")
    lookup = np.array([semanticLabelIds[sourceLabelMapping[name]] for name in names], dtype=np.uint8)
    return lookup[labels]


@dataclass(frozen=True)
class FaceSemanticResult:
    """Portable result with no dependency on torch or a specific parser.

    ``confidence`` is the uncalibrated maximum softmax probability, not an
    accuracy estimate. ``validMask`` excludes pixels outside the parser crop.
    Landmarks retain RetinaFace's upstream five-point order and XY pixel units;
    they are alignment anchors, not dense eyelid/lip boundary annotations.
    """

    labelMap: np.ndarray
    confidence: np.ndarray
    landmarks: np.ndarray
    validMask: np.ndarray | None = None
    boxXyxy: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    labelNames: tuple[str, ...] = semanticLabelNames

    def __post_init__(self) -> None:
        labels = np.asarray(self.labelMap)
        confidence = np.asarray(self.confidence)
        landmarks = np.asarray(self.landmarks)
        if tuple(self.labelNames) != semanticLabelNames:
            raise ValueError("FaceSemanticResult requires the standard semanticLabelNames schema")
        if (labels.ndim != 2 or min(labels.shape) < 1 or labels.dtype.kind not in "iu"
                or labels.min() < 0 or labels.max() >= len(semanticLabelNames)):
            raise ValueError("labelMap must be a nonempty HW array of standard integer labels")
        if (confidence.shape != labels.shape or confidence.dtype.kind != "f"
                or not np.isfinite(confidence).all() or np.any(confidence < 0)
                or np.any(confidence > 1)):
            raise ValueError("confidence must be a finite floating HW array in [0, 1]")
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("landmarks must contain five finite XY pixel coordinates")
        if self.validMask is not None:
            validMask = np.asarray(self.validMask)
            if validMask.shape != labels.shape or validMask.dtype != np.bool_:
                raise ValueError("validMask must be a boolean HW array")
        if self.boxXyxy is not None:
            box = np.asarray(self.boxXyxy)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or np.any(box[2:] <= box[:2])):
                raise ValueError("boxXyxy must be finite XYXY with positive extent")
            object.__setattr__(self, "boxXyxy", box.astype(np.float32))
        object.__setattr__(self, "labelMap", labels.astype(np.uint8))
        object.__setattr__(self, "confidence", confidence.astype(np.float32))
        object.__setattr__(self, "landmarks", landmarks.astype(np.float32))
        if self.validMask is not None:
            object.__setattr__(self, "validMask", np.array(self.validMask, copy=True))


def semanticMask(result: FaceSemanticResult, names: Iterable[str] | str,
                 minConfidence: float = 0.0) -> np.ndarray:
    """Select named classes, excluding unsupported and low-confidence pixels."""
    if (isinstance(minConfidence, (bool, np.bool_)) or not isinstance(minConfidence, Real)
            or not np.isfinite(minConfidence) or not 0 <= minConfidence <= 1):
        raise ValueError("minConfidence must be finite and in [0, 1]")
    requestedNames = (names,) if isinstance(names, str) else tuple(names)
    unknownNames = set(requestedNames) - semanticLabelIds.keys()
    if unknownNames:
        raise ValueError(f"Unknown semantic labels: {sorted(unknownNames)}")
    mask = np.isin(result.labelMap, [semanticLabelIds[name] for name in requestedNames])
    mask &= result.confidence >= minConfidence
    if result.validMask is not None:
        mask &= result.validMask
    return mask
