"""Offline STFR eye-landmark ONNX adapter, using the existing OpenCV CPU backend.

Channel order and crops follow STFR's eyelid_detector and to_wrap.py. Image-side
names are deliberate: mapping these two detections onto parser labels 4/5 is a
separate spatial correspondence decision, not an inferred anatomical identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock

import cv2
import numpy as np


def decodeEyelidHeatmaps(heatmaps: np.ndarray, *, radiusFraction: float = .1) -> dict:
    """Match upstream positive local-centroid decoding with +0.5 heatmap centers.

    Peak scores are raw regressed heatmap amplitudes, NOT probabilities. The
    local centroid variance is a descriptive spread, not calibrated uncertainty.
    Empty/nonpositive maps receive validMask=False and zero coordinates.
    """
    values = np.asarray(heatmaps, np.float32)
    if (values.ndim != 4 or min(values.shape, default=0) < 1 or not np.isfinite(values).all()
            or isinstance(radiusFraction, (bool, np.bool_)) or not np.isscalar(radiusFraction)
            or not np.isfinite(radiusFraction) or radiusFraction <= 0):
        raise ValueError("Expected finite BCHW heatmaps and positive radiusFraction")
    height, width = values.shape[-2:]
    flat = values.reshape(*values.shape[:2], -1)
    peakIndex = flat.argmax(-1)
    peakY, peakX = np.divmod(peakIndex, width)
    yy, xx = np.indices((height, width), dtype=np.float32)
    radiusSquared = radiusFraction ** 2 * height * width
    region = ((xx - peakX[..., None, None]) ** 2 + (yy - peakY[..., None, None]) ** 2 <= radiusSquared)
    local = np.maximum(values, 0) * region
    mass = local.sum((-2, -1))
    valid = mass > np.finfo(np.float32).eps
    denominator = np.maximum(mass, np.finfo(np.float32).eps)
    points = np.stack([(local * (xx + .5)).sum((-2, -1)) / denominator,
                       (local * (yy + .5)).sum((-2, -1)) / denominator], -1)
    variance = (local * ((xx + .5 - points[..., 0, None, None]) ** 2
                        + (yy + .5 - points[..., 1, None, None]) ** 2)).sum((-2, -1)) / denominator
    points[~valid] = 0
    return {"points": points.astype(np.float32), "rawPeakScores": flat.max(-1),
            "validMask": valid, "localSpread": np.sqrt(variance).astype(np.float32)}


def _cropWithPadding(image: np.ndarray, leftTop: np.ndarray, rightBottom: np.ndarray) -> np.ndarray:
    x0, y0 = map(int, leftTop)
    x1, y1 = map(int, rightBottom)
    result = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    height, width = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if sx1 > sx0 and sy1 > sy0:
        result[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image[sy0:sy1, sx0:sx1]
    return result


def _makeInputs(image: np.ndarray, anchors: np.ndarray) -> tuple[dict, dict]:
    height, width = image.shape[:2]
    box = np.stack([anchors.min(0), anchors.max(0)]).astype(np.float64)
    padding = .5 * (box[1] - box[0]).mean()
    box[0] = np.maximum(box[0] - padding, 0)
    box[1] = np.minimum(box[1] + padding, [width, height])
    center = np.round(box.mean(0)).astype(np.int32)
    size = int(np.round((box[1] - box[0]).max()))
    faceLeftTop, faceRightBottom = center - size // 2, center + size // 2
    faceCrop = _cropWithPadding(image, faceLeftTop, faceRightBottom)
    eyeRadius = int(np.round(np.linalg.norm(anchors[1] - anchors[0]) * .33))
    centers = np.round(anchors[:2]).astype(np.int32)
    eyeLeftTop = centers - eyeRadius
    eyeRightBottom = centers + eyeRadius
    crops = [_cropWithPadding(image, lo, hi) for lo, hi in zip(eyeLeftTop, eyeRightBottom)]
    inputs = {}
    for name, crop, resolution, flip in (("input.1", faceCrop, 256, False),
                                        ("leye_img_flip", crops[0], 128, True),
                                        ("reye_img", crops[1], 128, False)):
        resized = cv2.resize(crop, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        if flip:
            resized = resized[:, ::-1]
        inputs[name] = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], np.float32) / 255
    return inputs, {"faceLeftTop": faceLeftTop, "faceSize": np.array(faceCrop.shape[1::-1]),
                    "eyeLeftTop": eyeLeftTop, "eyeSize": eyeRadius * 2,
                    "eyeCropBoxes": np.concatenate([eyeLeftTop, eyeRightBottom], axis=1)}


class EyelidLandmarksAdapter:
    """Preload once; serialized CPU inference without network access or downloads.

    infer accepts original-image RGB uint8 and four ordered XY anchors:
    image-left eye center, image-right eye center, nose, mouth. Five RetinaFace
    anchors are also accepted, replacing the two mouth corners by their mean;
    this is recorded as a crop-anchor adaptation from the upstream 68-point FAN.
    """

    def __init__(self, projectRoot: Path, *, minimumPeakScore: float = .2):
        if (isinstance(minimumPeakScore, (bool, np.bool_))
                or not isinstance(minimumPeakScore, (float, int, np.floating, np.integer))
                or not np.isfinite(minimumPeakScore) or not 0 < minimumPeakScore <= 1):
            raise ValueError("minimumPeakScore must be finite in (0,1]")
        self.projectRoot = Path(projectRoot).resolve()
        self.minimumPeakScore = float(minimumPeakScore)
        self.lock = RLock()
        self.isLoaded = False
        self.net = None
        self.modelMetadata = {}

    def assetPath(self) -> Path:
        return self.projectRoot / "04_STFR/weights/registration/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx"

    def loadBackend(self, path: Path):
        """Replaceable external dependency seam for lifecycle tests."""
        net = cv2.dnn.readNetFromONNX(str(path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if tuple(net.getUnconnectedOutLayersNames()) != ("1246", "1442"):
            raise ValueError("Unexpected STFR eye model output signature")
        return net

    def loadModels(self) -> EyelidLandmarksAdapter:
        with self.lock:
            if self.isLoaded:
                return self
            path = self.assetPath()
            if not path.is_file():
                raise FileNotFoundError(f"Missing local eyelid model: {path}")
            net = self.loadBackend(path)
            self.modelMetadata = {"backend": "OpenCV DNN CPU", "backendVersion": cv2.__version__,
                                  "weightPath": str(path), "weightSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "upstreamImplementation": "04_STFR/stfr/registration/eyelid_detector/eyelid_detector.py",
                                  "upstreamOrderEvidence": "04_STFR/stfr/registration/to_wrap.py:make_landmarks"}
            self.net = net
            self.isLoaded = True
            return self

    def runBackend(self, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
        for name, values in inputs.items():
            self.net.setInput(values, name)
        faceHeatmaps, eyeHeatmaps = self.net.forward(["1246", "1442"])
        if (faceHeatmaps.shape != (1, 68, 64, 64) or eyeHeatmaps.shape != (2, 43, 32, 32)
                or not np.isfinite(faceHeatmaps).all() or not np.isfinite(eyeHeatmaps).all()):
            raise ValueError("Unexpected or nonfinite STFR eye model heatmaps")
        return faceHeatmaps, eyeHeatmaps

    def infer(self, imageRgb: np.ndarray, *, faceAnchors: np.ndarray) -> dict:
        image = np.asarray(imageRgb)
        anchors = np.asarray(faceAnchors, np.float32)
        if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                or min(image.shape[:2], default=0) < 4):
            raise ValueError("Expected original-image RGB uint8 HWC")
        if anchors.shape not in ((4, 2), (5, 2)) or not np.isfinite(anchors).all():
            raise ValueError("faceAnchors must contain four or five finite original-image XY anchors")
        anchorSource = "providedFourPoints"
        if len(anchors) == 5:
            anchors = np.concatenate([anchors[:3], anchors[3:].mean(0, keepdims=True)])
            anchorSource = "retinaFaceFivePointsWithMeanMouthCorners"
        height, width = image.shape[:2]
        if (np.any(anchors < 0) or np.any(anchors >= [width, height])
                or anchors[0, 0] >= anchors[1, 0] or np.linalg.norm(anchors[1] - anchors[0]) < 6):
            raise ValueError("Anchors must be in bounds with image-left then image-right eyes at least six pixels apart")
        with self.lock:
            if not self.isLoaded:
                raise RuntimeError("Call loadModels before eyelid inference")
            inputs, crop = _makeInputs(image, anchors)
            faceHeatmaps, eyeHeatmaps = self.runBackend(inputs)
        face = decodeEyelidHeatmaps(faceHeatmaps)
        eye = decodeEyelidHeatmaps(eyeHeatmaps)
        facePoints = face["points"][0] * crop["faceSize"] / 64 + crop["faceLeftTop"]
        eyePoints = eye["points"] * crop["eyeSize"] / 32
        # Exact upstream flip-back convention, including its pixel-center offset.
        eyePoints[0, :, 0] = crop["eyeSize"] - 1 - eyePoints[0, :, 0]
        eyePoints += crop["eyeLeftTop"][:, None]
        curves = eyePoints[:, 19:].reshape(2, 2, 12, 2).copy()
        scores = eye["rawPeakScores"][:, 19:].reshape(2, 2, 12).copy()
        valid = eye["validMask"][:, 19:].reshape(2, 2, 12).copy()
        spread = eye["localSpread"][:, 19:].reshape(2, 2, 12).copy() * crop["eyeSize"] / 32
        # to_wrap.py explicitly reverses the second eye's lower and upper lines.
        curves[1] = curves[1, :, ::-1]
        scores[1] = scores[1, :, ::-1]
        valid[1] = valid[1, :, ::-1]
        spread[1] = spread[1, :, ::-1]
        valid &= (scores >= self.minimumPeakScore)
        valid &= ((curves >= 0) & (curves < [width, height])).all(-1)
        corners = np.stack([curves[:, :, 0].mean(1), curves[:, :, -1].mean(1)], axis=1)
        endpointGaps = np.linalg.norm(curves[:, 0, [0, -1]] - curves[:, 1, [0, -1]], axis=-1)
        confidence = np.where(valid, np.clip(scores, 0, 1), 0)
        cornerConfidence = np.stack([confidence[:, :, 0].min(1), confidence[:, :, -1].min(1)], axis=1)
        qualities = []
        for index in range(2):
            span = float(np.linalg.norm(corners[index, 1] - corners[index, 0]))
            lower, upper = curves[index]
            direction = corners[index, 1] - corners[index, 0]
            unit = direction / max(span, 1e-6)
            normal = np.array([-unit[1], unit[0]])
            opening = (lower - upper) @ normal
            reasons = []
            if not valid[index].all():
                reasons.append("lowPeakOrOutOfImagePoints")
            if span < 8:
                reasons.append("eyeSpanBelowEightPixels")
            if endpointGaps[index].max() > max(2., span * .08):
                reasons.append("upperLowerEndpointDisagreement")
            if np.quantile(opening[1:-1], .25) < -.5:
                reasons.append("crossedUpperLowerCurves")
            if ((np.diff(lower, axis=0) @ unit < -1).any()
                    or (np.diff(upper, axis=0) @ unit < -1).any()):
                reasons.append("nonmonotonicAlongEyeAxis")
            qualities.append({"imageSide": "left" if index == 0 else "right", "valid": not reasons,
                              "rejectionReasons": reasons, "eyeSpanPixels": span,
                              "medianOpeningPixels": float(np.median(opening[1:-1])),
                              "minimumRawPeakScore": float(scores[index].min()),
                              "meanRawPeakScore": float(scores[index].mean()),
                              "endpointGapPixels": endpointGaps[index].tolist(),
                              "meanLocalHeatmapSpreadPixels": float(spread[index].mean())})
        return {"eyelidCurves": curves.astype(np.float32), "corners": corners.astype(np.float32),
                "curveConfidence": confidence.astype(np.float32), "cornerConfidence": cornerConfidence.astype(np.float32),
                "rawPeakScores": scores, "pointValidMask": valid, "localSpreadPixels": spread,
                "rawEyePoints": eyePoints.astype(np.float32), "rawEyePeakScores": eye["rawPeakScores"],
                "faceLandmarks": facePoints.astype(np.float32), "faceRawPeakScores": face["rawPeakScores"][0],
                "cropBoxes": crop["eyeCropBoxes"], "faceAnchors": anchors.copy(), "quality": qualities,
                "metadata": {**self.modelMetadata, "device": "cpu", "originalSize": [width, height],
                             "anchorSource": anchorSource,
                             "coordinateConvention": "originalImagePixelXY; upstream heatmap centers and flip-back preserved",
                             "eyeOrder": ["imageLeft", "imageRight"], "curveOrder": ["lower", "upper"],
                             "pointOrder": "12 points per curve, image-left endpoint to image-right endpoint",
                             "cornerOrder": ["imageLeftEndpoint", "imageRightEndpoint"],
                             "cornerDefinition": "mean of corresponding upper/lower endpoint predictions",
                             "rawModelOrder": "43 points per eye: iris[0:19], lowerLid[19:31], upperLid[31:43]",
                             "secondEyeCurveReversed": True,
                             "confidenceKind": "raw heatmap peak clipped to [0,1] for weighting; not a calibrated probability",
                             "minimumPeakScore": self.minimumPeakScore,
                             "limitations": ["Quality checks do not prove visibility or identify a hallucinated occluded eye.",
                                             "RetinaFace anchors differ from upstream FAN68 eye means/nose30/lowerLip57.",
                                             "No temporal filtering, image warping, geometric fitting or semantic identity assignment."]}}
