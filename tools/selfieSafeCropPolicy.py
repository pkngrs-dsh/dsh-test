"""Frozen input-only routing for a four-photo UFF crop validation experiment.

This is deliberately an experiment helper, not a production default. It reads
original-photo semantics, detections and crop metadata only. A whole sequence
uses either m110_headGray or m135_raw; reconstruction outputs are never inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_opening, label


POLICY_VERSION = "selfie-safe-crop-input-only-v1"
CONFIG = {
    "expectedViewCount": 4,
    "candidateVariantId": "m110_headGray",
    "fallbackVariantId": "m135_raw",
    "semanticMinConfidence": 0.55,
    "mainFaceLabelIds": [1, 2, 3, 4, 5, 10, 11, 12, 13],
    "openingRadiusFaceScale": 0.006,
    "anchorRadiusFaceScale": 0.05,
    "minMainAreaFaceScaleSquared": 0.15,
    "maxMainAreaFaceScaleSquared": 1.40,
    "minReliableMainFraction": 0.55,
    "boundScanMinSpanFaceScale": 0.15,
    "boundScanMinRunFaceScale": 0.01,
    "verticalClearanceFaceScale": 0.01,
    "horizontalClearanceFaceScale": 0.006,
    "organClearanceFaceScale": 0.015,
    "sourceClearanceFaceScale": 0.008,
    "maxPaddingFraction": 0.10,
    "minOrganPixels": 6,
    "minOrganAreaFaceScaleSquared": 0.00005,
    "minDetectionScore": 0.90,
    "lowResolutionFaceScalePixels": 256,
}

RATIONALE = {
    "openingRadiusFaceScale": "Remove thin semantic fragments at 0.6% of detected face size; resolution scales with source pixels.",
    "mainFace": "Select the opened component supported around the frozen nose anchor, then require plausible area and reliable semantics. Hair, ears and hat never enter this mask.",
    "robustBounds": "Rows/columns need 15% of face-scale support for a continuous 1% run. This suppresses narrow attached hair mislabeled as skin; these bounds are conservative input proxies, not a true anatomical contour.",
    "clearance": "Require 1% face-scale above/below and 0.6% at sides, roughly 4.7 and 2.8 pixels in a 512 tight crop; organs require 1.5%. These are fixed engineering buffers, not statistically learned accuracy thresholds.",
    "padding": "Over 10% copied-edge pixels is a context-risk gate. A wider crop can worsen padding; baseline routing alone does not recover unseen image detail.",
    "source": "Face or detector near/outside original edges requires recapture or review, even if a padded crop contains the coordinates.",
    "semanticFailure": "Missing/uncertain organs or a missing reliable main face triggers baseline routing and manual review. A hidden far-side eye/brow can be normal self-occlusion, so missing semantics alone never sets retakeRequired. This conservative policy can decline safe tight crops.",
    "lowResolution": "Below 256 source face-scale pixels is only a quality warning; no rejection based on resolution alone.",
    "development": "Thresholds developed with only original input/semantic/crop QA from the prior 10 groups; no geometry or reconstruction metric is read by this module.",
}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bounds(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)] if len(x) else None


def _supported_extent(counts, scale):
    valid = counts >= CONFIG["boundScanMinSpanFaceScale"] * scale
    regions, count = label(valid)
    sizes = np.bincount(regions)
    keep = sizes >= max(1, int(np.ceil(CONFIG["boundScanMinRunFaceScale"] * scale)))
    keep[0] = False
    locations = np.flatnonzero(keep[regions])
    if not len(locations):
        return None
    return [int(locations[0]), int(locations[-1] + 1)]


def _robust_bounds(mask, scale):
    x = _supported_extent(mask.sum(axis=0), scale)
    y = _supported_extent(mask.sum(axis=1), scale)
    return [x[0], y[0], x[1], y[1]] if x is not None and y is not None else None


def _clearance(bounds, rectangle, scale):
    b, r = np.asarray(bounds), np.asarray(rectangle)
    return ((np.r_[b[:2] - r[:2], r[2:] - b[2:]]) / scale).tolist()


def _largest(mask):
    components, _ = label(mask)
    counts = np.bincount(components.ravel())
    counts[0] = 0
    return components == counts.argmax() if counts.max() else np.zeros_like(mask)


def _crop_check(item, bounds, organs, scale):
    rectangle = item["cropBoxXyxy"]
    clearance = _clearance(bounds, rectangle, scale)
    required = [CONFIG["horizontalClearanceFaceScale"], CONFIG["verticalClearanceFaceScale"],
                CONFIG["horizontalClearanceFaceScale"], CONFIG["verticalClearanceFaceScale"]]
    names = ["left", "forehead", "right", "chin"]
    reasons = ["mainFaceInsufficientClearance:" + name for name, value, minimum in zip(names, clearance, required)
               if value < minimum]
    organ_clearances = {}
    for name, organ in organs.items():
        values = _clearance(organ["boundsXyxy"], rectangle, scale)
        organ_clearances[name] = values
        if min(values) < CONFIG["organClearanceFaceScale"]:
            reasons.append("organInsufficientClearance:" + name)
    w, h = item["orientedSize"]
    x0, y0, x1, y1 = rectangle
    observed = max(0, min(w, x1) - max(0, x0)) * max(0, min(h, y1) - max(0, y0))
    padding = 1 - observed / ((x1 - x0) * (y1 - y0))
    if padding > CONFIG["maxPaddingFraction"]:
        reasons.append("excessiveCopiedEdgePadding")
    return {"accepted": not reasons, "reasons": reasons, "cropBoxXyxy": rectangle,
            "mainFaceClearanceFaceScaleLeftTopRightBottom": clearance,
            "organClearanceFaceScaleLeftTopRightBottom": organ_clearances,
            "paddingFractionExactSourceArea": padding}


def evaluate_view(detection, labels, confidence, tight_item, fallback_item, *, view_index=0):
    """Evaluate only original-input observations; any invalid semantics fails closed."""
    result = {"viewIndex": view_index, "accepted": False, "fallbackReasons": [],
              "retakeRequired": False, "manualReviewRequired": False,
              "retakeOrReviewRecommendation": "none", "retakeOrReviewReasons": [], "qualityWarnings": []}
    try:
        labels, confidence = np.asarray(labels), np.asarray(confidence)
        box = np.asarray(detection["boxXyxy"], dtype=float)
        landmarks = np.asarray(detection["landmarks"], dtype=float)
        if box.shape != (4,) or landmarks.shape != (5, 2) or not np.isfinite(box).all() or not np.isfinite(landmarks).all():
            raise ValueError("invalidDetectionCoordinates")
        scale = float(max(box[2:] - box[:2]))
        if scale <= 0 or detection["score"] < CONFIG["minDetectionScore"]:
            raise ValueError("unreliableDetection")
        w, h = tight_item["orientedSize"]
        if labels.shape != (h, w) or confidence.shape != labels.shape or not np.isfinite(confidence).all():
            raise ValueError("invalidOriginalSemanticShapeOrConfidence")
        if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() > 18 or confidence.min() < 0 or confidence.max() > 1:
            raise ValueError("invalidSemanticValues")
        if tight_item["sha256"] != fallback_item["sha256"] or tight_item["orientedSize"] != fallback_item["orientedSize"]:
            raise ValueError("cropPhotoIdentityMismatch")
        if scale < CONFIG["lowResolutionFaceScalePixels"]:
            result["qualityWarnings"].append("lowSourceFaceResolution")
        raw = np.isin(labels, CONFIG["mainFaceLabelIds"])
        reliable = raw & (confidence >= CONFIG["semanticMinConfidence"])
        radius = max(1, int(round(scale * CONFIG["openingRadiusFaceScale"])))
        opened = binary_opening(reliable, structure=np.ones((2 * radius + 1, 2 * radius + 1)))
        components, _ = label(opened)
        ar = max(1, int(round(scale * CONFIG["anchorRadiusFaceScale"])))
        ax, ay = np.round(landmarks[2]).astype(int)
        near = components[max(0, ay-ar):min(h, ay+ar+1), max(0, ax-ar):min(w, ax+ar+1)]
        counts = np.bincount(near.ravel())
        if not len(counts):
            raise ValueError("noseAnchorOutsideSemanticImage")
        counts[0] = 0
        if not counts.max():
            raise ValueError("noReliableMainFaceAtNoseAnchor")
        main = components == counts.argmax()
        area = float(main.sum() / scale**2)
        if not CONFIG["minMainAreaFaceScaleSquared"] <= area <= CONFIG["maxMainAreaFaceScaleSquared"]:
            raise ValueError("implausibleMainFaceArea")
        # The largest unthresholded connected component estimates support; this
        # check rejects a confident little patch inside an otherwise uncertain face.
        raw_main = _largest(raw)
        fraction = float((raw_main & reliable).sum() / max(1, raw_main.sum()))
        if fraction < CONFIG["minReliableMainFraction"]:
            raise ValueError("insufficientReliableFaceSemantics")
        bounds = _robust_bounds(main, scale)
        if bounds is None:
            raise ValueError("noStableMainFaceBounds")
        organs = {}
        for name, ids in {"browA": [2], "browB": [3], "eyeA": [4], "eyeB": [5],
                          "nose": [10], "mouth": [11, 12, 13]}.items():
            mask = _largest(np.isin(labels, ids) & (confidence >= CONFIG["semanticMinConfidence"]))
            minimum = max(CONFIG["minOrganPixels"], CONFIG["minOrganAreaFaceScaleSquared"] * scale**2)
            if mask.sum() < minimum:
                raise ValueError("missingReliableOrgan:" + name)
            organs[name] = {"boundsXyxy": _bounds(mask), "reliablePixels": int(mask.sum())}
        source_clearance = _clearance(bounds, [0, 0, w, h], scale)
        source_reasons = []
        if min(source_clearance) < CONFIG["sourceClearanceFaceScale"]:
            source_reasons.append("mainFaceNearOriginalPhotoBoundary")
        if min(box[:2]) < 0 or box[2] > w or box[3] > h or (landmarks < 0).any() or (landmarks >= [w, h]).any():
            source_reasons.append("detectorOrLandmarksOutsideOriginalPhoto")
        tight = _crop_check(tight_item, bounds, organs, scale)
        fallback = _crop_check(fallback_item, bounds, organs, scale)
        result.update({"faceScalePixels": scale, "openingRadiusPixels": radius,
            "reliableMainAreaFaceScaleSquared": area, "reliableMainFraction": fraction,
            "mainBoundsXyxy": bounds, "openedMainFullBoundsXyxy": _bounds(main),
            "originalClearanceFaceScaleLeftTopRightBottom": source_clearance,
            "organs": organs, "candidate": tight, "fallback": fallback})
        result["fallbackReasons"] = source_reasons + tight["reasons"]
        result["accepted"] = not result["fallbackReasons"]
        result["retakeOrReviewReasons"] = source_reasons + (["fallbackStillUnsafe:" + r for r in fallback["reasons"]] if not result["accepted"] else [])
        result["retakeRequired"] = bool(result["retakeOrReviewReasons"])
        result["retakeOrReviewRecommendation"] = "recaptureOrAdjustFraming" if result["retakeRequired"] else "none"
        result["fallbackRepairsObservedInputRisk"] = not result["accepted"] and not result["retakeRequired"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        reason = "inputOrSemanticFailure:" + str(exc)
        result["fallbackReasons"].append(reason)
        result["retakeOrReviewReasons"].append(reason)
        result["manualReviewRequired"] = True
        result["retakeOrReviewRecommendation"] = "manualReview"
        result["fallbackRepairsObservedInputRisk"] = False
    return result


def evaluate_case(case_root, *, case_id=None):
    """Read existing frozen original semantics and crop metadata, never geometry."""
    root = Path(case_root)
    evidence = [root / "detections.json", root / "m110_headGray/inputs.json", root / "m135_raw/inputs.json"]
    result = {"schemaVersion": 1, "policyVersion": POLICY_VERSION, "caseId": case_id or root.name,
              "config": CONFIG.copy(), "configSha256": _json_hash(CONFIG),
              "implementationSha256": _file_hash(__file__), "routingUnit": "wholeFourPhotoSequence",
              "perViewChecks": [], "inputEvidenceSha256": {}, "retakeRequired": False, "manualReviewRequired": False}
    try:
        detections = _read(evidence[0])["detections"]
        tight, fallback = _read(evidence[1]), _read(evidence[2])
        if len(detections) != CONFIG["expectedViewCount"] or len(tight) != len(detections) or len(fallback) != len(detections):
            raise ValueError("expectedExactlyFourConsistentViews")
        for i, (detection, item, base) in enumerate(zip(detections, tight, fallback)):
            path = root / "semantics" / f"view{i:02d}.npz"
            evidence.append(path)
            with np.load(path, allow_pickle=False) as archive:
                check = evaluate_view(detection, archive["labelMap"], archive["confidence"], item, base, view_index=i)
            result["perViewChecks"].append(check)
        result["accepted"] = all(v["accepted"] for v in result["perViewChecks"])
        result["retakeRequired"] = any(v["retakeRequired"] for v in result["perViewChecks"])
        result["manualReviewRequired"] = any(v["manualReviewRequired"] for v in result["perViewChecks"])
        result["fallbackReasons"] = [{"viewIndex": v["viewIndex"], "reasons": v["fallbackReasons"]}
                                     for v in result["perViewChecks"] if v["fallbackReasons"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(accepted=False, manualReviewRequired=True,
                      fallbackReasons=[{"viewIndex": None, "reasons": ["missingOrInvalidInputEvidence:" + str(exc)]}])
    result["inputEvidenceSha256"] = {str(p): _file_hash(p) for p in evidence if p.exists()}
    result["selectedVariantId"] = CONFIG["candidateVariantId"] if result["accepted"] else CONFIG["fallbackVariantId"]
    result["selectedVariant"] = result["selectedVariantId"]
    result["retakeOrReviewRecommendation"] = ("recaptureOrAdjustFraming" if result["retakeRequired"] else
                                               "manualReview" if result["manualReviewRequired"] else "none")
    result["deliveryRecommendation"] = ("recaptureOrManualReviewBeforeDelivery" if result["retakeRequired"] or result["manualReviewRequired"]
                                         else "eligibleForOfflineQualityEvaluation")
    result["limits"] = ["No reconstruction result, UV, camera, eyelid metric or render is read.",
        "Input safety is not evidence that the selected reconstruction is more accurate.",
        "Automatic segmentation can miss real truncation or mislabel broad hair/neck regions; robust bounds suppress thin true protrusions too.",
        "Normal self-occlusion can hide an eye/brow; missing semantic support causes conservative fallback/manual review, not a claim that the crop cut an organ off.",
        "A fallback route does not restore original-photo content or certify a usable model.",
        "All four frames must pass; a side/up frame can cause the entire sequence to use the baseline."]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, default=Path("runs/uffSelfieCropStudy20261012"))
    parser.add_argument("--output", type=Path, default=Path("runs/uffSelfieSafeCropValidation20261012/policyDevelopment.json"))
    args = parser.parse_args()
    # A development report deliberately loads only the case identifiers from
    # the original input selection, avoiding the reconstruction manifest.
    selection = _read(args.development_root / "selection/selection.json")
    cases = selection["cases"]
    results = [evaluate_case(args.development_root / "cases" / c["caseId"]) for c in cases]
    report = {"schemaVersion": 1, "status": "frozenForProspectiveValidation", "policyVersion": POLICY_VERSION,
              "config": CONFIG, "configSha256": _json_hash(CONFIG), "implementationSha256": _file_hash(__file__),
              "rationale": RATIONALE, "developmentInputSelectionSha256": _file_hash(args.development_root / "selection/selection.json"),
              "usesReconstructionResults": False, "cases": results,
              "summary": {"caseCount": len(results), "candidateSelectedCount": sum(r["accepted"] for r in results),
                          "fallbackCount": sum(not r["accepted"] for r in results),
                          "recaptureOrReviewCount": sum(r["retakeRequired"] or r["manualReviewRequired"] for r in results)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "configSha256": report["configSha256"],
                      "implementationSha256": report["implementationSha256"]}, indent=2))


if __name__ == "__main__":
    main()
