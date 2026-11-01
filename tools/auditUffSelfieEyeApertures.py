"""Compare rendered visible eye holes with fixed original-photo eye segmentation.

This is a secondary 2D automatic proxy, independent of the eye-curve binding.
It is not a calibrated anatomical or 3D accuracy measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.artifacts import writeJson
from renderUffMultiCase import grid
from renderUffSelfieCrop import VARIANTS, LABELS, BASELINE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def centroid(mask):
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()]) if len(x) else np.full(2, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/uffSelfieCropStudy20261012")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(root / "cropManifest.json")
    rows = []
    overview = []
    for case in manifest["cases"]:
        caseId = case["caseId"]
        directory = root / "preview" / caseId
        if not (directory / "diagnostics.json").exists():
            rows.append({"caseId": caseId, "status": "unavailable", "reason": "renderMissing"})
            continue
        rendered = read(directory / "diagnostics.json")
        front = rendered["selectedFrontView"]
        view = next(v for v in rendered["views"] if v["viewIndex"] == front)
        frame = directory / f"view{front:02d}"
        output = root / "apertureAudit" / caseId
        output.mkdir(parents=True, exist_ok=True)
        semanticPath = root / "cases" / caseId / "semantics" / f"view{front:02d}.npz"
        with np.load(semanticPath, allow_pickle=False) as data:
            labels, confidence = data["labelMap"], data["confidence"]
        size = tuple(view["renderSize"])
        targets = [np.asarray(Image.fromarray((labels == i).astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0 for i in (4, 5)]
        reliable = [int(((labels == i) & (confidence >= .65)).sum()) >= 12 for i in (4, 5)]
        if not all(reliable) or any(t.sum() < 8 for t in targets):
            rows.append({"caseId": caseId, "status": "unavailable", "reason": "tooFewReliableSemanticEyePixels"})
            continue
        centers = np.stack([centroid(t) for t in targets])
        widths = np.array([np.ptp(np.nonzero(t)[1]) + 1 for t in targets])
        yy, xx = np.nonzero(np.logical_or.reduce(targets))
        pad = max(10, int(widths.max() * .6))
        eyeCrop = (max(0, xx.min()-pad), max(0, yy.min()-pad), min(size[0], xx.max()+pad+1), min(size[1], yy.max()+pad+1))
        photo = Image.open(frame / "photo.png").convert("RGB")
        caseRows, panels, titles = [], [photo.crop(eyeCrop)], ["Original photograph"]
        hashes = {str(semanticPath): sha(semanticPath), str(directory / "diagnostics.json"): sha(directory / "diagnostics.json")}
        for variant in VARIANTS:
            path = frame / f"{variant}_fixed_visible.png"
            if not path.exists():
                continue
            visible = np.asarray(Image.open(path).convert("L")) > 127
            if visible.shape != (size[1], size[0]):
                raise ValueError("Rendered mask size disagrees with frame")
            holes = binary_fill_holes(visible) & ~visible
            components, count = label(holes)
            candidates = [components == index for index in range(1, count + 1)
                          if np.count_nonzero(components == index) >= 2]
            candidates = [m for m in candidates if np.any(np.linalg.norm(centers-centroid(m), axis=1) / widths < 1.5)]
            Image.fromarray(holes.astype(np.uint8) * 255).save(output / f"{variant}_all_holes.png")
            candidateMap = np.zeros_like(visible, dtype=np.uint8)
            for componentIndex, component in enumerate(candidates, 1):
                candidateMap[component] = min(255, componentIndex * 40)
            Image.fromarray(candidateMap).save(output / f"{variant}_candidate_holes.png")
            predicted = [np.zeros_like(visible), np.zeros_like(visible)]
            matched = [False, False]
            if candidates:
                cost = np.linalg.norm(centers[:, None] - np.stack([centroid(m) for m in candidates])[None], axis=-1) / widths[:, None]
                indices, matches = linear_sum_assignment(cost)
                for i, j in zip(indices, matches):
                    if cost[i, j] < 1.5:
                        predicted[i], matched[i] = candidates[j], True
            eyes = []
            paint = np.asarray(photo).copy()
            for index, (target, prediction) in enumerate(zip(targets, predicted)):
                intersection, union = int((target & prediction).sum()), int((target | prediction).sum())
                eyes.append({"semanticLabel": index+4, "matchedRenderedHole": matched[index],
                    "unmatchedReason": None if matched[index] else "No closed silhouette hole matched near target; inspect visibility/occlusion/connectivity in saved masks",
                    "targetArea": int(target.sum()), "predictedArea": int(prediction.sum()),
                    "iou": intersection/union if union else None,
                    "predictedAreaOverTarget": float(prediction.sum()/target.sum()),
                    "centerErrorOverTargetWidth": float(np.linalg.norm(centroid(prediction)-centers[index])/widths[index]) if matched[index] else None})
                paint[target & ~prediction] = [233, 110, 73]
                paint[prediction & ~target] = [70, 145, 242]
                paint[prediction & target] = [97, 205, 127]
            score = float(np.mean([e["iou"] for e in eyes]))
            area = float(np.mean([e["predictedAreaOverTarget"] for e in eyes]))
            caseRows.append({"id": variant, "meanEyeIoU": score, "meanAreaRatio": area, "eyes": eyes,
                             "closedComponentCount": int(count), "nearEyeCandidateCount": len(candidates)})
            panels.append(Image.fromarray(paint).crop(eyeCrop))
            shortLabel = variant.replace("backgroundGray", "env-gray").replace("headGray", "head-gray")
            titles.append(f"{shortLabel} | IoU {score:.3f} | area {area:.2f}")
            hashes[str(path)] = sha(path)
        baseline = next(r for r in caseRows if r["id"] == BASELINE)
        for row in caseRows:
            row["deltaIoUFromBaseline"] = row["meanEyeIoU"]-baseline["meanEyeIoU"]
        result = {"caseId": caseId, "status": "measured", "viewIndex": front, "renderSize": size,
                  "fixedCameraAccepted": view["fixedCameraAccepted"], "eyeCrop": eyeCrop,
                  "variants": caseRows, "sourceHashes": hashes}
        image = grid(panels, titles, 3, panel_size=420, panel_height=165,
                     subtitle=f"{caseId} | orange: photo eye only / blue: mesh hole only / green: overlap")
        image.save(output / "comparison.png")
        writeJson(output / "result.json", result)
        overview.append(output / "comparison.png")
        rows.append(result)
    eligible = [row for row in rows if row["status"] == "measured" and row["fixedCameraAccepted"]]
    summary = {}
    for variant in VARIANTS:
        values = [v for c in eligible for v in c["variants"] if v["id"] == variant]
        summary[variant] = {"caseCount": len(values), "meanIoU": float(np.mean([v["meanEyeIoU"] for v in values])) if values else None,
            "meanDeltaIoU": float(np.mean([v["deltaIoUFromBaseline"] for v in values])) if values else None,
            "betterThanBaselineCount": sum(v["deltaIoUFromBaseline"] > 1e-8 for v in values),
            "worseThanBaselineCount": sum(v["deltaIoUFromBaseline"] < -1e-8 for v in values),
            "meanAreaRatio": float(np.mean([v["meanAreaRatio"] for v in values])) if values else None}
    report = {"status": "complete", "cases": rows, "summary": summary,
        "scriptSha256": sha(__file__), "manifestSha256": sha(root / "cropManifest.json"),
        "method": "Fill rendered mesh silhouette holes, pair nearby connected components to fixed original-photo semantic eye centers; nearest resize into same rendered pixel frame",
        "limits": ["Secondary automatic 2D eye-region overlap proxy, not true 3D/anatomical accuracy.",
            "Independent of ordered curve binding only; it shares the original input photos, baseline camera and automatic semantics.",
            "FaRL eye segmentation differs from STFR target; boundaries can be imperfect, especially makeup and low-resolution images.",
            "Confidence >= 0.65 is a case eligibility gate; the overlap target contains all pixels assigned semantic labels 4 and 5.",
            "Fixed m135_raw camera can favor baseline. No registration or camera refit performed.",
            "Finite preview pixel resolution quantizes areas. Missing hole matches have empty predicted mask and zero IoU.",
            "No ordered eye-curve binding is used; this measures visible aperture only, not upper/lower lid geometry."]}
    writeJson(root / "apertureAudit.json", report)
    print(json.dumps({"cases": len(rows), "eligibleCases": len(eligible), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
