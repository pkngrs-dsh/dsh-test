#!/usr/bin/env python3
"""Freeze six photograph-only, exact-file-disjoint safe-crop validation groups.

CPU only. Selection never reads reconstruction outputs or alters source photos.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "runs/uffMultiCase20261011/dataInventory.json"
PRIOR = ROOT / "runs/uffMultiCase20261011/selectedCases.json"
DEVELOPMENT = ROOT / "runs/uffSelfieCropStudy20261012/selection/selection.json"
OUT = ROOT / "runs/uffSelfieSafeCropValidation20261012/selection"

# Chosen only after visually inspecting all four source photos of nine candidates.
SPECS = [
    (13, "nearPlaid4", "近距离格子衫", "nearFaceBoundary",
     "灰墙、椅背与肩部总体稳定，左右转头及抬头明确；视线有变化。",
     "720×1280 竖图，面部占比较大；侧转时额头、耳部与下巴靠近横向边界。"),
    (40, "softWhiteWall4", "白墙低清 · 侧转模糊", "lowerResolutionAndMotionBlur",
     "白墙、肩部与衣领较稳定，左右转头和抬头可见；侧转时视线明显移动。",
     "544×960 输入，侧转照片有轻微运动模糊；原目录名含 flip，不能假定为未经镜像处理的原始采集。"),
    (95, "brightBacklight4", "窗前逆光 · 胡须", "strongDirectionalLight",
     "窗帘、肩部与座椅总体一致，左右转头及抬头明确；有少量构图变化。",
     "1080×1920 输入，正面窗光有明显眩光和过曝背景，转头后脸部光照不同；胡须边界可见。"),
    (128, "tiedHairForehead4", "扎发露额 · 大侧转", "foreheadAndChinBoundary",
     "木墙和肩部总体稳定，左右大幅转头、抬头明确；一张侧转图有轻微模糊。",
     "1080×1920 输入，发束高于头顶、额头完整露出；侧转的额头和下巴边界接近原图左右边缘。"),
    (152, "dimWindowEarbuds4", "暗正面窗边 · 耳机", "dimFaceAndAccessory",
     "窗框和肩部较稳定，左右转头及较大抬头；相机仍可能有手持位移。",
     "1080×1920 输入，正面偏暗，背景较亮；耳机邻近耳缘，转头时曝光和面部受光变化。"),
    (181, "woodFringe4", "木墙刘海 · 常规自拍", "ordinarySelfieAndHair",
     "木墙、肩部与衣领总体稳定，左右转头及抬头完整；面部保持轻微笑意。",
     "1080×1920 输入，作为较常规自拍条件；刘海遮挡部分额头，抬头时眼睛向上看。"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standard_four(group: dict) -> list[dict]:
    return [next(f for f in group["files"] if Path(f["filename"]).stem == str(i))
            for i in range(1, 5)]


def contact(cases: list[dict], path: Path, width: int = 1600, row_height: int = 420) -> None:
    image = Image.new("RGB", (width, row_height * len(cases)), "#eeeeee")
    draw = ImageDraw.Draw(image)
    for row, case in enumerate(cases):
        draw.text((8, row * row_height + 4),
                  f"{case['caseId']} | idx {case['inventoryIndex']} | source 1,2,3,4", fill="black")
        for col, source_path in enumerate(case["paths"]):
            with Image.open(source_path) as source:
                photo = ImageOps.exif_transpose(source).convert("RGB")
            photo.thumbnail((width // 4 - 8, row_height - 30), Image.Resampling.LANCZOS)
            x = col * width // 4 + (width // 4 - photo.width) // 2
            image.paste(photo, (x, row * row_height + 26))
    image.save(path, quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(INVENTORY.read_text())
    previous = json.loads(PRIOR.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    previous_hashes = {f["sha256"] for c in previous["cases"] for f in c["files"]}
    development_hashes = {f["sha256"] for c in development["cases"] for f in c["files"]}
    eligible, exclusions, duplicate_sets = [], [], set()
    for group in inventory["groups"]:
        reason = None
        if not group["hasStandardFour"]:
            reason = "missingStandardFour"
        elif not group["standardFourUnique"]:
            reason = "repeatedImageWithinFour"
        elif group["sharedFileCountWithKnownBaselines"]:
            reason = "exactFileOverlapWithKnownBaseline"
        elif any(f["sha256"] in previous_hashes for f in standard_four(group)):
            reason = "exactFileOverlapWithPreviousSelectedCases"
        elif any(f["sha256"] in development_hashes for f in standard_four(group)):
            reason = "exactFileOverlapWithDevelopmentCases"
        else:
            key = tuple(sorted(f["sha256"] for f in standard_four(group)))
            if key in duplicate_sets:
                reason = "duplicateFourFileSetOfEarlierEligibleDirectory"
            duplicate_sets.add(key)
        if reason:
            exclusions.append({"inventoryIndex": group["index"],
                               "group": group["relativeGroupPath"],
                               "reason": reason, "stage": "inventoryFilter"})
        else:
            eligible.append(group)
    manual_exclusions = [
        (39, "192×192 原图细节很少，本轮先用 544×960 且带模糊的候选覆盖低清；可另作极低清压力测试。", "fourViewVisualReview"),
        (65, "四视图有明显滚转及背景/镜头构图变化，采集模式更不明确；本轮优先选择视觉上更符合扭头的组。", "fourViewVisualReview"),
        (70, "侧视和抬头照片背景中可见旁人，存在多脸或分割歧义。", "fourViewVisualReview"),
        (154, "棚拍肖像式正面候选，采集协议来源不清。", "frontThumbnailReview"),
        (160, "正面候选严重欠曝，优先选择可看清脸部但有暗光挑战的组。", "frontThumbnailReview"),
        (178, "正面候选严重欠曝，优先选择可看清脸部但有暗光挑战的组。", "frontThumbnailReview"),
    ]
    for index, reason, stage in manual_exclusions:
        exclusions.append({"inventoryIndex": index, "group": inventory["groups"][index]["relativeGroupPath"],
                           "reason": reason, "stage": stage, "status": "excludedForThisStudy"})
    eligible_indices = {g["index"] for g in eligible}
    selected, fresh_hashes, decoded_hashes = [], set(), set()
    for index, case_id, label, stratum, evidence, note in SPECS:
        assert index in eligible_indices, (index, "ineligible")
        group = inventory["groups"][index]
        files = []
        for old in standard_four(group):
            path = Path(old["path"])
            digest = sha256(path)
            assert digest == old["sha256"], f"Source changed since inventory: {path}"
            assert digest not in previous_hashes | development_hashes | fresh_hashes
            fresh_hashes.add(digest)
            with Image.open(path) as source:
                source.load()  # Pillow strict decode: truncated images are not enabled.
                raw_size = list(source.size)
                photo = ImageOps.exif_transpose(source).convert("RGB")
                oriented_size = list(photo.size)
                rgb_hash = hashlib.sha256(str(oriented_size).encode() + photo.tobytes()).hexdigest()
            assert oriented_size == old["orientedSize"]
            assert rgb_hash not in decoded_hashes, "Duplicate decoded RGB inside selected images"
            decoded_hashes.add(rgb_hash)
            files.append({"path": str(path), "filename": path.name, "sha256": digest,
                          "decodedOrientedRgbSha256": rgb_hash, "sizeBytes": path.stat().st_size,
                          "rawSize": raw_size, "orientedSize": oriented_size, "decodeStatus": "ok",
                          "reverifiedAgainstInventory": True})
        same_sizes = len({tuple(f["orientedSize"]) for f in files}) == 1
        case = {"caseId": case_id, "label": label, "inventoryIndex": index,
                "group": group["relativeGroupPath"], "relativeGroupPath": group["relativeGroupPath"],
                "sourceDirectory": group["path"], "sourceImageCount": group["fileCount"],
                "paths": [f["path"] for f in files], "inputOrder": [f["filename"] for f in files],
                "sourceHashes": [f["sha256"] for f in files], "files": files,
                "orientedSizes": [f["orientedSize"] for f in files],
                "allOrientedSizesIdentical": same_sizes, "sharedCropEligible": same_sizes,
                "sharedCropSkipReason": None if same_sizes else "Oriented image sizes differ",
                "referenceViewIndex": 0, "referenceViewSelection": "visuallyMostFrontalOfSelectedFour",
                "selectionStratum": stratum, "notes": [evidence, note],
                "captureModeHypothesis": "batchReportedPredominantlyHeadTurning",
                "captureModeEvidence": {"userStatement": "用户说明此前主实验女性为相机环绕，新这批基本都是扭头。",
                                        "visualObservation": evidence},
                "captureModeConfidence": "batchLevelOnly; noPerCaseCaptureLabel; someCameraMotionPossible",
                "exactOverlapWithPreviousSelectedFiles": False,
                "exactOverlapWithDevelopmentSelectedFiles": False,
                "exactOverlapWithKnownBaseline": False,
                "visualReview": {"allFourReviewed": True, "prominentAdditionalFaceSeen": False,
                                 "frontAndBothYawDirectionsPresent": True, "upwardViewPresent": True}}
        selected.append(case)
        contact([case], OUT / f"{case_id}.jpg")
    for page in range(0, len(selected), 3):
        contact(selected[page:page + 3], OUT / f"selectedFourViews{page // 3 + 1}.jpg")
    contact(selected, OUT / "selectedOverview.jpg", width=1000, row_height=270)
    result = {
        "schemaVersion": 1, "createdDate": "2026-10-12", "status": "frozenBeforeReconstruction",
        "sourceRoot": inventory["sourceRoot"], "inventoryPath": str(INVENTORY),
        "inventorySha256": sha256(INVENTORY), "priorSelectionPath": str(PRIOR),
        "priorSelectionSha256": sha256(PRIOR), "developmentSelectionPath": str(DEVELOPMENT),
        "developmentSelectionSha256": sha256(DEVELOPMENT),
        "selectionScriptSha256": sha256(Path(__file__)),
        "selectionMethod": "Purposeful photograph-only selection, front thumbnails of all 136 eligible unique four-file sets reviewed, then all four photos of nine candidates inspected. No reconstruction quality inspected.",
        "captureModePolicy": "User's head-turn label is batch-level, not verified per case; background/shoulder consistency supports but does not establish a stationary camera. Some hand-held motion is possible.",
        "identityNote": "Six file collections, not six independently verified participants. Exact-file exclusion and internal decoded-RGB deduplication only; no biometric identity or transformed-copy matching.",
        "statisticalScope": "Purposive convenience sample from the same source collection. File-disjoint validation of a frozen rule, not random sampling or a subject-independent external test.",
        "protocolRole": "Held-out file groups for validation; prior ten-group study supplies rule development. Threshold selection must not depend on these six groups' reconstruction quality.",
        "inventorySummary": {"groupCount": inventory["groupCount"], "fileCount": inventory["fileCount"],
                             "eligibleUniqueFourFileSets": len(eligible),
                             "eligibleDirectoryDistribution": dict(Counter(g["relativeGroupPath"].split("/")[0] for g in eligible))},
        "selectedCaseCount": len(selected), "uniqueSelectedFileCount": len(fresh_hashes),
        "uniqueDecodedOrientedRgbCount": len(decoded_hashes),
        "sharedSelectedFileCountWithPreviousCases": 0,
        "sharedSelectedFileCountWithDevelopmentCases": 0,
        "sharedSelectedFileCountWithKnownBaselines": 0,
        "allSelectedSourcesRehashedAndStrictlyDecoded": True,
        "frontThumbnailReviewedCandidateCount": len(eligible),
        "fourViewReviewedCandidateIndices": [13, 39, 40, 65, 70, 95, 128, 152, 181],
        "exclusions": exclusions, "cases": selected,
        "contactSheets": [str(OUT / "selectedFourViews1.jpg"), str(OUT / "selectedFourViews2.jpg")],
    }
    (OUT / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selection": str(OUT / "selection.json"), "caseIds": [c["caseId"] for c in selected],
                      "caseCount": len(selected), "imageCount": len(fresh_hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
