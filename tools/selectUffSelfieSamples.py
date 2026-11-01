#!/usr/bin/env python3
"""Freeze a visually reviewed, exact-file-disjoint selfie crop study sample.

Selection is made from photographs only, before observing reconstruction outputs.
This script is CPU-only and never modifies source photographs or older inventories.
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
OUT = ROOT / "runs/uffSelfieCropStudy20261012/selection"

# Every selected photograph was visually reviewed in the four-view contact sheets.
# These indices refer to the immutable source inventory, not an identity label.
SPECS = [
    (15, "lowResSweater4", "浅墙低清 · 毛衣", "lowResolution",
     "左右转头和抬头可见；背景墙图案及衣领变化较小，伴少量构图变化。",
     "354×476 左右的原始低清照片；细节损失单独标记，不能混同高分辨率准确性。"),
    (42, "closeupLongHair4", "近距离 · 长发", "closeup",
     "左右转头及抬头；肩部变化较小，画面非常贴近面部，存在少量手持位移。",
     "头部接近画面边缘，长发和项链邻近边界；表情有轻微微笑变化。"),
    (68, "ceilingBeard4", "吊顶背景 · 胡须", "detailedBackground",
     "吊顶、窗框与肩部较稳定，左右转头及抬头明确。",
     "2160×2880 高分辨率，背景有规则纹理及设备；额头高光，短胡须。"),
    (85, "woodSmallFace4", "木墙 · 小脸占比", "smallFaceOccupancy",
     "木墙和肩部稳定，左右转头及抬头；相机有轻微构图变化。",
     "原图上方背景及肩部较多，适合观察紧裁剪和宽裁剪差异；侧面眼裂较窄。"),
    (117, "woodBangs4", "木墙 · 刘海", "hairBoundary",
     "木墙和肩部较稳定，左右转头及抬头明确；部分侧视有轻微模糊。",
     "刘海覆盖部分额头，头发与浅色衣服边界清楚；抬头图视线有变化。"),
    (131, "windowChain4", "窗边侧光 · 吊顶", "directionalLight",
     "吊顶与椅背位置总体稳定，头部左右转动及明显抬头。",
     "面颊受方向光影响，黑色上衣和项链靠近颈部边界；抬头幅度较大。"),
    (153, "warmWoodTexture4", "暖色木墙 · 肤色纹理", "warmLight",
     "木墙与肩部总体稳定，左右转头和抬头可见。",
     "暖色光、面部颜色变化与胡须纹理明显；不是均匀肤色样本。"),
    (194, "windowCurly4", "窗后背景 · 卷发", "backWindow",
     "窗框与肩部大体稳定，左右转头及抬头；仍有手持构图差异。",
     "768×1024 输入，蓬松头发靠近图像边缘；窗后树木具有较多细节。"),
    (170, "dimCeiling4", "暗正面 · 转头光变", "unevenExposure",
     "吊顶和椅背较稳定，左右转头及较大抬头明确。",
     "正面偏暗，侧转与抬头时曝光/光照观感变亮；用于检查困难自拍情况。"),
    (134, "backwardCap4", "反戴帽 · 顶光", "headAccessory",
     "吊顶灯、显示器和肩部较稳定，左右转头及抬头可见。",
     "反戴帽改变头部轮廓，顶灯高亮；帽子及耳部应与环境背景分开理解。"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standard_four(group: dict) -> list[dict]:
    return [next(f for f in group["files"] if Path(f["filename"]).stem == str(i))
            for i in range(1, 5)]


def contact(cases: list[dict], output: Path, width: int = 1200, row_height: int = 330) -> None:
    image = Image.new("RGB", (width, row_height * len(cases)), "#eeeeee")
    draw = ImageDraw.Draw(image)
    for row, case in enumerate(cases):
        draw.text((8, row * row_height + 4),
                  f"{case['caseId']} | {case['relativeGroupPath']} | 1,2,3,4", fill="black")
        for col, path in enumerate(case["paths"]):
            with Image.open(path) as source:
                photo = ImageOps.exif_transpose(source).convert("RGB")
            photo.thumbnail((width // 4 - 8, row_height - 30), Image.Resampling.LANCZOS)
            x = col * width // 4 + (width // 4 - photo.width) // 2
            image.paste(photo, (x, row * row_height + 26))
    image.save(output, quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(INVENTORY.read_text())
    previous = json.loads(PRIOR.read_text())
    previous_hashes = {f["sha256"] for c in previous["cases"] for f in c["files"]}
    eligible = []
    exclusions = []
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
        if reason:
            exclusions.append({"inventoryIndex": group["index"], "group": group["relativeGroupPath"],
                               "reason": reason, "stage": "inventoryFilter"})
        else:
            eligible.append(group)

    manual_exclusions = [
        {"inventoryIndex": 182, "reason": "背景中可见旁人，四视图有多脸检测歧义；优先选择无明显旁人的组。",
         "stage": "fourViewVisualReview", "status": "excludedForThisStudy"},
        {"inventoryIndex": 160, "reason": "正面候选缩略图严重欠曝；未进入四视图精审。",
         "stage": "frontThumbnailReview", "status": "excludedForThisStudy"},
        {"inventoryIndex": 178, "reason": "正面候选缩略图严重欠曝；未进入四视图精审。",
         "stage": "frontThumbnailReview", "status": "excludedForThisStudy"},
        {"inventoryIndex": 154, "reason": "正面候选呈现棚拍肖像样式，采集协议来源不清；未进入四视图精审。",
         "stage": "frontThumbnailReview", "status": "excludedForThisStudy"},
    ]
    for item in manual_exclusions:
        item["group"] = inventory["groups"][item["inventoryIndex"]]["relativeGroupPath"]
    exclusions.extend(manual_exclusions)
    eligible_indices = {g["index"] for g in eligible}
    selected = []
    fresh_hashes = set()
    for index, case_id, label, stratum, evidence, note in SPECS:
        assert index in eligible_indices, (index, "ineligible")
        group = inventory["groups"][index]
        files = []
        for prior in standard_four(group):
            path = Path(prior["path"])
            digest = sha256(path)
            assert digest == prior["sha256"], f"Source changed since inventory: {path}"
            assert digest not in previous_hashes and digest not in fresh_hashes
            fresh_hashes.add(digest)
            with Image.open(path) as source:
                source.load()
                raw_size = list(source.size)
                photo = ImageOps.exif_transpose(source).convert("RGB")
                oriented_size = list(photo.size)
            assert oriented_size == prior["orientedSize"]
            files.append({"path": str(path), "filename": path.name, "sha256": digest,
                          "sizeBytes": path.stat().st_size, "rawSize": raw_size,
                          "orientedSize": oriented_size, "decodeStatus": "ok",
                          "reverifiedAgainstInventory": True})
        same_sizes = len({tuple(f["orientedSize"]) for f in files}) == 1
        case = {"caseId": case_id, "label": label, "inventoryIndex": index,
                "group": group["relativeGroupPath"], "relativeGroupPath": group["relativeGroupPath"],
                "sourceDirectory": group["path"], "sourceImageCount": group["fileCount"],
                "paths": [f["path"] for f in files], "inputOrder": [f["filename"] for f in files],
                "sourceHashes": [f["sha256"] for f in files], "files": files,
                "orientedSizes": [f["orientedSize"] for f in files],
                "allOrientedSizesIdentical": same_sizes, "sharedCropEligible": same_sizes,
                "sharedCropSkipReason": None if same_sizes else "Oriented source image dimensions differ across views; a common pixel crop is undefined without another resampling convention.",
                "referenceViewIndex": 0, "referenceViewSelection": "visuallyMostFrontalOfSelectedFour",
                "selectionStratum": stratum, "notes": [evidence, note],
                "captureModeHypothesis": "batchReportedPredominantlyHeadTurning",
                "captureModeEvidence": {"userStatement": "用户说明此前主实验女性为相机环绕，新这批基本都是扭头。",
                                        "visualObservation": evidence},
                "captureModeConfidence": "batchLevelOnly; noPerCaseCaptureLabel; someCameraMotionPossible",
                "exactOverlapWithPreviousSelectedFiles": False,
                "exactOverlapWithKnownBaseline": False,
                "visualReview": {"allFourReviewed": True, "prominentAdditionalFaceSeen": False,
                                 "frontAndBothYawDirectionsPresent": True, "upwardViewPresent": True}}
        selected.append(case)
        contact([case], OUT / f"{case_id}.jpg", width=1200, row_height=430)
    for page in range(0, len(selected), 5):
        contact(selected[page:page + 5], OUT / f"selectedFourViews{page // 5 + 1}.jpg")
    contact(selected, OUT / "selectedOverview.jpg", width=880, row_height=230)
    result = {
        "schemaVersion": 1, "createdDate": "2026-10-12", "status": "frozenBeforeReconstruction",
        "sourceRoot": inventory["sourceRoot"], "inventoryPath": str(INVENTORY),
        "inventorySha256": sha256(INVENTORY), "priorSelectionPath": str(PRIOR),
        "priorSelectionSha256": sha256(PRIOR), "selectionScriptSha256": sha256(Path(__file__)),
        "selectionMethod": "Purposeful coverage of image size, face occupancy, hair/accessories, lighting and background; all four selected views inspected before reconstruction.",
        "captureModePolicy": "Focus on visually head-turn-consistent inputs; user label is batch-level, not verified per case. Do not infer a strictly stationary camera from these stills.",
        "identityNote": "Ten file collections, not ten independently verified participants; no biometric identity matching performed.",
        "statisticalScope": "Purposive convenience sample, not a random population sample. Low-resolution challenge is labeled separately.",
        "inventorySummary": {"groupCount": inventory["groupCount"], "fileCount": inventory["fileCount"],
                             "eligibleGroupCount": len(eligible),
                             "eligibleUniqueFourFileSets": len({tuple(sorted(f["sha256"] for f in standard_four(g))) for g in eligible}),
                             "eligibleDirectoryDistribution": dict(Counter(g["relativeGroupPath"].split("/")[0] for g in eligible))},
        "selectedCaseCount": len(selected), "uniqueSelectedFileCount": len(fresh_hashes),
        "sharedSelectedFileCountWithPreviousCases": 0,
        "allSelectedSourcesRehashedAndStrictlyDecoded": True,
        "fourViewReviewedCandidateIndices": [15, 42, 68, 85, 117, 131, 153, 182, 194, 170, 134, 1],
        "reviewedNotSelected": [{"inventoryIndex": 1, "reason": "可作为后续备用；本轮限制为10组，木墙/刘海条件已有覆盖。"}],
        "exclusions": exclusions, "cases": selected,
        "contactSheets": [str(OUT / "selectedFourViews1.jpg"), str(OUT / "selectedFourViews2.jpg")],
    }
    (OUT / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selection": str(OUT / "selection.json"), "caseIds": [c["caseId"] for c in selected],
                      "caseCount": len(selected), "imageCount": len(fresh_hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
