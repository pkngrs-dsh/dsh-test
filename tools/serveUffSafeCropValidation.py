"""Read-only localhost viewer for the frozen safe-crop validation (port 18092)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import renderUffSelfieCrop as renderer
import serveUffSelfieCrop as server
from evaluateUffSafeCropValidation import BASELINE, CANDIDATE, GUARDED, DEFAULT_ROOT, decisionFor, read

DISPLAY = (BASELINE, CANDIDATE, GUARDED)
LABELS = {BASELINE: "基线 · 1.35 原背景", CANDIDATE: "直接紧裁 · 1.10 头部灰底", GUARDED: "带保护的整组选择"}


def substitute(text, old, new):
    if old not in text:
        raise ValueError(f"Inherited viewer changed; replacement missing: {old[:70]}")
    return text.replace(old, new)


def policyText(decision):
    if decision.get("status") != "available":
        return "策略尚未决定"
    selected = decision["selectedVariantId"]
    text = "本组选用：" + ("1.10 紧裁＋头部灰底。" if selected == CANDIDATE else "回退到 1.35 原背景。")
    if decision.get("retakeRequired"):
        retakeReasons = []
        for view in decision.get("perViewChecks", []):
            if view.get("retakeRequired"):
                retakeReasons.extend(f"视图 {view['viewIndex']}：{reasonText(reason)}" for reason in view.get("retakeOrReviewReasons", []))
        text += "建议调整取景后再拍。" + ("依据：" + "；".join(retakeReasons) + "。" if retakeReasons else "")
    elif decision.get("manualReviewRequired"):
        text += "自动判断的依据不足，需要人工看图；侧脸遮住远侧眼睛也可能触发保护。"
    reasons = decision.get("fallbackReasons", decision.get("reasons", []))
    if reasons:
        friendly = []
        for reason in reasons:
            if isinstance(reason, dict):
                descriptions = reason.get("reasons", [reason.get("message", reason.get("reason", reason.get("code", "输入覆盖不足")))])
                description = "、".join(reasonText(value) for value in descriptions)
                view = reason.get("viewIndex")
                friendly.append((f"视图 {view}：" if view is not None else "") + str(description))
            else:
                friendly.append(reasonText(reason))
        text += " 保护原因：" + "；".join(friendly)
    else:
        text += "四张照片全部通过输入覆盖保护。"
    return text


def reasonText(reason):
    value = str(reason)
    locations = {"forehead": "额头", "chin": "下巴", "left": "左脸", "right": "右脸"}
    if value.startswith("mainFaceInsufficientClearance:"):
        return locations.get(value.rsplit(":", 1)[1], "脸部") + "距裁剪边界太近"
    if value.startswith("organInsufficientClearance:"):
        return "五官距裁剪边界太近"
    if value == "excessiveCopiedEdgePadding":
        return "裁剪图中复制边缘占比过多"
    if value == "mainFaceNearOriginalPhotoBoundary":
        return "脸部距原照片边界太近"
    if value == "detectorOrLandmarksOutsideOriginalPhoto":
        return "检测到的脸部超出原照片"
    if "missingReliableOrgan" in value:
        return "部分五官无法可靠辨认"
    if value.startswith(("inputOrSemanticFailure:", "missingOrInvalidInputEvidence:")):
        return "自动识别的脸部信息不可靠"
    if value.startswith("fallbackStillUnsafe:"):
        if value == "fallbackStillUnsafe:excessiveCopiedEdgePadding":
            return "放宽裁剪后的复制边缘仍超过门槛"
        return "较宽裁剪仍未通过：" + reasonText(value.split(":", 1)[1])
    return "输入覆盖检查未通过"


def createApp(root):
    root = Path(root).resolve()
    decisions = {case["caseId"]: decisionFor(case, root) for case in read(root / "cropManifest.json")["cases"]}
    labels = {caseId: policyText(decision) for caseId, decision in decisions.items()}
    renderer.VARIANTS = DISPLAY
    renderer.GROUPS = {"validation": DISPLAY}
    renderer.LABELS = LABELS
    server.VARIANTS = DISPLAY
    server.GROUPS = renderer.GROUPS
    server.LABELS = LABELS
    html = server.HTML
    replacements = [
        ("<title>自拍裁剪 · UFF 对照</title>", "<title>安全紧裁验证 · UFF 对照</title>"),
        ("FaceAppearance · 自拍裁剪实验", "FaceAppearance · 安全紧裁验证"),
        ("哪种裁剪更适合扭头自拍", "紧裁加保护，是否更稳妥"),
        ("三个裁剪范围 × 三种背景处理，另对照整段固定裁剪框", "新增拍摄组：基线、直接紧裁、带额头和下巴保护的整组选择"),
        ('<p class="note" id="status"></p>', '<p class="note" id="status"></p><p class="note" id="policy"></p>'),
        ("variant=baseline,mode='clay'", "variant='guarded_selected',mode='clay'"),
        ("pairB='m135_headGray'", "pairB='guarded_selected'"),
        ("q.get('variant')||baseline", "q.get('variant')||'guarded_selected'"),
        ("const $=s=>document.querySelector(s)", "const policyDescriptions="+json.dumps(labels, ensure_ascii=False).replace("<", "\\u003c")+";\nconst $=s=>document.querySelector(s)"),
        ("$('#details').textContent='固定原照坐标，避免不同裁剪框的 512 像素相机混用。';", "$('#policy').textContent=policyDescriptions[id]||'';$('#details').textContent='保护规则只依据四帧输入决定；选择结果直接复用基线或紧裁几何。';"),
        ("每行依次是原背景、去环境背景灰底、仅保留头部灰底。前三行为逐帧框 1.10 / 1.35 / 1.65；最后一行是整段固定框。输入灰底预览为 PNG，模型实际使用 float32 0.5。", "依次是 1.35 原背景、直接采用 1.10 头部灰底、实际选用方案。整组四帧使用同一方案。灰底预览为 PNG，模型使用 float32 0.5。"),
        ("每行是同一裁剪范围：原照、原背景、去环境背景灰底、头部灰底。最后一行为整段固定框的两个版本。基准是逐帧框 1.35 · 原背景。", "从左到右：原照、1.35 原背景基线、直接采用 1.10 头部灰底、保护规则选择的结果。带保护的一列与所选版本完全相同。"),
    ]
    for old, new in replacements:
        html = substitute(html, old, new)
    server.HTML = html
    app = server.createApp(root)
    app.title = "UFF safe crop validation"

    @app.get("/policy")
    def policy():
        return decisions

    @app.get("/validation")
    def validation():
        path = root / "validationMetrics.json"
        return read(path) if path.is_file() else {"status": "pending"}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    server.uvicorn.run(createApp(args.root), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
