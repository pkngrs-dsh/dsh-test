"""Small reproducible figure summarizing the measured background ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ("backgroundGray", "headGray", "headBlack")
LABELS = ("B 仅去环境背景 · 灰底", "C 保留头部 · 灰底", "D 保留头部 · 黑底")
NAMES = {"warmRoom4": "室内暖色", "outdoorSideLight4": "室外侧光", "cyanWall4": "青墙",
         "backlitCap4": "逆光帽子", "highPitch4": "大俯仰", "orbitControl4": "环绕对照"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012")
    args = parser.parse_args()
    path = args.root / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["status"] != "complete" or len(data["cases"]) != 6:
        raise ValueError("Expected six completed cases")
    rows = []
    for case in data["cases"]:
        variants = {v["id"]: v for v in case["variants"]}
        eyes = case["frontalEyes"]["modes"]["fixedOriginalCamera"]["variants"]
        originalEye = eyes["original"]["commonVisibleContourErrorPixels"]
        row = {"caseId": case["caseId"], "label": NAMES[case["caseId"]],
               "commonEyeSampleCount": originalEye["count"], "originalEyeMeanPixels": originalEye["mean"],
               "repeatRmsPercent": variants["originalRepeat"]["geometryChangeFromOriginal"]["displacementOverOriginalBboxDiagonal"]["rms"] * 100,
               "variants": {}}
        for name in IDS:
            variant = variants[name]
            change = variant["geometryChangeFromOriginal"]
            eye = eyes[name]["commonVisibleContourErrorPixels"]
            row["variants"][name] = {
                "rawRmsPercent": change["displacementOverOriginalBboxDiagonal"]["rms"] * 100,
                "alignedRmsPercent": change["alignedResidualOverOriginalBboxDiagonal"]["rms"] * 100,
                "effectOverRepeat": variant["observedRepeatComparison"]["effectOverRepeatRms"],
                "eyeMeanPixels": eye["mean"], "eyeDeltaPixels": eye["mean"] - originalEye["mean"],
                "maximumAbsoluteFocalChangePercent": max(abs(x) for cam in variant["cameras"] for x in cam["focalFractionChange"]) * 100}
        rows.append(row)
    report = {"metricsSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "normalization": "fresh A final mesh bounding-box diagonal", "cases": rows,
              "interpretation": "Output sensitivity confirmed; eye proxy changes are mixed, not a 3D accuracy measurement."}
    (args.root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4), gridspec_kw={"width_ratios": [1.06, 1]}, facecolor="#f8f9fc")
    colors = ("#377a9f", "#d28b35", "#557f4b")
    y = np.arange(len(rows))
    for ax in axes:
        ax.set_facecolor("#f8f9fc")
        ax.set_yticks(y, [r["label"] for r in rows])
        ax.set_ylim(5.6, -.6)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    for i, (name, label, color) in enumerate(zip(IDS, LABELS, colors)):
        yy = y + (i - 1) * .23
        values = [r["variants"][name]["rawRmsPercent"] for r in rows]
        axes[0].barh(yy, values, height=.20, color=color, label=label)
        for yy0, value in zip(yy, values):
            axes[0].text(value + .015, yy0, f"{value:.2f}", va="center", fontsize=9, color=color)
        deltas = [r["variants"][name]["eyeDeltaPixels"] for r in rows]
        axes[1].plot(deltas, yy, "o", color=color, markersize=6)
        for yy0, value in zip(yy, deltas):
            axes[1].text(value + (.05 if value >= 0 else -.05), yy0, f"{value:+.2f}",
                         va="center", ha="left" if value >= 0 else "right", fontsize=9, color=color)
    axes[0].set_xlim(0, 1.22)
    axes[0].set_xlabel("网格顶点 RMS 变化 / A 脸框对角线（%）")
    axes[0].set_title("几何变化远高于重复运行差异", loc="left", fontsize=14, pad=15)
    axes[1].axvline(0, color="#8e98a5", lw=1)
    axes[1].set_xlim(-2.25, 1.35)
    axes[1].set_xlabel("固定 A 相机：眼睑平均误差变化（512 图像像素）")
    axes[1].set_title("眼睑代理：左侧改善，右侧退化", loc="left", fontsize=14, pad=15)
    fig.suptitle("背景引起明确的几何变化，眼睑代理改善不一致", x=.035, ha="left", fontsize=19, y=.985)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
    fig.text(.035, .067, "控制：6 组 × 4 张照片；冻结裁剪、UV、面部采样掩码。A / B / C / D + A′ 重复，共 30 次重建。", fontsize=10, color="#445366")
    fig.text(.035, .030, "A′ 网格 RMS：0.00055%–0.00065%。眼睑来自固定绑定与自动检测，各组共同可见样本 32–52；无真实 3D 标注。", fontsize=10, color="#445366")
    fig.subplots_adjust(top=.81, bottom=.19, left=.095, right=.98, wspace=.27)
    fig.savefig(args.root / "metricsOverview.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.root / "metricsOverview.png")


if __name__ == "__main__":
    main()
