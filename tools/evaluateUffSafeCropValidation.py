"""Evaluate a frozen input-only crop guard on new capture groups.

Actual reconstructions are baseline, unconditionally tight/head-gray, and a
baseline repeat. The guarded policy only selects an existing result. Pairwise
eye support differs from the earlier nine-variant study and is not comparable
to its absolute scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.artifacts import writeJson
import diagnoseUffSelfieCrop as diagnosis

BASELINE = "m135_raw"
CANDIDATE = "m110_headGray"
GUARDED = "guarded_selected"
REPEAT = "m135_repeat"
DEFAULT_ROOT = ROOT / "runs/uffSelfieSafeCropValidation20261012"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decisionFor(case, root):
    path = Path(case.get("policyDecisionPath", root / "cases" / case["caseId"] / "policyDecision.json"))
    if not path.is_absolute():
        path = root / path
    decision = read(path) if path.is_file() else case.get("policyDecision")
    if not decision:
        return {"status": "unavailable", "reason": "policyDecisionMissing"}
    if path.is_file() and case.get("policyDecisionSha256") and diagnosis.fileHash(path) != case["policyDecisionSha256"]:
        raise ValueError(f"Saved policy decision changed: {case['caseId']}")
    if path.is_file() and case.get("policyDecision") and decision != case["policyDecision"]:
        raise ValueError(f"Manifest and saved policy decisions differ: {case['caseId']}")
    selected = decision.get("selectedVariantId", decision.get("selectedVariant"))
    if selected not in (BASELINE, CANDIDATE):
        raise ValueError(f"Unknown selected variant for {case['caseId']}: {selected}")
    accepted = decision.get("accepted", selected == CANDIDATE)
    if bool(accepted) != (selected == CANDIDATE):
        raise ValueError("Accepted guard and selected reconstruction disagree")
    return {"status": "available", **decision, "selectedVariantId": selected,
            "accepted": bool(accepted), "decisionPath": str(path),
            "decisionSha256": diagnosis.fileHash(path) if path.is_file() else None}


def diagnose(root, manifest, overwrite):
    output = root / "pairMetrics.json"
    if output.exists() and not overwrite:
        raise FileExistsError("pairMetrics.json already exists; use --overwrite for analysis only")
    diagnosis.CORE = [BASELINE, CANDIDATE]
    report = {"schemaVersion": 1, "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(root / "cropManifest.json"),
        "sourceManifestSha256": diagnosis.fileHash(root / "cropManifest.json"),
        "implementationHashes": {name: diagnosis.fileHash(ROOT / name) for name in (
            "tools/evaluateUffSafeCropValidation.py", "tools/diagnoseUffSelfieCrop.py",
            "tools/diagnoseUffBackgroundAblation.py", "tools/renderCropComparison.py",
            "tools/runEyeRefinementExperiment.py", "src/faceAppearance/eyeContours.py",
            "src/faceAppearance/semanticProjection.py", "src/faceAppearance/surfaceVisibility.py")},
        "supportVariantIds": [BASELINE, CANDIDATE], "cases": [], "notGroundTruth": True,
        "definitions": {"coreSupport": "intersection of baseline and unconditional tight/head-gray candidate only",
            "allCompletedSupport": "intersection of those two plus the repeated baseline",
            "primary": "front-view common-visible upper/lower eye contour error / original-photo eye width; fixed fresh baseline original-photo camera and binding",
            "support": "two reliable eyes, >=3 common visible samples on each upper/lower contour, >=16 total contour samples",
            "repeat": "one baseline repeat estimates observed numerical variation only",
            "guarded": "derived by frozen input-only routing, not separately inferred or used to define support"},
        "limits": ["Pairwise scores cannot be compared numerically to earlier nine-variant common-support scores.",
            "No calibrated camera, held-out photo, independent 3D ground truth, or verified independent identities.",
            "Baseline camera/binding may favor baseline; own PnP fit quality is not used to rank reconstructions.",
            "Crop guard only addresses input truncation/padding/segmentation; it cannot certify geometric improvement."]}
    begin = perf_counter()
    for originalCase in manifest["cases"]:
        case = {**originalCase, "variants": [v for v in originalCase["variants"] if v["id"] in (BASELINE, CANDIDATE, REPEAT)]}
        print(f"ANALYZE {case['caseId']}", flush=True)
        try:
            row = diagnosis.diagnoseCase(case, root, root / "pairAnalysis" / case["caseId"], int(manifest.get("seed", 2026)))
        except Exception as error:
            row = {"caseId": case["caseId"], "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        report["cases"].append(row)
        writeJson(output, report)
        print(f"ANALYZED {case['caseId']} {row['status']}", flush=True)
    report["summary"] = diagnosis.aggregate(report["cases"])
    # Remove irrelevant old-study factor contrasts, keeping pair scopes only.
    report["summary"].pop("pairedFactorContrasts", None)
    report["status"] = "complete" if all(c["status"] == "measured" and all(v["status"] == "measured" for v in c["variants"]) for c in report["cases"]) else "partial"
    report["wallSeconds"] = perf_counter() - begin
    report["completedAt"] = datetime.now(timezone.utc).isoformat()
    writeJson(output, report)
    return report


def runApertures(root):
    import auditUffSelfieEyeApertures as audit
    audit.VARIANTS = (BASELINE, CANDIDATE)
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / "tools/auditUffSelfieEyeApertures.py"), "--root", str(root)]
        audit.main()
    finally:
        sys.argv = previous
    report = read(root / "apertureAudit.json")
    report["wrapperSha256"] = diagnosis.fileHash(__file__)
    report["supportVariantIds"] = [BASELINE, CANDIDATE]
    report["guardedPolicyMeasuredSeparately"] = False
    writeJson(root / "apertureAudit.json", report)
    return report


def scoresForPrimary(case):
    ranking = case.get("proxyRankings", {}).get("coreSupport", {})
    if ranking.get("status") != "ranked":
        return {"status": "unavailable", "reason": ranking.get("reason", case.get("eyes", {}).get("reason", "pairMetricsMissing"))}
    scores = {v["variantId"]: v["score"] for v in ranking["ranking"]}
    if not all(key in scores for key in (BASELINE, CANDIDATE)):
        return {"status": "unavailable", "reason": "pairedReconstructionMissing"}
    eyes = case["eyes"]
    front = eyes["views"][eyes["referenceViewIndex"]]["coreSupport"]
    return {"status": "measured", "scores": {key: scores[key] for key in (BASELINE, CANDIDATE)},
            "commonVisibleContourCount": front["variants"][BASELINE]["commonVisibleContourErrorPixels"]["count"]}


def scoresForAperture(case):
    if case.get("status") != "measured" or not case.get("fixedCameraAccepted"):
        return {"status": "unavailable", "reason": case.get("reason", "missingApertureOrAcceptedCamera")}
    values = {v["id"]: v for v in case["variants"]}
    if not all(key in values for key in (BASELINE, CANDIDATE)):
        return {"status": "unavailable", "reason": "pairedReconstructionMissing"}
    return {"status": "measured", "scores": {key: values[key]["meanEyeIoU"] for key in (BASELINE, CANDIDATE)},
            "meanAreaRatios": {key: values[key]["meanAreaRatio"] for key in (BASELINE, CANDIDATE)},
            "unmatchedHoleCounts": {key: sum(not e["matchedRenderedHole"] for e in values[key]["eyes"]) for key in (BASELINE, CANDIDATE)}}


def enrichMetric(metric, selected, lowerBetter):
    if metric.get("status") != "measured":
        return metric
    scores = {**metric["scores"], GUARDED: metric["scores"][selected]}
    baseline = scores[BASELINE]
    return {**metric, "scores": scores, "lowerIsBetter": lowerBetter,
        "deltaFromBaseline": {key: value-baseline for key, value in scores.items()},
        "relativeChangeFromBaseline": {key: value/baseline-1 if baseline > 0 else None for key, value in scores.items()},
        "guardedMinusUnconditionalCandidate": scores[GUARDED]-scores[CANDIDATE]}


def aggregateMetric(cases, metricName, lowerBetter):
    eligible = [c for c in cases if c[metricName].get("status") == "measured"]
    sign = -1 if lowerBetter else 1
    variants = {}
    for variant in (BASELINE, CANDIDATE, GUARDED):
        rows = [{"caseId": c["caseId"], "score": c[metricName]["scores"][variant],
                 "deltaFromBaseline": c[metricName]["deltaFromBaseline"][variant],
                 "relativeChangeFromBaseline": c[metricName]["relativeChangeFromBaseline"][variant]} for c in eligible]
        variants[variant] = {"caseCount": len(rows), "caseScoreStats": diagnosis.stats([v["score"] for v in rows]),
            "pairedDeltaStats": diagnosis.stats([v["deltaFromBaseline"] for v in rows]),
            "pairedRelativeChangeStats": diagnosis.stats([v["relativeChangeFromBaseline"] for v in rows if v["relativeChangeFromBaseline"] is not None]),
            "betterCaseIds": [v["caseId"] for v in rows if sign*v["deltaFromBaseline"] > 1e-8],
            "worseCaseIds": [v["caseId"] for v in rows if sign*v["deltaFromBaseline"] < -1e-8],
            "tiedCaseIds": [v["caseId"] for v in rows if abs(v["deltaFromBaseline"]) <= 1e-8], "cases": rows}
    rejected = [c for c in eligible if c["selectedVariantId"] == BASELINE]
    return {"eligibleCaseCount": len(eligible), "eligibleCaseIds": [c["caseId"] for c in eligible],
        "unavailableCases": [{"caseId": c["caseId"], "reason": c[metricName].get("reason")} for c in cases if c not in eligible],
        "lowerIsBetter": lowerBetter, "caseWeighting": "equal weight per capture group, not per point or photo",
        "variants": variants,
        "fallbackCounterfactual": {"measurableRejectedCaseCount": len(rejected),
            "avoidedWorseProxyCaseIds": [c["caseId"] for c in rejected if sign*c[metricName]["deltaFromBaseline"][CANDIDATE] < -1e-8],
            "lostBetterProxyCaseIds": [c["caseId"] for c in rejected if sign*c[metricName]["deltaFromBaseline"][CANDIDATE] > 1e-8],
            "guardedMinusUnconditionalCandidateStats": diagnosis.stats([c[metricName]["guardedMinusUnconditionalCandidate"] for c in eligible])}}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = report["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .6*len(cases)+1.8)), sharey=True)
    y = np.arange(len(cases))
    colors = {CANDIDATE: "#b2783b", GUARDED: "#2a8079"}
    for axis, name, scale, title in zip(axes, ("eyeContour", "eyeAperture"), (100, 100),
            ("Eye contour error change (%) | lower is better", "Eye aperture IoU change (points) | higher is better")):
        for variant, offset in ((CANDIDATE, -.16), (GUARDED, .16)):
            values = []
            for case in cases:
                value = None
                if case[name].get("status") == "measured":
                    value = case[name]["relativeChangeFromBaseline"][variant] if name == "eyeContour" else case[name]["deltaFromBaseline"][variant]
                values.append(value*scale if value is not None else np.nan)
            axis.barh(y+offset, values, height=.29, color=colors[variant], label="Unconditional tight/head-gray" if variant == CANDIDATE else "Guarded selection")
            for yi, value in zip(y+offset, values):
                if np.isfinite(value):
                    axis.annotate(f"{value:+.1f}", (value, yi), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=8)
                else:
                    axis.text(.5, yi, "unavailable", transform=axis.get_yaxis_transform(), va="center", ha="center", fontsize=8, color="#737373")
        axis.axvline(0, color="#9b9b9b", linewidth=.8)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=.15)
        axis.margins(x=.28)
    axes[0].set_yticks(y, [c["caseId"]+(" [tight]" if c.get("selectedVariantId") == CANDIDATE else " [fallback]") for c in cases], fontsize=9)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("New capture groups | frozen input-only crop guard | baseline: 1.35 original", fontsize=12)
    fig.text(.5, .012, "Fixed baseline photo camera; automatic 2D proxies, no 3D ground truth. Missing contour support remains unavailable.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, .96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(root, manifest):
    metricsPath, aperturePath = root / "pairMetrics.json", root / "apertureAudit.json"
    metrics = read(metricsPath) if metricsPath.exists() else {}
    apertures = read(aperturePath) if aperturePath.exists() else {}
    byMetric = {c["caseId"]: c for c in metrics.get("cases", [])}
    byAperture = {c["caseId"]: c for c in apertures.get("cases", [])}
    cases = []
    for case in manifest["cases"]:
        decision = decisionFor(case, root)
        row = {"caseId": case["caseId"], "label": case.get("label", case["caseId"]), "decision": decision,
               "selectedVariantId": decision.get("selectedVariantId"), "retakeRequired": decision.get("retakeRequired", False),
               "manualReviewRequired": decision.get("manualReviewRequired", False)}
        primary = scoresForPrimary(byMetric.get(case["caseId"], {}))
        secondary = scoresForAperture(byAperture.get(case["caseId"], {}))
        if decision["status"] == "available":
            row["eyeContour"] = enrichMetric(primary, row["selectedVariantId"], True)
            row["eyeAperture"] = enrichMetric(secondary, row["selectedVariantId"], False)
        else:
            row["eyeContour"] = row["eyeAperture"] = {"status": "unavailable", "reason": "policyDecisionMissing"}
        data = byMetric.get(case["caseId"], {})
        row["repeat"] = {key: data.get(key) for key in ("observedRepeatVertexRms", "observedRepeatFixedEyeProxyDelta")}
        cases.append(row)
    report = {"schemaVersion": 1, "status": "complete" if metrics.get("status") == "complete" and apertures.get("status") == "complete" and all(c["decision"]["status"] == "available" for c in cases) else "partial",
        "createdAt": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "routing": {"caseCount": len(cases), "tightCaseIds": [c["caseId"] for c in cases if c["selectedVariantId"] == CANDIDATE],
            "fallbackCaseIds": [c["caseId"] for c in cases if c["selectedVariantId"] == BASELINE],
            "retakeCaseIds": [c["caseId"] for c in cases if c["retakeRequired"]],
            "manualReviewCaseIds": [c["caseId"] for c in cases if c["manualReviewRequired"]]},
        "eyeContour": aggregateMetric(cases, "eyeContour", True),
        "eyeAperture": aggregateMetric(cases, "eyeAperture", False),
        "sourceHashes": {str(p): diagnosis.fileHash(p) for p in (Path(__file__), root / "cropManifest.json", root / "policyFrozen.json", metricsPath, aperturePath) if p.exists()},
        "limits": ["Frozen input-only policy routes all four frames together; it does not inspect reconstruction scores.",
            "Guarded results reuse baseline or unconditional candidate exactly and require no additional UFF inference.",
            "Routing protects input coverage; geometric or perceptual benefit is a separate empirical question.",
            "Eye contour uses two-variant common visibility, unlike the earlier nine-variant study; absolute scores are not comparable.",
            "Each capture group receives equal weight; identities and fixed-camera acquisition are not independently verified.",
            "Input photographs also supplied reconstruction. Both automatic 2D proxies use the frozen baseline camera and are not true 3D accuracy.",
            "Aperture missing-hole matches score zero; inspect segmentation, occlusion, and connected-component diagnostics.",
            "One baseline repeat reports observed numerical variation without confidence intervals or significance claims.",
            "Retake-required groups remain visible for diagnostics; selecting a fallback does not certify acceptable capture quality."]}
    writeJson(root / "validationMetrics.json", report)
    plot(report, root / "validationOverview.png")
    print(json.dumps({"status": report["status"], "routing": report["routing"],
        "primaryEligible": report["eyeContour"]["eligibleCaseCount"], "secondaryEligible": report["eyeAperture"]["eligibleCaseCount"]}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--diagnose", action="store_true", help="Run pairwise CPU diagnostics before aggregating")
    parser.add_argument("--apertures", action="store_true", help="Evaluate saved render masks before aggregating")
    parser.add_argument("--overwrite", action="store_true", help="Replace pairwise analysis only, never inference")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = read(root / "cropManifest.json")
    if args.diagnose:
        diagnose(root, manifest, args.overwrite)
    if args.apertures:
        runApertures(root)
    evaluate(root, manifest)


if __name__ == "__main__":
    main()
