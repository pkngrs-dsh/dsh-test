"""Read-only numerical and scientific-plot audit of completed WildCap ablations."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from runWildCapUsageValidation import ROOT, BRANCHES, arrays, readJson, sha, frozen, completedBranch
from faceAppearance.artifacts import writeJson


def describe(value):
    return {"min": float(value.min()), "median": float(np.median(value)),
            "max": float(value.max()), "mean": float(value.mean())}


def analyze(out):
    plan, report = readJson(out / "plan.json"), readJson(out / "report.json")
    if report["status"] != "completed" or sha(out / "request.npz") != plan["requestSha256"]:
        raise ValueError("Analysis needs a complete, unchanged experiment")
    frozen(plan)
    request = arrays(out / "request.npz")
    materials = {n: arrays(out / "branches" / n / "ictMaterials.npz") for n in BRANCHES}
    diagnostics = {n: completedBranch(out / "branches" / n, plan, n) for n in BRANCHES}
    baseline = materials["baseline"]
    observed = baseline["validMask"]
    local = materials["localTgl"]["effectiveLocalMask"]
    candidate = request["artifactMask"] & observed
    sharedSkin = np.logical_and.reduce([m["skinMask"] > .99 for m in materials.values()]) & observed
    skinMasksEqual = all(np.array_equal(m["skinMask"], baseline["skinMask"]) for m in materials.values())
    metrics = {}
    for name, m in materials.items():
        d = diagnostics[name]
        if (d["lightingSchedule"] != BRANCHES[name][0] or d["workingResolution"] != 512
                or d["outputResolution"] != 512 or d["skinTone"] != plan["fixedSkinTone"]):
            raise ValueError(f"Incorrect experimental conditions: {name}")
        if not np.array_equal(m["validMask"], observed) or not np.array_equal(m["guidance"], baseline["guidance"]):
            raise ValueError(f"Guidance or output support changes: {name}")
        localDelta = m["fittedShading"] - m["fittedMainShading"]
        if name != "localTgl" and (np.any(m["effectiveLocalMask"]) or np.any(localDelta)):
            raise ValueError("An unmasked branch has active local shading")
        if np.any(localDelta[~m["effectiveLocalMask"]]):
            raise ValueError("Local shading leaks outside its effective mask")
        diff = m["renderedGuidance"] - m["guidance"]
        regions = {"allObserved": observed, "candidateShadow": candidate, "sharedReliableSkin": sharedSkin}
        fit = {region: {"texels": int(mask.sum()), "linearMse": float(np.mean(diff[mask] ** 2)),
                        "linearMae": float(np.mean(np.abs(diff[mask])))} for region, mask in regions.items()}
        traces = [t["localGradientMaxAbs"] for t in d["lossTrace"]
                  if t["optimizer"] == "local" and t["localGradientMaxAbs"] is not None]
        metrics[name] = {"seconds": d["seconds"], "peakAllocatedGiB": d["peakAllocatedGiB"],
                         "globalSteps": d["globalOptimizerSteps"], "localSteps": d["localOptimizerSteps"],
                         "effectiveLocalSteps": d["effectiveLocalOptimizerSteps"],
                         "effectiveLocalTexels": int(m["effectiveLocalMask"].sum()),
                         "localGradientSamples": traces,
                         "localShadingDelta": describe(localDelta[local]) if local.any() else None,
                         "fittedShading": describe(m["fittedShading"][observed]),
                         "negativeShadingFraction": float(np.mean(m["fittedShading"][observed] < 0)),
                         "linearReconstructionFit": fit,
                         "diffuseChangeVsReplayEncodedMae": float(np.mean(np.abs(m["diffuse"][observed] - baseline["diffuse"][observed])))}
    if not local.any() or diagnostics["localTgl"]["effectiveLocalOptimizerSteps"] != 200:
        raise ValueError("The mask failed to activate local lighting")
    # The request protection includes a three-pixel safety margin. Upstream's
    # 5x5 local edge band may consume two pixels of that margin, but must not
    # reach the annotated feature cores themselves.
    annotation = readJson(out / "maskAnnotation.json")
    coreImage = Image.new("L", (512, 512)); draw = ImageDraw.Draw(coreImage)
    for box in annotation["protectedBoxesIct512Pixels"]:
        draw.rectangle(box, fill=255)
    for x, y, radius in ((358, 187, 7), (306, 361, 6), (143, 351, 6)):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    featureCore = np.asarray(coreImage) > 0
    if not np.array_equal(binary_dilation(featureCore, iterations=3), request["protectedMask"]):
        raise ValueError("Reconstructed feature cores do not match prepared protection")
    protectedOverlap = int((local & featureCore).sum())
    marginOverlap = int((local & request["protectedMask"] & ~featureCore).sum())
    if protectedOverlap:
        raise ValueError(f"Effective local mask overlaps {protectedOverlap} protected feature core pixels")
    result = {"status": "completed", "conditionsVerified": True,
              "baselineReplay": report["baselineReplay"], "metrics": metrics,
              "candidateMaskTexels": int(candidate.sum()), "effectiveMaskProtectedOverlap": protectedOverlap,
              "effectiveMaskSafetyMarginOverlap": marginOverlap,
              "allGuidanceEqual": True, "allIctValidMasksEqual": True,
              "allSkinMasksExactlyEqual": skinMasksEqual,
              "candidateMaskCoverage": float((candidate & local).sum() / candidate.sum()),
              "localShadingConfinedToEffectiveMask": True,
              "limits": ["Reconstruction fit is agreement with the input, not albedo accuracy",
                         "The adapter excludes artifacts from skin statistics (an extra adaptation, not the original upstream rule); this subject's candidate does not overlap that region and saved skin masks are checked for exact equality",
                         "A single visually chosen candidate mask is not a shadow ground truth annotation",
                         "Small border jump after feathering follows by construction and does not establish better delighting"],
              "sourceReportSha256": sha(out / "report.json")}
    destination = out / "analysis"
    destination.mkdir(exist_ok=True)
    plot(destination, request, materials, observed, local)
    frozen(plan)
    writeJson(destination / "report.json", result)
    return result


def plot(destination, request, materials, observed, local):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    def masked(x):
        if x.ndim == 3:
            rgba = np.concatenate([np.clip(x, 0, 1), observed[..., None].astype(float)], axis=2)
            return rgba
        return np.ma.masked_where(~observed, x)

    luminance = lambda x: x @ np.array([.2126, .7152, .0722])
    crop = (64, 448, 416, 32)
    names = list(BRANCHES)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, name in enumerate(names):
        m = materials[name]
        axes[row, 0].imshow(masked(m["diffuse"]))
        shade = np.log2(np.clip(luminance(m["fittedShading"]), 1/64, 64))
        shadeMap = axes[row, 1].imshow(masked(shade), cmap="coolwarm", norm=Normalize(-3, 1))
        delta = luminance(m["fittedShading"] - m["fittedMainShading"])
        deltaMap = axes[row, 2].imshow(masked(delta), cmap="PuOr_r", norm=Normalize(-.35, .35))
        residual = np.mean(np.abs(m["renderedGuidance"] - m["guidance"]), axis=2)
        errorMap = axes[row, 3].imshow(masked(residual), cmap="magma", norm=Normalize(0, .08))
        axes[row, 0].set_ylabel(name, fontsize=12)
        for col, ax in enumerate(axes[row]):
            if row == 0:
                ax.set_title(["Exported diffuse (gamma 2.2)", "Fitted shading: log2 luminance",
                              "Local minus global shading", "Reconstruction |error|, linear RGB"][col], fontsize=10)
            ax.set_xlim(crop[:2]); ax.set_ylim(crop[2:]); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(shadeMap, ax=axes[:, 1], shrink=.5)
    fig.colorbar(deltaMap, ax=axes[:, 2], shrink=.5)
    fig.colorbar(errorMap, ax=axes[:, 3], shrink=.5)
    fig.suptitle("Controlled WildCap validation: same input, geometry, seed and skin anchor\nShading and reconstruction precede export skin normalization; fitted error is not albedo accuracy", fontsize=12)
    fig.savefig(destination / "lightingAudit.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(request["texture"]); axes[0].set_title("Fixed SH + prior input")
    axes[1].imshow(request["texture"])
    overlay = np.zeros((*local.shape, 4)); overlay[request["artifactMask"]] = [0, 1, .5, .5]
    axes[1].imshow(overlay); axes[1].set_title("Candidate artifact mask")
    axes[2].imshow(request["texture"])
    overlay[:] = 0; overlay[local] = [0, 1, .5, .5]
    axes[2].imshow(overlay); axes[2].set_title("Effective local support incl. edge band")
    for ax in axes:
        ax.set_xlim(crop[:2]); ax.set_ylim(crop[2:]); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(destination / "maskAudit.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/wildCapUsageValidation")
    result = analyze(parser.parse_args().experiment.resolve())
    print({"status": result["status"], "metrics": result["metrics"]})
