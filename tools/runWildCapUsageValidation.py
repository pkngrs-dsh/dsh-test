"""Controlled WildCap usage validation on the current cached SH-prior subject.

prepare creates a reviewed candidate artifact mask and frozen requests. run
replays the original baseline, changes only the lighting schedule, then changes
only the artifact mask. A fifth display variant feathers the local result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.atlasTransfer import transferTexture
from faceAppearance.registration import RegistrationAssets
from runShExperiment import validateCompletedWildCap

BRANCHES = {"baseline": ("upstream", False), "global301": ("global301", False),
            "localTgl": ("upstream", True)}
LABELS = {"input": "SH + 先验输入", "baseline": "原配置复跑", "global301": "全局光更新 301 次",
          "localTgl": "启用局部阴影掩码", "localTglFeather": "局部掩码 + 边缘过渡"}
DESCRIPTIONS = {"input": "固定的 SH + OpenDelight 先验输入，未加 WildCap。",
                "baseline": "原配置复跑：全局光更新 101 次，无局部掩码，直接回传。",
                "global301": "仅将全局光更新延长至 301 次，其他条件与基线相同。",
                "localTgl": "原分段光照调度，加保守的候选残余阴影掩码；肤色锚固定。",
                "localTglFeather": "同一 localTgl 输出，仅在回传边界向内做线性颜色过渡。"}


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen(plan):
    for path, expected in plan["sourceHashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen validation input changed: {path}")


def candidateMask(valid):
    """Subject-specific visual annotation of residual darkness, not shadow truth.

    Pixel centers refer to the inspected 512 ICT input. Eyelashes, brows,
    nostrils, lips and visible moles are protected with an additional margin.
    """
    regions = {
        "leftUnderEye": [(143, 157), (177, 155), (215, 159), (220, 176), (200, 187), (168, 181), (145, 171)],
        "rightUnderEye": [(296, 159), (332, 155), (365, 157), (364, 171), (339, 181), (316, 184), (293, 175)],
        "leftNoseSide": [(211, 177), (227, 183), (222, 202), (214, 211), (201, 205), (198, 192)],
        "rightNoseSide": [(284, 183), (300, 177), (312, 192), (309, 205), (296, 211), (287, 202)],
    }
    maskImage = Image.new("L", (512, 512)); draw = ImageDraw.Draw(maskImage)
    for polygon in regions.values():
        draw.polygon(polygon, fill=255)
    protection = Image.new("L", (512, 512)); draw = ImageDraw.Draw(protection)
    boxes = [(102, 76, 236, 127), (277, 76, 410, 127), (127, 128, 230, 152),
             (279, 128, 382, 152), (184, 241, 329, 339), (201, 209, 244, 242), (266, 209, 311, 242)]
    for box in boxes:
        draw.rectangle(box, fill=255)
    for x, y, radius in ((358, 187, 7), (306, 361, 6), (143, 351, 6)):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    protected = binary_dilation(np.asarray(protection) > 0, iterations=3)
    mask = (np.asarray(maskImage) > 0) & valid & ~protected
    if not mask.any() or np.any(mask & protected):
        raise ValueError("Candidate mask has no useful support or overlaps protected details")
    return mask, protected, {"regionsIct512Pixels": regions, "protectedBoxesIct512Pixels": boxes,
                            "extraProtectionPixels": 3, "candidateTexels": int(mask.sum()),
                            "meaning": "Conservative visually chosen candidate artifacts, not annotated shadow ground truth",
                            "scope": "Only this inspected subject and fixed ICT512 input; not an automatic detector"}


def featherComposite(base, diffuse, valid, width=12):
    if width <= 0:
        raise ValueError("Feather width must be positive")
    x = np.clip((distance_transform_edt(valid) - 1) / width, 0, 1).astype(np.float32)
    alpha = x*x*(3-2*x)
    linear = base ** 2.2
    mixed = linear * (1-alpha[..., None]) + diffuse ** 2.2 * alpha[..., None]
    composite = np.clip(mixed, 0, 1) ** (1/2.2)
    composite[~valid] = base[~valid]
    composite[alpha == 1] = diffuse[alpha == 1]
    return composite.astype(np.float32), alpha


def edgeDeltaMetric(color, base, valid, coverage):
    """Measure added discontinuity across the compositing boundary only."""
    delta = color ** 2.2 - base ** 2.2
    values = []
    for axis in (0, 1):
        first, second = [slice(None)]*2, [slice(None)]*2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        crossing = (valid[first] != valid[second]) & coverage[first] & coverage[second]
        values.append(np.abs(delta[first] - delta[second])[crossing])
    values = np.concatenate(values)
    return {"edgePairCount": len(values), "meanAddedJumpLinear": float(values.mean()),
            "p95AddedJumpLinear": float(np.quantile(values, .95)),
            "meaning": "Added color-delta jump at the replacement boundary, not albedo accuracy"}


def prepare(args):
    out, experiment = args.output.resolve(), args.sourceExperiment.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh validation output directory")
    main = readJson(experiment / "experimentReport.json")
    source = Path(main["sourceRun"])
    frontend = "shPrior"
    branch = experiment / frontend
    previous = readJson(branch / "result.json")
    diagnostic = validateCompletedWildCap(branch, previous, main["parameters"]["wildCapPlanned"], 1024)
    if diagnostic["lightingMode"] != "globalOnly" or diagnostic["artifactMaskProvided"]:
        raise ValueError("Expected the saved no-mask global-only baseline")
    texture = arrays(branch / "texture.npz")
    forward = RegistrationAssets(ROOT / "assets/registration/wrapV1", 512)
    request = transferTexture(texture["albedo"], texture["coverageMask"], forward.atlasMap("uffToIct"),
                              materialKind="color", colorEncoding="gamma2.2")
    savedPng = np.asarray(Image.open(branch / "wildcapIct/input.png").convert("RGB"))
    if not np.array_equal(savedPng, np.rint(request["texture"] * 255).astype(np.uint8)):
        raise ValueError("Recomputed ICT request differs from saved original input")
    geometry = forward.toIctGeometry(arrays(source / "geometry.npz"))
    mask, protected, annotation = candidateMask(request["validMask"])
    out.mkdir(parents=True)
    np.savez_compressed(out / "request.npz", texture=request["texture"], validMask=request["validMask"],
                        artifactMask=mask, protectedMask=protected,
                        **{k: v for k, v in geometry.items() if isinstance(v, np.ndarray)})
    writeImage(out / "inputIct.png", request["texture"])
    writeImage(out / "artifactMask.png", mask)
    writeImage(out / "protectedFeatures.png", protected)
    overlay = request["texture"].copy()
    overlay[mask] = .55*overlay[mask] + .45*np.array([0, 1, .5])
    writeImage(out / "artifactMaskOverlay.png", overlay)
    writeJson(out / "maskAnnotation.json", annotation)
    paths = [source / name for name in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json", "delight.npz")]
    paths += [branch / "texture.npz", branch / "wildcapIct/materials.npz", branch / "wildcapIct/diagnostics.json",
              branch / "wildcapUff/materials.npz", ROOT / "assets/registration/wrapV1/manifest.json"]
    paths += [source / f"input{i:02d}.png" for i in range(4)]
    plan = {"status": "prepared", "sourceRun": str(source), "sourceExperiment": str(experiment),
            "frontend": frontend, "geometryFrame": geometry["geometryFrame"], "resolution": 512,
            "iterations": 599, "seed": 2026, "fixedSkinTone": diagnostic["skinTone"],
            "skinToneMeaning": "Original SH-prior anchor held fixed; not a calibrated true skin tone",
            "branches": BRANCHES, "featherWidthUffPixels": 12, "requestSha256": sha(out / "request.npz"),
            "sourceHashes": {str(p): sha(p) for p in paths},
            "limits": ["One subject and one candidate mask, no albedo ground truth",
                       "This adapter excludes artifacts from reliable skin support; verify actual overlap separately (this is an adaptation, not the upstream rule)",
                       "Normal/specular maps remain excluded from controlled diffuse preview"]}
    writeJson(out / "plan.json", plan)
    frozen(plan)
    print(json.dumps({"status": "prepared", "maskTexels": int(mask.sum()), "output": str(out)}), flush=True)


def completedBranch(path, plan, name):
    record = readJson(path / "result.json")
    if record.get("status") != "succeeded" or record.get("requestSha256") != plan["requestSha256"]:
        raise ValueError(f"Uncommitted validation result: {name}")
    for filename, expected in record["artifactHashes"].items():
        if sha(path / filename) != expected:
            raise ValueError(f"Changed completed result: {name}/{filename}")
    diagnostic = readJson(path / "diagnostics.json")
    expectedCounts = {"baseline": (101, 200, 0), "global301": (301, 0, 0), "localTgl": (101, 200, 200)}
    for key, count in zip(("globalOptimizerSteps", "localOptimizerSteps", "effectiveLocalOptimizerSteps"), expectedCounts[name]):
        if diagnostic.get(key) != count:
            raise ValueError(f"Unexpected effective optimizer updates: {name}/{key}={diagnostic.get(key)}")
    if diagnostic["iterations"] != 599 or diagnostic["seed"] != plan["seed"]:
        raise ValueError("Validation sampling conditions changed")
    for filename in ("ictMaterials.npz", "uffMaterials.npz"):
        for key, value in arrays(path / filename).items():
            if value.dtype.kind not in "bifu" or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite or invalid result: {filename}/{key}")
    return diagnostic


def exportVariants(out, plan):
    source = Path(plan["sourceExperiment"]) / plan["frontend"]
    texture = arrays(source / "texture.npz")
    base, coverage = texture["albedo"], texture["coverageMask"]
    loaded = {name: arrays(out / "branches" / name / "uffMaterials.npz") for name in BRANCHES}
    support = loaded["baseline"]["validMask"]
    for name, data in loaded.items():
        if not np.array_equal(data["validMask"], support):
            raise ValueError(f"Output support differs: {name}")
    variants = {"input": base, **{name: value["baseWithWildCap"] for name, value in loaded.items()}}
    feather, alpha = featherComposite(base, loaded["localTgl"]["diffuse"], support, plan["featherWidthUffPixels"])
    variants["localTglFeather"] = feather
    writeImage(out / "featherAlpha.png", alpha)
    reports = {}
    for name, color in variants.items():
        destination = out / "variants" / name
        if destination.exists():
            raise FileExistsError(f"Partial presentation output requires review: {destination}")
        destination.mkdir(parents=True)
        if not np.array_equal(color[~support], base[~support]):
            raise ValueError(f"Variant changes outside its valid WildCap patch: {name}")
        saved = dict(texture, albedo=color, linearAlbedo=color**2.2)
        saved["previewAlbedo"] = color.copy()
        np.savez_compressed(destination / "texture.npz", **saved)
        writeImage(destination / "albedo.png", color)
        record = {"status": "succeeded", "label": LABELS[name], "description": DESCRIPTIONS[name],
                  "seconds": None if name == "input" else readJson(out / "branches" / ("localTgl" if name == "localTglFeather" else name) / "diagnostics.json")["seconds"],
                  "edgeDeltaMetric": edgeDeltaMetric(color, base, support, coverage),
                  "textureSha256": sha(destination / "texture.npz")}
        writeJson(destination / "result.json", record)
        reports[name] = record
    original = arrays(source / "wildcapIct/materials.npz")
    replay = arrays(out / "branches/baseline/ictMaterials.npz")
    replayCheck = {key: {"maxAbs": float(np.max(np.abs(original[key].astype(float)-replay[key].astype(float)))),
                         "meanAbs": float(np.mean(np.abs(original[key].astype(float)-replay[key].astype(float))))}
                   for key in ("diffuse", "specular", "tangentNormal", "validMask")}
    report = {"status": "completed", "sourceRun": plan["sourceRun"], "frontend": plan["frontend"],
              "baselineReplay": replayCheck, "variants": reports,
              "branches": {n: completedBranch(out / "branches" / n, plan, n) for n in BRANCHES},
              "commonCoverageTexels": int(coverage.sum()), "wildCapValidTexels": int(support.sum()),
              "featherCoreIdentical": bool(np.array_equal(feather[alpha == 1], loaded["localTgl"]["diffuse"][alpha == 1])),
              "sourceUnchanged": True, "limits": plan["limits"]}
    frozen(plan)
    writeJson(out / "report.json", report)
    print(json.dumps({"status": "completed", "baselineReplay": replayCheck}), flush=True)


def run(args):
    import torch
    from faceAppearance.runtime import configureRuntime
    configureRuntime(ROOT)
    from faceAppearance.adapters.wildcap import WildCapAdapter
    out = args.output.resolve(); plan = readJson(out / "plan.json")
    frozen(plan)
    if sha(out / "request.npz") != plan["requestSha256"]:
        raise ValueError("Prepared request changed")
    request = arrays(out / "request.npz")
    back = RegistrationAssets(ROOT / "assets/registration/wrapV1", 1024)
    sourceTexture = arrays(Path(plan["sourceExperiment"]) / plan["frontend"] / "texture.npz")
    currentImplementation = {str(p.relative_to(ROOT)): sha(p) for p in
                             (ROOT / "src/faceAppearance/adapters/wildcap.py", Path(__file__))}
    adapter = None
    for name, (schedule, masked) in BRANCHES.items():
        dest = out / "branches" / name
        if (dest / "result.json").exists():
            completedBranch(dest, plan, name)
            if readJson(dest / "result.json")["implementationHashes"] != currentImplementation:
                raise ValueError("Implementation changed across resumed branches")
            print("Verified completed " + name, flush=True)
            continue
        if dest.exists():
            raise FileExistsError(f"Preserve incomplete branch before restarting: {dest}")
        if adapter is None:
            adapter = WildCapAdapter(ROOT, args.device).loadModels()
        dest.mkdir(parents=True)
        torch.cuda.reset_peak_memory_stats()
        print(f"Running {name}: schedule={schedule}, mask={masked}, 599 updates", flush=True)
        result = adapter.optimize(request["vertices"], request["faces"], request["uvs"], request["texture"],
                                  request["validMask"], request["artifactMask"] if masked else None,
                                  uvFaces=request["uvFaces"], geometryFrame=plan["geometryFrame"],
                                  workingResolution=512, iterations=599, seed=plan["seed"],
                                  skinTone=plan["fixedSkinTone"], lightingSchedule=schedule)
        np.savez_compressed(dest / "ictMaterials.npz", **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        returned = transferTexture(result["diffuse"], result["validMask"], back.atlasMap("ictToUff"),
                                   materialKind="color", colorEncoding="gamma2.2")
        valid = returned["validMask"] & sourceTexture["coverageMask"]
        composite = sourceTexture["albedo"].copy(); composite[valid] = returned["texture"][valid]
        np.savez_compressed(dest / "uffMaterials.npz", diffuse=returned["texture"], validMask=valid, baseWithWildCap=composite)
        for filename,key in (("diffuse.png","diffuse"),("localMask.png","effectiveLocalMask"),("fittedShading.png","fittedShading")):
            if key in result:
                writeImage(dest / filename, result[key])
        diagnostic = dict(result["diagnostics"], peakAllocatedGiB=torch.cuda.max_memory_allocated()/1024**3)
        writeJson(dest / "diagnostics.json", diagnostic)
        writeJson(dest / "result.json", {"status": "succeeded", "name": name,
                  "requestSha256": plan["requestSha256"], "implementationHashes": currentImplementation,
                  "artifactHashes": {f:sha(dest/f) for f in ("ictMaterials.npz","uffMaterials.npz","diagnostics.json")}})
        completedBranch(dest, plan, name)
        frozen(plan)
        print(f"Completed {name}: {diagnostic['seconds']:.1f}s", flush=True)
    exportVariants(out, plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourceExperiment",type=Path,default=ROOT/"runs/shWildCapClientComparison")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/wildCapUsageValidation")
    parser.add_argument("--stage",choices=("prepare","run"),required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    (prepare if args.stage=="prepare" else run)(args)


if __name__=="__main__":
    main()
