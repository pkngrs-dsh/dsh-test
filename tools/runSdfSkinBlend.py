"""Build same-geometry client assets before/after semantic skin extrapolation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as connectedLabels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.clientAsset import exportClientAsset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def srgb(value):
    x = np.clip(value, 0, 1)
    return np.where(x <= .0031308, x*12.92, 1.055*x**(1/2.4)-.055)


def panels(images, labels, path, crop=None, width=700):
    height = round(width * (images[0].height if crop is None else crop[3]-crop[1]) /
                   (images[0].width if crop is None else crop[2]-crop[0]))
    result = Image.new("RGB", (width*len(images), height+40), (25, 31, 36))
    draw = ImageDraw.Draw(result)
    for index, (im, label) in enumerate(zip(images, labels)):
        draw.text((index*width+12, 12), label, fill="white")
        selected = im if crop is None else im.crop(crop)
        result.paste(selected.resize((width, height), Image.Resampling.LANCZOS), (index*width, 40))
    result.save(path)


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    sourceTexture, supportPath = args.texture.resolve(), args.support.resolve()
    if (output / "clientAssets").exists():
        raise FileExistsError("Use a fresh output: clientAssets already exists")
    semanticReport = readJson(supportPath.with_name("semanticSupport.json"))
    if semanticReport["status"] != "completed" or Path(semanticReport["sourceRun"]) != source:
        raise ValueError("Skin support must belong to this source reconstruction")
    if semanticReport.get("support", {}).get("repairableNonSkinLabels") != ["background", "neck", "hair"]:
        raise ValueError("Regenerate skin support with separate donor exclusion and feature preservation")
    if digest(supportPath) != semanticReport["artifactSha256"]:
        raise ValueError("Skin support changed after preparation")
    for name, expected in semanticReport["sourceHashes"].items():
        if digest(source/name) != expected:
            raise ValueError(f"Frozen skin-support input changed: {name}")
    textureHash = digest(sourceTexture)
    texture, support = arrays(sourceTexture), arrays(supportPath)
    coverage = texture["coverageMask"]
    texture["safeSkinMask"] = support["safeSkinMask"] & coverage
    texture["protectedFeatureMask"] = support["protectedFeatureMask"] & coverage
    geometry = arrays(source / "geometry.npz")
    config = readJson(source / "config.json")
    assetRoot = Path(config["assemblyAsset"])
    if not assetRoot.is_absolute():
        assetRoot = ROOT / assetRoot
    template = AssemblyAssets(assetRoot)
    if template.manifestSha256 != readJson(source/"assembly/diagnostics.json")["assetManifestSha256"]:
        raise ValueError("Template differs from the original reconstruction")
    template.skinExtensionPixels = args.extension
    template.skinDonorInsetPixels = args.inset
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "textureWithSkinSupport.npz", **texture)
    provenance, entries = {}, []
    for name, mode, label, description in (
            ("legacy", "legacy", "原融合", "同一原贴图，覆盖边缘向内羽化后直接混合模板。"),
            ("sdf", "sdfSkin", "SDF 肤色外扩", "内侧可靠肤色先延伸至边界外，再按距离平滑过渡到模板。")):
        started = perf_counter()
        result = template.assemble(geometry, texture, textureBlendMode=mode)
        elapsed = perf_counter()-started
        exportClientAsset(output / "clientAssets" / name, result)
        provenance[name] = result["provenanceMaps"]
        if result["textureBlendDebug"]:
            np.savez_compressed(output / "sdfDebug.npz", **result["textureBlendDebug"])
            for key in ("projectedLinear", "extendedSkinLinear"):
                writeImage(output / (key+".png"), srgb(result["textureBlendDebug"][key]))
        entries.append({"name": name, "label": label, "description": description,
                        "directory": "clientAssets/"+name, "seconds": elapsed,
                        "textureDiagnostics": result["diagnostics"]["texture"]})
        print(f"Exported {name}: assembly {elapsed:.3f}s", flush=True)
    before = arrays(output / "clientAssets/legacy/clientMesh.npz")
    after = arrays(output / "clientAssets/sdf/clientMesh.npz")
    original = arrays(source / "assembly/clientMesh.npz")
    for key in before:
        if not np.array_equal(before[key], after[key]) or not np.array_equal(before[key], original[key]):
            raise ValueError(f"Boundary-only assembly changed client geometry: {key}")
    if not np.array_equal(provenance["legacy"]["sourceCoverage"], provenance["sdf"]["sourceCoverage"]):
        raise ValueError("Extrapolation must not create new measured coverage")
    photos = [Image.open(output / f"clientAssets/{n}/head_color.png").convert("RGB") for n in ("legacy", "sdf")]
    panels(photos, ["Original inward feather", "SDF safe-skin extension"], output / "atlasComparison.png")
    components, _ = connectedLabels(provenance["legacy"]["sourceCoverage"])
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    ys, xs = np.nonzero(components == sizes.argmax())
    crop = (max(0,int(xs.min())-70),max(0,int(ys.min())-70),
            min(photos[0].width,int(xs.max())+71), min(photos[0].height,int(ys.max())+71))
    panels(photos, ["Original boundary", "SDF skin then template"], output / "boundaryComparison.png", crop=crop)
    mapNames = ["sourceCoverage", "safeSkinDonor", "skinRepairWeight", "skinExtrapolated", "photoBlendAlpha"]
    images = [Image.fromarray(np.rint(provenance["sdf"][n].astype(float)*255).astype(np.uint8)).convert("RGB") for n in mapNames]
    panels(images, mapNames, output / "boundaryMasks.png", crop=crop, width=360)
    for name, expected in semanticReport["sourceHashes"].items():
        if digest(source/name) != expected:
            raise ValueError("Source reconstruction changed during assembly")
    if digest(sourceTexture) != textureHash or digest(supportPath) != semanticReport["artifactSha256"]:
        raise ValueError("Texture or semantic support changed during assembly")
    implementation = {name: digest(ROOT/name) for name in ("src/faceAppearance/headAssembly.py",
                       "src/faceAppearance/skinExtrapolation.py", "tools/runSdfSkinBlend.py")}
    for name in implementation:
        dest = output/"implementation"/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    writeJson(output / "comparison.json", {"status": "completed", "sourceRun": str(source),
                "sourceTexture": str(sourceTexture), "sourceTextureSha256": textureHash,
                "skinSupport": str(supportPath), "skinSupportSha256": digest(supportPath),
                "entries": entries, "implementationHashes": implementation,
                "geometryIdentical": True, "measuredCoverageIdentical": True,
                "sourceStage": "SH + prior before WildCap", "modelsRerun": False,
                "parameters": {"extensionPixels": args.extension, "insetPixels": args.inset},
                "limits": ["Extrapolated skin is estimated low-frequency color, not a new observation",
                           "Image semantics and existing UV correspondences may have errors",
                           "No change to internal texture defects, geometry, material parameters or rig"]})
    print(json.dumps({"status":"completed", "output":str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"runs/service/b775794b61894a8b804fddb0620db45d/output")
    parser.add_argument("--texture", type=Path, default=ROOT/"runs/shWildCapClientComparison/shPrior/texture.npz")
    parser.add_argument("--support", type=Path, default=ROOT/"runs/sdfSkinTemplateBlendV2/sourceSkinSupport.npz")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/sdfSkinTemplateBlendFinal")
    parser.add_argument("--extension", type=float, default=48)
    parser.add_argument("--inset", type=float, default=24)
    run(parser.parse_args())
