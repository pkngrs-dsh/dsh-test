"""Validate, export and render the five saved WildCap usage candidates.

This reads completed experiment outputs only. It does not run a neural model,
fit cameras, change geometry, or enable WildCap specular/normal maps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.artifacts import writeImage, writeJson as writeArtifactJson
from exportShClientComparison import (checkMesh, digest, loadVerifiedAsset,
                                      readArrays, readJson, writeJson)

NAMES = ("input", "baseline", "global301", "localTgl", "localTglFeather")
LABELS = {"input": "SH + 先验输入", "baseline": "原 WildCap 配置",
          "global301": "延长全局光照拟合", "localTgl": "局部阴影掩码",
          "localTglFeather": "局部阴影 + 边界过渡"}
ENGLISH = {"input": "Input SH + prior", "baseline": "WC baseline",
           "global301": "WC global 301", "localTgl": "WC local mask",
           "localTglFeather": "WC local + feather"}
NOTES = {"input": "固定的 SH + OpenDelight 先验输入，未执行 WildCap。",
         "baseline": "复现上一轮 WildCap 配置，用于定位差异。",
         "global301": "仅调整全局光照拟合调度；其它设置由实验记录固定。",
         "localTgl": "加入局部阴影掩码，保留原回传边界用于对照。",
         "localTglFeather": "使用同一局部阴影结果，单独改善有效回传区域的边界过渡。"}
SOURCE_FILES = ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json",
                "config.json", "result.json", "assembly/clientMesh.npz",
                "assembly/manifest.json", "assembly/diagnostics.json")


def absolutePath(value) -> Path:
    value = Path(value)
    return (value if value.is_absolute() else ROOT / value).resolve()


def validateExperiment(experiment: Path) -> dict:
    """Check the committed data contract before export, rendering or serving."""
    experiment = Path(experiment).resolve()
    required = ["plan.json", "report.json", "request.npz"]
    required += [f"variants/{name}/{filename}" for name in NAMES
                 for filename in ("texture.npz", "result.json")]
    required += [f"branches/{name}/{filename}" for name in ("baseline", "global301", "localTgl")
                 for filename in ("ictMaterials.npz", "uffMaterials.npz", "diagnostics.json", "result.json")]
    missing = [name for name in required if not (experiment / name).is_file()]
    if missing:
        raise FileNotFoundError("WildCap validation is incomplete; missing: " + ", ".join(missing))
    plan, report = readJson(experiment / "plan.json"), readJson(experiment / "report.json")
    if report.get("status") != "completed":
        raise ValueError("WildCap validation requires report.json status=completed")
    if plan.get("frontend") != "shPrior":
        raise ValueError("This presentation requires the planned fixed shPrior frontend")
    source, frontend = absolutePath(plan["sourceRun"]), absolutePath(plan["sourceExperiment"]) / "shPrior"
    if absolutePath(report["sourceRun"]) != source or report.get("frontend") != plan["frontend"]:
        raise ValueError("Completed report refers to a different source or frontend")
    if digest(experiment / "request.npz") != plan["requestSha256"]:
        raise ValueError("Saved WildCap request changed after planning")
    for filename, expected in plan["sourceHashes"].items():
        if digest(absolutePath(filename)) != expected:
            raise ValueError(f"A planned source artifact changed: {filename}")
    if readJson(source / "result.json").get("status") != "succeeded":
        raise ValueError("Source reconstruction has not succeeded")
    sourceHashes = {name: digest(source / name) for name in SOURCE_FILES}
    sourceHashes.update({path.name: digest(path) for path in sorted(source.glob("input*.png"))})
    for filename in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"):
        if digest(frontend / filename) != sourceHashes[filename]:
            raise ValueError(f"Frontend differs from the fixed source: {filename}")
    base = readArrays(frontend / "texture.npz")
    common = base["coverageMask"]
    if common.dtype != np.bool_ or common.ndim != 2 or not common.any():
        raise ValueError("Frontend support must be a nonempty boolean UV mask")
    variants = {}
    for name in NAMES:
        directory = experiment / "variants" / name
        result, texture = readJson(directory / "result.json"), readArrays(directory / "texture.npz")
        if result.get("status") != "succeeded":
            raise ValueError(f"Candidate is not complete: {name}")
        if result.get("textureSha256") != digest(directory / "texture.npz"):
            raise ValueError(f"Candidate texture differs from its committed identity: {name}")
        if report["variants"].get(name) != result:
            raise ValueError(f"Completed report and candidate metadata differ: {name}")
        for field, shape in (("albedo", (*common.shape, 3)), ("linearAlbedo", (*common.shape, 3)),
                             ("coverageMask", common.shape), ("confidence", common.shape),
                             ("sourceIndex", common.shape)):
            if field not in texture or texture[field].shape != shape or not np.isfinite(texture[field]).all():
                raise ValueError(f"Invalid or missing texture field: {name}/{field}")
        if texture["coverageMask"].dtype != np.bool_ or not np.array_equal(texture["coverageMask"], common):
            raise ValueError(f"Candidate changes the observed UV support: {name}")
        for field in ("albedo", "linearAlbedo", "confidence"):
            if texture[field].min() < 0 or texture[field].max() > 1:
                raise ValueError(f"Texture values outside [0, 1]: {name}/{field}")
        if not np.allclose(texture["linearAlbedo"], texture["albedo"] ** 2.2, atol=2e-6, rtol=1e-5):
            raise ValueError(f"Stale linearAlbedo or incorrect gamma conversion: {name}")
        if name == "input" and not np.array_equal(texture["albedo"], base["albedo"]):
            raise ValueError("Input candidate differs from the cached SH + prior atlas")
        variants[name] = {"texture": texture, "result": result}
    for name in ("baseline", "global301", "localTgl"):
        directory = experiment / "branches" / name
        result = readJson(directory / "result.json")
        if result.get("status") != "succeeded" or result.get("requestSha256") != plan["requestSha256"]:
            raise ValueError(f"WildCap model branch has not succeeded: {name}")
        for filename in ("ictMaterials.npz", "uffMaterials.npz", "diagnostics.json"):
            if result["artifactHashes"].get(filename) != digest(directory / filename):
                raise ValueError(f"Model artifact differs from its committed identity: {name}/{filename}")
        if report["branches"].get(name) != readJson(directory / "diagnostics.json"):
            raise ValueError(f"Completed report and model diagnostics differ: {name}")
    identity = {"sourceRun": str(source), "sourceHashes": sourceHashes,
                "experimentHashes": {name: digest(experiment / name) for name in required},
                "frontendTextureSha256": digest(frontend / "texture.npz"),
                "templateManifestSha256": readJson(source / "assembly/diagnostics.json")["assetManifestSha256"]}
    return {"experiment": experiment, "source": source, "plan": plan, "report": report,
            "identity": identity, "variants": variants, "coverage": common,
            "geometry": readArrays(source / "geometry.npz")}


def variantIdentity(data, name):
    return {**data["identity"], "variant": name,
            "colorInput": "linearAlbedo validated against gamma2.2 albedo",
            "implementationHashes": {path: digest(ROOT / path) for path in (
                "tools/presentWildCapValidation.py", "src/faceAppearance/headAssembly.py",
                "src/faceAppearance/clientAsset.py")}}


def variantSeconds(data, name):
    result = data["variants"][name]["result"]
    for key in ("wildCapSeconds", "seconds", "durationSeconds"):
        if isinstance(result.get(key), (int, float)):
            return result[key]
    branch = "localTgl" if name == "localTglFeather" else name
    path = data["experiment"] / "branches" / branch / "diagnostics.json"
    if path.is_file():
        value = readJson(path).get("seconds")
        if isinstance(value, (int, float)):
            return value
    return None


def exportClients(data):
    from faceAppearance.clientAsset import exportClientAsset
    from faceAppearance.headAssembly import AssemblyAssets

    source, experiment = data["source"], data["experiment"]
    config = readJson(source / "config.json")
    assetRoot = Path(config["assemblyAsset"])
    if not assetRoot.is_absolute():
        assetRoot = Path(config.get("projectRoot", ROOT)) / assetRoot
    template = AssemblyAssets(assetRoot)
    if template.manifestSha256 != data["identity"]["templateManifestSha256"]:
        raise ValueError("Client template differs from the original reconstruction")
    templateManifest = readJson(source / "assembly/manifest.json")
    sourceMesh = readArrays(source / "assembly/clientMesh.npz")
    destination = experiment / "clientAssets"
    destination.mkdir(exist_ok=True)
    entries = []
    for name in NAMES:
        output, identity = destination / name, variantIdentity(data, name)
        if output.exists() and any(output.iterdir()):
            asset, metadata = loadVerifiedAsset(output, templateManifest, identity)
            checkMesh(sourceMesh, asset.mesh)
        else:
            started = perf_counter()
            assembly = template.assemble(data["geometry"], data["variants"][name]["texture"], colorEncoding="gamma2.2")
            exportClientAsset(output, assembly)
            archive = readJson(output / "archive.json")
            metadata = {"schemaVersion": 1, "name": name, "inputIdentity": identity,
                        "archive": {key: archive[key] for key in ("sha256", "bytes")},
                        "assemblyAndExportSeconds": perf_counter() - started}
            writeJson(output / "comparison.json", metadata)
            asset, _ = loadVerifiedAsset(output, templateManifest, identity)
            metadata["identicalClientArrays"] = checkMesh(sourceMesh, asset.mesh)
            writeJson(output / "comparison.json", metadata)
        result = data["variants"][name]["result"]
        entries.append({"name": name, "label": result.get("label", LABELS[name]),
                        "description": result.get("description", NOTES[name]),
                        "directory": f"clientAssets/{name}", "archive": metadata["archive"],
                        "inputIdentity": identity, "seconds": variantSeconds(data, name),
                        "commonCoverageTexels": int(data["coverage"].sum()),
                        "candidateOnly": True})
        print(f"Verified client asset: {name}", flush=True)
    if validateExperiment(experiment)["identity"] != data["identity"]:
        raise ValueError("Experiment/source inputs changed during client export")
    manifest = {"schemaVersion": 1, "status": "complete", "experiment": str(experiment),
                "sourceRun": str(source), "inputIdentity": data["identity"],
                "templateId": templateManifest["templateId"], "templateHashes": templateManifest["templateHashes"],
                "commonCoverageTexels": int(data["coverage"].sum()),
                "comparison": "Candidate validation; fixed geometry, geometry normals, preset roughness; diffuse only",
                "entries": entries}
    writeJson(experiment / "clientComparison.json", manifest)
    return manifest


def renderPresentation(data, *, device="cuda", imageSize=768, panelSize=384, detailView=0):
    from renderCropComparison import asImage, makeGrid, prepareGeometry, validatePhotoPair
    from renderShExperiment import atlasPreview, renderCanonical, renderPhotoView, renderVariant
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer
    import torch

    started = perf_counter()
    source, output = data["source"], data["experiment"] / "presentation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Presentation requires an empty directory: {output}")
    if min(imageSize, panelSize) < 32:
        raise ValueError("imageSize and panelSize must be at least 32")
    inputs = readJson(source / "inputs.json")
    photos = [validatePhotoPair(item, item) for item in inputs]
    if not 0 <= detailView < len(photos):
        raise ValueError("detailView must name an existing source photo")
    cameras, transforms = readJson(source / "cameras.json"), readJson(source / "viewTransforms.json")
    if not (len(cameras) == len(transforms) == len(photos) == len(data["geometry"]["faceMasks"])):
        raise ValueError("Source photo/camera/transform/segmentation counts differ")
    for index, camera in enumerate(cameras):
        if not camera.get("valid") or camera.get("viewIndex", index) != index:
            raise ValueError("Rendering requires accepted ordered saved PnP cameras")
    reference = {"inputs": inputs, "cameras": cameras, "transforms": transforms,
                 "savedGeometry": data["geometry"]}
    coverage = data["coverage"]
    geometry = prepareGeometry(data["geometry"], coverage.shape[0])
    variants = [{"name": name, "label": ENGLISH[name], "albedo": data["variants"][name]["texture"]["albedo"]}
                for name in NAMES]
    output.mkdir(exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    writeImage(output / "commonUvCoverage.png", coverage)
    atlasPanels = []
    for variant in variants:
        values = atlasPreview(variant["albedo"], coverage)
        writeImage(output / f"atlas{variant['name'].title()}.png", values)
        atlasPanels.append(asImage(values))
    makeGrid(atlasPanels, [item["label"] for item in variants], columns=5, panelSize=panelSize,
             subtitle="Same UV support | linear RGB displayed as sRGB | five usage candidates").save(output / "atlasOverview.png")
    panels = {mode: {"panels": [], "labels": []} for mode in ("render", "overlay", "unlit")}
    views = []
    for index, photo in enumerate(photos):
        viewPanels, viewReport = renderPhotoView(reference, geometry, variants, coverage, photo, index,
                                                output, device, panelSize, index == detailView)
        for mode in panels:
            for key in ("panels", "labels"):
                panels[mode][key].extend(viewPanels[mode][key])
        views.append(viewReport)
        print(f"Rendered source photo {index + 1}/{len(photos)}", flush=True)
    for mode, filename in (("render", "overview.png"), ("overlay", "overlayOverview.png"), ("unlit", "unlitOverview.png")):
        makeGrid(**panels[mode], columns=6, panelSize=panelSize,
                 subtitle="Original photo + five candidates | frozen PnP and geometry | " + mode).save(output / filename)
    canonical = renderCanonical(geometry, variants, coverage, output, device, imageSize, panelSize, False)
    renderer = Pytorch3dFaceRenderer(geometry, device=device, imageSize=imageSize, frameGeometry=True)
    renderer.setView(azimuth=0, elevation=0)
    unlitPanels, frontPanels = [], []
    for variant in variants:
        images = renderVariant(renderer, variant, coverage, np.array([0., .3, 1.]))
        unlitPanels.append(asImage(images["unlit"]))
        frontPanels.append(asImage(images["render"]))
        writeImage(output / f"canonical{variant['name'].title()}Unlit.png", images["unlit"])
    labels = [item["label"] for item in variants]
    makeGrid(unlitPanels, labels, columns=5, panelSize=panelSize,
             subtitle="No illumination | same front camera and mesh | diffuse colors only").save(output / "canonicalUnlit.png")
    makeGrid(frontPanels, labels, columns=5, panelSize=panelSize,
             subtitle="Same front light, geometry normals and roughness | diffuse colors only").save(output / "candidateOverview.png")
    del renderer
    if validateExperiment(data["experiment"])["identity"] != data["identity"]:
        raise ValueError("Experiment/source inputs changed while rendering")
    report = {"status": "rendered", "inputIdentity": data["identity"], "seconds": perf_counter() - started,
              "device": device, "variants": list(NAMES), "views": views, "canonicalRender": canonical,
              "commonCoverageTexels": int(coverage.sum()), "geometryAndCamerasFrozen": True,
              "modelInferenceRepeated": False, "wildCapNormalAndSpecularUsed": False,
              "shading": {"roughness": .45, "specularMultiplier": .22, "f0": .08,
                          "keyIntensity": 2.5, "ambient": .18, "normalSource": "shared geometry"},
              "limitations": ["All variants are usage-validation candidates, not calibrated reflectance estimates.",
                              "Fixed preview lights are not estimated photograph illumination.",
                              "Input photographs participated in reconstruction; there is no held-out albedo truth.",
                              "Feathering changes the transfer composite only, not WildCap inference.",
                              "WildCap detail normals and specular maps are intentionally not enabled."]}
    if device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    writeArtifactJson(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/wildCapUsageValidation")
    parser.add_argument("--stage", choices=("validate", "export", "render", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imageSize", type=int, default=768)
    parser.add_argument("--panelSize", type=int, default=384)
    parser.add_argument("--detailView", type=int, default=0)
    args = parser.parse_args()
    data = validateExperiment(args.experiment)
    if args.stage in ("export", "all"):
        exportClients(data)
    if args.stage in ("render", "all"):
        renderPresentation(data, device=args.device, imageSize=args.imageSize,
                           panelSize=args.panelSize, detailView=args.detailView)
    print(json.dumps({"status": "complete", "stage": args.stage, "variants": list(NAMES),
                      "experiment": str(data["experiment"])}), flush=True)


if __name__ == "__main__":
    main()
