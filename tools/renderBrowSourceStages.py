"""Render cached UFF geometry stages with explicit observed support, entirely on CPU.

No model is loaded and no missing surface is filled. Gray panels use matched
cameras and a common preset light, not the illumination from the photograph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.adapters.geometry import sampleDiagnosticPartials
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from compareClientAssembly import loadPhotoReference
from renderClientAssembly import ClientPreviewRenderer
from runBrowRefinementExperiment import clayAsset, cropReference, fileHash, makeRow, readArrays, summary


stageLabels = {
    "inputMean": "输入支持均值（缺测留空）",
    "network": "融合网络原始输出",
    "blurred": "blur + 最终采样",
    "savedReconstruction": "旧缓存 UFF 重建",
    "savedAppearance": "旧缓存眼修正后",
}


def supportedMesh(vertices, faces, valid):
    """Drop unsupported triangles and compact away unused/nonfinite vertices."""
    vertices, faces, valid = np.asarray(vertices), np.asarray(faces), np.asarray(valid, bool)
    usable = valid & np.isfinite(vertices).all(axis=1)
    faceIds = np.flatnonzero(usable[faces].all(axis=1))
    selectedFaces = faces[faceIds]
    vertexIds = np.unique(selectedFaces)
    remap = np.full(len(vertices), -1, np.int64)
    remap[vertexIds] = np.arange(len(vertexIds))
    return {"vertices": vertices[vertexIds], "faces": remap[selectedFaces],
            "originalVertexIndices": vertexIds, "originalFaceIndices": faceIds}


def sampleStage(pointMap, mask, uvs):
    sampled = sampleDiagnosticPartials(pointMap[None], mask[None], uvs)
    return {"vertices": sampled["partialVertices"][0], "valid": sampled["partialValid"][0]}


def loadStages(trace, source, appearance):
    """Keep measurement support distinct from the model's completed template."""
    stages = {
        "inputMean": sampleStage(trace["modelInputMean"], trace["inputSupportMask"], trace["uvs"]),
        "network": sampleStage(trace["predUvPointMap"], trace["predValidMask"], trace["uvs"]),
        # This is the actual final sampler output, including its original thresholds.
        "blurred": {"vertices": trace["vertices"], "valid": np.isfinite(trace["vertices"]).all(axis=1)},
        "savedReconstruction": {"vertices": source["vertices"], "valid": np.isfinite(source["vertices"]).all(axis=1)},
        "savedAppearance": {"vertices": appearance["vertices"], "valid": np.isfinite(appearance["vertices"]).all(axis=1)},
    }
    return stages


def renderSupported(renderer, reference, light):
    if renderer is None:
        return Image.new("RGB", (reference["imageShape"][1], reference["imageShape"][0]), "#dbe0e7"), {
            "empty": True, "reason": "No triangle has three supported finite vertices"}
    camera, projection = renderer.photoCamera(reference["camera"], reference["imageShape"])
    colors, visible, stats = renderer.render(camera, reference["imageShape"], light, perspective=True)
    return Image.fromarray(np.rint(colors * 255).astype(np.uint8)), {
        "projectionCheck": projection, "raster": stats, "visiblePixelCount": int(visible.sum())}


def runCase(caseDir, traceDir, regionDir, output, panelWidth):
    manifestPath = traceDir / "manifest.json"
    manifest = json.loads(manifestPath.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Trace is not complete: {traceDir}")
    expectedHash = manifest["stages"]["fusion"]["sha256"]
    if fileHash(traceDir / "fusion.npz") != expectedHash:
        raise ValueError("Fusion trace hash does not match its manifest")
    trace = readArrays(traceDir / "fusion.npz")
    source = readArrays(caseDir / "reconstruction/geometry.npz")
    appearance = readArrays(caseDir / "appearance/geometryRefined.npz")
    region = readArrays(regionDir / "region.npz")
    faces, uvs = trace["faces"], trace["uvs"]
    if not np.array_equal(faces, source["faces"]) or not np.array_equal(faces, appearance["faces"]):
        raise ValueError("Stage topology mismatch")
    if not np.array_equal(uvs, source["uvs"]) or not np.array_equal(uvs, appearance["uvs"]):
        raise ValueError("Stage UV mismatch")
    if len(region["regionWeights"]) != len(uvs):
        raise ValueError("Region vertices do not match stage topology")
    output.mkdir(parents=True, exist_ok=False)
    stages = loadStages(trace, source, appearance)
    partialCounts = trace["partialValid"].sum(axis=0)
    roi = region["regionWeights"] > 0
    seeds = region["eyebrowSeedMask"].astype(bool)
    canonicalSpan = float(np.ptp(trace["canonicalVertices"], axis=0).max())
    references = [loadPhotoReference(caseDir / "appearance", index) for index in (0, 1)]
    semantics = [readArrays(caseDir / "semantics" / f"view{index:02d}.npz") for index in (0, 1)]
    meshes = {name: supportedMesh(stage["vertices"], faces, stage["valid"]) for name, stage in stages.items()}
    renderers = {name: ClientPreviewRenderer(clayAsset(mesh["vertices"], mesh["faces"]),
                    device="cpu", facesPerPixel=2) if len(mesh["faces"]) else None
                 for name, mesh in meshes.items()}
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    light, rows, renderReports = [-.5, .6, 1.], {}, {}
    for viewIndex, viewName in enumerate(("front", "side")):
        reference = cropReference(references[viewIndex], semantics[viewIndex]["landmarks"], panelWidth)
        panels, labels, report = [reference["photo"]], ["原照片"], {}
        for name, renderer in renderers.items():
            panel, statistics = renderSupported(renderer, reference, light)
            panel.save(output / f"{viewName}_{name}.png")
            panels.append(panel)
            labels.append(stageLabels[name])
            report[name] = statistics
        row = makeRow(panels, labels, f"{caseDir.name} | {'正面' if viewIndex == 0 else '侧面'} · 阶段来源对照 · 缺测不补面", font)
        row.save(output / f"{viewName}Comparison.png")
        rows[viewName] = row
        renderReports[viewName] = {"cameraChanged": False, "crop": reference["crop"], "branches": report}
    # Actual partial-view evidence is shown separately; the fusion network may
    # invent a completed surface where these independent panels contain holes.
    reference = cropReference(references[0], semantics[0]["landmarks"], panelWidth)
    panels, labels, partialReports = [reference["photo"]], ["原照片"], []
    for index, vertices in enumerate(trace["partialVertices"]):
        mesh = supportedMesh(vertices, faces, trace["partialValid"][index])
        renderer = ClientPreviewRenderer(clayAsset(mesh["vertices"], mesh["faces"]),
                      device="cpu", facesPerPixel=2) if len(mesh["faces"]) else None
        panel, stats = renderSupported(renderer, reference, light)
        panel.save(output / f"partialView{index:02d}.png")
        panels.append(panel)
        validRoi = int((trace["partialValid"][index] & roi).sum())
        labels.append(f"视角 {int(trace['selectedViewIndices'][index])} · ROI {validRoi}/{int(roi.sum())}")
        partialReports.append({"selectedViewIndex": int(trace["selectedViewIndices"][index]),
            "supportedVertexCount": int(trace["partialValid"][index].sum()),
            "supportedFaceCount": len(mesh["faces"]), "supportedRoiVertexCount": validRoi, "render": stats})
    makeRow(panels, labels, f"{caseDir.name} | 各视角 canonical 局部表面 · 同一正面相机 · 非独立真值", font).save(output / "partialComparison.png")
    # Counts are placed at saved baseline vertices only to identify the same ROI.
    # This overlay is not a projection of the missing partial surface itself.
    supportImage = reference["photo"].copy()
    draw = ImageDraw.Draw(supportImage)
    projected, depth = projectPoints(source["vertices"], **reference["camera"])
    colors = [(185, 30, 48), (240, 145, 35), (244, 210, 57), (55, 178, 101), (25, 113, 204)]
    for vertexId in np.flatnonzero(roi & (depth > 0)):
        x, y = projected[vertexId]
        count = int(partialCounts[vertexId])
        draw.ellipse((x - 1.7, y - 1.7, x + 1.7, y + 1.7), fill=colors[min(count, 4)])
    supportImage.save(output / "partialSupportOverlay.png")
    stageReports = {}
    for name, stage in stages.items():
        valid = stage["valid"]
        observedRoi = roi & valid
        distance = np.linalg.norm(stage["vertices"][observedRoi].astype(float) - source["vertices"][observedRoi], axis=1)
        stageReports[name] = {"validVertexCount": int(valid.sum()), "validFaceCount": len(meshes[name]["faces"]),
            "validRoiVertexCount": int(observedRoi.sum()), "roiVertexCount": int(roi.sum()),
            "validRoiFraction": float(observedRoi.sum() / max(1, roi.sum())),
            "distanceFromSavedReconstructionPercentCanonicalSpan": summary(100 * distance / canonicalSpan),
            "supportMeaning": "Observed UV support only" if name == "inputMean" else "Model/template completion, not measured coverage"}
    stageDifferences = {}
    for before, after in (("inputMean", "network"), ("network", "blurred"),
                          ("blurred", "savedReconstruction"), ("savedReconstruction", "savedAppearance")):
        common = roi & stages[before]["valid"] & stages[after]["valid"]
        difference = np.linalg.norm(stages[after]["vertices"][common].astype(float) - stages[before]["vertices"][common], axis=1)
        stageDifferences[f"{before}To{after}"] = summary(100 * difference / canonicalSpan)
    np.savez_compressed(output / "sampledStages.npz", faces=faces, uvs=uvs, partialCounts=partialCounts,
        regionWeights=region["regionWeights"], **{f"{name}Vertices": stage["vertices"] for name, stage in stages.items()},
        **{f"{name}Valid": stage["valid"] for name, stage in stages.items()})
    report = {"caseId": caseDir.name, "status": "completed", "device": "cpu", "traceManifestSha256": fileHash(manifestPath),
        "fusionSha256": expectedHash, "sourceReconstructionSha256": fileHash(caseDir / "reconstruction/geometry.npz"),
        "sourceAppearanceSha256": fileHash(caseDir / "appearance/geometryRefined.npz"),
        "regionSha256": fileHash(regionDir / "region.npz"), "roiVertexCount": int(roi.sum()),
        "partialCountHistogramRoi": np.bincount(partialCounts[roi], minlength=len(trace["partialValid"]) + 1),
        "partialCountHistogramEyebrowSeeds": np.bincount(partialCounts[seeds], minlength=len(trace["partialValid"]) + 1),
        "partialSupportOverlayColors": {"0": "red", "1": "orange", "2": "yellow", "3": "green", "4OrMore": "blue"},
        "stages": stageReports, "stageDifferencesPercentCanonicalSpan": stageDifferences,
        "partials": partialReports, "render": renderReports, "lightDirectionCanonical": light,
        "limits": ["Photo illumination differs from preset gray-model illumination; no 3D ground truth.",
            "VGGT partial depth is model evidence, not an independent measurement.",
            "Canonical partial maps already include per-view similarity and [-1,1] coordinate clipping.",
            "Missing partial vertices remain NaN; only fully supported triangles are rendered.",
            "Network panels may contain completed regions. Coverage does not mean correctness.",
            "Mean and raw-network stages use the same conservative local sampler (>1 accumulated support).",
            "Blurred stage uses saved actual final vertices and their original sampler thresholds.",
            "ROI is the existing projected semantic-eyebrow neighborhood, not anatomical bone annotation."]}
    writeJson(output / "report.json", report)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=projectRoot / "runs/clientBatchV1")
    parser.add_argument("--traceRoot", type=Path, default=projectRoot / "runs/browNormalRefinementV1")
    parser.add_argument("--regionRoot", type=Path, default=projectRoot / "runs/browRefinementV1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["windowLight4", "diffuseLight4", "backlightGlasses4", "overheadHair4"])
    parser.add_argument("--panelWidth", type=int, default=340)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    reports, rows = {}, {"front": [], "side": []}
    for caseId in args.cases:
        report, caseRows = runCase(args.batch / caseId, args.traceRoot / caseId / "trace",
            args.regionRoot / caseId, args.output / caseId, args.panelWidth)
        reports[caseId] = report
        for name, row in caseRows.items():
            rows[name].append(row)
        writeJson(args.output / "report.json", {"status": "running", "cases": reports})
        print(f"{caseId}: stage comparison complete", flush=True)
    for name, panels in rows.items():
        if panels:
            overview = Image.new("RGB", (max(item.width for item in panels), sum(item.height for item in panels)))
            top = 0
            for panel in panels:
                overview.paste(panel, (0, top))
                top += panel.height
            overview.save(args.output / f"{name}Overview.png")
    writeJson(args.output / "report.json", {"status": "completed", "cases": reports})


if __name__ == "__main__":
    main()
