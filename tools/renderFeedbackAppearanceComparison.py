"""Compare completed appearance candidates with delivered ZIPs on saved cameras.

Read-only CPU rendering; no reconstruction, camera fit, texture edits or service
requests. A completed experiment report and unchanged non-head payloads are
required before the candidate can be rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saveSrgb(image, path):
    info = PngImagePlugin.PngInfo()
    info.add(b"sRGB", b"\x00")
    image.save(path, pnginfo=info)


def run(caseDirectory, experimentDirectory, outputDirectory, size=320, candidateLabel="SH 候选",
        baselineArchive=None):
    case, experiment, output = map(Path, (caseDirectory, experimentDirectory, outputDirectory))
    candidateReport = json.loads((experiment / "report.json").read_text())
    if candidateReport.get("status") != "complete" or candidateReport.get("sourceTaskId") != case.name:
        raise ValueError("Candidate must be complete and belong to this task")
    archives = {"production": Path(baselineArchive) if baselineArchive is not None else case / "asset/legacyAsset.zip",
                "sh": experiment / "asset/legacyAsset.zip"}
    hashes = {key: fileHash(path) for key, path in archives.items()}
    sourceKey = 'parentArchiveSha256' if baselineArchive is not None else 'sourceArchiveSha256'
    if candidateReport[sourceKey] != hashes["production"]:
        raise ValueError("Candidate source differs from the delivered production ZIP")
    if candidateReport["variant"]["sha256"] != hashes["sh"]:
        raise ValueError("Candidate ZIP differs from its completed experiment report")
    assets = {key: _load(path) for key, path in archives.items()}
    if assets["production"].payloads.keys() != assets["sh"].payloads.keys():
        raise ValueError("Candidate payload names changed")
    unchanged = {name: payload == assets["sh"].payloads[name]
                 for name, payload in assets["production"].payloads.items()
                 if name not in ("head_color.png", "face_nicp.png")}
    if not all(unchanged.values()):
        raise ValueError("Appearance-only comparison requires identical geometry and other materials")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty rendering output directory")
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    recon = case / "reconstruction"
    cameras = json.loads((recon / "cameras.json").read_text())
    inputs = json.loads((recon / "inputs.json").read_text())
    if len(cameras) != len(inputs):
        raise ValueError("Camera/input count mismatch")
    similarity = assets["production"].manifest["assemblyDiagnostics"]["templateToUff"]
    renderers = {key: EvidenceRenderer(path, similarity, device="cpu", facesPerPixel=12)
                 for key, path in archives.items()}
    views, allPanels, allLabels = [], [], []
    for index, (camera, record) in enumerate(zip(cameras, inputs)):
        if not camera.get("valid") or camera.get("viewIndex", index) != index:
            raise ValueError("Invalid saved camera or view order")
        photoPath = recon / f"input{index:02d}.png"
        with Image.open(photoPath) as source:
            if list(source.size) != record["workingSize"]:
                raise ValueError("Saved working photo dimensions differ")
            photo = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        sx, sy = size / record["workingSize"][0], size / record["workingSize"][1]
        displayTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])
        scaled = {key: np.asarray(camera[key], np.float64) for key in ("intrinsics", "rotation", "translation")}
        scaled["intrinsics"] = displayTransform @ scaled["intrinsics"]
        images, stats, filenames = {}, {}, {}
        photoName = f"view{index:02d}_photo.png"
        saveSrgb(photo, output / photoName)
        for branch, renderer in renderers.items():
            images[branch], stats[branch] = renderer.variants(scaled, (size, size))
            filenames[branch] = {}
            for mode in ("baseColor", "lit", "clay"):
                filename = f"view{index:02d}_{branch}_{mode}.png"
                saveSrgb(images[branch][mode], output / filename)
                filenames[branch][mode] = filename
        if not np.array_equal(images["production"]["clay"], images["sh"]["clay"]):
            raise ValueError("Identical geometry/alpha must produce identical clay renders")
        panels, labels = [], []
        for mode, modeLabel in (("baseColor", "无光照基色"), ("lit", "共同 PBR 光照")):
            panels.extend([photo, solid(images["production"][mode]), solid(images["sh"][mode])])
            baselineLabel = '上一版完整候选' if baselineArchive is not None else '生产 OpenDelight'
            labels.extend([f"照片 {index + 1} · 原生产工作图", f"{baselineLabel} · {modeLabel}", f"{candidateLabel} · {modeLabel}"])
        comparison = f"view{index:02d}_comparison.png"
        saveSrgb(evidenceGrid(panels, labels, columns=3, panelSize=size,
            subtitle=f"{case.name[:8]} / 视角 {index + 1} | 保存相机；同一几何；原照片不是反照率真值"), output / comparison)
        views.append({"viewIndex": index, "photo": photoName, "photoSourceSha256": fileHash(photoPath),
                      "comparison": comparison, "renders": filenames, "stats": stats,
                      "camera": {key: value.tolist() for key, value in scaled.items()},
                      "displayPixelTransform": displayTransform.tolist(), "clayPixelsIdentical": True})
        allPanels.extend(panels); allLabels.extend(labels)
        print(json.dumps({"case": case.name[:8], "view": index, "seconds": round(perf_counter() - start, 2)}), flush=True)
    saveSrgb(evidenceGrid(allPanels, allLabels, columns=3, panelSize=size,
        subtitle="每视角上行无光照基色，下行共同 PBR 光照；相机/几何一致；未拟合照片光照"), output / "overview.png")
    report = {"status": "complete", "sourceTaskId": case.name, "seconds": perf_counter() - start,
              "size": size, "device": "cpu", "candidateLabel": candidateLabel,
              "sourceArchiveSha256": hashes,
              "sourceExperimentReportSha256": fileHash(experiment / "report.json"),
              "cameraSourceSha256": fileHash(recon / "cameras.json"),
              "inputsSourceSha256": fileHash(recon / "inputs.json"),
              "protectedPayloadsByteIdentical": unchanged, "views": views, "overview": "overview.png",
              "cameraRefitted": False, "geometryModified": False, "photoColorModified": False,
              "renderer": "Existing EvidenceRenderer CPU, 12 alpha-tested depth layers, perspective correct UVs",
              "lighting": {"baseColor": "unlit", "lit": "Lambert + GGX/Smith/Schlick",
                           "keyDirectionCamera": [-.5, -.5, -1.], "keyIntensity": 2.5, "ambient": .22,
                           "exposureAdjustment": False, "photoIlluminationFitted": False},
              "limitations": ["Production working PNGs intentionally preserve the old ICC handling, isolating the SH experiment.",
                               ("Local appearance candidate compared against its recorded parent archive." if baselineArchive is not None
                                else "SH candidate also recalculates fallback color gain; it is not just a display brightness adjustment."),
                               "Input photographs contain illumination and are not measured albedo ground truth.",
                               "This inspection light is not the original scene light and does not reproduce the external client renderer.",
                               "No cast shadows, ambient occlusion, subsurface scattering, environment light or normal map."]}
    writeJson(output / "report.json", report)
    print(json.dumps({"status": "complete", "output": str(output), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--candidate-label", default="SH 候选")
    parser.add_argument("--baseline-archive", dest="baselineArchive", type=Path)
    args = parser.parse_args()
    run(args.case.resolve(), args.experiment.resolve(), args.output.resolve(), args.size, args.candidate_label,
        args.baselineArchive)
