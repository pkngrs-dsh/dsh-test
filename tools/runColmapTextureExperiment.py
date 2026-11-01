"""Compare cached UFF fusion with COLMAP mesh_texturer; no model inference.

Only a fresh experiment directory is written. Marker and validity passes use
the same geometry-only view selection and padding as the color pass. Direct
source-mask sampling prevents padding from becoming photographic support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.artifacts import exportMesh, writeImage, writeJson
from faceAppearance.clientAsset import exportClientAsset
from faceAppearance.colmapTexture import (prepareWorkspace, loadTexturedMesh, rebakeAtlas,
                                          viewMarkerColors, sourceSupportFromMarkers)
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.texture import fuseTexture


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exportBranch(output, name, result, geometry):
    data = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output / "branches" / f"{name}.npz", **data)
    writeJson(output / "branches" / f"{name}.json", result.get("diagnostics", {}))
    directory = output / "meshes" / name
    directory.mkdir(parents=True)
    mask = data["coverageMask"]
    preview = data["albedo"].copy()
    if mask.any():
        _, nearest = distance_transform_edt(~mask, return_indices=True)
        preview[~mask] = preview[nearest[0][~mask], nearest[1][~mask]]
    writeImage(directory / "albedo.png", data["albedo"])
    writeImage(directory / "albedoPreview.png", preview)
    writeImage(directory / "coverage.png", mask)
    exportMesh(directory, geometry)


def prepare(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Experiment requires an empty output directory")
    result = json.loads((source / "result.json").read_text())
    if result.get("status") != "succeeded":
        raise ValueError("Source reconstruction must have succeeded")
    config = json.loads((source / "config.json").read_text())
    geometry, delight, baseline = (arrays(source / f"{name}.npz") for name in ("geometry", "delight", "texture"))
    cameras = json.loads((source / "cameras.json").read_text())
    resolution = baseline["albedo"].shape[0]
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir()
    sourceNames = ["geometry.npz", "delight.npz", "texture.npz", "cameras.json", "inputs.json",
                   "viewTransforms.json", "config.json", "result.json", "assembly/clientMesh.npz",
                   "assembly/diagnostics.json", "assembly/manifest.json"]
    sourceHashes = {name: digest(source / name) for name in sourceNames}
    for name in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"):
        shutil.copy2(source / name, output / name)
    report = {"status": "preparing", "source": str(source), "sourceHashes": sourceHashes,
              "resolution": resolution, "geometryAndCamerasFrozen": True,
              "sourceSampling": config["textureSampling"],
              "colorEncoding": "OpenDelight native encoded RGB; no extra gamma conversion before COLMAP",
              "assemblyColorEncoding": config["assemblyColorEncoding"],
              "limitations": ["Cached denseBaseline versus projectionMean also changes image sampling, not only fusion.",
                  "ProjectionMean and COLMAP receive the same PNG-quantized OpenDelight RGB.",
                  "COLMAP selects a source image per face, smooths labels, then corrects patch colors; it does not average all source views per texel.",
                  "Mask pass verifies valid source-image support; it is not a calibrated uncertainty estimate.",
                  "Legacy SH delight, Lab brightness repair and multi-band template blending are not reproduced."]}
    writeJson(output / "experimentReport.json", report)
    baseline["diagnostics"] = result["textureDiagnostics"]
    exportBranch(output, "denseBaseline", baseline, geometry)
    quantized = np.rint(delight["albedoImages"] * 255).astype(np.uint8)
    projectionInput = {**delight, "albedoImages": quantized.astype(np.float32) / 255}
    print("Computing frozen-camera projection fusion", flush=True)
    start = perf_counter()
    projection = fuseTexture(geometry, projectionInput, cameras, resolution,
                             colorBalance=config["colorBalance"], samplingMode="projection")
    exportBranch(output, "projectionMean", projection, geometry)
    report["projectionFusionSeconds"] = perf_counter() - start
    report["pngQuantizationMaxError"] = float(np.abs(projectionInput["albedoImages"] - delight["albedoImages"]).max())
    report["colorWorkspace"] = prepareWorkspace(output / "colorWorkspace", geometry, cameras, quantized)
    masks = np.repeat((delight["validMasks"].astype(np.uint8) * 255)[..., None], 3, axis=-1)
    report["supportWorkspace"] = prepareWorkspace(output / "supportWorkspace", geometry, cameras, masks)
    markers = np.zeros_like(quantized)
    activeViews = report["colorWorkspace"]["activeViewIndices"]
    markerColors = viewMarkerColors(len(activeViews))
    for markerIndex, viewIndex in enumerate(activeViews):
        markers[viewIndex] = markerColors[markerIndex]
    report["markerWorkspace"] = prepareWorkspace(output / "markerWorkspace", geometry, cameras, markers)
    report["markerColors"] = markerColors.tolist()
    report["status"] = "prepared"
    writeJson(output / "experimentReport.json", report)
    print(f"Prepared {output}", flush=True)
    return report


def texturer(output, imageId, workspaceName, resultName, *, support=False, executable=None):
    destination = output / resultName
    destination.mkdir(exist_ok=True)
    options = {"texture_scale_factor": 4, "atlas_patch_padding": 50,
               "min_cos_normal_angle": .1, "min_visible_vertices": 3,
               "view_selection_smoothing_iterations": 3, "inpaint_radius": 5,
               "apply_color_correction": 0 if support else 1, "color_correction_regularization": .1,
               "num_threads": 4}
    if executable:
        command = [str(executable), "mesh_texturer", "--input_path", str(output / workspaceName / "mesh.ply"),
                   "--workspace_path", str(output / workspaceName), "--output_path", str(destination)]
    else:
        command = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                   "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=512m", "--mount",
                   f"type=bind,source={output / workspaceName},target=/workspace,readonly", "--mount",
                   f"type=bind,source={destination},target=/output", imageId, "colmap", "mesh_texturer",
                   "--input_path", "/workspace/mesh.ply", "--workspace_path", "/workspace", "--output_path", "/output"]
    for name, value in options.items():
        command.extend([f"--MeshTextureMapping.{name}", str(value)])
    writeJson(destination / "command.json", {"argv": command, "options": options})
    print(f"Running COLMAP {resultName}", flush=True)
    start = perf_counter()
    with (destination / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if completed.returncode:
        raise RuntimeError(f"COLMAP exited {completed.returncode}; see {destination / 'run.log'}")
    return {"seconds": perf_counter() - start, "options": options,
            "log": str(destination / "run.log"), "argv": command}


def finish(source, output, image, report, executable=None):
    for name, expected in report["sourceHashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Cached source changed: {name}")
    imageId = None
    if executable:
        executable = Path(executable).resolve(strict=True)
        helpResult = subprocess.run([str(executable), "mesh_texturer", "--help"], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, check=True, timeout=30)
        (output / "colmapHelp.txt").write_text(helpResult.stdout)
        report["nativeColmap"] = {"path": str(executable), "sha256": digest(executable),
                                  "help": str(output / "colmapHelp.txt")}
    else:
        inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
        imageId = inspected["Id"]
        report["dockerImage"] = {"id": imageId, "repoDigests": inspected.get("RepoDigests", []),
                                 "created": inspected.get("Created"), "architecture": inspected.get("Architecture")}
    report["status"] = "texturing"
    writeJson(output / "experimentReport.json", report)
    report["colorRun"] = texturer(output, imageId, "colorWorkspace", "colorOutput", executable=executable)
    report["supportRun"] = texturer(output, imageId, "supportWorkspace", "supportOutput", support=True, executable=executable)
    report["markerRun"] = texturer(output, imageId, "markerWorkspace", "markerOutput", support=True, executable=executable)
    geometry = arrays(source / "geometry.npz")
    colorMesh = loadTexturedMesh(output / "colorOutput/mesh.ply", geometry)
    maskMesh = loadTexturedMesh(output / "supportOutput/mesh.ply", geometry)
    markerMesh = loadTexturedMesh(output / "markerOutput/mesh.ply", geometry)
    # Image content must not change geometry-only view selection/atlas packing.
    for otherMesh in (maskMesh, markerMesh):
        if (not np.array_equal(colorMesh["validFaces"], otherMesh["validFaces"])
                or not np.array_equal(colorMesh["faceUvs"], otherMesh["faceUvs"])):
            raise ValueError("Auxiliary atlas layout differs; cannot infer photo support")
    report["supportAtlasLayoutIdentical"] = True
    with Image.open(output / "colorOutput/texture.png") as imageFile:
        colorAtlas = np.asarray(imageFile.convert("RGB"))
    with Image.open(output / "supportOutput/texture.png") as imageFile:
        maskAtlas = np.asarray(imageFile.convert("RGB"))
    with Image.open(output / "markerOutput/texture.png") as imageFile:
        markerAtlas = np.asarray(imageFile.convert("RGB"))
    if colorAtlas.shape != maskAtlas.shape or colorAtlas.shape != markerAtlas.shape:
        raise ValueError("Color and auxiliary atlas sizes differ")
    support = rebakeAtlas(geometry, maskMesh, maskAtlas, report["resolution"])
    sourceSupport = sourceSupportFromMarkers(geometry, markerMesh, markerAtlas,
        json.loads((source / "cameras.json").read_text()), arrays(source / "delight.npz")["validMasks"],
        report["colorWorkspace"]["activeViewIndices"], resolution=report["resolution"])
    observed = (sourceSupport["coverageMask"] & support["assignedMask"]
                & (support["rawRebaked"].min(axis=-1) >= .999))
    if not observed.any():
        raise ValueError("COLMAP produced no qualified photo support after rebaking")
    texture = rebakeAtlas(geometry, colorMesh, colorAtlas, report["resolution"],
                         coverage={"coverageMask": observed, "sourceIndex": sourceSupport["sourceIndex"]})
    texture["faceSourceIndex"] = sourceSupport["faceSourceIndex"]
    texture["supportFraction"] = support["rawRebaked"][..., 0]
    texture["diagnostics"].update(photoSupportMethod="Direct source-mask sampling from recovered face view AND matching validity atlas with the same inpaint radius as RGB; both >= .999",
                                   sourceViewIdentityAvailable=True,
                                   sourceSupport=sourceSupport["diagnostics"])
    exportBranch(output, "colmap", texture, geometry)
    report["colmapMesh"] = colorMesh["diagnostics"]
    report["colmapRebake"] = texture["diagnostics"]
    print("Exporting client template asset", flush=True)
    config = json.loads((source / "config.json").read_text())
    assetRoot = Path(config["assemblyAsset"])
    if not assetRoot.is_absolute():
        assetRoot = Path(config["projectRoot"]) / assetRoot
    assets = AssemblyAssets(assetRoot)
    baselineDiagnostics = json.loads((source / "assembly/diagnostics.json").read_text())
    if assets.manifestSha256 != baselineDiagnostics["assetManifestSha256"]:
        raise ValueError("Client template settings differ from the cached baseline")
    report["assemblyManifestSha256"] = assets.manifestSha256
    assembly = assets.assemble(geometry, texture, colorEncoding=config["assemblyColorEncoding"])
    exportClientAsset(output / "assembly", assembly)
    oldMesh, newMesh = arrays(source / "assembly/clientMesh.npz"), arrays(output / "assembly/clientMesh.npz")
    geometryChecks = {key: np.array_equal(oldMesh[key], newMesh[key]) for key in
                      ("vertices", "templateVertices", "faces", "uvs", "uvFaces", "materialIds")}
    if not all(geometryChecks.values()):
        raise ValueError(f"Client geometry changed: {geometryChecks}")
    report["identicalClientArrays"] = geometryChecks
    report["clientZipSha256"] = digest(output / "assembly/clientAsset.zip")
    report["sourceFilesUnchanged"] = all(digest(source / name) == expected for name, expected in report["sourceHashes"].items())
    if not report["sourceFilesUnchanged"]:
        raise ValueError("Source artifacts changed during the experiment")
    codeFiles = [Path(__file__), ROOT / "src/faceAppearance/colmapTexture.py", ROOT / "src/faceAppearance/texture.py",
                 ROOT / "src/faceAppearance/headAssembly.py", ROOT / "src/faceAppearance/clientAsset.py"]
    report["implementationHashes"] = {str(path.relative_to(ROOT)): digest(path) for path in codeFiles}
    report["coverage"] = {}
    for name in ("denseBaseline", "projectionMean", "colmap"):
        count = int(arrays(output / "branches" / f"{name}.npz")["coverageMask"].sum())
        report["coverage"][name] = {"observedTexels": count, "fractionOfFaceAtlas": count / texture["atlasMask"].sum()}
    report["status"] = "succeeded"
    writeJson(output / "experimentReport.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "coverage": report["coverage"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-image", default="colmap/colmap:latest")
    parser.add_argument("--colmap-executable", type=Path, help="Use an existing native COLMAP CLI instead of Docker")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an already prepared experiment after inspecting its report")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.resume:
        report = json.loads((output / "experimentReport.json").read_text())
        if report["source"] != str(source) or report["status"] not in {"prepared", "texturing"}:
            raise ValueError("Only a matching prepared/unfinished experiment can resume")
    else:
        report = prepare(source, output)
    if not args.prepare_only:
        finish(source, output, args.docker_image, report, args.colmap_executable)


if __name__ == "__main__":
    main()
