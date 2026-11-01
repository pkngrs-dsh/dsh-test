"""Compare local brow fairing on saved UFF meshes without neural inference.

Semantic eyebrows select a surface neighborhood, not anatomical ground truth.
All branches retain source topology, cameras, and reference-view projections.
The gray renders are geometry comparisons under preset, unmatched illumination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import exportMesh, writeJson
from faceAppearance.browRegion import buildBrowRegion
from faceAppearance.camera import projectPoints
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.surfaceFairing import refineSurfaceAlongDirections
from faceAppearance.texture import vertexNormals
from compareClientAssembly import loadPhotoReference
from renderClientAssembly import ClientPreviewRenderer


profiles = {"mild": 8., "medium": 64., "strong": 512.}
branchLabels = {"baseline": "当前 UFF", "mild": "平滑权重 8", "medium": "平滑权重 64", "strong": "平滑权重 512"}


def readArrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "maximum": float(values.max())}


def clayAsset(vertices, faces):
    return {"vertices": vertices, "faces": faces,
        "previewUvsVDown": np.full((len(vertices), 2), .5, np.float32),
        "previewUvFaces": faces,
        "atlasLinearRgba": np.tile([.4, .4, .4, 1.], (2, 2, 1)).astype(np.float32),
        "faceMaterialIds": np.zeros(len(faces), np.int64), "alphaCutoff": np.array([0.]),
        "roughness": np.array([.6]), "metallic": np.array([0.])}


def cropReference(reference, eyeLandmarks, panelWidth):
    eyeDistance = float(np.linalg.norm(eyeLandmarks[0] - eyeLandmarks[1]))
    eyeCenter = np.mean(eyeLandmarks[:2], axis=0)
    crop = np.r_[np.floor(eyeCenter + [-1.12 * eyeDistance, -.78 * eyeDistance]),
                 np.ceil(eyeCenter + [1.12 * eyeDistance, .40 * eyeDistance])].astype(int).tolist()
    photo = reference["photo"].crop(crop)
    panelHeight = max(32, round(photo.height * panelWidth / photo.width))
    sx, sy = panelWidth / photo.width, panelHeight / photo.height
    affine = np.array([[sx, 0, -sx * crop[0] + (sx - 1) / 2],
                       [0, sy, -sy * crop[1] + (sy - 1) / 2], [0, 0, 1.]])
    return {"photo": photo.resize((panelWidth, panelHeight), Image.Resampling.LANCZOS),
        "camera": {**reference["camera"], "intrinsics": affine @ reference["camera"]["intrinsics"]},
        "imageShape": (panelHeight, panelWidth), "crop": crop, "affine": affine}


def makeRow(panels, labels, title, font):
    width, height = panels[0].size
    row = Image.new("RGB", (width * len(panels), height + 62), "#eef0f4")
    draw = ImageDraw.Draw(row)
    draw.text((8, 3), title, font=font, fill="#172434")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        row.paste(panel, (index * width, 62))
        draw.text((index * width + 8, 33), label, font=font, fill="#172434")
    return row


def renderCase(output, caseId, geometries, references, semantics, region, panelWidth):
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    light = [-.5, .6, 1.]
    renderers = {name: ClientPreviewRenderer(clayAsset(mesh["vertices"], mesh["faces"]),
                                            device="cpu", facesPerPixel=2)
                 for name, mesh in geometries.items()}
    viewReports, viewRows = {}, {}
    for viewIndex, viewName in ((0, "front"), (1, "side")):
        reference = cropReference(references[viewIndex], semantics[viewIndex]["landmarks"], panelWidth)
        panels, labels, branchReports = [reference["photo"]], ["原照片"], {}
        for name, renderer in renderers.items():
            camera, check = renderer.photoCamera(reference["camera"], reference["imageShape"])
            values, visible, stats = renderer.render(camera, reference["imageShape"], light, perspective=True)
            if not visible.any():
                raise ValueError(f"Empty brow render: {caseId}/{viewName}/{name}")
            panel = Image.fromarray(np.rint(values * 255).astype(np.uint8))
            panel.save(output / f"{viewName}_{name}.png")
            panels.append(panel)
            labels.append(branchLabels[name])
            branchReports[name] = {"projectionCheck": check, "raster": stats}
        title = f"{caseId} | {'正面' if viewIndex == 0 else '侧面'} · 相同相机/灯光 · 灰模几何对照"
        row = makeRow(panels, labels, title, font)
        row.save(output / f"{viewName}Comparison.png")
        viewRows[viewName] = row
        viewReports[viewName] = {"viewIndex": viewIndex, "originalCrop": reference["crop"],
            "originalToRasterPixelCenters": reference["affine"], "cameraPoseChanged": False,
            "lightDirectionCanonical": light, "branches": branchReports}
    # Exact reference-view selection can be inspected independently of gray shading.
    reference = cropReference(references[0], semantics[0]["landmarks"], panelWidth * 2)
    photo = reference["photo"].copy()
    draw = ImageDraw.Draw(photo)
    points, depth = projectPoints(geometries["baseline"]["vertices"], **reference["camera"])
    weights = region["regionWeights"]
    for index in np.flatnonzero(weights > 0):
        x, y = points[index]
        if depth[index] > 0:
            color = (int(255 * weights[index]), int(220 * (1 - weights[index])), 75)
            draw.ellipse((x - 1.3, y - 1.3, x + 1.3, y + 1.3), fill=color)
    photo.save(output / "regionOverlay.png")
    writeJson(output / "renderReport.json", viewReports)
    return viewRows


def runCase(caseDir, output, *, panelWidth, render):
    started = perf_counter()
    output.mkdir(parents=True)
    sourcePath = caseDir / "appearance/geometryRefined.npz"
    source = readArrays(sourcePath)
    vertices, faces = source["vertices"], source["faces"]
    references = [loadPhotoReference(caseDir / "appearance", index) for index in range(4)]
    semantics = [readArrays(caseDir / "semantics" / f"view{index:02d}.npz") for index in range(4)]
    camera = references[0]["camera"]
    boundaryPath = projectRoot / "assets/anatomical/uff6986/boundaries.json"
    loops = validateBoundaryAsset(vertices, faces, json.loads(boundaryPath.read_text()))
    eyeIds = np.unique(np.concatenate([ids for name, ids in loops.items() if name.endswith("Eye")]))
    visibility = visibleVertices(vertices, faces, [camera], [references[0]["imageShape"]],
                                 minimumCosine=.15, visibilityMode="exact")[0]
    eyeDistance = float(np.linalg.norm(semantics[0]["landmarks"][0] - semantics[0]["landmarks"][1]))
    region = buildBrowRegion(vertices, faces, camera, semantics[0]["labelMap"], semantics[0]["confidence"],
                            visibility, eyeIds, eyeDistancePixels=eyeDistance)
    np.savez_compressed(output / "region.npz", **{key: value for key, value in region.items() if isinstance(value, np.ndarray)})
    writeJson(output / "regionReport.json", region["diagnostics"])
    weights = region["regionWeights"]
    if not np.any(weights > 0):
        raise ValueError("No trusted brow surface selected")
    cameraCenter = -np.asarray(camera["translation"]) @ np.asarray(camera["rotation"])
    directions = vertices.astype(np.float64) - cameraCenter
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    normals = vertexNormals(vertices, faces)
    extent = float(np.ptp(vertices, axis=0).max())
    branches, branchReports = {"baseline": source}, {}
    for name, smoothWeight in profiles.items():
        fitted = refineSurfaceAlongDirections(vertices, faces, directions, weights,
                                               smoothWeight=smoothWeight, maxDisplacementFraction=.015)
        changed = fitted["vertices"]
        if not np.array_equal(changed[weights == 0], vertices[weights == 0]):
            raise AssertionError("Unselected vertices moved")
        if not np.array_equal(changed[eyeIds], vertices[eyeIds]):
            raise AssertionError("Eye boundary moved")
        target = {key: changed if key == "vertices" else source[key] for key in ("vertices", "faces", "uvs")}
        directory = output / name
        directory.mkdir()
        np.savez_compressed(directory / "geometry.npz", **target)
        exportMesh(directory, target, includeMaterial=False)
        normalsAfter = vertexNormals(changed, faces)
        angles = np.degrees(np.arccos(np.clip((normals * normalsAfter).sum(1), -1, 1)))
        displacement = np.linalg.norm(changed.astype(float) - vertices, axis=1)
        projectionChanges = []
        for index, reference in enumerate(references):
            before, beforeDepth = projectPoints(vertices, **reference["camera"])
            after, afterDepth = projectPoints(changed, **reference["camera"])
            if np.any((beforeDepth > 0) & (afterDepth <= 0)):
                raise AssertionError("Previously positive-depth vertex crossed a camera plane")
            valid = (weights > 0) & (beforeDepth > 0) & (afterDepth > 0)
            drift = np.linalg.norm(after[valid] - before[valid], axis=1)
            projectionChanges.append({"viewIndex": index, **summary(drift)})
            if index == 0 and len(drift) and drift.max() > .002:
                raise AssertionError("Reference projection changed beyond float-cast tolerance")
        visibilityAfter = visibleVertices(changed, faces, [camera], [references[0]["imageShape"]],
                                          minimumCosine=0., visibilityMode="exact")[0]
        lostSeedCount = int((region["eyebrowSeedMask"] & ~visibilityAfter).sum())
        if lostSeedCount:
            raise AssertionError("A trusted eyebrow seed lost front-facing depth visibility")
        branchReports[name] = {"smoothWeight": smoothWeight, "solver": fitted["diagnostics"],
            "selectedVertexCount": int(np.count_nonzero(weights)),
            "maximumDisplacementPercentOfHeadSpan": 100 * float(displacement.max()) / extent,
            "surfaceNormalChangeDegrees": summary(angles[weights > 0]),
            "projectionDriftOriginalPixels": projectionChanges,
            "positiveCameraDepthPreserved": True, "newlyOccludedEyebrowSeeds": lostSeedCount,
            "eyeBoundaryExactlyUnchanged": True, "outsideRegionExactlyUnchanged": True,
            "topologyAndUvExactlyUnchanged": True}
        writeJson(directory / "report.json", branchReports[name])
        branches[name] = target
        print(caseDir.name, name, json.dumps({"maxShiftPercent": branchReports[name]["maximumDisplacementPercentOfHeadSpan"],
              "normalP95": branchReports[name]["surfaceNormalChangeDegrees"]["p95"]}), flush=True)
    rows = renderCase(output, caseDir.name, branches, references, semantics, region, panelWidth) if render else {}
    report = {"status": "completed", "source": sourcePath, "sourceSha256": fileHash(sourcePath),
        "eyeBoundaryAssetSha256": fileHash(boundaryPath), "referenceHashes": [item["sourceHashes"] for item in references],
        "semanticsSha256": [fileHash(caseDir / "semantics" / f"view{index:02d}.npz") for index in range(4)],
        "baseline": "Saved final UFF geometry before client transfer; existing eye refinement retained",
        "region": region["diagnostics"], "branches": branchReports, "seconds": perf_counter() - started,
        "limits": ["Semantic eyebrows are region seeds, not anatomical brow-ridge truth.",
            "Fairing is a geometric prior, not a recovery of independently observed bone depth.",
            "Side-view projection drift is displacement, not a photo error score.",
            "Source textures were not resampled; comparisons use uniform gray only.",
            "No client transfer or production configuration changed."]}
    writeJson(output / "report.json", report)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=projectRoot / "runs/clientBatchV1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["windowLight4", "diffuseLight4", "backlightGlasses4", "overheadHair4"])
    parser.add_argument("--panelWidth", type=int, default=380)
    parser.add_argument("--noRender", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new experiment output directory")
    args.output.mkdir(parents=True)
    reports, rows = {}, {"front": [], "side": []}
    for caseId in args.cases:
        report, caseRows = runCase(args.batch / caseId, args.output / caseId,
                                  panelWidth=args.panelWidth, render=not args.noRender)
        reports[caseId] = report
        for name, row in caseRows.items():
            rows[name].append(row)
        writeJson(args.output / "report.json", {"status": "running", "profiles": profiles, "cases": reports})
    for name, panels in rows.items():
        if panels:
            overview = Image.new("RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels)))
            top = 0
            for panel in panels:
                overview.paste(panel, (0, top))
                top += panel.height
            overview.save(args.output / f"{name}Overview.png")
    writeJson(args.output / "report.json", {"status": "completed", "profiles": profiles, "cases": reports,
        "implementationHashes": {str(path.relative_to(projectRoot)): fileHash(path) for path in
            (Path(__file__), projectRoot / "src/faceAppearance/browRegion.py", projectRoot / "src/faceAppearance/surfaceFairing.py")}})


if __name__ == "__main__":
    main()
