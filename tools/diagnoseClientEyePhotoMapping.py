"""Project frozen eye mappings and current client geometry onto the same photos.

Diagnostic only: no camera fitting, geometry changes, inference or asset export.
Client topology-hole rims are hidden rear openings, not visible eyelid margins.
Template X-extrema corners and cached detector curves remain explicit proxies.
"""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.correspondence import sampleSurface
from runEyeRefinementExperiment import loadEyeCache
from runReferenceExperiment import fileHash, readArrays
from renderCropComparison import photoCamera, validatePhotoPair


def project(points, camera):
    xy, depth = projectPoints(points, camera["intrinsics"], camera["rotation"], camera["translation"])
    if np.any(depth <= 0):
        raise ValueError("Eye diagnostic point is behind the fixed camera")
    return xy


def resample(points, count=64):
    points = np.asarray(points, float)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-9]
    if distance[-1] < 1e-8:
        raise ValueError("Collapsed eye path")
    sample = np.linspace(0, distance[-1], count)
    return np.stack([np.interp(sample, distance[keep], points[keep, axis]) for axis in (0, 1)], 1)


def shapeMetrics(curves, corners, observed):
    upper, lower = [resample(curves[key]) for key in ("upper", "lower")]
    targetUpper, targetLower = [resample(observed[key+"Curve"]) for key in ("upper", "lower")]
    targetCorners = np.array([observed["innerCorner"], observed["outerCorner"]])
    axis = targetCorners[1]-targetCorners[0]
    width = np.linalg.norm(axis)
    down = np.array([-axis[1], axis[0]]) / width
    if down[1] < 0:
        down *= -1
    opening = np.max((lower-upper) @ down)
    targetOpening = np.max((targetLower-targetUpper) @ down)
    error = np.linalg.norm(corners-targetCorners, axis=1)
    return {"cornerErrorPixels": error, "meanCornerErrorPixels": float(error.mean()),
            "upperArcSampleMeanErrorPixels": float(np.linalg.norm(upper-targetUpper, axis=1).mean()),
            "lowerArcSampleMeanErrorPixels": float(np.linalg.norm(lower-targetLower, axis=1).mean()),
            "cornerWidthPixels": float(np.linalg.norm(corners[1]-corners[0])),
            "detectorWidthPixels": float(width), "openingPixels": float(opening),
            "detectorOpeningPixels": float(targetOpening), "openingRatioToDetector": float(opening/targetOpening),
            "meaning": "Topology-hole path against detector; client rear opening is not visible eyelid. These are diagnostic offsets, not anatomical accuracy"}


def run(args):
    source, asset, template, output = [p.resolve() for p in (args.source, args.asset, args.template, args.output)]
    if (output / "photoMapping.json").exists():
        raise FileExistsError("Choose a fresh diagnostic output directory")
    mappingPath = args.mapping.resolve()
    mapping = json.loads(mappingPath.read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = [photoCamera(camera, item, "fitted") for camera, item in
               zip(json.loads((source / "cameras.json").read_text()), inputs)]
    observations, _, cacheHashes = loadEyeCache(args.cache.resolve(), inputs, minimumRawPeak=.6)
    data = readArrays(template / "template.npz")
    geometry, client = readArrays(source / "geometry.npz"), readArrays(asset / "clientMesh.npz")
    original = readArrays(source / "assembly/clientMesh.npz")
    if not all(np.array_equal(value, original[key]) for key, value in client.items()):
        raise ValueError("This diagnostic requires the unchanged current service client geometry")
    transform = json.loads((asset / "diagnostics.json").read_text())["templateToUff"]
    def place(value):
        return value @ np.asarray(transform["rotation"]) * transform["scale"] + transform["translation"]
    stages = {"restTemplate": place(data["vertices"][data["nicpToFullIds"]]),
              "uff": geometry["vertices"],
              "mappedTarget": sampleSurface(geometry["vertices"], geometry["faces"], data),
              "finalClient": place(client["vertices"][data["nicpToFullIds"]]*100)}
    sourceFiles = [source/name for name in ("geometry.npz", "cameras.json", "inputs.json")]
    sourceFiles += [asset / "clientMesh.npz", asset / "diagnostics.json", template / "template.npz", mappingPath]
    hashes = {str(path): fileHash(path) for path in sourceFiles}
    eyes = mapping["eyes"]
    names, labels = list(eyes), list(observations[0]["eyes"])
    centers = [project(stages["restTemplate"][eyes[name]["clientNicpVertexIds"]], cameras[0]).mean(0) for name in names]
    targets = {label: (observations[0]["eyes"][label]["innerCorner"]+observations[0]["eyes"][label]["outerCorner"])/2 for label in labels}
    costs = [(sum(np.linalg.norm(center-targets[label]) for center,label in zip(centers, perm)), perm)
             for perm in permutations(labels)]
    costs.sort()
    labelMapping = dict(zip(names, costs[0][1]))
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    stageTitles = {"restTemplate": "Template rear opening (hidden)", "uff": "UFF original eye-hole boundary",
                   "mappedTarget": "Rear-hole UV targets (unweighted)", "finalClient": "Final rear opening (hidden)"}
    records, overviews = [], []
    for viewIndex, (item, camera, observation) in enumerate(zip(inputs, cameras, observations)):
        photo = validatePhotoPair(item, item)
        projectedStages = {}
        allPoints = []
        for stage, vertices in stages.items():
            projectedStages[stage] = {}
            for name, eye in eyes.items():
                isUff = stage == "uff"
                paths = eye["sourcePaths"] if isUff else eye["clientPaths"]
                suffix = "VertexIds" if isUff else "NicpVertexIds"
                corners = eye["sourceCorners"] if isUff else eye["clientCorners"]
                cornerSuffix = "Vertex" if isUff else "NicpVertex"
                curves = {part: project(vertices[paths[part+suffix]], camera) for part in ("upper", "lower")}
                points = project(vertices[[corners["inner"+cornerSuffix], corners["outer"+cornerSuffix]]], camera)
                observed = observation["eyes"][labelMapping[name]]
                row = {"viewIndex": viewIndex, "eye": name, "labelId": labelMapping[name], "stage": stage,
                       "corners": points, "curves": curves, "detectorAccepted": observed["valid"]}
                if observed["valid"]:
                    row["metrics"] = shapeMetrics(curves, points, observed)
                    allPoints += [np.asarray(observed["upperCurve"]), np.asarray(observed["lowerCurve"])]
                else:
                    row["rejectionReason"] = observed["reason"]
                allPoints += list(curves.values())
                records.append(row); projectedStages[stage][name] = row
        allPoints = np.concatenate(allPoints)
        minimum = np.maximum(np.floor(allPoints.min(0)-14), 0).astype(int)
        maximum = np.minimum(np.ceil(allPoints.max(0)+14), photo.size).astype(int)
        crop = (*minimum.tolist(), *maximum.tolist())
        width, height = maximum-minimum
        scale = 510/width
        panelHeight = round(height*scale)
        canvas = Image.new("RGB", (510*4, panelHeight+100), "#141e24")
        for column, (stage, title) in enumerate(stageTitles.items()):
            tile = photo.crop(crop).resize((510, panelHeight), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(tile)
            def xy(points):
                return [tuple(value) for value in (np.asarray(points)-minimum+.5)*scale-.5]
            for name, eye in eyes.items():
                row = projectedStages[stage][name]
                observed = observation["eyes"][labelMapping[name]]
                if observed["valid"]:
                    for part in ("upperCurve", "lowerCurve"):
                        draw.line(xy(observed[part]), fill="#ffdb54", width=2)
                for part in ("upper", "lower"):
                    draw.line(xy(row["curves"][part]), fill="#35e4ed", width=2)
                for index, point in enumerate(xy(row["corners"])):
                    x,y=point;draw.ellipse((x-4,y-4,x+4,y+4), outline="#ffffff", width=2)
                    draw.text((x+5,y-16), "I" if index==0 else "O", font=font, fill="white")
                if stage == "mappedTarget":
                    ids = np.asarray(eye["clientNicpVertexIds"])
                    coords = xy(project(stages[stage][ids], camera))
                    for vertex, (x,y) in zip(ids,coords):
                        color = "#75ff73" if data["geometryWeights"][vertex]>0 else "#ff667a"
                        draw.ellipse((x-3,y-3,x+3,y+3), fill=color)
            canvas.paste(tile, (column*510, 62))
            labelDraw = ImageDraw.Draw(canvas)
            labelDraw.text((column*510+10, 10), title, font=font, fill="white")
            labelDraw.text((column*510+10, 33), "Yellow: detector | Cyan: geometry", font=font, fill="#b5cad2")
        ImageDraw.Draw(canvas).text((10,panelHeight+72),
            f"Photo {viewIndex+1} | Same saved camera | UV target dots: green=active, red=zero weight | I/O: corner proxies",
            font=font, fill="#cad8df")
        canvas.save(output / f"view{viewIndex:02d}EyeMapping.png")
        overviews.append(canvas)
    maxHeight=sum(im.height for im in overviews)
    grid=Image.new("RGB", (2040,maxHeight),"#141e24")
    cursor=0
    for im in overviews:
        grid.paste(im,(0,cursor));cursor+=im.height
    grid.save(output / "photoMappingOverview.png")
    if any(fileHash(Path(path)) != value for path,value in hashes.items()):
        raise ValueError("Frozen source changed during diagnostic")
    report = {"status":"completed", "sourceRun": str(source), "clientAsset": str(asset),
              "sourceHashes": hashes, "eyeCacheHashes": cacheHashes, "eyeCache": str(args.cache.resolve()),
              "labelMapping": labelMapping, "referenceAssignmentCostPixels": [float(v[0]) for v in costs],
              "placement": transform, "cameras": cameras, "records": records,
              "geometryModified": False, "cameraRefitted": False,
              "clientBoundaryScope": "Hidden rear topological opening, not visible eyelid margin; see visibleEyeMapping.json for exact ray evidence",
              "limitations": ["Static template X extrema and detector curves are proxies, not manually labeled ground truth",
                               "Side-view detector rejection is respected; no new depth/occlusion certification",
                               "Mapped targets include zero-weight locations only for diagnosis; they are not all applied constraints",
                               "Client opening width/height and detector offsets must not be interpreted as visible eyelid measurements",
                               "Photo expressions and saved-camera residuals remain in projection error"]}
    writeJson(output / "photoMapping.json", report)
    (output / "diagnoseClientEyePhotoMapping.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status":"completed", "output":str(output), "frontal":[
        {"stage":r["stage"],"eye":r["eye"],**r["metrics"]} for r in records if r["viewIndex"]==0]}, default=lambda a:a.tolist()),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"runs/service/b775794b61894a8b804fddb0620db45d/output")
    parser.add_argument("--asset",type=Path,default=ROOT/"runs/sdfSkinTemplateBlendFinal/clientAssets/sdf")
    parser.add_argument("--template",type=Path,default=ROOT/"assets/clientTemplates/legacyFemaleV1")
    parser.add_argument("--cache",type=Path,default=ROOT/"runs/eyeLandmarkValidatedCache")
    parser.add_argument("--mapping",type=Path,default=ROOT/"runs/clientEyeMappingAudit/staticMapping.json")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/clientEyeMappingAudit/photoTopology")
    run(parser.parse_args())
