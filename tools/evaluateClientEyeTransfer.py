"""Evaluate fixed client eye-hole endpoints against cached frontal photo corners.

This is an explicitly labeled proxy: topology identifies the eye holes, while
their rest-shape X extrema define reproducible endpoints. Detector landmarks
are observations, not anatomical ground truth. Nothing is optimized here.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot/"src"))
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from renderCropComparison import photoCamera


def evaluate(assets, source, template, observations, placement, output):
    source, template, output = Path(source), Path(template), Path(output)
    with np.load(template/"template.npz", allow_pickle=False) as archive:
        data = dict(archive)
    rest = data["vertices"][data["nicpToFullIds"]]
    boundaries = extractMeshBoundaryLoops(rest, data["nicpFaces"])
    # This identity has three audited holes: a neck opening and two 10-vertex
    # eye openings. Fail on another topology instead of guessing an anatomy.
    if boundaries["diagnostics"]["loopVertexCounts"] != [40, 10, 10]:
        raise ValueError("Expected the audited female/260429 skin boundaries")
    eyeRecords = []
    for loop in boundaries["loops"][1:]:
        positiveX = rest[loop, 0].mean() > 0
        eyeRecords.append({"eye": "positiveXEye" if positiveX else "negativeXEye",
                           "labelId": 5 if positiveX else 4,
                           "fullVertexIds": data["nicpToFullIds"][loop[[np.abs(rest[loop, 0]).argmin(), np.abs(rest[loop, 0]).argmax()]]].tolist()})
    inputs = json.loads((source/"inputs.json").read_text())
    cameras = json.loads((source/"cameras.json").read_text())
    contours = json.loads(Path(observations).read_text())[0]["eyes"]
    camera = photoCamera(cameras[0], inputs[0], "fitted")
    transform = json.loads(Path(placement).read_text())["templateToUff"]
    with Image.open(inputs[0]["path"]) as image:
        photo = ImageOps.exif_transpose(image).convert("RGB")
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Frontal fixed eye-hole endpoint proxies against cached STFR detector corners",
              "meaning": "Diagnostic proxy only; no manual ground truth or independent visibility test",
              "placement": transform, "camera": camera, "templateEyeBindings": eyeRecords,
              "sourcePhoto": inputs[0]["path"], "branches": {}}
    for name, path in assets.items():
        with np.load(Path(path)/"clientMesh.npz", allow_pickle=False) as archive:
            vertices = archive["vertices"]*100
            if not np.array_equal(archive["faces"], data["faces"]):
                raise ValueError("Client eye binding requires unchanged template faces")
        vertices = vertices @ np.asarray(transform["rotation"])*transform["scale"]+np.asarray(transform["translation"])
        projection, depth = projectPoints(vertices, camera["intrinsics"], camera["rotation"], camera["translation"])
        rows, errors = [], []
        annotated = photo.copy()
        draw = ImageDraw.Draw(annotated)
        for eye in eyeRecords:
            record = contours[str(eye["labelId"])]
            if not record["valid"]:
                raise ValueError("Frontal eye detector rejected a required observation")
            target = np.array([record["innerCorner"], record["outerCorner"]])
            predicted = projection[eye["fullVertexIds"]]
            if np.any(depth[eye["fullVertexIds"]] <= 0):
                raise ValueError("Eye proxy behind saved camera")
            error = np.linalg.norm(predicted-target, axis=1)
            rows.append({**eye, "target": target, "projected": predicted, "errorPixels": error})
            errors.extend(error.tolist())
            for first, second in zip(target, predicted):
                draw.line([tuple(first), tuple(second)], fill=(255, 220, 30), width=2)
                for point, color in [(first, (30, 230, 80)), (second, (255, 70, 80))]:
                    x, y = point
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        report["branches"][name] = {"eyes": rows, "meanErrorPixels": float(np.mean(errors)),
                                      "maxErrorPixels": float(np.max(errors))}
        annotated.save(output/f"{name}EyeProxies.png")
    writeJson(output/"eyeProxyMetrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=projectRoot/"runs/eyeCornerPriorityFinal")
    parser.add_argument("--template", type=Path, default=projectRoot/"assets/clientTemplates/legacyFemaleV1")
    parser.add_argument("--observations", type=Path, default=projectRoot/"runs/eyeCornerRefinementFinal/photoContours.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate({"baseline": args.baseline, "candidate": args.candidate}, args.source, args.template,
                      args.observations, args.baseline/"diagnostics.json", args.output)
    print(json.dumps({key: value["meanErrorPixels"] for key, value in report["branches"].items()}))


if __name__ == "__main__":
    main()
