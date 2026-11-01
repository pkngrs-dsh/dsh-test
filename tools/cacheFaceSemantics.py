"""Cache offline, original-resolution face semantics with explicit provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeImage, writeJson
from renderCropComparison import asImage, fileHash, makeGrid, validatePhotoPair

# Display palette only. The stored label IDs are the stable 19-class contract.
palette = np.array([[0,0,0], [180,155,130], [250,90,50], [250,140,30],
                    [30,180,255], [30,240,210], [200,100,200], [160,120,180],
                    [160,100,140], [200,200,80], [70,210,80], [90,40,130],
                    [245,65,145], [210,40,70], [100,130,180], [200,180,70],
                    [60,80,120], [90,75,40], [120,100,90]], np.float32) / 255


def cacheSemantics(source: Path, output: Path, *, device="cuda", parserName="celebm", adapter=None):
    from faceAppearance.adapters.faceParsing import FaceParsingAdapter

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty semantic cache directory")
    inputs = json.loads((source / "inputs.json").read_text())
    photos = [validatePhotoPair(item, item) for item in inputs]
    output.mkdir(parents=True, exist_ok=True)
    writeJson(output / "manifest.json", {"status": "parsing", "source": source})
    start = perf_counter()
    if adapter is None:
        adapter = FaceParsingAdapter(projectRoot, device=device, parserName=parserName).loadModels()
    elif not adapter.isLoaded or adapter.device != device or adapter.parserName != parserName:
        raise ValueError("Preloaded parser must match the requested device and parser name")
    loadSeconds = perf_counter() - start
    views, panels, labels = [], [], []
    for index, (photo, item) in enumerate(zip(photos, inputs)):
        print(f"Parsing original photo {index + 1}/{len(inputs)}", flush=True)
        begin = perf_counter()
        result = adapter.infer(np.asarray(photo))
        np.savez_compressed(output / f"view{index:02d}.npz", labelMap=result.labelMap,
                            confidence=result.confidence, landmarks=result.landmarks)
        Image.fromarray(result.labelMap.astype(np.uint8)).save(output / f"view{index:02d}Labels.png")
        writeImage(output / f"view{index:02d}Confidence.png", result.confidence)
        rgb = np.asarray(photo, np.float32) / 255
        foreground = np.isin(result.labelMap, (1,2,3,4,5,10,11,12,13))
        overlay = rgb.copy()
        overlay[foreground] = .55 * rgb[foreground] + .45 * palette[result.labelMap[foreground]]
        writeImage(output / f"view{index:02d}Overlay.png", overlay)
        ys, xs = np.nonzero(foreground)
        if not len(xs):
            raise ValueError("No facial semantics found")
        crop = (int(xs.min())-30, int(ys.min())-30, int(xs.max())+31, int(ys.max())+31)
        panels.extend([photo.crop(crop), asImage(overlay).crop(crop)])
        labels.extend([f"View {index+1}: original", f"View {index+1}: semantics"])
        views.append({"viewIndex": index, "photoSha256": item["sha256"],
                      "imageSizeWh": photo.size, "seconds": perf_counter()-begin,
                      "metadata": result.metadata,
                      "cacheSha256": fileHash(output / f"view{index:02d}.npz"),
                      "pixelsPerLabel": np.bincount(result.labelMap.ravel(), minlength=19)})
    makeGrid(panels, labels, columns=4, panelSize=320,
             subtitle="Original-resolution FaRL segmentation | automatic labels, not manual ground truth").save(output / "overview.png")
    report = {"status": "complete", "source": source, "device": device, "parserName": parserName,
              "loadSeconds": loadSeconds, "totalSeconds": perf_counter()-start,
              "inputsSha256": fileHash(source / "inputs.json"), "labelNames": result.labelNames,
              "views": views, "confidenceMeaning": "model softmax score, not calibrated geometric certainty",
              "modelWeightHashes": {name: fileHash(path) for name, path in adapter.assetPaths().items()},
              "implementationHashes": {name: fileHash(projectRoot / name) for name in (
                  "src/faceAppearance/adapters/faceParsing.py", "src/faceAppearance/faceSemantics.py",
                  "tools/cacheFaceSemantics.py")}}
    writeJson(output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "totalSeconds": report["totalSeconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/referenceViewBlend4Experiment")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/faceSemanticCache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parserName", choices=("celebm", "lapa"), default="celebm")
    args = parser.parse_args()
    cacheSemantics(args.input, args.output, device=args.device, parserName=args.parserName)
