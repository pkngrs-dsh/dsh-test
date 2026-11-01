"""Rebake saved ICT materials with the current CPU transfer contract, without inference."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.atlasTransfer import transferTexture
from faceAppearance.registration import RegistrationAssets


def rebakeResult(resultDir: Path):
    start = perf_counter()
    report = json.loads((resultDir / "result.json").read_text())
    diagnostics = report["wildCapDiagnostics"]
    config = report["config"]
    assets = RegistrationAssets(projectRoot / config["registrationAsset"], config["workingResolution"])
    if assets.manifest["resultSha256"] != diagnostics["registrationResultSha256"]:
        raise ValueError("Saved optimization and current registration versions differ")
    sourcePath = resultDir / "wildcapIct/materials.npz"
    with np.load(sourcePath, allow_pickle=False) as saved:
        source = {name: saved[name] for name in saved.files}
    with np.load(resultDir / "texture.npz", allow_pickle=False) as saved:
        base = saved["albedo"].copy()
    maps = {}
    for key in ("diffuse", "specular"):
        transferred = transferTexture(
            source[key], source["validMask"], assets.atlasMap("ictToUff"),
            materialKind="color" if key == "diffuse" else "scalar",
            colorEncoding="gamma2.2" if key == "diffuse" else "native")
        maps[key], maps["validMask"] = transferred["texture"], transferred["validMask"]
    if not maps["validMask"].any():
        raise ValueError("No valid shared surface survived rebaking")
    base[maps["validMask"]] = maps["diffuse"][maps["validMask"]]
    maps["baseWithWildCap"] = base
    outputDir = resultDir / "wildcapUff"
    for key in ("diffuse", "specular", "baseWithWildCap"):
        writeImage(outputDir / f"{key}.png", maps[key])
    writeImage(outputDir / "refinedCoverage.png", maps["validMask"])
    np.savez_compressed(outputDir / "materials.npz", **maps)
    operation = {"operation": "gamma2.2DiffuseResampledInLinearLight",
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "sourceMaterialsSha256": hashlib.sha256(sourcePath.read_bytes()).hexdigest(),
                 "seconds": perf_counter() - start, "modelInferenceRepeated": False}
    report.setdefault("postprocessing", []).append(operation)
    diagnostics["refinedUffPixelCount"] = int(maps["validMask"].sum())
    diagnostics["diffuseTransferSpace"] = "linearLight; gamma2.2 decoded then reencoded"
    if not diagnostics["artifactMaskProvided"]:
        diagnostics["lightingMode"] = "globalOnly"
        diagnostics["localLightingTexelCount"] = 0
    writeJson(resultDir / "wildcapIct/diagnostics.json", diagnostics)
    writeJson(resultDir / "result.json", report)
    print(json.dumps(operation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resultDir", type=Path)
    rebakeResult(parser.parse_args().resultDir)
