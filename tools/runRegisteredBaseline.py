"""Bake a verified real-photo baseline to ICT, optionally optimize, and return maps."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.atlasTransfer import transferTexture
from faceAppearance.registration import RegistrationAssets
from faceAppearance.runtime import configureRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/quality1024_16views")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/registeredBaseline")
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skipOptimization", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configureRuntime(projectRoot)
    assets = RegistrationAssets(projectRoot / "assets/registration/wrapV1", args.resolution)
    geometry = dict(np.load(args.input / "geometry.npz"))
    texture = dict(np.load(args.input / "texture.npz"))
    ictGeometry = assets.toIctGeometry(geometry)
    np.savez_compressed(args.output / "ictGeometry.npz", **{
        key: value for key, value in ictGeometry.items() if isinstance(value, np.ndarray)})
    transferred = transferTexture(texture["albedo"], texture["coverageMask"], assets.atlasMap("uffToIct"))
    writeImage(args.output / "ictInput.png", transferred["texture"])
    writeImage(args.output / "ictObserved.png", transferred["validMask"])
    roundTrip = transferTexture(transferred["texture"], transferred["validMask"], assets.atlasMap("ictToUff"))
    writeImage(args.output / "uffRoundTrip.png", roundTrip["texture"])
    writeImage(args.output / "uffRoundTripCoverage.png", roundTrip["validMask"])
    report = {"status": "atlasBakingPassed", "resolution": args.resolution,
              "atlasTransfer": {"uffToIct": transferred["diagnostics"],
                                "nativeRoundTrip": roundTrip["diagnostics"], "ictToUff": {}},
              "ictObservedTexels": int(transferred["validMask"].sum()),
              "uffRoundTripObservedTexels": int(roundTrip["validMask"].sum()),
              "registrationResultSha256": assets.manifest["resultSha256"],
              "geometryFrame": "uvFaceFusionCanonical", "inputColorInterpretation": "OpenDelight native values assumed gamma-2.2 encoded RGB for the WildCap experiment"}
    writeJson(args.output / "report.json", report)
    if args.skipOptimization:
        print(json.dumps(report, indent=2)); return
    from faceAppearance.adapters.wildcap import WildCapAdapter
    import torch
    try:
        start = time.perf_counter()
        adapter = WildCapAdapter(projectRoot).loadModels()
        report["loadSeconds"] = time.perf_counter() - start
        print(f"WildCap loaded in {report['loadSeconds']:.1f}s; optimizing {args.iterations} updates", flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = adapter.optimize(ictGeometry["vertices"], ictGeometry["faces"], ictGeometry["uvs"],
                                  transferred["texture"], transferred["validMask"],
                                  uvFaces=ictGeometry["uvFaces"], geometryFrame=ictGeometry["geometryFrame"],
                                  workingResolution=args.resolution, iterations=args.iterations)
        report["wildCap"] = result["diagnostics"]
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        np.savez_compressed(args.output / "ictMaterials.npz", **{
            key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        for key in ("diffuse", "specular"):
            writeImage(args.output / f"ict{key.title()}.png", result[key])
            back = transferTexture(result[key], result["validMask"], assets.atlasMap("ictToUff"),
                                    materialKind="scalar" if key == "specular" else "color",
                                    colorEncoding="gamma2.2" if key == "diffuse" else "native")
            report["atlasTransfer"]["ictToUff"][key] = back["diagnostics"]
            writeImage(args.output / f"uff{key.title()}.png", back["texture"])
        writeImage(args.output / "ictNormalEncoded.png", (result["tangentNormal"] + 1) / 2)
        report["normalTransferStatus"] = "keptInIctAtlasUntilSourceTangentConventionIsVerified"
        report["status"] = "wildCapExecuted"
        writeJson(args.output / "report.json", report)
        print(json.dumps({"status": report["status"], "seconds": report["wildCap"]["seconds"],
                          "peakAllocatedGiB": report["peakAllocatedGiB"]}, indent=2), flush=True)
    except Exception as error:
        writeJson(args.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
