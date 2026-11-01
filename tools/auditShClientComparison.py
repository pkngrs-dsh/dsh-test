"""Read-only CPU audit of saved SH / WildCap comparison artifacts.

Writes one verification report after every requested check succeeds. No model
inference, camera fitting, rendering, or optimization is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeJson
from runShExperiment import detailMetric, validateCompletedWildCap

branchNames = ("openDelight", "shOnly", "shPrior")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fileHash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readArrays(path):
    with np.load(path, allow_pickle=False) as saved:
        result = dict(saved)
    require(bool(result), f"Empty array archive: {path}")
    for name, value in result.items():
        require(value.dtype.kind in "bifu" and np.isfinite(value).all(),
                f"Nonfinite or nonnumeric field {name}: {path}")
    return result


def requireSame(left, right, name):
    require(np.array_equal(left, right), f"Values differ: {name}")


def compareMetric(saved, actual, name):
    for key in ("highPassRmsLinear", "relativeHighPassRms", "pixelCount", "sigmaUvPixels"):
        expected, observed = saved.get(key), actual[key]
        require((expected is None and observed is None) or
                (expected is not None and observed is not None and
                 np.isclose(expected, observed, rtol=1e-5, atol=1e-9)),
                f"Saved descriptive metric differs: {name}.{key}")


def auditComparison(inputRoot, output, *, withWildCap=True):
    start = perf_counter()
    root, output = Path(inputRoot).resolve(), Path(output).resolve()
    require(not output.exists(), f"Choose a new verification output file: {output}")
    experiment = readJson(root / "experimentReport.json")
    require(experiment.get("status") in ("prepared", "completedWithWildCap"),
            "Experiment has not committed its preparation")
    if withWildCap:
        require(experiment.get("status") == "completedWithWildCap",
                "All three WildCap branches must finish before the full audit")
    parameters = experiment["parameters"]
    resolution = parameters["atlasResolution"]
    source = Path(experiment["sourceRun"]).resolve()
    require(readJson(source / "result.json").get("status") == "succeeded",
            "Source service reconstruction must be successful")
    require(parameters.get("geometryAndCamerasFrozen") is True,
            "Experiment must freeze geometry and fitted cameras")
    require(parameters.get("rawSampling") == "working" and
            parameters.get("originalResolutionRawSampling") is False,
            "This audit requires the saved-working-image SH comparison")

    sourceNames = ("geometry.npz", "cameras.json", "inputs.json", "delight.npz")
    copiedNames = ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json")
    sourceHashes = {name: fileHash(source / name) for name in set(sourceNames + copiedNames)}
    inputs = readJson(source / "inputs.json")
    photoRecords = []
    for index, metadata in enumerate(inputs):
        original = Path(metadata["path"])
        require(fileHash(original) == metadata["sha256"], f"Source photo changed: view {index}")
        imageName = f"input{index:02d}.png"
        imageHash = fileHash(source / imageName)
        require(parameters["workingImageHashes"][imageName] == imageHash,
                f"Working input changed: {imageName}")
        photoRecords.append({"viewIndex": index, "originalSha256": metadata["sha256"],
                             "workingImageSha256": imageHash, "workingSizeWh": metadata["workingSize"]})
    codeHashes = {}
    for filename, expected in parameters["implementationHashes"].items():
        actual = fileHash(projectRoot / filename)
        require(actual == expected, f"SH implementation changed since preparation: {filename}")
        codeHashes[filename] = actual
    require(fileHash(projectRoot / "tools/runShBandwidthControl.py") == parameters["workingSamplerSha256"],
            "Saved working image sampler changed since preparation")

    observations = readArrays(root / "observations.npz")
    common = readArrays(root / "commonSupport.npz")
    mask, weights = common["coverageMask"], common["weights"]
    require(mask.shape == (resolution, resolution) and mask.dtype.kind == "b" and mask.any(),
            "Expected nonempty binary common UV coverage at the planned resolution")
    require(weights.shape == (len(inputs), resolution, resolution) and np.all(weights >= 0),
            "Common observation weights have invalid shape or range")
    require(observations["weights"].shape == weights.shape and
            np.all(weights <= observations["weights"]), "Common weights exceed original shared weights")
    expectedWeights = observations["weights"].copy()
    shSeconds = {}
    for name in ("shOnly", "shPrior"):
        fit = readArrays(root / f"{name}Fit.npz")
        require(fit["validWeights"].shape == weights.shape, f"SH support shape differs: {name}")
        expectedWeights = np.minimum(expectedWeights, fit["validWeights"])
        fitDiagnostic = readJson(root / f"{name}Fit.json")
        shSeconds[name] = fitDiagnostic["seconds"]
    requireSame(weights, expectedWeights, "common weights versus original and both SH supports")
    require(experiment["commonCoverageTexels"] == int(mask.sum()), "Common coverage count differs")
    require(experiment["commonObservationCount"] == int(np.count_nonzero(weights)),
            "Common observation count differs")

    branchReports, wildCapMasks = {}, []
    for name in branchNames:
        branch = root / name
        record = readJson(branch / "result.json")
        require(record.get("status") == "succeeded", f"Branch is incomplete: {name}")
        require(Path(record["sourceRun"]).resolve() == source, f"Branch uses another source: {name}")
        require(record["experiment"] == parameters, f"Branch experiment parameters differ: {name}")
        for filename in sourceNames:
            require(record["sourceHashes"][filename] == sourceHashes[filename],
                    f"Source hash differs: {name}/{filename}")
        for filename in copiedNames:
            require(fileHash(branch / filename) == sourceHashes[filename],
                    f"Frozen source copy differs: {name}/{filename}")
        for index, photo in enumerate(photoRecords):
            require(fileHash(branch / f"input{index:02d}.png") == photo["workingImageSha256"],
                    f"Branch working input differs: {name}, view {index}")
        texture = readArrays(branch / "texture.npz")
        requireSame(texture["coverageMask"], mask, f"{name} common coverage")
        albedo, linear = texture["albedo"], texture["linearAlbedo"]
        require(albedo.shape == (resolution, resolution, 3) and linear.shape == albedo.shape,
                f"Branch albedo shape differs: {name}")
        require(np.all((albedo >= 0) & (albedo <= 1)) and np.all((linear >= 0) & (linear <= 1)),
                f"Branch color range differs: {name}")
        require(np.allclose(albedo ** 2.2, linear, rtol=1e-5, atol=1e-7),
                f"Gamma-2.2 and linear fields disagree: {name}")
        require(np.all(albedo[~mask] == 0) and np.all(linear[~mask] == 0),
                f"Unobserved albedo must be zero: {name}")
        before = detailMetric(linear, mask)
        compareMetric(record["detailMetric"], before, name)
        branchReport = {"textureSha256": fileHash(branch / "texture.npz"),
                        "commonCoverageTexels": int(mask.sum()), "detailBefore": before}
        if withWildCap:
            diagnostic = validateCompletedWildCap(branch, record, parameters["wildCapPlanned"], resolution)
            require(experiment["wildCapBranches"][name] == diagnostic,
                    f"Aggregate WildCap diagnostics differ: {name}")
            materials = readArrays(branch / "wildcapUff/materials.npz")
            valid, diffuse, composite = materials["validMask"].astype(bool), materials["diffuse"], materials["baseWithWildCap"]
            require(not np.any(valid & ~mask), f"WildCap support exceeds shared base support: {name}")
            requireSame(composite[~valid], albedo[~valid], f"{name} composite outside WildCap support")
            requireSame(composite[valid], diffuse[valid], f"{name} composite inside WildCap support")
            require(np.all((composite >= 0) & (composite <= 1)), f"WildCap composite range differs: {name}")
            after = detailMetric(composite ** 2.2, mask)
            compareMetric(record["wildCapDetailMetric"], after, name + "WildCap")
            wildCapMasks.append(valid)
            branchReport.update(wildCapSeconds=diagnostic["seconds"],
                                peakAllocatedGiB=diagnostic["peakAllocatedGiB"],
                                iterations=diagnostic["iterations"], resolution=diagnostic["workingResolution"],
                                seed=diagnostic["seed"], lightingMode=diagnostic["lightingMode"],
                                wildCapValidTexels=int(valid.sum()), detailAfter=after,
                                outsideWildCapPatchUnchanged=True, insideWildCapPatchEqualsDiffuse=True,
                                wildCapMaterialsSha256=fileHash(branch / "wildcapUff/materials.npz"))
        branchReports[name] = branchReport
    report = {"status": "verified", "experimentStatus": experiment["status"],
              "inputRoot": str(root), "sourceRun": str(source), "withWildCap": withWildCap,
              "sourceHashes": sourceHashes, "photos": photoRecords, "implementationHashes": codeHashes,
              "identicalGeometryCamerasInputsTransforms": True, "identicalCommonSupport": True,
              "commonWeightsMatchBothShSafetyAndConsensusGates": True,
              "allFinite": True, "commonCoverageTexels": int(mask.sum()),
              "commonObservationCount": int(np.count_nonzero(weights)), "branches": branchReports,
              "timing": {"preparationSeconds": experiment["seconds"], "shFitSeconds": shSeconds,
                         "auditSeconds": perf_counter() - start},
              "metricMeaning": "Descriptive texture energy, including residual shading/noise; not reflectance accuracy",
              "limitations": ["Shared working inputs do not make internal OpenDelight and SH resampling identical",
                              "Final robust fusion weights depend on each branch's color residuals",
                              "WildCap comparison uses diffuse composite; specular and normals are held fixed in preview",
                              "SH scales and OpenDelight gamma assumptions are not physical calibration"],
              "inferenceOrTestsRun": False}
    if withWildCap:
        report["identicalWildCapSupport"] = all(np.array_equal(wildCapMasks[0], value) for value in wildCapMasks[1:])
        report["commonWildCapCoverageTexels"] = int(np.logical_and.reduce(wildCapMasks).sum())
        report["timing"]["wildCapLoadSeconds"] = experiment.get("wildCapLoadSeconds")
        report["timing"]["wildCapOptimizationSeconds"] = sum(value["wildCapSeconds"] for value in branchReports.values())
    writeJson(output, report)
    print(json.dumps({"status": report["status"], "withWildCap": withWildCap,
                      "commonCoverageTexels": report["commonCoverageTexels"], "output": str(output)}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/shWildCapClientComparison")
    parser.add_argument("--output", type=Path, help="Defaults to input/verificationReport.json")
    parser.add_argument("--withoutWildCap", action="store_true", help="Audit the completed three frontends only")
    args = parser.parse_args()
    auditComparison(args.input, args.output or args.input / "verificationReport.json",
                    withWildCap=not args.withoutWildCap)


if __name__ == "__main__":
    main()
