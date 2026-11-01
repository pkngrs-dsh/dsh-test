"""CPU-only bandwidth control for saved SH delighting experiments.

Resample saved 512-pixel inputs at the exact original-photo UV sample locations,
refit the two SH modes, then re-fuse seven branches on one common support. Raw
photo atlases retain illumination and must not be interpreted as albedo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion, gaussian_filter

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot / "src") not in sys.path:
    sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.appearanceObservations import fuseLinearObservations, linearToGamma22, srgbToLinear
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.shDelight import solveShDelight
from faceAppearance.texture import samplePixels


branchLabels = {
    "openDelight": "OpenDelight 512 | assumed albedo",
    "rawOriginalLit": "Original photo | LIT, NOT ALBEDO",
    "raw512Lit": "512 input photo | LIT, NOT ALBEDO",
    "shOnlyOriginal": "SH-only | original photo samples",
    "shOnly512": "SH-only | 512 input samples",
    "shPriorOriginal": "SH + OD prior | original samples",
    "shPrior512": "SH + OD prior | 512 input samples",
}


def fileHash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readArrays(path: Path, names: tuple[str, ...] | None = None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in (names or data.files)}


def sampleSavedWorkingImages(observations: dict, inputs: list[dict], source: Path) -> dict:
    """Map original-photo sample centers back into saved working RGB images."""
    pixels = np.asarray(observations["originalSamplingPixels"], np.float64)
    weights = np.asarray(observations["weights"])
    if (weights.ndim != 3 or pixels.shape != (*weights.shape, 2)
            or len(inputs) != len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Sampling coordinates, weights and input views must have matching finite shapes")
    sampled = np.zeros((*weights.shape, 3), dtype=np.float32)
    validMasks = np.zeros(weights.shape, dtype=bool)
    records = []
    for index, metadata in enumerate(inputs):
        imagePath = source / f"input{index:02d}.png"
        with Image.open(imagePath) as image:
            image = image.convert("RGB")
            if list(image.size) != metadata["workingSize"]:
                raise ValueError(f"Saved working image dimensions differ from metadata: {imagePath}")
            linear = srgbToLinear(np.asarray(image, np.float32) / 255)
        transform = np.asarray(metadata["inputToWorking"], np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("Expected a finite invertible inputToWorking transform")
        selected = np.flatnonzero(weights[index].ravel() > 0)
        original = pixels[index].reshape(-1, 2)[selected]
        if not np.isfinite(original).all():
            raise ValueError("Observed original sampling coordinates must be finite")
        projected = np.column_stack([original, np.ones(len(original))]) @ transform.T
        if np.any(np.abs(projected[:, 2]) < 1e-12):
            raise ValueError("Saved sampling coordinates map to nonfinite working positions")
        working = projected[:, :2] / projected[:, 2:3]
        height, width = linear.shape[:2]
        # Stored sampling coordinates are float32; tolerate subpixel roundoff
        # only, not genuine out-of-frame locations.
        tolerance = 1e-4
        inside = ((working[:, 0] >= -tolerance) & (working[:, 0] <= width - 1 + tolerance)
                  & (working[:, 1] >= -tolerance) & (working[:, 1] <= height - 1 + tolerance))
        working = np.clip(working[inside], [0, 0], [width - 1, height - 1])
        retained = selected[inside]
        sampled[index].reshape(-1, 3)[retained] = samplePixels(linear, working)
        validMasks[index].ravel()[retained] = True
        records.append({"viewIndex": index, "workingSizeWh": metadata["workingSize"],
                        "originalSizeWh": metadata["orientedSize"],
                        "sourcePhotoSha256": metadata["sha256"],
                        "savedWorkingImageSha256": fileHash(imagePath),
                        "inputToWorking": transform.tolist(),
                        "originalObservedSamples": int(len(selected)), "retainedSamples": int(len(retained)),
                        "outsideWorkingImageSamples": int((~inside).sum())})
    return {"raw512Linear": sampled, "samplingValidMask": validMasks,
            "diagnostics": {"views": records,
                            "encoding": "Saved working PNG interpreted as sRGB and decoded before bilinear sampling",
                            "sampleLocation": "Original common UV sampling locations transformed by saved inputToWorking",
                            "roundoffToleranceWorkingPixels": 1e-4}}


def comparisonMetrics(values: np.ndarray, fusion: dict, commonWeights: np.ndarray,
                       commonMask: np.ndarray) -> dict:
    """Measure texture energy and multi-view agreement; neither is accuracy."""
    linear = fusion["linearAlbedo"]
    luminance = linear @ np.array([.2126, .7152, .0722])
    inner = binary_erosion(commonMask, iterations=5)
    filteredMask = gaussian_filter(commonMask.astype(float), 1.2)
    smooth = np.divide(gaussian_filter(luminance * commonMask, 1.2), filteredMask,
                       out=np.zeros_like(luminance), where=filteredMask > 1e-10)
    highPassRms = float(np.sqrt(np.mean((luminance[inner] - smooth[inner]) ** 2))) if inner.any() else None
    meanLuminance = float(luminance[inner].mean()) if inner.any() else None
    multiView = commonMask & ((commonWeights > 0).sum(axis=0) >= 2)
    metricWeights = commonWeights * multiView[None]
    totalWeight = float(metricWeights.sum())
    residual = values - linear[None]
    residualRmse = float(np.sqrt(np.sum(metricWeights[..., None] * residual ** 2)
                                 / (3 * totalWeight))) if totalWeight > 0 else None
    referenceMean = float(np.sum(metricWeights * (values @ np.array([.2126, .7152, .0722])))
                          / totalWeight) if totalWeight > 0 else None
    return {"highPassRmsLinear": highPassRms,
            "relativeHighPassRms": highPassRms / max(meanLuminance, 1e-8) if highPassRms is not None else None,
            "meanLuminanceLinear": meanLuminance, "highPassSigmaUvPixels": 1.2,
            "highPassMetricTexels": int(inner.sum()),
            "crossViewRmseLinear": residualRmse,
            "relativeCrossViewRmse": residualRmse / max(referenceMean, 1e-8) if residualRmse is not None else None,
            "multiViewMetricTexels": int(multiView.sum()),
            "crossViewMetricWeights": "Identical common input weights, before branch-dependent robust fusion",
            "meaning": "Descriptive texture energy and view agreement only; includes illumination, noise and UV errors, not accuracy"}


def makeComparisonBoard(textures: dict, commonMask: np.ndarray) -> tuple[Image.Image, list[int]]:
    ys, xs = np.nonzero(commonMask)
    if not len(xs):
        raise ValueError("No common observed UV region remains for the comparison board")
    height, width = commonMask.shape
    crop = [max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8),
            min(width, int(xs.max()) + 9), min(height, int(ys.max()) + 9)]
    panelWidth, panelHeight, labelHeight, header = 410, 410, 46, 74
    canvas = Image.new("RGB", (4 * panelWidth, header + 2 * (panelHeight + labelHeight)), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    fontPath = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(fontPath), 16) if fontPath.exists() else ImageFont.load_default()
    draw.text((16, 12), "CPU bandwidth control | same UV samples, mask, base weights and display exposure", fill="white", font=font)
    draw.text((16, 39), "Raw photo panels retain light. Texture energy and agreement are NOT albedo accuracy.", fill=(202, 212, 224), font=font)
    for index, (name, label) in enumerate(branchLabels.items()):
        values = linearToGamma22(textures[name]).copy()
        values[~commonMask] = np.array([.12, .14, .17])
        image = Image.fromarray(np.rint(values * 255).astype(np.uint8)).crop(crop)
        image = ImageOps.contain(image, (panelWidth, panelHeight), Image.Resampling.LANCZOS)
        left, top = (index % 4) * panelWidth, header + (index // 4) * (panelHeight + labelHeight)
        draw.text((left + 10, top + 12), label, fill="white", font=font)
        canvas.paste(image, (left + (panelWidth - image.width) // 2,
                            top + labelHeight + (panelHeight - image.height) // 2))
    return canvas, crop


def runBandwidthControl(experiment: Path, source: Path, output: Path, iterations: int = 8) -> dict:
    start = perf_counter()
    experiment, source, output = experiment.resolve(), source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new bandwidth control output directory; existing results are never overwritten")
    parentReport = json.loads((experiment / "experimentReport.json").read_text())
    if Path(parentReport["sourceRun"]).resolve() != source:
        raise ValueError("Bandwidth control source must match the original SH experiment source")
    if parentReport["parameters"]["shIterations"] != iterations:
        raise ValueError("SH iteration count must match the original experiment")
    implementationPaths = [projectRoot / "src/faceAppearance/shDelight.py",
                           projectRoot / "src/faceAppearance/appearanceObservations.py"]
    implementationHashes = {str(path.relative_to(projectRoot)): fileHash(path) for path in implementationPaths}
    for name, expected in parentReport["parameters"].get("implementationHashes", {}).items():
        if name in implementationHashes and implementationHashes[name] != expected:
            raise ValueError(f"Algorithm changed after the original SH experiment: {name}")
    observations = readArrays(experiment / "observations.npz")
    inputs = json.loads((source / "inputs.json").read_text())
    expectedInputs = json.loads((experiment / "openDelight/inputs.json").read_text())
    if inputs != expectedInputs:
        raise ValueError("Input transforms or original-photo identities differ from the main experiment")
    print("Sampling cached 512 inputs at the original experiment UV positions", flush=True)
    sampled = sampleSavedWorkingImages(observations, inputs, source)
    originalWeights = observations["weights"]
    if np.any((originalWeights > 0) & ~sampled["samplingValidMask"]):
        raise ValueError("Some original observations cannot be sampled from the saved working image; inspect the coordinate contract")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "raw512Observations.npz", raw512Linear=sampled["raw512Linear"],
                        samplingValidMask=sampled["samplingValidMask"])
    writeJson(output / "samplingDiagnostics.json", sampled["diagnostics"])
    # Match the main experiment: original observation weights and the exact
    # same OD fusion define both fit input support and the optional prior.
    prior = fuseLinearObservations(observations["openDelightLinear"], originalWeights)["linearAlbedo"]
    fits = {}
    for mode in ("shOnly", "shPrior"):
        print(f"Fitting {mode} from 512 samples ({iterations} iterations, CPU)", flush=True)
        fitStart = perf_counter()
        result = solveShDelight(sampled["raw512Linear"], originalWeights, observations["normals"],
                               mode=mode, priorAlbedo=prior if mode == "shPrior" else None,
                               iterations=iterations)
        result["diagnostics"]["seconds"] = perf_counter() - fitStart
        fits[mode] = result
        np.savez_compressed(output / f"{mode}512Fit.npz", **{
            name: value for name, value in result.items() if isinstance(value, np.ndarray)})
        writeJson(output / f"{mode}512Fit.json", result["diagnostics"])
    mainSupport = readArrays(experiment / "commonSupport.npz", ("weights",))["weights"]
    commonWeights = np.minimum(mainSupport, fits["shOnly"]["validWeights"])
    commonWeights = np.minimum(commonWeights, fits["shPrior"]["validWeights"])
    if not commonWeights.any():
        raise ValueError("No observations are safe for both bandwidths and both SH modes")
    originalFits = {mode: readArrays(experiment / f"{mode}Fit.npz", ("correctedObservations",))
                    for mode in ("shOnly", "shPrior")}
    branchValues = {"openDelight": observations["openDelightLinear"],
                    "rawOriginalLit": observations["rawLinear"], "raw512Lit": sampled["raw512Linear"],
                    "shOnlyOriginal": originalFits["shOnly"]["correctedObservations"],
                    "shPriorOriginal": originalFits["shPrior"]["correctedObservations"],
                    "shOnly512": fits["shOnly"]["correctedObservations"],
                    "shPrior512": fits["shPrior"]["correctedObservations"]}
    print("Re-fusing all seven branches on one common support", flush=True)
    fusions = {name: fuseLinearObservations(values, commonWeights) for name, values in branchValues.items()}
    commonMask = np.logical_and.reduce([fusion["coverageMask"] for fusion in fusions.values()])
    if not commonMask.any():
        raise ValueError("Final fusions have no common UV coverage")
    np.savez_compressed(output / "commonSupport.npz", weights=commonWeights, coverageMask=commonMask)
    metrics, textures = {}, {}
    for name, values in branchValues.items():
        fusion = fusions[name]
        linear = fusion["linearAlbedo"].copy()
        linear[~commonMask] = 0
        textures[name] = linear
        np.savez_compressed(output / f"{name}.npz", linearRgb=linear, coverageMask=commonMask,
                            confidence=fusion["confidence"], sourceIndex=fusion["sourceIndex"])
        writeImage(output / f"{name}.png", linearToGamma22(linear))
        metrics[name] = comparisonMetrics(values, {**fusion, "linearAlbedo": linear}, commonWeights, commonMask)
    board, crop = makeComparisonBoard(textures, commonMask)
    board.save(output / "atlasComparison.png")
    sourcePaths = [experiment / name for name in ("observations.npz", "commonSupport.npz", "shOnlyFit.npz", "shPriorFit.npz")]
    sourcePaths += [source / "inputs.json"] + [source / f"input{index:02d}.png" for index in range(len(inputs))]
    report = {
        "status": "completed", "seconds": perf_counter() - start, "executionDevice": "CPU",
        "sourceRun": str(source), "mainExperiment": str(experiment), "shIterations": iterations,
        "mainExperimentWasModified": False, "wildCapWasRun": False,
        "originalViewSizesWh": [item["orientedSize"] for item in inputs],
        "workingViewSizesWh": [item["workingSize"] for item in inputs],
        "atlasResolutionHw": list(commonMask.shape), "boardUvCropXyxy": crop,
        "commonCoverageTexels": int(commonMask.sum()),
        "mainCommonCoverageTexels": int((mainSupport.sum(axis=0) > 0).sum()),
        "commonViewTexelCount": int(np.count_nonzero(commonWeights)),
        "mainCommonViewTexelCount": int(np.count_nonzero(mainSupport)),
        "sharedInputs": "Frozen geometry, normals, UV sample locations, same original fit weights and OD prior; min(main support, both new SH validWeights) for every final fusion",
        "fusionWeights": "Common input weights; identical robust fusion code may assign branch-dependent residual weights",
        "displayEncoding": "gamma2.2, identical exposure and common UV crop; no auto-contrast",
        "branchLabels": branchLabels, "metrics": metrics,
        "sampling": sampled["diagnostics"], "implementationHashes": implementationHashes,
        "toolSha256": fileHash(Path(__file__)),
        "sourceHashes": {str(path): fileHash(path) for path in sourcePaths},
        "scopeLimits": [
            "Raw photo atlases retain illumination and are not albedo.",
            "Texture energy and cross-view consistency are descriptive, not reconstruction accuracy.",
            "The 512 control includes the original sRGB-domain Lanczos resize and PNG quantization; it is not an ideal linear-light bandwidth filter alone.",
            "Geometry, dense UV errors, unknown photo processing, and approximate OD gamma2.2 encoding remain fixed.",
            "SH-only and SH-prior use different explicit RGB gauges; relative metrics reduce scale sensitivity without removing intrinsic ambiguity.",
            "Common support is selected by all SH safety/consensus gates, so findings concern this intersection rather than the whole face.",
            "No WildCap refinement is included in this bandwidth control.",
        ],
    }
    writeJson(output / "bandwidthControlReport.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "commonCoverageTexels": report["commonCoverageTexels"], "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=projectRoot / "runs/shDelightExperiment")
    parser.add_argument("--source", type=Path, default=projectRoot / "runs/faceCrop1024_4viewsObserved")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    runBandwidthControl(args.experiment, args.source, args.output or args.experiment / "bandwidthControl", args.iterations)


if __name__ == "__main__":
    main()
