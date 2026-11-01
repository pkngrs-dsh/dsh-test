"""Project verified cached face semantics into the current UFF texture on CPU.

This prepares conservative donor/protection masks for later SDF extrapolation.
It does not edit colors, enlarge observed coverage, fit geometry, or run models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.appearanceObservations import collectAppearanceObservations
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.faceSemantics import semanticLabelNames
from faceAppearance.semanticProjection import sampleSemanticObservations


def fileHash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readArrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def buildSkinSupport(labels, confidence, weights, viewingCosines, coverage, *,
                     skinLabelIds=(1,), minConfidence=.8, minCosine=.35,
                     minRelativeWeight=.15, minimumSkinShare=.9,
                     protectionHaloPixels=2) -> tuple[dict, dict]:
    """Separate excluded skin donors from anatomy whose colors must survive.

    View weights express visibility/view quality, not semantic probability.
    Relative-weight gating avoids a negligible side-view vote protecting an
    otherwise reliable texel. Nose is protected by default because this label
    schema has no separate nostril class. Skin colors are not classified by RGB.
    """
    labels, confidence = np.asarray(labels), np.asarray(confidence, np.float32)
    weights, cosines = np.asarray(weights, np.float32), np.asarray(viewingCosines, np.float32)
    coverage = np.asarray(coverage)
    if (weights.ndim != 3 or labels.shape != weights.shape or confidence.shape != weights.shape
            or cosines.shape != weights.shape or coverage.shape != weights.shape[1:]
            or coverage.dtype != np.bool_ or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= len(semanticLabelNames))
            or any(not np.isfinite(a).all() for a in (confidence, weights, cosines))
            or np.any(weights < 0) or np.any(confidence < 0) or np.any(confidence > 1)
            or np.any(cosines < 0) or np.any(cosines > 1)):
        raise ValueError("Expected matching finite NHW observations and boolean HW coverage")
    if (not skinLabelIds or not set(skinLabelIds).issubset({1, 10})
            or not 0 < minConfidence <= 1 or not 0 <= minCosine < 1
            or not 0 < minRelativeWeight <= 1 or not .5 < minimumSkinShare <= 1
            or not isinstance(protectionHaloPixels, int) or protectionHaloPixels < 0):
        raise ValueError("Invalid conservative semantic support parameters")
    maximumWeight = weights.max(axis=0)
    meaningful = ((weights > 0) & (weights >= minRelativeWeight * maximumWeight[None])
                  & (cosines >= minCosine) & coverage[None])
    reliable = meaningful & (confidence >= minConfidence)
    skin = np.isin(labels, skinLabelIds)
    donorExcludedCore = np.any(reliable & ~skin, axis=0)
    # Background/hair/neck near a projected silhouette can be contamination.
    # They must never donate skin, but keeping their old blend immutable leaves
    # bright template islands when the surrounding boundary alpha is raised.
    protectedLabelIds = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18)
    protectedCore = np.any(reliable & np.isin(labels, protectedLabelIds), axis=0)
    donorExcluded = donorExcludedCore.copy()
    protected = protectedCore.copy()
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
    for _ in range(protectionHaloPixels):
        donorExcluded = binary_dilation(donorExcluded, structure=structure) & coverage
        protected = binary_dilation(protected, structure=structure) & coverage

    meaningfulWeights = np.where(meaningful, weights, 0)
    totalWeight = meaningfulWeights.sum(axis=0)
    safeWeight = np.where(reliable & skin, weights, 0).sum(axis=0)
    skinShare = np.divide(safeWeight, totalWeight, out=np.zeros_like(safeWeight), where=totalWeight > 0)
    safe = coverage & (skinShare >= minimumSkinShare) & ~donorExcluded

    # Weighted winning-label evidence is an uncalibrated support score. It is
    # zero outside qualified observations and is never a measured skin certainty.
    evidence = np.stack([np.where(meaningful & (labels == labelId), weights * confidence, 0).sum(axis=0)
                         for labelId in range(len(semanticLabelNames))])
    dominant = evidence.argmax(axis=0).astype(np.uint8)
    semanticConfidence = np.divide(evidence.max(axis=0), totalWeight,
                                   out=np.zeros_like(totalWeight), where=totalWeight > 0)
    semanticConfidence = np.clip(semanticConfidence, 0, 1).astype(np.float32)
    dominant[~coverage] = 0
    arrays = {"safeSkinMask": safe, "protectedFeatureMask": protected,
              "protectedFeatureCoreMask": protectedCore, "semanticConfidence": semanticConfidence,
              "donorExcludedMask": donorExcluded, "donorExcludedCoreMask": donorExcludedCore,
              "dominantSemanticLabel": dominant, "skinSupportFraction": skinShare,
              "sourceCoverageMask": coverage.copy(),
              "qualifiedSemanticSupportMask": meaningful.any(axis=0)}
    diagnostics = {"skinLabelIds": list(skinLabelIds),
                   "skinLabels": [semanticLabelNames[i] for i in skinLabelIds],
                   "donorExcludedLabels": [name for i, name in enumerate(semanticLabelNames) if i not in skinLabelIds],
                   "protectedLabels": [semanticLabelNames[i] for i in protectedLabelIds],
                   "semanticBoundaryPolicy": "all reliable non-skin labels exclude donors; only facial features/accessories preserve the existing blend",
                   "repairableNonSkinLabels": ["background", "neck", "hair"],
                   "minimumParserConfidence": minConfidence, "minimumViewingCosine": minCosine,
                   "minimumRelativeViewWeight": minRelativeWeight, "minimumReliableSkinWeightShare": minimumSkinShare,
                   "protectionHaloUffPixels": protectionHaloPixels,
                   "sourceCoverageTexels": int(coverage.sum()), "safeSkinTexels": int(safe.sum()),
                   "protectedFeatureCoreTexels": int(protectedCore.sum()),
                   "protectedFeatureTexels": int(protected.sum()),
                   "donorExcludedCoreTexels": int(donorExcludedCore.sum()),
                   "donorExcludedTexels": int(donorExcluded.sum()),
                   "qualifiedSemanticTexels": int(meaningful.any(axis=0).sum()),
                   "dominantLabelTexels": {name: int(((dominant == i) & coverage).sum())
                                             for i, name in enumerate(semanticLabelNames)}}
    return arrays, diagnostics


def prepareSupport(sourceRun: Path, sourceExperiment: Path, semanticCache: Path, output: Path) -> dict:
    start = perf_counter()
    sourceRun, sourceExperiment, semanticCache, output = [Path(p).resolve() for p in
                                                        (sourceRun, sourceExperiment, semanticCache, output)]
    artifactPath, reportPath = output / "sourceSkinSupport.npz", output / "semanticSupport.json"
    if artifactPath.exists() or reportPath.exists():
        raise FileExistsError("Semantic support artifacts already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    sourceResult = json.loads((sourceRun / "result.json").read_text())
    experimentReport = json.loads((sourceExperiment / "experimentReport.json").read_text())
    verification = json.loads((sourceExperiment / "verificationReport.json").read_text())
    branch = sourceExperiment / "shPrior"
    branchReport = json.loads((branch / "result.json").read_text())
    if (sourceResult.get("status") != "succeeded"
            or experimentReport.get("status") not in ("prepared", "completedWithWildCap")
            or Path(experimentReport["sourceRun"]).resolve() != sourceRun
            or not verification.get("identicalGeometryCamerasInputsTransforms")):
        raise ValueError("A completed experiment sharing the exact source reconstruction is required")
    required = ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json", "delight.npz")
    sourceHashes = {name: fileHash(sourceRun / name) for name in required}
    for name, digest in sourceHashes.items():
        if digest != verification["sourceHashes"].get(name):
            raise ValueError(f"Frozen source changed since observation export: {name}")
        if name in branchReport.get("sourceHashes", {}) and digest != branchReport["sourceHashes"][name]:
            raise ValueError(f"Branch source identity mismatch: {name}")
        if name != "delight.npz" and fileHash(branch / name) != digest:
            raise ValueError(f"Saved branch differs from current source: {name}")
    sourceHashes["texture.npz"] = fileHash(sourceRun / "texture.npz")
    inputs = json.loads((sourceRun / "inputs.json").read_text())
    cameras = json.loads((sourceRun / "cameras.json").read_text())
    manifest = json.loads((semanticCache / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or tuple(manifest.get("labelNames", ())) != semanticLabelNames
            or len(manifest.get("views", [])) != len(inputs)):
        raise ValueError("Completed cached 19-class semantics for every original photo are required")
    cacheInputsPath = Path(manifest["source"]) / "inputs.json"
    if fileHash(cacheInputsPath) != manifest["inputsSha256"]:
        raise ValueError("Semantic cache input manifest changed")
    cachedInputs = json.loads(cacheInputsPath.read_text())
    if [item["sha256"] for item in cachedInputs] != [item["sha256"] for item in inputs]:
        raise ValueError("Semantic cache photo identities/order differ from current source")
    labels, confidences, views = [], [], []
    for index, (record, item) in enumerate(zip(manifest["views"], inputs)):
        cachePath = semanticCache / f"view{index:02d}.npz"
        digest = fileHash(cachePath)
        if (record["viewIndex"] != index or record["photoSha256"] != item["sha256"]
                or digest != record["cacheSha256"] or fileHash(Path(item["path"])) != item["sha256"]
                or record["imageSizeWh"] != item["orientedSize"]):
            raise ValueError(f"Semantic cache/photo hash or image-frame mismatch at view {index}")
        cached = readArrays(cachePath)
        if (cached["labelMap"].shape != tuple(reversed(item["orientedSize"]))
                or cached["confidence"].shape != cached["labelMap"].shape):
            raise ValueError(f"Semantic map is not in the original image frame: view {index}")
        labels.append(cached["labelMap"])
        confidences.append(cached["confidence"])
        views.append({"viewIndex": index, "photoSha256": item["sha256"], "semanticCacheSha256": digest})

    observationsPath = sourceExperiment / "observations.npz"
    observationHash = fileHash(observationsPath)
    observations = readArrays(observationsPath)
    needed = {"originalSamplingPixels", "weights", "viewingCosines", "atlasMask"}
    if not needed.issubset(observations):
        raise ValueError("Saved observations lack exact original-photo coordinates or visibility weights")
    texture = readArrays(sourceRun / "texture.npz")
    coverage = np.asarray(texture["coverageMask"])
    if coverage.ndim != 2 or coverage.shape[0] != coverage.shape[1] or coverage.dtype != np.bool_:
        raise ValueError("Expected the current square UFF texture with boolean measured coverage")
    print("Rechecking saved observation coordinates against frozen source geometry/cameras on CPU", flush=True)
    reconstructed = collectAppearanceObservations(readArrays(sourceRun / "geometry.npz"), cameras, inputs,
                                                  readArrays(sourceRun / "delight.npz"), coverage.shape[0])
    exactChecks = {key: bool(np.array_equal(observations[key], reconstructed[key])) for key in needed}
    if not all(exactChecks.values()):
        raise ValueError(f"Saved sampling no longer matches source geometry/cameras: {exactChecks}")
    del reconstructed
    if observations["weights"].shape != (len(inputs), *coverage.shape):
        raise ValueError("Observation dimensions differ from the current UFF texture")
    print("Projecting cached semantics through verified original-image samples", flush=True)
    projected = sampleSemanticObservations(labels, confidences, observations["originalSamplingPixels"], observations["weights"])
    arrays, diagnostics = buildSkinSupport(projected["semanticLabels"], projected["semanticConfidence"],
                                          observations["weights"], observations["viewingCosines"], coverage)
    if not arrays["safeSkinMask"].any():
        raise ValueError("No reliable skin donors remain after conservative semantic protection")
    np.savez_compressed(artifactPath, **arrays)
    writeImage(output / "sourceSafeSkinMask.png", arrays["safeSkinMask"])
    writeImage(output / "sourceProtectedFeatureMask.png", arrays["protectedFeatureMask"])
    writeImage(output / "sourceDonorExcludedMask.png", arrays["donorExcludedMask"])
    writeImage(output / "sourceSemanticConfidence.png", arrays["semanticConfidence"])
    if (fileHash(observationsPath) != observationHash
            or any(fileHash(sourceRun / name) != digest for name, digest in sourceHashes.items())
            or any(fileHash(semanticCache / f"view{view['viewIndex']:02d}.npz") != view["semanticCacheSha256"] for view in views)):
        raise ValueError("Source artifacts changed during semantic projection")
    report = {"schemaVersion": 1, "status": "completed", "sourceRun": sourceRun,
              "sourceExperiment": sourceExperiment, "semanticCache": semanticCache,
              "atlasResolution": coverage.shape[0], "sourceHashes": sourceHashes,
              "observationsSha256": observationHash, "exactObservationReplay": exactChecks,
              "semanticManifestSha256": fileHash(semanticCache / "manifest.json"), "views": views,
              "semanticProjection": projected["diagnostics"], "support": diagnostics,
              "semanticConfidenceMeaning": "winning label weight-times-parser-confidence / qualified view weight; not calibrated certainty",
              "nosePolicy": "whole nose label protected; parser has no separate nostril class",
              "coveragePolicy": "all output masks are subsets of the unchanged measured source coverage",
              "implementationHashes": {"runner": fileHash(Path(__file__)),
                                       "semanticProjection": fileHash(projectRoot / "src/faceAppearance/semanticProjection.py"),
                                       "appearanceObservations": fileHash(projectRoot / "src/faceAppearance/appearanceObservations.py")},
              "artifactSha256": fileHash(artifactPath), "seconds": perf_counter() - start,
              "limitations": ["Automatic image semantics can mislabel skin, shadows, facial hair or feature boundaries.",
                              "Frozen dense UV and geometry/camera correspondences retain their existing alignment error.",
                              "Masks identify candidate donors/protection; extrapolated colors are not measured observations."]}
    writeJson(reportPath, report)
    print(json.dumps({"status": report["status"], "safeSkinTexels": diagnostics["safeSkinTexels"],
                      "protectedFeatureTexels": diagnostics["protectedFeatureTexels"], "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourceRun", type=Path, default=projectRoot / "runs/service/b775794b61894a8b804fddb0620db45d/output")
    parser.add_argument("--sourceExperiment", type=Path, default=projectRoot / "runs/shWildCapClientComparison")
    parser.add_argument("--semanticCache", type=Path, default=projectRoot / "runs/faceSemanticCache")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/sdfSkinTemplateBlendV2")
    args = parser.parse_args()
    prepareSupport(args.sourceRun, args.sourceExperiment, args.semanticCache, args.output)
