"""Opt-in fixed-camera frontal eyelid postprocessing for frozen UFF runs.

All fits use the saved 512px working frame. Original-photo metrics and overlays
remain available; no pose is re-estimated and no source artifact is overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.eyeContours import bindTemplateEyeContours, buildEyeContourConstraints
from faceAppearance.eyeRefinement import refineEyeGeometry
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.texture import fuseTexture
from runEyeRefinementExperiment import loadEyeCache, eyeMetrics, drawComparison
from runReferenceExperiment import readArrays, fileHash
from renderCropComparison import photoCamera


def workingObservations(observations, inputs):
    result = copy.deepcopy(observations)
    for photo, item in zip(result, inputs):
        matrix = np.asarray(item["inputToWorking"], float)
        photo["coordinateFrame"] = "savedWorkingPixels"
        if "diagnostics" in photo:
            photo["originalPhotoDiagnostics"] = photo.pop("diagnostics")
        photo["imageShape"] = list(reversed(item["workingSize"]))
        for eye in photo["eyes"].values():
            if "diagnostics" in eye:
                eye["originalPhotoDiagnostics"] = eye.pop("diagnostics")
            if not eye["valid"]:
                continue
            for key in ("innerCorner", "outerCorner", "upperCurve", "lowerCurve"):
                points = np.asarray(eye[key])
                homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)@matrix.T
                eye[key] = homogeneous[..., :2]/homogeneous[..., 2:]
    return result


def summarizeMetrics(metrics):
    views = {}
    for row in metrics:
        error = (row["upperMeanErrorPixels"]+row["lowerMeanErrorPixels"])/2
        width = row["targetMetrics"]["cornerDistancePixels"]
        views.setdefault(row["viewIndex"], []).append((error, error/width))
    return {view: {"meanContourErrorPixels": float(np.mean([v[0] for v in values])),
                   "meanContourErrorEyeWidthFraction": float(np.mean([v[1] for v in values])),
                   "eyeCount": len(values)} for view, values in views.items()}


def runCase(record, root, *, support=.24, iterations=100, fitName="fit"):
    start = perf_counter()
    caseId = record["caseId"]
    source = Path(record["outputDirectory"])
    case = root / "cases" / caseId
    output = case / fitName
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    geometry = readArrays(source / "geometry.npz")
    inputs = json.loads((source / "inputs.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())
    cameras = [{**camera, **{key: np.asarray(camera[key], float) for key in
                 ("intrinsics", "rotation", "translation")}} for camera in cameras]
    originalCameras = [photoCamera(camera, item, "fitted") for camera, item in zip(cameras, inputs)]
    observations, cacheManifest, hashes = loadEyeCache(case / "eyelids", inputs, minimumRawPeak=.6,
                                                      inputsSha256=fileHash(source / "inputs.json"))
    if not all(observations[0]["eyes"].get(label, {}).get("valid") for label in (4, 5)):
        raise ValueError("Reference photo needs two reliable, separately identified eyes")
    workObs = workingObservations(observations, inputs)
    vertices, faces = geometry["vertices"], geometry["faces"]
    asset = ROOT / "assets/anatomical/uff6986/boundaries.json"
    loops = validateBoundaryAsset(vertices, faces, json.loads(asset.read_text()))
    binding = bindTemplateEyeContours(vertices, faces, {key: value for key, value in loops.items()
                                    if key.endswith("Eye")}, cameras[0], workObs[0])
    shapes = [tuple(reversed(item["workingSize"])) for item in inputs]
    visibility = visibleVertices(vertices, faces, cameras, shapes, minimumCosine=.05, visibilityMode="exact")
    bound = buildEyeContourConstraints(vertices, faces, cameras, binding, workObs, visibility,
                                       cornerWeight=8., maxTargetDistance=28.)
    frontGroups = [item for item in bound["constraints"] if item["viewIndex"] == 0]
    writeJson(output / "constraints.json", bound)
    frontNames = {item["name"] for item in frontGroups}
    if any(eye+"."+part not in frontNames for eye in binding["eyes"] for part in ("upper", "lower")):
        report = {"status": "skipped", "reason": "insufficientVisibleFrontalEyelidCorrespondences",
            "caseId": caseId, "constraintDiagnostics": bound["diagnostics"],
            "sourceHashes": {name: fileHash(source / name) for name in ("geometry.npz", "cameras.json", "inputs.json")},
            "eyeCacheHashes": hashes, "geometryUnchanged": True,
            "limits": ["Reliable photo detections cannot substitute for a visible mesh correspondence."]}
        writeJson(output / "report.json", report)
        return {"caseId": caseId, "label": record["label"], "status": "skipped", "sourceDirectory": str(source),
                "resultPath": str(output / "report.json"), "frontViewIndex": 0,
                "summary": "正面上睑网格对应被背向法线/深度可见性门控排除，保留原始 UFF。"}
    fitStart = perf_counter()
    fitted = refineEyeGeometry(vertices, faces, cameras, bound["constraints"],
            supportRadius=support*float(np.ptp(vertices, axis=0).max()), controlCount=24,
            referenceWeight=12., referenceViewIndex=0, iterations=iterations,
            deformationFrame="referenceImagePlane")
    fitSeconds = perf_counter()-fitStart
    refined = fitted["vertices"]
    finalVisibility = visibleVertices(refined, faces, cameras, shapes, minimumCosine=.05, visibilityMode="exact")
    checks = {"outsideSupportExactlyUnchanged": bool(np.array_equal(vertices[~fitted["activeVertexMask"]],
                                                                 refined[~fitted["activeVertexMask"]])),
              "maximumReferenceDepthChange": fitted["diagnostics"]["maximumReferenceDepthChange"]}
    if not checks["outsideSupportExactlyUnchanged"] or checks["maximumReferenceDepthChange"] > 1e-6:
        raise AssertionError(checks)
    revisedGeometry = {**geometry, "vertices": refined}
    np.savez_compressed(output / "geometryRefined.npz", **revisedGeometry)
    np.savez_compressed(output / "fitState.npz", activeVertexMask=fitted["activeVertexMask"],
        controlPoints=fitted["controlPoints"], controlOffsets=fitted["controlOffsets"],
        initialVisibility=visibility, finalVisibility=finalVisibility)
    metrics = {"before": eyeMetrics(vertices, binding, originalCameras, observations),
               "after": eyeMetrics(refined, binding, originalCameras, observations)}
    workMetrics = {"before": eyeMetrics(vertices, binding, cameras, workObs),
                   "after": eyeMetrics(refined, binding, cameras, workObs)}
    visibilityAudit = [{"name": c["name"], "viewIndex": c["viewIndex"], "samples": len(c["vertexIndices"]),
                        "visibleAfterFit": int(finalVisibility[c["viewIndex"], c["vertexIndices"]].all(1).sum())}
                       for c in bound["constraints"]]
    for name, value in (("templateBinding", binding), ("photoContours", observations),
                        ("workingPhotoContours", workObs), ("constraints", bound),
                        ("history", fitted["history"]), ("eyeMetrics", metrics), ("workingEyeMetrics", workMetrics)):
        writeJson(output / (name+".json"), value)
    drawComparison(output, inputs, originalCameras, geometry, refined, binding, observations)
    fusionStart = perf_counter()
    originalTexture = readArrays(source / "texture.npz")
    texture = fuseTexture(revisedGeometry, readArrays(source / "delight.npz"), cameras,
                          originalTexture["albedo"].shape[0], colorBalance=True, samplingMode="denseUv")
    np.savez_compressed(output / "texture.npz", **{key: value for key, value in texture.items() if isinstance(value, np.ndarray)})
    writeJson(output / "textureDiagnostics.json", texture["diagnostics"])
    fusionSeconds = perf_counter()-fusionStart
    summary = {stage: summarizeMetrics(value) for stage, value in workMetrics.items()}
    report = {"status": "complete", "caseId": caseId, "source": source,
        "geometryDiagnostics": fitted["diagnostics"], "constraintDiagnostics": bound["diagnostics"],
        "eyeMetrics": metrics, "workingEyeMetrics": workMetrics, "workingSummary": summary,
        "fitCoordinateFrame": "saved512pxWorkingCrop", "frontViewIndex": 0,
        "sourceHashes": {name: fileHash(source / name) for name in
            ("geometry.npz", "texture.npz", "delight.npz", "cameras.json", "inputs.json")},
        "eyeCacheHashes": hashes, "detector": cacheManifest["model"], "boundaryAssetSha256": fileHash(asset),
        "checks": checks, "visibilityAfterFit": visibilityAudit,
        "fitSeconds": fitSeconds, "fusionSeconds": fusionSeconds, "totalSeconds": perf_counter()-start,
        "supportRadiusFraction": support, "camerasFrozen": True, "topologyAndUVsPreserved": True,
        "textureSamplingMode": "denseUv", "limits": [
            "Automatic eye curves and template corner extrema are proxies, not anatomical ground truth.",
            "Scores are in-sample; fixed-camera visual comparison is required.",
            "Preserved frontal depth does not establish correct 3D shape or camera pose.",
            "No reliable lip/nose surface binding is available; only compact eye displacement is fitted.",
            "denseUv retains UFF pixel-to-UV correspondences; geometry changes re-evaluate visibility/weights, not those correspondences.",
            "No eyeballs are added. Triangle quality gates do not prove absence of self-intersection."]}
    report["outputHashes"] = {name: fileHash(output / name) for name in ("geometryRefined.npz", "texture.npz", "fitState.npz")}
    writeJson(output / "report.json", report)
    print(json.dumps({"caseId": caseId, "status": fitted["diagnostics"]["status"], "summary": summary,
                      "fitSeconds": fitSeconds, "fusionSeconds": fusionSeconds}), flush=True)
    return {"caseId": caseId, "label": record["label"], "status": "succeeded", "sourceDirectory": str(source),
            "geometryPath": str(output / "geometryRefined.npz"), "texturePath": str(output / "texture.npz"),
            "resultPath": str(output / "report.json"), "frontViewIndex": 0, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/uffFrontPriority20261011")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/uffMultiCase20261011/batchManifest.json")
    parser.add_argument("--case", action="append")
    parser.add_argument("--support", type=float, default=.24)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fit-name", default="fit")
    parser.add_argument("--refit-success", action="store_true")
    args = parser.parse_args()
    if args.fit_name not in ("fit", "fitV2"):
        raise ValueError("Unsupported fit output name")
    args.root = args.root.resolve()
    source = json.loads(args.source.read_text())
    target = args.root / "frontManifest.json"
    manifest = json.loads(target.read_text()) if target.exists() else {
        "schemaVersion": 1, "sourceManifest": str(args.source.resolve()), "cases": []}
    selected = [row for row in source["cases"] if row["status"] == "succeeded"
                and (not args.case or row["caseId"] in args.case)]
    for row in selected:
        previous = next((item for item in manifest["cases"] if item["caseId"] == row["caseId"]), None)
        if previous and previous["status"] == "succeeded" and args.refit_success:
            manifest.setdefault("previousCandidates", []).append(previous)
            manifest["cases"].remove(previous)
            previous = None
        if previous and previous["status"] == "failed" and args.retry_failed:
            from datetime import datetime, timezone
            fitPath = args.root / "cases" / row["caseId"] / "fit"
            if fitPath.exists():
                fitPath.rename(fitPath.with_name("failedFit-"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
            manifest.setdefault("failedAttempts", []).append(previous)
            manifest["cases"].remove(previous)
        if any(item["caseId"] == row["caseId"] for item in manifest["cases"]):
            continue
        try:
            item = runCase(row, args.root, support=args.support, iterations=args.iterations, fitName=args.fit_name)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            item = {"caseId": row["caseId"], "label": row["label"], "status": "failed", "error": repr(exc),
                    "sourceDirectory": row["outputDirectory"]}
        manifest["cases"].append(item)
        manifest["status"] = "running"
        writeJson(target, manifest)
    manifest["status"] = "complete" if len(manifest["cases"]) == 6 else "partial"
    manifest["outcomes"] = {state: sum(row["status"] == state for row in manifest["cases"])
                            for state in ("succeeded", "skipped", "failed")}
    writeJson(target, manifest)
    snapshot = args.root / "implementation"
    for name in ("tools/runUffFrontPriority.py", "tools/runEyeRefinementExperiment.py",
                 "src/faceAppearance/eyeRefinement.py", "src/faceAppearance/eyeContours.py",
                 "src/faceAppearance/texture.py"):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    if manifest["outcomes"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
