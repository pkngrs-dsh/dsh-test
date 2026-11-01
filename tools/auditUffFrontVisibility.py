"""Read-only separation of frontal eyelid normal and exact-depth visibility gates."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.eyeContours import bindTemplateEyeContours
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from faceAppearance.texture import vertexNormals
from renderCropComparison import fileHash
from runEyeRefinementExperiment import loadEyeCache
from runUffFrontPriority import workingObservations


def independentRayDepth(cameraVertices, faces, query):
    """3D Moller-Trumbore, no screen-space bins or projected barycentrics."""
    triangles = np.asarray(cameraVertices, float)[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    offset = -triangles[:, 0]
    q = np.cross(offset, edge1)
    numerator = np.sum(edge2*q, axis=1)
    result = []
    for point in query:
        direction = point / point[2]
        p = np.cross(direction, edge2)
        determinant = np.sum(edge1*p, axis=1)
        nondegenerate = np.abs(determinant) > 1e-12
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=nondegenerate)
        u = np.sum(offset*p, axis=1) * inverse
        v = np.sum(direction*q, axis=1) * inverse
        depth = numerator * inverse
        valid = (nondegenerate & (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
                 & (depth > 1e-8) & (triangles[:, :, 2] > 1e-8).all(1))
        result.append(float(depth[valid].min()) if valid.any() else None)
    return result


def statistics(values):
    values = np.asarray(values)
    return dict(zip(("minimum", "median", "maximum"), map(float, np.quantile(values, (0, .5, 1)))))


def run():
    root = ROOT / "runs/uffFrontPriority20261011"
    report = {"schemaVersion": 1, "status": "complete", "viewIndex": 0,
              "minimumCosine": .05, "depthToleranceFraction": .005,
              "coordinateFrame": "saved512pxWorkingCrop", "cases": [],
              "method": "Exact screen-space nearest triangle depth cross-checked against independent 3D Moller-Trumbore rays",
              "limits": ["Fixed automatic corner/path bindings remain proxies, not reviewed anatomy.",
                         "No threshold, model, fit, camera, input or cached observation is modified.",
                         "Hidden topology boundary points need not coincide with the visible silhouette."]}
    for caseId in ("outdoorSideLight4", "cyanWall4", "orbitControl4"):
        source = ROOT / "runs/uffMultiCase20261011/cases" / caseId / "reconstruction"
        with np.load(source / "geometry.npz") as archive:
            vertices, faces = archive["vertices"], archive["faces"]
        inputs = json.loads((source / "inputs.json").read_text(encoding="utf-8"))
        cameras = json.loads((source / "cameras.json").read_text(encoding="utf-8"))
        camera = {**cameras[0], **{key: np.asarray(cameras[0][key], float)
                                  for key in ("intrinsics", "rotation", "translation")}}
        observations, _, _ = loadEyeCache(root / "cases" / caseId / "eyelids", inputs,
                                           inputsSha256=fileHash(source / "inputs.json"))
        work = workingObservations(observations, inputs)
        asset = ROOT / "assets/anatomical/uff6986/boundaries.json"
        loops = validateBoundaryAsset(vertices, faces, json.loads(asset.read_text(encoding="utf-8")))
        loops = {name: value for name, value in loops.items() if name.endswith("Eye")}
        binding = bindTemplateEyeContours(vertices, faces, loops, camera, work[0])
        shape = tuple(reversed(inputs[0]["workingSize"]))
        transformed = vertices @ camera["rotation"].T + camera["translation"]
        normals = vertexNormals(vertices, faces) @ camera["rotation"].T
        directions = -transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
        cosine = np.sum(normals*directions, axis=1)
        points, depth = projectPoints(vertices, camera["intrinsics"], camera["rotation"], camera["translation"])
        sampler = ProjectedSurfaceVisibility(transformed, faces, camera["intrinsics"], shape)
        allIndices = np.unique(np.concatenate(list(loops.values())))
        exact = sampler.closestDepth(transformed[allIndices])
        independent = np.asarray(independentRayDepth(transformed, faces, transformed[allIndices]), float)
        if not np.isfinite(independent).all() or not np.isfinite(exact["depth"]).all():
            raise ValueError("Eye rim ray unexpectedly lacks a triangle hit")
        deltas = depth[allIndices] - exact["depth"]
        tolerance = .005 * float(np.ptp(vertices, axis=0).max())
        computedVisible = visibleVertices(vertices, faces, [camera], [shape], minimumCosine=.05,
                                          visibilityMode="exact")[0]
        indexToRow = {int(index): row for row, index in enumerate(allIndices)}
        directMask = ((cosine[allIndices] > .05) & (np.abs(deltas) <= tolerance)
                      & (depth[allIndices] > 0) & exact["insidePhoto"])
        if not np.array_equal(directMask, computedVisible[allIndices]):
            raise AssertionError("Independent gate decomposition does not reproduce visibleVertices")
        result = {"caseId": caseId, "depthTolerance": tolerance, "eyeRimVertexCount": len(allIndices),
                  "independentRayMaximumDepthDifference": float(np.max(np.abs(independent-exact["depth"]))),
                  "allEyeRimPointsInsidePhotoAndPositiveDepth": bool(exact["insidePhoto"].all() and (depth[allIndices] > 0).all()),
                  "visibilityGateExactlyReproduced": True,
                  "sourceHashes": {name: fileHash(source / name) for name in ("geometry.npz", "cameras.json", "inputs.json")},
                  "eyes": {}}
        for eyeName, eye in binding["eyes"].items():
            eyeResult = {"labelId": eye["labelId"], "loopVertexCount": len(loops[eyeName]), "parts": {}}
            targetEye = work[0]["eyes"][eye["labelId"]]
            for part in ("upper", "lower"):
                path = np.asarray(eye[part]["pathVertexIndices"])
                rows = np.asarray([indexToRow[int(index)] for index in path])
                co, dz = cosine[path], deltas[rows]
                normalFailed, depthFailed = co <= .05, np.abs(dz) > tolerance
                data = {"pathVertexCountIncludingBothCorners": len(path),
                        "normalGateFailedCount": int(normalFailed.sum()),
                        "strictlyBackFacingVertexNormalCount": int((co <= 0).sum()),
                        "grazingPositiveBelowThresholdCount": int(((co > 0) & (co <= .05)).sum()),
                        "exactDepthGateFailedCount": int(depthFailed.sum()),
                        "bothFailedCount": int((normalFailed & depthFailed).sum()),
                        "normalOnlyFailedCount": int((normalFailed & ~depthFailed).sum()),
                        "depthOnlyFailedCount": int((~normalFailed & depthFailed).sum()),
                        "acceptedVertexCount": int(computedVisible[path].sum()),
                        "cosine": statistics(co), "signedDepthDelta": statistics(dz),
                        "absoluteDepthDeltaToleranceRatio": statistics(np.abs(dz)/tolerance),
                        "vertices": [{"vertexId": int(index), "pixel": points[index],
                                      "cosine": float(cosine[index]), "queryDepth": float(depth[index]),
                                      "closestDepth": float(exact["depth"][row]),
                                      "depthDelta": float(deltas[row]), "normalAccepted": bool(cosine[index] > .05),
                                      "depthAccepted": bool(abs(deltas[row]) <= tolerance),
                                      "nearestFaceId": int(exact["faceIds"][row]),
                                      "nearestFaceVertices": faces[exact["faceIds"][row]],
                                      "visible": bool(computedVisible[index])}
                                     for index, row in zip(path, rows)]}
                indices = np.asarray(eye[part]["vertexIndices"])
                weights = np.asarray(eye[part]["barycentric"])
                samplePoints = (vertices[indices]*weights[..., None]).sum(1)
                samplePixels, sampleDepth = projectPoints(samplePoints, camera["intrinsics"], camera["rotation"], camera["translation"])
                target = np.asarray(targetEye[part+"Curve"])
                confidence = np.asarray(targetEye[part+"Confidence"])
                distance = np.linalg.norm(samplePixels-target, axis=1)
                selected = np.arange(1, len(indices)-1)
                sampleNormal = (cosine[indices] > .05).all(1)
                sampleDelta = np.asarray([[deltas[indexToRow[int(index)]] for index in pair] for pair in indices])
                sampleDepthGate = (np.abs(sampleDelta) <= tolerance).all(1)
                sampleVisibility = computedVisible[indices].all(1)
                sampleValid = sampleVisibility & (distance <= 28.) & (confidence >= .4) & (sampleDepth > 0)
                data["curveSamplesExcludingCorners"] = {
                    "candidateCount": len(selected), "normalEndpointGateFailedCount": int((~sampleNormal[selected]).sum()),
                    "depthEndpointGateFailedCount": int((~sampleDepthGate[selected]).sum()),
                    "visibilityFailedCount": int((~sampleVisibility[selected]).sum()),
                    "distanceOver28PixelsCount": int((distance[selected] > 28.).sum()),
                    "confidenceBelowPointFourCount": int((confidence[selected] < .4).sum()),
                    "retainedCount": int(sampleValid[selected].sum()), "distancePixels": statistics(distance[selected])}
                eyeResult["parts"][part] = data
            result["eyes"][eyeName] = eyeResult
        report["cases"].append(result)
        print(caseId, "ray error", result["independentRayMaximumDepthDifference"],
              "negative upper", result["eyes"]["negativeXEye"]["parts"]["upper"]["curveSamplesExcludingCorners"])
    report["implementationHashes"] = {name: fileHash(ROOT / name) for name in (
        "tools/auditUffFrontVisibility.py", "src/faceAppearance/semanticProjection.py",
        "src/faceAppearance/surfaceVisibility.py", "src/faceAppearance/eyeContours.py")}
    writeJson(root / "frontVisibilityAudit.json", report)


if __name__ == "__main__":
    run()
