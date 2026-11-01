"""Inspect original client eye semantics without fitting or relabelling them.

The 32 legacy MP samples, internal topological holes and rendered eye-material
boundaries are three different objects.  This diagnostic records their evidence
separately.  Original centimeters and original face/barycentric bindings survive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.clientLandmarks import loadClientLandmarks, sampleClientLandmarks
from faceAppearance.eyeSocketVisibility import _raster, buildVisibleEyeRimBinding


EYE_PATHS = {
    "rightEye": {"upper": [33, 246, 161, 160, 159, 158, 157, 173, 133],
                 "lower": [33, 7, 163, 144, 145, 153, 154, 155, 133]},
    "leftEye": {"upper": [263, 466, 388, 387, 386, 385, 384, 398, 362],
                "lower": [263, 249, 390, 373, 374, 380, 381, 382, 362]},
}
VIEWS = {"front": 0., "leftOblique": 35., "rightOblique": -35., "profile": 90.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def nearestContour(point, contours):
    if not contours:
        return None
    starts = np.concatenate(contours).astype(float)
    ends = np.concatenate([np.roll(contour, -1, axis=0) for contour in contours]).astype(float)
    segment = ends-starts
    lengths = (segment*segment).sum(1)
    fraction = np.divide(((point-starts)*segment).sum(1), lengths,
                         out=np.zeros(len(starts)), where=lengths > 0).clip(0, 1)
    nearest = starts+fraction[:, None]*segment
    distances = np.linalg.norm(nearest-point, axis=1)
    index = int(distances.argmin())
    return float(distances[index]), nearest[index]


def exactFrontHit(point, vertices, faces, opaqueIds):
    """Independent exact orthographic ray; no rounded-pixel visibility verdict."""
    triangles = vertices[faces[opaqueIds]]
    first, second, third = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = ((second[:, 1]-third[:, 1])*(first[:, 0]-third[:, 0])
                   +(third[:, 0]-second[:, 0])*(first[:, 1]-third[:, 1]))
    valid = np.abs(denominator) > 1e-12
    safe = np.where(valid, denominator, 1.)
    w0 = ((second[:, 1]-third[:, 1])*(point[0]-third[:, 0])
          +(third[:, 0]-second[:, 0])*(point[1]-third[:, 1]))/safe
    w1 = ((third[:, 1]-first[:, 1])*(point[0]-third[:, 0])
          +(first[:, 0]-third[:, 0])*(point[1]-third[:, 1]))/safe
    weights = np.c_[w0, w1, 1-w0-w1]
    valid &= (weights >= -1e-8).all(1)
    if not valid.any():
        return None
    ids = np.flatnonzero(valid)
    depths = (weights[ids]*triangles[ids, :, 2]).sum(1)
    local = int(ids[int(depths.argmax())])
    return int(opaqueIds[local]), weights[local], float(depths.max())


def crossCheckSemanticModule(arrays, originalBinding, allPoints):
    """Cross-check the new module against this independently decoded source."""
    from faceAppearance.semanticEyelidBinding import buildSemanticEyelidBinding, sampleSemanticEyelids

    semantic = buildSemanticEyelidBinding(arrays["nicpFaces"])
    sampled = sampleSemanticEyelids(arrays["vertices"][arrays["nicpToFullIds"]], semantic)
    checkedIds = set()
    rowsChecked = 0
    for name, moduleName in [("rightEye", "subjectRight"), ("leftEye", "subjectLeft")]:
        for role in ("upper", "lower", "innerCorner", "outerCorner"):
            ids = (list(reversed(EYE_PATHS[name][role])) if role in ("upper", "lower")
                   else [EYE_PATHS[name]["upper"][-1 if role == "innerCorner" else 0]])
            rows = semantic["eyes"][moduleName][role]
            assert np.array_equal(rows["mpIds"], ids)
            for key in ("faceIds", "vertexIndices", "barycentricWeights"):
                assert np.array_equal(rows[key], originalBinding[key][ids])
            expected = allPoints[ids][0] if role.endswith("Corner") else allPoints[ids]
            assert np.array_equal(sampled[moduleName][role], expected)
            rowsChecked += len(ids)
            checkedIds.update(ids)
    return dict(accepted=True, exactSourceRows=True, exactSamplePositions=True,
                uniquePointCount=len(checkedIds), rowsIncludingSharedCorners=rowsChecked,
                direction="Module inner-to-outer is the exact reverse of audit outer-to-inner",
                modulePath="src/faceAppearance/semanticEyelidBinding.py",
                moduleSha256=digest(ROOT / "src/faceAppearance/semanticEyelidBinding.py"),
                bindingSha256=semantic["bindingSha256"], reviewStatus=semantic["reviewStatus"])


def drawView(path, viewName, angle, eyePlots, resolution):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), layout="constrained")
    for axis, plot in zip(axes, eyePlots):
        halfExtent = plot["halfExtent"]
        extent = (-halfExtent, halfExtent, -halfExtent, halfExtent)
        axis.imshow(plot["image"], extent=extent, origin="upper", interpolation="nearest")
        visibleFaces = np.unique(plot["hits"][plot["hits"] >= 0])
        visibleSkin = visibleFaces[plot["materials"][visibleFaces] == 0]
        triangles = plot["vertices"][plot["faces"][visibleSkin], :2]
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        axis.add_collection(LineCollection(edges, colors="#72808c", linewidths=.35, alpha=.55))
        hole = plot["hole"]
        axis.plot(*np.vstack([hole, hole[0]])[:, :2].T, color="#9846cf", linewidth=1.2,
                  linestyle="--", label="Internal hole (X-ray overlay)")
        for label, color in [("upper", "#d96d13"), ("lower", "#2474b5")]:
            points = plot["paths"][label]
            axis.plot(points[:, 0], points[:, 1], color=color, alpha=.7,
                      linewidth=1.2, label=f"Original MP {label} sequence (projection)")
        for contour in plot["contours"]:
            xy = (contour-(resolution-1)/2)*plot["unitsPerPixel"]
            xy[:, 1] *= -1
            xy = np.vstack([xy, xy[0]])
            axis.plot(xy[:, 0], xy[:, 1], color="#ef3333", linewidth=1.3)
        for record, point in zip(plot["records"], plot["points"]):
            hidden = record["exactRayOccluded"]
            color = "#bd1038" if hidden else "#087a35"
            axis.scatter(point[0], point[1], s=23, facecolors="white" if hidden else color,
                         edgecolors=color, linewidth=.9, zorder=10)
            axis.annotate(str(record["mpId"]), point[:2], xytext=(3, 4),
                          textcoords="offset points", fontsize=7, color=color, zorder=11)
        axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal",
                 xlabel="Camera X relative to eye sphere center (cm)",
                 ylabel="Camera Y relative to eye sphere center (cm)",
                 title=f"{plot['eyeName']} | {sum(r['exactRayOccluded'] is True for r in plot['records'])}/16 MP samples occluded")
        axis.legend(loc="lower left", fontsize=7, framealpha=.85)
    fig.suptitle(f"Original male template: {viewName}, yaw {angle:+g} degrees\n"
                 "Red: rendered eye-material boundary. Green solid: visible MP. Red open: occluded MP.\n"
                 "Blue: target eye. Purple fill: other eye. Hole/path lines are X-ray projections, NOT anatomical truth. Alpha cards excluded.", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import cv2

    templateDir = ROOT / "assets/clientTemplates/legacyMaleDirectServiceV1"
    output = ROOT / "reports/semanticEyelid20261024"
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(templateDir / "template.npz", allow_pickle=False))
    components = json.loads((templateDir / "components.json").read_text())
    vertices = arrays["vertices"].astype(float)
    faces = arrays["faces"]
    materials = arrays["materialIds"]
    faceComponents = np.full(len(faces), "unassigned", dtype=object)
    for component in components:
        faceComponents[component["faceIndices"]] = component["name"]
    nicpIds = arrays["nicpToFullIds"]
    binding = loadClientLandmarks(arrays["nicpFaces"])
    # Independently replay original JSON rather than using the loader as oracle.
    original = json.loads(Path(binding["sourcePath"]).read_text())
    rows = {int(key): value for key, value in original.items()}
    allPoints = sampleClientLandmarks(vertices[nicpIds], binding)
    loops = extractMeshBoundaryLoops(vertices[nicpIds], arrays["nicpFaces"])["loops"]
    originalIds = sorted(set(sum([value for paths in EYE_PATHS.values() for value in paths.values()], [])))
    for mpId in originalIds:
        faceId, u, v = rows[mpId]
        weights = np.array([u, v, 1-float(u)-float(v)], float)
        assert int(faceId) == binding["faceIds"][mpId]
        assert np.array_equal(weights, binding["barycentricWeights"][mpId])
        expected = (vertices[nicpIds[arrays["nicpFaces"][int(faceId)]]]*weights[:, None]).sum(0)
        assert np.array_equal(expected, allPoints[mpId])
    opaqueIds = np.flatnonzero(np.isin(materials, [0, 1, 2]))
    resolution = 192
    eyes = {}
    for name in EYE_PATHS:
        sign = -1 if name == "rightEye" else 1
        component = next(c for c in components if c["materialId"] == 2
                         and vertices[c["vertexIndices"], 0].mean()*sign > 0)
        rim = buildVisibleEyeRimBinding(arrays, component, resolution=resolution)
        hole = next(np.asarray(loop, int) for loop in loops
                    if len(loop) == 10 and vertices[nicpIds[loop], 0].mean()*sign > 0)
        eyes[name] = dict(component=component, rim=rim, hole=hole,
                          ids=sorted(set(EYE_PATHS[name]["upper"]+EYE_PATHS[name]["lower"])))
    # Equal physical crop across both eyes and every direction. This differs
    # slightly from the individual front-only buildVisibleEyeRimBinding crop.
    halfExtent = max(e["rim"]["diagnostics"]["cropHalfExtent"] for e in eyes.values())
    unitsPerPixel = 2*halfExtent/(resolution-1)
    report = dict(schemaVersion=1, status="readOnlyTemplateAuditComplete", templateId=templateDir.name,
                  sourceHashes={str(templateDir / name): digest(templateDir / name)
                                for name in ["template.npz", "components.json", "manifest.json"]},
                  originalBindingSource=binding["sourcePath"], originalBindingSha256=binding["sourceSha256"],
                  auditToolSha256=digest(__file__), originalJsonReplayExact=True,
                  semanticModuleCrossCheck=crossCheckSemanticModule(arrays, binding, allPoints),
                  pointCount=32, fittingPerformed=False, geometryChanged=False,
                  anatomicalValidation=False, anatomicalErrorInferredFromBoundaryDistance=False,
                  coordinates="Original template centimeters; per-eye sphere center subtracted before rotation",
                  projection="Orthographic; camera looks toward negative camera Z; larger Z occludes smaller Z",
                  resolution=resolution, sharedCropHalfExtentCm=halfExtent, unitsPerPixelCm=unitsPerPixel,
                  opaqueMaterials=[0, 1, 2], excludedAlphaMaterial=3, twoSidedOcclusion=True,
                  visibilityToleranceCm=1e-7,
                  limitations=["MP IDs are legacy semantic candidates, not reviewed anatomical truth.",
                               "Boundary distances measure projected eye-material contours, not anatomical error; canthi may differ legitimately.",
                               "Raster boundary is quantized at 192 pixels; exact point visibility uses an independent continuous ray query.",
                               "Alpha eyelash/brow cards are excluded consistently with eyeSocketVisibility.",
                               "An occluded point or asymmetric depth alone does not prove incorrect annotation.",
                               "X-ray hole/path overlay lines are not surface curves or visible silhouettes."],
                  views={})
    for viewName, angle in VIEWS.items():
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        viewReport = dict(yawDegrees=angle, rotationColumnVector=rotation, eyes={})
        eyePlots = []
        for name, eye in eyes.items():
            center = eye["rim"]["sphereCenter"]
            cameraVertices = (vertices-center) @ rotation.T
            cameraPoints = (allPoints[eye["ids"]]-center) @ rotation.T
            pixels = cameraVertices[:, :2]/unitsPerPixel
            pixels[:, 1] *= -1
            pixels += (resolution-1)/2
            pointPixels = cameraPoints[:, :2]/unitsPerPixel
            pointPixels[:, 1] *= -1
            pointPixels += (resolution-1)/2
            hits, rasterBary = _raster(cameraVertices, faces, opaqueIds, pixels, resolution)
            hitValid = hits >= 0
            depthImage = np.full(hits.shape, np.nan)
            depthImage[hitValid] = (cameraVertices[faces[hits[hitValid]], 2]*rasterBary[hitValid]).sum(1)
            mask = hitValid & np.isin(hits, eye["component"]["faceIndices"])
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [contour[:, 0].astype(float) for contour in contours if len(contour)]
            records = []
            for mpId, point, pixel in zip(eye["ids"], cameraPoints, pointPixels):
                corner = "outerCanthus" if mpId == EYE_PATHS[name]["upper"][0] else (
                    "innerCanthus" if mpId == EYE_PATHS[name]["upper"][-1] else None)
                role = corner or ("upperEyelid" if mpId in EYE_PATHS[name]["upper"] else "lowerEyelid")
                pixelInt = np.rint(pixel).astype(int)
                inCrop = bool((pixelInt >= 0).all() and (pixelInt < resolution).all())
                hit = int(hits[pixelInt[1], pixelInt[0]]) if inCrop else -1
                if hit >= 0:
                    centerPoint = np.r_[(pixelInt-(resolution-1)/2)*unitsPerPixel, 0.]
                    centerPoint[1] *= -1
                    centerRay = exactFrontHit(centerPoint, cameraVertices, faces, opaqueIds)
                    assert centerRay is not None
                    assert abs(centerRay[2]-depthImage[pixelInt[1], pixelInt[0]]) < 1e-8
                ray = exactFrontHit(point, cameraVertices, faces, opaqueIds)
                boundary = nearestContour(pixel, contours)
                boundTriangle = cameraVertices[nicpIds[binding["vertexIndices"][mpId]]]
                boundNormal = np.cross(boundTriangle[1]-boundTriangle[0], boundTriangle[2]-boundTriangle[0])
                boundNormal /= np.linalg.norm(boundNormal)
                record = dict(mpId=mpId, semanticRole=role, originalNicpFaceId=int(binding["faceIds"][mpId]),
                              originalNicpVertexIds=binding["vertexIndices"][mpId],
                              originalFullVertexIds=nicpIds[binding["vertexIndices"][mpId]],
                              originalBarycentricWeights=binding["barycentricWeights"][mpId],
                              originalRestPointCm=allPoints[mpId], cameraPointCm=point, pixelXY=pixel,
                              originalBoundFaceUnitNormalCamera=boundNormal,
                              rasterSamplePixelXY=pixelInt, rasterInCrop=inCrop,
                              rasterPixelCenterOffsetPx=float(np.linalg.norm(pixel-pixelInt)),
                              rasterFrontFaceId=hit, rasterFrontMaterialId=int(materials[hit]) if hit >= 0 else None,
                              rasterFrontDepthCm=float(depthImage[pixelInt[1], pixelInt[0]]) if hit >= 0 else None,
                              rasterFrontMinusSampleDepthCm=float(depthImage[pixelInt[1], pixelInt[0]]-point[2]) if hit >= 0 else None,
                              exactRayFrontFaceId=ray[0] if ray else None,
                              exactRayFrontMaterialId=int(materials[ray[0]]) if ray else None,
                              exactRayFrontComponentName=str(faceComponents[ray[0]]) if ray else None,
                              exactRayFrontBarycentricWeights=ray[1] if ray else None,
                              exactRayFrontDepthCm=ray[2] if ray else None,
                              exactRayFrontMinusSampleDepthCm=ray[2]-float(point[2]) if ray else None,
                              exactRayOccluded=bool(ray[2]-point[2] > 1e-7) if ray else None,
                              visibleEyeBoundaryDistancePixels=boundary[0] if boundary else None,
                              visibleEyeBoundaryDistanceCm=boundary[0]*unitsPerPixel if boundary else None,
                              nearestVisibleEyeBoundaryPixelXY=boundary[1] if boundary else None,
                              templateFrontSampledSkinRimDistance3dCm=float(np.linalg.norm(eye["rim"]["restPoints"]-allPoints[mpId], axis=1).min()))
                records.append(record)
            viewReport["eyes"][name] = dict(sphereCenterOriginalCm=center, sphereRadiusCm=eye["rim"]["sphereRadius"],
                                              holeNicpVertexIds=eye["hole"], holeFullVertexIds=nicpIds[eye["hole"]],
                                              opaqueEyePixels=int(mask.sum()), contourCount=len(contours),
                                              contourClipped=bool(mask[[0, -1]].any() or mask[:, [0, -1]].any()),
                                              exactRayOccludedPointCount=sum(r["exactRayOccluded"] is True for r in records),
                                              rasterPixelDepthVerifiedByIndependentRay=True,
                                              points=records)
            image = np.ones((resolution, resolution, 3))*.98
            palette = {0: [.81, .81, .79], 1: [.65, .47, .40], 2: [.72, .57, .80]}
            for materialId, color in palette.items():
                image[hitValid & (materials[np.maximum(hits, 0)] == materialId)] = color
            image[mask] = [.35, .65, .86]
            eyePlots.append(dict(eyeName=name, halfExtent=halfExtent, unitsPerPixel=unitsPerPixel,
                                 image=image, vertices=cameraVertices, faces=faces, materials=materials,
                                 hits=hits, hole=cameraVertices[nicpIds[eye["hole"]]], records=records,
                                 paths={role: (allPoints[ids]-center) @ rotation.T
                                        for role, ids in EYE_PATHS[name].items()},
                                 contours=contours, points=cameraPoints))
        figureName = f"template_{viewName}.png"
        drawView(output / figureName, viewName, angle, eyePlots, resolution)
        viewReport["figure"] = figureName
        report["views"][viewName] = viewReport
    (output / "templateAudit.json").write_text(json.dumps(report, default=serial, indent=2)+"\n")
    print(json.dumps({"output": str(output / "templateAudit.json"), "sourceReplayExact": True,
                      "views": {name: {eye: {key: data[key] for key in ["opaqueEyePixels", "exactRayOccludedPointCount"]}
                                       for eye, data in value["eyes"].items()}
                                for name, value in report["views"].items()}}, indent=2))


if __name__ == "__main__":
    main()
