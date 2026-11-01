"""Inspect actual UFF mesh holes and optionally fit independent photo contours.

The template annotations below are specific to the verified UFF6986 template.
Unknown topologies are rejected. Nose-region segmentation is never used as a
nostril target, and outer lip contours are never used as an oral aperture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion, binary_dilation

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.anatomicalBoundaries import (boundaryVisibilityDiagnostics, buildAnatomicalConstraints, createBoundaryAsset,
    extractMeshBoundaryLoops, lipInterfaceMask, pairBoundaryRims, rasterizeBoundaryAnnotations,
    surfacePathBindings, validateBoundaryAsset)
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.geometryRefinement import refineLocalGeometry
from faceAppearance.semanticProjection import visibleVertices
from renderCropComparison import fileHash, photoCamera
from runReferenceExperiment import readArrays
from runSemanticRefinementExperiment import loadSemanticCache


def readTemplate(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts and parts[0] == "f":
            faces.append([int(value.split("/")[0])-1 for value in parts[1:]])
    return np.asarray(vertices), np.asarray(faces)


def runDiagnostic(source, semanticCache, output, *, fit=False, iterations=180,
                  includeLipInterface=False, targetAnnotations=None, surfaceFeatures=None):
    start = perf_counter()
    source, semanticCache, output = (Path(path).resolve() for path in (source, semanticCache, output))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Completed diagnostic exists; choose another output directory")
    geometry = readArrays(source / "geometry.npz")
    vertices, faces = geometry["vertices"], geometry["faces"]
    inputs = json.loads((source / "inputs.json").read_text())
    savedCameras = json.loads((source / "cameras.json").read_text())
    cameras = [photoCamera(camera, item, "fitted") for camera, item in zip(savedCameras, inputs)]
    semantics, semanticManifest = loadSemanticCache(semanticCache, inputs)
    template = projectRoot / "assets/wrapPreparation/uvFaceFusionFixed.obj"
    templateVertices, templateFaces = readTemplate(template)
    if len(vertices) != 6986 or not np.array_equal(faces, templateFaces):
        raise ValueError("Diagnostic annotations only apply to numbered UFF6986 template")
    extracted = extractMeshBoundaryLoops(templateVertices, templateFaces)
    if extracted["diagnostics"]["loopVertexCounts"] != [112, 64, 60, 60, 26, 26]:
        raise ValueError("Unexpected UFF topology boundary structure")
    annotations = {
        0: {"name": "outerFace", "description": "Open face-patch perimeter; not a full-head silhouette"},
        1: {"name": "oralAperture", "description": "True oral mesh hole rim; not outer lip boundary"},
        2: {"name": "positiveXEye", "description": "Eye hole at positive template X; anatomical laterality not assumed"},
        3: {"name": "negativeXEye", "description": "Eye hole at negative template X; anatomical laterality not assumed"},
        4: {"name": "positiveXNostril", "description": "Nostril hole at positive template X; no parser nostril target"},
        5: {"name": "negativeXNostril", "description": "Nostril hole at negative template X; no parser nostril target"}}
    asset = createBoundaryAsset(templateVertices, templateFaces, annotations,
                source=str(template), reviewStatus="template and four-view overlays inspected; revise names if needed")
    asset["sourceFileSha256"] = fileHash(template)
    loops = validateBoundaryAsset(vertices, faces, asset)
    writeJson(output / "boundaryAsset.json", asset)
    projected = [projectPoints(vertices, camera["intrinsics"], camera["rotation"], camera["translation"])[0]
                 for camera in cameras]
    bindings = {name: {"vertexIndices": loop[:, None], "barycentric": np.ones((len(loop), 1))}
                for name, loop in loops.items() if "Eye" in name or "Nostril" in name}
    mouthBinding = pairBoundaryRims(projected[0], loops["oralAperture"])
    bindings["oralCenterline"] = mouthBinding
    featureTemplate = {"schemaVersion": 1, "topologySha256": asset["topologySha256"],
        "instructions": "Optional reviewed surface features, separate from topology hole rims. The UFF oral aperture can be occluded behind lips: do not equate it with the visible lip seam. Add surfacePaths records containing unique name (for example visibleLipSeam), connected zero-based original vertexIndices, and explicit closed boolean. Supply a matching photo contour under the same name. Paths are topology-validated and depth-tested.",
        "surfacePaths": []}
    writeJson(output / "surfaceFeatures.template.json", featureTemplate)
    if surfaceFeatures is not None:
        featureAsset = json.loads(Path(surfaceFeatures).read_text())
        additional = surfacePathBindings(vertices, faces, featureAsset)
        if set(additional) & set(bindings):
            raise ValueError("Surface feature names must not replace topology boundary names")
        bindings.update(additional)
    targetContours, targetReports = [], []
    for viewIndex, result in enumerate(semantics):
        labels, confidence = result["labelMap"], result["confidence"]
        eyeRegion = np.isin(labels, [4, 5])
        eyeContour = eyeRegion & ~binary_erosion(eyeRegion)
        reliable = np.where(confidence >= .55, confidence, 0)
        seam = lipInterfaceMask(labels, confidence, minConfidence=.55)
        targets = {name: {"contourMask": eyeContour, "confidence": reliable}
                   for name in bindings if "Eye" in name}
        # An open mouth's upper/lower interface is interrupted and does not
        # define its complete centerline. Exclude that photo rather than join it.
        innerMouthCount = int((labels == 11).sum())
        if includeLipInterface and seam.sum() >= 24 and innerMouthCount == 0:
            targets["oralCenterline"] = {"contourMask": seam, "confidence": reliable}
        targetContours.append(targets)
        targetReports.append({"viewIndex": viewIndex, "eyeContourPixels": int(eyeContour.sum()),
             "confidentLipInterfacePixels": int(seam.sum()), "innerMouthPixels": innerMouthCount,
             "mouthConstraint": "upperLowerLipContactInterface" if "oralCenterline" in targets
                                 else "skippedDisabledOrOpenOrUncertainMouthInterface",
             "nostrilConstraint": "skippedNoIndependentNostrilLabels"})
    annotationTemplate = {"schemaVersion": 1, "coordinateFrame": "originalExifOrientedPixels",
        "instructions": "Add reviewed contours to each views.contours list. Built-in names: positiveXEye, negativeXEye, positiveXNostril, negativeXNostril, oralCenterline. Additional names must have an explicitly annotated mesh surface path via --surfaceFeatures. Use true visible hole boundaries for eyes/nostrils. UFF oralCenterline spans an internal rim, may be fully occluded, and will then provide zero constraints even with a manual photo seam; use a reviewed visibleLipSeam surface path instead. Do not use outer lip or whole nose regions as replacements. Each contour: name, points [[x,y],...], closed boolean, confidence (0,1]. View hashes must remain unchanged. Empty lists leave automatic targets unchanged. Supplied same-name targets replace automatic ones.",
        "views": [{"viewIndex": index, "photoSha256": item["sha256"],
                   "imageSize": item["orientedSize"], "contours": []} for index, item in enumerate(inputs)]}
    writeJson(output / "photoAnnotations.template.json", annotationTemplate)
    if targetAnnotations is not None:
        annotationInput = json.loads(Path(targetAnnotations).read_text())
        if (annotationInput.get("schemaVersion") != 1
                or annotationInput.get("coordinateFrame") != "originalExifOrientedPixels"):
            raise ValueError("Manual contours require the original-photo coordinate contract")
        seenViews = set()
        for record in annotationInput.get("views", []):
            index = record["viewIndex"]
            if (not isinstance(index, int) or not 0 <= index < len(inputs) or index in seenViews
                    or record.get("photoSha256") != inputs[index]["sha256"]
                    or record.get("imageSize") != inputs[index]["orientedSize"]):
                raise ValueError("Manual contour view, image dimensions or photo hash mismatch")
            seenViews.add(index)
            targets = rasterizeBoundaryAnnotations(semantics[index]["labelMap"].shape, record.get("contours", []))
            if set(targets)-set(bindings):
                raise ValueError("Manual annotation references unknown anatomical binding")
            targetContours[index].update(targets)
            targetReports[index]["manualTargetNames"] = sorted(targets)
    visibility = visibleVertices(vertices, faces, cameras, [result["labelMap"].shape for result in semantics],
                                 minimumCosine=.05)
    visibilityAudit = boundaryVisibilityDiagnostics(vertices, faces, cameras,
                        [result["labelMap"].shape for result in semantics], loops)
    writeJson(output / "boundaryVisibility.json", visibilityAudit)
    bound = buildAnatomicalConstraints(vertices, cameras, bindings, targetContours, visibility,
                                      maxTargetDistance=24.)
    writeJson(output / "constraintBinding.json", bound["diagnostics"])
    packed = {"vertexVisibility": visibility, "mouthVertexPairs": mouthBinding["vertexIndices"]}
    constraintMeta = []
    for index, constraint in enumerate(bound["constraints"]):
        record = {}
        for key, value in constraint.items():
            if isinstance(value, np.ndarray):
                packed[f"constraint{index:02d}_{key}"] = value
            else:
                record[key] = value
        constraintMeta.append(record)
    np.savez_compressed(output / "constraints.npz", **packed)
    writeJson(output / "constraints.json", constraintMeta)
    refined = None
    if fit:
        import torch
        torch.set_num_threads(1)
        extent = float(np.ptp(vertices, axis=0).max())
        refined = refineLocalGeometry(vertices, faces, cameras, bound["constraints"], iterations=iterations,
             maxDisplacement=.0125*extent, deformationMode="smooth", smoothControlCount=96,
             smoothRadius=.035*extent, supportRings=5, referenceWeight=4., referenceAnchorWeight=0.,
             minAreaRatio=.35, minNormalCosine=.3, maxRelativeEdgeChange=.3,
             maxViewDegradation=.5)
        np.savez_compressed(output / "geometryRefined.npz", **{**geometry, "vertices": refined["vertices"]})
        writeJson(output / "geometryRefinement.json", {"diagnostics": refined["diagnostics"], "history": refined["history"]})
    palette = {"outerFace": "#ff5050", "oralAperture": "#00eaff", "positiveXEye": "#ffdf00",
               "negativeXEye": "#60ff30", "positiveXNostril": "#ff40ef", "negativeXNostril": "#ffaa20"}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    panels, featurePanels = [], []
    for viewIndex, item in enumerate(inputs):
        original = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        for name, loop in loops.items():
            coords = [tuple(value) for value in projected[viewIndex][loop]]
            draw.line(coords + [coords[0]], fill=palette[name], width=2)
        draw.rectangle((0, 0, annotated.width, 32), fill="#202027")
        draw.text((10, 7), f"View {viewIndex}: actual topology boundaries", font=font, fill="white")
        annotated.save(output / f"view{viewIndex:02d}Topology.png")
        annotated.thumbnail((360, 640))
        panels.append(annotated)
        featureIndices = np.concatenate([loop for name, loop in loops.items() if name != "outerFace"])
        coords = projected[viewIndex][featureIndices]
        minimum, maximum = np.floor(coords.min(0)-35).astype(int), np.ceil(coords.max(0)+35).astype(int)
        minimum = np.maximum(minimum, 0)
        maximum = np.minimum(maximum, original.size)
        box = (*minimum, *maximum)
        tiles = []
        for stage, heading in [("photo", "Photo"), ("before", "Initial hole rims + targets"),
                               ("after", "Fitted hole rims + targets")]:
            if stage == "after" and refined is None:
                continue
            current = original.copy()
            currentDraw = ImageDraw.Draw(current)
            if stage != "photo":
                pixels = (projected[viewIndex] if stage == "before" else projectPoints(refined["vertices"],
                          cameras[viewIndex]["intrinsics"], cameras[viewIndex]["rotation"], cameras[viewIndex]["translation"])[0])
                array = np.asarray(current).copy()
                for name, target in targetContours[viewIndex].items():
                    array[binary_dilation(target["contourMask"])] = [255, 70, 70]
                current = Image.fromarray(array)
                currentDraw = ImageDraw.Draw(current)
                for name, loop in loops.items():
                    if name == "outerFace":
                        continue
                    coords = [tuple(value) for value in pixels[loop]]
                    currentDraw.line(coords+[coords[0]], fill=palette[name], width=2)
                centerlinePoints = (pixels[mouthBinding["vertexIndices"]]*.5).sum(1)
                for x, y in centerlinePoints:
                    currentDraw.ellipse((x-1, y-1, x+1, y+1), fill="white")
            tile = current.crop(box)
            tile = ImageOps.contain(tile, (460, 480))
            tilePanel = Image.new("RGB", (460, 520), "#202027")
            tilePanel.paste(tile, ((460-tile.width)//2, 40))
            ImageDraw.Draw(tilePanel).text((10, 10), f"View {viewIndex}: {heading}", font=font, fill="white")
            tiles.append(tilePanel)
        row = Image.new("RGB", (460*len(tiles), 520), "#202027")
        for index, tile in enumerate(tiles):
            row.paste(tile, (index*460, 0))
        row.save(output / f"view{viewIndex:02d}Features.png")
        featurePanels.append(row)
    overview = Image.new("RGB", (360*len(panels), 640), "#202027")
    for index, panel in enumerate(panels):
        overview.paste(panel, (index*360, 0))
    overview.save(output / "topologyOverview.png")
    features = Image.new("RGB", (featurePanels[0].width, 520*len(featurePanels)), "#202027")
    for index, panel in enumerate(featurePanels):
        features.paste(panel, (0, index*520))
    features.save(output / "featureOverview.png")
    report = {"status": "complete", "source": str(source), "semanticCache": str(semanticCache),
              "topology": extracted["diagnostics"], "targetReports": targetReports,
              "mouthBinding": mouthBinding["diagnostics"], "geometryFitted": fit,
              "boundaryVisibility": visibilityAudit,
              "includeLipInterface": includeLipInterface,
              "manualTargetAnnotations": str(targetAnnotations) if targetAnnotations is not None else None,
              "surfaceFeatures": str(surfaceFeatures) if surfaceFeatures is not None else None,
              "geometryDiagnostics": refined["diagnostics"] if refined else None,
              "seconds": perf_counter()-start,
              "sourceHashes": {name: fileHash(source/name) for name in ("geometry.npz", "cameras.json", "inputs.json")},
              "implementationHashes": {name: fileHash(projectRoot/name) for name in (
                  "src/faceAppearance/anatomicalBoundaries.py", "tools/diagnoseAnatomicalBoundaries.py")},
              "semanticManifestSha256": fileHash(semanticCache / "manifest.json"),
              "parserName": semanticManifest["parserName"],
              "limits": ["Anatomical loop names visually inspected, not author-provided template landmarks",
                         "Lip target is parser upper/lower contact, not measured subpixel lip seam",
                         "Mouth bindings span the hole and fit its centerline; they do not close it or add inner mouth",
                         "Nostrils and outer face not fitted because independent target not available",
                         "Initial visibility frozen; downstream rendering must recompute occlusion",
                         "Contour residual is in-sample, not ground truth; no texture or photo score computed here"]}
    writeJson(output / "report.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "topology": report["topology"], "geometryDiagnostics": report["geometryDiagnostics"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/semanticFeatherRefinementExperiment")
    parser.add_argument("--semanticCache", type=Path, default=projectRoot / "runs/faceSemanticCache")
    parser.add_argument("--output", type=Path, default=projectRoot / "runs/anatomicalBoundaryExperiment")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--includeLipInterface", action="store_true")
    parser.add_argument("--targetAnnotations", type=Path)
    parser.add_argument("--surfaceFeatures", type=Path)
    parser.add_argument("--iterations", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error("Expected 1..1000 iterations")
    runDiagnostic(args.input, args.semanticCache, args.output, fit=args.fit, iterations=args.iterations,
                  includeLipInterface=args.includeLipInterface, targetAnnotations=args.targetAnnotations,
                  surfaceFeatures=args.surfaceFeatures)
