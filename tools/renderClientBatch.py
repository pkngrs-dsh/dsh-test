"""Render successful multi-case A/B/C exports with explicit shared controls.

Manifest schema: {"schemaVersion": 1, "cases": [{"caseId": "case01",
"label": "Readable label", "status": "succeeded", "outputDir": "case01",
"referenceViewIndex": 0, "sideViewIndex": 2}]}. outputDir is absolute or
relative to the manifest. Each case contains appearance/ and assembly/A|B|C/.
Failed/skipped cases remain in the report; no failed result is presented as a
rendered comparison. --prepareOnly performs CPU validation without torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot/"src") not in sys.path:
    sys.path.insert(0, str(projectRoot/"src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    # runpy.run_path does not add the script directory as direct CLI execution
    # does; the batch completion gate intentionally uses that entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from faceAppearance.artifacts import writeImage, writeJson
from compareClientAssembly import clayAsset, imageFromArray, loadPhotoReference, prepareAssets
from renderClientAssembly import ClientPreviewRenderer, fileHash


branchNames = ("A", "B", "C")
lightDirectionUff = [-.5, .6, 1.]


def previewFont(size: int) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def eyeStatusSuffix(case: dict) -> str:
    refinement = case.get("eyeRefinement")
    return " / 眼区未精修" if isinstance(refinement, dict) and refinement.get("status") == "skipped" else ""


def summaryGrid(panels: list, labels: list[str], *, subtitle: str, panelSize: int = 384) -> Image.Image:
    """CJK labels and a separate status line prevent silent degraded-case labels."""
    if not panels or len(panels) != len(labels):
        raise ValueError("Expected one label per comparison panel")
    columns, header, labelHeight = 4, 58, 54
    rows = math.ceil(len(panels)/columns)
    canvas = Image.new("RGB", (columns*panelSize, header+rows*(panelSize+labelHeight)), (22, 27, 33))
    draw, font = ImageDraw.Draw(canvas), previewFont(15)
    draw.text((12, 15), subtitle, font=previewFont(18), fill=(225, 235, 244))
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = index % columns*panelSize, header+index//columns*(panelSize+labelHeight)
        draw.multiline_text((x+9, y+6), label, font=font, spacing=2, fill=(235, 239, 244))
        resized = ImageOps.contain(panel.convert("RGB"), (panelSize, panelSize), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x+(panelSize-resized.width)//2, y+labelHeight+(panelSize-resized.height)//2))
    return canvas


def readManifest(path: Path, *, payload: bytes | None = None) -> list[dict]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_bytes() if payload is None else payload)
    if manifest.get("schemaVersion", 1) != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("Batch manifest requires schemaVersion=1 and a cases array")
    cases, identifiers = [], set()
    for entry in manifest["cases"]:
        if not isinstance(entry, dict):
            raise ValueError("Every batch case must be an object")
        caseId = entry.get("caseId")
        if (not isinstance(caseId, str) or not caseId or caseId in {".", ".."}
                or "/" in caseId or "\\" in caseId or "\0" in caseId or caseId in identifiers):
            raise ValueError("caseId must be a unique nonempty directory name")
        identifiers.add(caseId)
        label = entry.get("label", caseId)
        if not isinstance(label, str) or not label:
            raise ValueError("Case label must be a nonempty string")
        directory = Path(entry.get("outputDir", caseId))
        if not directory.is_absolute():
            directory = path.parent/directory
        cases.append({**entry, "caseId": caseId, "label": label, "outputDir": directory.resolve()})
    return cases


def chooseViews(case: dict, sourceDir: Path) -> tuple[int, int, dict]:
    cameras = json.loads((sourceDir/"cameras.json").read_text())
    referenceIndex = case.get("referenceViewIndex", 0)
    if isinstance(referenceIndex, bool) or not isinstance(referenceIndex, int) or not 0 <= referenceIndex < len(cameras):
        raise ValueError("referenceViewIndex must identify an existing saved view")
    if not cameras[referenceIndex].get("valid"):
        raise ValueError("Saved reference camera was rejected")
    valid = []
    referenceRotation = np.asarray(cameras[referenceIndex]["rotation"], float)
    if referenceRotation.shape != (3, 3) or not np.isfinite(referenceRotation).all():
        raise ValueError("Invalid saved reference rotation")
    referenceAxis = referenceRotation[2]/np.linalg.norm(referenceRotation[2])
    for index, camera in enumerate(cameras):
        rotation = np.asarray(camera.get("rotation", []), float)
        if index == referenceIndex or not camera.get("valid"):
            continue
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or np.linalg.norm(rotation[2]) < 1e-10:
            continue
        axis = rotation[2]/np.linalg.norm(rotation[2])
        separation = float(np.degrees(np.arccos(np.clip(referenceAxis@axis, -1, 1))))
        valid.append((index, separation))
    if not valid:
        raise ValueError("At least one additional valid saved camera is needed for a side comparison")
    sideIndex = case.get("sideViewIndex")
    if sideIndex is None:
        sideIndex, separation = max(valid, key=lambda item: item[1])
        policy = "Valid non-reference saved camera with maximum optical-axis angular separation"
    else:
        if isinstance(sideIndex, bool) or not isinstance(sideIndex, int) or sideIndex not in dict(valid):
            raise ValueError("sideViewIndex must identify a different valid saved camera")
        separation = dict(valid)[sideIndex]
        policy = ("Explicit --sideViewIndex command-line override" if case.get("sideViewOverrideSource")
                  else "Explicit sideViewIndex from batch manifest")
    return referenceIndex, sideIndex, {"referenceViewIndex": referenceIndex, "sideViewIndex": sideIndex,
        "sideSelectionPolicy": policy, "opticalAxisSeparationDegrees": separation,
        "commandLineSideOverride": case.get("sideViewOverrideSource"),
        "manifestSideViewIndex": case.get("manifestSideViewIndex", case.get("sideViewIndex")),
        "weakSideAngle": separation < 15.,
        "meaning": "Saved camera selection only; no pose fitting. A weak angle is disclosed rather than relabelled as a strong profile."}


def resizeReference(reference: dict, maxPhotoDimension: int) -> dict:
    """Resize only the preview, retaining exact original-camera provenance.

    For integer-centered pixels: pPreview = S*pOriginal + (S-1)/2. This is
    raster-resolution conversion, not a new camera pose or an independent fit.
    """
    photo = reference["photo"]
    width, height = photo.size
    scale = min(1., maxPhotoDimension/max(width, height)) if maxPhotoDimension else 1.
    previewWidth, previewHeight = max(1, round(width*scale)), max(1, round(height*scale))
    sx, sy = previewWidth/width, previewHeight/height
    affine = np.array([[sx, 0, (sx-1)*.5], [0, sy, (sy-1)*.5], [0, 0, 1.]])
    camera = {**reference["camera"], "intrinsics": affine@reference["camera"]["intrinsics"]}
    crops = dict(reference["crops"])
    # Display crops use pixel boundaries, so edge coordinates scale without the
    # half-pixel shift used for landmark/projection center coordinates above.
    for name in ("fullFace", "eyes", "nose", "mouth"):
        box = np.asarray(reference["crops"][name], float)*[sx, sy, sx, sy]
        crops[name] = [*np.floor(box[:2]).astype(int).tolist(), *np.ceil(box[2:]).astype(int).tolist()]
    if previewWidth != width or previewHeight != height:
        photo = photo.resize((previewWidth, previewHeight), Image.Resampling.LANCZOS)
    return {**reference, "photo": photo, "camera": camera, "crops": crops,
            "imageShape": (previewHeight, previewWidth),
            "originalCamera": reference["camera"], "originalCrops": reference["crops"],
            "previewConversion": {"originalImageSizeHw": [height, width],
                                  "previewImageSizeHw": [previewHeight, previewWidth],
                                  "originalToPreviewPixelCenters": affine,
                                  "maximumPhotoDimension": maxPhotoDimension,
                                  "cameraPoseChanged": False, "cameraRefitted": False}}


def prepareCase(case: dict, *, maxTextureSize: int, maxPhotoDimension: int) -> dict:
    directory = case["outputDir"]
    sourceDir = directory/"appearance"
    pairB, reportB = prepareAssets(directory/"assembly/A", directory/"assembly/B", maxTextureSize)
    pairC, reportC = prepareAssets(directory/"assembly/A", directory/"assembly/C", maxTextureSize)
    if pairB["baseline"]["diagnostics"]["sourceHashes"] != pairC["baseline"]["diagnostics"]["sourceHashes"]:
        raise ValueError("Baseline A changed while preparing this case")
    assets = {"A": pairB["baseline"], "B": pairB["candidate"], "C": pairC["candidate"]}
    referenceIndex, sideIndex, selection = chooseViews(case, sourceDir)
    references = {name: resizeReference(loadPhotoReference(sourceDir, index), maxPhotoDimension)
                  for name, index in (("front", referenceIndex), ("side", sideIndex))}
    # Only compare after the shared A placement. Templates/indices/UVs have
    # already passed prepareAssets' strict legacy-binding identity checks.
    geometryEqualBc = bool(np.array_equal(assets["B"]["vertices"], assets["C"]["vertices"]))
    return {"assets": assets, "references": references,
            "controls": {"appliedTemplateToUff": reportB["appliedTemplateToUff"],
                         "placementSource": reportB["placementSource"],
                         "placementPolicy": "All three original meter-space meshes use A placement; no per-branch alignment",
                         "originalTemplateToUff": {name: asset["diagnostics"]["templateToUff"] for name, asset in assets.items()},
                         "sameLegacyBindingArrays": reportB["identicalBindingArrays"],
                         "bDifferenceFromACm": reportB["vertexDifferenceCm"],
                         "cDifferenceFromACm": reportC["vertexDifferenceCm"],
                         "bAndCPlacedVerticesBitExact": geometryEqualBc,
                         "viewSelection": selection},
            "assetDiagnostics": {name: asset["diagnostics"] for name, asset in assets.items()}}


def saveFeatureFigure(path: Path, photo: Image.Image, branches: dict, crops: dict, *, title: str,
                      mode: str, panelWidth: int = 384) -> None:
    """Four columns with compact variable-height rows instead of empty squares."""
    images = [photo]+[imageFromArray(branches[name]) for name in branchNames]
    labels = ["Reference photo", "A: default", "B: barycentric geometry", "C: corrected UV"]
    parts = ("fullFace", "eyes", "nose", "mouth")
    heights = [max(32, min(panelWidth*2, math.ceil(panelWidth*(crops[p][3]-crops[p][1])/
                                                    max(1, crops[p][2]-crops[p][0])))) for p in parts]
    header, labelHeight, footer = 70, 32, 32
    canvas = Image.new("RGB", (panelWidth*4, header+sum(heights)+len(parts)*labelHeight+footer), (22, 27, 33))
    draw = ImageDraw.Draw(canvas)
    font, titleFont = previewFont(15), previewFont(19)
    draw.text((12, 10), title, font=titleFont, fill=(236, 241, 245))
    draw.text((12, 40), f"{mode} | one A placement, saved camera, fixed light; source-defined crops", font=font, fill=(201, 215, 227))
    y = header
    for part, height in zip(parts, heights):
        for index, (image, label) in enumerate(zip(images, labels)):
            draw.text((index*panelWidth+8, y+7), f"{part}: {label}", font=font, fill=(236, 241, 245))
            resized = ImageOps.contain(image.crop(crops[part]), (panelWidth, height), Image.Resampling.LANCZOS)
            canvas.paste(resized, (index*panelWidth+(panelWidth-resized.width)//2,
                                  y+labelHeight+(height-resized.height)//2))
        y += height+labelHeight
    draw.text((12, y+8), "Preview only; no photometric score or anatomical ground-truth claim.", font=font, fill=(201, 215, 227))
    canvas.save(path)


def renderCase(data: dict, case: dict, output: Path, *, device: str, facesPerPixel: int) -> dict:
    renders, viewResults, grayChecks = {}, {}, {}
    assets = data["assets"]
    for viewName, reference in data["references"].items():
        viewDir = output/viewName
        viewDir.mkdir()
        reference["photo"].save(viewDir/"referencePhoto.png")
        renders[viewName], viewResults[viewName] = {}, {}
        for mode in ("textured", "clay"):
            renders[viewName][mode], branchImages = {}, {}
            for branch in branchNames:
                preview = clayAsset(assets[branch], assets["A"]) if mode == "clay" else assets[branch]
                renderer = ClientPreviewRenderer(preview, device=device, facesPerPixel=facesPerPixel)
                camera, projection = renderer.photoCamera(reference["camera"], reference["imageShape"])
                image, visible, stats = renderer.render(camera, reference["imageShape"], lightDirectionUff, perspective=True)
                if not visible.any():
                    raise ValueError(f"No visible head: {case['caseId']}/{viewName}/{mode}/{branch}")
                branchImages[branch] = image
                writeImage(viewDir/f"{branch}_{mode}.png", image)
                writeImage(viewDir/f"{branch}_{mode}Mask.png", visible)
                for part in ("eyes", "nose", "mouth"):
                    imageFromArray(image).crop(reference["crops"][part]).save(viewDir/f"{branch}_{mode}_{part}.png")
                renders[viewName][mode][branch] = {"projectionCheck": projection, **stats}
                del renderer, preview
            saveFeatureFigure(viewDir/f"{mode}Abc.png", reference["photo"], branchImages,
                              reference["crops"], title=f"{case['label']} / {viewName}{eyeStatusSuffix(case)}", mode=mode)
            viewResults[viewName][mode] = {"reference": reference["photo"].crop(reference["crops"]["fullFace"]),
                **{branch: imageFromArray(image).crop(reference["crops"]["fullFace"]) for branch, image in branchImages.items()}}
            if mode == "clay":
                left = np.rint(np.clip(branchImages["B"], 0, 1)*255).astype(np.int16)
                right = np.rint(np.clip(branchImages["C"], 0, 1)*255).astype(np.int16)
                difference = np.abs(left-right)
                grayChecks[viewName] = {"pixelIdentical": bool(np.array_equal(left, right)),
                    "changedPixels": int(np.any(difference > 0, axis=-1).sum()),
                    "maximum8bitChannelDifference": int(difference.max()),
                    "meaning": "Renderer control check; B/C equal geometry is reported separately, not assumed"}
        for part in ("eyes", "nose", "mouth"):
            reference["photo"].crop(reference["crops"][part]).save(viewDir/f"reference_{part}.png")
    panels, labels = [], []
    for viewName in ("front", "side"):
        for branch in ("reference", *branchNames):
            panels.append(viewResults[viewName]["textured"][branch])
            labels.append(f"{viewName}: {branch}{eyeStatusSuffix(case)}")
    summaryGrid(panels, labels,
                subtitle=f"{case['label']}{eyeStatusSuffix(case)} | original photo / A / B / C | fixed controls").save(output/"caseOverview.png")
    return {"renders": renders, "bAndCGrayControlChecks": grayChecks}


def sourceRecords(data: dict) -> dict:
    return {name: {"viewIndex": reference["viewIndex"], "originalCamera": reference["originalCamera"],
                  "previewCamera": reference["camera"], "previewConversion": reference["previewConversion"],
                  "originalDisplayCrops": reference["originalCrops"], "previewDisplayCrops": reference["crops"],
                  "sourceHashes": reference["sourceHashes"], "sameForAllBranches": True}
            for name, reference in data["references"].items()}


def saveBatchFigures(output: Path, rendered: list[dict]) -> list[str]:
    paths = []
    for view in ("front", "side"):
        for mode in ("textured", "clay"):
            panels, labels = [], []
            for case in rendered:
                caseDir = output/"cases"/case["caseId"]
                report = json.loads((caseDir/"caseReport.json").read_text())
                crop = report["views"][view]["previewDisplayCrops"]["fullFace"]
                for branch in ("reference", *branchNames):
                    imagePath = caseDir/view/("referencePhoto.png" if branch == "reference" else f"{branch}_{mode}.png")
                    with Image.open(imagePath) as image:
                        panels.append(image.convert("RGB").crop(crop))
                    labels.append(f"{case['label'][:24]}\n{branch}{eyeStatusSuffix(case)}")
            if panels:
                filename = f"batch{view.title()}{mode.title()}.png"
                summaryGrid(panels, labels,
                            subtitle=f"{view} / {mode}: reference, A, B, C | only successfully rendered cases").save(output/filename)
                paths.append(filename)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maxTextureSize", type=int, default=1024)
    parser.add_argument("--maxPhotoDimension", type=int, default=1280,
                        help="Shared preview raster limit per view; 0 keeps the original resolution")
    parser.add_argument("--facesPerPixel", type=int, default=16)
    parser.add_argument("--sideViewIndex", type=int, default=None,
                        help="Override every case's saved side-camera index without modifying the manifest")
    parser.add_argument("--prepareOnly", action="store_true", help="Validate successful cases without torch or GPU")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("Choose a new or empty batch preview directory")
    if (not 32 <= args.maxTextureSize <= 4096 or not 2 <= args.facesPerPixel <= 128
            or args.maxPhotoDimension != 0 and not 64 <= args.maxPhotoDimension <= 4096):
        raise ValueError("Invalid texture/raster/depth-layer preview settings")
    if args.sideViewIndex is not None and args.sideViewIndex < 0:
        raise ValueError("sideViewIndex must be nonnegative")
    manifestBytes = args.manifest.read_bytes()
    manifestSha256 = hashlib.sha256(manifestBytes).hexdigest()
    cases = readManifest(args.manifest, payload=manifestBytes)
    if args.sideViewIndex is not None:
        for case in cases:
            case["manifestSideViewIndex"] = case.get("sideViewIndex")
            case["sideViewIndex"] = args.sideViewIndex
            case["sideViewOverrideSource"] = "commandLine --sideViewIndex"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"inputManifest.json").write_bytes(manifestBytes)
    start = perf_counter()
    scriptPath = Path(__file__).resolve()
    sourceFiles = [scriptPath, scriptPath.with_name("compareClientAssembly.py"),
                   scriptPath.with_name("renderClientAssembly.py"), scriptPath.with_name("renderCropComparison.py"),
                   projectRoot/"src/faceAppearance/pytorch3dRenderer.py", projectRoot/"src/faceAppearance/photoCameras.py"]
    # Snapshot before rendering so a later summary failure cannot lose the
    # implementation provenance of already completed, expensive image work.
    for sourcePath in sourceFiles:
        destination = args.output/"implementation"/sourcePath.relative_to(projectRoot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sourcePath.read_bytes())
    records, rendered = [], []
    runtimeInitialized = False
    torch = None
    for case in cases:
        record = {"caseId": case["caseId"], "label": case["label"], "inputStatus": case.get("status"),
                  "inputOutputDir": str(case["outputDir"]), "eyeRefinement": case.get("eyeRefinement")}
        if case.get("status") != "succeeded":
            record.update(status="skipped", reason="Input case has no explicit succeeded status", inputError=case.get("error"))
            records.append(record)
            continue
        caseDir = args.output/"cases"/case["caseId"]
        caseDir.mkdir(parents=True)
        phase = "preparation"
        caseStart = perf_counter()
        try:
            data = prepareCase(case, maxTextureSize=args.maxTextureSize, maxPhotoDimension=args.maxPhotoDimension)
            report = {"caseId": case["caseId"], "label": case["label"], "status": "prepared",
                      "eyeRefinement": case.get("eyeRefinement"), "degradationLabel": eyeStatusSuffix(case).strip(" /"),
                      "inputOutputDir": str(case["outputDir"]), "controls": data["controls"],
                      "assetDiagnostics": data["assetDiagnostics"], "views": sourceRecords(data),
                      "photometricEvaluation": False, "anatomicalLandmarkEvaluation": False,
                      "originalAssetsModified": False}
            if not args.prepareOnly:
                phase = "rendering"
                if not runtimeInitialized:
                    import torch as torchRuntime
                    import pytorch3d
                    torch = torchRuntime
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    runtimeInitialized = True
                report.update(renderCase(data, case, caseDir, device=args.device, facesPerPixel=args.facesPerPixel))
                report.update(status="rendered", pytorch3dVersion=pytorch3d.__version__)
            report["seconds"] = perf_counter()-caseStart
            report["outputHashes"] = {path.relative_to(caseDir).as_posix(): fileHash(path)
                                       for path in sorted(caseDir.rglob("*.png"))}
            writeJson(caseDir/"caseReport.json", report)
            record.update(status=report["status"], report=str((caseDir/"caseReport.json").relative_to(args.output)),
                          reportSha256=fileHash(caseDir/"caseReport.json"), seconds=report["seconds"],
                          viewSelection=data["controls"]["viewSelection"])
            if report["status"] == "rendered":
                rendered.append(record)
            del data
        except Exception as error:
            record.update(status="failed", phase=phase, error={"type": type(error).__name__, "message": str(error)},
                          seconds=perf_counter()-caseStart)
            writeJson(caseDir/"caseFailure.json", record)
            # A failed case's partial images are never included in the summary.
            if runtimeInitialized and args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        records.append(record)
        print(json.dumps({"caseId": record["caseId"], "status": record["status"]}), flush=True)
    summaryFiles = saveBatchFigures(args.output, rendered)
    successfulCount = sum(record["status"] in {"prepared", "rendered"} for record in records)
    failedCount = sum(record["status"] == "failed" for record in records)
    skippedCount = sum(record["status"] == "skipped" for record in records)
    status = ("noSuccessfulCases" if not successfulCount else "partial" if failedCount else
              "prepared" if args.prepareOnly else "renderedWithSkippedCases" if skippedCount else "rendered")
    batchReport = {"schemaVersion": 1, "status": status, "manifest": str(args.manifest.resolve()),
        "manifestSha256": manifestSha256, "manifestSnapshot": "inputManifest.json",
        "manifestChangedDuringRun": fileHash(args.manifest) != manifestSha256, "cases": records,
        "counts": {"requested": len(cases), "successful": successfulCount, "failed": failedCount, "skipped": skippedCount},
        "prepareOnly": args.prepareOnly, "device": None if args.prepareOnly else args.device,
        "seconds": perf_counter()-start, "summaryFiles": summaryFiles,
        "summaryHashes": {name: fileHash(args.output/name) for name in summaryFiles},
        "implementationHashes": {str(path.relative_to(projectRoot)): fileHash(path) for path in sourceFiles},
        "lightingPreset": {"directionUff": lightDirectionUff, "sameForAllCasesViewsBranches": True,
                           "f0Dielectric": .04, "keyIntensity": 2.5, "ambient": .22, "fittedToPhoto": False},
        "clayMaterial": {"baseColorLinear": [.4, .4, .4], "roughness": .6, "metallic": 0.,
                         "alpha": "Every branch uses its case A alpha atlas/cutoffs; geometry remains branch-specific"},
        "previewSettings": {"maxPhotoDimension": args.maxPhotoDimension, "maxTextureSize": args.maxTextureSize,
                            "facesPerPixel": args.facesPerPixel, "cullBackfaces": True, "supersampling": False,
                            "sideViewIndexOverride": args.sideViewIndex},
        "photometricEvaluation": False, "anatomicalLandmarkEvaluation": False, "originalAssetsModified": False,
        "limits": ["Failed/skipped cases remain listed and are excluded from image summaries.",
                   "Input status=succeeded does not bypass required asset/camera/photo hash validation.",
                   "Cameras and placements are shared within each case; no branch is independently realigned.",
                   "Raster-size conversion rescales saved intrinsics with the declared half-pixel affine, without changing camera pose.",
                   "Photo detector points define crops only; they are not anatomical ground truth.",
                   "Fixed illumination is not the source photo lighting; no photometric scores are reported.",
                   "Shader has no shadows, ambient occlusion, SSS, measured normal maps or rig validation."]}
    if runtimeInitialized and args.device.startswith("cuda"):
        batchReport["peakAllocatedGiB"] = torch.cuda.max_memory_allocated()/1024**3
    writeJson(args.output/"batchReport.json", batchReport)
    print(json.dumps({"status": status, "counts": batchReport["counts"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
