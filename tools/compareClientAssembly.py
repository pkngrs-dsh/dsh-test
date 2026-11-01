"""Compare two client exports without independently aligning their geometry.

Both original meter-space meshes use the baseline's templateToUff transform,
one saved fitted photo camera and one shared set of full-head cameras. Lighting
is a fixed inspection preset: no photometric error or anatomical accuracy is
inferred from these images. --prepareOnly validates the comparison on the CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot / "src") not in sys.path:
    sys.path.insert(0, str(projectRoot / "src"))

from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.photoCameras import originalImageIntrinsics
from renderClientAssembly import ClientPreviewRenderer, fileHash, loadClientAsset
from renderCropComparison import makeGrid


def applyPlacement(verticesMeters: np.ndarray, placement: dict) -> np.ndarray:
    """Read original exported positions; never undo/reapply rounded previews."""
    vertices = np.asarray(verticesMeters, np.float64)
    rotation = np.asarray(placement["rotation"], np.float64)
    translation = np.asarray(placement["translation"], np.float64)
    scale = float(placement["scale"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.isfinite(scale) or scale <= 0
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("Expected finite meter vertices and a proper positive-scale placement")
    return (vertices*100.)@rotation*scale+translation


def prepareAssets(baselineDir: Path, candidateDir: Path, maxTextureSize: int) -> tuple:
    """Use one placement and require the same legacy template/binding arrays."""
    assets, meshes = {}, {}
    for branch, directory in (("baseline", baselineDir), ("candidate", candidateDir)):
        assets[branch] = loadClientAsset(directory, maxTextureSize)
        with np.load(directory/"clientMesh.npz", allow_pickle=False) as archive:
            meshes[branch] = {key: archive[key] for key in
                              ("vertices", "templateVertices", "faces", "uvs", "uvFaces", "materialIds")}
    baseline, candidate = meshes["baseline"], meshes["candidate"]
    for key in ("templateVertices", "faces", "uvs", "uvFaces", "materialIds"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"Comparison requires identical legacy template/binding arrays: {key} differs")
    if baseline["vertices"].shape != candidate["vertices"].shape:
        raise ValueError("Branch vertex counts differ")
    originalPlacements = {name: asset["diagnostics"]["templateToUff"] for name, asset in assets.items()}
    placement = originalPlacements["baseline"]
    for name, asset in assets.items():
        asset["vertices"] = applyPlacement(meshes[name]["vertices"], placement)
    allVertices = np.concatenate([asset["vertices"] for asset in assets.values()])
    center = (allVertices.min(0)+allVertices.max(0))*.5
    radius = float(np.linalg.norm(allVertices-center, axis=-1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid shared geometry framing radius")
    displacementCm = np.linalg.norm((candidate["vertices"]-baseline["vertices"])*100., axis=1)
    return assets, {"originalTemplateToUff": originalPlacements, "appliedTemplateToUff": placement,
                    "placementSource": str((baselineDir/"diagnostics.json").resolve()),
                    "placementPolicy": "Both exported meter arrays use the baseline similarity; no branch alignment or camera fitting",
                    "exportVerticesToUff": "(verticesMeters * 100) @ baselineRotation * baselineScale + baselineTranslation",
                    "identicalBindingArrays": ["templateVertices", "faces", "uvs", "uvFaces", "materialIds"],
                    "vertexDifferenceCm": {"median": float(np.median(displacementCm)),
                                           "p95": float(np.percentile(displacementCm, 95)),
                                           "max": float(displacementCm.max()),
                                           "meaning": "Index-wise shape difference, not reconstruction accuracy"},
                    "sharedFraming": {"center": center, "radius": radius, "orthographicExtent": radius*1.08,
                                      "policy": "One bounding sphere around both meshes after shared placement"}}


def displayCrops(metadata: dict) -> dict:
    """Use saved source-photo detector points, never rendered feature locations."""
    if "faceDetection" in metadata and "landmarks" in metadata["faceDetection"]:
        landmarks = np.asarray(metadata["faceDetection"]["landmarks"], np.float64)
        landmarkSource = "Saved original-photo five-point faceDetection landmarks"
    elif "faceLandmarksWorking" in metadata:
        working = np.asarray(metadata["faceLandmarksWorking"], np.float64)
        transform = np.asarray(metadata["inputToWorking"], np.float64)
        mapped = np.column_stack([working, np.ones(len(working))])@np.linalg.inv(transform).T
        landmarks = mapped[:, :2]/mapped[:, 2:3]
        landmarkSource = "Saved working five-point face landmarks converted by inverse inputToWorking"
    else:
        raise ValueError("Source metadata needs five-point landmarks for fixed closeup crops")
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError("Invalid source five-point landmarks")
    eyeDistance = float(np.linalg.norm(landmarks[1]-landmarks[0]))
    if eyeDistance <= 1:
        raise ValueError("Degenerate source eye landmarks")

    def box(center, width, height):
        minimum = np.floor(center-np.array([width, height])*.5).astype(int)
        maximum = minimum+np.ceil([width, height]).astype(int)
        return [*minimum.tolist(), *maximum.tolist()]

    eyeCenter = landmarks[:2].mean(0)-[0, .12*eyeDistance]
    mouthCenter = landmarks[3:5].mean(0)
    full = metadata.get("cropBoxXyxy", [0, 0, *metadata["orientedSize"]])
    return {"fullFace": list(full), "eyes": box(eyeCenter, 1.85*eyeDistance, .85*eyeDistance),
            "nose": box(landmarks[2], .86*eyeDistance, .78*eyeDistance),
            "mouth": box(mouthCenter, 1.05*eyeDistance, .65*eyeDistance),
            "source": landmarkSource, "landmarksOriginalPixels": landmarks.tolist(),
            "meaning": "Display crops only; detector points are not evaluation ground truth"}


def loadPhotoReference(sourceDir: Path, viewIndex: int) -> dict:
    paths = {name: sourceDir/name for name in ("inputs.json", "cameras.json", "viewTransforms.json", "geometry.npz")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    inputs = json.loads(paths["inputs.json"].read_text())
    cameras = json.loads(paths["cameras.json"].read_text())
    transforms = json.loads(paths["viewTransforms.json"].read_text())
    if not 0 <= viewIndex < len(inputs) or len(inputs) != len(cameras) or len(inputs) != len(transforms):
        raise ValueError("Invalid saved view index or camera metadata counts")
    metadata, camera = inputs[viewIndex], cameras[viewIndex]
    if not camera.get("valid") or camera.get("viewIndex", viewIndex) != viewIndex:
        raise ValueError("Reference fitted camera is invalid or mismatched")
    sourcePath = Path(metadata["path"])
    if fileHash(sourcePath) != metadata["sha256"]:
        raise ValueError("Original reference photo differs from its saved hash")
    with Image.open(sourcePath) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    if list(photo.size) != metadata["orientedSize"]:
        raise ValueError("Original EXIF-oriented photo dimensions differ")
    camera = {"intrinsics": originalImageIntrinsics(camera["intrinsics"], metadata["inputToWorking"]),
              "rotation": np.asarray(camera["rotation"], np.float64),
              "translation": np.asarray(camera["translation"], np.float64)}
    return {"photo": photo, "camera": camera, "viewIndex": viewIndex, "metadata": metadata,
            "crops": displayCrops(metadata), "imageShape": (photo.height, photo.width),
            "sourceHashes": {**{name: fileHash(path) for name, path in paths.items()},
                             "originalPhoto": metadata["sha256"]}}


def clayAsset(asset: dict, baseline: dict) -> dict:
    """Uniform surfaces and shared baseline cutouts isolate visible geometry."""
    if asset["atlasLinearRgba"].shape != baseline["atlasLinearRgba"].shape:
        raise ValueError("Clay comparison requires identical atlas layouts")
    atlas = baseline["atlasLinearRgba"].copy()
    atlas[..., :3] = .4
    return {**asset, "atlasLinearRgba": atlas, "alphaCutoff": baseline["alphaCutoff"].copy(),
            "roughness": np.full_like(baseline["roughness"], .6), "metallic": np.zeros_like(baseline["metallic"])}


def imageFromArray(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(values, 0, 1)*255).astype(np.uint8))


def saveComparisons(directory: Path, photo: Image.Image, branches: dict, crops: dict, mode: str) -> None:
    fixedSubtitle = "Shared baseline placement, saved camera and preset light | no photo-error score"
    panels = [photo.crop(crops["fullFace"])]+[imageFromArray(branches[name]["photo"]).crop(crops["fullFace"])
                                             for name in ("baseline", "candidate")]
    makeGrid(panels, ["Reference photo", f"A: baseline / {mode}", f"B: candidate / {mode}"],
             columns=3, panelSize=512, subtitle=fixedSubtitle).save(directory/"photoComparison.png")
    closeups, labels = [], []
    for part in ("eyes", "nose", "mouth"):
        closeups.extend([photo.crop(crops[part])]+[imageFromArray(branches[name]["photo"]).crop(crops[part])
                                                    for name in ("baseline", "candidate")])
        labels.extend([f"{part}: reference", f"{part}: A baseline", f"{part}: B candidate"])
    makeGrid(closeups, labels, columns=3, panelSize=448,
             subtitle="Identical source-defined feature crops | fixed preview lighting").save(directory/"closeups.png")
    headPanels, headLabels = [], []
    for view in ("front", "leftThreeQuarter", "rightThreeQuarter"):
        for name in ("baseline", "candidate"):
            headPanels.append(imageFromArray(branches[name][view]))
            headLabels.append(f"{'A' if name == 'baseline' else 'B'}: {name} / {view}")
    makeGrid(headPanels, headLabels, columns=2, panelSize=512,
             subtitle="Shared full-head cameras and light; no independent alignment").save(directory/"fullHeadComparison.png")


def renderComparison(assets: dict, reference: dict, placement: dict, output: Path, *,
                     device: str, size: int, facesPerPixel: int, includeClay: bool) -> dict:
    import torch
    from pytorch3d.renderer import FoVOrthographicCameras, look_at_view_transform

    camera = reference["camera"]
    photoLight = camera["rotation"].T@np.array([-.5, -.5, -1.])
    framing = placement["sharedFraming"]
    center, radius, extent = framing["center"], framing["radius"], framing["orthographicExtent"]
    headCameras, headCameraRecords = {}, []
    for name, azimuth in (("front", 0.), ("leftThreeQuarter", -40.), ("rightThreeQuarter", 40.)):
        rotation, translation = look_at_view_transform(dist=4*radius, elev=0., azim=azimuth,
                                                       at=(tuple(center),), device=device)
        headCameras[name] = FoVOrthographicCameras(device=device, R=rotation, T=translation,
            min_x=-extent, max_x=extent, min_y=-extent, max_y=extent, znear=.01*radius, zfar=10*radius)
        headCameraRecords.append({"name": name, "azimuthDegrees": azimuth, "elevationDegrees": 0.,
                                  "R": rotation.cpu().numpy(), "T": translation.cpu().numpy()})
    diagnostics = {}
    for mode in (["textured", "clay"] if includeClay else ["textured"]):
        modeDir = output/mode
        modeDir.mkdir()
        diagnostics[mode], renderedBranches = {}, {}
        for branch, asset in assets.items():
            branchDir = modeDir/branch
            branchDir.mkdir()
            previewAsset = clayAsset(asset, assets["baseline"]) if mode == "clay" else asset
            renderer = ClientPreviewRenderer(previewAsset, device=device, facesPerPixel=facesPerPixel)
            photoCamera, projection = renderer.photoCamera(camera, reference["imageShape"])
            rendered, visible, stats = renderer.render(photoCamera, reference["imageShape"], photoLight, perspective=True)
            if not visible.any():
                raise ValueError(f"No visible client head in photo camera: {mode}/{branch}")
            writeImage(branchDir/"photoCameraRender.png", rendered)
            writeImage(branchDir/"photoCameraMask.png", visible)
            overlay = np.asarray(reference["photo"], np.float32)/255.
            overlay[visible] = .5*overlay[visible]+.5*rendered[visible]
            writeImage(branchDir/"photoCameraOverlay50.png", overlay)
            for part in ("eyes", "nose", "mouth"):
                imageFromArray(rendered).crop(reference["crops"][part]).save(branchDir/f"{part}.png")
            renderedBranches[branch] = {"photo": rendered}
            diagnostics[mode][branch] = {"photo": {"projectionCheck": projection, **stats}, "headViews": {}}
            for name, headCamera in headCameras.items():
                image, mask, stats = renderer.render(headCamera, (size, size), [-.5, .6, 1.], perspective=False)
                if not mask.any():
                    raise ValueError(f"No visible client head in full-head camera: {mode}/{branch}/{name}")
                writeImage(branchDir/f"head{name[0].upper()+name[1:]}.png", image)
                renderedBranches[branch][name] = image
                diagnostics[mode][branch]["headViews"][name] = stats
            # One renderer at a time; retained comparison panels are CPU arrays.
            del renderer, previewAsset
        saveComparisons(modeDir, reference["photo"], renderedBranches, reference["crops"], mode)
    return {"renders": diagnostics, "fullHeadCameras": headCameraRecords,
            "lightingPreset": {"shader": "renderClientAssembly.ClientPreviewRenderer linear Lambert + GGX/Smith/Schlick",
                               "f0Dielectric": .04, "keyIntensity": 2.5, "ambient": .22,
                               "photoLightDirectionUff": photoLight, "fullHeadLightDirectionUff": [-.5, .6, 1.],
                               "sameBetweenBranches": True, "fittedToPhoto": False},
            "clayMaterial": ({"baseColorLinear": [.4, .4, .4], "roughness": .6, "metallic": 0.,
                              "alpha": "Both branches use baseline alpha atlas and cutoffs; geometry remains branch-specific"}
                             if includeClay else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=projectRoot/"runs/clientAssemblyReady/assembly")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=projectRoot/"runs/eyeCornerPriorityFinal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewIndex", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--maxTextureSize", type=int, default=1024)
    parser.add_argument("--facesPerPixel", type=int, default=16)
    parser.add_argument("--includeClay", action="store_true")
    parser.add_argument("--prepareOnly", action="store_true", help="Validate and save the comparison plan on CPU; no torch/GPU imports")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("Choose a new or empty comparison output directory")
    if not 64 <= args.size <= 2048 or not 2 <= args.facesPerPixel <= 128:
        raise ValueError("size must be 64–2048 and facesPerPixel must be 2–128")
    start = perf_counter()
    assets, placement = prepareAssets(args.baseline, args.candidate, args.maxTextureSize)
    reference = loadPhotoReference(args.source, args.viewIndex)
    sourceFiles = [Path(__file__), Path(__file__).with_name("renderClientAssembly.py"),
                   Path(__file__).with_name("renderCropComparison.py"), projectRoot/"src/faceAppearance/pytorch3dRenderer.py",
                   projectRoot/"src/faceAppearance/photoCameras.py"]
    report = {"status": "prepared", "baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve()),
        "source": str(args.source.resolve()), "placement": placement,
        "assetDiagnostics": {name: value["diagnostics"] for name, value in assets.items()},
        "sourceHashes": reference["sourceHashes"], "implementationHashes": {str(path.relative_to(projectRoot)): fileHash(path) for path in sourceFiles},
        "photoCamera": {**reference["camera"], "kind": "savedFitted", "refitted": False, "viewIndex": args.viewIndex,
                        "imageSizeHw": reference["imageShape"], "sameBetweenBranches": True},
        "displayCrops": reference["crops"], "fullHeadImageSizeHw": [args.size, args.size],
        "originalAssetsModified": False, "photometricEvaluation": False, "anatomicalLandmarkEvaluation": False,
        "alpha": {"mode": "MASK before depth selection; inherited validated renderer", "facesPerPixel": args.facesPerPixel,
                  "cullBackfaces": True, "supersampling": False},
        "limits": ["Fixed lighting is not fitted to the reference photo; no photometric scores are computed.",
                   "Source landmarks define display crops only and are not anatomical ground truth.",
                   "Projection checks test PyTorch3D/OpenCV conversion, not anatomical fit.",
                   "Original candidate placement is recorded but not applied; both meshes use baseline placement.",
                   "Shared full-head framing uses both placed meshes and never centers branches independently.",
                   "Preview has no shadow maps, ambient occlusion, SSS, measured normal maps or rig validation."]}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepareOnly:
        report["seconds"] = perf_counter()-start
        writeJson(args.output/"comparisonPlan.json", report)
        print(json.dumps({"status": "prepared", "output": str(args.output.resolve()), "seconds": report["seconds"]}))
        return
    import torch
    import pytorch3d

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    reference["photo"].save(args.output/"referencePhoto.png")
    for part in ("eyes", "nose", "mouth"):
        reference["photo"].crop(reference["crops"][part]).save(args.output/f"reference{part.title()}.png")
    report.update(renderComparison(assets, reference, placement, args.output, device=args.device, size=args.size,
                                   facesPerPixel=args.facesPerPixel, includeClay=args.includeClay))
    report.update(status="rendered", device=args.device, pytorch3dVersion=pytorch3d.__version__, seconds=perf_counter()-start)
    if args.device.startswith("cuda"):
        report["peakAllocatedGiB"] = torch.cuda.max_memory_allocated()/1024**3
    for sourcePath in sourceFiles:
        destination = args.output/"implementation"/sourcePath.relative_to(projectRoot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sourcePath.read_bytes())
    report["outputHashes"] = {path.relative_to(args.output).as_posix(): fileHash(path)
                              for path in sorted(args.output.rglob("*.png"))}
    writeJson(args.output/"comparisonReport.json", report)
    print(json.dumps({"status": "rendered", "output": str(args.output.resolve()), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
