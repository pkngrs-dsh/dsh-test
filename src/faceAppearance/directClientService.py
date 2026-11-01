"""Per-request direct client registration and baking for the legacy male API.

All geometry, cameras and observations come from the current request. The
female template continues to use the separately configured baseline pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from .artifacts import writeImage, writeJson


def colmapPath(config):
    path = Path(os.environ.get("FACE_APPEARANCE_COLMAP", config.colmapExecutable))
    return (path if path.is_absolute() else config.projectRoot / path).resolve()


def checkDirectRuntime(config, assets):
    """Fail startup before accepting jobs if the selected runtime is incomplete."""
    if assets.geometryMethod != "directSurface":
        raise ValueError("directColmap requires a directSurface male asset")
    options = assets.geometryTransfer
    if options.get("restWeight") != 0 or options.get("staticConstraint") != "hard":
        raise ValueError("directColmap requires zero template prior and hard static constraints")
    executable = colmapPath(config)
    result = subprocess.run([str(executable), "mesh_texturer", "--help"],
                            capture_output=True, text=True, timeout=30, check=True)
    if "MeshTextureMapping" not in result.stdout + result.stderr:
        raise ValueError("COLMAP mesh_texturer is unavailable")
    from .adapters.eyelidLandmarks import EyelidLandmarksAdapter
    adapter = EyelidLandmarksAdapter(config.projectRoot).loadModels()
    from .clientLandmarks import loadClientLandmarks
    loadClientLandmarks(assets.arrays["nicpFaces"],
        config.projectRoot / "assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json")
    jawRuntime = {}
    from .renderedJawRuntime import getDetector
    from .clientReferenceView import METHOD as REFERENCE_METHOD
    detector = getDetector(config.projectRoot)
    if config.enableRenderedJawRefinement:
        from .clientRenderedJaw import METHOD
        jawRuntime = {"renderedJawMethod": METHOD,
                      "renderedJawWeightSha256": detector.faceAppearanceWeightSha256}
    return {"mode": "directColmap", "colmapExecutable": str(executable),
            **jawRuntime,
            "landmarkWeightSha256": adapter.modelMetadata["weightSha256"],
            "referenceSelection": REFERENCE_METHOD,
            "referenceWeightSha256": detector.faceAppearanceWeightSha256,
            "inputConvention": "One reliable frontal photograph is selected from all current inputs"}


def _load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def assembleDirectClient(assets, geometry, texture, reconstructionDirectory, photoPaths, config,
                         *, referenceViewIndex=None):
    from .clientGeometryRefinement import refineClientGeometry
    from .clientTextureBaking import bakeClientTexture

    start = perf_counter()
    root = Path(reconstructionDirectory)
    output = root / "directClient"
    output.mkdir(parents=True, exist_ok=False)
    cameras = json.loads((root / "cameras.json").read_text(encoding="utf-8"))
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    delight = _load(root / "delight.npz")
    from .clientReferenceView import selectClientReference
    from .renderedJawRuntime import observePhotos
    stageStart = perf_counter()
    referenceObservations = observePhotos(config.projectRoot, photoPaths, inputs)
    referenceSelection = selectClientReference(cameras, inputs, referenceObservations,
                                               referenceViewIndex=referenceViewIndex)
    referenceViewIndex = referenceSelection["referenceViewIndex"]
    referenceSeconds = perf_counter() - stageStart
    writeJson(output / "referenceSelection.json", referenceSelection)
    np.savez_compressed(output / "referenceObservations.npz", **{
        key: value for key, value in referenceObservations.items() if isinstance(value, np.ndarray)})
    writeJson(output / "referenceObservations.json", referenceObservations["diagnostics"])
    stageStart = perf_counter()
    refined = refineClientGeometry(assets, geometry, cameras, inputs,
        photoPaths=photoPaths, delight=delight, projectRoot=config.projectRoot,
        outputDirectory=output / "refinement", referenceViewIndex=referenceViewIndex)
    stageSeconds = {"clientReferenceSelection": referenceSeconds,
                    "directRegistrationAndRefinement": perf_counter() - stageStart}
    jawDiagnostics = {"status": "disabled"}
    accessoryReference = None
    if config.enableRenderedJawRefinement and referenceViewIndex is not None:
        from .clientRenderedJaw import refineRenderedClientJaw
        stageStart = perf_counter()
        accessoryReference = np.array(refined["vertices"], copy=True)
        jaw = refineRenderedClientJaw(assets, refined["vertices"], refined["faces"],
            refined["templateToUff"], cameras, inputs, photoPaths,
            projectRoot=config.projectRoot, outputDirectory=output / "renderedJaw",
            referenceViewIndex=referenceViewIndex, photoObservations=referenceObservations)
        refined["vertices"] = jaw["vertices"]
        transform = refined["templateToUff"]
        refined["templateVertices"] = ((jaw["vertices"] - np.asarray(transform["translation"]))
            @ np.asarray(transform["rotation"]).T) / transform["scale"]
        if refined.get("mouthState") is not None:
            refined["mouthState"]["afterCanonical"] = jaw["vertices"].copy()
            np.savez_compressed(output / "mouthState.npz", **refined["mouthState"])
        jawDiagnostics = jaw["diagnostics"]
        stageSeconds["renderedFrontalJaw"] = perf_counter() - stageStart
    elif config.enableRenderedJawRefinement:
        jawDiagnostics = {"status": "unchanged_no_shared_reference", "referenceViewIndex": None,
                          "reason": "No reliable current photograph; retain direct registration"}
    arrays = assets.arrays
    faces = np.asarray(refined["faces"], np.int64)
    if not np.array_equal(faces, arrays["nicpFaces"]):
        raise ValueError("Refined skin topology differs from client template")
    lookup = {tuple(face): index for index, face in enumerate(arrays["faces"])}
    fullIds = np.array([lookup[tuple(face)] for face in arrays["nicpToFullIds"][faces]])
    if not np.all(arrays["materialIds"][fullIds] == 0):
        raise ValueError("Client skin must use head material")
    finalGeometry = {"vertices": refined["vertices"], "faces": faces,
                     "uvs": arrays["uvs"], "uvFaces": arrays["uvFaces"][fullIds],
                     "uvConvention": "topLeftPixelEdges"}
    np.savez_compressed(output / "geometry.npz", **{
        key: value for key, value in finalGeometry.items() if isinstance(value, np.ndarray)})
    # Preserve the experiment's template-only global gain. This legacy lookup
    # determines three scalar gains; none of its transferred pixels are used.
    encoding = "linear" if "linearAlbedo" in texture else config.assemblyColorEncoding
    _, _, priorTexture = assets._transferTexture(texture, encoding)
    gain = np.asarray(priorTexture["templateLinearColorGain"])
    stageStart = perf_counter()
    baked = bakeClientTexture(finalGeometry, cameras, delight,
        assets.materials[0]["baseColor"][..., :3], output / "baking",
        colmapExecutable=colmapPath(config), mouthState=refined.get("mouthState"),
        referenceView=referenceViewIndex, fallbackGain=gain)
    stageSeconds["clientColmapAndMouthTexture"] = perf_counter() - stageStart
    provenance = {key: value for key, value in baked.items()
                  if isinstance(value, np.ndarray) and key != "linearRGB"}
    np.savez_compressed(output / "texture.npz", linearAlbedo=baked["linearRGB"], **provenance)
    # Display PNG only. Export consumes the floating point linear array below.
    linear = np.clip(baked["linearRGB"], 0, 1)
    srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    writeImage(output / "headColor.png", srgb)
    diagnostics = {"pipeline": ("directSurfaceFrontWeightedSafeJawColmapV3"
                                if config.enableRenderedJawRefinement
                                else "directSurfaceFrontLandmarksLipCreaseColmapV2"),
                   "geometry": refined["diagnostics"], "baking": baked["diagnostics"],
                   "referenceSelection": referenceSelection,
                   "renderedJaw": jawDiagnostics,
                   "templateGainSource": "Previous UV lookup used only for three fallback color gains",
                   "stageSeconds": stageSeconds,
                   "limitations": ["Qualified locally on one male turning-head sample; broad input quality is not certified.",
                       "Existing mouth-corner intersections and deep folds may remain.",
                       "Front-view refinement can worsen upward views; guarded stages may be skipped."]}
    stageStart = perf_counter()
    assembled = assets.assembleFinalClientSurface(finalGeometry, refined["templateToUff"],
        baked["linearRGB"], provenance, diagnostics=diagnostics,
        accessoryReferenceCanonical=accessoryReference)
    stageSeconds["finalClientAssembly"] = perf_counter() - stageStart
    diagnostics["seconds"] = perf_counter() - start
    writeJson(output / "report.json", diagnostics)
    return assembled, diagnostics
