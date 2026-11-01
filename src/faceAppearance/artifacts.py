"""Portable mesh, image and diagnostic exports with explicit conventions."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def jsonValue(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def writeJson(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonValue,
                               allow_nan=False), encoding="utf-8")


def writeImage(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(path)


def exportMesh(outputDir: Path, geometry: dict, includeMaterial: bool = True) -> None:
    """Convert top-left array UVs to bottom-left OBJ UVs exactly once."""
    with (outputDir / "face.obj").open("w", encoding="utf-8") as handle:
        handle.write("# UVFaceFusion canonical coordinates; units are not calibrated meters\n")
        if includeMaterial:
            handle.write("mtllib face.mtl\n")
        handle.write("o reconstructedFace\n")
        for vertex in geometry["vertices"]:
            handle.write("v " + " ".join(f"{float(v):.9g}" for v in vertex) + "\n")
        for u, v in geometry["uvs"]:
            handle.write(f"vt {float(u):.9g} {1 - float(v):.9g}\n")
        if includeMaterial:
            handle.write("usemtl facePreview\n")
        for face in geometry["faces"]:
            handle.write("f " + " ".join(f"{int(i)+1}/{int(i)+1}" for i in face) + "\n")
    if includeMaterial:
        (outputDir / "face.mtl").write_text(
            "# Preview material; roughness/specular are not reconstructed\n"
            "newmtl facePreview\nKd 1 1 1\nKs 0 0 0\nmap_Kd albedoPreview.png\n")


def saveCameraOverlays(outputDir: Path, images: np.ndarray, cameras: list[dict]):
    import cv2

    for index, (image, camera) in enumerate(zip(images, cameras)):
        overlay = np.rint(image * 255).astype(np.uint8).copy()
        originals = camera.get("diagnosticImagePoints", [])
        projected = camera.get("diagnosticProjectedPoints", [])
        for source, target in zip(originals, projected):
            if not np.isfinite(target).all() or np.abs(target).max() > 10000:
                continue
            p, q = tuple(np.rint(source).astype(int)), tuple(np.rint(target).astype(int))
            cv2.line(overlay, p, q, (255, 80, 0), 1)
            cv2.circle(overlay, p, 2, (20, 255, 20), -1)
        Image.fromarray(overlay).save(outputDir / f"camera{index:02d}.png")
