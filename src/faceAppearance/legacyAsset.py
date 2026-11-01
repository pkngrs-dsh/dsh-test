"""Export the notebook's root-level ZIP names without altering client v1 output.

The legacy consumer uses the original gender-specific mesh indices and separate
head/body/brow/eye image slots. Positions are converted from template centimeters
to meters once. A normal image is required by that protocol, so the default is
an explicitly documented neutral tangent-space map, not an inferred normal map.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

from .clientAsset import _arrayIdentity, _fileIdentity, _validateAssembly, _writeBaseColor, _writeJson


LEGACY_FILES = (
    "face_nicp.obj", "face_nicp.mtl", "face_nicp.png", "head_color.png",
    "head_normals.png", "body_color.png", "brow_color.png", "eye_color.png",
)


def exportLegacyAsset(outputDir: str | Path, assembly: dict, *, gender: str,
                      normalMapPath: str | Path | None = None) -> dict:
    """Write a separate legacy ZIP and return its path, identity and file list.

    ``assembly`` is the unchanged result of ``AssemblyAssets.assemble``. The
    output directory must be new or empty. Its ``legacyAsset.zip`` contains only
    root-level payload names and ``legacyManifest.json``. The ZIP's own identity
    lives outside it in ``archive.json`` to avoid self-referential hashes.

    Unlike the notebook's single-material OBJ writer, this exporter preserves
    the four original material assignments, so ordinary OBJ viewers also bind
    the accessory textures correctly. All old payload filenames, indices,
    triangle order, UV corner indices and meter coordinates remain compatible.
    The head material references ``face_nicp.png``, an exact head-color alias.
    """
    if gender not in {"male", "female"}:
        raise ValueError("Legacy gender must be male or female")
    output = Path(outputDir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Legacy export requires a new or empty directory")
    data = _validateAssembly(assembly)
    expectedPrefix = "legacyMale" if gender == "male" else "legacyFemale"
    if not data["templateId"].startswith(expectedPrefix):
        raise ValueError("Legacy template identity does not match requested gender")
    textureNames = ["head_color.png", "body_color.png", "eye_color.png", "brow_color.png"]
    if (len(data["materials"]) != 4 or
            [item["legacyTextureName"] for item in data["materials"]] != textureNames):
        raise ValueError("Legacy material order must be head, body, eye, brow")
    verticesMeters = data["vertices"].astype(np.float64) * .01
    if not np.isfinite(verticesMeters).all():
        raise ValueError("Legacy meter coordinates must be finite")
    materialBase = "male00SG" if gender == "male" else "body_headSG"
    materialNames = [materialBase + (str(index) if index else "") for index in range(4)]
    normalPixels = None
    if normalMapPath is not None:
        normalSource = Path(normalMapPath).resolve()
        with Image.open(normalSource) as image:
            image.load()
            if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
                raise ValueError("Template normal map must be an RGB or RGBA PNG")
            normalPixels = np.asarray(image.convert("RGB")).copy()
        if normalPixels.shape[:2] != data["materials"][0]["baseColor"].shape[:2]:
            raise ValueError("Template normal map must match head atlas dimensions")
        normalProvenance = {
            "source": "callerProvidedTemplateTangentNormalMapNotReconstructed",
            "sourceFile": normalSource.name,
            "sourceSha256": _fileIdentity(normalSource)["sha256"],
            "rgbOperation": "original encoded RGB values, alpha omitted, no color transfer",
        }
    else:
        normalPixels = np.empty((*data["materials"][0]["baseColor"].shape[:2], 3), np.uint8)
        normalPixels[...] = (128, 128, 255)
        normalProvenance = {
            "source": "neutralTangentSpacePlaceholderNotReconstructed",
            "encodedRgb": [128, 128, 255],
            "intendedDirection": [0, 0, 1],
            "quantization": "8-bit UNORM approximates zero X/Y with 128/255",
        }
    output.mkdir(parents=True, exist_ok=True)
    for material, name in zip(data["materials"], textureNames):
        _writeBaseColor(output / name, material["baseColor"])
    shutil.copyfile(output / "head_color.png", output / "face_nicp.png")
    Image.fromarray(normalPixels).save(output / "head_normals.png")

    with (output / "face_nicp.obj").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Legacy template indices, meter coordinates, four original material slots\n")
        handle.write("mtllib face_nicp.mtl\no face_nicp\n")
        for vertex in verticesMeters:
            handle.write("v " + " ".join(format(float(value), ".17g") for value in vertex) + "\n")
        for u, v in data["uvs"]:
            handle.write(f"vt {float(u):.17g} {1-float(v):.17g}\n")
        current = None
        for face, uvFace, materialId in zip(data["faces"], data["uvFaces"], data["materialIds"]):
            if int(materialId) != current:
                current = int(materialId)
                handle.write(f"usemtl {materialNames[current]}\n")
            handle.write("f " + " ".join(f"{int(vertex)+1}/{int(uv)+1}"
                                           for vertex, uv in zip(face, uvFace)) + "\n")
    with (output / "face_nicp.mtl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Separate legacy color slots; normal-map provenance is in legacyManifest.json\n")
        for index, (material, name) in enumerate(zip(data["materials"], materialNames)):
            texture = "face_nicp.png" if index == 0 else textureNames[index]
            handle.write(f"newmtl {name}\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 2\n")
            handle.write(f"map_Kd {texture}\n\n")

    identities = {name: _fileIdentity(output / name) for name in LEGACY_FILES}
    manifest = {
        "schemaVersion": 1, "protocol": "main_v1_notebook_root_zip",
        "gender": gender, "templateId": data["templateId"], "templateHashes": data["templateHashes"],
        "files": identities,
        "coordinates": {"sourceUnit": "centimeter", "exportUnit": "meter", "scale": .01,
                        "axisTransform": "identity", "translation": [0, 0, 0]},
        "binding": {"vertexOrderPreserved": True, "faceOrderPreserved": True,
                    "uvOrderPreserved": True, "uvFaceOrderPreserved": True,
                    "materialFaceOrderPreserved": True, "rigGenerated": False},
        "counts": {"vertices": len(data["vertices"]), "faces": len(data["faces"]),
                   "uvs": len(data["uvs"]), "materials": len(data["materials"])},
        "arrayIdentities": {name: _arrayIdentity(data[name])
                            for name in ("faces", "uvs", "uvFaces", "materialIds", "templateVertices")},
        "uv": {"source": "array top-left, V down", "obj": "bottom-left, V up",
               "transform": "1 - source V exactly once", "imageRows": "unchanged top to bottom"},
        "materials": [{"index": index, "originalName": name, "role": material["name"],
                       "colorTexture": textureNames[index], "colorSpace": "sRGB",
                       "alphaMode": material["alphaMode"], "alphaCutoff": material["alphaCutoff"],
                       "roughnessPreset": material["roughness"], "metallicPreset": material["metallic"]}
                      for index, (name, material) in enumerate(zip(materialNames, data["materials"]))],
        "headColorAlias": {"path": "face_nicp.png", "identicalTo": "head_color.png"},
        "normalMap": {"path": "head_normals.png", "space": "tangent", "colorSpace": "linearData",
                      "encoding": "RGB uint8 UNORM, decode 2 * (value / 255) - 1",
                      "width": int(normalPixels.shape[1]), "height": int(normalPixels.shape[0]),
                      "estimatedFromPhotos": False, **normalProvenance},
        "assemblyDiagnostics": data["diagnostics"],
        "limitations": [
            "Normal-map filename compatibility does not imply recovered surface detail.",
            "This is a standalone legacy archive; the standard client v1 manifest and ZIP are unchanged.",
            "Four source material assignments replace the notebook writer's single head material; legacy root image slots remain available.",
            "OBJ/MTL alpha support varies; the consumer must apply brow alpha as declared in this manifest.",
            "Template topology compatibility does not certify anatomical fit or accessory collision freedom.",
        ],
    }
    _writeJson(output / "legacyManifest.json", manifest)
    entries = [*LEGACY_FILES, "legacyManifest.json"]
    archivePath = output / "legacyAsset.zip"
    with zipfile.ZipFile(archivePath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in entries:
            archive.write(output / name, arcname=name)
    archiveIdentity = _fileIdentity(archivePath)
    result = {"archivePath": str(archivePath), "archiveSha256": archiveIdentity["sha256"],
              "sizeBytes": archiveIdentity["bytes"], "gender": gender, "templateId": data["templateId"],
              "manifestPath": str(output / "legacyManifest.json"),
              "manifestSha256": _fileIdentity(output / "legacyManifest.json")["sha256"],
              "files": [{"name": name, "size": (output / name).stat().st_size} for name in entries]}
    _writeJson(output / "archive.json", result)
    return result
