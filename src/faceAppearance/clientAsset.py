"""Export an assembled head without changing template vertex/face/UV indexing.

The caller supplies the established template coordinate frame in centimeters.
Only a uniform cm-to-m conversion is applied. This module does not infer a rig,
estimate PBR parameters, recalculate topology, split UV vertices, or add normals.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from PIL import Image, PngImagePlugin


def _jsonValue(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Client metadata must be JSON-compatible, received {type(value).__name__}")


def _jsonCopy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_jsonValue))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Client metadata must be finite JSON-compatible data: {error}") from error


def _writeJson(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_jsonValue)+"\n", encoding="utf-8")


def _fileIdentity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _arrayIdentity(values: np.ndarray) -> dict:
    values = np.ascontiguousarray(values)
    descriptor = {"dtype": values.dtype.str, "shape": list(values.shape), "order": "C"}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")+b"\0"+values.tobytes())
    return {**descriptor, "sha256": digest.hexdigest(), "hashEncoding": "canonical JSON dtype/shape/order, NUL, C-order array bytes"}


def _realScalar(value, name: str, *, minimum=0., maximum=1.) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be a finite scalar in [{minimum}, {maximum}]")
    return float(value)


def _finiteArray(values, name: str, columns: int) -> np.ndarray:
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != columns or len(values) < 1
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be a nonempty finite real Nx{columns} array")
    return values


def _indices(values, name: str, shape: tuple, upperBound: int) -> np.ndarray:
    values = np.asarray(values)
    if (values.shape != shape or values.dtype.kind not in "iu" or np.any(values < 0)
            or np.any(values >= upperBound)):
        raise ValueError(f"{name} must be an integer array of shape {shape} in [0,{upperBound})")
    return values


def _validateAssembly(assembly: dict) -> dict:
    if not isinstance(assembly, dict) or not isinstance(assembly.get("geometry"), dict):
        raise ValueError("assembly must contain a geometry dictionary")
    geometry = assembly["geometry"]
    required = ("vertices", "faces", "uvs", "uvFaces", "materialIds")
    if any(name not in geometry for name in required) or "templateVertices" not in assembly:
        raise ValueError("Assembly requires vertices/faces/uvs/uvFaces/materialIds and templateVertices")
    vertices = _finiteArray(geometry["vertices"], "vertices", 3)
    templateVertices = _finiteArray(assembly["templateVertices"], "templateVertices", 3)
    if vertices.shape != templateVertices.shape:
        raise ValueError("vertices and templateVertices must have identical template vertex counts/order")
    rawFaces = np.asarray(geometry["faces"])
    if rawFaces.ndim != 2 or rawFaces.shape[1:] != (3,) or len(rawFaces) < 1:
        raise ValueError("faces must contain a nonempty Fx3 triangle index array")
    faces = _indices(rawFaces, "faces", rawFaces.shape, len(vertices))
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("A triangle may not repeat a vertex index")
    uvs = _finiteArray(geometry["uvs"], "uvs", 2)
    uvFaces = _indices(geometry["uvFaces"], "uvFaces", faces.shape, len(uvs))
    materials = assembly.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("materials must be a nonempty list")
    materialIds = _indices(geometry["materialIds"], "materialIds", (len(faces),), len(materials))
    preparedMaterials = []
    legacyTextureNames = set()
    for index, material in enumerate(materials):
        if not isinstance(material, dict) or any(key not in material for key in ("name", "baseColor", "roughness", "metallic", "alphaMode")):
            raise ValueError(f"Material {index} requires name, baseColor, roughness, metallic and alphaMode")
        name = material["name"]
        if not isinstance(name, str) or not name.strip() or any(ord(character) < 32 for character in name):
            raise ValueError("Material names must be nonempty display strings without control characters")
        colors = np.asarray(material["baseColor"])
        if (colors.ndim != 3 or colors.shape[-1] not in (3, 4) or min(colors.shape[:2]) < 1
                or colors.dtype.kind != "f" or not np.isfinite(colors).all()
                or np.any(colors < 0) or np.any(colors > 1)):
            raise ValueError("baseColor must be finite linear RGB/RGBA floating-point HWC in [0,1]")
        roughness = _realScalar(material["roughness"], "roughness")
        metallic = _realScalar(material["metallic"], "metallic")
        if metallic != 0:
            raise ValueError("This head-material contract requires dielectric metallic=0")
        alphaMode = material["alphaMode"]
        if alphaMode not in ("OPAQUE", "MASK"):
            raise ValueError("alphaMode must be OPAQUE or MASK")
        alphaCutoff = _realScalar(material.get("alphaCutoff", .5), "alphaCutoff")
        legacyTextureName = material.get("legacyTextureName")
        if legacyTextureName is not None:
            if (not isinstance(legacyTextureName, str)
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", legacyTextureName) is None
                    or len(legacyTextureName) > 128
                    or legacyTextureName.lower().split(".")[0] in
                    {"con", "prn", "aux", "nul", *[f"com{number}" for number in range(1, 10)],
                     *[f"lpt{number}" for number in range(1, 10)]}):
                raise ValueError("legacyTextureName must be a portable simple PNG filename without directory components")
            if legacyTextureName.lower() in legacyTextureNames:
                raise ValueError("legacyTextureName must be unique across materials, ignoring case")
            legacyTextureNames.add(legacyTextureName.lower())
        preparedMaterials.append({"index": index, "name": name, "materialKey": f"material_{index:03d}",
                                  "baseColor": colors, "roughness": roughness, "metallic": metallic,
                                  "alphaMode": alphaMode, "alphaCutoff": alphaCutoff,
                                  "legacyTextureName": legacyTextureName})
    templateId, templateHashes = assembly.get("templateId"), assembly.get("templateHashes")
    if not isinstance(templateId, str) or not templateId.strip():
        raise ValueError("templateId must be a nonempty string identifying the established template")
    if (not isinstance(templateHashes, dict) or not templateHashes
            or any(not isinstance(key, str) or not key or not isinstance(value, str)
                   or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None for key, value in templateHashes.items())):
        raise ValueError("templateHashes must map nonempty source identifiers to SHA-256 hexadecimal strings")
    components, diagnostics = assembly.get("components"), assembly.get("diagnostics")
    if not isinstance(components, list) or any(not isinstance(component, dict) for component in components):
        raise ValueError("components must be a list of JSON metadata dictionaries")
    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be a JSON dictionary")
    for component in components:
        for key, bound in (("vertexIndices", len(vertices)), ("faceIndices", len(faces)), ("uvIndices", len(uvs))):
            if key in component:
                indices = np.asarray(component[key])
                if indices.ndim != 1:
                    raise ValueError(f"Component {key} must be a one-dimensional index list")
                if len(indices):
                    _indices(indices, f"component {key}", indices.shape, bound)
    provenanceMaps = assembly.get("provenanceMaps", {})
    if not isinstance(provenanceMaps, dict):
        raise ValueError("provenanceMaps must be a dictionary of finite float/bool head texture maps")
    preparedProvenance = {}
    provenanceNames = set()
    headShape = preparedMaterials[0]["baseColor"].shape[:2]
    for name, values in provenanceMaps.items():
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None
                or name.lower() in provenanceNames or name == "file"
                or name.lower() in {"con", "prn", "aux", "nul", *[f"com{number}" for number in range(1, 10)],
                                    *[f"lpt{number}" for number in range(1, 10)]}):
            raise ValueError("provenanceMaps names must be unique portable identifiers, excluding reserved filenames and 'file'")
        values = np.asarray(values)
        if (values.shape != headShape or values.dtype.kind not in "fb" or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)):
            raise ValueError(f"provenanceMaps[{name}] must be a finite float/bool HW array in [0,1] matching material 0 head texture {headShape}")
        provenanceNames.add(name.lower())
        preparedProvenance[name] = values
    return {"vertices": vertices, "templateVertices": templateVertices, "faces": faces,
            "uvs": uvs, "uvFaces": uvFaces, "materialIds": materialIds, "materials": preparedMaterials,
            "templateId": templateId, "templateHashes": _jsonCopy(templateHashes),
            "components": _jsonCopy(components), "diagnostics": _jsonCopy(diagnostics),
            "provenanceMaps": preparedProvenance}


def _writeBaseColor(path: Path, colors: np.ndarray) -> None:
    linear = colors[..., :3].astype(np.float64)
    srgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
    if colors.shape[-1] == 4:
        srgb = np.concatenate([srgb, colors[..., 3:4]], axis=-1)
    encoded = np.rint(np.clip(srgb, 0, 1)*255).astype(np.uint8)
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b"sRGB", b"\x00")
    Image.fromarray(encoded).save(path, pnginfo=metadata)


def exportClientAsset(outputDir: Path, assembly: dict) -> dict:
    """Export a binding-preserving OBJ/NPZ/PBR package into a new/empty directory.

    NPZ positions, template positions and deltas are float64 meters. Index and
    UV arrays retain their input dtypes, order and values. OBJ uses the same
    meter positions and independent face/UV indices; only its ``vt`` V is
    flipped once from top-left array UV to bottom-left OBJ UV. Material changes
    emit usemtl in the existing face order instead of grouping/reordering faces.

    Component metadata is serialized unchanged and declares its original cm
    coordinate convention. No unknown pivot field is silently transformed.
    Optional material legacyTextureName values create root-level PNG aliases,
    referenced by the MTL; the standard numbered textures remain authoritative.
    Optional provenanceMaps contain scalar float/bool [0,1] HW maps at material
    0 head-texture resolution. NPZ retains values/dtypes; preview PNGs encode
    linear weights directly to uint8 without sRGB transfer or image-row changes.
    The manifest hashes payload files, while archive.json outside the ZIP
    hashes the ZIP and its manifest, avoiding circular/self-referential hashes.
    """
    output = Path(outputDir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Client export requires a new or empty directory")
    data = _validateAssembly(assembly)
    verticesMeters = data["vertices"].astype(np.float64)*.01
    templateMeters = data["templateVertices"].astype(np.float64)*.01
    vertexDelta = verticesMeters-templateMeters
    if not np.isfinite(vertexDelta).all():
        raise ValueError("Vertex delta overflow after meter conversion")
    output.mkdir(parents=True, exist_ok=True)
    (output/"textures").mkdir()
    np.savez_compressed(output/"clientMesh.npz", vertices=verticesMeters, templateVertices=templateMeters,
                        vertexDelta=vertexDelta, faces=data["faces"], uvs=data["uvs"],
                        uvFaces=data["uvFaces"], materialIds=data["materialIds"])
    materials = []
    for material in data["materials"]:
        texturePath = f"textures/baseColor_{material['index']:03d}.png"
        _writeBaseColor(output/texturePath, material["baseColor"])
        if material["legacyTextureName"] is not None:
            shutil.copyfile(output/texturePath, output/material["legacyTextureName"])
        materials.append({"index": material["index"], "name": material["name"], "materialKey": material["materialKey"],
            "baseColorTexture": {"path": texturePath, "colorSpace": "sRGB", "alphaColorSpace": "linear",
                                 "sourceColorSpace": "linear", "encoding": "PNG uint8, standard sRGB OETF on RGB only",
                                 "width": material["baseColor"].shape[1], "height": material["baseColor"].shape[0],
                                 "channels": material["baseColor"].shape[2], "pixelRowOrder": "topToBottom"},
            "baseColorFactor": [1., 1., 1., 1.], "roughnessFactor": material["roughness"], "metallicFactor": material["metallic"],
            "alphaMode": material["alphaMode"], "alphaCutoff": material["alphaCutoff"],
            "parameterProvenance": {"roughnessFactor": "callerProvidedPresetNotEstimated",
                                    "metallicFactor": "dielectricPresetNotEstimated",
                                    "alphaMode": "callerProvidedAssetConfiguration"}})
        if material["legacyTextureName"] is not None:
            materials[-1]["legacyTexturePath"] = material["legacyTextureName"]
    _writeJson(output/"materials.json", {"schemaVersion": 1, "workflow": "metallicRoughness",
                "materialIds": "zero-based material index per face in clientMesh.npz; existing face order retained",
                "materials": materials,
                "limitations": ["No normal map, tangent frame, rig, skin weights or specular estimate is generated.",
                                "OBJ/MTL PBR and alpha-mask support varies; materials.json is authoritative."]})
    with (output/"face_nicp.obj").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Original template frame; meters; vertex/face/vt order preserved\nmtllib face_nicp.mtl\no face_nicp\n")
        for vertex in verticesMeters:
            handle.write("v "+" ".join(format(float(value), ".17g") for value in vertex)+"\n")
        for u, v in data["uvs"]:
            handle.write(f"vt {float(u):.17g} {1-float(v):.17g}\n")
        currentMaterial = None
        for face, uvFace, materialId in zip(data["faces"], data["uvFaces"], data["materialIds"]):
            if int(materialId) != currentMaterial:
                currentMaterial = int(materialId)
                handle.write(f"usemtl material_{currentMaterial:03d}\n")
            handle.write("f "+" ".join(f"{int(vertex)+1}/{int(uv)+1}" for vertex, uv in zip(face, uvFace))+"\n")
    with (output/"face_nicp.mtl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# PBR/alpha details are authoritative in materials.json; roughness/metallic are presets\n")
        for material in materials:
            handle.write(f"newmtl {material['materialKey']}\nKd 1 1 1\nd 1\n")
            handle.write(f"Pr {material['roughnessFactor']:.17g}\nPm {material['metallicFactor']:.17g}\n")
            mtlTexturePath = material.get("legacyTexturePath", material["baseColorTexture"]["path"])
            handle.write(f"map_Kd {mtlTexturePath}\n\n")
    _writeJson(output/"components.json", {"schemaVersion": 1, "coordinateFrame": "originalTemplateUnchanged",
                "metadataLengthUnit": "centimeter", "metadataScaleToExportedMesh": .01,
                "indexConvention": "zeroBasedOriginalTemplateIndicesUnchanged", "components": data["components"]})
    _writeJson(output/"diagnostics.json", data["diagnostics"])
    provenancePaths = {}
    if data["provenanceMaps"]:
        np.savez_compressed(output/"textureProvenance.npz", **data["provenanceMaps"])
        (output/"textureProvenance").mkdir()
        for name, values in data["provenanceMaps"].items():
            path = f"textureProvenance/{name}.png"
            Image.fromarray(np.rint(values.astype(np.float64)*255).astype(np.uint8)).save(output/path)
            provenancePaths[name] = path
    payloadPaths = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    manifest = {"schemaVersion": 1, "status": "exported", "templateId": data["templateId"], "templateHashes": data["templateHashes"],
        "paths": {"meshNpz": "clientMesh.npz", "obj": "face_nicp.obj", "mtl": "face_nicp.mtl",
                  "materials": "materials.json", "components": "components.json", "diagnostics": "diagnostics.json",
                  "archive": "clientAsset.zip", "archiveMetadata": "archive.json"}, "diagnosticsPath": "diagnostics.json",
        "coordinates": {"sourceLengthUnit": "centimeter", "exportLengthUnit": "meter", "sourceToExportScale": .01,
                        "coordinateFrame": "originalTemplateUnchanged", "axisTransform": "identity",
                        "translation": [0., 0., 0.], "positionDtype": "float64",
                        "vertexDeltaDefinition": "vertices - templateVertices in exported meter coordinates",
                        "componentMetadata": "preserved in source centimeters; components.json declares conversion"},
        "uv": {"npzConvention": "arrayNormalizedTopLeftOriginVDown", "objConvention": "bottomLeftOriginVUp",
               "objVTransform": "1 - inputV exactly once", "textureRows": "topToBottomUnchanged",
               "separateUvIndices": True, "outOfUnitRangeUvAllowed": True},
        "binding": {"vertexOrderPreserved": True, "faceOrderPreserved": True, "uvOrderPreserved": True,
                    "uvFaceOrderPreserved": True, "materialFaceOrderPreserved": True,
                    "verticesDuplicatedForUvSeams": False, "coordinateAxesChanged": False,
                    "originalBindingIndicesRemainValid": True,
                    "scope": "Existing index correspondence is retained; caller must verify this source template ID/hash matches its rig.",
                    "rigGenerated": False, "normalsGenerated": False},
        "counts": {"vertices": len(verticesMeters), "faces": len(data["faces"]), "uvs": len(data["uvs"]),
                   "materials": len(materials), "components": len(data["components"])},
        "arrayIdentities": {key: _arrayIdentity(data[key]) for key in ("faces", "uvs", "uvFaces", "materialIds", "templateVertices")},
        "arrayIdentityCoordinateUnit": {"templateVertices": "centimeter input before export conversion"},
        "files": {name: _fileIdentity(output/name) for name in payloadPaths},
        "archivePolicy": "ZIP contains payload and manifest.json with relative POSIX entries; external archive.json hashes ZIP and manifest"}
    if provenancePaths:
        manifest["paths"]["textureProvenance"] = "textureProvenance.npz"
        manifest["paths"]["textureProvenanceMasks"] = provenancePaths
        manifest["textureProvenance"] = {"materialIndex": 0,
            "height": data["materials"][0]["baseColor"].shape[0], "width": data["materials"][0]["baseColor"].shape[1],
            "source": "callerProvidedAssemblyProvenance", "valueRange": [0., 1.],
            "npzEncoding": "original values and dtypes; authoritative scalar maps",
            "pngEncoding": "uint8 linear scalar weight; round(value * 255); no sRGB transfer",
            "pixelRowOrder": "topToBottomUnchanged",
            "maps": {name: {"npzKey": name, "pngPath": provenancePaths[name], "arrayIdentity": _arrayIdentity(values)}
                     for name, values in data["provenanceMaps"].items()}}
    _writeJson(output/"manifest.json", manifest)
    archiveEntries = payloadPaths+["manifest.json"]
    with zipfile.ZipFile(output/"clientAsset.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in archiveEntries:
            archive.write(output/name, arcname=name)
    _writeJson(output/"archive.json", {"schemaVersion": 1, "path": "clientAsset.zip",
                **_fileIdentity(output/"clientAsset.zip"), "manifestSha256": _fileIdentity(output/"manifest.json")["sha256"],
                "entries": archiveEntries, "entryPaths": "relativePosix", "includesArchiveMetadata": False})
    return manifest
