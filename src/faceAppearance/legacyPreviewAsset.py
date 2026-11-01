"""Read the deployed legacy ZIP into the existing Three.js preview contract.

The ZIP is never extracted and OBJ/MTL references are never fetched. File hashes
establish consistency with the packaged manifest, not authenticity of a template
or rig. A caller should separately pin the downloaded archive's transport hash.
Only the deployed schema-1 legacy export is supported; older notebook ZIPs that
lack its manifest must not silently bypass validation.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import struct
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
from PIL import Image

from .clientImport import ClientAssetImportError, _json, _require, _sha, _templateHashes, _zipEntries
from .webPreview import buildPreview


MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_TOTAL_IMAGE_PIXELS = 128 * 1024 * 1024
MAX_VERTICES = 200_000
MAX_FACES = 400_000
_TEXTURES = ("head_color.png", "body_color.png", "eye_color.png", "brow_color.png")
_PAYLOADS = {"face_nicp.obj", "face_nicp.mtl", "face_nicp.png", "head_normals.png", *_TEXTURES}


def _displayString(value, name):
    _require(isinstance(value, str) and 0 < len(value) <= 128
             and value.strip() == value and all(ord(character) >= 32 for character in value),
             f"Invalid legacy {name}")
    return value


def _unitFloat(value, name):
    _require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
             f"Invalid legacy {name}")
    return float(value)


def _materials(manifest):
    values = manifest["materials"]
    _require(isinstance(values, list) and len(values) == 4, "Legacy export requires four materials")
    result, names = [], set()
    for index, material in enumerate(values):
        _require(isinstance(material, dict) and type(material["index"]) is int
                 and material["index"] == index, "Invalid legacy material index")
        name = _displayString(material["originalName"], "material name")
        _require(name not in names, "Duplicate legacy material name")
        names.add(name)
        _require(material["colorTexture"] == _TEXTURES[index]
                 and material["colorSpace"] == "sRGB", "Unsupported legacy texture slots")
        _require(material["alphaMode"] in ("OPAQUE", "MASK"), "Unsupported legacy alpha mode")
        _require(index != 3 or material["alphaMode"] == "MASK", "Legacy brows require alpha cutout")
        metallic = _unitFloat(material["metallicPreset"], "metallicPreset")
        _require(metallic == 0, "Legacy face materials must be dielectric")
        result.append({"index": index, "name": _displayString(material["role"], "material role"),
                       "roughnessFactor": _unitFloat(material["roughnessPreset"], "roughnessPreset"),
                       "metallicFactor": metallic, "alphaMode": material["alphaMode"],
                       "alphaCutoff": _unitFloat(material["alphaCutoff"], "alphaCutoff"),
                       "baseColorFactor": [1, 1, 1, 1],
                       "baseColorTexture": {"path": _TEXTURES[index]}})
    return result


def _obj(payload, manifest):
    """Parse bytes using readLegacyObj's index/UV convention without extraction.

    Its public parser requires a filesystem Path, so this bounded stream parser
    retains the same triangle and independent UV indices directly from the ZIP.
    Material indices follow the manifest, even if first face use differs.
    """
    vertices, uvs, faces, uvFaces, materialIds = [], [], [], [], []
    lookup = {item["originalName"]: item["index"] for item in manifest["materials"]}
    current = None
    materialLibrarySeen = False

    def index(value, count):
        parsed = int(value)
        _require(parsed != 0, "OBJ index zero is invalid")
        resolved = parsed - 1 if parsed > 0 else count + parsed
        _require(0 <= resolved < count, "OBJ index is out of range")
        return resolved

    for lineNumber, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        _require(len(line) <= 4096, "OBJ line is too long")
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        kind = parts[0]
        try:
            if kind == "v":
                _require(len(parts) in (4, 5), "OBJ requires three vertex coordinates")
                values = [float(value) for value in parts[1:]]
                _require(len(values) == 3 or values[3] == 1, "Unsupported OBJ homogeneous weight")
                vertices.append(values[:3])
                _require(len(vertices) <= MAX_VERTICES, "Too many OBJ vertices")
            elif kind == "vt":
                _require(len(parts) in (3, 4), "OBJ requires two texture coordinates")
                values = [float(value) for value in parts[1:]]
                _require(len(values) == 2 or values[2] == 0, "Unsupported third OBJ UV coordinate")
                uvs.append([values[0], 1 - values[1]])
                _require(len(uvs) <= MAX_VERTICES * 3, "Too many OBJ texture coordinates")
            elif kind == "usemtl":
                name = " ".join(parts[1:])
                _require(name in lookup, "OBJ references an undeclared material")
                current = lookup[name]
            elif kind == "mtllib":
                _require(parts == ["mtllib", "face_nicp.mtl"] and not materialLibrarySeen,
                         "OBJ must reference only the packaged face_nicp.mtl")
                materialLibrarySeen = True
            elif kind == "f":
                _require(len(parts) == 4 and current is not None, "OBJ needs material-assigned triangles")
                vertexIds, uvIds = [], []
                for corner in parts[1:]:
                    fields = corner.split("/")
                    _require(len(fields) in (2, 3) and fields[0] and fields[1],
                             "OBJ face corner requires vertex and UV indices")
                    vertexIds.append(index(fields[0], len(vertices)))
                    uvIds.append(index(fields[1], len(uvs)))
                _require(len(set(vertexIds)) == 3, "OBJ triangle repeats a vertex")
                faces.append(vertexIds)
                uvFaces.append(uvIds)
                materialIds.append(current)
                _require(len(faces) <= MAX_FACES, "Too many OBJ faces")
            else:
                _require(kind in ("o", "g", "s", "vn"), "Unsupported OBJ record")
        except (ValueError, OverflowError) as error:
            raise ClientAssetImportError(f"Invalid legacy OBJ line {lineNumber}: {error}") from error
    _require(materialLibrarySeen and vertices and uvs and faces, "Missing legacy OBJ geometry or material library")
    vertexArray = np.asarray(vertices, dtype=np.float64)
    uvArray = np.asarray(uvs, dtype=np.float64)
    _require(np.isfinite(vertexArray).all() and np.max(np.abs(vertexArray)) <= 10_000,
             "OBJ vertices are outside the finite meter preview range")
    _require(np.isfinite(uvArray).all() and np.max(np.abs(uvArray)) <= 1_000_000,
             "OBJ UVs are outside the finite preview range")
    expected = {"vertices": len(vertices), "faces": len(faces), "uvs": len(uvs), "materials": 4}
    _require(manifest["counts"] == expected and all(type(value) is int for value in manifest["counts"].values()),
             "Legacy manifest counts do not match OBJ")
    return {"vertices": vertexArray, "uvs": uvArray, "faces": np.asarray(faces, dtype=np.int64),
            "uvFaces": np.asarray(uvFaces, dtype=np.int64), "materialIds": np.asarray(materialIds, dtype=np.int64)}


def _mtl(payload, manifest):
    expected = {item["originalName"]: ("face_nicp.png" if item["index"] == 0 else item["colorTexture"])
                for item in manifest["materials"]}
    seen, mapped, current = set(), set(), None
    for line in payload.decode("utf-8-sig").splitlines():
        _require(len(line) <= 4096, "MTL line is too long")
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            current = " ".join(parts[1:])
            _require(current in expected and current not in seen, "Unexpected or duplicate MTL material")
            seen.add(current)
        elif parts[0] == "map_Kd":
            _require(current is not None and current not in mapped
                     and parts == ["map_Kd", expected[current]], "MTL references an unexpected texture")
            mapped.add(current)
        else:
            _require(current is not None and parts[0] in ("Ka", "Kd", "Ks", "d", "illum", "Ns", "Ni", "Tr"),
                     "Unsupported MTL record")
    _require(seen == mapped == set(expected), "MTL material texture bindings are incomplete")


def _load(archivePath):
    try:
        path = Path(archivePath)
        _require(path.is_file() and 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES,
                 "Legacy archive is missing or exceeds the compressed size limit")
        with zipfile.ZipFile(path) as archive:
            entries, _ = _zipEntries(archive, MAX_UNCOMPRESSED_BYTES)
            _require({entry.filename for entry in entries} == _PAYLOADS | {"legacyManifest.json"},
                     "Legacy ZIP has missing or unexpected files")
            for name, limit in (("legacyManifest.json", 4 * 1024 * 1024),
                                ("face_nicp.obj", 32 * 1024 * 1024), ("face_nicp.mtl", 1024 * 1024)):
                _require(archive.getinfo(name).file_size <= limit, f"Legacy {name} exceeds its size limit")
            manifest = _json(archive.read("legacyManifest.json"), "legacyManifest.json")
            _require(type(manifest["schemaVersion"]) is int and manifest["schemaVersion"] == 1
                     and manifest["protocol"] == "main_v1_notebook_root_zip", "Unsupported legacy manifest")
            _require(manifest["gender"] in ("male", "female"), "Invalid legacy gender")
            templateId = _displayString(manifest["templateId"], "template ID")
            _require(templateId.startswith("legacyMale" if manifest["gender"] == "male" else "legacyFemale"),
                     "Legacy template gender mismatch")
            _templateHashes(manifest["templateHashes"])
            files = manifest["files"]
            _require(isinstance(files, dict) and set(files) == _PAYLOADS, "Invalid legacy payload manifest")
            payloads = {}
            for name, identity in files.items():
                _require(isinstance(identity, dict) and type(identity["bytes"]) is int
                         and identity["bytes"] == archive.getinfo(name).file_size,
                         f"Legacy payload byte count mismatch: {name}")
                payload = archive.read(name)
                _require(len(payload) == identity["bytes"]
                         and hashlib.sha256(payload).hexdigest() == _sha(identity["sha256"], name),
                         f"Legacy payload SHA-256 mismatch: {name}")
                payloads[name] = payload
        coordinates, uv = manifest["coordinates"], manifest["uv"]
        _require(coordinates["exportUnit"] == "meter" and coordinates["axisTransform"] == "identity"
                 and coordinates["translation"] == [0, 0, 0], "Unsupported legacy coordinates")
        _require(uv["obj"] == "bottom-left, V up" and uv["imageRows"] == "unchanged top to bottom",
                 "Unsupported legacy UV convention")
        for name in ("vertexOrderPreserved", "faceOrderPreserved", "uvOrderPreserved",
                     "uvFaceOrderPreserved", "materialFaceOrderPreserved"):
            _require(manifest["binding"][name] is True, f"Legacy binding does not preserve {name}")
        _require(manifest["headColorAlias"] == {"path": "face_nicp.png", "identicalTo": "head_color.png"}
                 and payloads["face_nicp.png"] == payloads["head_color.png"], "Legacy head color alias differs")
        materials = _materials(manifest)
        imageModes, totalPixels = {}, 0
        for name in sorted(_PAYLOADS):
            if not name.endswith(".png"):
                continue
            with Image.open(io.BytesIO(payloads[name])) as image:
                pixels = image.width * image.height
                totalPixels += pixels
                _require(image.format == "PNG" and image.mode in ("RGB", "RGBA")
                         and 0 < pixels <= MAX_IMAGE_PIXELS and totalPixels <= MAX_TOTAL_IMAGE_PIXELS
                         and getattr(image, "n_frames", 1) == 1, f"Unsupported legacy PNG: {name}")
                imageModes[name] = image.mode
                image.verify()
            with Image.open(io.BytesIO(payloads[name])) as image:
                image.load()
        for material in materials:
            _require(material["alphaMode"] != "MASK"
                     or imageModes[material["baseColorTexture"]["path"]] == "RGBA",
                     "Legacy alpha-cutout texture requires a PNG alpha channel")
        mesh = _obj(payloads["face_nicp.obj"], manifest)
        _mtl(payloads["face_nicp.mtl"], manifest)
        return SimpleNamespace(manifest=manifest, mesh=mesh, materials={"materials": materials}, payloads=payloads)
    except ClientAssetImportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError,
            zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, struct.error, zlib.error,
            Image.DecompressionBombError) as error:
        raise ClientAssetImportError("Invalid legacy preview archive") from error


def loadLegacyPreview(archivePath: str | Path, taskId: str) -> dict:
    """Return scene.js preview JSON; preserve source vertices, winding and UVs."""
    _require(isinstance(taskId, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", taskId),
             "Invalid preview task ID")
    asset = _load(archivePath)
    preview = buildPreview(asset, taskId)
    for material in preview["materials"]:
        material["texture"]["url"] = f"/api/jobs/{taskId}/textures/{material['index']}"
    preview["warnings"].extend(["legacy_template_identity_not_independently_verified",
                                "legacy_normal_map_not_applied_to_preview"])
    return preview


def readLegacyTexture(archivePath: str | Path, materialIndex: int) -> bytes:
    """Return a verified packaged RGB/RGBA PNG, including original brow alpha."""
    _require(type(materialIndex) is int and 0 <= materialIndex < len(_TEXTURES),
             "Unknown legacy material index")
    asset = _load(archivePath)
    return asset.payloads[_TEXTURES[materialIndex]]
