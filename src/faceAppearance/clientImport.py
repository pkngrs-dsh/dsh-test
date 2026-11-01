"""Read client ZIPs without extraction or coordinate/index transformations.

Positions and deltas remain in exported meters; array UVs remain V-down. A
package's own template claims are not trusted for binding: callers must pin
the expected template ID and source hashes, then audit their local template.
This module checks static asset correspondence, not skeletons or animation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile
import zlib

import numpy as np


DEFAULT_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


class ClientAssetImportError(ValueError):
    """The downloaded asset or its expected template violates the contract."""


@dataclass(frozen=True)
class LoadedClientAsset:
    archivePath: Path
    archiveSha256: str
    archiveBytes: int
    manifest: dict
    mesh: dict[str, np.ndarray]
    materials: dict
    components: dict
    diagnostics: dict
    payloads: dict[str, bytes]
    templateIdentityVerified: bool


def _require(condition, message):
    if not condition:
        raise ClientAssetImportError(message)


def _json(payload: bytes, name: str) -> dict:
    def objectPairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def rejectConstant(value):
        raise ClientAssetImportError(f"Nonfinite JSON value in {name}: {value}")

    def finiteFloat(value):
        number = float(value)
        _require(math.isfinite(number), f"Nonfinite JSON value in {name}: {value}")
        return number

    try:
        value = json.loads(payload, object_pairs_hook=objectPairs, parse_constant=rejectConstant,
                           parse_float=finiteFloat)
    except (UnicodeError, ValueError) as error:
        raise ClientAssetImportError(f"Invalid JSON in {name}: {error}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value),
             f"{name} must be a SHA-256 hexadecimal string")
    return value.lower()


def _templateHashes(value):
    _require(isinstance(value, dict) and bool(value), "templateHashes must be a nonempty dictionary")
    _require(all(isinstance(key, str) and key for key in value), "Invalid template hash identifier")
    return {key: _sha(digest, f"templateHashes.{key}") for key, digest in value.items()}


def _safeName(name):
    _require(isinstance(name, str) and name and "\\" not in name and ":" not in name
             and all(ord(character) >= 32 for character in name), f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
             f"Unsafe archive path: {name!r}")
    return name


def _zipEntries(archive, limit):
    entries = archive.infolist()
    _require(0 < len(entries) <= 4096, "Invalid archive entry count")
    names = set()
    total = 0
    for entry in entries:
        _safeName(entry.filename)
        _require(entry.orig_filename == entry.filename, "Archive path contains a NUL character")
        key = entry.filename.casefold()
        _require(key not in names, f"Duplicate archive path: {entry.filename}")
        names.add(key)
        mode = entry.external_attr >> 16
        _require(not entry.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                 f"Archive entry is not a regular file: {entry.filename}")
        _require(not entry.flag_bits & 1, "Encrypted ZIP entries are unsupported")
        total += entry.file_size
        _require(total <= limit, "Archive exceeds maxUncompressedBytes")
    for name in names:
        _require(not any(parent.as_posix() in names for parent in PurePosixPath(name).parents),
                 f"Archive file/directory path collision: {name}")
    return entries, total


def _readNpz(payload, limit):
    """Bound the inner ZIP and NPY shape before NumPy allocates array memory."""
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries, total = _zipEntries(archive, limit)
        for entry in entries:
            _require(entry.filename.endswith(".npy") and "/" not in entry.filename,
                     "NPZ must contain simple NPY array entries")
            data = archive.read(entry)
            stream = io.BytesIO(data)
            version = np.lib.format.read_magic(stream)
            _require(version in ((1, 0), (2, 0)), "Unsupported NPY header version")
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = reader(stream)
            _require(not dtype.hasobject and dtype.kind in "biuf", "NPZ arrays must contain real numeric data without pickle")
            count = math.prod(shape)
            _require(count * dtype.itemsize == len(data) - stream.tell(),
                     "NPY shape does not match bounded payload size")
            arrays[entry.filename[:-4]] = np.load(io.BytesIO(data), allow_pickle=False)
    return arrays, total


def _finiteArray(values, name, columns):
    _require(values.ndim == 2 and values.shape[1] == columns and len(values) > 0
             and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             f"{name} must be a nonempty finite Nx{columns} real array")


def _indices(values, name, shape, upper):
    _require(values.shape == shape and values.dtype.kind in "iu"
             and not np.any(values < 0) and not np.any(values >= upper),
             f"{name} has invalid shape or index range")


def _arrayIdentity(values):
    values = np.ascontiguousarray(values)
    descriptor = {"dtype": values.dtype.str, "shape": list(values.shape), "order": "C"}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
                            + b"\0" + values.tobytes()).hexdigest()
    return {**descriptor, "sha256": digest}


def _validateMesh(mesh, manifest, materials, components):
    required = {"vertices", "templateVertices", "vertexDelta", "faces", "uvs", "uvFaces", "materialIds"}
    _require(set(mesh) == required, "Unexpected or missing clientMesh arrays")
    for name in ("vertices", "templateVertices", "vertexDelta"):
        _finiteArray(mesh[name], name, 3)
        _require(mesh[name].dtype == np.dtype("float64"), f"{name} must contain float64 meters")
        _require(mesh[name].shape == mesh["vertices"].shape, "Vertex array shapes differ")
    _require(np.allclose(mesh["vertexDelta"], mesh["vertices"] - mesh["templateVertices"], rtol=1e-12, atol=1e-12),
             "vertexDelta does not equal vertices - templateVertices in meters")
    faces = mesh["faces"]
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0, "faces must contain triangles")
    _indices(faces, "faces", faces.shape, len(mesh["vertices"]))
    _require(not np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0), "Triangle repeats a vertex")
    _finiteArray(mesh["uvs"], "uvs", 2)
    _indices(mesh["uvFaces"], "uvFaces", faces.shape, len(mesh["uvs"]))
    _indices(mesh["materialIds"], "materialIds", (len(faces),), len(materials["materials"]))
    counts = {"vertices": len(mesh["vertices"]), "faces": len(faces), "uvs": len(mesh["uvs"]),
              "materials": len(materials["materials"]), "components": len(components["components"])}
    _require(manifest["counts"] == counts and all(type(value) is int for value in manifest["counts"].values()),
             "Manifest counts do not match payload")
    for name in ("faces", "uvs", "uvFaces", "materialIds"):
        actual = _arrayIdentity(mesh[name])
        declared = manifest["arrayIdentities"][name]
        _require(all(declared.get(key) == value for key, value in actual.items()), f"Array identity mismatch: {name}")
    for component in components["components"]:
        _require(isinstance(component, dict), "Invalid component metadata")
        for name, upper in (("vertexIndices", counts["vertices"]), ("faceIndices", counts["faces"]),
                            ("uvIndices", counts["uvs"])):
            if name in component:
                values = np.asarray(component[name])
                _require(values.ndim == 1, f"Component {name} must be one-dimensional")
                if len(values):
                    _indices(values, f"Component {name}", values.shape, upper)


def loadClientAsset(archivePath: Path, *, expectedTemplateId: str | None = None,
                    expectedTemplateHashes: dict[str, str] | None = None,
                    expectedArchiveSha256: str | None = None, expectedArchiveBytes: int | None = None,
                    maxUncompressedBytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES) -> LoadedClientAsset:
    """Validate a downloaded schema-1 ZIP and return its unchanged mesh arrays.

    Supply both expected template fields from a trusted client template before
    reusing any external binding. ``templateIdentityVerified`` is false when
    neither is supplied. The uncompressed budget covers the outer ZIP and the
    expanded mesh NPZ together. No archive files are written or extracted.
    """
    _require(type(maxUncompressedBytes) is int and maxUncompressedBytes > 0, "Invalid maxUncompressedBytes")
    _require((expectedTemplateId is None) == (expectedTemplateHashes is None),
             "Expected template ID and hashes must be supplied together")
    try:
        path = Path(archivePath).resolve()
        with path.open("rb") as source:
            digest = hashlib.sha256()
            archiveBytes = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                archiveBytes += len(block)
                digest.update(block)
            archiveSha256 = digest.hexdigest()
            if expectedArchiveBytes is not None:
                _require(type(expectedArchiveBytes) is int and expectedArchiveBytes >= 0
                         and archiveBytes == expectedArchiveBytes, "Downloaded archive byte count mismatch")
            if expectedArchiveSha256 is not None:
                _require(archiveSha256 == _sha(expectedArchiveSha256, "expectedArchiveSha256"),
                         "Downloaded archive SHA-256 mismatch")
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                entries, outerBytes = _zipEntries(archive, maxUncompressedBytes)
                _require("manifest.json" in archive.namelist(), "Missing packaged manifest.json")
                manifest = _json(archive.read("manifest.json"), "manifest.json")
                _require(type(manifest["schemaVersion"]) is int and manifest["schemaVersion"] == 1
                         and manifest["status"] == "exported", "Unsupported client manifest")
                _require(isinstance(manifest["templateId"], str) and bool(manifest["templateId"].strip()), "Missing template ID")
                hashes = _templateHashes(manifest["templateHashes"])
                if expectedTemplateId is not None:
                    _require(manifest["templateId"] == expectedTemplateId, "Template ID mismatch")
                    _require(hashes == _templateHashes(expectedTemplateHashes), "Template source hashes mismatch")
                files = manifest["files"]
                _require(isinstance(files, dict) and files, "Missing manifest payload identities")
                _require({entry.filename for entry in entries} == set(files) | {"manifest.json"}
                         and not {"manifest.json", "archive.json", "clientAsset.zip"}.intersection(files),
                         "ZIP entries do not match manifest payloads")
                payloads = {}
                for name, identity in files.items():
                    _safeName(name)
                    _require(isinstance(identity, dict) and type(identity["bytes"]) is int
                             and identity["bytes"] == archive.getinfo(name).file_size, f"Payload byte count mismatch: {name}")
                    payload = archive.read(name)
                    _require(hashlib.sha256(payload).hexdigest() == _sha(identity["sha256"], name),
                             f"Payload SHA-256 mismatch: {name}")
                    payloads[name] = payload
        paths = manifest["paths"]
        for key in ("meshNpz", "obj", "mtl", "materials", "components", "diagnostics"):
            _require(paths[key] in payloads, f"Missing required payload: {key}")
        coordinates, binding, uv = manifest["coordinates"], manifest["binding"], manifest["uv"]
        _require(coordinates["exportLengthUnit"] == "meter" and coordinates["sourceLengthUnit"] == "centimeter"
                 and coordinates["sourceToExportScale"] == .01 and coordinates["coordinateFrame"] == "originalTemplateUnchanged"
                 and coordinates["axisTransform"] == "identity" and coordinates["translation"] == [0., 0., 0.]
                 and coordinates["positionDtype"] == "float64", "Unsupported coordinate contract")
        for key in ("vertexOrderPreserved", "faceOrderPreserved", "uvOrderPreserved", "uvFaceOrderPreserved",
                    "materialFaceOrderPreserved", "originalBindingIndicesRemainValid"):
            _require(binding[key] is True, f"Binding contract does not preserve {key}")
        _require(binding["coordinateAxesChanged"] is False and binding["verticesDuplicatedForUvSeams"] is False,
                 "Unsupported binding transformations")
        _require(uv["npzConvention"] == "arrayNormalizedTopLeftOriginVDown"
                 and uv["textureRows"] == "topToBottomUnchanged" and uv["separateUvIndices"] is True,
                 "Unsupported UV contract")
        materials = _json(payloads[paths["materials"]], paths["materials"])
        components = _json(payloads[paths["components"]], paths["components"])
        diagnostics = _json(payloads[paths["diagnostics"]], paths["diagnostics"])
        _require(type(materials["schemaVersion"]) is int and materials["schemaVersion"] == 1
                 and materials["workflow"] == "metallicRoughness" and isinstance(materials["materials"], list)
                 and materials["materials"], "Invalid materials payload")
        _require(type(components["schemaVersion"]) is int and components["schemaVersion"] == 1
                 and isinstance(components["components"], list) and components["metadataLengthUnit"] == "centimeter"
                 and components["metadataScaleToExportedMesh"] == .01
                 and components["coordinateFrame"] == "originalTemplateUnchanged"
                 and components["indexConvention"] == "zeroBasedOriginalTemplateIndicesUnchanged",
                 "Invalid component metadata contract")
        for index, material in enumerate(materials["materials"]):
            _require(type(material["index"]) is int and material["index"] == index
                     and material["baseColorTexture"]["path"] in payloads,
                     "Invalid material index or missing base color texture")
            texture = material["baseColorTexture"]
            _require(texture["colorSpace"] == "sRGB" and texture["alphaColorSpace"] == "linear"
                     and texture["pixelRowOrder"] == "topToBottom", "Unsupported base color texture contract")
            _require(material["alphaMode"] in ("OPAQUE", "MASK"), "Unsupported material alpha mode")
            for key in ("roughnessFactor", "metallicFactor", "alphaCutoff"):
                _require(type(material[key]) in (float, int) and 0 <= material[key] <= 1,
                         f"Invalid material {key}")
            _require(material["metallicFactor"] == 0, "Head material must be dielectric")
        mesh, _ = _readNpz(payloads[paths["meshNpz"]], maxUncompressedBytes - outerBytes)
        _validateMesh(mesh, manifest, materials, components)
        return LoadedClientAsset(path, archiveSha256, archiveBytes, manifest, mesh, materials, components,
                                 diagnostics, payloads, expectedTemplateId is not None)
    except ClientAssetImportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, zipfile.BadZipFile,
            NotImplementedError, RuntimeError, EOFError, struct.error, zlib.error) as error:
        raise ClientAssetImportError(f"Invalid client asset: {error}") from error


def auditTemplateCompatibility(asset: LoadedClientAsset, assetDir: Path) -> dict:
    """Compare a loaded asset to a locally trusted compiled template bundle.

    Verifies source identity, the bundle NPZ hash, original topology/UV/material
    correspondence, centimeter-to-meter rest positions, and protected static
    vertices. This cannot validate a rig, skin weights, or blendshape motion.
    """
    try:
        root = Path(assetDir).resolve()
        templateManifest = _json((root / "manifest.json").read_bytes(), "template manifest")
        _require(templateManifest["schemaVersion"] == 1 and templateManifest["templateUnits"] == "centimeters"
                 and templateManifest["outputUnits"] == "meters" and templateManifest["outputUnitScale"] == .01
                 and templateManifest["arrayUvOrigin"] == "vDown", "Unsupported local template contract")
        _require(asset.manifest["templateId"] == templateManifest["templateId"], "Local template ID mismatch")
        _require(_templateHashes(asset.manifest["templateHashes"]) == _templateHashes(templateManifest["templateHashes"]),
                 "Local template source hashes mismatch")
        templatePath = root / "template.npz"
        _require(not templatePath.is_symlink() and templatePath.is_file(), "Local template NPZ must be a regular file")
        payload = templatePath.read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == _sha(templateManifest["files"]["template.npz"], "template.npz"),
                 "Local template NPZ SHA-256 mismatch")
        template, _ = _readNpz(payload, DEFAULT_MAX_UNCOMPRESSED_BYTES)
        for name in ("faces", "uvs", "uvFaces", "materialIds"):
            _require(template[name].dtype == asset.mesh[name].dtype and np.array_equal(template[name], asset.mesh[name]),
                     f"Local template {name} correspondence changed")
        vertices = template["vertices"]
        _finiteArray(vertices, "template vertices", 3)
        _require(np.array_equal(vertices.astype(np.float64) * .01, asset.mesh["templateVertices"]),
                 "Local template rest positions differ after centimeter-to-meter conversion")
        declared = asset.manifest["arrayIdentities"]["templateVertices"]
        _require(all(declared.get(key) == value for key, value in _arrayIdentity(vertices).items()),
                 "Source template vertex identity mismatch")
        mapping, staticMask = template["nicpToFullIds"], template["staticMask"]
        _require(mapping.ndim == 1 and len(mapping) > 0, "Invalid local template NICP mapping")
        _indices(mapping, "nicpToFullIds", mapping.shape, len(vertices))
        _require(len(np.unique(mapping)) == len(mapping) and staticMask.dtype.kind == "b" and staticMask.shape == mapping.shape,
                 "Invalid local template static mask")
        protected = mapping[staticMask]
        _require(np.array_equal(asset.mesh["vertices"][protected], asset.mesh["templateVertices"][protected]),
                 "Protected static template vertices moved")
        renderMesh = buildRenderMesh(asset)
        return {"status": "passed", "templateId": templateManifest["templateId"],
                "sourceTemplateHashesMatched": True, "localTemplateNpzHashMatched": True,
                "topologyUvMaterialIndicesUnchanged": True, "templateRestPositionsMatched": True,
                "staticProtectedVerticesUnchanged": True, "staticProtectedVertexCount": int(len(protected)),
                "vertices": len(vertices), "sourceVertexCount": len(vertices),
                "renderVertexCount": len(renderMesh["vertices"]), "faces": len(template["faces"]),
                "coordinateTransformApplied": False, "lengthUnit": "meter", "arrayUvOrigin": "vDown",
                "rigValidated": False, "skinWeightsValidated": False, "blendshapesValidated": False,
                "limitations": ["Static index and template correspondence only; external skeleton, skin weights and blendshape motion require engine validation."]}
    except ClientAssetImportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, zipfile.BadZipFile,
            NotImplementedError, RuntimeError, EOFError, struct.error, zlib.error) as error:
        raise ClientAssetImportError(f"Invalid local template: {error}") from error


def buildRenderMesh(asset: LoadedClientAsset) -> dict[str, np.ndarray]:
    """Split UV/material seams with an explicit map back to binding vertices.

    Render vertices follow first triangle-corner encounter. Triangle order,
    winding, material IDs, meter positions and V-down UVs remain unchanged.
    Use ``sourceVertexIds`` to copy source skin weights or blendshape deltas
    into these render vertices; this function does not validate either driver.
    """
    mesh = asset.mesh
    keys = {}
    sourceVertices, sourceUvs, vertexMaterials = [], [], []
    faces = np.empty(mesh["faces"].shape, dtype=np.int64)
    for faceIndex, (triangle, uvTriangle, materialId) in enumerate(zip(mesh["faces"], mesh["uvFaces"], mesh["materialIds"])):
        for corner, (vertexId, uvId) in enumerate(zip(triangle, uvTriangle)):
            key = (int(vertexId), int(uvId), int(materialId))
            if key not in keys:
                keys[key] = len(keys)
                sourceVertices.append(key[0])
                sourceUvs.append(key[1])
                vertexMaterials.append(key[2])
            faces[faceIndex, corner] = keys[key]
    sourceVertexIds = np.asarray(sourceVertices, dtype=np.int64)
    sourceUvIds = np.asarray(sourceUvs, dtype=np.int64)
    return {"vertices": mesh["vertices"][sourceVertexIds], "uvs": mesh["uvs"][sourceUvIds],
            "faces": faces, "materialIds": mesh["materialIds"].copy(),
            "sourceVertexIds": sourceVertexIds, "sourceUvIds": sourceUvIds,
            "vertexMaterialIds": np.asarray(vertexMaterials, dtype=np.int64)}
