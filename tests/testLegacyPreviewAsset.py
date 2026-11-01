"""CPU-only legacy ZIP preview and malformed-package checks."""

import hashlib
import io
import json
import stat
import warnings
import zipfile

import numpy as np
from PIL import Image
import pytest

from faceAppearance.clientImport import ClientAssetImportError
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import loadLegacyPreview, readLegacyTexture
import faceAppearance.legacyPreviewAsset as legacyPreview


@pytest.fixture
def package(tmp_path):
    vertices = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2], [0, 1, 2]], dtype=np.int64)
    uvs = np.array([[.1, .2], [.9, .2], [.1, .8], [.45, .65]])
    geometry = {"vertices": vertices, "faces": faces, "uvs": uvs,
                "uvFaces": np.array([[0, 1, 2], [3, 2, 1], [0, 2, 1], [1, 3, 2], [0, 1, 2]], dtype=np.int64),
                "materialIds": np.array([2, 0, 1, 3, 2], dtype=np.int64)}
    materials = []
    for index, name in enumerate(("head", "oral", "eyes", "browsLashes")):
        pixels = np.ones((2, 3, 4 if index == 3 else 3), dtype=np.float64) * .4
        if index == 3:
            pixels[..., 3] = np.array([[0, 1, 0], [1, 0, 1]])
        materials.append({"name": name, "baseColor": pixels, "roughness": .6, "metallic": 0,
                          "alphaMode": "MASK" if index == 3 else "OPAQUE", "alphaCutoff": .5,
                          "legacyTextureName": ("head_color.png", "body_color.png", "eye_color.png", "brow_color.png")[index]})
    assembly = {"geometry": geometry, "templateVertices": vertices, "materials": materials,
                "components": [], "diagnostics": {}, "templateId": "legacyMaleV1",
                "templateHashes": {"originalTemplate": "a" * 64}}
    archive = exportLegacyAsset(tmp_path / "asset", assembly, gender="male")
    return archive["archivePath"], assembly


def rewrite(path, *, changes=None, manifestChange=None, refreshHashes=True, extra=None, drop=None):
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(payloads["legacyManifest.json"])
    for name, payload in (changes or {}).items():
        payloads[name] = payload
        if refreshHashes:
            manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if manifestChange:
        manifestChange(manifest)
    payloads["legacyManifest.json"] = json.dumps(manifest).encode()
    if drop:
        del payloads[drop]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
            if extra:
                archive.writestr(*extra)


def testPreviewPreservesSourceTopologyUvMaterialOrderAndSmoothNormals(package):
    archive, assembly = package
    preview = loadLegacyPreview(archive, "preview-123")
    source = assembly["geometry"]
    mesh = preview["mesh"]
    sourceIds = np.asarray(mesh["sourceVertexIds"])
    indices = np.asarray(mesh["indices"]).reshape(-1, 3)
    np.testing.assert_array_equal(sourceIds[indices], source["faces"])
    np.testing.assert_array_equal(np.asarray(mesh["positions"]).reshape(-1, 3), source["vertices"][sourceIds] * .01)
    np.testing.assert_allclose(np.asarray(mesh["uvs"]).reshape(-1, 2)[indices], source["uvs"][source["uvFaces"]], atol=1e-15)
    normals = np.zeros_like(source["vertices"])
    points = source["vertices"][source["faces"]]
    for face, normal in zip(source["faces"], np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])):
        normals[face] += normal
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    np.testing.assert_allclose(np.asarray(mesh["normals"]).reshape(-1, 3), normals[sourceIds])
    assert preview["coordinates"] == {"lengthUnit": "meter", "uvOrigin": "vDown"}
    assert [group["materialIndex"] for group in mesh["groups"]] == [2, 0, 1, 3, 2]
    assert preview["counts"]["sourceVertices"] == 4
    assert preview["counts"]["renderVertices"] > 4
    assert preview["materials"][3]["alphaMode"] == "MASK"
    assert preview["materials"][3]["alphaCutoff"] == .5
    assert "legacy_template_identity_not_independently_verified" in preview["warnings"]
    for index, material in enumerate(preview["materials"]):
        assert material["texture"]["url"] == f"/api/jobs/preview-123/textures/{index}"
        payload = readLegacyTexture(archive, index)
        assert hashlib.sha256(payload).hexdigest() == material["texture"]["sha256"]
        assert len(payload) == material["texture"]["sizeBytes"]
    with Image.open(io.BytesIO(readLegacyTexture(archive, 3))) as brows:
        np.testing.assert_array_equal(np.asarray(brows)[..., 3], [[0, 255, 0], [255, 0, 255]])


@pytest.mark.parametrize("kind", ["hash", "size", "count", "coordinate", "uv", "missing", "extra", "alpha", "mtl", "alias", "nan"])
def testRejectsManifestPayloadAndContractDamage(package, kind):
    archive, _ = package
    if kind == "hash":
        rewrite(archive, manifestChange=lambda m: m["files"]["face_nicp.obj"].update(sha256="0" * 64))
    elif kind == "size":
        rewrite(archive, manifestChange=lambda m: m["files"]["face_nicp.obj"].update(bytes=10))
    elif kind == "count":
        rewrite(archive, manifestChange=lambda m: m["counts"].update(vertices=100))
    elif kind == "coordinate":
        rewrite(archive, manifestChange=lambda m: m["coordinates"].update(exportUnit="centimeter"))
    elif kind == "uv":
        rewrite(archive, manifestChange=lambda m: m["uv"].update(obj="top-left, V down"))
    elif kind == "missing":
        rewrite(archive, drop="head_normals.png")
    elif kind == "extra":
        rewrite(archive, extra=("surprise.txt", b"unexpected"))
    elif kind == "alpha":
        rewrite(archive, manifestChange=lambda m: m["materials"][3].update(alphaMode="OPAQUE"))
    elif kind == "mtl":
        rewrite(archive, changes={"face_nicp.mtl": b"newmtl male00SG\nmap_Kd https://outside.example/texture.png\n"})
    elif kind == "alias":
        rewrite(archive, changes={"face_nicp.png": b"not the head alias"})
    elif kind == "nan":
        with zipfile.ZipFile(archive) as source:
            obj = source.read("face_nicp.obj").replace(b"v 0 0 0", b"v nan 0 0", 1)
        rewrite(archive, changes={"face_nicp.obj": obj})
    with pytest.raises(ClientAssetImportError):
        loadLegacyPreview(archive, "fixture")


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a\\b", "C:unsafe", "a/../b", "face_nicp.obj", "FACE_NICP.OBJ", "a\x00hidden"])
def testRejectsTraversalDuplicateAndAmbiguousEntries(package, name):
    archive, _ = package
    rewrite(archive, extra=(name, b"bad"))
    with pytest.raises(ClientAssetImportError):
        loadLegacyPreview(archive, "fixture")


def testRejectsSymlinkEntry(package):
    archive, _ = package
    entry = zipfile.ZipInfo("outside")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    rewrite(archive, extra=(entry, b"/etc/passwd"))
    with pytest.raises(ClientAssetImportError):
        loadLegacyPreview(archive, "fixture")


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_UNCOMPRESSED_BYTES", "MAX_IMAGE_PIXELS", "MAX_TOTAL_IMAGE_PIXELS", "MAX_VERTICES", "MAX_FACES"])
def testEnforcesAllocationLimits(package, monkeypatch, limit):
    archive, _ = package
    monkeypatch.setattr(legacyPreview, limit, 1)
    with pytest.raises(ClientAssetImportError):
        loadLegacyPreview(archive, "fixture")


@pytest.mark.parametrize("index", [-1, 4, True, 1.0, "0"])
def testRejectsUnknownMaterial(package, index):
    with pytest.raises(ClientAssetImportError):
        readLegacyTexture(package[0], index)


@pytest.mark.parametrize("taskId", ["../task", "a?x=1", "", "x" * 129, None])
def testRejectsUnsafeTaskId(package, taskId):
    with pytest.raises(ClientAssetImportError):
        loadLegacyPreview(package[0], taskId)


def testTextureEndpointAlsoChecksWholePackage(package):
    archive, _ = package
    rewrite(archive, changes={"face_nicp.obj": b"tampered"}, refreshHashes=False)
    with pytest.raises(ClientAssetImportError):
        readLegacyTexture(archive, 3)
