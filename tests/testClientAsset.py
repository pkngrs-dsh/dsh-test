"""CPU-only contracts for an index-preserving client asset package."""

import copy
import hashlib
import json
from pathlib import PurePosixPath
import zipfile

import numpy as np
from PIL import Image
import pytest

from faceAppearance.clientAsset import exportClientAsset


@pytest.fixture
def assembly():
    template = np.array([[0., 0., 0.], [100., 0., 0.], [100., 200., 0.], [0., 200., 0.]], np.float32)
    vertices = template.copy()
    vertices[2, 2] = 3.25
    return {
        "geometry": {
            "vertices": vertices,
            "faces": np.array([[2, 3, 0], [0, 1, 2], [0, 2, 3]], np.int32),
            # UV seams use independent indices; retaining vertex order must not split vertices.
            "uvs": np.array([[.1, .2], [.8, .2], [.8, .9], [.1, .9], [1.1, -.2], [.4, .6]], np.float64),
            "uvFaces": np.array([[2, 3, 4], [0, 1, 2], [4, 2, 5]], np.int64),
            "materialIds": np.array([1, 0, 1], np.uint8),
        },
        "templateVertices": template,
        "materials": [
            {"name": "skin / 中文", "baseColor": np.array([[[0., .0031308, .5], [1., .25, .1]],
                                                             [[.2, .3, .4], [.6, .7, .8]]], np.float64),
             "roughness": .63, "metallic": 0, "alphaMode": "OPAQUE"},
            {"name": "lashes", "baseColor": np.array([[[.5, .0031308, 1., .5]]], np.float32),
             "roughness": .8, "metallic": 0, "alphaMode": "MASK", "alphaCutoff": .45},
        ],
        "components": [{"name": "eyeball", "pivot": np.array([1., 2., 3.]),
                        "vertexIndices": np.array([3, 0]), "faceIndices": [2], "uvIndices": [5, 4]}],
        "diagnostics": {"fixture": True, "confidence": np.float32(.25)},
        "templateId": "legacyTemplateFixture",
        "templateHashes": {"originalTemplateObj": "a"*64},
    }


def readJson(path):
    return json.loads(path.read_text(encoding="utf-8"))


def testNpzRetainsTopologyAndSeparateUvsInMetersWithoutInputMutation(assembly, tmp_path):
    original = copy.deepcopy(assembly)
    output = tmp_path/"client"
    manifest = exportClientAsset(output, assembly)
    with np.load(output/"clientMesh.npz", allow_pickle=False) as mesh:
        assert set(mesh.files) == {"vertices", "templateVertices", "vertexDelta", "faces", "uvs", "uvFaces", "materialIds"}
        for name in ("faces", "uvs", "uvFaces", "materialIds"):
            np.testing.assert_array_equal(mesh[name], original["geometry"][name])
            assert mesh[name].dtype == original["geometry"][name].dtype
        np.testing.assert_array_equal(mesh["vertices"], original["geometry"]["vertices"].astype(np.float64)*.01)
        np.testing.assert_array_equal(mesh["templateVertices"], original["templateVertices"].astype(np.float64)*.01)
        np.testing.assert_array_equal(mesh["vertexDelta"], mesh["vertices"]-mesh["templateVertices"])
        assert mesh["vertices"].dtype == np.float64
        assert mesh["vertexDelta"][2, 2] == .0325
        assert mesh["vertices"].shape == (4, 3) and mesh["uvs"].shape == (6, 2)
    for name, values in original["geometry"].items():
        np.testing.assert_array_equal(assembly["geometry"][name], values)
    np.testing.assert_array_equal(assembly["templateVertices"], original["templateVertices"])
    for actual, expected in zip(assembly["materials"], original["materials"]):
        np.testing.assert_array_equal(actual["baseColor"], expected["baseColor"])
    np.testing.assert_array_equal(assembly["components"][0]["pivot"], original["components"][0]["pivot"])
    assert manifest["binding"]["originalBindingIndicesRemainValid"]
    assert manifest["binding"]["verticesDuplicatedForUvSeams"] is False
    assert manifest["coordinates"]["sourceToExportScale"] == .01


def testObjPreservesFaceMaterialOrderAndFlipsOnlyV(assembly, tmp_path):
    exportClientAsset(tmp_path, assembly)
    vertices, uvs, faces, uvFaces, materialNames = [], [], [], [], []
    currentMaterial = None
    for line in (tmp_path/"face_nicp.obj").read_text().splitlines():
        words = line.split()
        if words[0] == "v":
            vertices.append([float(value) for value in words[1:]])
        elif words[0] == "vt":
            uvs.append([float(value) for value in words[1:]])
        elif words[0] == "usemtl":
            currentMaterial = words[1]
        elif words[0] == "f":
            pairs = [word.split("/") for word in words[1:]]
            assert all(len(pair) == 2 for pair in pairs)
            faces.append([int(pair[0])-1 for pair in pairs])
            uvFaces.append([int(pair[1])-1 for pair in pairs])
            materialNames.append(currentMaterial)
        assert words[0] != "vn"
    np.testing.assert_array_equal(vertices, assembly["geometry"]["vertices"].astype(np.float64)*.01)
    expectedUvs = assembly["geometry"]["uvs"].copy()
    expectedUvs[:, 1] = 1-expectedUvs[:, 1]
    np.testing.assert_array_equal(uvs, expectedUvs)
    np.testing.assert_array_equal(faces, assembly["geometry"]["faces"])
    np.testing.assert_array_equal(uvFaces, assembly["geometry"]["uvFaces"])
    assert materialNames == ["material_001", "material_000", "material_001"]
    mtl = (tmp_path/"face_nicp.mtl").read_text()
    assert "map_Kd textures/baseColor_000.png" in mtl
    assert "map_Kd textures/baseColor_001.png" in mtl
    assert "skin /" not in mtl


def testBaseColorUsesSrgbOetfButAlphaStaysLinearAndRowsStayUnchanged(assembly, tmp_path):
    exportClientAsset(tmp_path, assembly)
    skin = np.asarray(Image.open(tmp_path/"textures/baseColor_000.png"))
    np.testing.assert_array_equal(skin[0, 0], [0, 10, 188])
    np.testing.assert_array_equal(skin[0, 1], [255, 137, 89])
    np.testing.assert_array_equal(skin[1, 0], [124, 149, 170])
    lashes = Image.open(tmp_path/"textures/baseColor_001.png")
    assert lashes.mode == "RGBA" and lashes.info["srgb"] == 0
    np.testing.assert_array_equal(np.asarray(lashes)[0, 0], [188, 10, 255, 128])
    materials = readJson(tmp_path/"materials.json")["materials"]
    assert materials[0]["name"] == "skin / 中文"
    assert materials[0]["roughnessFactor"] == .63 and materials[0]["metallicFactor"] == 0
    assert materials[1]["alphaMode"] == "MASK" and materials[1]["alphaCutoff"] == .45
    assert materials[1]["baseColorTexture"]["alphaColorSpace"] == "linear"
    assert "NotEstimated" in materials[0]["parameterProvenance"]["roughnessFactor"]


def testComponentMetadataRetainsBindingAndDeclaresCentimeterConversion(assembly, tmp_path):
    manifest = exportClientAsset(tmp_path, assembly)
    components = readJson(tmp_path/"components.json")
    assert components["metadataLengthUnit"] == "centimeter"
    assert components["metadataScaleToExportedMesh"] == .01
    assert components["components"][0] == {"name": "eyeball", "pivot": [1., 2., 3.],
                                          "vertexIndices": [3, 0], "faceIndices": [2], "uvIndices": [5, 4]}
    assert readJson(tmp_path/"diagnostics.json") == {"fixture": True, "confidence": .25}
    assert not manifest["binding"]["rigGenerated"] and not manifest["binding"]["normalsGenerated"]
    assert manifest["templateHashes"] == assembly["templateHashes"]
    assert manifest["arrayIdentities"]["faces"]["dtype"] == assembly["geometry"]["faces"].dtype.str
    assert manifest["arrayIdentities"]["uvs"]["shape"] == [6, 2]


def testManifestAndZipHaveCompleteRelativeNonCircularHashes(assembly, tmp_path):
    manifest = exportClientAsset(tmp_path, assembly)
    assert manifest == readJson(tmp_path/"manifest.json")
    assert manifest["status"] == "exported"
    for name, identity in manifest["files"].items():
        payload = (tmp_path/name).read_bytes()
        assert identity == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    assert "manifest.json" not in manifest["files"]
    assert "clientAsset.zip" not in manifest["files"] and "archive.json" not in manifest["files"]
    archiveIdentity = readJson(tmp_path/"archive.json")
    with zipfile.ZipFile(tmp_path/"clientAsset.zip") as archive:
        assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
        assert set(archive.namelist()) == set(archiveIdentity["entries"])
        for name in archive.namelist():
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            assert "\\" not in name
            assert archive.read(name) == (tmp_path/name).read_bytes()
    assert archiveIdentity["sha256"] == hashlib.sha256((tmp_path/"clientAsset.zip").read_bytes()).hexdigest()
    assert archiveIdentity["manifestSha256"] == hashlib.sha256((tmp_path/"manifest.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("field,value", [
    ("vertices", np.array([[np.nan, 0, 0]])),
    ("vertices", np.ones((4, 3), bool)),
    ("faces", np.array([[0., 1., 2.]])),
    ("faces", np.array([[0, 1, 4]])),
    ("faces", np.array([[0, -1, 2]])),
    ("faces", np.array([[0, 0, 2]])),
    ("uvFaces", np.array([[0, 1, 6], [0, 1, 2], [0, 1, 2]])),
    ("uvFaces", np.array([[0, 1, 2]])),
    ("uvs", np.array([[np.inf, 0.]])),
    ("materialIds", np.array([0, 2, 0])),
    ("materialIds", np.array([[0, 1, 0]])),
])
def testInvalidMeshFailsBeforeCreatingOutput(assembly, tmp_path, field, value):
    assembly["geometry"][field] = value
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        exportClientAsset(output, assembly)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("baseColor", np.ones((2, 2, 3), np.uint8)),
    ("baseColor", np.full((2, 2, 3), 1.01)),
    ("baseColor", np.full((2, 2, 3), np.nan)),
    ("baseColor", np.ones((2, 2, 2))),
    ("roughness", -.1), ("roughness", True), ("roughness", np.inf),
    ("metallic", .1), ("alphaMode", "BLEND"), ("alphaCutoff", 1.01), ("name", "bad\nname"),
])
def testInvalidMaterialFailsBeforeCreatingOutput(assembly, tmp_path, field, value):
    assembly["materials"][0][field] = value
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        exportClientAsset(output, assembly)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("templateVertices", np.zeros((3, 3))),
    ("templateId", ""), ("templateHashes", {}), ("templateHashes", {"source": "unverified"}),
    ("components", [{"vertexIndices": [4]}]), ("components", [{"pivot": [0, np.nan, 1]}]),
    ("diagnostics", {"nonFinite": np.inf}),
])
def testInvalidIdentityOrMetadataFailsBeforeCreatingOutput(assembly, tmp_path, field, value):
    assembly[field] = value
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        exportClientAsset(output, assembly)
    assert not output.exists()


def testExistingNonemptyOutputIsNeverOverwritten(assembly, tmp_path):
    sentinel = tmp_path/"existing.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        exportClientAsset(tmp_path, assembly)
    assert sentinel.read_text() == "unchanged" and list(tmp_path.iterdir()) == [sentinel]


def testOptionalLegacyTextureAliasesAreIdenticalAndRecorded(assembly, tmp_path):
    assembly["materials"][0]["legacyTextureName"] = "head_color.png"
    assembly["materials"][1]["legacyTextureName"] = "brow_color.png"
    manifest = exportClientAsset(tmp_path, assembly)
    materials = readJson(tmp_path/"materials.json")["materials"]
    for material, alias in zip(materials, ("head_color.png", "brow_color.png")):
        assert material["legacyTexturePath"] == alias
        assert (tmp_path/alias).read_bytes() == (tmp_path/material["baseColorTexture"]["path"]).read_bytes()
        assert alias in manifest["files"]
    assert "map_Kd head_color.png" in (tmp_path/"face_nicp.mtl").read_text()
    with zipfile.ZipFile(tmp_path/"clientAsset.zip") as archive:
        assert archive.read("brow_color.png") == (tmp_path/"brow_color.png").read_bytes()


@pytest.mark.parametrize("name", ["../head.png", "/head.png", "textures/head.png", "dir\\head.png",
                                  "head.jpg", "head.png\nnewmtl injected", "con.png"])
def testUnsafeLegacyTextureAliasesFailBeforeExport(assembly, tmp_path, name):
    assembly["materials"][0]["legacyTextureName"] = name
    output = tmp_path/"invalid"
    with pytest.raises(ValueError, match="legacyTextureName"):
        exportClientAsset(output, assembly)
    assert not output.exists()


def testLegacyTextureAliasesCannotOverwriteEachOther(assembly, tmp_path):
    assembly["materials"][0]["legacyTextureName"] = "Head.png"
    assembly["materials"][1]["legacyTextureName"] = "head.png"
    with pytest.raises(ValueError, match="unique"):
        exportClientAsset(tmp_path/"invalid", assembly)


def testProvenanceMapsPreserveWeightsAndMetadataInsideArchive(assembly, tmp_path):
    maps = {"photoBlendAlpha": np.array([[0., .5], [.25, 1.]], np.float32),
            "sourceCoverage": np.array([[False, True], [True, True]]),
            "templateFallback": np.array([[1., .5], [.75, 0.]], np.float64)}
    assembly["provenanceMaps"] = maps
    original = copy.deepcopy(maps)
    manifest = exportClientAsset(tmp_path, assembly)
    assert manifest == readJson(tmp_path/"manifest.json")
    assert manifest["paths"]["textureProvenance"] == "textureProvenance.npz"
    metadata = manifest["textureProvenance"]
    assert metadata["materialIndex"] == 0 and metadata["width"] == 2 and metadata["height"] == 2
    assert metadata["valueRange"] == [0., 1.] and "no sRGB" in metadata["pngEncoding"]
    with np.load(tmp_path/"textureProvenance.npz", allow_pickle=False) as exported:
        assert set(exported.files) == set(maps)
        for name, expected in maps.items():
            np.testing.assert_array_equal(exported[name], expected)
            np.testing.assert_array_equal(maps[name], original[name])
            assert exported[name].dtype == expected.dtype
            assert metadata["maps"][name]["arrayIdentity"]["dtype"] == expected.dtype.str
    for name, pixels in {"photoBlendAlpha": [[0, 128], [64, 255]],
                         "sourceCoverage": [[0, 255], [255, 255]],
                         "templateFallback": [[255, 128], [191, 0]]}.items():
        path = manifest["paths"]["textureProvenanceMasks"][name]
        image = Image.open(tmp_path/path)
        assert image.mode == "L" and "srgb" not in image.info
        np.testing.assert_array_equal(np.asarray(image), pixels)
    with zipfile.ZipFile(tmp_path/"clientAsset.zip") as archive:
        paths = [manifest["paths"]["textureProvenance"], *manifest["paths"]["textureProvenanceMasks"].values()]
        for path in paths:
            payload = (tmp_path/path).read_bytes()
            assert archive.read(path) == payload
            assert manifest["files"][path]["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("maps", [
    None, {"sourceCoverage": np.zeros((3, 2), bool)}, {"photoBlendAlpha": np.zeros((2, 2), np.uint8)},
    {"photoBlendAlpha": np.full((2, 2), np.nan)}, {"photoBlendAlpha": np.full((2, 2), -0.1)},
    {"photoBlendAlpha": np.full((2, 2), 1.1)}, {"sourceCoverage": np.zeros((2, 2, 1), bool)},
    {"../mask": np.zeros((2, 2))}, {"file": np.zeros((2, 2))}, {"con": np.zeros((2, 2))},
    {"mask": np.zeros((2, 2)), "Mask": np.zeros((2, 2))},
])
def testInvalidProvenanceFailsBeforeCreatingOutput(assembly, tmp_path, maps):
    assembly["provenanceMaps"] = maps
    with pytest.raises(ValueError, match="provenanceMaps"):
        exportClientAsset(tmp_path/"invalid", assembly)
    assert not (tmp_path/"invalid").exists()


def testAbsentProvenanceRetainsExistingExportContract(assembly, tmp_path):
    manifest = exportClientAsset(tmp_path, assembly)
    assert "textureProvenance" not in manifest and "textureProvenance" not in manifest["paths"]
    assert not (tmp_path/"textureProvenance.npz").exists()
