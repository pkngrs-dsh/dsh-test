"""CPU integration checks using the real client exporter and corrupt ZIPs."""

import hashlib
import io
import json
import stat
import warnings
import zipfile

import numpy as np
import pytest

from faceAppearance.clientAsset import exportClientAsset
from faceAppearance.clientImport import (
    ClientAssetImportError, auditTemplateCompatibility, buildRenderMesh, loadClientAsset,
)


@pytest.fixture
def package(tmp_path):
    template = np.array([[0., 0., 0.], [100., 0., 0.], [100., 200., 0.], [0., 200., 0.]])
    vertices = template.copy()
    vertices[2, 2] = 3.25
    assembly = {
        "geometry": {"vertices": vertices, "faces": np.array([[2, 3, 0], [0, 1, 2], [0, 2, 3]], np.int64),
                     "uvs": np.array([[.1, .2], [.8, .2], [.8, .9], [.1, .9], [1.1, -.2], [.4, .6]]),
                     "uvFaces": np.array([[2, 3, 4], [0, 1, 2], [4, 2, 5]], np.int64),
                     "materialIds": np.array([1, 0, 1], np.int64)},
        "templateVertices": template,
        "materials": [{"name": "skin", "baseColor": np.ones((2, 2, 3)) * .5,
                       "roughness": .6, "metallic": 0, "alphaMode": "OPAQUE"},
                      {"name": "eyes", "baseColor": np.ones((2, 2, 4)) * .7,
                       "roughness": .3, "metallic": 0, "alphaMode": "MASK"}],
        "components": [{"name": "fixture", "vertexIndices": [0, 1, 2, 3], "faceIndices": [0, 1, 2]}],
        "diagnostics": {"fixture": True}, "templateId": "importFixtureV1",
        "templateHashes": {"originalTemplateObj": "a" * 64},
    }
    output = tmp_path / "package"
    exportClientAsset(output, assembly)
    return output / "clientAsset.zip", assembly


def rewrite(path, changes=None, *, manifestChange=None, refreshHashes=True, extra=None):
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(payloads["manifest.json"])
    for name, payload in (changes or {}).items():
        payloads[name] = payload
        if refreshHashes:
            manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if manifestChange:
        manifestChange(manifest)
    payloads["manifest.json"] = json.dumps(manifest).encode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
            if extra:
                archive.writestr(*extra)


def rewriteMesh(path, change):
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("clientMesh.npz")), allow_pickle=False) as data:
            mesh = {name: data[name] for name in data.files}
    change(mesh)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **mesh)
    rewrite(path, {"clientMesh.npz": buffer.getvalue()})


def templateBundle(tmp_path, assembly):
    root = tmp_path / "template"
    root.mkdir()
    geometry = assembly["geometry"]
    np.savez_compressed(root / "template.npz", vertices=assembly["templateVertices"],
                        **{name: geometry[name] for name in ("faces", "uvs", "uvFaces", "materialIds")},
                        nicpToFullIds=np.arange(4), staticMask=np.array([True, True, False, True]))
    manifest = {"schemaVersion": 1, "templateId": assembly["templateId"],
                "templateHashes": assembly["templateHashes"], "templateUnits": "centimeters",
                "outputUnits": "meters", "outputUnitScale": .01, "arrayUvOrigin": "vDown",
                "files": {"template.npz": hashlib.sha256((root / "template.npz").read_bytes()).hexdigest()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def testExporterPackageLoadsWithoutCoordinateOrIndexChanges(package):
    path, assembly = package
    asset = loadClientAsset(path, expectedTemplateId=assembly["templateId"],
                            expectedTemplateHashes=assembly["templateHashes"],
                            expectedArchiveSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            expectedArchiveBytes=path.stat().st_size)
    assert asset.templateIdentityVerified
    assert asset.archivePath == path.resolve()
    for name in ("faces", "uvs", "uvFaces", "materialIds"):
        np.testing.assert_array_equal(asset.mesh[name], assembly["geometry"][name])
    np.testing.assert_array_equal(asset.mesh["vertices"], assembly["geometry"]["vertices"] * .01)
    np.testing.assert_array_equal(asset.mesh["templateVertices"], assembly["templateVertices"] * .01)
    assert asset.diagnostics == {"fixture": True}
    assert asset.materials["materials"][0]["baseColorTexture"]["path"] in asset.payloads
    assert not loadClientAsset(path).templateIdentityVerified


@pytest.mark.parametrize("arguments,message", [
    ({"expectedArchiveSha256": "0" * 64}, "archive SHA-256"),
    ({"expectedArchiveBytes": 5}, "byte count"),
    ({"expectedTemplateId": "fixture"}, "supplied together"),
    ({"expectedTemplateId": "other", "expectedTemplateHashes": {"originalTemplateObj": "a" * 64}}, "Template ID"),
    ({"expectedTemplateId": "importFixtureV1", "expectedTemplateHashes": {"originalTemplateObj": "b" * 64}}, "source hashes"),
])
def testRejectsUnexpectedDownloadOrTemplateIdentity(package, arguments, message):
    with pytest.raises(ClientAssetImportError, match=message):
        loadClientAsset(package[0], **arguments)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "folder\\file", "a//b", "./relative", "clientMesh.npz"])
def testRejectsUnsafeOrDuplicateArchiveEntries(package, name):
    rewrite(package[0], extra=(name, b"payload"))
    with pytest.raises(ClientAssetImportError, match="archive path"):
        loadClientAsset(package[0])


def testRejectsSymbolicLinkEntry(package):
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    rewrite(package[0], extra=(link, b"clientMesh.npz"))
    with pytest.raises(ClientAssetImportError, match="regular file"):
        loadClientAsset(package[0])


def testRejectsFileDirectoryCollision(package):
    rewrite(package[0], extra=("textures", b"payload"))
    with pytest.raises(ClientAssetImportError, match="path collision"):
        loadClientAsset(package[0])


def testRejectsNonfiniteJsonExponent(package):
    rewrite(package[0], {"diagnostics.json": b'{"overflow": 1e400}'})
    with pytest.raises(ClientAssetImportError, match="Nonfinite JSON"):
        loadClientAsset(package[0])


def testRejectsTamperedPayloadWithoutRefreshingHash(package):
    rewrite(package[0], {"diagnostics.json": b'{"fixture":false}'}, refreshHashes=False)
    with pytest.raises(ClientAssetImportError, match="Payload"):
        loadClientAsset(package[0])


@pytest.mark.parametrize("change,message", [
    (lambda m: m.update(schemaVersion=2), "Unsupported"),
    (lambda m: m["counts"].update(vertices=123), "counts"),
    (lambda m: m["coordinates"].update(exportLengthUnit="centimeter"), "coordinate"),
    (lambda m: m["binding"].update(vertexOrderPreserved=False), "Binding"),
    (lambda m: m["uv"].update(npzConvention="bottomLeftOriginVUp"), "UV"),
    (lambda m: m["arrayIdentities"]["faces"].update(sha256="b" * 64), "Array identity"),
])
def testRejectsIncorrectManifestClaims(package, change, message):
    rewrite(package[0], manifestChange=change)
    with pytest.raises(ClientAssetImportError, match=message):
        loadClientAsset(package[0])


@pytest.mark.parametrize("change,message", [
    (lambda m: m["vertices"].__setitem__((0, 0), np.nan), "finite"),
    (lambda m: m["vertexDelta"].__setitem__((0, 0), 1), "vertexDelta"),
    (lambda m: m["faces"].__setitem__((0, 0), 4), "index range"),
    (lambda m: m["uvFaces"].__setitem__((0, 0), -1), "index range"),
    (lambda m: m["materialIds"].__setitem__(0, 2), "index range"),
    (lambda m: m.update(vertices=m["vertices"].astype(object)), "without pickle"),
    (lambda m: m.update(uvs=np.ones((6, 3))), "Nx2"),
])
def testRejectsMalformedMeshEvenWithUpdatedPayloadHash(package, change, message):
    rewriteMesh(package[0], change)
    with pytest.raises(ClientAssetImportError, match=message):
        loadClientAsset(package[0])


def testBudgetIncludesInnerNpzExpansion(package):
    path = package[0]
    with zipfile.ZipFile(path) as outer:
        outerSize = sum(entry.file_size for entry in outer.infolist())
        with zipfile.ZipFile(io.BytesIO(outer.read("clientMesh.npz"))) as inner:
            innerSize = sum(entry.file_size for entry in inner.infolist())
    with pytest.raises(ClientAssetImportError, match="maxUncompressedBytes"):
        loadClientAsset(path, maxUncompressedBytes=outerSize - 1)
    with pytest.raises(ClientAssetImportError, match="maxUncompressedBytes"):
        loadClientAsset(path, maxUncompressedBytes=outerSize + innerSize - 1)
    assert loadClientAsset(path, maxUncompressedBytes=outerSize + innerSize)


def testRejectsHugeNpyShapeBeforeAllocation(package):
    data = io.BytesIO()
    np.lib.format.write_array_header_1_0(data, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 3)})
    mesh = io.BytesIO()
    with zipfile.ZipFile(mesh, "w", zipfile.ZIP_DEFLATED) as inner:
        inner.writestr("vertices.npy", data.getvalue())
    rewrite(package[0], {"clientMesh.npz": mesh.getvalue()})
    with pytest.raises(ClientAssetImportError, match="bounded payload"):
        loadClientAsset(package[0])


def testAuditLocalTemplateAndRenderSeamMapping(package, tmp_path):
    path, assembly = package
    root = templateBundle(tmp_path, assembly)
    asset = loadClientAsset(path)
    report = auditTemplateCompatibility(asset, root)
    assert report["status"] == "passed" and report["staticProtectedVertexCount"] == 3
    assert not report["rigValidated"] and not report["skinWeightsValidated"] and not report["blendshapesValidated"]
    assert not report["coordinateTransformApplied"]
    render = buildRenderMesh(asset)
    assert report["renderVertexCount"] == len(render["vertices"]) > report["sourceVertexCount"]
    np.testing.assert_array_equal(render["sourceVertexIds"][render["faces"]], asset.mesh["faces"])
    np.testing.assert_array_equal(render["sourceUvIds"][render["faces"]], asset.mesh["uvFaces"])
    np.testing.assert_array_equal(render["vertices"], asset.mesh["vertices"][render["sourceVertexIds"]])
    np.testing.assert_array_equal(render["uvs"], asset.mesh["uvs"][render["sourceUvIds"]])
    np.testing.assert_array_equal(render["materialIds"], asset.mesh["materialIds"])
    np.testing.assert_array_equal(render["vertexMaterialIds"][render["faces"]], np.repeat(render["materialIds"][:, None], 3, axis=1))
    assert render["sourceVertexIds"].tolist()[:3] == [2, 3, 0]


def testAuditRejectsChangedStaticVertexEvenWithConsistentDelta(package, tmp_path):
    root = templateBundle(tmp_path, package[1])
    def change(mesh):
        mesh["vertices"][0, 2] += .01
        mesh["vertexDelta"] = mesh["vertices"] - mesh["templateVertices"]
    rewriteMesh(package[0], change)
    with pytest.raises(ClientAssetImportError, match="static template vertices moved"):
        auditTemplateCompatibility(loadClientAsset(package[0]), root)


def testAuditRejectsChangedLocalTemplateBytes(package, tmp_path):
    root = templateBundle(tmp_path, package[1])
    with (root / "template.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ClientAssetImportError, match="NPZ SHA-256"):
        auditTemplateCompatibility(loadClientAsset(package[0]), root)


def testAuditRejectsWrongLocalTemplateHashes(package, tmp_path):
    root = templateBundle(tmp_path, package[1])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["templateHashes"]["originalTemplateObj"] = "f" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ClientAssetImportError, match="source hashes"):
        auditTemplateCompatibility(loadClientAsset(package[0]), root)


@pytest.mark.parametrize("name,message", [("faces", "faces correspondence"), ("vertices", "rest positions")])
def testAuditRejectsLocalTemplateMismatchWithValidFileHash(package, tmp_path, name, message):
    root = templateBundle(tmp_path, package[1])
    with np.load(root / "template.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays[name][0, 0] += 1
    np.savez_compressed(root / "template.npz", **arrays)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["template.npz"] = hashlib.sha256((root / "template.npz").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ClientAssetImportError, match=message):
        auditTemplateCompatibility(loadClientAsset(package[0]), root)
