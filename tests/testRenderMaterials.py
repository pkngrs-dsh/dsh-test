"""CPU checks of PyTorch3D-consistent experimental normal preparation."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch3d.structures")

from faceAppearance import renderMaterials
from faceAppearance.normalTransfer import convertUvs
from faceAppearance.texture import buildUvLookup


def planeTransfer(sourceResolution=16, targetResolution=32):
    vertices = np.array([[0, 0, 0], [1, 0, 0.4], [0, 1, 0.2]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    objUvs = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32)
    nativeUvs = convertUvs(objUvs, sourceOrigin="vUp", targetOrigin="vDown")
    source = {"vertices": vertices, "faces": faces, "uvs": objUvs, "uvFaces": faces}
    target = {"vertices": vertices, "faces": faces, "uvs": nativeUvs}
    lookup = buildUvLookup(nativeUvs, faces, targetResolution)
    mapping = {"sourceUvMap": np.zeros((targetResolution, targetResolution, 2), np.float32),
               "validMask": lookup["validMask"]}
    mask = lookup["validMask"]
    mapping["sourceUvMap"][mask] = lookup["barycentrics"][mask] @ nativeUvs
    normals = np.zeros((sourceResolution, sourceResolution, 3), np.float32)
    normals[..., 2] = 1
    return source, target, mapping, normals


def testPytorch3dAreaWeightedNormalsAreTheActualCornerNormalBasis():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 2]], np.float32)
    faces = np.array([[0, 1, 2], [0, 3, 1]], np.int64)
    uvs = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4],
                    [0.6, 0.6], [0.6, 0.9], [0.9, 0.6]], np.float32)
    geometry = {"vertices": vertices, "faces": faces, "uvs": uvs,
                "uvFaces": np.array([[0, 1, 2], [3, 4, 5]])}
    basis = renderMaterials.buildRenderBasis(geometry, 16, uvOrigin="vUp")
    sharedNormal = np.array([0, 4, 2]) / np.sqrt(20)
    np.testing.assert_allclose(basis["vertexNormals"][0], sharedNormal, atol=1e-7)
    np.testing.assert_allclose(basis["vertexNormals"][1], sharedNormal, atol=1e-7)
    np.testing.assert_allclose(basis["vertexNormals"][2:], [[0, 0, 1], [0, 1, 0]], atol=1e-7)
    np.testing.assert_allclose(basis["corners"]["cornerFrames"][..., 2],
                               basis["vertexNormals"][faces], atol=1e-7)


def testFlatNormalsMatchSlantedGeometryAcrossDifferentResolutions():
    source, target, mapping, normals = planeTransfer()
    result = renderMaterials.transferRenderNormal(source, target, normals, np.ones((16, 16)), mapping,
                                                   sourceConvention="openGL")
    valid = result["validMask"]
    assert result["tangentNormal"].shape == (32, 32, 3)
    assert valid.sum() > 200
    expected = np.array([-0.4, -0.2, 1]) / np.sqrt(1.2)
    np.testing.assert_allclose(result["objectNormal"][valid], np.tile(expected, (valid.sum(), 1)), atol=1e-6)
    np.testing.assert_allclose(result["tangentNormal"], np.broadcast_to([0, 0, 1], (32, 32, 3)), atol=1e-6)
    assert result["metadata"]["sourceConventionVerified"] is False
    assert result["metadata"]["targetConvention"] == "openGL"


def testExperimentalGreenDirectionAndNeutralMaskedFallback():
    source, target, mapping, normals = planeTransfer()
    normals[:] = [0, 0.6, 0.8]
    coverage = np.ones((16, 16), bool)
    coverage[8:13, 2:5] = False
    openGl = renderMaterials.transferRenderNormal(source, target, normals, coverage, mapping,
                                                   sourceConvention="openGL")
    directX = renderMaterials.transferRenderNormal(source, target, normals, coverage, mapping,
                                                    sourceConvention="directX")
    valid = openGl["validMask"]
    np.testing.assert_array_equal(valid, directX["validMask"])
    np.testing.assert_allclose(openGl["tangentNormal"][valid], np.tile([0, 0.6, 0.8], (valid.sum(), 1)), atol=1e-6)
    np.testing.assert_allclose(directX["tangentNormal"][valid], np.tile([0, -0.6, 0.8], (valid.sum(), 1)), atol=1e-6)
    np.testing.assert_array_equal(directX["tangentNormal"][~valid], np.tile([0, 0, 1], ((~valid).sum(), 1)))
    assert not valid[18:24, 5:8].any()
    assert openGl["metadata"]["sourceConventionIsAssumed"] is True
    np.testing.assert_allclose(normals[0, 0], [0, 0.6, 0.8])


def testUnknownSourceConventionIsRejectedBeforeLoadingAnyGeometry():
    with pytest.raises(ValueError, match="experimental sourceConvention"):
        renderMaterials.transferRenderNormal({}, {}, None, None, {}, sourceConvention="wildcap")


def testSavedMaterialsArePreparedWithoutRewritingRun(tmp_path, monkeypatch):
    source, target, mapping, normals = planeTransfer(sourceResolution=8, targetResolution=16)

    class RegistrationFixture:
        manifest = {"resultSha256": "registeredShape"}
        geometry = {"uffValidFaces": np.ones(1, bool)}

        def __init__(self, assetRoot, resolution):
            assert resolution == 16

        def toIctGeometry(self, geometry):
            np.testing.assert_array_equal(geometry["vertices"], target["vertices"])
            return {**source, "validFaceMask": np.ones(1, bool), "geometryFrame": "uvFaceFusionCanonical"}

        def atlasMap(self, direction):
            assert direction == "ictToUff"
            return mapping

    monkeypatch.setattr(renderMaterials, "RegistrationAssets", RegistrationFixture)
    resultDir = tmp_path / "savedRun"
    (resultDir / "wildcapIct").mkdir(parents=True)
    np.savez(resultDir / "geometry.npz", **target)
    np.savez(resultDir / "texture.npz", albedo=np.full((16, 16, 3), 0.4, np.float32),
             previewAlbedo=np.full((16, 16, 3), 0.45, np.float32), coverageMask=np.ones((16, 16), bool))
    np.savez(resultDir / "wildcapIct/materials.npz", diffuse=np.full((8, 8, 3), 0.7, np.float32),
             specular=np.full((8, 8), 0.2, np.float32), tangentNormal=normals, validMask=np.ones((8, 8), bool))
    (resultDir / "result.json").write_text(json.dumps({"status": "succeeded",
        "wildCapDiagnostics": {"registrationResultSha256": "registeredShape"}}))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in resultDir.rglob("*") if path.is_file()}
    prepared = renderMaterials.prepareRenderMaterials(tmp_path, resultDir)
    materialMask = prepared["materialValidMask"]
    np.testing.assert_allclose(prepared["wildCapAlbedo"][materialMask], 0.7, atol=1e-6)
    np.testing.assert_allclose(prepared["wildCapAlbedo"][~materialMask], 0.4, atol=1e-6)
    assert np.all(prepared["specular"][~materialMask] == 0)
    np.testing.assert_allclose(prepared["geometry"]["objUvs"], source["uvs"], atol=1e-7)
    assert prepared["metadata"]["diffuseTransfer"]["interpolationSpace"] == "linearLight"
    assert prepared["metadata"]["sourceConventionVerified"] is False
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in resultDir.rglob("*") if path.is_file()}
    assert before == after
