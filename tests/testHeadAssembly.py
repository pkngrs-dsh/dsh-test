"""CPU assembly invariants: binding identity, static skin, rigid parts and support."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from faceAppearance.correspondence import arrayHash
from faceAppearance.headAssembly import AssemblyAssets, decodeColor, fitSimilarity, guardDisplacement
from faceAppearance.texture import buildUvLookup


def writeJson(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/"syntheticTemplate"
    (root/"textures").mkdir(parents=True)
    xs, ys = np.meshgrid(np.arange(3), np.arange(3))
    skin = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(9))).astype(np.float64)
    skin[4, 2] = .2
    # Deliberately retain an original full-template order unlike the NICP order.
    skinIds = np.array([2, 0, 4, 1, 3, 5, 6, 7, 8], np.int64)
    vertices = np.empty((13, 3), np.float64)
    vertices[skinIds] = skin
    vertices[9:] = [[.7, .8, .3], [1.3, .8, .3], [1., 1.3, .3], [1., 1., .9]]
    skinFaces = []
    for y in range(2):
        for x in range(2):
            index = y*3+x
            skinFaces.extend([[index, index+1, index+4], [index, index+4, index+3]])
    skinFaces = np.array(skinFaces, np.int64)
    accessoryFaces = np.array([[9, 11, 10], [9, 10, 12], [10, 11, 12], [11, 9, 12]], np.int64)
    faces = np.concatenate((skinIds[skinFaces], accessoryFaces))
    sourceUvs = np.column_stack((xs.ravel()/2, ys.ravel()/2)).astype(np.float32)
    fullUvs = np.vstack((np.tile([[.2, .8]], (13, 1)), [[0., 0.], [1., 0.], [1., 1.]])).astype(np.float32)
    uvFaces = faces.copy()
    uvFaces[0] = [13, 14, 15]
    triangleIds, barycentrics = [], []
    for index in range(len(skin)):
        triangleIndex, cornerIndex = np.argwhere(skinFaces == index)[0]
        triangleIds.append(triangleIndex)
        barycentrics.append(np.eye(3)[cornerIndex])
    staticMask = np.zeros(len(skin), bool)
    staticMask[[0, 2, 6, 8]] = True
    gridX, gridY = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    atlasValid = np.zeros((16, 16), bool)
    atlasValid[2:-2, 2:-2] = True
    atlasHead = np.zeros((16, 16), bool)
    atlasHead[1:-1, 1:-1] = True
    arrays = {"vertices": vertices, "faces": faces, "uvs": fullUvs, "uvFaces": uvFaces,
              "materialIds": np.r_[np.zeros(8, np.int32), np.ones(4, np.int32)],
              "nicpToFullIds": skinIds, "nicpFaces": skinFaces, "sourceVertices": skin,
              "sourceFaces": skinFaces.copy(), "sourceUvs": sourceUvs,
              "triangleIds": np.array(triangleIds, np.int64), "barycentricWeights": np.array(barycentrics, np.float32),
              "geometryWeights": (~staticMask).astype(np.float32), "staticMask": staticMask,
              "atlasSourceUvs": np.stack((gridX, gridY), axis=-1).astype(np.float32),
              "atlasValidMask": atlasValid, "atlasHeadMask": atlasHead}
    np.savez_compressed(root/"template.npz", **arrays)
    components = [{"name": "skin", "vertexIndices": skinIds.tolist(), "faceIndices": list(range(8)), "materialId": 0},
                  {"name": "eye", "vertexIndices": list(range(9, 13)), "faceIndices": list(range(8, 12)), "materialId": 1}]
    writeJson(root/"components.json", components)
    head = np.broadcast_to(np.array([100, 150, 190, 255], np.uint8), (16, 16, 4)).copy()
    accessory = np.broadcast_to(np.array([70, 100, 130, 255], np.uint8), (2, 2, 4)).copy()
    accessory[..., 3] = [[0, 64], [128, 255]]
    Image.fromarray(head).save(root/"textures/head.png")
    Image.fromarray(accessory).save(root/"textures/eye.png")
    materials = [{"name": "head", "texture": "textures/head.png", "roughness": .6, "metallic": 0, "alphaMode": "OPAQUE"},
                 {"name": "eye", "texture": "textures/eye.png", "roughness": .2, "metallic": 0,
                  "alphaMode": "MASK", "alphaCutoff": .5}]
    manifest = {"schemaVersion": 1, "templateId": "syntheticTemplate", "templateHashes": {"fullMesh": "a"*64},
                "sourceTopologyHash": arrayHash(skinFaces.astype(np.int64), sourceUvs.astype(np.float32)),
                "materials": materials, "mappingStatus": "syntheticExactCorrespondence",
                "files": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in root.rglob("*") if path.is_file()}}
    writeJson(root/"manifest.json", manifest)
    return root, arrays


def reconstructedGeometry(arrays, *, deform=True):
    vertices = arrays["sourceVertices"].copy()
    if deform:
        vertices[4, 2] += .35
        vertices[1, 0] += .13
    angle = .31
    rotation = np.array([[np.cos(angle), np.sin(angle), 0.], [-np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    vertices = vertices @ rotation*.13+[.7, -2., .3]
    return {"vertices": vertices, "faces": arrays["sourceFaces"].copy(), "uvs": arrays["sourceUvs"].copy()}


def observedTexture(value=.2, *, covered=True):
    return {"linearAlbedo": np.full((16, 16, 3), value, np.float32),
            "coverageMask": np.full((16, 16), covered, bool)}


@pytest.fixture(scope="module")
def realAssemblyAssets():
    root = Path(__file__).resolve().parents[1]/"assets/clientTemplates/legacyFemaleV1"
    if not (root/"manifest.json").is_file():
        pytest.skip("Local compiled legacyFemaleV1 bundle is unavailable")
    return AssemblyAssets(root)


def realCoverageTexture(mode):
    coverage = np.ones((128, 128), bool)
    if mode == "none":
        coverage[:] = False
    elif mode == "lowU":
        coverage[:, 64:] = False
    elif mode == "highU":
        coverage[:, :64] = False
    elif mode != "both":
        raise ValueError("Unknown fixture coverage")
    return {"linearAlbedo": np.full((128, 128, 3), .2, np.float32), "coverageMask": coverage}


def realComponentMask(assets, labels):
    parts = [part for part in assets.components if part.get("label") in labels]
    assert parts and all(part["materialId"] == 3 for part in parts)
    faces = np.concatenate([part["faceIndices"] for part in parts])
    return buildUvLookup(assets.arrays["uvs"], assets.arrays["uvFaces"][faces],
                         assets.materials[3]["baseColor"].shape[0])["validMask"]


def testProperSimilarityRecoversRotationScaleAndTranslationWithZeroWeightOutlier():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [2., 3., 4.]])
    expectedRotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    target = source @ expectedRotation*2.4+[3., -4., 2.]
    target[-1] = [1000., -2000., 3000.]
    transform = fitSimilarity(source, target, np.array([1., 2., 1., 3., 0.]))
    np.testing.assert_allclose(transform["rotation"], expectedRotation, atol=1e-12)
    assert transform["scale"] == pytest.approx(2.4, abs=1e-12)
    np.testing.assert_allclose(transform["translation"], [3., -4., 2.], atol=1e-12)
    assert np.linalg.det(transform["rotation"]) == pytest.approx(1.)


def testSimilarityNeverAcceptsReflectionAsProperRotation():
    source = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., .5]])
    mirrored = source*np.array([-1, 1, 1])
    transform = fitSimilarity(source, mirrored, np.ones(4))
    assert np.linalg.det(transform["rotation"]) == pytest.approx(1.)
    assert transform["scale"] > 0
    residual = source @ transform["rotation"]*transform["scale"]+transform["translation"]-mirrored
    assert np.linalg.norm(residual) > .1


def testSimilarityRejectsCollinearOrUnqualifiedPoints():
    line = np.column_stack((np.arange(4), np.zeros((4, 2))))
    with pytest.raises(ValueError, match="collinear"):
        fitSimilarity(line, line, np.ones(4))
    with pytest.raises(ValueError, match="at least three"):
        fitSimilarity(line, line, np.array([1., 1., 0., 0.]))


def testLocalGuardBacktracksUnsafeTriangleWithoutReducingDistantValidDetail():
    rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                     [10., 0., 0.], [11., 0., 0.], [10., 1., 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    displacement = np.zeros_like(rest)
    displacement[2, 1] = -2.  # Full transfer flips only the first triangle.
    displacement[3:, 2] = .2
    deformed, retention, diagnostics = guardDisplacement(rest, displacement, faces)
    assert (retention[:3] < 1.).all()
    np.testing.assert_array_equal(retention[3:], 1.)
    np.testing.assert_array_equal(deformed[3:], rest[3:]+displacement[3:])
    assert diagnostics["invertedFaceCount"] == 0 and diagnostics["minNormalDot"] > .1
    assert diagnostics["minAreaRatio"] >= .08 and diagnostics["maxAreaRatio"] <= 12.


def testGuardCapsDisplacementInCentimeters():
    rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    displacement = np.tile([0., 0., 6.], (3, 1))
    deformed, retention, _ = guardDisplacement(rest, displacement, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(retention, .5)
    np.testing.assert_allclose(np.linalg.norm(deformed-rest, axis=1), 3.)


@pytest.mark.parametrize("encoding", ["srgb", "gamma2.2", "linear"])
def testColorDecodingPreservesAlphaAndInput(encoding):
    colors = np.array([[[.04045, .5, 1., .37], [0., .2, .75, 0.]]], np.float32)
    before = colors.copy()
    decoded = decodeColor(colors, encoding)
    np.testing.assert_array_equal(colors, before)
    np.testing.assert_array_equal(decoded[..., 3], before[..., 3])
    if encoding == "srgb":
        np.testing.assert_allclose(decoded[0, 0, :3], [.04045/12.92, ((.5+.055)/1.055)**2.4, 1.], atol=1e-8)
    elif encoding == "gamma2.2":
        np.testing.assert_allclose(decoded[..., :3], before[..., :3]**2.2, atol=1e-7)
    else:
        np.testing.assert_array_equal(decoded, before)


def testAssembleRetainsOriginalIndicesUvsMaterialsAndStaticVertices(bundle):
    root, arrays = bundle
    assets = AssemblyAssets(root)
    output = assets.assemble(reconstructedGeometry(arrays), observedTexture())
    geometry = output["geometry"]
    for name in ("faces", "uvs", "uvFaces", "materialIds"):
        np.testing.assert_array_equal(geometry[name], arrays[name])
        assert geometry[name].dtype == arrays[name].dtype
    np.testing.assert_array_equal(output["templateVertices"], arrays["vertices"])
    staticIds = arrays["nicpToFullIds"][arrays["staticMask"]]
    np.testing.assert_array_equal(geometry["vertices"][staticIds], arrays["vertices"][staticIds])
    freeIds = arrays["nicpToFullIds"][~arrays["staticMask"]]
    assert np.linalg.norm(geometry["vertices"][freeIds]-arrays["vertices"][freeIds]) > .01
    assert output["diagnostics"]["topologySafety"]["invertedFaceCount"] == 0
    assert output["diagnostics"]["staticVertexCount"] == len(staticIds)


def testGlobalSourceSimilarityDoesNotChangeTemplateShapeOrFrame(bundle):
    root, arrays = bundle
    output = AssemblyAssets(root).assemble(reconstructedGeometry(arrays, deform=False), observedTexture())
    np.testing.assert_allclose(output["geometry"]["vertices"], arrays["vertices"], atol=1e-12)
    assert output["diagnostics"]["templateToUff"]["scale"] == pytest.approx(.13, abs=1e-12)


def testAccessoryPlacementPreservesRigidDistancesAndDeclaredTransform(bundle):
    root, arrays = bundle
    output = AssemblyAssets(root).assemble(reconstructedGeometry(arrays), observedTexture())
    component = output["components"][1]
    ids = component["vertexIndices"]
    before, after = arrays["vertices"][ids], output["geometry"]["vertices"][ids]
    np.testing.assert_allclose(np.linalg.norm(before[:, None]-before[None, :], axis=-1),
                               np.linalg.norm(after[:, None]-after[None, :], axis=-1), atol=1e-12)
    matrix = component["transformRow4x4"]
    np.testing.assert_allclose((np.c_[before, np.ones(len(ids))] @ matrix)[:, :3], after, atol=1e-12)
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.)
    assert component["transformAppliesToComponent"] and component["placement"] == "localSkinRigidFit"
    np.testing.assert_allclose(component["pivotAfterCm"], after.mean(0), atol=1e-12)
    assert np.linalg.norm(after-before) > .001


def testBlackIsAnObservedColorAndAccessoryRgbaIsUnchanged(bundle):
    root, arrays = bundle
    assets = AssemblyAssets(root)
    output = assets.assemble(reconstructedGeometry(arrays), observedTexture(0.))
    maps = output["provenanceMaps"]
    np.testing.assert_array_equal(maps["sourceCoverage"], arrays["atlasValidMask"])
    assert maps["photoBlendAlpha"][7, 7] == 1.
    np.testing.assert_array_equal(output["materials"][0]["baseColor"][7, 7, :3], 0.)
    np.testing.assert_array_equal(output["materials"][1]["baseColor"], assets.materials[1]["baseColor"])
    np.testing.assert_allclose(output["materials"][1]["baseColor"][..., 3], np.array([[0, 64], [128, 255]])/255, atol=1e-8)
    assert output["materials"][1]["alphaMode"] == "MASK"


def testMissingCoverageUsesTemplateWithoutPretendingItWasObserved(bundle):
    root, arrays = bundle
    assets = AssemblyAssets(root)
    output = assets.assemble(reconstructedGeometry(arrays), observedTexture(1., covered=False))
    maps = output["provenanceMaps"]
    assert not maps["sourceCoverage"].any() and not maps["photoBlendAlpha"].any()
    np.testing.assert_array_equal(maps["templateFallback"], arrays["atlasHeadMask"].astype(np.float32))
    np.testing.assert_array_equal(output["materials"][0]["baseColor"], assets.materials[0]["baseColor"])
    assert output["diagnostics"]["texture"]["headObservedTexelFraction"] == 0.
    assert output["diagnostics"]["texture"]["headPhotoBlendFraction"] == 0.


def testCoverageHoleAndUnmappedHeadRemainFallback(bundle):
    root, arrays = bundle
    assets = AssemblyAssets(root)
    texture = observedTexture(0.)
    texture["coverageMask"][6:10, 6:10] = False
    output = assets.assemble(reconstructedGeometry(arrays), texture)
    maps = output["provenanceMaps"]
    assert not maps["sourceCoverage"][7, 7] and maps["photoBlendAlpha"][7, 7] == 0.
    assert maps["templateFallback"][7, 7] == 1.
    assert not maps["sourceCoverage"][1, 1] and maps["templateFallback"][1, 1] == 1.
    assert maps["templateFallback"][0, 0] == 0.
    np.testing.assert_array_equal(maps["photoBlendAlpha"]+maps["templateFallback"], arrays["atlasHeadMask"].astype(np.float32))
    np.testing.assert_array_equal(output["materials"][0]["baseColor"][7, 7], assets.materials[0]["baseColor"][7, 7])


def testRepeatedAssemblyIsIsolatedFromReturnedArrayAndMetadataMutation(bundle):
    root, arrays = bundle
    assets = AssemblyAssets(root)
    geometry, texture = reconstructedGeometry(arrays), observedTexture()
    baseline = assets.assemble(geometry, texture)
    first = assets.assemble(geometry, texture)
    first["geometry"]["vertices"][:] = 999.
    first["geometry"]["faces"][:] = 0
    first["templateVertices"][:] = 999.
    first["materials"][0]["baseColor"][:] = 1.
    first["materials"][1]["baseColor"][:] = 1.
    first["components"][1]["vertexIndices"][0] = 999
    first["components"][1]["faceIndices"][0] = 999
    first["components"][1]["pivotAfterCm"][:] = 999.
    first["templateHashes"]["fullMesh"] = "b"*64
    first["provenanceMaps"]["photoBlendAlpha"][:] = 0.
    second = assets.assemble(geometry, texture)
    for name, expected in baseline["geometry"].items():
        np.testing.assert_array_equal(second["geometry"][name], expected)
    for actual, expected in zip(second["materials"], baseline["materials"]):
        np.testing.assert_array_equal(actual["baseColor"], expected["baseColor"])
    for name in ("vertexIndices", "faceIndices", "pivotAfterCm"):
        np.testing.assert_array_equal(second["components"][1][name], baseline["components"][1][name])
    assert second["templateHashes"]["fullMesh"] == "a"*64
    assert second["provenanceMaps"]["photoBlendAlpha"].max() > 0
    for name, expected in arrays.items():
        np.testing.assert_array_equal(assets.arrays[name], expected)
        assert not assets.arrays[name].flags.writeable


@pytest.mark.parametrize("changed", ["faces", "uvs", "vertices"])
def testReconstructedSourceIdentityMismatchIsRejected(bundle, changed):
    root, arrays = bundle
    geometry = reconstructedGeometry(arrays)
    if changed == "faces":
        geometry["faces"] = geometry["faces"][::-1].copy()
    elif changed == "uvs":
        geometry["uvs"][0, 0] += .01
    else:
        geometry["vertices"] = geometry["vertices"][:-1]
    with pytest.raises(ValueError, match="identity"):
        AssemblyAssets(root).assemble(geometry, observedTexture())


@pytest.mark.parametrize("filename", ["template.npz", "components.json", "textures/head.png"])
def testPreparedFileHashMismatchFailsBeforeUse(bundle, filename):
    root, _ = bundle
    with (root/filename).open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        AssemblyAssets(root)


def testPreparedSourceTopologyHashMismatchIsRejected(bundle):
    root, _ = bundle
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["sourceTopologyHash"] = "0"*64
    writeJson(root/"manifest.json", manifest)
    with pytest.raises(ValueError, match="topology identity"):
        AssemblyAssets(root)


@pytest.mark.parametrize("name", ["vertices", "sourceVertices"])
def testPreparedIntegerCoordinatesCannotSilentlyTruncateDisplacement(bundle, name):
    root, arrays = bundle
    arrays[name] = arrays[name].astype(np.int64)
    np.savez_compressed(root/"template.npz", **arrays)
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"]["template.npz"] = hashlib.sha256((root/"template.npz").read_bytes()).hexdigest()
    writeJson(root/"manifest.json", manifest)
    with pytest.raises(ValueError, match="coordinate"):
        AssemblyAssets(root)


def testManifestCannotLoadFilesOutsidePreparedBundle(bundle):
    root, _ = bundle
    outside = root.parent/"outside.txt"
    outside.write_text("not an asset")
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"]["../outside.txt"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    writeJson(root/"manifest.json", manifest)
    with pytest.raises(ValueError, match="Invalid template asset path"):
        AssemblyAssets(root)


def testRealCompiledBundleLoadsAndRejectsChangedSourceIdentity(realAssemblyAssets):
    assets = realAssemblyAssets
    assert assets.manifest["templateId"] == "legacyFemaleV1"
    assert len(assets.arrays["vertices"]) == 3544
    assert len(assets.arrays["nicpToFullIds"]) == 2795
    assert len(assets.materials) == 4
    assert np.unique(assets.arrays["materialIds"]).tolist() == [0, 1, 2, 3]
    geometry = {"vertices": assets.arrays["sourceVertices"].copy(),
                "faces": assets.arrays["sourceFaces"].copy(), "uvs": assets.arrays["sourceUvs"].copy()}
    geometry["uvs"][0, 0] += .01
    with pytest.raises(ValueError, match="identity"):
        assets._deform(geometry)


def testRealHairTintIsConstantPresetRgbWithoutMultiplyingWhiteCarrierOrAlpha(realAssemblyAssets, monkeypatch):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    original = assets.materials[3]["baseColor"]
    # The legacy image contains carrier RGB variation, so multiplication is not equivalent.
    assert np.ptp(original[..., :3]) > .1
    materials, _, diagnostics = assets._transferTexture(realCoverageTexture("none"), "linear")
    expectedTint = decodeColor(np.array([[[.12, .085, .06]]], np.float32), "srgb")[0, 0]
    np.testing.assert_array_equal(materials[3]["baseColor"][..., :3], np.broadcast_to(expectedTint, original[..., :3].shape))
    np.testing.assert_array_equal(materials[3]["baseColor"][..., 3], original[..., 3])
    policy = next(item for item in diagnostics["hairMaterialPolicy"] if item["materialId"] == 3)
    assert policy["rgbPolicy"] == "constantPresetTintWithOriginalAlpha" and not policy["estimatedFromPhoto"]


def testBothObservedRealBrowsHideOnlyTheirExactTrianglesAndPreserveLashes(realAssemblyAssets, monkeypatch):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    materials, _, diagnostics = assets._transferTexture(realCoverageTexture("both"), "linear")
    support = diagnostics["eyebrowPhotoSupport"]
    assert len(support) == 2 and min(item["fraction"] for item in support) >= assets.eyebrowCoverageThreshold
    assert diagnostics["hiddenEyebrowMaterialIds"] == [3]
    browMask = realComponentMask(assets, {"eyebrow"})
    originalAlpha = assets.materials[3]["baseColor"][..., 3]
    changedAlpha = materials[3]["baseColor"][..., 3]
    assert np.count_nonzero(originalAlpha[browMask]) > 100
    np.testing.assert_array_equal(changedAlpha[browMask], 0.)
    np.testing.assert_array_equal(changedAlpha[~browMask], originalAlpha[~browMask])
    for label in ("upperEyelash", "lowerEyelash"):
        lashMask = realComponentMask(assets, {label})
        assert not (browMask & lashMask).any()
        assert np.count_nonzero(originalAlpha[lashMask]) > 100
        np.testing.assert_array_equal(changedAlpha[lashMask], originalAlpha[lashMask])


@pytest.mark.parametrize("mode", ["lowU", "highU"])
def testEitherUnsupportedRealBrowKeepsSharedAtlasAlphaForBoth(realAssemblyAssets, monkeypatch, mode):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    materials, _, diagnostics = assets._transferTexture(realCoverageTexture(mode), "linear")
    fractions = sorted(item["fraction"] for item in diagnostics["eyebrowPhotoSupport"])
    assert len(fractions) == 2 and fractions[0] == 0. and fractions[1] >= assets.eyebrowCoverageThreshold
    assert diagnostics["hiddenEyebrowMaterialIds"] == []
    np.testing.assert_array_equal(materials[3]["baseColor"][..., 3], assets.materials[3]["baseColor"][..., 3])


def testNoRealPhotoCoveragePreservesBrowAlphaAndDoesNotReusePreviousHiddenResult(realAssemblyAssets, monkeypatch):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    assets._transferTexture(realCoverageTexture("both"), "linear")
    materials, maps, diagnostics = assets._transferTexture(realCoverageTexture("none"), "linear")
    assert diagnostics["hiddenEyebrowMaterialIds"] == []
    assert all(item["fraction"] == 0. for item in diagnostics["eyebrowPhotoSupport"])
    assert not maps["sourceCoverage"].any() and not maps["photoBlendAlpha"].any()
    np.testing.assert_array_equal(materials[3]["baseColor"][..., 3], assets.materials[3]["baseColor"][..., 3])


def testHiddenRealBrowComponentsRetainTheirMeshPositionsIndicesAndBinding(realAssemblyAssets, monkeypatch):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    geometry = {"vertices": assets.arrays["sourceVertices"].copy(),
                "faces": assets.arrays["sourceFaces"].copy(), "uvs": assets.arrays["sourceUvs"].copy()}
    hidden = assets.assemble(geometry, realCoverageTexture("both"), colorEncoding="linear")
    visible = assets.assemble(geometry, realCoverageTexture("none"), colorEncoding="linear")
    assert hidden["diagnostics"]["texture"]["hiddenEyebrowMaterialIds"] == [3]
    assert visible["diagnostics"]["texture"]["hiddenEyebrowMaterialIds"] == []
    for name, expected in visible["geometry"].items():
        np.testing.assert_array_equal(hidden["geometry"][name], expected)
    for hiddenPart, visiblePart in zip(hidden["components"], visible["components"]):
        assert hiddenPart["name"] == visiblePart["name"]
        for name in ("vertexIndices", "faceIndices", "pivotAfterCm", "transformRow4x4"):
            np.testing.assert_array_equal(hiddenPart[name], visiblePart[name])
    monkeypatch.setattr(assets, "hidePresetEyelashes", True)
    hiddenLashes = assets.assemble(geometry, realCoverageTexture("none"), colorEncoding="linear")
    assert hiddenLashes["diagnostics"]["texture"]["hiddenEyelashMaterialIds"] == [3]
    for name, expected in visible["geometry"].items():
        np.testing.assert_array_equal(hiddenLashes["geometry"][name], expected)
    for hiddenPart, visiblePart in zip(hiddenLashes["components"], visible["components"]):
        for name in ("vertexIndices", "faceIndices", "pivotAfterCm", "transformRow4x4"):
            np.testing.assert_array_equal(hiddenPart[name], visiblePart[name])


def testReturnedRealHairTintMetadataCannotPolluteFollowingCalls(realAssemblyAssets, monkeypatch):
    assets = realAssemblyAssets
    monkeypatch.setattr(assets, "hidePresetEyelashes", False)
    first, _, _ = assets._transferTexture(realCoverageTexture("none"), "linear")
    # Assert isolation before mutating to keep this shared fixture intact on a regression.
    assert first[3]["hairTintSrgb"] is not assets.materials[3]["hairTintSrgb"]
    first[3]["hairTintSrgb"][:] = [1., 1., 1.]
    second, _, _ = assets._transferTexture(realCoverageTexture("none"), "linear")
    assert second[3]["hairTintSrgb"] == [.12, .085, .06]
    assert assets.materials[3]["hairTintSrgb"] == [.12, .085, .06]


def testDefaultRealEyelashHidingIsExactAndClearlyPresetEvenWithoutPhotoCoverage(realAssemblyAssets):
    assets = realAssemblyAssets
    assert assets.manifest["appearanceOptions"]["hidePresetEyelashes"] is True
    assert assets.hidePresetEyelashes is True
    materials, maps, diagnostics = assets._transferTexture(realCoverageTexture("none"), "linear")
    lashMask = realComponentMask(assets, {"upperEyelash", "lowerEyelash"})
    browMask = realComponentMask(assets, {"eyebrow"})
    originalAlpha = assets.materials[3]["baseColor"][..., 3]
    changedAlpha = materials[3]["baseColor"][..., 3]
    assert np.count_nonzero(originalAlpha[lashMask]) > 100
    np.testing.assert_array_equal(changedAlpha[lashMask], 0.)
    np.testing.assert_array_equal(changedAlpha[~lashMask], originalAlpha[~lashMask])
    np.testing.assert_array_equal(changedAlpha[browMask], originalAlpha[browMask])
    assert diagnostics["hiddenEyelashMaterialIds"] == [3]
    assert diagnostics["eyelashPolicy"] == "hiddenByPresetConfiguration"
    assert diagnostics["hiddenEyebrowMaterialIds"] == []
    assert diagnostics["headObservedTexelFraction"] == 0.
    assert not maps["sourceCoverage"].any() and not maps["photoBlendAlpha"].any()
