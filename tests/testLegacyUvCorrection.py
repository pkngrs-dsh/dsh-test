"""CPU evidence for conservative per-corner UV correction and runtime assets."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from faceAppearance.barycentricDeformation import BarycentricDeformer
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyUvCorrection import compileUvCorrection


def gridMesh(side):
    xs, ys = np.meshgrid(np.linspace(0., 1., side), np.linspace(0., 1., side))
    uvs = np.column_stack((xs.ravel(), ys.ravel()))
    faces = []
    for y in range(side-1):
        for x in range(side-1):
            index = y*side+x
            faces.extend([[index, index+1, index+side+1], [index, index+side+1, index+side]])
    return uvs, np.asarray(faces, np.int64)


@pytest.fixture
def uvFixture():
    uvs, faces = gridMesh(3)
    skinIds = np.array([2, 0, 4, 1, 3, 5, 6, 7, 8], np.int64)
    rest = np.c_[uvs*2, np.zeros(len(uvs))]
    fullVertices = np.empty_like(rest)
    fullVertices[skinIds] = rest
    resolution = 33
    atlasX, atlasY = np.meshgrid(np.linspace(0., 1., resolution), np.linspace(0., 1., resolution))
    eligible = np.ones((resolution, resolution), bool)
    eligible[14:17, 14:17] = False
    arrays = {"vertices": fullVertices, "faces": skinIds[faces][::-1].copy(),
              "uvs": uvs.copy(), "uvFaces": faces[::-1].copy(), "materialIds": np.zeros(len(faces), np.int32),
              "nicpFaces": faces, "nicpToFullIds": skinIds, "staticMask": np.zeros(len(rest), bool),
              "atlasValidMask": eligible, "atlasHeadMask": np.ones_like(eligible),
              "atlasSourceUvs": (.2+.6*np.stack((atlasX, atlasY), axis=-1)).astype(np.float32),
              "sourceUvs": uvs.copy(), "sourceFaces": faces.copy(), "sourceVertices": rest.copy()}
    samples = {"sampleNicpFaceIds": np.repeat(np.arange(len(faces)), 3),
               "sampleNicpWeights": np.tile([[.6, .2, .2], [.2, .6, .2], [.2, .2, .6]], (len(faces), 1)),
               "sampleWeights": np.ones(3*len(faces))}
    return arrays, .2+.6*uvs, faces.copy(), samples


def compileKnownField(fixture, correction):
    arrays, skinUvs, skinUvFaces, samples = fixture
    faceIds, bary = samples["sampleNicpFaceIds"], samples["sampleNicpWeights"]
    currentCorners = skinUvs[skinUvFaces[faceIds]]
    sampleOffset = (correction[arrays["nicpFaces"][faceIds]]*bary[..., None]).sum(axis=1)
    samples["sampleUvs"] = (currentCorners*bary[..., None]).sum(axis=1)+sampleOffset
    deformer = BarycentricDeformer(arrays["vertices"][arrays["nicpToFullIds"]], arrays["nicpFaces"],
        faceIds, bary, samples["sampleWeights"], arrays["staticMask"], smoothWeight=0., restWeight=0.)
    return compileUvCorrection(arrays, skinUvs, skinUvFaces, samples, deformer)


def testKnownUvTranslationMovesSamplingInCorrectDirectionWithoutChangingEligibility(uvFixture):
    arrays, _, _, _ = uvFixture
    shift = np.array([.035, -.025])
    corrected, diagnostics = compileKnownField(uvFixture, np.broadcast_to(shift, (9, 2)).copy())
    np.testing.assert_allclose(corrected["uvCorrectionPerNicpVertex"], np.broadcast_to(shift, (9, 2)), atol=1e-14)
    np.testing.assert_array_equal(corrected["atlasValidMask"], arrays["atlasValidMask"])
    valid = corrected["atlasValidMask"]
    np.testing.assert_allclose(corrected["atlasSourceUvs"][valid], arrays["atlasSourceUvs"][valid]+shift, atol=1e-7)
    np.testing.assert_array_equal(corrected["atlasSourceUvs"][~valid], 0.)
    assert diagnostics["medianSampleErrorAfterTexels"] < 1e-12
    assert diagnostics["medianSampleErrorBeforeTexels"] > 1.
    assert diagnostics["anatomicalAccuracyCertified"] is False


def testStaticVerticesArePinnedInKnownUvOffsetField(uvFixture):
    arrays, _, _, _ = uvFixture
    arrays["staticMask"][[0, 2, 6, 8]] = True
    expected = np.tile([.025, -.015], (9, 1))
    expected[arrays["staticMask"]] = 0.
    corrected, diagnostics = compileKnownField(uvFixture, expected)
    np.testing.assert_allclose(corrected["uvCorrectionPerNicpVertex"], expected, atol=1e-14)
    np.testing.assert_array_equal(corrected["uvCorrectionPerNicpVertex"][arrays["staticMask"]], 0.)
    assert diagnostics["staticVertexCorrectionsExactlyZero"]
    assert diagnostics["medianSampleErrorAfterTexels"] < 1e-12


def testIndependentPerCornerSeamIsPreservedInActualAtlasSampling(uvFixture):
    arrays, skinUvs, _, samples = uvFixture
    faces = arrays["nicpFaces"]
    corners = skinUvs[faces].copy()
    leftFaces = arrays["sourceUvs"][faces, 0].mean(axis=1) < .5
    corners[leftFaces, :, 0] -= .04
    corners[~leftFaces, :, 0] += .04
    independentUvs = corners.reshape(-1, 2)
    independentFaces = np.arange(len(independentUvs)).reshape(-1, 3)
    shift = np.array([.01, .015])
    corrected, _ = compileKnownField((arrays, independentUvs, independentFaces, samples), np.tile(shift, (9, 1)))
    atlas = corrected["atlasSourceUvs"]
    assert corrected["atlasValidMask"][8, 15] and corrected["atlasValidMask"][8, 17]
    np.testing.assert_allclose(atlas[8, 15], [.2+.6*15/32-.04+.01, .2+.6*8/32+.015], atol=1e-7)
    np.testing.assert_allclose(atlas[8, 17], [.2+.6*17/32+.04+.01, .2+.6*8/32+.015], atol=1e-7)
    assert atlas[8, 17, 0]-atlas[8, 15, 0] == pytest.approx(.6*2/32+.08, abs=1e-7)
    assert len(independentUvs) > len(arrays["vertices"])


def testCorrectedDomainNeverAddsPreviouslyUnsupportedAtlasPixels(uvFixture):
    arrays, _, _, _ = uvFixture
    arrays["atlasValidMask"][:5, :] = False
    arrays["atlasValidMask"][25:, 20:] = False
    corrected, diagnostics = compileKnownField(uvFixture, np.tile([.1, .02], (9, 1)))
    assert not (corrected["atlasValidMask"] & ~arrays["atlasValidMask"]).any()
    np.testing.assert_array_equal(corrected["atlasSourceUvs"][~arrays["atlasValidMask"]], 0.)
    assert diagnostics["correctedEligibleAtlasTexels"] <= diagnostics["baseEligibleAtlasTexels"]


def testReversedUvTrianglesAreRejectedInsteadOfRendered(uvFixture):
    arrays, skinUvs, _, _ = uvFixture
    correction = np.zeros((9, 2))
    correction[:, 0] = 1.-2*skinUvs[:, 0]
    corrected, diagnostics = compileKnownField(uvFixture, correction)
    assert not corrected["correctedUvFaceValid"].any()
    assert not corrected["atlasValidMask"].any()
    np.testing.assert_array_equal(corrected["atlasSourceUvs"], 0.)
    assert diagnostics["rejectedUvFaceCount"] == len(arrays["nicpFaces"])


def testCollapsedUvTrianglesAreRejected(uvFixture):
    arrays, skinUvs, _, _ = uvFixture
    correction = np.zeros((9, 2))
    correction[:, 0] = .5-skinUvs[:, 0]
    corrected, diagnostics = compileKnownField(uvFixture, correction)
    assert not corrected["correctedUvFaceValid"].any()
    assert not corrected["atlasValidMask"].any()
    assert diagnostics["rejectedUvFaceCount"] == len(arrays["nicpFaces"])


def testSamplingInsideSourceUvHoleIsRejectedEvenWhenOldAtlasWasEligible(uvFixture):
    arrays, _, _, _ = uvFixture
    sourceUvs, sourceFaces = gridMesh(5)
    centers = sourceUvs[sourceFaces].mean(axis=1)
    holeFaces = ((centers[:, 0] > .25) & (centers[:, 0] < .5)
                 & (centers[:, 1] > .25) & (centers[:, 1] < .5))
    arrays["sourceUvs"], arrays["sourceFaces"] = sourceUvs, sourceFaces[~holeFaces]
    arrays["sourceVertices"] = np.c_[sourceUvs, np.zeros(len(sourceUvs))]
    assert arrays["atlasValidMask"][9, 9]
    corrected, diagnostics = compileKnownField(uvFixture, np.zeros((9, 2)))
    assert not corrected["atlasValidMask"][9, 9]  # Native UV around (.369,.369) falls inside the hole.
    np.testing.assert_array_equal(corrected["atlasSourceUvs"][9, 9], 0.)
    assert corrected["atlasValidMask"][26, 26]
    assert diagnostics["rejectedUvFaceCount"] == 0
    assert diagnostics["correctedEligibleAtlasTexels"] < diagnostics["baseEligibleAtlasTexels"]


def testOutOfChartSamplesAreRejectedWithoutInventingCoverage(uvFixture):
    corrected, diagnostics = compileKnownField(uvFixture, np.tile([.5, 0.], (9, 1)))
    assert corrected["atlasValidMask"][8, 2]
    assert not corrected["atlasValidMask"][8, 26]
    np.testing.assert_array_equal(corrected["atlasSourceUvs"][8, 26], 0.)
    assert diagnostics["rejectedUvFaceCount"] == 0


def testCompilationDoesNotMutateInputsOrReuseOutputStorage(uvFixture):
    arrays, skinUvs, skinUvFaces, samples = uvFixture
    faceIds, bary = samples["sampleNicpFaceIds"], samples["sampleNicpWeights"]
    samples["sampleUvs"] = (skinUvs[skinUvFaces[faceIds]]*bary[..., None]).sum(axis=1)+[.02, .01]
    deformer = BarycentricDeformer(arrays["vertices"][arrays["nicpToFullIds"]], arrays["nicpFaces"],
        faceIds, bary, samples["sampleWeights"], arrays["staticMask"], smoothWeight=0., restWeight=0.)
    beforeArrays = {name: values.copy() for name, values in arrays.items()}
    beforeSamples = {name: values.copy() for name, values in samples.items()}
    beforeUvs, beforeFaces = skinUvs.copy(), skinUvFaces.copy()
    for values in (*arrays.values(), *samples.values(), skinUvs, skinUvFaces):
        values.setflags(write=False)
    first, _ = compileUvCorrection(arrays, skinUvs, skinUvFaces, samples, deformer)
    expected = {name: values.copy() for name, values in first.items()}
    for values in first.values():
        values[:] = 0
    second, _ = compileUvCorrection(arrays, skinUvs, skinUvFaces, samples, deformer)
    for name, values in second.items():
        np.testing.assert_array_equal(values, expected[name])
    for name, values in arrays.items():
        np.testing.assert_array_equal(values, beforeArrays[name])
    for name, values in samples.items():
        np.testing.assert_array_equal(values, beforeSamples[name])
    np.testing.assert_array_equal(skinUvs, beforeUvs)
    np.testing.assert_array_equal(skinUvFaces, beforeFaces)


@pytest.fixture(scope="module")
def realBundles():
    templateRoot = Path(__file__).resolve().parents[1]/"assets/clientTemplates"
    paths = [templateRoot/name for name in ("legacyFemaleBarycentricV1", "legacyFemaleBarycentricUvV1")]
    if not all((path/"manifest.json").is_file() for path in paths):
        pytest.skip("Local barycentric and UV-corrected experimental bundles are unavailable")
    return tuple(AssemblyAssets(path) for path in paths)


def gradientTexture():
    x, y = np.meshgrid(np.linspace(0., 1., 128), np.linspace(0., 1., 128))
    return {"linearAlbedo": np.stack((x, y, .2+.2*x+.3*y), axis=-1).astype(np.float32),
            "coverageMask": np.ones((128, 128), bool)}


def sourceGeometry(assets):
    return {"vertices": assets.arrays["sourceVertices"].copy(), "faces": assets.arrays["sourceFaces"].copy(),
            "uvs": assets.arrays["sourceUvs"].copy()}


def testRealCorrectedBundleHashesLoadAndBindingUvsStayUnchanged(realBundles):
    baseline, corrected = realBundles
    path = corrected.assetRoot/"uvCorrection.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == corrected.manifest["files"]["uvCorrection.npz"]
    with np.load(path, allow_pickle=False) as saved:
        for name in ("atlasSourceUvs", "atlasValidMask"):
            np.testing.assert_array_equal(saved[name], corrected.arrays[name])
        np.testing.assert_array_equal(saved["uvCorrectionPerNicpVertex"][corrected.arrays["staticMask"]], 0.)
    for name in ("vertices", "faces", "uvs", "uvFaces", "materialIds", "sourceUvs"):
        np.testing.assert_array_equal(corrected.arrays[name], baseline.arrays[name])
    assert not (corrected.arrays["atlasValidMask"] & ~baseline.arrays["atlasValidMask"]).any()
    common = corrected.arrays["atlasValidMask"] & baseline.arrays["atlasValidMask"]
    assert np.max(np.abs(corrected.arrays["atlasSourceUvs"][common]-baseline.arrays["atlasSourceUvs"][common])) > .001
    assert corrected.manifest["textureTransfer"]["anatomicalAccuracyCertified"] is False


def testRealUvCorrectionChangesSampledTextureButNotGeometryOrBinding(realBundles):
    baseline, corrected = realBundles
    geometry, texture = sourceGeometry(baseline), gradientTexture()
    before = baseline.assemble(geometry, texture, colorEncoding="linear")
    after = corrected.assemble(geometry, texture, colorEncoding="linear")
    for name, values in before["geometry"].items():
        np.testing.assert_array_equal(after["geometry"][name], values)
    common = (before["provenanceMaps"]["photoBlendAlpha"] == 1) & (after["provenanceMaps"]["photoBlendAlpha"] == 1)
    assert common.sum() > 100
    difference = np.abs(before["materials"][0]["baseColor"][common, :3]-after["materials"][0]["baseColor"][common, :3])
    assert difference.mean() > .001
    for materialId in (1, 2, 3):
        np.testing.assert_array_equal(before["materials"][materialId]["baseColor"], after["materials"][materialId]["baseColor"])


def testRealChangedSourceGeometryChangesMeshWithoutChangingFrozenUvTransfer(realBundles):
    _, assets = realBundles
    geometry, texture = sourceGeometry(assets), gradientTexture()
    before = assets.assemble(geometry, texture, colorEncoding="linear")
    row = np.flatnonzero(assets.barycentricArrays["sampleWeights"] > 0)[0]
    faceId = assets.barycentricArrays["sourceTriangleIds"][row]
    vertexId = assets.arrays["sourceFaces"][faceId, 0]
    geometry["vertices"][vertexId, 2] += .02
    after = assets.assemble(geometry, texture, colorEncoding="linear")
    assert np.max(np.abs(after["geometry"]["vertices"]-before["geometry"]["vertices"])) > 1e-6
    for name in ("faces", "uvs", "uvFaces", "materialIds"):
        np.testing.assert_array_equal(after["geometry"][name], before["geometry"][name])
    for name, values in before["provenanceMaps"].items():
        np.testing.assert_array_equal(after["provenanceMaps"][name], values)
    for beforeMaterial, afterMaterial in zip(before["materials"], after["materials"]):
        np.testing.assert_array_equal(beforeMaterial["baseColor"], afterMaterial["baseColor"])
