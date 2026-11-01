"""CPU integration of preloaded legacy barycentric constraints into assembly."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import faceAppearance.barycentricDeformation as deformationModule
import faceAppearance.headAssembly as assemblyModule
from faceAppearance.headAssembly import AssemblyAssets
from testHeadAssembly import bundle, observedTexture, reconstructedGeometry


def writeManifest(root, manifest):
    (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def saveConstraints(root, constraints):
    np.savez_compressed(root/"legacyBarycentric.npz", **constraints)
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"]["legacyBarycentric.npz"] = hashlib.sha256((root/"legacyBarycentric.npz").read_bytes()).hexdigest()
    writeManifest(root, manifest)


@pytest.fixture
def legacyBundle(bundle, tmp_path):
    baselineRoot, arrays = bundle
    candidateRoot = tmp_path/"legacyCandidate"
    shutil.copytree(baselineRoot, candidateRoot)
    faceIds = np.repeat(np.arange(len(arrays["nicpFaces"])), 2)
    barycentrics = np.tile([[.6, .25, .15], [.2, .25, .55]], (len(arrays["nicpFaces"]), 1))
    weights = np.linspace(.4, 1., len(faceIds))
    weights[0] = 0.
    constraints = {"sampleNicpFaceIds": faceIds.astype(np.int64),
                   "sampleNicpWeights": barycentrics.astype(np.float64), "sampleWeights": weights,
                   "sourceTriangleIds": faceIds.astype(np.int64),
                   "sourceBarycentricWeights": barycentrics.astype(np.float64)}
    manifest = json.loads((candidateRoot/"manifest.json").read_text())
    manifest["geometryTransfer"] = {"method": "legacyBarycentric", "asset": "legacyBarycentric.npz",
                                    "alignmentMode": "sharedUvVertices", "dataWeight": 20.,
                                    "smoothWeight": 1., "restWeight": .03}
    writeManifest(candidateRoot, manifest)
    saveConstraints(candidateRoot, constraints)
    return baselineRoot, candidateRoot, arrays, constraints


def testLegacyAssetAndSparseSolveAreLoadedExactlyOnce(legacyBundle, monkeypatch):
    _, root, arrays, constraints = legacyBundle
    loads, factorizations = [], []
    originalLoad, originalFactorized = np.load, deformationModule.factorized

    def recordLoad(path, *args, **kwargs):
        loads.append(Path(path).name)
        return originalLoad(path, *args, **kwargs)

    def recordFactorization(matrix):
        factorizations.append(matrix.shape)
        return originalFactorized(matrix)

    monkeypatch.setattr(np, "load", recordLoad)
    monkeypatch.setattr(deformationModule, "factorized", recordFactorization)
    assets = AssemblyAssets(root)
    preloaded = assets.barycentricDeformer
    assert loads.count("legacyBarycentric.npz") == 1 and len(factorizations) == 1
    # Runtime must work from verified preloaded memory, without opening the asset again.
    (root/"legacyBarycentric.npz").unlink()
    for _ in range(2):
        assets.assemble(reconstructedGeometry(arrays), observedTexture())
    assert assets.barycentricDeformer is preloaded
    assert loads.count("legacyBarycentric.npz") == 1 and len(factorizations) == 1
    for name, expected in constraints.items():
        np.testing.assert_array_equal(assets.barycentricArrays[name], expected)
        assert not assets.barycentricArrays[name].flags.writeable


def testCandidateSharesBaselineAlignmentAndPreservesOriginalBindingWithStaticPins(legacyBundle):
    baselineRoot, candidateRoot, arrays, _ = legacyBundle
    geometry, texture = reconstructedGeometry(arrays), observedTexture()
    baseline = AssemblyAssets(baselineRoot).assemble(geometry, texture)
    candidate = AssemblyAssets(candidateRoot).assemble(geometry, texture)
    for name in ("rotation", "translation", "scale"):
        np.testing.assert_array_equal(candidate["diagnostics"]["templateToUff"][name],
                                      baseline["diagnostics"]["templateToUff"][name])
    for name in ("faces", "uvs", "uvFaces", "materialIds"):
        np.testing.assert_array_equal(candidate["geometry"][name], arrays[name])
        assert candidate["geometry"][name].dtype == arrays[name].dtype
    np.testing.assert_array_equal(candidate["templateVertices"], arrays["vertices"])
    staticIds = arrays["nicpToFullIds"][arrays["staticMask"]]
    np.testing.assert_array_equal(candidate["geometry"]["vertices"][staticIds], arrays["vertices"][staticIds])
    assert candidate["geometry"]["vertices"].shape == arrays["vertices"].shape
    assert candidate["geometry"]["uvs"].shape[0] != len(candidate["geometry"]["vertices"])
    assert np.linalg.norm(candidate["geometry"]["vertices"]-baseline["geometry"]["vertices"]) > 1e-5
    transfer = candidate["diagnostics"]["geometryTransfer"]
    assert transfer["method"] == "legacyBarycentric" and transfer["alignmentMode"] == "sharedUvVertices"
    assert transfer["solve"]["positiveSampleCount"] == 15
    assert candidate["diagnostics"]["topologySafety"]["invertedFaceCount"] == 0


def testLegacyBranchKeepsEveryTextureAlphaAndCoverageMapIdenticalToBaseline(legacyBundle):
    baselineRoot, candidateRoot, arrays, _ = legacyBundle
    geometry, texture = reconstructedGeometry(arrays), observedTexture(0.)
    texture["coverageMask"][6:10, 6:10] = False
    baseline = AssemblyAssets(baselineRoot).assemble(geometry, texture)
    candidate = AssemblyAssets(candidateRoot).assemble(geometry, texture)
    assert len(candidate["materials"]) == len(baseline["materials"])
    for actual, expected in zip(candidate["materials"], baseline["materials"]):
        np.testing.assert_array_equal(actual["baseColor"], expected["baseColor"])
        assert {key: value for key, value in actual.items() if key != "baseColor"} == {
            key: value for key, value in expected.items() if key != "baseColor"}
    for name, expected in baseline["provenanceMaps"].items():
        np.testing.assert_array_equal(candidate["provenanceMaps"][name], expected)
    for name, expected in baseline["diagnostics"]["texture"].items():
        actual = candidate["diagnostics"]["texture"][name]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def testGuardedSampleResidualIsRecomputedFromAcceptedVertices(legacyBundle, monkeypatch):
    _, root, arrays, constraints = legacyBundle
    originalGuard = assemblyModule.guardDisplacement

    def partialGuard(rest, displacement, faces):
        vertices, retention, quality = originalGuard(rest, displacement*.25, faces)
        return vertices, retention*.25, quality

    monkeypatch.setattr(assemblyModule, "guardDisplacement", partialGuard)
    geometry = reconstructedGeometry(arrays)
    result = AssemblyAssets(root).assemble(geometry, observedTexture())
    transform = result["diagnostics"]["templateToUff"]
    sourceCorners = geometry["vertices"][arrays["sourceFaces"][constraints["sourceTriangleIds"]]]
    sourceTargets = (sourceCorners*constraints["sourceBarycentricWeights"][..., None]).sum(axis=1)
    targetsCm = (sourceTargets-transform["translation"]) @ transform["rotation"].T/transform["scale"]
    acceptedSkin = result["geometry"]["vertices"][arrays["nicpToFullIds"]]
    targetCorners = acceptedSkin[arrays["nicpFaces"][constraints["sampleNicpFaceIds"]]]
    acceptedSamples = (targetCorners*constraints["sampleNicpWeights"][..., None]).sum(axis=1)
    errors = np.linalg.norm(acceptedSamples-targetsCm, axis=1)[constraints["sampleWeights"] > 0]
    diagnostics = result["diagnostics"]["geometryTransfer"]
    assert result["diagnostics"]["deformationRetentionMean"] <= .25
    assert diagnostics["guardedSampleResidualMedianCm"] == pytest.approx(np.median(errors), abs=1e-14)
    assert diagnostics["guardedSampleResidualP95Cm"] == pytest.approx(np.percentile(errors, 95), abs=1e-14)
    assert abs(diagnostics["guardedSampleResidualMedianCm"]-
               diagnostics["solve"]["sampleResidualMedianAfterCm"]) > 1e-4


def testLegacyRepeatedCallsCannotBePollutedByReturnedData(legacyBundle):
    _, root, arrays, constraints = legacyBundle
    assets = AssemblyAssets(root)
    geometry, texture = reconstructedGeometry(arrays), observedTexture()
    first = assets.assemble(geometry, texture)
    expectedVertices = first["geometry"]["vertices"].copy()
    expectedColors = first["materials"][0]["baseColor"].copy()
    expectedResidual = first["diagnostics"]["geometryTransfer"]["guardedSampleResidualMedianCm"]
    first["geometry"]["vertices"][:] = 999.
    first["templateVertices"][:] = 999.
    first["materials"][0]["baseColor"][:] = 0.
    first["components"][1]["vertexIndices"][0] = 999
    first["templateHashes"]["fullMesh"] = "b"*64
    first["diagnostics"]["geometryTransfer"]["solve"]["sampleResidualMedianAfterCm"] = -1.
    second = assets.assemble(geometry, texture)
    np.testing.assert_array_equal(second["geometry"]["vertices"], expectedVertices)
    np.testing.assert_array_equal(second["materials"][0]["baseColor"], expectedColors)
    assert second["diagnostics"]["geometryTransfer"]["guardedSampleResidualMedianCm"] == expectedResidual
    assert second["components"][1]["vertexIndices"][0] == 9
    assert second["templateHashes"]["fullMesh"] == "a"*64
    for name, expected in constraints.items():
        np.testing.assert_array_equal(assets.barycentricArrays[name], expected)


@pytest.mark.parametrize("failure", ["unhashed", "missing", "changed"])
def testLegacyAssetMustExistAndMatchItsManifestHash(legacyBundle, failure):
    _, root, _, _ = legacyBundle
    if failure == "unhashed":
        manifest = json.loads((root/"manifest.json").read_text())
        del manifest["files"]["legacyBarycentric.npz"]
        writeManifest(root, manifest)
    elif failure == "missing":
        (root/"legacyBarycentric.npz").unlink()
    else:
        with (root/"legacyBarycentric.npz").open("ab") as output:
            output.write(b"changed")
    with pytest.raises(ValueError, match="manifest|asset path|hash mismatch"):
        AssemblyAssets(root)


@pytest.mark.parametrize("name,value", [("method", "inventedMethod"), ("alignmentMode", "separateLegacySimilarity"),
                                        ("asset", None)])
def testUnsupportedGeometryTransferConfigurationFails(legacyBundle, name, value):
    _, root, _, _ = legacyBundle
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["geometryTransfer"][name] = value
    writeManifest(root, manifest)
    with pytest.raises(ValueError, match="Unsupported|covered"):
        AssemblyAssets(root)


@pytest.mark.parametrize("name,value", [
    ("sourceTriangleIds", np.full(16, 8, np.int64)),
    ("sourceTriangleIds", np.full(16, -1, np.int64)),
    ("sourceTriangleIds", np.zeros(16, np.float64)),
    ("sourceTriangleIds", np.zeros(15, np.int64)),
    ("sourceBarycentricWeights", np.full((16, 3), .5)),
    ("sourceBarycentricWeights", np.full((16, 3), np.nan)),
    ("sampleNicpFaceIds", np.full(16, 8, np.int64)),
    ("sampleNicpWeights", np.full((16, 3), -.1)),
    ("sampleWeights", np.zeros(16)),
])
def testInvalidLegacySourceOrTargetSamplingFailsAtPreload(legacyBundle, name, value):
    _, root, _, constraints = legacyBundle
    constraints[name] = value
    saveConstraints(root, constraints)
    with pytest.raises(ValueError):
        AssemblyAssets(root)


def testMissingLegacyArrayIsRejectedAtPreload(legacyBundle):
    _, root, _, constraints = legacyBundle
    del constraints["sourceBarycentricWeights"]
    saveConstraints(root, constraints)
    with pytest.raises(ValueError, match="Incomplete"):
        AssemblyAssets(root)


@pytest.mark.parametrize("changed", ["faces", "uvs", "vertices", "uvFaces"])
def testLegacyBranchStillRejectsIncompatibleReconstructedGeometry(legacyBundle, changed):
    _, root, arrays, _ = legacyBundle
    geometry = reconstructedGeometry(arrays)
    if changed == "faces":
        geometry["faces"] = geometry["faces"][::-1].copy()
    elif changed == "uvs":
        geometry["uvs"][0, 0] += .01
    elif changed == "uvFaces":
        geometry["uvFaces"] = geometry["faces"][::-1].copy()
    else:
        geometry["vertices"][0, 0] = np.nan
    with pytest.raises(ValueError, match="identity|native UV"):
        AssemblyAssets(root).assemble(geometry, observedTexture())
