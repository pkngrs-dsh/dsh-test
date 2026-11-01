"""Independent CPU contracts for preloaded weighted surface deformation."""

import numpy as np
import pytest

import faceAppearance.barycentricDeformation as deformationModule
from faceAppearance.barycentricDeformation import BarycentricDeformer


@pytest.fixture
def inputs():
    return {"restVertices": np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]),
            "faces": np.array([[0, 1, 2]], np.int32),
            "sampleFaceIds": np.zeros(3, np.int64),
            "sampleBarycentrics": np.array([[.5, .25, .25], [.25, .5, .25], [.25, .25, .5]]),
            "sampleWeights": np.ones(3), "staticMask": np.zeros(3, bool)}


def testNonVertexBarycentricTargetsRecoverKnownVertexDisplacements(inputs):
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    expected = inputs["restVertices"]+np.array([[.1, -.2, 1.], [.2, .3, 2.], [-.1, .4, 3.]])
    targets = inputs["sampleBarycentrics"] @ expected
    actual, diagnostics = deformer.deform(targets)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(deformer.restSamplePoints, inputs["sampleBarycentrics"] @ inputs["restVertices"], atol=1e-15)
    assert actual.dtype == np.float64 and actual.shape == inputs["restVertices"].shape
    assert diagnostics["sampleResidualMedianBeforeCm"] > 1.
    assert diagnostics["sampleResidualP95AfterCm"] < 1e-12
    assert diagnostics["geometryGuardApplied"] is False and diagnostics["lengthUnit"] == "centimeter"


def testStaticVerticesAreExactHardConstraintsWithNonzeroTargets(inputs):
    inputs["staticMask"][0] = True
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    expected = inputs["restVertices"].copy()
    expected[1:] += [[.3, .2, 1.], [-.2, .1, 2.]]
    actual, diagnostics = deformer.deform(inputs["sampleBarycentrics"] @ expected)
    np.testing.assert_array_equal(actual[0], inputs["restVertices"][0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert diagnostics["staticVertexCount"] == 1 and diagnostics["freeVertexCount"] == 2


def testSampleWeightsEnterObjectiveOnceRatherThanBeingSquared(inputs):
    inputs["sampleFaceIds"] = np.array([0])
    inputs["sampleBarycentrics"] = np.array([[.5, .25, .25]])
    inputs["sampleWeights"] = np.array([4.])
    inputs["staticMask"] = np.array([False, True, True])
    deformer = BarycentricDeformer(**inputs, dataWeight=2., smoothWeight=0., restWeight=3.)
    target = deformer.restSamplePoints+[[0., 0., 1.]]
    actual, diagnostics = deformer.deform(target)
    # One scalar unknown: delta = (data * weight * C * residual)/(data * weight * C² + prior).
    assert actual[0, 2] == pytest.approx(4./5.)
    np.testing.assert_array_equal(actual[1:], inputs["restVertices"][1:])
    assert diagnostics["sampleResidualMedianBeforeCm"] == pytest.approx(1.)
    assert diagnostics["sampleResidualMedianAfterCm"] == pytest.approx(.6)


def testSmoothingSpreadsDisplacementAndRestPriorReducesUnobservedMotion():
    options = {"restVertices": np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
               "faces": np.array([[0, 1, 2], [0, 2, 3]]), "sampleFaceIds": np.array([0]),
               "sampleBarycentrics": np.array([[1., 0., 0.]]), "sampleWeights": np.array([1.]),
               "staticMask": np.zeros(4, bool)}
    target = np.array([[0., 0., 1.]])
    unsmoothed, _ = BarycentricDeformer(**options, smoothWeight=0., restWeight=.03).deform(target)
    smoothed, _ = BarycentricDeformer(**options, smoothWeight=10., restWeight=.03).deform(target)
    stronglyAnchored, _ = BarycentricDeformer(**options, smoothWeight=10., restWeight=20.).deform(target)
    np.testing.assert_array_equal(unsmoothed[1:, 2], 0.)
    assert (smoothed[1:, 2] > 0).all()
    assert np.ptp(smoothed[:, 2]) < np.ptp(unsmoothed[:, 2])
    assert np.linalg.norm(stronglyAnchored[:, 2]) < np.linalg.norm(smoothed[:, 2])


def testZeroWeightSamplesHaveNoEffectAndAreExcludedFromDiagnostics(inputs):
    withIgnored = {**inputs, "sampleFaceIds": np.r_[inputs["sampleFaceIds"], 0],
                   "sampleBarycentrics": np.vstack((inputs["sampleBarycentrics"], [.2, .3, .5])),
                   "sampleWeights": np.r_[inputs["sampleWeights"], 0.]}
    reference = BarycentricDeformer(**inputs)
    augmented = BarycentricDeformer(**withIgnored)
    targets = reference.restSamplePoints+np.array([[0., 0., 1.], [0., 0., .5], [0., 0., .25]])
    expected, referenceDiagnostics = reference.deform(targets)
    actual, diagnostics = augmented.deform(np.vstack((targets, [1e100, -1e100, 1e100])))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert diagnostics["sampleCount"] == 4 and diagnostics["positiveSampleCount"] == 3
    for name in ("sampleResidualMedianBeforeCm", "sampleResidualP95BeforeCm",
                 "sampleResidualMedianAfterCm", "sampleResidualP95AfterCm"):
        assert diagnostics[name] == pytest.approx(referenceDiagnostics[name], abs=1e-14)


def testZeroWeightSamplesCanBeOnStaticVertices(inputs):
    inputs["staticMask"][0] = True
    inputs["sampleBarycentrics"][0] = [1., 0., 0.]
    inputs["sampleWeights"][0] = 0.
    deformer = BarycentricDeformer(**inputs)
    result, diagnostics = deformer.deform(deformer.restSamplePoints+[[99., 99., 99.], [0., 0., 1.], [0., 0., 1.]])
    np.testing.assert_array_equal(result[0], inputs["restVertices"][0])
    assert diagnostics["positiveSampleCount"] == 2
    assert diagnostics["sampleResidualMedianBeforeCm"] == pytest.approx(1.)


def testAllStaticReturnsFreshRestWithoutFactorizationButHonestResidual(inputs, monkeypatch):
    inputs["staticMask"][:] = True
    monkeypatch.setattr(deformationModule, "factorized", lambda *args: pytest.fail("All-static system needs no factorization"))
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    result, diagnostics = deformer.deform(deformer.restSamplePoints+[[0., 0., 2.]])
    np.testing.assert_array_equal(result, inputs["restVertices"])
    assert diagnostics["allVerticesStatic"] and diagnostics["freeVertexCount"] == 0
    assert diagnostics["sampleResidualMedianBeforeCm"] == diagnostics["sampleResidualMedianAfterCm"] == 2.
    assert diagnostics["maxDisplacementCm"] == 0.
    result[:] = 999.
    np.testing.assert_array_equal(deformer.restVertices, inputs["restVertices"])


def testFactorizationIsPreloadedOnceAndRepeatedSolvesAreIsolated(inputs, monkeypatch):
    calls = []
    originalFactorized = deformationModule.factorized

    def recordFactorization(matrix):
        calls.append(matrix.shape)
        return originalFactorized(matrix)

    monkeypatch.setattr(deformationModule, "factorized", recordFactorization)
    original = {name: value.copy() for name, value in inputs.items()}
    deformer = BarycentricDeformer(**inputs)
    assert calls == [(3, 3)]
    target = deformer.restSamplePoints+[[0., 0., 1.]]
    targetBefore = target.copy()
    first, firstDiagnostics = deformer.deform(target)
    expected = first.copy()
    for name, values in inputs.items():
        np.testing.assert_array_equal(values, original[name])
        if values.dtype.kind == "b":
            values[:] = True
        else:
            values[:] = 999
    first[:] = -999.
    firstDiagnostics["sampleResidualMedianAfterCm"] = -1.
    deformer.deform(deformer.restSamplePoints-[[0., 0., .5]])
    repeated, repeatedDiagnostics = deformer.deform(target)
    np.testing.assert_array_equal(repeated, expected)
    np.testing.assert_array_equal(target, targetBefore)
    assert calls == [(3, 3)] and repeatedDiagnostics["sampleResidualMedianAfterCm"] >= 0
    for name, values in original.items():
        np.testing.assert_array_equal(getattr(deformer, name), values)
        assert not getattr(deformer, name).flags.writeable
    for matrix in (deformer.sampleMatrix, deformer.laplacian, deformer.freeOperator):
        assert not matrix.data.flags.writeable and not matrix.indices.flags.writeable and not matrix.indptr.flags.writeable


def testUnconstrainedFreeVerticesWithoutPriorFailAtPreload(inputs):
    inputs["sampleFaceIds"] = np.array([0])
    inputs["sampleBarycentrics"] = np.array([[1., 0., 0.]])
    inputs["sampleWeights"] = np.array([1.])
    with pytest.raises(ValueError, match="singular"):
        BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)


@pytest.mark.parametrize("name,value", [
    ("restVertices", np.ones((3, 3), int)), ("restVertices", np.full((3, 3), np.nan)),
    ("restVertices", np.zeros((3, 2))), ("faces", np.array([[0., 1., 2.]])),
    ("faces", np.array([[0, 1, 3]])), ("faces", np.array([[0, 0, 2]])),
    ("faces", np.array([[-1, 1, 2]])), ("sampleFaceIds", np.array([1, 0, 0])),
    ("sampleFaceIds", np.array([], np.int64)), ("sampleFaceIds", np.array([0., 0., 0.])),
    ("sampleBarycentrics", np.full((3, 3), .5)),
    ("sampleBarycentrics", np.array([[1., -.1, .1], [1., 0., 0.], [0., 1., 0.]])),
    ("sampleBarycentrics", np.full((3, 3), np.nan)),
    ("sampleWeights", np.zeros(3)), ("sampleWeights", np.array([1., -1., 1.])),
    ("sampleWeights", np.ones(3, int)), ("sampleWeights", np.array([1., np.inf, 1.])),
    ("staticMask", np.zeros(3, int)), ("staticMask", np.zeros(2, bool)),
])
def testInvalidPreloadInputsAreRejected(inputs, name, value):
    inputs[name] = value
    with pytest.raises(ValueError):
        BarycentricDeformer(**inputs)


@pytest.mark.parametrize("name,value", [("dataWeight", 0.), ("dataWeight", True), ("dataWeight", np.inf),
                                        ("smoothWeight", -1.), ("smoothWeight", np.nan),
                                        ("restWeight", -1.), ("restWeight", "1")])
def testInvalidObjectiveParametersAreRejected(inputs, name, value):
    with pytest.raises(ValueError, match=name):
        BarycentricDeformer(**inputs, **{name: value})


@pytest.mark.parametrize("targets", [np.zeros((2, 3)), np.zeros((3, 3), int), np.full((3, 3), np.nan)])
def testInvalidTargetsAreRejectedWithoutPollutingNextSolve(inputs, targets):
    deformer = BarycentricDeformer(**inputs)
    with pytest.raises(ValueError, match="targetPoints"):
        deformer.deform(targets)
    result, _ = deformer.deform(deformer.restSamplePoints)
    np.testing.assert_array_equal(result, inputs["restVertices"])


def testNoSupportIsStillRejectedForAllStaticMesh(inputs):
    inputs["staticMask"][:] = True
    inputs["sampleWeights"][:] = 0.
    with pytest.raises(ValueError, match="positive observation"):
        BarycentricDeformer(**inputs)


def testTwoChannelTransferRecoversNonVertexUvChangesUsingExistingFactorization(inputs, monkeypatch):
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    solver = deformer._solve
    monkeypatch.setattr(deformationModule, "factorized", lambda *args: pytest.fail("Attribute transfer must reuse the existing solver"))
    restUvs = np.array([[.2, .3], [.8, .1], [.1, .9]])
    expected = restUvs+np.array([[.03, -.02], [-.01, .04], [.02, -.03]])
    transferred = deformer.transferValues(restUvs, inputs["sampleBarycentrics"] @ expected)
    np.testing.assert_allclose(transferred, expected, atol=1e-14)
    assert transferred.shape == (3, 2) and transferred.dtype == np.float64
    assert deformer._solve is solver
    # The same solve also applies to a zero-rest correction field, without a UV range clamp.
    expectedDelta = np.array([[.02, -.05], [-.03, .04], [.01, -.02]])
    delta = deformer.transferValues(np.zeros((3, 2)), inputs["sampleBarycentrics"] @ expectedDelta)
    np.testing.assert_allclose(delta, expectedDelta, atol=1e-14)
    assert (delta < 0).any()


@pytest.mark.parametrize("channelCount", [1, 4, 7])
def testAttributeChannelCountIsIndependentOfGeometryDimensions(inputs, channelCount):
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    rest = np.arange(3*channelCount, dtype=np.float64).reshape(3, channelCount)
    expected = rest+np.linspace(-.5, .5, 3*channelCount).reshape(3, channelCount)
    actual = deformer.transferValues(rest, inputs["sampleBarycentrics"] @ expected)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert actual.shape == (3, channelCount)


def testTransferPreservesStaticAttributeValuesRatherThanGeometryCoordinates(inputs):
    inputs["staticMask"][0] = True
    deformer = BarycentricDeformer(**inputs, smoothWeight=0., restWeight=0.)
    rest = np.array([[.2, .7], [.5, .4], [.9, .3]])
    expected = rest.copy()
    expected[1:] += [[.1, -.2], [-.15, .05]]
    result = deformer.transferValues(rest, inputs["sampleBarycentrics"] @ expected)
    np.testing.assert_array_equal(result[0], rest[0])
    np.testing.assert_allclose(result, expected, atol=1e-14)


def testAttributeTransferIgnoresZeroWeightTargets(inputs):
    inputs["sampleWeights"][0] = 0.
    deformer = BarycentricDeformer(**inputs)
    rest = np.array([[.1, .2], [.3, .4], [.5, .6]])
    target = inputs["sampleBarycentrics"] @ rest+[[.2, -.1]]
    baseline = deformer.transferValues(rest, target)
    target[0] = [-1e100, 1e100]
    np.testing.assert_array_equal(deformer.transferValues(rest, target), baseline)


def testAttributeTransferAcceptsReadonlyInputsAndIsolatesRepeatedReturns(inputs):
    deformer = BarycentricDeformer(**inputs)
    rest = np.array([[.1, .7], [.2, .9], [.3, .8]], np.float32)
    target = (inputs["sampleBarycentrics"] @ rest+[[.01, -.02]]).astype(np.float32)
    restBefore, targetBefore = rest.copy(), target.copy()
    rest.setflags(write=False)
    target.setflags(write=False)
    first = deformer.transferValues(rest, target)
    expected = first.copy()
    first[:] = 999.
    otherRest = np.zeros((3, 4))
    deformer.transferValues(otherRest, np.ones((3, 4)))
    np.testing.assert_array_equal(deformer.transferValues(rest, target), expected)
    np.testing.assert_array_equal(rest, restBefore)
    np.testing.assert_array_equal(target, targetBefore)
    np.testing.assert_array_equal(deformer.restVertices, inputs["restVertices"])
    # Calling generic transfer does not change the existing geometry API or its diagnostics.
    vertices, diagnostics = deformer.deform(deformer.restSamplePoints)
    np.testing.assert_array_equal(vertices, inputs["restVertices"])
    assert diagnostics["sampleResidualMedianAfterCm"] == 0.


def testAllStaticAttributeTransferReturnsIndependentOriginalValues(inputs):
    inputs["staticMask"][:] = True
    deformer = BarycentricDeformer(**inputs)
    rest = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = deformer.transferValues(rest, np.full((3, 2), 999.))
    np.testing.assert_array_equal(result, rest)
    assert result.dtype == np.float64
    result[:] = 0.
    np.testing.assert_array_equal(rest, np.arange(6).reshape(3, 2))


@pytest.mark.parametrize("rest,target", [
    (np.zeros((2, 2)), np.zeros((3, 2))), (np.zeros((3, 0)), np.zeros((3, 0))),
    (np.zeros(3), np.zeros((3, 1))), (np.zeros((3, 2, 1)), np.zeros((3, 2))),
    (np.zeros((3, 2)), np.zeros((2, 2))), (np.zeros((3, 2)), np.zeros((3, 3))),
    (np.zeros((3, 2), int), np.zeros((3, 2))), (np.zeros((3, 2)), np.zeros((3, 2), int)),
    (np.full((3, 2), np.nan), np.zeros((3, 2))), (np.zeros((3, 2)), np.full((3, 2), np.inf)),
])
def testInvalidAttributeShapesDtypesAndFiniteValuesAreRejected(inputs, rest, target):
    deformer = BarycentricDeformer(**inputs)
    with pytest.raises(ValueError, match="restValues|targetValues"):
        deformer.transferValues(rest, target)
    np.testing.assert_array_equal(deformer.transferValues(np.zeros((3, 2)), np.zeros((3, 2))), 0.)
