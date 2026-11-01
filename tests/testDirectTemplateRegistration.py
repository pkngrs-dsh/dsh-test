"""CPU numerical checks of the old masked registration objective and frame."""

import numpy as np
import pytest

from faceAppearance.directTemplateRegistration import registerDirectTemplate, _laplacian


@pytest.fixture
def inputs():
    rest = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.], [0., 0., 1.4]])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]], np.int64)
    faceIds = np.array([0, 0, 1, 2, 0, 1, 3, 4], np.int64)
    bary = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.],
                     [.2, .3, .5], [.4, .2, .4], [.2, .6, .2]])
    samples = (rest[faces[faceIds]] * bary[..., None]).sum(axis=1)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = samples @ rotation * 1.7 + [4., -3., 2.]
    target[4] += [.17, -.23, .31]
    target[6] += [-.05, .07, .08]
    return {"restVertices": rest, "faces": faces, "sampleFaceIds": faceIds,
            "sampleBarycentrics": bary, "targetPoints": target, "validMask": np.ones(len(faceIds), bool),
            "staticVertexIds": np.array([0, 3], np.int64)}


def independentMatrices(data):
    n = len(data["restVertices"])
    faces = data["faces"]
    C = np.zeros((int(data["validMask"].sum()), n))
    for row, sample in enumerate(np.flatnonzero(data["validMask"])):
        C[row, faces[data["sampleFaceIds"][sample]]] = data["sampleBarycentrics"][sample]
    adjacency = np.zeros((n, n))
    for triangle in faces:
        for i in triangle:
            for j in triangle:
                if i != j:
                    adjacency[i, j] = 1
    L = np.diag(adjacency.sum(axis=1)) - adjacency
    S = np.eye(n)[data["staticVertexIds"]]
    return C, S, L


def independentLoss(vertices, aligned, data, dataWeight, smoothWeight, restWeight):
    C, _, L = independentMatrices(data)
    observation = C @ vertices - data["targetPoints"][data["validMask"]]
    anchor = vertices[data["staticVertexIds"]] - aligned[data["staticVertexIds"]]
    return {"data": dataWeight * np.mean(np.concatenate([observation, anchor])**2),
            "smooth": smoothWeight * np.mean((L @ vertices)**2),
            "rest": restWeight * np.mean((vertices - aligned)**2)}


@pytest.mark.parametrize("restWeight", [1000., 100., 0.])
def testSparseSolutionMatchesIndependentDenseNormalEquationsAndObjective(inputs, restWeight):
    result = registerDirectTemplate(**inputs, restWeight=restWeight)
    aligned, vertices = result["alignedRest"], result["rawVertices"]
    C, S, L = independentMatrices(inputs)
    A = np.vstack([C, S])
    Q = np.concatenate([inputs["targetPoints"], aligned[inputs["staticVertexIds"]]])
    n, k = len(vertices), len(Q)
    H = 10000. / (3*k) * A.T @ A + 3000. / (3*n) * L.T @ L + restWeight / (3*n) * np.eye(n)
    rhs = 10000. / (3*k) * A.T @ Q + restWeight / (3*n) * aligned
    np.testing.assert_allclose(vertices, np.linalg.solve(H, rhs), rtol=2e-13, atol=2e-13)
    loss = independentLoss(vertices, aligned, inputs, 10000., 3000., restWeight)
    report = result["diagnostics"]
    for name, value in loss.items():
        assert report["objectiveAfter"][name] == pytest.approx(value, rel=1e-13)
    assert report["objectiveAfter"]["total"] == pytest.approx(sum(loss.values()), rel=1e-13)
    assert report["objectiveAfter"]["total"] <= report["objectiveBefore"]["total"]
    assert report["mseDenominators"] == {"jointData": 30, "smooth": 15, "rest": 15}
    assert report["normalEquationRelativeResidual"] < 1e-12
    assert report["normalEquationGradientMaxAbs"] < 1e-9
    assert report["guardApplied"] is False and report["legacyFiniteStepTrajectoryMatched"] is False
    assert report["solver"] == "cpuFloat64SparseQuadraticOptimum"


def testLaplacianIsUnnormalizedDegreeMinusUniqueAdjacency(inputs):
    expected = independentMatrices(inputs)[2]
    laplacian, _ = _laplacian(len(inputs["restVertices"]), inputs["faces"])
    np.testing.assert_array_equal(laplacian.toarray(), expected)
    np.testing.assert_array_equal(laplacian.diagonal(), [4., 3., 4., 3., 4.])
    np.testing.assert_array_equal(laplacian @ np.ones((5, 3)), np.zeros((5, 3)))
    # Repeated faces do not increase edge weights in upstream Large Steps.
    repeated, _ = _laplacian(5, np.tile(inputs["faces"], (2, 1)))
    np.testing.assert_array_equal(repeated.toarray(), expected)


def testUmeyamaUsesClientSurfaceSamplesAndReturnsTargetFrame(inputs):
    C, _, _ = independentMatrices(inputs)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    scale, translation = 2.8, np.array([7.3, -3.1, .28])
    inputs["targetPoints"] = C @ inputs["restVertices"] @ rotation * scale + translation
    result = registerDirectTemplate(**inputs, smoothWeight=0., restWeight=0.)
    transform = result["transform"]
    assert transform["scale"] == pytest.approx(scale, abs=1e-13)
    np.testing.assert_allclose(transform["rotation"], rotation, atol=1e-13)
    np.testing.assert_allclose(transform["translation"], translation, atol=1e-13)
    expected = inputs["restVertices"] @ rotation * scale + translation
    np.testing.assert_allclose(result["alignedRest"], expected, atol=1e-13)
    np.testing.assert_allclose(result["rawVertices"], expected, atol=1e-13)
    inverse = ((result["rawVertices"] - translation) @ rotation.T) / scale
    np.testing.assert_allclose(inverse, inputs["restVertices"], atol=1e-13)


def testAbsoluteLaplacianAndSoftStaticAnchorsAreNotCurrentDisplacementSolver(inputs):
    C, _, _ = independentMatrices(inputs)
    inputs["targetPoints"] = C @ inputs["restVertices"]
    result = registerDirectTemplate(**inputs)
    static = inputs["staticVertexIds"]
    # Even observations identical to rest are opposed by smoothing L@V: rest
    # is curved. L@(V-rest) would instead have zero initial energy and no move.
    assert np.linalg.norm(result["rawVertices"] - result["alignedRest"]) > .1
    assert np.linalg.norm(result["rawVertices"][static] - result["alignedRest"][static]) > .01
    assert result["diagnostics"]["objectiveBefore"]["smooth"] > 0
    assert result["diagnostics"]["objectiveAfter"]["dataStaticAnchorPart"] > 0


def testRestPriorPresetsHaveExpectedOrdering(inputs):
    results = [registerDirectTemplate(**inputs, restWeight=weight) for weight in (0., 100., 1000.)]
    movement = [np.linalg.norm(item["rawVertices"] - item["alignedRest"]) for item in results]
    assert movement[0] > movement[1] > movement[2]


@pytest.mark.parametrize("restWeight", [1000., 100., 0.])
def testHardBoundaryUsesCorrectEliminationAndPreservesOldDataDenominator(inputs, restWeight):
    result = registerDirectTemplate(**inputs, restWeight=restWeight, staticConstraint="hard")
    vertices, aligned = result["rawVertices"], result["alignedRest"]
    static = inputs["staticVertexIds"]
    free = np.setdiff1d(np.arange(len(vertices)), static)
    C, S, L = independentMatrices(inputs)
    A = np.vstack([C, S])
    Q = np.concatenate([inputs["targetPoints"], aligned[static]])
    H = 10000. / (3*len(Q)) * A.T @ A + 3000. / (3*len(vertices)) * L.T @ L
    H += restWeight / (3*len(vertices)) * np.eye(len(vertices))
    rhs = 10000. / (3*len(Q)) * A.T @ Q + restWeight / (3*len(vertices)) * aligned
    expected = np.linalg.solve(H[np.ix_(free, free)], rhs[free] - H[np.ix_(free, static)] @ aligned[static])
    np.testing.assert_array_equal(vertices[static], aligned[static])
    np.testing.assert_allclose(vertices[free], expected, atol=1e-13)
    assert result["diagnostics"]["mseDenominators"]["jointData"] == 30
    assert result["diagnostics"]["objectiveAfter"]["dataStaticAnchorPart"] == 0
    assert result["diagnostics"]["fixedVertexCount"] == 2
    assert result["diagnostics"]["normalEquationResidualDomain"] == "freeVerticesOnly"


def testHardBoundaryCanFixEveryVertex(inputs):
    inputs["staticVertexIds"] = np.arange(len(inputs["restVertices"]), dtype=np.int64)
    result = registerDirectTemplate(**inputs, restWeight=0., staticConstraint="hard")
    np.testing.assert_array_equal(result["rawVertices"], result["alignedRest"])
    assert result["diagnostics"]["freeVertexCount"] == 0
    assert result["diagnostics"]["normalEquationGradientMaxAbs"] == 0


def testMaskedObservationsAndLegacyExtrapolationDoNotAffectFit(inputs):
    expected = registerDirectTemplate(**inputs)
    inputs["sampleFaceIds"] = np.r_[inputs["sampleFaceIds"], 0]
    inputs["sampleBarycentrics"] = np.vstack([inputs["sampleBarycentrics"], [-1., 2., 0.]])
    inputs["targetPoints"] = np.vstack([inputs["targetPoints"], [1e6, -1e6, 2e6]])
    inputs["validMask"] = np.r_[inputs["validMask"], False]
    actual = registerDirectTemplate(**inputs)
    np.testing.assert_array_equal(actual["rawVertices"], expected["rawVertices"])
    np.testing.assert_array_equal(actual["alignedRest"], expected["alignedRest"])
    assert actual["diagnostics"]["sampleCount"] == 9
    assert actual["diagnostics"]["validSampleCount"] == 8


def testTargetSimilarityChangesOnlyOutputFrameAndObjectiveUnits(inputs):
    first = registerDirectTemplate(**inputs, restWeight=100.)
    r = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    s, t = 3., np.array([37., -85., .2])
    inputs["targetPoints"] = inputs["targetPoints"] @ r * s + t
    second = registerDirectTemplate(**inputs, restWeight=100.)
    np.testing.assert_allclose(second["rawVertices"], first["rawVertices"] @ r * s + t, atol=2e-12)
    np.testing.assert_allclose(second["alignedRest"], first["alignedRest"] @ r * s + t, atol=2e-12)
    assert second["diagnostics"]["objectiveAfter"]["total"] == pytest.approx(first["diagnostics"]["objectiveAfter"]["total"] * s*s)


def testNoStaticAnchorsOrRestPriorIsWellDefinedOnObservedConnectedMesh(inputs):
    inputs["staticVertexIds"] = np.empty(0, np.int64)
    result = registerDirectTemplate(**inputs, restWeight=0.)
    assert np.isfinite(result["rawVertices"]).all()
    assert result["diagnostics"]["objectiveAfter"]["staticAnchorResidual"]["count"] == 0
    assert result["diagnostics"]["mseDenominators"]["jointData"] == 24


def testUnobservedDisconnectedComponentWithoutPriorIsRejected(inputs):
    inputs["restVertices"] = np.vstack([inputs["restVertices"], [10., 10., 10.]])
    with pytest.raises(ValueError, match="Unobserved mesh components"):
        registerDirectTemplate(**inputs, restWeight=0.)
    result = registerDirectTemplate(**inputs, restWeight=100.)
    np.testing.assert_allclose(result["rawVertices"][-1], result["alignedRest"][-1], atol=1e-13)


def testInputsAndTopologyAreNeverMutated(inputs):
    copies = {key: value.copy() for key, value in inputs.items()}
    result = registerDirectTemplate(**inputs)
    for key, before in copies.items():
        np.testing.assert_array_equal(inputs[key], before)
    result["rawVertices"][:] = 0
    result["alignedRest"][:] = 0
    np.testing.assert_array_equal(inputs["restVertices"], copies["restVertices"])


def testDisplacementSmoothingHasNoNeutralShapeShrinkage(inputs):
    C, _, _ = independentMatrices(inputs)
    inputs["targetPoints"] = C @ inputs["restVertices"]
    result = registerDirectTemplate(**inputs, smoothnessMode="displacement",
                                    restWeight=0., staticConstraint="hard")
    np.testing.assert_allclose(result["rawVertices"], inputs["restVertices"], atol=2e-14)
    assert result["diagnostics"]["objectiveBefore"]["smooth"] == 0
    assert result["diagnostics"]["objectiveAfter"]["smooth"] < 1e-25
    assert result["diagnostics"]["objectiveAfter"]["absoluteLaplacianSse"] > 1
    assert result["diagnostics"]["differentialRegularization"]["templateCurvaturePriorPresent"]


def testWeightedDifferentialRowsMatchIndependentDenseSystem(inputs):
    weights = np.array([0., .2, .6, 1., 1.])
    originalWeights = weights.copy()
    result = registerDirectTemplate(**inputs, smoothnessMode="displacement",
                                    differentialWeights=weights, restWeight=0., staticConstraint="hard")
    C, S, L = independentMatrices(inputs)
    aligned = result["alignedRest"]
    design = np.vstack([C, S]); targets = np.vstack([inputs["targetPoints"], aligned[inputs["staticVertexIds"]]])
    cd, cs = 10000./(3*len(targets)), 3000./(3*len(aligned))
    H = cd * design.T @ design + cs * L.T @ L
    differential = weights[:, None] * (L @ aligned)
    rhs = cd * design.T @ targets + cs * L.T @ differential
    fixed = inputs["staticVertexIds"]; free = np.setdiff1d(np.arange(len(aligned)), fixed)
    expected = aligned.copy()
    expected[free] = np.linalg.solve(H[np.ix_(free, free)], rhs[free] - H[np.ix_(free, fixed)] @ aligned[fixed])
    np.testing.assert_allclose(result["rawVertices"], expected, atol=5e-13)
    residual = L @ expected - differential
    assert result["diagnostics"]["objectiveAfter"]["smooth"] == pytest.approx(cs*np.sum(residual**2), rel=1e-12)
    np.testing.assert_array_equal(result["rawVertices"][fixed], aligned[fixed])
    np.testing.assert_array_equal(weights, originalWeights)


def testZeroDifferentialRowsExactlyRecoverLegacyGeometry(inputs):
    legacy = registerDirectTemplate(**inputs, restWeight=0., staticConstraint="hard")
    candidate = registerDirectTemplate(**inputs, restWeight=0., staticConstraint="hard",
        smoothnessMode="displacement", differentialWeights=np.zeros(len(inputs["restVertices"])))
    np.testing.assert_array_equal(candidate["rawVertices"], legacy["rawVertices"])
    assert candidate["diagnostics"]["objectiveAfter"]["total"] == legacy["diagnostics"]["objectiveAfter"]["total"]


def testDifferentialSmoothingIsSimilarityEquivariant(inputs):
    original = registerDirectTemplate(**inputs, smoothnessMode="displacement", restWeight=0., staticConstraint="hard")
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    inputs["targetPoints"] = inputs["targetPoints"] @ rotation * 3.7 + [11., -2., 6.]
    transformed = registerDirectTemplate(**inputs, smoothnessMode="displacement", restWeight=0., staticConstraint="hard")
    np.testing.assert_allclose(transformed["rawVertices"], original["rawVertices"] @ rotation * 3.7 + [11., -2., 6.], atol=2e-12)


@pytest.mark.parametrize("options", [
    {"smoothnessMode": "unknown"},
    {"differentialWeights": np.ones(5)},
    {"smoothnessMode": "displacement", "solver": "legacyAdamCpu"},
    {"smoothnessMode": "displacement", "differentialWeights": np.ones(4)},
    {"smoothnessMode": "displacement", "differentialWeights": np.full(5, np.nan)},
    {"smoothnessMode": "displacement", "differentialWeights": np.full(5, 1.01)},
    {"smoothnessMode": "displacement", "differentialWeights": np.full(5, -.01)},
])
def testInvalidDifferentialConfigurationFails(inputs, options):
    with pytest.raises(ValueError):
        registerDirectTemplate(**inputs, **options)


@pytest.mark.parametrize("restWeight", [1000., 100., 0.])
def testCpuFiniteStepReplayMatchesIndependentTorchAutograd(inputs, restWeight):
    torch = pytest.importorskip("torch")
    epochs = 17
    actual = registerDirectTemplate(**inputs, restWeight=restWeight, solver="legacyAdamCpu", epochs=epochs)
    C, S, L = independentMatrices(inputs)
    A = torch.tensor(np.vstack([C, S]), dtype=torch.float32)
    aligned = torch.tensor(actual["alignedRest"], dtype=torch.float32)
    Q = torch.cat([torch.tensor(inputs["targetPoints"], dtype=torch.float32), aligned[inputs["staticVertexIds"]]])
    laplacian = torch.tensor(L, dtype=torch.float32)
    M = torch.eye(len(aligned), dtype=torch.float32) + 19 * laplacian
    u = (M @ aligned).detach().requires_grad_()
    # SGD only applies the independently computed AdamUniform direction; its
    # optimizer-owned LR enables the actual Torch cosine scheduler used before.
    optimizer = torch.optim.SGD([u], lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.003)
    mean, variance = torch.zeros_like(u), torch.zeros_like(u)
    history = []
    for step in range(1, epochs + 1):
        optimizer.zero_grad()
        vertices = torch.linalg.solve(M, u)
        loss = 10000 * ((A @ vertices - Q)**2).mean()
        loss = loss + 3000 * ((laplacian @ vertices)**2).mean() + restWeight * ((vertices - aligned)**2).mean()
        loss.backward()
        with torch.no_grad():
            gradient = u.grad.clone()
            mean.mul_(.9).add_(gradient * .1)
            variance.mul_(.999).add_(gradient.square() * .001)
            corrected = mean / (1 - .9**step)
            denominator = (variance / (1 - .999**step)).sqrt().amax() + 1e-8
            u.grad.copy_(corrected / denominator)
        optimizer.step()
        scheduler.step()
        history.append(loss.item())
    np.testing.assert_allclose(actual["rawVertices"], vertices.detach().numpy(), atol=3e-5, rtol=3e-6)
    np.testing.assert_allclose(actual["diagnostics"]["algorithmReplay"]["lossHistoryFloat32"], history,
                               atol=.04, rtol=3e-5)
    assert actual["diagnostics"]["iterationsPerformed"] == epochs
    assert actual["diagnostics"]["legacyFiniteStepAlgorithmReplayed"] is True
    assert actual["diagnostics"]["legacyFiniteStepTrajectoryMatched"] is False
    assert actual["diagnostics"]["convergedObjectiveSolution"] is False


def testCpuReplayPreservesVendorPreFinalUpdateReturnConvention(inputs):
    one = registerDirectTemplate(**inputs, solver="legacyAdamCpu", epochs=1)
    np.testing.assert_allclose(one["rawVertices"], one["alignedRest"], atol=3e-5, rtol=3e-6)
    assert one["diagnostics"]["algorithmReplay"]["returnedAfterParameterUpdates"] == 0
    many = registerDirectTemplate(**inputs, solver="legacyAdamCpu", epochs=600)
    assert many["diagnostics"]["objectiveAfter"]["total"] < many["diagnostics"]["objectiveBefore"]["total"]
    assert len(many["diagnostics"]["algorithmReplay"]["lossHistoryFloat32"]) == 600
    assert many["diagnostics"]["algorithmReplay"]["returnedAfterParameterUpdates"] == 599


@pytest.mark.parametrize("damage", ["vertex_dtype", "vertex_nan", "faces_range", "faces_float", "face_repeated",
    "sample_range", "bary_shape", "bary_negative", "bary_sum", "target_nan", "mask_type", "mask_few",
    "static_duplicate", "static_range", "static_type", "source_collinear", "target_collapsed"])
def testMalformedInputsFailExplicitly(inputs, damage):
    if damage == "vertex_dtype": inputs["restVertices"] = inputs["restVertices"].astype(np.int64)
    elif damage == "vertex_nan": inputs["restVertices"][0, 0] = np.nan
    elif damage == "faces_range": inputs["faces"][0, 0] = 99
    elif damage == "faces_float": inputs["faces"] = inputs["faces"].astype(np.float64)
    elif damage == "face_repeated": inputs["faces"][0, 0] = inputs["faces"][0, 1]
    elif damage == "sample_range": inputs["sampleFaceIds"][0] = -1
    elif damage == "bary_shape": inputs["sampleBarycentrics"] = inputs["sampleBarycentrics"][:, :2]
    elif damage == "bary_negative": inputs["sampleBarycentrics"][0] = [-.1, 1.1, 0.]
    elif damage == "bary_sum": inputs["sampleBarycentrics"][0] = [.1, .1, .1]
    elif damage == "target_nan": inputs["targetPoints"][0, 0] = np.nan
    elif damage == "mask_type": inputs["validMask"] = inputs["validMask"].astype(np.int64)
    elif damage == "mask_few": inputs["validMask"][2:] = False
    elif damage == "static_duplicate": inputs["staticVertexIds"] = np.array([0, 0], np.int64)
    elif damage == "static_range": inputs["staticVertexIds"][0] = 20
    elif damage == "static_type": inputs["staticVertexIds"] = np.array([0., 1.])
    elif damage == "source_collinear": inputs["restVertices"][:, 1:] = 0
    elif damage == "target_collapsed": inputs["targetPoints"][:] = 0
    with pytest.raises(ValueError):
        registerDirectTemplate(**inputs)


@pytest.mark.parametrize("kwargs", [{"dataWeight": 0}, {"dataWeight": True}, {"smoothWeight": -1},
    {"restWeight": np.inf}, {"restWeight": -1}, {"epochs": 0}, {"epochs": 600.}, {"solver": "faithful600"},
    {"staticConstraint": "invalid"}, {"solver": "legacyAdamCpu", "staticConstraint": "hard"}])
def testInvalidOptionsAndUnavailableTrajectoryFail(inputs, kwargs):
    with pytest.raises(ValueError):
        registerDirectTemplate(**inputs, **kwargs)
