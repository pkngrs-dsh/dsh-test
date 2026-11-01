"""Independent CPU checks for caller-normalized, target-frame XYZ priors."""

from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

from faceAppearance.directTemplateRegistration import registerDirectTemplate


@pytest.fixture
def inputs():
    rest = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.], [0., 0., 1.4]])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]])
    faceIds = np.array([0, 0, 1, 2, 0, 1, 3, 4])
    bary = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.],
                     [.2, .3, .5], [.4, .2, .4], [.2, .6, .2]])
    samples = (rest[faces[faceIds]] * bary[..., None]).sum(axis=1)
    target = samples @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) * 1.7 + [4., -3., 2.]
    target[4] += [.17, -.23, .31]
    target[6] += [-.05, .07, .08]
    return {"restVertices": rest, "faces": faces, "sampleFaceIds": faceIds,
            "sampleBarycentrics": bary, "targetPoints": target,
            "validMask": np.ones(len(faceIds), bool), "staticVertexIds": np.array([0, 3])}


def makePrior(n=5, weight=170.):
    # Each row couples multiple vertices AND different axes. Targets deliberately
    # differ from rest, so an axis-separated or ignored prior cannot pass.
    matrix = np.zeros((3, 3*n))
    matrix[0, [0, 4, 14]] = [.7, -.2, .9]
    matrix[1, [2, 6, 13]] = [-.5, .6, -.1]
    matrix[2, [3, 8, 12]] = [.8, -.4, .2]
    return {"operator": sparse.csr_matrix(matrix), "targets": np.array([1.2, -.7, 2.3]), "weight": weight}


def independentProblem(data, aligned, prior, weights, differential=None):
    """Construct a dense least-squares design, independently of production H."""
    n = len(aligned)
    C = np.zeros((int(data["validMask"].sum()), n))
    for row, sample in enumerate(np.flatnonzero(data["validMask"])):
        for vertex, amount in zip(data["faces"][data["sampleFaceIds"][sample]], data["sampleBarycentrics"][sample]):
            C[row, vertex] += amount
    adjacency = np.zeros((n, n))
    for triangle in data["faces"]:
        for i in triangle:
            for j in triangle:
                if i != j:
                    adjacency[i, j] = 1.
    L = np.diag(adjacency.sum(axis=1)) - adjacency
    static = data["staticVertexIds"]
    D = np.vstack([C, np.eye(n)[static]])
    q = np.vstack([data["targetPoints"][data["validMask"]], aligned[static]])
    cd, cs, cr = weights[0] / q.size, weights[1] / aligned.size, weights[2] / aligned.size
    smoothTarget = np.zeros_like(aligned) if differential is None else differential[:, None] * (L @ aligned)
    design = np.vstack([np.sqrt(cd)*np.kron(D, np.eye(3)),
                        np.sqrt(cs)*np.kron(L, np.eye(3)),
                        np.sqrt(cr)*np.eye(3*n),
                        np.sqrt(prior["weight"])*prior["operator"].toarray()])
    target = np.concatenate([np.sqrt(cd)*q.ravel(), np.sqrt(cs)*smoothTarget.ravel(),
                             np.sqrt(cr)*aligned.ravel(), np.sqrt(prior["weight"])*prior["targets"]])
    return design, target


@pytest.mark.parametrize("staticConstraint", ["soft", "hard"])
@pytest.mark.parametrize("smoothnessMode", ["absolute", "displacement"])
def testCoupledOptimumAndDiagnosticsMatchIndependentDenseLeastSquares(inputs, staticConstraint, smoothnessMode):
    prior = makePrior()
    weights = (103., 17., 3.)
    differential = np.array([0., .2, .6, 1., .8]) if smoothnessMode == "displacement" else None
    result = registerDirectTemplate(**inputs, dataWeight=weights[0], smoothWeight=weights[1], restWeight=weights[2],
        quadraticPrior=prior, staticConstraint=staticConstraint, smoothnessMode=smoothnessMode,
        differentialWeights=differential)
    aligned, vertices = result["alignedRest"], result["rawVertices"]
    design, target = independentProblem(inputs, aligned, prior, weights, differential)
    fixed = np.concatenate([3*inputs["staticVertexIds"] + axis for axis in range(3)]) if staticConstraint == "hard" else np.empty(0, int)
    free = np.setdiff1d(np.arange(vertices.size), fixed)
    expected = aligned.ravel().copy()
    expected[free] = np.linalg.lstsq(design[:, free], target - design[:, fixed] @ expected[fixed], rcond=None)[0]
    np.testing.assert_allclose(vertices.ravel(), expected, rtol=2e-13, atol=2e-13)
    if staticConstraint == "hard":
        np.testing.assert_array_equal(vertices[inputs["staticVertexIds"]], aligned[inputs["staticVertexIds"]])
    report = result["diagnostics"]
    residual = design @ vertices.ravel() - target
    independentGradient = (2*design.T @ residual)[free]
    assert np.max(np.abs(independentGradient)) < 2e-11
    assert report["normalEquationGradientMaxAbs"] < 2e-11
    assert report["normalEquationRelativeResidual"] < 1e-12
    assert report["objectiveAfter"]["total"] == pytest.approx(float(residual @ residual), rel=1e-13)
    for label, value in [("Before", aligned), ("After", vertices)]:
        item = report["objective" + label]
        priorResidual = prior["operator"] @ value.ravel() - prior["targets"]
        assert item["quadraticPrior"] == pytest.approx(prior["weight"] * (priorResidual @ priorResidual))
        assert item["total"] == pytest.approx(sum(item[key] for key in ("data", "smooth", "rest", "quadraticPrior")))
    assert report["objectiveAfter"]["total"] < report["objectiveBefore"]["total"]
    assert report["mseDenominators"] == {"jointData": 30, "smooth": 15, "rest": 15}
    assert report["quadraticPrior"]["rowNormalization"] == "callerSuppliedUnchanged"
    baseline = registerDirectTemplate(**inputs)
    np.testing.assert_array_equal(aligned, baseline["alignedRest"])
    assert np.linalg.norm(vertices - baseline["rawVertices"]) > .1


def testAllHardFixedVerticesStayExactEvenWithIncompatiblePrior(inputs):
    inputs["staticVertexIds"] = np.arange(5)
    result = registerDirectTemplate(**inputs, staticConstraint="hard", quadraticPrior=makePrior(weight=1e8))
    np.testing.assert_array_equal(result["rawVertices"], result["alignedRest"])
    report = result["diagnostics"]
    assert report["freeVertexCount"] == 0
    assert report["normalEquationGradientMaxAbs"] == 0
    assert report["objectiveAfter"]["quadraticPrior"] > 0
    assert report["objectiveBefore"] == report["objectiveAfter"]


@pytest.mark.parametrize("constraint", ["soft", "hard"])
@pytest.mark.parametrize("mode", ["absolute", "displacement"])
def testNoneAndZeroAreBitwiseOriginalPathIncludingDiagnostics(inputs, constraint, mode):
    kwargs = {"staticConstraint": constraint, "smoothnessMode": mode, "restWeight": 0.}
    baseline = registerDirectTemplate(**inputs, **kwargs)
    for prior in (None, makePrior(weight=0.)):
        actual = registerDirectTemplate(**inputs, **kwargs, quadraticPrior=prior)
        for key in ("rawVertices", "alignedRest"):
            np.testing.assert_array_equal(actual[key], baseline[key])
        for key in ("rotation", "translation", "scale"):
            np.testing.assert_array_equal(actual["transform"][key], baseline["transform"][key])
        actualReport, expectedReport = deepcopy(actual["diagnostics"]), deepcopy(baseline["diagnostics"])
        actualReport.pop("seconds")
        expectedReport.pop("seconds")
        assert actualReport == expectedReport


def testTargetFramePriorRotatesAndTranslatesWithoutChangingItsMeaning(inputs):
    prior = makePrior()
    first = registerDirectTemplate(**inputs, quadraticPrior=prior, staticConstraint="hard")
    angle = .42
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    translation = np.array([8.2, -1.7, 4.3])
    inputs["targetPoints"] = inputs["targetPoints"] @ rotation + translation
    # row vertices V'=V@R+t; column vector inverse is blockdiag(R).
    rotatedOperator = prior["operator"] @ sparse.kron(sparse.eye(5), rotation)
    transformed = {"operator": rotatedOperator.tocsr(),
                   "targets": prior["targets"] + rotatedOperator @ np.tile(translation, 5), "weight": prior["weight"]}
    second = registerDirectTemplate(**inputs, quadraticPrior=transformed, staticConstraint="hard")
    np.testing.assert_allclose(second["rawVertices"], first["rawVertices"] @ rotation + translation, atol=2e-13)
    assert second["diagnostics"]["objectiveAfter"]["total"] == pytest.approx(first["diagnostics"]["objectiveAfter"]["total"], rel=1e-13)
    # Passing the untransformed prior must have a different effect: the API must
    # not silently interpret this operator in template coordinates.
    wrongFrame = registerDirectTemplate(**inputs, quadraticPrior=prior, staticConstraint="hard")
    assert np.linalg.norm(wrongFrame["rawVertices"] - second["rawVertices"]) > .05


def testNoHiddenRowNormalizationAndSparseDuplicateInputsRemainUnchanged(inputs):
    prior = makePrior()
    original = registerDirectTemplate(**inputs, quadraticPrior=prior)
    coo = prior["operator"].tocoo()
    duplicate = sparse.coo_matrix((np.r_[coo.data*.3, coo.data*.7],
        (np.r_[coo.row, coo.row], np.r_[coo.col, coo.col])), shape=coo.shape)
    inputCopies = {key: value.copy() for key, value in inputs.items()}
    oldData, oldRows, oldColumns = duplicate.data.copy(), duplicate.row.copy(), duplicate.col.copy()
    targets = prior["targets"].copy()
    actual = registerDirectTemplate(**inputs, quadraticPrior={"operator": duplicate, "targets": targets, "weight": prior["weight"]})
    np.testing.assert_allclose(actual["rawVertices"], original["rawVertices"], atol=3e-14)
    for key, value in inputCopies.items():
        np.testing.assert_array_equal(inputs[key], value)
    for actualArray, before in [(duplicate.data, oldData), (duplicate.row, oldRows), (duplicate.col, oldColumns), (targets, prior["targets"])]:
        np.testing.assert_array_equal(actualArray, before)
    rescaled = {"operator": prior["operator"]*7, "targets": targets*7, "weight": prior["weight"]/49}
    second = registerDirectTemplate(**inputs, quadraticPrior=rescaled)
    np.testing.assert_allclose(second["rawVertices"], original["rawVertices"], atol=3e-14)


def testPriorCanConstrainPreviouslyUnobservedComponentAndRejectsMissingAxes(inputs):
    inputs["restVertices"] = np.vstack([inputs["restVertices"], [10., 10., 10.]])
    target = np.array([7., 9., -2.])
    full = sparse.csr_matrix((np.ones(3), (np.arange(3), np.arange(15, 18))), shape=(3, 18))
    result = registerDirectTemplate(**inputs, restWeight=0., quadraticPrior={"operator": full, "targets": target, "weight": 11.})
    np.testing.assert_allclose(result["rawVertices"][-1], target, atol=2e-14)
    with pytest.raises(ValueError, match="singular|unconstrained"):
        registerDirectTemplate(**inputs, restWeight=0., quadraticPrior={"operator": full[:1], "targets": target[:1], "weight": 11.})
    with pytest.raises(ValueError, match="Unobserved"):
        registerDirectTemplate(**inputs, restWeight=0., quadraticPrior={"operator": full, "targets": target, "weight": 0.})


def testRelativeOnlyPriorCannotHideUnobservedComponentTranslationNullspace(inputs):
    inputs["restVertices"] = np.vstack([inputs["restVertices"], [[10., 0., 0.], [11., 0., 0.], [10., 1., 0.]]])
    inputs["faces"] = np.vstack([inputs["faces"], [5, 6, 7]])
    # The disconnected triangle has smoothing and a cross-axis relative prior,
    # but still no measurement of its world translation.
    matrix = sparse.csr_matrix(([.3, .4, -.3, -.4], ([0]*4, [15, 16, 18, 19])), shape=(1, 24))
    with pytest.raises(ValueError, match="singular|unconstrained"):
        registerDirectTemplate(**inputs, restWeight=0., quadraticPrior={"operator": matrix, "targets": np.array([.2]), "weight": 19.})


@pytest.mark.parametrize("invalid", [
    {}, {"operator": sparse.eye(15), "targets": np.zeros(15), "weight": 1., "extra": True},
    {"operator": np.eye(15), "targets": np.zeros(15), "weight": 1.},
    {"operator": sparse.csr_matrix((1, 14)), "targets": np.zeros(1), "weight": 1.},
    {"operator": sparse.csr_matrix((0, 15)), "targets": np.zeros(0), "weight": 1.},
    {"operator": sparse.eye(15, format="lil"), "targets": np.zeros(15), "weight": 1.},
    {"operator": sparse.eye(15, format="csr", dtype=complex), "targets": np.zeros(15), "weight": 1.},
    {"operator": sparse.eye(15, format="csr")*np.nan, "targets": np.zeros(15), "weight": 0.},
    {"operator": sparse.eye(15, format="csr"), "targets": np.zeros((15, 1)), "weight": 1.},
    {"operator": sparse.eye(15, format="csr"), "targets": np.zeros(15, int), "weight": 1.},
    {"operator": sparse.eye(15, format="csr"), "targets": np.full(15, np.inf), "weight": 1.},
    *[{"operator": sparse.eye(15, format="csr"), "targets": np.zeros(15), "weight": weight}
      for weight in (-1., float("nan"), float("inf"), True, "1")],
])
def testInvalidPriorsAreRejectedBeforeSolving(inputs, invalid):
    with pytest.raises(ValueError, match="quadraticPrior"):
        registerDirectTemplate(**inputs, quadraticPrior=invalid)


@pytest.mark.parametrize("weight", [0., 1.])
def testLegacyReplayExplicitlyRejectsPriorRatherThanIgnoringIt(inputs, weight):
    with pytest.raises(ValueError, match="exact solver"):
        registerDirectTemplate(**inputs, solver="legacyAdamCpu", quadraticPrior=makePrior(weight=weight))


def testSparseSizeAndOuterProductBudgetsRejectDenseRisk(inputs):
    with pytest.raises(ValueError, match="storage budget"):
        registerDirectTemplate(**inputs, quadraticPrior={"operator": sparse.csr_matrix((1_000_001, 15)),
            "targets": np.empty(0), "weight": 1.})
    inputs["restVertices"] = np.vstack([inputs["restVertices"], np.zeros((1496, 3))])
    matrix = sparse.csr_matrix(np.ones((1, 4503)))
    with pytest.raises(ValueError, match="product budget"):
        registerDirectTemplate(**inputs, quadraticPrior={"operator": matrix, "targets": np.zeros(1), "weight": 1.})


def testFiniteInputsWhoseNormalMatrixOverflowsAreRejected(inputs):
    matrix = sparse.csr_matrix(([1e200], ([0], [0])), shape=(1, 15))
    with pytest.raises(ValueError, match="nonfinite coefficients"):
        registerDirectTemplate(**inputs, quadraticPrior={"operator": matrix, "targets": np.zeros(1), "weight": 1.})
