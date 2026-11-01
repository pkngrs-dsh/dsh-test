"""CPU checks of the subject-specific WildCap presentation controls."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


toolPath = Path(__file__).resolve().parents[1] / "tools/runWildCapUsageValidation.py"
specification = importlib.util.spec_from_file_location("wildCapUsageValidationTestTool", toolPath)
validation = importlib.util.module_from_spec(specification)
with patch.object(sys, "path", [str(toolPath.parent), *sys.path]):
    specification.loader.exec_module(validation)


def testCandidateMaskProtectsDetailsAndRejectsUnobservedTexels():
    valid = np.ones((512, 512), bool)
    mask, protected, metadata = validation.candidateMask(valid)
    assert mask.dtype == protected.dtype == bool
    assert mask.shape == protected.shape == valid.shape
    assert mask.any() and protected.any()
    assert not np.any(mask & protected)
    assert metadata["candidateTexels"] == mask.sum()
    assert protected[100, 180]  # Brow
    assert protected[140, 180]  # Eye/lashes
    assert protected[280, 256]  # Mouth
    assert protected[187, 358]  # Inspected mole
    assert "not annotated shadow ground truth" in metadata["meaning"]
    valid[:, :256] = False
    clipped, sameProtection, _ = validation.candidateMask(valid)
    np.testing.assert_array_equal(clipped, mask & valid)
    np.testing.assert_array_equal(sameProtection, protected)
    assert not clipped[:, :256].any()
    assert clipped[:, 256:].any()


def testCandidateMaskRejectsNoUsefulSupport():
    with pytest.raises(ValueError, match="no useful support"):
        validation.candidateMask(np.zeros((512, 512), bool))


def testFeatherUsesLinearColorAndPreservesOutsideAndCore():
    base = np.full((11, 11, 3), .2, np.float32)
    diffuse = np.full_like(base, .8)
    valid = np.zeros((11, 11), bool)
    valid[1:10, 1:10] = True
    baseBefore, diffuseBefore, validBefore = base.copy(), diffuse.copy(), valid.copy()
    composite, alpha = validation.featherComposite(base, diffuse, valid, width=2)
    assert composite.dtype == alpha.dtype == np.float32
    assert alpha[1, 5] == 0 and alpha[2, 5] == .5 and alpha[3, 5] == 1
    expected = ((.2 ** 2.2 + .8 ** 2.2) / 2) ** (1 / 2.2)
    np.testing.assert_allclose(composite[2, 5], expected, atol=1e-7)
    np.testing.assert_array_equal(composite[~valid], base[~valid])
    np.testing.assert_array_equal(composite[alpha == 1], diffuse[alpha == 1])
    np.testing.assert_allclose(composite[alpha == 0], base[alpha == 0], atol=1e-7)
    np.testing.assert_array_equal(base, baseBefore)
    np.testing.assert_array_equal(diffuse, diffuseBefore)
    np.testing.assert_array_equal(valid, validBefore)


@pytest.mark.parametrize("width", [0, -1])
def testFeatherRejectsNonpositiveWidth(width):
    with pytest.raises(ValueError, match="must be positive"):
        validation.featherComposite(np.zeros((3, 3, 3)), np.ones((3, 3, 3)),
                                    np.ones((3, 3), bool), width)


def testFeatherEmptySupportLeavesInputUnchanged():
    base = np.random.default_rng(7).random((8, 8, 3), dtype=np.float32)
    composite, alpha = validation.featherComposite(base, np.ones_like(base),
                                                   np.zeros((8, 8), bool))
    np.testing.assert_array_equal(composite, base)
    assert not alpha.any()


def testBoundaryMetricMeasuresAddedLinearJumpAndCoverage():
    valid = np.zeros((9, 9), bool)
    valid[3:6, 3:6] = True
    coverage = np.ones_like(valid)
    base = np.full((9, 9, 3), .2, np.float32)
    color = base.copy()
    color[valid] = .8
    report = validation.edgeDeltaMetric(color, base, valid, coverage)
    assert report["edgePairCount"] == 12
    expected = .8 ** 2.2 - .2 ** 2.2
    assert report["meanAddedJumpLinear"] == pytest.approx(expected)
    assert report["p95AddedJumpLinear"] == pytest.approx(expected)
    unchanged = validation.edgeDeltaMetric(base, base, valid, coverage)
    assert unchanged["meanAddedJumpLinear"] == unchanged["p95AddedJumpLinear"] == 0
    coverage[2] = False
    assert validation.edgeDeltaMetric(color, base, valid, coverage)["edgePairCount"] == 9


def testFeatherBoundaryMetricIsContinuityNotReflectanceAccuracy():
    base = np.full((31, 31, 3), .2, np.float32)
    diffuse = np.full_like(base, .8)
    valid = np.zeros((31, 31), bool)
    valid[2:29, 2:29] = True
    composite, alpha = validation.featherComposite(base, diffuse, valid, width=3)
    metric = validation.edgeDeltaMetric(composite, base, valid, np.ones_like(valid))
    # This reduction is imposed at the patch boundary even for an arbitrary
    # interior color; it cannot establish a more accurate de-lit albedo.
    assert metric["meanAddedJumpLinear"] < 1e-7
    assert alpha[15, 15] == 1
    np.testing.assert_array_equal(composite[15, 15], diffuse[15, 15])
