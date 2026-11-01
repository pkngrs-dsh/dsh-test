"""Deterministic CPU checks for the read-only triangle intersection audit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("directSafety", ROOT / "tools/auditDirectRegistrationSafety.py")
SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY)


class DirectRegistrationSafetyTests(unittest.TestCase):
    def testSyntheticRegressionCases(self):
        self.assertEqual(SAFETY.selfTest()["status"], "passed")

    def testRigidRotationAndTranslationPreserveIntersection(self):
        a = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        b = np.array([[.5, .5, -1.], [.5, .5, 1.], [1.5, .5, 0.]])
        rng = np.random.default_rng(20261012)
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            shift = rng.normal(size=3) * 100
            aa, bb = a @ rotation + shift, b @ rotation + shift
            result = SAFETY.intersectTriangles(aa, bb, 1e-8)
            self.assertEqual(result["kind"], "nonCoplanarSegment")
            self.assertTrue(SAFETY.satIntersects(aa, bb, 1e-8))
            self.assertAlmostEqual(result["segmentLengthCm"], 1.)

    def testRandomPlaneClippingMatchesIndependentSeparatingAxes(self):
        rng = np.random.default_rng(4492)
        for _ in range(500):
            a, b = rng.normal(size=(2, 3, 3))
            self.assertEqual(SAFETY.intersectTriangles(a, b, 1e-9) is not None,
                             bool(SAFETY.satIntersects(a, b, 1e-9)))

    def testCoincidentPositionsAreNotConfusedWithSharedVertexIndices(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        pairs, excluded = SAFETY.broadPhase(v, np.array([[0, 1, 2], [3, 4, 5]]), 1e-9)
        self.assertEqual(excluded, 0)
        np.testing.assert_array_equal(pairs, [[0, 1]])
        hit = SAFETY.intersectTriangles(v[:3], v[3:], 1e-9)
        self.assertEqual(hit["kind"], "coplanarAreaOverlap")
        self.assertAlmostEqual(hit["overlapAreaCm2"], .5)


if __name__ == "__main__":
    unittest.main()
