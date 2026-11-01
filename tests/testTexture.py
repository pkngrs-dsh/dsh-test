"""Geometric tests for native UV convention, visibility and masked observations."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceAppearance.texture import buildUvLookup, interpolateSurface, rasterizeDepth, fuseTexture, samplePixels


class TextureTests(unittest.TestCase):
    def setUp(self):
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)
        self.uvs = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float32)
        self.vertices = np.array([[0, 0, 2], [0, 4, 2], [4, 0, 2]], dtype=np.float32)
        self.geometry = {"vertices": self.vertices, "faces": self.faces, "uvs": self.uvs}
        self.camera = {"valid": True, "intrinsics": np.diag([2., 2., 1.]),
                       "rotation": np.eye(3), "translation": np.zeros(3), "medianErrorPixels": 0.}

    def delight(self, values):
        return {"albedoImages": np.stack([np.full((5, 5, 3), value, np.float32) for value in values]),
                "validMasks": np.ones((len(values), 5, 5), dtype=bool)}

    def testUvCoordinatesUseTopLeftIntegerCenters(self):
        lookup = buildUvLookup(self.uvs, self.faces, 5)
        self.assertEqual(int(lookup["validMask"].sum()), 15)
        np.testing.assert_array_equal(lookup["barycentrics"][0, 0], [1, 0, 0])
        np.testing.assert_array_equal(lookup["barycentrics"][4, 0], [0, 1, 0])
        np.testing.assert_array_equal(lookup["barycentrics"][0, 4], [0, 0, 1])
        self.assertEqual(lookup["faceIds"][4, 4], -1)
        surface = interpolateSurface(self.vertices, self.faces, lookup)
        np.testing.assert_array_equal(surface[3, 1], [1, 3, 2])
        np.testing.assert_array_equal(surface[4, 4], [0, 0, 0])

    def testDepthIsPerspectiveCorrect(self):
        vertices = np.array([[0, 0, 1], [4, 0, 2], [0, 4, 2]], dtype=float)
        depth = rasterizeDepth(vertices, self.faces, np.eye(3), 3, 3)
        self.assertAlmostEqual(float(depth[0, 1]), 4 / 3, places=6)
        self.assertTrue(np.isinf(depth[2, 2]))

    def testBlackRgbIsAValidObservation(self):
        result = fuseTexture(self.geometry, self.delight([0]), [self.camera], resolution=5)
        self.assertEqual(int(result["coverageMask"].sum()), 15)
        np.testing.assert_array_equal(result["albedo"], 0)
        self.assertGreater(result["confidence"][0, 0], 0)
        self.assertEqual(result["sourceIndex"][0, 0], 0)

    def testRgbSamplingDoesNotFlipVerticalDirection(self):
        data = self.delight([0])
        data["albedoImages"][0, :, :, 0] = np.arange(5)[:, None] / 4
        result = fuseTexture(self.geometry, data, [self.camera], resolution=5)
        self.assertAlmostEqual(float(result["albedo"][3, 1, 0]), 0.75)
        self.assertAlmostEqual(float(result["albedo"][1, 3, 0]), 0.25)

    def testTranslationUsesCanonicalToCameraConvention(self):
        data = self.delight([0])
        data["albedoImages"][0, :, :, 0] = np.arange(5)[None, :] / 4
        camera = dict(self.camera, translation=np.array([1., 0., 0.]))
        result = fuseTexture(self.geometry, data, [camera], resolution=5)
        self.assertAlmostEqual(float(result["albedo"][1, 1, 0]), 0.5)

    def testColorBalanceFitsOverlappingObservations(self):
        result = fuseTexture(self.geometry, self.delight([0.3, 0.4]), [self.camera] * 2,
                             resolution=33, colorBalance=True)
        np.testing.assert_allclose(result["diagnostics"]["colorGains"][1], [0.75] * 3, atol=1e-6)
        np.testing.assert_allclose(result["albedo"][result["coverageMask"]], 0.3, atol=1e-6)

    def testCloserSurfaceOccludesRearSurface(self):
        geometry = {"vertices": np.vstack([self.vertices, self.vertices / 2]),
                    "faces": np.array([[0, 1, 2], [3, 4, 5]]),
                    "uvs": np.array([[0, 0], [0, .5], [.5, 0], [.5, .5], [.5, 1], [1, .5]])}
        result = fuseTexture(geometry, self.delight([0.5]), [self.camera], resolution=9)
        self.assertFalse(result["coverageMask"][1, 1])
        self.assertTrue(result["coverageMask"][5, 5])
        self.assertEqual(result["sourceIndex"][1, 1], -1)

    def testHoleFillNeverClaimsMeasuredCoverage(self):
        data = self.delight([0.5])
        data["validMasks"][0, 0, 0] = False
        result = fuseTexture(self.geometry, data, [self.camera], resolution=5)
        self.assertFalse(result["coverageMask"][0, 0])
        np.testing.assert_array_equal(result["albedo"][0, 0], 0)
        np.testing.assert_allclose(result["previewAlbedo"][0, 0], 0.5)
        self.assertEqual(result["sourceIndex"][0, 0], -1)
        self.assertEqual(result["confidence"][0, 0], 0)

    def testBackfacesDoNotReceiveTexture(self):
        geometry = dict(self.geometry, faces=self.faces[:, ::-1])
        result = fuseTexture(geometry, self.delight([0.5]), [self.camera], resolution=5)
        self.assertFalse(result["coverageMask"].any())

    def testInvalidCamerasRemainExplicitlyMissing(self):
        result = fuseTexture(self.geometry, self.delight([0.5]), [{"valid": False}], resolution=5)
        self.assertFalse(result["coverageMask"].any())
        self.assertEqual(result["diagnostics"]["observedTexelCount"], 0)
        np.testing.assert_array_equal(result["sourceIndex"], -1)
        self.assertTrue(result["diagnostics"]["warnings"])

    def testColorOutlierDoesNotDominateConsensus(self):
        result = fuseTexture(self.geometry, self.delight([0.2, 0.2, 0.9]), [self.camera] * 3,
                             resolution=5, colorBalance=False)
        self.assertLess(float(result["albedo"][1, 1, 0]), 0.21)
        self.assertGreaterEqual(float(result["albedo"][1, 1, 0]), 0.2 - 1e-6)

    def testLargeSamplingBatchHasNoRasterAxisLimit(self):
        samples = samplePixels(np.full((2, 2, 3), 0.25, np.float32), np.zeros((40000, 2)))
        self.assertEqual(samples.shape, (40000, 3))
        np.testing.assert_array_equal(samples, 0.25)

    def testInvalidInputsFailBeforeSampling(self):
        with self.assertRaises(ValueError):
            buildUvLookup(self.uvs, np.array([[0, 1, 99]]), 5)
        with self.assertRaises(ValueError):
            buildUvLookup(self.uvs, self.faces, 1)
        data = self.delight([0.5])
        data["albedoImages"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fuseTexture(self.geometry, data, [self.camera], resolution=5)


if __name__ == "__main__":
    unittest.main()
