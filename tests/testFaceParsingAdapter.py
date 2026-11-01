"""CPU tests exercise real tensor warping with tiny injected model kernels."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from faceAppearance.adapters.faceParsing import FaceParsingAdapter


class FaceParsingAdapterTests(unittest.TestCase):
    def makeAdapter(self, root):
        import torch
        import torch.nn.functional as functional

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.faceCount = 1
                self.faces = None

            def forward(self, image):
                self.calls += 1
                self.inputShape = image.shape
                if self.faces is not None:
                    return self.faces
                return {"scores": torch.ones(self.faceCount),
                        "rects": torch.tensor([[2., 3., 28., 29.]]).repeat(self.faceCount, 1),
                        "points": torch.tensor([[[4., 5.], [24., 5.], [14., 13.], [7., 23.], [21., 23.]]])
                        .repeat(self.faceCount, 1, 1)}

        class Parser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.active = 0
                self.maxActive = 0

            def forward(self, image):
                self.calls += 1
                self.active += 1
                self.maxActive = max(self.active, self.maxActive)
                time.sleep(0.002)
                # The red channel controls left/right labels. This makes a
                # dropped inverse warp or RGB/channel-order mistake observable.
                logits = torch.cat([torch.full_like(image[:, :1], -10),
                                    12 * (image[:, :1] - 0.5),
                                    12 * (0.5 - image[:, :1])], dim=1)
                self.active -= 1
                return logits, None

        def identityGrid(*, matrix, orig_shape):
            return functional.affine_grid(torch.eye(2, 3)[None], (1, 3, *orig_shape), align_corners=False)

        class StubAdapter(FaceParsingAdapter):
            def __init__(self, projectRoot):
                super().__init__(projectRoot, device="cpu", detectionMaxSize=32)
                self.loads = 0

            def loadBackend(self, paths):
                self.loads += 1
                return ({"faceDetector": Detector(), "faceParser": Parser()},
                        {"label_names": ["background", "lb", "rb"],
                         "get_matrix_fn": lambda points: points,
                         "get_grid_fn": identityGrid, "get_inv_grid_fn": identityGrid},
                        {"backend": "test"})

        adapter = StubAdapter(root)
        for path in adapter.assetPaths().values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return adapter

    def testImportDoesNotImportTorchOrFacer(self):
        result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                                 "import faceAppearance.adapters.faceParsing; "
                                 "assert 'torch' not in sys.modules; assert 'facer' not in sys.modules",
                                 str(Path(__file__).resolve().parents[1] / "src")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def testPreloadIsExplicitIdempotentAndKeepsOriginalFrame(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory))
            image = np.zeros((64, 40, 3), np.uint8)
            image[:, :20, 0] = 255
            with self.assertRaisesRegex(RuntimeError, "loadModels"):
                adapter.infer(image)
            adapter.loadModels().loadModels()
            firstModels = dict(adapter.models)
            first = adapter.infer(image)
            second = adapter.infer(image)
            self.assertEqual(adapter.loads, 1)
            self.assertEqual(adapter.models, firstModels)
            self.assertEqual(first.labelMap.shape, image.shape[:2])
            np.testing.assert_array_equal(first.labelMap[:, :20], 2)
            np.testing.assert_array_equal(first.labelMap[:, 20:], 3)
            np.testing.assert_array_equal(first.labelMap, second.labelMap)
            np.testing.assert_allclose(first.landmarks[0], [8.5, 10.5])
            np.testing.assert_allclose(first.boxXyxy, [4, 6, 56, 58])
            self.assertEqual(first.metadata["sideConvention"], "sourceDatasetLabels")
            self.assertEqual(first.metadata["selection"],
                             {"strategy": "largestArea", "detectedCount": 1, "selectedIndex": 0})
            self.assertEqual(tuple(adapter.models["faceDetector"].inputShape), (1, 3, 32, 20))
            self.assertTrue(first.validMask.all())

    def testProvidedLandmarksBypassDetectionAndInputIsUnchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            image = np.full((20, 30, 3), 120, np.uint8)
            points = np.array([[5, 5], [20, 5], [13, 10], [7, 15], [18, 15]], np.float32)
            imageBefore, pointsBefore = image.copy(), points.copy()
            result = adapter.infer(image, faceLandmarks=points)
            self.assertEqual(adapter.models["faceDetector"].calls, 0)
            np.testing.assert_array_equal(result.landmarks, points)
            np.testing.assert_array_equal(image, imageBefore)
            np.testing.assert_array_equal(points, pointsBefore)
            self.assertIsNone(result.boxXyxy)
            self.assertEqual(result.metadata["landmarkSource"], "provided")
            self.assertNotIn("selection", result.metadata)

    def testDirectDetectionKeepsThreeItemContractAndSelectionIsOptIn(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            imageTensor = torch.zeros((1, 3, 64, 40), dtype=torch.uint8)
            points, box, score = adapter.detectFace(imageTensor)
            selectedPoints, selectedBox, selectedScore, selection = adapter.detectFace(
                imageTensor, includeSelection=True)
            torch.testing.assert_close(points, selectedPoints)
            np.testing.assert_array_equal(box, selectedBox)
            self.assertEqual(score, selectedScore)
            np.testing.assert_allclose(points[0, 0].numpy(), [8.5, 10.5])
            np.testing.assert_allclose(box, [4, 6, 56, 58])
            self.assertEqual(score, 1.0)
            self.assertEqual(selection,
                             {"strategy": "largestArea", "detectedCount": 1, "selectedIndex": 0})

    def testNoDetectionFailsAndNextCallRecovers(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            detector = adapter.models["faceDetector"]
            detector.faceCount = 0
            for faces in (None, {"scores": torch.empty(0), "rects": torch.empty(0),
                                 "points": torch.empty(0)}):
                detector.faces = faces
                with self.assertRaisesRegex(ValueError, "No faces detected"):
                    adapter.infer(np.zeros((32, 32, 3), np.uint8))
            self.assertEqual(adapter.models["faceParser"].calls, 0)
            detector.faces = None
            detector.faceCount = 1
            adapter.infer(np.zeros((32, 32, 3), np.uint8))
            self.assertEqual(adapter.loads, 1)

    def testMultipleFacesSelectLargestAndKeepCoordinatesAndScoreAligned(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            detected = {
                "scores": torch.tensor([0.99, 0.85, 0.95]),
                "rects": torch.tensor([[1., 1., 9., 9.], [2., 3., 18., 29.], [2., 2., 12., 12.]]),
                "points": torch.tensor([
                    [[2., 2.], [7., 2.], [4., 4.], [3., 7.], [6., 7.]],
                    [[4., 5.], [16., 5.], [10., 13.], [6., 23.], [14., 23.]],
                    [[3., 3.], [10., 3.], [6., 6.], [4., 10.], [9., 10.]],
                ]),
            }
            originals = {key: value.clone() for key, value in detected.items()}
            adapter.models["faceDetector"].faces = detected
            parsedPoints = []
            originalParse = adapter.parseImage

            def recordPoints(imageTensor, points):
                parsedPoints.append(points.clone())
                return originalParse(imageTensor, points)

            adapter.parseImage = recordPoints
            result = adapter.infer(np.zeros((64, 40, 3), np.uint8))
            expectedPoints = (detected["points"][1].numpy() + 0.5) * 2 - 0.5
            np.testing.assert_allclose(result.landmarks, expectedPoints)
            np.testing.assert_allclose(parsedPoints[0][0].numpy(), expectedPoints)
            self.assertEqual(tuple(parsedPoints[0].shape), (1, 5, 2))
            np.testing.assert_allclose(result.boxXyxy, [4, 6, 36, 58])
            self.assertAlmostEqual(result.metadata["detectionScore"], 0.85)
            self.assertEqual(result.metadata["selection"],
                             {"strategy": "largestArea", "detectedCount": 3, "selectedIndex": 1})
            for key in detected:
                torch.testing.assert_close(detected[key], originals[key])

    def testEqualAreaFacesUseConfidenceThenOriginalIndex(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            adapter.models["faceDetector"].faceCount = 3
            detected = adapter.models["faceDetector"](torch.zeros((1, 3, 32, 32), dtype=torch.uint8))
            adapter.models["faceDetector"].faces = detected
            for scores, selectedIndex in (([0.8, 0.9, 0.85], 1), ([0.9, 0.9, 0.85], 0)):
                with self.subTest(scores=scores):
                    detected["scores"] = torch.tensor(scores)
                    result = adapter.infer(np.zeros((32, 32, 3), np.uint8))
                    self.assertEqual(result.metadata["selection"]["selectedIndex"], selectedIndex)
                    self.assertAlmostEqual(result.metadata["detectionScore"], 0.9)

    def testMalformedMultipleFaceDetectionFailsBeforeParsing(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            detector = adapter.models["faceDetector"]
            detector.faceCount = 2
            detected = detector(torch.zeros((1, 3, 32, 32), dtype=torch.uint8))
            malformedFields = (
                ("rects", torch.tensor([[1., 1., 1., 2.], [2., 3., 28., 29.]])),
                ("rects", torch.tensor([[float("nan"), 1., 2., 2.], [2., 3., 28., 29.]])),
                ("scores", torch.tensor([0.8])),
                ("points", detected["points"][:1]),
            )
            for key, value in malformedFields:
                with self.subTest(field=key, shape=value.shape):
                    detector.faces = {**detected, key: value}
                    with self.assertRaises(ValueError):
                        adapter.infer(np.zeros((32, 32, 3), np.uint8))
            self.assertEqual(adapter.models["faceParser"].calls, 0)

    def testInverseWarpPaddingIsNotConfidentObservedBackground(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            originalInverse = adapter.parserSettings["get_inv_grid_fn"]

            def shiftedInverse(**kwargs):
                grid = originalInverse(**kwargs)
                grid[:, :, :4] = torch.tensor([-2., -2.])
                return grid

            adapter.parserSettings["get_inv_grid_fn"] = shiftedInverse
            result = adapter.infer(np.zeros((32, 32, 3), np.uint8))
            self.assertFalse(result.validMask[:, :4].any())
            self.assertTrue(result.validMask[:, 4:].all())
            np.testing.assert_array_equal(result.labelMap[:, :4], 0)
            np.testing.assert_array_equal(result.confidence[:, :4], 0)

    def testConcurrentInferenceIsSerialized(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            image = np.zeros((32, 32, 3), np.uint8)
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(adapter.infer, [image] * 6))
            self.assertEqual(len(results), 6)
            self.assertEqual(adapter.models["faceParser"].maxActive, 1)
            self.assertEqual(adapter.loads, 1)

    def testMissingAssetsFailWithoutBackendLoading(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FaceParsingAdapter(Path(directory))
            with self.assertRaisesRegex(FileNotFoundError, "Missing face parsing assets"):
                adapter.loadModels()
            self.assertFalse(adapter.isLoaded)
            self.assertFalse(adapter.models)

    def testFailedPreloadCanRetryWithoutPublishingPartialState(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory))
            originalLoader = adapter.loadBackend

            def invalidSchema(paths):
                models, settings, metadata = originalLoader(paths)
                settings["label_names"] = ["unrecognized"]
                return models, settings, metadata

            adapter.loadBackend = invalidSchema
            with self.assertRaisesRegex(ValueError, "Unknown parser labels"):
                adapter.loadModels()
            self.assertFalse(adapter.isLoaded)
            self.assertFalse(adapter.models)
            self.assertFalse(adapter.parserSettings)
            adapter.loadBackend = originalLoader
            adapter.loadModels()
            self.assertTrue(adapter.isLoaded)
            self.assertEqual(adapter.loads, 2)

    def testTorchScriptParserPreloadsAndFreezesParameters(self):
        import torch

        class TinyParser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(12.))

            def forward(self, image):
                logits = torch.cat([torch.full_like(image[:, :1], -10),
                                    self.scale * (image[:, :1] - 0.5),
                                    self.scale * (0.5 - image[:, :1])], dim=1)
                return logits, image.mean()

        scripted = torch.jit.trace(TinyParser(), torch.zeros((1, 3, 32, 32)))
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory))
            originalLoader = adapter.loadBackend

            def scriptedBackend(paths):
                models, settings, metadata = originalLoader(paths)
                models["faceParser"] = scripted
                return models, settings, metadata

            adapter.loadBackend = scriptedBackend
            adapter.loadModels().loadModels()
            self.assertFalse(any(parameter.requires_grad for parameter in scripted.parameters()))
            result = adapter.infer(np.zeros((32, 32, 3), np.uint8))
            np.testing.assert_array_equal(result.labelMap, 3)
            self.assertEqual(adapter.loads, 1)

    def testInvalidImagesAndLandmarksFailClearly(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.makeAdapter(Path(directory)).loadModels()
            for image in (np.zeros((20, 30), np.uint8), np.zeros((20, 30, 3), np.float32),
                          np.zeros((0, 30, 3), np.uint8)):
                with self.assertRaisesRegex(ValueError, "RGB uint8 HWC"):
                    adapter.infer(image)
            for points in (np.zeros((68, 2)), np.full((5, 2), np.nan), np.zeros((5, 2))):
                with self.assertRaises(ValueError):
                    adapter.infer(np.zeros((20, 30, 3), np.uint8), faceLandmarks=points)


if __name__ == "__main__":
    unittest.main()
