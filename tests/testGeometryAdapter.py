"""CPU contract tests: cameras, model lifetime, validation and failure cleanup."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from faceAppearance.adapters.geometry import GeometryAdapter, resizeIntrinsics, validateImages


class FakeModel:
    def __init__(self) -> None:
        self.moves = []

    def eval(self):
        return self

    def requires_grad_(self, enabled):
        return self

    def to(self, device):
        self.moves.append(str(device))
        return self


class StubGeometryAdapter(GeometryAdapter):
    """Replace heavyweight kernels, keeping the real lifecycle and composition."""

    def __init__(self, projectRoot):
        super().__init__(projectRoot, device="cpu")
        self.loads = []

    def loadTemplate(self, paths):
        self.loads.append("template")

    def loadUvPredictor(self, checkpointPath):
        self.loads.append("uv")
        return FakeModel()

    def loadFaceModels(self, paths):
        self.loads.append("face")
        return FakeModel(), FakeModel()

    def loadVggt(self, checkpointPath):
        self.loads.append("vggt")
        return FakeModel()

    def loadFusion(self, checkpointPath):
        self.loads.append("fusion")
        return FakeModel()

    def predictUvs(self, images):
        return np.zeros((*images.shape[:3], 2), dtype=np.float32)

    def predictFaceMasks(self, images):
        return np.ones(images.shape[:3], dtype=np.float32)

    def predictPointMaps(self, images):
        count = len(images)
        return {
            "pointMaps": np.zeros_like(images),
            "cameraIntrinsics": np.repeat(np.eye(3)[None], count, axis=0).astype(np.float32),
            "cameraExtrinsics": np.zeros((count, 3, 4), dtype=np.float32),
        }

    def fuseGeometry(self, uvMaps, faceMasks, pointMaps):
        return {
            "vertices": np.zeros((3, 3), dtype=np.float32),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "uvs": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            "viewTransforms": [{"s": 1.0, "R": np.eye(3), "t": np.zeros(3)} for _ in uvMaps],
            "supportMask": np.ones((512, 512), dtype=bool),
        }


class GeometryAdapterTests(unittest.TestCase):
    def testOriginalFaceDetectionReturnsOriginalCoordinatesWithoutReloading(self):
        import torch

        class Detector(FakeModel):
            def __call__(self, image):
                self.inputShape = tuple(image.shape)
                return {"rects": torch.tensor([[100., 50., 300., 250.]]),
                        "points": torch.tensor([[[120., 100.], [240., 100.], [180., 150.],
                                                  [150., 200.], [220., 200.]]]),
                        "scores": torch.tensor([0.98])}

        adapter = GeometryAdapter(Path("."), device="cpu")
        detector = Detector()
        adapter.models["faceDetector"], adapter.isLoaded = detector, True
        result = adapter.detectInputFace(np.zeros((2048, 1024, 3), dtype=np.uint8))
        self.assertEqual(detector.inputShape, (1, 3, 1024, 512))
        np.testing.assert_allclose(result["boxXyxy"], [200, 100, 600, 500])
        np.testing.assert_allclose(result["landmarks"][0], [240.5, 200.5])
        self.assertEqual(result["selection"], {"strategy": "largestArea", "detectedCount": 1, "selectedIndex": 0})
        self.assertEqual(detector.moves, ["cpu", "cpu"])

    def testOriginalFaceDetectionSelectsLargestAndPreservesHalfPixelMapping(self):
        import torch

        class Detector(FakeModel):
            def __call__(self, image):
                return {"rects": torch.tensor([[10., 20., 40., 50.], [100., 50., 300., 250.]]),
                        "points": torch.tensor([[[20., 30.], [30., 30.], [25., 40.]],
                                                [[120., 100.], [240., 100.], [180., 150.]]]),
                        "scores": torch.tensor([.99, .85])}

        adapter = GeometryAdapter(Path("."), device="cpu")
        detector = Detector()
        adapter.models["faceDetector"], adapter.isLoaded = detector, True
        result = adapter.detectInputFace(np.zeros((2048, 1024, 3), dtype=np.uint8))
        np.testing.assert_allclose(result["boxXyxy"], [200, 100, 600, 500])
        np.testing.assert_allclose(result["landmarks"][0], [240.5, 200.5])
        self.assertAlmostEqual(result["score"], .85)
        self.assertEqual(result["selection"], {"strategy": "largestArea", "detectedCount": 2, "selectedIndex": 1})
        self.assertEqual(detector.moves, ["cpu", "cpu"])

    def testOriginalFaceDetectionRejectsNoFacesAndReleasesModel(self):
        import torch

        class Detector(FakeModel):
            def __call__(self, image):
                return {"rects": torch.empty(0), "scores": torch.empty(0)}

        adapter = GeometryAdapter(Path("."), device="cpu")
        detector = Detector()
        adapter.models["faceDetector"], adapter.isLoaded = detector, True
        with self.assertRaisesRegex(ValueError, "No faces detected"):
            adapter.detectInputFace(np.zeros((80, 80, 3), dtype=np.uint8))
        self.assertEqual(detector.moves, ["cpu", "cpu"])

    def testFaceMasksPassOnlySelectedFaceWithAlignedFieldsToParser(self):
        import torch

        class Detector(FakeModel):
            def __init__(self, count):
                super().__init__()
                allFaces = {"rects": torch.tensor([[1., 2., 4., 5.], [10., 10., 60., 70.]]),
                            "points": torch.tensor([[[2., 3.], [3., 3.], [2., 4.]],
                                                    [[20., 30.], [40., 30.], [30., 50.]]]),
                            "scores": torch.tensor([.99, .85]), "image_ids": torch.tensor([0, 0])}
                self.faces = {key: value[-count:] for key, value in allFaces.items()}

            def __call__(self, image):
                return self.faces

        class Parser(FakeModel):
            def __call__(self, image, faces):
                self.received = faces
                height, width = image.shape[-2:]
                logits = torch.full((1, 19, height, width), -100.)
                logits[:, 2] = 0.
                return {"seg": {"logits": logits}}

        for count in (1, 2):
            with self.subTest(faceCount=count):
                adapter = GeometryAdapter(Path("."), device="cpu")
                detector, parser = Detector(count), Parser()
                adapter.models.update(faceDetector=detector, faceParser=parser)
                result = adapter.predictFaceMasks(np.zeros((1, 80, 80, 3), np.float32))
                np.testing.assert_allclose(result, 1.)
                self.assertEqual(result.shape, (1, 80, 80))
                for key in ("rects", "points", "scores", "image_ids"):
                    torch.testing.assert_close(parser.received[key], detector.faces[key][-1:])
                    self.assertEqual(len(detector.faces[key]), count)
                self.assertEqual(detector.moves, ["cpu", "cpu"])
                self.assertEqual(parser.moves, ["cpu", "cpu"])

    def testFaceMaskDetectionFailureReleasesBothModels(self):
        import torch

        class Detector(FakeModel):
            def __call__(self, image):
                return {"rects": torch.empty(0), "scores": torch.empty(0)}

        adapter = GeometryAdapter(Path("."), device="cpu")
        detector, parser = Detector(), FakeModel()
        adapter.models.update(faceDetector=detector, faceParser=parser)
        with self.assertRaisesRegex(ValueError, "No faces detected"):
            adapter.predictFaceMasks(np.zeros((1, 80, 80, 3), np.float32))
        self.assertEqual(detector.moves, ["cpu", "cpu"])
        self.assertEqual(parser.moves, ["cpu", "cpu"])

    def testResizeIntrinsicsPreservesProjectedPixelCenters(self):
        intrinsics = np.array([[730, 1, 257.4], [0, 718, 256.2], [0, 0, 1]], dtype=np.float32)
        points = np.array([[0.1, -0.2, 1], [0.4, 0.2, 2], [-0.3, 0.1, 3]], dtype=np.float32)
        projected = points @ intrinsics.T
        projected = projected[:, :2] / projected[:, 2:]
        resized = points @ resizeIntrinsics(intrinsics).T
        resized = resized[:, :2] / resized[:, 2:]
        expected = (projected + 0.5) * (512 / 518) - 0.5
        np.testing.assert_allclose(resized, expected, rtol=1e-6, atol=3e-5)

    def testResizeIntrinsicsHandlesBatchWithoutMutatingInput(self):
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
        original = intrinsics.copy()
        resized = resizeIntrinsics(intrinsics)
        self.assertEqual(resized.shape, (4, 3, 3))
        np.testing.assert_array_equal(intrinsics, original)
        np.testing.assert_array_equal(resized[:, 2], original[:, 2])

    def testImageValidationRejectsInvalidInputs(self):
        invalidImages = [
            np.zeros((3, 512, 512, 3), dtype=np.float32),
            np.zeros((17, 512, 512, 3), dtype=np.float32),
            np.zeros((4, 3, 512, 512), dtype=np.float32),
            np.zeros((4, 512, 512, 3), dtype=np.uint8),
            np.full((4, 512, 512, 3), np.nan, dtype=np.float32),
            np.full((4, 512, 512, 3), 1.1, dtype=np.float32),
        ]
        for images in invalidImages:
            with self.subTest(shape=images.shape, dtype=images.dtype):
                with self.assertRaises(ValueError):
                    validateImages(images)

    def testMissingAssetsFailBeforeLoadingAnyModel(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = StubGeometryAdapter(Path(directory))
            with self.assertRaisesRegex(FileNotFoundError, "Missing geometry assets"):
                adapter.loadModels()
            self.assertEqual(adapter.loads, [])
            self.assertFalse(adapter.isLoaded)

    def testPreloadAndRepeatedReconstructionDoNotReloadWeights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = StubGeometryAdapter(Path(directory))
            for path in adapter.assetPaths().values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            images = np.zeros((4, 512, 512, 3), dtype=np.float32)
            with self.assertRaisesRegex(RuntimeError, "loadModels"):
                adapter.reconstruct(images)
            adapter.loadModels()
            firstModels = dict(adapter.models)
            adapter.loadModels()
            for _ in range(2):
                result = adapter.reconstruct(images)
                self.assertEqual(result["uvMaps"].shape, (4, 512, 512, 2))
                self.assertEqual(result["faceMasks"].shape, (4, 512, 512))
                self.assertEqual(result["cameraIntrinsics"].shape, (4, 3, 3))
                self.assertEqual(result["cameraExtrinsics"].shape, (4, 3, 4))
                self.assertEqual(len(result["viewTransforms"]), 4)
            self.assertEqual(adapter.loads, ["template", "uv", "face", "vggt", "fusion"])
            self.assertEqual(adapter.models, firstModels)

    def testStagedModelsReturnToCpuAfterFailure(self):
        adapter = GeometryAdapter(Path("."), device="cpu", memoryPolicy="staged")
        model = FakeModel()
        adapter.models["uvPredictor"] = model
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with adapter.activateModels("uvPredictor"):
                raise RuntimeError("synthetic failure")
        self.assertEqual(model.moves, ["cpu", "cpu"])

    def testPaddedImagePixelsNeverBecomeSurfaceOrTextureObservations(self):
        class MaskProbe(StubGeometryAdapter):
            def fuseGeometry(self, uvMaps, faceMasks, pointMaps):
                self.fusionMasks = faceMasks.copy()
                return super().fuseGeometry(uvMaps, faceMasks, pointMaps)

        adapter = MaskProbe(Path("."))
        adapter.isLoaded = True
        images = np.zeros((4, 512, 512, 3), dtype=np.float32)
        observed = np.ones(images.shape[:3], dtype=bool)
        observed[:, :, :25] = False
        result = adapter.reconstruct(images, validImageMasks=observed)
        np.testing.assert_array_equal(adapter.fusionMasks, observed)
        np.testing.assert_array_equal(result["faceMasks"], observed)
        with self.assertRaisesRegex(ValueError, "boolean"):
            adapter.reconstruct(images, validImageMasks=observed.astype(np.float32))

    def testVendorSimilarityUsesRowVectorConvention(self):
        from faceAppearance.vendor.uvFusion.utilities import umeyama

        generator = np.random.default_rng(24)
        source = generator.normal(size=(40, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 1.75 * (source @ rotation) + np.array([0.2, -0.4, 0.8])
        scale, fittedRotation, translation = umeyama(source, target)
        np.testing.assert_allclose(scale * (source @ fittedRotation) + translation, target, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
