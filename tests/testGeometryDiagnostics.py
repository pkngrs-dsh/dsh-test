"""CPU-only contracts for lossless UFF diagnostic caching and cached-UV replay."""

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from faceAppearance.adapters.geometry import (
    GeometryAdapter, _writeDiagnosticStage, sampleDiagnosticPartials,
)
from testGeometryAdapter import FakeModel, StubGeometryAdapter


class GeometryDiagnosticTests(unittest.TestCase):
    def testOnlyDiagnosticAssetsAreRequiredAndFullLoadingReusesModels(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = StubGeometryAdapter(Path(directory))
            for name in ("mesh", "uvs", "vggt", "fusion"):
                path = adapter.assetPaths()[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            adapter.loadDiagnosticModels()
            oldModels = dict(adapter.models)
            adapter.loadDiagnosticModels()
            self.assertEqual(adapter.loads, ["template", "vggt", "fusion"])
            self.assertFalse(adapter.isLoaded)
            self.assertTrue(adapter.diagnosticModelsLoaded)
            self.assertEqual(set(adapter.models), {"vggt", "fusion"})
            with self.assertRaisesRegex(RuntimeError, "loadModels"):
                adapter.reconstruct(np.zeros((4, 512, 512, 3), np.float32))
            for path in adapter.assetPaths().values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            adapter.loadModels()
            self.assertEqual(adapter.loads, ["template", "vggt", "fusion", "uv", "face"])
            self.assertTrue(adapter.isLoaded)
            self.assertIs(adapter.models["vggt"], oldModels["vggt"])
            self.assertIs(adapter.models["fusion"], oldModels["fusion"])

    def testFailedSubsetLoadDoesNotClaimReadiness(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = StubGeometryAdapter(Path(directory))
            for name in ("mesh", "uvs", "vggt", "fusion"):
                path = adapter.assetPaths()[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(adapter, "loadFusion", side_effect=RuntimeError("bad checkpoint")):
                with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                    adapter.loadDiagnosticModels()
            self.assertFalse(adapter.isLoaded)
            self.assertFalse(adapter.diagnosticModelsLoaded)
            self.assertEqual(adapter.models, {})

    def testFailedFullLoadPreservesReadyDiagnosticModels(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = StubGeometryAdapter(Path(directory))
            for path in adapter.assetPaths().values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            adapter.loadDiagnosticModels()
            oldModels = dict(adapter.models)
            with patch.object(adapter, "loadFaceModels", side_effect=RuntimeError("bad parser")):
                with self.assertRaisesRegex(RuntimeError, "bad parser"):
                    adapter.loadModels()
            self.assertFalse(adapter.isLoaded)
            self.assertTrue(adapter.diagnosticModelsLoaded)
            self.assertEqual(adapter.models, oldModels)

    def testCachedReplayUsesProvidedMasksAndNeverCallsUvOrFacePrediction(self):
        adapter = StubGeometryAdapter(Path("."))
        adapter.diagnosticModelsLoaded = True
        images = np.zeros((4, 512, 512, 3), np.float32)
        uvs = np.full((4, 512, 512, 2), 0.25, np.float32)
        masks = np.ones((4, 512, 512), np.float32)
        masks[:, :, :20] = 0
        with patch.object(adapter, "predictUvs", side_effect=AssertionError("must not rerun")), \
                patch.object(adapter, "predictFaceMasks", side_effect=AssertionError("must not rerun")):
            result = adapter.reconstructFromCachedUvs(images, uvs, masks)
        self.assertIs(result["uvMaps"], uvs)
        self.assertIs(result["faceMasks"], masks)
        self.assertFalse(result["faceMasks"][:, :, :20].any())

    def testCachedReplayRejectsInvalidDataBeforeInference(self):
        adapter = StubGeometryAdapter(Path("."))
        adapter.diagnosticModelsLoaded = True
        images = np.zeros((4, 512, 512, 3), np.float32)
        uvs = np.zeros((4, 512, 512, 2), np.float32)
        masks = np.ones((4, 512, 512), np.float32)
        invalidCases = [(uvs[:3], masks), (uvs.astype(np.uint8), masks),
                        (uvs + np.nan, masks), (uvs + 2, masks),
                        (uvs, masks * 0), (uvs, masks + 1), (uvs, masks[:, :, :4])]
        with patch.object(adapter, "predictPointMaps", side_effect=AssertionError("must not infer")):
            for badUvs, badMasks in invalidCases:
                with self.subTest(uvShape=badUvs.shape, maskShape=badMasks.shape):
                    with self.assertRaises(ValueError):
                        adapter.reconstructFromCachedUvs(images, badUvs, badMasks)

    def testPartialSamplingMissingSupportIsNanAndRetainsSupportWeights(self):
        maps = np.ones((2, 9, 9, 3), np.float32) * np.array([0.2, 0.3, 0.7], np.float32)
        masks = np.zeros((2, 9, 9), bool)
        masks[0, :4, :4] = True
        uvs = np.array([[0.125, 0.125], [0.875, 0.875]], np.float32)
        sampled = sampleDiagnosticPartials(maps, masks, uvs)
        np.testing.assert_array_equal(sampled["partialValid"], [[True, False], [False, False]])
        np.testing.assert_allclose(sampled["partialVertices"][0, 0], [0.2, 0.3, 0.7], atol=1e-7)
        self.assertTrue(np.isnan(sampled["partialVertices"][~sampled["partialValid"]]).all())
        self.assertGreater(sampled["partialSampleWeight"][0, 0], 1)
        self.assertEqual(sampled["partialSampleWeight"][1, 0], 0)

    def testCacheHashesPreserveDtypesAndRejectOverwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {"nativeFloat64": np.array([1.234567890123456, np.nan], np.float64),
                      "mask": np.array([True, False]), "indices": np.array([1, 2], np.int64)}
            _writeDiagnosticStage(root, "example", arrays, {"source": "test"})
            with np.load(root / "example.npz") as cached:
                for name, values in arrays.items():
                    np.testing.assert_array_equal(cached[name], values)
                    self.assertEqual(cached[name].dtype, values.dtype)
            manifest = json.loads((root / "manifest.json").read_text())
            stage = manifest["stages"]["example"]
            self.assertEqual(stage["sha256"], hashlib.sha256((root / "example.npz").read_bytes()).hexdigest())
            self.assertEqual(stage["arrays"]["nativeFloat64"]["dataSha256"],
                             hashlib.sha256(arrays["nativeFloat64"].view(np.uint8)).hexdigest())
            with self.assertRaises(FileExistsError):
                _writeDiagnosticStage(root, "example", arrays, {})
            self.assertEqual(manifest["status"], "partial")

    def testVggtCacheSavesActualDepthAndConfidenceWithoutChangingOutput(self):
        class VggtModel(FakeModel):
            def aggregator(self, rgb):
                return [torch.ones(1)], 1

            def camera_head(self, tokens):
                return [torch.zeros((1, 4, 9))]

            def depth_head(self, tokens, rgb, patchStart, frames_chunk_size):
                self.chunkSize = frames_chunk_size
                depth = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1, 1).expand(1, 4, 518, 518, 1)
                confidence = torch.arange(7, 11, dtype=torch.float32).reshape(1, 4, 1, 1).expand(1, 4, 518, 518)
                return depth, confidence

        def poseDecode(pose, shape):
            return torch.eye(4)[None, None, :3].repeat(1, 4, 1, 1), torch.eye(3)[None, None].repeat(1, 4, 1, 1)

        def unproject(depth, extrinsics, intrinsics):
            return np.repeat(depth.numpy(), 3, axis=-1)

        modules = {"vggt.utils.pose_enc": types.SimpleNamespace(pose_encoding_to_extri_intri=poseDecode),
                   "vggt.utils.geometry": types.SimpleNamespace(unproject_depth_map_to_point_map=unproject)}
        adapter = GeometryAdapter(Path("."), device="cpu", depthChunkSize=2)
        adapter.models["vggt"] = model = VggtModel()
        images = np.zeros((4, 512, 512, 3), np.float32)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            original = adapter.predictPointMaps(images)
            diagnostic = adapter.predictPointMaps(images, diagnosticsDirectory=Path(directory))
            for name in original:
                np.testing.assert_array_equal(diagnostic[name], original[name])
            with np.load(Path(directory) / "vggt.npz") as cache:
                self.assertEqual(cache["depthNative518"].shape, (4, 518, 518, 1))
                self.assertEqual(cache["depthConfidenceNative518"].shape, (4, 518, 518))
                np.testing.assert_array_equal(cache["depthConfidenceNative518"][:, 0, 0], [7, 8, 9, 10])
                np.testing.assert_array_equal(cache["pointMaps512"], original["pointMaps"])
                np.testing.assert_array_equal(cache["pointMapsNative518"][:, 0, 0, 0], [1, 2, 3, 4])
        self.assertEqual(model.chunkSize, 2)
        self.assertEqual(model.moves, ["cpu", "cpu", "cpu", "cpu"])

    def testVendorMeanCaptureUsesModelMethodAndDoesNotChangeNetworkOutput(self):
        from faceAppearance.vendor.uvFusion.inference import infer_uv_point_map_with_model

        class MeanModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def compute_input_fusion(self, values, mask):
                self.calls += 1
                return (values * mask).sum(1) / mask.sum(1).clamp_min(1e-8), mask.sum(1) / values.shape[1]

            def forward(self, values, mask):
                return self.compute_input_fusion(values * mask, mask)[0] + 0.25

        def canonicalize(**kwargs):
            count = len(kwargs["uvs_array"])
            return kwargs["pred_uv_point_map"], kwargs["valid_mask"], {
                "c": 1., "R": np.eye(3), "t": np.zeros(3), "vertices_mask": np.ones(count, bool)}

        uvs = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 8), np.linspace(0.1, 0.9, 8)), -1).astype(np.float32)
        uvs = np.stack([uvs, uvs])
        masks = np.ones((2, 8, 8), np.float32)
        masks[1, :, :4] = 0
        points = np.ones((2, 8, 8, 3), np.float32)
        points[1] *= 3
        model = MeanModel()
        kwargs = {"device": "cpu", "uv_size": 8}
        with patch("faceAppearance.vendor.uvFusion.inference.canonicalize_partial_uv_point_map", canonicalize):
            original = infer_uv_point_map_with_model(model, uvs, masks, points, uvs[0, 0], points[0, 0], **kwargs)
            diagnostic = infer_uv_point_map_with_model(model, uvs, masks, points, uvs[0, 0], points[0, 0],
                                                       capture_input_fusion=True, **kwargs)
        np.testing.assert_array_equal(original["pred_uv_point_map"], diagnostic["pred_uv_point_map"])
        np.testing.assert_array_equal(diagnostic["model_input_mean"] + 0.25, diagnostic["pred_uv_point_map"])
        np.testing.assert_array_equal(diagnostic["model_input_coverage"], diagnostic["input_valid_masks"].mean(0))
        self.assertNotIn("model_input_mean", original)
        self.assertEqual(model.calls, 3)

    def testFusionStageCacheMatchesDefaultMeshAndSamplesRealConfidence(self):
        adapter = GeometryAdapter(Path("."), device="cpu")
        adapter.models["fusion"] = FakeModel()
        adapter.canonicalUvs = np.array([[.2, .2], [.4, .2], [.2, .4]], np.float32)
        adapter.canonicalVertices = np.array([[0., 0., .1], [.1, 0., .1], [0., .1, .1]], np.float32)
        adapter.canonicalFaces = np.array([[0, 1, 2]], np.int64)
        adapter.templateMask = np.ones((8, 8), bool)
        grid = np.stack(np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8)), -1).astype(np.float32)
        uvs = np.stack([grid] * 4)
        masks = np.ones((4, 8, 8), np.float32)
        pointMaps = np.concatenate([uvs, np.full((4, 8, 8, 1), .5, np.float32)], axis=-1)
        transforms = [{"c": np.float64(1.234567890123456), "R": np.eye(3, dtype=np.float32),
                       "t": np.zeros(3, np.float32), "vertices_mask": np.ones(3, bool)} for _ in range(4)]

        def infer(*args, **kwargs):
            result = {"input_uv_point_maps": pointMaps, "input_valid_masks": masks.astype(bool),
                      "input_support_mask": masks[0].astype(bool), "pred_valid_mask": adapter.templateMask,
                      "selected_view_indices": list(range(4)), "transforms": transforms,
                      "pred_uv_point_map": pointMaps.mean(0)}
            if kwargs.get("capture_input_fusion"):
                result.update(model_input_mean=pointMaps.mean(0), model_input_coverage=np.ones((8, 8), np.float32))
            return result

        with tempfile.TemporaryDirectory() as directory, \
                patch("faceAppearance.vendor.uvFusion.inference.infer_uv_point_map_with_model", infer):
            baseline = adapter.fuseGeometry(uvs, masks, pointMaps)
            result = adapter.fuseGeometry(uvs, masks, pointMaps, diagnosticsDirectory=Path(directory),
                                           depthConfidenceMaps=np.full(masks.shape, 9., np.float32))
            for key in ("vertices", "faces", "uvs", "supportMask"):
                np.testing.assert_array_equal(result[key], baseline[key])
            with np.load(Path(directory) / "fusion.npz") as cache:
                np.testing.assert_array_equal(cache["vertices"], baseline["vertices"])
                np.testing.assert_array_equal(cache["viewScales"], [item["c"] for item in transforms])
                self.assertTrue(cache["partialValid"].all())
                np.testing.assert_allclose(cache["partialDepthConfidence"], 9., atol=2e-6)
                np.testing.assert_array_equal(cache["modelInputMean"], pointMaps.mean(0))
        self.assertEqual(adapter.models["fusion"].moves, ["cpu", "cpu", "cpu", "cpu"])


if __name__ == "__main__":
    unittest.main()
