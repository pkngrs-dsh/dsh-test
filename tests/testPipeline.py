import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from faceAppearance.artifacts import exportMesh
from faceAppearance.config import PipelineConfig
from faceAppearance.pipeline import MissingCorrespondenceError, Pipeline


def testMissingIctCorrespondenceStopsBeforeLoadingGpu(tmp_path):
    pipeline = Pipeline(PipelineConfig(tmp_path, enableWildCap=True))
    with pytest.raises(MissingCorrespondenceError, match="registered ICT"):
        pipeline.loadModels()
    assert pipeline.geometryAdapter is None


@pytest.mark.parametrize("cropPolicy", ["centerSquare", "faceSquare"])
def testGeometryStagePreservesWorldToCanonicalContractAndHasNoMissingTexture(tmp_path, cropPolicy):
    class GeometryFixture:
        detectionCount = 0

        def detectInputFace(self, image):
            assert image.shape == (80, 80, 3) and image.dtype == np.uint8
            self.detectionCount += 1
            return {"boxXyxy": [-20, -10, 85, 95],
                    "landmarks": [[30, 25], [50, 25], [40, 45]], "score": 0.99}

        def reconstruct(self, images, *, validImageMasks=None):
            assert validImageMasks.shape == images.shape[:3]
            assert validImageMasks[:, 200:300, 200:300].all()
            assert not validImageMasks[:, 0, :].any()
            if cropPolicy == "faceSquare":
                assert validImageMasks.mean() < 0.5
            return {"vertices": np.eye(3, dtype=np.float32), "faces": np.array([[0, 1, 2]]),
                    "uvs": np.array([[0, 0.2], [1, 0.4], [0.2, 1.]]),
                    "cameraExtrinsics": np.zeros((4, 3, 4)),
                    "viewTransforms": [{"s": 2, "R": np.eye(3), "t": np.ones(3)}] * 4}

    paths = []
    for index in range(4):
        path = tmp_path / f"photo{index}.png"
        Image.new("RGB", (80, 80), (index * 20, 0, 0)).save(path)
        paths.append(path)
    pipeline = Pipeline(PipelineConfig(tmp_path, device="cpu", inputCropPolicy=cropPolicy))
    pipeline.loaded, pipeline.geometryAdapter = True, GeometryFixture()
    output = tmp_path / "output"
    report = pipeline.run(paths, output, "geometry")
    assert report["status"] == "succeeded"
    assert not (output / "delight.npz").exists()
    transforms = json.loads((output / "viewTransforms.json").read_text())
    assert len(transforms) == 4 and transforms[0]["s"] == 2
    assert pipeline.geometryAdapter.detectionCount == (4 if cropPolicy == "faceSquare" else 0)
    if cropPolicy == "faceSquare":
        assert "edgePaddingApplied" in report["inputs"][0]["qualityFlags"]
    assert "mtllib" not in (output / "face.obj").read_text()
    with pytest.raises(FileExistsError):
        pipeline.run(paths, output, "geometry")


def testExportObjFlipsOnlyVAndPreservesGeometryOrder(tmp_path):
    geometry = {"vertices": np.eye(3), "faces": np.array([[2, 0, 1]]),
                "uvs": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])}
    exportMesh(tmp_path, geometry)
    lines = (tmp_path / "face.obj").read_text().splitlines()
    assert "vt 0.1 0.8" in lines
    assert "f 3/3 1/1 2/2" in lines


def testIncompleteOutputCannotContaminateARetry(tmp_path):
    pipeline = Pipeline(PipelineConfig(tmp_path, device="cpu"))
    pipeline.loaded = True
    output = tmp_path / "failedEarlier"
    staleDir = output / "wildcapUff"
    staleDir.mkdir(parents=True)
    staleMaterial = staleDir / "diffuse.png"
    staleMaterial.write_bytes(b"partial previous request")
    with pytest.raises(FileExistsError, match="not empty"):
        pipeline.run([], output, "geometry")
    assert staleMaterial.read_bytes() == b"partial previous request"


@pytest.fixture
def materialPipeline(tmp_path):
    """Exercise real atlas resampling and exports without model/GPU work."""
    config = PipelineConfig(tmp_path, device="cpu", workingResolution=512,
                            wildCapResolution=512, enableWildCap=True, wildCapIterations=1)
    pipeline = Pipeline(config)
    yy, xx = np.mgrid[:512, :512]
    uvMap = np.stack([xx / 511, yy / 511], axis=-1).astype(np.float32)

    class RegistrationFixture:
        manifest = {"resultSha256": "reviewed-test-registration"}

        def __init__(self):
            self.reverseValid = np.ones((512, 512), bool)

        def atlasMap(self, direction):
            return {"sourceUvMap": uvMap,
                    "validMask": self.reverseValid if direction == "ictToUff" else np.ones((512, 512), bool)}

        def toIctGeometry(self, geometry):
            return {**geometry, "uvFaces": geometry["faces"], "geometryFrame": "uvFaceFusionCanonical"}

    class WildCapFixture:
        def __init__(self):
            self.received = None
            self.emptyResult = False

        def optimize(self, vertices, faces, uvs, image, validMask, **options):
            self.received = options
            normal = np.zeros_like(image)
            normal[..., 2] = 1
            mask = np.zeros_like(validMask) if self.emptyResult else validMask.copy()
            return {"diffuse": np.full_like(image, 0.75), "specular": np.full(image.shape[:2], 0.25),
                    "tangentNormal": normal, "validMask": mask,
                    "diagnostics": {"status": "experimental"}}

    registration = RegistrationFixture()
    pipeline.registrationAssets[512] = registration
    adapter = WildCapFixture()
    pipeline.wildCapAdapter = adapter
    geometry = {"vertices": np.eye(3), "faces": np.array([[0, 1, 2]]),
                "uvs": np.array([[0, 0], [1, 0], [0, 1]])}
    coverage = np.ones((512, 512), bool)
    coverage[100:120, 100:120] = False
    texture = {"albedo": np.full((512, 512, 3), 0.25, np.float32), "coverageMask": coverage}
    return pipeline, registration, adapter, geometry, texture


@pytest.mark.parametrize("provideMask", [False, True])
def testRefinementPassesExplicitIctMaskAndPreservesUnobservedCoverage(materialPipeline, tmp_path, provideMask):
    pipeline, _, adapter, geometry, texture = materialPipeline
    artifactMask = np.zeros((512, 512), bool) if provideMask else None
    if provideMask:
        artifactMask[50:100, 100:150] = True
    output = tmp_path / "result"
    diagnostics = pipeline.refineMaterials(geometry, texture, output, artifactMaskIct=artifactMask)
    assert adapter.received["artifactMask"] is artifactMask
    assert adapter.received["geometryFrame"] == "uvFaceFusionCanonical"
    np.testing.assert_array_equal(adapter.received["uvFaces"], geometry["faces"])
    with np.load(output / "wildcapUff/materials.npz", allow_pickle=False) as materials:
        valid = materials["validMask"]
        assert not valid[101:119, 101:119].any()
        assert valid[250, 250]
        np.testing.assert_array_equal(materials["baseWithWildCap"][~valid], texture["albedo"][~valid])
        assert materials["baseWithWildCap"][250, 250, 0] == pytest.approx(0.75, abs=1e-6)
        assert diagnostics["refinedUffPixelCount"] == int(valid.sum())


@pytest.mark.parametrize("emptyStage", ["ict", "uff"])
def testRefinementCannotSucceedWithoutObservedReturnedPixels(materialPipeline, tmp_path, emptyStage):
    pipeline, registration, adapter, geometry, texture = materialPipeline
    if emptyStage == "ict":
        adapter.emptyResult = True
        message = "no valid observed ICT"
    else:
        registration.reverseValid[:] = False
        message = "survived the ICT-to-UFF transfer"
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=message):
        pipeline.refineMaterials(geometry, texture, output)
    assert not (output / "wildcapUff/baseWithWildCap.png").exists()
