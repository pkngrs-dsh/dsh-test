import json

import numpy as np
import pytest

from faceAppearance import camera
from faceAppearance.cameraDiagnostics import compareSavedCameras


def makeGeometry(viewCount=1, size=50):
    ys, xs = np.mgrid[:size, :size]
    uvMap = np.stack([xs, ys], axis=-1).astype(np.float32) / 511
    intrinsics = np.diag([2., 2., 1.])
    return {
        "vertices": np.array([[0., 0., 2.], [511., 0., 2.], [511., 511., 2.], [0., 511., 2.]]),
        "faces": np.array([[0, 1, 2], [0, 2, 3]]),
        "uvs": np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        "uvMaps": np.repeat(uvMap[None], viewCount, axis=0),
        "faceMasks": np.ones((viewCount, size, size), dtype=np.float32),
        "cameraIntrinsics": np.repeat(intrinsics[None], viewCount, axis=0),
    }


def makeCamera(translation=(0, 0, 0), **overrides):
    return {"intrinsics": np.diag([2., 2., 1.]), "rotation": np.eye(3),
            "translation": np.asarray(translation, dtype=float), "valid": True,
            "source": "testSavedCamera", **overrides}


def testIdentityAndKnownProjectionDisplacementWithoutRefitting(monkeypatch):
    def rejectFit(*args, **kwargs):
        raise AssertionError("Diagnostics must not fit any camera")

    monkeypatch.setattr(camera, "fitCamera", rejectFit)
    monkeypatch.setattr(camera, "fitViewCameras", rejectFit)
    geometry = makeGeometry()
    baseline = makeCamera(source="nativeSaved")
    shifted = makeCamera((4, 0, 0), source="fittedSaved")
    result = compareSavedCameras(geometry, [baseline], [shifted])[0]
    assert result["notGroundTruth"] is True
    assert result["pointCount"] == 100
    assert result["heldOutPointCount"] == 20
    assert result["native"]["source"] == "nativeSaved"
    assert result["fitted"]["source"] == "fittedSaved"
    assert result["native"]["allMedianErrorPixels"] < 1e-5
    assert result["fitted"]["allMedianErrorPixels"] == pytest.approx(4, abs=1e-5)
    assert result["fitted"]["positiveDepthFraction"] == 1
    assert result["nativeVsFittedVertices"]["maximumDisplacementPixels"] == pytest.approx(4)
    assert result["nativeVsFittedVertices"]["bothPositiveDepthVertexCount"] == 4
    same = compareSavedCameras(geometry, [baseline], [baseline])[0]
    assert same["nativeVsFittedVertices"]["maximumDisplacementPixels"] == 0
    json.dumps(result, allow_nan=False)


def testCorrespondencesAndHeldOutMatchOriginalFitViewCameras(monkeypatch):
    geometry = makeGeometry(viewCount=2)
    geometry["faceMasks"][0, 0, 0] = 0.5
    geometry["uvMaps"][0, 5, 5] = np.nan
    geometry["uvMaps"][0, 5, 10] = [-0.1, 0]
    geometry["uvMaps"][0, 5, 15] = [1.1, 0]
    geometry["uvMaps"][1, ..., 0] += np.arange(50)[:, None] / 511
    captured = []

    def captureFit(points, pixels, intrinsics, maxErrorPixels, seed):
        captured.append((points.copy(), pixels.copy(), seed))
        return {"valid": False}

    monkeypatch.setattr(camera, "fitCamera", captureFit)
    camera.fitViewCameras(geometry, seed=71)
    cameras = [makeCamera((0, 0, 2)), makeCamera((0, 0, 2))]
    results = compareSavedCameras(geometry, cameras, cameras, seed=71)
    assert len(captured) == 2
    for viewIndex, (points, pixels, seed) in enumerate(captured):
        projected, _ = camera.projectPoints(points, cameras[viewIndex]["intrinsics"], np.eye(3), [0, 0, 2])
        errors = np.linalg.norm(projected - pixels, axis=1)
        heldOut = np.random.default_rng(seed).permutation(len(points))[:max(12, len(points) // 5)]
        result = results[viewIndex]
        assert result["pointCount"] == len(points)
        assert result["seed"] == seed
        assert result["native"]["allMedianErrorPixels"] == pytest.approx(np.median(errors))
        assert result["native"]["allP90ErrorPixels"] == pytest.approx(np.percentile(errors, 90))
        assert result["native"]["heldOutMedianErrorPixels"] == pytest.approx(np.median(errors[heldOut]))
        assert result["native"]["heldOutP90ErrorPixels"] == pytest.approx(np.percentile(errors[heldOut], 90))
    assert results[0]["pointCount"] == 96


def testMissingCameraAndSavedRejectedPoseRemainExplicit():
    geometry = makeGeometry()
    failed = {"valid": False, "reason": "Robust camera fitting failed"}
    result = compareSavedCameras(geometry, [makeCamera(valid=False)], [failed])[0]
    assert result["native"]["status"] == "measured"
    assert result["native"]["savedValid"] is False
    assert result["fitted"]["status"] == "unavailable"
    assert result["fitted"]["allMedianErrorPixels"] is None
    assert result["nativeVsFittedVertices"]["maximumDisplacementPixels"] is None
    json.dumps(result, allow_nan=False)


def testBehindCameraAndEmptySupportDoNotReportMisleadingErrors():
    geometry = makeGeometry()
    result = compareSavedCameras(geometry, [makeCamera((0, 0, -3))], [makeCamera()])[0]
    assert result["native"]["positiveDepthFraction"] == 0
    assert result["native"]["heldOutPositiveDepthFraction"] == 0
    assert result["native"]["status"] == "noPositiveDepthCorrespondences"
    assert result["native"]["allMedianErrorPixels"] is None
    assert result["nativeVsFittedVertices"]["status"] == "noSharedPositiveDepthVertices"
    geometry["faceMasks"][:] = 0
    empty = compareSavedCameras(geometry, [makeCamera()], [makeCamera()])[0]
    assert empty["pointCount"] == empty["heldOutPointCount"] == 0
    assert empty["native"]["status"] == "noCorrespondences"
    assert empty["native"]["positiveDepthFraction"] is None
    assert empty["nativeVsFittedVertices"]["maximumDisplacementPixels"] == 0
    json.dumps([result, empty], allow_nan=False)


def testFewCorrespondencesHaveNoFictitiousOriginalHeldOutSet():
    result = compareSavedCameras(makeGeometry(size=20), [makeCamera()], [makeCamera()])[0]
    assert result["pointCount"] == 16
    assert result["heldOutPointCount"] == 0
    assert result["native"]["allMedianErrorPixels"] < 1e-5
    assert result["native"]["heldOutMedianErrorPixels"] is None


def testRejectsMismatchedViewsAndMaskDimensions():
    geometry = makeGeometry()
    with pytest.raises(ValueError, match="counts must match"):
        compareSavedCameras(geometry, [], [makeCamera()])
    geometry["faceMasks"] = geometry["faceMasks"][0]
    with pytest.raises(ValueError, match="matching faceMasks"):
        compareSavedCameras(geometry, [makeCamera()], [makeCamera()])
