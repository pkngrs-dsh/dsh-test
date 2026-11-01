import struct

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from faceAppearance.colmapTexture import (loadTexturedMesh, prepareWorkspace, rebakeAtlas,
                                         sourceSupportFromMarkers, viewMarkerColors)


def plane():
    return {"vertices": np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
            "faces": np.array([[0, 1, 2], [0, 2, 3]]),
            "uvs": np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])}


def writePly(path, geometry, topLeftFaceUvs, binary=False):
    vertices, faces = geometry["vertices"], geometry["faces"]
    colmapUvs = np.asarray(topLeftFaceUvs).copy()
    colmapUvs[..., 1] = 1 - colmapUvs[..., 1]
    formatName = "binary_little_endian" if binary else "ascii"
    header = (f"ply\nformat {formatName} 1.0\ncomment TextureFile texture.png\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
              "property list uchar float texcoord\nend_header\n")
    with path.open("wb") as output:
        output.write(header.encode("ascii"))
        for vertex in vertices:
            output.write(struct.pack("<3f", *vertex) if binary else
                         (" ".join(map(str, vertex)) + "\n").encode("ascii"))
        for face, uv in zip(faces, colmapUvs):
            output.write(struct.pack("<B3iB6f", 3, *face, 6, *uv.ravel()) if binary else
                         ("3 " + " ".join(map(str, face)) + " 6 " +
                          " ".join(map(str, uv.ravel())) + "\n").encode("ascii"))


@pytest.mark.parametrize("binary", [False, True])
def testRectangularAtlasUsesEdgeUvAndOneVerticalFlip(tmp_path, binary):
    geometry = plane()
    height, width = 5, 9
    # Map original UV corners to actual source texel centers. A (size-1)
    # sampler or a second vertical flip visibly changes this coordinate ramp.
    edgeUvs = (geometry["uvs"] * [width - 1, height - 1] + .5) / [width, height]
    path = tmp_path / "mesh.ply"
    writePly(path, geometry, edgeUvs[geometry["faces"]], binary)
    mesh = loadTexturedMesh(path, geometry)
    yy, xx = np.mgrid[:height, :width]
    atlas = np.stack([xx / (width - 1), yy / (height - 1), (xx + 2 * yy) / 16], axis=-1)
    result = rebakeAtlas(geometry, mesh, atlas, resolution=5)
    targetY, targetX = np.mgrid[:5, :5]
    expected = np.stack([targetX / 4, targetY / 4, (targetX * 2 + targetY * 2) / 16], axis=-1)
    np.testing.assert_allclose(result["rawRebaked"], expected, atol=2e-7)
    assert result["assignedMask"].all()
    assert result["coverageMask"].all()
    assert mesh["textureFile"] == "texture.png"
    assert "confidence" not in result and "sourceIndex" not in result
    assert "atlasAssignmentOnly" in result["diagnostics"]["coverageMeaning"]


def testOriginalUvFaceIndicesSupportSeams(tmp_path):
    geometry = plane()
    geometry["uvs"] = np.concatenate([np.zeros((3, 2)), geometry["uvs"]])
    geometry["uvFaces"] = geometry["faces"] + 3
    sourceUv = (plane()["uvs"] * 3 + .5) / 4
    path = tmp_path / "mesh.ply"
    writePly(path, geometry, sourceUv[geometry["faces"]])
    result = rebakeAtlas(geometry, loadTexturedMesh(path, geometry),
                         np.full((4, 4, 3), .25), resolution=4)
    assert result["coverageMask"].all()
    np.testing.assert_allclose(result["albedo"], .25)


def testBlackColorIsObservedAndExternalMaskControlsSupport(tmp_path):
    geometry = plane()
    sourceUv = (geometry["uvs"] * 3 + .5) / 4
    path = tmp_path / "mesh.ply"
    writePly(path, geometry, sourceUv[geometry["faces"]])
    support = np.ones((4, 4)); support[1, 2] = 0
    result = rebakeAtlas(geometry, loadTexturedMesh(path, geometry),
                         np.zeros((4, 4, 3), np.uint8), resolution=4,
                         coverage={"coverageMask": support, "confidence": support * .75,
                                   "sourceIndex": np.full((4, 4), 2)})
    np.testing.assert_array_equal(result["coverageMask"], support.astype(bool))
    assert result["assignedMask"].all()
    assert not result["albedo"].any()
    assert result["confidence"][1, 2] == 0 and result["sourceIndex"][1, 2] == -1
    assert result["sourceIndex"][0, 0] == 2


def testUnassignedFaceCannotBecomePhotoCoverage(tmp_path):
    geometry = plane()
    sourceUv = (geometry["uvs"] * 3 + .5) / 4
    faceUvs = sourceUv[geometry["faces"]]
    faceUvs[1] = [0, 1]  # Raw COLMAP all-zero triangle after V flip.
    path = tmp_path / "mesh.ply"
    writePly(path, geometry, faceUvs)
    mesh = loadTexturedMesh(path, geometry)
    np.testing.assert_array_equal(mesh["validFaces"], [True, False])
    result = rebakeAtlas(geometry, mesh, np.ones((4, 4, 3)), resolution=4,
                         coverage=np.ones((4, 4), bool))
    assert result["coverageMask"][0, 3]
    assert not result["coverageMask"][3, 0]
    np.testing.assert_array_equal(result["albedo"][3, 0], [0, 0, 0])


@pytest.mark.parametrize("change", ["faces", "vertices"])
def testChangedTopologyOrVertexOrderRejected(tmp_path, change):
    geometry = plane()
    changed = {key: value.copy() for key, value in geometry.items()}
    changed[change] = changed[change][::-1]
    path = tmp_path / "mesh.ply"
    writePly(path, changed, geometry["uvs"][geometry["faces"]])
    with pytest.raises(ValueError, match="changed"):
        loadTexturedMesh(path, geometry)


def testTruncatedBinaryRejected(tmp_path):
    geometry = plane()
    path = tmp_path / "mesh.ply"
    writePly(path, geometry, geometry["uvs"][geometry["faces"]], binary=True)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        loadTexturedMesh(path, geometry)


def testCameraExportRetainsFxFyPoseAndSourceViewIndices(tmp_path):
    geometry = plane()
    rotation = Rotation.from_euler("xyz", [15, -12, 7], degrees=True).as_matrix()
    camera = {"intrinsics": [[83., 0, 4.1], [0, 72., 2.3], [0, 0, 1]],
              "rotation": rotation, "translation": [0.1, -0.2, 4],
              "width": 9, "height": 5, "viewIndex": 1}
    image = np.full((5, 9, 3), .25, np.float32)
    output = prepareWorkspace(tmp_path / "workspace", geometry,
                              [{"valid": False}, camera], [None, image])
    assert output["activeViewIndices"] == [1]
    record = output["cameras"][0]
    assert record["colmapImageId"] == 1 and record["sourceViewIndex"] == 1
    assert output["exportProjectionChecks"][0]["maxPixelError"] < 1e-10
    modelLine = (tmp_path / "workspace/sparse/cameras.txt").read_text().splitlines()[-1].split()
    assert modelLine[:4] == ["1", "PINHOLE", "9", "5"]
    np.testing.assert_allclose(np.array(modelLine[4:], float), [83, 72, 4.1, 2.3])
    np.testing.assert_array_equal(np.asarray(Image.open(record["imagePath"])), 64)
    poseLines = (tmp_path / "workspace/sparse/images.txt").read_text().splitlines()
    assert poseLines[-1] == ""  # Required empty POINTS2D observations line.
    with pytest.raises(FileExistsError):
        prepareWorkspace(tmp_path / "workspace", geometry, [camera], [None, image])


def testInvalidCameraRejectedBeforeWritingWorkspace(tmp_path):
    camera = {"intrinsics": [[80., 1, 4], [0, 75., 2], [0, 0, 1]],
              "rotation": np.eye(3), "translation": [0, 0, 4]}
    with pytest.raises(ValueError, match="PINHOLE"):
        prepareWorkspace(tmp_path / "workspace", plane(), [camera], [np.zeros((5, 9, 3))])
    assert not (tmp_path / "workspace").exists()


def markerMesh(geometry, faceUvs):
    return {"vertices": geometry["vertices"], "faces": geometry["faces"],
            "faceUvs": np.asarray(faceUvs), "validFaces": np.ones(len(geometry["faces"]), bool),
            "uvConvention": "topLeftPixelEdges"}


def testMarkerPaletteIsSeparatedFromOtherViewsAndBackground():
    palette = viewMarkerColors(64).astype(float)
    distances = np.max(np.abs(palette[:, None] - palette[None]), axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    assert distances.min() >= 64
    assert np.max(palette, axis=1).min() >= 48
    np.testing.assert_array_equal(viewMarkerColors(4), palette[:4].astype(np.uint8))
    with pytest.raises(ValueError):
        viewMarkerColors(0)


def testMarkersRecoverOriginalViewIndicesButDoNotGrantPhotoSupport():
    geometry = plane()
    faceUvs = np.array([[[.1, .1], [.4, .1], [.4, .9]],
                       [[.6, .1], [.9, .9], [.6, .9]]])
    palette = viewMarkerColors(2)
    atlas = np.empty((16, 32, 3), np.uint8)
    atlas[:, :16] = palette[0]; atlas[:, 16:] = palette[1]
    camera = {"intrinsics": [[2., 0, 3], [0, 3., 2], [0, 0, 1]],
              "rotation": np.eye(3), "translation": [0, 0, 2]}
    cameras = [{**camera, "viewIndex": 0}, {"valid": False}, {**camera, "viewIndex": 2}]
    masks = [np.zeros((8, 9)), None, np.ones((8, 9))]
    result = sourceSupportFromMarkers(geometry, markerMesh(geometry, faceUvs), atlas,
                                      cameras, masks, [2, 0], resolution=5)
    np.testing.assert_array_equal(result["faceSourceIndex"], [2, 0])
    assert result["coverageMask"][0, 4] and not result["coverageMask"][4, 0]
    assert result["sourceIndex"][0, 4] == 2 and result["sourceIndex"][4, 0] == -1
    assert result["diagnostics"]["decodedFaceCount"] == 2
    assert result["diagnostics"]["markerInpaintProvesPhotoSupport"] is False


def testSourceMaskCoordinatesUseAffineProjectedCornersAndSeparateFxFy():
    geometry = {"vertices": np.array([[0., 0., 1.], [4., 0., 2.], [0., 4., 4.]]),
                "faces": np.array([[0, 1, 2]]), "uvs": np.array([[0., 0.], [1., 0.], [0., 1.]])}
    mesh = markerMesh(geometry, [[[.2, .2], [.8, .2], [.2, .8]]])
    atlas = np.broadcast_to(viewMarkerColors(1)[0], (16, 16, 3)).copy()
    camera = {"intrinsics": [[2., 0, 1], [0, 3., 1], [0, 0, 1]],
              "rotation": np.eye(3), "translation": [0, 0, 0]}
    mask = np.zeros((8, 8)); mask[:, :4] = 1
    result = sourceSupportFromMarkers(geometry, mesh, atlas, [camera], [mask], [0], resolution=3)
    # Midpoint of edge 0->1: affine projected x=3 (valid); projecting the
    # interpolated 3D point instead yields x=3.666... (invalid interpolation).
    assert result["coverageMask"][0, 1]
    assert result["sampledValidity"][0, 1] == 1
    assert not result["coverageMask"][0, 2]


def testMarkerConsensusRejectsMixedColorsAndTinyUnresolvedFaces():
    geometry = plane()
    faceUvs = np.array([[[.1, .1], [.9, .1], [.9, .9]],
                       [[.499, .2], [.501, .21], [.499, .21]]])
    atlas = np.empty((16, 16, 3), np.uint8)
    palette = viewMarkerColors(2)
    atlas[:, :8] = palette[0]; atlas[:, 8:] = palette[1]
    camera = {"intrinsics": [[2., 0, 3], [0, 3., 2], [0, 0, 1]],
              "rotation": np.eye(3), "translation": [0, 0, 2]}
    result = sourceSupportFromMarkers(geometry, markerMesh(geometry, faceUvs), atlas,
                                      [camera, camera], [np.ones((8, 9))] * 2, [0, 1], resolution=5)
    assert (result["faceSourceIndex"] == -1).all()
    assert not result["coverageMask"].any()
    assert result["diagnostics"]["undecodedAssignedFaceCount"] == 2


def testColmapSourceBilinearDomainExcludesFinalRowAndColumn():
    geometry = plane()
    mesh = markerMesh(geometry, (geometry["uvs"][geometry["faces"]] * .6 + .2))
    atlas = np.broadcast_to(viewMarkerColors(1)[0], (16, 16, 3)).copy()
    camera = {"intrinsics": [[3., 0, 0], [0, 3., 0], [0, 0, 1]],
              "rotation": np.eye(3), "translation": [0, 0, 1]}
    result = sourceSupportFromMarkers(geometry, mesh, atlas, [camera], [np.ones((4, 4))], [0], resolution=4)
    assert result["coverageMask"][:3, :3].all()
    assert not result["coverageMask"][-1].any() and not result["coverageMask"][:, -1].any()
