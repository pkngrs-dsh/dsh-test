import copy

import numpy as np
import pytest

from faceAppearance.anatomicalBoundaries import (boundaryVisibilityDiagnostics, buildAnatomicalConstraints, createBoundaryAsset,
    extractMeshBoundaryLoops, lipInterfaceMask, meshTopologyHash, pairBoundaryRims,
    rasterizeBoundaryAnnotations, surfacePathBindings, validateBoundaryAsset)


def ringMesh():
    vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0],
                         [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
                      [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7]])
    return vertices, faces


def testBoundaryExtractionFindsActualHoleAndPerimeter():
    vertices, faces = ringMesh()
    result = extractMeshBoundaryLoops(vertices, faces)
    assert [loop.tolist() for loop in result["loops"]] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert result["diagnostics"]["boundaryEdgeCount"] == 8
    # Geometry changes do not change numbered boundary rings.
    moved = vertices + np.array([1., 8., -3.])
    assert [loop.tolist() for loop in extractMeshBoundaryLoops(moved, faces)["loops"]] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert result["topologySha256"] == meshTopologyHash(len(vertices), faces.astype(np.int32))


def testClosedMeshHasNoFakeBoundary():
    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    assert extractMeshBoundaryLoops(vertices, faces)["loops"] == []


def testRejectsAmbiguousDuplicateOrNonManifoldTopology():
    vertices, faces = ringMesh()
    with pytest.raises(ValueError, match="Duplicate"):
        extractMeshBoundaryLoops(vertices, np.concatenate([faces, faces[:1]]))
    with pytest.raises(ValueError, match="Non-manifold"):
        extractMeshBoundaryLoops(vertices, np.concatenate([faces, [[1, 4, 7]]]))
    # Two triangles touching only at a vertex produce a branching boundary.
    with pytest.raises(ValueError, match="branches"):
        extractMeshBoundaryLoops(vertices, np.array([[0, 1, 2], [0, 4, 5]]))


@pytest.mark.parametrize("kind", ["wrongVertexCount", "wrongFaces", "editedLoop", "repeatedName", "partialLoop"])
def testAssetFailsClosedOnStaleOrAlteredIndices(kind):
    vertices, faces = ringMesh()
    asset = createBoundaryAsset(vertices, faces, {0: {"name": "outer"}, 1: {"name": "hole"}})
    if kind == "wrongVertexCount":
        asset["vertexCount"] += 1
    elif kind == "wrongFaces":
        faces = faces[::-1]
    elif kind == "editedLoop":
        asset["boundaries"][1]["vertexIndices"][0] = 0
    elif kind == "repeatedName":
        asset["boundaries"][1]["name"] = "outer"
    else:
        asset["boundaries"][1]["vertexIndices"].pop()
    with pytest.raises(ValueError):
        validateBoundaryAsset(vertices, faces, asset)


def testAssetPreservesTopologyAcrossSubjectShapeAndNamesAreExplicit():
    vertices, faces = ringMesh()
    asset = createBoundaryAsset(vertices, faces, {1: {"name": "reviewedHole"}})
    result = validateBoundaryAsset(vertices*1.3, faces, asset)
    assert list(result) == ["reviewedHole"]
    np.testing.assert_array_equal(result["reviewedHole"], [4, 5, 6, 7])
    assert asset["reviewStatus"] == "unreviewed"
    with pytest.raises(ValueError):
        createBoundaryAsset(vertices, faces, {0: {"name": "same"}, 1: {"name": "same"}})


def testManualSurfaceFeatureCanFollowVisibleInteriorEdgesInsteadOfHole():
    vertices, faces = ringMesh()
    asset = {"topologySha256": meshTopologyHash(len(vertices), faces),
             "surfacePaths": [{"name": "visibleSeam", "vertexIndices": [0, 4, 5, 1], "closed": True}]}
    bindings = surfacePathBindings(vertices, faces, asset)
    assert bindings["visibleSeam"]["vertexIndices"].tolist() == [[0], [4], [5], [1]]
    np.testing.assert_array_equal(bindings["visibleSeam"]["barycentric"], np.ones((4, 1)))
    asset["surfacePaths"][0]["vertexIndices"] = [0, 6, 1]
    with pytest.raises(ValueError, match="connected"):
        surfacePathBindings(vertices, faces, asset)


def testLipInterfaceExcludesOuterLipAndDoesNotBridgeMouthOpening():
    labels = np.zeros((20, 30), np.uint8)
    labels[5:10, 7:24], labels[10:15, 7:24] = 12, 13
    confidence = np.ones(labels.shape)
    mask = lipInterfaceMask(labels, confidence)
    assert mask.sum() == 34
    assert mask[9:11, 7:24].all()
    assert not mask[5].any() and not mask[14].any()
    labels[9:11, 7:24] = 11
    assert not lipInterfaceMask(labels, confidence).any()


def testLipInterfaceRequiresBothSidesConfidence():
    labels = np.array([[12, 12, 12], [13, 13, 13]], np.uint8)
    confidence = np.ones(labels.shape)
    confidence[0] = .1
    assert not lipInterfaceMask(labels, confidence).any()


def testManualAnnotationRasterizesOpenAndClosedContours():
    contours = [{"name": "oralCenterline", "points": [[4, 6], [14, 6]], "closed": False, "confidence": .8},
                {"name": "nostril", "points": [[4, 12], [14, 12], [10, 16]], "closed": True}]
    result = rasterizeBoundaryAnnotations((20, 20), contours)
    assert result["oralCenterline"]["contourMask"].sum() == 11
    assert result["oralCenterline"]["confidence"][6, 8] == pytest.approx(.8)
    assert result["nostril"]["contourMask"][15, 9]


@pytest.mark.parametrize("mutation", ["outside", "noClosed", "duplicate", "collapsed"])
def testInvalidManualAnnotationIsRejected(mutation):
    contour = {"name": "mouth", "points": [[4, 6], [14, 6]], "closed": False}
    contours = [contour]
    if mutation == "outside":
        contour["points"][0][0] = -1
    elif mutation == "noClosed":
        contour.pop("closed")
    elif mutation == "duplicate":
        contours.append(copy.deepcopy(contour))
    else:
        contour["points"] = [[4, 6], [4.1, 6.1]]
    with pytest.raises(ValueError):
        rasterizeBoundaryAnnotations((20, 20), contours)


def testMouthRimPairsBindCenterWithoutCollapsingAperture():
    angle = np.linspace(0, 2*np.pi, 40, endpoint=False)
    pixels = np.stack([50+30*np.cos(angle), 40+3*np.sin(angle)], axis=1)
    binding = pairBoundaryRims(pixels, np.arange(40), sampleCount=14)
    points = (pixels[binding["vertexIndices"]]*binding["barycentric"][..., None]).sum(1)
    np.testing.assert_allclose(points[:, 1], 40, atol=1e-10)
    assert len(points) >= 10
    assert (np.ptp(pixels[binding["vertexIndices"], 1], axis=1) > 0).all()
    with pytest.raises(ValueError, match="thin"):
        pairBoundaryRims(np.stack([30*np.cos(angle), 30*np.sin(angle)], 1), np.arange(40))


def constraintFixture():
    vertices = np.array([[-1, -.1, 0], [-.3, -.1, 0], [.3, -.1, 0], [1, -.1, 0.]])
    camera = {"intrinsics": np.array([[20, 0, 20], [0, 20, 20], [0, 0, 1.]]),
              "rotation": np.eye(3), "translation": np.array([0, 0, 2.])}
    bindings = {"eyeRim": {"vertexIndices": np.arange(4)[:, None], "barycentric": np.ones((4, 1))}}
    contour = np.zeros((40, 40), bool)
    contour[20, 8:33] = True
    targets = [{"eyeRim": {"contourMask": contour, "confidence": contour.astype(float)*.8}}]
    visibility = np.ones((1, 4), bool)
    return vertices, [camera], bindings, targets, visibility


def testAnatomicalConstraintsAttachExactRimVerticesAndOriginalPhotoTargets():
    vertices, cameras, bindings, targets, visibility = constraintFixture()
    result = buildAnatomicalConstraints(vertices, cameras, bindings, targets, visibility)
    constraint = result["constraints"][0]
    np.testing.assert_array_equal(constraint["vertexIndices"], np.arange(4)[:, None])
    np.testing.assert_allclose(constraint["weights"], .8)
    assert result["diagnostics"]["groups"][0]["initialMeanDistancePixels"] == pytest.approx(1.)
    assert not np.shares_memory(constraint["vertexIndices"], bindings["eyeRim"]["vertexIndices"])


def testAnatomicalConstraintsRejectOcclusionMissingAndFarTargets():
    vertices, cameras, bindings, targets, visibility = constraintFixture()
    visibility[0, :2] = False
    result = buildAnatomicalConstraints(vertices, cameras, bindings, targets, visibility)
    assert not result["constraints"]
    visibility[:] = True
    assert not buildAnatomicalConstraints(vertices, cameras, bindings, targets, visibility,
                                          maxTargetDistance=.5)["constraints"]
    assert not buildAnatomicalConstraints(vertices, cameras, bindings, [{}], visibility)["constraints"]


def testAnatomicalConstraintsRequireAllVerticesForSpanningFeature():
    vertices, cameras, _, targets, visibility = constraintFixture()
    bindings = {"eyeRim": {"vertexIndices": np.array([[0, 1], [2, 3]]),
                           "barycentric": np.full((2, 2), .5)}}
    visibility[0, 1] = False
    result = buildAnatomicalConstraints(vertices, cameras, bindings, targets, visibility, minimumSamples=1)
    assert result["constraints"][0]["vertexIndices"].tolist() == [[2, 3]]


@pytest.mark.parametrize("depthValue,finiteCount,passedCount", [(4., 4, 4), (3.8, 4, 0), (np.inf, 0, 0)])
def testVisibilityDiagnosticsDistinguishOccludedGeometryFromHoleSampling(monkeypatch, depthValue, finiteCount, passedCount):
    vertices, faces = ringMesh()
    camera = {"intrinsics": np.array([[20, 0, 20], [0, 20, 20], [0, 0, 1.]]),
              "rotation": np.diag([1., -1., -1.]), "translation": np.array([0, 0, 4.])}
    monkeypatch.setattr("faceAppearance.texture.rasterizeDepth", lambda *args: np.full((40, 40), depthValue))
    report = boundaryVisibilityDiagnostics(vertices, faces, [camera], [(40, 40)], {"hole": np.arange(4, 8)})
    group = report["groups"][0]
    assert group["boundaryVertices"]["finiteRasterSupportCount"] == finiteCount
    assert group["boundaryVertices"]["depthTolerancePassedCount"] == passedCount
    assert group["boundaryVertices"]["frontFacingCount"] == 4
    assert group["incidentFaceCentroids"]["count"] == 8
