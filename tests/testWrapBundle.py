"""Structural and integrity regression checks for the manual-registration handoff."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "tools"))
from prepareWrapBundle import ObjMesh, prepareBundle, readObj, sha256File, writeObj
from validateWrapResult import validateResult


class WrapBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporaryDir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporaryDir.cleanup)
        self.tempRoot = Path(self.temporaryDir.name)
        self.sourcePath = self.tempRoot / "source.ply"
        self.uvPath = self.tempRoot / "sourceUvs.npy"
        self.ictPath = self.tempRoot / "sourceIct.obj"
        self.bundleDir = self.tempRoot / "bundle"
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [0, 3, 1]])
        self.texCoords = np.array([[0.1, 0.1], [0.8, 0.1], [0.1, 0.8], [0.8, 0.8]])
        trimesh.Trimesh(vertices, faces, process=False).export(self.sourcePath)
        np.save(self.uvPath, self.texCoords)
        # UV indices deliberately differ from vertex indices, as happens at real seams.
        self.originalMesh = ObjMesh(vertices * 10 + 3, self.texCoords, faces, np.array([[2, 0, 1], [2, 3, 0]]))
        writeObj(self.ictPath, self.originalMesh)
        self.manifest = prepareBundle(self.sourcePath, self.uvPath, self.ictPath, self.bundleDir)
        self.resultPath = self.tempRoot / "ictRegisteredToUvFaceFusion.obj"
        self.resultMesh = ObjMesh(vertices.copy(), self.texCoords.copy(), faces.copy(), self.originalMesh.uvFaces.copy())
        writeObj(self.resultPath, self.resultMesh)

    def testNativeGeometryAndUvsArePreserved(self) -> None:
        fixedMesh = readObj(self.bundleDir / "uvFaceFusionFixed.obj")
        loadedSource = trimesh.load(self.sourcePath, process=False)
        np.testing.assert_array_equal(fixedMesh.vertices, loadedSource.vertices)
        np.testing.assert_array_equal(fixedMesh.faces, loadedSource.faces)
        np.testing.assert_array_equal(fixedMesh.texCoords, self.texCoords)
        floatingMesh = readObj(self.bundleDir / "ictFloating.obj")
        np.testing.assert_array_equal(floatingMesh.vertices, self.originalMesh.vertices)
        np.testing.assert_array_equal(floatingMesh.uvFaces, self.originalMesh.uvFaces)
        self.assertEqual(sha256File(self.ictPath), sha256File(self.bundleDir / "originals/open_eye_template.obj"))
        self.assertEqual(self.manifest["automaticTransformsApplied"], [])

    def testStructureDoesNotApproveSemanticRegistration(self) -> None:
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertTrue(result["structuralValid"], result)
        self.assertFalse(result["registrationAccepted"])
        self.assertTrue(result["semanticReviewRequired"])

    def testMissingResultOnlyChecksInputs(self) -> None:
        result = validateResult(self.bundleDir)
        self.assertTrue(result["inputIntegrityValid"])
        self.assertIsNone(result["structuralValid"])
        self.assertFalse(result["registrationAccepted"])

    def testUnregisteredTemplateRemainsUnaccepted(self) -> None:
        result = validateResult(self.bundleDir, self.bundleDir / "ictFloating.obj")
        self.assertFalse(result["geometryChanged"])
        self.assertFalse(result["registrationAccepted"])
        self.assertTrue(any("unregistered" in warning for warning in result["warnings"]))

    def testModifiedFixedTemplateFailsIntegrity(self) -> None:
        fixedPath = self.bundleDir / "uvFaceFusionFixed.obj"
        fixedPath.write_text(fixedPath.read_text() + "# altered\n")
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertFalse(result["inputIntegrityValid"])
        self.assertFalse(result["structuralValid"])

    def testReorderedFacesAreRejected(self) -> None:
        self.resultMesh.faces = self.resultMesh.faces[::-1]
        writeObj(self.resultPath, self.resultMesh)
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertFalse(result["structuralValid"])
        self.assertIn("Triangle vertex indices/order/winding changed", result["errors"])

    def testChangedUvSeamIndicesAreRejected(self) -> None:
        self.resultMesh.uvFaces = self.resultMesh.faces.copy()
        writeObj(self.resultPath, self.resultMesh)
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertFalse(result["structuralValid"])
        self.assertIn("Triangle UV indices/order changed", result["errors"])

    def testUvEditsAndFlipsAreRejected(self) -> None:
        self.resultMesh.texCoords[:, 1] = 1 - self.resultMesh.texCoords[:, 1]
        writeObj(self.resultPath, self.resultMesh)
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertFalse(result["structuralValid"])
        self.assertIn("Texture-coordinate values changed", result["errors"])

    def testNonfiniteCoordinatesAreRejected(self) -> None:
        self.resultMesh.vertices[0, 0] = float("nan")
        writeObj(self.resultPath, self.resultMesh)
        self.assertFalse(validateResult(self.bundleDir, self.resultPath)["structuralValid"])

    def testCollapsedTrianglesAreRejected(self) -> None:
        self.resultMesh.vertices[1] = self.resultMesh.vertices[0]
        writeObj(self.resultPath, self.resultMesh)
        result = validateResult(self.bundleDir, self.resultPath)
        self.assertFalse(result["structuralValid"])
        self.assertGreater(result["nearZeroTriangleCount"], 0)

    def testAddedVertexIsRejected(self) -> None:
        self.resultMesh.vertices = np.vstack([self.resultMesh.vertices, [[0.2, 0.2, 0.2]]])
        writeObj(self.resultPath, self.resultMesh)
        self.assertFalse(validateResult(self.bundleDir, self.resultPath)["structuralValid"])

    def testBundleCannotBeSilentlyReplaced(self) -> None:
        with self.assertRaises(FileExistsError):
            prepareBundle(self.sourcePath, self.uvPath, self.ictPath, self.bundleDir)

    def testMissingOrInvalidIndicesFail(self) -> None:
        for invalidFace in ["f 1 2 3", "f 0/1 2/2 3/3", "f 1/1 2/2 99/3", "f 1/1 2/2 3/3 4/4"]:
            with self.subTest(invalidFace=invalidFace):
                self.resultPath.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n" + invalidFace + "\n")
                self.assertFalse(validateResult(self.bundleDir, self.resultPath)["structuralValid"])


if __name__ == "__main__":
    unittest.main()
