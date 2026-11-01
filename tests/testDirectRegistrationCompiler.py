"""CPU checks of the immutable legacy surface correspondence compiler."""

import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepareDirectRegistration", ROOT / "tools/prepareDirectRegistration.py")
COMPILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILER)


class DirectRegistrationCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.audit = ROOT / "reports/legacyBarycentricAudit"
        cls.manifests = {}
        cls.arrays = {}
        for gender in ("Male", "Female"):
            base = ROOT / f"assets/clientTemplates/legacy{gender}V1"
            output = cls.root / f"legacy{gender}DirectV1"
            cls.manifests[gender] = COMPILER.compileDirectRegistration(base, cls.audit, output)
            cls.arrays[gender] = COMPILER._readArchive(output / "directRegistration.npz")
        cls.mapping = COMPILER._readArchive(cls.audit / "mapping.npz")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def testBothGendersRetainExactValidOriginalSamples(self):
        mapping = self.mapping
        kept = np.flatnonzero(mapping["sampleValidMask"])
        self.assertEqual(len(kept), 5303)
        for gender in ("Male", "Female"):
            arrays = self.arrays[gender]
            np.testing.assert_array_equal(arrays["originalSampleIds"], kept)
            for key in ("sampleNicpFaceIds", "sampleNicpWeights", "sourceTriangleIds", "sourceBarycentricWeights"):
                np.testing.assert_array_equal(arrays[key], mapping[key][kept])
            np.testing.assert_array_equal(arrays["sampleWeights"], np.ones(len(kept)))
            self.assertNotIn(731, arrays["originalSampleIds"])
            self.assertEqual(self.manifests[gender]["geometryTransfer"]["method"], "directSurface")
            self.assertFalse(self.manifests[gender]["textureTransfer"]["geometryConsistencyValidated"])

    def testRestNormalAndStaticAdjacentSamplesAreNotGated(self):
        for gender in ("Male", "Female"):
            arrays = self.arrays[gender]
            ids = arrays["originalSampleIds"]
            normal = self.mapping[f"rawNativeVDown_{gender.lower()}NormalDots"][ids]
            distance = self.mapping[f"rawNativeVDown_{gender.lower()}RestDistances"][ids]
            self.assertGreater(int((normal < .35).sum()), 0)
            self.assertGreater(int((distance > .12).sum()), 0)
            base = COMPILER._readArchive(ROOT / f"assets/clientTemplates/legacy{gender}V1/template.npz")
            np.testing.assert_array_equal(arrays["staticMask"], base["staticMask"])
            np.testing.assert_array_equal(arrays["staticVertexIds"], np.flatnonzero(base["staticMask"]))
            adjacent = base["staticMask"][base["nicpFaces"][arrays["sampleNicpFaceIds"]]].any(axis=1)
            self.assertEqual(int(adjacent.sum()), 4)
            self.assertEqual(int(arrays["staticMask"].sum()), 538)

    def testBaseGeometryAndTextureFilesRemainByteIdentical(self):
        for gender in ("Male", "Female"):
            base = ROOT / f"assets/clientTemplates/legacy{gender}V1"
            output = self.root / f"legacy{gender}DirectV1"
            source = json.loads((base / "manifest.json").read_text())
            for relative, digest in source["files"].items():
                self.assertEqual(COMPILER.fileHash(output / relative), digest)
            for relative, digest in self.manifests[gender]["files"].items():
                self.assertEqual(COMPILER.fileHash(output / relative), digest)

    def testRefusesOverwrite(self):
        with self.assertRaises(FileExistsError):
            COMPILER.compileDirectRegistration(ROOT / "assets/clientTemplates/legacyMaleV1", self.audit,
                                                self.root / "legacyMaleDirectV1")

    def testDetectsUnapprovedMappingHash(self):
        with TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit"
            audit.mkdir()
            shutil.copyfile(self.audit / "audit.json", audit / "audit.json")
            (audit / "mapping.npz").write_bytes(b"not the audited mapping")
            with self.assertRaisesRegex(ValueError, "mapping hash mismatch"):
                COMPILER.compileDirectRegistration(ROOT / "assets/clientTemplates/legacyMaleV1", audit,
                                                    Path(temporary) / "result")

    def testOriginalJsonIndependentlyChecksCachedCorrespondences(self):
        with TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit"
            audit.mkdir()
            mapping = {key: value.copy() for key, value in self.mapping.items()}
            mapping["sampleNicpFaceIds"][0] += 1
            np.savez_compressed(audit / "mapping.npz", **mapping)
            report = json.loads((self.audit / "audit.json").read_text())
            report["mappingSha256"] = COMPILER.fileHash(audit / "mapping.npz")
            (audit / "audit.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "sampleNicpFaceIds"):
                COMPILER.compileDirectRegistration(ROOT / "assets/clientTemplates/legacyMaleV1", audit,
                                                    Path(temporary) / "result")

    def testVertexUvGeometryWeightsDoNotControlCompiledConstraints(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary) / "base"
            shutil.copytree(ROOT / "assets/clientTemplates/legacyMaleV1", base)
            arrays = COMPILER._readArchive(base / "template.npz")
            arrays["geometryWeights"][:] = 0
            np.savez_compressed(base / "template.npz", **arrays)
            manifest = json.loads((base / "manifest.json").read_text())
            manifest["files"]["template.npz"] = COMPILER.fileHash(base / "template.npz")
            (base / "manifest.json").write_text(json.dumps(manifest))
            output = Path(temporary) / "result"
            COMPILER.compileDirectRegistration(base, self.audit, output)
            actual = COMPILER._readArchive(output / "directRegistration.npz")
            for key, value in self.arrays["Male"].items():
                np.testing.assert_array_equal(actual[key], value)


if __name__ == "__main__":
    unittest.main()
