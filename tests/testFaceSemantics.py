"""Parser-independent label and confidence contracts."""

import unittest

import numpy as np

from faceAppearance.faceSemantics import (FaceSemanticResult, normalizeParserLabels,
                                         semanticLabelIds, semanticLabelNames, semanticMask)


class FaceSemanticsTests(unittest.TestCase):
    def testCelebmAndLapaNormalizeToStableIds(self):
        celebm = ("background", "neck", "face", "cloth", "rr", "lr", "rb", "lb", "re",
                  "le", "nose", "imouth", "llip", "ulip", "hair", "eyeg", "hat", "earr", "neck_l")
        lapa = ("background", "face", "rb", "lb", "re", "le", "nose", "ulip", "imouth", "llip", "hair")
        for source, expected in ((celebm, [0, 14, 1, 16, 8, 7, 3, 2, 5, 4, 10, 11, 13, 12, 17, 6, 18, 9, 15]),
                                 (lapa, [0, 1, 3, 2, 5, 4, 10, 12, 11, 13, 17])):
            result = normalizeParserLabels(np.arange(len(source), dtype=np.uint8)[None], source)
            np.testing.assert_array_equal(result[0], expected)
            self.assertEqual(result.dtype, np.uint8)

    def testRemapRejectsUnknownDuplicateAndOutOfRangeSchema(self):
        for labels, names in ((np.array([[0]]), ["other"]), (np.array([[0]]), ["face", "face"]),
                              (np.array([[-1]]), ["face"]), (np.array([[1]]), ["face"]),
                              (np.array([[0.0]]), ["face"]), (np.array([], int), ["face"])):
            with self.subTest(labels=labels, names=names), self.assertRaises(ValueError):
                normalizeParserLabels(labels, names)

    def testSelectionRespectsConfidenceAndValidSupport(self):
        labels = np.array([[2, 3, 12, 13], [2, 3, 11, 1]], np.uint8)
        confidence = np.array([[0.9, 0.2, 0.9, 0.9], [0.9, 0.8, 0.9, 0.9]], np.float32)
        validMask = np.ones((2, 4), bool)
        validMask[1, 0] = False
        result = FaceSemanticResult(labels, confidence, np.zeros((5, 2)), validMask=validMask)
        np.testing.assert_array_equal(semanticMask(result, ["leftBrow", "rightBrow"], 0.5),
                                      [[True, False, False, False], [False, True, False, False]])
        self.assertEqual(int(semanticMask(result, "upperLip").sum()), 1)
        self.assertFalse(semanticMask(result, []).any())
        with self.assertRaisesRegex(ValueError, "Unknown semantic"):
            semanticMask(result, "face")
        for confidenceThreshold in (-0.1, 1.1, np.nan, True):
            with self.assertRaises(ValueError):
                semanticMask(result, ["skin"], confidenceThreshold)

    def testResultRejectsInvalidExternalPayloads(self):
        validArgs = {"labelMap": np.zeros((2, 3), np.uint8),
                     "confidence": np.ones((2, 3), np.float32), "landmarks": np.zeros((5, 2))}
        badValues = {"labelMap": np.full((2, 3), 19, np.uint8),
                     "confidence": np.full((2, 3), np.nan), "landmarks": np.zeros((68, 2)),
                     "validMask": np.zeros((3, 2), bool), "boxXyxy": np.array([1, 1, 0, 0]),
                     "labelNames": tuple(reversed(semanticLabelNames))}
        for key, value in badValues.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                FaceSemanticResult(**{**validArgs, key: value})

    def testSemanticPairsRetainExplicitSourceSides(self):
        self.assertEqual(semanticLabelIds["leftBrow"], 2)
        self.assertEqual(semanticLabelIds["rightBrow"], 3)
        labels = normalizeParserLabels(np.array([[0, 1, 0]], np.uint8), ["rb", "lb"])
        np.testing.assert_array_equal(labels, [[3, 2, 3]])


if __name__ == "__main__":
    unittest.main()
