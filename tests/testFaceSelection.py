import numpy as np
import pytest

from faceAppearance.faceSelection import selectLargestFaceIndex


def testLargestAreaWinsOverHigherConfidenceWithoutMutatingInputs():
    boxes = np.array([[2, 3, 12, 13], [-10, 5, 30, 55]], dtype=np.float32)
    scores = np.array([.99, .81], dtype=np.float32)
    before = boxes.copy(), scores.copy()
    assert selectLargestFaceIndex(boxes, scores) == 1
    np.testing.assert_array_equal(boxes, before[0])
    np.testing.assert_array_equal(scores, before[1])


def testAreaTiesUseConfidenceThenStableOriginalIndex():
    boxes = [[0, 0, 20, 20], [30, 10, 40, 50], [50, 10, 60, 50]]
    assert selectLargestFaceIndex(boxes, [.85, .9, .9]) == 1
    assert selectLargestFaceIndex(boxes, [.9, .9, .9]) == 0
    assert selectLargestFaceIndex([boxes[0]], [0.]) == 0


@pytest.mark.parametrize("boxes", [[], np.empty((0, 4))])
def testNoFacesFailsExplicitly(boxes):
    with pytest.raises(ValueError, match="No faces detected"):
        selectLargestFaceIndex(boxes, [])


@pytest.mark.parametrize("boxes,scores,message", [
    ([[0, 0, 10]], [.9], "shape"),
    ([[0, 0, 10, 10]], [.9, .8], "matching"),
    ([[0, 0, 10, 10]], [[.9]], "shape"),
    ([[0, 0, np.nan, 10]], [.9], "finite"),
    ([[0, 0, 10, 10]], [np.inf], "finite"),
    ([[0, 0, 10, 10]], [1.1], r"\[0, 1\]"),
    ([[0, 0, 10, 10]], [-.1], r"\[0, 1\]"),
    ([[0, 0, 0, 10]], [.9], "positive"),
    ([[10, 10, 0, 0]], [.9], "positive"),
    ([[0, 0, 10, 10], [0, 0, 0, 3]], [.9, .8], "positive"),
    ([[-1e308, 0, 1e308, 10]], [.9], "finite"),
    ([[0, 0, 1e200, 1e200]], [.9], "finite"),
    ([["0", "0", "10", "10"]], [.9], "numeric"),
    ([[0, 0, 10, 10]], [True], "numeric"),
])
def testMalformedDetectionsAreRejected(boxes, scores, message):
    with pytest.raises(ValueError, match=message):
        selectLargestFaceIndex(boxes, scores)
