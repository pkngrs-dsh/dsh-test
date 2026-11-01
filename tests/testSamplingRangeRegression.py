"""Regression for saturated-photo SH failures discovered by the second batch."""

import numpy as np
import pytest

from faceAppearance.texture import samplePixels
from faceAppearance.shDelight import solveShDelight


@pytest.mark.parametrize("channels", [False, True])
def testConstantWhiteDoesNotOvershootItsConvexBound(channels):
    shape = (2, 2, 3) if channels else (2, 2)
    point = np.array([[.12909374542414298, .2829922190380785]])
    result = samplePixels(np.ones(shape, np.float32), point)
    np.testing.assert_array_equal(result, 1.)
    assert result.dtype == np.float32


def testZeroBorderRemainsPartOfTheConvexCombination():
    points = np.array([[-.5, .5], [-2., 0.], [1.5, .5]])
    np.testing.assert_array_equal(samplePixels(np.ones((2, 2), np.float32), points), [.5, 0., .5])
    np.testing.assert_array_equal(samplePixels(np.zeros((2, 2), np.float32), points), 0.)


def testSamplerPreservesSignedHdrAndNonSaturatedValues():
    image = np.broadcast_to(np.array([-2., .25, 4.], np.float32), (2, 2, 3))
    result = samplePixels(image, np.array([[.5, .5]]))
    np.testing.assert_array_equal(result, [[-2., .25, 4.]])


def testActualOutOfRangeSourceStillFailsShValidation():
    color = samplePixels(np.full((2, 2, 3), 1.001, np.float32), np.array([[.5, .5]]))[0]
    assert np.all(color > 1)
    observations = np.broadcast_to(color, (4, 4, 4, 3)).copy()
    with pytest.raises(ValueError, match="linear RGB"):
        solveShDelight(observations, np.ones((4, 4, 4), np.float32),
                      np.broadcast_to([0., 0., 1.], (4, 4, 3)).copy())
