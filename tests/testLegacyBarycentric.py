import json

import numpy as np
import pytest

from faceAppearance.legacyBarycentric import compileLegacyUvSamples, readLegacyBarycentric


def read(tmpPath, records, count=3):
    path = tmpPath / "bary.json"
    path.write_text(records if isinstance(records, str) else json.dumps(records))
    return readLegacyBarycentric(path, np.array([[0, 1, 2], [2, 3, 4]]), expectedSampleCount=count)


def testSampleOrderingIsNumericNotJsonInsertionOrLexicographicOrder(tmp_path):
    records = {str(index): [index % 2, index/30, .2] for index in reversed(range(12))}
    result = read(tmp_path, records, count=12)
    np.testing.assert_array_equal(result["sampleIds"], np.arange(12))
    np.testing.assert_array_equal(result["sampleNicpFaceIds"], np.arange(12) % 2)
    np.testing.assert_allclose(result["sampleNicpWeights"][:, 0], np.arange(12)/30)
    np.testing.assert_allclose(result["sampleNicpWeights"].sum(1), 1)


@pytest.mark.parametrize("records", [
    '{"0":[0,.2,.3],"0":[0,.1,.2]}'.replace(".", "0."),
    {"0": [0, .2, .3], "00": [0, .2, .3], "1": [0, .2, .3]},
    {"0": [0, .2, .3], "1": [0, .2, .3]},
    {"0": [0, .2, .3], "1": [0, .2, .3], "3": [0, .2, .3]},
    {"0": [0, .2, .3], "1": [0, .2, .3], "-1": [0, .2, .3]},
    {"0": [2, .2, .3], "1": [0, .2, .3], "2": [0, .2, .3]},
    {"0": [.5, .2, .3], "1": [0, .2, .3], "2": [0, .2, .3]},
    {"0": [True, .2, .3], "1": [0, .2, .3], "2": [0, .2, .3]},
    {"0": [0, float("nan"), .3], "1": [0, .2, .3], "2": [0, .2, .3]},
])
def testMalformedOrAmbiguousTableIsRejected(tmp_path, records):
    with pytest.raises(ValueError):
        read(tmp_path, records)


def testOutsideWeightsArePreservedAndInvalidRatherThanClipped(tmp_path):
    result = read(tmp_path, {"0": [0, .3, .7], "1": [1, -.000017, .5], "2": [0, 0, 0]})
    np.testing.assert_array_equal(result["sampleBaryValidMask"], [True, False, True])
    assert result["sampleNicpWeights"][1, 0] == -.000017
    np.testing.assert_allclose(result["sampleNicpWeights"][1], [-.000017, .5, .500017])


def testUvCompileInterpolatesExactSurfaceAndMarksMissingSupport():
    vertices = np.array([[0., 0, 2], [2, 0, 4], [0, 2, 6]])
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0., 0], [1, 0], [0, 1]])
    result = compileLegacyUvSamples([[.25, .25], [.8, .8]], vertices, faces, uv)
    np.testing.assert_allclose(result["sampleSourcePoints"][0], [.5, .5, 3.5])
    np.testing.assert_array_equal(result["sampleUvValidMask"], [True, False])
    np.testing.assert_allclose(result["sourceBarycentricWeights"][0], [.5, .25, .25])
    np.testing.assert_array_equal(result["sampleUvs"], [[.25, .25], [.8, .8]])


def testDegenerateUvFaceRemovalRetainsOriginalFaceIdentity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 3], [0, 1, 2]])
    uv = np.array([[0., 0], [1, 0], [0, 1], [.5, 0]])
    result = compileLegacyUvSamples([[.2, .2]], vertices, faces, uv)
    np.testing.assert_array_equal(result["sourceTriangleIds"], [1])
    np.testing.assert_allclose(result["sampleSourcePoints"], [[.2, .2, 0]])


def testExplicitUvDirectionIsNeverChangedAutomatically():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    uv = vertices[:, :2]
    direct = compileLegacyUvSamples([[.1, .2]], vertices, faces, uv)
    flipped = compileLegacyUvSamples([[.1, .8]], vertices, faces, uv)
    assert direct["sampleSourcePoints"][0, 1] == pytest.approx(.2)
    assert flipped["sampleSourcePoints"][0, 1] == pytest.approx(.8)
