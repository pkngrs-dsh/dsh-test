import numpy as np
import pytest

from faceAppearance.clientTemplate import connectedMaterialComponents, readLegacyObj


def writeObj(tmpPath, text):
    path = tmpPath / "template.obj"
    path.write_text(text)
    return path


HEADER = """v 3 2 1
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 0
vt 0.2 0.1
vt 0.9 0.2
vt 0.4 0.8
vt 0.6 0.3
"""


def testParserPreservesOrderDuplicatesUvSeamsAndFirstUsedMaterialOrder(tmp_path):
    path = writeObj(tmp_path, HEADER + """usemtl unused
usemtl eyes
f 2/1 3/2 4/3
usemtl head
f 5/4 4/3 3/2
usemtl eyes
f 1/1 2/4 3/2
""")
    result = readLegacyObj(path)
    assert result["vertices"].shape == (5, 3)
    assert result["uvs"].shape == (4, 2)
    assert result["vertices"].dtype == np.float64
    assert result["faces"].dtype == np.int64
    np.testing.assert_array_equal(result["vertices"][[1, 4]], [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result["faces"], [[1, 2, 3], [4, 3, 2], [0, 1, 2]])
    np.testing.assert_array_equal(result["uvFaces"], [[0, 1, 2], [3, 2, 1], [0, 3, 1]])
    np.testing.assert_allclose(result["uvs"], [[.2, .9], [.9, .8], [.4, .2], [.6, .7]])
    assert result["materialNames"] == ["eyes", "head"]
    np.testing.assert_array_equal(result["materialIds"], [0, 1, 0])


def testNegativeIndicesAreRelativeToTheirOwnRecordsAtFaceLine(tmp_path):
    result = readLegacyObj(writeObj(tmp_path, HEADER + "f -4/-4/-1 -3/-3/-1 -2/-2/-1\nv 9 9 9\nvt .8 .2\n"))
    np.testing.assert_array_equal(result["faces"], [[1, 2, 3]])
    np.testing.assert_array_equal(result["uvFaces"], [[0, 1, 2]])
    assert result["materialNames"] == ["__default__"]


@pytest.mark.parametrize("face", ["f 1/1 2/2 3/3 4/4", "f 1 2 3", "f 1//1 2//1 3//1",
                                  "f 0/1 2/2 3/3", "f 1/0 2/2 3/3", "f -6/1 2/2 3/3",
                                  "f 6/1 2/2 3/3", "f 1/5 2/2 3/3"])
def testParserRejectsTopologyOrUvAmbiguity(tmp_path, face):
    with pytest.raises(ValueError, match="template.obj:"):
        readLegacyObj(writeObj(tmp_path, HEADER + face + "\n"))


@pytest.mark.parametrize("extra", ["v nan 0 0\n", "vt inf 0\n", "v 0 0 0 2\n", "vt 0 0 1\n"])
def testInvalidCoordinatesAreRejectedEvenOnUnusedRecords(tmp_path, extra):
    with pytest.raises(ValueError):
        readLegacyObj(writeObj(tmp_path, HEADER + extra + "f 2/1 3/2 4/3\n"))


def testComponentsPreserveOriginalIndicesAndDoNotConnectCoincidentSeparateVertices():
    mesh = {"vertices": np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0],
                                  [0, 1, 0], [0, 0, 1], [20, 20, 20]]),
            "faces": np.array([[0, 1, 2], [3, 4, 5], [0, 2, 6], [0, 1, 6]]),
            "materialIds": np.array([0, 0, 0, 1]), "materialNames": ["head", "oral"]}
    before = {key: value.copy() for key, value in mesh.items() if isinstance(value, np.ndarray)}
    result = connectedMaterialComponents(mesh)
    assert [record["name"] for record in result] == ["head.component000", "head.component001", "oral.component000"]
    assert [record["faceIndices"].tolist() for record in result] == [[0, 2], [1], [3]]
    assert [record["vertexIndices"].tolist() for record in result] == [[0, 1, 2, 6], [3, 4, 5], [0, 1, 6]]
    np.testing.assert_array_equal(result[0]["bounds"], [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_allclose(result[0]["center"], [.25, .25, .25])
    np.testing.assert_array_equal(np.sort(np.concatenate([record["faceIndices"] for record in result])), np.arange(4))
    for key, value in before.items():
        np.testing.assert_array_equal(mesh[key], value)


def testComponentsRejectInvalidMaterialIndices():
    with pytest.raises(ValueError, match="out of range"):
        connectedMaterialComponents({"vertices": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]]),
                                     "materialIds": np.array([1]), "materialNames": ["head"]})
