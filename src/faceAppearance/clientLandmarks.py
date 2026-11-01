"""Fixed semantic surface bindings on the legacy client NICP topology.

The 468 IDs retain their MediaPipe meaning and map to a triangle plus its
original ``(u, v, 1-u-v)`` weights.  They are never rebound by nearest neighbour.
The iBUG-68 adapter is an approximate semantic correspondence from the old
project; it is not a manual annotation or an anatomical ground truth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .correspondence import arrayHash
from .legacyBarycentric import readLegacyBarycentric


_DEFAULT_PATH = (Path(__file__).resolve().parents[2] / "assets" / "clientLandmarks"
                 / "legacyMediapipe468" / "template_tri_lms_468_bicentroid.json")
_SOURCE_SHA256 = "66dc9bab1ebe088575a3e18ef79245418d30ee6573fb87365dd089a046ced73f"
_NICP_FACES_SHA256 = "38dcb50f30500c2962cd2c08f3c022bd7ba509dbfb088bf072be67819556cc2a"

# ID membership follows Google's MediaPipe face_mesh_connections.py.  Left and
# right mean the subject's sides; no image mirroring or relabeling is performed.
_GROUPS = {
    "rightEye": (33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246),
    "leftEye": (263, 249, 390, 373, 374, 380, 381, 382, 362,
                398, 384, 385, 386, 387, 388, 466),
    "outerLips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                  409, 270, 269, 267, 0, 37, 39, 40, 185),
    "innerLips": (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                  415, 310, 311, 312, 13, 82, 81, 80, 191),
    "rightBrow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "leftBrow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "nose": (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327,
             294, 278, 344, 440, 275, 45, 220, 115, 48, 64),
    "faceOval": (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
                 103, 67, 109),
}
_GROUPS.update({"eyes": _GROUPS["rightEye"] + _GROUPS["leftEye"],
                "mouth": _GROUPS["outerLips"] + _GROUPS["innerLips"],
                "brows": _GROUPS["rightBrow"] + _GROUPS["leftBrow"],
                "all": tuple(range(468))})

# Correspondences from the old v1_utils/face_landmarking/mp2dlib.py by
# Peizhi Yan (2024), https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks.
# Zero-based iBUG ID indexes each row. Two distinct MP points mean an average,
# which is not a single surface binding and is excluded by the subset adapter.
IBUG68_MEDIAPIPE_IDS = (
    (127,), (234,), (93,), (132, 58), (58, 172), (136,), (150,),
    (176,), (152,), (400,), (379,), (365,), (397, 288), (361,),
    (323,), (454,), (356,), (70,), (63,), (105,), (66,), (107,),
    (336,), (296,), (334,), (293,), (300,), (168, 6), (197, 195),
    (5,), (4,), (75,), (97,), (2,), (326,), (305,),
    (33,), (160,), (158,), (133,), (153,), (144,),
    (362,), (385,), (387,), (263,), (373,), (380,),
    (61,), (39,), (37,), (0,), (267,), (269,), (291,),
    (321,), (314,), (17,), (84,), (91,),
    (78,), (82,), (13,), (312,), (308,), (317,), (14,), (87,),
)
_IBUG_GROUPS = {
    "rightEye": tuple(range(36, 42)), "leftEye": tuple(range(42, 48)),
    "eyes": tuple(range(36, 48)), "mouth": tuple(range(48, 68)),
    "outerLips": tuple(range(48, 60)), "innerLips": tuple(range(60, 68)),
    "brows": tuple(range(17, 27)), "nose": tuple(range(27, 36)),
    "jaw": tuple(range(17)), "all": tuple(range(68)),
}


def _freeze(arrays):
    for value in arrays.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return arrays


def _groupIds(groups, table):
    if isinstance(groups, str):
        groups = (groups,)
    selected = []
    for group in groups:
        if group not in table:
            raise ValueError(f"Unknown landmark group {group!r}; choose from {tuple(table)}")
        selected.extend(table[group])
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def loadClientLandmarks(nicpFaces, path=None):
    """Load all 468 bindings, requiring the audited NICP face/vertex order.

    ``nicpFaces`` must index NICP vertices, not the full client mesh. The male
    and female legacy bundles have the same ordered topology. Out-of-triangle
    finite weights are preserved and marked false in ``validMask``. Malformed
    records or a different topology raise. ``path`` may select a revised JSON,
    but does not disable the topology check or imply anatomical validation.
    """
    nicpFaces = np.asarray(nicpFaces)
    if (nicpFaces.shape != (5532, 3) or nicpFaces.dtype.kind not in "iu"
            or arrayHash(nicpFaces.astype(np.int64)) != _NICP_FACES_SHA256):
        raise ValueError("Landmark bindings require the audited legacy NICP face and vertex order")
    source = Path(path) if path is not None else _DEFAULT_PATH
    sourceHash = hashlib.sha256(source.read_bytes()).hexdigest()
    if path is None and sourceHash != _SOURCE_SHA256:
        raise ValueError("Bundled client landmark source hash mismatch")
    table = readLegacyBarycentric(source, nicpFaces, expectedSampleCount=468,
                                  boundaryTolerance=0.)
    faceIds = table["sampleNicpFaceIds"]
    return _freeze({
        "landmarkIds": table["sampleIds"],
        "mediapipeIds": table["sampleIds"].copy(),
        "faceIds": faceIds,
        "vertexIndices": nicpFaces[faceIds].astype(np.int64),
        "barycentricWeights": table["sampleNicpWeights"],
        "validMask": table["sampleBaryValidMask"],
        "nicpVertexCount": 2795,
        "nicpFacesHash": _NICP_FACES_SHA256,
        "sourcePath": str(source.resolve()), "sourceSha256": sourceHash,
        "isOriginalBinding": sourceHash == _SOURCE_SHA256,
        "semanticConvention": "MediaPipe 468, subject left/right, fixed client surface bindings",
    })


def selectClientLandmarks(binding, groups=("eyes", "mouth")):
    """Select fixed MediaPipe feature memberships, sorted by numeric MP ID."""
    ids = _groupIds(groups, _GROUPS)
    result = {name: np.asarray(binding[name])[ids].copy() for name in
              ("landmarkIds", "mediapipeIds", "faceIds", "vertexIndices",
               "barycentricWeights", "validMask")}
    result["nicpVertexCount"] = binding["nicpVertexCount"]
    result["semanticConvention"] = binding["semanticConvention"]
    return _freeze(result)


def ibugClientLandmarks(binding, groups=("eyes", "mouth")):
    """Return exact fixed surface bindings for single-MP iBUG approximations.

    Default output has 32 rows: zero-based iBUG 36:48 (eyes), 48:68 (mouth).
    Mapping *between conventions* is approximate; barycentric sampling of the
    chosen MP surface location is exact. Multi-MP rows are omitted explicitly
    as ``excludedAveragedIbugIds`` rather than collapsed to a nearest triangle.
    """
    requested = _groupIds(groups, _IBUG_GROUPS)
    ids = np.asarray([i for i in requested if len(IBUG68_MEDIAPIPE_IDS[i]) == 1], np.int64)
    mpIds = np.asarray([IBUG68_MEDIAPIPE_IDS[i][0] for i in ids], np.int64)
    result = {name: np.asarray(binding[name])[mpIds].copy() for name in
              ("faceIds", "vertexIndices", "barycentricWeights", "validMask")}
    result.update({"ibugIds": ids, "landmarkIds": ids.copy(), "mediapipeIds": mpIds,
                   "nicpVertexCount": binding["nicpVertexCount"],
                   "excludedAveragedIbugIds": np.asarray(
                       [i for i in requested if len(IBUG68_MEDIAPIPE_IDS[i]) != 1], np.int64),
                   "semanticConvention": "Approximate MP-to-iBUG68 correspondence, zero-based IDs; not ground truth"})
    return _freeze(result)


def sampleClientLandmarks(nicpVertices, binding, *, invalidValue=np.nan):
    """Sample fixed locations on deformed NICP vertices; invalid rows use NaN.

    Positions and output share the caller's coordinate frame and units. This
    does not estimate pose, project into an image, or change correspondence.
    """
    vertices = np.asarray(nicpVertices, np.float64)
    if vertices.shape != (binding["nicpVertexCount"], 3) or not np.isfinite(vertices).all():
        raise ValueError("Expected finite ordered NICP vertices with shape (2795, 3)")
    points = (vertices[binding["vertexIndices"]] * binding["barycentricWeights"][..., None]).sum(axis=1)
    points[~binding["validMask"]] = invalidValue
    return points
