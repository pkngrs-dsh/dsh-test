import numpy as np
import hashlib
import json
import pytest

from faceAppearance.correspondence import SurfaceIndex
from faceAppearance.registration import RegistrationAssets, transferRegisteredGeometry


def testOffsetTransferPreservesRestAndRigidScaleEquivariance():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    faces = np.array([[0, 1, 2]])
    target = np.array([[0.2, 0.3, 0.1], [0.1, 0.2, -0.3]])
    mapping = SurfaceIndex(vertices, faces).query(target)
    unchanged = transferRegisteredGeometry(vertices, vertices, faces, target, mapping)
    np.testing.assert_allclose(unchanged, target, atol=1e-6)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = vertices @ rotation.T * 3 + [1, 2, 3]
    actual = transferRegisteredGeometry(vertices, transformed, faces, target, mapping)
    np.testing.assert_allclose(actual, target @ rotation.T * 3 + [1, 2, 3], atol=1e-6)


def testRegistrationRejectsChangedAssetBeforeUsingItsArrays(tmp_path):
    files = {}
    for name in ("geometryMaps.npz", "atlas512.npz"):
        path = tmp_path / name
        np.savez_compressed(path, fixture=np.zeros(1))
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "status": "prototypeCommonSurfaceQualified", "files": files}))
    RegistrationAssets(tmp_path, 512)
    with (tmp_path / "atlas512.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity check failed"):
        RegistrationAssets(tmp_path, 512)


def testUnreviewedRegistrationIsRejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "status": "structuralValid"}))
    with pytest.raises(ValueError, match="prototype review"):
        RegistrationAssets(tmp_path, 512)
