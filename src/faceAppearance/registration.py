"""Versioned registration assets, canonical geometry transfer and atlas access."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .correspondence import arrayHash, sampleSurface


def triangleFrames(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(first, second)
    areaScale = np.linalg.norm(cross, axis=1)
    if np.any(areaScale < 1e-14):
        raise ValueError("Cannot transfer offsets through degenerate triangles")
    normal = cross / np.sqrt(areaScale)[:, None]
    return np.stack([first, second, normal], axis=-1)


def transferRegisteredGeometry(sourceRest: np.ndarray, sourceDeformed: np.ndarray,
                               sourceFaces: np.ndarray, registeredTarget: np.ndarray,
                               mapping: dict) -> np.ndarray:
    """Preserve local rest offsets, including explicitly unqualified boundary vertices.

    Offset transport is equivariant to rigid motion and uniform scale. The
    caller must still retain valid-face masks; extrapolation is not evidence
    that an unmatched neck/inner surface has been reconstructed from photos.
    """
    if np.shape(sourceRest) != np.shape(sourceDeformed) or not np.isfinite(sourceDeformed).all():
        raise ValueError("Source geometry must preserve vertex order and finite coordinates")
    triangleIds = mapping["triangleIds"]
    restPoints = sampleSurface(sourceRest, sourceFaces, mapping)
    deformedPoints = sampleSurface(sourceDeformed, sourceFaces, mapping)
    restFrames = triangleFrames(sourceRest[sourceFaces[triangleIds]])
    deformedFrames = triangleFrames(sourceDeformed[sourceFaces[triangleIds]])
    offsets = np.linalg.solve(restFrames, (registeredTarget - restPoints)[..., None])
    return (deformedPoints + (deformedFrames @ offsets)[..., 0]).astype(np.float32)


class RegistrationAssets:
    def __init__(self, assetRoot: Path, resolution: int):
        self.assetRoot = Path(assetRoot).resolve()
        self.resolution = resolution
        self.manifest = json.loads((self.assetRoot / "manifest.json").read_text())
        if self.manifest.get("schemaVersion") != 1 or self.manifest.get("status") != "prototypeCommonSurfaceQualified":
            raise ValueError("Registration has not passed the common-surface prototype review")
        for fileName in ("geometryMaps.npz", f"atlas{resolution}.npz"):
            path = self.assetRoot / fileName
            record = self.manifest["files"][fileName]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Registration asset integrity check failed: {fileName}")
        with np.load(self.assetRoot / "geometryMaps.npz", allow_pickle=False) as data:
            self.geometry = {name: data[name] for name in data.files}
        with np.load(self.assetRoot / f"atlas{resolution}.npz", allow_pickle=False) as data:
            self.atlases = {name: data[name] for name in data.files}

    def atlasMap(self, direction: str) -> dict:
        if direction not in {"uffToIct", "ictToUff"}:
            raise ValueError("Unknown transfer direction")
        return {key[0].lower() + key[1:]: value for name, value in self.atlases.items()
                if name.startswith(direction) for key in [name[len(direction):]]}

    def toIctGeometry(self, geometry: dict) -> dict:
        actualHash = arrayHash(np.asarray(geometry["faces"], np.int64), np.asarray(geometry["uvs"], np.float32))
        if actualHash != self.manifest["uffTopologyHash"]:
            raise ValueError("Input UFF topology/UV differs from registered asset version")
        data = self.geometry
        mapping = {"triangleIds": data["uffToIctTriangleIds"],
                   "barycentricWeights": data["uffToIctBarycentricWeights"]}
        vertices = transferRegisteredGeometry(data["uffVertices"], geometry["vertices"], data["uffFaces"],
                                              data["ictRegisteredVertices"], mapping)
        return {"vertices": vertices, "faces": data["ictFaces"], "uvs": data["ictUvs"],
                "uvFaces": data["ictUvFaces"], "validFaceMask": data["ictValidFaces"],
                "geometryFrame": "uvFaceFusionCanonical"}
