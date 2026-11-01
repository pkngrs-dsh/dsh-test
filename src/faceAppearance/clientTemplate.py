"""Read fixed client topology without welding, reordering, or triangulating it.

Geometry remains in the file's native coordinates. OBJ UVs have a bottom-left
origin; returned array UVs use a top-left origin. Material names identify file
groups only: anatomical names and binding semantics must come from an audited
asset manifest, not from component size or position.
"""

from pathlib import Path

import numpy as np


def readLegacyObj(path: str | Path) -> dict:
    """Read a triangular, textured OBJ while retaining every source index.

    Negative indices are resolved against the records present on that line.
    Normals, smoothing groups, object names, and external MTL files do not alter
    vertex identity. Missing materials receive the explicit ``__default__``
    name. ``materialNames`` follows first use by a face, including revisits.
    """
    path = Path(path)
    vertices, uvs, faces, uvFaces, materialIds, materialNames = [], [], [], [], [], []
    materialLookup = {}
    currentMaterial = "__default__"

    def resolveIndex(value, count, kind):
        index = int(value)
        if index == 0:
            raise ValueError(f"{kind} index zero is invalid")
        index = index - 1 if index > 0 else count + index
        if not 0 <= index < count:
            raise ValueError(f"{kind} index is out of range")
        return index

    with path.open(encoding="utf-8-sig") as stream:
        for lineNumber, line in enumerate(stream, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            try:
                if parts[0] == "v":
                    if len(parts) not in (4, 5):
                        raise ValueError("expected three vertex coordinates and optional homogeneous weight")
                    values = [float(value) for value in parts[1:]]
                    if len(values) == 4 and values[3] != 1:
                        raise ValueError("nonunit homogeneous vertex weights are unsupported")
                    vertices.append(values[:3])
                elif parts[0] == "vt":
                    if len(parts) not in (3, 4):
                        raise ValueError("expected two texture coordinates and optional zero third coordinate")
                    values = [float(value) for value in parts[1:]]
                    if len(values) == 3 and values[2] != 0:
                        raise ValueError("three-dimensional texture coordinates are unsupported")
                    uvs.append([values[0], 1 - values[1]])
                elif parts[0] == "usemtl":
                    if len(parts) < 2:
                        raise ValueError("empty material name")
                    currentMaterial = " ".join(parts[1:])
                elif parts[0] == "f":
                    if len(parts) != 4:
                        raise ValueError("only existing triangles are allowed; automatic triangulation changes topology")
                    vertexIds, uvIds = [], []
                    for corner in parts[1:]:
                        fields = corner.split("/")
                        if len(fields) not in (2, 3) or not fields[1]:
                            raise ValueError("every face corner requires a texture-coordinate index")
                        vertexIds.append(resolveIndex(fields[0], len(vertices), "vertex"))
                        uvIds.append(resolveIndex(fields[1], len(uvs), "texture"))
                    if currentMaterial not in materialLookup:
                        materialLookup[currentMaterial] = len(materialNames)
                        materialNames.append(currentMaterial)
                    faces.append(vertexIds)
                    uvFaces.append(uvIds)
                    materialIds.append(materialLookup[currentMaterial])
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{path}:{lineNumber}: {error}") from error
    vertexArray = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    uvArray = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if not len(vertexArray) or not len(uvArray) or not faces:
        raise ValueError(f"{path}: expected a nonempty triangular mesh with UVs")
    if not np.isfinite(vertexArray).all() or not np.isfinite(uvArray).all():
        raise ValueError(f"{path}: mesh coordinates must be finite")
    return {"vertices": vertexArray, "faces": np.asarray(faces, dtype=np.int64),
            "uvs": uvArray, "uvFaces": np.asarray(uvFaces, dtype=np.int64),
            "materialIds": np.asarray(materialIds, dtype=np.int64),
            "materialNames": materialNames}


def connectedMaterialComponents(mesh: dict) -> list[dict]:
    """Return deterministic vertex-connected components within each material.

    All indices address the original mesh. Faces belong to exactly one record;
    vertices shared by two material groups appear in both records. No vertices
    or UV seams are split, welded, or reordered. Isolated unused vertices have
    no material membership and therefore do not create a component.
    """
    vertices = np.asarray(mesh["vertices"])
    faces, materialIds = np.asarray(mesh["faces"]), np.asarray(mesh["materialIds"])
    names = mesh["materialNames"]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or materialIds.shape != (len(faces),) or materialIds.dtype.kind not in "iu"):
        raise ValueError("Invalid vertex, triangle, or material arrays")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)
                       or materialIds.min() < 0 or materialIds.max() >= len(names)):
        raise ValueError("Triangle or material indices are out of range")
    records = []
    for materialId, materialName in enumerate(names):
        faceIds = np.flatnonzero(materialIds == materialId)
        if not len(faceIds):
            continue
        parent = {int(index): int(index) for index in np.unique(faces[faceIds])}

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for triangle in faces[faceIds]:
            first = root(int(triangle[0]))
            for index in triangle[1:]:
                parent[root(int(index))] = first
        groups = {}
        for faceId in faceIds:
            groups.setdefault(root(int(faces[faceId, 0])), []).append(int(faceId))
        for componentIndex, group in enumerate(sorted(groups.values(), key=lambda values: values[0])):
            componentFaces = np.asarray(group, dtype=np.int64)
            componentVertices = np.unique(faces[componentFaces])
            points = vertices[componentVertices]
            records.append({"name": f"{materialName}.component{componentIndex:03d}",
                            "materialId": materialId, "vertexIndices": componentVertices,
                            "faceIndices": componentFaces, "center": points.mean(axis=0),
                            "bounds": np.stack([points.min(axis=0), points.max(axis=0)])})
    return records
