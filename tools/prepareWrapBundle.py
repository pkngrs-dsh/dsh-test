#!/usr/bin/env python3
"""Prepare immutable, unaligned UVFaceFusion/ICT assets for manual Wrap registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjMesh:
    vertices: np.ndarray
    texCoords: np.ndarray
    faces: np.ndarray
    uvFaces: np.ndarray


def sha256File(filePath: Path) -> str:
    fileHash = hashlib.sha256()
    with filePath.open("rb") as sourceFile:
        for block in iter(lambda: sourceFile.read(1024 * 1024), b""):
            fileHash.update(block)
    return fileHash.hexdigest()


def readObj(filePath: Path) -> ObjMesh:
    """Read triangular OBJ indices without welding, sorting, or repairing geometry."""
    vertices, texCoords, faces, uvFaces = [], [], [], []

    def resolveIndex(rawIndex: str, count: int) -> int:
        index = int(rawIndex)
        if index == 0:
            raise ValueError("OBJ indices must not be zero")
        resolvedIndex = index - 1 if index > 0 else count + index
        if not 0 <= resolvedIndex < count:
            raise ValueError(f"OBJ index {index} is outside the available {count} entries")
        return resolvedIndex

    for lineNumber, sourceLine in enumerate(filePath.read_text(encoding="utf-8").splitlines(), 1):
        fields = sourceLine.split("#", 1)[0].split()
        if not fields:
            continue
        try:
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError("vertex requires three coordinates")
                if len(fields) == 5 and float(fields[4]) != 1:
                    raise ValueError("homogeneous vertex coordinates are unsupported")
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "vt":
                if len(fields) not in (3, 4):
                    raise ValueError("texture coordinate requires two coordinates")
                if len(fields) == 4 and float(fields[3]) != 0:
                    raise ValueError("nonzero third texture coordinate is unsupported")
                texCoords.append([float(value) for value in fields[1:3]])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError("expected triangle faces; automatic triangulation is forbidden")
                face, uvFace = [], []
                for field in fields[1:]:
                    components = field.split("/")
                    face.append(resolveIndex(components[0], len(vertices)))
                    if len(components) < 2 or not components[1]:
                        raise ValueError("every face corner must retain its texture-coordinate index")
                    uvFace.append(resolveIndex(components[1], len(texCoords)))
                faces.append(face)
                uvFaces.append(uvFace)
        except (ValueError, IndexError) as error:
            raise ValueError(f"{filePath.name}:{lineNumber}: {error}") from error
    if not vertices or not texCoords or not faces:
        raise ValueError(f"{filePath.name}: missing vertices, UVs, or faces")
    mesh = ObjMesh(np.asarray(vertices, dtype=np.float64), np.asarray(texCoords, dtype=np.float64),
                   np.asarray(faces, dtype=np.int64), np.asarray(uvFaces, dtype=np.int64))
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.texCoords).all():
        raise ValueError(f"{filePath.name}: coordinates contain NaN or infinity")
    return mesh


def writeObj(filePath: Path, mesh: ObjMesh) -> None:
    """Export indices in input order; 17 significant digits preserve float64 values."""
    with filePath.open("w", encoding="utf-8", newline="\n") as targetFile:
        targetFile.write("# Native template coordinates: no alignment, scaling, or axis conversion.\n")
        targetFile.write("mtllib uvChecker.mtl\n")
        for vertex in mesh.vertices:
            targetFile.write("v " + " ".join(f"{value:.17g}" for value in vertex) + "\n")
        for texCoord in mesh.texCoords:
            targetFile.write("vt " + " ".join(f"{value:.17g}" for value in texCoord) + "\n")
        targetFile.write("usemtl uvChecker\n")
        for face, uvFace in zip(mesh.faces, mesh.uvFaces):
            corners = [f"{vertexIndex + 1}/{uvIndex + 1}" for vertexIndex, uvIndex in zip(face, uvFace)]
            targetFile.write("f " + " ".join(corners) + "\n")


def createUvChecker(filePath: Path, resolution: int = 1024, cellCount: int = 16) -> None:
    """Numeric UV grid: U rises rightwards, V rises upwards in standard OBJ UVs."""
    from PIL import Image, ImageDraw, ImageFont

    checker = Image.new("RGB", (resolution, resolution))
    drawing = ImageDraw.Draw(checker)
    font = ImageFont.load_default()
    cellSize = resolution // cellCount
    for rowIndex in range(cellCount):
        for columnIndex in range(cellCount):
            vIndex = cellCount - rowIndex - 1
            cellColor = (212, 224, 234) if (rowIndex + columnIndex) % 2 == 0 else (62, 86, 110)
            textColor = (10, 18, 25) if (rowIndex + columnIndex) % 2 == 0 else (255, 255, 255)
            x, y = columnIndex * cellSize, rowIndex * cellSize
            drawing.rectangle((x, y, x + cellSize - 1, y + cellSize - 1), fill=cellColor)
            drawing.text((x + 4, y + 8), f"{columnIndex:02d},{vIndex:02d}", fill=textColor, font=font)
            drawing.text((x + 4, y + 30), "U> V^", fill=textColor, font=font)
            drawing.line((x + 4, y + cellSize - 5, x + cellSize - 5, y + cellSize - 5), fill=(230, 70, 60), width=2)
            drawing.line((x + 4, y + cellSize - 5, x + 4, y + 3), fill=(40, 180, 80), width=2)
    checker.save(filePath)


def meshSummary(mesh: ObjMesh) -> dict:
    return {"vertexCount": len(mesh.vertices), "triangleCount": len(mesh.faces),
            "uvCount": len(mesh.texCoords), "bounds": [mesh.vertices.min(axis=0).tolist(), mesh.vertices.max(axis=0).tolist()],
            "coordinateSpace": "unmodified source-native coordinates; unit is not asserted to be meters or centimeters"}


def prepareBundle(uvFaceFusionMesh: Path, uvFaceFusionUvs: Path, ictMesh: Path, outputDir: Path) -> dict:
    import trimesh

    if (outputDir / "manifest.json").exists():
        raise FileExistsError("Bundle already exists. Choose a new outputDir so reviewed templates cannot be overwritten.")
    sourceMesh = trimesh.load(uvFaceFusionMesh, process=False, maintain_order=True)
    if not isinstance(sourceMesh, trimesh.Trimesh):
        raise ValueError("UVFaceFusion source must be one triangle mesh")
    sourceUvs = np.load(uvFaceFusionUvs, allow_pickle=False)
    if sourceUvs.shape != (len(sourceMesh.vertices), 2) or not np.isfinite(sourceUvs).all():
        raise ValueError("UVFaceFusion must have one finite 2D UV coordinate per vertex")
    fixedMesh = ObjMesh(np.asarray(sourceMesh.vertices), sourceUvs, np.asarray(sourceMesh.faces), np.asarray(sourceMesh.faces))
    if not np.isfinite(fixedMesh.vertices).all() or fixedMesh.faces.ndim != 2 or fixedMesh.faces.shape[1] != 3:
        raise ValueError("UVFaceFusion must have finite vertices and triangle faces")
    floatingMesh = readObj(ictMesh)
    originalsDir = outputDir / "originals"
    originalsDir.mkdir(parents=True, exist_ok=True)
    originalCopies = [(uvFaceFusionMesh, originalsDir / "canonical_mesh_flame.ply"),
                      (uvFaceFusionUvs, originalsDir / "canonical_mesh_flame_uvs.npy"),
                      (ictMesh, originalsDir / "open_eye_template.obj")]
    for sourcePath, copyPath in originalCopies:
        shutil.copyfile(sourcePath, copyPath)
    writeObj(outputDir / "uvFaceFusionFixed.obj", fixedMesh)
    writeObj(outputDir / "ictFloating.obj", floatingMesh)
    createUvChecker(outputDir / "uvChecker.png")
    (outputDir / "uvChecker.mtl").write_text("newmtl uvChecker\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd uvChecker.png\n", encoding="utf-8")
    fileRoles = {"uvFaceFusionFixed.obj": "immutableFixedGeometry", "ictFloating.obj": "immutableFloatingReference",
                 "uvChecker.png": "uvDiagnosticOnly", "uvChecker.mtl": "uvDiagnosticOnly",
                 "originals/canonical_mesh_flame.ply": "originalUvFaceFusionGeometry",
                 "originals/canonical_mesh_flame_uvs.npy": "originalUvFaceFusionUvs",
                 "originals/open_eye_template.obj": "originalIctGeometryAndUvs"}
    manifest = {"schemaVersion": 1, "status": "awaitingManualRegistration", "fixedMesh": "uvFaceFusionFixed.obj",
                "floatingMesh": "ictFloating.obj", "expectedResult": "ictRegisteredToUvFaceFusion.obj",
                "direction": "deform ICT floating onto unchanged UVFaceFusion fixed",
                "automaticTransformsApplied": [], "resultCoordinates": "UVFaceFusion original coordinate system and native numerical units",
                "fixed": meshSummary(fixedMesh), "floating": meshSummary(floatingMesh),
                "semanticReviewRequired": True, "files": {}}
    for relativePath, role in fileRoles.items():
        filePath = outputDir / relativePath
        manifest["files"][relativePath] = {"sha256": sha256File(filePath), "byteSize": filePath.stat().st_size, "role": role}
    manifest["sources"] = [{"fileName": sourcePath.name, "sha256": sha256File(sourcePath),
                             "bundleCopy": str(copyPath.relative_to(outputDir))} for sourcePath, copyPath in originalCopies]
    (outputDir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    projectRoot = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvFaceFusionMesh", type=Path, default=projectRoot / "01_UVFaceFusion/uvfacefusion/assets/canonical_mesh_flame.ply")
    parser.add_argument("--uvFaceFusionUvs", type=Path, default=projectRoot / "01_UVFaceFusion/uvfacefusion/assets/canonical_mesh_flame_uvs.npy")
    parser.add_argument("--ictMesh", type=Path, default=projectRoot / "04_STFR/stfr/registration/assets/open_eye_template.obj")
    parser.add_argument("--outputDir", type=Path, default=projectRoot / "assets/wrapPreparation")
    args = parser.parse_args()
    manifest = prepareBundle(args.uvFaceFusionMesh, args.uvFaceFusionUvs, args.ictMesh, args.outputDir)
    print(json.dumps({"bundleDir": str(args.outputDir.resolve()), "fixed": manifest["fixed"], "floating": manifest["floating"],
                      "status": manifest["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
