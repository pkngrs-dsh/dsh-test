#!/usr/bin/env python3
"""Check immutable Wrap inputs and returned ICT structure; never approve semantic registration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .prepareWrapBundle import meshSummary, readObj, sha256File
except ImportError:
    from prepareWrapBundle import meshSummary, readObj, sha256File


def validateResult(bundleDir: Path, resultPath: Path | None = None, uvTolerance: float = 1e-6) -> dict:
    if not np.isfinite(uvTolerance) or uvTolerance < 0 or uvTolerance > 1e-5:
        raise ValueError("uvTolerance must be finite and between 0 and 1e-5")
    errors, warnings = [], []
    report = {"bundleDir": str(bundleDir.resolve()), "resultPath": str(resultPath.resolve()) if resultPath else None,
              "inputIntegrityValid": False, "structuralValid": None, "registrationAccepted": False,
              "semanticReviewRequired": True, "uvTolerance": uvTolerance, "errors": errors, "warnings": warnings}
    try:
        manifest = json.loads((bundleDir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schemaVersion") != 1:
            raise ValueError("Unsupported manifest schemaVersion")
        requiredFiles = {"uvFaceFusionFixed.obj", "ictFloating.obj", "originals/canonical_mesh_flame.ply",
                         "originals/canonical_mesh_flame_uvs.npy", "originals/open_eye_template.obj", "uvChecker.mtl", "uvChecker.png"}
        if not requiredFiles.issubset(manifest["files"]):
            raise ValueError("Manifest lacks required immutable inputs")
        for relativePath, record in manifest["files"].items():
            filePath = (bundleDir / relativePath).resolve()
            if not filePath.is_relative_to(bundleDir.resolve()):
                raise ValueError("Manifest path must stay inside the bundle")
            if not filePath.is_file():
                errors.append(f"Missing immutable input: {relativePath}")
            elif filePath.stat().st_size != record["byteSize"] or sha256File(filePath) != record["sha256"]:
                errors.append(f"Immutable input changed: {relativePath}")
        report["inputIntegrityValid"] = not errors
        if not resultPath:
            warnings.append("No Wrap result supplied; input integrity only, no registration has been validated.")
            return report
        report["structuralValid"] = False
        if errors:
            return report
        referenceMesh = readObj(bundleDir / "ictFloating.obj")
        fixedMesh = readObj(bundleDir / "uvFaceFusionFixed.obj")
        resultMesh = readObj(resultPath)
        report["result"] = meshSummary(resultMesh)
        report["resultSha256"] = sha256File(resultPath)
        if resultMesh.vertices.shape != referenceMesh.vertices.shape:
            errors.append(f"Vertex count changed: expected {len(referenceMesh.vertices)}, got {len(resultMesh.vertices)}")
        if not np.array_equal(resultMesh.faces, referenceMesh.faces):
            errors.append("Triangle vertex indices/order/winding changed")
        if not np.array_equal(resultMesh.uvFaces, referenceMesh.uvFaces):
            errors.append("Triangle UV indices/order changed")
        if resultMesh.texCoords.shape != referenceMesh.texCoords.shape:
            errors.append("Texture-coordinate count changed")
        elif not np.allclose(resultMesh.texCoords, referenceMesh.texCoords, atol=uvTolerance, rtol=0):
            errors.append("Texture-coordinate values changed")
        triangleVertices = resultMesh.vertices[resultMesh.faces]
        areas = np.linalg.norm(np.cross(triangleVertices[:, 1] - triangleVertices[:, 0],
                                       triangleVertices[:, 2] - triangleVertices[:, 0]), axis=1) / 2
        fixedExtent = float(np.max(np.ptp(fixedMesh.vertices, axis=0)))
        tinyAreaThreshold = max(fixedExtent ** 2 * 1e-12, np.finfo(float).tiny)
        degenerateCount = int(np.count_nonzero(areas <= tinyAreaThreshold))
        report["nearZeroTriangleCount"] = degenerateCount
        if degenerateCount:
            errors.append(f"Result contains {degenerateCount} near-zero-area triangles")
        if resultMesh.vertices.shape == referenceMesh.vertices.shape:
            report["geometryChanged"] = not np.array_equal(resultMesh.vertices, referenceMesh.vertices)
            if not report["geometryChanged"]:
                warnings.append("ICT coordinates are unchanged; this appears to be an unregistered template.")
        resultExtent = float(np.max(np.ptp(resultMesh.vertices, axis=0)))
        extentRatio = resultExtent / fixedExtent if fixedExtent else None
        report["resultToFixedExtentRatio"] = extentRatio
        if extentRatio is not None and not 0.5 <= extentRatio <= 2.0:
            warnings.append("Result/fixed extent ratio is outside [0.5, 2]; check alignment and export units manually.")
        warnings.append("Structural checks cannot prove anatomical correspondence, absence of folds, UV seam quality, or alignment. Manual semantic review is required.")
        report["structuralValid"] = not errors
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
        if resultPath:
            report["structuralValid"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundleDir", type=Path, default=Path(__file__).resolve().parents[1] / "assets/wrapPreparation")
    parser.add_argument("--result", type=Path, help="ICT OBJ returned by Wrap; omit to check bundle inputs only")
    parser.add_argument("--uvTolerance", type=float, default=1e-6)
    parser.add_argument("--reportPath", type=Path)
    args = parser.parse_args()
    report = validateResult(args.bundleDir, args.result, args.uvTolerance)
    reportText = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.reportPath:
        protectedPaths = {(args.bundleDir / "manifest.json").resolve()}
        manifestPath = args.bundleDir / "manifest.json"
        if manifestPath.is_file():
            manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
            protectedPaths.update((args.bundleDir / relativePath).resolve() for relativePath in manifest.get("files", {}))
        if args.result:
            protectedPaths.add(args.result.resolve())
        if args.reportPath.resolve() in protectedPaths:
            parser.error("reportPath must not overwrite a bundle input, manifest, or returned result")
        args.reportPath.parent.mkdir(parents=True, exist_ok=True)
        args.reportPath.write_text(reportText, encoding="utf-8")
    print(reportText, end="")
    raise SystemExit(0 if report["inputIntegrityValid"] and not report["errors"] else 1)


if __name__ == "__main__":
    main()
