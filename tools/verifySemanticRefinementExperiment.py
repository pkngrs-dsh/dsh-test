"""Verify saved refinement contracts independently of the fitting objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeJson
from renderCropComparison import fileHash
from runReferenceExperiment import readArrays


def verifyExperiment(source: Path, comparison: Path):
    source, comparison = Path(source).resolve(), Path(comparison).resolve()
    report = json.loads((source / "experimentReport.json").read_text())
    rendering = json.loads((comparison / "comparisonReport.json").read_text())
    assert report["status"] == "prepared" and rendering["status"] == "rendered"
    assert rendering["experimentReportSha256"] == fileHash(source / "experimentReport.json")
    for name, digest in report["outputHashes"].items():
        assert fileHash(source / name) == digest, name
    previous = Path(report["source"])
    for name in ("cameras.json", "inputs.json", "viewTransforms.json", "geometry.npz"):
        assert fileHash(source / name) == fileHash(previous / name), name
    original, refined = (readArrays(source / name) for name in ("geometry.npz", "geometryRefined.npz"))
    for name in ("faces", "uvs"):
        np.testing.assert_array_equal(original[name], refined[name])
    baseline, semantic = (readArrays(source / "branches" / f"{name}.npz")
                           for name in ("baseline", "semanticProtected"))
    cached = readArrays(previous / "branches/referenceFirst.npz")
    for name in ("linearAlbedo", "coverageMask"):
        np.testing.assert_array_equal(baseline[name], cached[name])
    np.testing.assert_array_equal(semantic["coverageMask"], baseline["coverageMask"])
    core = cached["referenceCoreMask"]
    np.testing.assert_array_equal(semantic["linearAlbedo"][core], cached["linearAlbedo"][core])
    fitted = readArrays(source / "semanticProtectedShFit.npz")
    protection = semantic["protectionMask"]
    referenceIndex = report["referenceViewIndex"]
    np.testing.assert_array_equal(semantic["linearAlbedo"][protection],
                                  fitted["correctedObservations"][referenceIndex][protection])
    assert not np.any(protection & (fitted["validWeights"][referenceIndex] <= 0))
    faces = original["faces"]
    def triangleCross(vertices):
        return np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    rest, changed = triangleCross(original["vertices"].astype(float)), triangleCross(refined["vertices"].astype(float))
    restLength, changedLength = np.linalg.norm(rest, axis=1), np.linalg.norm(changed, axis=1)
    areaRatio = changedLength/restLength
    normalCosine = np.sum(rest*changed, axis=1)/(restLength*changedLength)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1), axis=0)
    edgeLength = lambda verts: np.linalg.norm(verts[edges[:, 1]].astype(float)-verts[edges[:, 0]], axis=1)
    relativeEdge = np.abs(edgeLength(refined["vertices"])/edgeLength(original["vertices"])-1)
    displacement = np.linalg.norm(refined["vertices"].astype(float)-original["vertices"], axis=1)
    assert areaRatio.min() >= .25-1e-4 and normalCosine.min() >= .1-1e-4
    assert relativeEdge.max() <= .35+1e-3  # Exported float32 rounding near very short edges.
    assert displacement.max() <= report["geometryDiagnostics"]["maxDisplacement"]*(1+1e-4)
    for view in rendering["views"]:
        directory = comparison / f"view{view['viewIndex']:02d}"
        common = readArrays(directory / "commonPhotoData.npz")["evaluationMask"]
        for name in report["branches"]:
            saved = readArrays(directory / f"{name}PhotoFit.npz")
            np.testing.assert_array_equal(saved["evaluationMask"], common)
            assert np.isfinite(saved["renderedLinear"]).all()
    checks = {"status": "passed", "cameraInputsAndTransformsByteIdentical": True,
              "vertexFaceUvOrderPreserved": True, "baselineTextureExact": True,
              "semanticCoverageUnchanged": True, "originalReferenceCoreExact": True,
              "semanticProtectedColorExact": True, "everyBranchUsesSameEvaluationPixels": True,
              "savedFloat32Geometry": {"minimumAreaRatio": float(areaRatio.min()),
                                       "minimumNormalCosine": float(normalCosine.min()),
                                       "maximumRelativeEdgeChange": float(relativeEdge.max()),
                                       "maximumDisplacement": float(displacement.max())},
              "experimentReportSha256": fileHash(source / "experimentReport.json"),
              "comparisonReportSha256": fileHash(comparison / "comparisonReport.json"),
              "scope": "Artifact consistency and local mesh safety; not identity, anatomical or physical material accuracy"}
    writeJson(source / "verificationReport.json", checks)
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    verifyExperiment(args.input, args.comparison)
