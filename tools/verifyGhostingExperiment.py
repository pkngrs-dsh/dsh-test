"""Independently inspect saved anti-ghosting experiment contracts and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / "src"), str(projectRoot / "tools")]
from faceAppearance.artifacts import writeJson
from runReferenceExperiment import fileHash, readArrays


def verifyExperiment(directory: Path):
    directory = directory.resolve()
    report = json.loads((directory / "experimentReport.json").read_text())
    source = Path(report["source"])
    for name, digest in report["sourceHashes"].items():
        assert fileHash(source / name) == digest, f"Source changed: {name}"
    for name, digest in report["outputHashes"].items():
        assert fileHash(directory / name) == digest, f"Output changed: {name}"
    for name, digest in report["implementationHashes"].items():
        assert fileHash(directory / "implementation" / name) == digest, f"Snapshot changed: {name}"
    for name in ("geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"):
        assert fileHash(source / name) == fileHash(directory / name), f"Frozen frame changed: {name}"
    fitted = readArrays(source / "semanticProtectedShFit.npz")
    baseline = readArrays(directory / "branches/baseline.npz")
    previous = readArrays(source / "branches/semanticProtected.npz")
    alignment = readArrays(directory / "alignment.npz")
    reference = report["referenceViewIndex"]
    assert np.array_equal(baseline["linearAlbedo"], previous["linearAlbedo"])
    assert np.array_equal(alignment["alignedObservations"][reference], fitted["correctedObservations"][reference])
    assert np.array_equal(alignment["alignedWeights"][reference], fitted["validWeights"][reference])
    assert not alignment["acceptedMask"][reference].any()
    assert not alignment["samplingFlow"][reference].any()
    assert np.array_equal(alignment["alignedWeights"] > 0, fitted["validWeights"] > 0)
    assert np.all(alignment["alignedWeights"] <= fitted["validWeights"])
    rejected = ~alignment["acceptedMask"]
    assert np.array_equal(alignment["alignedObservations"][rejected], fitted["correctedObservations"][rejected])
    magnitudes = np.linalg.norm(alignment["samplingFlow"], axis=-1)
    assert magnitudes.max() <= report["maxDisplacementUvPixels"]
    jacobianRanges = []
    for flow in alignment["samplingFlow"]:
        dxDy, dxDx = np.gradient(flow[..., 0])
        dyDy, dyDx = np.gradient(flow[..., 1])
        determinant = (1+dxDx)*(1+dyDy)-dxDy*dyDx
        assert determinant.min() >= .35 and determinant.max() <= 2.5
        jacobianRanges.append([float(determinant.min()), float(determinant.max())])
    core = baseline["referenceCoreMask"] | baseline["protectionMask"]
    counts = {}
    for name in report["branches"]:
        branch = readArrays(directory / f"branches/{name}.npz")
        coverage = branch["coverageMask"]
        assert np.array_equal(coverage, baseline["coverageMask"])
        assert np.array_equal(branch["linearAlbedo"][core], fitted["correctedObservations"][reference][core])
        assert np.isfinite(branch["linearAlbedo"]).all()
        assert np.min(branch["linearAlbedo"]) >= 0 and np.max(branch["linearAlbedo"]) <= 1
        if "highFrequencySourceIndex" in branch:
            owners = branch["highFrequencySourceIndex"]
            assert np.all((owners[coverage] >= 0) & (owners[coverage] < len(fitted["validWeights"])))
            yy, xx = np.nonzero(coverage)
            weights = alignment["alignedWeights"] if name == "combined" else fitted["validWeights"]
            assert np.all(weights[owners[coverage], yy, xx] > 0)
        counts[name] = int(coverage.sum())
    result = {"status": "verified", "experimentSha256": fileHash(directory / "experimentReport.json"),
              "sourceAndSnapshotHashesVerified": True, "fixedGeometryAndCamerasVerified": True,
              "referenceObservationsAndProtectedColorsBitExact": True,
              "coverageBitExactAcrossBranches": True, "rejectedSamplesBitExact": True,
              "maximumFlowUvPixels": float(magnitudes.max()), "jacobianRanges": jacobianRanges,
              "branchCoverageTexels": counts,
              "limits": ["Invariants verify implementation contracts, not visual quality or true correspondences"]}
    writeJson(directory / "verificationReport.json", result)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=projectRoot / "runs/ghostingRefinementExperiment")
    args = parser.parse_args()
    verifyExperiment(args.input)
