"""Compare legacy NICP UV interpolation to the independent 5346-sample chart.

No solver or geometry deformation is run. The frozen barycentric audit and
mapping cache are inputs only; a separate consistency report is written.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceAppearance.clientTemplate import readLegacyObj
from faceAppearance.legacyBarycentric import compileLegacyUvSamples


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return {"count": len(values), "mean": float(values.mean()),
            **dict(zip(("min", "median", "p90", "p95", "p99", "max"),
                       map(float, np.quantile(values, [0, .5, .9, .95, .99, 1]))))}


def auditConsistency(auditDir, nicpUvPath):
    auditDir, nicpUvPath = Path(auditDir), Path(nicpUvPath)
    reportPath = auditDir / "uvAssetConsistency.json"
    if reportPath.exists():
        raise FileExistsError(f"Refusing to replace an existing audit: {reportPath}")
    frozenHashes = {name: fileHash(auditDir / name) for name in ("audit.json", "mapping.npz")}
    audit = json.loads((auditDir / "audit.json").read_text())
    mapping = np.load(auditDir / "mapping.npz", allow_pickle=False)
    old = readLegacyObj(nicpUvPath)
    female = readLegacyObj(audit["sources"]["femaleNicp"]["path"])
    if not np.array_equal(old["faces"], female["faces"]):
        raise ValueError("NICP UV asset changed the triangle order addressed by the barycentric table")
    source = readLegacyObj(audit["sources"]["sourceMesh"]["path"])
    sourceUvs = np.load(audit["sources"]["sourceUvs"]["path"], allow_pickle=False)
    faceIds, weights = mapping["sampleNicpFaceIds"], mapping["sampleNicpWeights"]
    corners = old["uvs"][old["uvFaces"][faceIds]]
    baselineUvs = (corners*weights[..., None]).sum(1)
    refinedUvs = mapping["sampleUvs"]
    differences = baselineUvs-refinedUvs
    uvDistances = np.linalg.norm(differences, axis=1)
    texelDistances = uvDistances*1023
    baselineMapping = compileLegacyUvSamples(baselineUvs, source["vertices"], source["faces"], sourceUvs)
    validBary = mapping["sampleBaryValidMask"]
    refinedValid = mapping["sampleValidMask"]
    jointValid = refinedValid & baselineMapping["sampleUvValidMask"]
    geometryDistances = np.linalg.norm(baselineMapping["sampleSourcePoints"]-mapping["sampleSourcePoints"], axis=1)

    # Identify nearby chart seams without claiming every large difference is
    # caused by a seam. UV endpoints sharing one geometry vertex may belong to
    # different islands; such a face needs separate review.
    vertexUvIds = [[] for _ in old["vertices"]]
    for vertexId, uvId in zip(old["faces"].ravel(), old["uvFaces"].ravel()):
        vertexUvIds[vertexId].append(int(uvId))
    vertexSeamSpan = np.zeros(len(vertexUvIds))
    for vertexId, ids in enumerate(vertexUvIds):
        points = old["uvs"][np.unique(ids)]
        vertexSeamSpan[vertexId] = np.linalg.norm(points[:, None]-points[None, :], axis=-1).max()
    sampleSeamSpan = vertexSeamSpan[old["faces"][faceIds]].max(1)*1023
    cornerSpan = np.linalg.norm(corners[:, :, None]-corners[:, None, :], axis=-1).max(axis=(1, 2))*1023

    def summary(mask):
        comparable = mask & validBary
        joint = mask & jointValid
        return {"baryValidSamples": int(comparable.sum()),
                "refinedUvValidSamples": int((mask & refinedValid).sum()),
                "bothUvValidSamples": int(joint.sum()),
                "refinedValidButBaselineUvUnsupported": int((mask & refinedValid & ~baselineMapping["sampleUvValidMask"]).sum()),
                "uvDistanceAllBaryValid": stats(uvDistances[comparable]),
                "texelDistance1024AllBaryValid": stats(texelDistances[comparable]),
                "uvDistanceBothSupported": stats(uvDistances[joint]),
                "texelDistance1024BothSupported": stats(texelDistances[joint]),
                "geometryDistanceBothSupported": stats(geometryDistances[joint]),
                "signedTexelOffset1024MeanAllBaryValid": (differences[comparable]*1023).mean(0).tolist() if comparable.any() else None,
                "largeUvDifferenceCountsAllBaryValid": {str(threshold): int((comparable & (texelDistances > threshold)).sum())
                                                         for threshold in (8, 16, 32, 64, 128)},
                "touchesGeometryVertexWithUvBranchesOver32Texels": int((comparable & (sampleSeamSpan > 32)).sum())}

    def detail(sampleId):
        index = int(sampleId)
        return {"sampleId": index, "nicpFaceId": int(faceIds[index]),
                "refinedUv": refinedUvs[index].tolist(), "nicpInterpolatedUv": baselineUvs[index].tolist(),
                "texelDistance1024": float(texelDistances[index]),
                "refinedUvSupported": bool(refinedValid[index]),
                "nicpUvSupported": bool(baselineMapping["sampleUvValidMask"][index]),
                "geometryDistanceIfBothSupported": float(geometryDistances[index]) if jointValid[index] else None,
                "adjacentVertexUvBranchSpanTexels1024": float(sampleSeamSpan[index]),
                "sampleFaceUvCornerSpanTexels1024": float(cornerSpan[index])}

    regions = {}
    for name, region in audit["regions"].items():
        mask = np.zeros(len(faceIds), bool)
        mask[region["sampleIds"]] = True
        ids = np.flatnonzero(mask & validBary)
        worst = ids[np.argsort(texelDistances[ids])[-5:][::-1]]
        regions[name] = {"selection": region["selection"], **summary(mask),
                         "largestUvDifferences": [detail(index) for index in worst]}
    comparableIds = np.flatnonzero(validBary)
    worst = comparableIds[np.argsort(texelDistances[comparableIds])[-30:][::-1]]
    seamRelated = validBary & (texelDistances > 64) & (sampleSeamSpan > 32)
    largeWithoutSeam = validBary & (texelDistances > 64) & ~(sampleSeamSpan > 32)
    result = {"schemaVersion": 1, "status": "legacyUvAssetsDisagreeBeforeAnyNewSolver",
              "comparison": "For each old refined sample i: NICP face+bary interpolated through old NICP standard-OBJ UV, versus refined RAW native-V-down UV",
              "sampleCount": len(faceIds), "uvConvention": "Both compared arrays are V-down; only standard NICP OBJ vt receives 1-v",
              "texelConvention": "1024 texture integer centers: uv*(1024-1), Euclidean pixel distance",
              "geometryConvention": "Both UVs independently query the identical unchanged UFF canonical surface; no fitting, registration, pose, or solver applied",
              "geometryUnits": "uncalibrated native UFF canonical units",
              "global": summary(np.ones(len(faceIds), bool)), "regions": regions,
              "largeJumpReview": {"definition": "UV separation >64 texels; seam risk means an adjacent NICP geometry vertex has multiple UV branches separated >32 texels",
                                  "seamRiskSampleIds": np.flatnonzero(seamRelated).tolist(),
                                  "largeDifferenceWithoutDetectedSeamSampleIds": np.flatnonzero(largeWithoutSeam).tolist(),
                                  "largestDifferences": [detail(index) for index in worst],
                                  "note": "Large differences are not automatically attributed to seams; this separates seam-adjacent candidates from other chart correspondence disagreements."},
              "refinedValidBaselineUnsupportedSampleIds": np.flatnonzero(refinedValid & ~baselineMapping["sampleUvValidMask"]).tolist(),
              "interpretation": ["The two historical UV correspondences are not interchangeable, despite sharing NICP face indices.",
                                 "These differences exist on unchanged rest geometry before a new solver runs; they cannot be attributed solely to its deformation.",
                                 "A geometry branch driven by refined RAW UV can mismatch a texture still transferred through the old NICP UV chart.",
                                 "The oralAperture subset refers to the internal topology rim, not the visible lip-contact line.",
                                 "Unsupported UV samples use nearest-surface output only as diagnostics; their 3D discrepancy is excluded from supported comparison statistics.",
                                 "No geometry, texture, frozen mapping, frozen audit, or active assets were changed."],
              "sources": {"frozenAudit": {"path": str(auditDir / "audit.json"), "sha256": frozenHashes["audit.json"]},
                          "frozenMapping": {"path": str(auditDir / "mapping.npz"), "sha256": frozenHashes["mapping.npz"]},
                          "nicpUvAsset": {"path": str(nicpUvPath), "sha256": fileHash(nicpUvPath)},
                          "uffMesh": audit["sources"]["sourceMesh"], "uffNativeUvs": audit["sources"]["sourceUvs"]},
              "implementationSha256": fileHash(__file__)}
    for name, expected in frozenHashes.items():
        if fileHash(auditDir / name) != expected:
            raise RuntimeError(f"Frozen input changed during audit: {name}")
    reportPath.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditDir", type=Path, default=root / "reports/legacyBarycentricAudit")
    parser.add_argument("--nicpUvAsset", type=Path, default=Path("/path/to/legacy-template/template_nicp_aligned_uv_in_flame.obj"))
    args = parser.parse_args()
    result = auditConsistency(args.auditDir, args.nicpUvAsset)
    print(json.dumps({"report": str(args.auditDir / "uvAssetConsistency.json"), "global": result["global"],
                      "largeSeamRiskCount": len(result["largeJumpReview"]["seamRiskSampleIds"]),
                      "largeNonSeamCount": len(result["largeJumpReview"]["largeDifferenceWithoutDetectedSeamSampleIds"])}, indent=2))


if __name__ == "__main__":
    main()
