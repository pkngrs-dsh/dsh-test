"""Compile the old 5346 surface correspondences for direct registration.

This creates separate experimental client bundles. Client topology, materials
and the existing texture lookup remain byte-identical to their base bundles.
The compiled geometry constraints use only valid old JSON/UV surface samples;
vertex-UV geometry weights and rest-distance/normal gates are not consulted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from faceAppearance.clientTemplate import readLegacyObj
from faceAppearance.correspondence import arrayHash, sampleSurface
from faceAppearance.legacyBarycentric import readLegacyBarycentric


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _readArchive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _sourcePath(audit, name):
    record = audit["sources"][name]
    path = Path(record["path"])
    if not path.is_file() or fileHash(path) != record["sha256"]:
        raise ValueError(f"Audited source hash mismatch: {name}")
    return path


def _sameArrays(actual, expected, label, *, tolerance=None):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Audited array shape differs: {label}")
    matched = (np.array_equal(actual, expected) if tolerance is None else
               np.allclose(actual, expected, atol=tolerance, rtol=0))
    if not matched:
        raise ValueError(f"Audited array differs: {label}")


def _nondegenerate(vertices, faces):
    triangles = vertices[faces]
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-14


def compileDirectRegistration(baseAsset, auditDir, outputDir):
    """Create a new bundle and return its manifest; never overwrite an asset."""
    baseAsset, auditDir, outputDir = (Path(value).resolve()
                                      for value in (baseAsset, auditDir, outputDir))
    if outputDir.exists() and (not outputDir.is_dir() or any(outputDir.iterdir())):
        raise FileExistsError("Choose a new or empty direct-registration asset directory")
    baseManifestPath = baseAsset / "manifest.json"
    baseManifest = json.loads(baseManifestPath.read_text(encoding="utf-8"))
    gender = baseManifest.get("gender")
    if gender not in ("male", "female"):
        raise ValueError("Base asset must declare male or female")
    for relative, expectedHash in baseManifest["files"].items():
        path = (baseAsset / relative).resolve()
        if not path.is_relative_to(baseAsset) or not path.is_file() or fileHash(path) != expectedHash:
            raise ValueError(f"Base bundle hash mismatch: {relative}")
    base = _readArchive(baseAsset / "template.npz")
    auditPath = auditDir / "audit.json"
    audit = json.loads(auditPath.read_text(encoding="utf-8"))
    mappingPath = auditDir / "mapping.npz"
    if audit.get("mappingFile") != "mapping.npz" or fileHash(mappingPath) != audit["mappingSha256"]:
        raise ValueError("Audited mapping hash mismatch")
    if audit.get("sampleCount") != 5346 or audit.get("chosenUvConvention") != "rawNativeVDown":
        raise ValueError("Expected the audited 5346-sample raw V-down mapping")
    mapping = _readArchive(mappingPath)
    sourcePaths = {name: _sourcePath(audit, name) for name in
                   ("femaleNicp", "maleNicp", "barycentric", "refined", "sourceMesh", "sourceUvs")}
    meshes = {name: readLegacyObj(sourcePaths[name]) for name in
              ("femaleNicp", "maleNicp", "refined", "sourceMesh")}
    _sameArrays(meshes["femaleNicp"]["faces"], meshes["maleNicp"]["faces"], "male/female NICP face order")
    _sameArrays(base["nicpFaces"], meshes[f"{gender}Nicp"]["faces"], "base NICP face order")
    _sameArrays(base["vertices"][base["nicpToFullIds"]], meshes[f"{gender}Nicp"]["vertices"],
                "base NICP rest vertices", tolerance=1e-9)
    if baseManifest["templateHashes"]["nicpMesh"] != audit["sources"][f"{gender}Nicp"]["sha256"]:
        raise ValueError("Base NICP source identity differs from the audited source")
    sourceUvs = np.load(sourcePaths["sourceUvs"], allow_pickle=False).astype(np.float32)
    sourceMesh = meshes["sourceMesh"]
    topologyHash = arrayHash(sourceMesh["faces"].astype(np.int64), sourceUvs)
    if topologyHash != audit["topology"]["sourceTopologyHash"] or topologyHash != baseManifest["sourceTopologyHash"]:
        raise ValueError("UFF topology identity differs from the audit or base asset")
    _sameArrays(base["sourceFaces"], sourceMesh["faces"], "UFF face order")
    _sameArrays(base["sourceUvs"], sourceUvs, "UFF native UVs")
    _sameArrays(base["sourceVertices"], sourceMesh["vertices"], "UFF canonical rest positions", tolerance=1e-9)
    refined = meshes["refined"]
    if len(refined["vertices"]) != 5346 or len(refined["uvs"]) != 5346:
        raise ValueError("Refined sample count differs from the 5346-row table")
    _sameArrays(refined["faces"], refined["uvFaces"], "refined vertex/UV corner order")
    table = readLegacyBarycentric(sourcePaths["barycentric"], base["nicpFaces"], boundaryTolerance=0.)
    for key in ("sampleIds", "sampleNicpFaceIds", "sampleNicpWeights", "sampleBaryValidMask"):
        _sameArrays(mapping[key], table[key], key)
    nativeUvs = refined["uvs"].copy()
    nativeUvs[:, 1] = 1 - nativeUvs[:, 1]
    _sameArrays(mapping["sampleUvs"], nativeUvs, "audited raw native sample UVs")
    sourceIds, sourceWeights = mapping["sourceTriangleIds"], mapping["sourceBarycentricWeights"]
    if (sourceIds.shape != (5346,) or sourceIds.dtype.kind not in "iu"
            or (sourceIds < 0).any() or (sourceIds >= len(base["sourceFaces"])).any()
            or sourceWeights.shape != (5346, 3) or not np.isfinite(sourceWeights).all()
            or (sourceWeights < 0).any() or (sourceWeights > 1).any()
            or not np.allclose(sourceWeights.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Audited UFF triangle/barycentric coordinates are malformed")
    sourceMapping = {"triangleIds": sourceIds, "barycentricWeights": sourceWeights}
    sourcePoints = sampleSurface(base["sourceVertices"], base["sourceFaces"], sourceMapping)
    _sameArrays(sourcePoints, mapping["sampleSourcePoints"], "UFF sample rest points", tolerance=1e-6)
    uvPoints = (base["sourceUvs"][base["sourceFaces"][sourceIds]] * sourceWeights[..., None]).sum(axis=1)
    uvDistances = np.linalg.norm(uvPoints - nativeUvs, axis=1)
    # Distances in the audited cache were float32; re-evaluation allows only
    # that numerical rounding, while qualification retains the audited cutoff.
    _sameArrays(uvDistances, mapping["sourceUvDistances"], "UV support distances", tolerance=2e-7)
    uvValid = mapping["sourceUvDistances"] <= 1e-5
    _sameArrays(mapping["sampleUvValidMask"], uvValid, "UV support mask")
    valid = table["sampleBaryValidMask"] & uvValid
    _sameArrays(mapping["sampleValidMask"], valid, "combined validity mask")
    if int(valid.sum()) != audit["validSamples"]:
        raise ValueError("Valid sample count differs from the audit")
    # Degenerate geometry is not a usable surface constraint. This is separate
    # from subjective rest-distance or normal-agreement qualification.
    geometryValid = (_nondegenerate(base["sourceVertices"], base["sourceFaces"])[sourceIds]
                     & _nondegenerate(meshes[f"{gender}Nicp"]["vertices"], base["nicpFaces"])
                       [table["sampleNicpFaceIds"]])
    keep = valid & geometryValid
    if int(keep.sum()) < 100:
        raise ValueError("Too few valid direct surface constraints")
    staticMask = np.asarray(base["staticMask"])
    if staticMask.dtype.kind != "b" or staticMask.shape != (len(base["nicpToFullIds"]),):
        raise ValueError("Expected the base template's boolean NICP static mask")
    arrays = {
        "originalSampleIds": table["sampleIds"][keep],
        "sampleNicpFaceIds": table["sampleNicpFaceIds"][keep],
        "sampleNicpWeights": table["sampleNicpWeights"][keep],
        "sourceTriangleIds": sourceIds[keep],
        "sourceBarycentricWeights": sourceWeights[keep],
        "sampleWeights": np.ones(int(keep.sum()), dtype=np.float64),
        "staticMask": staticMask.copy(), "staticVertexIds": np.flatnonzero(staticMask),
    }
    staticAdjacent = staticMask[base["nicpFaces"][arrays["sampleNicpFaceIds"]]].any(axis=1)
    qualification = {
        "rule": "Valid old JSON barycentrics AND UV support distance <= 1e-5 AND nondegenerate surface triangles",
        "totalLegacySamples": 5346, "retainedSamples": int(keep.sum()),
        "invalidBarycentricSampleIds": table["sampleIds"][~table["sampleBaryValidMask"]].tolist(),
        "outOfUvSupportSampleIds": table["sampleIds"][~uvValid].tolist(),
        "degenerateGeometrySampleIds": table["sampleIds"][valid & ~geometryValid].tolist(),
        "excludedSampleIds": table["sampleIds"][~keep].tolist(),
        "retainedStaticAdjacentSamples": int(staticAdjacent.sum()),
        "usesVertexUvGeometryWeights": False, "usesRestDistanceGate": False, "usesNormalGate": False,
        "rejectStaticAdjacentFaces": False, "retainedSampleWeight": 1.,
    }
    provenance = {
        "schemaVersion": 1, "gender": gender, "baseTemplateId": baseManifest["templateId"],
        "baseManifestSha256": fileHash(baseManifestPath), "auditSha256": fileHash(auditPath),
        "mappingSha256": fileHash(mappingPath), "sourceTopologyHash": topologyHash,
        "nicpFaceOrderHash": arrayHash(base["nicpFaces"]), "maleFemaleNicpFaceOrderEqual": True,
        "sourceHashes": {name: audit["sources"][name]["sha256"] for name in sourcePaths},
        "staticMaskHash": arrayHash(staticMask), "staticVertexCount": int(staticMask.sum()),
        "staticAnchorPolicy": "Base template staticMask is available for soft rest-position anchors; no sample rejection",
        "qualification": qualification,
        "textureStatus": "Existing base atlas lookup retained byte-for-byte; consistency with direct geometry is unresolved",
    }
    outputDir.mkdir(parents=True, exist_ok=True)
    for relative in baseManifest["files"]:
        destination = outputDir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseAsset / relative, destination)
    np.savez_compressed(outputDir / "directRegistration.npz", **arrays)
    (outputDir / "directRegistrationSources.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = copy.deepcopy(baseManifest)
    manifest.update(templateId=outputDir.name, baseTemplateId=baseManifest["templateId"],
                    mappingStatus="directSurfaceLegacySamplesAnatomyUnverified",
                    mappingMethod="Old sample ID -> client face/barycentrics AND independently -> UFF face/barycentrics")
    manifest["templateHashes"].update(baseAssetManifest=fileHash(baseManifestPath),
                                      directRegistrationAudit=fileHash(auditPath),
                                      directRegistrationMapping=fileHash(mappingPath))
    manifest["geometryTransfer"] = {
        "method": "directSurface", "asset": "directRegistration.npz",
        "alignmentMode": "directSurfaceSamples", "qualification": qualification,
        "staticAnchors": {"source": "template.staticMask", "mode": "soft", "count": int(staticMask.sum())},
    }
    manifest["textureTransfer"] = {"method": "baseAtlasLookupUnchanged",
                                   "baseTemplateId": baseManifest["templateId"],
                                   "geometryConsistencyValidated": False}
    manifest["baseGeometryDiagnostics"] = manifest.pop("diagnostics", {})
    manifest["baseGeometryLimits"] = manifest.pop("limits", {})
    manifest["limitations"] = [
        "Experimental surface registration correspondence; no anatomical, registration-quality or likeness certification.",
        "Texture sampling retains the base bundle lookup and its qualification; geometry/texture consistency remains unresolved.",
        "Static regions supply soft anchors; downstream registration may move them rather than fixing them exactly.",
        "Original vertex/triangle/UV/material numbering is retained; skeleton, skin weights and expression behavior are unverified.",
    ]
    manifest["compiler"] = {"path": str(Path(__file__).resolve()), "sha256": fileHash(__file__)}
    manifest["files"] = {path.relative_to(outputDir).as_posix(): fileHash(path)
                         for path in sorted(outputDir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    (outputDir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseAsset", type=Path, required=True)
    parser.add_argument("--auditDir", type=Path, default=PROJECT_ROOT / "reports/legacyBarycentricAudit")
    parser.add_argument("--outputDir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compileDirectRegistration(**vars(args))
    print(json.dumps({"outputDir": str(args.outputDir), "templateId": manifest["templateId"],
                      **manifest["geometryTransfer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
