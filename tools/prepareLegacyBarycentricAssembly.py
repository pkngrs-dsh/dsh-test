"""Compile the audited legacy surface constraints into an optional client asset.

The base mesh, bound UVs, materials and attachment identities are copied
unchanged. Skin deformation constraints change; optional UV correction changes
the source-texture sampling lookup, not the client UVs. Existing bundles are
never overwritten, and the original legacy files are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot/"src"))
from faceAppearance.artifacts import writeJson
from faceAppearance.correspondence import sampleSurface
from faceAppearance.headAssembly import AssemblyAssets


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compileBarycentricAssembly(baseAsset, auditDir, outputDir, legacyRoot, *,
                               dataWeight=20., smoothWeight=1., restWeight=.03,
                               maxRestDistance=.12, minNormalDot=.35, correctTextureUvs=False):
    baseAsset, auditDir, outputDir, legacyRoot = map(Path, (baseAsset, auditDir, outputDir, legacyRoot))
    if outputDir.exists() and any(outputDir.iterdir()):
        raise FileExistsError("Choose a new or empty compiled asset directory")
    if (not np.isfinite(maxRestDistance) or maxRestDistance <= 0
            or not np.isfinite(minNormalDot) or not -1 <= minNormalDot <= 1):
        raise ValueError("Invalid correspondence qualification thresholds")
    base = AssemblyAssets(baseAsset)
    if base.manifest.get("gender") != "female":
        raise ValueError("This audit currently describes the female/260429 client template")
    with np.load(auditDir/"mapping.npz", allow_pickle=False) as archive:
        mapping = {key: archive[key] for key in archive.files}
    sourceMapping = {"triangleIds": mapping["sourceTriangleIds"],
                     "barycentricWeights": mapping["sourceBarycentricWeights"]}
    sourcePoints = sampleSurface(base.arrays["sourceVertices"], base.arrays["sourceFaces"], sourceMapping)
    if not np.allclose(sourcePoints, mapping["sampleSourcePoints"], atol=1e-6, rtol=0):
        raise ValueError("Audited UFF surface differs from the base assembly template")
    oldNicpHash = fileHash(legacyRoot/"female/260429/template_female_tri_nicp.obj")
    if oldNicpHash != base.manifest["templateHashes"]["nicpMesh"]:
        raise ValueError("Legacy client topology/rest version differs from the base asset")
    # Strictly omit malformed/out-of-support rows. The original row IDs survive
    # so the compiled subset can always be traced back to the legacy JSON.
    rowIds = np.flatnonzero(mapping["sampleValidMask"])
    sampleFaces = mapping["sampleNicpFaceIds"][rowIds]
    if sampleFaces.dtype.kind not in "iu" or np.any(sampleFaces < 0) or np.any(sampleFaces >= len(base.arrays["nicpFaces"])):
        raise ValueError("Audited constraints must refer to the existing NICP faces")
    static = base.arrays["staticMask"][base.arrays["nicpFaces"][sampleFaces]].any(axis=1)
    distances = mapping["femaleRestDistances"][rowIds]
    normalDots = mapping["femaleNormalDots"][rowIds]
    qualified = ~static & (distances < maxRestDistance) & (normalDots > minNormalDot)
    if qualified.sum() < 100:
        raise ValueError("Too few qualified surface constraints; inspect the legacy audit")
    arrays = {"originalSampleIds": mapping["sampleIds"][rowIds],
              "sampleUvs": mapping["sampleUvs"][rowIds],
              "sampleNicpFaceIds": sampleFaces,
              "sampleNicpWeights": mapping["sampleNicpWeights"][rowIds],
              "sourceTriangleIds": mapping["sourceTriangleIds"][rowIds],
              "sourceBarycentricWeights": mapping["sourceBarycentricWeights"][rowIds],
              "sampleWeights": qualified.astype(np.float64),
              "restDistancesCanonical": distances, "normalDots": normalDots}
    outputDir.mkdir(parents=True, exist_ok=True)
    for name in base.manifest["files"]:
        target = outputDir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseAsset/name, target)
    np.savez_compressed(outputDir/"barycentricGeometry.npz", **arrays)
    textureDiagnostics = None
    if correctTextureUvs:
        from faceAppearance.barycentricDeformation import BarycentricDeformer
        from faceAppearance.clientTemplate import readLegacyObj
        from faceAppearance.legacyUvCorrection import compileUvCorrection

        mapped = readLegacyObj(legacyRoot/"template_nicp_aligned_uv_in_flame.obj")
        if not np.array_equal(mapped["faces"], base.arrays["nicpFaces"]):
            raise ValueError("Texture correspondence must preserve NICP face order")
        deformer = BarycentricDeformer(base.arrays["vertices"][base.arrays["nicpToFullIds"]], base.arrays["nicpFaces"],
            arrays["sampleNicpFaceIds"], arrays["sampleNicpWeights"], arrays["sampleWeights"], base.arrays["staticMask"],
            dataWeight=dataWeight, smoothWeight=smoothWeight, restWeight=restWeight)
        corrected, textureDiagnostics = compileUvCorrection(base.arrays, mapped["uvs"], mapped["uvFaces"], arrays, deformer)
        np.savez_compressed(outputDir/"template.npz", **{**base.arrays, **{key: corrected[key] for key in ("atlasSourceUvs", "atlasValidMask")}})
        np.savez_compressed(outputDir/"uvCorrection.npz", **corrected)
    shutil.copyfile(auditDir/"audit.json", outputDir/"legacyCorrespondenceAudit.json")
    manifest = json.loads((baseAsset/"manifest.json").read_text())
    manifest["templateId"] = outputDir.name
    manifest["baseTemplateId"] = base.manifest["templateId"]
    manifest["templateHashes"].update({
        "legacyBarycentricJson": fileHash(legacyRoot/"landmarks/flame_template_refined_aligned_bicentroid.json"),
        "legacyRefinedMesh": fileHash(legacyRoot/"template_flame_refined_mesh.obj"),
        "baseAssetManifest": fileHash(baseAsset/"manifest.json"),
        "barycentricAuditMapping": fileHash(auditDir/"mapping.npz")})
    manifest["geometryTransfer"] = {"method": "legacyBarycentric", "asset": "barycentricGeometry.npz",
        "alignmentMode": "sharedUvVertices", "dataWeight": float(dataWeight),
        "smoothWeight": float(smoothWeight), "restWeight": float(restWeight),
        "qualification": {"maxRestDistanceCanonical": maxRestDistance, "minNormalDot": minNormalDot,
                          "rejectStaticAdjacentFaces": True},
        "totalLegacySamples": len(mapping["sampleIds"]), "validSamples": len(rowIds),
        "qualifiedSamples": int(qualified.sum()), "staticAdjacentSamples": int(static.sum())}
    manifest["mappingStatus"] = "legacyBarycentricUvOrientationAuditedAnatomyUnverified"
    manifest["mappingMethod"] = "Legacy refined raw V-down UV -> UFF surface targets; original NICP face+bary constraints; fixed-topology solve"
    if textureDiagnostics is not None:
        manifest["textureTransfer"] = textureDiagnostics
    manifest["limitations"] = ["UV direction and table structure were audited; local anatomical correspondence is not certified.",
        ("Skin geometry and UV correction use the same legacy constraints, whose anatomical quality remains unverified."
         if correctTextureUvs else "Skin geometry uses legacy barycentric constraints, while texture transfer retains the existing UV correspondence."),
        "Static vertices, original numbering and attachment shapes are preserved; external rig/expression behavior is unverified."]
    manifest["compiler"] = {"path": str(Path(__file__).resolve()), "sha256": fileHash(__file__)}
    manifest["files"] = {str(path.relative_to(outputDir)): fileHash(path) for path in sorted(outputDir.rglob("*"))
                         if path.is_file() and path.name != "manifest.json"}
    writeJson(outputDir/"manifest.json", manifest)
    # Execute the same loader used by the service before declaring compilation
    # successful: validates/factors the solve and verifies every referenced file.
    AssemblyAssets(outputDir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseAsset", type=Path, default=projectRoot/"assets/clientTemplates/legacyFemaleV1")
    parser.add_argument("--auditDir", type=Path, default=projectRoot/"reports/legacyBarycentricAudit")
    parser.add_argument("--outputDir", type=Path, default=projectRoot/"assets/clientTemplates/legacyFemaleBarycentricV1")
    parser.add_argument("--legacyRoot", type=Path, default=Path("/path/to/legacy-template"))
    parser.add_argument("--dataWeight", type=float, default=20.)
    parser.add_argument("--smoothWeight", type=float, default=1.)
    parser.add_argument("--restWeight", type=float, default=.03)
    parser.add_argument("--maxRestDistance", type=float, default=.12)
    parser.add_argument("--minNormalDot", type=float, default=.35)
    parser.add_argument("--correctTextureUvs", action="store_true")
    args = parser.parse_args()
    manifest = compileBarycentricAssembly(**vars(args))
    print(json.dumps({"outputDir": str(args.outputDir), **manifest["geometryTransfer"],
                      "textureTransfer": manifest.get("textureTransfer")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
