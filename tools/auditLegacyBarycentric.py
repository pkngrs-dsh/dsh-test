"""Audit legacy 5346 NICP surface samples against both UFF UV conventions.

Read-only with respect to templates and pipeline assets. Writes only this
audit's report and a solver-independent sample mapping cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceAppearance.clientTemplate import readLegacyObj
from faceAppearance.correspondence import arrayHash
from faceAppearance.legacyBarycentric import compileLegacyUvSamples, readLegacyBarycentric
from faceAppearance.texture import vertexNormals


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {"count": int(values.size), "mean": float(np.mean(values)),
            **dict(zip(("min", "p05", "median", "p95", "max"),
                       map(float, np.quantile(values, [0, .05, .5, .95, 1]))))}


def properSimilarity(source, target, valid):
    sourceMean, targetMean = source[valid].mean(0), target[valid].mean(0)
    a, b = source[valid]-sourceMean, target[valid]-targetMean
    left, singularValues, right = np.linalg.svd(a.T @ b)
    reflectedOptimum = float(np.linalg.det(left @ right))
    correction = np.eye(3)
    correction[-1, -1] = 1 if reflectedOptimum >= 0 else -1
    rotation = left @ correction @ right
    scale = np.sum(singularValues*np.diag(correction))/np.sum(a*a)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid proper-similarity fit")
    translation = targetMean-sourceMean @ rotation*scale
    matrixRow = np.eye(4)
    matrixRow[:3, :3] = scale*rotation
    matrixRow[3, :3] = translation
    points = source @ rotation*scale+translation
    return {"rotation": rotation, "scale": scale, "translation": translation, "matrixRow": matrixRow,
            "points": points, "residuals": np.linalg.norm(points-target, axis=1),
            "unconstrainedRotationDeterminant": reflectedOptimum,
            "axisCorrelations": np.corrcoef(source[valid].T, target[valid].T)[:3, 3:]}


def sampleNormals(mesh, faceIds, weights):
    values = (vertexNormals(mesh["vertices"], mesh["faces"])[mesh["faces"][faceIds]]*weights[..., None]).sum(1)
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def nondegenerateFaces(mesh):
    t = mesh["vertices"][mesh["faces"]]
    return np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) > 1e-14


def audit(legacyRoot, projectRoot, outputDir):
    legacyRoot, projectRoot, outputDir = Path(legacyRoot), Path(projectRoot), Path(outputDir)
    if outputDir.exists() and any(outputDir.iterdir()):
        raise FileExistsError("Use a new audit directory rather than overwriting prior evidence")
    paths = {"refined": legacyRoot / "template_flame_refined_mesh.obj",
             "alignedRefined": legacyRoot / "landmarks/flame_template_refined_aligned.obj",
             "alignedRefinedJson": legacyRoot / "landmarks/flame_template_refined_aligned.json",
             "barycentric": legacyRoot / "landmarks/flame_template_refined_aligned_bicentroid.json",
             "femaleNicp": legacyRoot / "female/260429/template_female_tri_nicp.obj",
             "maleNicp": legacyRoot / "male/template_male_tri_nicp.obj",
             "sourceMesh": projectRoot / "assets/wrapPreparation/uvFaceFusionFixed.obj",
             "sourceUvs": projectRoot / "assets/wrapPreparation/originals/canonical_mesh_flame_uvs.npy",
             "boundaries": projectRoot / "assets/anatomical/uff6986/boundaries.json"}
    meshes = {name: readLegacyObj(paths[name]) for name in ("refined", "alignedRefined", "femaleNicp", "maleNicp", "sourceMesh")}
    source = meshes["sourceMesh"]
    sourceUvs = np.asarray(np.load(paths["sourceUvs"], allow_pickle=False), np.float32)
    table = readLegacyBarycentric(paths["barycentric"], meshes["femaleNicp"]["faces"], boundaryTolerance=0.)
    count = len(table["sampleIds"])
    refined, alignedRefined = meshes["refined"], meshes["alignedRefined"]
    if len(refined["vertices"]) != count or len(refined["uvs"]) != count or not np.array_equal(refined["faces"], refined["uvFaces"]):
        raise ValueError("The claimed refined vertex/UV sample-order identity does not hold")
    if not np.array_equal(meshes["femaleNicp"]["faces"], meshes["maleNicp"]["faces"]):
        raise ValueError("Female and male NICP triangle order differs")
    rawUvs = refined["uvs"].copy()
    rawUvs[:, 1] = 1-rawUvs[:, 1]
    mappings = {"rawNativeVDown": compileLegacyUvSamples(rawUvs, source["vertices"], source["faces"], sourceUvs),
                "standardObjFlipV": compileLegacyUvSamples(refined["uvs"], source["vertices"], source["faces"], sourceUvs)}
    fits, reportCases, extraArrays = {}, {}, {}
    pointsByGender = {}
    faceIds, weights = table["sampleNicpFaceIds"], table["sampleNicpWeights"]
    for gender in ("female", "male"):
        mesh = meshes[f"{gender}Nicp"]
        pointsByGender[gender] = (mesh["vertices"][mesh["faces"][faceIds]]*weights[..., None]).sum(1)
    for mode, mapping in mappings.items():
        reportCases[mode] = {"uvValidCount": int(mapping["sampleUvValidMask"].sum()),
                             "uvDistances": stats(mapping["sourceUvDistances"]), "genders": {}}
        sourceNormals = sampleNormals(source, mapping["sourceTriangleIds"], mapping["sourceBarycentricWeights"])
        for gender in ("female", "male"):
            mesh = meshes[f"{gender}Nicp"]
            valid = mapping["sampleUvValidMask"] & table["sampleBaryValidMask"] & nondegenerateFaces(mesh)[faceIds]
            fit = properSimilarity(pointsByGender[gender], mapping["sampleSourcePoints"], valid)
            normals = sampleNormals(mesh, faceIds, weights) @ fit["rotation"]
            dots = np.sum(normals*sourceNormals, axis=1)
            fits[(mode, gender)] = {**fit, "normalDots": dots, "valid": valid}
            reportCases[mode]["genders"][gender] = {
                "validSampleCount": int(valid.sum()), "restDistances": stats(fit["residuals"][valid]),
                "normalDots": stats(dots[valid]), "negativeNormalCount": int(((dots < 0) & valid).sum()),
                "rotationDeterminant": float(np.linalg.det(fit["rotation"])),
                "unconstrainedRotationDeterminant": fit["unconstrainedRotationDeterminant"],
                "scale": float(fit["scale"]), "matrixRow": fit["matrixRow"].tolist(),
                "axisCorrelationsBeforeAlignment": fit["axisCorrelations"].tolist()}
            extraArrays[f"{mode}_{gender}RestDistances"] = fit["residuals"].astype(np.float32)
            extraArrays[f"{mode}_{gender}NormalDots"] = dots.astype(np.float32)
    # This report keeps both hypotheses. The cache exports RAW only after it
    # wins independently for both rest shapes, with positive orientation.
    if any(reportCases["rawNativeVDown"]["genders"][gender]["restDistances"]["median"] >=
           reportCases["standardObjFlipV"]["genders"][gender]["restDistances"]["median"]*.5
           for gender in ("female", "male")):
        raise ValueError("UV direction is not decisive enough to publish a preferred sample cache")
    selected = mappings["rawNativeVDown"]
    selectedValid = selected["sampleUvValidMask"] & table["sampleBaryValidMask"]
    femaleFit = fits[("rawNativeVDown", "female")]
    regions = {}
    boundaries = json.loads(paths["boundaries"].read_text())
    if boundaries["vertexCount"] != len(source["vertices"]):
        raise ValueError("Boundary annotation vertex count does not match source topology")
    for boundary in boundaries["boundaries"]:
        if boundary["name"] == "outerFace":
            continue
        distances, _ = cKDTree(source["vertices"][boundary["vertexIndices"]]).query(selected["sampleSourcePoints"])
        mask = (distances <= .035) & selectedValid
        sampleIds = np.flatnonzero(mask)
        perGender = {}
        for gender in ("female", "male"):
            fit = fits[("rawNativeVDown", gender)]
            perGender[gender] = {"restDistances": stats(fit["residuals"][mask]),
                                "normalDots": stats(fit["normalDots"][mask]),
                                "normalBelow0_35Count": int((fit["normalDots"][mask] < .35).sum()),
                                "restAbove0_12Count": int((fit["residuals"][mask] > .12).sum())}
        worst = sampleIds[np.argsort(femaleFit["residuals"][mask])[-10:][::-1]]
        regions[boundary["name"]] = {"selection": "Distance <=0.035 native UFF units from reviewed template hole-rim vertices",
                                     "sampleCount": len(sampleIds), "sampleIds": sampleIds.tolist(),
                                     "genders": perGender, "worstFemaleSampleIds": worst.tolist()}
    alignedPositions = np.array([[point[axis] for axis in ("x", "y", "z")]
                                 for point in json.loads(paths["alignedRefinedJson"].read_text())])
    alignedComparison = {}
    for gender in ("female", "male"):
        valid = table["sampleBaryValidMask"]
        fit = properSimilarity(pointsByGender[gender], alignedRefined["vertices"], valid)
        alignedComparison[gender] = {"directResidualCentimeters": stats(np.linalg.norm(pointsByGender[gender]-alignedRefined["vertices"], axis=1)[valid]),
                                     "properSimilarityResidualInAlignedUnits": stats(fit["residuals"][valid]),
                                     "scale": float(fit["scale"]), "rotationDeterminant": float(np.linalg.det(fit["rotation"]))}
    outputDir.mkdir(parents=True)
    outputArrays = {**table, **selected, "sampleValidMask": selectedValid,
                    "femaleNormalDots": femaleFit["normalDots"].astype(np.float32),
                    "femaleRestDistances": femaleFit["residuals"].astype(np.float32),
                    "femaleProperSimilarityMatrixRow": femaleFit["matrixRow"],
                    "sampleFemaleNicpPoints": pointsByGender["female"],
                    "sampleMaleNicpPoints": pointsByGender["male"], **extraArrays}
    np.savez_compressed(outputDir / "mapping.npz", **outputArrays)
    auditData = {"schemaVersion": 1, "status": "auditedLegacySamplingCandidate", "sampleCount": count,
                 "chosenUvConvention": "rawNativeVDown", "units": "UFF canonical numerical units are uncalibrated",
                 "mappingDirection": "Each old refined sample index -> NICP face+bary and independently -> UFF UV-surface face+bary",
                 "barycentricRule": "Numeric sample keys 0..5345; original weights [u,v,1-u-v]; finite data and face IDs required; valid weights >=0 and <=1, sum constructed as 1; no clipping or renormalization",
                 "sampleValidRule": "sampleUvValidMask (distance<=1e-5) AND sampleBaryValidMask; no rest/normal hard gate in this adapter",
                 "validSamples": int(selectedValid.sum()), "invalidBarySampleIds": np.flatnonzero(~table["sampleBaryValidMask"]).tolist(),
                 "invalidUvSampleIds": np.flatnonzero(~selected["sampleUvValidMask"]).tolist(),
                 "weightExtrema": [float(weights.min()), float(weights.max())],
                 "sampleWeightsSumErrorMax": float(np.abs(weights.sum(1)-1).max()),
                 "topology": {"refinedVertices": len(refined["vertices"]), "refinedFaces": len(refined["faces"]),
                              "refinedUvs": len(refined["uvs"]), "refinedFacesEqualUvFaces": np.array_equal(refined["faces"], refined["uvFaces"]),
                              "refinedAndAlignedFaceOrderEqual": np.array_equal(refined["faces"], alignedRefined["faces"]),
                              "refinedAndAlignedUvFacesEqual": np.array_equal(refined["uvFaces"], alignedRefined["uvFaces"]),
                              "refinedAndAlignedUvValuesMaxDifference": float(np.max(np.abs(refined["uvs"]-alignedRefined["uvs"]))),
                              "femaleMaleNicpFaceOrderEqual": np.array_equal(meshes["femaleNicp"]["faces"], meshes["maleNicp"]["faces"]),
                              "femaleMaleNicpUvFacesEqual": np.array_equal(meshes["femaleNicp"]["uvFaces"], meshes["maleNicp"]["uvFaces"]),
                              "femaleMaleNicpUvValuesEqual": np.array_equal(meshes["femaleNicp"]["uvs"], meshes["maleNicp"]["uvs"]),
                              "degenerateGeometryFaces": {key: int((~nondegenerateFaces(mesh)).sum()) for key, mesh in meshes.items()},
                              "sourceTopologyHash": arrayHash(source["faces"].astype(np.int64), sourceUvs.astype(np.float32))},
                 "uvHypotheses": reportCases, "regions": regions, "alignedRefinedComparison": alignedComparison,
                 "alignedRefinedJsonVsObjMaxDifference": float(np.max(np.abs(alignedPositions-alignedRefined["vertices"]))),
                 "interpretation": ["RAW refined OBJ vt is numerically native V-down for this file; do not apply conventional OBJ 1-v here.",
                                    "The separate template_nicp_aligned_uv_in_flame.obj is a standard OBJ chart and requires 1-v; conventions are file-specific.",
                                    "5346 samples need not equal UFF's 6986 vertices: both surfaces are evaluated through UV and barycentric interpolation.",
                                    "Male and female share topology but have different rest shapes and absolute positions; compare after a proper similarity, not direct coordinate displacement.",
                                    "The oralAperture region is an internal mesh rim, not a validated visible lip-contact line.",
                                    "Low rest error is supporting evidence, not anatomical certification or proof of injectivity. These results do not require redoing Wrap before testing reuse."],
                 "sources": {name: {"path": str(path.resolve()), "sha256": fileHash(path)} for name, path in paths.items()},
                 "implementationHashes": {"auditTool": fileHash(__file__), "adapter": fileHash(projectRoot / "src/faceAppearance/legacyBarycentric.py")},
                 "mappingFile": "mapping.npz", "mappingSha256": fileHash(outputDir / "mapping.npz")}
    (outputDir / "audit.json").write_text(json.dumps(auditData, ensure_ascii=False, indent=2)+"\n")
    return auditData


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacyRoot", type=Path, default=Path("/path/to/legacy-template"))
    parser.add_argument("--outputDir", type=Path, default=root / "reports/legacyBarycentricAudit")
    args = parser.parse_args()
    result = audit(args.legacyRoot, root, args.outputDir)
    print(json.dumps({"outputDir": str(args.outputDir), "chosenUvConvention": result["chosenUvConvention"],
                      "validSamples": result["validSamples"], "invalidBarySampleIds": result["invalidBarySampleIds"],
                      "femaleRaw": result["uvHypotheses"]["rawNativeVDown"]["genders"]["female"],
                      "regionCounts": {name: region["sampleCount"] for name, region in result["regions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
