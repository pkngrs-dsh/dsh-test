"""Compile the notebook male template as an independent portable asset bundle.

This is an offline adapter for the existing UV chart, not an anatomical
registration certificate. Unqualified skin and all accessories keep template
fallback data. No learned models, GPU, or old barycentric JSON are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceAppearance.clientTemplate import connectedMaterialComponents, readLegacyObj
from faceAppearance.correspondence import SurfaceIndex, arrayHash, sampleSurface
from faceAppearance.texture import buildUvLookup, vertexNormals


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def similarity(source, target, valid):
    """Proper similarity with deterministic 80% residual trimming for outliers."""
    selected = np.asarray(valid, bool).copy()
    for _ in range(8):
        if selected.sum() < 12:
            raise ValueError("Too few eligible points for template-to-UFF similarity")
        sourceMean, targetMean = source[selected].mean(0), target[selected].mean(0)
        centeredSource, centeredTarget = source[selected]-sourceMean, target[selected]-targetMean
        left, _, right = np.linalg.svd(centeredSource.T @ centeredTarget)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        scale = np.sum((centeredSource @ rotation)*centeredTarget)/np.sum(centeredSource**2)
        if not scale > 0:
            raise ValueError("Similarity fit did not produce a positive scale")
        translation = targetMean - sourceMean @ rotation * scale
        aligned = source @ rotation * scale + translation
        residual = np.linalg.norm(aligned-target, axis=1)
        selected = valid & (residual <= np.quantile(residual[valid], .8))
    return aligned, rotation, scale, translation, residual


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(zip(("min", "median", "p95", "max"), map(float, np.quantile(values, [0, .5, .95, 1]))))


def compileLegacyMaleTemplate(legacyRoot, outputDir, sourceObj, sourceUvPath, *, resolution=1024):
    legacyRoot, outputDir = Path(legacyRoot), Path(outputDir)
    if (outputDir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing compiled bundle: {outputDir}")
    sourcePaths = {
        "fullMesh": legacyRoot / "male/template_male_tri_full.obj",
        "nicpMesh": legacyRoot / "male/template_male_tri_nicp.obj",
        "nicpUvMapping": legacyRoot / "template_nicp_aligned_uv_in_flame.obj",
        "staticFaceMask": legacyRoot / "landmarks/nicp_none_facial_region_mask_with_ear.json",
        "leftEyeFaces": legacyRoot / "landmarks/left_eye_ball_json.json",
        "rightEyeFaces": legacyRoot / "landmarks/right_eye_ball_json.json",
        "sourceMesh": Path(sourceObj), "sourceUvs": Path(sourceUvPath),
        "headTexture": legacyRoot / "male/Tex_male00_head_C.png",
        "oralTexture": legacyRoot / "male/Tex_male00_body_C.png",
        "eyeTexture": legacyRoot / "male/Tex_Default_Eyeball_C2.png",
        "browTexture": legacyRoot / "male/Eyebrows_Default_Nan.png",
    }
    full, nicp, mapped, source = [readLegacyObj(sourcePaths[key]) for key in
                                 ("fullMesh", "nicpMesh", "nicpUvMapping", "sourceMesh")]
    sourceVertices = source["vertices"]
    sourceFaces = source["faces"].astype(np.int64)
    sourceUvs = np.asarray(np.load(sourcePaths["sourceUvs"], allow_pickle=False), np.float32)
    if sourceUvs.shape != (len(sourceVertices), 2):
        raise ValueError("UFF native UVs must be one per source vertex")
    # The Wrap preparation OBJ deliberately stored UFF native V-down as vt.
    # It is NOT a conventional OBJ texture chart: undo the parser conversion
    # only for this explicit consistency check; all sampling uses the NPY.
    restoredObjUvs = source["uvs"].copy()
    restoredObjUvs[:, 1] = 1-restoredObjUvs[:, 1]
    if not np.array_equal(source["uvFaces"], sourceFaces) or not np.allclose(restoredObjUvs, sourceUvs, atol=1e-7, rtol=0):
        raise ValueError("UFF source OBJ and original native UV array disagree")
    if not np.array_equal(mapped["faces"], nicp["faces"]):
        raise ValueError("Legacy UV mapping changed NICP triangle indices or order")
    if full["materialNames"] != ["male00SG", "male00SG1", "male00SG2", "male00SG3"]:
        raise ValueError("Male template material identity differs from the notebook asset")
    if not np.array_equal(mapped["vertices"], nicp["vertices"]):
        raise ValueError("Legacy UV mapping must use the actual male NICP rest vertices")
    if len(nicp["vertices"]) != 2795 or len(full["vertices"]) != 3544:
        raise ValueError("Unexpected legacy template identity")
    distances, nicpToFullIds = cKDTree(full["vertices"]).query(nicp["vertices"])
    if distances.max() != 0 or len(np.unique(nicpToFullIds)) != len(nicpToFullIds):
        raise ValueError("NICP vertices are not a unique exact subset of the full template")
    nicpToFullIds = nicpToFullIds.astype(np.int64)
    fullFaceLookup = {tuple(face): index for index, face in enumerate(full["faces"])}
    nicpToFullFaceIds = np.array([fullFaceLookup.get(tuple(face), -1)
                                for face in nicpToFullIds[nicp["faces"]]], np.int64)
    if (nicpToFullFaceIds < 0).any() or len(np.unique(nicpToFullFaceIds)) != len(nicpToFullFaceIds):
        raise ValueError("NICP triangles do not have a unique identical full-template counterpart")
    if not (full["materialIds"][nicpToFullFaceIds] == 0).all():
        raise ValueError("NICP triangles must all belong to the head material")

    planarUvs = np.c_[sourceUvs, np.zeros(len(sourceUvs))]
    triangles = planarUvs[sourceFaces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    nondegenerateIds = np.flatnonzero(areas > 1e-14)
    uvIndex = SurfaceIndex(planarUvs, sourceFaces[nondegenerateIds])
    uvMapping = uvIndex.query(np.c_[mapped["uvs"], np.zeros(len(mapped["uvs"]))])
    uvMapping["triangleIds"] = nondegenerateIds[uvMapping["triangleIds"]]
    uvPoints = sampleSurface(sourceVertices, sourceFaces, uvMapping)
    selectedUvIds = np.empty(len(nicp["vertices"]), np.int64)
    seamSpread = np.zeros(len(selectedUvIds), np.float64)
    vertexUvIds = [[] for _ in selectedUvIds]
    for vertexId, uvId in zip(mapped["faces"].ravel(), mapped["uvFaces"].ravel()):
        vertexUvIds[vertexId].append(int(uvId))
    for vertexId, candidates in enumerate(vertexUvIds):
        candidates = np.unique(candidates)
        best = np.lexsort((candidates, uvMapping["distances"][candidates]))[0]
        selectedUvIds[vertexId] = candidates[best]
        validCandidates = candidates[uvMapping["distances"][candidates] <= 1e-5]
        if len(validCandidates) > 1:
            seamSpread[vertexId] = np.max(np.linalg.norm(uvPoints[validCandidates]-uvPoints[candidates[best]], axis=1))
    triangleIds = uvMapping["triangleIds"][selectedUvIds].astype(np.int64)
    barycentricWeights = uvMapping["barycentricWeights"][selectedUvIds].astype(np.float32)
    targetPoints = uvPoints[selectedUvIds]
    uvDistances = uvMapping["distances"][selectedUvIds]
    uvQualified = uvDistances <= 1e-5
    aligned, rotation, scale, translation, restDistances = similarity(nicp["vertices"], targetPoints, uvQualified)
    sourceNormals = vertexNormals(sourceVertices, sourceFaces)
    mappedNormals = (sourceNormals[sourceFaces[triangleIds]]*barycentricWeights[..., None]).sum(1)
    mappedNormals /= np.maximum(np.linalg.norm(mappedNormals, axis=1, keepdims=True), 1e-12)
    templateNormals = vertexNormals(nicp["vertices"], nicp["faces"]) @ rotation
    normalDots = np.sum(mappedNormals*templateNormals, axis=1)
    staticFaceIds = np.asarray(json.loads(sourcePaths["staticFaceMask"].read_text()), np.int64)
    if (staticFaceIds < 0).any() or (staticFaceIds >= len(nicp["faces"])).any():
        raise ValueError("Legacy static face indices do not match the NICP mesh")
    staticMask = np.zeros(len(nicp["vertices"]), bool)
    staticMask[np.unique(nicp["faces"][staticFaceIds])] = True
    qualified = uvQualified & (restDistances < .1) & (normalDots > .35) & ~staticMask
    # Chart seam conflicts decrease geometric influence; severe conflicts are
    # rejected. No correspondence is invented across disjoint UV islands.
    seamWeight = np.clip((.05-seamSpread)/.04, 0, 1)
    geometryWeights = (qualified*seamWeight).astype(np.float32)
    if np.count_nonzero(geometryWeights) < 100:
        raise ValueError("Fewer than 100 skin vertices qualified; manual registration is required")

    # Raster in the full template's head chart, then follow the exact NICP face
    # identity and its existing UV-corner correspondence into UFF's chart.
    headFaceIds = np.flatnonzero(full["materialIds"] == 0)
    lookup = buildUvLookup(full["uvs"], full["uvFaces"][headFaceIds], resolution)
    atlasHeadMask = lookup["validMask"]
    fullToNicpFace = np.full(len(full["faces"]), -1, np.int64)
    fullToNicpFace[nicpToFullFaceIds] = np.arange(len(nicpToFullFaceIds))
    targetNicpFace = np.full((resolution, resolution), -1, np.int64)
    targetNicpFace[atlasHeadMask] = fullToNicpFace[headFaceIds[lookup["faceIds"][atlasHeadMask]]]
    eligibleFaces = ((geometryWeights[nicp["faces"]] > 0).all(1)
                     & ~staticMask[nicp["faces"]].any(1)
                     & (uvMapping["distances"][mapped["uvFaces"]] <= 1e-5).all(1))
    atlasValidMask = (targetNicpFace >= 0)
    atlasValidMask[atlasValidMask] &= eligibleFaces[targetNicpFace[atlasValidMask]]
    atlasSourceUvs = np.zeros((resolution, resolution, 2), np.float32)
    ids = targetNicpFace[atlasValidMask]
    atlasSourceUvs[atlasValidMask] = (mapped["uvs"][mapped["uvFaces"][ids]]*
                                     lookup["barycentrics"][atlasValidMask, :, None]).sum(1)
    # Even when all target corners are supported, a large chart triangle could
    # bridge a source hole. Reject those pixels using the source UV coverage.
    sourceLookup = buildUvLookup(sourceUvs, sourceFaces, resolution)
    sampleUvs = atlasSourceUvs[atlasValidMask]
    pixels = np.rint(sampleUvs*(resolution-1)).astype(np.int64)
    inside = ((pixels >= 0) & (pixels < resolution)).all(1)
    sourceSupport = np.zeros(len(pixels), bool)
    sourceSupport[inside] = sourceLookup["validMask"][pixels[inside, 1], pixels[inside, 0]]
    atlasValidMask[atlasValidMask] = sourceSupport
    atlasSourceUvs[~atlasValidMask] = 0

    components = connectedMaterialComponents(full)
    eyeFaces = {side: set(json.loads(sourcePaths[f"{side}EyeFaces"].read_text())) for side in ("left", "right")}
    for component in components:
        vertexIds, faceIds = component["vertexIndices"], component["faceIndices"]
        materialId = component["materialId"]
        if materialId == 0:
            label = "head" if len(vertexIds) == 2795 else "headAccessory"
        elif materialId == 1:
            knownOral = {(2801, 2915): "upperTeeth", (2916, 3032): "lowerTeeth", (3033, 3073): "tongue"}
            label = knownOral.get((int(vertexIds.min()), int(vertexIds.max())), "oralAccessory")
        elif materialId == 2:
            matches = [side for side, expected in eyeFaces.items() if set(faceIds) == expected]
            if len(matches) != 1:
                raise ValueError("Eye component does not exactly match its legacy face annotation")
            label = f"{matches[0]}Eye"
        else:
            # Roles audited against Y position and the original RGBA atlas.
            # Left/right counterparts share UV coordinates; their identities
            # remain distinct even though the texture region is shared.
            hairRoles = {(3182, 3531): "lowerEyelash", (3278, 3543): "lowerEyelash",
                         (3204, 3507): "upperEyelash", (3316, 3519): "upperEyelash",
                         (3374, 3434): "eyebrow", (3435, 3495): "eyebrow"}
            label = hairRoles.get((int(vertexIds.min()), int(vertexIds.max())), "browOrEyelash")
        component["label"] = label
        for key, value in list(component.items()):
            if isinstance(value, np.ndarray):
                component[key] = value.tolist()
    materials = []
    for name, sourceKey, legacyName, roughness, alphaMode in [
            ("head", "headTexture", "head_color.png", .6, "OPAQUE"),
            ("oral", "oralTexture", "body_color.png", .45, "OPAQUE"),
            ("eyes", "eyeTexture", "eye_color.png", .2, "OPAQUE"),
            ("browLashes", "browTexture", "brow_color.png", .7, "MASK")]:
        materials.append({"name": name, "texture": f"textures/{legacyName}", "roughness": roughness,
                          "metallic": 0, "alphaMode": alphaMode, "legacyTextureName": legacyName,
                          "sourceKey": sourceKey})
        if alphaMode == "MASK":
            materials[-1]["alphaCutoff"] = .5
        if name == "browLashes":
            materials[-1]["hairTintSrgb"] = [.12, .085, .06]
    outputDir.mkdir(parents=True, exist_ok=True)
    (outputDir / "textures").mkdir(exist_ok=True)
    for material in materials:
        shutil.copyfile(sourcePaths[material.pop("sourceKey")], outputDir / material["texture"])
    np.savez_compressed(outputDir / "template.npz", vertices=full["vertices"], faces=full["faces"],
                        uvs=full["uvs"], uvFaces=full["uvFaces"], materialIds=full["materialIds"],
                        nicpToFullIds=nicpToFullIds, nicpFaces=nicp["faces"], sourceVertices=sourceVertices,
                        sourceFaces=sourceFaces, sourceUvs=sourceUvs, triangleIds=triangleIds,
                        barycentricWeights=barycentricWeights, geometryWeights=geometryWeights,
                        staticMask=staticMask, atlasSourceUvs=atlasSourceUvs, atlasValidMask=atlasValidMask,
                        atlasHeadMask=atlasHeadMask)
    (outputDir / "components.json").write_text(json.dumps(components, indent=2)+"\n")
    diagnostics = {"fullVertices": len(full["vertices"]), "fullTriangles": len(full["faces"]),
                   "nicpVertices": len(nicp["vertices"]), "sourceVertices": len(sourceVertices),
                   "uvQualifiedVertices": int(uvQualified.sum()), "staticVertices": int(staticMask.sum()),
                   "distanceQualifiedVertices": int((uvQualified & (restDistances < .1)).sum()),
                   "normalQualifiedVertices": int((uvQualified & (normalDots > .35)).sum()),
                   "geometryQualifiedVertices": int((geometryWeights > 0).sum()),
                   "seamConflictsOver0_01CanonicalUnits": int((seamSpread > .01).sum()),
                   "seamRejectedVertices": int((qualified & (seamWeight == 0)).sum()),
                   "eligibleNicpFaces": int(eligibleFaces.sum()), "atlasHeadTexels": int(atlasHeadMask.sum()),
                   "atlasMappedTexels": int(atlasValidMask.sum()),
                   "sourceUvDegenerateTriangles": int(len(sourceFaces)-len(nondegenerateIds)),
                   "qualifiedRestDistances": distribution(restDistances[geometryWeights > 0]),
                   "qualifiedNormalDots": distribution(normalDots[geometryWeights > 0]),
                   "similarityScale": float(scale), "similarityRotationRowVector": rotation.tolist(),
                   "similarityTranslation": translation.tolist(), "componentCount": len(components)}
    manifest = {"schemaVersion": 1, "templateId": "legacyMaleV1", "gender": "male",
                "sourceTopologyHash": arrayHash(sourceFaces.astype(np.int64), sourceUvs.astype(np.float32)),
                "templateHashes": {key: fileHash(path) for key, path in sourcePaths.items()},
                "sources": {key: str(path.resolve()) for key, path in sourcePaths.items()},
                "materialNames": full["materialNames"], "materials": materials,
                "appearanceOptions": {"seamFeatherPixels": 24, "hideObservedEyebrows": True,
                                      "eyebrowCoverageThreshold": .7, "hidePresetEyelashes": True},
                "templateUnits": "centimeters", "outputUnitScale": .01, "outputUnits": "meters",
                "arrayUvOrigin": "vDown", "sourceGeometryFrame": "uvFaceFusionCanonical",
                "sourceMetricUnits": "uncalibrated", "mappingStatus": "candidateLegacyUvUnverifiedAnatomy",
                "mappingMethod": "Existing standard OBJ UV chart, converted 1-v, queried against original UFF native V-down NPY",
                "limits": {"maxUvDistance": 1e-5, "maxRestDistance": .1, "minNormalDot": .35,
                           "seamFullWeightDistance": .01, "seamZeroWeightDistance": .05,
                           "staticMask": "All vertices of legacy nonfacial-with-ear face mask",
                           "graphBoundaryTaper": False, "atlasResolution": resolution},
                "limitations": ["Candidate mapping has not been anatomically certified in Wrap.",
                                "Legacy barycentric registration JSON is not used.",
                                "Unqualified skin, neck, ears, and accessories retain template fallback.",
                                "Same topology and order preserve external binding identity; this bundle contains no skeleton or skin weights.",
                                "Existing eye components are open meshes, not complete spherical eyeballs.",
                                "Roughness values are preset material defaults, not inferred from photographs."],
                "diagnostics": diagnostics,
                "compiler": {"path": str(Path(__file__).resolve()), "sha256": fileHash(__file__)}}
    # The manifest cannot include its own hash recursively; every other asset
    # is covered. The runtime validates this closed list before loading.
    manifest["legacyNotebookAsset"] = {"mesh": "Template/male/template_male_tri_full.obj",
                                       "nicp": "Template/male/template_male_tri_nicp.obj",
                                       "coordinateSource": "male rest mesh, not a female substitute",
                                       "normalMapDefault": "neutral tangent-space placeholder at legacy export"}
    manifest["files"] = {str(path.relative_to(outputDir)): fileHash(path) for path in sorted(outputDir.rglob("*"))
                         if path.is_file() and path.name != "manifest.json"}
    (outputDir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacyRoot", type=Path, default=Path("/path/to/legacy-template"))
    parser.add_argument("--outputDir", type=Path, default=root / "assets/clientTemplates/legacyMaleV1")
    parser.add_argument("--sourceObj", type=Path, default=root / "assets/wrapPreparation/uvFaceFusionFixed.obj")
    parser.add_argument("--sourceUvs", type=Path, default=root / "assets/wrapPreparation/originals/canonical_mesh_flame_uvs.npy")
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()
    manifest = compileLegacyMaleTemplate(args.legacyRoot, args.outputDir, args.sourceObj, args.sourceUvs, resolution=args.resolution)
    print(json.dumps({"outputDir": str(args.outputDir), **manifest["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
