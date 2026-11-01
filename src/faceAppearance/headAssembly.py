"""Retarget a reconstructed face into a fixed, binding-compatible client mesh.

Asset preparation is offline. Runtime jobs use frozen topology/correspondence,
screened displacement smoothing, rigid accessory placement and linear-light
atlas blending. No topology operation or rig inference is performed here.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from .correspondence import arrayHash, sampleSurface
from .texture import buildUvLookup, samplePixels


def decodeColor(colors: np.ndarray, encoding: str) -> np.ndarray:
    """Decode RGB; alpha, when present, is always linear coverage."""
    values = np.asarray(colors, dtype=np.float64)
    if (values.ndim != 3 or values.shape[-1] not in (3, 4)
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError("Color must be finite RGB/RGBA in [0,1]")
    result = values.copy()
    if encoding == "srgb":
        result[..., :3] = np.where(values[..., :3] <= .04045, values[..., :3]/12.92,
                                   ((values[..., :3]+.055)/1.055)**2.4)
    elif encoding == "gamma2.2":
        result[..., :3] **= 2.2
    elif encoding != "linear":
        raise ValueError("Color encoding must be linear, srgb or gamma2.2")
    return result.astype(np.float32)


def fitSimilarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict:
    """Robust proper similarity: target = source @ rotation * scale + translation."""
    source, target, weights = map(lambda item: np.asarray(item, np.float64), (source, target, weights))
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape
            or weights.shape != (len(source),) or not np.isfinite(source).all()
            or not np.isfinite(target).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.count_nonzero(weights > 0) < 3):
        raise ValueError("Similarity requires at least three finite weighted correspondences")
    robustWeights = weights.copy()
    for _ in range(5):
        normalized = robustWeights/robustWeights.sum()
        sourceMean, targetMean = normalized @ source, normalized @ target
        x, y = source-sourceMean, target-targetMean
        covariance = x.T @ (normalized[:, None]*y)
        u, singularValues, vt = np.linalg.svd(covariance)
        if singularValues[1] < 1e-12:
            raise ValueError("Similarity correspondences are collinear or collapsed")
        correction = np.ones(3)
        correction[-1] = np.linalg.det(u @ vt)
        rotation = (u*correction) @ vt
        variance = np.sum(normalized[:, None]*x*x)
        scale = float(singularValues @ correction / variance)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Similarity has an invalid scale")
        translation = targetMean-sourceMean @ rotation*scale
        residual = np.linalg.norm(source @ rotation*scale+translation-target, axis=1)
        cutoff = max(float(np.median(residual[weights > 0]))*2.5, 1e-7)
        robustWeights = weights*np.minimum(1., cutoff/np.maximum(residual, 1e-12))
    return {"scale": scale, "rotation": rotation, "translation": translation}


def _meshLaplacian(vertexCount: int, faces: np.ndarray) -> sparse.csr_matrix:
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                               faces[:, [2, 0]]]), axis=1), axis=0)
    rows, cols = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(vertexCount, vertexCount))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags((degree > 0).astype(np.float64))-sparse.diags(1/np.maximum(degree, 1)) @ adjacency


def _triangleQuality(before: np.ndarray, after: np.ndarray, faces: np.ndarray) -> dict:
    first, second = before[faces], after[faces]
    beforeCross = np.cross(first[:, 1]-first[:, 0], first[:, 2]-first[:, 0])
    afterCross = np.cross(second[:, 1]-second[:, 0], second[:, 2]-second[:, 0])
    beforeArea, afterArea = np.linalg.norm(beforeCross, axis=1), np.linalg.norm(afterCross, axis=1)
    valid = beforeArea > 1e-10
    dots = np.sum(beforeCross[valid]*afterCross[valid], axis=1)/np.maximum(beforeArea[valid]*afterArea[valid], 1e-20)
    return {"minNormalDot": float(dots.min()),
            "minAreaRatio": float(np.min(afterArea[valid]/beforeArea[valid])),
            "maxAreaRatio": float(np.max(afterArea[valid]/beforeArea[valid])),
            "invertedFaceCount": int(np.count_nonzero(dots <= 0))}


def guardDisplacement(rest: np.ndarray, displacement: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Backtrack only unsafe triangle neighborhoods, retaining other face detail.

    This checks local orientation/area, not nonlocal self-intersections. A
    deficient correspondence must remain visible in the returned retention map.
    """
    triangles = rest[faces]
    beforeCross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    beforeArea = np.linalg.norm(beforeCross, axis=1)
    valid = beforeArea > 1e-10
    retention = np.minimum(1., 3./np.maximum(np.linalg.norm(displacement, axis=1), 1e-12))
    for _ in range(32):
        deformed = rest+displacement*retention[:, None]
        triangles = deformed[faces]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        area = np.linalg.norm(cross, axis=1)
        dots = np.sum(cross*beforeCross, axis=1)/np.maximum(area*beforeArea, 1e-20)
        ratio = area/np.maximum(beforeArea, 1e-20)
        bad = valid & ((dots <= .1) | (ratio < .08) | (ratio > 12))
        if not bad.any():
            return deformed, retention, _triangleQuality(rest, deformed, faces)
        core = np.zeros(len(rest), bool)
        core[np.unique(faces[bad])] = True
        halo = np.unique(faces[core[faces].any(axis=1)])
        retention[halo] *= .75
        retention[core] *= 2/3
    raise ValueError("Client deformation failed topology safety checks; correspondence requires review")


class AssemblyAssets:
    """Preload and verify one explicitly versioned client template on CPU.

    Coordinates in this module are the original template centimeters. Export
    converts to meters once. The legacy paths use hard static constraints;
    direct registration explicitly selects soft or hard boundary constraints.
    Accessories retain their internal shape. The qualified direct male eye
    adapter permits bounded uniform shrinking while preserving iris direction.
    """

    def __init__(self, assetRoot: str | Path):
        self.assetRoot = Path(assetRoot).resolve()
        manifestBytes = (self.assetRoot/"manifest.json").read_bytes()
        self.manifest = json.loads(manifestBytes)
        self.manifestSha256 = hashlib.sha256(manifestBytes).hexdigest()
        if self.manifest.get("schemaVersion") != 1:
            raise ValueError("Unsupported client template schema")
        files = self.manifest.get("files", {})
        if not {"template.npz", "components.json"}.issubset(files):
            raise ValueError("Template manifest must hash geometry and component metadata")
        for name, expectedHash in files.items():
            path = (self.assetRoot/name).resolve()
            if not path.is_relative_to(self.assetRoot) or not path.is_file():
                raise ValueError(f"Invalid template asset path: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expectedHash:
                raise ValueError(f"Template asset hash mismatch: {name}")
        with np.load(self.assetRoot/"template.npz", allow_pickle=False) as saved:
            self.arrays = {key: saved[key].copy() for key in saved.files}
        self.components = json.loads((self.assetRoot/"components.json").read_text())
        self._validateArrays()
        self.materials = []
        for material in self.manifest["materials"]:
            if material["texture"] not in files:
                raise ValueError("All material images must be integrity checked")
            with Image.open(self.assetRoot/material["texture"]) as image:
                colors = np.asarray(image.convert("RGBA"), np.float32)/255.
            colors = decodeColor(colors, "srgb")
            colors.setflags(write=False)
            self.materials.append({**material, "baseColor": colors})
        if (not self.materials or self.materials[0]["baseColor"].shape[:2]
                != self.arrays["atlasValidMask"].shape):
            raise ValueError("Head atlas and template base color resolutions differ")
        options = self.manifest.get("appearanceOptions", {})
        self.seamFeatherPixels = float(options.get("seamFeatherPixels", 6))
        self.textureBlendMode = options.get("textureBlendMode", "auto")
        self.skinExtensionPixels = float(options.get("skinExtensionPixels", 48))
        self.skinDonorInsetPixels = float(options.get("skinDonorInsetPixels", 24))
        self.hideObservedEyebrows = options.get("hideObservedEyebrows", False)
        self.hidePresetEyelashes = options.get("hidePresetEyelashes", False)
        self.eyebrowCoverageThreshold = float(options.get("eyebrowCoverageThreshold", .7))
        if (not np.isfinite(self.seamFeatherPixels) or not 1 <= self.seamFeatherPixels <= 128
                or not isinstance(self.hideObservedEyebrows, bool)
                or not isinstance(self.hidePresetEyelashes, bool)
                or not 0 < self.eyebrowCoverageThreshold <= 1):
            raise ValueError("Invalid template appearance options")
        if (self.textureBlendMode not in ("auto", "legacy", "sdfSkin")
                or not np.isfinite(self.skinExtensionPixels) or not 1 <= self.skinExtensionPixels <= 256
                or not np.isfinite(self.skinDonorInsetPixels) or not 1 <= self.skinDonorInsetPixels <= 128):
            raise ValueError("Invalid skin extrapolation options")
        # Exclude eye-corner/accessory UV islands which happen to share material 0.
        skinFaces = [part["faceIndices"] for part in self.components
                     if part.get("label") == "head" and part["materialId"] == 0]
        self.atlasSkinDomain = (buildUvLookup(self.arrays["uvs"],
                                self.arrays["uvFaces"][np.concatenate(skinFaces)],
                                self.arrays["atlasHeadMask"].shape[0])["validMask"]
                                if skinFaces else np.zeros_like(self.arrays["atlasHeadMask"]))
        self.browMasks = {}
        self.eyelashMasks = {}
        for materialId, material in enumerate(self.materials):
            if "hairTintSrgb" in material:
                tint = np.asarray(material["hairTintSrgb"], np.float32)
                if tint.shape != (3,):
                    raise ValueError("Preset hair tint must have three sRGB channels")
                decodeColor(tint[None, None], "srgb")  # Validate without changing asset RGB.
            eyebrows = [part for part in self.components if part.get("label") == "eyebrow" and part["materialId"] == materialId]
            if eyebrows:
                height, width = material["baseColor"].shape[:2]
                if height != width:
                    raise ValueError("Eyebrow coverage adapter requires a square preset atlas")
                faceIds = np.concatenate([part["faceIndices"] for part in eyebrows])
                self.browMasks[materialId] = buildUvLookup(self.arrays["uvs"], self.arrays["uvFaces"][faceIds], height)["validMask"]
            eyelashes = [part for part in self.components if part.get("label") in ("upperEyelash", "lowerEyelash") and part["materialId"] == materialId]
            if eyelashes:
                height, width = material["baseColor"].shape[:2]
                if height != width:
                    raise ValueError("Eyelash adapter requires a square preset atlas")
                faceIds = np.concatenate([part["faceIndices"] for part in eyelashes])
                self.eyelashMasks[materialId] = buildUvLookup(self.arrays["uvs"], self.arrays["uvFaces"][faceIds], height)["validMask"]
        weights = self.arrays["geometryWeights"]
        self.freeIds = np.flatnonzero(~self.arrays["staticMask"])
        laplacian = _meshLaplacian(len(weights), self.arrays["nicpFaces"])
        # Offline qualification is fixed, so the displacement solve is factored
        # once and reused. A weak zero-displacement prior stabilizes unmapped skin.
        operator = sparse.diags(20*weights+.03)+laplacian.T @ laplacian
        self.solveDisplacement = factorized(operator[self.freeIds][:, self.freeIds].tocsc()) if len(self.freeIds) else None
        self.geometryTransfer = deepcopy(self.manifest.get("geometryTransfer", {"method": "uvVertices"}))
        self.geometryMethod = self.geometryTransfer.get("method", "uvVertices")
        self.barycentricDeformer = None
        self.directArrays = None
        if self.geometryMethod == "directSurface":
            name = self.geometryTransfer.get("asset")
            if not isinstance(name, str) or name not in files:
                raise ValueError("Direct registration asset must be covered by the template manifest")
            if self.geometryTransfer.get("alignmentMode") != "directSurfaceSamples":
                raise ValueError("Direct registration requires surface-sample alignment")
            with np.load(self.assetRoot/name, allow_pickle=False) as saved:
                self.directArrays = {key: saved[key].copy() for key in saved.files}
            needed = {"sampleNicpFaceIds", "sampleNicpWeights", "sampleWeights",
                      "sourceTriangleIds", "sourceBarycentricWeights"}
            if not needed.issubset(self.directArrays):
                raise ValueError("Incomplete direct registration asset")
            direct = self.directArrays
            count = len(direct["sampleNicpFaceIds"])
            if (direct["sampleWeights"].shape != (count,)
                    or not np.isfinite(direct["sampleWeights"]).all()
                    or not np.isin(direct["sampleWeights"], [0., 1.]).all()):
                raise ValueError("Direct registration currently requires binary sample validity")
            # Verify both sides of the correspondence before serving jobs.
            sampleSurface(self.arrays["vertices"][self.arrays["nicpToFullIds"]],
                          self.arrays["nicpFaces"],
                          {"triangleIds": direct["sampleNicpFaceIds"],
                           "barycentricWeights": direct["sampleNicpWeights"]})
            sampleSurface(self.arrays["sourceVertices"], self.arrays["sourceFaces"],
                          {"triangleIds": direct["sourceTriangleIds"],
                           "barycentricWeights": direct["sourceBarycentricWeights"]})
            for values in direct.values():
                values.setflags(write=False)
        elif self.geometryMethod == "legacyBarycentric":
            from .barycentricDeformation import BarycentricDeformer

            name = self.geometryTransfer.get("asset")
            if not isinstance(name, str) or name not in files:
                raise ValueError("Barycentric geometry asset must be covered by the template manifest")
            if self.geometryTransfer.get("alignmentMode", "sharedUvVertices") != "sharedUvVertices":
                raise ValueError("Unsupported barycentric alignment mode")
            with np.load(self.assetRoot/name, allow_pickle=False) as saved:
                self.barycentricArrays = {key: saved[key].copy() for key in saved.files}
            needed = {"sampleNicpFaceIds", "sampleNicpWeights", "sampleWeights",
                      "sourceTriangleIds", "sourceBarycentricWeights"}
            if not needed.issubset(self.barycentricArrays):
                raise ValueError("Incomplete barycentric geometry asset")
            sampling = self._barycentricSourceMapping()
            if (sampling["triangleIds"].dtype.kind not in "iu"
                    or sampling["triangleIds"].shape != self.barycentricArrays["sampleWeights"].shape):
                raise ValueError("Barycentric source sampling must address each template sample")
            sampleSurface(self.arrays["sourceVertices"], self.arrays["sourceFaces"], sampling)
            self.barycentricDeformer = BarycentricDeformer(
                self.arrays["vertices"][self.arrays["nicpToFullIds"]], self.arrays["nicpFaces"],
                self.barycentricArrays["sampleNicpFaceIds"], self.barycentricArrays["sampleNicpWeights"],
                self.barycentricArrays["sampleWeights"], self.arrays["staticMask"],
                **{key: self.geometryTransfer.get(key, default)
                   for key, default in (("dataWeight", 20.), ("smoothWeight", 1.), ("restWeight", .03))})
            for values in self.barycentricArrays.values():
                values.setflags(write=False)
        elif self.geometryMethod != "uvVertices":
            raise ValueError(f"Unsupported client geometry transfer method: {self.geometryMethod}")
        self.skinTree = cKDTree(self.arrays["vertices"][self.arrays["nicpToFullIds"]])
        self.eyeSocketBindings = {}
        # Qualified only for this original male bundle. Existing female and
        # legacy assembly paths retain their separately established behavior.
        if (self.geometryMethod == "directSurface"
                and self.manifest.get("templateId") == "legacyMaleDirectServiceV1"):
            from .eyeSocketPlacement import buildEyeSocketBindings
            self.eyeSocketBindings = buildEyeSocketBindings(self.arrays, self.components)
        for values in self.arrays.values():
            values.setflags(write=False)

    def _barycentricSourceMapping(self) -> dict:
        return {"triangleIds": self.barycentricArrays["sourceTriangleIds"],
                "barycentricWeights": self.barycentricArrays["sourceBarycentricWeights"]}

    def _validateArrays(self) -> None:
        arrays = self.arrays
        required = {"vertices", "faces", "uvs", "uvFaces", "materialIds", "nicpToFullIds", "nicpFaces",
                    "sourceVertices", "sourceFaces", "sourceUvs", "triangleIds", "barycentricWeights",
                    "geometryWeights", "staticMask", "atlasSourceUvs", "atlasValidMask", "atlasHeadMask"}
        if not required.issubset(arrays):
            raise ValueError(f"Template arrays missing: {sorted(required-set(arrays))}")
        for name, values in arrays.items():
            if values.dtype.kind not in "fbiu" or not np.isfinite(values).all():
                raise ValueError(f"Invalid template array: {name}")
        for name, width in (("vertices", 3), ("sourceVertices", 3), ("uvs", 2), ("sourceUvs", 2)):
            if (arrays[name].ndim != 2 or arrays[name].shape[1] != width or not len(arrays[name])
                    or arrays[name].dtype.kind != "f"):
                raise ValueError(f"Invalid template coordinate shape: {name}")
        for name, bound in (("faces", len(arrays["vertices"])), ("uvFaces", len(arrays["uvs"])),
                            ("sourceFaces", len(arrays["sourceVertices"])), ("nicpFaces", len(arrays["nicpToFullIds"]))):
            values = arrays[name]
            if (values.ndim != 2 or values.shape[1] != 3 or not len(values) or values.dtype.kind not in "iu"
                    or np.any(values < 0) or np.any(values >= bound)):
                raise ValueError(f"Invalid template triangle indices: {name}")
        ids = arrays["nicpToFullIds"]
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= len(arrays["vertices"]))):
            raise ValueError("NICP subset must address unique original vertices")
        weights, static = arrays["geometryWeights"], arrays["staticMask"]
        if (weights.shape != ids.shape or np.any(weights < 0) or np.any(weights > 1)
                or static.shape != ids.shape or static.dtype.kind != "b" or not np.any(weights > 0)
                or np.any(weights[static] > 0)):
            raise ValueError("Invalid correspondence weights or static boundary")
        if arrays["triangleIds"].shape != ids.shape or arrays["triangleIds"].dtype.kind not in "iu":
            raise ValueError("Mapping must address each NICP vertex")
        sampleSurface(arrays["sourceVertices"], arrays["sourceFaces"], arrays)
        valid, head = arrays["atlasValidMask"], arrays["atlasHeadMask"]
        if (valid.ndim != 2 or min(valid.shape) < 2 or valid.dtype.kind != "b" or head.dtype.kind != "b"
                or head.shape != valid.shape or arrays["atlasSourceUvs"].shape != (*valid.shape, 2)
                or np.any(valid & ~head)):
            raise ValueError("Invalid atlas transfer lookup")
        materialIds = arrays["materialIds"]
        if (arrays["uvFaces"].shape != arrays["faces"].shape or materialIds.shape != (len(arrays["faces"]),)
                or materialIds.dtype.kind not in "iu" or np.any(materialIds < 0)
                or np.any(materialIds >= len(self.manifest["materials"]))):
            raise ValueError("Invalid full-head material or UV indices")
        if (arrays["sourceUvs"].shape != (len(arrays["sourceVertices"]), 2)
                or self.manifest["sourceTopologyHash"] != arrayHash(arrays["sourceFaces"].astype(np.int64), arrays["sourceUvs"].astype(np.float32))):
            raise ValueError("UFF topology identity mismatch in prepared template")

    def _deform(self, geometry: dict) -> tuple[dict, list, dict]:
        arrays = self.arrays
        vertices = np.asarray(geometry["vertices"], np.float64)
        sourceFaces, sourceUvs = np.asarray(geometry["faces"]), np.asarray(geometry["uvs"])
        if sourceFaces.dtype.kind not in "iu":
            raise ValueError("Reconstructed topology must use integer indices")
        if "uvFaces" in geometry and not np.array_equal(geometry["uvFaces"], sourceFaces):
            raise ValueError("The prepared UFF correspondence requires one native UV per vertex")
        if (vertices.shape != arrays["sourceVertices"].shape or not np.isfinite(vertices).all()
                or arrayHash(sourceFaces.astype(np.int64), sourceUvs.astype(np.float32)) != self.manifest["sourceTopologyHash"]):
            raise ValueError("Reconstructed geometry does not match this template's UFF topology/UV identity")
        rest = arrays["vertices"][arrays["nicpToFullIds"]]
        displacement = np.zeros_like(rest)
        transferDiagnostics = {"method": self.geometryMethod, "alignmentMode": "sharedUvVertices"}
        if self.directArrays is not None:
            from .directTemplateRegistration import registerDirectTemplate

            direct = self.directArrays
            targets = sampleSurface(vertices, arrays["sourceFaces"],
                         {"triangleIds": direct["sourceTriangleIds"],
                          "barycentricWeights": direct["sourceBarycentricWeights"]})
            result = registerDirectTemplate(rest, arrays["nicpFaces"], direct["sampleNicpFaceIds"],
                         direct["sampleNicpWeights"], targets, direct["sampleWeights"] > 0,
                         np.flatnonzero(arrays["staticMask"]),
                         **{key: self.geometryTransfer.get(key, default) for key, default in
                            (("dataWeight", 10000.), ("smoothWeight", 3000.), ("restWeight", 1000.),
                             ("staticConstraint", "soft"), ("solver", "exact"), ("epochs", 600))})
            transform = result["transform"]
            deformed = ((result["rawVertices"]-transform["translation"]) @ transform["rotation"].T)/transform["scale"]
            if self.geometryTransfer.get("staticConstraint", "soft") == "hard":
                deformed[arrays["staticMask"]] = rest[arrays["staticMask"]]
            displacement = deformed-rest
            retention = np.ones(len(rest))
            quality = _triangleQuality(rest, deformed, arrays["nicpFaces"])
            quality["scope"] = "Orientation/area relative to rest, not a self-intersection or deployment acceptance check"
            samples = sampleSurface(deformed, arrays["nicpFaces"],
                         {"triangleIds": direct["sampleNicpFaceIds"],
                          "barycentricWeights": direct["sampleNicpWeights"]})
            sampleTargetsCm = ((targets-transform["translation"]) @ transform["rotation"].T)/transform["scale"]
            errors = np.linalg.norm(samples-sampleTargetsCm, axis=1)[direct["sampleWeights"] > 0]
            transferDiagnostics = {"method": self.geometryMethod, "alignmentMode": "directSurfaceSamples",
                         "solve": result["diagnostics"], "postSolveVertexGuardApplied": False,
                         "releaseStatus": self.geometryTransfer.get("releaseStatus", "localExperiment"),
                         "staticDriftMaxCm": float(np.linalg.norm(displacement[arrays["staticMask"]], axis=1).max(initial=0.))}
        else:
            target = sampleSurface(vertices, arrays["sourceFaces"], arrays)
            # Existing UV and barycentric paths retain their original alignment.
            transform = fitSimilarity(rest, target, arrays["geometryWeights"])
            desired = ((target-transform["translation"]) @ transform["rotation"].T)/transform["scale"]-rest
        if self.barycentricDeformer is not None:
            sampleTargets = sampleSurface(vertices, arrays["sourceFaces"], self._barycentricSourceMapping())
            sampleTargetsCm = ((sampleTargets-transform["translation"]) @ transform["rotation"].T)/transform["scale"]
            solvedVertices, solveDiagnostics = self.barycentricDeformer.deform(sampleTargetsCm)
            displacement = solvedVertices-rest
            transferDiagnostics["solve"] = solveDiagnostics
        elif self.directArrays is None and self.solveDisplacement is not None:
            displacement[self.freeIds] = self.solveDisplacement((20*arrays["geometryWeights"][:, None]*desired)[self.freeIds])
        if self.directArrays is None:
            deformed, retention, quality = guardDisplacement(rest, displacement, arrays["nicpFaces"])
        if self.barycentricDeformer is not None:
            mapped = sampleSurface(deformed, arrays["nicpFaces"],
                                   {"triangleIds": self.barycentricArrays["sampleNicpFaceIds"],
                                    "barycentricWeights": self.barycentricArrays["sampleNicpWeights"]})
            errors = np.linalg.norm(mapped-sampleTargetsCm, axis=1)[self.barycentricArrays["sampleWeights"] > 0]
            transferDiagnostics["guardedSampleResidualMedianCm"] = float(np.median(errors))
            transferDiagnostics["guardedSampleResidualP95Cm"] = float(np.percentile(errors, 95))
        assembled, components = self._assembleSkin(deformed)
        if self.directArrays is None:
            errors = np.linalg.norm(deformed-(rest+desired), axis=1)[arrays["geometryWeights"] > 0]
        diagnostics = {"templateToUff": transform, "deformationRetentionMean": float(retention.mean()),
                       "geometryTransfer": transferDiagnostics,
                       "deformationRetentionMin": float(retention.min()),
                       "guardedVertexCount": int(np.count_nonzero(retention < 1)),
                       "deformationRetention": retention, "topologySafety": quality,
                       "qualifiedVertexCount": (int(np.count_nonzero(arrays["geometryWeights"]))
                           if self.directArrays is None else int(len(np.unique(arrays["nicpFaces"][
                               self.directArrays["sampleNicpFaceIds"][self.directArrays["sampleWeights"] > 0]])))),
                       "staticVertexCount": int(arrays["staticMask"].sum()),
                       "maxSkinDisplacementCm": float(np.linalg.norm(deformed-rest, axis=1).max()),
                       "targetResidualMedianCm": float(np.median(errors)), "targetResidualP95Cm": float(np.percentile(errors, 95))}
        if self.directArrays is not None:
            diagnostics["qualifiedSampleCount"] = int(np.count_nonzero(self.directArrays["sampleWeights"]))
            diagnostics["targetResidualPopulation"] = "Client triangle surface samples, not vertex targets"
        return assembled, components, diagnostics

    def _assembleSkin(self, deformed: np.ndarray, *, eyePlacementMode="legacyNearest") -> tuple[dict, list]:
        """Insert client-centimeter skin and place each original accessory once."""
        if eyePlacementMode not in ("assetDefault", "legacyNearest"):
            raise ValueError("eyePlacementMode must be assetDefault or legacyNearest")
        arrays = self.arrays
        rest = arrays["vertices"][arrays["nicpToFullIds"]]
        deformed = np.asarray(deformed, np.float64)
        if deformed.shape != rest.shape or not np.isfinite(deformed).all():
            raise ValueError("Final client skin must be finite and match the original skin vertex count")
        assembled = arrays["vertices"].copy()
        assembled[arrays["nicpToFullIds"]] = deformed
        skinIds = set(arrays["nicpToFullIds"].tolist())
        components, movedIds = [], set()
        for component in self.components:
            ids = np.asarray(component["vertexIndices"], np.int64)
            if not len(ids) or np.any(ids < 0) or np.any(ids >= len(assembled)):
                raise ValueError("Invalid accessory component vertex indices")
            pivot = arrays["vertices"][ids].mean(axis=0)
            rotation, translation = np.eye(3), np.zeros(3)
            placement = "retargetedSkin"
            placementDetails = {}
            if not set(ids.tolist()).issubset(skinIds):
                if set(ids.tolist()) & (skinIds | movedIds):
                    raise ValueError("Accessory components must not share skinned or other accessory vertices")
                socket = (self.eyeSocketBindings.get(component.get("label"))
                          if eyePlacementMode == "assetDefault" else None)
                if socket is not None:
                    from .eyeSocketPlacement import placeEyeInSocket
                    fitted = placeEyeInSocket(socket, deformed)
                    if not np.array_equal(ids, socket["componentVertexIds"]):
                        raise ValueError("Eye component identity changed after socket binding")
                    assembled[ids] = fitted["vertices"]
                    rotation = np.eye(3)*fitted["uniformScale"]
                    translation = fitted["translationCm"]
                    placement = "visibleSocketTranslation"
                    placementDetails = {"uniformScale": fitted["uniformScale"],
                                        "eyeSocketPlacement": fitted["diagnostics"]}
                else:
                    # Legacy initialization for teeth, tongue and other parts.
                    # Qualified eyes use their visible socket anchors instead.
                    _, anchors = self.skinTree.query(arrays["vertices"][ids], k=min(4, len(rest)))
                    anchors = np.unique(anchors)
                    source, targetAnchors = rest[anchors], deformed[anchors]
                    sourceMean, targetMean = source.mean(axis=0), targetAnchors.mean(axis=0)
                    u, _, vt = np.linalg.svd((source-sourceMean).T @ (targetAnchors-targetMean))
                    fix = np.diag([1., 1., np.linalg.det(u @ vt)])
                    rotation = u @ fix @ vt
                    translation = targetMean-sourceMean @ rotation
                    assembled[ids] = arrays["vertices"][ids] @ rotation+translation
                    placement = "localSkinRigidFit"
                movedIds.update(ids.tolist())
            matrix = np.eye(4)
            matrix[:3, :3], matrix[3, :3] = rotation, translation
            components.append({**deepcopy(component), "placement": placement, **placementDetails,
                               "pivotBeforeCm": pivot, "pivotAfterCm": assembled[ids].mean(axis=0),
                               "translationCm": translation, "transformRow4x4": matrix,
                               "transformAppliesToComponent": not set(ids.tolist()).issubset(skinIds)})
        return {key: assembled if key == "vertices" else arrays[key].copy()
                for key in ("vertices", "faces", "uvs", "uvFaces", "materialIds")}, components

    def _materialsFromHead(self, headLinear: np.ndarray, browCoverage: list):
        """Apply the shared preset hair color/alpha policy without rebaking RGB."""
        materials = [{**{key: deepcopy(value) for key, value in material.items() if key != "baseColor"},
                      "baseColor": headLinear if index == 0 else material["baseColor"].copy()}
                     for index, material in enumerate(self.materials)]
        hairPolicy = []
        for materialId, material in enumerate(materials):
            if "hairTintSrgb" in material:
                material["baseColor"][..., :3] = decodeColor(np.asarray(material["hairTintSrgb"], np.float32)[None, None], "srgb")[0, 0]
                hairPolicy.append({"materialId": materialId, "rgbPolicy": "constantPresetTintWithOriginalAlpha",
                                   "tintSrgb": deepcopy(material["hairTintSrgb"]), "estimatedFromPhoto": False})
        hiddenBrowMaterials = []
        if self.hideObservedEyebrows:
            for materialId, mask in self.browMasks.items():
                fractions = [item["fraction"] for item in browCoverage if item["materialId"] == materialId]
                # Both sides can share the same atlas. Hide that shared region
                # only if every affected eyebrow has qualified photo support.
                if fractions and min(fractions) >= self.eyebrowCoverageThreshold:
                    materials[materialId]["baseColor"][mask, 3] = 0
                    hiddenBrowMaterials.append(materialId)
        hiddenEyelashMaterials = []
        if self.hidePresetEyelashes:
            for materialId, mask in self.eyelashMasks.items():
                materials[materialId]["baseColor"][mask, 3] = 0
                hiddenEyelashMaterials.append(materialId)
        return materials, hairPolicy, hiddenBrowMaterials, hiddenEyelashMaterials

    def _transferTexture(self, texture: dict, colorEncoding: str, *, textureBlendMode=None, includeDebug=False):
        arrays = self.arrays
        if "linearAlbedo" in texture:
            colors, colorEncoding = decodeColor(texture["linearAlbedo"], "linear"), "linear"
        else:
            colors = decodeColor(texture["albedo"], colorEncoding)
        coverage = np.asarray(texture["coverageMask"])
        if (coverage.shape != colors.shape[:2] or coverage.dtype.kind not in "bfiu"
                or not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1)):
            raise ValueError("Texture coverage must be finite HW in [0,1], independent of RGB")
        valid, head = arrays["atlasValidMask"], arrays["atlasHeadMask"]
        points = arrays["atlasSourceUvs"][valid]*np.array([colors.shape[1]-1, colors.shape[0]-1])
        projected = np.zeros((*valid.shape, 3), np.float32)
        projected[valid] = samplePixels(colors[..., :3], points)
        observed = np.zeros(valid.shape, bool)
        observed[valid] = samplePixels(coverage.astype(np.float32), points) >= .999
        alpha = np.minimum(distance_transform_edt(observed)/self.seamFeatherPixels, 1.).astype(np.float32)
        template = self.materials[0]["baseColor"].copy()
        colorMask = observed & (template[..., :3].min(axis=-1) > .08) & (projected.min(axis=-1) > .03)
        gain = np.ones(3)
        if colorMask.sum() >= 128:
            gain = np.clip(np.median(projected[colorMask], axis=0)/np.maximum(np.median(template[..., :3][colorMask], axis=0), .01), .65, 1.5)
        # Color-match only the fallback. Fully observed reconstructed texels do
        # not change; eyebrow/eyelash alpha and other preset materials survive.
        template[..., :3] = np.clip(template[..., :3]*gain, 0, 1)
        mode = self.textureBlendMode if textureBlendMode is None else textureBlendMode
        requestedMode = mode
        if mode == "auto":
            mode = "sdfSkin" if "safeSkinMask" in texture and "protectedFeatureMask" in texture else "legacy"
        blendMaps, blendDebug, blendDiagnostics = {}, {}, {"method": "legacyInwardFeather"}
        if mode == "sdfSkin":
            from .skinExtrapolation import blendSkinExterior

            if "safeSkinMask" not in texture or "protectedFeatureMask" not in texture:
                raise ValueError("sdfSkin requires explicit safeSkinMask and protectedFeatureMask")

            def projectMask(key, *, conservative=False):
                if key not in texture:
                    return None
                value = np.asarray(texture[key])
                if value.shape != coverage.shape or value.dtype != np.bool_:
                    raise ValueError(f"{key} must be a boolean source-texture mask")
                mapped = np.zeros(valid.shape, bool)
                sampled = samplePixels(value.astype(np.float32), points)
                mapped[valid] = sampled > 0 if conservative else sampled >= .999
                return mapped

            confidence = None
            if "confidence" in texture:
                value = np.asarray(texture["confidence"])
                if value.shape != coverage.shape or not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("Texture confidence must be finite nonnegative HW weights")
                confidence = np.zeros(valid.shape, np.float32)
                confidence[valid] = samplePixels(value.astype(np.float32), points)
            blended = blendSkinExterior(projected, observed, template[..., :3], self.atlasSkinDomain,
                        safeSkinMask=projectMask("safeSkinMask"),
                        protectedMask=projectMask("protectedFeatureMask", conservative=True),
                        confidence=confidence, seamFeatherPixels=self.seamFeatherPixels,
                        insetPixels=self.skinDonorInsetPixels, extensionPixels=self.skinExtensionPixels)
            template[..., :3], alpha = blended["color"], blended["alpha"]
            blendMaps, blendDebug, blendDiagnostics = blended["maps"], blended["debug"], blended["diagnostics"]
        elif mode == "legacy":
            template[..., :3] = projected*alpha[..., None]+template[..., :3]*(1-alpha[..., None])
        else:
            raise ValueError("textureBlendMode must be auto, legacy or sdfSkin")
        blendDiagnostics["requestedMode"] = requestedMode
        blendDiagnostics["resolvedMode"] = mode
        browCoverage = []
        for component in self.components:
            if component.get("label") != "eyebrow":
                continue
            ids = np.asarray(component["vertexIndices"], np.int64)
            _, nearest = self.skinTree.query(arrays["vertices"][ids])
            mappedUvs = sampleSurface(arrays["sourceUvs"], arrays["sourceFaces"],
                                      {"triangleIds": arrays["triangleIds"][nearest],
                                       "barycentricWeights": arrays["barycentricWeights"][nearest]})
            supported = samplePixels(coverage.astype(np.float32), mappedUvs*np.array([colors.shape[1]-1, colors.shape[0]-1])) >= .999
            supported &= arrays["geometryWeights"][nearest] > 0
            browCoverage.append({"name": component["name"], "materialId": component["materialId"],
                                 "fraction": float(supported.mean())})
        materials, hairPolicy, hiddenBrowMaterials, hiddenEyelashMaterials = self._materialsFromHead(template, browCoverage)
        maps = {**blendMaps, "photoBlendAlpha": alpha, "sourceCoverage": observed,
                "templateFallback": (head.astype(np.float32)*(1-alpha)).astype(np.float32)}
        diagnostics = {"sourceColorEncoding": colorEncoding, "templateColorEncoding": "srgb",
                       "correspondence": deepcopy(self.manifest.get("textureTransfer", {"method": "legacyNicpUvChart"})),
                       "headObservedTexelFraction": float(observed.sum()/max(head.sum(), 1)),
                       "headPhotoBlendFraction": float(alpha.sum()/max(head.sum(), 1)),
                       "headDirectPhotoBlendFraction": float(blendMaps.get("directPhotoBlendAlpha", alpha).sum()/max(head.sum(), 1)),
                       "headEstimatedSkinBlendFraction": float(blendMaps.get("estimatedSkinBlendAlpha", np.zeros_like(alpha)).sum()/max(head.sum(), 1)),
                       "photoBlendAlphaMeaning": "Total source-derived contribution, including explicitly marked estimated skin; sourceCoverage alone records observations",
                       "templateLinearColorGain": gain,
                       "boundaryBlend": blendDiagnostics,
                       "seamFeatherPixels": self.seamFeatherPixels, "roughnessSource": "presetNotEstimated",
                       "hairMaterialPolicy": hairPolicy, "eyebrowPhotoSupport": browCoverage,
                       "hiddenEyebrowMaterialIds": hiddenBrowMaterials,
                       "hiddenEyelashMaterialIds": hiddenEyelashMaterials,
                       "eyelashPolicy": "hiddenByPresetConfiguration" if self.hidePresetEyelashes else "templateAlpha",
                       "unobservedRegions": ("bounded skin extrapolation at exterior boundary, then color-matched template"
                                             if mode == "sdfSkin" else "original template with bounded global color matching")}
        result = (materials, maps, diagnostics)
        return (*result, blendDebug) if includeDebug else result

    def assemble(self, geometry: dict, texture: dict, *, colorEncoding: str = "gamma2.2", textureBlendMode=None) -> dict:
        start = perf_counter()
        assembled, components, geometryDiagnostics = self._deform(geometry)
        materials, maps, textureDiagnostics, blendDebug = self._transferTexture(
            texture, colorEncoding, textureBlendMode=textureBlendMode, includeDebug=True)
        diagnostics = {**geometryDiagnostics, "texture": textureDiagnostics,
                       "assemblySeconds": perf_counter()-start,
                       "assetManifestSha256": self.manifestSha256,
                       "inputIdentities": {"geometryArrayHash": arrayHash(geometry["vertices"], geometry["faces"], geometry["uvs"]),
                                           "colorKey": "linearAlbedo" if "linearAlbedo" in texture else "albedo",
                                           "textureArrayHash": arrayHash(texture.get("linearAlbedo", texture.get("albedo")), texture["coverageMask"])},
                       "mappingStatus": self.manifest.get("mappingStatus", "prototypeUvCorrespondence"),
                       "bindingContract": "Original vertex/face/UV order retained; client reuses its existing rig and skin weights.",
                       "limitations": ["Legacy correspondence has structural/UV checks; independent anatomical accuracy remains unverified.",
                                       "Rigid accessory placement is an initialization, not a collision or expression validation.",
                                       "Eyes, teeth, tongue, eyebrows/eyelashes and unseen head areas use template assets.",
                                       "No skeleton, skin weights, blendshapes, calibrated roughness or normal map is generated."]}
        if "safeSkinMask" in texture and "protectedFeatureMask" in texture:
            diagnostics["inputIdentities"]["skinSupportArrayHash"] = arrayHash(texture["safeSkinMask"], texture["protectedFeatureMask"])
        return {"geometry": assembled, "templateVertices": self.arrays["vertices"].copy(),
                "materials": materials, "components": components, "diagnostics": diagnostics,
                "templateId": self.manifest["templateId"], "templateHashes": deepcopy(self.manifest["templateHashes"]),
                "provenanceMaps": maps, "textureBlendDebug": blendDebug}

    def assembleFinalClientSurface(self, finalSkinCanonical: dict, templateToUff: dict,
                                   headLinearClientAtlas: np.ndarray, provenance: dict, *,
                                   diagnostics: dict | None = None,
                                   eyePlacementMode: str = "assetDefault",
                                   accessoryReferenceCanonical: np.ndarray | None = None) -> dict:
        """Package a completed client-UV skin without repeating UFF transfer.

        ``finalSkinCanonical`` has original client-skin ``faces``, complete
        client ``uvs`` and corresponding skin ``uvFaces``, but its ``vertices``
        are in the same UFF frame as the supplied fixed camera/registration.
        ``templateToUff`` uses ``canonical = (clientCm @ rotation)*scale +
        translation``. Head RGB is already linear and in the original client
        atlas; no resampling, blending, gamma decoding or UV flip happens here.

        ``provenance['coverageMask']`` is a boolean HW observation map, excluding
        filtering padding. Optional finite float/bool HW maps are carried to
        the exporter; integer source-index maps are hashed in diagnostics.
        Other materials retain the common preset hair/alpha policy. Accessories
        are fitted from original template positions to final skin by default.
        Optional ``accessoryReferenceCanonical`` is a finite floating-point
        skin vertex array in the same canonical frame and vertex order, taken
        before rendered-jaw refinement. It freezes accessory placement against
        that reference while retaining the supplied final skin exactly.
        ``eyePlacementMode='assetDefault'`` enables the visible-socket eye
        adapter only on its qualified male direct template; ``legacyNearest``
        replays the earlier accessory fitting for controlled comparisons.
        Geometry/collision acceptance belongs to the caller's refinement stage.
        """
        start = perf_counter()
        arrays = self.arrays
        if not isinstance(finalSkinCanonical, dict) or not isinstance(templateToUff, dict):
            raise ValueError("Final surface and its template-to-UFF transform must be dictionaries")
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise ValueError("Final surface diagnostics must be a dictionary when supplied")
        if not {"vertices", "faces", "uvs", "uvFaces"}.issubset(finalSkinCanonical):
            raise ValueError("Final client surface requires vertices, faces, uvs and uvFaces")
        canonical = np.asarray(finalSkinCanonical["vertices"])
        rest = arrays["vertices"][arrays["nicpToFullIds"]]
        if canonical.shape != rest.shape or canonical.dtype.kind != "f" or not np.isfinite(canonical).all():
            raise ValueError("Final canonical skin must be a finite floating-point array in original vertex order")
        canonical = canonical.astype(np.float64)
        accessoryReference = None
        if accessoryReferenceCanonical is not None:
            accessoryReference = np.asarray(accessoryReferenceCanonical)
            if (accessoryReference.shape != canonical.shape or accessoryReference.dtype.kind != "f"
                    or not np.isfinite(accessoryReference).all()):
                raise ValueError("Accessory reference skin must be a finite floating-point array in original vertex order")
            accessoryReference = accessoryReference.astype(np.float64)
        faces, uvs, uvFaces = (np.asarray(finalSkinCanonical[key]) for key in ("faces", "uvs", "uvFaces"))
        if (faces.dtype.kind not in "iu" or not np.array_equal(faces, arrays["nicpFaces"])
                or uvs.dtype.kind != "f" or not np.array_equal(uvs, arrays["uvs"])
                or uvFaces.dtype.kind not in "iu"):
            raise ValueError("Final client skin must preserve the template's exact face and UV identity")
        fullLookup = {tuple(face): index for index, face in enumerate(arrays["faces"])}
        try:
            fullFaceIds = np.asarray([fullLookup[tuple(face)] for face in arrays["nicpToFullIds"][faces]], np.int64)
        except KeyError:
            raise ValueError("Client skin faces are missing from the full template") from None
        if (not np.all(arrays["materialIds"][fullFaceIds] == 0)
                or not np.array_equal(uvFaces, arrays["uvFaces"][fullFaceIds])):
            raise ValueError("Final client skin UV corners or head material assignment changed")

        try:
            rotation = np.asarray(templateToUff["rotation"], np.float64)
            translation = np.asarray(templateToUff["translation"], np.float64)
            scale = float(templateToUff["scale"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("Invalid template-to-UFF similarity transform") from None
        if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all()
                or not np.isfinite(translation).all() or not np.isfinite(scale) or scale <= 0
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10, rtol=0)):
            raise ValueError("Template-to-UFF transform must have finite translation, positive scale and proper rotation")
        clientCm = ((canonical-translation) @ rotation.T)/scale
        if not np.isfinite(clientCm).all():
            raise ValueError("Final skin overflowed while restoring client centimeters")
        static = arrays["staticMask"]
        roundTripScale = max(1., float(np.abs(rest).max()),
                             float(np.abs(canonical).max())/scale, float(np.abs(translation).max())/scale)
        tolerance = 1024*np.finfo(np.float64).eps*roundTripScale
        drift = float(np.abs(clientCm[static]-rest[static]).max(initial=0.))
        if drift > tolerance:
            raise ValueError("Final client skin moved fixed interface vertices beyond floating-point round-trip tolerance")
        # Only remove algebraic forward/inverse rounding; actual movement fails.
        clientCm[static] = rest[static]
        accessoryPlacementReference = "finalSkin"
        accessoryReferenceDiagnostics = {}
        accessoryClientCm = clientCm
        if accessoryReference is not None:
            accessoryClientCm = ((accessoryReference-translation) @ rotation.T)/scale
            if not np.isfinite(accessoryClientCm).all():
                raise ValueError("Accessory reference skin overflowed while restoring client centimeters")
            referenceRoundTripScale = max(1., float(np.abs(rest).max()),
                float(np.abs(accessoryReference).max())/scale, float(np.abs(translation).max())/scale)
            referenceTolerance = 1024*np.finfo(np.float64).eps*referenceRoundTripScale
            referenceDrift = float(np.abs(accessoryClientCm[static]-rest[static]).max(initial=0.))
            if referenceDrift > referenceTolerance:
                raise ValueError("Accessory reference skin moved fixed interface vertices beyond floating-point round-trip tolerance")
            accessoryClientCm[static] = rest[static]
            accessoryPlacementReference = "preRenderedJaw"
            accessoryReferenceDiagnostics = {
                "accessoryReferenceStaticRoundTripToleranceCm": referenceTolerance,
                "accessoryReferenceStaticDriftBeforeRoundTripCorrectionCm": referenceDrift}
        assembled, components = self._assembleSkin(accessoryClientCm, eyePlacementMode=eyePlacementMode)
        if accessoryReference is not None:
            # Keep reference-placed accessories and their transforms intact.
            # Only skin positions and the corresponding skin pivots change.
            assembled["vertices"][arrays["nicpToFullIds"]] = clientCm
            for component in components:
                if not component["transformAppliesToComponent"]:
                    ids = np.asarray(component["vertexIndices"], np.int64)
                    component["pivotAfterCm"] = assembled["vertices"][ids].mean(axis=0)

        colors = np.asarray(headLinearClientAtlas)
        headShape = self.materials[0]["baseColor"].shape[:2]
        if (colors.ndim != 3 or colors.shape[:2] != headShape or colors.shape[-1] not in (3, 4)
                or colors.dtype.kind != "f" or not np.isfinite(colors).all()
                or np.any(colors < 0) or np.any(colors > 1)):
            raise ValueError("Final head atlas must be finite linear RGB/RGBA in [0,1] at the original client resolution")
        if not isinstance(provenance, dict) or "coverageMask" not in provenance:
            raise ValueError("Final client texture requires explicit observation coverageMask")
        coverage = np.asarray(provenance["coverageMask"])
        if coverage.shape != headShape or coverage.dtype != np.bool_:
            raise ValueError("Final client coverageMask must be a boolean head-atlas map")
        headLinear = (colors.copy() if colors.shape[-1] == 4 else
                      np.concatenate([colors, self.materials[0]["baseColor"][..., 3:4]], axis=-1))
        maps = {"sourceCoverage": coverage.copy()}
        indexMapIdentities = {}
        for name, value in provenance.items():
            if name == "coverageMask":
                continue
            values = np.asarray(value)
            if name in {"sourceIndex", "faceSourceIndex"}:
                expectedShape = headShape if name == "sourceIndex" else (len(faces),)
                if values.dtype.kind not in "iu" or values.shape != expectedShape:
                    raise ValueError(f"Invalid final texture source-index map: {name}")
                indexMapIdentities[name] = {"arrayHash": arrayHash(values), "shape": list(values.shape),
                                             "meaning": "Source assignment, excluded from scalar blend masks"}
                continue
            if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None
                    or name.lower() in {"file", "con", "prn", "aux", "nul",
                                          *[f"com{number}" for number in range(1, 10)],
                                          *[f"lpt{number}" for number in range(1, 10)]}
                    or any(name != saved and name.lower() == saved.lower() for saved in maps)
                    or values.shape != headShape or values.dtype.kind not in "fb"
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Final texture provenance must contain named finite float/bool HW maps in [0,1]")
            if name == "sourceCoverage" and not np.array_equal(values, coverage):
                raise ValueError("Final sourceCoverage disagrees with coverageMask")
            maps[name] = values.copy()

        # This atlas already uses client UVs. Sample texel centers at uv*size-.5
        # and require support on every UV corner of a skin vertex at a seam.
        points = arrays["uvs"][uvFaces].reshape(-1, 2)*np.array([headShape[1], headShape[0]])-.5
        supportedCorners = samplePixels(coverage.astype(np.float32), points) >= .999
        supportedCorners &= (points >= 0).all(axis=1) & (points <= np.array([headShape[1]-1, headShape[0]-1])).all(axis=1)
        supportedVertices = np.ones(len(rest), bool)
        np.logical_and.at(supportedVertices, faces.reshape(-1), supportedCorners)
        usedVertices = np.zeros(len(rest), bool)
        usedVertices[faces.reshape(-1)] = True
        supportedVertices &= usedVertices
        browCoverage = []
        for component in self.components:
            if component.get("label") != "eyebrow":
                continue
            ids = np.asarray(component["vertexIndices"], np.int64)
            _, nearest = self.skinTree.query(arrays["vertices"][ids])
            supported = supportedVertices[nearest] & (arrays["geometryWeights"][nearest] > 0)
            browCoverage.append({"name": component["name"], "materialId": component["materialId"],
                                 "fraction": float(supported.mean())})
        materials, hairPolicy, hiddenBrows, hiddenLashes = self._materialsFromHead(headLinear, browCoverage)
        if not np.array_equal(materials[0]["baseColor"][..., :3], colors[..., :3]):
            raise ValueError("Preset material policy unexpectedly changed final head RGB")
        resultDiagnostics = {
            "assemblySeconds": perf_counter()-start, "assetManifestSha256": self.manifestSha256,
            "templateToUff": {"rotation": rotation, "translation": translation, "scale": scale},
            "geometryTransfer": {"method": "finalClientSurface", "coordinateInput": "uffCanonical",
                                 "coordinateOutput": "originalTemplateCentimeters", "geometryRefitted": False,
                                 "staticRoundTripToleranceCm": tolerance, "staticDriftBeforeRoundTripCorrectionCm": drift,
                                 "accessoryPlacement": "Original template to designated reference skin once; qualified eyes use visible socket translation and bounded uniform scale, other accessories use rigid fits",
                                 "accessoryPlacementReference": accessoryPlacementReference,
                                 **accessoryReferenceDiagnostics,
                                 "eyePlacementMode": eyePlacementMode},
            "staticVertexCount": int(static.sum()),
            "eyeSocketPlacement": {component["label"]: deepcopy(component["eyeSocketPlacement"])
                                   for component in components if "eyeSocketPlacement" in component},
            "maxSkinDisplacementCm": float(np.linalg.norm(clientCm-rest, axis=1).max()),
            "topologySafety": {**_triangleQuality(rest, clientCm, arrays["nicpFaces"]),
                               "scope": "Relative-to-template diagnostic only; refinement/collision acceptance belongs to caller"},
            "texture": {"sourceColorEncoding": "linear", "templateColorEncoding": "srgb",
                        "correspondence": {"method": "alreadyBakedOriginalClientUv", "uvSampling": "pixelEdges"},
                        "headRgbCopiedExactly": True, "textureTransferredAgain": False,
                        "boundaryBlend": {"method": "alreadyCompletedByCaller"},
                        "observedTexelCount": int(coverage.sum()),
                        "filterPaddingMeaning": "Filtering support is not added to sourceCoverage",
                        "hairMaterialPolicy": hairPolicy, "eyebrowPhotoSupport": browCoverage,
                        "eyebrowSupportSampling": "Conservative support over original skin UV corners at nearest skin anchors",
                        "hiddenEyebrowMaterialIds": hiddenBrows, "hiddenEyelashMaterialIds": hiddenLashes,
                        "eyelashPolicy": "hiddenByPresetConfiguration" if self.hidePresetEyelashes else "templateAlpha"},
            "inputIdentities": {"geometryArrayHash": arrayHash(canonical, faces, uvs, uvFaces),
                                **({"accessoryReferenceGeometryArrayHash": arrayHash(accessoryReference)}
                                   if accessoryReference is not None else {}),
                                "textureArrayHash": arrayHash(colors, coverage), "sourceIndexMaps": indexMapIdentities},
            "upstream": deepcopy(diagnostics) if diagnostics is not None else {},
            "bindingContract": "Original complete template vertex/face/UV/material order retained; export converts centimeters to meters once.",
            "limitations": ["Caller must qualify final skin geometry and texture before packaging.",
                            "Socket-based eye placement and other rigid accessories do not certify collision freedom, gaze accuracy or expressions.",
                            "No skeleton, skin weights, blendshapes or estimated normal map is generated."]}
        return {"geometry": assembled, "templateVertices": arrays["vertices"].copy(),
                "materials": materials, "components": components, "diagnostics": resultDiagnostics,
                "templateId": self.manifest["templateId"], "templateHashes": deepcopy(self.manifest["templateHashes"]),
                "provenanceMaps": maps, "textureBlendDebug": {}}
