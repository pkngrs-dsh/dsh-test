"""Compile independently directed surface/UV maps from a reviewed Wrap export."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from prepareWrapBundle import readObj, sha256File
from validateWrapResult import validateResult
from faceAppearance.artifacts import writeJson
from faceAppearance.atlasTransfer import compileAtlasMap
from faceAppearance.correspondence import SurfaceIndex, arrayHash, qualifyMapping
from faceAppearance.texture import vertexNormals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=projectRoot / "assets/ictRegisteredToUvFaceFusion.obj")
    parser.add_argument("--wrapProject", type=Path, default=projectRoot / "assets/flame_to_ict.wrap")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=projectRoot / "assets/registration/wrapV1")
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new asset version; do not overwrite a compiled registration")
    bundle = projectRoot / "assets/wrapPreparation"
    structural = validateResult(bundle, args.result)
    if not structural["structuralValid"]:
        raise ValueError(structural["errors"])
    review = json.loads(args.review.read_text())
    if review.get("resultSha256") != sha256File(args.result) or review.get("status") != "prototypeCommonSurfaceQualified":
        raise ValueError("A matching common-surface review is required before compiling usable assets")
    fixed = readObj(bundle / "uvFaceFusionFixed.obj")
    registered = readObj(args.result)
    native = readObj(bundle / "ictFloating.obj")
    project = json.loads(args.wrapProject.read_text())
    freeFaces = np.asarray(project["nodes"]["SelectPolygons"]["params"]["selection"]["value"], dtype=np.int64)
    if (freeFaces < 0).any() or (freeFaces >= len(registered.faces)).any():
        raise ValueError("Invalid free polygon indices")
    forward = qualifyMapping(SurfaceIndex(fixed.vertices, fixed.faces).query(registered.vertices),
                             SurfaceIndex(fixed.vertices, fixed.faces),
                             vertexNormals(registered.vertices, registered.faces), 0.01, 0.5)
    reverse = qualifyMapping(SurfaceIndex(registered.vertices, registered.faces).query(fixed.vertices),
                             SurfaceIndex(registered.vertices, registered.faces),
                             vertexNormals(fixed.vertices, fixed.faces), 0.01, 0.5)
    ictValidFaces = forward["validMask"][registered.faces].all(axis=1)
    ictValidFaces[freeFaces] = False
    uffValidFaces = reverse["validMask"][fixed.faces].all(axis=1)
    # Guard the reverse map against the author's intentionally unconstrained region.
    reverse["validMask"] &= ictValidFaces[reverse["triangleIds"]]
    uffValidFaces &= reverse["validMask"][fixed.faces].all(axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {"uffVertices": fixed.vertices, "uffFaces": fixed.faces, "uffUvs": fixed.texCoords,
              "ictRegisteredVertices": registered.vertices, "ictNativeVertices": native.vertices,
              "ictFaces": native.faces, "ictUvs": native.texCoords, "ictUvFaces": native.uvFaces,
              "ictValidFaces": ictValidFaces, "uffValidFaces": uffValidFaces, "freeIctFaceIds": freeFaces}
    for prefix, mapping in (("uffToIct", forward), ("ictToUff", reverse)):
        for key, value in mapping.items():
            arrays[prefix + key[0].upper() + key[1:]] = value
    np.savez_compressed(args.output / "geometryMaps.npz", **arrays)
    nativeIctUvs = native.texCoords * [1, -1] + [0, 1]
    manifest = {"schemaVersion": 1, "status": "prototypeCommonSurfaceQualified",
                "review": review, "resultSha256": sha256File(args.result),
                "wrapProjectSha256": sha256File(args.wrapProject),
                "geometryFrame": "uvFaceFusionCanonical", "metricUnits": "uncalibrated",
                "uffTopologyHash": arrayHash(fixed.faces.astype(np.int64), fixed.texCoords.astype(np.float32)),
                "ictTopologyHash": arrayHash(native.faces.astype(np.int64), native.texCoords.astype(np.float32)),
                "arrayUvConvention": "u right, v down; ICT OBJ v is converted as 1-v",
                "coordinateRestoration": "No estimated RigidAlignment transform is applied",
                "maxSurfaceDistance": 0.01, "minNormalDot": 0.5,
                "freeIctFaceCount": len(freeFaces), "resolutions": {}, "files": {}}
    for resolution in (512, 1024):
        atlasArrays, atlasSummary = {}, {}
        for direction, source, target, sourceUv, targetUv, sourceMask, targetMask in (
                ("uffToIct", fixed, registered, fixed.texCoords, nativeIctUvs, uffValidFaces, ictValidFaces),
                ("ictToUff", registered, fixed, nativeIctUvs, fixed.texCoords, ictValidFaces, uffValidFaces)):
            print(f"Compiling {direction} {resolution}", flush=True)
            mapping = compileAtlasMap(source.vertices, source.faces, sourceUv, source.uvFaces,
                                       target.vertices, target.faces, targetUv, target.uvFaces,
                                       resolution, sourceMask, targetMask)
            for key, value in mapping.items():
                atlasArrays[direction + key[0].upper() + key[1:]] = value
            atlasSummary[direction] = {"validTexels": int(mapping["validMask"].sum()),
                                        "atlasTexels": int(mapping["atlasMask"].sum())}
        fileName = f"atlas{resolution}.npz"
        np.savez_compressed(args.output / fileName, **atlasArrays)
        manifest["resolutions"][str(resolution)] = atlasSummary
    for filePath in args.output.glob("*.npz"):
        manifest["files"][filePath.name] = {"sha256": sha256File(filePath), "bytes": filePath.stat().st_size}
    writeJson(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["resolutions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
