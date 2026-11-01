"""Offline COLMAP mesh-texturing I/O; never reconstruct or change topology.

COLMAP exports bottom-left per-face UVs at normalized pixel *edges*.
This adapter flips V once while reading PLY, then samples atlas pixel centers
as ``uv * [width, height] - 0.5``. UFF's original UV lookup keeps its own
top-left integer-center ``uv * (resolution - 1)`` convention.
Destination client UVs may opt into normalized pixel-edge coordinates with
``geometry['uvConvention'] = 'topLeftPixelEdges'``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .texture import buildUvLookup, samplePixels, validateTriangles
from .imageColor import readSrgbImage


def viewMarkerColors(viewCount):
    """Deterministic separated uint8 RGB labels for at most 64 active views.

    These are identifiers, never validity values. Order follows the exported
    workspace's activeViewIndices. Black is reserved for an empty atlas.
    """
    if isinstance(viewCount, (bool, np.bool_)) or not isinstance(viewCount, (int, np.integer)) or not 1 <= viewCount <= 64:
        raise ValueError("Marker view count must be an integer between 1 and 64")
    grid = np.array([[r, g, b] for r in (48, 112, 176, 240)
                     for g in (48, 112, 176, 240) for b in (48, 112, 176, 240)], np.float64)
    selected = []
    distances = np.sum(grid ** 2, axis=1)
    for _ in range(viewCount):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.sum((grid - grid[index]) ** 2, axis=1))
        distances[selected] = -1
    return grid[selected].astype(np.uint8)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(source):
    metadata = dict(source) if isinstance(source, dict) else {}
    value = metadata.get("image", metadata.get("path")) if metadata else source
    sourcePath = None
    sourceHash = None
    colorManagement = None
    if isinstance(value, (str, Path)):
        sourcePath = str(Path(value).resolve())
        sourceHash = _hash(sourcePath)
        image, colorManagement = readSrgbImage(sourcePath)
        with image:
            pixels = np.asarray(image).copy()
    else:
        pixels = np.asarray(value)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2:
            raise ValueError("Images must be nonempty HxWx3 RGB arrays or paths")
        if pixels.dtype == np.uint8:
            pixels = pixels.copy()
        elif pixels.dtype.kind == "f" and np.isfinite(pixels).all() and (pixels >= 0).all() and (pixels <= 1).all():
            pixels = np.rint(pixels * 255).astype(np.uint8)
        else:
            raise ValueError("RGB arrays must be uint8 or finite floats in [0,1]")
        if metadata.get("path") is not None:
            sourcePath = str(Path(metadata["path"]).resolve())
            sourceHash = _hash(sourcePath)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2:
        raise ValueError("Images must have positive dimensions >= 2")
    provenance = {"sourcePath": sourcePath, "sourceSha256": sourceHash,
                  "sourceRgbSha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
    if colorManagement is not None:
        provenance["colorManagement"] = colorManagement
    return pixels, provenance


def _camera(camera, width, height):
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    translation = np.asarray(camera["translation"], dtype=np.float64).reshape(-1)
    if intrinsics.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected camera K/R/t shaped (3,3), (3,3), (3,)")
    if not all(np.isfinite(value).all() for value in (intrinsics, rotation, translation)):
        raise ValueError("Camera parameters must be finite")
    expected = np.array([[intrinsics[0, 0], 0, intrinsics[0, 2]],
                         [0, intrinsics[1, 1], intrinsics[1, 2]], [0, 0, 1]])
    if not np.allclose(intrinsics, expected, rtol=0, atol=1e-10) or np.min(np.diag(intrinsics)[:2]) <= 0:
        raise ValueError("PINHOLE requires positive fx/fy and zero skew/projective terms")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("Camera rotation must be proper orthonormal world-to-camera R")
    if camera.get("width", width) != width or camera.get("height", height) != height:
        raise ValueError("Camera dimensions differ from the supplied image frame")
    xyzw = Rotation.from_matrix(rotation).as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    if wxyz[0] < 0:
        wxyz = -wxyz
    return intrinsics, rotation, translation, wxyz


def prepareWorkspace(directory, geometry, cameras, images) -> dict:
    """Export a NEW COLMAP workspace from already fitted cameras and RGB images.

    ``images`` is indexed by the original zero-based camera ``viewIndex``;
    entries may be paths (EXIF oriented), RGB arrays, or ``{image, path}``
    descriptors. Cameras with ``valid=False`` are excluded. Caller must supply
    intrinsics in the exact frame of these images; no resizing/cropping occurs.

    Returns JSON-compatible ``directory, imageDirectory, sparseDirectory,
    meshPath, meshSha256, imagePaths, activeViewIndices, cameras`` (per-view
    IDs, paths, hashes, byte sizes, K/R/t/quaternion), and
    ``exportProjectionChecks``. IDs start at 1; filenames are image_001.png.
    Text model has independent PINHOLE fx/fy and empty observations/points3D.
    """
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f"COLMAP workspace must be new: {directory}")
    vertices, faces = validateTriangles(geometry["vertices"], geometry["faces"], 3)
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("Mesh vertex count exceeds COLMAP int32 face indices")
    images = list(images)
    prepared, seen = [], set()
    for cameraIndex, camera in enumerate(cameras):
        if not camera.get("valid", True):
            continue
        viewIndex = camera.get("viewIndex", cameraIndex)
        if isinstance(viewIndex, (bool, np.bool_)) or not isinstance(viewIndex, (int, np.integer)) or not 0 <= viewIndex < len(images):
            raise ValueError("Accepted camera viewIndex must address an input image")
        if viewIndex in seen:
            raise ValueError("Accepted cameras have duplicate source viewIndex")
        seen.add(viewIndex)
        pixels, source = _image(images[viewIndex])
        height, width = pixels.shape[:2]
        K, R, t, quaternion = _camera(camera, width, height)
        prepared.append((int(viewIndex), pixels, source, K, R, t, quaternion))
    if not prepared:
        raise ValueError("At least one valid fitted camera is required")

    directory.mkdir(parents=True)
    imageDirectory, sparseDirectory = directory / "images", directory / "sparse"
    imageDirectory.mkdir()
    sparseDirectory.mkdir()
    cameraLines = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy"]
    imageLines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# Empty POINTS2D line follows every image"]
    records, checks = [], []
    for identifier, (viewIndex, pixels, source, K, R, t, quaternion) in enumerate(prepared, 1):
        height, width = pixels.shape[:2]
        name = f"image_{identifier:03d}.png"
        path = imageDirectory / name
        Image.fromarray(pixels).save(path)
        params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        cameraLines.append(f"{identifier} PINHOLE {width} {height} " + " ".join(format(float(value), ".17g") for value in params))
        poseText = " ".join(format(float(value), ".17g") for value in np.r_[quaternion, t])
        imageLines.extend([f"{identifier} {poseText} {identifier} {name}", ""])
        # Round-trip the actual serialized quaternion/translation, not just R.
        serialized = np.fromstring(poseText, sep=" ")
        exportedR = Rotation.from_quat(serialized[:4][[1, 2, 3, 0]]).as_matrix()
        referenceCamera = vertices @ R.T + t
        exportedCamera = vertices @ exportedR.T + serialized[4:]
        positive = (referenceCamera[:, 2] > 1e-8) & (exportedCamera[:, 2] > 1e-8)
        referencePixels = referenceCamera[positive] @ K.T
        exportedPixels = exportedCamera[positive] @ K.T
        referencePixels = referencePixels[:, :2] / referencePixels[:, 2:3]
        exportedPixels = exportedPixels[:, :2] / exportedPixels[:, 2:3]
        delta = np.linalg.norm(referencePixels - exportedPixels, axis=1)
        maxDelta = float(delta.max()) if len(delta) else None
        if maxDelta is not None and (not np.isfinite(maxDelta) or maxDelta > 1e-3):
            raise ValueError("Quaternion export changed vertex projections by more than 0.001 pixels")
        checks.append({"sourceViewIndex": viewIndex, "colmapImageId": identifier,
                       "comparedVertexCount": int(positive.sum()), "maxPixelError": maxDelta,
                       "positiveDepthFraction": float((referenceCamera[:, 2] > 1e-8).mean()),
                       "worldToCameraConvention": "X @ rotation.T + translation"})
        records.append({**source, "sourceViewIndex": viewIndex, "viewIndex": viewIndex,
                        "colmapImageId": identifier, "cameraId": identifier, "imageName": name,
                        "imagePath": str(path), "sha256": _hash(path), "sizeBytes": path.stat().st_size,
                        "width": width, "height": height, "intrinsics": K.tolist(),
                        "rotation": R.tolist(), "translation": t.tolist(), "quaternionWxyz": quaternion.tolist()})
    (sparseDirectory / "cameras.txt").write_text("\n".join(cameraLines) + "\n", encoding="ascii")
    (sparseDirectory / "images.txt").write_text("\n".join(imageLines) + "\n", encoding="ascii")
    (sparseDirectory / "points3D.txt").write_text("", encoding="ascii")
    meshPath = directory / "mesh.ply"
    with meshPath.open("w", encoding="ascii") as output:
        output.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
                     f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            output.write(" ".join(format(float(value), ".17g") for value in vertex) + "\n")
        for face in faces:
            output.write("3 " + " ".join(str(int(value)) for value in face) + "\n")
    return {"schemaVersion": 1, "directory": str(directory), "imageDirectory": str(imageDirectory),
            "sparseDirectory": str(sparseDirectory), "meshPath": str(meshPath), "meshSha256": _hash(meshPath),
            "vertexCount": len(vertices), "faceCount": len(faces),
            "imagePaths": [record["imagePath"] for record in records],
            "activeViewIndices": [record["sourceViewIndex"] for record in records],
            "cameras": records, "exportProjectionChecks": checks}


_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
              "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
              "float": "f", "float32": "f", "double": "d", "float64": "d"}


def _plyHeader(stream):
    if stream.readline().rstrip(b"\r\n") != b"ply":
        raise ValueError("Expected PLY magic/header")
    elements, textureFile, formatName = [], None, None
    while True:
        if stream.tell() > 1024 * 1024:
            raise ValueError("PLY header exceeds 1 MiB")
        raw = stream.readline()
        if not raw:
            raise ValueError("Truncated PLY header")
        fields = raw.decode("ascii").split()
        if not fields:
            continue
        if fields[0] == "end_header":
            break
        if fields[0] == "format":
            if formatName or len(fields) != 3 or fields[2] != "1.0" or fields[1] not in {"ascii", "binary_little_endian"}:
                raise ValueError("Only ASCII or binary little-endian PLY 1.0 is supported")
            formatName = fields[1]
        elif fields[0] == "element":
            if len(fields) != 3 or any(element[0] == fields[1] for element in elements):
                raise ValueError("Invalid or duplicate PLY element")
            count = int(fields[2])
            if not 0 <= count <= 10_000_000:
                raise ValueError("Invalid PLY element count")
            elements.append((fields[1], count, []))
        elif fields[0] == "property":
            if not elements:
                raise ValueError("PLY property has no element")
            if len(fields) == 3 and fields[1] in _PLY_TYPES:
                item = (fields[2], None, _PLY_TYPES[fields[1]])
            elif len(fields) == 5 and fields[1] == "list" and fields[2] in _PLY_TYPES and fields[3] in _PLY_TYPES:
                item = (fields[4], _PLY_TYPES[fields[2]], _PLY_TYPES[fields[3]])
                if item[1] in "fd":
                    raise ValueError("PLY list count type must be integral")
            else:
                raise ValueError("Unsupported PLY property declaration")
            if any(prop[0] == item[0] for prop in elements[-1][2]):
                raise ValueError("Duplicate PLY property")
            elements[-1][2].append(item)
        elif fields[:2] == ["comment", "TextureFile"]:
            textureFile = raw.decode("ascii").strip().split(maxsplit=2)[2]
        elif fields[0] not in {"comment", "obj_info"}:
            raise ValueError("Unsupported PLY header directive")
    if not formatName:
        raise ValueError("Missing PLY format")
    return formatName, elements, textureFile


def _plyRecord(stream, properties, binary):
    tokens = None if binary else iter(stream.readline().decode("ascii").split())

    def scalar(code):
        if binary:
            size = struct.calcsize("<" + code)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("Truncated binary PLY record")
            return struct.unpack("<" + code, data)[0]
        try:
            value = next(tokens)
        except StopIteration as error:
            raise ValueError("Truncated ASCII PLY record") from error
        return float(value) if code in "fd" else int(value)

    values = {}
    for name, countType, valueType in properties:
        if countType is None:
            values[name] = scalar(valueType)
        else:
            count = scalar(countType)
            if not 0 <= count <= 1024:
                raise ValueError("Invalid PLY property list size")
            values[name] = [scalar(valueType) for _ in range(count)]
    if tokens is not None and next(tokens, None) is not None:
        raise ValueError("Unexpected values at end of ASCII PLY record")
    return values


def _sameGeometry(vertices, faces, expectedGeometry):
    expectedVertices, expectedFaces = validateTriangles(expectedGeometry["vertices"], expectedGeometry["faces"], 3)
    vertices, faces = validateTriangles(vertices, faces, 3)
    if vertices.shape != expectedVertices.shape or not np.allclose(vertices, expectedVertices, rtol=2e-6, atol=2e-7):
        raise ValueError("COLMAP changed vertex positions/count/order")
    if faces.shape != expectedFaces.shape or not np.array_equal(faces, expectedFaces):
        raise ValueError("COLMAP changed triangle topology/count/order")
    return float(np.max(np.abs(vertices - expectedVertices)))


def loadTexturedMesh(plyPath, expectedGeometry) -> dict:
    """Read exact PLY vertex/triangle order; accept ASCII or little-endian binary.

    Returns ``vertices, faces, faceUvs`` (Fx3x2, native top-left/V-down),
    ``validFaces`` (finite in-range nondegenerate UV triangles), ``textureFile``
    (PLY comment only, no file read), ``uvConvention``, and ``diagnostics``.
    Optional ``faceSourceIndex`` is retained only when a scalar integer
    ``view_id`` or ``source_index`` property is actually present. COLMAP's
    ordinary output omits this: validFaces proves atlas assignment only,
    never semantic/photographic support and never a source-view identity.
    """
    plyPath = Path(plyPath)
    expectedVertices, expectedFaces = validateTriangles(expectedGeometry["vertices"], expectedGeometry["faces"], 3)
    vertices, faces, faceUvs, sourceIndices = [], [], [], []
    with plyPath.open("rb") as stream:
        formatName, elements, textureFile = _plyHeader(stream)
        counts = {name: count for name, count, _ in elements}
        if counts.get("vertex") != len(expectedVertices) or counts.get("face") != len(expectedFaces):
            raise ValueError("COLMAP changed vertex/triangle counts")
        for name, count, properties in elements:
            for _ in range(count):
                item = _plyRecord(stream, properties, formatName != "ascii")
                if name == "vertex":
                    if not all(key in item and np.isscalar(item[key]) for key in ("x", "y", "z")):
                        raise ValueError("PLY vertex is missing scalar x/y/z")
                    vertices.append([item["x"], item["y"], item["z"]])
                elif name == "face":
                    indices = item.get("vertex_indices", item.get("vertex_index"))
                    uv = item.get("texcoord")
                    if not isinstance(indices, list) or len(indices) != 3 or not all(isinstance(value, int) for value in indices):
                        raise ValueError("Expected three integer vertex indices per PLY face")
                    if not isinstance(uv, list) or len(uv) != 6:
                        raise ValueError("Expected six texture coordinates per PLY face")
                    faces.append(indices)
                    faceUvs.append(uv)
                    source = item.get("source_index", item.get("view_id"))
                    if source is not None:
                        if not isinstance(source, int) or source < -1:
                            raise ValueError("Invalid per-face source-view index")
                        sourceIndices.append(source)
        remainder = stream.read()
        if remainder and (formatName != "ascii" or remainder.strip()):
            raise ValueError("Unexpected trailing PLY data")
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    maximumError = _sameGeometry(vertices, faces, expectedGeometry)
    faceUvs = np.asarray(faceUvs, np.float64).reshape(-1, 3, 2)
    faceUvs[..., 1] = 1 - faceUvs[..., 1]
    finite = np.isfinite(faceUvs).all(axis=(1, 2))
    inRange = (faceUvs >= -1e-6).all(axis=(1, 2)) & (faceUvs <= 1 + 1e-6).all(axis=(1, 2))
    edges = faceUvs[:, 1:] - faceUvs[:, :1]
    twiceArea = np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    validFaces = finite & inRange & (twiceArea > 1e-14)
    result = {"vertices": vertices, "faces": faces, "faceUvs": faceUvs, "validFaces": validFaces,
              "textureFile": textureFile, "uvConvention": "topLeftPixelEdges",
              "diagnostics": {"plyFormat": formatName, "vertexCount": len(vertices), "faceCount": len(faces),
                              "maxVertexCoordinateError": maximumError, "validFaceCount": int(validFaces.sum()),
                              "degenerateUvFaceCount": int((finite & (twiceArea <= 1e-14)).sum()),
                              "invalidUvFaceCount": int((~finite | ~inRange).sum()),
                              "coverageMeaning": "atlasAssignmentOnly; not photo/semantic support",
                              "sourceViewIdentityAvailable": bool(sourceIndices)}}
    if sourceIndices:
        result["faceSourceIndex"] = np.asarray(sourceIndices, np.int64)
    return result


def _destinationUvLookup(geometry, resolution):
    """Rasterize target UVs without changing triangle order or UV seams."""
    uvFaces = np.asarray(geometry.get("uvFaces", geometry["faces"]))
    if uvFaces.shape != np.asarray(geometry["faces"]).shape:
        raise ValueError("Original UV triangles must match geometry face order")
    convention = geometry.get("uvConvention", "topLeftPixelCenters")
    if convention not in ("topLeftPixelCenters", "topLeftPixelEdges"):
        raise ValueError("Target uvConvention must be topLeftPixelCenters or topLeftPixelEdges")
    uvs = np.asarray(geometry["uvs"], dtype=np.float64)
    if convention == "topLeftPixelEdges":
        if not isinstance(resolution, (int, np.integer)) or resolution < 2:
            raise ValueError("resolution must be an integer >= 2")
        # buildUvLookup addresses integer centers using u*(resolution-1).
        # Change only its raster coordinates so normalized client UVs instead
        # address centers at (i+.5)/resolution, as in the WebGL texture sampler.
        uvs = (uvs * resolution - .5) / (resolution - 1)
        sampling = "client native V-down UV * resolution - 0.5"
    else:
        sampling = "original native V-down UV * (resolution-1)"
    return buildUvLookup(uvs, uvFaces, resolution), sampling


def rebakeAtlas(geometry, colmapMesh, atlasRgb, resolution=1024, coverage=None) -> dict:
    """Resample COLMAP atlas onto ORIGINAL UVs using matching face barycentrics.

    ``atlasRgb`` is top-row-first uint8 or float RGB in [0,1]. ``coverage`` may
    be an original-UV-resolution support mask, or a dict with coverageMask and
    optional confidence/sourceIndex. It is intersected with COLMAP assignment;
    neither black nor white RGB implies coverage. Without coverage, the result's
    coverageMask means atlas assignment ONLY, explicitly flagged in diagnostics.

    Returns float ``albedo`` (zero outside final support), ``rawRebaked`` (before
    support masking), bool ``coverageMask, assignedMask, atlasMask`` and
    ``diagnostics``. confidence/sourceIndex appear only if explicitly supplied
    by coverage or per-face source metadata; unsupported pixels become 0/-1.

    Destination UVs default to UFF integer-center coordinates. Set
    ``geometry['uvConvention']='topLeftPixelEdges'`` for normalized client
    texture coordinates. Separate ``uvFaces`` retain per-corner UV seams.
    """
    _sameGeometry(colmapMesh["vertices"], colmapMesh["faces"], geometry)
    if colmapMesh.get("uvConvention") != "topLeftPixelEdges":
        raise ValueError("Use loadTexturedMesh: rebake requires already V-flipped COLMAP edge UVs")
    atlas = np.asarray(atlasRgb)
    if atlas.ndim != 3 or atlas.shape[2] != 3 or min(atlas.shape[:2]) < 1:
        raise ValueError("COLMAP atlas must be an HxWx3 RGB image")
    if atlas.dtype == np.uint8:
        atlas = atlas.astype(np.float32) / 255
    elif atlas.dtype.kind == "f" and np.isfinite(atlas).all() and (atlas >= 0).all() and (atlas <= 1).all():
        atlas = atlas.astype(np.float32)
    else:
        raise ValueError("COLMAP atlas must be uint8 or finite floats in [0,1]")
    faceUvs = np.asarray(colmapMesh["faceUvs"], np.float64)
    validFaces = np.asarray(colmapMesh["validFaces"])
    faceCount = len(geometry["faces"])
    if faceUvs.shape != (faceCount, 3, 2) or validFaces.shape != (faceCount,) or validFaces.dtype.kind != "b":
        raise ValueError("Invalid COLMAP face UVs/assignment mask")
    if not np.isfinite(faceUvs[validFaces]).all():
        raise ValueError("Assigned COLMAP UVs must be finite")
    lookup, destinationSampling = _destinationUvLookup(geometry, resolution)
    atlasMask = lookup["validMask"]
    assigned = atlasMask.copy()
    assigned[atlasMask] &= validFaces[lookup["faceIds"][atlasMask]]
    selected = lookup["faceIds"][assigned]
    positions = np.sum(faceUvs[selected] * lookup["barycentrics"][assigned, :, None], axis=1)
    positions = positions * [atlas.shape[1], atlas.shape[0]] - 0.5
    # Border texel centers must exist; never count partial zero-border samples
    # as photographic evidence. COLMAP pads packed patches in ordinary output.
    inside = (positions >= -1e-6).all(axis=1) & (positions <= [atlas.shape[1] - 1 + 1e-6, atlas.shape[0] - 1 + 1e-6]).all(axis=1)
    raw = np.zeros((*atlasMask.shape, 3), np.float32)
    raw[assigned] = samplePixels(atlas, np.clip(positions, [0, 0], [atlas.shape[1] - 1, atlas.shape[0] - 1]))
    assigned[assigned] = inside
    raw[~assigned] = 0
    final = assigned.copy()
    extras = coverage if isinstance(coverage, dict) else {}
    if coverage is not None:
        mask = np.asarray(extras.get("coverageMask") if isinstance(coverage, dict) else coverage)
        if mask.shape != final.shape or mask.dtype.kind not in "bifu" or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError("Coverage must match the original UV atlas and lie in [0,1]")
        final &= mask >= .999
    albedo = raw.copy()
    albedo[~final] = 0
    result = {"albedo": albedo, "rawRebaked": raw, "coverageMask": final, "assignedMask": assigned,
              "atlasMask": atlasMask, "diagnostics": {"resolution": int(resolution),
                "atlasWidth": int(atlas.shape[1]), "atlasHeight": int(atlas.shape[0]),
                "atlasTexelCount": int(atlasMask.sum()), "assignedTexelCount": int(assigned.sum()),
                "coveredTexelCount": int(final.sum()), "externalCoverageApplied": coverage is not None,
                "coverageMeaning": "atlasAssignmentIntersectedWithExternalSupport" if coverage is not None else "atlasAssignmentOnly; not photo/semantic support",
                "sourceSampling": "COLMAP top-left edge UV * [width,height] - 0.5",
                "destinationSampling": destinationSampling,
                "verticalFlipsDuringRebake": 0, "inferredCoverageFromRgb": False}}
    if "confidence" in extras:
        confidence = np.asarray(extras["confidence"], np.float32)
        if confidence.shape != final.shape or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError("Confidence must match the original atlas and lie in [0,1]")
        result["confidence"] = np.where(final, confidence, 0).astype(np.float32)
    if "sourceIndex" in extras:
        source = np.asarray(extras["sourceIndex"])
        if source.shape != final.shape or source.dtype.kind not in "iu" or (source < -2).any():
            raise ValueError("Source indices must match original atlas and be integer >= -2")
        result["sourceIndex"] = np.where(final, source, -1).astype(np.int64)
    elif "faceSourceIndex" in colmapMesh:
        source = np.asarray(colmapMesh["faceSourceIndex"])
        if source.shape != (faceCount,) or source.dtype.kind not in "iu" or (source < -1).any():
            raise ValueError("Invalid COLMAP per-face source indices")
        result["sourceIndex"] = np.full(final.shape, -1, np.int64)
        result["sourceIndex"][final] = source[lookup["faceIds"][final]]
    return result


def sourceSupportFromMarkers(geometry, colmapMesh, markerAtlas, cameras,
                             validMasks, activeViewIndices, resolution=1024):
    """Recover selected views, then sample ORIGINAL validity at COLMAP positions.

    Caller must run constant ``viewMarkerColors`` images with color correction
    disabled, matching color-pass geometry/atlas options, and check color and
    marker PLY face UVs plus atlas dimensions match before calling this helper.
    Marker inpaint may match the color pass: it carries VIEW IDENTITY only.

    A face is decoded only when seven interior barycentric samples all match
    one marker within 2.5 uint8 levels in every channel. No majority vote or
    relaxed tiny-triangle threshold is used. Tiny/ambiguous faces remain -1.
    The markers never establish photographic validity. Coverage comes solely
    from the original selected-view validMasks (bilinear >= .999), intersected
    with assigned/decoded faces and the source bilinear image bounds.

    COLMAP 4.1.0 BakeTexture uses affine interpolation of PROJECTED CORNERS,
    not perspective projection of a barycentrically interpolated 3D point.
    Bitmap::InterpolateBilinear uses integer pixel centers, no half-pixel shift,
    and needs x+1/y+1 neighbors. Its exact domain is [0,W-1) x [0,H-1).

    Returns coverageMask, sourceIndex (-1 outside coverage), faceSourceIndex
    (original zero-based source indices), decodedFaceMask, sampledValidity,
    atlasMask and diagnostics. No RGB albedo or confidence is inferred.
    Destination UV convention and separate uvFaces match rebakeAtlas.
    """
    vertices, faces = validateTriangles(geometry["vertices"], geometry["faces"], 3)
    _sameGeometry(colmapMesh["vertices"], colmapMesh["faces"], geometry)
    if colmapMesh.get("uvConvention") != "topLeftPixelEdges":
        raise ValueError("Marker recovery requires V-down COLMAP pixel-edge UVs")
    active = np.asarray(activeViewIndices)
    masks = list(validMasks)
    if active.ndim != 1 or active.dtype.kind not in "iu" or len(np.unique(active)) != len(active) or (active < 0).any() or (active >= len(masks)).any():
        raise ValueError("activeViewIndices must be unique original mask indices")
    palette = viewMarkerColors(len(active)).astype(np.float32)
    marker = np.asarray(markerAtlas)
    if marker.ndim != 3 or marker.shape[2] != 3 or min(marker.shape[:2]) < 2:
        raise ValueError("Marker atlas must be HxWx3 RGB")
    if marker.dtype == np.uint8:
        marker = marker.astype(np.float32)
    elif marker.dtype.kind == "f" and np.isfinite(marker).all() and (marker >= 0).all() and (marker <= 1).all():
        marker = marker.astype(np.float32) * 255
    else:
        raise ValueError("Marker atlas must be uint8 or finite float RGB in [0,1]")
    faceUvs = np.asarray(colmapMesh["faceUvs"], np.float64)
    validFaces = np.asarray(colmapMesh["validFaces"])
    if faceUvs.shape != (len(faces), 3, 2) or validFaces.shape != (len(faces),) or validFaces.dtype.kind != "b":
        raise ValueError("Invalid COLMAP face UVs/assignment mask")
    if not np.isfinite(faceUvs[validFaces]).all():
        raise ValueError("Assigned marker UVs must be finite")
    barySamples = np.array([[1 / 3, 1 / 3, 1 / 3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6],
                           [.45, .45, .1], [.45, .1, .45], [.1, .45, .45]])
    assignedFaces = np.flatnonzero(validFaces)
    positions = np.einsum("sc,fcd->fsd", barySamples, faceUvs[assignedFaces])
    positions = positions * [marker.shape[1], marker.shape[0]] - .5
    inside = (positions >= 0).all(axis=2) & (positions <= [marker.shape[1] - 1, marker.shape[0] - 1]).all(axis=2)
    samples = samplePixels(marker, positions.reshape(-1, 2)).reshape(-1, len(barySamples), 3)
    errors = np.max(np.abs(samples[:, :, None, :] - palette[None, None, :, :]), axis=3)
    nearest = np.argmin(errors, axis=2)
    minimum = np.min(errors, axis=2)
    accepted = inside & (minimum <= 2.5)
    sameView = (nearest == nearest[:, :1]).all(axis=1)
    decoded = accepted.all(axis=1) & sameView
    faceSource = np.full(len(faces), -1, np.int64)
    faceSource[assignedFaces[decoded]] = active[nearest[decoded, 0]]

    cameraByView = {}
    for index, camera in enumerate(cameras):
        if camera.get("valid", True):
            view = camera.get("viewIndex", index)
            if isinstance(view, (bool, np.bool_)) or not isinstance(view, (int, np.integer)) or view in cameraByView:
                raise ValueError("Accepted cameras require unique integer viewIndex")
            cameraByView[view] = camera
    lookup, destinationSampling = _destinationUvLookup(geometry, resolution)
    atlasMask = lookup["validMask"]
    selectedMap = np.full(atlasMask.shape, -1, np.int64)
    selectedMap[atlasMask] = faceSource[lookup["faceIds"][atlasMask]]
    validity = np.zeros(atlasMask.shape, np.float32)
    coverage = np.zeros(atlasMask.shape, bool)
    outsideCount = 0
    perView = []
    # PLY coordinates and mvs::Image's projection matrix are float32 in COLMAP.
    points = np.c_[np.asarray(colmapMesh["vertices"], np.float32), np.ones(len(vertices), np.float32)]
    for view in active.tolist():
        if view not in cameraByView:
            raise ValueError("An active marker view has no accepted camera")
        mask = np.asarray(masks[view])
        if mask.ndim != 2 or min(mask.shape) < 2 or mask.dtype.kind not in "bifu" or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError("Original validity masks must be finite HW values in [0,1]")
        height, width = mask.shape
        K, R, t, _ = _camera(cameraByView[view], width, height)
        projection = K.astype(np.float32) @ np.c_[R.astype(np.float32), t.astype(np.float32)]
        projected = points @ projection.T
        positive = projected[:, 2] > 0
        xy = np.divide(projected[:, :2], projected[:, 2:3],
                       out=np.zeros((len(vertices), 2), np.float32), where=positive[:, None])
        selected = selectedMap == view
        faceIndices = lookup["faceIds"][selected]
        # Affine source pixel position mirrors BakeTexture's face_projections.
        sourcePixels = np.sum(xy[faces[faceIndices]] * lookup["barycentrics"][selected, :, None], axis=1)
        inFrame = ((sourcePixels >= 0).all(axis=1)
                   & (sourcePixels < [width - 1, height - 1]).all(axis=1)
                   & positive[faces[faceIndices]].all(axis=1))
        values = samplePixels(mask.astype(np.float32), sourcePixels)
        values[~inFrame] = 0
        validity[selected] = values
        supported = inFrame & (values >= .999)
        coverage[selected] = supported
        outsideCount += int((~inFrame).sum())
        perView.append({"sourceViewIndex": view, "decodedFaceCount": int((faceSource == view).sum()),
                        "selectedTexelCount": int(selected.sum()), "supportedTexelCount": int(supported.sum())})
    unresolved = atlasMask & (selectedMap < 0)
    sourceIndex = np.where(coverage, selectedMap, -1)
    return {"coverageMask": coverage, "sourceIndex": sourceIndex, "faceSourceIndex": faceSource,
            "decodedFaceMask": faceSource >= 0, "sampledValidity": validity, "atlasMask": atlasMask,
            "diagnostics": {"method": "constantViewMarkersThenOriginalValidityMasks",
                "markerColorsRgb8": palette.astype(np.uint8).tolist(), "activeViewIndices": active.tolist(),
                "markerInteriorSamplesPerFace": len(barySamples), "markerMaxChannelErrorRgb8": 2.5,
                "markerConsensus": "all seven interior samples must match the same view; no tiny-face relaxation",
                "assignedFaceCount": int(validFaces.sum()), "decodedFaceCount": int(decoded.sum()),
                "undecodedAssignedFaceCount": int((~decoded).sum()),
                "mixedMarkerFaceCount": int((~sameView).sum()),
                "markerMismatchFaceCount": int((~accepted.all(axis=1)).sum()),
                "maximumAcceptedMarkerChannelErrorRgb8": float(minimum[decoded].max()) if decoded.any() else None,
                "atlasTexelCount": int(atlasMask.sum()), "coveredTexelCount": int(coverage.sum()),
                "unresolvedTexelCount": int(unresolved.sum()), "outsideSourceBilinearDomainTexelCount": outsideCount,
                "validityThreshold": .999, "sourcePosition": "affine barycentric interpolation of projected face corners",
                "destinationSampling": destinationSampling,
                "sourcePixelConvention": "integer centers, no half-pixel offset; 0<=x<W-1 and 0<=y<H-1",
                "sourceReference": "COLMAP 4.1.0 mvs/texture_mapping.cc BakeTexture; sensor/bitmap.cc InterpolateBilinear",
                "markerInpaintProvesPhotoSupport": False, "sourceViews": perView}}
