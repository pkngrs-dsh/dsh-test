"""Export a geometry-only legacy variant without touching texture payloads.

The caller supplies the complete vertex array in the source archive's recorded
UFF canonical frame. Only changed OBJ coordinate tokens are rewritten. This
preserves vertex numbering, UV/material records, binding metadata and every
other packaged payload; it does not validate anatomical fit or intersections.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .artifacts import jsonValue
from .clientAsset import _fileIdentity
from .legacyPreviewAsset import _load


def _similarity(manifest):
    try:
        value = manifest['assemblyDiagnostics']['templateToUff']
        rotation = np.asarray(value['rotation'], np.float64)
        translation = np.asarray(value['translation'], np.float64)
        scale = np.asarray(value['scale'], np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Source archive needs a finite templateToUff similarity') from error
    if (rotation.shape != (3, 3) or translation.shape != (3,) or scale.shape != ()
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.isfinite(scale) or scale <= 0
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-8)):
        raise ValueError('Source archive needs a finite proper templateToUff similarity')
    return rotation, translation, float(scale)


def _replaceVertexCoordinates(payload, vertices, changed):
    """Retain all original bytes except the XYZ tokens of changed v records."""
    if not changed.any():
        return payload, 0
    bom = b'\xef\xbb\xbf' if payload.startswith(b'\xef\xbb\xbf') else b''
    lines = payload.decode('utf-8-sig').splitlines(keepends=True)
    vertexIndex, replaced = 0, 0
    for lineIndex, line in enumerate(lines):
        tokens = list(re.finditer(r'\S+', line.split('#', 1)[0]))
        if not tokens or tokens[0].group() != 'v':
            continue
        if vertexIndex >= len(vertices) or len(tokens) not in (4, 5):
            raise ValueError('Source OBJ vertex records differ from the validated mesh')
        if changed[vertexIndex]:
            # Reverse token replacement preserves whitespace, optional w=1,
            # comments and the source line-ending convention, including CRLF.
            for coordinate in (2, 1, 0):
                token = tokens[coordinate + 1]
                line = (line[:token.start()] + format(vertices[vertexIndex, coordinate], '.17g')
                        + line[token.end():])
            lines[lineIndex] = line
            replaced += 1
        vertexIndex += 1
    if vertexIndex != len(vertices) or replaced != int(changed.sum()):
        raise ValueError('Source OBJ vertex count changed during geometry export')
    return bom + ''.join(lines).encode('utf-8'), replaced


def exportLegacyGeometryVariant(sourceArchive, canonicalFullVertices, outputDirectory, *, diagnostics):
    """Write and verify a fixed-texture legacy ZIP in a new/empty directory.

    ``canonicalFullVertices`` is finite Nx3 float/integer data, in the complete
    source OBJ vertex order. Its UFF frame is defined by the source manifest:
    ``canonical = (objMeters * 100) @ rotation * scale + translation``.
    Unchanged means exact equality to that computed canonical source array;
    those vertex lines and their parsed meter coordinates remain exact.

    All PNGs, MTL, OBJ non-coordinate tokens and binding metadata are retained.
    The returned report includes archive identity, changed-vertex count and
    post-load checks. Safety/likeness checks belong to the caller, not export.
    """
    source = Path(sourceArchive)
    asset = _load(source)
    proposed = np.asarray(canonicalFullVertices)
    if (proposed.shape != asset.mesh['vertices'].shape or proposed.dtype.kind not in 'fiu'
            or not np.isfinite(proposed).all()):
        raise ValueError('Expected finite Nx3 canonicalFullVertices in the complete source vertex order')
    proposed = proposed.astype(np.float64, copy=True)
    if not isinstance(diagnostics, dict):
        raise ValueError('Geometry variant diagnostics must be a dictionary')
    rotation, translation, scale = _similarity(asset.manifest)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        baseline = asset.mesh['vertices'] * 100 @ rotation * scale + translation
        changed = np.any(proposed != baseline, axis=1)
        meters = asset.mesh['vertices'].copy()
        # Solve using the actual stored matrix rather than assuming its rounded
        # transpose is a perfectly exact inverse.
        meters[changed] = np.linalg.solve(rotation.T, ((proposed[changed] - translation) / scale).T).T * .01
    if (not np.isfinite(baseline).all() or not np.isfinite(meters).all()
            or np.abs(meters).max() > 10_000):
        raise ValueError('Canonical geometry exceeds the finite legacy meter range')
    obj, count = _replaceVertexCoordinates(asset.payloads['face_nicp.obj'], meters, changed)
    payloads = {**asset.payloads, 'face_nicp.obj': obj}
    sourceHash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = copy.deepcopy(asset.manifest)
    manifest['files']['face_nicp.obj'] = {'sha256': hashlib.sha256(obj).hexdigest(), 'bytes': len(obj)}
    manifest['assemblyDiagnostics']['localGeometryVariant'] = {
        'sourceArchiveSha256': sourceHash, 'changedVertexCount': count,
        'coordinateFrame': 'source assemblyDiagnostics.templateToUff canonical coordinates',
        'texturePayloadsUnchanged': True, 'bindingMetadataUnchanged': True,
        'diagnostics': copy.deepcopy(diagnostics)}
    manifestBytes = json.dumps(manifest, ensure_ascii=False, indent=2,
                               default=jsonValue, allow_nan=False).encode('utf-8')
    output = Path(outputDirectory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Geometry variant output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    (output / 'legacyManifest.json').write_bytes(manifestBytes)
    archivePath = output / 'legacyAsset.zip'
    with zipfile.ZipFile(archivePath, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*payloads, 'legacyManifest.json']:
            archive.write(output / name, arcname=name)
    verified = _load(archivePath)
    protected = {name: payload == verified.payloads[name] for name, payload in asset.payloads.items()
                 if name != 'face_nicp.obj'}
    topology = {name: bool(np.array_equal(asset.mesh[name], verified.mesh[name]))
                for name in ('faces', 'uvs', 'uvFaces', 'materialIds')}
    unchangedVertices = bool(np.array_equal(asset.mesh['vertices'][~changed], verified.mesh['vertices'][~changed]))
    binding = verified.manifest['binding'] == asset.manifest['binding']
    metadata = all(verified.manifest[name] == asset.manifest[name]
                   for name in ('arrayIdentities', 'counts', 'coordinates', 'uv', 'materials'))
    with np.errstate(over='ignore', invalid='ignore'):
        actual = verified.mesh['vertices'] * 100 @ rotation * scale + translation
    if not np.isfinite(actual).all():
        raise ValueError('Geometry variant produced a nonfinite canonical coordinate round trip')
    maximumError = float(np.abs(actual - proposed).max(initial=0))
    tolerance = max(1e-12, 64 * np.finfo(np.float64).eps
                    * max(1., float(np.abs(proposed).max()), float(np.abs(translation).max())))
    if (not all(protected.values()) or not all(topology.values()) or not unchangedVertices
            or not binding or not metadata or maximumError > tolerance):
        raise ValueError('Geometry variant failed its protected-payload or coordinate round-trip verification')
    report = {'archivePath': str(archivePath.resolve()), **_fileIdentity(archivePath),
        'sourceArchiveSha256': sourceHash, 'changedVertexCount': count,
        'protectedPayloadsIdentical': protected, 'topologyUvMaterialArraysIdentical': topology,
        'unchangedVerticesExactlyPreserved': unchangedVertices, 'bindingMetadataIdentical': binding,
        'templateArrayIdentitiesAndMaterialMetadataIdentical': metadata,
        'maximumCanonicalRoundTripError': maximumError, 'canonicalRoundTripTolerance': tolerance}
    (output / 'archive.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
