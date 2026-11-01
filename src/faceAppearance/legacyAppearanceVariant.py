"""Replace only the head albedo of a verified legacy archive.

Useful for appearance experiments: original OBJ/MTL, binding metadata and every
other material payload remain byte-identical. This does not reassemble a mesh.
"""
from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .clientAsset import _writeBaseColor, _fileIdentity
from .legacyPreviewAsset import _load


def exportLegacyAppearanceVariant(sourceArchive, linearHeadColor, outputDirectory, *, diagnostics):
    source = Path(sourceArchive)
    asset = _load(source)
    colors = np.asarray(linearHeadColor)
    with Image.open(BytesIO(asset.payloads['head_color.png'])) as image:
        expectedShape = (image.height, image.width, 3)
        sourceAlpha = (image.convert('RGBA').getchannel('A').copy()
                       if 'A' in image.getbands() or 'transparency' in image.info else None)
    if (colors.shape != expectedShape or colors.dtype.kind != 'f'
            or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any()):
        raise ValueError('Expected finite linear RGB matching the original head atlas')
    output = Path(outputDirectory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Appearance variant output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    _writeBaseColor(output / 'head_color.png', colors)
    if sourceAlpha is not None:
        with Image.open(output / 'head_color.png') as encoded:
            rgba = encoded.convert('RGBA')
        rgba.putalpha(sourceAlpha)
        rgba.save(output / 'head_color.png')
    newHead = (output / 'head_color.png').read_bytes()
    payloads = {**asset.payloads, 'head_color.png': newHead, 'face_nicp.png': newHead}
    manifest = copy.deepcopy(asset.manifest)
    for name, content in payloads.items():
        (output / name).write_bytes(content)
        manifest['files'][name] = _fileIdentity(output / name)
    manifest['assemblyDiagnostics']['localAppearanceVariant'] = {
        'sourceArchiveSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        **diagnostics, 'geometryPayloadUnchanged': True, 'otherMaterialPayloadsUnchanged': True,
        'headAlphaPreserved': sourceAlpha is not None}
    (output / 'legacyManifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    archivePath = output / 'legacyAsset.zip'
    with zipfile.ZipFile(archivePath, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*payloads, 'legacyManifest.json']:
            archive.write(output / name, arcname=name)
    verified = _load(archivePath)
    unchanged = {name: content == verified.payloads[name] for name, content in asset.payloads.items()
                 if name not in ('head_color.png', 'face_nicp.png')}
    if not all(unchanged.values()):
        raise ValueError('Appearance variant changed a protected source payload')
    with Image.open(BytesIO(verified.payloads['head_color.png'])) as encoded:
        alphaIdentical = (np.array_equal(np.asarray(encoded.getchannel('A')), np.asarray(sourceAlpha))
                          if sourceAlpha is not None else 'A' not in encoded.getbands())
    if not alphaIdentical:
        raise ValueError('Appearance variant changed the source head alpha channel')
    report = {'archivePath': str(archivePath.resolve()), **_fileIdentity(archivePath),
              'protectedPayloadsIdentical': unchanged,
              'headAlphaIdentical': bool(alphaIdentical),
              'headColorAliasIdentical': verified.payloads['head_color.png'] == verified.payloads['face_nicp.png']}
    (output / 'archive.json').write_text(json.dumps(report, indent=2))
    return report
