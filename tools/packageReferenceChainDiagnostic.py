"""Package a gated local geometry chain with exact production RGBA pixels.

This ZIP is a diagnostic base for subsequent rebaking, never a final appearance
result. Original image samples (including alpha) remain unchanged. Assembly
uses the new surface and its saved pre-jaw accessory placement reference.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case, chain):
    reportPath = chain / 'report.json'
    report = json.loads(reportPath.read_text())
    if (report['case'] != case.name or not report['finalSafety']['accepted']
            or report['finalSafety']['newIntersectionPairsAgainstOriginalDirect'] != 0):
        raise ValueError('Require the accepted cumulative chain for this exact case')
    geometry = load(chain / 'finalGeometry.npz')
    geometry['uvConvention'] = 'topLeftPixelEdges'
    transform = load(chain / 'templateToUff.npz')
    transform['scale'] = float(transform['scale'])
    reference = load(chain / 'accessoryReference.npz')['vertices']
    mouth = load(chain / 'mouthState.npz')
    selectedIndex = report['referenceSelection']['referenceViewIndex']
    if int(mouth['referenceViewIndex']) != selectedIndex:
        raise ValueError('Final mouth state differs from shared reference')
    originalZip = case / 'asset/legacyAsset.zip'
    original = _load(originalZip)
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates' / original.manifest['templateId'])
    if assets.manifest['templateHashes'] != original.manifest['templateHashes']:
        raise ValueError('Template identity differs from delivered production ZIP')
    texture = load(case / 'reconstruction/directClient/texture.npz')
    provenance = {key: value for key, value in texture.items() if key != 'linearAlbedo'}
    assembly = assets.assembleFinalClientSurface(geometry, transform, texture['linearAlbedo'], provenance,
        diagnostics={'diagnosticOnly': True, 'appearanceRebaked': False,
            'referenceSelection': report['referenceSelection'], 'finalChainSafety': report['finalSafety'],
            'geometrySource': str(chain / 'finalGeometry.npz'),
            'limitation': 'Original production textures on changed geometry; pending SH rebake before evaluation or delivery'},
        accessoryReferenceCanonical=reference)
    imageNames = ('head_color.png', 'body_color.png', 'eye_color.png', 'brow_color.png')
    sourcePixels = {}
    with zipfile.ZipFile(originalZip) as archive:
        for material, name in zip(assembly['materials'], imageNames):
            with Image.open(io.BytesIO(archive.read(name))) as image:
                rgba = np.array(image.convert('RGBA'))
            sourcePixels[name] = rgba
            material['baseColor'] = decodeColor(rgba.astype(np.float32) / 255., 'srgb')
        normalPath = chain / 'diagnosticSourceNormals.png'
        normalPath.write_bytes(archive.read('head_normals.png'))
    exported = exportLegacyAsset(chain / 'diagnosticAsset', assembly,
        gender=original.manifest['gender'], normalMapPath=normalPath)
    outputArchive = Path(exported['archivePath'])
    checks = {}
    with zipfile.ZipFile(outputArchive) as archive:
        for name, pixels in sourcePixels.items():
            with Image.open(io.BytesIO(archive.read(name))) as image:
                current = np.array(image.convert('RGBA'))
            checks[name] = {'rgbaPixelsExactlyPreserved': bool(np.array_equal(current, pixels)),
                           'alphaExactlyPreserved': bool(np.array_equal(current[..., 3], pixels[..., 3]))}
            if not checks[name]['rgbaPixelsExactlyPreserved']:
                raise ValueError('Diagnostic package changed original material pixels')
    packaged = _load(outputArchive)
    skin = packaged.mesh['vertices'][assets.arrays['nicpToFullIds']] * 100
    canonical = skin @ transform['rotation'] * transform['scale'] + transform['translation']
    maximumError = float(np.abs(canonical - geometry['vertices']).max())
    if not np.allclose(canonical, geometry['vertices'], rtol=0, atol=2e-7):
        raise ValueError('Diagnostic ZIP skin differs from final canonical geometry')
    report['referenceViewIndex'] = selectedIndex
    report['selection'] = report['referenceSelection']
    report['finalChainSafety'] = report['finalSafety']
    report['exported'] = {**exported, 'sha256': exported['archiveSha256'],
        'diagnosticOnly': True, 'appearanceRebaked': False,
        'originalZipSha256': digest(originalZip), 'materialChecks': checks,
        'maximumSkinCanonicalExportError': maximumError,
        'fullVertexCount': len(packaged.mesh['vertices']), 'fullFaceCount': len(packaged.mesh['faces'])}
    report['outputHashes'] = {name: digest(chain / name) for name in
        ('finalGeometry.npz', 'templateToUff.npz', 'accessoryReference.npz', 'mouthState.npz')}
    report['limitations'] = [value for value in report['limitations']
                            if value != 'No texture baking or ZIP export; final surface requires rebaking before delivery.']
    report['limitations'].append('Diagnostic ZIP preserves production RGBA; final surface requires texture rebaking before delivery.')
    report['packagingImplementationSha256'] = digest(Path(__file__))
    writeJson(reportPath, report)
    print(json.dumps({'archivePath': str(outputArchive), 'sha256': exported['archiveSha256'],
                     'allRgbaPixelsPreserved': True, 'maximumSkinCanonicalExportError': maximumError}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--chain', type=Path, required=True)
    args = parser.parse_args()
    run(args.case.resolve(), args.chain.resolve())
