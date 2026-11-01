"""Build four complete-head comparisons from the same deployed reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/legacyPreview/preview-fb21ed26fc4b41969d581365622054e5/asset.zip')
    parser.add_argument('--texture', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/comparison')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    geometry = load(args.source/'directClient/geometry.npz')
    refined = load(args.source/'directClient/refinement/clientGeometry.npz')
    transform = {key: refined['templateToUff'+key.title()] for key in ('rotation', 'translation', 'scale')}
    old = load(args.source/'directClient/texture.npz')
    new = load(args.texture)
    newLinear = new.pop('linearRGB', None)
    if newLinear is None:
        newLinear = new.pop('linearAlbedo')
    oldLinear = old.pop('linearAlbedo')
    baseline = _load(args.baseline)
    eyeIds = np.unique(np.concatenate([c['vertexIndices'] for c in assets.components if c['label'] in {'leftEye', 'rightEye'}]))
    nonEye = np.ones(len(assets.arrays['vertices']), bool)
    nonEye[eyeIds] = False
    baseDir = args.output/'baseline'
    baseDir.mkdir(exist_ok=True)
    shutil.copyfile(args.baseline, baseDir/'legacyAsset.zip')
    report = {'baselineSha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              'scope': 'Identical deployed skin geometry, UVs and camera reconstruction; change only eyeballs and exterior texture blend.',
              'branches': {}}
    for name, linear, provenance, eyeMode in [('eyes', oldLinear, old, 'assetDefault'),
                                             ('texture', newLinear, new, 'legacyNearest'),
                                             ('combined', newLinear, new, 'assetDefault')]:
        assembly = assets.assembleFinalClientSurface(geometry, transform, linear, provenance,
            eyePlacementMode=eyeMode, diagnostics={'comparisonBranch': name, 'source': str(args.source)})
        mesh = assembly['geometry']
        # Packaging must not change the underlying skin or unrelated attachments.
        if not np.allclose(mesh['vertices'][nonEye]*.01, baseline.mesh['vertices'][nonEye], rtol=0, atol=1e-13):
            raise ValueError('Comparison changed vertices outside the eyeballs')
        for key in ('faces', 'uvs', 'uvFaces', 'materialIds'):
            if not np.allclose(mesh[key], baseline.mesh[key], rtol=0, atol=1e-15):
                raise ValueError('Comparison changed topology or UV identity: '+key)
        result = exportLegacyAsset(args.output/name, assembly, gender='male')
        packaged = _load(args.output/name/'legacyAsset.zip')
        if name=='eyes' and any(packaged.payloads[n] != baseline.payloads[n] for n in
            ('head_color.png','body_color.png','eye_color.png','brow_color.png')):
            raise ValueError('Eye-only branch changed a texture')
        if name=='texture' and not np.allclose(packaged.mesh['vertices'], baseline.mesh['vertices'], rtol=0, atol=1e-13):
            raise ValueError('Texture-only branch changed geometry')
        writeJson(args.output/name/'assemblyDiagnostics.json', assembly['diagnostics'])
        writeJson(args.output/name/'components.json', assembly['components'])
        report['branches'][name] = {'export': result,
            'eyeMode': eyeMode, 'eyeVertexCount': len(eyeIds),
            'nonEyeMaxDifferenceMeters': float(np.max(np.abs(packaged.mesh['vertices'][nonEye]-baseline.mesh['vertices'][nonEye]))),
            'sourceCoverageUnchanged': bool(np.array_equal(old['coverageMask'], provenance['coverageMask']))}
    writeJson(args.output/'report.json', report)
    print(json.dumps({'output': str(args.output), 'branches': list(report['branches'])}))


if __name__=='__main__':
    main()
