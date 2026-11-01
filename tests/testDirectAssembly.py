"""Direct surface registration keeps the existing client export contract."""
import hashlib
import json

import numpy as np
import pytest

import faceAppearance.headAssembly as module
from faceAppearance.headAssembly import AssemblyAssets
from testHeadAssembly import bundle, reconstructedGeometry, observedTexture


@pytest.fixture
def directBundle(bundle):
    root, arrays = bundle
    faceIds = np.repeat(np.arange(len(arrays['nicpFaces'])), 2).astype(np.int64)
    bary = np.tile([[.6, .25, .15], [.2, .25, .55]], (len(arrays['nicpFaces']), 1))
    np.savez_compressed(root/'direct.npz', sampleNicpFaceIds=faceIds, sampleNicpWeights=bary,
                        sourceTriangleIds=faceIds, sourceBarycentricWeights=bary,
                        sampleWeights=np.ones(len(faceIds)))
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['files']['direct.npz'] = hashlib.sha256((root/'direct.npz').read_bytes()).hexdigest()
    manifest['geometryTransfer'] = {'method': 'directSurface', 'asset': 'direct.npz',
        'alignmentMode': 'directSurfaceSamples', 'restWeight': 0., 'smoothWeight': 30.,
        'staticConstraint': 'hard', 'releaseStatus': 'localExperiment'}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root, arrays


def testDirectAssemblyDoesNotUseUvAlignmentOrVertexGuard(directBundle, monkeypatch):
    root, arrays = directBundle
    assets = AssemblyAssets(root)
    def forbidden(*args, **kwargs):
        raise AssertionError('UV alignment/guard called by direct surface registration')
    monkeypatch.setattr(module, 'fitSimilarity', forbidden)
    monkeypatch.setattr(module, 'guardDisplacement', forbidden)
    output = assets.assemble(reconstructedGeometry(arrays), observedTexture())
    for key in ('faces', 'uvs', 'uvFaces', 'materialIds'):
        np.testing.assert_array_equal(output['geometry'][key], arrays[key])
    protected = arrays['nicpToFullIds'][arrays['staticMask']]
    np.testing.assert_array_equal(output['geometry']['vertices'][protected], arrays['vertices'][protected])
    assert output['diagnostics']['geometryTransfer']['postSolveVertexGuardApplied'] is False
    assert output['diagnostics']['geometryTransfer']['alignmentMode'] == 'directSurfaceSamples'
    assert output['diagnostics']['geometryTransfer']['solve']['weights']['rest'] == 0
    assert output['geometry']['vertices'].shape == arrays['vertices'].shape


def testDirectCorrespondenceIsPreloadedAndHashChecked(directBundle):
    root, arrays = directBundle
    assets = AssemblyAssets(root)
    assert all(not value.flags.writeable for value in assets.directArrays.values())
    (root/'direct.npz').unlink()
    assets._deform(reconstructedGeometry(arrays))
    with pytest.raises(ValueError, match='Invalid template asset'):
        AssemblyAssets(root)


def testDirectAlignmentRequiresExplicitMode(directBundle):
    root, _ = directBundle
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['geometryTransfer']['alignmentMode'] = 'sharedUvVertices'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='surface-sample alignment'):
        AssemblyAssets(root)


def testSoftBoundaryDriftIsReported(directBundle):
    root, arrays = directBundle
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['geometryTransfer']['staticConstraint'] = 'soft'
    (root/'manifest.json').write_text(json.dumps(manifest))
    _, _, diagnostics = AssemblyAssets(root)._deform(reconstructedGeometry(arrays))
    assert diagnostics['geometryTransfer']['staticDriftMaxCm'] > 0
    assert diagnostics['geometryTransfer']['releaseStatus'] == 'localExperiment'
