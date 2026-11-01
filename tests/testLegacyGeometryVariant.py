import hashlib
from pathlib import Path
import numpy as np
import pytest

from faceAppearance.legacyGeometryVariant import exportLegacyGeometryVariant
from faceAppearance.legacyPreviewAsset import _load
from testLegacyPreviewAsset import package, rewrite


def sourceWithSimilarity(package):
    source, _ = package
    angle = .37
    transform = {'rotation': [[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]],
                 'translation': [.15, -14.4, .03], 'scale': .086}
    rewrite(source, manifestChange=lambda manifest: manifest['assemblyDiagnostics'].update(templateToUff=transform))
    asset = _load(source)
    canonical = asset.mesh['vertices'] * 100 @ np.asarray(transform['rotation']) * transform['scale'] + transform['translation']
    return Path(source), asset, canonical, transform


def testGeometryVariantChangesOnlySelectedVertexTokensAndRetainsAllTextures(package, tmp_path):
    source, asset, canonical, transform = sourceWithSimilarity(package)
    # A valid OBJ may use indentation, tabs, w=1, comments, CRLF and a BOM.
    lines = asset.payloads['face_nicp.obj'].decode().splitlines(keepends=True)
    vertexLine = next(index for index, line in enumerate(lines) if line.startswith('v '))
    lines[vertexLine] = '\t v\t0  0\t0 1 # keep this binding annotation\r\n'
    decorated = b'\xef\xbb\xbf' + ''.join(lines).encode()
    rewrite(source, changes={'face_nicp.obj': decorated})
    asset = _load(source)
    sourceHash = hashlib.sha256(source.read_bytes()).hexdigest()
    target = canonical.copy()
    target[0] += [.003, -.001, .002]
    target[2] += [-.001, .002, 0.]
    report = exportLegacyGeometryVariant(source, target, tmp_path/'variant', diagnostics={'purpose': 'geometry-only test'})
    result = _load(report['archivePath'])
    assert report['changedVertexCount'] == 2
    assert all(report['protectedPayloadsIdentical'].values())
    assert all(report['topologyUvMaterialArraysIdentical'].values())
    assert report['unchangedVerticesExactlyPreserved']
    assert report['bindingMetadataIdentical']
    assert result.manifest['binding'] == asset.manifest['binding']
    assert result.manifest['arrayIdentities'] == asset.manifest['arrayIdentities']
    for name, payload in asset.payloads.items():
        if name != 'face_nicp.obj':
            assert result.payloads[name] == payload
    beforeLines = decorated.decode('utf-8-sig').splitlines(keepends=True)
    afterLines = result.payloads['face_nicp.obj'].decode('utf-8-sig').splitlines(keepends=True)
    indices = [index for index, line in enumerate(beforeLines) if line.split()[0] == 'v']
    assert result.payloads['face_nicp.obj'].startswith(b'\xef\xbb\xbf')
    for index, line in enumerate(beforeLines):
        if index not in (indices[0], indices[2]):
            assert afterLines[index] == line
    assert afterLines[indices[0]].startswith('\t v\t')
    assert afterLines[indices[0]].endswith(' 1 # keep this binding annotation\r\n')
    np.testing.assert_array_equal(result.mesh['vertices'][[1, 3]], asset.mesh['vertices'][[1, 3]])
    actual = result.mesh['vertices'] * 100 @ np.asarray(transform['rotation']) * transform['scale'] + transform['translation']
    np.testing.assert_allclose(actual, target, rtol=0, atol=1e-12)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sourceHash


def testGeometryVariantNoopPreservesObjBytes(package, tmp_path):
    source, asset, canonical, _ = sourceWithSimilarity(package)
    report = exportLegacyGeometryVariant(source, canonical, tmp_path/'noop', diagnostics={})
    assert report['changedVertexCount'] == 0
    result = _load(report['archivePath'])
    assert result.payloads == asset.payloads
    assert report['maximumCanonicalRoundTripError'] == 0


@pytest.mark.parametrize('kind', ['shape', 'nan', 'inf', 'complex', 'bool', 'meterRange'])
def testGeometryVariantRejectsInvalidVerticesBeforeWriting(package, tmp_path, kind):
    source, _, canonical, _ = sourceWithSimilarity(package)
    if kind == 'shape':
        canonical = canonical[:-1]
    elif kind == 'nan':
        canonical[0, 0] = np.nan
    elif kind == 'inf':
        canonical[0, 0] = np.inf
    elif kind == 'complex':
        canonical = canonical.astype(np.complex128)
    elif kind == 'bool':
        canonical = canonical.astype(bool)
    else:
        canonical[0, 0] = 1e9
    with pytest.raises(ValueError, match='finite'):
        exportLegacyGeometryVariant(source, canonical, tmp_path/'invalid', diagnostics={})
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('kind', ['missing', 'zeroScale', 'reflection', 'translationShape'])
def testGeometryVariantRejectsMalformedSourceSimilarity(package, tmp_path, kind):
    source, _, canonical, _ = sourceWithSimilarity(package)
    def damage(manifest):
        transform = manifest['assemblyDiagnostics']['templateToUff']
        if kind == 'missing':
            del manifest['assemblyDiagnostics']['templateToUff']
        elif kind == 'zeroScale':
            transform['scale'] = 0.
        elif kind == 'reflection':
            transform['rotation'] = np.diag([-1., 1., 1.]).tolist()
        else:
            transform['translation'] = [0., 0.]
    rewrite(source, manifestChange=damage)
    with pytest.raises(ValueError, match='templateToUff'):
        exportLegacyGeometryVariant(source, canonical, tmp_path/'invalid', diagnostics={})
    assert not (tmp_path/'invalid').exists()


def testGeometryVariantRefusesToOverwriteExistingOutput(package, tmp_path):
    source, _, canonical, _ = sourceWithSimilarity(package)
    output = tmp_path/'occupied'
    output.mkdir()
    sentinel = output/'keep.bin'
    sentinel.write_bytes(b'preserve existing experiment')
    with pytest.raises(FileExistsError):
        exportLegacyGeometryVariant(source, canonical, output, diagnostics={})
    assert sentinel.read_bytes() == b'preserve existing experiment'
    assert list(output.iterdir()) == [sentinel]
