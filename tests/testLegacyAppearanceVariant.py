import io
import numpy as np
from PIL import Image
import pytest

from faceAppearance.legacyAppearanceVariant import exportLegacyAppearanceVariant
from faceAppearance.legacyPreviewAsset import _load
from testLegacyPreviewAsset import package, rewrite


def testAppearanceVariantRetainsBindingAndAllOtherPayloadBytes(package, tmp_path):
    source, _ = package
    baseline = _load(source)
    colors = np.full((2,3,3), .5, np.float32)
    report = exportLegacyAppearanceVariant(source, colors, tmp_path/'variant', diagnostics={'experiment':'test'})
    result = _load(report['archivePath'])
    assert all(report['protectedPayloadsIdentical'].values())
    assert result.manifest['binding'] == baseline.manifest['binding']
    assert result.manifest['arrayIdentities'] == baseline.manifest['arrayIdentities']
    for name in ('vertices','faces','uvs','uvFaces','materialIds'):
        np.testing.assert_array_equal(result.mesh[name], baseline.mesh[name])
    assert result.payloads['head_color.png'] == result.payloads['face_nicp.png']
    with Image.open(io.BytesIO(result.payloads['head_color.png'])) as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), np.full((2,3,3),188,np.uint8))
    assert _load(source).payloads == baseline.payloads


def testAppearanceVariantRejectsWrongAtlasOrNonFiniteColorBeforeWriting(package, tmp_path):
    source, _ = package
    for colors in [np.zeros((3,2,3)), np.full((2,3,3),np.nan), np.full((2,3,3),1.1)]:
        with pytest.raises(ValueError, match='linear RGB'):
            exportLegacyAppearanceVariant(source, colors, tmp_path/'invalid', diagnostics={})
        assert not (tmp_path/'invalid').exists()


def testAppearanceVariantPreservesNonOpaqueHeadAlphaExactly(package, tmp_path):
    source, _ = package
    rgba = np.full((2, 3, 4), 93, dtype=np.uint8)
    rgba[..., 3] = [[0, 1, 127], [128, 228, 255]]
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format='PNG')
    rewrite(source, changes={name: stream.getvalue() for name in ('head_color.png', 'face_nicp.png')})
    result = exportLegacyAppearanceVariant(source, np.full((2, 3, 3), .5, np.float32),
                                           tmp_path/'rgbaVariant', diagnostics={})
    assert result['headAlphaIdentical']
    verified = _load(result['archivePath'])
    with Image.open(io.BytesIO(verified.payloads['head_color.png'])) as image:
        assert image.mode == 'RGBA'
        np.testing.assert_array_equal(np.asarray(image)[..., :3], 188)
        np.testing.assert_array_equal(np.asarray(image)[..., 3], rgba[..., 3])
