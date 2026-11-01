import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
import pytest

from faceAppearance.data import readImage
from faceAppearance.faceCrop import cropFaceSquare
from faceAppearance.imageColor import readSrgbImage


# Locally generated LittleCMS matrix/TRC fixture, not a user's photograph:
# D65 white (.3127, .3290), P3 primaries (.680,.320)/(.265,.690)/(.150,.060),
# and the standard sRGB transfer curve. No test-time native library is needed.
displayP3Profile = (Path(__file__).parent / "fixtures/displayP3.icc").read_bytes()


def displayP3ToSrgbReference(rgb):
    """Independent D65 colorimetry reference, before uint8 quantization."""
    encoded = np.asarray(rgb, np.float64) / 255
    linear = np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)
    p3ToXyz = np.array([[.4865709486, .2656676932, .1982172852],
                       [.2289745641, .6917385218, .0792869141],
                       [0., .0451133819, 1.0439443689]])
    srgbToXyz = np.array([[.4123907993, .3575843394, .1804807884],
                         [.2126390059, .7151686788, .0721923154],
                         [.0193308187, .1191947798, .9505321522]])
    converted = np.clip(linear @ p3ToXyz.T @ np.linalg.inv(srgbToXyz).T, 0, 1)
    output = np.where(converted <= .0031308, 12.92 * converted,
                      1.055 * converted ** (1 / 2.4) - .055)
    return np.rint(output * 255).astype(np.uint8)


def testDisplayP3EmbeddedProfileConvertsActualPixelsAndRecordsProvenance(tmp_path):
    rgb = np.array([[[184, 145, 122], [80, 180, 120], [220, 80, 150],
                     [32, 32, 32], [0, 0, 0], [255, 255, 255]]], np.uint8)
    path = tmp_path / "p3.png"
    Image.fromarray(rgb).save(path, icc_profile=displayP3Profile)
    converted, metadata = readSrgbImage(path)
    actual = np.asarray(converted)
    np.testing.assert_allclose(actual, displayP3ToSrgbReference(rgb), atol=1)
    assert np.max(np.abs(actual.astype(int) - rgb)) > 10
    assert converted.mode == "RGB"
    assert metadata["conversion"] == "iccToSrgb"
    assert metadata["sourceIccPresent"]
    assert metadata["sourceIccSha256"] == hashlib.sha256(displayP3Profile).hexdigest()
    assert "Display P3" in metadata["sourceIccDescription"]
    assert metadata["outputColorSpace"] == "sRGB"
    assert metadata["renderingIntent"] == "relativeColorimetric"
    # The detached result carries its new profile, so re-opening it cannot
    # accidentally interpret corrected pixels as Display P3 a second time.
    saved = tmp_path / "corrected.png"
    converted.save(saved)
    replay, _ = readSrgbImage(saved)
    np.testing.assert_array_equal(replay, actual)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def testNoProfilePreservesExistingRgbAndExifBehavior(tmp_path, mode):
    rgb = np.zeros((80, 120, 3), np.uint8)
    rgb[:30, :50] = [170, 90, 50]
    source = Image.fromarray(rgb).convert(mode)
    exif = source.getexif(); exif[274] = 6
    path = tmp_path / "unprofiled.png"
    source.save(path, exif=exif)
    with Image.open(path) as opened:
        expected = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    actual, metadata = readSrgbImage(path)
    np.testing.assert_array_equal(actual, expected)
    assert actual.size == (80, 120)
    assert actual.getexif().get(274) is None
    assert metadata["conversion"] == "assumeSrgb"
    assert not metadata["sourceIccPresent"]
    assert metadata["sourceIccSha256"] is None


def testEmbeddedSrgbIsNotDoubleGammaConverted():
    rgb = np.arange(256, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
    stream = io.BytesIO()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(rgb).save(stream, format="PNG", icc_profile=profile)
    stream.seek(0)
    result, metadata = readSrgbImage(stream)
    np.testing.assert_array_equal(result, rgb)
    assert metadata["conversion"] == "iccToSrgb"


@pytest.mark.parametrize("profile", [b"invalid ICC bytes",
    ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()])
def testMalformedOrIncompatibleEmbeddedProfileIsRejected(tmp_path, profile):
    path = tmp_path / "invalid.png"
    Image.new("RGB", (80, 80), (170, 90, 50)).save(path, icc_profile=profile)
    with pytest.raises(ValueError, match="Embedded ICC color profile"):
        readSrgbImage(path)
    with pytest.raises(ValueError, match="Embedded ICC color profile"):
        readImage(path)


def testEmptyEmbeddedProfileDecoderStateIsRejected(monkeypatch):
    opened = Image.new("RGB", (80, 80))
    opened.info["icc_profile"] = b""
    monkeypatch.setattr("faceAppearance.imageColor.Image.open", lambda _: opened)
    with pytest.raises(ValueError, match="empty or invalid"):
        readSrgbImage("unused")


def testIngestionConvertsBeforeDetectionWithSameExifAndCropCoordinates(tmp_path):
    source = Image.new("RGB", (80, 120), (184, 145, 122))
    source.paste((80, 180, 120), (0, 0, 30, 40))
    exif = source.getexif(); exif[274] = 6
    path = tmp_path / "orientedP3.png"
    source.save(path, exif=exif, icc_profile=displayP3Profile)
    with Image.open(path) as opened:
        expected = ImageCms.profileToProfile(
            ImageOps.exif_transpose(opened), ImageCms.ImageCmsProfile(io.BytesIO(displayP3Profile)),
            ImageCms.createProfile("sRGB"), renderingIntent=1, outputMode="RGB")
    face = {"boxXyxy": [20, 10, 90, 70], "landmarks": [[30, 25], [70, 25]], "score": .95}
    observed = []

    def detector(rgb):
        np.testing.assert_array_equal(rgb, expected)
        observed.append(rgb.shape)
        return face

    actual, record = readImage(path, imageSize=64, cropPolicy="faceSquare", faceDetector=detector)
    cropped, crop = cropFaceSquare(np.asarray(expected), face)
    expectedWorking = np.asarray(Image.fromarray(cropped).resize((64, 64), Image.Resampling.LANCZOS))
    np.testing.assert_array_equal(np.rint(actual * 255).astype(np.uint8), expectedWorking)
    assert observed == [(80, 120, 3)]
    assert record["cropBoxXyxy"] == crop["cropBoxXyxy"]
    assert record["orientedSize"] == [120, 80]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["colorManagement"]["conversion"] == "iccToSrgb"


def testColmapPathConvertsButModelNativeArrayDescriptorDoesNot(tmp_path):
    from faceAppearance.colmapTexture import _image

    native = np.full((4, 5, 3), [184, 145, 122], np.uint8)
    path = tmp_path / "p3.png"
    Image.fromarray(native).save(path, icc_profile=displayP3Profile)
    corrected, metadata = _image(path)
    np.testing.assert_allclose(corrected, displayP3ToSrgbReference(native), atol=1)
    assert metadata["colorManagement"]["conversion"] == "iccToSrgb"
    assert metadata["sourceSha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # Production supplies OD-native colors/markers as arrays and only includes
    # a source path for provenance. Its ICC must not reinterpret those arrays.
    untouched, metadata = _image({"image": native, "path": path})
    np.testing.assert_array_equal(untouched, native)
    assert "colorManagement" not in metadata


def testOriginalLinearSamplingAppliesIccBeforeSrgbDecoding(tmp_path):
    from faceAppearance.appearanceObservations import readOriginalLinear, srgbToLinear

    path = tmp_path / "p3.png"
    pixels = np.full((4, 5, 3), [184, 145, 122], np.uint8)
    Image.fromarray(pixels).save(path, icc_profile=displayP3Profile)
    metadata = {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "orientedSize": [5, 4]}
    actual = readOriginalLinear(metadata)
    expected = srgbToLinear(displayP3ToSrgbReference(pixels).astype(np.float32) / 255)
    np.testing.assert_allclose(actual, expected, atol=.007)
    assert not np.allclose(actual, srgbToLinear(pixels.astype(np.float32) / 255))
