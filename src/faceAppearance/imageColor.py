"""Decode user photographs into EXIF-oriented, color-managed sRGB pixels.

An absent profile retains the existing assumption that RGB values are sRGB.
An embedded profile must be usable: silently discarding it changes the input
colors. Model-native arrays, normal maps and template assets do not use this
photograph reader.
"""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageCms, ImageOps


def readSrgbImage(source) -> tuple[Image.Image, dict]:
    """Return a detached RGB image and color provenance from a path or stream.

    EXIF orientation is applied before callers detect, crop or resize. ICC
    conversion uses relative colorimetric intent; malformed or incompatible
    embedded profiles raise ValueError rather than falling back to raw RGB.
    """
    with Image.open(source) as opened:
        opened.load()  # Truncated image decoding remains an error.
        sourceMode = opened.mode
        profilePresent = "icc_profile" in opened.info
        profileBytes = opened.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(opened)
        metadata = {"sourceMode": sourceMode, "sourceIccPresent": profilePresent,
                    "sourceIccSha256": None, "sourceIccDescription": None,
                    "outputColorSpace": "sRGB", "conversion": "assumeSrgb",
                    "renderingIntent": None}
        if not profilePresent:
            return oriented.convert("RGB"), metadata
        if not isinstance(profileBytes, bytes) or not profileBytes:
            raise ValueError("Embedded ICC color profile is empty or invalid")
        metadata["sourceIccSha256"] = hashlib.sha256(profileBytes).hexdigest()
        try:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(profileBytes))
            # Palette and alpha channels are not model inputs. Keep CMYK/LAB/L
            # modes for their matching profiles instead of first changing the
            # color coordinates with an unmanaged RGB conversion.
            if oriented.mode in {"P", "PA", "RGBA", "RGBX"}:
                oriented = oriented.convert("RGB")
            elif oriented.mode == "LA":
                oriented = oriented.convert("L")
            image = ImageCms.profileToProfile(
                oriented, profile, ImageCms.createProfile("sRGB"),
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB", flags=0)
            metadata.update(sourceIccDescription=ImageCms.getProfileDescription(profile).strip(),
                            conversion="iccToSrgb", renderingIntent="relativeColorimetric")
        except (ImageCms.PyCMSError, OSError, TypeError, ValueError) as error:
            raise ValueError("Embedded ICC color profile is invalid or incompatible with the image") from error
        return image, metadata
