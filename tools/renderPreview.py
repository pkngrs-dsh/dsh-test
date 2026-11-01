"""CPU front-view inspection of the exported geometry and native UV texture.

This is an orthographic diagnostic, not a PBR rendering or a likeness metric.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from faceAppearance.texture import samplePixels, trianglePixels


def renderPreview(resultDir: Path, texturePath: Path | None = None, outputPath: Path | None = None):
    geometry = dict(np.load(resultDir / "geometry.npz"))
    vertices, faces, uvs = geometry["vertices"], geometry["faces"], geometry["uvs"]
    texture = np.asarray(Image.open(texturePath or resultDir / "albedoPreview.png").convert("RGB")) / 255.0
    size = 512
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    extent = np.ptp(vertices[:, :2], axis=0).max()
    positions = (vertices - center) * (420 / extent)
    pixels = positions[:, :2] * [1, -1] + [255.5, 255.5]
    depth = np.full((size, size), -np.inf)
    colorImage = np.full((size, size, 3), 0.15)
    shapeImage = colorImage.copy()
    light = np.array([-0.35, 0.4, 1]);light /= np.linalg.norm(light)
    for face in faces:
        sample = trianglePixels(pixels[face], size, size)
        if sample is None:
            continue
        yy, xx, barycentrics = sample
        depths = barycentrics @ positions[face, 2]
        visible = depths > depth[yy, xx]
        yy, xx, barycentrics = yy[visible], xx[visible], barycentrics[visible]
        depth[yy, xx] = depths[visible]
        sampledUv = barycentrics @ uvs[face]
        colorImage[yy, xx] = samplePixels(texture, sampledUv * (texture.shape[0] - 1))
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= max(np.linalg.norm(normal), 1e-10)
        diffuse = 0.18 + 0.75 * max(float(normal @ light), 0)
        shapeImage[yy, xx] = diffuse
    panels = [Image.open(resultDir / "input00.png").convert("RGB"),
              Image.fromarray(np.rint(np.clip(colorImage, 0, 1) * 255).astype(np.uint8)),
              Image.fromarray(np.rint(np.clip(shapeImage, 0, 1) * 255).astype(np.uint8))]
    canvas = Image.new("RGB", (size * 3, size + 32), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, ("Input view", "Canonical face / unlit texture", "Geometry / diagnostic light"))):
        canvas.paste(panel, (index * size, 32))
        draw.text((index * size + 10, 9), label, fill=(245, 245, 245))
    canvas.save(outputPath or resultDir / "inspection.png")
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resultDir", type=Path)
    renderPreview(parser.parse_args().resultDir)
