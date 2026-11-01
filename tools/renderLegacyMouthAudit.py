"""Fixed orthographic geometry-only comparison of FLAME and client transfer."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(vertices, faces, bounds, size=700):
    xmin, xmax, ymin, ymax = bounds
    width, height = size, round(size * (ymax - ymin) / (xmax - xmin))
    uv = np.column_stack(((vertices[:, 0] - xmin) / (xmax - xmin) * (width - 1),
                          (ymax - vertices[:, 1]) / (ymax - ymin) * (height - 1)))
    normals = np.zeros_like(vertices)
    points = vertices[faces]
    faceNormals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], faceNormals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-.25, .4, 1.]); light /= np.linalg.norm(light)
    luminance = .3 + .65 * np.maximum(normals @ light, 0)
    image = np.full((height, width, 3), [28, 35, 39], dtype=np.uint8)
    depth = np.full((height, width), -np.inf)
    for face in faces:
        tri = uv[face]
        lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(tri.max(0)).astype(int), [width - 1, height - 1])
        if np.any(lo > hi):
            continue
        a, b, c = tri
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        wc = 1 - wa - wb
        z = wa * vertices[face[0], 2] + wb * vertices[face[1], 2] + wc * vertices[face[2], 2]
        target = depth[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        valid = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6) & (z > target)
        target[valid] = z[valid]
        shade = wa * luminance[face[0]] + wb * luminance[face[1]] + wc * luminance[face[2]]
        colors = np.clip(shade[..., None] * [208, 218, 222], 0, 255).astype(np.uint8)
        image[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1][valid] = colors[valid]
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('reports/legacyWebPreview20261012/mouthAudit'))
    args = parser.parse_args()
    with np.load(args.root / 'transferStages.npz') as data:
        stages = {key: data[key] for key in data.files}
    labels = [('Raw FLAME / similarity alignment only', 'source', 'sourceFaces'),
              ('Male template / before transfer', 'rest', 'faces'),
              ('Delivered client / after transfer', 'assembled', 'faces')]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    for name, bounds in [('face', (-8, 8, 156, 176)), ('mouth', (-4, 4, 159, 165))]:
        panels = [render(stages[key], stages[faceKey], bounds) for _, key, faceKey in labels]
        width, height = panels[0].size
        result = Image.new('RGB', (width * 3, height + 82), (28, 35, 39))
        draw = ImageDraw.Draw(result)
        draw.text((18, 10), 'Same orthographic camera and light; no texture; coordinates in template centimeters', fill='white', font=font)
        for index, (panel, (label, _, _)) in enumerate(zip(panels, labels)):
            result.paste(panel, (index * width, 82))
            draw.text((index * width + 18, 48), label, fill='white', font=font)
        result.save(args.root / f'{name}Comparison.png')
    print('Rendered faceComparison.png and mouthComparison.png')


if __name__ == '__main__':
    main()
