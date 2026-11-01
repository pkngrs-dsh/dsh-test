"""Create a before/after diagnostic and check saved material arrays on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from renderPreview import renderPreview


def inspectResult(resultDir: Path):
    base = renderPreview(resultDir, resultDir / "albedo.png")
    refined = renderPreview(resultDir, resultDir / "wildcapUff/baseWithWildCap.png",
                            resultDir / "wildcapInspection.png")
    width, height = 512, 544
    comparison = Image.new("RGB", (width * 3, height), (25, 25, 25))
    panels = [base.crop((0, 32, width, height)),
              base.crop((width, 32, width * 2, height)),
              refined.crop((width, 32, width * 2, height))]
    draw = ImageDraw.Draw(comparison)
    for index, (panel, label) in enumerate(zip(panels, (
            "Input view", "Base delighted / unlit", "WildCap diffuse / unlit (experimental)"))):
        comparison.paste(panel, (index * width, 32))
        draw.text((index * width + 8, 9), label, fill=(245, 245, 245))
    comparison.save(resultDir / "materialComparison.png")
    with np.load(resultDir / "wildcapIct/materials.npz", allow_pickle=False) as saved:
        ict = {name: saved[name] for name in saved.files}
    with np.load(resultDir / "wildcapUff/materials.npz", allow_pickle=False) as saved:
        uff = {name: saved[name] for name in saved.files}
    assert all(np.isfinite(value).all() for value in [*ict.values(), *uff.values()])
    assert ict["validMask"].any() and uff["validMask"].any()
    lengths = np.linalg.norm(ict["tangentNormal"][ict["generatedMask"]], axis=-1)
    assert np.allclose(lengths, 1, atol=1e-5)
    result = json.loads((resultDir / "result.json").read_text())
    report = {"status": "finiteAndNonempty", "normalUnitLengthMaxError": float(np.abs(lengths - 1).max()),
              "refinedUffPixelCount": int(uff["validMask"].sum()),
              "observedIctPixelCount": int(ict["validMask"].sum()),
              "generatedIctPixelCount": int(ict["generatedMask"].sum()),
              "lightingMode": "globalOnly" if not result["wildCapDiagnostics"]["artifactMaskProvided"]
                              else "seeRuntimeDiagnostics",
              "note": "CPU array checks and unlit comparison; no likeness or PBR accuracy claim"}
    (resultDir / "materialValidation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resultDir", type=Path)
    inspectResult(parser.parse_args().resultDir)
