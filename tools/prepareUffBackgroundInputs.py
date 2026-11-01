"""Prepare CPU-only background ablations from frozen UFF crops and FaRL caches.

No models are loaded. Baseline pixels are decoded from the saved input PNGs.
The normalized semantic cache's explicit label schema is validated before use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.faceSemantics import semanticLabelNames

HEAD_NAMES = ("skin", "leftBrow", "rightBrow", "leftEye", "rightEye", "glasses",
              "leftEar", "rightEar", "earring", "nose", "innerMouth", "upperLip",
              "lowerLip", "hair", "hat")
VARIANTS = ("baseline", "bgRemovedGray", "headOnlyGray", "headOnlyBlack")
VARIANT_TITLES = ("A | baseline crop", "B | background -> gray",
                  "C | head only -> gray", "D | head only -> black")
PALETTE = np.array([[0,0,0], [180,155,130], [250,90,50], [250,140,30],
                    [30,180,255], [30,240,210], [200,100,200], [160,120,180],
                    [160,100,140], [200,200,80], [70,210,80], [90,40,130],
                    [245,65,145], [210,40,70], [100,130,180], [200,180,70],
                    [60,80,120], [90,75,40], [120,100,90]], np.uint8)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "sizeBytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def rgb_image(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


def contact_sheet(rows, column_titles, title, notes, output, panel=256):
    left, top, gap, line_height = 90, 84, 8, 24
    foot = 22 + len(notes) * line_height
    canvas = Image.new("RGB", (left + len(column_titles) * (panel + gap) + gap,
                               top + len(rows) * (panel + gap) + foot), (20, 25, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, fill=(235, 241, 245), font=font(21))
    for col, text in enumerate(column_titles):
        draw.text((left + col * (panel + gap), 51), text, fill=(205, 220, 226), font=font(15))
    for row, images in enumerate(rows):
        y = top + row * (panel + gap)
        draw.text((12, y + 9), f"View {row+1}", fill=(220, 227, 232), font=font(15))
        for col, im in enumerate(images):
            canvas.paste(im.resize((panel, panel), Image.Resampling.LANCZOS),
                         (left + col * (panel + gap), y))
    for i, note in enumerate(notes):
        draw.text((16, top + len(rows) * (panel + gap) + 6 + i * line_height),
                  note, fill=(174, 191, 201), font=font(14))
    canvas.save(output)


def transform_labels(labels, record):
    """Inverse-map working integer pixel centers, nearest, edge replication.

    inputToWorking already includes the half-pixel offset used by PIL resize.
    At a nearest-neighbor tie choose the larger source pixel index.
    """
    if record["workingSize"] != [512, 512]:
        raise ValueError("Expected the frozen 512x512 input crop")
    width, height = record["orientedSize"]
    if labels.shape != (height, width):
        raise ValueError("Cached semantics must match the EXIF-oriented original")
    matrix = np.asarray(record["inputToWorking"], np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid inputToWorking")
    inverse = np.linalg.inv(matrix)
    if not np.allclose(inverse, record["workingToInput"], rtol=0, atol=1e-9):
        raise ValueError("Stored forward/inverse pixel transforms disagree")
    left, top, right, bottom = record["cropBoxXyxy"]
    scale_x, scale_y = 512 / (right-left), 512 / (bottom-top)
    expected = np.array([[scale_x, 0, scale_x * (0.5-left)-0.5],
                         [0, scale_y, scale_y * (0.5-top)-0.5], [0,0,1]])
    if not np.allclose(matrix, expected, rtol=0, atol=1e-9):
        raise ValueError("Crop metadata and half-pixel transform disagree")
    yy, xx = np.mgrid[:512, :512]
    original = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ inverse.T
    source_x = original[..., 0] / original[..., 2]
    source_y = original[..., 1] / original[..., 2]
    observed = ((source_x >= -1e-7) & (source_x <= width-1+1e-7) &
                (source_y >= -1e-7) & (source_y <= height-1+1e-7))
    nearest_x = np.floor(source_x + 0.5 + 1e-10).astype(np.int64).clip(0, width-1)
    nearest_y = np.floor(source_y + 0.5 + 1e-10).astype(np.int64).clip(0, height-1)
    return labels[nearest_y, nearest_x], observed


def verify_baseline_crop(photo, record, baseline):
    """Reproduce ingestion's EXIF/RGB, crop, edge pad and PIL Lanczos resize."""
    with Image.open(photo) as image:
        original = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    height, width = original.shape[:2]
    left, top, right, bottom = record["cropBoxXyxy"]
    expected_padding = [max(0,-left), max(0,-top), max(0,right-width), max(0,bottom-height)]
    if expected_padding != record["paddingLeftTopRightBottom"]:
        raise ValueError("Source crop padding metadata disagrees")
    crop = original[max(0,top):min(height,bottom), max(0,left):min(width,right)]
    if any(expected_padding):
        crop = np.pad(crop, ((expected_padding[1], expected_padding[3]),
                            (expected_padding[0], expected_padding[2]), (0,0)), mode="edge")
    rebuilt = np.asarray(Image.fromarray(crop).resize((512,512), Image.Resampling.LANCZOS), np.float32) / np.float32(255)
    changed = int(np.any(rebuilt != baseline, axis=-1).sum())
    maximum = float(np.abs(rebuilt-baseline).max())
    if changed:
        raise AssertionError(f"Saved PNG differs from recreated readImage pixels: {changed} pixels, max {maximum}")
    return {"method": "EXIF transpose + RGB convert + saved cropBox + saved edge padding + PIL Lanczos resize to 512 + float32 / 255",
            "changedRgbPixels": changed, "maxAbsoluteFloat32Delta": maximum,
            "exactFloat32Equality": True}


def prepare_case(case, semantics_root, output_root, dilation, feather):
    case_id = case["caseId"]
    source = Path(case["outputDirectory"])
    semantic_dir = semantics_root / "cases" / case_id / "semantics"
    output = output_root / "cases" / case_id
    if (output / "inputs.npz").exists() or (output / "inputs.json").exists():
        raise FileExistsError(f"Inputs already exist; choose a new output directory: {output}")
    records = json.loads((source / "inputs.json").read_text())
    semantic_manifest = json.loads((semantic_dir / "manifest.json").read_text())
    if (semantic_manifest["status"] != "complete" or semantic_manifest["parserName"] != "celebm"
            or tuple(semantic_manifest["labelNames"]) != semanticLabelNames):
        raise ValueError("Expected complete CelebM cache in the normalized 19-label schema")
    if semantic_manifest["inputsSha256"] != file_hash(source / "inputs.json"):
        raise ValueError("Semantic cache does not match frozen input provenance")
    if len(records) != 4 or len(semantic_manifest["views"]) != len(records):
        raise ValueError("Expected exactly four matching views")
    with np.load(source / "geometry.npz", allow_pickle=False) as geometry:
        face_masks = geometry["faceMasks"].copy()
        if face_masks.shape != (4, 512, 512) or not np.isfinite(face_masks).all():
            raise ValueError("Invalid source geometry face masks")
        if geometry["uvMaps"].shape != (4, 512, 512, 2):
            raise ValueError("Invalid source UV map shape")
    head_ids = [semanticLabelNames.index(name) for name in HEAD_NAMES]
    arrays = {name: [] for name in (*VARIANTS, "keepForeground", "keepHead",
              "alphaForeground", "alphaHead", "semanticLabels", "semanticHead",
              "faceProtected", "observedOriginal")}
    views = []
    for index, (record, semantic_view) in enumerate(zip(records, semantic_manifest["views"])):
        photo = Path(record["path"])
        png_path = source / f"input{index:02d}.png"
        cache_path = semantic_dir / f"view{index:02d}.npz"
        photo_record, png_record, cache_record = map(file_record, (photo, png_path, cache_path))
        if not (photo_record["sha256"] == record["sha256"] == semantic_view["photoSha256"]):
            raise ValueError("Original photo hash mismatch")
        if cache_record["sha256"] != semantic_view["cacheSha256"] or semantic_view["viewIndex"] != index:
            raise ValueError("Semantic view provenance mismatch")
        with Image.open(photo) as original:
            if list(ImageOps.exif_transpose(original).size) != record["orientedSize"]:
                raise ValueError("Original oriented image dimensions changed")
        with Image.open(png_path) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError("Baseline PNG must be RGB 512x512")
            baseline = np.asarray(image, np.float32) / np.float32(255)
        baseline_verification = verify_baseline_crop(photo, record, baseline)
        with np.load(cache_path, allow_pickle=False) as cache:
            labels = cache["labelMap"]
            if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= len(semanticLabelNames):
                raise ValueError("Unexpected cached label IDs")
            labels, observed = transform_labels(labels, record)
        protected = face_masks[index] > .5
        semantic_head = np.isin(labels, head_ids)
        foreground = (labels != semanticLabelNames.index("background")) | protected
        head_core = semantic_head | protected
        # Euclidean expansion measured from the nearest foreground pixel center.
        distance = distance_transform_edt(~head_core)
        keep_head = distance <= dilation
        alpha_head = np.clip((dilation + feather - distance) / feather, 0, 1).astype(np.float32)
        alpha_head[protected] = 1
        alpha_foreground = foreground.astype(np.float32)
        bg_gray = baseline * alpha_foreground[..., None] + np.float32(.5) * (1-alpha_foreground[..., None])
        head_gray = baseline * alpha_head[..., None] + np.float32(.5) * (1-alpha_head[..., None])
        head_black = baseline * alpha_head[..., None]
        # Explicit copy makes exact source-face preservation independent of blend arithmetic.
        for changed in (bg_gray, head_gray, head_black):
            changed[protected] = baseline[protected]
        variants = (baseline, bg_gray, head_gray, head_black)
        for name, value in zip(VARIANTS, variants):
            arrays[name].append(value)
        for name, value in (("keepForeground", foreground), ("keepHead", keep_head),
                ("alphaForeground", alpha_foreground), ("alphaHead", alpha_head),
                ("semanticLabels", labels), ("semanticHead", semantic_head),
                ("faceProtected", protected), ("observedOriginal", observed)):
            arrays[name].append(value)
        changed_counts = {name: int(np.any(value[protected] != baseline[protected], axis=-1).sum())
                          for name, value in zip(VARIANTS[1:], variants[1:])}
        if any(changed_counts.values()) or not np.all(alpha_head[protected] == 1):
            raise AssertionError("Face-protected RGB changed")
        for value, alpha in ((bg_gray, alpha_foreground), (head_gray, alpha_head), (head_black, alpha_head)):
            if not np.array_equal(value[alpha == 1], baseline[alpha == 1]):
                raise AssertionError("RGB changed where alpha equals 1")
        views.append({"viewIndex": index, "sourcePhoto": photo_record, "sourceCropPng": png_record,
            "semanticCache": cache_record, "orientedSize": record["orientedSize"],
            "workingSize": record["workingSize"], "inputToWorking": record["inputToWorking"],
            "workingToInput": record["workingToInput"], "cropBoxXyxy": record["cropBoxXyxy"],
            "sourcePaddingMode": record["paddingMode"], "paddingPixels": int((~observed).sum()),
            "baselineReproduction": baseline_verification, "alphaOneRgbExactlyUnchanged": True,
            "pixelsPerWorkingLabel": np.bincount(labels.ravel(), minlength=19).tolist(),
            "faceProtectedPixels": int(protected.sum()),
            "semanticBackgroundOverriddenByFaceProtection": int(((labels == 0) & protected).sum()),
            "nonHeadSemanticPixelsOverriddenByFaceProtection": int((~semantic_head & protected).sum()),
            "keepForegroundPixels": int(foreground.sum()), "keepHeadPixels": int(keep_head.sum()),
            "headFeatherPixels": int(((alpha_head > 0) & (alpha_head < 1)).sum()),
            "removedPixelEquivalent": {"B": float((1-alpha_foreground).sum()), "C_D": float((1-alpha_head).sum())},
            "faceRgbChangedPixels": changed_counts,
            "faceRgbMaxAbsoluteDelta": {name: float(np.abs(value[protected]-baseline[protected]).max(initial=0))
                                      for name, value in zip(VARIANTS[1:], variants[1:])}})
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(value) for name, value in arrays.items()}
    np.savez_compressed(output / "inputs.npz", **arrays)
    rows = [[rgb_image(arrays[name][v]) for name in VARIANTS] for v in range(4)]
    contact_sheet(rows, VARIANT_TITLES, case_id + " | frozen 512px inputs",
        ["B: label 0 replaced; body retained. C/D: head labels + 6px expansion + 4px outer feather.",
         "Source geometry faceMask > 0.5 is copied exactly in B/C/D. Display PNGs quantize gray; NPZ uses 0.5."],
        output / "inputContactSheet.png")
    mask_rows = []
    for v in range(4):
        mask_rows.append([Image.fromarray(PALETTE[arrays["semanticLabels"][v]]),
                         rgb_image(arrays["faceProtected"][v].astype(np.float32)),
                         rgb_image(arrays["alphaForeground"][v]), rgb_image(arrays["alphaHead"][v])])
        rgb_image(arrays["alphaForeground"][v]).save(output / f"alphaForeground{v:02d}.png")
        rgb_image(arrays["alphaHead"][v]).save(output / f"alphaHead{v:02d}.png")
        Image.fromarray(arrays["semanticLabels"][v]).save(output / f"semanticLabels{v:02d}.png")
        for name in VARIANTS:
            rgb_image(arrays[name][v]).save(output / f"{name}{v:02d}.png")
    contact_sheet(mask_rows, ("Normalized labels", "Protected face > 0.5", "B foreground alpha", "C/D head alpha"),
        case_id + " | mask audit", ["White = preserve RGB; black = replace RGB; gray = feather blend.",
        "Label colors: background black, neck blue, cloth dark blue, hair brown; see inputs.json for all IDs."],
        output / "maskContactSheet.png")
    report = {"schemaVersion": 1, "status": "prepared", "caseId": case_id,
        "createdAt": datetime.now(timezone.utc).isoformat(), "sourceGeometry": file_record(source / "geometry.npz"),
        "sourceInputs": file_record(source / "inputs.json"), "semanticManifest": file_record(semantic_dir / "manifest.json"),
        "preparationScript": file_record(Path(__file__)),
        "labelSchemaSource": file_record(ROOT / "src/faceAppearance/faceSemantics.py"),
        "inputCoordinateImplementation": file_record(ROOT / "src/faceAppearance/data.py"),
        "labelNames": list(semanticLabelNames), "headLabelNames": list(HEAD_NAMES), "headLabelIds": head_ids,
        "headExcludedNames": ["background", "neck", "necklace", "cloth"],
        "variantOrder": list(VARIANTS), "grayValueRgb": [.5,.5,.5], "blackValueRgb": [0,0,0],
        "colorConvention": "RGB stored PNG code values / 255, no linearization or enhancement; float32 in [0,1]",
        "maskRecipe": {"background": "semantic label != background OR original geometry.faceMasks > 0.5; binary, no feather",
            "headCore": "head semantic labels OR original geometry.faceMasks > 0.5",
            "headDilationPixelsAt512": dilation, "headOuterFeatherPixelsAt512": feather,
            "keepHead": "distance_transform_edt(~headCore) <= headDilationPixelsAt512",
            "alphaHead": "clip((dilation + feather - distance_transform_edt(~headCore))/feather, 0, 1)",
            "faceProtection": "alpha forced to 1 and baseline RGB copied at source geometry.faceMasks > 0.5",
            "blend": "baseline * alpha + replacementRgb * (1-alpha)",
            "coordinateSampling": "Inverse inputToWorking on integer output pixel centers; floor(originalXY + 0.5 + 1e-10); source edge clamp",
            "pixelCenterOffset": "inputToWorking already includes (scale*0.5 - 0.5); no extra offset is added",
            "confidenceThreshold": None},
        "arraySchema": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name,value in arrays.items()},
        "views": views, "inputsNpz": file_record(output / "inputs.npz"),
        "contactSheets": {"inputs": str(output / "inputContactSheet.png"), "masks": str(output / "maskContactSheet.png")},
        "automaticChecks": {"protectedFaceRgbExactlyEqualAllVariants": True,
            "baselineExactlyMatchesRecreatedReadImage": True, "alphaOneRgbExactlyUnchanged": True,
            "originalPhotoHashesMatch": True, "semanticCacheHashesMatch": True,
            "semanticSchemaMatchesExplicitNormalizedContract": True, "sourceCropTransformsConsistent": True},
        "visualQa": {"status": "awaitingInspection", "notes": []},
        "limitations": ["Automatic semantic labels can miss wispy hair, earrings or shadowed ears and can include spurious regions.",
            "The source crop can already cut off the top of the head; no missing head content is synthesized.",
            "Head expansion/feather deliberately retains up to 10 pixels of nearby background, neck or clothes at 512 resolution.",
            "Source face-mask protection overrides semantic removal, including any mislabeled or non-head pixels inside that mask.",
            "Nearest semantic resampling follows the saved crop but cannot reproduce the antialiased RGB boundary exactly.",
            "Semantic edge replication follows baseline crop padding and can retain repeated foreground where the source image ends.",
            "B changes environmental label 0 only; C/D also change neck, necklace and clothes, and therefore answer a broader context question.",
            "Preview PNGs round float RGB to uint8; use inputs.npz for inference to retain exact gray 0.5 and feather values."]}
    write_json(output / "inputs.json", report)
    print(json.dumps({"caseId": case_id, "status": "prepared", "inputs": str(output / "inputs.npz"),
                      "protectedFaceChangedPixels": sum(sum(v["faceRgbChangedPixels"].values()) for v in views)}), flush=True)
    return report


def make_overview(reports, output):
    tile_width = 600
    sheets = []
    for report in reports:
        im = Image.open(report["contactSheets"]["inputs"]).convert("RGB")
        im = im.resize((tile_width, round(im.height * tile_width / im.width)), Image.Resampling.LANCZOS)
        sheets.append(im)
    height = max(im.height for im in sheets)
    canvas = Image.new("RGB", (tile_width * 3, height * ((len(sheets)+2)//3)), (20,25,29))
    for index, im in enumerate(sheets):
        canvas.paste(im, (index % 3 * tile_width, index // 3 * height))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, default=ROOT / "runs/uffMultiCase20261011/batchManifest.json")
    parser.add_argument("--semantics-root", type=Path, default=ROOT / "runs/uffFrontPriority20261011")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--head-dilation", type=float, default=6)
    parser.add_argument("--head-feather", type=float, default=4)
    args = parser.parse_args()
    if not np.isfinite([args.head_dilation, args.head_feather]).all() or args.head_dilation < 0 or args.head_feather <= 0:
        raise ValueError("Expected nonnegative dilation and positive feather")
    manifest = json.loads(args.baseline_manifest.read_text())
    cases = [c for c in manifest["cases"] if c["status"] == "succeeded" and (not args.case or c["caseId"] in args.case)]
    if not cases or (args.case and set(args.case) != {c["caseId"] for c in cases}):
        raise ValueError("Requested cases must have successful baselines")
    reports = [prepare_case(case, args.semantics_root.resolve(), args.output.resolve(), args.head_dilation, args.head_feather)
               for case in cases]
    # Include previously prepared cases when completing a selectively started run.
    all_reports = [json.loads(path.read_text()) for path in sorted((args.output / "cases").glob("*/inputs.json"))]
    order = {case["caseId"]: i for i,case in enumerate(manifest["cases"])}
    all_reports.sort(key=lambda report: order[report["caseId"]])
    make_overview(all_reports, args.output / "inputOverview.png")
    write_json(args.output / "inputManifest.json", {"schemaVersion": 1, "status": "prepared",
        "baselineManifest": file_record(args.baseline_manifest), "preparationScript": file_record(Path(__file__)),
        "caseIds": [report["caseId"] for report in all_reports], "caseCount": len(all_reports),
        "cases": [{"caseId": report["caseId"], "inputsJson": file_record(args.output / "cases" / report["caseId"] / "inputs.json"),
                   "inputsNpz": report["inputsNpz"], "contactSheets": report["contactSheets"]} for report in all_reports],
        "overview": str((args.output / "inputOverview.png").resolve()),
        "automaticChecks": {"allProtectedFacePixelsUnchanged": all(report["automaticChecks"]["protectedFaceRgbExactlyEqualAllVariants"] for report in all_reports)}})


if __name__ == "__main__":
    main()
