"""Run explicitly selected UFF cases with one model load and frozen settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import platform
import random
import re
import shutil
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.config import PipelineConfig
from faceAppearance.pipeline import Pipeline


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/clientService.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "batchManifest.json"
    if manifest_path.exists() or (args.output / "cases").exists():
        raise FileExistsError("Use a new batch output; existing runs are never overwritten")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    cases = selection["cases"]
    ids = [item["caseId"] for item in cases]
    if not cases or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", i) for i in ids):
        raise ValueError("Case IDs must be unique simple directory names")
    config = PipelineConfig.fromFile(args.config, projectRoot=ROOT)
    if not config.enableAssembly or config.enableWildCap or config.assemblySourceStage != "fusion":
        raise ValueError("This baseline requires fusion assembly and no WildCap")
    manifest = {
        "schemaVersion": 1, "status": "preparing", "createdAt": datetime.now(timezone.utc).isoformat(),
        "selection": str(args.selection.resolve()), "selectionSha256": sha(args.selection),
        "config": config.toDict(), "cases": [],
        "captureModeNote": "User identifies the earlier main case as camera orbit; this new batch predominantly uses a fixed camera and head turning. Individual new cases are not independently labelled.",
        "limits": ["Input views participate in reconstruction; not held-out photo validation",
                   "UFF native geometry and unchanged legacy client assembly are saved separately",
                   "No SH, WildCap, eyelid refinement, semantic fusion or SDF template blend applied",
                   "Canonical units are uncalibrated; camera errors are learned-UV consistency, not 3D ground truth",
                   "Directory count is not a verified count of distinct people"],
    }
    for case in cases:
        paths = [Path(p).resolve() for p in case["paths"]]
        if not config.minViews <= len(paths) <= config.maxViews:
            raise ValueError("Each selected case must have 4–16 ordered views")
        hashes = [sha(p) for p in paths]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"Duplicate views within {case['caseId']}")
        manifest["cases"].append({**case, "paths": list(map(str, paths)),
            "label": case.get("label", case["caseId"]),
            "sourceDirectory": str(paths[0].parent), "photoSha256": hashes,
            "outputDirectory": str(args.output / "cases" / case["caseId"] / "reconstruction"),
            "status": "pending"})
    source_files = list((ROOT / "src/faceAppearance").rglob("*.py")) + [Path(__file__), args.config]
    manifest["sourceHashes"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source_files))}
    shutil.copy2(__file__, args.output / "runUffMultiCase.py")
    save(args.output / "config.json", config.toDict())
    save(manifest_path, manifest)
    import numpy as np
    import torch
    manifest["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "torchThreads": torch.get_num_threads()}
    start = perf_counter()
    try:
        pipeline = Pipeline(config).loadModels()
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        save(manifest_path, manifest)
        raise
    manifest["preloadSeconds"] = pipeline.loadSeconds
    manifest["assemblyAssetLoadSeconds"] = pipeline.assemblyAssetLoadSeconds
    manifest["geometryAssetHashes"] = {str(p.relative_to(ROOT)): sha(p)
                                        for p in pipeline.geometryAdapter.assetPaths().values()}
    manifest["status"] = "running"
    save(manifest_path, manifest)
    for case in manifest["cases"]:
        directory = Path(case["outputDirectory"])
        case["status"] = "running"
        save(manifest_path, manifest)
        print(f"CASE START {case['caseId']}", flush=True)
        case_start = perf_counter()
        try:
            for path, expected in zip(case["paths"], case["photoSha256"]):
                if sha(path) != expected:
                    raise ValueError(f"Input changed: {path}")
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            result = pipeline.run([Path(p) for p in case["paths"]], directory, "assembly")
            case.update(status="succeeded", inferenceSeconds=result["inferenceSeconds"],
                        stageSeconds=result["stageSeconds"], peakAllocatedGiB=result["peakAllocatedGiB"],
                        resultSha256=sha(directory / "result.json"))
            print(f"CASE DONE {case['caseId']} {result['inferenceSeconds']:.2f}s", flush=True)
        except Exception as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
            directory.parent.mkdir(parents=True, exist_ok=True)
            (directory.parent / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"CASE FAILED {case['caseId']}: {case['error']}", flush=True)
        case["wallSeconds"] = perf_counter() - case_start
        save(manifest_path, manifest)
        gc.collect()
        torch.cuda.empty_cache()
    manifest["status"] = "complete"
    manifest["succeededCases"] = sum(c["status"] == "succeeded" for c in manifest["cases"])
    manifest["failedCases"] = sum(c["status"] == "failed" for c in manifest["cases"])
    manifest["wallSecondsIncludingLoadAndHashing"] = perf_counter() - start
    manifest["completedAt"] = datetime.now(timezone.utc).isoformat()
    save(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "succeededCases", "failedCases", "preloadSeconds", "wallSecondsIncludingLoadAndHashing")}), flush=True)


if __name__ == "__main__":
    main()
