"""One thin command interface over reusable Python modules."""

import argparse
import json
import os
from pathlib import Path

from .artifacts import writeJson
from .config import PipelineConfig
from .data import auditDataset, listImages


def main():
    projectRoot = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Modular multi-view face appearance prototype")
    parser.add_argument("--config", type=Path, help="Defaults to clientService.json for service commands; preview512.json otherwise")
    commands = parser.add_subparsers(dest="command", required=True)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("--input", type=Path, required=True, help="Directory with 4–16 distinct photos")
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--stopAfter", choices=("geometry", "delight", "fusion", "wildcap", "assembly"), default="fusion")
    assemble = commands.add_parser("assemble", help="Assemble saved geometry/material arrays on the CPU")
    assemble.add_argument("--geometry", type=Path, required=True, help="Saved UFF geometry NPZ")
    assemble.add_argument("--texture", type=Path, required=True, help="Saved albedo/coverage NPZ")
    assemble.add_argument("--output", type=Path, required=True, help="Client asset package directory")
    assemble.add_argument("--colorEncoding", choices=("linear", "srgb", "gamma2.2"), default=None)
    audit = commands.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=projectRoot / "reports/datasetAuditLatest.json")
    for name in ("serve", "serve-client"):
        serve = commands.add_parser(name, help="Start the HTTP service" if name == "serve" else
                                    "Start the client asset HTTP service with assembly enabled")
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8000)
        serve.add_argument("--outputRoot", type=Path, help="Persistent job directory")
        serve.add_argument("--queueCapacity", type=int, default=2, help="Number of waiting jobs; one additional running job")
        serve.add_argument("--apiTokenEnv", default="FACE_APPEARANCE_API_TOKEN", help="Environment variable containing optional Bearer token")
    legacy = commands.add_parser("serve-legacy", help="Serve the previous notebook JSON/URL protocol")
    legacy.add_argument("--host", default="127.0.0.1")
    legacy.add_argument("--port", type=int, default=7860)
    legacy.add_argument("--outputRoot", type=Path, help="Persistent legacy task directory")
    legacy.add_argument("--queueCapacity", type=int, default=2)
    legacy.add_argument("--publicBaseUrl", default=None, help="Public origin for result downloads when OSS is not configured")
    legacy.add_argument("--maleAssemblyAsset", default="assets/clientTemplates/legacyMaleV1")
    legacy.add_argument("--femaleAssemblyAsset", default="assets/clientTemplates/legacyFemaleV1")
    args = parser.parse_args()
    if args.command == "audit":
        report = auditDataset(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writeJson(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key.endswith("Count")}))
        return
    configPath = args.config or projectRoot / "configs" / (
        "clientService.json" if args.command in {"serve-client", "serve-legacy"} else "preview512.json")
    config = PipelineConfig.fromFile(configPath)
    if args.command == "assemble":
        # This explicit offline entry point does not load reconstruction,
        # delighting or WildCap models, even when those are in the config.
        import numpy as np
        from .headAssembly import AssemblyAssets
        from .clientAsset import exportClientAsset

        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Output directory is not empty; choose a new directory")
        with np.load(args.geometry, allow_pickle=False) as savedGeometry:
            geometry = {key: savedGeometry[key] for key in savedGeometry.files}
        with np.load(args.texture, allow_pickle=False) as savedTexture:
            texture = {key: savedTexture[key] for key in savedTexture.files}
        assets = AssemblyAssets(config.projectRoot / config.assemblyAsset)
        encoding = "linear" if "linearAlbedo" in texture else (args.colorEncoding or config.assemblyColorEncoding)
        assembled = assets.assemble(geometry, texture, colorEncoding=encoding)
        manifest = exportClientAsset(args.output, assembled)
        print(json.dumps({"status": manifest.get("status", "exported"),
                          "output": str(args.output.resolve()),
                          "manifest": str((args.output / "manifest.json").resolve())}, ensure_ascii=False))
        return
    if args.command == "serve-legacy":
        import uvicorn
        from .legacyService import createLegacyApp

        app = createLegacyApp(config, outputRoot=args.outputRoot,
                              queueCapacity=args.queueCapacity,
                              publicBaseUrl=args.publicBaseUrl,
                              maleAssemblyAsset=args.maleAssemblyAsset,
                              femaleAssemblyAsset=args.femaleAssemblyAsset)
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
        return
    if args.command in {"serve", "serve-client"}:
        import uvicorn
        from .service import createApp
        if args.command == "serve-client" and not config.enableAssembly:
            parser.error("serve-client requires a configuration with enableAssembly=true")
        token = os.environ.get(args.apiTokenEnv)
        app = createApp(config, outputRoot=args.outputRoot, queueCapacity=args.queueCapacity,
                        apiToken=token)
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
        return
    from .pipeline import Pipeline
    paths = listImages(args.input)
    from .data import loadViews
    loadViews(paths, config.minViews, config.maxViews)  # Fail cheaply before loading weights.
    pipeline = Pipeline(config).loadModels()
    report = pipeline.run(paths, args.output, args.stopAfter)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "inferenceSeconds": report["inferenceSeconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
