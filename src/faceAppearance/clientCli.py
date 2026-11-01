"""Standalone client and offline import checks; never initializes reconstruction models."""

import argparse
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile

from .clientHttp import FaceAppearanceClient, JobFailedError, positiveSeconds
from .clientImport import loadClientAsset, auditTemplateCompatibility, buildRenderMesh


def writeJson(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".client-", suffix=".json",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def templateIdentity(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    identity = {key: manifest[key] for key in ("templateId", "templateHashes")}
    if not isinstance(identity["templateId"], str) or not identity["templateId"]:
        raise ValueError("Template manifest must provide templateId")
    if not isinstance(identity["templateHashes"], dict) or not identity["templateHashes"]:
        raise ValueError("Template manifest must provide templateHashes")
    return identity


def inspectAsset(archive, template, *, expectedDownload=None):
    identity = templateIdentity(template)
    expected = expectedDownload or {}
    asset = loadClientAsset(archive, expectedTemplateId=identity["templateId"],
                            expectedTemplateHashes=identity["templateHashes"],
                            expectedArchiveSha256=expected.get("sha256"),
                            expectedArchiveBytes=expected.get("sizeBytes"))
    report = auditTemplateCompatibility(asset, template)
    report.update(archiveSha256=asset.archiveSha256, archiveBytes=asset.archiveBytes)
    return asset, report


def writeRenderMesh(path, asset):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".client-", suffix=".npz", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **buildRenderMesh(asset))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finishJob(client, state, output, template, timeout):
    statePath = output / "clientJob.json"
    try:
        job = client.waitForJob(state["jobId"], timeout=timeout)
    except JobFailedError as error:
        state["lastJob"] = error.record
        writeJson(statePath, state)
        raise
    state["lastJob"] = job
    writeJson(statePath, state)
    archive = output / "clientAsset.zip"
    if not archive.exists():
        client.download(job, archive)
    asset, report = inspectAsset(archive, template, expectedDownload=job["download"])
    writeRenderMesh(output / "renderMesh.npz", asset)
    writeJson(output / "importReport.json", report)
    return {"status": "succeeded", "jobId": state["jobId"], "archive": str(archive.resolve()),
            "importReport": str((output / "importReport.json").resolve()),
            "renderMesh": str((output / "renderMesh.npz").resolve()),
            "rigValidated": False, "blendshapesValidated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Service HTTP(S) origin; submit defaults to http://127.0.0.1:8000")
    parser.add_argument("--tokenEnv", default="FACE_APPEARANCE_API_TOKEN")
    parser.add_argument("--requestTimeout", type=float, default=30)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="Upload once, save job state, then download and inspect")
    submit.add_argument("--input", required=True, type=Path, help="Directory of 4–16 images")
    submit.add_argument("--output", required=True, type=Path, help="New or empty client output directory")
    submit.add_argument("--template", required=True, type=Path, help="Local compiled template asset directory")
    submit.add_argument("--timeout", type=float, default=600, help="Polling deadline in seconds")
    submit.add_argument("--noWait", action="store_true", help="Return after saving the accepted job ID")
    resume = commands.add_parser("resume", help="Resume a saved job without uploading photos again")
    resume.add_argument("--output", required=True, type=Path, help="Directory containing clientJob.json")
    resume.add_argument("--template", required=True, type=Path)
    resume.add_argument("--timeout", type=float, default=600)
    inspect = commands.add_parser("inspect", help="Inspect a ZIP offline against a local template")
    inspect.add_argument("--archive", required=True, type=Path)
    inspect.add_argument("--template", required=True, type=Path)
    inspect.add_argument("--report", required=True, type=Path)
    inspect.add_argument("--renderMesh", type=Path, help="Optional mesh with source vertex mapping")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            if args.renderMesh is not None and args.report.resolve() == args.renderMesh.resolve():
                raise ValueError("Report and render mesh must use distinct output paths")
            for path in (args.report, args.renderMesh):
                if path is not None and (path.exists() or path.is_symlink()):
                    raise FileExistsError(f"Choose a new output path: {path}")
            asset, report = inspectAsset(args.archive, args.template)
            if args.renderMesh:
                writeRenderMesh(args.renderMesh, asset)
            writeJson(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        positiveSeconds(args.timeout, "timeout")
        output = args.output.resolve()
        identity = templateIdentity(args.template)
        if args.command == "resume":
            state = json.loads((output / "clientJob.json").read_text(encoding="utf-8"))
            if state.get("schemaVersion") != 1 or state.get("expectedTemplate") != identity:
                raise ValueError("Saved state does not match the supplied template")
            if args.url is not None and args.url.rstrip("/") != state["baseUrl"]:
                raise ValueError("Resume must use the original service origin")
            baseUrl = state["baseUrl"]
        else:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise FileExistsError("Choose a new or empty client output directory")
            baseUrl = args.url or "http://127.0.0.1:8000"
        client = FaceAppearanceClient(baseUrl, token=os.environ.get(args.tokenEnv),
                                      requestTimeout=args.requestTimeout)
        if args.command == "submit":
            if not args.input.is_dir():
                raise FileNotFoundError(f"Photo directory not found: {args.input}")
            allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            photos = sorted(path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in allowed)
            if client.getCapabilities().get("template") != identity:
                raise ValueError("Service template identity differs from the local binding template")
            output.mkdir(parents=True, exist_ok=True)
            job = client.submit(photos, templateId=identity["templateId"])
            # Always print the ID before local I/O, so an accepted request can be located after disk errors.
            print(f"Accepted job {job['jobId']}; state: {output / 'clientJob.json'}", file=sys.stderr, flush=True)
            state = {"schemaVersion": 1, "baseUrl": client.baseUrl, "jobId": job["jobId"],
                     "expectedTemplate": identity, "lastJob": job}
            writeJson(output / "clientJob.json", state)
            if args.noWait:
                print(json.dumps({"status": job["status"], "jobId": job["jobId"],
                                  "state": str(output / "clientJob.json")}, ensure_ascii=False))
                return
        result = finishJob(client, state, output, args.template, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, http.client.HTTPException) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
