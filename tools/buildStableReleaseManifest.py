#!/usr/bin/env python3
"""Build a release inventory from the explicit deployment allowlist.

This command inventories files; it neither installs packages nor starts/tests
the service. Previously read model digests are reused after size/mtime checks,
so packaging does not reread the nine large checkpoint files. Such reuse is
recorded explicitly and is not a fresh checksum verification of those files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile


EXCLUDED_DIRECTORIES = {
    ".git", "__pycache__", "build", ".venv", ".runtimes", ".cache",
    "testdata", "runs", "notebooks",
}
SECRET_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}
CHECKSUM = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_path(text: str) -> PurePosixPath:
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or "\\" in text
            or any(part in {"", ".", ".."} for part in text.split("/"))
            or any(ord(character) < 32 for character in text)):
        raise ValueError(f"Allowlist contains an invalid relative path: {text!r}")
    return path


def excluded_reason(path: PurePosixPath) -> str | None:
    parts = tuple(part.casefold() for part in path.parts)
    if any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in parts):
        return "cacheBuildOrUnreleasedDataDirectory"
    if any(first == "reports" and second.startswith("deployment")
           for first, second in zip(parts, parts[1:])):
        return "privateDeploymentRecords"
    name = parts[-1]
    if name.startswith(".release-manifest-"):
        return "unfinishedManifestTemporaryFile"
    # Only the reviewed empty template is publishable. This also catches
    # service.env.backup and .env.local without opening their contents.
    if ".env" in name and path.as_posix() != "deploy/service.env.example":
        return "privateEnvironmentFile"
    if name in {"id_rsa", "id_ed25519", "id_ecdsa", "authorized_keys", "known_hosts"}:
        return "sshMaterial"
    if any(word in name for word in ("credential", "secret", "private-key", "private_key")):
        return "possibleCredentialFile"
    if path.suffix.casefold() in SECRET_EXTENSIONS:
        return "credentialOrKeyFile"
    if path.suffix.casefold() in {".pyc", ".pyo", ".so", ".ipynb"}:
        return "compiledBinaryOrNotebook"
    return None


def signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def model_cache(root: Path, manifest_path: Path) -> tuple[dict, dict, int]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    generated = datetime.fromisoformat(manifest["generatedAt"])
    if generated.tzinfo is None:
        raise ValueError("Model hash snapshot timestamp must include a timezone")
    generated_ns = int(generated.timestamp() * 1_000_000_000)
    cached = {}
    for name, entry in manifest["models"].items():
        path = relative_path(entry["path"]).as_posix()
        if path in cached or not CHECKSUM.fullmatch(entry["sha256"]):
            raise ValueError("Duplicate model path or invalid model digest")
        if not isinstance(entry["sizeBytes"], int) or entry["sizeBytes"] <= 0:
            raise ValueError("Invalid model file size")
        cached[path] = {**entry, "modelName": name}
    provenance = {
        "path": manifest_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generatedAt": manifest["generatedAt"],
        "reusePolicy": "Size must match and model mtime must not exceed the previous hash snapshot time; no checkpoint bytes reread.",
    }
    return cached, provenance, generated_ns


def build_manifest(root: Path, allowlist: Path, output: Path, models: Path) -> dict:
    allowlist_bytes = allowlist.read_bytes()
    entries = []
    for line in allowlist_bytes.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(relative_path(stripped))
    if not entries:
        raise ValueError("Release allowlist is empty")
    cached, model_provenance, generated_ns = model_cache(root, models)
    self_path = output.relative_to(root).as_posix()
    files, excluded = {}, {}

    def visit(relative: PurePosixPath, ancestors: frozenset[Path]) -> None:
        name = relative.as_posix()
        reason = "manifestSelfReference" if name == self_path else excluded_reason(relative)
        if reason:
            excluded[name] = reason
            return
        source = root.joinpath(*relative.parts)
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Missing, unreadable or cyclic release path: {name}") from error
        # Dereference curated symlinks but never use them to smuggle excluded
        # data or environment files into an innocently named allowlist entry.
        resolved_reason = excluded_reason(PurePosixPath(resolved.as_posix()))
        if resolved_reason and not (name == "deploy/service.env.example" and resolved == source):
            raise ValueError(f"Release symlink resolves to an excluded target: {name}")
        info = source.stat()
        if stat.S_ISDIR(info.st_mode):
            if resolved in ancestors:
                raise ValueError(f"Directory symlink cycle in release: {name}")
            for child in sorted(source.iterdir(), key=lambda item: item.name):
                visit(relative / child.name, ancestors | {resolved})
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Release path is not a regular file: {name}")
        if name in files:
            return
        if name in cached:
            item = cached[name]
            if info.st_size != item["sizeBytes"] or info.st_mtime_ns > generated_ns:
                raise ValueError(f"Model changed since its checksum snapshot; explicitly regenerate modelHashes.json: {name}")
            digest = item["sha256"]
            hash_source = "reusedModelSnapshotAfterSizeAndMtimeCheck"
        else:
            digest = sha256(source)
            hash_source = "computedFromFile"
        if signature(info) != signature(source.stat()):
            raise ValueError(f"File changed while constructing release manifest: {name}")
        files[name] = {
            "path": name, "sizeBytes": info.st_size, "sha256": digest,
            "mode": format(stat.S_IMODE(info.st_mode), "04o"),
            "hashSource": hash_source,
        }

    for entry in entries:
        visit(entry, frozenset())
    missing_models = sorted(set(cached) - set(files))
    if missing_models:
        raise ValueError(f"Release allowlist omits required models: {missing_models}")
    # Ensure the provenance inputs were not edited while file hashes ran.
    if allowlist.read_bytes() != allowlist_bytes or sha256(models) != model_provenance["sha256"]:
        raise ValueError("Allowlist or model snapshot changed during manifest generation")
    ordered = [files[name] for name in sorted(files)]
    manifest = {
        "schemaVersion": 1,
        "releaseId": "faceappearance-stable-20261012",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "pathBase": "releaseRoot",
        "allowlist": {"path": allowlist.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(allowlist_bytes).hexdigest()},
        "modelHashSnapshot": model_provenance,
        "fileCount": len(ordered),
        "totalBytes": sum(item["sizeBytes"] for item in ordered),
        "reusedModelHashCount": sum(item["hashSource"].startswith("reused") for item in ordered),
        "files": ordered,
        "excluded": [{"path": name, "reason": excluded[name]} for name in sorted(excluded)],
        "notes": [
            "Only relative allowlist paths are published; resolved host paths are omitted.",
            "The output manifest does not include itself.",
            "A reused model digest is previously measured provenance, not a fresh reread; verify transferred bytes independently on the destination.",
            "This inventory does not assert successful installation, inference, protocol checks or service readiness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".release-manifest-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allowlist", default="deploy/release-files.txt")
    parser.add_argument("--models", default="deploy/modelHashes.json")
    parser.add_argument("--output", default="deploy/releaseManifest.json")
    parser.add_argument("--release-id", default=None)
    arguments = parser.parse_args()
    root = arguments.project_root.resolve(strict=True)
    selected = [root.joinpath(*relative_path(value).parts) for value in
                (arguments.allowlist, arguments.output, arguments.models)]
    for path in selected:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Manifest control paths must remain inside the release root")
    report = build_manifest(root, *selected)
    if arguments.release_id:
        report["releaseId"] = arguments.release_id
        selected[1].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": arguments.output, "fileCount": report["fileCount"],
                      "totalBytes": report["totalBytes"],
                      "reusedModelHashCount": report["reusedModelHashCount"]}, indent=2))


if __name__ == "__main__":
    main()
