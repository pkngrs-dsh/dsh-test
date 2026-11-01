"""Validate the deliverable before a client job can report success."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


class InvalidClientAssetError(ValueError):
    code = "invalid_client_asset"


def fileHash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validateClientOutput(outputDir: Path) -> dict:
    """Require a consistent v1 manifest, ZIP and archive identity; never extract."""
    try:
        root = Path(outputDir).resolve()
        assembly = root / "assembly"
        if assembly.is_symlink() or not assembly.resolve().is_relative_to(root):
            raise ValueError("Assembly directory escaped output")
        for name in ("manifest.json", "archive.json", "clientAsset.zip"):
            path = assembly / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing regular file: {name}")
        manifestBytes = (assembly / "manifest.json").read_bytes()
        manifest = json.loads(manifestBytes)
        archive = json.loads((assembly / "archive.json").read_text(encoding="utf-8"))
        if manifest["schemaVersion"] != 1 or manifest["status"] != "exported":
            raise ValueError("Unsupported client manifest")
        if not manifest["templateId"] or not manifest["templateHashes"]:
            raise ValueError("Template identity missing")
        if manifest["coordinates"]["exportLengthUnit"] != "meter":
            raise ValueError("Expected meter coordinates")
        if not manifest["binding"]["originalBindingIndicesRemainValid"]:
            raise ValueError("Binding indices were not preserved")
        if manifest["paths"]["archive"] != "clientAsset.zip":
            raise ValueError("Unexpected archive path")
        for key in ("meshNpz", "obj", "mtl", "materials", "components", "diagnostics"):
            if manifest["paths"][key] not in manifest["files"]:
                raise ValueError("Required client payload missing from manifest")
        if not isinstance(manifest["templateId"], str) or not isinstance(manifest["templateHashes"], dict):
            raise ValueError("Invalid template identity")
        if not isinstance(manifest["counts"], dict) or any(
                type(value) is not int or value < 0 for value in manifest["counts"].values()):
            raise ValueError("Invalid mesh counts")
        zipPath = assembly / "clientAsset.zip"
        digest = fileHash(zipPath)
        size = zipPath.stat().st_size
        if (archive["schemaVersion"] != 1 or archive["path"] != "clientAsset.zip"
                or archive["sha256"] != digest or archive["bytes"] != size
                or archive["manifestSha256"] != hashlib.sha256(manifestBytes).hexdigest()):
            raise ValueError("Archive identity does not match")
        expected = set(manifest["files"]) | {"manifest.json"}
        with zipfile.ZipFile(zipPath) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != expected or set(archive["entries"]) != expected:
                raise ValueError("Archive entries do not match manifest")
            for name in names:
                path = PurePosixPath(name)
                if (path.is_absolute() or ".." in path.parts or "\\" in name
                        or ":" in name or path.as_posix() != name):
                    raise ValueError("Unsafe archive entry")
                if name == "manifest.json":
                    if bundle.read(name) != manifestBytes:
                        raise ValueError("Packaged manifest differs")
                    continue
                identity = manifest["files"][name]
                checksum = hashlib.sha256()
                count = 0
                with bundle.open(name) as payload:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                        checksum.update(chunk)
                        count += len(chunk)
                if count != identity["bytes"] or checksum.hexdigest() != identity["sha256"]:
                    raise ValueError("Packaged payload differs from manifest")
        return {key: manifest[key] for key in (
            "schemaVersion", "templateId", "templateHashes", "coordinates", "binding", "counts"
        )} | {"archive": {"sha256": digest, "sizeBytes": size}}
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        raise InvalidClientAssetError("Client asset package is missing or failed integrity checks") from error
