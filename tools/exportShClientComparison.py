"""Export three cached delighting results, optionally before/after WildCap.

The source reconstruction is read-only. Existing exports are reused only when
their input identities, client ZIP, template and complete mesh arrays verify.
No models are loaded; WildCap must already have a committed completion marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.clientImport import loadClientAsset
from runShExperiment import methodNames, validateCompletedWildCap

NAMES = {"openDelight": "OpenDelight", "shOnly": "SH 去光", "shPrior": "SH + OpenDelight 先验"}


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def readArrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writeJson(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkMesh(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Comparison client mesh fields differ")
    checks = {key: reference[key].dtype == candidate[key].dtype and
              np.array_equal(reference[key], candidate[key]) for key in reference}
    if not all(checks.values()):
        raise ValueError(f"Comparison requires identical geometry/UV/index arrays: {checks}")
    return checks


def loadVerifiedAsset(directory, templateManifest, expectedIdentity=None):
    directory = Path(directory)
    comparison = readJson(directory / "comparison.json")
    if expectedIdentity is not None and comparison.get("inputIdentity") != expectedIdentity:
        raise ValueError(f"Existing export input identity differs: {directory}")
    archive = readJson(directory / "archive.json")
    if comparison.get("archive") != {key: archive[key] for key in ("sha256", "bytes")}:
        raise ValueError(f"Export archive identity differs from committed comparison metadata: {directory}")
    asset = loadClientAsset(directory / "clientAsset.zip",
                           expectedTemplateId=templateManifest["templateId"],
                           expectedTemplateHashes=templateManifest["templateHashes"],
                           expectedArchiveSha256=archive["sha256"], expectedArchiveBytes=archive["bytes"])
    return asset, comparison


def validateInputs(source, experiment, withWildCap):
    """Shared source/support checks used both at export and viewer startup."""
    source, experiment = Path(source).resolve(), Path(experiment).resolve()
    report = readJson(experiment / "experimentReport.json")
    if Path(report["sourceRun"]).resolve() != source:
        raise ValueError("Comparison experiment refers to a different source reconstruction")
    if readJson(source / "result.json").get("status") != "succeeded":
        raise ValueError("Source reconstruction must have succeeded")
    parameters = report["parameters"]
    if parameters.get("geometryAndCamerasFrozen") is not True or parameters.get("commonViewTexelWeights") is not True:
        raise ValueError("Comparison requires frozen geometry/cameras and shared observation weights")
    common = readArrays(experiment / "commonSupport.npz")["coverageMask"]
    if common.dtype != np.bool_ or not common.any():
        raise ValueError("Comparison support must be a nonempty boolean mask")
    branches = {}
    sharedSourceHashes = None
    for name in methodNames:
        branch = experiment / name
        branchReport = readJson(branch / "result.json")
        if (branchReport.get("status") != "succeeded" or branchReport.get("method") != name
                or Path(branchReport["sourceRun"]).resolve() != source
                or branchReport["experiment"] != parameters):
            raise ValueError(f"Branch identity or parameters differ: {name}")
        sourceHashes = branchReport["sourceHashes"]
        if sharedSourceHashes is not None and sourceHashes != sharedSourceHashes:
            raise ValueError("Branches refer to different source files")
        sharedSourceHashes = sourceHashes
        for filename, expected in sourceHashes.items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts or digest(source / relative) != expected:
                raise ValueError(f"Source identity differs: {filename}")
        copiedFiles = ["geometry.npz", "cameras.json", "inputs.json", "viewTransforms.json"]
        copiedFiles.extend(path.name for path in source.glob("input*.png"))
        for filename in copiedFiles:
            if digest(branch / filename) != digest(source / filename):
                raise ValueError(f"Frozen branch source differs: {name}/{filename}")
        texture = readArrays(branch / "texture.npz")
        if not np.array_equal(texture["coverageMask"], common):
            raise ValueError(f"Branch coverage differs from shared experiment mask: {name}")
        if texture["linearAlbedo"].shape != (*common.shape, 3):
            raise ValueError(f"Unexpected texture shape: {name}")
        if withWildCap:
            validateCompletedWildCap(branch, branchReport, parameters["wildCapPlanned"], parameters["atlasResolution"])
            materials = readArrays(branch / "wildcapUff/materials.npz")
            valid = materials["validMask"].astype(bool)
            if np.any(valid & ~common):
                raise ValueError(f"WildCap support exceeds common photo support: {name}")
            if not np.array_equal(materials["baseWithWildCap"][~valid], texture["albedo"][~valid]):
                raise ValueError(f"WildCap composite changes unqualified texels: {name}")
            if not np.array_equal(materials["baseWithWildCap"][valid], materials["diffuse"][valid]):
                raise ValueError(f"WildCap composite differs from qualified diffuse: {name}")
        branches[name] = branchReport
    sourceSnapshot = dict(sharedSourceHashes)
    for filename in ("viewTransforms.json", "config.json", "result.json", "assembly/clientMesh.npz",
                     "assembly/manifest.json", "assembly/diagnostics.json"):
        sourceSnapshot[filename] = digest(source / filename)
    sourceSnapshot.update({path.name: digest(path) for path in source.glob("input*.png")})
    return report, branches, sourceSnapshot


def exportComparison(source, experiment, withWildCap=True):
    from faceAppearance.clientAsset import exportClientAsset
    from faceAppearance.headAssembly import AssemblyAssets

    source, experiment = Path(source).resolve(), Path(experiment).resolve()
    report, branchReports, sourceHashes = validateInputs(source, experiment, withWildCap)
    config = readJson(source / "config.json")
    assetRoot = Path(config["assemblyAsset"])
    if not assetRoot.is_absolute():
        assetRoot = Path(config.get("projectRoot", ROOT)) / assetRoot
    template = AssemblyAssets(assetRoot)
    if template.manifestSha256 != readJson(source / "assembly/diagnostics.json")["assetManifestSha256"]:
        raise ValueError("Client template settings differ from the source reconstruction")
    templateManifest = readJson(source / "assembly/manifest.json")
    sourceMesh = readArrays(source / "assembly/clientMesh.npz")
    geometry = readArrays(source / "geometry.npz")
    destination = experiment / "clientAssets"
    destination.mkdir(exist_ok=True)
    codeHashes = {name: digest(ROOT / name) for name in (
        "src/faceAppearance/headAssembly.py", "src/faceAppearance/clientAsset.py",
        "tools/exportShClientComparison.py")}
    sourceStageSeconds = readJson(source / "result.json").get("stageSeconds", {})
    entries = []
    for method in methodNames:
        branch = experiment / method
        for after in ((False, True) if withWildCap else (False,)):
            name = method + ("WildCap" if after else "")
            output = destination / name
            identity = {"method": method, "wildCap": after, "sourceRun": str(source),
                        "sourceHashes": sourceHashes, "textureSha256": digest(branch / "texture.npz"),
                        "viewTransformsSha256": digest(source / "viewTransforms.json"),
                        "commonSupportSha256": digest(experiment / "commonSupport.npz"),
                        "parameters": report["parameters"], "templateManifestSha256": template.manifestSha256,
                        "implementationHashes": codeHashes,
                        "colorInput": "WildCap composite gamma2.2 decoded to linear" if after else "linearAlbedo"}
            if after:
                identity["wildCapMaterialsSha256"] = digest(branch / "wildcapUff/materials.npz")
                identity["wildCapDiagnosticsSha256"] = digest(branch / "wildcapIct/diagnostics.json")
            if output.exists() and any(output.iterdir()):
                asset, metadata = loadVerifiedAsset(output, templateManifest, identity)
                checkMesh(sourceMesh, asset.mesh)
                print(f"Verified existing client asset: {name}", flush=True)
            else:
                texture = readArrays(branch / "texture.npz")
                if after:
                    # Assembly prefers linearAlbedo. Replacing albedo alone
                    # would silently re-export the original pre-WildCap colors.
                    composite = readArrays(branch / "wildcapUff/materials.npz")["baseWithWildCap"]
                    texture["albedo"] = composite
                    texture["linearAlbedo"] = np.asarray(composite, np.float32) ** 2.2
                started = perf_counter()
                assembly = template.assemble(geometry, texture, colorEncoding="gamma2.2")
                exportClientAsset(output, assembly)
                archive = readJson(output / "archive.json")
                metadata = {"schemaVersion": 1, "name": name, "inputIdentity": identity,
                            "archive": {key: archive[key] for key in ("sha256", "bytes")},
                            "assemblyAndExportSeconds": perf_counter() - started}
                writeJson(output / "comparison.json", metadata)
                asset, _ = loadVerifiedAsset(output, templateManifest, identity)
                metadata["identicalClientArrays"] = checkMesh(sourceMesh, asset.mesh)
                writeJson(output / "comparison.json", metadata)
                print(f"Exported client asset: {name}", flush=True)
            branchReport = branchReports[method]
            frontEndSeconds = (sourceStageSeconds.get("delight") or sourceStageSeconds.get("delightSeconds")) if method == "openDelight" else branchReport["shDiagnostics"]["seconds"]
            entries.append({"name": name, "method": method, "wildCap": after,
                            "label": NAMES[method] + (" + WildCap" if after else ""),
                            "directory": f"clientAssets/{name}", "archive": metadata["archive"],
                            "inputIdentity": identity, "frontEndSeconds": frontEndSeconds,
                            "wildCapSeconds": branchReport["wildCapDiagnostics"]["seconds"] if after else None,
                            "wildCapValidTexels": int(readArrays(branch / "wildcapUff/materials.npz")["validMask"].sum()) if after else None,
                            "commonCoverageTexels": report["commonCoverageTexels"]})
    # Recheck the immutable source after the longer assembly/export work.
    for filename, expected in sourceHashes.items():
        if digest(source / filename) != expected:
            raise ValueError(f"Source changed during export: {filename}")
    manifest = {"schemaVersion": 1, "status": "complete" if withWildCap else "frontEndsOnly",
                "sourceRun": str(source), "experiment": str(experiment), "sourceHashes": sourceHashes,
                "templateId": templateManifest["templateId"], "templateHashes": templateManifest["templateHashes"],
                "parameters": report["parameters"], "commonCoverageTexels": report["commonCoverageTexels"],
                "comparison": "Fixed client geometry, geometry-derived normals and preset roughness; diffuse colors only",
                "entries": entries}
    writeJson(experiment / "clientComparison.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--withoutWildCap", action="store_true", help="Export the three front ends before WildCap completes")
    args = parser.parse_args()
    manifest = exportComparison(args.source, args.experiment, not args.withoutWildCap)
    print(json.dumps({"status": manifest["status"], "assetCount": len(manifest["entries"]),
                      "manifest": str(args.experiment.resolve() / "clientComparison.json")}), flush=True)


if __name__ == "__main__":
    main()
