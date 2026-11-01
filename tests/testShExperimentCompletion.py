"""CPU checks for interrupted WildCap experiment artifacts, without model inference."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


toolPath = Path(__file__).resolve().parents[1] / "tools/runShExperiment.py"
specification = importlib.util.spec_from_file_location("shExperimentCompletionTestTool", toolPath)
experiment = importlib.util.module_from_spec(specification)
specification.loader.exec_module(experiment)


@pytest.fixture
def completedBranch(tmp_path):
    branch = tmp_path / "openDelight"
    for name in ("wildcapIct", "wildcapUff"):
        (branch / name).mkdir(parents=True)
    planned = {"iterations": 599, "resolution": 8, "seed": 2026}
    diagnostic = {"status": "experimental", "iterations": 599, "workingResolution": 8,
                  "outputResolution": 8, "seed": 2026, "inputMethod": branch.name,
                  "lossTrace": [{"photometricSum": 1.25}], "seconds": 2.0}
    report = {"status": "succeeded", "stopAfter": "wildcap", "wildCapDiagnostics": diagnostic}
    (branch / "wildcapIct/diagnostics.json").write_text(json.dumps(diagnostic))
    for name, resolution in (("wildcapIct", 8), ("wildcapUff", 16)):
        rgb = np.full((resolution, resolution, 3), .5, np.float32)
        fields = {"diffuse": rgb, "specular": rgb[..., 0], "validMask": np.ones(rgb.shape[:2], bool)}
        fields["tangentNormal" if name == "wildcapIct" else "baseWithWildCap"] = rgb
        np.savez(branch / name / "materials.npz", **fields)
    return branch, report, planned


def validate(fixture):
    branch, report, planned = fixture
    return experiment.validateCompletedWildCap(branch, report, planned, atlasResolution=16)


def testCompleteArtifactsValidateWithoutModifyingAnyFile(completedBranch):
    branch, report, _ = completedBranch
    before = {path: experiment.hashFile(path) for path in branch.rglob("*") if path.is_file()}
    assert validate(completedBranch) == report["wildCapDiagnostics"]
    assert all(experiment.hashFile(path) == digest for path, digest in before.items())


@pytest.mark.parametrize("key", ["iterations", "resolution", "seed"])
def testOldCompletionCannotBeReusedForChangedPlan(completedBranch, key):
    branch, report, planned = completedBranch
    changed = {**planned, key: planned[key] + 1}
    with pytest.raises(ValueError, match="experiment plan"):
        validate((branch, report, changed))


@pytest.mark.parametrize("filename", ["wildcapIct/materials.npz", "wildcapUff/materials.npz",
                                      "wildcapIct/diagnostics.json"])
def testCompletionMarkerDoesNotHideMissingArtifacts(completedBranch, filename):
    (completedBranch[0] / filename).unlink()
    with pytest.raises(FileNotFoundError, match="every branch"):
        validate(completedBranch)


@pytest.mark.parametrize("side", ["wildcapIct", "wildcapUff"])
@pytest.mark.parametrize("damage", ["truncated", "nonfinite", "wrongShape", "missingField"])
def testUnreadableOrInvalidMaterialsCannotCountAsCompleted(completedBranch, side, damage):
    path = completedBranch[0] / side / "materials.npz"
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:30])
    else:
        with np.load(path) as saved:
            fields = dict(saved)
        if damage == "nonfinite":
            fields["diffuse"][0, 0, 0] = np.nan
        elif damage == "wrongShape":
            fields["diffuse"] = fields["diffuse"][:-1]
        else:
            del fields["diffuse"]
        np.savez(path, **fields)
    with pytest.raises(ValueError, match="Invalid completed WildCap materials"):
        validate(completedBranch)


@pytest.mark.parametrize("damage", ["truncated", "nonfinite", "different"])
def testUncommittedOrInvalidDiagnosticsCannotCountAsCompleted(completedBranch, damage):
    branch, report, _ = completedBranch
    path = branch / "wildcapIct/diagnostics.json"
    if damage == "truncated":
        path.write_text('{"status":')
    else:
        diagnostic = copy.deepcopy(report["wildCapDiagnostics"])
        diagnostic["lossTrace"][0]["photometricSum"] = float("inf") if damage == "nonfinite" else 2.5
        path.write_text(json.dumps(diagnostic))
    with pytest.raises(ValueError, match="diagnostics"):
        validate(completedBranch)
