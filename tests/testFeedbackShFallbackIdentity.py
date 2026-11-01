"""Local input identity checks; baking and template matching are CPU stubs."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


toolPath = Path(__file__).resolve().parents[1] / 'tools/refineFeedbackShFallback.py'
specification = importlib.util.spec_from_file_location('feedbackShFallbackIdentityTool', toolPath)
experiment = importlib.util.module_from_spec(specification)
specification.loader.exec_module(experiment)


def identity(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


@pytest.fixture
def trial(tmp_path, monkeypatch):
    case, parent, output = (tmp_path / name for name in ('case-one', 'parent', 'new-result'))
    root = case / 'reconstruction'
    (root / 'directClient/refinement').mkdir(parents=True)
    (case / 'asset').mkdir()
    (case / 'asset/legacyAsset.zip').write_bytes(b'verified source fixture')
    np.savez(root / 'directClient/geometry.npz', vertices=np.zeros((3, 3)))
    np.savez(root / 'directClient/refinement/mouthState.npz', mask=np.ones((2, 2), bool))
    (root / 'cameras.json').write_text('[]')
    (root / 'input00.png').write_bytes(b'frozen working image')
    rgb = np.full((2, 2, 3), .3, np.float32)
    mask = np.ones((2, 2), bool)
    np.savez(root / 'directClient/texture.npz', linearAlbedo=rgb,
             boundarySafeSkinDonor=mask, boundaryDirectPhotoAlpha=np.zeros((2, 2)))
    (parent / 'baking').mkdir(parents=True)
    (parent / 'asset').mkdir()
    (parent / 'asset/legacyAsset.zip').write_bytes(b'verified parent fixture')
    sh = {name: mask.copy() for name in (
        'coverageMask', 'colmapCoverageMask', 'sourceIndex', 'faceSourceIndex', 'frontAlpha',
        'atlasMask', 'assignedMask', 'gutter', 'filterPadding', 'filterObservedKernelFraction')}
    sh.update(linearRGB=rgb, boundaryModified=np.zeros((2, 2), bool))
    np.savez(parent / 'baking/clientTexture.npz', **sh)
    np.savez(parent / 'shImages.npz', linearImages=rgb[None], validMasks=mask[None])
    names = ('directClient/geometry.npz', 'directClient/refinement/mouthState.npz', 'cameras.json', 'input00.png')
    report = {'status': 'complete', 'sourceTaskId': case.name,
              'sourceArchiveSha256': identity(case / 'asset/legacyAsset.zip')['sha256'],
              'sourceHashes': {name: identity(root / name)['sha256'] for name in names},
              'variant': identity(parent / 'asset/legacyAsset.zip'), 'referenceViewIndex': 0,
              'baking': {'templateLinearColorGain': [1, 1, 1]}, 'limitations': [],
              'shApplication': {'rawFallbackPixels': 2}}
    (parent / 'report.json').write_text(json.dumps(report))
    calls = {'bake': 0, 'export': 0}
    monkeypatch.setattr(experiment, 'AssemblyAssets', lambda _: SimpleNamespace(materials=[{'baseColor': rgb}]))
    monkeypatch.setattr(experiment, 'matchTemplateToSh', lambda *args: {
        'linearRGB': rgb.copy(), 'maps': {'templateMatchSkinWeight': mask.copy()}, 'diagnostics': {}})
    monkeypatch.setattr(experiment, 'matchTemplateDetail', lambda *args: {'linearRGB': rgb.copy(), 'diagnostics': {}})

    def bake(*args, **kwargs):
        calls['bake'] += 1
        destination = args[4]
        destination.mkdir()
        np.savez(destination / 'clientTexture.npz', **sh)
        return {**sh, 'diagnostics': {'templateLinearColorGain': [1, 1, 1]}}

    def export(source, colors, destination, *, diagnostics):
        calls['export'] += 1
        calls['exportDiagnostics'] = diagnostics
        destination.mkdir()
        archive = destination / 'legacyAsset.zip'
        archive.write_bytes(b'new candidate fixture')
        return {'archivePath': str(archive), **identity(archive)}

    monkeypatch.setattr(experiment, 'bakeClientTexture', bake)
    monkeypatch.setattr(experiment, 'exportLegacyAppearanceVariant', export)
    return SimpleNamespace(case=case, root=root, parent=parent, output=output, report=report, calls=calls)


def saveReport(trial):
    (trial.parent / 'report.json').write_text(json.dumps(trial.report))


def run(trial):
    experiment.run(trial.case, trial.parent, trial.output)
    return json.loads((trial.output / 'report.json').read_text())


@pytest.mark.parametrize('name', ['directClient/geometry.npz', 'cameras.json', 'input00.png'])
def testChangedRecordedSourceFailsBeforeReadingArraysOrBaking(trial, monkeypatch, name):
    (trial.root / name).write_bytes(b'changed after parent report')
    monkeypatch.setattr(experiment, 'loadArrays', lambda _: pytest.fail('read arrays before input validation'))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(trial)
    assert not trial.output.exists()
    assert trial.calls['bake'] == 0


@pytest.mark.parametrize('which', ['source', 'parent'])
def testChangedZipFailsBeforeBaking(trial, which):
    directory = trial.case if which == 'source' else trial.parent
    (directory / 'asset/legacyAsset.zip').write_bytes(b'wrong zip identity')
    with pytest.raises(ValueError, match='identity mismatch'):
        run(trial)
    assert not trial.output.exists()
    assert trial.calls['bake'] == 0


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'directClient/../geometry.npz',
                                 'directClient\\geometry.npz', 'C:/outside', './cameras.json'])
def testUnsafeSourceIdentityPathIsRejected(trial, name):
    trial.report['sourceHashes'][name] = '0' * 64
    saveReport(trial)
    with pytest.raises(ValueError, match='Unsafe input identity path'):
        run(trial)
    assert not trial.output.exists()


def testSourceIdentitySymlinkCannotEscapeReconstruction(trial, tmp_path):
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside bytes')
    (trial.root / 'linked').symlink_to(outside)
    trial.report['sourceHashes']['linked'] = identity(outside)['sha256']
    saveReport(trial)
    with pytest.raises(ValueError, match='escapes its root'):
        run(trial)
    assert not trial.output.exists()


@pytest.mark.parametrize('name', ['baking/clientTexture.npz', 'shImages.npz'])
def testRecordedParentArtifactIdentityDetectsChangedBytes(trial, name):
    path = trial.parent / name
    trial.report['artifactIdentities'] = {name: identity(path)}
    saveReport(trial)
    path.write_bytes(path.read_bytes() + b'changed archive suffix')
    with pytest.raises(ValueError, match='identity mismatch'):
        run(trial)
    assert not trial.output.exists()
    assert trial.calls['bake'] == 0


@pytest.mark.parametrize('historicalIdentities', [False, True])
def testResultIdentifiesActualInputsWithoutClaimingMissingHistory(trial, historicalIdentities):
    if historicalIdentities:
        trial.report['artifactIdentities'] = {name: identity(trial.parent / name)
            for name in ('baking/clientTexture.npz', 'shImages.npz')}
        trial.report['sourceHashes']['directClient/texture.npz'] = identity(trial.root / 'directClient/texture.npz')['sha256']
        saveReport(trial)
    originals = {path: path.read_bytes() for path in trial.parent.rglob('*') if path.is_file()}
    result = run(trial)
    provenance = result['inputProvenance']
    expected = 'verifiedAgainstParentReport' if historicalIdentities else 'snapshotOnly'
    for name, item in provenance['parentArtifacts'].items():
        assert item == {**identity(trial.parent / name), 'verification': expected}
    assert provenance['parentReport'] == {**identity(trial.parent / 'report.json'), 'verification': 'snapshotOnly'}
    extra = provenance['additionalSourceInputs']['directClient/texture.npz']
    assert extra == {**identity(trial.root / 'directClient/texture.npz'), 'verification': expected}
    assert result['sourceHashes']['directClient/texture.npz'] == extra['sha256']
    assert result['sourceArchiveSha256'] == identity(trial.case / 'asset/legacyAsset.zip')['sha256']
    assert result['shApplication'] == trial.report['shApplication']
    assert trial.calls['exportDiagnostics']['shApplication'] == trial.report['shApplication']
    for name, actual in result['artifactIdentities'].items():
        assert actual == identity(trial.output / name)
    assert 'shImages.npz' not in result['artifactIdentities']
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert trial.calls['bake'] == trial.calls['export'] == 1


def testSnapshotChangedAfterLoadIsRejectedBeforeBaking(trial, monkeypatch):
    original = experiment.matchTemplateDetail

    def change(*args):
        path = trial.parent / 'shImages.npz'
        path.write_bytes(path.read_bytes() + b'changed during run')
        return original(*args)

    monkeypatch.setattr(experiment, 'matchTemplateDetail', change)
    with pytest.raises(ValueError, match='identity mismatch'):
        run(trial)
    assert trial.calls['bake'] == trial.calls['export'] == 0
    assert not (trial.output / 'report.json').exists()


def testInputChangedDuringBakeCannotPublishACompletedCandidate(trial, monkeypatch):
    original = experiment.bakeClientTexture

    def change(*args, **kwargs):
        baked = original(*args, **kwargs)
        path = trial.root / 'directClient/texture.npz'
        path.write_bytes(path.read_bytes() + b'changed during bake')
        return baked

    monkeypatch.setattr(experiment, 'bakeClientTexture', change)
    with pytest.raises(ValueError, match='identity mismatch'):
        run(trial)
    assert trial.calls['bake'] == 1
    assert trial.calls['export'] == 0
    assert not (trial.output / 'report.json').exists()


def testExistingResultIsNotOverwritten(trial):
    trial.output.mkdir()
    existing = trial.output / 'report.json'
    existing.write_text('existing result')
    with pytest.raises(FileExistsError, match='new output directory'):
        run(trial)
    assert existing.read_text() == 'existing result'
    assert trial.calls['bake'] == 0


@pytest.mark.parametrize('field,value', [('status', 'running'), ('sourceTaskId', 'other-case')])
def testParentMustBeCompleteAndBelongToCase(trial, field, value):
    trial.report[field] = value
    saveReport(trial)
    with pytest.raises(ValueError, match='complete and belong'):
        run(trial)
    assert not trial.output.exists()


@pytest.mark.parametrize('fields', [
    {'geometryReplay': {'directory': '/another/geometry'}},
    {'geometryBaselineArchiveSha256': 'f' * 64},
])
def testGeometryReplayParentCannotSilentlyUseProductionGeometry(trial, fields):
    trial.report.update(fields)
    saveReport(trial)
    with pytest.raises(ValueError, match='Geometry replay parents are not supported'):
        run(trial)
    assert not trial.output.exists()
    assert trial.calls['bake'] == 0
