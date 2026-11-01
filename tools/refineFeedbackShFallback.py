"""Adapt only unobserved template skin to an existing local SH appearance trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.appearanceObservations import linearToGamma22
from faceAppearance.artifacts import writeJson
from faceAppearance.clientShTemplateBlend import matchTemplateToSh, matchTemplateDetail
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAppearanceVariant import exportLegacyAppearanceVariant
from experimentFeedbackShDelight import loadArrays


def _identity(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': size}


def _relativeInput(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or '\x00' in name or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe input identity path: {name!r}')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Input identity path escapes its root: {name!r}')
    return path


def _checkIdentity(actual, expected, name, *, requireSize=False):
    if (not isinstance(expected, dict)
            or not isinstance(expected.get('sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', expected['sha256']) is None
            or (requireSize and (type(expected.get('bytes')) is not int or expected['bytes'] < 0))):
        raise ValueError(f'Invalid recorded input identity: {name}')
    if (actual['sha256'] != expected['sha256']
            or ('bytes' in expected and actual['bytes'] != expected['bytes'])):
        raise ValueError(f'Input identity mismatch: {name}')


def _verifiedInputs(case, parent):
    reportPath = _relativeInput(parent, 'report.json')
    reportBytes = reportPath.read_bytes()
    report = json.loads(reportBytes)
    if (not isinstance(report, dict) or report.get('status') != 'complete'
            or report.get('sourceTaskId') != case.name):
        raise ValueError('Parent SH trial must be complete and belong to this case')
    if (report.get('geometryReplay') is not None
            or report.get('geometryBaselineArchiveSha256', report.get('sourceArchiveSha256'))
            != report.get('sourceArchiveSha256')):
        raise ValueError('Geometry replay parents are not supported by template fallback refinement')
    reportIdentity = {'sha256': hashlib.sha256(reportBytes).hexdigest(), 'bytes': len(reportBytes)}
    snapshots = {reportPath: reportIdentity}

    def capture(path, expected=None, *, requireSize=False):
        actual = _identity(path)
        if expected is not None:
            _checkIdentity(actual, expected, str(path), requireSize=requireSize)
        if path in snapshots and snapshots[path] != actual:
            raise ValueError(f'Input changed while taking snapshot: {path}')
        snapshots[path] = actual
        return {**actual, 'verification': 'verifiedAgainstParentReport' if expected is not None else 'snapshotOnly'}

    archive = _relativeInput(case, 'asset/legacyAsset.zip')
    archiveIdentity = capture(archive, {'sha256': report.get('sourceArchiveSha256')})
    root = case / 'reconstruction'
    recordedHashes = report.get('sourceHashes')
    required = {'directClient/geometry.npz', 'cameras.json', 'directClient/refinement/mouthState.npz'}
    if not isinstance(recordedHashes, dict) or not required.issubset(recordedHashes):
        raise ValueError('Parent sourceHashes must identify geometry, cameras and mouth state')
    sourceHashes = {}
    for name, digest in recordedHashes.items():
        identity = capture(_relativeInput(root, name), {'sha256': digest})
        sourceHashes[name] = identity['sha256']
    extraName = 'directClient/texture.npz'
    extraExpected = {'sha256': sourceHashes[extraName]} if extraName in sourceHashes else None
    extraIdentity = capture(_relativeInput(root, extraName), extraExpected)
    sourceHashes[extraName] = extraIdentity['sha256']

    recordedArtifacts = report.get('artifactIdentities', {})
    if not isinstance(recordedArtifacts, dict):
        raise ValueError('Parent artifactIdentities must be an object')
    parentArtifacts = {}
    for name in ('baking/clientTexture.npz', 'shImages.npz'):
        expected = recordedArtifacts.get(name)
        if name in recordedArtifacts and expected is None:
            raise ValueError(f'Invalid recorded input identity: {name}')
        parentArtifacts[name] = capture(_relativeInput(parent, name), expected, requireSize=True)
    parentArchive = None
    if 'variant' in report:
        if not isinstance(report['variant'], dict):
            raise ValueError('Invalid parent variant identity')
        parentArchive = capture(_relativeInput(parent, 'asset/legacyAsset.zip'),
                                report['variant'], requireSize=True)
    provenance = {'schemaVersion': 1,
        'parentReport': {**reportIdentity, 'verification': 'snapshotOnly'},
        'parentArtifacts': parentArtifacts, 'parentArchive': parentArchive,
        'sourceArchive': archiveIdentity, 'additionalSourceInputs': {extraName: extraIdentity},
        'snapshotMeaning': 'snapshotOnly records this run\'s actual input bytes; no historical identity was available to verify.'}
    return report, sourceHashes, provenance, snapshots


def _verifySnapshots(snapshots):
    for path, expected in snapshots.items():
        _checkIdentity(_identity(path), expected, str(path), requireSize=True)


def run(caseDirectory, parentDirectory, outputDirectory):
    case, parent, output = map(lambda p: Path(p).resolve(), (caseDirectory,parentDirectory,outputDirectory))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new output directory')
    report, sourceHashes, inputProvenance, snapshots = _verifiedInputs(case, parent)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    root = case / 'reconstruction'
    geometry = loadArrays(root / 'directClient/geometry.npz')
    geometry['uvConvention'] = 'topLeftPixelEdges'
    cameras = json.loads((root / 'cameras.json').read_text())
    mouth = loadArrays(root / 'directClient/refinement/mouthState.npz')
    sh = loadArrays(parent / 'baking/clientTexture.npz')
    od = loadArrays(root / 'directClient/texture.npz')
    images = loadArrays(parent / 'shImages.npz')
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    base = np.clip(assets.materials[0]['baseColor'][...,:3] * report['baking']['templateLinearColorGain'],0,1).astype(np.float32)
    provenance = {key:value for key,value in sh.items() if key != 'linearRGB'}
    provenance['openDelightBoundarySafeSkinDonor'] = od['boundarySafeSkinDonor']
    provenance['openDelightDirectPhotoAlpha'] = od['boundaryDirectPhotoAlpha']
    matched = matchTemplateToSh(base, sh['linearRGB'], od['linearAlbedo'], provenance)
    detail = matchTemplateDetail(matched['linearRGB'], sh['linearRGB'], provenance,
                                 matched['maps']['templateMatchSkinWeight'])
    writeJson(output / 'templateMatch.json', {'color': matched['diagnostics'], 'detail': detail['diagnostics']})
    np.savez_compressed(output / 'fallback.npz', linearRGB=detail['linearRGB'], **matched['maps'])
    _verifySnapshots(snapshots)
    baked = bakeClientTexture(geometry, cameras,
        {'albedoImages':linearToGamma22(images['linearImages']),'validMasks':images['validMasks']},
        detail['linearRGB'], output / 'baking', colmapExecutable=ROOT / '.runtimes/colmap/bin/colmap',
        mouthState=mouth, referenceView=report['referenceViewIndex'], fallbackGain=None, threads=2)
    sameKeys = ('coverageMask','colmapCoverageMask','sourceIndex','faceSourceIndex','frontAlpha',
                'atlasMask','assignedMask','gutter','filterPadding','filterObservedKernelFraction')
    frozen = {key:bool(np.array_equal(sh[key],baked[key])) for key in sameKeys}
    core = sh['coverageMask'] & ~sh['boundaryModified'] & ~baked['boundaryModified']
    frozen['observedCoreColors'] = bool(np.array_equal(sh['linearRGB'][core],baked['linearRGB'][core]))
    if not all(frozen.values()):
        raise ValueError('Template adaptation altered observed-source provenance or core colors: '+str(frozen))
    _verifySnapshots(snapshots)
    variantDiagnostics = {'method':'frontReferenceShWithMatchedTemplateFallback','sourceTaskId':case.name,
                          'referenceViewIndex':report['referenceViewIndex'],'observedCoreUnchanged':True}
    if 'shApplication' in report:
        variantDiagnostics['shApplication'] = report['shApplication']
    variant = exportLegacyAppearanceVariant(case/'asset/legacyAsset.zip',baked['linearRGB'],output/'asset',
        diagnostics=variantDiagnostics)
    _verifySnapshots(snapshots)
    result = {**report,'status':'complete','variant':variant,'parent':str(parent),
              'sourceHashes':sourceHashes,'sourceArchiveSha256':inputProvenance['sourceArchive']['sha256'],
              'inputProvenance':inputProvenance,
              'artifactIdentities':{name:_identity(output/name) for name in
                  ('baking/clientTexture.npz','fallback.npz','templateMatch.json')},
              'baking':baked['diagnostics'],'templateAdaptation':{'color':matched['diagnostics'],'detail':detail['diagnostics']},
              'frozenObservationChecks':frozen,'adaptationSeconds':perf_counter()-started,
              'limitations':report['limitations']+['Unobserved skin color and detail come from adapted template estimates, not new photo evidence.']}
    writeJson(output/'report.json',result)
    print(json.dumps({'stage':'complete','seconds':result['adaptationSeconds'],'archive':variant['archivePath']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.case,args.parent,args.output)
