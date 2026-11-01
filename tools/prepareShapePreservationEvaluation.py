"""Freeze input-only dev/holdout/domain-control data before shape-prior solves.

Only the unchanged initial direct objective is replayed. No new prior,
candidate geometry, rendering metric or image-based quality result is used to
choose the split. Historical project reuse is explicitly distinguished from
being held out of this shape-preservation experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate
from faceAppearance.geometryRefinement import _meshEdges
from faceAppearance.legacyGeometryVariant import _similarity
from faceAppearance.legacyPreviewAsset import _load
from experimentNoseSurfaceRegistration import arrays, digest

RUN_ROOT = ROOT / 'runs/shapePreservation20261024'
REPORT_ROOT = ROOT / 'reports/shapePreservation20261024'
REGIONS = ('frontImageLeft', 'frontImageRight')
OPTIONS = {'dataWeight': 10000., 'smoothWeight': 3000., 'restWeight': 0.,
           'staticConstraint': 'hard', 'solver': 'exact'}


def photoSetHash(records):
    identities = sorted(record['sha256'] for record in records)
    return hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()


def catalog():
    entries = []
    for shortId in ('110bac9f', '06732111', '10840519'):
        case = next((ROOT / 'reports/feedbackAudit20261024/cases').glob(shortId + '*'))
        entries.append({'caseId': shortId, 'sourceTaskId': case.name, 'split': 'development',
            'sourceDomain': 'humanReportedByUserFeedback',
            'domainEvidence': 'Existing user-feedback real-input grouping retained; not inferred from pixels.',
            'reconstruction': str(case / 'reconstruction'), 'workingPhotos': str(case / 'reconstruction'),
            'sourceArchive': str(ROOT / 'runs/feedbackFix20261024' / shortId / 'realEyelidCandidate/asset/legacyAsset.zip'),
            'previouslyUsedInThisAlgorithmDevelopment': True})
    for caseId in ('highPitch4', 'cyanWall4', 'outdoorSideLight4'):
        baseline = ROOT / 'runs/eyelidTextureBatch20261012' / caseId / 'baseline'
        entries.append({'caseId': caseId, 'split': 'holdout', 'sourceDomain': 'sourceProvenanceUnknown',
            'domainEvidence': 'Original local TestData source paths and bytes are available, but metadata does not explicitly authenticate human versus AI origin. Appearance is not used as provenance evidence.',
            'reconstruction': str(baseline / 'reconstruction'),
            'workingPhotos': str(ROOT / 'runs/uffMultiCase20261011/cases' / caseId / 'reconstruction'),
            'sourceArchive': str(baseline / 'asset/legacyAsset.zip'),
            'previouslyUsedInThisAlgorithmDevelopment': False,
            'historicalProjectUse': 'Previously used for UFF, eye, jaw or rendering experiments; not a globally unseen project dataset.'})
    case = next((ROOT / 'reports/feedbackAudit20261024/cases').glob('28696732*'))
    entries.append({'caseId': '28696732', 'sourceTaskId': case.name, 'split': 'aiDomainControl',
        'sourceDomain': 'aiReportedByUserInput',
        'domainEvidence': 'User-provided AI-input case already identified in earlier feedback; original filenames explicitly refer to generated photorealistic images.',
        'reconstruction': str(case / 'reconstruction'), 'workingPhotos': str(case / 'reconstruction'),
        'sourceArchive': str(ROOT / 'runs/feedbackFix20261024/28696732/eyelidCandidate/asset/legacyAsset.zip'),
        'previouslyUsedInThisAlgorithmDevelopment': True,
        'generalizationRole': 'Different input domain control; never counted as human generalization evidence.'})
    return entries


def inventoryCaches():
    files = subprocess.check_output(['rg', '--files', 'runs', 'reports'], cwd=ROOT, text=True).splitlines()
    groups = {}
    for relative in files:
        if not relative.endswith('/directClient/refinement/clientGeometry.npz'):
            continue
        directory = (ROOT / relative).parents[2]
        inputsPath = directory / 'inputs.json'
        if not inputsPath.exists():
            continue
        inputs = json.loads(inputsPath.read_text())
        identity = photoSetHash(inputs)
        row = groups.setdefault(identity, {'originalPhotoSetSha256': identity,
            'originalPhotoSha256': sorted(record['sha256'] for record in inputs), 'cacheRoots': []})
        row['cacheRoots'].append(str(directory))
    return {'scope': 'All rg-discovered runs/reports caches containing directClient/refinement/clientGeometry.npz and inputs.json',
        'directCacheDirectoryCount': sum(len(row['cacheRoots']) for row in groups.values()),
        'uniqueOriginalPhotoSetCount': len(groups), 'groups': list(groups.values()),
        'deduplication': 'Sorted original-photo SHA256 set; crop and replay duplicates are not independent subjects.',
        'subjectIdentityLimit': 'Disjoint file hashes do not establish distinct human identities or authenticate image provenance.'}


def prepareEntry(entry, templates, mapping, regionTemplate, referenceTransform):
    caseId = entry['caseId']
    output = RUN_ROOT / caseId / 'preparation'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing shape-preservation preparation outputs')
    output.mkdir(parents=True, exist_ok=True)
    reconstruction, photoDirectory, archivePath = map(Path, (entry['reconstruction'], entry['workingPhotos'], entry['sourceArchive']))
    inputsPath, camerasPath = reconstruction / 'inputs.json', reconstruction / 'cameras.json'
    inputs, cameras = json.loads(inputsPath.read_text()), json.loads(camerasPath.read_text())
    if len(inputs) != len(cameras) or len(inputs) < 4:
        raise ValueError('Evaluation requires paired original camera/input metadata and at least four views')
    sourcePath = reconstruction / 'geometry.npz'
    clientPath = reconstruction / 'directClient/refinement/clientGeometry.npz'
    source, client, archive = arrays(sourcePath), arrays(clientPath), _load(archivePath)
    faces, skinIds = templates['nicpFaces'], templates['nicpToFullIds']
    np.testing.assert_array_equal(source['faces'], templates['sourceFaces'])
    np.testing.assert_array_equal(source['uvs'].astype(np.float32), templates['sourceUvs'].astype(np.float32))
    np.testing.assert_array_equal(client['faces'], faces)
    for key in ('faces', 'uvs', 'uvFaces', 'materialIds'):
        np.testing.assert_array_equal(archive.mesh[key], templates[key])
    referenceManifest = json.loads((ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1/manifest.json').read_text())
    if archive.manifest['templateId'] != referenceManifest['templateId'] or archive.manifest['templateHashes'] != referenceManifest['templateHashes']:
        raise ValueError('Evaluation archive uses a different registered template')
    if not all(archive.manifest['binding'][key] for key in ('vertexOrderPreserved', 'faceOrderPreserved', 'uvOrderPreserved', 'uvFaceOrderPreserved', 'materialFaceOrderPreserved')):
        raise ValueError('Evaluation archive does not preserve original binding order')
    photoRecords, verifiedPaths = [], [archivePath, sourcePath, clientPath, inputsPath, camerasPath]
    if photoDirectory != reconstruction:
        for filename in ('inputs.json', 'cameras.json', 'geometry.npz'):
            if digest(photoDirectory / filename) != digest(reconstruction / filename):
                raise ValueError('Working-photo cache has different source geometry, crops or cameras')
            verifiedPaths.append(photoDirectory / filename)
    for index, (record, camera) in enumerate(zip(inputs, cameras)):
        photoPath = photoDirectory / f'input{index:02d}.png'
        with Image.open(photoPath) as photo:
            if list(photo.size) != record['workingSize']:
                raise ValueError('Working photograph dimensions differ from saved camera frame')
        if not camera.get('valid') or camera.get('viewIndex') != index:
            raise ValueError('Evaluation requires valid ordered saved cameras')
        original = Path(record['path'])
        originalVerified = original.exists() and digest(original) == record['sha256']
        if original.exists() and not originalVerified:
            raise ValueError('Original photo bytes differ from saved reconstruction input identity')
        if originalVerified:
            verifiedPaths.append(original)
        verifiedPaths.append(photoPath)
        photoRecords.append({'viewIndex': index, 'originalPath': str(original),
            'originalPhotoSha256': record['sha256'], 'originalFileLocallyVerified': originalVerified,
            'originalOrientedSize': record['orientedSize'], 'workingPhotoPath': str(photoPath),
            'workingPhotoSha256': digest(photoPath), 'workingSize': record['workingSize'],
            'inputToWorking': record['inputToWorking']})
    rest = templates['vertices'][skinIds]
    valid, static = mapping['sampleWeights'] > 0, np.flatnonzero(templates['staticMask'])
    targets = sampleSurface(source['vertices'].astype(float), source['faces'],
        {'triangleIds': mapping['sourceTriangleIds'], 'barycentricWeights': mapping['sourceBarycentricWeights']})
    replay = registerDirectTemplate(rest, faces, mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'],
        targets, valid, static, **OPTIONS)
    rotation, translation, scale = _similarity(archive.manifest)
    for key, value in (('rotation', rotation), ('translation', translation), ('scale', scale)):
        np.testing.assert_allclose(value, replay['transform'][key], rtol=0, atol=1e-12)
    cachedPureDirect = client['directTemplateVertices'] @ rotation * scale + translation
    error = float(np.abs(replay['rawVertices'] - cachedPureDirect).max())
    if error > 1e-12:
        raise ValueError(f'Cached initial registration differs from the common original objective: {error}')
    fullAppearance = archive.mesh['vertices'] * 100 @ rotation * scale + translation
    interiorMask = np.zeros(len(rest), bool)
    extra = {}
    for name in REGIONS:
        for suffix in ('rimVertexIdsOrdered', 'interiorVertexIds', 'faceIds', 'vertexIds'):
            extra[name + '_' + suffix] = regionTemplate[name + '_' + suffix]
        interiorMask[extra[name + '_interiorVertexIds']] = True
        templateInward = referenceTransform['rotation'] @ -regionTemplate[name + '_restPlaneBasis'][:, 2]
        extra[name + '_inwardDirection'] = rotation.T @ templateInward
    edges = _meshEdges(faces)
    eyeMouthIds = np.unique(selectClientLandmarks(loadClientLandmarks(faces), groups=('eyes', 'mouth'))['vertexIndices'])
    np.savez_compressed(output / 'geometry.npz', rawRest=rest, pureDirect=replay['rawVertices'],
        alignedRest=replay['alignedRest'], targetPoints=targets, validMask=valid,
        sampleFaceIds=mapping['sampleNicpFaceIds'], sampleBarycentric=mapping['sampleNicpWeights'],
        faces=faces, fullFaces=archive.mesh['faces'], fullAppearance=fullAppearance, skinIds=skinIds,
        staticIds=static, eyeMouthIds=eyeMouthIds, interiorMask=interiorMask, edges=edges,
        patchFaceMask=interiorMask[faces].any(1), interiorEdgeMask=interiorMask[edges].any(1),
        rotation=rotation, translation=translation, scale=scale, unit=scale / 10., **extra)
    report = {**entry, 'status': 'inputIdentityAndOriginalDirectReplayVerified',
        'newPriorOrCandidateRun': False, 'newAlgorithmResultsUsedForSelection': False,
        'originalPhotoSetSha256': photoSetHash(inputs), 'photos': photoRecords,
        'templateId': archive.manifest['templateId'], 'templateHashes': archive.manifest['templateHashes'],
        'topologyUvMaterialBindingMatchesCommonTemplate': True,
        'fullVertexCount': len(fullAppearance), 'skinVertexCount': len(skinIds),
        'sourceUffVertexCount': len(source['vertices']), 'cameraCount': len(cameras),
        'initialDirectReplay': {'available': True, 'passed': True, 'maximumCanonicalError': error,
            'options': OPTIONS, 'validSampleCount': int(valid.sum()),
            'cachedClientGeometryPath': str(clientPath), 'baselineDiagnostics': replay['diagnostics']},
        'sourceHashes': {str(path): digest(path) for path in verifiedPaths},
        'preparedGeometry': str(output / 'geometry.npz'), 'preparedGeometrySha256': digest(output / 'geometry.npz')}
    writeJson(output / 'report.json', report)
    print(json.dumps({'case': caseId, 'split': entry['split'], 'sourceDomain': entry['sourceDomain'],
        'replayError': error, 'originalsLocallyVerified': sum(row['originalFileLocallyVerified'] for row in photoRecords)}), flush=True)
    return report


def main():
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / 'split.json').exists():
        raise FileExistsError('The evaluation split is already frozen')
    entries = catalog()
    writeJson(REPORT_ROOT / 'cacheInventory.json', inventoryCaches())
    asset = ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1'
    templates, mapping = arrays(asset / 'template.npz'), arrays(asset / 'directRegistration.npz')
    regionPath = ROOT / 'reports/noseCavity20261024/topologyAudit/regions.npz'
    referencePath = ROOT / 'reports/feedbackAudit20261024/referenceGeometryReplay/ai286BoundarySafeChain/templateToUff.npz'
    reports = [prepareEntry(entry, templates, mapping, arrays(regionPath), arrays(referencePath)) for entry in entries]
    sets = {report['caseId']: {photo['originalPhotoSha256'] for photo in report['photos']} for report in reports}
    overlaps = []
    for index, first in enumerate(reports):
        for second in reports[index + 1:]:
            shared = sets[first['caseId']] & sets[second['caseId']]
            if shared:
                overlaps.append({'first': first['caseId'], 'second': second['caseId'], 'sharedPhotoHashes': sorted(shared)})
    if overlaps:
        raise ValueError(f'Frozen split has overlapping original photographs: {overlaps}')
    commonPaths = [asset / 'template.npz', asset / 'directRegistration.npz', asset / 'manifest.json',
        regionPath, referencePath, Path(__file__), ROOT / 'src/faceAppearance/directTemplateRegistration.py',
        ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json']
    split = {'status': 'frozenBeforeNewShapePriorCandidateResults', 'algorithmCandidateRun': False,
        'development': ['110bac9f', '06732111', '10840519'],
        'holdout': ['highPitch4', 'cyanWall4', 'outdoorSideLight4'], 'aiDomainControl': ['28696732'],
        'parameterSelection': 'Choose ONE lowest common predetermined weight passing every development case; freeze it before any holdout solve. Never choose separate per-case weights or tune on holdout.',
        'holdoutExecutionRule': 'If no single weight passes all development cases, do not run holdout for parameter selection; report unresolved development failures.',
        'noCaseSpecificAlgorithmParameters': True, 'originalPhotoHashOverlapPairs': overlaps,
        'humanGeneralizationLimit': 'Three development cases retain the user-reported real-input grouping. The three newly held-out local TestData groups lack explicit human/AI provenance metadata, so confirmed-human holdout availability remains unproven. AI control is not human generalization evidence.',
        'historicalExposureLimit': 'Held out of this shape-preservation algorithm, not unseen throughout the entire project; these caches were previously used for other modules.',
        'commonSourceHashes': {str(path): digest(path) for path in commonPaths},
        'cases': [{key: report[key] for key in ('caseId', 'split', 'sourceDomain', 'domainEvidence',
            'reconstruction', 'workingPhotos', 'sourceArchive', 'originalPhotoSetSha256',
            'preparedGeometry', 'preparedGeometrySha256')} for report in reports]}
    for row in split['cases']:
        path = RUN_ROOT / row['caseId'] / 'preparation/report.json'
        row['preparationReport'], row['preparationReportSha256'] = str(path), digest(path)
    writeJson(RUN_ROOT / 'split.json', split)
    writeJson(REPORT_ROOT / 'inputPreparationReport.json', {**split, 'splitPath': str(RUN_ROOT / 'split.json'),
        'splitSha256': digest(RUN_ROOT / 'split.json'), 'detailedCases': reports})


if __name__ == '__main__':
    main()
