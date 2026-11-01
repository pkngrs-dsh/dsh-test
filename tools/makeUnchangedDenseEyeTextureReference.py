"""Record a texture-only reference after a dense geometry binding rejection.

The actual original ZIP and geometry remain unchanged. Measurement-only
contour gates still apply; no failed fitting gate is relaxed or relabeled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, extractVisibleEyeContours
from runClientDenseEyeExperiment import compactBinding, curveMetrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case):
    case = case.resolve()
    source = case/'baseline/reconstruction'
    archive = case/'baseline/asset/legacyAsset.zip'
    output = case/'textureGeometryBaseline'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty texture geometry reference directory')
    failedBinding = json.loads((case/'dense/baselineBinding.json').read_text())
    rejections = {name:eye['reasons'] for name,eye in failedBinding['eyes'].items() if eye['status'] != 'accepted'}
    if not rejections:
        raise ValueError('This fallback is only for recorded geometry binding rejections')
    output.mkdir(parents=True, exist_ok=True)
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    inputs = json.loads((source/'inputs.json').read_text())
    camera = json.loads((source/'cameras.json').read_text())[0]
    manifest = json.loads((case/'observations/manifest.json').read_text())
    observation = np.load(case/'observations/view00.npz', allow_pickle=False)
    if (manifest['inputsSha256'] != digest(source/'inputs.json')
            or manifest['camerasSha256'] != digest(source/'cameras.json')
            or manifest['views'][0]['arraySha256'] != digest(case/'observations/view00.npz')
            or not observation['candidateEyeValidMask'].all()):
        raise ValueError('Observation identity or fixed quality gates failed')
    photoEyes = {name:{'upper':observation['eyelidCurvesWorking'][i,1],
                      'lower':observation['eyelidCurvesWorking'][i,0]}
                 for i,name in enumerate(('imageLeft','imageRight'))}
    scene = loadLegacyEyeBindingInputs(archive, assets.arrays['nicpToFullIds'])
    original = np.load(source/'directClient/geometry.npz', allow_pickle=False)
    before, faces = original['vertices'], original['faces']
    if not np.allclose(scene['vertices'][assets.arrays['nicpToFullIds']], before, rtol=0, atol=1e-12):
        raise ValueError('Original ZIP differs from source geometry')
    measured = extractVisibleEyeContours(scene['vertices'],scene['faces'],scene['materialIds'],
        assets.arrays['nicpToFullIds'],camera,photoEyes,imageShape=tuple(reversed(inputs[0]['workingSize'])),
        uvs=scene['uvs'],uvFaces=scene['uvFaces'],materials=scene['materials'],alphaTextures=scene['alphaTextures'])
    writeJson(output/'afterBinding.json', compactBinding(measured))
    report = {'status':'unchangedGeometryBindingRejectedTextureOnly',
              'geometryFittingAttempted':False, 'geometryFittingAccepted':False,
              'selectedScale':0., 'changedVertexCount':0,
              'geometryExactlyUnchanged':True, 'sourceArchiveSha256':digest(archive),
              'observationManifestSha256':digest(case/'observations/manifest.json'),
              'originalGeometryBindingRejections':rejections,
              'source':str(source), 'originalArchive':str(archive),
              'measurementOnly':True, 'productionModified':False,
              'textureRebaked':False,
              'scope':'Texture-only input on unchanged stable geometry; dense geometry binding was rejected'}
    if any(eye['status']!='accepted' for eye in measured['eyes'].values()):
        report['status']='textureOnlySkippedMeasurementRejected'
        report['measurementRejections']={name:eye['reasons'] for name,eye in measured['eyes'].items() if eye['status']!='accepted'}
        writeJson(output/'report.json', report)
        return report
    metrics = curveMetrics(measured, photoEyes, float(np.asarray(inputs[0]['inputToWorking'])[0,0]))
    report.update(before=metrics, after=metrics)
    np.savez_compressed(output/'geometry.npz',before=before,after=before.copy(),faces=faces,
                        fixedVertexIds=np.arange(len(before),dtype=np.int64))
    for branch in ('before','after'):
        (output/branch).mkdir()
        shutil.copyfile(archive,output/branch/'legacyAsset.zip')
    report['archiveBytesExactlyUnchanged']=(digest(output/'after/legacyAsset.zip')==digest(archive))
    writeJson(output/'report.json',report)
    implementation=output/'implementation'
    implementation.mkdir()
    shutil.copyfile(__file__,implementation/Path(__file__).name)
    statusPath=case/'denseStatus.json'
    state=json.loads(statusPath.read_text())
    state['textureOnlyReference']={'status':report['status'],'directory':str(output),
                                   'measurementAccepted':True,'denseGeometryFittingAccepted':False}
    writeJson(statusPath,state)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,action='append',required=True)
    args=parser.parse_args()
    for case in args.case:
        report=run(case)
        print(json.dumps({'case':case.name,'status':report['status'],'originalRejections':report['originalGeometryBindingRejections']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
