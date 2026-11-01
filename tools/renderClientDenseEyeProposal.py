"""Inspect dense-eyelid geometry without exporting a delivery ZIP.

The default proposed mode is explicitly diagnostic. The after mode requires an
accepted local experiment and cross-checks its report, baseline and exported skin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid, fileHash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'runs/liveJaw20261012/comparison/after/legacyAsset.zip')
    parser.add_argument('--source', type=Path, default=ROOT/'runs/liveJaw20261012/source/reconstruction')
    parser.add_argument('--geometry', type=Path, default=ROOT/'runs/denseClientEyes20261012/comparison/geometry.npz')
    parser.add_argument('--geometry-key', choices=('proposed','after'), default='proposed')
    parser.add_argument('--observations', type=Path, default=ROOT/'runs/denseClientEyes20261012/observations')
    parser.add_argument('--photos', type=Path, default=ROOT/'runs/legacyPreviewJaw20261012/preview-609d6c9b63154933b7e3ecd46bb37719')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/denseClientEyes20261012/proposalInspection')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Diagnostic output directory must be empty')
    started = perf_counter()
    import torch
    from pytorch3d.structures import Meshes
    torch.set_num_threads(1)
    inputs = json.loads((args.source/'inputs.json').read_text())
    cameras = json.loads((args.source/'cameras.json').read_text())
    similarity = json.loads((args.source/'directClient/renderedJaw/templateToUff.json').read_text())
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    with np.load(args.geometry, allow_pickle=False) as geom:
        before, proposed, faces = (geom[key].copy() for key in ('before', args.geometry_key, 'faces'))
        fullProposed = geom['proposed'].copy()
    if not np.array_equal(faces, assets.arrays['nicpFaces']):
        raise ValueError('Proposal skin topology differs from client template')
    accepted = args.geometry_key == 'after'
    branchName = 'afterAccepted' if accepted else 'proposedUnaccepted'
    branchLabel = '本地密集眼睑拟合 · 未部署' if accepted else '未采用候选 · 仅诊断'
    status = 'acceptedLocalExperiment' if accepted else 'diagnosticOnlyUnacceptedProposal'
    experiment = None
    acceptanceEvidence = None
    if accepted:
        experimentPath = args.geometry.parent/'report.json'
        experiment = json.loads(experimentPath.read_text())
        scale = experiment.get('selectedScale')
        changed = int(np.any(before != proposed,axis=1).sum())
        if (experiment.get('status') != 'acceptedLocalExperiment'
                or isinstance(scale,bool) or not isinstance(scale,(float,int))
                or not np.isfinite(scale) or not 0 < scale <= 1
                or changed == 0 or experiment.get('changedVertexCount') != changed
                or not np.allclose(proposed,before+scale*(fullProposed-before),rtol=0,atol=1e-12)
                or experiment.get('finalGeometryCheck',{}).get('accepted') is not True
                or experiment.get('newIntersectionPairCount') != 0
                or experiment.get('fixedVerticesExactlyUnchanged') is not True):
            raise ValueError('After geometry must be the report-selected accepted local experiment')
        if (experiment.get('sourceArchiveSha256') != fileHash(args.archive)
                or experiment.get('observationManifestSha256') != fileHash(args.observations/'manifest.json')):
            raise ValueError('Accepted experiment baseline archive or observation manifest hash differs')
        acceptedArchivePath = args.geometry.parent/'after/legacyAsset.zip'
        acceptedAsset = _load(acceptedArchivePath)
        transform = acceptedAsset.manifest['assemblyDiagnostics']['templateToUff']
        if any(not np.array_equal(np.asarray(transform[key]),np.asarray(similarity[key]))
               for key in ('scale','rotation','translation')):
            raise ValueError('Accepted export and saved source use different transforms')
        exportedSkin = ((acceptedAsset.mesh['vertices'][assets.arrays['nicpToFullIds']]*100)
            @ np.asarray(similarity['rotation'])*similarity['scale']+similarity['translation'])
        if not np.allclose(exportedSkin,proposed,rtol=0,atol=1e-12):
            raise ValueError('Selected after geometry differs from the accepted local export')
        acceptanceEvidence = {'reportSha256':fileHash(experimentPath),'selectedScale':scale,
            'changedVertexCount':changed,'acceptedArchiveSha256':fileHash(acceptedArchivePath),
            'acceptedExportMaximumCanonicalError':float(np.abs(exportedSkin-proposed).max())}
    with np.load(args.observations/'view00.npz', allow_pickle=False) as obs:
        curves = obs['eyelidCurvesWorking'].copy()
    item, camera = inputs[0], cameras[0]
    if not camera['valid'] or camera['viewIndex'] != 0:
        raise ValueError('Expected the saved accepted frontal camera')
    photoPath = args.photos/'00.image'
    if fileHash(photoPath) != item['sha256']:
        raise ValueError('Frontal photo does not match reconstruction')
    working, replay = readImage(photoPath, imageSize=item['workingSize'][0],
        cropPolicy=item['requestedCropPolicy'], cropMargin=item['cropMargin'],
        faceDetector=lambda _: item['faceDetection'])
    for key in ('inputToWorking', 'workingSize', 'cropBoxXyxy', 'paddingLeftTopRightBottom'):
        if replay[key] != item[key]:
            raise ValueError(f'Frontal crop replay differs in {key}')
    photo = Image.fromarray(np.rint(working*255).astype(np.uint8))
    renderer = EvidenceRenderer(args.archive, similarity, device=args.device, facesPerPixel=16)
    rotation, translation = (np.asarray(similarity[key]) for key in ('rotation', 'translation'))
    baselineFull = (renderer.source.mesh['vertices']*100) @ rotation * similarity['scale'] + translation
    skinIds = assets.arrays['nicpToFullIds']
    if not np.allclose(baselineFull[skinIds], before, rtol=0, atol=1e-12):
        raise ValueError('Delivered baseline skin differs from proposal baseline')
    fullProposal = baselineFull.copy()
    fullProposal[skinIds] = proposed
    accessoryMask = np.ones(len(fullProposal), bool)
    accessoryMask[skinIds] = False
    if not np.array_equal(fullProposal[accessoryMask], baselineFull[accessoryMask]):
        raise ValueError('Proposal changed an accessory')
    args.output.mkdir(parents=True, exist_ok=True)
    photo.save(args.output/'front-photo.png')
    eyeBoxes = []
    for eye in range(2):
        lo = np.floor(curves[eye].reshape(-1,2).min(0)-[15,17]).astype(int)
        hi = np.ceil(curves[eye].reshape(-1,2).max(0)+[15,17]).astype(int)
        eyeBoxes.append(np.r_[lo,hi])
    roi = np.r_[np.stack(eyeBoxes)[:,:2].min(0), np.stack(eyeBoxes)[:,2:].max(0)]
    factor = 4
    width, height = (roi[2:]-roi[:2])*factor
    pixelMap = np.array([[factor,0,factor*(.5-roi[0])-.5],
                         [0,factor,factor*(.5-roi[1])-.5],[0,0,1]],float)
    zoomCamera = {key:np.asarray(camera[key],float) for key in ('intrinsics','rotation','translation')}
    zoomCamera['intrinsics'] = pixelMap@zoomCamera['intrinsics']
    photoZoom = photo.crop(tuple(roi)).resize((int(width),int(height)),Image.Resampling.LANCZOS)
    photoZoom.save(args.output/'eye-roi-photo-4x.png')
    renders, stats = {}, []
    for name, vertices in [('baseline',baselineFull),(branchName,fullProposal)]:
        renderer.vertices = torch.as_tensor(vertices, dtype=torch.float32, device=args.device)
        renderer.mesh = Meshes(verts=[renderer.vertices],faces=[renderer.faces])
        images, report = renderer.variants(camera,(photo.height,photo.width))
        zoomImages, zoomReport = renderer.variants(zoomCamera,(int(height),int(width)))
        renders[name] = {'full':images,'zoom':zoomImages}
        for mode in ('baseColor','clay'):
            images[mode].save(args.output/f'front-{name}-{mode}.png')
            zoomImages[mode].save(args.output/f'eye-roi-{name}-{mode}-4x.png')
        stats.append({'branch':name,'full':report,'eyeRoi4x':zoomReport})
        print(json.dumps({'branch':name,'seconds':perf_counter()-started}),flush=True)
    labels = ['原照片','当前交付模型',branchLabel]
    panels, names = [], []
    for mode, modeLabel in [('baseColor','原有贴图'),('clay','统一灰模')]:
        panels.extend([photo,solid(renders['baseline']['full'][mode]),solid(renders[branchName]['full'][mode])])
        names.extend([label+' · '+modeLabel for label in labels])
    evidenceGrid(panels,names,columns=3,panelSize=512,
        subtitle=('本地密集眼睑拟合 · 未部署；固定原相机、贴图、眼球和附件' if accepted else
                  '未采用候选检查；固定原相机、贴图、眼球和附件；候选未被采纳')).save(args.output/'front-comparison.png')
    eyePanels, eyeLabels = [], []
    for eye, box in enumerate(eyeBoxes):
        mapped = np.r_[(box[:2]-roi[:2])*factor,(box[2:]-roi[:2])*factor]
        eyeName = '图左眼' if eye==0 else '图右眼'
        for mode, modeLabel in [('baseColor','原有贴图'),('clay','统一灰模')]:
            eyePanels.extend([photoZoom.crop(tuple(mapped)),
                solid(renders['baseline']['zoom'][mode]).crop(tuple(mapped)),
                solid(renders[branchName]['zoom'][mode]).crop(tuple(mapped))])
            eyeLabels.extend([eyeName+' · '+label+' · '+modeLabel for label in labels])
    evidenceGrid(eyePanels,eyeLabels,columns=3,panelSize=380,
        subtitle=('眼区 4 倍光栅；本地密集眼睑拟合 · 未部署；贴图未重新烘焙' if accepted else
                  '眼区 4 倍光栅；未采用候选，仅用于观察可见眼部结构；贴图未重新烘焙')).save(args.output/'eyes-comparison.png')
    writeJson(args.output/'report.json',{
        'status':status,'accepted':accepted,'acceptedLocalExperiment':accepted,'deployed':False,'notDeployableAsset':True,
        'geometryKey':args.geometry_key,'acceptanceEvidence':acceptanceEvidence,
        'seconds':perf_counter()-started,'sourceArchiveSha256':fileHash(args.archive),
        'proposalGeometrySha256':fileHash(args.geometry),'sourceInputsSha256':fileHash(args.source/'inputs.json'),
        'sourceCamerasSha256':fileHash(args.source/'cameras.json'),'photoSha256':item['sha256'],
        'implementationSha256':fileHash(__file__),'device':args.device,
        'proposalChangedSkinVertices':int(np.any(before!=proposed,axis=1).sum()),
        'accessoryVerticesUnchanged':True,'textureUnchanged':True,'cameraRefitted':False,
        'eyeBoxesWorking':eyeBoxes,'renderRoiWorking':roi,'roiRasterScale':factor,
        'roiPixelMap':pixelMap,'roiCameraIntrinsics':zoomCamera['intrinsics'],'renders':stats,
        'notes':[('The after array passed local selection and matches the accepted local archive; it is not deployed.' if accepted else
                  'The proposed array is inspected regardless of the fitter acceptance result.'),
            'All non-skin geometry, UV coordinates, material maps and alpha cutoffs come from the delivered ZIP.',
            'Full renders use the saved 512-pixel image camera. Eye ROI uses its pixel-center-preserving crop at 4x raster resolution.',
            'Texture is fixed for causal geometry inspection; no rebaking, image synthesis or camera fitting.',
            'Visible angular structures alone do not establish a background hole or open eyeball topology; material and face identity require separate audits.']})
    print(json.dumps({'status':status,'output':str(args.output),'seconds':perf_counter()-started}),flush=True)


if __name__=='__main__':
    main()
