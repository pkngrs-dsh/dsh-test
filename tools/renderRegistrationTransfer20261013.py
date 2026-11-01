"""Render fixed-camera UFF / transfer / postprocess geometry diagnostics.

No fitting, new correspondences, or texture synthesis occurs here. Skin-only
surfaces have different topology; holes are not an assembled visible-eye mask.
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
from faceAppearance.camera import projectPoints
from faceAppearance.clientLandmarks import loadClientLandmarks, sampleClientLandmarks, selectClientLandmarks
from faceAppearance.data import readImage
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderClientAssembly import fileHash
from runFastFrontEyeExperiment import CASES, paths, npz

BRANCHES = [('uff', '原始 UFF 表面'), ('direct', '纯配准皮肤'),
            ('stable', '稳定版后处理皮肤'), ('final', '快速眼形后处理皮肤')]


class SurfaceRenderer(EvidenceRenderer):
    """Use exactly the existing photo renderer's projection and clay shader."""
    def __init__(self, vertices, faces, *, device='cpu'):
        import torch
        from pytorch3d.structures import Meshes
        self.torch, self.device, self.facesPerPixel = torch, device, 1
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, device=device, dtype=dtype)
        self.vertices = tensor(vertices)
        self.faces = tensor(faces, torch.int64)
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces])
        self.faceUvs = tensor(np.zeros((len(faces), 3, 2)))
        self.materialIds = tensor(np.zeros(len(faces)), torch.int64)
        self.materials = [{'alphaMode': 'OPAQUE'}]
        self.maps = [tensor([[[[.42]], [[.42]], [[.42]], [[1.]]]])]
        self.roughness, self.metallic = tensor([.65]), tensor([0.])


def bounds(points, imageSize, margin):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(-1)]
    lo, hi = points.min(0), points.max(0)
    center, span = (lo + hi) / 2, np.maximum(hi - lo, 8)
    span = span + margin * span.max()
    lo, hi = center - span / 2, center + span / 2
    return [int(max(0, np.floor(lo[0]))), int(max(0, np.floor(lo[1]))),
            int(min(imageSize[0], np.ceil(hi[0]))), int(min(imageSize[1], np.ceil(hi[1])))]


def run(case, meshesPath, output, device):
    start = perf_counter()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    p = paths(case)
    meshes = npz(meshesPath)
    cameras = json.loads((p['source'] / 'cameras.json').read_text())
    inputs = json.loads((p['source'] / 'inputs.json').read_text())
    branches = [(name, label) for name, label in BRANCHES if name + 'Vertices' in meshes]
    if len(branches) < 3:
        raise ValueError('At least original UFF, pure registration, and final meshes required')
    binding = loadClientLandmarks(meshes['clientFaces'])
    renderers = {name: SurfaceRenderer(meshes[name + 'Vertices'],
                    meshes['uffFaces' if name == 'uff' else 'clientFaces'], device=device)
                 for name, _ in branches}
    views, rasterRecords = [], []
    for i, (record, camera) in enumerate(zip(inputs, cameras)):
        if camera.get('valid') is not True or camera.get('viewIndex') != i:
            raise ValueError('Invalid saved camera')
        photoPath = p['photos'] / f'{i:02d}.image'
        if fileHash(photoPath) != record['sha256']:
            raise ValueError('Original photo differs from saved input')
        image, replay = readImage(photoPath, imageSize=record['workingSize'][0],
            cropPolicy=record['requestedCropPolicy'], cropMargin=record['cropMargin'],
            faceDetector=lambda _, saved=record: saved['faceDetection'])
        for key in ('inputToWorking', 'workingSize', 'cropBoxXyxy', 'paddingLeftTopRightBottom'):
            if replay[key] != record[key]:
                raise ValueError(f'Crop replay mismatch: {key}')
        photo = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        photo.save(output / f'view{i}-photo.png')
        projected = {}
        for group in ('eyes', 'mouth', 'nose', 'all'):
            lm = selectClientLandmarks(binding, groups=(group,))
            points = sampleClientLandmarks(meshes['directVertices'], lm)
            projected[group], _ = projectPoints(points, *(np.asarray(camera[key], np.float64)
                for key in ('intrinsics', 'rotation', 'translation')))
        crops = {name: bounds(points, photo.size, .55 if name != 'all' else .15)
                 for name, points in projected.items()}
        crops['whole'] = [0, 0, photo.width, photo.height]
        view = {'id': i, 'photo': f'view{i}-photo.png', 'crops': crops, 'branches': {}}
        panels, labels = [photo], ['原照片 · 保存的 512 裁剪']
        for name, label in branches:
            variants, stats = renderers[name].variants(camera, (photo.height, photo.width))
            clay = variants['clay']
            filename = f'view{i}-{name}.png'
            clay.save(output / filename)
            view['branches'][name] = filename
            panels.append(solid(clay)); labels.append(label)
            rasterRecords.append({'view': i, 'branch': name, **stats})
        for cropName, box in crops.items():
            cropped = [panel.crop(box) for panel in panels]
            evidenceGrid(cropped, labels, columns=len(panels), panelSize=384,
                subtitle=f'{case} · view {i} · {cropName} | 同相机/光照；UFF 与客户端拓扑不同').save(output / f'view{i}-{cropName}-comparison.png')
        views.append(view)
        print(json.dumps({'case': case, 'view': i, 'seconds': perf_counter() - start}), flush=True)
    report = dict(caseId=case, branches=[dict(id=k, label=v) for k, v in branches], views=views,
        cameraRefitted=False, geometryFitted=False, texturesUsed=False,
        sourceMeshes=str(meshesPath.resolve()), sourceMeshesSha256=fileHash(meshesPath),
        camerasSha256=fileHash(p['source'] / 'cameras.json'), inputsSha256=fileHash(p['source'] / 'inputs.json'),
        implementationSha256=fileHash(Path(__file__)), renderRecords=rasterRecords, seconds=perf_counter()-start,
        limitations=['仅原 UFF 网格与客户端 NICP 皮肤，不附加客户端眼球、牙齿、舌和眉睫；洞边界不等于完整装配后可见眼裂/唇缝。',
            'UFF 和客户端拓扑不同。共同灰模消除贴图影响，但不等于真实照片照明或真实三维形状。',
            '相机来自原重建；没有为任何分支重拟合。局部裁剪仅用于展示。',
            '稳定版后处理包含原 landmark 与下巴修正；快速眼形是后续本地候选。'])
    writeJson(output / 'renderReport.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--meshes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args.case, args.meshes, args.output, args.device)
