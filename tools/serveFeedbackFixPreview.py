"""Read-only localhost viewer for pinned delivered assets and completed local candidates.

Candidate interface: runs/feedbackFix20261024/<shortId>/<branch>/report.json
must say status=complete, sourceTaskId, sourceArchiveSha256, and
variant={sha256, bytes}. Its ZIP is always <branch>/asset/legacyAsset.zip;
paths supplied inside reports are never used to locate files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from threading import RLock

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.webPreview import buildPreview

AUDIT = ROOT / 'reports/feedbackAudit20261024'
RUNS = ROOT / 'runs/feedbackFix20261024'
REPORTS = ROOT / 'reports/feedbackFix20261024'
WEB = REPORTS / 'previewWeb'
# Same pinned task identities as serveDeliveredAssets.py; never call its
# createApp(), which writes a verification file owned by the old viewer.
CASES = [
    ('110bac9f-1438-460e-acee-012964a2773d', 'IMG_0626 · 真人候选', 0),
    ('28696732-c03c-4855-bde2-fe06f8ccaee7', '生成照片 · AI 候选', 3),
    ('06732111-f6cf-4418-a7c9-5a5a4e814c37', 'IMG_0496 · 真人候选', 0),
    ('10840519-80f3-42ec-b4ce-f28a00ac225f', 'IMG_0718 · 真人候选', 0),
]
TEXTURES = ('head_color.png', 'body_color.png', 'eye_color.png', 'brow_color.png')
BRANCHES = {'sh': ('shAppearance', 'SH 去光候选'),
            'matched': ('shTemplateMatched', 'SH＋补色'),
            'geometry': ('geometryCandidate', '几何候选'),
            'eyelid': ('eyelidCandidate', '眼睑取色修正（同几何）'),
            'cavity': ('noseCavityCandidate', '鼻腔浅恢复（实验）'),
            'rim': ('noseRimCandidate', '左孔缘约束（实验）'),
            'cavityBatch': ('cavityGeneralizationCandidate', '鼻腔通用验证（实验）'),
            'eyelidBatch': ('realEyelidCandidate', '真人眼睑取色（实验）')}
SH_PROTECTED = ('face_nicp.obj', 'face_nicp.mtl', 'head_normals.png',
                'body_color.png', 'brow_color.png', 'eye_color.png')


def _identity(data):
    return hashlib.sha256(data).hexdigest()


def _verified(archive, expectedHash, expectedSize, shortId, variant):
    data = archive.read_bytes()
    digest = _identity(data)
    if digest != expectedHash or len(data) != expectedSize:
        raise ValueError('ZIP SHA-256 or size does not match its recorded identity')
    asset = _load(archive)
    if _identity(archive.read_bytes()) != digest:
        raise ValueError('ZIP changed during package validation')
    full = buildPreview(asset, f'{shortId}-{variant}')
    for material in full['materials']:
        material['texture']['url'] = (f'/api/cases/{shortId}/variants/{variant}/textures/'
            f'{material["index"]}?archive={digest}')
    sourceIds = np.asarray(full['mesh']['sourceVertexIds'])
    positions = np.asarray(full['mesh']['positions']).reshape(-1, 3)
    triangles = np.asarray(full['mesh']['indices']).reshape(-1, 3)
    if not np.array_equal(positions, asset.mesh['vertices'][sourceIds]):
        raise ValueError('Preview changed source vertex coordinates')
    if not np.array_equal(sourceIds[triangles], asset.mesh['faces']):
        raise ValueError('Preview changed triangle order or winding')
    skin = copy.deepcopy(full)
    selected = asset.mesh['materialIds'] == 0
    skin['mesh']['indices'] = triangles[selected].ravel().tolist()
    skin['mesh']['groups'] = [{'start': 0, 'count': int(selected.sum()) * 3, 'materialIndex': 0}]
    skin['counts']['faces'] = int(selected.sum())
    return {'bytes': data, 'sha256': digest, 'asset': asset, 'full': full, 'skin': skin,
            'identity': {'archiveSha256': digest, 'archiveBytes': len(data),
                'packageVerified': True, 'positionsEqualParsedObj': True,
                'triangleOrderAndWindingEqualParsedObj': True,
                'counts': full['counts'], 'skinFaces': int(selected.sum()),
                'normalMapApplied': False}}


def createApp():
    expected = {item['task_id']: item for item in
                json.loads((AUDIT / 'recent20Tasks.json').read_text())['tasks']}
    cases, production, candidateCache = {}, {}, {}
    lock = RLock()
    for taskId, label, photoIndex in CASES:
        shortId = taskId.split('-')[0]
        pinned = expected[taskId]
        identity = pinned['archive']
        production[shortId] = _verified(AUDIT / 'cases' / taskId / 'asset/legacyAsset.zip',
            identity['sha256'], identity['sizeBytes'], shortId, 'production')
        cases[shortId] = {'shortId': shortId, 'taskId': taskId, 'label': label,
                         'defaultPhotoIndex': photoIndex, 'inputNames': pinned['inputFilenames'],
                         'createdAtBeijing': pinned['createdAtBeijing']}

    def checkedCase(shortId):
        if shortId not in cases:
            raise HTTPException(404, 'Unknown task')
        return cases[shortId]

    def candidate(shortId, variant):
        checkedCase(shortId)
        if variant not in BRANCHES:
            raise HTTPException(404, 'Unknown candidate')
        folder, label = BRANCHES[variant]
        directory = RUNS / shortId / folder
        reportPath = directory / 'report.json'
        archivePath = directory / 'asset/legacyAsset.zip'
        with lock:
            if not reportPath.is_file():
                return None, '尚无完成报告'
            try:
                reportBytes = reportPath.read_bytes()
                report = json.loads(reportBytes)
                if report.get('status') != 'complete':
                    return None, '候选尚未完成'
                if report.get('sourceTaskId') != cases[shortId]['taskId']:
                    raise ValueError('Candidate report belongs to a different task')
                if report.get('sourceArchiveSha256') != production[shortId]['sha256']:
                    raise ValueError('Candidate refers to a different delivered ZIP')
                identity = report['variant']
                stat = archivePath.stat()
                key = (shortId, variant, _identity(reportBytes), stat.st_size, stat.st_mtime_ns)
                if key not in candidateCache:
                    value = _verified(archivePath, identity['sha256'], identity['bytes'], shortId, variant)
                    if variant in ('sh', 'matched'):
                        for name in SH_PROTECTED:
                            if value['asset'].payloads[name] != production[shortId]['asset'].payloads[name]:
                                raise ValueError(f'SH candidate changed protected payload: {name}')
                        value['identity']['protectedProductionPayloadsIdentical'] = list(SH_PROTECTED)
                    if variant in ('eyelid', 'eyelidBatch'):
                        parent, reason = candidate(shortId, 'geometry' if variant == 'eyelid' else 'cavityBatch')
                        if parent is None or report['parentArchiveSha256'] != parent['sha256']:
                            raise ValueError('Eyelid candidate does not match its completed parent')
                        protected = [name for name in parent['asset'].payloads
                                     if name not in ('head_color.png', 'face_nicp.png')]
                        for name in protected:
                            if value['asset'].payloads[name] != parent['asset'].payloads[name]:
                                raise ValueError(f'Eyelid texture candidate changed protected payload: {name}')
                        value['identity']['protectedGeometryCandidatePayloadsIdentical'] = protected
                        if value['asset'].manifest['binding'] != parent['asset'].manifest['binding']:
                            raise ValueError('Eyelid texture candidate changed binding metadata')
                        if variant == 'eyelidBatch':
                            value['identity']['diagnosticOnly'] = True
                    if variant in ('cavity', 'rim', 'cavityBatch'):
                        parentVariant = 'eyelid' if variant == 'cavity' else 'cavity'
                        allowedCount = 88 if variant == 'rim' else 69
                        parent, reason = ((production[shortId], None) if variant == 'cavityBatch'
                                          else candidate(shortId, parentVariant))
                        if parent is None or report['parentArchiveSha256'] != parent['sha256']:
                            raise ValueError('Nasal experiment does not match its completed parent baseline')
                        protected = [name for name in parent['asset'].payloads if name != 'face_nicp.obj']
                        for name in protected:
                            if value['asset'].payloads[name] != parent['asset'].payloads[name]:
                                raise ValueError(f'Cavity experiment changed protected payload: {name}')
                        for name in ('faces', 'uvs', 'uvFaces', 'materialIds'):
                            if not np.array_equal(value['asset'].mesh[name], parent['asset'].mesh[name]):
                                raise ValueError('Cavity experiment changed topology, UVs or materials')
                        if value['asset'].manifest['binding'] != parent['asset'].manifest['binding']:
                            raise ValueError('Cavity experiment changed binding metadata')
                        allowed = np.asarray(report['allowedFullVertexIds'])
                        if (allowed.dtype.kind not in 'iu' or allowed.ndim != 1 or len(allowed) != allowedCount
                                or len(np.unique(allowed)) != allowedCount or allowed.min() < 0
                                or allowed.max() >= len(parent['asset'].mesh['vertices'])):
                            raise ValueError('Invalid cavity experiment vertex region')
                        changed = np.flatnonzero(np.any(value['asset'].mesh['vertices'] != parent['asset'].mesh['vertices'], axis=1))
                        if not np.isin(changed, allowed).all() or len(changed) != report['changedVertexCount']:
                            raise ValueError('Cavity experiment changed vertices outside its audited region')
                        value['identity'].update(diagnosticOnly=True, changedVertexCount=len(changed),
                            protectedParentPayloadsIdentical=protected)
                    value['identity'].update(reportStatus='complete', reportSha256=_identity(reportBytes))
                    value['report'] = report
                    for old in [old for old in candidateCache if old[:2] == key[:2]]:
                        del candidateCache[old]
                    candidateCache[key] = value
                return candidateCache[key], None
            except (OSError, ValueError, KeyError, TypeError) as error:
                return None, '候选校验未通过：' + str(error)

    def variantValue(shortId, variant):
        checkedCase(shortId)
        if variant == 'production':
            return production[shortId]
        value, reason = candidate(shortId, variant)
        if value is None:
            raise HTTPException(409, reason)
        return value

    def renderEntries(shortId, renderKind='appearanceRender'):
        if renderKind not in ('appearanceRender', 'matchedRender', 'geometryRender', 'eyelidRender', 'realEyelidRender', 'proportionRender', 'noseRender', 'noseSurfaceRender', 'noseCavityRender', 'noseRimRender', 'cavityGeneralizationRender'):
            return []
        directory = REPORTS / shortId / renderKind
        reportPath = directory / 'report.json'
        if not reportPath.is_file():
            return []
        try:
            report = json.loads(reportPath.read_text())
            if report.get('status') != 'complete':
                return []
        except (OSError, ValueError):
            return []
        if renderKind == 'realEyelidRender':
            names = [('eyeComparison.png', '真人眼睑取色 · 正面局部对照')]
            names += [(f'view{index:02d}Comparison.png', f'真人眼睑取色 · 原相机视角 {index + 1}')
                      for index in range(4)]
            return [{'label': label, 'url': f'/api/cases/{shortId}/renders/{renderKind}/{name}'}
                    for name, label in names if (directory / name).is_file()]
        if renderKind == 'eyelidRender':
            return [{'label': label, 'url': f'/api/cases/{shortId}/renders/{renderKind}/{name}'}
                    for name, label in [('eyeComparison.png', '眼睑局部修正 · 同几何对照'),
                                        ('view03_comparison.png', '眼睑修正 · 正面完整对照')]
                    if (directory / name).is_file()]
        if renderKind == 'noseRender':
            prefix = '鼻部取色对照' if report.get('qualityAccepted') is True else '鼻部实验（未采用）'
            names = [('view03_baseColor_nose.png', f'{prefix} · 正面局部（同几何）'),
                     ('view02_baseColor_nose.png', f'{prefix} · 抬头局部（同几何）')]
            names += [(f'view{index:02d}Comparison.png', f'{prefix} · 视角 {index + 1}') for index in range(4)]
            return [{'label': label, 'url': f'/api/cases/{shortId}/renders/{renderKind}/{name}'}
                    for name, label in names if (directory / name).is_file()]
        if renderKind in ('noseSurfaceRender', 'noseCavityRender', 'noseRimRender', 'cavityGeneralizationRender'):
            prefix = ('鼻腔通用验证（实验）' if renderKind == 'cavityGeneralizationRender' else
                      '左孔缘约束（实验）' if renderKind == 'noseRimRender' else
                      '鼻腔浅恢复（实验）' if renderKind == 'noseCavityRender' else
                      '鼻外表面修复对照' if report.get('qualityAccepted') is True else '鼻外表面实验（未采用）')
            viewLabels = ([(0, '正面'), (3, '抬头')] if renderKind == 'cavityGeneralizationRender'
                          else [(3, '正面'), (2, '抬头')])
            names = [(f'view{view:02d}_{mode}_nose.png', f'{prefix} · {label} · {modeLabel}')
                     for view, label in viewLabels
                     for mode, modeLabel in [('baseColor', '同贴图'), ('clay', '灰模')]]
            names += [(f'view{index:02d}Comparison.png', f'{prefix} · 视角 {index + 1}') for index in range(4)]
            return [{'label': label, 'url': f'/api/cases/{shortId}/renders/{renderKind}/{name}'}
                    for name, label in names if (directory / name).is_file()]
        if renderKind in ('geometryRender', 'proportionRender'):
            prefix = '未采用轮廓实验 · 同贴图对照' if renderKind == 'proportionRender' else '几何＋肤色固定相机'
            return [{'label': f'{prefix} · 视角 {int(p.name[4:6]) + 1}',
                     'url': f'/api/cases/{shortId}/renders/{renderKind}/{p.name}'}
                    for p in sorted(directory.glob('view??Comparison.png'))
                    if re.fullmatch(r'view\d{2}Comparison\.png', p.name) and p.is_file()]
        if renderKind == 'matchedRender':
            combined = [p for p in sorted((REPORTS / shortId).glob('candidateView??.png'))
                        if re.fullmatch(r'candidateView\d{2}\.png', p.name) and p.is_file()]
            if combined:
                return [{'label': f'固定相机三分支 · 视角 {int(p.stem[-2:]) + 1}',
                         'url': f'/api/cases/{shortId}/renders/combined/{p.name}'} for p in combined]
        return [{'label': f'{"SH＋补色" if renderKind == "matchedRender" else "SH"}固定相机 · 视角 {int(p.name[4:6]) + 1}',
                 'url': f'/api/cases/{shortId}/renders/{renderKind}/{p.name}'}
                for p in sorted(directory.glob('view??_comparison.png'))
                if re.fullmatch(r'view\d{2}_comparison\.png', p.name) and p.is_file()]

    app = FastAPI(title='Local feedback candidate preview', docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def localOnly(request, callNext):
        if request.url.hostname not in ('127.0.0.1', 'localhost', 'testserver'):
            return JSONResponse({'detail': 'Local access only'}, status_code=403)
        if request.method not in ('GET', 'HEAD'):
            return JSONResponse({'detail': 'Read-only viewer'}, status_code=405)
        response = await callNext(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/health')
    def health():
        return {'status': 'ok', 'readOnly': True, 'productionCases': len(cases)}

    @app.get('/api/manifest')
    def manifest():
        entries = []
        for shortId, entry in cases.items():
            variants = [{'id': 'production', 'label': '生产交付', 'available': True,
                         **production[shortId]['identity']}]
            for variant, (_, label) in BRANCHES.items():
                value, reason = candidate(shortId, variant)
                variants.append({'id': variant, 'label': label, 'available': value is not None,
                    **(value['identity'] if value is not None else {'reason': reason})})
            comparisons = renderEntries(shortId, 'matchedRender') or renderEntries(shortId)
            comparisons += renderEntries(shortId, 'geometryRender')
            comparisons += renderEntries(shortId, 'eyelidRender')
            comparisons += renderEntries(shortId, 'proportionRender')
            comparisons += renderEntries(shortId, 'noseRender')
            comparisons += renderEntries(shortId, 'noseSurfaceRender')
            comparisons += renderEntries(shortId, 'noseCavityRender')
            comparisons += renderEntries(shortId, 'noseRimRender')
            comparisons += renderEntries(shortId, 'cavityGeneralizationRender')
            comparisons += renderEntries(shortId, 'realEyelidRender')
            entries.append({**entry, 'variants': variants, 'fixedCameraComparisons': comparisons})
        return {'cases': entries, 'readOnly': True,
                'cameraPolicy': 'First asset sets framing; every subsequent swap preserves camera, controls and light rig.'}

    @app.get('/api/cases/{shortId}/variants/{variant}/preview/{scope}')
    def preview(shortId: str, variant: str, scope: str):
        if scope not in ('full', 'skin'):
            raise HTTPException(404, 'Unknown mesh scope')
        return variantValue(shortId, variant)[scope]

    @app.get('/api/cases/{shortId}/variants/{variant}/textures/{index}')
    def texture(shortId: str, variant: str, index: int, archive: str = Query(...)):
        if not 0 <= index < len(TEXTURES):
            raise HTTPException(404, 'Unknown texture')
        value = variantValue(shortId, variant)
        if archive != value['sha256']:
            raise HTTPException(409, 'Candidate changed; refresh the manifest')
        return Response(value['asset'].payloads[TEXTURES[index]], media_type='image/png')

    @app.get('/api/cases/{shortId}/variants/{variant}/download')
    def download(shortId: str, variant: str):
        value = variantValue(shortId, variant)
        return Response(value['bytes'], media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="{shortId}-{variant}.zip"'})

    @app.get('/api/cases/{shortId}/variants/{variant}/report')
    def report(shortId: str, variant: str):
        value = variantValue(shortId, variant)
        return value.get('report', value['identity'])

    @app.get('/api/cases/{shortId}/photos/{index}')
    def photo(shortId: str, index: int):
        entry = checkedCase(shortId)
        if not 0 <= index < len(entry['inputNames']):
            raise HTTPException(404, 'Unknown source image')
        return FileResponse(AUDIT / 'cases' / entry['taskId'] / 'reconstruction' / f'input{index:02d}.png')

    @app.get('/api/cases/{shortId}/renders/{renderKind}/{name}')
    def fixedRender(shortId: str, renderKind: str, name: str):
        checkedCase(shortId)
        entries = renderEntries(shortId, 'matchedRender' if renderKind == 'combined' else renderKind)
        url = f'/api/cases/{shortId}/renders/{renderKind}/{name}'
        if url not in {item['url'] for item in entries}:
            raise HTTPException(404, 'Comparison not complete')
        directory = REPORTS / shortId if renderKind == 'combined' else REPORTS / shortId / renderKind
        return FileResponse(directory / name, media_type='image/png')

    app.mount('/vendor', StaticFiles(directory=ROOT / 'src/faceAppearance/web/vendor'), name='threeVendor')
    app.mount('/', StaticFiles(directory=WEB, html=True), name='feedbackPreview')
    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18095)
    args = parser.parse_args()
    uvicorn.run(createApp(), host='127.0.0.1', port=args.port)
