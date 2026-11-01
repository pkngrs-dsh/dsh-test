"""Local browser bridge for the deployed notebook-v1 service (no GPU imports)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest, urlopen
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[2]
TASK = re.compile(r'preview-[a-f0-9]{32}\Z')
REMOTE = os.environ.get('FACE_APPEARANCE_PREVIEW_REMOTE_URL', 'http://127.0.0.1:7860')
REMOTE_PYTHON = os.environ.get('FACE_APPEARANCE_PREVIEW_REMOTE_PYTHON', 'python3')
REMOTE_HELPER = os.environ.get('FACE_APPEARANCE_PREVIEW_REMOTE_HELPER',
                              '/opt/faceappearance/tools/legacyPreviewRemote.py')
SSH_HOST = os.environ.get('FACE_APPEARANCE_PREVIEW_SSH_HOST', 'faceappearance-remote')
SAMPLE_ROOT = Path(os.environ.get('FACE_APPEARANCE_PREVIEW_SAMPLE_ROOT', ROOT / 'examples/photos'))
SAMPLES = [
    {'id': 'orbit-female', 'label': '环绕基线 · 女模板', 'gender': 'female', 'directory': 'orbit-female'},
    {'id': 'selfie-male', 'label': '室内扭头 · 男模板', 'gender': 'male', 'directory': 'selfie-male'},
]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomicJson(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.chmod(0o600)
    temporary.replace(path)


def httpJson(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = UrlRequest(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            message = json.loads(error.read(65536)).get('message', f'服务返回 HTTP {error.code}')
        except (ValueError, AttributeError):
            message = f'服务返回 HTTP {error.code}'
        raise HTTPException(error.code, message) from None
    except (URLError, TimeoutError, OSError):
        raise HTTPException(502, '无法连接远端服务，请检查网络后继续读取任务') from None


class PreviewBridge:
    def __init__(self, root, remote=REMOTE, sshHost=SSH_HOST, sampleRoot=None, sshConfig=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.remote = remote.rstrip('/')
        self.sshHost = sshHost
        self.sshConfig = str(Path(sshConfig).expanduser().resolve(strict=True)) if sshConfig else None
        self.sampleRoot = Path(sampleRoot or SAMPLE_ROOT).expanduser().resolve()
        self.lock = threading.RLock()
        self.active = set()

    def directory(self, task):
        if not TASK.fullmatch(task):
            raise HTTPException(400, '任务 ID 格式无效')
        return self.root / task

    def read(self, task):
        with self.lock:
            path = self.directory(task) / 'job.json'
            if not path.is_file():
                raise HTTPException(404, '本地没有这个任务')
            return json.loads(path.read_text())

    def update(self, task, **changes):
        with self.lock:
            record = self.read(task)
            record.update(changes, updatedAt=stamp())
            atomicJson(self.directory(task) / 'job.json', record)
            return record

    def public(self, task):
        record = self.read(task)
        keys = ('task_id', 'status', 'phase', 'gender', 'createdAt', 'updatedAt', 'error', 'assetReady',
                'metrics', 'callback', 'callbackReceived', 'callbackFields', 'archive', 'cleanup', 'transportError', 'downloadSource')
        result = {key: record[key] for key in keys if key in record}
        start = datetime.fromisoformat(record['createdAt'])
        end = datetime.fromisoformat(record['finishedAt']) if record.get('finishedAt') else datetime.now(timezone.utc)
        result['elapsedSeconds'] = round((end - start).total_seconds(), 1)
        result['images'] = [{'name': image['name'], 'url': f'/api/jobs/{task}/images/{i}'}
                            for i, image in enumerate(record['images'])]
        return result

    def helper(self, action, task, tarPath=None):
        command = ['ssh', *(['-F', self.sshConfig] if self.sshConfig else []),
                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.sshHost,
                   REMOTE_PYTHON, REMOTE_HELPER, action, task]
        source = open(tarPath, 'rb') if tarPath else subprocess.DEVNULL
        try:
            result = subprocess.run(command, stdin=source, capture_output=True, timeout=180)
        finally:
            if tarPath:
                source.close()
        if result.returncode:
            raise RuntimeError(f'SSH 素材操作失败（{action}），请检查服务端权限')
        return json.loads(result.stdout)

    def relayReady(self):
        # A mounted POST-only route answers GET with 405; absent extensions give 404.
        try:
            httpJson(f'{self.remote}/_integration/callback/' + '0' * 64)
        except HTTPException as error:
            return error.status_code == 405
        return False

    def create(self, images, gender):
        if gender not in ('male', 'female'):
            raise HTTPException(400, '请选择男模板或女模板')
        if not 4 <= len(images) <= 16:
            raise HTTPException(400, '请选择 4–16 张照片')
        if sum(len(data) for _, data in images) > 200 * 1024**2:
            raise HTTPException(400, '照片总大小不能超过 200 MiB')
        hashes = set()
        for name, data in images:
            if not 0 < len(data) <= 25 * 1024**2:
                raise HTTPException(400, '单张照片须小于 25 MiB')
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                raise HTTPException(400, '请移除完全重复的照片')
            hashes.add(digest)
            try:
                with Image.open(io.BytesIO(data)) as photo:
                    if min(photo.size) < 64 or photo.width * photo.height > 32_000_000:
                        raise ValueError
                    photo.verify()
                with Image.open(io.BytesIO(data)) as photo:
                    photo.load()
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                raise HTTPException(400, '照片无法解码或尺寸超限') from None
        with self.lock:
            if len(self.active) >= 2:
                raise HTTPException(429, '本地已有两个任务，请等待完成后再提交')
            task = 'preview-' + uuid.uuid4().hex
            directory = self.directory(task)
            directory.mkdir(mode=0o700)
            metadata = []
            for index, (name, data) in enumerate(images):
                path = directory / f'{index:02d}.image'
                path.write_bytes(data)
                path.chmod(0o600)
                metadata.append({'name': Path(name).name[:160], 'file': path.name})
            atomicJson(directory / 'job.json', {'task_id': task, 'status': 'pending', 'phase': 'uploading',
                       'gender': gender, 'createdAt': stamp(), 'images': metadata,
                       'callback': {'state': 'pending'}, 'assetReady': False})
            self.start(task)
        return self.public(task)

    def start(self, task):
        with self.lock:
            if task in self.active:
                return
            self.active.add(task)
        threading.Thread(target=self.work, args=(task,), name=f'preview-{task[-6:]}', daemon=True).start()

    def recover(self):
        for path in sorted(self.root.glob('preview-*/job.json')):
            try:
                record = json.loads(path.read_text())
                if record['status'] not in ('completed', 'failed') or (record['status'] == 'completed' and not record.get('assetReady')):
                    self.start(record['task_id'])
            except (ValueError, KeyError):
                pass

    def stage(self, task):
        directory, record = self.directory(task), self.read(task)
        if record.get('request'):
            return record['request']
        tarPath = directory / 'inputs.tar'
        try:
            with tarfile.open(tarPath, 'w') as archive:
                for index, item in enumerate(record['images']):
                    archive.add(directory / item['file'], arcname=f'{index:02d}.image', recursive=False)
            staged = self.helper('prepare', task, tarPath)
        finally:
            tarPath.unlink(missing_ok=True)
        request = {**staged, 'task_id': task, 'gender': record['gender']}
        self.update(task, request=request, phase='submitting')
        return request

    def receive(self, task, url, expectedArchive):
        from .legacyPreviewAsset import loadLegacyPreview
        directory = self.directory(task)
        path, temporary = directory / 'asset.zip', directory / 'asset.download'
        with urlopen(url, timeout=12) as response, temporary.open('wb') as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 256 * 1024**2:
                    raise ValueError('asset too large')
                output.write(chunk)
        temporary.chmod(0o600)
        identity = {'sha256': hashlib.sha256(temporary.read_bytes()).hexdigest(), 'sizeBytes': temporary.stat().st_size}
        if identity != expectedArchive:
            temporary.unlink(missing_ok=True)
            raise ValueError('downloaded ZIP identity differs from server record')
        preview = loadLegacyPreview(temporary, task)
        temporary.replace(path)
        atomicJson(directory / 'preview.json', preview)
        self.update(task, assetReady=True, archive=identity)

    def work(self, task):
        terminal = False
        try:
            self.update(task, error=None, finishedAt=None, transportError=None)
            request = self.stage(task)
            # Query first: a restarted browser bridge must not submit a second GPU job.
            try:
                result = httpJson(f'{self.remote}/task/status/{task}')
            except HTTPException as error:
                if error.status_code != 404:
                    raise
                result = httpJson(f'{self.remote}/generate/', request)
            self.update(task, status='processing' if result['status'] == 'processing' else 'pending', phase='queued')
            while True:
                try:
                    result = httpJson(f'{self.remote}/task/status/{task}')
                except HTTPException as error:
                    self.update(task, transportError=str(error.detail))
                    time.sleep(5)
                    continue
                state = result['status']
                self.update(task, status=state, phase='reconstructing' if state == 'processing' else state, transportError=None)
                if state in ('completed', 'failed'):
                    terminal = True
                    if state == 'failed':
                        self.update(task, error=result.get('error', '远端重建失败'))
                    else:
                        self.update(task, phase='downloading')
                        details = self.helper('inspect', task)
                        expectedArchive = details['archive']
                        try:
                            self.receive(task, result['result_url'], expectedArchive)
                        except HTTPError as error:
                            if error.code not in (401, 403):
                                raise
                            # Expiration or storage ACL can prevent reading a signed URL.
                            self.receive(task, f'{self.remote}/tasks/{task}/download', expectedArchive)
                            self.update(task, downloadSource='service_fallback_oss_denied_or_expired')
                        except (URLError, TimeoutError, OSError):
                            self.receive(task, f'{self.remote}/tasks/{task}/download', expectedArchive)
                            self.update(task, downloadSource='service_fallback_oss_connection')
                        else:
                            self.update(task, downloadSource='oss_signed_url')
                    for _ in range(10):
                        details = self.helper('inspect', task)
                        self.update(task, **{key: details[key] for key in ('callback', 'callbackReceived', 'callbackFields', 'metrics', 'failureType', 'failureLocations') if key in details})
                        if details.get('callback', {}).get('state') in ('delivered', 'failed'):
                            break
                        time.sleep(2)
                    if state == 'completed' and details.get('archive') != self.read(task).get('archive'):
                        raise ValueError('downloaded ZIP identity differs from server record')
                    self.update(task, phase=state, finishedAt=stamp())
                    break
                time.sleep(2)
        except Exception as error:
            message = str(error.detail) if isinstance(error, HTTPException) else ('联调操作失败：' + type(error).__name__)
            self.update(task, status='failed', phase='failed', error=message, finishedAt=stamp(), assetReady=False)
        finally:
            if terminal:
                try:
                    self.update(task, cleanup=self.helper('cleanup', task))
                except Exception:
                    self.update(task, cleanup={'error': '临时 OSS 输入清理失败，可重新执行 cleanup'})
            with self.lock:
                self.active.discard(task)


def createApp(root=ROOT / 'runs/legacyPreview', *, remote=REMOTE, sshHost=SSH_HOST, sampleRoot=None, sshConfig=None):
    bridge = PreviewBridge(root, remote, sshHost, sampleRoot, sshConfig)
    app = FastAPI(title='FaceAppearance 本地远端联调')
    app.state.bridge = bridge

    @app.middleware('http')
    async def localhostOnly(request, callNext):
        hostname = request.url.hostname
        origin = request.headers.get('origin')
        if hostname not in ('127.0.0.1', 'localhost', '[::1]', '::1', 'testserver'):
            return JSONResponse({'detail': '仅允许本机访问'}, status_code=403)
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': '拒绝跨站请求'}, status_code=403)
        response = await callNext(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.on_event('startup')
    def startup():
        bridge.recover()

    @app.get('/api/health')
    def health():
        status = httpJson(remote + '/health')
        ready = bridge.relayReady()
        return {**status, 'remoteBaseUrl': remote, 'previewReady': ready,
                'server_accepting_jobs': status['accepting_jobs'], 'accepting_jobs': status['accepting_jobs'] and ready,
                'previewMessage': None if ready else '联调接收器待启用，暂不能提交'}

    def sample(id):
        value = next((entry for entry in SAMPLES if entry['id'] == id), None)
        if value is None:
            raise HTTPException(404, '未知样本')
        return value

    @app.get('/api/samples')
    def samples():
        return {'samples': [{key: entry[key] for key in ('id', 'label', 'gender')} |
                {'images': [{'name': f'{i}.png', 'url': f'/api/samples/{entry["id"]}/images/{i}'} for i in range(1, 5)]}
                for entry in SAMPLES if (bridge.sampleRoot / entry['directory'] / '1.png').is_file()]}

    @app.get('/api/samples/{id}/images/{index}')
    def sampleImage(id: str, index: int):
        if index not in range(1, 5):
            raise HTTPException(404)
        return FileResponse(bridge.sampleRoot / sample(id)['directory'] / f'{index}.png', media_type='image/png')

    @app.post('/api/samples/{id}/jobs')
    async def sampleJob(id: str, request: Request):
        if not await run_in_threadpool(bridge.relayReady):
            raise HTTPException(503, '联调接收器待启用，暂不能提交')
        entry = sample(id)
        data = await request.json()
        images = [(f'{i}.png', (bridge.sampleRoot / entry['directory'] / f'{i}.png').read_bytes()) for i in range(1, 5)]
        return await run_in_threadpool(bridge.create, images, data.get('gender', entry['gender']))

    @app.post('/api/jobs')
    async def createJob(gender: str = Form(...), photos: list[UploadFile] = File(...)):
        if not await run_in_threadpool(bridge.relayReady):
            raise HTTPException(503, '联调接收器待启用，暂不能提交')
        if not 4 <= len(photos) <= 16:
            raise HTTPException(400, '请选择 4–16 张照片')
        images, total = [], 0
        try:
            for photo in photos:
                data = await photo.read(25 * 1024**2 + 1)
                total += len(data)
                if len(data) > 25 * 1024**2 or total > 200 * 1024**2:
                    raise HTTPException(400, '照片超过大小限制')
                images.append((photo.filename or 'photo', data))
            return await run_in_threadpool(bridge.create, images, gender)
        finally:
            for photo in photos:
                await photo.close()

    @app.get('/api/jobs')
    def jobs():
        paths = sorted(bridge.root.glob('preview-*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return {'jobs': [bridge.public(path.parent.name) for path in paths[:50]]}

    @app.get('/api/jobs/{task}')
    def job(task: str):
        return bridge.public(task)

    @app.post('/api/jobs/{task}/resume')
    def resume(task: str):
        bridge.read(task)
        if task not in bridge.active:
            bridge.update(task, status='pending', phase='resuming', error=None, finishedAt=None)
        bridge.start(task)
        return bridge.public(task)

    @app.get('/api/jobs/{task}/images/{index}')
    def jobImage(task: str, index: int):
        record = bridge.read(task)
        if not 0 <= index < len(record['images']):
            raise HTTPException(404)
        path = bridge.directory(task) / record['images'][index]['file']
        with Image.open(path) as photo:
            media = Image.MIME.get(photo.format, 'application/octet-stream')
        return FileResponse(path, media_type=media)

    def asset(task):
        if not bridge.read(task).get('assetReady'):
            raise HTTPException(409, '资产尚未下载完成')
        return bridge.directory(task) / 'asset.zip'

    @app.get('/api/jobs/{task}/download')
    def download(task: str):
        return FileResponse(asset(task), media_type='application/zip', filename=f'{task}.zip')

    @app.get('/api/jobs/{task}/preview')
    def preview(task: str):
        asset(task)
        return json.loads((bridge.directory(task) / 'preview.json').read_text())

    @app.get('/api/jobs/{task}/textures/{index}')
    def texture(task: str, index: int):
        from .legacyPreviewAsset import readLegacyTexture
        return Response(readLegacyTexture(asset(task), index), media_type='image/png')

    app.mount('/viewer-common', StaticFiles(directory=Path(__file__).with_name('web')), name='common')
    app.mount('/', StaticFiles(directory=Path(__file__).with_name('legacyWeb'), html=True), name='preview')
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18093)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/legacyPreview')
    parser.add_argument('--remote', default=REMOTE)
    parser.add_argument('--sshHost', default=SSH_HOST)
    parser.add_argument('--sshConfig', type=Path, help='Explicit SSH client configuration for the local preview bridge')
    parser.add_argument('--sampleRoot', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    uvicorn.run(createApp(args.root, remote=args.remote, sshHost=args.sshHost,
                         sampleRoot=args.sampleRoot, sshConfig=args.sshConfig),
                host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
