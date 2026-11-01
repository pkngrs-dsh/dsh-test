"""Local bridge integration with mocked SSH, OSS and HTTP (no remote jobs)."""

import asyncio
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import pytest

import faceAppearance.legacyPreview as bridgeModule
from testLegacyPreviewAsset import package as legacyPackage


def photos(count=4, *, size=(64, 64)):
    images = []
    for index in range(count):
        stream = io.BytesIO()
        Image.new('RGB', size, (index * 11 % 256, 85, 145)).save(stream, 'PNG')
        images.append((f'{index}.png', stream.getvalue()))
    return images


@pytest.fixture
def localBridge(tmp_path, monkeypatch, legacyPackage):
    bridge = bridgeModule.PreviewBridge(tmp_path / 'bridge')
    monkeypatch.setattr(bridge, 'start', lambda task: None)
    task = bridge.create(photos(), 'male')['task_id']
    archive = Path(legacyPackage[0]).read_bytes()
    identity = {'sha256': hashlib.sha256(archive).hexdigest(), 'sizeBytes': len(archive)}
    actions = []

    def helper(action, requestedTask, tarPath=None):
        assert requestedTask == task
        actions.append(action)
        if action == 'prepare':
            with tarfile.open(tarPath) as tar:
                assert tar.getnames() == [f'{i:02d}.image' for i in range(4)]
            return {'image_urls': [f'https://input.example/{i}.png' for i in range(4)],
                    'notify_url': 'https://callback.example/controlled', 'upload_dir': 'preview/results'}
        if action == 'inspect':
            return {'archive': identity.copy(), 'callback': {'state': 'delivered'},
                    'callbackReceived': True, 'callbackFields': ['task_id', 'status', 'sh_coeffs', 'result_url'],
                    'metrics': {'elapsedSeconds': 7.5}}
        if action == 'cleanup':
            return {'deleted': 4}
        raise AssertionError(action)

    monkeypatch.setattr(bridge, 'helper', helper)
    monkeypatch.setattr(bridgeModule, 'urlopen', lambda *args, **kwargs: io.BytesIO(archive))
    monkeypatch.setattr(bridgeModule.time, 'sleep', lambda seconds: None)
    return bridge, task, identity, actions


def remoteSequence(monkeypatch, states):
    calls = []
    values = iter(states)

    def fakeHttp(url, payload=None):
        calls.append((url, payload))
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(bridgeModule, 'httpJson', fakeHttp)
    return calls


def testNewJobStagesSubmitsPollsDownloadsChecksIdentityAndCleans(localBridge, monkeypatch):
    bridge, task, identity, actions = localBridge
    calls = remoteSequence(monkeypatch, [HTTPException(404, 'unknown'), {'status': 'queued'},
        {'status': 'processing'}, {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    updates = []
    original = bridge.update

    def update(*args, **kwargs):
        updates.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, 'update', update)
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == result['phase'] == 'completed'
    assert result['assetReady'] is True
    assert result['archive'] == identity
    assert result['callbackReceived'] is True
    assert result['cleanup'] == {'deleted': 4}
    assert actions[0] == 'prepare' and actions[-1] == 'cleanup'
    assert actions[1:-1] and set(actions[1:-1]) == {'inspect'}
    assert sum(url.endswith('/generate/') for url, _ in calls) == 1
    submitted = next(payload for url, payload in calls if url.endswith('/generate/'))
    assert set(submitted) == {'task_id', 'gender', 'image_urls', 'upload_dir', 'notify_url'}
    assert submitted['task_id'] == task and submitted['gender'] == 'male'
    assert any(item.get('phase') == 'reconstructing' for item in updates)
    assert not (bridge.directory(task) / 'inputs.tar').exists()
    assert task not in bridge.active
    public = bridge.public(task)
    assert 'request' not in public and 'image_urls' not in json.dumps(public)


def testExistingRemoteTaskIsNeverResubmitted(localBridge, monkeypatch):
    bridge, task, _, actions = localBridge
    bridge.stage(task)
    actions.clear()
    calls = remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    assert bridge.read(task)['status'] == 'completed'
    assert not any(url.endswith('/generate/') for url, _ in calls)
    assert 'prepare' not in actions


def testLostSubmitResponseCanResumeWithoutResubmitting(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    first = remoteSequence(monkeypatch, [HTTPException(404, 'unknown'), HTTPException(502, 'response lost')])
    bridge.work(task)
    assert bridge.read(task)['status'] == 'failed'
    assert sum(url.endswith('/generate/') for url, _ in first) == 1
    second = remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    assert bridge.read(task)['status'] == 'completed'
    assert not any(url.endswith('/generate/') for url, _ in second)


def testExpiredSignedUrlFallsBackToServiceDownload(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    originalOpen = bridgeModule.urlopen
    downloaded = []

    def openWithExpiredFirstUrl(url, *args, **kwargs):
        downloaded.append(url)
        if url == 'https://asset.example/expired.zip':
            raise HTTPError(url, 403, 'expired', {}, None)
        return originalOpen(url, *args, **kwargs)

    monkeypatch.setattr(bridgeModule, 'urlopen', openWithExpiredFirstUrl)
    remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/expired.zip'}])
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == 'completed'
    assert result['downloadSource'] == 'service_fallback_oss_denied_or_expired'
    assert downloaded == ['https://asset.example/expired.zip', f'{bridge.remote}/tasks/{task}/download']


@pytest.mark.parametrize('failure', [URLError('mock OSS connection failure'), TimeoutError('mock OSS timeout')])
def testOssConnectionFailureFallsBackToVerifiedServiceDownload(localBridge, monkeypatch, failure):
    bridge, task, identity, _ = localBridge
    originalOpen = bridgeModule.urlopen
    downloaded = []

    def openWithFailedOss(url, *args, **kwargs):
        downloaded.append(url)
        if url == 'https://asset.example/unreachable.zip':
            raise failure
        return originalOpen(url, *args, **kwargs)

    monkeypatch.setattr(bridgeModule, 'urlopen', openWithFailedOss)
    remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/unreachable.zip'}])
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == 'completed' and result['assetReady'] is True
    assert result['downloadSource'] == 'service_fallback_oss_connection'
    assert result['archive'] == identity
    assert downloaded == ['https://asset.example/unreachable.zip', f'{bridge.remote}/tasks/{task}/download']


def testResumePublishesPendingBeforeWorkerAndClearsPriorFailure(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    bridge.update(task, status='failed', phase='failed', error='previous failure',
                  finishedAt='2026-10-12T00:00:00+00:00', transportError='previous network failure')
    monkeypatch.setattr(bridgeModule, 'PreviewBridge', lambda *args, **kwargs: bridge)
    app = bridgeModule.createApp(bridge.root)
    started = []

    def observeStart(requestedTask):
        record = bridge.read(requestedTask)
        assert record['status'] == 'pending' and record['phase'] == 'resuming'
        assert record['error'] is None and record['finishedAt'] is None
        started.append(requestedTask)

    monkeypatch.setattr(bridge, 'start', observeStart)
    endpoint = next(route.endpoint for route in app.routes if route.path == '/api/jobs/{task}/resume')
    response = endpoint(task)
    assert started == [task]
    assert response['status'] == 'pending' and response['phase'] == 'resuming'
    stage = bridge.stage

    def checkClearedBeforeWork(requestedTask):
        record = bridge.read(requestedTask)
        assert record['error'] is record['finishedAt'] is record['transportError'] is None
        return stage(requestedTask)

    monkeypatch.setattr(bridge, 'stage', checkClearedBeforeWork)
    calls = remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == 'completed' and result['error'] is None
    assert result['transportError'] is None and result['finishedAt'] != '2026-10-12T00:00:00+00:00'
    assert not any(url.endswith('/generate/') for url, _ in calls)


def testStartDoesNotDuplicateAnActiveWorker(tmp_path, monkeypatch):
    bridge = bridgeModule.PreviewBridge(tmp_path / 'bridge')
    started = []
    monkeypatch.setattr(bridgeModule.threading, 'Thread',
                        lambda **kwargs: SimpleNamespace(start=lambda: started.append(kwargs)))
    task = 'preview-' + 'd' * 32
    bridge.start(task)
    bridge.start(task)
    assert len(started) == 1 and bridge.active == {task}


def testRecoverStartsIncompleteTasksOnly(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    started = []
    monkeypatch.setattr(bridge, 'start', started.append)
    bridge.recover()
    assert started == [task]
    started.clear()
    bridge.update(task, status='completed', assetReady=False)
    bridge.recover()
    assert started == [task]
    started.clear()
    bridge.update(task, status='completed', assetReady=True)
    bridge.recover()
    assert started == []
    bridge.update(task, status='failed', assetReady=False)
    bridge.recover()
    assert started == []


def testTemporaryPollFailureDoesNotResubmit(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    calls = remoteSequence(monkeypatch, [{'status': 'processing'}, HTTPException(502, 'network'),
        {'status': 'processing'}, {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    assert bridge.read(task)['status'] == 'completed'
    assert bridge.read(task)['transportError'] is None
    assert not any(url.endswith('/generate/') for url, _ in calls)


def testRemoteFailureDoesNotDownloadButStillCleans(localBridge, monkeypatch):
    bridge, task, _, actions = localBridge
    remoteSequence(monkeypatch, [{'status': 'processing'}, {'status': 'failed', 'error': 'model failure'}])
    monkeypatch.setattr(bridgeModule, 'urlopen', lambda *a, **kw: pytest.fail('failed job downloaded an asset'))
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == 'failed' and result['error'] == 'model failure'
    assert not result['assetReady']
    assert actions[-1] == 'cleanup'


def testMismatchedOuterZipHashNeverPublishesAsset(localBridge, monkeypatch):
    bridge, task, identity, actions = localBridge
    identity['sha256'] = '0' * 64
    remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    result = bridge.read(task)
    assert result['status'] == 'failed'
    assert not result['assetReady'], 'A package rejected by server SHA comparison must not be downloadable'
    assert actions[-1] == 'cleanup'


def testAssetIsNotPublishedBeforeServerIdentityIsChecked(localBridge, monkeypatch):
    bridge, task, _, _ = localBridge
    helper = bridge.helper
    inspected = []

    def inspectBeforeReady(action, *args, **kwargs):
        if action == 'inspect' and not inspected:
            assert not bridge.read(task)['assetReady'], 'ZIP was exposed before its outer identity was verified'
            inspected.append(True)
        return helper(action, *args, **kwargs)

    monkeypatch.setattr(bridge, 'helper', inspectBeforeReady)
    remoteSequence(monkeypatch, [{'status': 'processing'},
        {'status': 'completed', 'result_url': 'https://asset.example/result.zip'}])
    bridge.work(task)
    assert bridge.read(task)['status'] == 'completed'
    assert inspected


@pytest.mark.parametrize('case', ['few', 'many', 'duplicate', 'small', 'bad_image', 'truncated_jpeg', 'bad_gender', 'oversize_one', 'oversize_total'])
def testInputValidationHappensBeforeRemoteWork(tmp_path, monkeypatch, case):
    bridge = bridgeModule.PreviewBridge(tmp_path / 'bridge')
    monkeypatch.setattr(bridge, 'start', lambda task: pytest.fail('invalid input started a task'))
    images, gender = photos(), 'male'
    if case == 'few':
        images = photos(3)
    elif case == 'many':
        images = photos(17)
    elif case == 'duplicate':
        images[1] = images[0]
    elif case == 'small':
        images = photos(size=(63, 64))
    elif case == 'bad_image':
        images[0] = ('bad.png', b'not an image')
    elif case == 'truncated_jpeg':
        jpeg = io.BytesIO()
        Image.new('RGB', (256, 256), (20, 80, 120)).save(jpeg, 'JPEG')
        images[0] = ('truncated.jpg', jpeg.getvalue()[:-200])
    elif case == 'bad_gender':
        gender = 'unknown'
    elif case == 'oversize_one':
        class TooLarge(bytes):
            def __len__(self):
                return 25 * 1024**2 + 1
        images[0] = ('large.png', TooLarge(images[0][1]))
    elif case == 'oversize_total':
        class Large(bytes):
            def __len__(self):
                return 25 * 1024**2
        images = [(name, Large(data)) for name, data in photos(9)]
    with pytest.raises(HTTPException) as error:
        bridge.create(images, gender)
    assert error.value.status_code == 400
    assert list(bridge.root.iterdir()) == []


@pytest.mark.parametrize(('base', 'origin', 'expected'), [
    ('http://127.0.0.1:18093', None, 200),
    ('http://localhost:18093', 'http://localhost:18093', 200),
    ('http://127.0.0.1:18093', 'https://foreign.example', 403),
    ('http://127.0.0.1:18093', 'null', 403),
    ('http://foreign.example', None, 403),
])
def testLocalhostAndOriginPolicy(tmp_path, monkeypatch, base, origin, expected):
    app = bridgeModule.createApp(tmp_path / 'bridge')
    parsed = urlsplit(base)
    headers = [(b'host', parsed.netloc.encode())]
    if origin is not None:
        headers.append((b'origin', origin.encode()))
    request = Request({'type': 'http', 'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
                       'path': '/api/health', 'raw_path': b'/api/health', 'root_path': '',
                       'query_string': b'', 'headers': headers, 'server': (parsed.hostname, parsed.port or 80),
                       'client': ('127.0.0.1', 40000)})

    async def downstream(request):
        return JSONResponse({'status': 'ok'})

    # Exercise the exact middleware without TestClient's cross-thread event loop,
    # whose wakeup pipe is unavailable in this restricted execution environment.
    dispatch = app.user_middleware[0].kwargs['dispatch']
    response = asyncio.run(dispatch(request, downstream))
    assert response.status_code == expected
    if expected == 200:
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['x-content-type-options'] == 'nosniff'


def testBridgeDeploymentSettingsAreConfigurableWithoutChangingSource(tmp_path, monkeypatch):
    settings = {
        'REMOTE_URL': 'https://preview.example/',
        'REMOTE_PYTHON': '/opt/preview-env/bin/python',
        'REMOTE_HELPER': '/opt/preview/tools/legacyPreviewRemote.py',
        'SSH_HOST': 'preview-test-host',
        'SAMPLE_ROOT': str(tmp_path / 'photos'),
    }
    for name, value in settings.items():
        monkeypatch.setenv('FACE_APPEARANCE_PREVIEW_' + name, value)
    spec = importlib.util.spec_from_file_location('configuredPreviewFixture', bridgeModule.__file__)
    configured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configured)
    bridge = configured.PreviewBridge(tmp_path / 'configured')
    assert bridge.remote == 'https://preview.example'
    assert bridge.sampleRoot == tmp_path / 'photos'
    calls = []

    def run(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b'{"status":"mocked"}')

    monkeypatch.setattr(configured.subprocess, 'run', run)
    task = 'preview-' + 'd' * 32
    assert bridge.helper('inspect', task) == {'status': 'mocked'}
    assert calls[0][-5:] == ['preview-test-host', '/opt/preview-env/bin/python',
                             '/opt/preview/tools/legacyPreviewRemote.py', 'inspect', task]


@pytest.fixture
def remoteHelper(tmp_path, monkeypatch):
    monkeypatch.setenv('FACE_APPEARANCE_PREVIEW_SHARED_ROOT', str(tmp_path / 'remote'))
    monkeypatch.setenv('FACE_APPEARANCE_PREVIEW_UPLOAD_PREFIX', 'test-preview')
    monkeypatch.setenv('FACE_APPEARANCE_PREVIEW_PUBLIC_URL', 'https://relay.example/')
    path = Path(__file__).resolve().parents[1] / 'tools/legacyPreviewRemote.py'
    spec = importlib.util.spec_from_file_location('isolatedLegacyPreviewRemote', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.BASE.mkdir()

    def resetInput(images=None):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w') as target:
            for index, (_, data) in enumerate(photos() if images is None else images):
                entry = tarfile.TarInfo(f'{index:02d}.image')
                entry.size = len(data)
                target.addfile(entry, io.BytesIO(data))
        monkeypatch.setattr(helper.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(archive.getvalue())))

    resetInput()
    return helper, resetInput


def testRemoteStagingRetryIsIdempotentAfterLostSshResponse(remoteHelper):
    helper, resetInput = remoteHelper
    task = 'preview-' + 'a' * 32
    first = helper.prepare(task)
    record = json.loads((helper.BASE / 'previewStaging' / task / 'staging.json').read_text())
    token = record['token']
    inputs = helper.BASE / 'integrationCallbacks' / token / 'inputs'
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in inputs.iterdir()}
    registration = helper.BASE / 'integrationCallbacks' / token / 'registration.json'
    expiry = json.loads(registration.read_text())['inputExpiresAt']
    resetInput()
    second = helper.prepare(task)
    assert first == second
    assert record['transport'] == 'relay' and record['prepared'] is True and record['imageCount'] == 4
    assert {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in inputs.iterdir()} == before
    assert len(before) == 4
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in inputs.iterdir())
    assert inputs.stat().st_mode & 0o777 == 0o700
    assert json.loads(registration.read_text())['inputExpiresAt'] >= expiry
    assert first['notify_url'].endswith(f'/_integration/callback/{token}')
    assert first['image_urls'] == [f'https://relay.example/_integration/inputs/{token}/{i}' for i in range(4)]
    assert first['upload_dir'] == 'test-preview/results'


def testRemotePartialStagingFailureCanBeRetried(remoteHelper):
    helper, resetInput = remoteHelper
    task = 'preview-' + 'b' * 32
    bad = photos()
    bad[1] = ('corrupt.png', b'not a valid image')
    resetInput(bad)
    with pytest.raises((ValueError, OSError)):
        helper.prepare(task)
    assert list((helper.BASE / 'integrationCallbacks').rglob('*.image')) == []
    resetInput()
    request = helper.prepare(task)
    assert len(request['image_urls']) == 4
    assert len(list((helper.BASE / 'integrationCallbacks').rglob('*.image'))) == 4


def testRemoteCleanupIsIdempotentAndDoesNotDeleteResults(remoteHelper):
    helper, _ = remoteHelper
    task = 'preview-' + 'c' * 32
    helper.prepare(task)
    record = json.loads((helper.BASE / 'previewStaging' / task / 'staging.json').read_text())
    directory = helper.BASE / 'integrationCallbacks' / record['token']
    helper.save(directory / 'callback.json', {'receivedAt': '2026-10-12T00:00:00Z', 'payload': {'task_id': task}})
    helper.save(helper.BASE / 'results' / task / 'kept.json', {'asset': 'unchanged'})
    assert helper.cleanup(task) == {'deleted': 4, 'transport': 'relay'}
    assert helper.cleanup(task) == {'deleted': 0, 'alreadyCleaned': True}
    assert not (directory / 'inputs').exists()
    assert (directory / 'callback.json').is_file()
    assert (directory / 'registration.json').is_file()
    assert (helper.BASE / 'results' / task / 'kept.json').is_file()


def testRemoteInterruptedWriteRecoveryLeavesNoOrphanPhotos(remoteHelper, monkeypatch):
    helper, resetInput = remoteHelper
    task = 'preview-' + 'e' * 32
    write = Path.write_bytes
    interrupted = []

    def interruptedWrite(path, data):
        if path.name == '01.image' and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt('simulate process interruption')
        return write(path, data)

    monkeypatch.setattr(Path, 'write_bytes', interruptedWrite)
    with pytest.raises(KeyboardInterrupt):
        helper.prepare(task)
    record = json.loads((helper.BASE / 'previewStaging' / task / 'staging.json').read_text())
    assert record['prepared'] is False and record['transport'] == 'relay'
    originalInputs = helper.BASE / 'integrationCallbacks' / record['token'] / 'inputs'
    assert (originalInputs / '00.image').is_file()
    resetInput()
    assert len(helper.prepare(task)['image_urls']) == 4
    assert not originalInputs.exists()
    assert len(list((helper.BASE / 'integrationCallbacks').rglob('*.image'))) == 4


def testOldOssStagingIsReportedWithoutChangingOrDeletingIt(remoteHelper):
    helper, _ = remoteHelper
    task = 'preview-' + 'f' * 32
    path = helper.BASE / 'previewStaging' / task / 'staging.json'
    record = {'keys': ['old-private-input'], 'prepared': True, 'token': 'a' * 64}
    helper.save(path, record)
    with pytest.raises(ValueError, match='old OSS staging'):
        helper.prepare(task)
    result = helper.cleanup(task)
    assert result['deleted'] == 0 and result['retainedPrivateObjects'] == 1
    assert json.loads(path.read_text()) == record


def testCleanupRejectsAPathInsteadOfAToken(remoteHelper):
    helper, _ = remoteHelper
    task = 'preview-' + '9' * 32
    helper.save(helper.BASE / 'previewStaging' / task / 'staging.json',
                {'transport': 'relay', 'token': '../unsafe', 'imageCount': 4, 'prepared': True})
    with pytest.raises(ValueError, match='token'):
        helper.cleanup(task)
