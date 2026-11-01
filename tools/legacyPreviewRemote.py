#!/usr/bin/env python3
"""SSH-only private photo staging and callback evidence for local integration."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import tarfile
import time

BASE = Path(os.environ.get('FACE_APPEARANCE_PREVIEW_SHARED_ROOT',
                          Path(__file__).resolve().parents[1] / 'shared')).expanduser().resolve()
PREFIX = os.environ.get('FACE_APPEARANCE_PREVIEW_UPLOAD_PREFIX', 'faceappearance-preview').strip('/')
PUBLIC_URL = os.environ.get('FACE_APPEARANCE_PREVIEW_PUBLIC_URL', 'http://127.0.0.1:7860').rstrip('/')
TASK = re.compile(r'preview-[a-f0-9]{32}\Z')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temp.chmod(0o600)
    temp.replace(path)


def prepare(task):
    from PIL import Image
    root = BASE / 'previewStaging' / task
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = root / 'staging.json'
    if existing.is_file():
        saved = json.loads(existing.read_text())
        if saved.get('prepared') and saved.get('transport') == 'relay' and not saved.get('cleaned'):
            # SSH may have completed after the caller lost its connection.
            token = saved['token']
            save(BASE / 'integrationCallbacks' / token / 'registration.json',
                 {'task_id': task, 'inputExpiresAt': time.time() + 7200})
            return {'image_urls': [f'{PUBLIC_URL}/_integration/inputs/{token}/{i}' for i in range(saved['imageCount'])],
                    'notify_url': f'{PUBLIC_URL}/_integration/callback/{saved["token"]}',
                    'upload_dir': f'{PREFIX}/results'}
        if saved.get('transport') != 'relay':
            raise ValueError('old OSS staging must be inspected separately')
        priorToken = saved.get('token', '')
        if re.fullmatch(r'[a-f0-9]{64}', priorToken):
            shutil.rmtree(BASE / 'integrationCallbacks' / priorToken / 'inputs', ignore_errors=True)
    token = secrets.token_hex(32)
    directory = BASE / 'integrationCallbacks' / token
    inputs = directory / 'inputs'
    inputs.mkdir(parents=True, mode=0o700)
    save(root / 'staging.json', {'transport': 'relay', 'token': token, 'prepared': False})
    urls, total, hashes = [], 0, set()
    try:
        with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as archive:
            for index, member in enumerate(archive):
                if index >= 16 or not member.isfile() or not re.fullmatch(r'\d{2}\.image', member.name):
                    raise ValueError('invalid input archive')
                if not 0 < member.size <= 25 * 1024**2:
                    raise ValueError('input image too large')
                total += member.size
                if total > 200 * 1024**2:
                    raise ValueError('input set too large')
                data = archive.extractfile(member).read()
                identity = hashlib.sha256(data).hexdigest()
                if identity in hashes:
                    raise ValueError('duplicate image')
                hashes.add(identity)
                with Image.open(io.BytesIO(data)) as photo:
                    if min(photo.size) < 64 or photo.width * photo.height > 32_000_000:
                        raise ValueError('invalid image dimensions')
                    photo.verify()
                with Image.open(io.BytesIO(data)) as photo:
                    photo.load()
                path = inputs / f'{index:02d}.image'
                path.write_bytes(data)
                path.chmod(0o600)
                urls.append(f'{PUBLIC_URL}/_integration/inputs/{token}/{index}')
        if not 4 <= len(urls) <= 16:
            raise ValueError('need 4–16 images')
        save(directory / 'registration.json', {'task_id': task, 'inputExpiresAt': time.time() + 7200})
        save(root / 'staging.json', {'transport': 'relay', 'imageCount': len(urls), 'token': token, 'prepared': True})
        return {'image_urls': urls, 'notify_url': f'{PUBLIC_URL}/_integration/callback/{token}',
                'upload_dir': f'{PREFIX}/results'}
    except Exception:
        shutil.rmtree(inputs, ignore_errors=True)
        raise


def inspect(task):
    digest = hashlib.sha256(task.encode()).hexdigest()
    path = BASE / 'tasks' / digest / 'job.json'
    record = json.loads(path.read_text()) if path.is_file() else {}
    result = {key: record[key] for key in ('status', 'callback', 'metrics', 'startedAt', 'finishedAt', 'failureType', 'failureLocations') if key in record}
    result['callbackReceived'] = False
    staging = BASE / 'previewStaging' / task / 'staging.json'
    if staging.is_file():
        staged = json.loads(staging.read_text())
        token = staged.get('token')
        if token:
            callback = BASE / 'integrationCallbacks' / token / 'callback.json'
            if callback.is_file():
                value = json.loads(callback.read_text())
                result['callbackReceived'] = True
                result['callbackReceivedAt'] = value.get('receivedAt')
                result['callbackFields'] = sorted(value['payload'])
    if record.get('archive'):
        result['archive'] = {key: record['archive'][key] for key in ('sha256', 'sizeBytes')}
    return result


def cleanup(task):
    path = BASE / 'previewStaging' / task / 'staging.json'
    if not path.is_file():
        return {'deleted': 0}
    record = json.loads(path.read_text())
    if record.get('cleaned'):
        return {'deleted': 0, 'alreadyCleaned': True}
    if record.get('transport') != 'relay':
        return {'deleted': 0, 'error': '旧 OSS 测试输入需要存储账号删除权限', 'retainedPrivateObjects': len(record.get('keys', []))}
    token = record['token']
    if not re.fullmatch(r'[a-f0-9]{64}', token):
        raise ValueError('invalid staging token')
    inputs = BASE / 'integrationCallbacks' / token / 'inputs'
    shutil.rmtree(inputs, ignore_errors=False) if inputs.exists() else None
    record['cleaned'] = True
    save(path, record)
    return {'deleted': record['imageCount'], 'transport': 'relay'}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'inspect', 'cleanup'])
    parser.add_argument('task')
    args = parser.parse_args()
    if not TASK.fullmatch(args.task):
        raise SystemExit('invalid preview task id')
    try:
        print(json.dumps(globals()[args.action](args.task), ensure_ascii=False))
    except Exception as exc:
        # Never echo SDK exceptions: they can include signed URLs and credentials.
        print(json.dumps({'error': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
