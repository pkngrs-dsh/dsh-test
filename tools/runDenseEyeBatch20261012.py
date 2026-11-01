"""Run unchanged dense-eye observation and fitting tools on prepared local cases.

The reconstruction and original asset are read-only inputs. A rejection is
recorded without lowering detector, visible-binding, or geometry gates. This
driver intentionally does not create a texture candidate or deploy anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case):
    case = case.resolve()
    source = case / 'baseline/reconstruction'
    archive = case / 'baseline/asset/legacyAsset.zip'
    required = [archive, source/'inputs.json', source/'cameras.json',
                source/'directClient/geometry.npz', source/'directClient/renderedJaw/geometry.npz',
                source/'directClient/renderedJaw/templateToUff.json', case/'photos/00.image']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {'case': case.name, 'status': 'baselineNotReady', 'missing': missing}
    record = {'case': case.name, 'status': 'running', 'source': str(source),
              'originalArchive': str(archive), 'originalArchiveSha256': digest(archive),
              'frozenThresholds': True, 'productionModified': False,
              'stages': []}
    statusPath = case/'denseStatus.json'
    write(statusPath, record)
    started = perf_counter()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT/'src') + ':' + str(ROOT/'tools'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for stage, script, args in (
        ('observations', 'cacheClientDenseEyes.py', ['--source', source, '--photos', case/'photos', '--output', case/'observations']),
        ('dense', 'runClientDenseEyeExperiment.py', ['--source', source, '--archive', archive,
            '--observations', case/'observations', '--output', case/'dense'])):
        output = case/stage
        existing = output / ('manifest.json' if stage == 'observations' else 'report.json')
        if existing.is_file():
            report = json.loads(existing.read_text())
            complete = report.get('status') == 'complete' if stage == 'observations' else report.get('status') in ('acceptedLocalExperiment', 'noImprovingSafeCandidate')
            if complete and stage == 'observations':
                complete = (report.get('inputsSha256') == digest(source/'inputs.json')
                            and report.get('camerasSha256') == digest(source/'cameras.json'))
            if complete and stage == 'dense':
                complete = (report.get('sourceArchiveSha256') == digest(archive)
                            and report.get('observationManifestSha256') == digest(case/'observations/manifest.json'))
            if complete:
                record['stages'].append({'stage': stage, 'status': report['status'], 'existing': True, 'report': str(existing)})
                continue
        command = [sys.executable, str(ROOT/'tools'/script), *map(str, args)]
        logPath = case/(stage+'.log')
        tick = perf_counter()
        with logPath.open('w') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        entry = {'stage': stage, 'returncode': completed.returncode, 'seconds': perf_counter()-tick,
                 'log': str(logPath), 'command': command}
        if completed.returncode:
            entry['status'] = 'rejectedOrFailed'
            entry['logTail'] = '\n'.join(logPath.read_text().splitlines()[-12:])
            binding = output/'baselineBinding.json'
            if binding.is_file():
                entry['baselineBinding'] = str(binding)
                data = json.loads(binding.read_text())
                entry['bindingEyes'] = {key: {'status': value.get('status'), 'reasons': value.get('reasons')}
                                        for key, value in data.get('eyes', {}).items()}
            record['stages'].append(entry)
            record['status'] = stage+'RejectedOrFailed'
            record['seconds'] = perf_counter()-started
            write(statusPath, record)
            return record
        report = json.loads(existing.read_text())
        entry.update(status=report['status'], report=str(existing))
        record['stages'].append(entry)
        write(statusPath, record)
    report = json.loads((case/'dense/report.json').read_text())
    record.update(status=report['status'], seconds=perf_counter()-started,
                  selectedScale=report['selectedScale'], changedVertexCount=report['changedVertexCount'],
                  beforeMeanContourDistanceOriginalPixels=report['before']['meanContourDistanceOriginalPixels'],
                  afterMeanContourDistanceOriginalPixels=report['after']['meanContourDistanceOriginalPixels'],
                  originalArchiveUnchanged=digest(archive)==record['originalArchiveSha256'])
    write(statusPath, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, action='append', required=True)
    args = parser.parse_args()
    for case in args.case:
        result = run(case)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
