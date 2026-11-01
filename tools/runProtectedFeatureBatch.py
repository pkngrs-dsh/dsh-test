"""Run independent cases with at most two single-threaded CPU jobs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
from time import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/protectedFeature20261024'
logs=RUN/'logs';logs.mkdir(exist_ok=True)
plan=json.loads((RUN/'experimentPlan.json').read_text())
env=os.environ.copy()
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
def one(case):
    record={'caseId':case,'started':time(),'stages':[]}
    for stage,tool in (('geometry','runProtectedFeatureExperiment.py'),('render','renderProtectedFeatureExperiment.py')):
        print(json.dumps({'case':case,'stage':stage,'status':'started'}),flush=True)
        with (logs/f'{case}_{stage}.log').open('w') as output:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'tools'/tool),'--case',case],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        record['stages'].append({'stage':stage,'returnCode':result.returncode,'finished':time()})
        print(json.dumps({'case':case,'stage':stage,'returnCode':result.returncode}),flush=True)
        if result.returncode:break
    (logs/f'{case}.json').write_text(json.dumps(record,indent=2));return record
with ThreadPoolExecutor(max_workers=2) as executor:
    results=[future.result() for future in as_completed([executor.submit(one,c) for c in plan['cases']])]
(RUN/'batchExecution.json').write_text(json.dumps({'maxConcurrentJobs':2,'device':'cpu','cases':results},indent=2))
sys.exit(int(any(any(s['returnCode'] for s in r['stages']) for r in results)))
