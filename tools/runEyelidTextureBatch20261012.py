"""Apply one frozen eyelid texture policy to prepared local cases and render."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]


def run(case, geometry):
    case,geometry=case.resolve(),geometry.resolve()
    source=case/'baseline/reconstruction'
    output=case/'texture'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'tools'),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    common=['--source',str(source),'--observations',str(case/'observations'),'--output',str(output)]
    stages=[('texture',['tools/runClientEyelidTextureExperiment.py','--comparison',str(geometry),
                        '--delight-source',str(source),'--filter-footprint',*common]),
            ('eyeRender',['tools/runClientEyelidTextureExperiment.py','--render-only',*common]),
            ('evidence',['tools/renderClientAppearanceEvidence.py','--comparison',str(output),
                '--source',str(source),'--photos',str(case/'photos'),'--device','cpu'])]
    report={'caseId':case.name,'status':'running','geometrySource':str(geometry),
            'output':str(output),'stages':[],'frozenTextureParameters':True,'deployed':False}
    tick=perf_counter()
    def write(): (case/'textureStatus.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write()
    for name,argv in stages:
        start=perf_counter()
        log=case/(name+'Run.log')
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,*argv],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        row={'stage':name,'exitCode':result.returncode,'seconds':perf_counter()-start,'log':str(log)}
        if result.returncode: row['logTail']='\n'.join(log.read_text().splitlines()[-10:])
        report['stages'].append(row);write()
        print(json.dumps({'case':case.name,**row},ensure_ascii=False),flush=True)
        if result.returncode:
            report['status']='failed';write();return
    report.update(status='completeLocalCandidate',seconds=perf_counter()-tick)
    write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--geometry-comparison',type=Path,required=True)
    args=parser.parse_args()
    run(args.case,args.geometry_comparison)
