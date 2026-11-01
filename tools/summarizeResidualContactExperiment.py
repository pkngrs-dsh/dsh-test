"""Read-only summary of the frozen residual/contact trial and independent audit."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from runResidualContactExperiment import RUN,read,sha,arrays,verify,plan
from faceAppearance.artifacts import writeJson
OUT=ROOT/'reports/residualContact20261024'

def summarize():
    config=plan();implementation=read(RUN/'implementationHashes.json');verify(implementation['sourceHashes'])
    auditImplementation=read(OUT/'independentAudit/auditImplementation.json')
    verify(auditImplementation['hashes'])
    validatorSha=auditImplementation['hashes'][str(ROOT/'tools/validateResidualContactExperiment.py')]
    cases={};images=0;proof={};local=absoluteChanging=0
    for case in config['cases']:
        context=read(RUN/case/'inputContext.json');verify(context['sourceHashes'])
        auditPath=OUT/'independentAudit'/f'{case}.json';audit=read(auditPath)
        if not audit['validationPassed'] or audit['validatorSha256']!=validatorSha:
            raise ValueError(f'Final independent evidence validation incomplete or failed: {case}')
        proof[str(auditPath)]=sha(auditPath)
        renderPath=OUT/case/'report.json';render=read(renderPath)
        verify(render['inputHashes']);verify(render['renderImplementationHashes'])
        for name,expected in render['images'].items():
            if sha(OUT/case/name)!=expected:raise ValueError('Rendered image changed')
        images+=len(render['images']);proof[str(renderPath)]=sha(renderPath)
        baseline=arrays(RUN/case/'baseline/fullGeometry.npz')['vertices'];branches={}
        for branch in config['branches']:
            directory=RUN/case/branch;report=read(directory/'report.json')
            verify({str(directory/name):expected for name,expected in report['outputHashes'].items()})
            full=arrays(directory/'fullGeometry.npz')['vertices'];changed=bool(np.any(full!=baseline))
            accepted=report['residualCandidateAccepted'];local+=int(accepted);absoluteChanging+=int(accepted and report['absoluteEligible'])
            if accepted!=changed:raise ValueError('Residual acceptance differs from actual changed output')
            entry=dict(status=report['diagnostics']['status'],geometryChanged=changed,incrementalSafe=report['incrementalSafe'],
                absoluteEligible=report['absoluteEligible'],localResidualAccepted=accepted,clientReady=False,
                changedVertexCount=int(np.any(full!=baseline,axis=1).sum()),
                maximumDisplacementNominalMm=float(np.linalg.norm(full-baseline,axis=1).max()/float(arrays(context['entry']['preparedGeometry'])['unit'])),
                newFullPairsAgainstOriginalPureDirect=len(report['newFullPairsAgainstOriginalPureDirect']),
                newFullPairsAgainstCurrentBounded=len(report['newFullPairsAgainstCurrentBounded']),
                newFullPairsAgainstPairedBaseline=len(report['newFullPairsAgainstPairedBaseline']),
                strictLegacyBindingAndPayloadPreserved=True)
            if branch!='baseline':
                fast=report['diagnostics'].get('fast',{})
                entry['apertureBefore']=fast.get('before');entry['apertureAfter']=fast.get('after')
                entry['stages']=[dict(round=s['round'],status=s['status'],activeVertexCount=s['solver']['activeVertexCount'],
                    requestedRadius=s['solver']['fixedBoundary']['requestedSupportRadius'],
                    effectiveRadius=s['solver']['fixedBoundary']['effectiveSupportRadius'],
                    initialPointError=s['solver']['initialMeanErrorPixels'],finalPointError=s['solver']['finalMeanErrorPixels']) for s in fast.get('stages',[])]
            branches[branch]=entry;proof[str(directory/'report.json')]=sha(directory/'report.json')
        cases[case]=dict(branches=branches,independentEvidenceVerified=True,renderedImages=len(render['images']))
    if images!=420:raise ValueError('Incomplete fixed render matrix')
    solverPath=OUT/'solverAudit/summary.json';solver=read(solverPath);proof[str(solverPath)]=sha(solverPath)
    result=dict(status='complete',cases=cases,caseCount=7,branchCount=21,cameraStates=84,imageCount=images,
        localImprovedBranches=local,changingAbsoluteEligibleBranches=absoluteChanging,
        promotedCandidates=0,productionDefaultsChanged=False,realLipsRefined=False,textureRebaked=False,
        planSha256=sha(RUN/'experimentPlan.json'),implementationFileCount=len(implementation['sourceHashes']),
        unitTests=read(RUN/'finalUnitTests.json'),independentAuditTests=auditImplementation['tests'],
        independentValidatorSha256=validatorSha,evidenceHashes=proof,summaryImplementationSha256=sha(__file__),
        conclusions=['Only outdoorSideLight4 has a small accepted local change; two methods return the same local result.',
            'No changed candidate passes original whole-pipeline acceptance. Six other cases retain the exact completed baseline.',
            'All controls still share a radius capped by the closest fixed vertex; actual active support is very small.',
            'Contact constraints preserve their frozen local gap envelope; this experiment does not establish an added visual benefit from contact.',
            'Original vertex order, topology, UVs, fixed attachment coordinates and bindings are preserved.'],
        limitations=['Common template textures isolate geometry; this does not evaluate photo-derived eyebrow or nostril texture quality.',
            'Photo contours are automatic proxies; nominal millimeters are template-derived, not metric scans.',
            'Previously seen regression cases are not independent unseen validation.'])
    verify(proof)
    index=OUT/'index.html';html=index.read_text()
    note='''<p id="experimentOutcome" style="padding:12px;border-left:4px solid #dca85b;background:#252c35">
本轮结果：只有 outdoorSideLight4 的两个分支有相同的小幅局部改善（平均眼孔 IoU 0.645 → 0.652）；其余六组返回完全相同的基线。
没有改变了几何且通过整体验收的新候选。067 的“几何门通过”属于未变化的原基线，不表示优化成功。
完整矩阵：7 组 × 3 分支 × 4 视角；此页使用共同模板贴图，只评估几何。
<a href="result.json">完整结果</a> · <a href="solverAudit/summary.json">形变限制诊断</a></p>'''
    if 'id="experimentOutcome"' not in html:
        html=html.replace('<label>样本',note+'\n<label>样本',1)
        index.write_text(html)
    result['viewerSha256']=sha(index)
    writeJson(OUT/'result.json',result)
    return result

if __name__=='__main__':
    result=summarize();print(json.dumps({k:result[k] for k in ('status','caseCount','branchCount','cameraStates','imageCount','localImprovedBranches','changingAbsoluteEligibleBranches')}))
