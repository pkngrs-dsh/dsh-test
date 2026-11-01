"""Publish a complete, read-only summary of the frozen seven-case experiment."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from runProtectedFeatureExperiment import RUN,plan,read,sha,verify,PLAN_SHA
from faceAppearance.artifacts import writeJson
OUT=ROOT/'reports/protectedFeature20261024'


def main():
    config=plan();cases=config['cases'];branches=list(config['branches'])
    execution=read(RUN/'batchExecution.json')
    assert len(execution['cases'])==7 and all(len(r['stages'])==2 and all(s['returnCode']==0 for s in r['stages']) for r in execution['cases'])
    verify(read(RUN/'implementationHashes.json')['sourceHashes'])
    independent=read(OUT/'independentAudit/validationSummary.json')
    assert independent['caseCount']==7 and independent['validationPassed']
    source=read(OUT/'sourceAudit/findings.json');eyes=read(OUT/'eyeAudit/summary.json')
    assert source['caseCount']==7 and len(eyes['cases'])==7
    rows=[];imageCount=0
    for case in cases:
        render=read(OUT/case/'report.json')
        assert len(render['views'])==4 and len(render['images'])==72
        verify({str(OUT/case/name):digest for name,digest in render['images'].items()});imageCount+=len(render['images'])
        reports={branch:read(RUN/case/branch/'report.json') for branch in branches}
        rows.append({'caseId':case,'branches':{b:{
            'geometrySafetyAccepted':r['geometrySafetyAccepted'],
            'newFullPairsAgainstPairedBaseline':len(r['newFullPairsAgainstPairedBaseline']),
            'pairedSemanticAlignmentAccepted':r['pairedSemanticAlignment']['accepted'] if r['pairedSemanticAlignment'] else None,
            'allNonSkinExactlyFixed':r['allNonSkinExactlyFixed'],
            'finalCandidateAccepted':r['finalCandidateAccepted'],
            'reportSha256':sha(RUN/case/b/'report.json')} for b,r in reports.items()}})
    counts={b:sum(row['branches'][b]['geometrySafetyAccepted'] for row in rows) for b in branches}
    audits=[OUT/'independentAudit/validationSummary.json',OUT/'sourceAudit/findings.json',OUT/'sourceAudit/index.json',
        OUT/'eyeAudit/summary.json',RUN/'finalUnitTests.json',RUN/'batchExecution.json']
    result={'status':'complete','planSha256':PLAN_SHA,'caseCount':7,'branchCount':28,'cameraStateCount':112,
        'renderedPngCount':imageCount,'unitTests':read(RUN/'finalUnitTests.json'),
        'geometrySafetyAcceptedCounts':counts,'newBranchesAccepted':sum(counts[b] for b in branches if b!='baseline'),
        'finalClientCandidatesAccepted':0,'productionDefaultsChanged':False,'textureRebaked':False,
        'uniformProspectiveParameters':True,'failedCandidatesRetained':True,'commonBaselineAttachments':True,
        'initialSourceAudit':source,'eyeAuditAggregate':eyes['aggregate'],'independentAudit':independent,
        'cases':rows,'auditHashes':{str(p):sha(p) for p in audits},'summaryToolSha256':sha(__file__),
        'interpretation':[
            'Unsupported geometry protection works, but source geometry still violates full safety gates.',
            'Restricting the existing coarse eye solve changes its feasible field and does not preserve its fitting accuracy.',
            'Template materials and baseline attachments isolate geometry; no claim of final adaptive assembly or photographic likeness.',
            'Successful execution, tests and independent reproduction do not establish acceptance of rejected candidates.']}
    writeJson(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ('status','caseCount','branchCount','cameraStateCount','renderedPngCount','geometrySafetyAcceptedCounts','newBranchesAccepted')}))

if __name__=='__main__':main()
