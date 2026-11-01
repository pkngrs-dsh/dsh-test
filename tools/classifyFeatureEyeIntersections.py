"""Classify recorded full-face intersection differences without recomputation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/featureGeometry20261024'
OUT=ROOT/'reports/featureGeometry20261024/eyeAudit'
TEMPLATE=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',action='append');args=parser.parse_args()
    with np.load(TEMPLATE/'template.npz',allow_pickle=False) as archive:
        faces=archive['faces'];materialIds=archive['materialIds'];mapping=archive['nicpToFullIds'];static=archive['staticMask']
    components=read(TEMPLATE/'components.json');faceComponent=np.full(len(faces),-1,np.int64)
    for index,component in enumerate(components):faceComponent[component['faceIndices']]=index
    if (faceComponent<0).any():raise ValueError('Original component manifest does not cover every full face')
    fixed=np.zeros(int(faces.max())+1,bool);fixed[mapping[static]]=True
    categories={0:'skin',1:'mouth',2:'eye',3:'alpha'}
    def classify(pairs):
        raw=np.asarray(pairs,np.int64).reshape(-1,2)
        if len(raw) and (raw.min()<0 or raw.max()>=len(faces)):raise ValueError('Pair outside original topology')
        counts=Counter();labels=Counter();materialPairs=Counter();staticCounts=Counter()
        for first,second in raw:
            ids=materialIds[[first,second]]
            counts['-'.join(sorted(categories[int(value)] for value in ids))]+=1
            labels[' / '.join(sorted(components[int(faceComponent[face])]['name']+':'+components[int(faceComponent[face])]['label'] for face in (first,second)))]+=1
            materialPairs['-'.join(map(str,sorted(map(int,ids))))]+=1
            skinFaces=[face for face in (first,second) if materialIds[face]==0]
            if skinFaces:
                state='allSkinFacesFullyStatic' if all(fixed[faces[face]].all() for face in skinFaces) else 'includesNonstaticSkin'
                staticCounts[state]+=1
        return dict(pairCount=len(raw),surfaceCategoryCounts=dict(counts),materialPairCounts=dict(materialPairs),
            componentPairCounts=dict(labels),skinStaticCounts=dict(staticCounts),
            classificationOnly=True,intersectionsRecomputed=False)
    cases=args.case or read(RUN/'experimentPlan.json')['cases'];results=[]
    for case in cases:
        root=RUN/case;branches={};pins={}
        baselineFull=None
        if (root/'baseline/fullGeometry.npz').exists():
            with np.load(root/'baseline/fullGeometry.npz',allow_pickle=False) as archive:baselineFull=archive['vertices'].copy()
        contextPath=root/'inputContext.json';unit=None
        if contextPath.exists():
            prepared=Path(read(contextPath)['entry']['preparedGeometry'])
            with np.load(prepared,allow_pickle=False) as archive:unit=float(archive['unit'])
        for name in ('baseline','sourcePrior','visibleEyes','combined'):
            path=root/name/'report.json'
            if not path.exists():branches[name]={'status':'pending'};continue
            report=read(path);pins[str(path)]=sha(path)
            if 'newFullPairsAgainstOriginalPureDirect' not in report:
                branches[name]={'status':report['status']};continue
            with np.load(root/name/'fullGeometry.npz',allow_pickle=False) as archive:
                if not np.array_equal(archive['faces'],faces):raise ValueError('Full face numbering changed')
                full=archive['vertices']
            with np.load(root/name/'fullBeforeEyes.npz',allow_pickle=False) as archive:before=archive['vertices']
            noOp=np.array_equal(full,before)
            row=dict(status='classifiedExistingRecordedPairs',geometrySafetyAccepted=report['geometrySafetyAccepted'],
                versusOriginalPureDirect=classify(report['newFullPairsAgainstOriginalPureDirect']),
                versusCurrentBounded=classify(report['newFullPairsAgainstCurrentBounded']),
                actualEyeOutputEqualsPreEye=bool(noOp),
                attribution=('All listed final differences already exist in this branch preEye assembly, since the eye-stage output equals preEye exactly.' if noOp else
                    'Final differences cannot be attributed to eye proposals without an independently computed preEye pair difference.'))
            if baselineFull is not None and unit is not None:
                componentChanges=[]
                for component in components:
                    ids=np.asarray(component['vertexIndices'],np.int64)
                    delta=(before[ids]-baselineFull[ids])/unit
                    distances=np.linalg.norm(delta,axis=1)
                    componentChanges.append(dict(name=component['name'],label=component['label'],materialId=component['materialId'],
                        changedVertexCount=int(np.any(delta!=0,axis=1).sum()),maximumDisplacementMm=float(distances.max()),
                        medianDisplacementMm=float(np.median(distances)),centroidDisplacementMm=float(np.linalg.norm(delta.mean(0)))))
                row['preEyeComponentMotionFromFactorialBaseline']=componentChanges
                row['componentMotionScope']='Direct original-index array difference, before the new eye solver. No rigid fitting or inferred collision severity.'
            evidencePath=root/name/'eyeEvidence.json'
            if evidencePath.exists():
                pins[str(evidencePath)]=sha(evidencePath);evidence=read(evidencePath)
                row['eyeStatus']=evidence['diagnostics']['status'];attempts=[]
                for proposal in evidence['candidateProposals']:
                    fullGate=proposal['strict'].get('fullAssembly',{})
                    attempts.append(dict(round=proposal['round'],scale=proposal['scale'],accepted=proposal['strict']['accepted'],
                        reasons=proposal['strict']['reasons'],newVersusOwnPreEye=classify(fullGate.get('newPairs',[])),
                        collisionEvaluated='newPairs' in fullGate))
                row['rawEyeBacktracks']=attempts
            branches[name]=row
        result=dict(caseId=case,branches=branches,sourceHashes=pins,
            limitations=['Intersection lists are copied from recorded strict gates, not newly recomputed.',
                'Triangle pair category does not prove penetration severity or visible artifact severity.',
                'Alpha material means eyelash/eyebrow cards. Existing strict full-pair rules include them; none are waived.',
                'Disabling old six-point eye stages changes the downstream input before the new local eye solver. A no-op eye solver is not an unchanged factorial branch.'])
        directory=OUT/case;directory.mkdir(parents=True,exist_ok=True)
        (directory/'intersectionClasses.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False));results.append(result)
    output=dict(cases=results,templateSha256=sha(TEMPLATE/'template.npz'),componentsSha256=sha(TEMPLATE/'components.json'),
        scriptSha256=sha(__file__),method='existing full-face pairs classified by original template materials and connected components')
    (OUT/'intersectionClasses.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False))
    for result in results:
        print(json.dumps({'case':result['caseId'],'branches':{name:{'beforeUnchanged':row.get('actualEyeOutputEqualsPreEye'),
            'finalVsOriginal':row.get('versusOriginalPureDirect',{}).get('surfaceCategoryCounts'),
            'rawEye':[(attempt['scale'],attempt['newVersusOwnPreEye']['surfaceCategoryCounts']) for attempt in row.get('rawEyeBacktracks',[])]}
            for name,row in result['branches'].items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
