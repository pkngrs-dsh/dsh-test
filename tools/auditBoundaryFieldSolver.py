"""Read frozen solver evidence without fitting, rasterizing or intersecting meshes.

Only scalar aggregation, saved-array comparisons, SVD of the existing small
basis, and template face-ID lookups are performed. Triangle-pair identities do
not establish that penetration became physically worse or visually apparent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/boundaryField20261024'
OUTPUT=ROOT/'reports/boundaryField20261024/solverAudit/summary.json'


def audit(output):
    hashes={}
    def sha(path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()
    def pin(path):
        path=Path(path).resolve();value=sha(path)
        if str(path) in hashes and hashes[str(path)]!=value:raise ValueError('Input changed during audit')
        hashes[str(path)]=value;return path
    def read(path):return json.loads(pin(path).read_text())
    def archive(path):return np.load(pin(path),allow_pickle=False)

    codeHash=sha(__file__)
    plan=read(RUN/'experimentPlan.json');manifest=read(RUN/'implementationHashes.json')
    if sha(RUN/'experimentPlan.json')!=manifest['planSha256']:raise ValueError('Frozen plan identity lost')
    templateRoot=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
    components=read(templateRoot/'components.json');materials=read(templateRoot/'manifest.json')['materials']
    with archive(templateRoot/'template.npz') as data:
        faces=data['faces'];materialIds=data['materialIds']
    labels={}
    for component in components:
        for face in component['faceIndices']:
            if face in labels:raise ValueError('Template component memberships overlap')
            labels[face]=component['label']
    if len(labels)!=len(faces):raise ValueError('Template components do not cover every full face')
    def classify(first,second):
        a,b=labels[first],labels[second]
        if a=='head' and b=='head':return 'skin-skin'
        if a!='head' and b!='head':return 'other-nonSkinPair'
        other=second if a=='head' else first;label=labels[other]
        if materials[int(materialIds[other])]['alphaMode']=='MASK':return 'skin-alpha'
        if label in ('leftEye','rightEye'):return 'skin-eye'
        if label=='headAccessory':return 'skin-canthus'
        return 'skin-otherOpaque'

    pairClassNames=('skin-skin','skin-canthus','skin-eye','skin-alpha','skin-otherOpaque','other-nonSkinPair')
    aggregateGn=Counter();aggregateCombinations=Counter();aggregateOuter=Counter();aggregatePairs=Counter({name:0 for name in pairClassNames})
    rows=[]
    for case in plan['cases']:
        folder=RUN/case;context=read(folder/'inputContext.json')
        with archive(context['entry']['preparedGeometry']) as data:unit=float(data['unit'])
        reports={branch:read(folder/branch/'report.json') for branch in ('legacyResidualReplay','boundaryFieldContact')}
        fieldRows=[];gn=Counter();combinations=Counter();rounds=[]
        for branch in ('legacyResidualReplay','boundaryFieldContact'):
            directory=folder/branch;report=reports[branch]
            evidence=read(directory/'solverEvidence.json')
            with archive(directory/'solverEvidence.npz') as saved:
                for index,roundEvidence in enumerate(evidence):
                    item=roundEvidence['evidence'];basis=saved[item['basis']['arrayKey']]
                    singular=np.linalg.svd(basis,compute_uv=False)
                    rankTolerance=max(basis.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0.
                    norm=np.linalg.norm(basis,axis=1)
                    stage=report['diagnostics']['fast']['stages'][index]['solver']
                    fieldRows.append(dict(branch=branch,round=roundEvidence['round'],controlCount=basis.shape[1],
                        activeRows=int(np.any(basis!=0,axis=1).sum()),meaningfulRows=int((norm>norm.max(initial=0)*1e-8).sum()),
                        numericalRank=int((singular>rankTolerance).sum()),rankTolerance=float(rankTolerance),
                        requestedRadius=stage['fixedBoundary']['requestedSupportRadius'],effectiveRadius=stage['fixedBoundary']['effectiveSupportRadius'],
                        initialPointMeanPixels=stage['initialMeanErrorPixels'],finalPointMeanPixels=stage['finalMeanErrorPixels'],
                        bestIteration=stage['bestIteration']))
                    if branch!='boundaryFieldContact':continue
                    acceptedNorms=[];acceptedMeans=[]
                    for trial in item['trials']:
                        safety=trial['safety'];additional=safety.get('additionalSafety',{})
                        failures=[]
                        if not safety['geometricSafetyAccepted']:failures.append('geometry')
                        if not additional.get('contact',{}).get('accepted',True):failures.append('contact')
                        if not additional.get('coarseAlignment',{}).get('accepted',True):failures.append('coarse')
                        combinations['+'.join(failures) or 'allSafetyPass']+=1
                        gn[trial['decision']]+=1
                        for reason in failures:gn[reason+'Failed']+=1
                        if trial['decision']=='acceptedStep':
                            difference=saved[trial['trialCoefficients']['arrayKey']]-saved[trial['beforeCoefficients']['arrayKey']]
                            acceptedNorms.append(float(np.linalg.norm(difference)))
                            acceptedMeans.append(trial['evaluation']['meanErrorPixels'])
                    finalCoefficients=saved[item['finalCoefficients']['arrayKey']]
                    outputVertices=saved[item['outputVertices']['arrayKey']]
                    referenceVertices=saved[item['referenceVertices']['arrayKey']]
                    rounds.append(dict(round=roundEvidence['round'],trialCount=len(item['trials']),
                        acceptedStepCount=len(acceptedNorms),zeroAcceptedCoefficientStepCount=sum(value==0 for value in acceptedNorms),
                        minimumAcceptedCoefficientStepNorm=min(acceptedNorms) if acceptedNorms else None,
                        maximumAcceptedCoefficientStepNorm=max(acceptedNorms) if acceptedNorms else None,
                        bestAcceptedPointMeanPixels=min(acceptedMeans) if acceptedMeans else None,
                        finalCoefficientsExactlyZero=bool(np.count_nonzero(finalCoefficients)==0),
                        savedSolverOutputExactlyReference=bool(np.array_equal(outputVertices,referenceVertices)),
                        initialObjective=item['initialObjective'],initialObjectiveComponents=item['initialObjectiveComponents'],
                        finalEvaluation=item['finalEvaluation'],bestIteration=stage['bestIteration'],
                        acceptedStepMeaning='An inner GN step; not an accepted outer/full-assembly or final effect result'))
        report=reports['boundaryFieldContact'];raw=read(folder/'boundaryFieldContact/rawRefinement.json')
        with archive(folder/'boundaryFieldContact/fullGeometry.npz') as data:
            if not np.array_equal(data['faces'],faces):raise ValueError('Template face IDs no longer describe candidate topology')
        attempts=[];pairCounts=Counter({name:0 for name in pairClassNames})
        for stage in report['diagnostics'].get('fast',{}).get('stages',[]):
            for attempt in stage['attempts']:
                gate=attempt['externalCandidateGate'];pairs=gate.get('fullAssembly',{}).get('newPairs',[])
                classes=Counter({name:0 for name in pairClassNames})
                classes.update(classify(int(a),int(b)) for a,b in pairs)
                pairCounts.update(classes);aggregatePairs.update(classes)
                aggregateOuter.update(gate.get('reasons',[]))
                attempts.append(dict(round=stage['round'],scale=attempt['scale'],reasons=gate.get('reasons',[]),
                    geometryAccepted=gate['geometry']['accepted'],coarseAccepted=gate['coarseAlignment']['accepted'],
                    contactProxyAccepted=gate['contact']['accepted'],legacyObservationAccepted=gate['externalCandidateGate']['legacyGate']['accepted'],
                    maximumDisplacementMm=attempt['geometry']['metrics']['maximumDisplacement']/unit,
                    newPairCount=len(pairs),pairClasses=dict(classes),savedNewFullFacePairs=pairs,
                    completeMaskMeasured='apertureMetrics' in attempt,apertureGate=attempt.get('apertureGate')))
        aggregateGn.update(gn);aggregateCombinations.update(combinations)
        row=dict(caseId=case,fieldComparison=fieldRows,gnCounts=dict(gn),exclusiveSafetyCombinations=dict(combinations),
            solverRounds=rounds,outerAttempts=attempts,pairRecordClasses=dict(pairCounts),
            rawStatus=raw['diagnostics']['status'],outputStatus=report['diagnostics']['status'],
            rawRefinementAccepted=report['rawRefinementAccepted'],geometryChanged=report['geometryChanged'],
            improvementAccepted=report['improvementAccepted'],finalEffectGate=report['finalEffectGate'])
        rows.append(row)

    totalTrials=sum(sum(item['trialCount'] for item in row['solverRounds']) for row in rows)
    result=dict(schemaVersion='boundaryFieldSavedSolverAttributionV1',cases=rows,
        aggregate=dict(caseCount=len(rows),totalGnTrials=totalTrials,gnCounts=dict(aggregateGn),
            exclusiveSafetyCombinations=dict(aggregateCombinations),outerFailureReasons=dict(aggregateOuter),
            pairRecordClasses=dict(aggregatePairs),pairRecordCount=sum(aggregatePairs.values()),
            pairCountsRepeatAcrossScales=True,changedOutputs=sum(row['geometryChanged'] for row in rows),
            acceptedImprovements=sum(row['improvementAccepted'] for row in rows)),
        method=dict(savedEvidenceOnly=True,geometryRefitted=False,geometryOrNormalsRecomputed=False,
            rasterRecomputed=False,intersectionsRecomputed=False,verticesReconstructedFromCoefficients=False,
            rank='SVD of saved B only; rank threshold=max(B.shape)*float64 epsilon*largest singular value',
            gnFailureCounters='geometry/contact/coarse single-condition counts overlap; exclusive combinations supplied separately',
            pairClassification='Original full-face IDs mapped through frozen template components and alphaMode; headAccessory denotes template canthus pieces',
            finalStep='Compare saved coefficients/output with saved original arrays; nonzero inner step does not imply selected output',
            numpyVersion=np.__version__),
        conclusions=[
            'The fixed-boundary basis restores active rows and rank; zero output is not evidence that the new field stayed inactive.',
            'Four cases improve inner point residuals but every nonzero outer trial fails the strict full-scene pair gate before complete-mask measurement.',
            'Three cases are contact-limited: tiny accepted inner steps do not reach the existing minimum point improvement, so the solver returns its original mesh.',
            'All final outputs return the baseline. The final effect gate therefore mostly measures an already unchanged result, not a rejected substantial mask improvement.'
        ],
        nextStepEvidence=[
            'Consider the same contact/coarse limits inside a constrained step calculation rather than unconstrained GN followed by rejection; this is a future hypothesis, not a demonstrated improvement.',
            'The existing point-to-eye proxy omits canthus/alpha interactions and does not certify triangle contacts. Any future full-scene contact treatment must retain the current strict gate and use common template-derived rules.',
            'Investigate existing overlap/sliding at attached surfaces before interpreting new triangle-pair IDs as deeper penetration; do not waive alpha or canthus pairs from this result.'
        ],
        limitations=[
            'Saved pair-set changes identify the exact current gate failure, not signed penetration depth, visible severity or continuous collision.',
            'A new pair may correspond to a neighboring triangle along an existing physical overlap; this audit does not establish or rule that out.',
            'Alpha cards and opaque parts are classified separately for diagnosis only. No original gate/material is exempted.',
            'Inner point-fit and objective reductions do not establish complete-aperture or photographic likeness improvement.',
            'This audit does not replace independent recomputation of gates and solver equations.'
        ],inputHashes=hashes,code=dict(path=str(Path(__file__).resolve()),sha256=codeHash),allReadInputsUnchanged=True)
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError('Input changed before audit completion: '+path)
    if sha(__file__)!=codeHash:raise ValueError('Audit code changed during read')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(output.resolve()),sha256=sha(output),inputCount=len(hashes),aggregate=result['aggregate']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    audit(parser.parse_args().output)
