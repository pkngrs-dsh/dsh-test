"""Classify saved strict full-scene pair lists; never recompute intersections."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from auditProtectedEyeGeometry import RUN,OUT,TEMPLATE,BRANCHES,read,sha,load,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append');args=parser.parse_args()
    template=load(TEMPLATE/'template.npz');faces=template['faces'];materials=template['materialIds']
    skin=np.zeros(int(faces.max())+1,bool);skin[template['nicpToFullIds']]=True
    skinFace=skin[faces].all(1)
    components=read(TEMPLATE/'components.json');faceComponent=np.full(len(faces),-1,np.int64)
    for index,component in enumerate(components):faceComponent[component['faceIndices']]=index
    if (faceComponent<0).any():raise ValueError('Unclassified original face')
    categories={0:'headAccessory',1:'mouth',2:'eye',3:'alpha'}
    def classify(pairs):
        raw=np.asarray(pairs,np.int64).reshape(-1,2)
        if len(raw) and (raw.min()<0 or raw.max()>=len(faces)):raise ValueError('Invalid saved full pair')
        surfaces=Counter();labels=Counter()
        for pair in raw:
            surfaces['-'.join(sorted('skin' if skinFace[face] else categories[int(materials[face])] for face in pair))]+=1
            labels[' / '.join(sorted(components[int(faceComponent[face])]['name']+':'+components[int(faceComponent[face])]['label'] for face in pair))]+=1
        return dict(pairCount=len(raw),surfaceCategoryCounts=dict(surfaces),componentPairCounts=dict(labels))
    results=[]
    for case in args.case or read(RUN/'experimentPlan.json')['cases']:
        result=dict(caseId=case,branches={},sourceHashes={})
        baselinePath=RUN/case/'baseline/fullGeometry.npz'
        if not baselinePath.exists():results.append(result);continue
        baseline=load(baselinePath)['vertices'];result['sourceHashes'][str(baselinePath)]=sha(baselinePath)
        context=read(RUN/case/'inputContext.json');unit=float(load(context['entry']['preparedGeometry'])['unit'])
        for name in BRANCHES:
            directory=RUN/case/name;path=directory/'report.json'
            if not path.exists():continue
            report=read(path);result['sourceHashes'][str(path)]=sha(path)
            if report['status']!='geometryDiagnosticComplete':result['branches'][name]={'status':report['status']};continue
            meshPath=directory/'fullGeometry.npz'
            if sha(meshPath)!=report['outputHashes']['fullGeometry.npz']:raise ValueError('Completed branch changed')
            mesh=load(meshPath)
            if not np.array_equal(mesh['faces'],faces):raise ValueError('Full topology identity changed')
            result['sourceHashes'][str(meshPath)]=sha(meshPath)
            componentMotion=[]
            for item in components:
                ids=np.asarray(item['vertexIndices']);difference=(mesh['vertices'][ids]-baseline[ids])/unit
                componentMotion.append(dict(name=item['name'],label=item['label'],materialId=item['materialId'],
                    exactUnchanged=bool(not np.any(difference)),changedVertexCount=int(np.any(difference!=0,axis=1).sum()),
                    maximumMm=float(np.linalg.norm(difference,axis=1).max())))
            result['branches'][name]=dict(status='classifiedSavedPairs',
                versusCommonBaseline=classify(report['newFullPairsAgainstPairedBaseline']),
                versusOriginalPureDirect=classify(report['newFullPairsAgainstOriginalPureDirect']),
                versusCurrentBounded=classify(report['newFullPairsAgainstCurrentBounded']),
                componentMotionFromCommonBaseline=componentMotion,
                allNonSkinCoordinatesExactlyFixed=bool(np.array_equal(mesh['vertices'][~skin],baseline[~skin])),
                reportedGeometrySafetyAccepted=report['geometrySafetyAccepted'])
        result['limitations']=['Only saved strict face-pair lists are classified; this does not recompute or measure penetration severity.',
            'Head-material canthi are explicitly headAccessory, not deformable skin.',
            'Eye/lash/brow coordinates are common baseline coordinates; altered skin-to-accessory relationships remain part of the strict rejection.',
            'Final differences include full downstream processing. They cannot all be attributed to one eye-stage without additional stage-specific collision calculation.']
        write(OUT/case/'intersectionClasses.json',result);results.append(result)
    write(OUT/'intersectionClasses.json',dict(cases=results,intersectionsRecomputed=False,
        templateSha256=sha(TEMPLATE/'template.npz'),componentSha256=sha(TEMPLATE/'components.json'),scriptSha256=sha(__file__)))
    for result in results:
        print(result['caseId'],{name:value.get('versusCommonBaseline',{}).get('surfaceCategoryCounts') for name,value in result['branches'].items()},flush=True)


if __name__=='__main__':main()
