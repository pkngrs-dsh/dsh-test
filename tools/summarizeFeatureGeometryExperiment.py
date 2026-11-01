"""Summarize the frozen factorial without selecting or retuning candidates."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/featureGeometry20261024'
OUT=ROOT/'reports/featureGeometry20261024'
def read(path):return json.loads(path.read_text())
plan=read(RUN/'experimentPlan.json');rows=[]
for case in plan['cases']:
    for branch in plan['branches']:
        path=RUN/case/branch/'report.json'
        if not path.exists():continue
        report=read(path)
        rows.append({'caseId':case,'branch':branch,'status':report['status'],
            'geometrySafetyAccepted':report['geometrySafetyAccepted'],
            'eyeEffectAccepted':report.get('eyeEffectAccepted'),
            'eyeStatus':report.get('eyeRefinement',{}).get('status'),
            'initialAccepted':report.get('initialAccepted'),
            'newPairsAgainstOriginalPureDirect':len(report.get('newFullPairsAgainstOriginalPureDirect',[])),
            'newPairsAgainstCurrentBounded':len(report.get('newFullPairsAgainstCurrentBounded',[])),
            'cavitySafety':report.get('finalCavityGuard',{}).get('accepted'),
            'cavityBenefit':report.get('finalCavityBenefit',{}),
            'originalBaselineReplayMaximumError':report.get('originalBaselineReplayMaximumError'),
            'report':str(path)})
completed=[case for case in plan['cases'] if (OUT/case/'report.json').exists()]
summary={'status':'complete' if len(rows)==28 and len(completed)==7 else 'inProgress',
    'caseCount':7,'branchCount':4,'geometryResultsCompleted':len(rows),'allFourViewsRenderedCases':completed,
    'renderedImageCount':sum(len(list((OUT/case).glob('*.png'))) for case in completed),
    'uniformParameters':True,'failedResultsRetained':True,'casesArePreviouslySeenRegressionGroups':True,
    'rows':rows,'newCandidateGeometrySafetyPassed':sum(row['geometrySafetyAccepted'] for row in rows if row['branch']!='baseline'),
    'visibleEyeModuleAcceptedCount':sum(bool(row['eyeEffectAccepted']) for row in rows if row['branch'] in ('visibleEyes','combined')),
    'sourcePriorInitialAcceptedCount':sum(read(RUN/case/'initial/sourcePrior/report.json')['accepted'] for case in plan['cases'] if (RUN/case/'initial/sourcePrior/report.json').exists()),
    'productionDefaultsChanged':False,'textureRebaked':False,'candidatePromoted':False,
    'eyeFactorInterpretation':'The eye factor replaces old semantic eye stages and then runs common assembly before the new local solver. Eyeballs and other accessories are fixed DURING that solver; their assembly positions can differ between factorial branches. This measures the replacement pipeline, not the isolated aperture solver at globally identical accessory positions.',
    'scope':'Geometry experiment only. Identical template materials; photo-derived brow/nose texture not rebaked. Original client vertex/face/UV/binding order retained.',
    'followupHypotheses':['Constrain unobserved internal lip surfaces before allowing source-shape residuals to propagate there.',
        'Keep the useful coarse eye alignment and restrict its distant deformation; replacing the entire stage reintroduces preexisting assembly contacts.',
        'Fit visible lids together with verified attachment relationships; retain strict geometry checks.',
        'After geometry acceptance, rebake feature-aware source selection for eyebrows and alar regions.']}
OUT.mkdir(exist_ok=True);(OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:summary[k] for k in ('status','geometryResultsCompleted','allFourViewsRenderedCases','newCandidateGeometrySafetyPassed','visibleEyeModuleAcceptedCount')}))
