"""Local bounded full-aperture geometry probe on frozen stable client archives."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.fastFrontEyeGeometry import refineFastFrontEyes
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs
from faceAppearance.clientLandmarks import loadClientLandmarks,ibugClientLandmarks
from faceAppearance.artifacts import writeJson
from runClientDenseEyeExperiment import compactBinding

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=('currentMale','highPitch4','cyanWall4','outdoorSideLight4'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Choose empty output')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.case=='currentMale':
        source=ROOT/'runs/liveJaw20261012/source/reconstruction'
        archive=ROOT/'runs/liveJaw20261012/comparison/after/legacyAsset.zip'
        observations=ROOT/'runs/denseClientEyes20261012/observations'
    else:
        case=ROOT/'runs/eyelidTextureBatch20261012'/args.case
        source=case/'baseline/reconstruction';archive=case/'baseline/asset/legacyAsset.zip';observations=case/'observations'
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1');a=assets.arrays
    inputs=json.loads((source/'inputs.json').read_text());camera=json.loads((source/'cameras.json').read_text())[0]
    obs=np.load(observations/'view00.npz',allow_pickle=False)
    if not obs['candidateEyeValidMask'].all():raise ValueError('Rejected photo curve')
    photoEyes={n:{'upper':obs['eyelidCurvesWorking'][i,1],'lower':obs['eyelidCurvesWorking'][i,0]} for i,n in enumerate(('imageLeft','imageRight'))}
    scene=loadLegacyEyeBindingInputs(archive,a['nicpToFullIds']);scene.pop('nicpToFullIds')
    jaw=np.load(source/'directClient/renderedJaw/geometry.npz',allow_pickle=False)
    mouth=ibugClientLandmarks(loadClientLandmarks(a['nicpFaces']),groups=('mouth',))
    fixedSkin=np.unique(np.r_[np.flatnonzero(a['staticMask']),mouth['vertexIndices'].ravel(),np.flatnonzero(np.any(jaw['before']!=jaw['after'],axis=1))])
    cornerFaces=np.concatenate([i['faceIndices'] for i in assets.components if i['label']=='headAccessory']).astype(np.int64)
    result=refineFastFrontEyes(**scene,camera=camera,photoEyes=photoEyes,skinVertexIds=a['nicpToFullIds'],cornerFaceIds=cornerFaces,
        fixedVertexIds=a['nicpToFullIds'][fixedSkin],imageShape=tuple(reversed(inputs[0]['workingSize'])),
        originalPixelScale=float(np.asarray(inputs[0]['inputToWorking'])[0,0]),rounds=args.rounds)
    np.savez_compressed(args.output/'geometry.npz',before=scene['vertices'],after=result['vertices'],faces=scene['faces'],
        skinVertexIds=a['nicpToFullIds'],cornerFaceIds=cornerFaces,fixedVertexIds=a['nicpToFullIds'][fixedSkin])
    writeJson(args.output/'report.json',result['diagnostics'])
    writeJson(args.output/'beforeBinding.json',compactBinding(result['beforeContours']))
    writeJson(args.output/'afterBinding.json',compactBinding(result['afterContours']))
    writeJson(args.output/'afterTextureBinding.json',compactBinding(result['afterTextureContours']))
    print(json.dumps({'case':args.case,**{k:v for k,v in result['diagnostics'].items() if k not in ('stages','before','after')},
        'beforeError':result['diagnostics']['before']['meanSymmetricBoundaryDistanceOriginalPixels'],
        'afterError':result['diagnostics']['after']['meanSymmetricBoundaryDistanceOriginalPixels']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
