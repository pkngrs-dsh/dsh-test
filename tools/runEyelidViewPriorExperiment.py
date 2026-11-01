"""Compare original and frontal-eyelid COLMAP selection on frozen client meshes."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyPreviewAsset import _load
from prepareFrontLandmarkTextures import encodeSrgb
from runClientEyelidTextureExperiment import archiveTexture, sha

BRANCHES = [('baseline','原始选图',0.), ('soft','温和正面优先',2.), ('strong','强正面优先',6.)]


def paths(case):
    if case == 'currentMale':
        return dict(source=ROOT/'runs/liveJaw20261012/source/reconstruction',
            delight=ROOT/'runs/clientAppearanceFix20261012/deployedSource',
            geometry=ROOT/'runs/denseClientEyes20261012/comparisonFinal',
            observations=ROOT/'runs/denseClientEyes20261012/observations',
            photos=ROOT/'runs/legacyPreviewJaw20261012/preview-609d6c9b63154933b7e3ecd46bb37719')
    root = ROOT/'runs/eyelidTextureBatch20261012'/case
    return dict(source=root/'baseline/reconstruction', delight=root/'baseline/reconstruction',
        geometry=root/('dense' if case=='cyanWall4' else 'textureGeometryBaseline'),
        observations=root/'observations', photos=root/'photos')


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def selectedLabels(path, faceCount, imageIds):
    lines=Path(path).read_text().splitlines()
    if lines[0].split()!=['COLMAP_FACE_VIEW_IDS_V1',str(faceCount),str(len(imageIds))]:
        raise ValueError('Unexpected frozen face label header')
    rows=np.asarray([list(map(int,line.split())) for line in lines[1:] if line.strip()],np.int64)
    if rows.shape!=(faceCount,2) or not np.array_equal(rows[:,0],np.arange(faceCount)):
        raise ValueError('Frozen labels must cover ordered original faces exactly once')
    if not np.isin(rows[:,1],[-1,*imageIds]).all():
        raise ValueError('Unknown COLMAP image ID in frozen labels')
    return rows[:,1]


def sourceMap(lookup, source, coverage):
    # Source indices: front red, side A blue, side B green, up yellow, fallback gray.
    palette = np.array([[238,90,82],[76,152,235],[69,191,132],[239,194,74]],np.uint8)
    colors = np.full((*coverage.shape,3),30,np.uint8)
    colors[lookup['validMask']] = [120,120,120]
    for index, color in enumerate(palette):
        colors[coverage & (source==index)] = color
    return colors


def main():
    from faceAppearance.eyelidViewPrior import buildEyelidViewPrior, writeEyelidViewPrior
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['currentMale','highPitch4','cyanWall4','outdoorSideLight4'], required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs/eyelidViewPrior20261012')
    parser.add_argument('--colmap',type=Path,default=ROOT/'.runtimes/colmap-eyelid/bin/colmap-eyelid')
    args=parser.parse_args()
    p=paths(args.case); out=args.output_root/args.case
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    inputs=json.loads((p['source']/'inputs.json').read_text())
    old=json.loads((p['delight']/'inputs.json').read_text())
    if len(inputs)!=len(old) or any(any(a[k]!=b[k] for k in ('sha256','inputToWorking','workingSize')) for a,b in zip(inputs,old)):
        raise ValueError('Delight photo identity or original crop mismatch')
    obsManifest=json.loads((p['observations']/'manifest.json').read_text())
    if obsManifest['inputsSha256']!=sha(p['source']/'inputs.json') or obsManifest['camerasSha256']!=sha(p['source']/'cameras.json'):
        raise ValueError('Observation camera/input identity mismatch')
    cameras=json.loads((p['source']/'cameras.json').read_text())
    if cameras[0].get('viewIndex')!=0 or not cameras[0].get('valid'):
        raise ValueError('First photo must be valid frontal reference')
    delight=npz(p['delight']/'delight.npz')
    sourceArchive=p['geometry']/'after/legacyAsset.zip'
    asset=_load(sourceArchive)
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    similarity=asset.manifest['assemblyDiagnostics']['templateToUff']
    full=asset.mesh['vertices']*100 @ np.asarray(similarity['rotation'])*similarity['scale']+similarity['translation']
    skinIds=assets.arrays['nicpToFullIds']
    geometry=npz(p['source']/'directClient/geometry.npz')
    geometry.update(vertices=full[skinIds],uvConvention='topLeftPixelEdges')
    if not np.allclose(geometry['vertices'], npz(p['geometry']/'geometry.npz')['after'],rtol=0,atol=1e-12):
        raise ValueError('Frozen mesh does not match accepted geometry')
    faceLookup={tuple(face):i for i,face in enumerate(asset.mesh['faces'])}
    fullFaceIds=np.asarray([faceLookup[tuple(face)] for face in skinIds[geometry['faces']]])
    if (not np.array_equal(geometry['uvs'],asset.mesh['uvs'])
            or not np.array_equal(geometry['uvFaces'],asset.mesh['uvFaces'][fullFaceIds])
            or not np.all(asset.mesh['materialIds'][fullFaceIds]==0)):
        raise ValueError('Baking geometry topology/UVs differs from the frozen ZIP skin')
    observation=npz(p['observations']/'view00.npz')
    if not observation['candidateEyeValidMask'].all(): raise ValueError('Rejected photo eyelid observations')
    binding=json.loads((p['geometry']/'afterBinding.json').read_text())
    curves=observation['eyelidCurvesWorking']
    photoEyes={n:{'lower':curves[i,0],'upper':curves[i,1]} for i,n in enumerate(('imageLeft','imageRight'))}
    modelEyes={n:{part:np.asarray(binding['eyes'][n][part]['sourcePixels']) for part in ('lower','upper')} for n in photoEyes}
    visible={'vertices':full,'faces':asset.mesh['faces'][asset.mesh['materialIds']!=3]}
    original=np.asarray(Image.open(io.BytesIO(asset.payloads['head_color.png']))).copy()
    size=original.shape[0]
    lookup,_=_destinationUvLookup(geometry,size)
    skinMember=np.zeros(len(full),bool);skinMember[skinIds]=True
    excludedFaces=(asset.mesh['materialIds']==0)&~skinMember[asset.mesh['faces']].all(1)
    excludedGeometry=dict(vertices=full, faces=asset.mesh['faces'][excludedFaces],uvs=asset.mesh['uvs'],
                          uvFaces=asset.mesh['uvFaces'][excludedFaces],uvConvention='topLeftPixelEdges')
    accessoryUv=_destinationUvLookup(excludedGeometry,size)[0]['validMask']
    # Preserve the exact original pixels under all non-NICP head accessories.
    protected=binary_dilation(accessoryUv,iterations=2)
    gain=json.loads((p['source']/'directClient/report.json').read_text())['baking']['templateLinearColorGain']
    mouthCandidates=[p['source']/'directClient/mouthState.npz',p['source']/'directClient/refinement/mouthState.npz',
                     p['delight']/'directClient/refinement/mouthState.npz']
    mouthPath=next((f for f in mouthCandidates if f.is_file()),None)
    mouth=npz(mouthPath) if mouthPath else None
    fallback=assets.materials[0]['baseColor'][...,:3]
    report=dict(caseId=args.case,status='running',deployed=False,
        inputs={k:str(v) for k,v in p.items()},sourceArchiveSha256=sha(sourceArchive),
        denseGeometryApplied=json.loads((p['geometry']/'report.json').read_text()).get('status')=='acceptedLocalExperiment',
        sameGeometryAllBranches=True,sourceDelightSha256=sha(p['delight']/'delight.npz'),
        sourceArchive=str(sourceArchive),mouthStateSource=str(mouthPath),
        sourceColor='OpenDelight native gamma2.2; all branches same cache',
        baseline='Fresh original COLMAP selection bake on the same final geometry; no previous eyelid resampling',
        parameters={'strengths':[0,2,6],'smoothingWeight':.35,'smoothingIterations':10},branches=[],previewDiagnostics=[])
    writeJson(out/'report.json',report)
    writeJson(out/'config.json',{'branches':[{'id':name,'label':label} for name,label,_ in BRANCHES],
        'defaultLeft':'baseline','defaultRight':'soft'})
    baseline=None
    baselineLabels=None
    for name,label,strength in BRANCHES:
        tick=perf_counter()
        prior=buildEyelidViewPrior(geometry,cameras[0],delight['validMasks'][0],photoEyes,modelEyes,
                                  visibleScene=visible,strength=strength,frontImageId=1)
        np.savez_compressed(out/(name+'Prior.npz'),**{k:v for k,v in prior.items() if isinstance(v,np.ndarray)})
        priorPath=out/(name+'Prior.txt')
        writeEyelidViewPrior(priorPath,prior,numImages=sum(bool(c.get('valid',True)) for c in cameras))
        baked=bakeClientTexture(geometry,cameras,delight,fallback,out/(name+'Bake'),
            colmapExecutable=args.colmap.resolve(),mouthState=mouth,referenceView=0,fallbackGain=gain,
            viewSelectionPrior=priorPath if strength else None,freezeViewLabels=True,
            viewSelectionSmoothingWeight=.35,viewSelectionSmoothingIterations=10)
        pixels=original.copy()
        pixels[~protected,:3]=np.rint(encodeSrgb(baked['linearRGB'][~protected])*255).astype(np.uint8)
        description={'deployed':False,'method':'COLMAP per-face frontal eyelid source preference',
            'strength':strength,'geometryChanged':False,'fullColmapRebake':True,
            'sourceArchiveSha256':sha(sourceArchive),'bakeDiagnostics':baked['diagnostics'],
            'priorDiagnostics':prior['diagnostics']}
        archive=archiveTexture(sourceArchive,out/name,pixels,description)
        Image.fromarray(pixels).save(out/(name+'Texture.png'))
        sourceImage=sourceMap(lookup,baked['sourceIndex'],baked['colmapCoverageMask'])
        Image.fromarray(sourceImage).save(out/(name+'Sources.png'))
        faceIds=lookup['faceIds'] if 'faceIds' in lookup else lookup['faceIndex']
        faceWeights=prior['eyelidWeight']
        region=lookup['validMask']&(faceWeights[np.maximum(faceIds,0)]>0)
        eligible=prior['frontEligible']&(faceWeights>0)
        labels=selectedLabels(out/(name+'Bake')/'colorOutput/faceViewIds.txt',len(geometry['faces']),
            [r['colmapImageId'] for r in baked['diagnostics']['workspaces']['colorWorkspace']['cameras']])
        if baseline is None: baseline=pixels.copy(); baselineLabels=labels.copy()
        changedLabels=labels!=baselineLabels
        if np.any(changedLabels&~eligible): raise ValueError('View selection changed outside eligible eyelid faces')
        row={'id':name,'label':label,'strength':strength,'seconds':perf_counter()-tick,'archive':archive,
            'prior':prior['diagnostics'],'changedFaceLabels':int(changedLabels.sum()),
            'frontFacesInEligibleRegion':int((eligible&(labels==1)).sum()),
            'eligibleFaces':int(eligible.sum()),
            'regionTexels':int(region.sum()),
            'frontObservedRegionTexels':int((region&baked['colmapCoverageMask']&(baked['sourceIndex']==0)).sum()),
            'unobservedRegionTexels':int((region&~baked['colmapCoverageMask']).sum()),
            'changedRgbTexelsVsFreshBaseline':int(np.any(pixels[...,:3]!=baseline[...,:3],axis=-1).sum()),
            'allGeometryPayloadsUnchanged':True,'protectedAccessoryPixelsUnchanged':True,
            'colorSupportMarkerLabelsIdentical':baked['diagnostics']['viewSelection']['labelsIdenticalAcrossThreePasses']}
        report['branches'].append(row)
        report['previewDiagnostics'].append({'label':label+'：来源（红正面/蓝绿侧面/黄抬头/灰回退）','path':name+'Sources.png'})
        writeJson(out/'report.json',report)
        print(json.dumps({'case':args.case,'branch':name,'seconds':row['seconds'],'changedFaces':row['changedFaceLabels'],
                          'frontEligible':row['frontFacesInEligibleRegion'],'eligible':row['eligibleFaces']},ensure_ascii=False),flush=True)
    heat=np.zeros((size,size),np.uint8)
    heat[lookup['validMask']]=np.rint(faceWeights[faceIds[lookup['validMask']]]*255).astype(np.uint8)
    Image.fromarray(heat).save(out/'eyelidWeight.png')
    report['previewDiagnostics'].append({'label':'眼睑区域权重','path':'eyelidWeight.png'})
    report.update(status='completeLocalCandidate',seconds=perf_counter()-start,
        implementationSha256={str(f):sha(ROOT/f) for f in ('src/faceAppearance/eyelidViewPrior.py',
            'src/faceAppearance/clientTextureBaking.py','tools/runEyelidViewPriorExperiment.py')},
        limitations=['Automatic eyelid curves; no manual truth.',
            'Preference does not repair geometry/photo misalignment or missing source coverage.',
            'COLMAP patch/color correction changes can affect texels outside the selected face region.',
            'No contour warp, local cleanup, new SH blend, or service deployment in this experiment.'])
    writeJson(out/'report.json',report)

if __name__=='__main__': main()
