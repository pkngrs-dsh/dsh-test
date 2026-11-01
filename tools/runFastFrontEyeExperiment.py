"""Fast frontal eye geometry + fresh client bake + bounded local eye appearance.

All branches start from the actual original stable delivered ZIP, not an older
experimental eye mesh. UFF/photo observations are cached; this is NOT a full
request latency benchmark. Camera, topology, UVs, and other materials stay fixed.
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys, zipfile
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets,decodeColor
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs
from faceAppearance.clientLandmarks import loadClientLandmarks,ibugClientLandmarks
from faceAppearance.fastFrontEyeGeometry import refineFastFrontEyes
from faceAppearance.fastFrontEyeTexture import reprojectFastFrontEyeTexture
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.colmapTexture import _destinationUvLookup
from runClientDenseEyeExperiment import compactBinding
from prepareFrontLandmarkTextures import encodeSrgb
from runClientEyelidTextureExperiment import sha

BRANCHES=[('baseline','原稳定几何 · 重新烘焙'),('geometry','快速眼形 · 重新烘焙'),('combined','快速眼形 + 正面眼周取色')]
CASES=('currentMale','highPitch4','cyanWall4','outdoorSideLight4')
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def paths(case):
    if case=='currentMale':
        return dict(source=ROOT/'runs/liveJaw20261012/source/reconstruction',
            archive=ROOT/'runs/liveJaw20261012/comparison/after/legacyAsset.zip',
            observations=ROOT/'runs/denseClientEyes20261012/observations',
            delight=ROOT/'runs/clientAppearanceFix20261012/deployedSource',
            photos=ROOT/'runs/legacyPreviewJaw20261012/preview-609d6c9b63154933b7e3ecd46bb37719')
    p=ROOT/'runs/eyelidTextureBatch20261012'/case
    return dict(source=p/'baseline/reconstruction',archive=p/'baseline/asset/legacyAsset.zip',
        observations=p/'observations',delight=p/'baseline/reconstruction',photos=p/'photos')

def archiveCandidate(original, output, canonical, pixels, description):
    """Replace only changed OBJ vertex lines and two head PNGs; hash and reload."""
    output.mkdir(parents=True,exist_ok=False)
    asset=_load(original)
    sim=asset.manifest['assemblyDiagnostics']['templateToUff']
    R=np.asarray(sim['rotation']);t=np.asarray(sim['translation']);s=float(sim['scale'])
    before=asset.mesh['vertices']*100@R*s+t
    changed=np.any(canonical!=before,axis=1)
    meter=asset.mesh['vertices'].copy()
    meter[changed]=((canonical[changed]-t)@R.T)/s*.01
    if not np.isfinite(meter).all():raise ValueError('Nonfinite candidate')
    payloads=dict(asset.payloads);lines=payloads['face_nicp.obj'].decode().splitlines(keepends=True)
    vertex=0
    for i,line in enumerate(lines):
        if line.startswith('v '):
            if changed[vertex]:lines[i]='v '+' '.join(format(x,'.17g') for x in meter[vertex])+'\n'
            vertex+=1
    if vertex!=len(meter):raise ValueError('OBJ vertex count mismatch')
    payloads['face_nicp.obj']=''.join(lines).encode()
    b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG')
    payloads['head_color.png']=payloads['face_nicp.png']=b.getvalue()
    manifest=asset.manifest
    manifest['files']={n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for n,v in payloads.items()}
    manifest['localFastFrontEyeExperiment']=description
    writeJson(output/'legacyManifest.json',manifest)
    for n,v in payloads.items():(output/n).write_bytes(v)
    with zipfile.ZipFile(output/'legacyAsset.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*payloads,'legacyManifest.json']:z.write(output/n,n)
    check=_load(output/'legacyAsset.zip')
    checks={k:bool(np.array_equal(asset.mesh[k],check.mesh[k])) for k in asset.mesh if k!='vertices'}
    actual=check.mesh['vertices']*100@R*s+t
    if not all(checks.values()) or not np.allclose(actual,canonical,rtol=0,atol=1e-12):raise ValueError('Export geometry changed')
    fixed=bool(np.array_equal(check.mesh['vertices'][~changed],asset.mesh['vertices'][~changed]))
    same={n:asset.payloads[n]==v for n,v in payloads.items() if n not in ('face_nicp.obj','head_color.png','face_nicp.png')}
    if not fixed or not all(same.values()):raise ValueError('Frozen asset bytes changed')
    record=dict(sha256=sha(output/'legacyAsset.zip'),changedVertexCount=int(changed.sum()),
        topologyUvMaterialArraysEqual=checks,unchangedVertexValuesExact=fixed,otherPayloadBytesEqual=same)
    writeJson(output/'archive.json',record)
    return record

def run(case,out):
    start=perf_counter();p=paths(case)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((p['source']/'inputs.json').read_text());old=json.loads((p['delight']/'inputs.json').read_text())
    if len(inputs)!=len(old) or any(any(a[k]!=b[k] for k in ('sha256','inputToWorking','workingSize')) for a,b in zip(inputs,old)):raise ValueError('Cached delight identity mismatch')
    obsManifest=json.loads((p['observations']/'manifest.json').read_text())
    if obsManifest['inputsSha256']!=sha(p['source']/'inputs.json') or obsManifest['camerasSha256']!=sha(p['source']/'cameras.json'):raise ValueError('Observation frame mismatch')
    obs=npz(p['observations']/'view00.npz')
    if not obs['candidateEyeValidMask'].all():raise ValueError('Unreliable photo eye')
    photoEyes={n:{'upper':obs['eyelidCurvesWorking'][i,1],'lower':obs['eyelidCurvesWorking'][i,0]} for i,n in enumerate(('imageLeft','imageRight'))}
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1');a=assets.arrays
    scene=loadLegacyEyeBindingInputs(p['archive'],a['nicpToFullIds']);scene.pop('nicpToFullIds')
    original=_load(p['archive']);full=scene['vertices'];skin=a['nicpToFullIds']
    cameras=json.loads((p['source']/'cameras.json').read_text());camera=cameras[0]
    if camera.get('viewIndex')!=0 or not camera.get('valid'):raise ValueError('Missing front camera')
    jaw=npz(p['source']/'directClient/renderedJaw/geometry.npz')
    mouthLm=ibugClientLandmarks(loadClientLandmarks(a['nicpFaces']),groups=('mouth',))
    fixedSkin=np.unique(np.r_[np.flatnonzero(a['staticMask']),mouthLm['vertexIndices'].ravel(),np.flatnonzero(np.any(jaw['before']!=jaw['after'],axis=1))])
    cornerFaces=np.concatenate([c['faceIndices'] for c in assets.components if c['label']=='headAccessory']).astype(np.int64)
    tick=perf_counter()
    fit=refineFastFrontEyes(**scene,camera=camera,photoEyes=photoEyes,skinVertexIds=skin,cornerFaceIds=cornerFaces,
        fixedVertexIds=skin[fixedSkin],imageShape=tuple(reversed(inputs[0]['workingSize'])),
        originalPixelScale=float(np.asarray(inputs[0]['inputToWorking'])[0,0]))
    fitSeconds=perf_counter()-tick
    np.savez_compressed(out/'geometry.npz',before=full,after=fit['vertices'],faces=scene['faces'],skinVertexIds=skin,cornerFaceIds=cornerFaces,fixedVertexIds=skin[fixedSkin])
    writeJson(out/'geometryReport.json',fit['diagnostics'])
    for when in ('before','after'):writeJson(out/(when+'Binding.json'),compactBinding(fit[when+'Contours']))
    writeJson(out/'textureBinding.json',compactBinding(fit['textureContours']))
    writeJson(out/'config.json',{'branches':[{'id':k,'label':v} for k,v in BRANCHES],'defaultLeft':'baseline','defaultRight':'combined'})
    geometry=npz(p['source']/'directClient/geometry.npz');geometry.update(vertices=full[skin],uvConvention='topLeftPixelEdges')
    faceLookup={tuple(f):i for i,f in enumerate(original.mesh['faces'])}
    fullFaceIds=np.array([faceLookup[tuple(f)] for f in skin[geometry['faces']]])
    if not np.array_equal(geometry['uvs'],original.mesh['uvs']) or not np.array_equal(geometry['uvFaces'],original.mesh['uvFaces'][fullFaceIds]):raise ValueError('UV/topology mismatch')
    pixels=np.asarray(Image.open(io.BytesIO(original.payloads['head_color.png']))).copy();size=pixels.shape[0]
    skinMember=np.zeros(len(full),bool);skinMember[skin]=True
    otherFaces=(original.mesh['materialIds']==0)&~skinMember[original.mesh['faces']].all(1)
    other=dict(vertices=full,faces=original.mesh['faces'][otherFaces],uvs=original.mesh['uvs'],uvFaces=original.mesh['uvFaces'][otherFaces],uvConvention='topLeftPixelEdges')
    protected=binary_dilation(_destinationUvLookup(other,size)[0]['validMask'],iterations=2)
    uvDomain=_destinationUvLookup(geometry,size)[0]['validMask'];writable=binary_dilation(uvDomain,iterations=1)&~protected
    delight=npz(p['delight']/'delight.npz')
    gain=json.loads((p['source']/'directClient/report.json').read_text())['baking']['templateLinearColorGain']
    mouthPaths=[p['source']/'directClient/mouthState.npz',p['source']/'directClient/refinement/mouthState.npz',p['delight']/'directClient/refinement/mouthState.npz']
    mouthPath=next((q for q in mouthPaths if q.is_file()),None);mouth=npz(mouthPath) if mouthPath else None
    report=dict(caseId=case,status='running',deployed=False,paths={k:str(v) for k,v in p.items()},
        sourceArchiveSha256=sha(p['archive']),sourceDelightSha256=sha(p['delight']/'delight.npz'),
        sourceColor='OpenDelight cached native gamma2.2; identical photos/crops verified',
        fullGenerationMeasured=False,geometrySeconds=fitSeconds,geometry=fit['diagnostics'],branches=[])
    writeJson(out/'report.json',report)
    baked=None
    for branch,label in BRANCHES:
        tick=perf_counter();candidate=full if branch=='baseline' else fit['vertices']
        geometry['vertices']=candidate[skin]
        if branch!='combined':
            baked=bakeClientTexture(geometry,cameras,delight,assets.materials[0]['baseColor'][...,:3],out/(branch+'Bake'),
                colmapExecutable=ROOT/'.runtimes/colmap-eyelid/bin/colmap-eyelid',mouthState=mouth,
                fallbackGain=gain,freezeViewLabels=True)
            updated=pixels.copy();updated[~protected,:3]=np.rint(encodeSrgb(baked['linearRGB'][~protected])*255).astype(np.uint8)
            textureDiagnostics=baked['diagnostics'];moduleSeconds=textureDiagnostics['seconds']
        else:
            # Complete aperture mask drives geometry; this separate measured
            # eye-material boundary provides ordered skin-only texture curves.
            # It does not claim that accessory fragments form a closed canthus.
            binding=fit['textureContours']
            if any(binding['eyes'][n]['status']!='accepted' for n in photoEyes):
                raise ValueError('Texture proxy does not have valid ordered eyelid curves')
            modelEyes={n:{part:np.asarray(binding['eyes'][n][part]['sourcePixels']) for part in ('lower','upper')} for n in photoEyes}
            if not all(np.isfinite(curve).all() for parts in modelEyes.values() for curve in parts.values()):raise ValueError('Model eyelid has nonfinite semantic points')
            result=reprojectFastFrontEyeTexture(geometry,camera,decodeColor(delight['albedoImages'][0],'gamma2.2'),delight['validMasks'][0],
                decodeColor(updated[...,:3]/255.,'srgb'),photoEyes,modelEyes,writableMask=writable,
                visibleScene={'vertices':candidate,'faces':scene['faces'][scene['materialIds']!=3]})
            modified=(result['alpha']>0)|result['filterPadding'];before=updated.copy()
            updated[modified,:3]=np.rint(encodeSrgb(result['linearRGB'][modified])*255).astype(np.uint8)
            if not np.array_equal(before[~modified],updated[~modified]) or np.any(modified&~writable):raise ValueError('Texture escaped editable support')
            np.savez_compressed(out/'textureProvenance.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)},writableMask=writable)
            Image.fromarray(np.rint(result['alpha']*255).astype(np.uint8)).save(out/'eyeAlpha.png')
            textureDiagnostics={**result['diagnostics'],
                'visibilityScope':'All opaque assembly surfaces; MASK brow/lash cards excluded from sampling occlusion and retained in final render',
                'modelTextureContours':'Measured eye-material proxy; complete aperture mask is evaluated separately'}
            moduleSeconds=textureDiagnostics['seconds']
        if not np.array_equal(updated[protected],pixels[protected]):raise ValueError('Accessory UV changed')
        info=archiveCandidate(p['archive'],out/branch,candidate,updated,dict(deployed=False,branch=branch,originalStableBaseline=True,geometryChanged=branch!='baseline',texture=textureDiagnostics))
        Image.fromarray(updated).save(out/(branch+'Texture.png'))
        row=dict(id=branch,label=label,wallSeconds=perf_counter()-tick,moduleSeconds=moduleSeconds,archive=info,texture=textureDiagnostics)
        report['branches'].append(row);writeJson(out/'report.json',report)
        print(json.dumps({'case':case,'branch':branch,'seconds':row['wallSeconds'],'geometrySeconds':fitSeconds}),flush=True)
    report.update(status='completeLocalExperiment',seconds=perf_counter()-start,
        measuredCandidatePostprocessSeconds=fitSeconds+report['branches'][1]['wallSeconds']+report['branches'][2]['wallSeconds'],
        scope='Cached original stable geometry, cameras, delight and photo semantics. UFF, registration, jaw, detectors, cold load and transmission excluded.',
        previewDiagnostics=[{'path':'eyeAlpha.png','label':'快速正面眼周取色权重','note':'受限的残差重采样/肤色外扩属于估计外观；不计为真实新增覆盖。'}])
    writeJson(out/'report.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--output-root',type=Path,default=ROOT/'runs/fastFrontEyes20261012/final');args=p.parse_args()
    from contextlib import ExitStack
    from unittest.mock import patch
    from faceAppearance import clientTextureBaking,colmapTexture
    from faceAppearance.clientUvLookupCache import ClientUvLookupCache
    geometry=npz(paths(args.case)['source']/'directClient/geometry.npz')
    geometry['uvConvention']='topLeftPixelEdges'
    tick=perf_counter()
    cache=ClientUvLookupCache(colmapTexture._destinationUvLookup)
    cache.warmMouthGrids(geometry,1024)
    warmSeconds=perf_counter()-tick
    with ExitStack() as context:
        for module in (clientTextureBaking,colmapTexture):
            context.enter_context(patch.object(module,'_destinationUvLookup',cache))
        report=run(args.case,args.output_root/args.case)
    report['startupUvWarmup']={'seconds':warmSeconds,**cache.diagnostics()}
    report['implementationSha256']={str(path.relative_to(ROOT)):sha(path) for path in (
        Path(__file__),ROOT/'src/faceAppearance/fastFrontEyeGeometry.py',ROOT/'src/faceAppearance/fastFrontEyeTexture.py',
        ROOT/'src/faceAppearance/clientUvLookupCache.py')}
    writeJson(args.output_root/args.case/'report.json',report)
if __name__=='__main__':main()
