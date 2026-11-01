"""Render freshly replayed paired assets and their geometry/head-color factorial.

Historical assets are context, not the paired control. The diagnostic factorial
uses one common set of non-head materials, so its two axes really are full
geometry and head base color. It creates images only, never a client asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import fileHash, noseGrid, scaledViewCamera
from renderShapePreservationComparison import baselinePatchCrop, loadArrays, pinHashes, verifyHashes

BRANCHES=('pureDirect','bounded')
MODES=(('baseColor','基色'),('clay','灰模'),('lit','共同检查光照'))
PRIMARY=('historical','pureDirect','bounded')
LABELS=('原工作照片','历史资产（非配对控制）','同流程 pureDirect 基线','同流程 bounded 新解')


def read(path):
    return json.loads(Path(path).read_text())


def banner(picture, caseId, gateLabel, *, factorial=False):
    lines=[f'下游重放与重新烘焙 · {caseId} · {gateLabel}',
        '统一公共眼口、下颌和附件流程；旧 AI 额外眼环实验未复制',
        ('2×2 诊断：行=完整几何，列=头部基色；交叉组合不是交付资产' if factorial else
         '历史资产仅作背景参照；两条配对分支均重新估计 SH、烘焙及眼睑贴图'),
        '保存相机与共同检查光照；模板投影裁剪；数值通过不等于客户端验收']
    result=Image.new('RGB',(picture.width,picture.height+122),(22,27,33))
    result.paste(picture.convert('RGB'),(0,122));draw=ImageDraw.Draw(result)
    for index,line in enumerate(lines):
        for fontSize in range(16,10,-1):
            font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontSize)
            if draw.textlength(line,font=font)<=result.width-24:break
        else:raise ValueError('Diagnostic banner would be clipped')
        draw.text((12,10+index*26),line,font=font,fill=(235,239,244))
    return result


def canonicalVertices(source, similarity):
    actual=source.manifest['assemblyDiagnostics']['templateToUff']
    for key in ('rotation','translation','scale'):
        np.testing.assert_array_equal(np.asarray(actual[key]),np.asarray(similarity[key]))
    return source.mesh['vertices']*100@np.asarray(similarity['rotation'])*similarity['scale']+similarity['translation']


def prepare(splitPath, runRoot, caseId):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',caseId) is None:
        raise ValueError('Expected one safe case identifier from the frozen split')
    splitPath=Path(splitPath).resolve();runRoot=Path(runRoot).resolve()
    split=read(splitPath);entries=[row for row in split['cases'] if row['caseId']==caseId]
    if len(entries)!=1:raise ValueError('Case must occur exactly once in the split')
    entry=entries[0]
    if caseId not in split[entry['split']]:raise ValueError('Inconsistent case split membership')
    identities={};pinHashes(identities,{str(splitPath):fileHash(splitPath)})
    pinHashes(identities,split['commonSourceHashes'])
    prepPath=Path(entry['preparationReport']).resolve();rawPath=Path(entry['preparedGeometry']).resolve()
    pinHashes(identities,{str(prepPath):entry['preparationReportSha256'],str(rawPath):entry['preparedGeometrySha256']})
    verifyHashes(identities)
    prep=read(prepPath);raw=loadArrays(rawPath)
    for key in ('caseId','reconstruction','workingPhotos','sourceArchive','preparedGeometry','sourceDomain'):
        if prep[key]!=entry[key]:raise ValueError(f'Preparation differs from split: {key}')
    pinHashes(identities,prep['sourceHashes'])
    planPath=runRoot/'experimentPlan.json';plan=read(planPath)
    if (Path(plan['parentSplit']).resolve()!=splitPath or plan['parentSplitSha256']!=fileHash(splitPath)
            or plan['initialBranches']!=list(BRANCHES) or plan['oldPostprocessingDeltasReused'] is not False):
        raise ValueError('Experiment plan differs from the frozen paired replay')
    pinHashes(identities,{str(planPath):fileHash(planPath)})
    archives={'historical':Path(entry['sourceArchive']).resolve()};reports={}
    for branch in BRANCHES:
        appearanceRoot=runRoot/caseId/'appearance'/branch
        geometryRoot=runRoot/caseId/'geometry'/branch
        appearancePath=appearanceRoot/'report.json';geometryPath=geometryRoot/'report.json'
        appearance=read(appearancePath);geometry=read(geometryPath)
        if (appearance['status']!='appearanceDiagnosticComplete'
                or geometry['status']!='geometryDiagnosticComplete'
                or any(item['caseId']!=caseId or item['branch']!=branch for item in (appearance,geometry))
                or Path(appearance['geometryReport']).resolve()!=geometryPath
                or any(item.get('finalCandidateAccepted') is not False for item in (appearance,geometry))):
            raise ValueError('Expected completed diagnostic-only paired geometry and appearance reports')
        for reportPath,item,parent in ((geometryPath,geometry,geometryRoot),(appearancePath,appearance,appearanceRoot)):
            pinHashes(identities,item['sourceHashes'])
            pinHashes(identities,{str(reportPath):fileHash(reportPath)})
            pinHashes(identities,{str((parent/name).resolve()):digest for name,digest in item['outputHashes'].items()})
        archive=Path(appearance['asset']['archivePath']).resolve()
        pinHashes(identities,{str(archive):appearance['asset']['sha256']})
        archives[branch]=archive
        reports[branch]={'geometry':geometry,'appearance':appearance,
                         'geometryPath':str(geometryPath),'appearancePath':str(appearancePath)}
    sourceDirectory=Path(entry['reconstruction']).resolve();photos=Path(entry['workingPhotos']).resolve()
    camerasPath=sourceDirectory/'cameras.json';inputsPath=sourceDirectory/'inputs.json'
    cameras=read(camerasPath);inputs=read(inputsPath)
    if len(cameras)!=4 or len(inputs)!=4 or len(prep['photos'])!=4:
        raise ValueError('Expected all four original camera/photo pairs')
    codePaths=[Path(__file__),ROOT/'tools/renderShapePreservationComparison.py',
        ROOT/'tools/renderNoseRepairComparison.py',ROOT/'tools/renderClientAppearanceEvidence.py',
        ROOT/'tools/renderClientAssembly.py',ROOT/'tools/renderFeedbackAppearanceComparison.py',
        ROOT/'src/faceAppearance/legacyPreviewAsset.py',ROOT/'src/faceAppearance/clientLandmarks.py']
    bindingPath=ROOT/'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'
    pinHashes(identities,{str(path.resolve()):fileHash(path) for path in
                         [*codePaths,bindingPath,camerasPath,inputsPath]})
    for index,row in enumerate(prep['photos']):
        path=photos/f'input{index:02d}.png'
        if (row['viewIndex']!=index or Path(row['workingPhotoPath']).resolve()!=path
                or row['workingSize']!=inputs[index]['workingSize']):
            raise ValueError('Working photo identity/order differs from preparation')
        pinHashes(identities,{str(path):row['workingPhotoSha256']})
    verifyHashes(identities)
    sources={name:_load(path) for name,path in archives.items()}
    similarity={key:raw[key].tolist() for key in ('rotation','translation','scale')}
    full={name:canonicalVertices(source,similarity) for name,source in sources.items()}
    reference=sources['pureDirect']
    materialComparison={}
    for name,source in sources.items():
        for key in ('faces','uvs','uvFaces','materialIds'):
            np.testing.assert_array_equal(source.mesh[key],reference.mesh[key])
        for key in ('templateId','templateHashes','binding','arrayIdentities'):
            if source.manifest[key]!=reference.manifest[key]:
                raise ValueError(f'Ordered template/binding differs for {name}: {key}')
        materialComparison[name]={'materialDefinitionsIdenticalToPairedBaseline':source.materials==reference.materials,
            'nonHeadTexturePayloadsIdenticalToPairedBaseline':{material['baseColorTexture']['path']:
                source.payloads[material['baseColorTexture']['path']]==reference.payloads[material['baseColorTexture']['path']]
                for material in reference.materials['materials'][1:]},
            'headPayloadSha256':hashlib.sha256(source.payloads['head_color.png']).hexdigest()}
        np.testing.assert_array_equal(source.mesh['faces'],raw['fullFaces'])
        np.testing.assert_allclose(full[name][raw['skinIds'][raw['staticIds']]],
                                   raw['pureDirect'][raw['staticIds']],rtol=0,atol=1e-12)
    for branch in BRANCHES:
        expected=loadArrays(runRoot/caseId/'geometry'/branch/'fullGeometry.npz')
        np.testing.assert_allclose(full[branch],expected['vertices'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(reference.mesh['faces'],expected['faces'])
        for key in ('rotation','translation','scale'):
            np.testing.assert_array_equal(np.asarray(reports[branch]['geometry']['templateToUff'][key]),raw[key])
    regions=sorted(key[:-len('_interiorVertexIds')] for key in raw if key.endswith('_interiorVertexIds'))
    noseIds=np.unique(np.concatenate([raw[name+suffix] for name in regions
                    for suffix in ('_interiorVertexIds','_rimVertexIdsOrdered')]))
    binding=loadClientLandmarks(raw['faces'],path=bindingPath)
    if not binding['isOriginalBinding']:raise ValueError('Unaudited eye binding for baseline crop')
    eyeBinding=ibugClientLandmarks(binding,groups=('eyes',))
    eyeIds=np.unique(eyeBinding['vertexIndices'][eyeBinding['validMask']])
    return dict(entry=entry,identities=identities,archives=archives,reports=reports,raw=raw,similarity=similarity,
        full=full,photos=photos,cameras=cameras,inputs=inputs,noseIds=noseIds,eyeIds=eyeIds,
        materials=materialComparison,splitPath=splitPath,runRoot=runRoot,plan=plan)


def run(splitPath, runRoot, caseId, outputDirectory, size=512):
    if type(size) is not int or not 256<=size<=1024:raise ValueError('Expected render size in [256,1024]')
    output=Path(outputDirectory).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh render directory')
    state=prepare(splitPath,runRoot,caseId)
    renderers={name:EvidenceRenderer(path,state['similarity'],device='cpu',facesPerPixel=12)
               for name,path in state['archives'].items()}
    factorial=EvidenceRenderer(state['archives']['pureDirect'],state['similarity'],device='cpu',facesPerPixel=12)
    factorial.torch.set_num_threads(1)
    from pytorch3d.structures import Meshes
    heads={branch:renderers[branch].maps[0].clone() for branch in BRANCHES}
    baseOtherMaps=[texture.clone() for texture in factorial.maps[1:]]
    materialDifference=not all(value['materialDefinitionsIdenticalToPairedBaseline']
        and all(value['nonHeadTexturePayloadsIdenticalToPairedBaseline'].values())
        for name,value in state['materials'].items() if name in BRANCHES)
    gates={branch:{'downstreamNumericalAccepted':state['reports'][branch]['geometry']['downstreamNumericalAccepted'],
                   'finalCavityGuard':state['reports'][branch]['geometry']['finalCavityGuard']}
           for branch in BRANCHES}
    gateLabel=('下游/鼻腔局部门通过；完整装配待审' if all(row['downstreamNumericalAccepted'] and row['finalCavityGuard']['accepted']
               for row in gates.values()) else '含未过门分支 · 仅供诊断')
    output.mkdir(parents=True,exist_ok=True);views=[]
    for index,metadata in enumerate(state['inputs']):
        camera=scaledViewCamera(state['cameras'][index],metadata,index,size)
        crops={};cropReports={}
        for name,ids in (('nose',state['noseIds']),('eyes',state['eyeIds'])):
            crops[name],cropReports[name]=baselinePatchCrop(state['raw']['pureDirect'],ids,camera,size)
        photoPath=state['photos']/f'input{index:02d}.png'
        with Image.open(photoPath) as source:
            if list(source.size)!=metadata['workingSize']:raise ValueError('Photo/frame dimensions differ')
            photo=source.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
        saveSrgb(photo,output/f'view{index:02d}_photo.png')
        primary={};stats={}
        for name,renderer in renderers.items():
            # Each asset has independent full geometry and fresh normal caches.
            renderer.mesh=Meshes(verts=[renderer.vertices],faces=[renderer.faces])
            primary[name],stats[name]=renderer.variants(camera,(size,size))
            for mode,_ in MODES:saveSrgb(solid(primary[name][mode]),output/f'view{index:02d}_{name}_{mode}.png')
        factorialImages={};factorialStats={}
        for geometry in BRANCHES:
            for texture in BRANCHES:
                key=geometry+'Geometry_'+texture+'Texture'
                factorial.vertices=factorial.torch.as_tensor(state['full'][geometry],dtype=factorial.torch.float32)
                factorial.maps=[heads[texture].clone(),*[value.clone() for value in baseOtherMaps]]
                factorial.mesh=Meshes(verts=[factorial.vertices],faces=[factorial.faces])
                factorialImages[key],factorialStats[key]=factorial.variants(camera,(size,size))
                for mode,_ in MODES:saveSrgb(solid(factorialImages[key][mode]),output/f'view{index:02d}_{key}_{mode}.png')
                if geometry==texture and not materialDifference:
                    for mode,_ in MODES:
                        np.testing.assert_array_equal(np.asarray(factorialImages[key][mode]),np.asarray(primary[geometry][mode]))
        light=stats['pureDirect']['lightDirectionCanonical']
        for row in [*stats.values(),*factorialStats.values()]:
            np.testing.assert_array_equal(row['lightDirectionCanonical'],light)
        panels=[];labels=[]
        for mode,modeLabel in MODES:
            row=[photo]+[solid(primary[branch][mode]) for branch in PRIMARY]
            panels.extend(row);labels.extend([LABELS[0]]+[label+' · '+modeLabel for label in LABELS[1:]])
            for cropName,crop in crops.items():
                picture=noseGrid(row,LABELS,crop,f'视角 {index+1} · {modeLabel} · '+('鼻部' if cropName=='nose' else '眼部')+'共用模板投影裁剪')
                saveSrgb(banner(picture,caseId,gateLabel),output/f'view{index:02d}_{mode}_{cropName}.png')
            factorialKeys=[g+'Geometry_'+t+'Texture' for g in BRANCHES for t in BRANCHES]
            factorialPanels=[solid(factorialImages[key][mode]) for key in factorialKeys]
            factorialLabels=[f'{g} 几何 × {t} 头贴图' for g in BRANCHES for t in BRANCHES]
            sheet=evidenceGrid(factorialPanels,factorialLabels,columns=2,panelSize=size,
                subtitle=f'视角 {index+1} · {modeLabel} · 非头部材质固定为配对 pureDirect 基线')
            saveSrgb(banner(sheet,caseId,gateLabel,factorial=True),output/f'view{index:02d}_{mode}_factorial.png')
            cropSheet=evidenceGrid([panel.crop(crops['nose']) for panel in factorialPanels],factorialLabels,
                columns=2,panelSize=256,subtitle=f'视角 {index+1} · {modeLabel} · 相同鼻部裁剪')
            saveSrgb(banner(cropSheet,caseId,gateLabel,factorial=True),output/f'view{index:02d}_{mode}_factorialNose.png')
        sheet=evidenceGrid(panels,labels,columns=4,panelSize=size,
            subtitle=f'视角 {index+1} · 原照片 / 历史参照 / 配对基线 / 新解 · 相机未重拟合')
        saveSrgb(banner(sheet,caseId,gateLabel),output/f'view{index:02d}Comparison.png')
        views.append({'viewIndex':index,'camera':camera,'photoSha256':fileHash(photoPath),'crops':crops,
                      'cropProjection':cropReports,'stats':stats,'factorialStats':factorialStats})
        print(json.dumps({'caseId':caseId,'renderedView':index,'pairedNumericalGates':gates}),flush=True)
    verifyHashes(state['identities'])
    report={'status':'completeDiagnosticRendering','renderer':'renderShapePipelineComparison','caseId':caseId,
        'split':state['entry']['split'],'sourceDomain':state['entry']['sourceDomain'],
        'domainEvidence':state['entry']['domainEvidence'],'size':size,'device':'cpu','views':views,
        'splitPath':str(state['splitPath']),'runRoot':str(state['runRoot']),
        'archives':{name:str(path) for name,path in state['archives'].items()},
        'gateReports':{branch:{key:state['reports'][branch][key] for key in ('geometryPath','appearancePath')} for branch in BRANCHES},
        'pairedNumericalGates':gates,'materialComparison':state['materials'],
        'inputHashesBeforeAndAfter':state['identities'],'inputHashesUnchanged':True,
        'imageSha256':{path.name:fileHash(path) for path in sorted(output.glob('*.png'))},
        'topologyUvMaterialOrderBindingIdentical':True,'staticCoordinatesPreservedWithinExportPrecision':True,
        'downstreamEyeMouthGeometryRerun':True,'textureRebakedForFinalGeometry':True,
        'historicalAssetIsPairedControl':False,'oldGeometryDeltasReused':False,
        'factorial':{'geometryBranches':list(BRANCHES),'headTextureBranches':list(BRANCHES),
            'nonHeadTexturesAndMaterialDefinitionsFixedTo':'pureDirect',
            'pairedNonHeadMaterialsOriginallyDiffered':materialDifference,
            'sameGeometrySameTexturePixelIdentityChecked':not materialDifference,
            'headColorDecode':'sRGB PNG decoded to linear once by EvidenceRenderer',
            'mixedPairsAreProposedAssets':False},
        'cropDerivedFromBaselinePatch':True,'noseCropVertexIds':state['noseIds'],'eyeCropVertexIds':state['eyeIds'],
        'newPhotoAnnotation':False,'cameraRefitted':False,'lightFrame':'cameraRelative',
        'photoIlluminationFitted':False,'usesGeometricNormals':True,'clientAssetExportedByRenderer':False,
        'deployed':False,'qualityAccepted':False,'finalCandidateAccepted':False,
        'limitations':['Numerical gates do not establish photographic or client acceptance.',
            'The historical AI asset contains an additional eye-ring experiment excluded from both current paired branches.',
            'The factorial includes crossed geometry/texture combinations only to diagnose causes; they are not deliverables.',
            'The fresh common photo inference can differ from historical original-image inference.',
            'Appearance is fitted to the input photographs and cannot independently establish relighting accuracy.',
            'Saved estimated cameras and geometric-normal inspection lighting are shared, not refitted to the photographs.',
            'Recorded source-domain provenance is retained; appearance does not authenticate human or AI origin.']}
    writeJson(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',type=Path,default=ROOT/'runs/shapePreservation20261024/split.json')
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/shapePipeline20261024')
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args()
    run(args.split,args.run_root,args.case,args.output,args.size)
