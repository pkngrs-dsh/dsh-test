"""Locate the actually visible eye/skin interface with fixed-camera CPU rays.

This separates hidden topology-hole rims from visible eyelid geometry. Opaque
materials 0/1/2 are used; current brow/lash cards must be hidden in the export.
"""
from pathlib import Path
import json
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, label

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from faceAppearance.camera import projectPoints
from faceAppearance.artifacts import writeJson
from runReferenceExperiment import fileHash, readArrays
from runEyeRefinementExperiment import loadEyeCache
from renderCropComparison import photoCamera, validatePhotoPair


def main():
    source=ROOT/'runs/service/b775794b61894a8b804fddb0620db45d/output'
    asset=ROOT/'runs/sdfSkinTemplateBlendFinal/clientAssets/sdf'
    output=ROOT/'runs/clientEyeMappingAudit/visibleEye'
    if output.exists():
        raise FileExistsError('Use a new diagnostic output; existing evidence is preserved')
    mesh=readArrays(asset/'clientMesh.npz')
    template=readArrays(ROOT/'assets/clientTemplates/legacyFemaleV1/template.npz')
    mappingPath=ROOT/'runs/clientEyeMappingAudit/staticMapping.json'
    mapping=json.loads(mappingPath.read_text())
    diagnostics=json.loads((asset/'diagnostics.json').read_text())
    if diagnostics['texture']['hiddenEyebrowMaterialIds'] != [3] or diagnostics['texture']['hiddenEyelashMaterialIds'] != [3]:
        raise ValueError('Current diagnostic assumes the exported brow/lash cards are hidden')
    inputs=json.loads((source/'inputs.json').read_text())
    observations,_,cacheHashes=loadEyeCache(ROOT/'runs/eyeLandmarkValidatedCache',inputs,minimumRawPeak=.6)
    observed=observations[0]
    camera=photoCamera(json.loads((source/'cameras.json').read_text())[0],inputs[0],'fitted')
    photo=validatePhotoPair(inputs[0],inputs[0])
    transform=diagnostics['templateToUff']
    vertices=mesh['vertices']*100@np.array(transform['rotation'])*transform['scale']+transform['translation']
    cameraVertices=vertices@np.array(camera['rotation']).T+camera['translation']
    projected,_=projectPoints(vertices,camera['intrinsics'],camera['rotation'],camera['translation'])
    height,width=tuple(reversed(inputs[0]['orientedSize']))
    headIds=np.flatnonzero(mesh['materialIds']==0)
    headSampler=ProjectedSurfaceVisibility(cameraVertices,mesh['faces'][headIds],camera['intrinsics'],(height,width))
    hiddenRecords={}
    for name,eye in mapping['eyes'].items():
        ids=np.asarray(eye['clientFullVertexIds']);hit=headSampler.closestDepth(cameraVertices[ids])
        gap=cameraVertices[ids,2]-hit['depth']
        tolerance=.005*np.ptp(vertices,axis=0).max()
        hiddenRecords[name]={'fullVertexIds':ids,'depthGapCanonical':gap,'occludedByHeadSkin':gap>tolerance,
                             'occludedCount':int((gap>tolerance).sum()),'depthToleranceCanonical':float(tolerance)}
    triangles=cameraVertices[mesh['faces']]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    front=np.sum(cross*(-triangles.mean(1)),axis=1)>0
    faceIds=np.flatnonzero((mesh['materialIds']!=3)&front)
    sampler=ProjectedSurfaceVisibility(cameraVertices,mesh['faces'][faceIds],camera['intrinsics'],(height,width))
    points=np.concatenate([np.r_[eye['upperCurve'],eye['lowerCurve']] for eye in observed['eyes'].values()])
    minimum=np.maximum(np.floor(points.min(0)-28),0).astype(int)
    maximum=np.minimum(np.ceil(points.max(0)+28),[width,height]).astype(int)
    xx,yy=np.meshgrid(np.arange(minimum[0],maximum[0]+1),np.arange(minimum[1],maximum[1]+1))
    pixels=np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])
    rays=pixels@np.linalg.inv(np.array(camera['intrinsics'])).T
    hits=sampler.closestDepth(rays)
    globalFaces=np.full(len(rays),-1,dtype=int);valid=hits['faceIds']>=0
    globalFaces[valid]=faceIds[hits['faceIds'][valid]]
    materials=np.full(len(rays),-1,dtype=int);materials[valid]=mesh['materialIds'][globalFaces[valid]]
    materials=materials.reshape(xx.shape);globalFaces=globalFaces.reshape(xx.shape)
    eyeMask=materials==2
    components,count=label(eyeMask)
    sizes=np.bincount(components.ravel());sizes[0]=0
    if np.count_nonzero(sizes>=20)!=2:
        raise ValueError(f'Expected two visible eye regions, got component sizes {sizes.tolist()}')
    inverse=np.full(len(vertices),-1,int);inverse[template['nicpToFullIds']]=np.arange(len(template['nicpToFullIds']))
    records=[]
    colors=['#4bea9c','#66bbff']
    crop=(*minimum.tolist(),*(maximum+1).tolist())
    scale=3
    base=photo.crop(crop).resize((xx.shape[1]*scale,xx.shape[0]*scale),Image.Resampling.LANCZOS)
    panels=[base.copy(),base.copy()]
    def xy(value):return [tuple(p) for p in (np.asarray(value)-minimum+.5)*scale-.5]
    for labelId,eye in observed['eyes'].items():
        for tile in panels:
            draw=ImageDraw.Draw(tile)
            for part in ('upperCurve','lowerCurve'):
                draw.line(xy(eye[part]),fill='#ffda55',width=2)
    for index,componentId in enumerate(np.flatnonzero(sizes>=20)):
        region=components==componentId
        yyEye,xxEye=np.nonzero(region)
        center=np.array([xxEye.mean(),yyEye.mean()])+minimum
        labelId=min(observed['eyes'],key=lambda k:np.linalg.norm(center-(observed['eyes'][k]['innerCorner']+observed['eyes'][k]['outerCorner'])/2))
        adjacent=binary_dilation(region)&(materials==0)
        adjacentFaces=np.unique(globalFaces[adjacent]);fullIds=np.unique(mesh['faces'][adjacentFaces])
        nicp=inverse[fullIds];skin=nicp>=0
        weights=np.full(len(fullIds),np.nan);weights[skin]=template['geometryWeights'][nicp[skin]]
        contours,_=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=len)[:,0]+minimum
        record={'labelId':int(labelId),'visiblePixels':int(region.sum()),'visibleEyeContourPixels':contour,
                'headFacesAdjacentToVisibleEye':adjacentFaces,'adjacentFullVertexIds':fullIds,
                'adjacentNicpVertexIds':nicp,'geometryWeights':np.nan_to_num(weights,nan=-1),
                'skinVertexCount':int(skin.sum()),'activeGeometryVertexCount':int(np.count_nonzero(weights>0)),
                'widthHeightPixels':[int(xxEye.max()-xxEye.min()+1),int(yyEye.max()-yyEye.min()+1)],
                'scope':'Vertices of head triangles next to visible eye pixels; an image-dependent neighborhood, not a bound eyelid curve'}
        records.append(record)
        for tile in panels:
            ImageDraw.Draw(tile).line(xy(np.r_[contour,contour[:1]]),fill=colors[index],width=2)
        draw=ImageDraw.Draw(panels[1])
        for vertex,weight in zip(fullIds,weights):
            x,y=xy([projected[vertex]])[0]
            color='#72ff72' if weight>0 else '#ff617c'
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    draw=ImageDraw.Draw(panels[0])
    for name,eye in mapping['eyes'].items():
        loop=np.array(eye['clientFullVertexIds'])
        draw.line(xy(projected[np.r_[loop,loop[:1]]]),fill='#ff709d',width=2)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    final=Image.new('RGB',(base.width*2,base.height+100),'#152128')
    titles=['Pink: hidden rear hole | Green/blue: visible eye boundary','Visible skin neighbors: green=active, red=zero weight']
    for i,tile in enumerate(panels):
        final.paste(tile,(i*base.width,62))
        ImageDraw.Draw(final).text((i*base.width+10,10),titles[i],font=font,fill='white')
        ImageDraw.Draw(final).text((i*base.width+10,34),'Yellow: automatic photo eyelid observation',font=font,fill='#ffdb55')
    ImageDraw.Draw(final).text((10,base.height+75),'Fixed frontal camera; opaque front faces; topology-hole boundary is NOT visible eyelid margin',font=font,fill='white')
    output.mkdir(parents=True)
    final.save(output/'visibleVersusTopology.png')
    np.savez_compressed(output/'visibleFaces.npz',pixelX=xx,pixelY=yy,materialIds=materials,faceIds=globalFaces,eyeMask=eyeMask)
    report={'status':'completed','camera':camera,'sourceRun':str(source),'client':str(asset),
            'hiddenTopologyRims':hiddenRecords,'visibleEyes':records,
            'method':'CPU exact ray nearest opaque front-facing triangle per original photo integer pixel',
            'excludedMaterial':3,'geometryChanged':False,'cameraRefitted':False,
            'inputHashes':{str(p):fileHash(p) for p in [source/'geometry.npz',source/'cameras.json',source/'inputs.json',asset/'clientMesh.npz',asset/'diagnostics.json',mappingPath]},
            'cacheHashes':cacheHashes,
            'limits':['Visible material boundary is view-dependent and includes template eye-piece geometry',
                      'Raster contour accuracy is about one original-photo pixel; no subpixel silhouette fitting',
                      'Neighbor triangle vertices are not an anatomical eyelid annotation',
                      'Rear-hole qualification must not be presented as qualification of visible eyelid']}
    writeJson(output/'visibleEyeMapping.json',report)
    (output/'diagnoseVisibleClientEyes.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':'completed','hidden':[v['occludedCount'] for v in hiddenRecords.values()],
        'visible':[ {k:r[k] for k in ('labelId','visiblePixels','skinVertexCount','activeGeometryVertexCount','widthHeightPixels')} for r in records]}))


if __name__=='__main__':main()
