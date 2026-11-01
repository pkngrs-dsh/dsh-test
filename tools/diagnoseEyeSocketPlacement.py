"""Controlled current-ZIP eye placement comparison; does not run model inference."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.legacyPreviewAsset import _obj


def render(mesh, vertices, textures, bounds, *, yaw=0., width=900):
    """Fixed orthographic camera, opaque depth and alpha-tested original maps."""
    xmin,xmax,ymin,ymax=bounds
    height=round(width*(ymax-ymin)/(xmax-xmin))
    angle=np.deg2rad(yaw)
    rotation=np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
    pivot=np.array([0,(ymin+ymax)/2,6.5])
    v=(vertices-pivot)@rotation+pivot
    screen=np.c_[(v[:,0]-xmin)/(xmax-xmin)*(width-1),(ymax-v[:,1])/(ymax-ymin)*(height-1)]
    image=np.full((height,width,3),[28,35,39],np.uint8)
    depth=np.full((height,width),-np.inf)
    faceBuffer=np.full((height,width),-1,np.int64)
    for fi,face in enumerate(mesh['faces']):
        tri=screen[face];lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),[width-1,height-1])
        if np.any(lo>hi):continue
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;wc=1-wa-wb
        z=wa*v[face[0],2]+wb*v[face[1],2]+wc*v[face[2],2]
        target=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)&(z>target)
        if not keep.any():continue
        uv=mesh['uvs'][mesh['uvFaces'][fi]];coords=wa[...,None]*uv[0]+wb[...,None]*uv[1]+wc[...,None]*uv[2]
        tex=textures[int(mesh['materialIds'][fi])]
        xy=np.clip(coords,0,1)*[tex.shape[1]-1,tex.shape[0]-1]
        q=np.floor(xy).astype(int);r=np.minimum(q+1,[tex.shape[1]-1,tex.shape[0]-1]);f=xy-q
        colors=((tex[q[...,1],q[...,0]]*(1-f[...,0,None])+tex[q[...,1],r[...,0]]*f[...,0,None])*(1-f[...,1,None])
               +(tex[r[...,1],q[...,0]]*(1-f[...,0,None])+tex[r[...,1],r[...,0]]*f[...,0,None])*f[...,1,None])
        if int(mesh['materialIds'][fi])==3:
            keep&=colors[...,3]>=127.5
        target[keep]=z[keep]
        image[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=np.rint(colors[...,:3][keep]).astype(np.uint8)
        faceBuffer[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=fi
    return Image.fromarray(image),faceBuffer


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--asset',type=Path,default=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        manifest=json.loads(z.read('legacyManifest.json'))
        mesh=_obj(z.read('face_nicp.obj'),manifest)
        textures=[np.asarray(Image.open(io.BytesIO(z.read(name))).convert('RGBA'),float) for name in
                  ('head_color.png','body_color.png','eye_color.png','brow_color.png')]
    arrays=dict(np.load(args.asset/'template.npz',allow_pickle=False))
    components=json.loads((args.asset/'components.json').read_text())
    before=mesh['vertices']*100.;after=before.copy();records=[]
    if not args.baseline_only:
        from faceAppearance.eyeSocketPlacement import buildEyeSocketBindings,placeEyeInSocket,_sphere
        bindings=buildEyeSocketBindings(arrays,components)
        for key,binding in bindings.items():
            placement=placeEyeInSocket(binding,before[arrays['nicpToFullIds']])
            after[binding['componentVertexIds']]=placement['vertices']
            ids=binding['componentVertexIds'];original=binding['templateEyeVertices'];old=before[ids]
            oldCenter,oldRadius=_sphere(old)
            u,_,vt=np.linalg.svd((original-original.mean(0)).T@(old-old.mean(0)))
            angle=float(np.degrees(np.arccos(np.clip((np.trace(u@vt)-1)/2,-1,1))))
            target=(before[arrays['nicpToFullIds']][binding['vertexIndices']]*binding['barycentricWeights'][...,None]).sum(1)
            restGap=np.linalg.norm(binding['restPoints']-binding['sphereCenter'],axis=1)-binding['sphereRadius']
            oldGap=np.linalg.norm(target-oldCenter,axis=1)-oldRadius
            records.append({'component':key,**placement['diagnostics'],
                'beforeSphereCenterCm':oldCenter,'beforeSphereRadiusCm':oldRadius,
                'beforeIrisRotationDegrees':angle,
                'centerChangeFromDeliveredMm':(placement['diagnostics']['placedSphereCenterCm']-oldCenter)*10,
                'beforeTemplateGapErrorRmsMm':float(np.sqrt(np.mean((oldGap-restGap)**2))*10),
                'afterTemplateGapErrorRmsMm':placement['diagnostics']['radialGapChangeCm']['rms']*10,
                'beforeWorstRimInsideSphereMm':float(max(0,-oldGap.min())*10),
                'afterWorstRimInsideSphereMm':float(max(0,-placement['diagnostics']['placedRadialGapCm']['min'])*10)})
        np.savez_compressed(args.output/'eyePlacementComparison.npz',before=before,after=after,**{k:mesh[k] for k in ('faces','uvs','uvFaces','materialIds')})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    panels=[]
    for label,yaw in [('Front',0.),('Yaw -25',-25.),('Yaw +25',25.)]:
        row=[]
        for name,vertices in [('Before',before),('After',after)]:
            image,fb=render(mesh,vertices,textures,(-6,6,167.7,171.1),yaw=yaw)
            for record in records:
                component=next(c for c in components if c.get('label')==record['component'])
                record.setdefault('visibleEyePixels',{}).setdefault(label,{})[name]=int(np.isin(fb,component['faceIndices']).sum())
            image.save(args.output/f'{name.lower()}_{int(yaw):+03d}.png')
            tile=Image.new('RGB',(image.width,image.height+38),(28,35,39));tile.paste(image,(0,38))
            ImageDraw.Draw(tile).text((14,7),f'{name} | {label} | same original textures',font=font,fill='white')
            row.append(tile)
        panels.append(row)
    result=Image.new('RGB',(panels[0][0].width*2,sum(row[0].height for row in panels)),(28,35,39));y=0
    for row in panels:
        result.paste(row[0],(0,y));result.paste(row[1],(row[0].width,y));y+=row[0].height
    result.save(args.output/'eyeComparison.png')
    if records:
        (args.output/'eyePlacement.json').write_text(json.dumps(plain(records),indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'eyeCount':len(records)}))


if __name__=='__main__':main()
