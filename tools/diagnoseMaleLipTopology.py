"""Camera-space and template-topology diagnosis of the frozen male lip crease.

The trace is the deepest *visible* surface point in a restricted mouth band,
not an anatomical landmark and not necessarily one connected mesh edge chain.
No frozen geometry or observations are changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.camera import projectPoints
from auditMaleMouthSeam import raster


def traceVisibleLipCrease(vertices, faces, camera, *, xRange=(218., 312.),
                         yRange=(344., 366.), samplesPerAxis=4):
    """Return the per-column visible depth peak inside a fixed mouth ROI.

    ``pixels`` has N by 2 working-image coordinates at 1/samplesPerAxis steps;
    ``faceIds``, ``barycentricWeights`` and ``points`` identify the same surface
    samples. ``validMask`` excludes missing rays and peaks on the y search
    boundary. ``depthProminence`` is peak depth minus the line interpolated
    between the two y-band endpoint depths. ``nextRowDepthDrop`` describes
    how abruptly the lower lip occludes the deeper surface beneath the peak.

    Faces are rasterized double-sided as in the cached scientific preview.
    All x/y values use integer working-image pixel centers. The trace is a
    geometric, camera-dependent fold/occlusion proxy; it does not claim to be
    an anatomical mouth boundary or a single connected mesh chain.
    """
    vertices=np.asarray(vertices, dtype=float); faces=np.asarray(faces, dtype=np.int64)
    if samplesPerAxis < 1 or int(samplesPerAxis) != samplesPerAxis:
        raise ValueError('samplesPerAxis must be a positive integer')
    if not xRange[0] < xRange[1] or not yRange[0] < yRange[1]:
        raise ValueError('Expected increasing mouth search ranges')
    box=[int(np.floor(xRange[0])),int(np.floor(yRange[0])),
         int(np.ceil(xRange[1]))+1,int(np.ceil(yRange[1]))+1]
    buffers=raster({'vertices':vertices,'faces':faces,'uvs':np.zeros((len(vertices),2)),
                    'uvFaces':faces},camera,box,samples=samplesPerAxis)
    s=samplesPerAxis
    xs=np.arange(buffers['depth'].shape[1])/s+box[0]-(s-1)/(2*s)
    ys=np.arange(buffers['depth'].shape[0])/s+box[1]-(s-1)/(2*s)
    cols=np.where((xs>=xRange[0])&(xs<=xRange[1]))[0]
    rows=np.where((ys>=yRange[0])&(ys<=yRange[1]))[0]
    window=buffers['depth'][np.ix_(rows,cols)]
    peaks=np.argmax(np.where(np.isfinite(window),window,-np.inf),axis=0)
    peakRows=rows[peaks]
    faceIds=buffers['faceMap'][peakRows,cols]
    valid=(faceIds>=0)&(peaks>0)&(peaks<len(rows)-1)
    pixelPoints=np.column_stack([xs[cols],ys[peakRows]])
    screen=buffers['pixels'][faces[np.maximum(faceIds,0)]]
    a,b,c=screen[:,0],screen[:,1],screen[:,2]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    valid &= np.abs(den)>1e-12
    den=np.where(np.abs(den)>1e-12,den,1.)
    wa=((b[:,1]-c[:,1])*(pixelPoints[:,0]-c[:,0])+(c[:,0]-b[:,0])*(pixelPoints[:,1]-c[:,1]))/den
    wb=((c[:,1]-a[:,1])*(pixelPoints[:,0]-c[:,0])+(a[:,0]-c[:,0])*(pixelPoints[:,1]-c[:,1]))/den
    weights=np.column_stack([wa,wb,1-wa-wb])/buffers['cameraVertices'][faces[np.maximum(faceIds,0)],2]
    weights/=np.maximum(weights.sum(-1,keepdims=True),1e-12)
    points=(vertices[faces[np.maximum(faceIds,0)]]*weights[...,None]).sum(1)
    peakDepth=buffers['depth'][peakRows,cols]
    fraction=(ys[peakRows]-ys[rows[0]])/(ys[rows[-1]]-ys[rows[0]])
    baseline=window[0]*(1-fraction)+window[-1]*fraction
    drop=peakDepth-buffers['depth'][np.minimum(peakRows+1,len(ys)-1),cols]
    return {'pixels':pixelPoints,'faceIds':faceIds,'vertexIndices':faces[np.maximum(faceIds,0)],
            'barycentricWeights':weights,'points':points,'cameraDepth':peakDepth,
            'validMask':valid,'depthProminence':peakDepth-baseline,'nextRowDepthDrop':drop,
            'sampleStepPixels':1/s,'xRange':list(xRange),'yRange':list(yRange),
            'definition':'visible z-buffer maximum along y inside fixed mouth band; camera-dependent fold/occlusion proxy'}


def _load(path):
    with np.load(path,allow_pickle=False) as value:return dict(value)


def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project(points,camera):
    return projectPoints(points,np.asarray(camera['intrinsics']),np.asarray(camera['rotation']),np.asarray(camera['translation']))[0]


def _rimPaths(rest,faces):
    """Topology hypotheses, seeded by inspected central upper/lower inner rows.

    Center seeds 995 and 1968 are on separate inward-folding lip rows. The
    short paths are verified mesh edges, but the corner routes are ambiguous
    in the compressed template and are not production semantic annotations.
    """
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    keep=(np.abs(rest[:,0])<2.45)&(rest[:,1]>161.65)&(rest[:,1]<162.15)&(rest[:,2]>7.)
    selected=edges[keep[edges].all(1)]
    cost=np.linalg.norm(rest[selected[:,0]]-rest[selected[:,1]],axis=1)*(1+5*np.abs(rest[selected].mean(1)[:,1]-161.84))
    graph=coo_matrix((np.tile(cost,2),(np.concatenate([selected[:,0],selected[:,1]]),
        np.concatenate([selected[:,1],selected[:,0]]))),shape=(len(rest),len(rest))).tocsr()
    results={}
    for name,seed in [('upperInner',995),('lowerInner',1968)]:
        _,predecessor=dijkstra(graph,indices=seed,return_predecessors=True)
        sides=[]
        for corner in [168,1143]:
            chain=[corner]
            while chain[-1]!=seed:
                parent=int(predecessor[chain[-1]])
                if parent<0:raise ValueError('No mouth-band topology path')
                chain.append(parent)
            sides.append(chain)
        ids=np.array(sides[0]+list(reversed(sides[1]))[1:])
        results[name]={'vertexIds':ids,'centerSeed':seed,
            'centralTrustedMask':np.abs(rest[ids,0])<=1.4,
            'scope':'edge-connected candidate inner row; center inspected, corner shortest-path route is approximate'}
    return results


def main():
    out=ROOT/'runs/maleLipGeometry20261012/topology';out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((ROOT/'reports/clientColmapBaseline20261012.json').read_text())
    for item in baseline['fixedReferenceArtifacts'].values():
        if _digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen input changed: '+item['path'])
    frozen=ROOT/'runs/clientColmap20261012/maleExactRegression'
    geometry=_load(frozen/'geometry.npz');state=_load(ROOT/'runs/frontLandmark20261012/maleExactRegression/landmarks.npz')
    arrays=_load(ROOT/'assets/clientTemplates/legacyMaleV1/template.npz')
    rest=arrays['vertices'][arrays['nicpToFullIds']];vertices=geometry['vertices'];faces=geometry['faces']
    camera=json.loads((ROOT/baseline['fixedReferenceArtifacts']['cameras']['path']).read_text())[0]
    trace=traceVisibleLipCrease(vertices,faces,camera)
    np.savez_compressed(out/'creaseTrace.npz',**{k:v for k,v in trace.items() if isinstance(v,np.ndarray)})
    paths=_rimPaths(rest,faces);pixels=_project(vertices,camera)
    binding=(vertices[state['vertexIndices']]*state['barycentricWeights'][...,None]).sum(1)
    bindingRest=(rest[state['vertexIndices']]*state['barycentricWeights'][...,None]).sum(1)
    bindingPixels=_project(binding,camera);traceXY=trace['pixels'][trace['validMask']]
    bindingReport=[]
    for i in np.where(state['ibugIds']>=60)[0]:
        bxy=bindingPixels[i];creaseY=float(np.interp(bxy[0],traceXY[:,0],traceXY[:,1]))
        bindingReport.append({'ibugId':int(state['ibugIds'][i]),'vertexIndices':state['vertexIndices'][i].tolist(),
            'barycentricWeights':state['barycentricWeights'][i].tolist(),'templatePointCm':bindingRest[i].tolist(),
            'projectedPixels':bxy.tolist(),'creaseYAtSameX':creaseY,'bindingYMinusVisibleCrease':float(bxy[1]-creaseY),
            'photoDetection':state['frontTargetPixels'][i].tolist()})
    # Sample the darkest observed photograph feature inside a fixed mouth band.
    # This is explicitly a diagnostic appearance proxy, not a labeled landmark.
    inputs=json.loads((ROOT/baseline['fixedReferenceArtifacts']['inputManifest']['path']).read_text())
    raw=ImageOps.exif_transpose(Image.open(baseline['sourceImages'][0]['path'])).convert('RGB')
    work=cv2.warpPerspective(np.asarray(raw),np.asarray(inputs[0]['inputToWorking']),tuple(inputs[0]['workingSize']))
    luma=cv2.GaussianBlur(work@np.array([.2126,.7152,.0722]),(0,0),.65)
    photoX=np.arange(222,303);photoY=343+np.argmin(luma[343:362,photoX],axis=0)
    photoTrace=np.column_stack([photoX,photoY]).astype(float)
    np.savez_compressed(out/'topologyChains.npz',**{name:record['vertexIds'] for name,record in paths.items()},
        upperCentralIds=paths['upperInner']['vertexIds'][paths['upperInner']['centralTrustedMask']],
        lowerCentralIds=paths['lowerInner']['vertexIds'][paths['lowerInner']['centralTrustedMask']],
        photoDarkFeaturePixels=photoTrace)
    hits=json.loads((ROOT/'runs/maleMouthSeam20261012/audit/report.json').read_text())['nonadjacentIntersections']['mouthPairs']
    collisionGroups={}
    for name,select in [('imageLeft',lambda x:x<260),('imageRight',lambda x:x>=260)]:
        selected=[hit for hit in hits if select(hit['projectedPoint'][0])];points=np.array([x['projectedPoint'] for x in selected])
        collisionGroups[name]={'pairCount':len(selected),'projectedBox':[*points.min(0).tolist(),*points.max(0).tolist()],
            'facePairs':[hit['faceIds'] for hit in selected]}
    # Count the trace's face sequence to show it is an occlusion envelope, not
    # a single mesh edge ring. Neighboring trace samples can switch surfaces.
    seq=trace['faceIds'][trace['validMask']];transitions=[]
    for i in np.where(seq[1:]!=seq[:-1])[0]:
        shared=np.intersect1d(faces[seq[i]],faces[seq[i+1]])
        transitions.append({'x':float(traceXY[i,0]),'faceIds':[int(seq[i]),int(seq[i+1])],
                            'sharedVertexCount':len(shared)})
    central=(photoX>=230)&(photoX<=295)
    offset=np.interp(photoX,traceXY[:,0],traceXY[:,1])-photoY
    report={'caseId':'maleExactRegression','geometrySha256':_digest(frozen/'geometry.npz'),
        'trace':{'definition':trace['definition'],'workingPixelXRange':trace['xRange'],'workingPixelYSearchRange':trace['yRange'],
            'samples':len(trace['pixels']),'validSamples':int(trace['validMask'].sum()),
            'distinctVisibleFaces':len(np.unique(seq)),'nonEdgeAdjacentVisibleFaceTransitions':sum(x['sharedVertexCount']<2 for x in transitions),
            'transitions':transitions,'depthProminenceMedianCanonical':float(np.median(trace['depthProminence'][trace['validMask']])),
            'surfaceSampleMaxReprojectionErrorPixels':float(np.linalg.norm(_project(trace['points'],camera)-trace['pixels'],axis=1).max())},
        'candidateInnerRows':{name:{**r,'vertexIds':r['vertexIds'].tolist(),'centralTrustedMask':r['centralTrustedMask'].tolist(),
            'projectedPixels':pixels[r['vertexIds']].tolist()} for name,r in paths.items()},
        'approximateInnerLipBinding':bindingReport,'existingCollisions':collisionGroups,
        'photoFeatureProxy':{'method':'columnwise darkest Gaussian-blurred sRGB luminance in y343:361, x222:302; not labeled anatomical seam',
            'pixels':photoTrace.tolist(),'centralXRange':[230,295],
            'geometryBelowPhotoMedianPixels':float(np.median(offset[central])),
            'geometryBelowPhotoMinMaxPixels':[float(offset[central].min()),float(offset[central].max())]},
        'findings':[
            'The gray crease is a recessed upper/inner lip surface that is abruptly occluded by the nearer lower lip; it is a visible-depth envelope, not a topological hole.',
            'Template center upper row is 1917 -> 995 -> 994 -> 2633 (outer rim to inward fold); lower row is 1185 -> 1968 -> 2680 -> 2623. Upper and lower surfaces are separate until their mouth-corner paths.',
            'iBUG 61/62/63 binding lies on the outer upper rim; iBUG 65/66/67 lies nearer the inward lower row. In particular 62 is mostly vertex 1917, while 66 is mostly vertex 1968. These are not corresponding depth rings.',
            'All 55 existing nonadjacent intersections are near projected corners: 8 left and 47 right. None occupies the central x230:290 visible seam, so central darkness cannot be attributed to those pairs.',
            'Lighting exaggerates the recess because the upper wall normal rotates away from the light while the lower surface has a contrasting normal. Depth is finite everywhere and no oral boundary ray misses the mesh.'],
        'recommendedTarget':{
            'firstPass':'Use the dense visible-depth trace as a camera-dependent positional proxy; estimate a smooth y-displacement toward the frontal photo dark feature over the central span, taper near corners, and apply one shared monotonic XY map to every lip layer.',
            'doNotInfer':'Do not treat the trace as one anatomical vertex chain. Do not close the mouth, flatten its depth or independently pull the approximate top/bottom iBUG bindings from this evidence alone.',
            'geometryChecks':'Keep camera depth, UV and fixed vertices unchanged; bound the XY-map Jacobian away from zero and recheck cumulative triangle areas, normals, edge lengths and the actual intersection pairs.',
            'evaluation':'Re-extract the visible trace on each candidate and compare all four views with unchanged light/camera. Texture must be rebaked after geometry changes; the dark-photo curve is an in-sample appearance proxy.'},
        'limits':['Topology path seeds are inspected on this frozen male template; corner shortest paths are ambiguous and are not reusable semantic annotations.',
            'Trace may switch between nonadjacent visible triangles because it follows an occlusion envelope.',
            'Photo darkest-pixel trace is automatic, expression-dependent and not manual ground truth.',
            'Existing collision diagnostics exclude faces sharing vertex indices. No geometry is changed by this script.']}
    # Fixed-camera scientific overlays, each using identical crop coordinates.
    box=[210,330,317,377];buffers=raster(geometry,camera,box,samples=6);light=np.array([-.25,-.4,-1.]);light/=np.linalg.norm(light)
    shade=.3+.65*np.maximum(buffers['normalMap']@light,0);clay=np.rint(np.clip(shade[...,None]*[208,218,222],0,255)).astype(np.uint8)
    size=(clay.shape[1],clay.shape[0]);photo=Image.fromarray(work).crop(tuple(box)).resize(size,Image.Resampling.LANCZOS)
    transform=lambda point:tuple((np.asarray(point)-box[:2])*6+2.5)
    def overlay(base,showPaths=False,showBindings=False):
        image=base.copy();draw=ImageDraw.Draw(image);draw.line([transform(p) for p in traceXY],fill=(30,230,245),width=3)
        if showPaths:
            for name,color in [('upperInner',(245,150,45)),('lowerInner',(110,235,100))]:
                draw.line([transform(p) for p in pixels[paths[name]['vertexIds']]],fill=color,width=2)
        if showBindings:
            for r in bindingReport:
                x,y=transform(r['projectedPixels']);draw.ellipse((x-4,y-4,x+4,y+4),fill=(255,225,35));draw.text((x+5,y-13),str(r['ibugId']),fill=(255,225,35))
        return image
    first=overlay(photo);draw=ImageDraw.Draw(first);draw.line([transform(p) for p in photoTrace],fill=(250,70,140),width=3)
    second=overlay(Image.fromarray(clay),showPaths=True,showBindings=True)
    third=Image.fromarray(clay);draw=ImageDraw.Draw(third)
    for hit in hits:
        x,y=transform(hit['projectedPoint']);draw.ellipse((x-3,y-3,x+3,y+3),fill=(250,75,70))
    width,height=size;figure=Image.new('RGB',(width*3,height+66),(28,35,39));draw=ImageDraw.Draw(figure)
    for i,(title,sub,image) in enumerate([
        ('Photo + geometry crease','cyan depth peak / magenta darkest photo proxy',first),
        ('Actual fold vs fixed iBUG bindings','orange upper / green lower / yellow approximate bindings',second),
        ('Existing intersections','red points: left 8 pairs / right 47 pairs',third)]):
        figure.paste(image,(i*width,66));draw.text((i*width+10,12),title,fill='white');draw.text((i*width+10,35),sub,fill=(178,191,196))
    figure.save(out/'visibleCreaseAndBindings.png')
    # Center template cross-section clarifies the nonmatching depth rings.
    profile=Image.new('RGB',(1000,520),(28,35,39));draw=ImageDraw.Draw(profile)
    draw.text((22,14),'Frozen template center profile | x = anterior depth Z (cm), y = template Y (cm)',fill='white')
    draw.text((22,37),'Orange upper wall / green lower wall. Yellow dots: existing iBUG 62 and 66 bindings.',fill=(178,191,196))
    toProfile=lambda p:(80+(p[2]-7.9)/1.6*830,470-(p[1]-161.25)/1.45*360)
    for ids,color in [([1920,1917,995,994,2633],(245,150,45)),([1188,1185,1968,2680,2623],(110,235,100))]:
        draw.line([toProfile(rest[i]) for i in ids],fill=color,width=3)
        for i in ids:
            x,y=toProfile(rest[i]);draw.ellipse((x-4,y-4,x+4,y+4),fill=color);draw.text((x+7,y-8),str(i),fill=color)
    for id in [62,66]:
        idx=np.where(state['ibugIds']==id)[0][0];x,y=toProfile(bindingRest[idx]);draw.ellipse((x-6,y-6,x+6,y+6),outline=(255,225,35),width=2);draw.text((x-62,y+9),'iBUG '+str(id),fill=(255,225,35))
    profile.save(out/'templateLipDepthRows.png')
    report['frozenSourcesUnchanged']=all(_digest(ROOT/item['path'])==item['sha256'] for item in baseline['fixedReferenceArtifacts'].values())
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'validTraceSamples':report['trace']['validSamples'],'distinctFaces':report['trace']['distinctVisibleFaces'],
        'centralOffsetMedian':report['photoFeatureProxy']['geometryBelowPhotoMedianPixels'],'sourcesUnchanged':report['frozenSourcesUnchanged']}))


if __name__=='__main__':main()
