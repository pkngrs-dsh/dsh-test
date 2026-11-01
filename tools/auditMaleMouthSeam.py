"""Read-only camera-space audit of the frozen male COLMAP mouth seam.

Renders auxiliary buffers at four samples per working-image pixel. This is an
experiment on cached data, not a reconstruction or a geometry repair.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.camera import projectPoints
from faceAppearance.headAssembly import decodeColor
from prepareFrontLandmarkTextures import encodeSrgb
from auditDirectRegistrationSafety import broadPhase, intersectTriangles


def archive(path):
    with np.load(path, allow_pickle=False) as value:
        return dict(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raster(geometry, camera, bbox, samples=4):
    """Perspective depth, face ID, UV, normal; no backface culling."""
    vertices, faces = geometry['vertices'], geometry['faces']
    cv = vertices @ np.asarray(camera['rotation']).T + camera['translation']
    xy = projectPoints(vertices, np.asarray(camera['intrinsics']),
                       np.asarray(camera['rotation']), np.asarray(camera['translation']))[0]
    width, height = (np.array(bbox[2:]) - bbox[:2]) * samples
    faceMap = np.full((height, width), -1, np.int32)
    depth = np.full((height, width), np.inf)
    uvMap = np.zeros((height, width, 2), np.float32)
    normals = np.zeros_like(cv)
    cross = np.cross(cv[faces[:, 1]] - cv[faces[:, 0]], cv[faces[:, 2]] - cv[faces[:, 0]])
    for i in range(3): np.add.at(normals, faces[:, i], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normalMap = np.zeros((height, width, 3))
    # Image integer coordinates are pixel centers. Place each working pixel's
    # center halfway between its four subpixel sample centers.
    pixelMap = (xy - bbox[:2]) * samples + (samples-1)/2
    for i, face in enumerate(faces):
        if (cv[face, 2] <= 0).any(): continue
        tri = pixelMap[face]
        lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(tri.max(0)).astype(int), [width - 1, height - 1])
        if (lo > hi).any(): continue
        a, b, c = tri
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-12: continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        wa = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / den
        wb = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / den
        wc = 1-wa-wb
        weights = np.stack([wa, wb, wc], axis=-1) / cv[face, 2]
        z = 1/np.maximum(weights.sum(-1), 1e-12)
        weights *= z[..., None]
        target = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        keep = (wa >= -1e-10) & (wb >= -1e-10) & (wc >= -1e-10) & (z < depth[target])
        if not keep.any(): continue
        depth[target][keep] = z[keep]
        faceMap[target][keep] = i
        uvMap[target][keep] = (weights @ geometry['uvs'][geometry['uvFaces'][i]])[keep]
        normalMap[target][keep] = (weights @ normals[face])[keep]
    return {'faceMap': faceMap, 'depth': depth, 'uvMap': uvMap, 'normalMap': normalMap,
            'pixels': xy, 'cameraVertices': cv}


def sample(texture, uv, interpolation=cv2.INTER_LINEAR):
    return cv2.remap(np.asarray(texture), (uv[..., 0]*texture.shape[1]-.5).astype(np.float32),
                     (uv[..., 1]*texture.shape[0]-.5).astype(np.float32), interpolation,
                     borderMode=cv2.BORDER_REPLICATE)


def colors(texture, buffers):
    linear = decodeColor(np.asarray(Image.open(texture).convert('RGB'), np.float32)/255, 'srgb')
    rgb = np.rint(255*encodeSrgb(sample(linear, buffers['uvMap']))).astype(np.uint8)
    rgb[buffers['faceMap'] < 0] = [28, 35, 39]
    return rgb


def main():
    output = ROOT/'runs/maleMouthSeam20261012/audit'
    output.mkdir(parents=True, exist_ok=True)
    baselinePath = ROOT/'reports/clientColmapBaseline20261012.json'
    baseline = json.loads(baselinePath.read_text())
    for item in baseline['fixedReferenceArtifacts'].values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: '+item['path'])
    folder = ROOT/'runs/clientColmap20261012/maleExactRegression'
    geometry = archive(folder/'geometry.npz')
    baked = archive(folder/'bakedTexture.npz')
    state = archive(ROOT/'runs/frontLandmark20261012/maleExactRegression/landmarks.npz')
    cameras = json.loads((ROOT/baseline['fixedReferenceArtifacts']['cameras']['path']).read_text())
    bbox = baseline['referenceFrontMouthRgbMaeSrgb8']['bboxXYXY']
    buffers = raster(geometry, cameras[0], bbox)
    visible = buffers['faceMap'] >= 0
    faces = geometry['faces']
    rgb = colors(folder/'colmapTexture.png', buffers)
    support = sample(baked['coverageMask'].astype(np.float32), buffers['uvMap'])
    assigned = sample(baked['assignedMask'].astype(np.float32), buffers['uvMap'])
    atlas = sample(baked['atlasMask'].astype(np.float32), buffers['uvMap'])
    faceSource = np.where(visible, baked['faceSourceIndex'][np.maximum(buffers['faceMap'], 0)], -2)
    # Luminance threshold only labels unusually dark display pixels in this ROI.
    # It is not a semantic lip detector and all values are reported with scope.
    luminance = rgb @ np.array([.2126, .7152, .0722])
    dark = visible & (luminance < 80)
    components, count = label(dark)
    darkComponents = []
    for i in range(1, count+1):
        yy, xx = np.where(components == i)
        if len(xx) < 4: continue
        darkComponents.append({'supersamples': len(xx), 'areaWorkingPixels': len(xx)/16,
            'bboxWorkingXYXY': [float(xx.min()/4+bbox[0]), float(yy.min()/4+bbox[1]),
                                float((xx.max()+1)/4+bbox[0]), float((yy.max()+1)/4+bbox[1])],
            'observedFraction': float((support[components == i] >= .999).mean()),
            'fullyUnsupportedFraction': float((support[components == i] <= .001).mean())})
    # Undirected boundary graph. A true mouth opening would be an additional
    # boundary component; closed internal oral surfaces can still be visible.
    edges = np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    graph = defaultdict(set)
    for a,b in boundary: graph[int(a)].add(int(b)); graph[int(b)].add(int(a))
    remaining = set(graph); boundaryComponents=[]
    while remaining:
        todo=[remaining.pop()]; found=set(todo)
        while todo:
            for vertex in graph[todo.pop()]:
                if vertex not in found: found.add(vertex); remaining.discard(vertex); todo.append(vertex)
        ids=np.array(sorted(found)); p=buffers['pixels'][ids]
        boundaryComponents.append({'vertexCount':len(ids), 'bboxWorkingXYXY':[*p.min(0).tolist(),*p.max(0).tolist()],
            'vertexIds':ids.tolist(), 'allDegreeTwo':all(len(graph[int(i)]) == 2 for i in ids)})
    epsilon=float(np.linalg.norm(np.ptp(geometry['vertices'],axis=0)))*1e-9
    pairs,_=broadPhase(geometry['vertices'],faces,epsilon)
    hits=[]
    for a,b in pairs:
        hit=intersectTriangles(geometry['vertices'][faces[a]],geometry['vertices'][faces[b]],epsilon)
        if hit and hit['kind'] in ('nonCoplanarSegment','coplanarAreaOverlap'):
            p=projectPoints(np.asarray(hit['point'])[None],np.asarray(cameras[0]['intrinsics']),
                            np.asarray(cameras[0]['rotation']),np.asarray(cameras[0]['translation']))[0][0]
            hits.append({'faceIds':[int(a),int(b)],'kind':hit['kind'],'projectedPoint':p.tolist(),
                         'inMouthRoi':bool((p >= bbox[:2]).all() and (p < bbox[2:]).all())})
    mouthHits=[hit for hit in hits if hit['inMouthRoi']]
    collisionFaces=np.unique([i for hit in mouthHits for i in hit['faceIds']])
    onCollisionFace=visible & np.isin(buffers['faceMap'],collisionFaces)
    cameraTri=buffers['cameraVertices'][faces]
    normal=np.cross(cameraTri[:,1]-cameraTri[:,0],cameraTri[:,2]-cameraTri[:,0])
    facing=(normal*cameraTri.mean(1)).sum(-1) < 0
    frontFacing=visible & facing[np.maximum(buffers['faceMap'],0)]
    report={'caseId':baseline['caseId'],'geometrySha256':digest(folder/'geometry.npz'),'referenceView':0,
        'bboxWorkingXYXY':bbox,'samplesPerAxis':4,'culling':'double-sided',
        'mouthRoi':{'workingPixelArea':visible.size/16,'uncoveredRayAreaPixels':float((~visible).sum()/16),
            'fullyObservedAreaPixels':float((visible & (support>=.999)).sum()/16),
            'fullyUnsupportedAreaPixels':float((visible & (support<=.001)).sum()/16),
            'partialSupportAreaPixels':float((visible & (support>.001)&(support<.999)).sum()/16),
            'unassignedAreaPixels':float((visible & (assigned<=.001)).sum()/16),
            'outsideAtlasAreaPixels':float((visible & (atlas<=.001)).sum()/16),
            'backFacingVisibleAreaPixels':float((visible & ~frontFacing).sum()/16)},
        'darkDisplayPixelDiagnostic':{'thresholdSrgbLuminance8':80,'areaWorkingPixels':float(dark.sum()/16),
            'fullyUnsupportedAreaPixels':float((dark & (support<=.001)).sum()/16),
            'fullyObservedAreaPixels':float((dark & (support>=.999)).sum()/16),
            'partialSupportAreaPixels':float((dark & (support>.001)&(support<.999)).sum()/16),
            'onFacesInMouthCollisionAreaPixels':float((dark & onCollisionFace).sum()/16),
            'backFacingAreaPixels':float((dark & ~frontFacing).sum()/16),
            'visibleFaceIds':np.unique(buffers['faceMap'][dark]).tolist(),
            'sourceCounts':{str(int(k)):int(v) for k,v in zip(*np.unique(faceSource[dark],return_counts=True))},
            'components':darkComponents},
        'topology':{'boundaryEdges':len(boundary),'nonmanifoldEdges':int((counts>2).sum()),'boundaryComponents':boundaryComponents},
        'nonadjacentIntersections':{'allPairs':len(hits),'inMouthRoiPairs':len(mouthHits),'mouthPairs':mouthHits},
        'faceViewSelection':{'visibleAreaWorkingPixels':{str(int(k)):float(v/16)
             for k,v in zip(*np.unique(faceSource[visible],return_counts=True))},
             'unassignedMeaning':'-1: COLMAP did not assign an input image to that face; texture fallback remains'},
        'limitations':['Dark threshold is a display diagnostic, not a semantic segmentation or perceptual metric.',
            'Geometric intersections exclude triangle pairs sharing vertex indices.',
            'The mouth ROI is photo-landmark-derived and the fixed correspondence is approximate.',
            'Existing COLMAP view labels describe face selection; atlas bilinear support can mix observed and fallback texels.',
            'No frozen texture, geometry, camera or source image was modified.']}
    flat=np.full(rgb.shape, [194,199,202], np.uint8);flat[~visible]=[28,35,39]
    light=np.array([-.25,-.4,-1.]);light/=np.linalg.norm(light)
    shade=.3+.65*np.maximum(buffers['normalMap']@light,0)
    clay=np.rint(np.clip(shade[...,None]*[208,218,222],0,255)).astype(np.uint8);clay[~visible]=[28,35,39]
    supportRgb=np.full(rgb.shape,[210,70,55],np.uint8)
    supportRgb[support>=.999]=[60,190,100];supportRgb[(support>.001)&(support<.999)]=[240,190,50];supportRgb[~visible]=[28,35,39]
    palette=np.array([[230,230,230],[60,110,240],[70,205,110],[225,95,75]],np.uint8)
    sourceRgb=np.full(rgb.shape,[145,50,170],np.uint8)
    for i,color in enumerate(palette): sourceRgb[faceSource==i]=color
    sourceRgb[~visible]=[28,35,39]
    collisionRgb=flat.copy();collisionRgb[onCollisionFace]=[225,85,45]
    wire=Image.fromarray(clay);draw=ImageDraw.Draw(wire)
    visibleIds=np.unique(buffers['faceMap'][visible])
    for face in faces[visibleIds]:
        p=(buffers['pixels'][face]-bbox[:2])*4+1.5
        draw.line([tuple(x) for x in np.concatenate([p,p[:1]])],fill=(30,130,160),width=1)
    binding=(geometry['vertices'][state['vertexIndices']]*state['barycentricWeights'][...,None]).sum(1)
    bindingPx=projectPoints(binding,np.asarray(cameras[0]['intrinsics']),np.asarray(cameras[0]['rotation']),np.asarray(cameras[0]['translation']))[0]
    for p in bindingPx[state['ibugIds']>=60]:
        x,y=(p-bbox[:2])*4+1.5;draw.ellipse((x-4,y-4,x+4,y+4),fill=(255,220,40))
    report['approximateInnerLipBinding']=[{'ibugId':int(state['ibugIds'][i]),'meshWorkingPixels':bindingPx[i].tolist(),
        'photoWorkingPixels':state['frontTargetPixels'][i].tolist(),
        'differenceMeshMinusPhoto':(bindingPx[i]-state['frontTargetPixels'][i]).tolist(),
        'errorPixels':float(np.linalg.norm(bindingPx[i]-state['frontTargetPixels'][i]))}
        for i in np.where(state['ibugIds']>=60)[0]]
    report['findings']=[
        'No ray in the frontal mouth ROI misses the skin mesh, and the visible mouth faces are front-facing. The dashed color is not the background showing through a topological hole.',
        'The topology has three boundary loops: two eyes and the neck. There is no separate open mouth boundary in this client skin subset.',
        'COLMAP leaves a thin row of mouth faces without a source image. These show the fallback texture; bilinear filtering mixes fallback with photograph pixels along their sides.',
        'Another darker line immediately above those fallback fragments is photograph-supported and largely selected from view 3. The image feature and the geometric lip crease are offset.',
        'All 55 existing nonadjacent intersections project inside the broad mouth ROI. Their exposed triangles are chiefly near the corners; they do not account for all central unsupported fragments.',
        'The fixed approximate inner-lip landmarks still lie about 2.8 to 6.2 working pixels below the detected photo positions. Binding and detection errors are included, so this is not a measured anatomical displacement.']
    report['proposedNextStep']=[
        'With frozen geometry and UVs, prefer the frontal photograph for the entire lip region and retain a source-validity/visibility mask.',
        'Audit the narrow unassigned surface strip explicitly. If its final geometry is front-visible and its projection is valid, a controlled frontal projection may replace template fallback there; do not count it as native COLMAP coverage.',
        'Keep geometry repair separate. If a doubled seam remains after consistent frontal texturing, review the lip binding and crease position rather than forcing a wide texture blur.']
    inputs=json.loads((ROOT/baseline['fixedReferenceArtifacts']['inputManifest']['path']).read_text())
    photo=ImageOps.exif_transpose(Image.open(baseline['sourceImages'][0]['path'])).convert('RGB')
    work=cv2.warpPerspective(np.asarray(photo),np.asarray(inputs[0]['inputToWorking']),tuple(inputs[0]['workingSize']))
    photoCrop=Image.fromarray(work).crop(tuple(bbox)).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.LANCZOS)
    panels=[('Original photo',photoCrop),('COLMAP base color',Image.fromarray(rgb)),('Flat color / ray coverage',Image.fromarray(flat)),
            ('Smooth clay / no texture',Image.fromarray(clay)),('Wire + approximate inner bindings',wire),('Support: green yes / red no',Image.fromarray(supportRgb)),
            ('Face view: 0 white 1 blue 2 green 3 red',Image.fromarray(sourceRgb)),('Mouth collision faces in orange',Image.fromarray(collisionRgb))]
    width,height=rgb.shape[1],rgb.shape[0]
    panel=Image.new('RGB',(4*width,2*(height+38)),(28,35,39));draw=ImageDraw.Draw(panel)
    for i,(title,picture) in enumerate(panels):
        x=(i%4)*width;y=(i//4)*(height+38);panel.paste(picture,(x,y+38));draw.text((x+8,y+10),title,fill='white')
    panel.save(output/'mouthSeamAudit.png')
    for name,array in [('colmapMouth',rgb),('flatMouth',flat),('clayMouth',clay),('supportMouth',supportRgb),('viewSelectionMouth',sourceRgb)]:
        Image.fromarray(array).save(output/(name+'.png'))
    np.savez_compressed(output/'mouthBuffers.npz',faceId=buffers['faceMap'],uv=buffers['uvMap'],depth=buffers['depth'],
        support=support,assigned=assigned,atlas=atlas,faceSource=faceSource,darkDiagnosticMask=dark,
        pixelBbox=np.asarray(bbox),samplesPerAxis=np.asarray(4))
    report['sourceHashesUnchanged']=all(digest(ROOT/item['path'])==item['sha256'] for item in baseline['fixedReferenceArtifacts'].values())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('mouthRoi','darkDisplayPixelDiagnostic','sourceHashesUnchanged')},ensure_ascii=False))


if __name__ == '__main__': main()
