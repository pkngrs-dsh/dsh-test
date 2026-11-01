"""Run one shared adaptive chin policy on frozen reconstructions and cameras."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from experimentFrontLandmarks import renderProjected


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    writeJson(path, serial(value))


def draw_curve(image, points, color):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2: return
    draw = ImageDraw.Draw(image)
    valid = np.isfinite(points).all(1)
    for a,b,keep in zip(points[:-1],points[1:],valid[:-1]&valid[1:]):
        if keep: draw.line([tuple(a),tuple(b)], fill=color, width=2)
    for x,y in points[valid]: draw.ellipse((x-2,y-2,x+2,y+2),fill=color)


def render_case(case, directory, result):
    cameras = json.loads(Path(case['cameraPath']).read_text())
    meshes = archive(directory/'geometry.npz')
    observations = Path(case['observationPath']).parent
    inputs = json.loads(Path(case['inputsPath']).read_text())
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    contours = result.get('contours', [])
    if isinstance(contours, dict): contours = list(contours.values())
    by_view = {int(item.get('viewIndex',i)): item for i,item in enumerate(contours)}
    rendered = []
    for i,(camera,record) in enumerate(zip(cameras,inputs)):
        camera={key:np.asarray(value) if key in ('rotation','translation','intrinsics') else value
                for key,value in camera.items()}
        photo = Image.open(observations/f'view{i}-photo.png').convert('RGB')
        width,height = photo.size
        photo.save(directory/f'view{i}-photo.png')
        pictures = [photo]
        contour = by_view.get(i,{})
        overlay = photo.copy()
        for key,color in [('targetPixels',(58,231,233)),('beforePixels',(251,167,61)),('afterPixels',(90,235,126))]:
            if key in contour: draw_curve(overlay,contour[key],color)
        overlay.save(directory/f'view{i}-contours.png')
        for branch in ('before','after'):
            image,mask = renderProjected(meshes[branch],meshes['faces'],camera,(width,height))
            image.save(directory/f'view{i}-{branch}.png')
            mixed = np.asarray(photo).copy()
            mixed[mask] = (.45*mixed[mask]+.55*np.asarray(image)[mask]).astype(np.uint8)
            Image.fromarray(mixed).save(directory/f'view{i}-{branch}-overlay.png')
            pictures.append(image)
        board = Image.new('RGB',(width*3,height+58),(22,29,34))
        draw = ImageDraw.Draw(board)
        draw.text((10,7),f"{case['caseId']} | view {i+1} | shared saved camera",font=font,fill='white')
        for col,(picture,title) in enumerate(zip(pictures,('Original photo','Before','Adaptive / unchanged fallback'))):
            draw.text((col*width+10,32),title,font=font,fill='white')
            board.paste(picture,(col*width,58))
        board.save(directory/f'view{i}-comparison.png')
        rendered.append({'viewIndex':i,'photo':f'view{i}-photo.png','before':f'view{i}-before.png',
            'after':f'view{i}-after.png','contours':f'view{i}-contours.png'})
    return rendered


def solve_case(case, directory, code_hash):
    from faceAppearance.adaptiveChinRefinement import refineAdaptiveChin
    if directory.exists() and any(directory.iterdir()): raise FileExistsError(str(directory))
    directory.mkdir(parents=True)
    for path_key,hash_key in [('geometryPath','geometrySha256'),('facesPath','facesSha256'),
            ('cameraPath','cameraSha256'),('inputsPath','inputsSha256'),('templateArrayPath','templateSha256')]:
        if case.get(hash_key) and sha(case[path_key]) != case[hash_key]:
            raise ValueError(f"Source changed: {case['caseId']} {path_key}")
    for image in case['sourceImages']:
        if sha(image['path']) != image['sha256']: raise ValueError('Source photograph changed')
    geometry = archive(case['geometryPath'])
    vertices = geometry[case.get('verticesKey','vertices')]
    faces = archive(case.get('facesPath',case['geometryPath']))[case.get('facesKey','faces')]
    assets = AssemblyAssets(Path(case['templatePath']))
    binding = archive(case['protectedBindingsPath'])
    fixed = np.flatnonzero(assets.arrays[case.get('fixedMaskKey','staticMask')])
    protected = np.unique(binding['vertexIndices'])
    observations = archive(case['observationPath'])
    inputs = json.loads(Path(case['inputsPath']).read_text())
    observations['workingImageShape'] = np.array(inputs[0]['workingSize'][::-1])
    cameras = json.loads(Path(case['cameraPath']).read_text())
    start = perf_counter()
    result = refineAdaptiveChin(vertices,faces,cameras,observations,fixedVertexIds=fixed,
        protectedVertexIds=protected,landmarkBinding=binding)
    elapsed = perf_counter()-start
    if not np.array_equal(vertices[np.union1d(fixed,protected)],result['vertices'][np.union1d(fixed,protected)]):
        raise ValueError('Protected source vertices changed')
    if sha(ROOT/'src/faceAppearance/adaptiveChinRefinement.py') != code_hash:
        raise RuntimeError('Solver changed during batch; use a new output after freezing code')
    np.savez_compressed(directory/'geometry.npz',before=vertices,after=result['vertices'],faces=faces,
        fixedVertexIds=fixed,protectedVertexIds=protected)
    report = {'caseId':case['caseId'],'seconds':elapsed,'solverSha256':code_hash,
        'diagnostics':result['diagnostics'],'contours':result.get('contours',[]),
        'captureMode':case.get('captureMode'),
        'role':case.get('role','exploratoryValidation'),
        'sourceHashes':{key:sha(case[key]) for key in ('geometryPath','facesPath','cameraPath','inputsPath',
            'observationPath','protectedBindingsPath','templateArrayPath','templateManifestPath') if key in case},
        'sourceGeometryPath':case['geometryPath'],'remoteModified':False,
        'limitations':['Automatic photo contours are observations, not manual anatomical ground truth.',
            'All input views may participate in candidate selection; they are not held-out views.',
            'Only geometry is evaluated in this batch; textures and accessories are not rebaked.']}
    write(directory/'report.json',report)
    reference = result['diagnostics'].get('referenceViewIndex')
    reference = 0 if reference is None else int(reference)
    camera = cameras[reference]
    rotation = np.asarray(camera['rotation'])
    base = vertices@rotation.T
    center = (base.min(0)+base.max(0))/2
    scale = 20./np.ptp(base,axis=0).max()
    branches = {key:((value@rotation.T-center)*[1,-1,-1]*scale).tolist()
        for key,value in [('before',vertices),('after',result['vertices'])]}
    write(directory/'mesh.json',{'faces':faces,'branches':branches,'referenceViewIndex':reference,
        'frame':'shared original reference-camera frame, normalized display units'})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientChinAdaptive20261012/evaluationV1')
    parser.add_argument('--stage',choices=('solve','render','all'),default='all')
    parser.add_argument('--case',action='append')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    selected=[case for case in manifest['cases'] if not args.case or case['caseId'] in args.case]
    code_hash=sha(ROOT/'src/faceAppearance/adaptiveChinRefinement.py')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ('solve','all'):
        implementation=args.output/'implementation'
        implementation.mkdir(exist_ok=True)
        for relative in ('src/faceAppearance/adaptiveChinRefinement.py',
                'src/faceAppearance/clientChinRefinement.py','src/faceAppearance/clientGeometrySafety.py',
                'src/faceAppearance/frontLandmarkRefinement.py','tools/runAdaptiveChinBatch.py'):
            target=implementation/Path(relative).name
            if target.exists() and sha(target)!=sha(ROOT/relative):
                raise ValueError('Output contains a different implementation snapshot')
            shutil.copyfile(ROOT/relative,target)
    rows=[]
    for case in selected:
        directory=args.output/case['caseId']
        report=(solve_case(case,directory,code_hash) if args.stage in ('solve','all')
                else json.loads((directory/'report.json').read_text()))
        if args.stage=='render':
            if rows and code_hash!=report['solverSha256']: raise ValueError('Mixed solver versions')
            code_hash=report['solverSha256']
        if args.stage in ('render','all'):
            render_case(case,directory,report)
        row={'caseId':case['caseId'],'label':case.get('label',case['caseId']),
            'captureMode':case.get('captureMode'),'role':report['role'],
            'diagnostics':report['diagnostics'],'seconds':report['seconds'],
            'solverSha256':report['solverSha256'],
            'viewCount':len(json.loads(Path(case['cameraPath']).read_text())),
            'reportUrl':f"/data/{case['caseId']}/report.json"}
        rows.append(row)
        write(args.output/'summary.json',{'status':'running','solverSha256':code_hash,'cases':rows})
        print(json.dumps(serial({'caseId':case['caseId'],'seconds':report['seconds'],
            'status':report['diagnostics'].get('status'),'referenceViewIndex':report['diagnostics'].get('referenceViewIndex')})),flush=True)
    write(args.output/'summary.json',{'status':'complete','solverSha256':code_hash,
        'manifestSha256':sha(args.manifest),'remoteModified':False,'cases':rows,
        'validationType':'One development dataset, four additional head-turn exploratory datasets, one orbital capture control; no independent ground-truth labels.'})


if __name__=='__main__': main()
