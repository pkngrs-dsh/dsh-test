"""Local read-only viewer for the adaptive chin experiment."""
from pathlib import Path
import argparse
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT=Path(__file__).resolve().parents[1]

def createApp(directory):
    directory=Path(directory).resolve()
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
    @app.middleware('http')
    async def headers(request,call_next):
        response=await call_next(request)
        response.headers.update({'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin'})
        return response
    @app.get('/api/summary')
    def summary():
        path=directory/'summary.json'
        if not path.is_file(): raise HTTPException(503,'实验尚未完成')
        return json.loads(path.read_text())
    @app.get('/data/{case_id}/{filename}')
    def artifact(case_id,filename):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',case_id): raise HTTPException(404)
        if not (filename in ('mesh.json','report.json','geometry.npz') or re.fullmatch(r'view\d+-(photo|before|after|contours|comparison|before-overlay|after-overlay)\.png',filename)):
            raise HTTPException(404)
        path=(directory/case_id/filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file(): raise HTTPException(404)
        return FileResponse(path)
    app.mount('/viewer-common',StaticFiles(directory=ROOT/'src/faceAppearance/web'))
    app.mount('/',StaticFiles(directory=ROOT/'src/faceAppearance/adaptiveChinWeb',html=True))
    return app

if __name__=='__main__':
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'runs/clientChinAdaptive20261012/evaluationV1')
    parser.add_argument('--port',type=int,default=18104)
    args=parser.parse_args()
    uvicorn.run(createApp(args.directory),host='127.0.0.1',port=args.port,access_log=False)
