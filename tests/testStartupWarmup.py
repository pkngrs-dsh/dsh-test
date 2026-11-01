"""Startup inference contract checks use CPU fakes, never inference weights."""
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faceAppearance.config import PipelineConfig
from faceAppearance.service import createApp
from faceAppearance.startupWarmup import runStartupWarmup


def config(tmp_path,**options):
    paths=[]
    for i in range(4):
        path=tmp_path/f'photo{i}.image'
        path.write_bytes(bytes([i+1])*20)
        paths.append(path.name)
    return PipelineConfig(projectRoot=tmp_path,device='cpu',warmupEnabled=True,warmupImages=paths,**options)


class FakePipeline:
    def __init__(self,config):
        self.config=config
        self.loaded=False
        self.events=[]
        self.outputs=[]
    def loadModels(self):
        self.events.append('load')
        self.loaded=True
    def run(self,paths,outputDir,stopAfter):
        assert self.loaded
        assert len(paths)==4 and all(p.is_file() for p in paths)
        self.events.append('run:'+stopAfter)
        self.outputs.append(Path(outputDir))
        Path(outputDir).mkdir()
        np.savez(Path(outputDir)/'geometry.npz',vertices=np.ones((3,3)))
        np.savez(Path(outputDir)/'texture.npz',linearAlbedo=np.ones((2,2,3)))
        return {'status':'succeeded','stageSeconds':{'geometry':.1,'delight':.2}}


def testWarmupDisabledDoesNotRunOrMisreportInference(tmp_path):
    cfg=PipelineConfig(projectRoot=tmp_path,device='cpu')
    result=runStartupWarmup(FakePipeline(cfg),cfg)
    assert result['status']=='disabled' and result['modelsLoadedOnly']


def testWarmupRunsAfterLoadBeforeReadyAndDoesNotCreateJobs(tmp_path):
    cfg=config(tmp_path)
    pipeline=FakePipeline(cfg)
    app=createApp(cfg,pipelineFactory=lambda cfg:pipeline)
    with TestClient(app) as client:
        assert pipeline.events==['load','run:fusion']
        health=client.get('/health').json()
        assert health['status']=='ready'
        warm=health['startupWarmup']
        assert warm['status']=='completed' and warm['temporaryOutputsRemoved']
        assert warm['modelLoadIncluded'] is False and warm['businessTaskCreated'] is False
        assert warm['clientMode']=='disabled' and warm['archive'] is None
        assert app.state.jobQueue.jobs=={}
        assert all(not p.exists() for p in pipeline.outputs)


def testWarmupFailurePreventsQueueReadinessAndCleansTemporaryFiles(tmp_path):
    cfg=config(tmp_path)
    class Failing(FakePipeline):
        def run(self,*args,**kwargs):
            super().run(*args,**kwargs)
            raise RuntimeError('deliberate inference failure')
    pipeline=Failing(cfg)
    app=createApp(cfg,pipelineFactory=lambda cfg:pipeline)
    with pytest.raises(RuntimeError,match='deliberate'):
        with TestClient(app):
            pytest.fail('Failed warmup became ready')
    assert not hasattr(app.state,'jobQueue')
    assert all(not p.exists() for p in pipeline.outputs)
    # Failure released the owner lock, permitting a clean process to start.
    with TestClient(createApp(replace(cfg,warmupEnabled=False),pipelineFactory=FakePipeline)):
        pass


def testWarmupLegacyCallsDirectAssemblyAndExportsRealZipWithoutTransport(tmp_path,monkeypatch):
    cfg=config(tmp_path,legacyMaleAssemblyMode='directColmap')
    pipeline=FakePipeline(cfg);pipeline.loadModels()
    events=[]
    def direct(asset,geometry,texture,directory,paths,config):
        events.append('direct')
        assert asset=='maleAsset' and 'vertices' in geometry and 'linearAlbedo' in texture
        return {'assembled':True},{'stageSeconds':{'colmap':.4}}
    def export(directory,assembled,gender):
        events.append('zip')
        assert assembled['assembled'] and gender=='male'
        directory.mkdir()
        archive=directory/'legacyAsset.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('mesh.obj','v 0 0 0\n')
        return {'archivePath':archive}
    monkeypatch.setattr('faceAppearance.directClientService.assembleDirectClient',direct)
    monkeypatch.setattr('faceAppearance.legacyAsset.exportLegacyAsset',export)
    result=runStartupWarmup(pipeline,cfg,legacyAssets={'male':'maleAsset'})
    assert events==['direct','zip']
    assert result['archive']['entryCount']==1 and result['clientMode']=='legacyMaleDirectColmap'
    assert result['temporaryOutputsRemoved'] and not result['uploaded'] and not result['callbackSent']


@pytest.mark.parametrize('change',[{'warmupImages':[]},{'warmupImages':['same']*4},
                                   {'warmupGender':'unknown'},{'warmupEnabled':'yes'}])
def testWarmupConfigurationFailsClearly(tmp_path,change):
    cfg=config(tmp_path)
    with pytest.raises(ValueError):replace(cfg,**change)


def testWarmupRejectsDuplicatedPhotoBytesAndNonLoadedModels(tmp_path):
    cfg=config(tmp_path);pipeline=FakePipeline(cfg)
    with pytest.raises(RuntimeError,match='Load pipeline'):runStartupWarmup(pipeline,cfg)
    pipeline.loadModels()
    (tmp_path/cfg.warmupImages[1]).write_bytes((tmp_path/cfg.warmupImages[0]).read_bytes())
    with pytest.raises(ValueError,match='distinct photographs'):runStartupWarmup(pipeline,cfg)


@pytest.mark.parametrize('fail',[False,True])
def testLegacyStartupWarmsBeforeAcceptingAndFailureBlocksReady(tmp_path,monkeypatch,fail):
    from faceAppearance.legacyService import createLegacyApp
    cfg=config(tmp_path)
    pipeline=FakePipeline(cfg)
    class Assets:
        def __init__(self,path):
            gender='female' if 'Female' in str(path) else 'male'
            self.manifest={'gender':gender,'templateId':'fake-'+gender}
    monkeypatch.setattr('faceAppearance.headAssembly.AssemblyAssets',Assets)
    def warm(pipeline,config,*,legacyAssets):
        assert pipeline.loaded and set(legacyAssets)=={'male','female'}
        pipeline.events.append('warmup')
        if fail:raise RuntimeError('legacy warmup failure')
        return {'status':'completed','seconds':.1,'enabled':True}
    monkeypatch.setattr('faceAppearance.startupWarmup.runStartupWarmup',warm)
    app=createLegacyApp(cfg,publicBaseUrl='http://127.0.0.1:8000',pipelineFactory=lambda cfg:pipeline)
    if fail:
        with pytest.raises(RuntimeError,match='legacy warmup failure'):
            with TestClient(app):pytest.fail('Failed legacy warmup became ready')
        assert not hasattr(app.state,'legacyQueue')
    else:
        with TestClient(app) as client:
            assert pipeline.events==['load','warmup']
            health=client.get('/health').json()
            assert health['accepting_jobs'] and health['startupWarmup']['status']=='completed'
            assert not app.state.legacyQueue.records
