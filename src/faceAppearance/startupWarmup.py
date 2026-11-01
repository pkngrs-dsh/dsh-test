"""Run a real local reconstruction before a service becomes ready.

This module never creates a queue task, uploads an asset or calls a callback.
The same loaded pipeline and selected client assembly execute real configured
four-view photographs; all generated files are deleted after the warmup. This
warms only the configured service path, not unrelated experimental algorithms.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile


logger = logging.getLogger(__name__)


def _synchronize(config):
    if config.device.startswith('cuda'):
        import torch
        torch.cuda.synchronize(config.device)


def _archive(path, root):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Warmup client export did not create a regular local archive')
    with zipfile.ZipFile(path) as bundle:
        if not bundle.namelist() or bundle.testzip() is not None:
            raise RuntimeError('Warmup client ZIP failed integrity validation')
        count=len(bundle.namelist())
    return {'sizeBytes':path.stat().st_size,'entryCount':count}


def runStartupWarmup(pipeline, config, *, legacyAssets=None):
    """Synchronously warm the configured path; an exception must block startup.

    Call after loadModels and before queue.start. ``legacyAssets`` is the
    already loaded gender->AssemblyAssets mapping from the notebook service.
    No model loading time is included in this report. Startup may take longer
    than a request, and this single shape does not warm all batch sizes.
    """
    if not config.warmupEnabled:
        return {'status':'disabled','enabled':False,'seconds':0.,'modelsLoadedOnly':True,
                'scope':'No inference warmup was requested by this configuration'}
    if not getattr(pipeline,'loaded',False):
        raise RuntimeError('Load pipeline models before startup warmup')
    paths=[]
    for item in config.warmupImages:
        path=Path(item)
        if not path.is_absolute():
            path=config.projectRoot/path
        path=path.resolve()
        if not path.is_file():
            raise ValueError('A configured warmup image is missing or is not a local file')
        paths.append(path)
    if len(paths)!=4 or len(set(paths))!=4 or not config.minViews<=4<=config.maxViews:
        raise ValueError('Startup warmup requires exactly four distinct supported local views')
    if len({hashlib.sha256(path.read_bytes()).digest() for path in paths})!=4:
        raise ValueError('Startup warmup images must contain distinct photographs')
    if legacyAssets is not None and config.warmupGender not in legacyAssets:
        raise ValueError('Warmup gender asset was not preloaded')
    _synchronize(config)
    start=perf_counter()
    stage='fusion' if legacyAssets is not None or not config.enableAssembly else 'assembly'
    logger.info('Starting real four-view inference warmup (%s)',stage)
    with TemporaryDirectory(prefix='faceappearance-startup-warmup-') as temporary:
        root=Path(temporary)
        output=root/'reconstruction'
        report=pipeline.run(paths,outputDir=output,stopAfter=stage)
        if not isinstance(report,dict) or report.get('status')!='succeeded':
            raise RuntimeError('Warmup reconstruction did not succeed')
        stages=dict(report.get('stageSeconds',{}))
        archive=None
        clientMode='disabled'
        if legacyAssets is not None:
            import numpy as np
            from .legacyAsset import exportLegacyAsset
            def load(name):
                with np.load(output/name,allow_pickle=False) as saved:
                    return {key:saved[key].copy() for key in saved.files}
            geometry,texture=load('geometry.npz'),load('texture.npz')
            gender=config.warmupGender
            tick=perf_counter()
            if gender=='male' and config.legacyMaleAssemblyMode=='directColmap':
                from .directClientService import assembleDirectClient
                assembled,clientReport=assembleDirectClient(legacyAssets[gender],geometry,texture,output,paths,config)
                stages.update(clientReport.get('stageSeconds',{}))
                clientMode='legacyMaleDirectColmap'
            else:
                encoding='linear' if 'linearAlbedo' in texture else config.assemblyColorEncoding
                assembled=legacyAssets[gender].assemble(geometry,texture,colorEncoding=encoding)
                clientMode='legacyUvTransfer'
            stages['warmupClientAssemblyWall']=perf_counter()-tick
            tick=perf_counter()
            exported=exportLegacyAsset(root/'asset',assembled,gender=gender)
            archive=_archive(exported['archivePath'],root)
            stages['warmupZipExportAndValidation']=perf_counter()-tick
        elif stage=='assembly':
            from .clientContract import validateClientOutput
            validateClientOutput(output)
            archive=_archive(output/'assembly/clientAsset.zip',root)
            clientMode='clientV1Assembly'
        _synchronize(config)
        seconds=perf_counter()-start
    result={'status':'completed','enabled':True,'seconds':seconds,'stageSeconds':stages,
            'modelsLoadedOnly':False,'modelLoadIncluded':False,'views':4,
            'workingResolution':config.workingResolution,'clientMode':clientMode,
            'gender':config.warmupGender if legacyAssets is not None else None,
            'archive':archive,'temporaryOutputsRemoved':not root.exists(),
            'businessTaskCreated':False,'uploaded':False,'callbackSent':False,
            'scope':'Configured service reconstruction and selected client path; excludes experimental postprocessing and other batch sizes'}
    logger.info('Real inference startup warmup completed in %.3fs (%s)',seconds,clientMode)
    return result
