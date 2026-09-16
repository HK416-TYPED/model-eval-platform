import json
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,SecretStr,Field
from .core import BACKENDS,inside,read_json,safe_id,write_json
from .store import Store
from . import __version__

class ConfigBody(BaseModel):
    name:str
    config:dict
class RunBody(BaseModel):
    run_id:str
    config_name:str
class StartBody(BaseModel):
    retry:bool=False
    limit:int|None=None
class ModelBody(BaseModel):
    model:dict
class ImportBody(BaseModel):
    spec:dict
    token:SecretStr|None=None
class ReviewBody(BaseModel):
    model_id:str
    case_id:str
    verdict:str
    note:str=Field(default='',max_length=4000)
    reviewer:str=Field(default='',max_length=80)
class SuiteBody(BaseModel):
    name:str
    dataset_ids:list[str]
    count:int=Field(default=5,ge=1,le=5000)
    selection_seed:int=20260915
    seeds:list[int]|None=None

def create_app(state_root):
    store=Store(state_root);app=FastAPI(title='Model Evaluation Platform',version=__version__)
    configs=store.root/'configs';configs.mkdir(exist_ok=True)
    app.mount('/static', StaticFiles(directory=Path(__file__).parent/'static'), name='static')
    @app.middleware('http')
    async def local_origin(request:Request,call_next):
        # Browser-controlled Fetch Metadata survives TLS/Host rewriting by a
        # reverse proxy. Cross-site pages cannot forge this forbidden header.
        # Older clients without it still require an exact Origin match.
        origin=request.headers.get('origin')
        same_origin_fetch=request.headers.get('sec-fetch-site')=='same-origin'
        if request.method not in ('GET','HEAD','OPTIONS') and origin and not same_origin_fetch and origin!=str(request.base_url).rstrip('/'):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Cross-origin write denied'},status_code=403)
        return await call_next(request)
    @app.exception_handler(ValueError)
    async def bad_value(request,error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':str(error)},status_code=400)
    @app.exception_handler(RuntimeError)
    async def busy(request,error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':str(error)},status_code=409)
    @app.exception_handler(FileNotFoundError)
    async def missing_file(request,error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':str(error)},status_code=404)
    @app.exception_handler(FileExistsError)
    async def existing_file(request,error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':str(error)},status_code=409)
    @app.exception_handler(KeyError)
    async def missing(request,error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':str(error)},status_code=404)
    @app.get('/')
    def home():return FileResponse(Path(__file__).parent/'static/index.html')
    @app.get('/api/capabilities')
    def capabilities():return BACKENDS
    @app.get('/api/runs')
    def runs():return store.list_runs()
    @app.get('/api/datasets')
    def datasets():
        from .data_library import list_datasets
        return [{k:v for k,v in r.items() if k not in {'selected_ids','binding','excluded'}} for r in list_datasets(store.root)]
    @app.get('/api/data-jobs')
    def data_jobs():
        from .data_jobs import list_jobs
        return list_jobs(store.root)
    @app.post('/api/datasets/import')
    def data_import(body:ImportBody):
        from .data_jobs import start_import
        return start_import(store.root,body.spec,body.token.get_secret_value() if body.token else None)
    @app.post('/api/uploads/{filename}')
    async def upload(filename:str,request:Request):
        import shutil,uuid
        filename=safe_id(filename)
        if not filename.endswith(('.tar','.tar.gz','.jsonl')):raise ValueError('只能上传 TAR 或 JSONL 文件')
        folder=store.root/'uploads';folder.mkdir(exist_ok=True)
        destination=folder/(uuid.uuid4().hex+'-'+filename);partial=destination.with_suffix(destination.suffix+'.partial')
        received=0
        try:
            with partial.open('xb') as stream:
                async for chunk in request.stream():
                    received+=len(chunk)
                    if received>8*1024**3:raise ValueError('上传文件超过 8 GiB')
                    if shutil.disk_usage(folder).free<len(chunk)+512*1024**2:raise ValueError('磁盘空间不足')
                    stream.write(chunk)
            if not received:raise ValueError('上传文件为空')
            partial.replace(destination)
            return {'path':str(destination),'bytes':received}
        finally:
            partial.unlink(missing_ok=True)
    @app.get('/api/datasets/{dataset_id}/preview')
    def data_preview(dataset_id:str,offset:int=0,limit:int=10):
        from itertools import islice
        from .suites import read_rows
        from .core import aspect_dimensions
        if offset<0 or not 1<=limit<=30:raise ValueError('Invalid preview page')
        folder=store.root/'datasets'/safe_id(dataset_id);info=read_json(folder/'dataset.json')
        rows=list(islice(read_rows(folder/'records.jsonl'),offset,offset+limit))
        for row in rows:
            row.pop('source_record',None)
            asset=row.get('target') or next(iter(row['inputs']),None)
            if asset and info['task']!='t2i':row['sampling_dimensions']=aspect_dimensions(asset['dimensions'])
        return {'id':dataset_id,'count':info['count'],'offset':offset,'records':rows}
    @app.get('/data/{dataset_id}/{relative:path}')
    def dataset_asset(dataset_id:str,relative:str):
        path=inside(store.root/'datasets'/safe_id(dataset_id),relative)
        if not path.is_file():raise HTTPException(404,'Dataset asset not found')
        return FileResponse(path)
    @app.post('/api/suites')
    def create_suite(body:SuiteBody):
        from .suites import freeze_suite
        if not body.dataset_ids:raise ValueError('请选择数据集')
        bindings=[]
        for dataset_id in body.dataset_ids:
            data=read_json(store.root/'datasets'/safe_id(dataset_id)/'dataset.json')
            if body.count>data['count']:raise ValueError('样本数超过数据集容量')
            bindings.append(data['binding'])
        folder=store.root/'suites'/safe_id(body.name)
        lock=freeze_suite(folder,bindings,body.selection_seed,body.seeds,body.count)
        return {'name':body.name,'suite_dir':str(folder),'cases':len(lock['cases']),'seeds':len(lock['seeds']),
                'planned_per_checkpoint':len(lock['cases'])*len(lock['seeds']),'tasks':[b['task'] for b in bindings],
                'digest':lock['digest']}
    @app.get('/api/runs/{run_id}')
    def run(run_id:str):
        run_id=safe_id(run_id);p=store.root/'runs'/run_id/'progress.json'
        jobs=store.jobs(run_id)
        from .review import summarize
        return {'summary':store.summary(run_id),'progress':read_json(p) if p.exists() else None,
                'snapshot_only':bool(store.run(run_id)['spec'].get('snapshot_only')),
                'analysis':summarize(jobs,store.reviews(run_id)),
                'models':[{k:v for k,v in m.items() if k!='identity'} for m in store.run(run_id)['spec']['models']],
                'jobs':[{k:v for k,v in j.items() if k!='request'} for j in jobs]}
    @app.get('/api/runs/{run_id}/reviews')
    def reviews(run_id:str):return store.reviews(safe_id(run_id))
    @app.put('/api/runs/{run_id}/reviews')
    def review(run_id:str,body:ReviewBody):
        run_id=safe_id(run_id)
        saved=store.save_review(run_id,body.model_id,body.case_id,body.verdict,body.note,body.reviewer)
        # Export regenerates snapshots; saving a note never starts GPU inference.
        return saved
    @app.post('/api/runs/{run_id}/report')
    def refresh_report(run_id:str):
        from .report import build_report
        run_id=safe_id(run_id);build_report(store,run_id)
        return {'url':'/files/'+run_id+'/report/index.html'}
    @app.get('/api/configs')
    def list_configs():return {p.stem:read_json(p) for p in configs.glob('*.json')}
    @app.put('/api/configs')
    def save_config(body:ConfigBody):
        write_json(configs/(safe_id(body.name)+'.json'),body.config);return {'saved':body.name}
    @app.post('/api/configs/{name}/preflight')
    def check(name:str):
        from .experiment import preflight
        result=preflight(read_json(configs/(safe_id(name)+'.json')))
        result['models']=[{'id':m['id'],'tasks':m['tasks'],'digest':m['identity']['digest']} for m in result['models']]
        return result
    @app.post('/api/runs')
    def create(body:RunBody):
        from .experiment import create_run
        from .report import build_report
        result=create_run(store,safe_id(body.run_id),read_json(configs/(safe_id(body.config_name)+'.json')))
        build_report(store,body.run_id);return result
    @app.post('/api/runs/{run_id}/start')
    def start(run_id:str,body:StartBody):
        from .cli import spawn_worker
        if store.run(safe_id(run_id))['spec'].get('snapshot_only'):raise ValueError('这是导入的报告快照，仅供评审和导出')
        if body.limit is not None and body.limit<1:raise ValueError('limit must be positive')
        return spawn_worker(store.root,safe_id(run_id),body.limit,body.retry)
    @app.post('/api/runs/{run_id}/cancel')
    def cancel(run_id:str):
        if store.run(safe_id(run_id))['spec'].get('snapshot_only'):raise ValueError('报告快照没有运行任务')
        store.cancel(safe_id(run_id));return store.summary(run_id)
    @app.post('/api/runs/{run_id}/append')
    def append(run_id:str,body:ModelBody):
        if store.run(safe_id(run_id))['spec'].get('snapshot_only'):raise ValueError('报告快照不支持追加推理')
        from .comparison import append_checkpoint
        from .worker import GpuLock
        with GpuLock(store.root,'cuda:0'):return append_checkpoint(store,safe_id(run_id),body.model)
    @app.post('/api/runs/{run_id}/export')
    def export(run_id:str):
        from .report import export_zip
        path=export_zip(store,safe_id(run_id));return {'url':'/files/'+run_id+'/'+Path(path).name}
    @app.post('/api/runs/{run_id}/pdf')
    def pdf(run_id:str):
        from .report import export_pdf
        run_id=safe_id(run_id)
        export_pdf(store,run_id)
        return {'url':'/files/'+run_id+'/report/report.pdf'}
    @app.get('/api/runs/{run_id}/log')
    def log(run_id:str):
        path=store.root/'runs'/safe_id(run_id)/'worker.log'
        if not path.exists():return {'text':''}
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-16000));return {'text':f.read().decode(errors='replace')}
    @app.get('/files/{run_id}/{relative:path}')
    def artifact(run_id:str,relative:str):
        path=inside(store.root/'runs'/safe_id(run_id),relative)
        if not path.is_file():raise HTTPException(404,'Artifact not found')
        if path.suffix.lower()=='.pdf':
            return FileResponse(path,media_type='application/pdf',filename=run_id+'-report.pdf')
        return FileResponse(path)
    return app
