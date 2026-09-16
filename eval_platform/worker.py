import json
import os
import time
import traceback
import subprocess
import sys
from pathlib import Path
from PIL import Image
from .core import digest, file_sha, inside, read_json, source_fingerprint, write_json
from .store import Store
from .suites import verify_suite

class Cancelled(Exception):pass

class GpuLock:
    """Kernel lock: automatically released after crashes; one worker per GPU."""
    def __init__(self,root,device):self.path=Path(root)/('gpu-'+device.replace(':','-')+'.lock')
    def __enter__(self):
        self.file=open(self.path,'a+b')
        try:
            if os.name=='nt':
                import msvcrt
                self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0)
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise RuntimeError('GPU worker already active')
        return self
    def __exit__(self,*_):self.file.close()

def valid_success(job, run_dir):
    try:
        result=job['result'];p=inside(run_dir,result['image'])
        if file_sha(p)!=result['sha256']:return False
        with Image.open(p) as im:
            im.load()
            return list(im.size)==result['dimensions']
    except (OSError,TypeError,KeyError,ValueError):return False

def recover(store,run_id,retry=False):
    path=store.root/'runs'/run_id
    for j in store.jobs(run_id):
        if j['state']=='running' or (j['state']=='succeeded' and not valid_success(j,path)) or (retry and j['state'] in ('failed','cancelled')):
            store.update_job(j['id'],'queued')
    if retry:store.clear_cancel(run_id)

def run_worker(state_root,run_id,device='cuda:0',limit=None,retry=False,model_id=None):
    from .backends import backend_for
    store=Store(state_root);run=store.run(run_id);spec=run['spec'];run_dir=store.root/'runs'/run_id
    if model_id is None:
        # Each model runs in the configured Python environment. Children acquire
        # the same kernel GPU lock; the dispatcher itself never loads a model.
        for model in spec['models']:
            if store.cancelled(run_id) and not retry:break
            args=[model.get('python',sys.executable),'-m','eval_platform.cli','--state',str(store.root),
                  'worker',run_id,'--device',device,'--model',model['id']]
            if limit is not None:args+=['--limit',str(limit)]
            if retry:args+=['--retry']
            proc=subprocess.run(args,cwd=str(Path(__file__).resolve().parent.parent))
            if proc.returncode:raise RuntimeError('Backend worker exited: '+str(proc.returncode))
            if limit is not None:break
        return store.summary(run_id)
    with GpuLock(store.root,device):
        suite=verify_suite(spec['suite_dir'])
        if suite['digest']!=spec['suite_digest']:raise ValueError('Suite changed after submission')
        recover(store,run_id,retry);finished=0
        for model in spec['models']:
            if model['id']!=model_id:continue
            pending=[j for j in store.jobs(run_id) if j['model_id']==model['id'] and j['state']=='queued']
            if not pending or store.cancelled(run_id):continue
            if limit is not None and finished>=limit:break
            backend=backend_for(model['backend']);loaded=False;load_seconds=None
            try:
                # Pin the executing code and all model components before model loading.
                from .experiment import model_identity
                if model_identity(model)['digest']!=model['identity']['digest']:raise ValueError('Model / runtime / environment changed since submission')
                import torch
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=False
                started=time.monotonic();backend.load(model,device);load_seconds=time.monotonic()-started;loaded=True
                write_json(run_dir/('load-'+model['id']+'.json'),{'seconds':load_seconds,'device':device,'gpu':torch.cuda.get_device_name(device)})
                for job in pending:
                    if limit is not None and finished>=limit:break
                    if store.cancelled(run_id):break
                    store.update_job(job['id'],'running')
                    job_dir=run_dir/'items'/job['id'];job_dir.mkdir(parents=True,exist_ok=True)
                    write_json(job_dir/'request.json',job['request'])
                    def cancel():
                        if store.cancelled(run_id):raise Cancelled('Cancelled by user')
                    def progress(step,total):
                        cancel();write_json(run_dir/'progress.json',{'job_id':job['id'],'model':model['id'],'step':step,'total':total,'updated':time.time()})
                    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
                    try:
                        for ref in job['request']['inputs']:
                            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen input bytes changed')
                        image,extra=backend.generate(job['request'],progress,cancel)
                        cancel();tmp=job_dir/'output.partial.png';image.save(tmp)
                        with Image.open(tmp) as check:check.load();dimensions=list(check.size)
                        if dimensions!=[job['request']['profile']['width'],job['request']['profile']['height']]:raise ValueError('Unexpected output dimensions')
                        processed=[]
                        for i,(input_id,im) in enumerate(extra.pop('preprocessed_images',[])):
                            p=job_dir/f'input-{i}-prepared.png';im.save(p)
                            processed.append({'id':input_id,'path':p.relative_to(run_dir).as_posix(),'sha256':file_sha(p),'dimensions':list(im.size)})
                        dest=job_dir/'output.png';os.replace(tmp,dest)
                        result={'image':dest.relative_to(run_dir).as_posix(),'sha256':file_sha(dest),'dimensions':dimensions,
                                'generation_seconds':time.monotonic()-started,'model_load_seconds':load_seconds,
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'seed':job['seed'],
                                'preprocessed_inputs':processed,'metrics_status':'not_configured',**extra}
                        write_json(job_dir/'result.json',result);store.update_job(job['id'],'succeeded',result)
                    except Cancelled as e:store.update_job(job['id'],'cancelled',error=str(e));break
                    except Exception as e:
                        (job_dir/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                        store.update_job(job['id'],'failed',error=type(e).__name__+': '+str(e))
                    finished+=1
                    from .report import build_report
                    build_report(store,run_id)
            except Exception as e:
                (run_dir/('load-'+model['id']+'-error.log')).write_text(traceback.format_exc(),encoding='utf-8')
                for job in pending:
                    if next(j for j in store.jobs(run_id) if j['id']==job['id'])['state']=='queued':
                        store.update_job(job['id'],'failed',error='MODEL_LOAD_FAILED: '+type(e).__name__+': '+str(e))
            finally:
                backend.unload()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:pass
        from .report import build_report
        build_report(store,run_id)
        return store.summary(run_id)
