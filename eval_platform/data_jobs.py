"""Detached import workers. Authentication arrives over stdin and is never persisted."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .core import read_json, write_json, safe_id
from .data_library import import_dataset, validate_import

SPEC_FIELDS={'dataset_id','source','format','repo','revision','filename','count','selection_seed',
             'archive_path','manifest_path','root','task','semantic_mode','input_fields','prompt_field','target_field'}


def start_import(state, spec, token=None):
    from .worker import GpuLock
    Path(state).mkdir(parents=True,exist_ok=True)
    with GpuLock(Path(state),'data-import-submit'):
        return _start_import(state,spec,token)


def _start_import(state, spec, token=None):
    if set(spec)-SPEC_FIELDS:raise ValueError('Unknown import fields: '+', '.join(sorted(set(spec)-SPEC_FIELDS)))
    validate_import(spec)
    state = Path(state).resolve()
    folder = state/'data-jobs'; folder.mkdir(parents=True, exist_ok=True)
    if (state/'datasets'/spec['dataset_id']).exists():
        raise FileExistsError('数据集 ID 已存在')
    for job in list_jobs(state):
        if job.get('dataset_id') == spec['dataset_id'] and job.get('status') in {'queued', 'running'}:
            raise RuntimeError('同名数据集正在导入')
    job_id = uuid.uuid4().hex
    record = {'id': job_id, 'dataset_id': spec['dataset_id'], 'status': 'queued',
              'phase': 'queued', 'created': time.time(), 'updated': time.time(), 'spec': spec}
    write_json(folder/(job_id+'.json'), record)
    options = {'start_new_session': True} if os.name != 'nt' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW}
    # Only this pipe carries the credential; no environment variable or argv value is used.
    process = subprocess.Popen([sys.executable, '-m', 'eval_platform.data_jobs', str(state), job_id],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               cwd=Path(__file__).resolve().parent.parent, **options)
    process.stdin.write((json.dumps({'token': token})+'\n').encode())
    process.stdin.close()
    return {'id': job_id, 'dataset_id': spec['dataset_id'], 'pid': process.pid}


def run_import(state, job_id, token):
    path = Path(state)/'data-jobs'/(safe_id(job_id)+'.json')
    record = read_json(path)
    def progress(**values):
        record.update(status='running', updated=time.time(), **values)
        write_json(path, record)
    try:
        progress(phase='starting',pid=os.getpid())
        result = import_dataset(state, record['spec'], token, progress)
        record.update(status='completed', phase='completed', count=result['count'])
    except BaseException as error:
        # Network exceptions can contain signed URLs; store only a type/status, never those URLs.
        import requests
        if isinstance(error, requests.RequestException):
            status = error.response.status_code if error.response is not None else None
            message = f'下载连接失败 ({type(error).__name__}, HTTP {status}); 可用相同分片重新导入，保留下载断点'
        else:
            message = str(error)
        if token:
            message = message.replace(token, '[redacted]')
        record.update(status='failed', phase='failed', error=message[:800])
    record['updated'] = time.time()
    write_json(path, record)


def list_jobs(state):
    jobs=[]
    for path in (Path(state)/'data-jobs').glob('*.json'):
        row=read_json(path)
        if row['status']=='running' and row.get('pid'):
            try:os.kill(row['pid'],0)
            except ProcessLookupError:
                row.update(status='failed',phase='interrupted',error='导入进程已退出；重新提交相同分片可从下载断点继续')
                write_json(path,row)
        jobs.append(row)
    return sorted(jobs,key=lambda j:j['created'],reverse=True)


if __name__ == '__main__':
    secret = json.loads(sys.stdin.readline() or '{}')
    run_import(sys.argv[1], sys.argv[2], secret.get('token'))
