"""Post-generation scoring. Python plugins and explicit HTTP endpoints share a contract."""
import base64
import importlib.util
import json
import math
import time
from pathlib import Path
from .core import canonical, digest, file_sha, inside, read_json

def validate_score(value):
    if value.get('status') not in ('scored','not_applicable','failed'):raise ValueError('Invalid score status')
    if value['status']=='scored':
        if not isinstance(value.get('metrics'),list) or not value['metrics']:raise ValueError('scored result requires metrics')
        for m in value['metrics']:
            if not isinstance(m.get('value'),(int,float)) or not math.isfinite(m['value']):raise ValueError('Nonfinite/missing score')
            if not m.get('name') or m.get('direction') not in ('higher','lower'):raise ValueError('Metric name/direction required')
    elif value.get('metrics'):raise ValueError('Unavailable/failed scores must not contain numeric metrics')
    return value

def score_run(store,run_id,config):
    spec=store.run(run_id)['spec'];suite=read_json(Path(spec['suite_dir'])/'suite.lock.json')
    cases={c['case_id']:c for c in suite['cases']};run_dir=store.root/'runs'/run_id
    identity={k:config[k] for k in ['id','version','transport','rubric','parameters']}
    if config['transport']=='python':
        path=Path(config['module_path']).resolve();identity['module_sha256']=file_sha(path)
        module_spec=importlib.util.spec_from_file_location('_eval_scorer_'+identity['module_sha256'][:12],path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        score=lambda request:module.score(request,config['parameters'])
    elif config['transport']=='http':
        import os,requests
        identity['endpoint']=config['endpoint']
        def score(request):
            headers={}
            if config.get('token_env'):headers['Authorization']='Bearer '+os.environ[config['token_env']]
            # Only explicit scoring actions send images to the configured endpoint.
            def encode(asset):
                return {'id':asset.get('id'),'sha256':asset['sha256'],'image_base64':base64.b64encode(Path(asset['path']).read_bytes()).decode()}
            payload={**request,'inputs':[encode(x) for x in request['inputs']],'output':encode(request['output'])}
            if 'target' in request:payload['target']=encode(request['target'])
            r=requests.post(config['endpoint'],json={'request':payload,'parameters':config['parameters'],'rubric':config['rubric']},headers=headers,timeout=config.get('timeout',120))
            r.raise_for_status();return r.json()
    else:raise ValueError('Unknown scoring transport')
    summary={'scored':0,'cached':0,'failed':0,'not_applicable':0}
    for j in store.jobs(run_id):
        if j['state']!='succeeded':continue
        c=cases[j['case_id']]
        request={'task':j['task'],'prompt':c['prompt'],'semantic_mode':c['semantic_mode'],'source_input_id':c.get('source_input_id'),
                 'inputs':[{**a,'path':str(inside(spec['suite_dir'],a['path']))} for a in c['inputs']],
                 'output':{'path':str(inside(run_dir,j['result']['image'])),'sha256':j['result']['sha256']},
                 'rubric':config['rubric']}
        if config.get('uses_target'):
            if 'dataset_target' not in c:
                result={'status':'not_applicable','reason':'No target in this case'}
            else:request['target']={**c['dataset_target'],'path':str(inside(spec['suite_dir'],c['dataset_target']['path']))};result=None
        else:result=None
        for a in request['inputs']+[request['output']]+([request['target']] if 'target' in request else []):
            if file_sha(a['path'])!=a['sha256']:raise ValueError('Scoring input changed')
        key=digest([j['id'],request,identity,config.get('uses_target',False)])
        with store.connect() as db:cached=db.execute('SELECT result FROM scores WHERE id=?',(key,)).fetchone()
        if cached and json.loads(cached[0])['status']!='failed':summary['cached']+=1;continue
        started=time.monotonic()
        try:result=validate_score(result or score(request))
        except Exception as e:result={'status':'failed','error':type(e).__name__+': '+str(e)}
        result.update(scorer_identity=identity,seconds=time.monotonic()-started)
        with store.connect() as db:
            db.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?)',(key,run_id,j['id'],config['id'],canonical(result),time.time()))
        summary[result['status']]+=1
    from .report import build_report
    build_report(store,run_id);return summary
