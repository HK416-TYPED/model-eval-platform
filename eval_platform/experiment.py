import importlib.metadata
import platform
import subprocess
import struct
import sys
from pathlib import Path
from .core import BACKENDS, digest, file_sha, generation_request, read_json, safe_id, source_fingerprint, validate_profile, write_json
from .suites import verify_suite

def safetensors_header(path):
    import json
    with open(path,'rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Incomplete safetensors header')
        n=struct.unpack('<Q',raw)[0]
        if n>100_000_000:raise ValueError('Invalid safetensors header length')
        header=json.loads(f.read(n))
    expected=8+n+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    if Path(path).stat().st_size!=expected:raise ValueError('Weight file is incomplete or has extra bytes')
    return header

def model_identity(model):
    backend=BACKENDS[model['backend']]
    if not backend['available']:raise ValueError(backend['reason'])
    if backend['family']!=model['family']:raise ValueError('Model family mismatch')
    files={};training_metadata={}
    for key in ['checkpoint','text_encoder','vae']:
        path=Path(model[key])
        if path.is_file():
            if path.suffix=='.safetensors':
                header=safetensors_header(path)
                if key=='checkpoint':
                    training_metadata={k:header.get('__metadata__',{}).get(k) for k in ['ss_epoch','ss_steps','ss_output_name','modelspec.title','modelspec.resolution']}
                if key=='checkpoint' and model['backend']=='anima_v7':
                    meta=header.get('__metadata__',{})
                    if meta.get('anima_native_reference_routing_mode')!='native_context_v1' or len(header)-int('__metadata__' in header)!=1569:
                        raise ValueError('Checkpoint is not Anima native context V7 (no fallback to V2)')
                    from .backends import load_package_runtime
                    load_package_runtime(model['runtime_dir']).validate_v7_checkpoint(path)
            files[key]={'sha256':file_sha(path),'bytes':path.stat().st_size}
        elif path.is_dir():
            entries={p.relative_to(path).as_posix():file_sha(p) for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts}
            if not entries:raise ValueError('Empty asset directory: '+str(path))
            files[key]={'sha256':digest(entries),'files':entries}
        else:raise FileNotFoundError(str(path))
    source=source_fingerprint(model['runtime_dir'])
    provenance_path=Path(model['checkpoint']).with_suffix('.provenance.json')
    provenance=read_json(provenance_path) if provenance_path.is_file() else None
    adapter=source_fingerprint(Path(__file__).parent,('.py',))
    python=model.get('python',sys.executable)
    probe="import sys,platform,json,importlib.metadata as m; names=['torch','transformers','diffusers','safetensors','Pillow']; installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{n:installed.get(n.lower()) for n in names}}))"
    environment=__import__('json').loads(subprocess.run([python,'-c',probe],check=True,capture_output=True,text=True,timeout=30).stdout)
    if model['backend']=='krea2_official':
        if tuple(map(int,environment['python'].split()[0].split('.')[:2]))<(3,12):raise ValueError('Official Krea2 requires Python >=3.12; configure model.python')
        version=environment['packages']['torch'] or '0.0'
        if tuple(map(int,version.split('+')[0].split('.')[:2]))<(2,9):raise ValueError('Official Krea2 requires torch >=2.9 in its own environment')
        if model.get('runtime_options'):raise ValueError('Unsupported Krea2 runtime options')
    identity={'backend':model['backend'],'assets':files,'source':source,'adapter':adapter,
              'environment':environment,'runtime_options':model.get('runtime_options',{}),'checkpoint_provenance':provenance,
              'training_metadata':training_metadata}
    identity['digest']=digest(identity)
    return identity

def preflight(config, verify_assets=True):
    problems=[];models=[];unavailable=[]
    try:suite=verify_suite(config['suite_dir'])
    except Exception as e:return {'ready':False,'problems':[str(e)],'models':[],'unavailable':[],'planned':0}
    ids=set();planned=0
    for model in config['models']:
        try:
            safe_id(model['id'])
            if model['id'] in ids:raise ValueError('Duplicate model ID')
            ids.add(model['id'])
            cap=BACKENDS[model['backend']]
            if not cap['available']:
                unavailable.append({'model':model['id'],'tasks':model.get('tasks',cap['tasks']),'reason':cap['reason']});continue
            tasks=model.get('tasks',cap['tasks'])
            if not tasks or not set(tasks)<=set(cap['tasks']):raise ValueError('Unsupported tasks')
            for task in tasks:
                validate_profile(config['profiles'][model['family']][task],model['backend'],task)
                if sum(c['task']==task for c in suite['cases'])!=suite['count_per_task']:raise ValueError(task+' has an inconsistent frozen case count')
                for case in suite['cases']:
                    if case['task']==task:generation_request(case,suite['seeds'][0],config['profiles'][model['family']][task],config['suite_dir'])
            if verify_assets:
                ident=model_identity(model)
            else:ident=None
            models.append({'id':model['id'],'identity':ident,'tasks':tasks})
            planned+=sum(c['task'] in tasks for c in suite['cases'])*len(suite['seeds'])
        except Exception as e:problems.append(model.get('id','?')+': '+str(e))
    if verify_assets and not problems:
        from .comparison import validate_comparison_models
        try:validate_comparison_models([{**m,'identity':next(v['identity'] for v in models if v['id']==m['id'])} for m in config['models'] if any(v['id']==m['id'] for v in models)])
        except ValueError as e:problems.append(str(e))
    active_families={m['family'] for m in config['models'] if any(v['id']==m['id'] for v in models)}
    if len(active_families)>1:problems.append('Create separate experiments for Anima and Krea2; epoch comparisons stay within one family')
    return {'ready':bool(models) and not problems,'problems':problems,'models':models,'unavailable':unavailable,'planned':planned,
            'suite_digest':suite['digest']}

def create_run(store, run_id, config):
    safe_id(run_id);checked=preflight(config)
    if not checked['ready']:raise ValueError('Preflight failed: '+'; '.join(checked['problems']))
    suite=verify_suite(config['suite_dir']);jobs=[];models=[]
    active={m['id']:m for m in checked['models']}
    # All active checkpoints of a family share precisely the same profile objects.
    for m in config['models']:
        if m['id'] not in active:continue
        model={**m,'identity':active[m['id']]['identity']};models.append(model)
        for case in suite['cases']:
            if case['task'] not in active[m['id']]['tasks']:continue
            profile=config['profiles'][m['family']][case['task']]
            for seed in suite['seeds']:
                key=digest([model['identity']['digest'],case['digest'],seed,profile])
                jobs.append({'id':digest([run_id,m['id'],key]),'model_id':m['id'],'case_id':case['case_id'],
                             'task':case['task'],'seed':seed,'request':generation_request(case,seed,profile,config['suite_dir'])})
    spec={'suite_dir':str(Path(config['suite_dir']).resolve()),'suite_digest':suite['digest'],
          'models':models,'profiles':config['profiles'],'unavailable':checked['unavailable']}
    path=store.root/'runs'/run_id
    if path.exists():raise FileExistsError('Run directory exists')
    path.mkdir(parents=True)
    write_json(path/'experiment.lock.json',spec)
    write_json(path/'models.lock.json',models)
    write_json(path/'environment.json',{m['id']:m['identity']['environment'] for m in models})
    store.create(run_id,spec,jobs)
    return store.summary(run_id)
