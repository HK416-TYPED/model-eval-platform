"""Compare epochs within a family; append weights without rerunning completed epochs."""
import copy
import time
from pathlib import Path
from .core import BACKENDS,canonical,digest,generation_request,safe_id,validate_profile,write_json
from .experiment import model_identity
from .suites import verify_suite

def comparison_contract(model):
    identity=model['identity']
    # Report/web changes do not change generation semantics. Keep the complete
    # source manifest in provenance while comparing the actual generation code.
    code=identity['adapter']['files']
    numeric_code={k:v for k,v in code.items() if k in ('core.py','backends.py','worker.py','checkpoint.py')}
    return {'family':model['family'],'backend':model['backend'],
            'text_encoder':identity['assets']['text_encoder'],'vae':identity['assets']['vae'],
            'source':identity['source']['digest'],'generation_code':numeric_code,
            'environment':identity['environment'],'runtime_options':identity['runtime_options']}

def validate_comparison_models(models):
    families={}
    for m in models:
        key=digest(comparison_contract(m))
        if m['family'] in families and key!=families[m['family']]:
            raise ValueError('Epoch comparison requires identical encoders, VAE, runtime and precision: '+m['family'])
        families[m['family']]=key

def append_checkpoint(store,run_id,model):
    run=store.run(run_id)
    if any(j['state']=='running' for j in store.jobs(run_id)):raise ValueError('Wait for or cancel the active generation before appending an epoch')
    spec=copy.deepcopy(run['spec']);safe_id(model['id'])
    if any(m['id']==model['id'] for m in spec['models']):raise ValueError('Checkpoint ID already exists')
    if model['family'] not in {m['family'] for m in spec['models']}:raise ValueError('Create a separate experiment for another model family')
    cap=BACKENDS[model['backend']]
    tasks=model.get('tasks',cap['tasks'])
    if not cap['available'] or not tasks or not set(tasks)<=set(cap['tasks']):raise ValueError('Unsupported backend/task')
    suite=verify_suite(spec['suite_dir'])
    if suite['digest']!=spec['suite_digest']:raise ValueError('Suite changed')
    model={**model,'identity':model_identity(model)}
    validate_comparison_models(spec['models']+[model])
    jobs=[]
    for case in suite['cases']:
        if case['task'] not in tasks:continue
        profile=spec['profiles'][model['family']][case['task']]
        validate_profile(profile,model['backend'],case['task'])
        for seed in suite['seeds']:
            key=digest([model['identity']['digest'],case['digest'],seed,profile])
            jobs.append((digest([run_id,model['id'],key]),run_id,model['id'],case['case_id'],case['task'],seed,'queued',
                         canonical(generation_request(case,seed,profile,spec['suite_dir'])),time.time()))
    if len(jobs)!=suite['count_per_task']*len(suite['seeds'])*len(tasks):raise ValueError('Requested tasks must match the frozen suite')
    spec['models'].append(model)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current=db.execute('SELECT spec FROM runs WHERE id=?',(run_id,)).fetchone()[0]
        if current!=canonical(run['spec']):raise ValueError('Experiment changed concurrently')
        db.execute('UPDATE runs SET spec=?,cancel=0 WHERE id=?',(canonical(spec),run_id))
        db.executemany('INSERT INTO jobs(id,run_id,model_id,case_id,task,seed,state,request,updated) VALUES(?,?,?,?,?,?,?,?,?)',jobs)
    path=store.root/'runs'/run_id
    write_json(path/'experiment.lock.json',spec);write_json(path/'models.lock.json',spec['models'])
    write_json(path/'environment.json',{m['id']:m['identity']['environment'] for m in spec['models']})
    from .report import build_report
    build_report(store,run_id);return store.summary(run_id)
