"""Freeze dataset candidates once; never sample separately per checkpoint."""
import json
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from .core import SEEDS, atomic_bytes, digest, file_sha, inside, read_json, safe_id, validate_case, write_json

def read_rows(path):
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def rank_rows(rows, selection_seed, task, source_id):
    unique = {}
    for row in rows:
        key = str(row['id'])
        if key in unique and digest(row) != digest(unique[key]): raise ValueError('Conflicting duplicate sample ID: '+key)
        unique[key] = row
    return sorted(unique.values(), key=lambda r: (digest([selection_seed,task,source_id,str(r['id'])]), str(r['id'])))

def normalize(row, binding):
    task = binding['task']
    if binding.get('exclude_invalid', True) and row.get('invalid_pair'): raise ValueError('invalid_pair flag')
    if binding.get('require_target',False) and not row.get(binding.get('target_field','target_file_name')):
        raise ValueError('Missing training target')
    refs = [{'id':f'input_{i}', 'path':row[field]} for i,field in enumerate(binding.get('input_fields',[]))]
    case = {'case_id':task+'-'+digest([binding['source_id'],str(row['id'])])[:16], 'task':task,
            'semantic_mode':binding.get('semantic_mode','t2i' if task=='t2i' else 'reference'),
            'prompt':row[binding.get('prompt_field','prompt')], 'inputs':refs,
            'source_input_id':'input_0' if binding.get('semantic_mode')=='edit' else None,
            'provenance':{'source_id':binding['source_id'],'row_id':str(row['id']),
                          'revision':binding.get('revision'),'split':binding.get('split','train'),
                          'text_policy':binding.get('text_policy','exact_dataset_prompt')}}
    target = row.get(binding.get('target_field','target_file_name'))
    if target and binding.get('include_target',True): case['dataset_target']={'path':target}
    validate_case(case)
    return case

def freeze_suite(destination, bindings, selection_seed=20260915, seeds=None, count=5):
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Suite already frozen: '+str(destination))
    seeds = SEEDS if seeds is None else seeds
    if type(count) is not int or not 1<=count<=5000: raise ValueError('Case count must be 1–5000')
    if not 1<=len(seeds)<=50 or len(set(seeds))!=len(seeds): raise ValueError('Use 1–50 unique seeds')
    if not bindings:raise ValueError('At least one dataset binding is required')
    if any(type(s) is not int or not 0<=s<=2**63-1 for s in seeds): raise ValueError('Invalid seeds')
    tasks = [b['task'] for b in bindings]
    if len(set(tasks))!=len(tasks): raise ValueError('Duplicate task binding')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.freeze-',dir=destination.parent))
    cases, sources, exclusions = [], [], []
    try:
        for binding in bindings:
            source = {k:v for k,v in binding.items() if k not in ('root','manifest')}
            source['manifest_sha256'] = file_sha(binding['manifest'])
            sources.append(source)
            selected = 0
            seen_content = set()
            for row in rank_rows(read_rows(binding['manifest']),selection_seed,binding['task'],binding['source_id']):
                try:
                    case = normalize(row,binding)
                    content_key=digest([case['prompt'],case['inputs'],case['source_input_id']])
                    if content_key in seen_content:
                        exclusions.append({'task':binding['task'],'row_id':str(row['id']),'reason':'Duplicate prompt/input combination'})
                        continue
                    assets = case['inputs'] + ([case['dataset_target']] if 'dataset_target' in case else [])
                    validated = []
                    for asset in assets:
                        path = inside(binding['root'],asset['path'])
                        with Image.open(path) as im:
                            im.load()
                            dimensions = list(im.size)
                        sha = file_sha(path)
                        relative = 'inputs/'+sha+path.suffix.lower()
                        validated.append((asset,path,sha,relative,dimensions))
                    for asset,path,sha,relative,dimensions in validated:
                        target=inside(tmp,relative);target.parent.mkdir(exist_ok=True)
                        if not target.exists(): shutil.copyfile(path,target)
                        if file_sha(target)!=sha: raise ValueError('Input changed while freezing')
                        asset.update(path=relative,sha256=sha,dimensions=dimensions)
                    case['digest'] = digest(case)
                    cases.append(case); selected += 1
                    seen_content.add(content_key)
                    if selected==count: break
                except (OSError,ValueError,KeyError) as e:
                    exclusions.append({'task':binding['task'],'row_id':str(row.get('id')),'reason':str(e)})
            if selected != count: raise ValueError(f"{binding['task']}: only {selected}/{count} valid cases; exclusions={exclusions[:10]}")
        lock = {'schema':'eval-suite/v1','selection_seed':selection_seed,'seeds':seeds,'count_per_task':count,
                'selection_algorithm':'sha256-rank-v1','sources':sources,'cases':cases,'excluded':exclusions,
                'label':'训练集固定案例对比'}
        lock['digest']=digest(lock)
        write_json(tmp/'suite.lock.json',lock)
        atomic_bytes(tmp/'cases.jsonl',''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases).encode())
        tmp.rename(destination)
        return lock
    finally:
        if tmp.exists(): shutil.rmtree(tmp)

def verify_suite(path):
    path=Path(path);lock=read_json(path/'suite.lock.json')
    if digest({k:v for k,v in lock.items() if k!='digest'})!=lock['digest']: raise ValueError('Suite lock digest mismatch')
    for c in lock['cases']:
        validate_case(c)
        if digest({k:v for k,v in c.items() if k!='digest'})!=c['digest']: raise ValueError('Case digest mismatch')
        for a in c['inputs']+([c['dataset_target']] if 'dataset_target' in c else []):
            if file_sha(inside(path,a['path']))!=a['sha256']: raise ValueError('Frozen input changed: '+a['path'])
    return lock
