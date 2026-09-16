"""Small, framework-free contracts shared by control and GPU environments."""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

TASK_INPUTS = {'t2i': 0, 'edit_single': 1, 'edit_dual': 2}
SEEDS = [20260915, 20260916, 20260917, 20260918, 20260919]
BACKENDS = {
    'anima_v7': {'family': 'anima', 'available': True, 'tasks': list(TASK_INPUTS)},
    'krea2_official': {'family': 'krea2', 'available': True, 'tasks': ['t2i']},
    'krea2_edit': {'family': 'krea2', 'available': False, 'tasks': ['edit_single', 'edit_dual'],
                   'reason': 'BACKEND_NOT_IMPLEMENTED: 等待 Krea2 Edit 运行时'},
}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())

def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value):
        raise ValueError('ID must contain only letters, digits, _, -, . (1–96 characters)')
    return value

def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes its declared root')
    return path

def validate_case(case):
    task = case['task']
    if task not in TASK_INPUTS: raise ValueError('Unknown task: '+task)
    refs = case.get('inputs', [])
    if len(refs) != TASK_INPUTS[task]: raise ValueError(f'{task} requires {TASK_INPUTS[task]} ordered inputs')
    if not isinstance(case['prompt'], str) or not case['prompt'].strip(): raise ValueError('Empty prompt')
    if len({x['id'] for x in refs}) != len(refs): raise ValueError('Duplicate input ID')
    mode = case.get('semantic_mode', 't2i' if task == 't2i' else 'reference')
    source = case.get('source_input_id')
    if mode == 'edit':
        if task != 'edit_single' or source != refs[0]['id']: raise ValueError('Aligned edit requires one identified source')
    elif mode not in ('t2i','reference') or source is not None:
        raise ValueError('Invalid semantic mode / source role')
    if (task == 't2i') != (mode == 't2i'): raise ValueError('T2I mode mismatch')

def validate_profile(profile, backend, task):
    common = {'width','height','steps','negative_prompt','resolution_policy','target_pixels'}
    allowed = common | ({'guidance_scale','flow_shift','native_reference_scale','reference_max_area'} if backend == 'anima_v7'
                        else {'guidance','y1','y2','mu','minres','maxres'})
    unknown = set(profile)-allowed
    if unknown: raise ValueError(f'Unknown sampling parameters: {sorted(unknown)}')
    for name in ['width','height','steps']:
        v = profile[name]
        if type(v) is not int or v <= 0: raise ValueError(name+' must be a positive integer')
    if profile['width'] % 16 or profile['height'] % 16: raise ValueError('Dimensions must be multiples of 16')
    if profile.get('resolution_policy', 'target_aspect' if task != 't2i' else 'fixed') not in {'fixed','target_aspect'}:
        raise ValueError('Unknown resolution policy')
    if 'target_pixels' in profile and (type(profile['target_pixels']) is not int or not 256 <= profile['target_pixels'] <= 16_777_216):
        raise ValueError('target_pixels must be an integer between 256 and 16777216')
    for k,v in profile.items():
        if isinstance(v,(int,float)) and not math.isfinite(v): raise ValueError('Nonfinite '+k)
    if backend == 'anima_v7':
        for k in ['guidance_scale','flow_shift']:
            if k not in profile: raise ValueError('Explicit '+k+' required')
        if task != 't2i' and not {'native_reference_scale','reference_max_area'} <= set(profile):
            raise ValueError('Explicit reference scale and geometry required')
        if task == 't2i' and set(profile) & {'native_reference_scale','reference_max_area'}:
            raise ValueError('T2I does not consume reference parameters')
    elif 'guidance' not in profile: raise ValueError('Explicit official guidance required')

def aspect_dimensions(dimensions, pixels=1048576, multiple=16):
    """Closest aligned aspect ratio under a fixed pixel budget; no cropping."""
    w, h = dimensions
    if any(type(v) is not int or v <= 0 for v in (w,h,pixels,multiple)):
        raise ValueError('Invalid geometry')
    ratio = w/h
    ideal_w = math.sqrt(pixels*ratio)
    ideal_h = math.sqrt(pixels/ratio)
    if min(ideal_w,ideal_h) < multiple:
        raise ValueError('Aspect ratio is too extreme for this pixel budget')
    candidates = []
    for width in range(max(multiple, int(ideal_w/multiple-3)*multiple), int(ideal_w/multiple+4)*multiple+1, multiple):
        for height in range(max(multiple, int(ideal_h/multiple-3)*multiple), int(ideal_h/multiple+4)*multiple+1, multiple):
            if width*height <= pixels:
                score = 4*math.log((width/height)/ratio)**2 + math.log(width*height/pixels)**2
                candidates.append((score, -width*height, width, height))
    if not candidates:raise ValueError('No aligned resolution under pixel budget')
    return list(min(candidates)[2:])


def resolve_geometry(case, profile, suite_dir):
    result = dict(profile)
    policy = result.pop('resolution_policy', 'fixed' if case['task']=='t2i' else 'target_aspect')
    pixels = result.pop('target_pixels', 1048576)
    if case['task']=='t2i' or policy=='fixed':
        return result, {'policy':'fixed','dimensions':[result.get('width'),result.get('height')]}
    asset = case.get('dataset_target') or case['inputs'][0]
    dimensions = asset.get('dimensions')
    if not dimensions:
        from PIL import Image
        with Image.open(inside(suite_dir,asset['path'])) as im:dimensions=list(im.size)
    width,height=aspect_dimensions(dimensions,pixels)
    result.update(width=width,height=height)
    return result, {'policy':'target_aspect','dimension_source':'dataset_target' if case.get('dataset_target') else 'input_0',
                    'source_dimensions':dimensions,'target_pixels':pixels,'alignment':16,
                    'dimensions':[width,height],'actual_pixels':width*height}


def generation_request(case, seed, profile, suite_dir):
    """Explicit allowlist: target images can never reach a generation backend."""
    validate_case(case)
    if type(seed) is not int or not 0 <= seed <= 2**63-1: raise ValueError('Invalid seed')
    refs = [{**x, 'path':str(inside(suite_dir, x['path']))} for x in case['inputs']]
    effective, geometry = resolve_geometry(case,profile,suite_dir)
    return {'case_id':case['case_id'], 'task':case['task'], 'semantic_mode':case['semantic_mode'],
            'prompt':case['prompt'], 'inputs':refs, 'source_input_id':case.get('source_input_id'),
            'seed':seed, 'profile':effective, 'geometry':geometry}

def source_fingerprint(root, suffixes=('.py','.json','.txt','.model','.toml')):
    root = Path(root)
    entries = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in suffixes and not set(p.relative_to(root).parts) & {'.git','__pycache__','.venv','state'}:
            entries[p.relative_to(root).as_posix()] = file_sha(p)
    if not entries: raise ValueError('Empty source or asset directory: '+str(root))
    return {'digest':digest(entries), 'files':entries}
