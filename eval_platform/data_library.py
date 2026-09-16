"""Verified TAR downloads and reproducible, complete evaluation datasets."""
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from PIL import Image, ImageOps
from .core import digest, file_sha, inside, read_json, safe_id, write_json, atomic_bytes

FORMATS = {'character_tar': ('edit_dual', 'reference', 2),
           'webdataset': ('edit_single', 'edit', 1)}
HF_ORIGIN = 'https://huggingface.co'


def validate_import(spec):
    safe_id(spec['dataset_id'])
    if spec.get('format') not in {*FORMATS, 'manifest', 'tar'}:
        raise ValueError('请选择 TAR 或 JSONL 清单格式')
    if type(spec.get('count', 500)) is not int or not 1 <= spec.get('count', 500) <= 5000:
        raise ValueError('样本数应为 1-5000')
    if type(spec.get('selection_seed', 20260915)) is not int:
        raise ValueError('抽样种子应为整数')
    if spec.get('source') == 'hf':
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', spec.get('repo', '')):
            raise ValueError('HF 仓库应为 owner/dataset')
        if spec['format'] == 'manifest':
            raise ValueError('HF 导入请选择 TAR 格式')
    elif spec.get('source') not in {'local_tar', 'manifest'}:
        raise ValueError('不支持的数据来源')
    if (spec.get('source')=='manifest') != (spec['format']=='manifest'):
        raise ValueError('JSONL 来源与格式必须同时选择')
    if spec.get('source')=='local_tar' and not spec.get('archive_path'):
        raise ValueError('请指定 TAR 路径')
    if spec.get('source')=='manifest' or spec.get('format')=='tar':
        from .core import TASK_INPUTS
        if spec.get('task') not in TASK_INPUTS:raise ValueError('请选择任务类型')
        if len(spec.get('input_fields',[]))!=TASK_INPUTS[spec['task']]:raise ValueError('输入字段数量与任务不符')
    if spec.get('source')=='manifest':
        if not spec.get('manifest_path'):raise ValueError('请填写 JSONL 路径')
        if spec['task']!='t2i' and not spec.get('root'):raise ValueError('请填写图片根目录')


def _hf_info(repo, token, revision=None):
    suffix = '/revision/' + quote(revision, safe='') if revision else ''
    response = requests.get(HF_ORIGIN+'/api/datasets/'+repo+suffix, params={'blobs': 'true'},
                            headers={'Authorization': 'Bearer '+token} if token else {}, timeout=(15, 45))
    response.raise_for_status()
    return response.json()


def download_file(repo, revision, entry, destination, token, progress=lambda **k: None):
    """Resume complete files; never commit a partial or checksum-mismatched TAR."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry.get('size') or entry.get('lfs', {}).get('size')
    expected_sha = entry.get('lfs', {}).get('sha256')
    if destination.is_file() and (not expected_size or destination.stat().st_size == expected_size):
        actual = file_sha(destination)
        if expected_sha and actual == expected_sha:
            return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': actual}
    partial = destination.with_name(destination.name+'.partial')
    url = HF_ORIGIN+'/datasets/'+repo+'/resolve/'+quote(revision, safe='')+'/'+quote(entry['rfilename'], safe='/')
    # Requests removes Authorization on cross-host redirects to HF's signed storage URLs.
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size and offset == expected_size:
            break
        if expected_size and offset > expected_size:
            raise ValueError('下载临时文件超过索引声明大小，请更换下载目录')
        headers = {'Authorization': 'Bearer '+token} if token else {}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('下载 Range 与本地断点不一致')
                    mode = 'ab'
                else:
                    offset = 0
                    mode = 'wb'
                progress(phase='downloading', file=entry['rfilename'], downloaded_bytes=offset, total_bytes=expected_size)
                last = time.monotonic()
                with partial.open(mode) as stream:
                    for block in response.iter_content(4*1024*1024):
                        if not block:
                            continue
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic()-last > 2:
                            progress(phase='downloading', file=entry['rfilename'], downloaded_bytes=offset, total_bytes=expected_size)
                            last = time.monotonic()
            if expected_size and offset != expected_size:
                raise OSError('下载尚未达到声明大小')
            break
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
    progress(phase='verifying_archive', file=entry['rfilename'], downloaded_bytes=offset, total_bytes=expected_size)
    actual = file_sha(partial)
    if expected_sha and actual != expected_sha:
        raise ValueError('完整文件 SHA256 与 HF 索引不一致，保留临时文件供检查')
    if expected_size and partial.stat().st_size != expected_size:
        raise ValueError('完整文件大小与 HF 索引不一致')
    partial.replace(destination)
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': actual}


def fetch_archive(state, spec, token, progress):
    info = _hf_info(spec['repo'], token, spec.get('revision'))
    entries = info.get('siblings', [])
    archives = [e for e in entries if e['rfilename'].endswith(('.tar', '.tar.gz'))]
    if spec.get('filename'):
        archives = [e for e in archives if e['rfilename'] == spec['filename']]
    if not archives:
        raise ValueError('仓库中未找到指定 TAR')
    entry = min(archives, key=lambda e: (e.get('size', 2**63), e['rfilename']))
    root = Path(state)/'downloads'/spec['repo'].replace('/', '--')/info['sha']
    root.mkdir(parents=True, exist_ok=True)
    # Keep room for selected images, suite copies and future result artifacts.
    if shutil.disk_usage(root).free < entry.get('size', 0)*1.5 + 512*1024**2:
        raise ValueError('磁盘剩余空间不足以下载完整 TAR 并提取样本')
    archive = download_file(spec['repo'], info['sha'], entry, inside(root, entry['rfilename']), token, progress)
    provenance = {'repo': spec['repo'], 'revision': info['sha'], 'filename': entry['rfilename'],
                  'archive_sha256': archive['sha256'], 'archive_bytes': archive['bytes'],
                  'transport': 'https://huggingface.co (official endpoint)', 'complete_archive': True}
    manifest = None
    if spec['format'] in {'character_tar','tar'}:
        matches = [e for e in entries if e['rfilename'] == 'metadata.jsonl']
        if not matches and spec['format']=='character_tar':
            raise ValueError('角色 TAR 仓库缺少 metadata.jsonl')
        if matches:
            metadata = download_file(spec['repo'], info['sha'], matches[0], root/'metadata.jsonl', token, progress)
            manifest = metadata['path']
            provenance['metadata_sha256'] = metadata['sha256']
    write_json(root/(Path(entry['rfilename']).name+'.verified.json'), provenance)
    return Path(archive['path']), manifest, provenance


def _tar_members(tf):
    members = {}
    for member in tf:
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or '\\' in member.name:
            raise ValueError('TAR 包含不安全路径')
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError('TAR 图片和元数据必须为普通文件，不接受链接')
        if member.name in members:
            raise ValueError('TAR 中存在重复文件路径')
        members[member.name] = member
    return members


def _member_bytes(tf, members, name, max_bytes=64*1024**2):
    member = members[name]
    if member.size > max_bytes:
        raise ValueError('样本成员文件超过大小限制')
    stream = tf.extractfile(member)
    raw = stream.read()
    if len(raw) != member.size:
        raise ValueError('TAR 成员截断')
    return raw


def _candidates(tf, members, format_name, manifest):
    rejected = Counter()
    candidates = []
    if format_name == 'character_tar':
        if not manifest:
            raise ValueError('角色 TAR 需要对应 metadata.jsonl 索引')
        with open(manifest, encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        for row in rows:
            if not row.get('has_target', bool(row.get('target_file_name'))):
                rejected['reference_only'] += 1
                continue
            try:
                refs = ['train/'+row['file_name'], 'train/'+row['ref2_file_name']]
                target = 'train/'+row['target_file_name']
                prompt_member = 'train/'+row['prompt_path'] if row.get('prompt_path') else None
                if not all(name in members for name in refs+[target]+([prompt_member] if prompt_member else [])):
                    rejected['outside_selected_tar_or_incomplete'] += 1
                    continue
                candidates.append({'id': str(row['id']), 'prompt': row['prompt'], 'refs': refs,
                                   'target': target, 'prompt_member': prompt_member, 'original': row})
            except KeyError:
                rejected['missing_fields'] += 1
    else:
        for name in sorted(members):
            if not name.endswith('.json'):
                continue
            row = json.loads(_member_bytes(tf, members, name, 2*1024**2))
            if row.get('invalid_pair'):
                rejected['invalid_pair'] += 1
                continue
            try:
                key = str(row['id'])
                ref = row.get('ref1_member', key+'.ref1.webp')
                target = row.get('target_member', key+'.target.webp')
                prompt_member = row.get('prompt_member', key+'.prompt.txt')
                if not all(n in members for n in [ref, target, prompt_member]):
                    rejected['incomplete_pair'] += 1
                    continue
                candidates.append({'id': key, 'prompt': row['edit_instruction'], 'refs': [ref],
                                   'target': target, 'prompt_member': prompt_member, 'original': row})
            except KeyError:
                rejected['missing_fields'] += 1
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('样本 ID 不唯一')
    return candidates, rejected


def _store_image(root, raw, suffix):
    sha = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        dimensions = list(image.size)
        thumb = ImageOps.exif_transpose(image).convert('RGB')
        thumb.thumbnail((360, 360), Image.Resampling.LANCZOS)
        preview = io.BytesIO()
        thumb.save(preview, format='WEBP', quality=82)
    suffix = suffix.lower() if suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'} else '.img'
    relative = 'images/'+sha+suffix
    if not (root/relative).exists():
        atomic_bytes(root/relative, raw)
        atomic_bytes(root/'previews'/(sha+'.webp'), preview.getvalue())
    return {'path': relative, 'sha256': sha, 'dimensions': dimensions, 'bytes': len(raw),
            'preview': 'previews/'+sha+'.webp'}


def _generic_candidates(tf, members, spec, manifest):
    """Task and field mapping are independent of a dataset's subject matter."""
    rows=[];rejected=Counter();candidates=[]
    if manifest:
        with open(manifest,encoding='utf-8-sig') as stream:
            rows=[(json.loads(line),'') for line in stream if line.strip()]
    else:
        manifests=[n for n in members if n.endswith('.jsonl')]
        if len(manifests)>1:raise ValueError('TAR 内有多个 JSONL，请提供明确的配套清单')
        if manifests:
            name=manifests[0];parent=str(PurePosixPath(name).parent)
            rows=[(json.loads(line),parent) for line in _member_bytes(tf,members,name).decode('utf-8-sig').splitlines() if line.strip()]
        else:
            rows=[(json.loads(_member_bytes(tf,members,n,2*1024**2)),str(PurePosixPath(n).parent)) for n in sorted(members) if n.endswith('.json')]
    if not rows:raise ValueError('TAR 中未找到 JSONL / 样本 JSON，请提供配套 JSONL 字段清单')
    def resolve(value,parent):
        if not isinstance(value,str) or not value:raise ValueError('图片路径必须是非空字符串')
        path=PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or '\\' in value:raise ValueError('清单包含不安全路径')
        options={value,str(PurePosixPath(parent)/value),'train/'+value}
        found=[n for n in options if n in members]
        if len(found)!=1:raise ValueError('图片路径不存在或有歧义: '+value)
        return found[0]
    for row,parent in rows:
        try:
            if row.get('invalid_pair'):raise ValueError('invalid_pair')
            key=str(row['id']);fields=spec['input_fields']
            # Compatibility aliases describe storage, never image subject matter.
            values=[]
            for i,field in enumerate(fields):
                value=row.get(field)
                if value is None and field in ('file_name','ref2_file_name'):
                    value=row.get('ref'+str(i+1)+'_member')
                    if value is None and 'edit_instruction' in row:value=key+'.ref'+str(i+1)+'.webp'
                values.append(resolve(value,parent))
            prompt_field=spec.get('prompt_field','prompt');prompt=row.get(prompt_field)
            if prompt is None and prompt_field=='prompt':prompt=row.get('edit_instruction')
            target_field=spec.get('target_field','target_file_name');target=row.get(target_field) if target_field else None
            if target is None and target_field=='target_file_name':
                target=row.get('target_member')
                if target is None and key+'.target.webp' in members:target=key+'.target.webp'
            target=resolve(target,parent) if target else None
            prompt_member=row.get('prompt_member') or row.get('prompt_path')
            if prompt_member:prompt_member=resolve(prompt_member,parent)
            elif key+'.prompt.txt' in members:prompt_member=key+'.prompt.txt'
            checksums=[row.get(field+'_sha256') or row.get('ref'+str(i+1)+'_sha256') for i,field in enumerate(fields)]
            if target:checksums.append(row.get(target_field+'_sha256') or row.get('target_sha256'))
            candidates.append({'id':key,'prompt':prompt,'refs':values,'target':target,'prompt_member':prompt_member,'checksums':checksums,'original':row})
        except (KeyError,ValueError,TypeError) as error:rejected[str(error)[:120]]+=1
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('样本 ID 不唯一')
    return candidates,rejected


def _prune_unselected(root, records):
    keep=set()
    for row in records:
        for asset in row['inputs']+([row['target']] if row.get('target') else []):
            keep.update([asset['path'],asset['preview']])
    for folder in ('images','previews'):
        for path in (root/folder).glob('*'):
            if path.relative_to(root).as_posix() not in keep:path.unlink()


def extract_dataset(state, spec, archive, manifest=None, provenance=None, progress=lambda **k: None):
    validate_import(spec)
    root = Path(state)/'datasets'
    root.mkdir(parents=True, exist_ok=True)
    destination = root/safe_id(spec['dataset_id'])
    if destination.exists():
        raise FileExistsError('数据集 ID 已存在，请使用新的名称')
    if spec['format']=='tar':
        task=spec['task'];mode={'edit_dual':'reference','edit_single':'edit','t2i':'t2i'}[task];input_count=len(spec['input_fields'])
    else:task, mode, input_count = FORMATS[spec['format']]
    provenance = provenance or {'archive_path': str(Path(archive).resolve()), 'archive_sha256': file_sha(archive),
                                'archive_bytes': Path(archive).stat().st_size, 'complete_archive': True}
    temp = Path(tempfile.mkdtemp(prefix='.import-', dir=root))
    selected, case_records = [], []
    wanted = spec.get('count', 500)
    seed = spec.get('selection_seed', 20260915)
    try:
        progress(phase='scanning', selected=0, requested=wanted)
        with tarfile.open(archive, 'r:*') as tf:
            members = _tar_members(tf)
            candidates, excluded = _generic_candidates(tf,members,spec,manifest) if spec['format']=='tar' else _candidates(tf, members, spec['format'], manifest)
            candidates.sort(key=lambda c: (digest([seed, provenance['archive_sha256'], c['id']]), c['id']))
            seen = set()
            for candidate in candidates:
                try:
                    if not isinstance(candidate['prompt'], str) or not candidate['prompt'].strip():
                        raise ValueError('empty_prompt')
                    if candidate['prompt_member']:
                        text = _member_bytes(tf, members, candidate['prompt_member'], 2*1024**2).decode('utf-8-sig')
                        if text.rstrip('\r\n') != candidate['prompt'].rstrip('\r\n'):
                            raise ValueError('prompt_metadata_mismatch')
                    assets = []
                    for index, name in enumerate(candidate['refs']+([candidate['target']] if candidate['target'] else [])):
                        raw = _member_bytes(tf, members, name)
                        if spec['format']=='tar':
                            declared=candidate['checksums'][index]
                            if declared and hashlib.sha256(raw).hexdigest()!=declared:raise ValueError('image_sha256_mismatch')
                        if spec['format'] == 'webdataset':
                            declared = candidate['original'].get('ref1_sha256' if index == 0 else 'target_sha256')
                            if declared and hashlib.sha256(raw).hexdigest() != declared:
                                raise ValueError('image_sha256_mismatch')
                        assets.append(_store_image(temp, raw, PurePosixPath(name).suffix))
                    pair_key = digest([candidate['prompt'],[a['sha256'] for a in assets]])
                    if pair_key in seen:
                        excluded['duplicate_pair'] += 1
                        continue
                    seen.add(pair_key)
                    inputs=assets[:input_count];target=assets[-1] if candidate['target'] else None
                    row = {'id': candidate['id'], 'prompt': candidate['prompt'],
                           **{f'input_{i}': a['path'] for i, a in enumerate(inputs)},
                           **({'target_file_name':target['path']} if target else {})}
                    selected.append(row)
                    case_records.append({'id': row['id'], 'prompt': row['prompt'], 'inputs': inputs,
                                         'target': target, 'source_record': candidate['original']})
                    if len(selected) % 10 == 0:
                        progress(phase='extracting', selected=len(selected), requested=wanted, candidates=len(candidates))
                    if len(selected) == wanted:
                        break
                except (ValueError, OSError, KeyError) as error:
                    excluded[type(error).__name__+': '+str(error)[:100]] += 1
            if len(selected) != wanted:
                raise ValueError(f'选定 TAR 只有 {len(selected)}/{wanted} 个合格完整样本；请换分片或降低数量')
        _prune_unselected(temp,case_records)
        atomic_bytes(temp/'metadata.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected).encode())
        atomic_bytes(temp/'records.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in case_records).encode())
        binding = {'task': task, 'semantic_mode': mode, 'source_id': spec.get('repo') or spec['dataset_id'],
                   'revision': provenance.get('revision'), 'root': str(destination.resolve()),
                   'manifest': str((destination/'metadata.jsonl').resolve()), 'prompt_field': 'prompt',
                   'input_fields': [f'input_{i}' for i in range(input_count)], 'target_field': 'target_file_name',
                   'require_target': spec['format']!='tar', 'snapshot_note': f'{wanted} complete cases sampled from one verified TAR',
                   'archive_sha256': provenance['archive_sha256']}
        record = {'schema': 'eval-dataset/v1', 'id': spec['dataset_id'], 'status': 'ready', 'task': task,
                  'format': spec['format'], 'count': wanted, 'selection_seed': seed,
                  'selection_algorithm': 'sha256(seed,archive_sha256,sample_id)', 'source': provenance,
                  'candidate_count': len(candidates), 'excluded': dict(excluded),
                  'selected_ids': [r['id'] for r in selected], 'binding': binding,
                  'manifest_sha256': file_sha(temp/'metadata.jsonl'), 'records_sha256': file_sha(temp/'records.jsonl'),
                  'image_files': len(list((temp/'images').glob('*'))),
                  'image_bytes': sum(p.stat().st_size for p in (temp/'images').glob('*'))}
        write_json(temp/'dataset.json', record)
        temp.rename(destination)
        progress(phase='completed', selected=wanted, requested=wanted, dataset_id=spec['dataset_id'])
        return record
    finally:
        if temp.exists() and temp.resolve().is_relative_to(root.resolve()):
            shutil.rmtree(temp)


def import_dataset(state, spec, token=None, progress=lambda **k: None):
    validate_import(spec)
    if (Path(state)/'datasets'/spec['dataset_id']).exists():
        raise FileExistsError('数据集 ID 已存在')
    if spec['source']=='manifest':return import_manifest(state,spec,progress)
    if spec['source'] == 'hf':
        from .worker import GpuLock
        with GpuLock(Path(state),'download-'+digest([spec['repo'],spec.get('revision'),spec.get('filename')])[:24]):
            archive, manifest, provenance = fetch_archive(state, spec, token, progress)
    else:
        archive = Path(spec['archive_path']).resolve()
        manifest = spec.get('manifest_path')
        provenance = None
    return extract_dataset(state, spec, archive, manifest, provenance, progress)


def list_datasets(state):
    return [read_json(path) for path in sorted((Path(state)/'datasets').glob('*/dataset.json'))]


def import_manifest(state,spec,progress):
    """Copy a generic local JSONL and its images into the immutable data library."""
    from .suites import read_rows, normalize
    root=Path(state)/'datasets';root.mkdir(parents=True,exist_ok=True)
    destination=root/safe_id(spec['dataset_id'])
    if destination.exists():raise FileExistsError('数据集 ID 已存在')
    spec=dict(spec)
    manifest=Path(spec['manifest_path']);source_sha=file_sha(manifest)
    spec['root']=spec.get('root') or str(manifest.resolve().parent)
    task=spec['task'];fields=spec['input_fields'];seed=spec.get('selection_seed',20260915);wanted=spec.get('count',500)
    binding={'task':task,'semantic_mode':spec.get('semantic_mode','t2i' if task=='t2i' else 'reference'),
             'source_id':spec['dataset_id'],'root':spec['root'],'input_fields':fields,
             'prompt_field':spec.get('prompt_field','prompt'),'target_field':spec.get('target_field','target_file_name')}
    rows=list(read_rows(manifest))
    if len({str(r['id']) for r in rows})!=len(rows):raise ValueError('样本 ID 不唯一')
    rows.sort(key=lambda r:(digest([seed,source_sha,str(r['id'])]),str(r['id'])))
    temp=Path(tempfile.mkdtemp(prefix='.import-',dir=root));selected=[];records=[];excluded=Counter();seen=set()
    try:
        for row in rows:
            try:
                case=normalize(row,binding)
                # A reference target is optional for new local data; inputs remain mandatory.
                source_assets=case['inputs']+([case['dataset_target']] if case.get('dataset_target') else [])
                assets=[]
                for asset in source_assets:
                    path=inside(spec['root'],asset['path'])
                    if path.stat().st_size>64*1024**2:raise ValueError('Image exceeds 64 MiB')
                    assets.append(_store_image(temp,path.read_bytes(),path.suffix))
                key=digest([case['prompt'],[a['sha256'] for a in assets]])
                if key in seen:raise ValueError('Duplicate prompt/image combination')
                seen.add(key);inputs=assets[:len(fields)];target=assets[-1] if case.get('dataset_target') else None
                selected.append({'id':str(row['id']),'prompt':case['prompt'],**{f'input_{i}':a['path'] for i,a in enumerate(inputs)},
                                 **({'target_file_name':target['path']} if target else {})})
                records.append({'id':str(row['id']),'prompt':case['prompt'],'inputs':inputs,'target':target})
                if len(selected)%10==0:progress(phase='extracting',selected=len(selected),requested=wanted)
                if len(selected)==wanted:break
            except (ValueError,KeyError,OSError) as error:excluded[type(error).__name__+': '+str(error)[:100]]+=1
        if len(selected)!=wanted:raise ValueError(f'只有 {len(selected)}/{wanted} 个有效样本')
        _prune_unselected(temp,records)
        atomic_bytes(temp/'metadata.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode())
        atomic_bytes(temp/'records.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records).encode())
        binding.update(root=str(destination.resolve()),manifest=str((destination/'metadata.jsonl').resolve()),
                       input_fields=[f'input_{i}' for i in range(len(fields))],prompt_field='prompt',target_field='target_file_name')
        record={'schema':'eval-dataset/v1','id':spec['dataset_id'],'status':'ready','task':task,'format':'manifest',
                'count':wanted,'selection_seed':seed,'selection_algorithm':'sha256(seed,manifest_sha256,sample_id)',
                'source':{'manifest_sha256':source_sha,'manifest_path':str(manifest.resolve())},'candidate_count':len(rows),
                'excluded':dict(excluded),'selected_ids':[r['id'] for r in selected],'binding':binding,
                'manifest_sha256':file_sha(temp/'metadata.jsonl'),'records_sha256':file_sha(temp/'records.jsonl'),
                'image_files':len(list((temp/'images').glob('*'))),'image_bytes':sum(p.stat().st_size for p in (temp/'images').glob('*'))}
        write_json(temp/'dataset.json',record);temp.rename(destination);return record
    finally:
        if temp.exists() and temp.resolve().is_relative_to(root.resolve()):shutil.rmtree(temp)
