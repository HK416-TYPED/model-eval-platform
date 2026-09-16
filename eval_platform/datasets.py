"""Dataset ingestion, separate from immutable suite generation.

Hub JSONL downloads only selected image groups. WebDataset accepts local TAR
shards and reads metadata without extracting arbitrary archive paths.
"""
import json
import tarfile
from pathlib import Path
from .core import atomic_bytes,digest,inside,write_json
from .suites import normalize,rank_rows,read_rows

def import_hf_jsonl(binding,destination):
    from huggingface_hub import HfApi,hf_hub_download
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    repo=binding['source_id'];info=HfApi().dataset_info(repo,revision=binding.get('revision') or 'main')
    revision=info.sha
    manifest=hf_hub_download(repo,binding.get('manifest_file','train/metadata.jsonl'),repo_type='dataset',revision=revision)
    selected=[];seen=set()
    for row in rank_rows(read_rows(manifest),binding.get('selection_seed',20260915),binding['task'],repo):
        try:c=normalize(row,binding)
        except (ValueError,KeyError):continue
        key=digest([c['prompt'],c['inputs'],c['source_input_id']])
        if key in seen:continue
        seen.add(key);selected.append((row,c))
        if len(selected)==5:break
    if len(selected)!=5:raise ValueError('Fewer than five valid candidates')
    destination.mkdir(parents=True)
    import shutil
    for row,c in selected:
        assets=c['inputs']+([c['dataset_target']] if 'dataset_target' in c else [])
        for asset in assets:
            target=inside(destination,asset['path'])
            remote=(Path(binding.get('path_prefix','train'))/asset['path']).as_posix()
            # Transport/access failures abort; they never silently choose easier-to-download cases.
            cached=hf_hub_download(repo,remote,repo_type='dataset',revision=revision)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cached,target)
    atomic_bytes(destination/'metadata.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r,c in selected).encode())
    result={**binding,'revision':revision,'manifest':str((destination/'metadata.jsonl').resolve()),'root':str(destination.resolve()),
            'snapshot_note':'Selected from pinned full upstream manifest; only selected assets downloaded'}
    write_json(destination/'binding.json',result);return result

def import_webdataset(binding,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    rows=[];members={}
    for shard in binding['shards']:
        shard=str(Path(shard).resolve())
        with tarfile.open(shard,'r:*') as tf:
            for member in tf:
                if not member.isfile():continue
                if member.size>50*1024**2:continue
                # WDS groups are flat and cannot traverse filesystem directories.
                name=Path(member.name).name
                if name!=member.name:raise ValueError('Expected flat WebDataset members')
                members[(shard,name)]=member
                if name.endswith('.json'):
                    if member.size>2*1024**2:raise ValueError('Oversized metadata')
                    row=json.load(tf.extractfile(member));row['_shard']=shard;rows.append(row)
    selected=[];seen=set()
    for row in rank_rows(rows,binding.get('selection_seed',20260915),binding['task'],binding['source_id']):
        try:
            if row.get('invalid_pair') and binding.get('exclude_invalid',True):continue
            prompt=row[binding.get('prompt_field','edit_instruction')]
            key=str(row['id']);ref=row.get('ref1_member',key+'.ref1.webp');target=row.get('target_member',key+'.target.webp')
            logical=digest([prompt,ref,row['_shard']])
            if logical in seen:continue
            for name in [ref,target]:
                if (row['_shard'],name) not in members:raise ValueError('Incomplete WebDataset sample: '+key)
            selected.append((row,ref,target));seen.add(logical)
            if len(selected)==5:break
        except KeyError:continue
    if len(selected)!=5:raise ValueError('Fewer than five valid WebDataset cases')
    destination.mkdir(parents=True);result_rows=[]
    for row,ref,target in selected:
        with tarfile.open(row['_shard'],'r:*') as tf:
            for name in [ref,target]:
                asset=inside(destination,name)
                atomic_bytes(asset,tf.extractfile(members[(row['_shard'],name)]).read())
        result_rows.append({**{k:v for k,v in row.items() if k!='_shard'},'ref1_file_name':ref,'target_file_name':target})
    atomic_bytes(destination/'metadata.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result_rows).encode())
    result={**binding,'root':str(destination.resolve()),'manifest':str((destination/'metadata.jsonl').resolve()),
            'input_fields':['ref1_file_name'],'prompt_field':binding.get('prompt_field','edit_instruction')}
    write_json(destination/'binding.json',result);return result
