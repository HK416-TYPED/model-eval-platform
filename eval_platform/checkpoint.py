"""Bridge training V7 metadata to the stricter published runtime format.

Never modifies the source, key names, tensor shapes, dtypes or tensor bytes.
Only three absent release-side declarations may be filled. Conflicts fail.
"""
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from .backends import load_package_runtime
from .core import file_sha,write_json
from .experiment import safetensors_header

def prepare_v7(source,destination,runtime_dir):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:raise ValueError('Source checkpoint must remain untouched')
    if destination.exists():raise FileExistsError(destination)
    module=load_package_runtime(runtime_dir);header=safetensors_header(source)
    metadata=header.get('__metadata__',{})
    permitted={'runtime_prompt_rewrite':'false','runtime_mask_input':'false','semantic_slot_roles':'false'}
    # All substantive architecture and prompt-contract declarations must already agree.
    for key,expected in module.V7_REQUIRED_METADATA.items():
        if key not in permitted and metadata.get(key)!=expected:raise ValueError('Incompatible V7 declaration: '+key)
        if key in permitted and key in metadata and metadata[key]!=expected:raise ValueError('Conflicting release declaration: '+key)
    added={k:v for k,v in permitted.items() if k not in metadata}
    if not added:module.validate_v7_checkpoint(source);return {'path':str(source),'changed':False}
    source_sha=file_sha(source)
    header['__metadata__']={**metadata,**added,'eval_original_sha256':source_sha,'eval_metadata_bridge':'v7-release-declarations-v1'}
    encoded=json.dumps(header,ensure_ascii=False,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < source.stat().st_size+512*1024**2:raise ValueError('Not enough free space for immutable checkpoint copy')
    tmp=destination.with_suffix('.partial');payload_hash=hashlib.sha256()
    try:
        with source.open('rb') as src,tmp.open('xb') as dst:
            n=struct.unpack('<Q',src.read(8))[0];src.seek(8+n)
            dst.write(struct.pack('<Q',len(encoded)));dst.write(encoded)
            for block in iter(lambda:src.read(8*1024*1024),b''):payload_hash.update(block);dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        # The upstream release validator checks every required native tensor shape,
        # dtype, configuration value and prohibited legacy module prefix.
        candidate=destination.with_name(destination.stem+'.validating.safetensors')
        tmp.rename(candidate)
        try:module.validate_v7_checkpoint(candidate)
        except BaseException:candidate.unlink();raise
        check_hash=hashlib.sha256()
        with candidate.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];f.seek(8+n)
            for block in iter(lambda:f.read(8*1024*1024),b''):check_hash.update(block)
        if check_hash.digest()!=payload_hash.digest():candidate.unlink();raise ValueError('Tensor byte verification failed')
        if file_sha(source)!=source_sha:candidate.unlink();raise ValueError('Source changed during copy')
        candidate.rename(destination)
        record={'schema':'checkpoint-metadata-bridge/v1','source':str(source),'source_sha256':source_sha,
                'path':str(destination),'sha256':file_sha(destination),'added_metadata':added,
                'tensor_payload_sha256':payload_hash.hexdigest(),'tensor_payload_identical':True}
        write_json(destination.with_suffix('.provenance.json'),record);return record
    finally:
        if tmp.exists():tmp.unlink()
