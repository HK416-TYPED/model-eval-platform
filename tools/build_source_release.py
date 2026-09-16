"""Build an auditable source ZIP from an explicit allowlist."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
TOOLS={'start_server.py','audit_data_library.py','import_dataset.py','build_source_release.py','install_pdf_font.sh'}
SECRET_PATTERNS=[rb'hf_[A-Za-z0-9]{20,}',rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']

def source_files(root):
    paths=[root/n for n in ['README.md','THIRD_PARTY_NOTICES.md','pyproject.toml','.gitignore']]
    for path in (root/'eval_platform').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and (path.suffix in {'.py','.js','.css','.html','.ttc'} or path.name=='COPYRIGHT.wqy-microhei'):paths.append(path)
    paths+=list((root/'tests').glob('test_*.py'))+list((root/'configs').glob('*.example.json'))
    paths += [root/'tools'/n for n in TOOLS]
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Refusing linked source: '+str(path))
        yield path

def build(root,destination):
    root=Path(root);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    entries={};manifest={}
    for path in source_files(root):
        name=path.relative_to(root).as_posix();raw=path.read_bytes()
        if any(re.search(pattern,raw) for pattern in SECRET_PATTERNS):raise ValueError('Credential-like content in '+name)
        if len(raw)>10*1024**2:raise ValueError('Unexpected large source asset: '+name)
        entries[name]=raw;manifest[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    entries['SOURCE_MANIFEST.json']=(json.dumps({'schema':'source-release/v1','files':manifest},indent=2)+'\n').encode()
    partial=destination.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted(entries.items()):
            info=zipfile.ZipInfo('model-eval-platform/'+name,date_time=(2026,9,15,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,raw)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():raise ValueError('ZIP integrity check failed')
        for name,raw in entries.items():
            if archive.read('model-eval-platform/'+name)!=raw:raise ValueError('Packaged bytes differ')
    partial.replace(destination);sha=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(sha+'  '+destination.name+'\n',encoding='utf-8')
    result={'path':str(destination.resolve()),'files':len(entries),'bytes':destination.stat().st_size,'sha256':sha}
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'dist/model-eval-platform-source-20260915.zip'))
    args=parser.parse_args();build(ROOT,args.output)
