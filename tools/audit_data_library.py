"""Verify every imported sample and complete HF archive; produce a portable audit."""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from eval_platform.core import file_sha,inside,read_json,write_json,aspect_dimensions
from eval_platform.data_library import list_datasets
from eval_platform.suites import read_rows

def audit(state):
    state=Path(state);results=[]
    for info in list_datasets(state):
        root=state/'datasets'/info['id'];source=info['source'];records=list(read_rows(root/'records.jsonl'))
        assert len(records)==info['count']==len(set(r['id'] for r in records))
        assert file_sha(root/'metadata.jsonl')==info['manifest_sha256']
        assert file_sha(root/'records.jsonl')==info['records_sha256']
        verified={};dimensions=Counter();sampling=Counter()
        for row in records:
            assert row['prompt'].strip()
            expected={'t2i':0,'edit_single':1,'edit_dual':2}[info['task']]
            assert len(row['inputs'])==expected
            if info['format']!='manifest':assert row.get('target')
            for asset in row['inputs']+([row['target']] if row.get('target') else []):
                if asset['path'] in verified:continue
                path=inside(root,asset['path']);assert file_sha(path)==asset['sha256']
                assert path.stat().st_size==asset['bytes']
                with Image.open(path) as image:image.load();assert list(image.size)==asset['dimensions']
                verified[asset['path']]=asset['sha256']
            asset=row.get('target') or next(iter(row['inputs']),None)
            if asset:
                dimensions['x'.join(map(str,asset['dimensions']))]+=1
                sampling['x'.join(map(str,aspect_dimensions(asset['dimensions'])))]+=1
        archive=None
        if source.get('repo'):
            path=state/'downloads'/source['repo'].replace('/','--')/source['revision']/source['filename']
            assert path.stat().st_size==source['archive_bytes'];assert file_sha(path)==source['archive_sha256']
            archive={k:source[k] for k in ['repo','revision','filename','archive_bytes','archive_sha256','complete_archive']}
        results.append({'id':info['id'],'count':len(records),'task':info['task'],'unique_image_files':len(verified),
                        'image_bytes':info['image_bytes'],'archive':archive,'selection_seed':info['selection_seed'],
                        'manifest_sha256':info['manifest_sha256'],'records_sha256':info['records_sha256'],
                        'candidate_count':info['candidate_count'],'excluded':info['excluded'],
                        'target_dimensions':dict(dimensions),'sampling_dimensions':dict(sampling),'status':'verified'})
        print(json.dumps({'dataset':info['id'],'verified_samples':len(records),'verified_images':len(verified)}),flush=True)
    return {'schema':'data-audit/v1','checked_at_unix':time.time(),'datasets':results,'total_samples':sum(r['count'] for r in results)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--state',default='state');parser.add_argument('--output',default='state/data-audit.json')
    args=parser.parse_args();write_json(args.output,audit(args.state))
