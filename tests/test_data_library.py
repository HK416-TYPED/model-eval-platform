import hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from eval_platform.core import aspect_dimensions,generation_request,read_json
from eval_platform.data_library import extract_dataset,import_dataset,_tar_members,download_file
from eval_platform.data_jobs import start_import
from eval_platform.experiment import preflight
from eval_platform.suites import freeze_suite


class DataLibrary(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def tar(self, count=8, invalid=False):
        path=self.root/'samples.tar'
        with tarfile.open(path,'w') as tf:
            def add(name,raw):
                item=tarfile.TarInfo(name);item.size=len(raw);tf.addfile(item,io.BytesIO(raw))
            for i in range(count):
                key=str(i);prompt=f'original prompt {i}\n第二行'
                row={'id':key,'edit_instruction':prompt,'invalid_pair':invalid and i==0}
                for suffix,size in [('ref1',(48,32)),('target',(32,48))]:
                    out=io.BytesIO();Image.new('RGB',size,(i,30 if suffix=='ref1' else 80,0)).save(out,'PNG');raw=out.getvalue()
                    row[suffix+'_sha256']=hashlib.sha256(raw).hexdigest();add(key+'.'+suffix+'.webp',raw)
                add(key+'.prompt.txt',prompt.encode());add(key+'.json',json.dumps(row).encode())
        return path
    def spec(self,id='test',count=5):return {'dataset_id':id,'format':'webdataset','source':'local_tar','archive_path':str(self.root/'samples.tar'),'count':count,'selection_seed':123}
    def test_complete_pair_selection_is_reproducible_and_excludes_invalid(self):
        tar=self.tar(invalid=True)
        a=extract_dataset(self.root,self.spec(),tar);b=extract_dataset(self.root,self.spec('repeat'),tar)
        self.assertEqual(a['count'],5);self.assertNotIn('0',a['selected_ids']);self.assertEqual(a['selected_ids'],b['selected_ids'])
        records=[json.loads(l) for l in (self.root/'datasets/test/records.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(records),5)
        for row in records:
            self.assertEqual(row['target']['dimensions'],[32,48]);self.assertEqual(len(row['inputs']),1)
        self.assertEqual(a['source']['archive_bytes'],tar.stat().st_size)
    def test_insufficient_pairs_leave_no_ready_dataset(self):
        tar=self.tar(2)
        with self.assertRaises(ValueError):extract_dataset(self.root,self.spec(),tar)
        self.assertFalse((self.root/'datasets/test').exists())
    def test_tar_rejects_traversal_links_and_duplicate_names(self):
        for kind in ('traversal','link','duplicate'):
            path=self.root/(kind+'.tar')
            with tarfile.open(path,'w') as tf:
                item=tarfile.TarInfo('../escape' if kind=='traversal' else 'sample')
                if kind=='link':item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd'
                tf.addfile(item)
                if kind=='duplicate':tf.addfile(item)
            with tarfile.open(path) as tf,self.assertRaises(ValueError):_tar_members(tf)
    def test_aspect_ratio_and_pixel_budget(self):
        for dimensions in ([1024,1024],[1920,1080],[1080,1920],[1200,800],[800,1200]):
            w,h=aspect_dimensions(dimensions)
            self.assertEqual(w%16,0);self.assertEqual(h%16,0)
            self.assertLessEqual(w*h,1048576);self.assertGreater(w*h,1048576*.96)
            self.assertLess(abs((w/h)/(dimensions[0]/dimensions[1])-1),.025)
    def test_target_geometry_takes_precedence_and_never_exposes_target_pixels(self):
        case={'case_id':'x','task':'edit_dual','semantic_mode':'reference','prompt':'x',
              'inputs':[{'id':'a','path':'a.png','dimensions':[1600,900]},{'id':'b','path':'b.png','dimensions':[500,500]}],
              'dataset_target':{'path':'secret-target.png','dimensions':[800,1200]}}
        p={'width':1024,'height':1024,'target_pixels':1048576,'resolution_policy':'target_aspect'}
        a=generation_request(case,1,p,self.root);b=generation_request(case,2,p,self.root)
        self.assertLess(a['profile']['width'],a['profile']['height']);self.assertEqual(a['profile'],b['profile'])
        self.assertEqual(p['width'],1024);self.assertNotIn('secret-target',json.dumps(a));self.assertNotIn('dataset_target',a)
        case.pop('dataset_target');c=generation_request(case,1,p,self.root)
        self.assertGreater(c['profile']['width'],c['profile']['height']);self.assertEqual(c['geometry']['dimension_source'],'input_0')
        case.update(task='t2i',semantic_mode='t2i',inputs=[])
        self.assertEqual(generation_request(case,1,p,self.root)['profile'],{'width':1024,'height':1024})
    def test_500_cases_generate_correct_plan(self):
        rows=self.root/'rows.jsonl';rows.write_text(''.join(json.dumps({'id':str(i),'prompt':f'prompt {i}'})+'\n' for i in range(505)))
        binding={'source_id':'fixture','task':'t2i','manifest':str(rows),'root':str(self.root),'input_fields':[]}
        lock=freeze_suite(self.root/'suite',[binding],count=500)
        config={'suite_dir':str(self.root/'suite'),'models':[{'id':'m','backend':'anima_v7','family':'anima','tasks':['t2i']}],
                'profiles':{'anima':{'t2i':{'width':1024,'height':1024,'steps':1,'guidance_scale':3.5,'flow_shift':5}}}}
        result=preflight(config,verify_assets=False)
        self.assertTrue(result['ready'],result);self.assertEqual(result['planned'],2500);self.assertEqual(len(lock['cases']),500)
    def test_generic_manifest_preserves_input_order(self):
        for name,size in [('a',(64,32)),('b',(32,64)),('target',(48,64))]:Image.new('RGB',size).save(self.root/(name+'.png'))
        rows=self.root/'rows.jsonl';rows.write_text(json.dumps({'id':'1','instruction':'exact text','first':'a.png','second':'b.png','target':'target.png'}))
        spec={'dataset_id':'generic','source':'manifest','format':'manifest','root':str(self.root),'manifest_path':str(rows),
              'task':'edit_dual','input_fields':['first','second'],'prompt_field':'instruction','target_field':'target','count':1}
        result=import_dataset(self.root,spec)
        record=json.loads((self.root/'datasets/generic/records.jsonl').read_text(encoding='utf-8'))
        self.assertEqual([a['dimensions'] for a in record['inputs']],[[64,32],[32,64]])
        self.assertEqual(result['count'],1);self.assertEqual(record['prompt'],'exact text')
    def test_credentials_only_travel_through_worker_stdin(self):
        token='fixture-secret-value'
        process=type('Process',(),{'pid':111,'stdin':io.BytesIO()})()
        process.stdin.close=lambda:None
        with patch('eval_platform.data_jobs.subprocess.Popen',return_value=process) as launch:
            job=start_import(self.root,self.spec(),token)
        stored=(self.root/'data-jobs'/(job['id']+'.json')).read_text(encoding='utf-8')
        self.assertNotIn(token,stored);self.assertNotIn(token,str(launch.call_args));self.assertIn(token,process.stdin.getvalue().decode())
        with self.assertRaises(ValueError):start_import(self.root,{**self.spec('bad'),'token':token})
    def test_download_rejects_checksum_mismatch(self):
        destination=self.root/'download.tar';destination.with_suffix('.tar.partial').write_bytes(b'wrong')
        entry={'rfilename':'download.tar','size':5,'lfs':{'sha256':hashlib.sha256(b'valid').hexdigest()}}
        with self.assertRaisesRegex(ValueError,'SHA256'):download_file('owner/repo','revision',entry,destination,None)
        self.assertFalse(destination.exists())
    def test_download_resumes_exact_range_and_checks_complete_file(self):
        destination=self.root/'download.tar';destination.with_suffix('.tar.partial').write_bytes(b'abc')
        entry={'rfilename':'download.tar','size':6,'lfs':{'sha256':hashlib.sha256(b'abcdef').hexdigest()}}
        class Response:
            status_code=206;headers={'Content-Range':'bytes 3-5/6'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
            def iter_content(self,*args):yield b'def'
        with patch('eval_platform.data_library.requests.get',return_value=Response()) as request:
            result=download_file('owner/repo','revision',entry,destination,None)
        self.assertEqual(request.call_args.kwargs['headers']['Range'],'bytes=3-')
        self.assertEqual(destination.read_bytes(),b'abcdef');self.assertEqual(result['sha256'],entry['lfs']['sha256'])

if __name__=='__main__':unittest.main()
