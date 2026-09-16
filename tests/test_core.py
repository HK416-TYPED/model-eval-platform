import copy
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from eval_platform.core import SEEDS, digest, generation_request, inside, validate_case, validate_profile, write_json
from eval_platform.suites import freeze_suite,rank_rows,verify_suite
from eval_platform.store import Store
from eval_platform.worker import GpuLock,recover
from eval_platform.scoring import validate_score
from eval_platform.report import build_report,export_zip

class Invariants(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data=self.root/'data';self.data.mkdir()
        Image.new('RGB',(32,48),'blue').save(self.data/'input.png')
        Image.new('RGB',(32,48),'green').save(self.data/'target.png')
        self.rows=[{'id':str(i),'prompt':f'Original prompt {i} --seed 123\n第二行','source':'input.png','target':'target.png'} for i in range(12)]
        self.manifest=self.data/'metadata.jsonl';self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.binding={'task':'edit_single','semantic_mode':'edit','source_id':'fixture/test-only','manifest':str(self.manifest),
                      'root':str(self.data),'input_fields':['source'],'target_field':'target'}
    def tearDown(self):self.temp.cleanup()
    def freeze(self):return freeze_suite(self.root/'suite',[self.binding])
    def test_order_independent_sampling(self):
        a=rank_rows(self.rows,7,'t2i','x');b=rank_rows(list(reversed(self.rows)),7,'t2i','x')
        self.assertEqual(a,b)
        self.assertNotEqual(a,rank_rows(self.rows,8,'t2i','x'))
    def test_5_by_5_and_target_never_in_request(self):
        lock=self.freeze();self.assertEqual(len(lock['cases'])*len(lock['seeds']),25)
        for c in lock['cases']:
            request=generation_request(c,SEEDS[0],{'width':32,'height':32},self.root/'suite')
            self.assertNotIn('dataset_target',request)
            self.assertNotIn('target.png',json.dumps(request))
            self.assertIn('--seed 123\n第二行',request['prompt'])
    def test_suite_cannot_be_overwritten(self):
        self.freeze()
        with self.assertRaises(FileExistsError):self.freeze()
    def test_corrupt_frozen_bytes_fail(self):
        lock=self.freeze();(self.root/'suite'/lock['cases'][0]['inputs'][0]['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):verify_suite(self.root/'suite')
    def test_insufficient_cases_do_not_repeat(self):
        self.manifest.write_text(json.dumps(self.rows[0]))
        with self.assertRaises(ValueError):self.freeze()
        self.assertFalse((self.root/'suite').exists())
    def test_duplicate_t2i_prompts_do_not_fill_five_cases(self):
        self.binding.update(task='t2i',semantic_mode='t2i',input_fields=[])
        self.manifest.write_text(''.join(json.dumps({**r,'prompt':'identical'})+'\n' for r in self.rows))
        with self.assertRaises(ValueError):self.freeze()
    def test_dual_order_and_source_constraints(self):
        c={'task':'edit_dual','prompt':'unchanged','semantic_mode':'reference','inputs':[{'id':'a'},{'id':'b'}]}
        validate_case(c);bad=copy.deepcopy(c);bad['inputs'].pop()
        with self.assertRaises(ValueError):validate_case(bad)
        bad=copy.deepcopy(c);bad['source_input_id']='a'
        with self.assertRaises(ValueError):validate_case(bad)
        self.assertNotEqual(digest(c),digest({**c,'inputs':list(reversed(c['inputs']))}))
    def test_unsafe_path_is_rejected(self):
        with self.assertRaises(ValueError):inside(self.data,'../secret')
    def test_unknown_sampling_parameters_fail(self):
        with self.assertRaises(ValueError):validate_profile({'width':32,'height':32,'steps':2,'guidance':3,'hidden_cfg':4},'krea2_official','t2i')
    def test_scores_absent_are_not_zero(self):
        validate_score({'status':'not_applicable','reason':'no target'})
        with self.assertRaises(ValueError):validate_score({'status':'failed','metrics':[{'value':0}]})
        with self.assertRaises(ValueError):validate_score({'status':'scored','metrics':[{'name':'quality','direction':'higher','value':float('nan')}]})
    def test_resume_and_report_keep_failed_cells(self):
        lock=self.freeze();store=Store(self.root/'state');run_dir=store.root/'runs'/'r';run_dir.mkdir(parents=True)
        model={'id':'m','family':'anima','identity':{'assets':{'checkpoint':{'sha256':'0'*64}}}}
        spec={'suite_dir':str(self.root/'suite'),'models':[model],'profiles':{'anima':{'edit_single':{}}},'unavailable':[]}
        jobs=[]
        for i,c in enumerate(lock['cases']):
            for s in SEEDS:jobs.append({'id':digest([i,s]),'model_id':'m','case_id':c['case_id'],'task':c['task'],'seed':s,'request':{}})
        store.create('r',spec,jobs);store.update_job(jobs[0]['id'],'running');store.update_job(jobs[1]['id'],'failed',error='<script>bad</script>')
        store.update_job(jobs[2]['id'],'succeeded',result={'image':'missing.png','sha256':'wrong'})
        recover(store,'r')
        states={j['id']:j['state'] for j in store.jobs('r')}
        self.assertEqual(states[jobs[0]['id']],'queued');self.assertEqual(states[jobs[1]['id']],'failed');self.assertEqual(states[jobs[2]['id']],'queued')
        page=Path(build_report(store,'r')).read_text(encoding='utf-8')
        self.assertIn('&lt;script&gt;bad&lt;/script&gt;',page)
        self.assertEqual(page.count('data-seed='),25)
        store.cancel('r');self.assertEqual(store.summary('r')['state'],'cancelled')
        recover(store,'r',retry=True);self.assertEqual(store.summary('r')['counts']['queued'],25)
    def test_gpu_lock_excludes_second_worker(self):
        with GpuLock(self.root,'cuda:0'):
            with self.assertRaises(RuntimeError):
                with GpuLock(self.root,'cuda:0'):pass

if __name__=='__main__':unittest.main()
