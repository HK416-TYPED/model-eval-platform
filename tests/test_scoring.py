import json,tempfile,unittest
from pathlib import Path
from PIL import Image
from eval_platform.core import file_sha,write_json
from eval_platform.scoring import score_run
from eval_platform.store import Store

class ScoringTests(unittest.TestCase):
    def test_plugin_cache_and_target_opt_in_without_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);suite=root/'suite';suite.mkdir();Image.new('RGB',(16,16),'blue').save(suite/'target.png')
            case={'case_id':'c','task':'t2i','semantic_mode':'t2i','prompt':'fixture prompt','inputs':[],
                  'dataset_target':{'path':'target.png','sha256':file_sha(suite/'target.png')},'provenance':{'source_id':'fixture','row_id':'1'},'digest':'fixture'}
            write_json(suite/'suite.lock.json',{'cases':[case],'seeds':[1],'sources':[]})
            store=Store(root/'state');run=store.root/'runs/r';run.mkdir(parents=True);Image.new('RGB',(16,16),'red').save(run/'output.png')
            model={'id':'ep1','family':'anima','identity':{'assets':{'checkpoint':{'sha256':'fixture'}}}}
            store.create('r',{'suite_dir':str(suite),'models':[model],'profiles':{'anima':{'t2i':{}}},'unavailable':[]},
                [{'id':'j','model_id':'ep1','case_id':'c','task':'t2i','seed':1,'request':{}}])
            store.update_job('j','succeeded',result={'image':'output.png','sha256':file_sha(run/'output.png'),'generation_seconds':1,'peak_allocated_bytes':0})
            plugin=root/'plugin.py';plugin.write_text("def score(request, parameters):\n    assert 'target' not in request\n    return {'status':'scored','metrics':[{'name':'fixture_only','value':0.5,'direction':'higher'}]}\n")
            config={'id':'fixture','version':'1','transport':'python','module_path':str(plugin),'rubric':{},'parameters':{},'uses_target':False}
            first=score_run(store,'r',config);second=score_run(store,'r',config)
            self.assertEqual(first['scored'],1);self.assertEqual(second['cached'],1)
            self.assertEqual(store.jobs('r')[0]['attempts'],0)
            config['parameters']={'revision':2}
            self.assertEqual(score_run(store,'r',config)['scored'],1)

if __name__=='__main__':unittest.main()
