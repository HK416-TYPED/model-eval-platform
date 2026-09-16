import json
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from eval_platform.store import Store
from eval_platform.web import create_app
from eval_platform.review import summarize


class Reviews(unittest.TestCase):
    def test_reverse_proxy_same_origin_write_and_cross_site_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            store.create('run',{'models':[]},[{'id':'j','model_id':'m','case_id':'c','task':'t2i','seed':1,'request':{}}])
            payload={'model_id':'m','case_id':'c','verdict':'mixed'}
            with TestClient(create_app(directory)) as client:
                url='/api/runs/run/reviews'
                self.assertEqual(client.put(url,json=payload,headers={'Origin':'https://public.example','Sec-Fetch-Site':'same-origin'}).status_code,200)
                for site in ['cross-site','same-site','none','']:
                    self.assertEqual(client.put(url,json=payload,headers={'Origin':'https://other.example','Sec-Fetch-Site':site}).status_code,403)
                self.assertEqual(client.put(url,json=payload,headers={'Origin':'https://other.example'}).status_code,403)

    def test_saved_review_is_scoped_validated_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            job={'id':'j','model_id':'m','case_id':'c','task':'t2i','seed':1,'request':{}}
            store.create('run',{'models':[]},[job])
            with TestClient(create_app(directory)) as client:
                payload={'model_id':'m','case_id':'c','verdict':'mixed','note':'原始 <script>文字</script>','reviewer':'测试'}
                self.assertEqual(client.put('/api/runs/run/reviews',json=payload).status_code,200)
                self.assertEqual(client.put('/api/runs/run/reviews',json={**payload,'verdict':'invalid'}).status_code,400)
                self.assertEqual(client.put('/api/runs/run/reviews',json={**payload,'case_id':'missing'}).status_code,404)
                self.assertEqual(client.put('/api/runs/run/reviews',json={**payload,'note':'x'*4001}).status_code,422)
                self.assertEqual(client.put('/api/runs/run/reviews',json={**payload,'note':'updated'}).status_code,200)
            reviews=Store(directory).reviews('run')
            self.assertEqual(len(reviews),1)
            self.assertEqual(reviews[0]['note'],'updated')
            self.assertEqual(store.jobs('run')[0]['state'],'queued')

    def test_pending_cancelled_and_missing_measurements_not_quality_scores(self):
        jobs=[{'id':str(i),'model_id':'m','task':'t2i','case_id':'c','seed':i,'state':state,'result':{'generation_seconds':float('nan')}} for i,state in enumerate(['succeeded','failed','queued','cancelled'])]
        value=summarize(jobs)
        self.assertEqual(value['groups'][0]['success_rate'],0.5)
        self.assertIsNone(value['groups'][0]['median_seconds'])
        self.assertEqual(value['reviewed_cases'],0)
        self.assertEqual(len(value['failures']),1)
