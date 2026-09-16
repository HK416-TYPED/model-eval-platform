import tempfile,unittest
from pathlib import Path
try:
    from fastapi.testclient import TestClient
    from eval_platform.web import create_app
except ImportError:
    TestClient=None

@unittest.skipIf(TestClient is None,'FastAPI/httpx unavailable in this Python environment')
class WebTests(unittest.TestCase):
    def test_data_upload_and_empty_library(self):
        with tempfile.TemporaryDirectory() as d,TestClient(create_app(d)) as client:
            self.assertEqual(client.get('/api/datasets').json(),[])
            self.assertEqual(client.get('/api/data-jobs').json(),[])
            self.assertEqual(client.post('/api/uploads/not-a-tar.exe',content=b'bad').status_code,400)
            self.assertEqual(client.post('/api/uploads/empty.tar',content=b'').status_code,400)
            value=client.post('/api/uploads/metadata.jsonl',content=b'{"id":"1"}\n').json()
            self.assertTrue(Path(value['path']).is_relative_to(Path(d)))
            self.assertEqual(Path(value['path']).read_bytes(),b'{"id":"1"}\n')
            self.assertEqual(client.post('/api/suites',json={'name':'empty','dataset_ids':[]}).status_code,400)
            self.assertEqual(client.get('/api/datasets/no-such-data/preview').status_code,404)
            self.assertEqual(client.post('/api/datasets/import',json={'spec':{'dataset_id':'x','source':'local_tar','format':'webdataset','count':0}}).status_code,400)

    def test_local_api_config_and_cross_origin_write_rejection(self):
        with tempfile.TemporaryDirectory() as d,TestClient(create_app(d)) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.get('/static/theme.css').status_code,200)
            self.assertEqual(client.get('/static/app.js').status_code,200)
            self.assertEqual(client.post('/api/runs/missing/pdf').status_code,404)
            self.assertIn('anima_v7',client.get('/api/capabilities').json())
            self.assertEqual(client.get('/api/runs').json(),[])
            config={'name':'fixture','config':{'suite_dir':'missing','models':[]}}
            self.assertEqual(client.put('/api/configs',json=config,headers={'Origin':'https://unrelated.example'}).status_code,403)
            self.assertEqual(client.get('/api/configs').json(),{})
            self.assertEqual(client.put('/api/configs',json=config).status_code,200)
            check=client.post('/api/configs/fixture/preflight').json()
            self.assertFalse(check['ready'])
            self.assertIn('problems',check)
            self.assertEqual(client.put('/api/configs',json={'name':'../escape','config':{}}).status_code,400)
            self.assertEqual(client.get('/api/runs/missing').status_code,404)

if __name__=='__main__':unittest.main()
