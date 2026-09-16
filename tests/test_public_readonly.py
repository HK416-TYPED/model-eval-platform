import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from eval_platform.public_readonly import create_public_app
from eval_platform.web import create_app


class PublicReadOnlyTests(unittest.TestCase):
    def test_writes_blocked_and_existing_artifacts_readable(self):
        with tempfile.TemporaryDirectory() as root:
            writable = create_app(root)
            paths = [(method, route.path.replace('{run_id}', 'missing').replace('{name}', 'x').replace('{filename}', 'x.jsonl'))
                     for route in writable.routes for method in getattr(route, 'methods', [])
                     if method not in {'GET', 'HEAD', 'OPTIONS'}]
            with TestClient(create_public_app(root)) as client:
                for method, path in paths:
                    with self.subTest(method=method, path=path):
                        r = client.request(method, path, json={}, headers={'Sec-Fetch-Site': 'same-origin'})
                        self.assertEqual(r.status_code, 403)
                for method in ['DELETE', 'PATCH', 'POST', 'PUT']:
                    self.assertEqual(client.request(method, '/static/index.html').status_code, 403)
                self.assertEqual(client.get('/api/runs').json(), [])
                self.assertEqual(client.get('/api/configs').json(), {})
                self.assertIn('只读访问', client.get('/').text)
                self.assertIn('只读访问', client.get('/static/index.html').text)
                self.assertEqual(client.get('/static/app.js').status_code, 200)
                folder = Path(root)/'runs'/'sample'/'report'
                folder.mkdir(parents=True)
                (folder/'index.html').write_text('existing report')
                self.assertEqual(client.get('/files/sample/report/index.html').text, 'existing report')
                (folder/'report.pdf').write_bytes(b'%PDF-fixture')
                pdf=client.get('/files/sample/report/report.pdf?v=3')
                self.assertEqual(pdf.content,b'%PDF-fixture')
                self.assertEqual(pdf.headers['cache-control'],'no-store')
                schema = client.get('/openapi.json').json()
                self.assertTrue(all(set(verbs) <= {'get', 'head', 'options'} for verbs in schema['paths'].values()))
            with TestClient(writable) as client:
                self.assertEqual(client.put('/api/configs', json={'name':'fixture','config':{}}).status_code, 200)
