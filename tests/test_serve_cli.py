import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from eval_platform.cli import main


class ServeCLI(unittest.TestCase):
    def test_public_readonly_and_private_admin(self):
        for extra,expected in [([],200),(['--read-only','--host','0.0.0.0','--port','6008'],403)]:
            with tempfile.TemporaryDirectory() as state,patch('sys.argv',['evalctl','--state',state,'serve',*extra]),patch('uvicorn.run') as run:
                main()
                with TestClient(run.call_args.args[0]) as client:
                    self.assertEqual(client.put('/api/configs',json={'name':'fixture','config':{}}).status_code,expected)
                    self.assertEqual(client.get('/openapi.json').json()['info']['version'],'0.3.0')

    def test_writable_non_loopback_rejected(self):
        with patch('sys.argv',['evalctl','serve','--host','0.0.0.0']),patch('uvicorn.run') as run:
            with self.assertRaises(SystemExit):main()
            run.assert_not_called()
