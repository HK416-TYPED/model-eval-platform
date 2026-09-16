import json
from pathlib import Path
import shutil
import tempfile
import unittest
from html.parser import HTMLParser
from tests import test_pdf
from eval_platform.store import Store
from eval_platform.core import file_sha
from eval_platform.report import build_report


def fixture(root, model_count=3):
    suite=root/'suite';suite.mkdir(parents=True,exist_ok=True)
    data=json.loads(test_pdf.PdfReports().fixture(suite,model_count).read_text())
    data['suite']['cases'][0]['digest']='fixture'
    (suite/'suite.lock.json').write_text(json.dumps(data['suite']),encoding='utf-8')
    store=Store(root/'state');run=root/'state/runs/layout-fixture';run.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(suite/'output.png',run/'output.png')
    spec={**data['experiment'],'suite_dir':str(suite)}
    store.create('layout-fixture',spec,[{**j,'request':{}} for j in data['jobs']])
    for j in data['jobs']:
        store.update_job(j['id'],j['state'],result={**j['result'],'image':'output.png','sha256':file_sha(run/'output.png')},error=j['error'])
    return Path(build_report(store,'layout-fixture'))


class ReportLayoutTests(unittest.TestCase):
    def test_complete_escaped_single_and_multiple_checkpoint_reports(self):
        for count in (1,3):
            with self.subTest(models=count),tempfile.TemporaryDirectory() as temp:
                path=fixture(Path(temp),count);text=path.read_text(encoding='utf-8')
                self.assertEqual(text.count('class="result-card"'),count*5)
                self.assertEqual(text.count('class="checkpoint-group"'),count)
                self.assertEqual(text.count('EXPECTED-FAILURE'),count)
                self.assertIn('&lt;script&gt;literal&lt;/script&gt;',text)
                self.assertIn('PROMPT-END',text)
                self.assertNotIn('<table class="compare-table">',text)
                for name in ['report.css','report-ui.js','viewer.js']:
                    self.assertTrue((path.parent/name).is_file())
                parser=HTMLParser();parser.feed(text)


if __name__=='__main__':unittest.main()
