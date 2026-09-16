import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from eval_platform.pdf_report import build_pdf


class PdfReports(unittest.TestCase):
    def fixture(self, root, model_count=1):
        Image.new('RGB', (80, 120), '#35664a').save(root/'input.png')
        Image.new('RGB', (120, 80), '#a8b153').save(root/'output.png')
        case = {'case_id': 'case-1', 'task': 'edit_single', 'semantic_mode': 'edit',
                'prompt': '原始提示词 <script>literal</script> & fixed seed\n' + '长文本不能截断。'*180 + 'PROMPT-END',
                'inputs': [{'id': 'source', 'path': 'input.png'}],
                'dataset_target': {'id': 'target', 'path': 'input.png'},
                'provenance': {'row_id': '1', 'source_id': 'test-only'}}
        models = [{'id': f'checkpoint-{i}', 'family': 'anima',
                   'identity': {'assets': {'checkpoint': {'sha256': str(i)*64}}}} for i in range(model_count)]
        jobs = []
        for model in models:
            for seed in range(101, 106):
                failed = seed == 103
                jobs.append({'id': model['id']+str(seed), 'model_id': model['id'], 'case_id': 'case-1',
                             'task': 'edit_single', 'seed': seed, 'state': 'failed' if failed else 'succeeded',
                             'report_image': None if failed else 'output.png', 'error': 'EXPECTED-FAILURE' if failed else None,
                             'result': {'generation_seconds': 2.5, 'peak_allocated_bytes': 123456}})
        report = {'summary': {'id': 'pdf-fixture', 'state': 'completed_with_errors', 'planned': 5*model_count,
                              'counts': {'succeeded': 4*model_count, 'failed': model_count}},
                  'suite': {'cases': [case], 'seeds': list(range(101, 106)), 'sources': [], 'digest': 'fixture'},
                  'experiment': {'models': models, 'profiles': {'anima': {'edit_single': {'width': 1024}}}},
                  'jobs': jobs, 'scores': []}
        path = root/'report.json'; path.write_text(json.dumps(report), encoding='utf-8')
        return path

    def test_embedded_chinese_full_prompt_images_and_failed_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = build_pdf(self.fixture(root))
            pdf = PdfReader(output)
            text = '\n'.join(page.extract_text() for page in pdf.pages)
            self.assertIn('模型效果测试报告', text)
            self.assertIn('PROMPT-END', text)
            self.assertIn('<script>literal</script>', text)
            self.assertIn('EXPECTED-FAILURE', text)
            self.assertIn('未评分', text)
            for seed in range(101, 106):
                self.assertIn('SEED '+str(seed), text)
            case_page = next(page for page in pdf.pages if 'SEED 101' in page.extract_text())
            for seed in range(101, 106):
                self.assertIn('SEED '+str(seed), case_page.extract_text())
            self.assertIn('输入 1', case_page.extract_text())
            self.assertTrue(case_page.images)
            fonts = [font.get_object() for page in pdf.pages for font in page['/Resources']['/Font'].values()]
            self.assertTrue(any('/FontFile2' in font.get('/FontDescriptor', {}) for font in fonts))
            self.assertFalse(list(root.glob('*.partial')))

    def test_multiple_checkpoints_paginate_without_losing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = PdfReader(build_pdf(self.fixture(root, 4)))
            text = '\n'.join(page.extract_text() for page in pdf.pages)
            for index in range(4):
                self.assertIn(f'checkpoint-{index}', text)
            # Pagination can become denser, but no checkpoint/seed may disappear.
            self.assertGreater(len(pdf.pages), 1)
            self.assertEqual(text.count('EXPECTED-FAILURE'), 8)  # case cells plus complete failure appendix
            for seed in range(101, 106):
                self.assertEqual(text.count('SEED '+str(seed)), 4)

    def test_two_single_checkpoint_cases_share_one_page(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=self.fixture(root);data=json.loads(source.read_text())
            case=copy.deepcopy(data['suite']['cases'][0]);case['case_id']='case-2'
            data['suite']['cases'].append(case)
            data['jobs'] += [{**j,'id':j['id']+'-2','case_id':'case-2'} for j in list(data['jobs'])]
            source.write_text(json.dumps(data),encoding='utf-8')
            pdf=PdfReader(build_pdf(source))
            pages=[page.extract_text() for page in pdf.pages if 'SEED 101' in page.extract_text()]
            self.assertEqual(len(pages),1)
            self.assertEqual(pages[0].count('SEED '),10)
            self.assertIn('case-1',pages[0]);self.assertIn('case-2',pages[0])


if __name__ == '__main__':
    unittest.main()
