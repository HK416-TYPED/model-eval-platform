import io,json,tarfile,tempfile,unittest
from pathlib import Path
from PIL import Image
from eval_platform.datasets import import_webdataset
from eval_platform.suites import freeze_suite

class WebDatasetTests(unittest.TestCase):
    def test_flat_shards_sample_complete_pairs_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shard=root/'fixture.tar';png=io.BytesIO();Image.new('RGB',(32,32),'blue').save(png,format='PNG');pixels=png.getvalue()
            with tarfile.open(shard,'w') as tf:
                for i in range(7):
                    key=f'{i:06d}'
                    entries={key+'.json':json.dumps({'id':key,'edit_instruction':f'edit {i}','invalid_pair':i==0}).encode(),
                             key+'.ref1.webp':pixels,key+'.target.webp':pixels}
                    for name,data in entries.items():
                        ti=tarfile.TarInfo(name);ti.size=len(data);tf.addfile(ti,io.BytesIO(data))
            binding=import_webdataset({'source_id':'fixture/test-only','task':'edit_single','semantic_mode':'edit','shards':[str(shard)]},root/'import')
            lock=freeze_suite(root/'suite',[binding])
            self.assertEqual(len(lock['cases']),5)
            self.assertNotIn('000000',{c['provenance']['row_id'] for c in lock['cases']})
            self.assertEqual(len(list((root/'import').glob('*.webp'))),10)

if __name__=='__main__':unittest.main()
