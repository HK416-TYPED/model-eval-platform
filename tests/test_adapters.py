import copy
import hashlib
import json
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from eval_platform.backends import hosted_anima_runtime,Krea2EditBackend
from eval_platform.checkpoint import prepare_v7
from eval_platform.comparison import validate_comparison_models,append_checkpoint
from eval_platform.core import SEEDS,digest,file_sha
from eval_platform.store import Store
from eval_platform.suites import freeze_suite
from PIL import Image

class Adapters(unittest.TestCase):
    def test_descriptor_bridge_preserves_functions(self):
        class Primitives:
            def _validate_sample_inputs(*,width):return width
            def _comfy_image_to_pil(image,*,name):return image,name
            def _normalise_reference_request(images,slots,*,mode):return images,slots,mode
        class Runtime(Primitives):pass
        module=types.SimpleNamespace(base=types.SimpleNamespace(NativeContextRuntimePrimitives=Primitives),AnimaNativeContextV7Runtime=Runtime)
        cls=hosted_anima_runtime(module);runtime=cls()
        self.assertEqual(runtime._validate_sample_inputs(width=16),16)
        self.assertEqual(runtime._comfy_image_to_pil('img',name='ref'),('img','ref'))
        self.assertEqual(runtime._normalise_reference_request(['a','b'],[0,1],mode='reference'),(['a','b'],[0,1],'reference'))
        self.assertIs(cls.__dict__['_comfy_image_to_pil'].__func__,Primitives.__dict__['_comfy_image_to_pil'])

    def test_metadata_bridge_leaves_tensor_bytes_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'original.safetensors';dst=root/'derived.safetensors'
            payload=bytes(range(32))
            header={'__metadata__':{'anima_prompt_contract':'exact'},'weight':{'dtype':'U8','shape':[32],'data_offsets':[0,32]}}
            encoded=json.dumps(header).encode();src.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            original_sha=file_sha(src)
            declarations={'anima_prompt_contract':'exact','runtime_prompt_rewrite':'false','runtime_mask_input':'false','semantic_slot_roles':'false'}
            def validate(path):
                with open(path,'rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));actual=f.read()
                self.assertEqual(actual,payload)
                self.assertTrue(all(h['__metadata__'][k]==v for k,v in declarations.items()))
            module=types.SimpleNamespace(V7_REQUIRED_METADATA=declarations,validate_v7_checkpoint=validate)
            with patch('eval_platform.checkpoint.load_package_runtime',return_value=module):
                result=prepare_v7(src,dst,root)
            self.assertEqual(file_sha(src),original_sha)
            self.assertEqual(result['tensor_payload_sha256'],hashlib.sha256(payload).hexdigest())
            self.assertTrue(result['tensor_payload_identical'])
            self.assertNotEqual(result['sha256'],original_sha)

    def test_krea_edit_cannot_produce_placeholder_success(self):
        with self.assertRaises(NotImplementedError):Krea2EditBackend().load({})

    def identity(self,checkpoint='a'):
        return {'digest':digest(checkpoint),'assets':{'checkpoint':{'sha256':checkpoint},'text_encoder':{'sha256':'t'},'vae':{'sha256':'v'}},
                'source':{'digest':'same-upstream'},'adapter':{'files':{'backends.py':'same-backend','core.py':'same-core','report.py':'report1'}},
                'environment':{'torch':'test-only'},'runtime_options':{}}

    def test_epoch_comparison_rejects_encoder_change(self):
        a={'id':'ep1','family':'anima','backend':'anima_v7','identity':self.identity('a')}
        b={**a,'id':'ep2','identity':self.identity('b')}
        validate_comparison_models([a,b])
        b['identity']['adapter']['files']['report.py']='report2'
        validate_comparison_models([a,b])
        b['identity']['assets']['text_encoder']['sha256']='different'
        with self.assertRaises(ValueError):validate_comparison_models([a,b])

    def test_append_epoch_preserves_existing_job_ids_and_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir()
            (data/'rows.jsonl').write_text(''.join(json.dumps({'id':str(i),'prompt':f'prompt {i}'})+'\n' for i in range(6)))
            lock=freeze_suite(root/'suite',[{'task':'t2i','source_id':'test/fixture','root':str(data),'manifest':str(data/'rows.jsonl'),'input_fields':[]}])
            store=Store(root/'state');model={'id':'ep1','family':'anima','backend':'anima_v7','tasks':['t2i'],'identity':self.identity('a')}
            profile={'width':32,'height':32,'steps':2,'guidance_scale':3.5,'flow_shift':5}
            spec={'suite_dir':str(root/'suite'),'suite_digest':lock['digest'],'models':[model],'profiles':{'anima':{'t2i':profile}},'unavailable':[]}
            jobs=[{'id':digest([c['case_id'],s]),'model_id':'ep1','case_id':c['case_id'],'task':'t2i','seed':s,'request':{}} for c in lock['cases'] for s in SEEDS]
            store.create('r',spec,jobs);(store.root/'runs/r').mkdir(parents=True)
            old=store.jobs('r');new_model={k:v for k,v in model.items() if k!='identity'};new_model['id']='ep2'
            with patch('eval_platform.comparison.model_identity',return_value=self.identity('b')):
                summary=append_checkpoint(store,'r',new_model)
            self.assertEqual(summary['planned'],50)
            self.assertEqual([j for j in store.jobs('r') if j['model_id']=='ep1'],old)
            expected={(j['case_id'],j['seed']) for j in old}
            self.assertEqual({(j['case_id'],j['seed']) for j in store.jobs('r') if j['model_id']=='ep2'},expected)

if __name__=='__main__':unittest.main()
