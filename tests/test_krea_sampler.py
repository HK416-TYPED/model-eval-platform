"""CPU contract test using the pinned official sampler and tiny synthetic modules.

This verifies adapter parity only. It is not a Krea2 weight/quality evaluation.
"""
import importlib.util,os,unittest
from pathlib import Path
from eval_platform.backends import Krea2OfficialBackend

source=Path(os.environ.get('KREA2_SOURCE_DIR','vendor/krea2-official'))/'sampling.py'
try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None or not source.is_file(),'Pinned official source and torch required')
class OfficialSamplerContract(unittest.TestCase):
    def test_adapter_matches_official_seed_guidance_and_prompt(self):
        module_spec=importlib.util.spec_from_file_location('_krea_sampling_test',source)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        from types import SimpleNamespace
        class Dit(torch.nn.Module):
            config=SimpleNamespace(patch=2)
            def forward(self,img,context,**kwargs):return torch.zeros_like(img)+context.mean()
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.seen=[]
            def forward(self,prompts):
                self.seen+=prompts
                value=0.01 if prompts[0] else 0.0
                return torch.full((len(prompts),2,1),value,dtype=torch.bfloat16),torch.ones(len(prompts),2,dtype=torch.bool)
        class AE(torch.nn.Module):
            channels=16;compression=8
            def decode(self,x):
                self.latent=x.clone()
                return torch.nn.functional.interpolate(x[:,:3],scale_factor=8,mode='nearest')
        prompt='Original instruction --seed 999\nNo parsing'
        profile={'width':32,'height':32,'steps':2,'guidance':3.5,'y1':0.5,'y2':1.15,'negative_prompt':''}
        direct_ae=AE();direct_encoder=Encoder()
        expected=module.sample(Dit(),direct_ae,direct_encoder,[prompt],device='cpu',seed=42,**{k:v for k,v in profile.items() if k!='negative_prompt'})[0]
        backend=Krea2OfficialBackend();backend.torch=torch;backend.device='cpu';backend.dit=Dit();backend.encoder=Encoder();backend.ae=AE();backend.sample=module.sample
        request={'case_id':'fixture','task':'t2i','semantic_mode':'t2i','source_input_id':None,'inputs':[],
                 'prompt':prompt,'seed':42,'profile':profile}
        steps=[]
        actual,metadata=backend.generate(request,lambda n,total:steps.append((n,total)),lambda:None)
        self.assertEqual(actual.tobytes(),expected.tobytes())
        self.assertTrue(torch.equal(backend.ae.latent,direct_ae.latent))
        self.assertEqual(backend.encoder.seen,[prompt,''])
        self.assertEqual(steps,[(1,2),(2,2)])
        self.assertIn('cond + guidance',metadata['guidance_formula'])
        repeated,_=backend.generate(request,lambda *a:None,lambda:None)
        self.assertEqual(repeated.tobytes(),actual.tobytes())

if __name__=='__main__':unittest.main()
