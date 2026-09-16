"""Backend adapters. Heavy dependencies are imported only inside GPU workers."""
import gc
import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from .core import validate_case, validate_profile

def load_package_runtime(path):
    path=Path(path).resolve(); name='_eval_anima_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(path)];package.__package__=name
        sys.modules[name]=package
    return importlib.import_module(name+'.runtime')

def hosted_anima_runtime(module):
    # The pinned publication snapshot omitted @staticmethod on these three
    # utility functions. Restore only their descriptors in our subclass, leaving
    # the upstream files and all numerical code unchanged.
    primitives=module.base.NativeContextRuntimePrimitives
    overrides={}
    for name in ['_validate_sample_inputs','_comfy_image_to_pil','_normalise_reference_request']:
        method=inspect.getattr_static(primitives,name)
        if isinstance(method,staticmethod):continue
        if list(inspect.signature(method).parameters)[0] in ('self','cls'):
            raise RuntimeError('Unexpected upstream descriptor signature: '+name)
        overrides[name]=staticmethod(method)
    return type('HostedAnimaV7Runtime',(module.AnimaNativeContextV7Runtime,),overrides)

class AnimaV7Backend:
    def load(self, model, device='cuda:0'):
        self.model=model
        module=load_package_runtime(model['runtime_dir'])
        # Strict upstream validator checks all tensor shapes and metadata contracts.
        module.validate_v7_checkpoint(model['checkpoint'])
        self.runtime=hosted_anima_runtime(module)(
            model['checkpoint'],model['text_encoder'],model['vae'],device=device,
            **model.get('runtime_options',{}))

    def generate(self, request, progress, cancel):
        import torch
        import numpy as np
        from PIL import Image
        validate_case(request);validate_profile(request['profile'],'anima_v7',request['task'])
        args={**request['profile'],'seed':request['seed'],'progress_callback':progress,'interrupt_callback':cancel}
        preprocessed=[]
        if request['task']=='t2i':output=self.runtime.generate_t2i(request['prompt'],**args)
        else:
            refs=[]
            for item in request['inputs']:
                with Image.open(item['path']) as im:
                    refs.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).float().div(255).unsqueeze(0))
            # Save exactly the runtime's deterministic reference preparation, when supported.
            prepare=getattr(self.runtime,'_preprocess_reference',None)
            if prepare:
                for i,ref in enumerate(refs):
                    pil,_=prepare(ref,name=f'reference_image_{i+1}',
                        preprocess_mode='match_output_edit' if request['semantic_mode']=='edit' else 'independent_reference',
                        reference_max_area=args['reference_max_area'],output_width=args['width'],output_height=args['height'])
                    preprocessed.append((request['inputs'][i]['id'],pil))
            output=self.runtime.generate(refs,list(range(len(refs))),request['prompt'],
                    task_mode=request['semantic_mode'],
                    aligned_source_slot_id=0 if request['source_input_id'] is not None else None,**args)
        pixels=(output[0].detach().float().cpu().clamp(0,1)*255).round().to(torch.uint8).numpy()
        return Image.fromarray(pixels), {'effective_profile':request['profile'],'dtype':'bfloat16','attention':'torch',
                                        'geometry':request.get('geometry'),
                                        'preprocessed_images':preprocessed}

    def unload(self):
        if hasattr(self,'runtime'):self.runtime.close();del self.runtime
        gc.collect()

class Krea2EditBackend:
    def load(self, model, device='cuda:0'):
        raise NotImplementedError('BACKEND_NOT_IMPLEMENTED: Krea2 Edit runtime is not connected')
    def generate(self, request, progress, cancel):
        validate_case(request)
        raise NotImplementedError('BACKEND_NOT_IMPLEMENTED')
    def unload(self):pass

class Krea2OfficialBackend:
    """Official model and sampler, with CPU/GPU movement outside their mathematics."""
    def load(self, model, device='cuda:0'):
        import torch
        from safetensors.torch import load_file
        from diffusers import AutoencoderKLQwenImage
        self.torch=torch;self.device=device;self.model=model
        sys.path.insert(0,str(Path(model['runtime_dir']).resolve()))
        from inference import single_mmdit_large_wide
        from encoder import Qwen3VLConditioner
        from mmdit import SingleStreamDiT
        from autoencoder import QwenAutoencoder
        from sampling import sample
        self.sample=sample
        with torch.device('meta'):self.dit=SingleStreamDiT(single_mmdit_large_wide)
        self.dit.load_state_dict(load_file(model['checkpoint']),strict=True,assign=True)
        self.dit=self.dit.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        self.encoder=Qwen3VLConditioner(model['text_encoder']).to(dtype=torch.bfloat16).eval().requires_grad_(False)
        # The official wrapper hardcodes an online repo. Initialize the same class
        # from the explicitly locked local Diffusers VAE instead (decode unchanged).
        self.ae=QwenAutoencoder.__new__(QwenAutoencoder);torch.nn.Module.__init__(self.ae)
        self.ae.ae=AutoencoderKLQwenImage.from_pretrained(model['vae'],local_files_only=True,torch_dtype=torch.bfloat16)
        self.ae.compression=8;self.ae.channels=16
        self.ae.register_buffer('latents_mean',torch.tensor(self.ae.ae.latents_mean).view(1,-1,1,1,1))
        self.ae.register_buffer('latents_std',torch.tensor(self.ae.ae.latents_std).view(1,-1,1,1,1))
        self.ae=self.ae.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    def generate(self, request, progress, cancel):
        validate_case(request);validate_profile(request['profile'],'krea2_official',request['task'])
        if request['task']!='t2i':raise NotImplementedError('Krea2 Edit unavailable')
        b=self;torch=self.torch;profile=dict(request['profile']);negative=profile.pop('negative_prompt','')
        calls=0;branches=2 if profile['guidance']>0 else 1;started=False
        class Encoder:
            def __call__(self,text):
                cancel();b.dit.to('cpu');b.ae.to('cpu');b.encoder.to(b.device)
                return b.encoder(text)
        class Model:
            config=b.dit.config
            def __call__(self,**kwargs):
                nonlocal calls,started
                cancel()
                if not started:
                    b.encoder.to('cpu');torch.cuda.empty_cache();b.dit.to(b.device);started=True
                value=b.dit(**kwargs);calls+=1
                if calls%branches==0:progress(calls//branches,profile['steps'])
                return value
        class AE:
            compression=8;channels=16
            def decode(self,x):
                cancel();b.dit.to('cpu');torch.cuda.empty_cache();b.ae.to(b.device)
                return b.ae.decode(x)
        with torch.inference_mode():
            images=self.sample(Model(),AE(),Encoder(),[request['prompt']],negative_prompts=[negative],
                        seed=request['seed'],device=self.device,dtype=torch.bfloat16,**profile)
        return images[0], {'effective_profile':request['profile'],'guidance_formula':'cond + guidance * (cond - uncond)',
                           'implementation':'official sampling.sample','offload':'sequential CPU/GPU','dtype':'bfloat16'}

    def unload(self):
        for k in ['dit','encoder','ae']:
            if hasattr(self,k):delattr(self,k)
        gc.collect()

def backend_for(name):
    return {'anima_v7':AnimaV7Backend,'krea2_official':Krea2OfficialBackend,'krea2_edit':Krea2EditBackend}[name]()
