"""Compact broad-coverage replay protocol; no full QKV tensors on disk."""
import json
from pathlib import Path
import shutil
import torch
from .learned_cache.io import atomic,digest,RunLock

ROOT=Path('artifacts/broad_coverage_20261030')
CHUNKS=(5,20,40)
PROMPTS={
 'train':[
 ('city_walk','A pedestrian in a yellow coat walks along a wet city sidewalk at dusk. The camera tracks smoothly beside them, with reflections and moving traffic in the background.'),
 ('forest_run','A runner wearing a blue jacket jogs along a winding forest trail. Sunlight passes through leaves as the camera follows from behind.'),
 ('reef_fish','A school of colorful fish swims around a coral reef in clear blue water. The underwater camera moves slowly forward through the scene.'),
 ('robot_work','A small industrial robot arm sorts colored wooden blocks on a workbench. A steady camera shows repeated precise movements and changing block positions.'),
 ('flower_wind','Red and white flowers sway in a garden breeze, with bees moving between blossoms. A close camera slowly pans across the flower bed.'),
 ('fabric_dance','A dancer in a long flowing green dress turns in a sunlit studio. The camera tracks the movement while fabric folds change continuously.'),
 ('waterfall','Water cascades over dark rocks in a mountain stream. The camera slowly moves upstream, revealing foam, ripples, and mist.'),
 ('toy_train','A red toy train circles a wooden track through a miniature town. The camera stays low beside the tracks as the train passes buildings and trees.')],
 'val':[
 ('horse_field','A brown horse trots across a grassy field beside a white fence. A side tracking shot follows the horse under soft afternoon sunlight.'),
 ('kitchen_cook','A cook chops vegetables on a kitchen counter and places them in a steaming pan. A steady medium shot shows the hands, utensils, and rising steam.'),
 ('snow_ski','A skier in an orange jacket carves turns down a snowy slope. The camera follows smoothly, with snow spraying and distant mountains visible.'),
 ('group_dance','Three people perform a synchronized dance in a bright courtyard. A wide camera shot slowly moves sideways while all three remain visible.')],
 'test':[
 ('tennis_match','Two tennis players exchange shots on an outdoor court. A stable wide shot shows the players moving, the ball crossing the net, and shifting shadows.'),
 ('bird_flight','A white seabird flies along a rocky coastline over turquoise waves. The camera tracks the bird from the side as its wings flap and the landscape moves behind it.'),
 ('traffic_crossing','Cars, bicycles, and pedestrians move through a busy urban intersection in daylight. A fixed elevated camera shows traffic flowing in several directions.'),
 ('fireworks_night','Bright fireworks expand above a city river at night. Reflections ripple on the water while boats drift slowly below the illuminated skyline.'),
 ('portrait_turn','A woman with curly hair turns her head and smiles while standing near a sunny window. A close steady camera captures natural facial movement and shifting hair.'),
 ('macro_insect','A tiny beetle walks along a green leaf covered with dew drops. A macro camera follows its movement as its legs and antennae move and the leaf sways.')]
}

def cases(split=None):
    rows=[]
    for part,items in PROMPTS.items():
        if split is not None and part!=split:continue
        for name,prompt in items:
            for seed in ([202,303] if part=='test' else [0] if part=='train' else [101]):
                rows.append(dict(id=f'{part}_{name}_s{seed}',split=part,name=name,prompt=prompt,seed=seed))
    return rows


def guard():
    ROOT.mkdir(parents=True,exist_ok=True);free=shutil.disk_usage(ROOT).free
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if used>4*2**30 or free<50*2**30:raise RuntimeError(f'Disk guard used={used},free={free}')
    return dict(used_mib=used/2**20,free_gib=free/2**30)


def manifest():
    ROOT.mkdir(parents=True,exist_ok=True)
    value=dict(cases=cases(),frames=480,fps=16,chunks=41,height=512,width=768,
        sample_chunks=list(CHUNKS),layers=list(range(30)),heads=list(range(12)),query_samples=32,
        epochs=6,updates_per_layer_per_epoch=96,prototype_count=4,history_chunks=5,
        checkpoint=digest('checkpoints/self_forcing_dmd.pt'),code={str(p):digest(p) for p in Path('local_experiments').glob('broad_*.py')})
    path=ROOT/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in value:
            if key!='code':assert old[key]==value[key],key
    else:atomic(path,value)
    return value


def noise(seed,chunk,phase):
    rng=torch.Generator(device='cuda').manual_seed(1000003*seed+1009*chunk+phase+12345)
    return torch.randn((1,3,16,64,96),device='cuda',dtype=torch.bfloat16,generator=rng)


class Engine:
    def __init__(self):
        import sys
        from streaming_sparse.v08.backbone import load_teacher,native_steps,native_cache
        self.wrapper=load_teacher();self.steps,_=native_steps(self.wrapper);self.wrapper.model.num_frame_per_block=3
        self.wrapper.model.local_attn_size=18
        for b in self.wrapper.model.blocks:
            b.self_attn.local_attn_size=18;b.self_attn.max_attention_size=27648;b.self_attn.sink_size=0
        def dense(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),**kw):
            assert q_lens is None and k_lens is None and window_size==(-1,-1)
            if q_scale is not None:q=q*q_scale
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                dropout_p=dropout_p,is_causal=causal,scale=softmax_scale).transpose(1,2)
        for name,module in list(sys.modules.items()):
            if name.startswith('wan.modules.'):
                for attr in ['attention','flash_attention']:
                    if callable(getattr(module,attr,None)):setattr(module,attr,dense)
        import wan.modules.causal_model as causal
        self.dense=dense;self.current={};self.callback=None
        self.handles=[]
        for i,b in enumerate(self.wrapper.model.blocks):
            def hook(module,args,index=i):self.current['layer']=index
            self.handles.append(b.self_attn.register_forward_pre_hook(hook))
        def observed(q,k,v,*args,**kwargs):
            if self.callback is not None:self.callback(self.current,q,k,v)
            return dense(q,k,v,*args,**kwargs)
        causal.attention=observed
        self.cache=native_cache(self.wrapper,'cuda',torch.bfloat16,tokens=27648)
        self.cross=[dict(k=torch.zeros(1,512,12,128,device='cuda',dtype=torch.bfloat16),v=torch.zeros(1,512,12,128,device='cuda',dtype=torch.bfloat16),is_init=False) for _ in range(30)]
        self.reset()

    def reset(self):
        for c in self.cache:c['global_end_index'].zero_();c['local_end_index'].zero_()
        for c in self.cross:c['is_init']=False

    @torch.no_grad()
    def forward(self,x,embeds,chunk,step):
        timestep=0. if step==4 else self.steps[step]
        self.current.update(chunk=chunk,step=step)
        return self.wrapper(noisy_image_or_video=x,conditional_dict={'prompt_embeds':embeds},
            timestep=torch.full((1,3),timestep,device='cuda',dtype=torch.float32),kv_cache=self.cache,
            crossattn_cache=self.cross,current_start=chunk*4608)[1]

    @torch.no_grad()
    def replay(self,case,callback,start_chunk=0,stop_chunk=40):
        folder=ROOT/'trajectories'/case['id'];emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
        self.reset();self.callback=None
        for chunk in range(stop_chunk+1):
            saved=torch.load(folder/f'chunk{chunk:02d}.pt',weights_only=True)
            if chunk in CHUNKS and chunk>=start_chunk:
                self.callback=callback
                for step in range(4):self.forward(saved['noisy'][step].cuda(),emb,chunk,step)
                self.callback=None
            self.forward(saved['clean'].cuda(),emb,chunk,4)
