import importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
here=Path(__file__).resolve().parent
repo=here.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from phase import set_phase
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.deployment import configure
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.backbone_ops import install

def variant(name,upstream,old,new):
    src=Path(upstream.__file__).read_text()
    assert old in src
    dest=here/(name+'.py')
    dest.write_text(src.replace(old,new))
    spec=importlib.util.spec_from_file_location(upstream.__package__+'.'+name,dest)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

@torch.inference_mode()
def run(method,n):
    torch.manual_seed(701)
    torch.set_num_threads(3)
    cfg=config('base',local_assets=repo);cfg.mode='dense_native'
    configure(cfg);engine=Engine();install(engine,cfg)
    for block in engine.wrapper.model.blocks:block.self_attn.sink_size=1
    import wan.modules.causal_model as causal
    dense=causal.attention
    if method=='pisa':
        sys.path.insert(0,str(repo/'external_baselines/PISA'))
        import piecewise_attn.kernels.piecewise_sparse_attn_hyd as upstream
        fn=variant('pisa_anchor',upstream,'top_k = max(1, int(density * NT))','score[..., :1536//block_size] = float("inf")\n    top_k = max(1536//block_size, int(density * NT))').piecewise_sparse_attention
    if method=='solattn':
        sys.path.insert(0,str(repo/'external_baselines/Sana/techniques/sparse_backends'))
        import sol_attn.triton_ref.fwd as upstream
        mod=variant('sol_anchor',upstream,'exact = exact & valid','exact = (exact | (block_indices < 24)) & valid')
        fn=mod.sol_attn
    if method=='svgear':
        from ear_anchor import EAR
        ear=EAR()
    seen=[0,0]
    def forward(q,k,v,*args,**kwargs):
        c,l,s=(engine.current[key] for key in ('chunk','layer','step'))
        if c<2 or s==4 or l==0:
            seen[0]+=1
            return dense(q,k,v,*args,**kwargs)
        seen[1]+=1
        if method=='svgear':return ear(q,k,v,l)
        padded=torch.zeros_like(k);padded[:,-q.size(1):]=q
        if method=='pisa':
            out=fn(padded.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),density=.1).transpose(1,2)
        else:out=fn(padded.contiguous(),k.contiguous(),v.contiguous(),tau=1.,sink_start=k.size(1)-q.size(1),sink_tokens=q.size(1))
        return out[:,-q.size(1):].contiguous()
    causal.attention=forward
    rt=SimpleNamespace(engine=engine,stores={})
    emb=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    times=[]
    for _ in range(3):
        engine.reset();seen[:]=[0,0];torch.manual_seed(701)
        if method=='svgear':ear.reset()
        parts=[];torch.cuda.synchronize();t=time.monotonic()
        for c in range(n):
            if c==6:canonical=torch.cat([x['k'][:,:1536] for x in engine.cache])
            if c>=6:set_phase(rt,canonical,3*c-15)
            parts.append(generate_chunk(engine,emb,701,c))
        z=torch.cat(parts,1);torch.cuda.synchronize();times.append(time.monotonic()-t)
        assert z.isfinite().all() and seen[1]==(n-2)*116
        print(method,n,times[-1],seen,flush=True)
    torch.save(z.cpu(),here/f'{method}_{n}.pt')
    (here/f'{method}_{n}.json').write_text(json.dumps(dict(times=times,calls=seen)))
run(sys.argv[1],int(sys.argv[2]))
