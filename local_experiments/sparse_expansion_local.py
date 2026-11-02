"""Same-input route/conditional-V diagnostics; no QKV saved; resumable by case."""
import gc,json,math,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .route_vector_runtime import VectorRuntime
from .sparse_expansion_runtime import ExpandedRouter
from .high_sparse_learn_kernels import pool
from .rollout_suite import generate_chunk
from .improve_accept import Telemetry
from .learned_cache.io import atomic,RunLock,disk_guard,digest
ROOT=Path('artifacts/sparse_expansion_20261031');MODEL='artifacts/full_route_grad_20261031/epoch1.pt'
CONFIGS=[(m,s) for m in [4,8,16] for s in [1,2,4]]

def code_options(rt,l,values):
    h,p,_,d=values.shape;basis=rt.basis[l].float();center=rt.centers[l]
    group=values.reshape(h,p,4,16,d).mean(3)
    pairs=torch.einsum('hprd,hdk->hprk',group-center[:,None,None],basis).to(torch.bfloat16).float()
    base=rt.active_codes.float();haar=torch.tensor([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],device=values.device,dtype=torch.float32)
    residual=torch.einsum('kr,hprd->hpkd',haar,pairs)/4
    choice=residual.square().sum(-1).argmax(-1);direction=residual.gather(2,choice[...,None,None].expand(-1,-1,1,8)).squeeze(2).to(torch.bfloat16).float()
    signs=haar[choice];compressed=base[:,:,None]+signs[...,None]*direction[:,:,None]
    return {'single':base[:,:,None].expand(-1,-1,4,-1),'paired4':pairs,'haar1':compressed},dict(paired4_extra_history_bytes=29*12*338*3*8*2,haar1_extra_history_bytes=29*12*338*(8*2+1))

@torch.no_grad()
def local_event(rt,q,k,v,luts):
    h,d=12,128;p=rt.active_pk.shape[1];tiles=torch.tensor([17,53],device='cuda');idx=(tiles[:,None]*64+torch.arange(64,device='cuda')).flatten();query=q[0].transpose(0,1).index_select(1,idx).float()
    keys=k[0].transpose(0,1).float().reshape(h,p,64,d);vals=v[0].transpose(0,1).float().reshape_as(keys)
    logits=query@keys.flatten(1,2).transpose(1,2)/math.sqrt(d);lse=logits.logsumexp(-1)
    weights=(logits-lse[...,None]).exp().reshape(h,128,p,64);pagez=weights.sum(-1).reshape(h,2,64,p)
    pagen=torch.einsum('hxpq,hpqd->hxpd',weights,vals).reshape(h,2,64,p,d);target=pagen.sum(-2);del logits,weights
    az=query@rt.active_pk.flatten(1,2).transpose(1,2)/math.sqrt(d)
    ar=(az+math.log(16)-lse[...,None]).exp().reshape(h,2,64,p,4);codes,cost=code_options(rt,rt.active_layer,vals);del keys,vals,az
    basis=rt.basis[rt.active_layer].float();center=rt.centers[rt.active_layer];result=[]
    def evaluate(lut,cname):
        mask=torch.zeros(h,2,p,device='cuda',dtype=torch.bool).scatter_(-1,lut[0,:,tiles].long(),True)
        exactn=(pagen*mask[:,:,None,:,None]).sum(-2);exactz=(pagez*mask[:,:,None]).sum(-1)
        a=ar*(~mask)[:,:,None,:,None];z=a.sum((-2,-1));c=torch.einsum('htqpr,hprd->htqd',a,codes[cname])
        n=torch.einsum('htqr,hdr->htqd',c,basis)+z[...,None]*center[:,None,None]
        y=(exactn+n)/(exactz+z).clamp_min(1e-12)[...,None]
        error=(y-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-8)
        return dict(nmse=float(error.mean()),p95=float(torch.quantile(error.flatten(),.95)),retained_mass=float(exactz.mean()))
    for name,lut in luts.items():result.append(dict(route=name,compensation='single',**evaluate(lut,'single')))
    for name in ['paired4','haar1']:result.append(dict(route='p4_s1',compensation=name,**evaluate(luts['p4_s1'],name)))
    return result,cost

class Observe:
    def __init__(self,rt,base,callback):self.rt=rt;self.base=base;self.callback=callback
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            e=self.rt.engine.current
            if e['layer'] in [4,14,24] and e['chunk'] in [2,5] and e['step'] in [0,3]:self.callback(self.rt,*args[:3],args[5])
            return self.base[grid](*args,**kwargs)
        return call

@torch.no_grad()
def main():
    lock=RunLock(ROOT/'local');stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'checks.json').read_text())['passed']
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=['sparse_expansion_runtime.py','sparse_expansion_local.py'];identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},model=digest(MODEL),configs=CONFIGS,cases=cases('val'),tiles=[17,53],all64_queries=True,source='frozen p4_s1 optimized student; no teacher used in routing; fixed 41/410 budget')
        ip=ROOT/'local/identity.json'
        if ip.exists():assert json.loads(ip.read_text())==json.loads(json.dumps(identity))
        else:atomic(ip,identity)
        engine=Engine()
        for case in cases('val'):
            dest=ROOT/'local'/(case['id']+'.json')
            if dest.exists():continue
            if stop[0]:atomic(ROOT/'local/status.json',dict(status='interrupted'));return
            disk_guard(ROOT,50,1);atomic(ROOT/'local/status.json',dict(status='running',case=case['id'],pid=os.getpid()))
            rt=VectorRuntime(engine,MODEL);routers={(m,s):ExpandedRouter(m,s) for m,s in CONFIGS}
            for r in routers.values():r.bind(rt)
            rows=[];timings=[];tele=Telemetry();tele.start()
            def observe(rt,q,k,v,lut):
                e=dict(engine.current);l=e['layer'];p=rt.n[l];luts={}
                for (m,s),r in routers.items():
                    r.current_var.copy_(rt.router.current_var);new,pk,code=r.route(l,q,p);luts[f'p{m}_s{s}']=new.clone()
                    assert torch.equal(pk,rt.active_pk) and torch.equal(code,rt.active_codes)
                    if (m,s)==(4,1):assert torch.equal(new,lut),'Baseline expansion differs'
                    assert (new[0]==(torch.arange(72,device='cuda')+p)[None,:,None]).any(-1).all()
                results,cost=local_event(rt,q,k,v,luts);rows.append(dict(event=e,metrics=results,cost=cost))
                if l==14 and e['chunk']==5 and e['step']==0:
                    for (m,s),r in routers.items():
                        def fn():
                            pool(engine.cache[l],rt.basis[l],rt.centers[l],rt.uk[l],rt.uv[l],r.code,r.current_var,offset=p*64,weighted=False)
                            return r.route(l,q,p)
                        for _ in range(3):fn()
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();a.record();t=time.perf_counter()
                        for _ in range(20):fn()
                        b.record();b.synchronize();timings.append(dict(probes=m,steps=s,gpu_ms=a.elapsed_time(b)/20,wall_ms=(time.perf_counter()-t)*50))
                print(json.dumps(dict(case=case['id'],event=e,rows=len(rows))),flush=True)
            rt.kernel=Observe(rt,rt.kernel,observe)
            try:
                emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                assert len(rows)==12;load=tele.finish();atomic(dest,dict(case=case,rows=rows,timings=timings,telemetry=load));print('completed',case['id'],flush=True)
            finally:
                if not tele.stop.is_set():tele.finish()
                rt.close();del rt,routers;gc.collect();torch.cuda.empty_cache()
        atomic(ROOT/'local/status.json',dict(status='complete',cases=4,events=48))
    except Exception as exc:atomic(ROOT/'local/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
