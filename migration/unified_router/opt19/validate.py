"""All96 Val events; full-Q execution, original128 supervised positions.

Timing: two fixed prompts x three history lengths, step3; includes preparation.
No Test data, no generated-video claims, no changes to previous checkpoints.
"""
import argparse,gc,json,math,statistics,time,traceback
import torch
from ..common import config,atomic_json
from ..opt18.setup import ROOT as PARENT
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt17.route import _select
from ..opt16.grouping import order,gather,restore
from ..opt9.segmented_attention import attention
from ..opt5.probe import capture,timing
from .cache import Cache
from . import tile
from .probe import ROOT,nmse
import triton as tr

CONFIGS=[dict(name='native32',group=32,mode='native'),dict(name='native16',group=16,mode='native')]+[
    dict(name=f'{mode}{g}_b{round(b*100)}',group=g,mode=mode,budget=b)
    for mode,g,b in [('shared',g,b) for g in [16,32] for b in [.15,.2,.25]]+
    [('cap',32,.2),('cap',32,.25)]]

def native_mask(fine,candidates,b):
    h,n,c=fine.shape;g=candidates.shape[1]
    mask=torch.zeros((h,n,b),device=fine.device,dtype=torch.bool)
    _select[(h*n,)](fine,candidates,mask,n,b,c,g,math.ceil(b*.15),tr.next_power_of_2(c),num_warps=4)
    return mask

def mask_for(fine,candidates,b,cfg):
    if cfg['mode']=='native':return native_mask(fine,candidates,b)
    ids=tile.select(fine,candidates,b,cfg['group'],cfg['budget'])
    mask=tile.expand(ids,fine.shape[1],b,cfg['group'])
    if cfg['mode']=='cap':
        inter=mask&native_mask(fine,candidates,b)
        mask=torch.where(inter.any(-1,keepdim=True),inter,mask)
    return mask

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=256
    folder=ROOT/f'validation/layer{a.layer:02d}_{a.variant}';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json';atomic_json(status,dict(status='loading',completed=0,total=96))
    _,m=config(PARENT);events=Events(m,a.layer)
    model=JointGroupedRouter(m,a.layer,a.variant).cuda().eval()
    model.load_state_dict(torch.load(PARENT/f'training/layer{a.layer:02d}/{a.variant}/best.pt',map_location='cuda',weights_only=True)['model'])
    frac,reps=model.options['candidate_fraction'],model.options['representatives']
    prompts=sorted({x['row']['prompt_id'] for x in events.val})[:2]
    rows={cfg['name']:[] for cfg in CONFIGS};timings=[];oldrows=[]
    for index,item in enumerate(events.val):
        q,raw,truth,targets=events.gpu(item)
        old,diag=model.deployed(q,raw,targets)
        oldrows.append(dict(sample_id=item['row']['sample_id'],nmse=nmse(old,truth),keep=float(diag['keep']),executed=float(diag['executed'])))
        cache=Cache(model.pipeline,raw,'fp16');kv=cache.current(raw['kc'],raw['vc'])
        ix=order(q,model.tree('q'));qq=gather(q,ix);fine,candidates=tile.scores(qq,kv.state,frac,reps)
        b=kv.state['key'].shape[1];nk=kv.k.shape[1];nh=kv.history_tokens;n=q.shape[1];h=q.shape[0]
        cost=torch.full((b,),64.,device=q.device);cost[73::74]=8.
        def apply(kv,ix,qq,mask,cfg):
            y=attention(qq,kv.k[:,:nh],kv.v[:,:nh],kv.k[:,nh:],kv.v[:,nh:],mask,group=cfg['group'],pack_v=True,fuse_pack=True)
            return restore(y,ix)
        def full(cfg):
            kk=cache.current(raw['kc'],raw['vc']);ii=order(q,model.tree('q'));qr=gather(q,ii)
            f,ci=tile.scores(qr,kk.state,frac,reps);mask=mask_for(f,ci,b,cfg)
            return apply(kk,ii,qr,mask,cfg)
        def dense():return torch.nn.functional.scaled_dot_product_attention(q[None],torch.cat((raw['kh'],raw['kc']),1)[None],torch.cat((raw['vh'],raw['vc']),1)[None])[0]
        need_timing=item['row']['prompt_id'] in prompts and item['row']['step']==3 and item['row']['generation_seed']==0
        for cfg in CONFIGS:
            mask=mask_for(fine,candidates,b,cfg);g=cfg['group']
            out=apply(kv,ix,qq,mask,cfg)[:,targets].float()
            assert torch.isfinite(out).all(),(a.layer,a.variant,index,cfg)
            executed=torch.nn.functional.pad(mask,(0,0,0,(-n)%g)).reshape(h,-1,g,b).any(2).sum()*g*64/(h*n*nk)
            row=dict(sample_id=item['row']['sample_id'],nmse=nmse(out,truth),nmse_vs_native=nmse(out,old),keep=float((mask*cost).sum()/(h*n*nk)),executed=float(executed))
            rows[cfg['name']].append(row)
            if need_timing:
                ng,yy=capture(lambda:full(cfg));dg,_=capture(dense)
                ng.replay();torch.cuda.synchronize()
                assert torch.equal(yy[:,targets].float(),out)
                timing(ng,80);timing(dg,80);samples=[]
                for j in range(4):
                    if j%2:d=timing(dg,30);s=timing(ng,30)
                    else:s=timing(ng,30);d=timing(dg,30)
                    samples.append([s,d])
                timings.append(dict(config=cfg['name'],sample_id=item['row']['sample_id'],history_tokens=nh,warm_ms=statistics.median(x[0] for x in samples),dense_ms=statistics.median(x[1] for x in samples),samples=samples))
                del ng,dg,yy
        atomic_json(status,dict(status='validating',completed=index+1,total=96,sample=item['row']['sample_id']))
        atomic_json(folder/'partial.json',dict(rows=rows,timings=timings,native_rows=oldrows))
        if index%12==11:print(a.layer,a.variant,index+1,flush=True)
        del cache,kv,diag,qq,fine,candidates
    summary=[]
    for cfg in CONFIGS:
        rr=rows[cfg['name']];tt=[x for x in timings if x['config']==cfg['name']]
        assert len(rr)==96 and len(tt)==6
        summary.append(dict(**cfg,**{k:statistics.mean(x[k] for x in rr) for k in ['nmse','nmse_vs_native','keep','executed']},
            warm_ms=statistics.mean(x['warm_ms'] for x in tt),dense_ms=statistics.mean(x['dense_ms'] for x in tt)))
    result=dict(layer=a.layer,variant=a.variant,configs=summary,rows=rows,timings=timings,native_rows=oldrows,scope='Val96; fullQ4680; accuracy original128 positions; timing6blocks; no video evaluation')
    atomic_json(folder/'result.json',result);atomic_json(status,dict(status='completed',completed=96,total=96))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
