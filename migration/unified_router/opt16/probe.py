import argparse,json,statistics,time
from types import SimpleNamespace
import torch
from .setup import ROOT,CAPTURE,BASE
from .grouping import Tree,fit_tree,order,gather
from .pipeline import GroupedRouter
from .history import HistoryBuilder
from .route import route
from ..common import atomic_json
from ..opt5.probe import capture,timing
from ..opt9.segmented_attention import attention
from ..opt14.network import Top3Route
from ..learning import ActionHead

def nmse(a,b):return float((a.float()-b.float()).square().sum()/b.float().square().sum().clamp_min(1e-20))

def measured(fn,reps=30):
    graph,_=capture(fn);timing(graph,reps)
    return statistics.median(timing(graph,reps) for _ in range(3))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--file',default='layer24_chunk3.pt');p.add_argument('--depth',type=int,default=6)
    p.add_argument('--variants',default='Identity1,Both1,Both25,Both35,Both50');p.add_argument('--tag',default='results');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/a.file.removesuffix('.pt');root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'status.json',dict(status='calibrating',file=a.file))
    d=torch.load(CAPTURE/a.file,map_location='cuda',weights_only=False)
    q,k,v,m=[d[key] for key in ['q','k','v','mask']];h,n,_=q.shape;nh=d['sketch'].history_tokens
    layer=int(a.file[5:7]);params=torch.load(BASE/f'action_basis/layer{layer:02d}.pt',map_location='cuda',weights_only=True)['parameters']
    # Layer24 chunk1 is the only calibration event used for all its other chunks.
    calfile='layer24_chunk1.pt' if layer==24 else a.file
    treefile=ROOT/f'layer{layer:02d}_depth{a.depth}_trees.pt'
    if treefile.exists():
        td=torch.load(treefile,map_location='cuda',weights_only=True)
        assert td['calibration_file']==calfile
        qt=Tree(td['q_weight'],td['q_threshold'],a.depth);kt=Tree(td['k_weight'],td['k_threshold'],a.depth)
    else:
        cd=torch.load(CAPTURE/calfile,map_location='cuda',weights_only=False)
        qt=fit_tree(cd['q'][:,::2],cd['k'][:,::2],a.depth)
        kt=fit_tree(cd['k'][:,::2],cd['q'][:,::2],a.depth)
        torch.save(dict(q_weight=qt.weight.cpu(),q_threshold=qt.threshold.cpu(),k_weight=kt.weight.cpu(),k_threshold=kt.threshold.cpu(),depth=a.depth,calibration_file=calfile),treefile)
        del cd
    pipeline=GroupedRouter(params,qt,kt);builder=HistoryBuilder(pipeline)
    ydense=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
    base=lambda:attention(q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],m,pack_v=True,fuse_pack=True)
    ybase=base()
    head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval()
    oldroute=Top3Route(batch=n);oldstate=oldroute.begin(d['prepared'],d['sketch'])
    router=SimpleNamespace(kind='K32-Set',beta=d['beta'],heads={f'{layer}_K32-Set':head})
    step=q.new_tensor(3.,dtype=torch.float32)
    def baseline():
        mask=oldroute.run(layer,q,d['sketch'],oldstate,step,router)
        return attention(q,k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],mask,pack_v=True,fuse_pack=True)
    current_baseline=baseline()
    reference=dict(captured_attention_nmse=nmse(ybase,ydense),opt14_attention_nmse=nmse(current_baseline,ydense),dense_ms=measured(lambda:torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]),captured_attention_ms=measured(base),opt14_route_attention_ms=measured(baseline))
    atomic_json(root/'baseline.json',reference);print('baseline',reference,flush=True)
    rows=[]
    for variant in a.variants.split(','):
        atomic_json(root/'status.json',dict(status='running',variant=variant,completed=len(rows)))
        gq=variant.startswith(('Both','Q'));gk=variant.startswith(('Both','K'))
        suffix=''.join(c for c in variant if c.isdigit());fraction=1. if suffix=='1' else int(suffix)/100.
        assert .15<=fraction<=1.
        pipeline.candidate_fraction=fraction
        kv=pipeline.pack_kv(k,v,nh,gk)
        history=builder.history(k[:,:nh],v[:,:nh],gk)
        warm_kv=builder.current(history,k[:,nh:],v[:,nh:],gk)
        assert torch.equal(kv.k,warm_kv.k) and torch.equal(kv.v,warm_kv.v)
        assert torch.allclose(kv.state['key'],warm_kv.state['key'],atol=.02,rtol=.002)
        out,aux=pipeline.run(q,kv,gq,True)
        assert torch.isfinite(out).all()
        ids=aux['q_order'];assert torch.equal(ids.sort(-1).values,torch.arange(n,device=q.device).expand(h,n))
        ki=kv.order
        assert torch.equal(ki.sort(-1).values,torch.arange(k.shape[1],device=k.device).expand(h,k.shape[1]))
        assert torch.equal(ki//4680,(torch.arange(k.shape[1],device=k.device)//4680)[None].expand_as(ki))
        mask=aux['mask'];cost=d['sketch'].costs
        pad=torch.nn.functional.pad(mask,(0,0,0,(-n)%32));union=pad.reshape(h,-1,32,mask.shape[-1]).any(2)
        oracle=route(aux['packed_q'],kv.state,.15,1.)
        recall=float((oracle&mask).sum()/oracle.sum())
        fn=lambda:pipeline.run(q,kv,gq)
        cold=lambda:pipeline(q,k,v,nh,gq,gk)
        warm=lambda:builder.run(q,history,k[:,nh:],v[:,nh:],gq,gk)
        warm_y=warm();assert nmse(warm_y,out)<1e-4,nmse(warm_y,out)
        bg,_=capture(baseline);fg,_=capture(fn);cg,_=capture(cold);wg,_=capture(warm)
        timing(bg,50);timing(fg,50);timing(cg,50);timing(wg,50);pairs=[]
        for i in range(4):
            if i%2:wt=timing(wg,30);ct=timing(cg,30);ft=timing(fg,30);bt=timing(bg,30)
            else:bt=timing(bg,30);ft=timing(fg,30);ct=timing(cg,30);wt=timing(wg,30)
            pairs.append([bt,ft,ct,wt])
        row=dict(variant=variant,candidate_fraction=fraction,group_q=gq,group_k=gk,attention_nmse_dense=nmse(out,ydense),attention_nmse_opt14=nmse(out,current_baseline),keep_tokens=float((mask*cost).sum()/(h*n*k.shape[1])),executed_fraction=float(union.sum()*32*64/(h*n*k.shape[1])),fine_mask_recall_vs_all_candidates=recall,opt14_route_attention_ms=statistics.median(x[0]for x in pairs),frozen_all_kv_ms=statistics.median(x[1]for x in pairs),cold_all_kv_ms=statistics.median(x[2]for x in pairs),samples=pairs,
                 components_ms=dict(q_assignment_and_order=measured(lambda:order(q,qt)) if gq else 0,kv_grouping_prototypes=measured(lambda:pipeline.pack_kv(k,v,nh,gk)),route_only=measured(lambda:route(aux['packed_q'],kv.state,.15,fraction)),attention_only=measured(lambda:attention(aux['packed_q'],kv.k[:,:nh],kv.v[:,:nh],kv.k[:,nh:],kv.v[:,nh:],mask,pack_v=True,fuse_pack=True))),
                 memory=dict(old_weight_elements=h*n*sum(part.count.shape[1]*part.count.shape[2] for part in d['sketch'].parts),new_fine_score_elements=aux['scores'].numel(),boolean_mask_elements=mask.numel()))
        row['history_cached_current_rebuilt_ms']=statistics.median(x[3]for x in pairs)
        row['cached_vs_uncached_nmse']=nmse(warm_y,out)
        rows.append(row);atomic_json(root/(a.tag+'.json'),dict(file=a.file,calibration_file=calfile,calibration_same_event=calfile==a.file,depth=a.depth,baseline=reference,rows=rows));print(row,flush=True)
        del bg,fg,cg,wg
    atomic_json(root/'status.json',dict(status='completed',variants=len(rows)))

if __name__=='__main__':main()
