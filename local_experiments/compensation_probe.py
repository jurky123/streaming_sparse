"""Fixed-mask full-layer tail decomposition and streaming fitted rank-8 compensation.
No videos/large QKV dumps. Each replay chunk is an atomic resumable unit.
"""
import argparse,json,math,os,signal
from pathlib import Path
import torch
from .broad_common import Engine,cases,CHUNKS,ROOT as DATA
from .improve_models import Events,scores
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/compensation_probe_20261030')
LAYERS=list(range(1,30))
STOP=False

def stop(*_):
    global STOP
    STOP=True

def proto(k,w):
    groups=k.reshape(12,338,4,16,128);x=groups-groups.mean(3,keepdim=True)
    a=(2*torch.tanh(torch.einsum('hprtd,hrd->hprt',x,w)/2)).softmax(-1)
    return torch.einsum('hprt,hprtd->hprd',a,groups)

def state(e,pk,alpha):
    z=scores((pk,None,torch.full(pk.shape[:3],16.,device=pk.device)),e['route'],e['var'],alpha)
    selected=z.topk(72,-1).indices
    mask=torch.zeros(12,8,339,device=z.device,dtype=torch.bool).scatter_(-1,selected,True);mask[...,-1]=True
    mask=mask.repeat_interleave(4,1);tail=~mask
    assert (mask.sum(-1)==73).all() and not tail[...,-1].any()
    zs=e['z'].masked_fill(~mask,-torch.inf).logsumexp(-1)
    zt=e['z'].masked_fill(~tail,-torch.inf).logsumexp(-1)
    ys=(e['z'].masked_fill(~mask,-torch.inf).softmax(-1)[...,None]*e['values']).sum(-2)
    yt=(e['z'].masked_fill(~tail,-torch.inf).softmax(-1)[...,None]*e['values']).sum(-2)
    logits=(e['q']@pk.flatten(1,2).transpose(1,2)/math.sqrt(128)).reshape(12,32,338,4).logsumexp(-1)+math.log(16)
    vm=e['h']['v'].reshape(12,338,64,128).mean(2)
    return dict(mask=mask,tail=tail[...,:338],zs=zs,zt=zt,ys=ys,yt=yt,zp=logits,vm=vm,rho=(zt-zs).sigmoid())

def blend(s,logzt,yt):
    mix=(logzt-s['zs']).sigmoid()
    return (1-mix[...,None])*s['ys']+mix[...,None]*yt

def approx(s,zp,vm):
    logits=zp.masked_fill(~s['tail'],-torch.inf)
    return logits.logsumexp(-1),logits.softmax(-1)@vm

def empty_stats():
    return dict(n=torch.zeros(12,dtype=torch.float64),vsum=torch.zeros(12,128,dtype=torch.float64),vv=torch.zeros(12,128,128,dtype=torch.float64),xx=torch.zeros(12,2,2,dtype=torch.float64),xy=torch.zeros(12,2,dtype=torch.float64))

def accumulate(stats,e,s):
    # A shared learned V subspace fits the actual omitted conditional output,
    # weighted by rho^2, rather than generic raw-V variance.
    w=s['rho'].double().square();v=s['yt'].double()
    stats['n']+=w.sum(1).cpu();stats['vsum']+=(w[...,None]*v).sum(1).cpu()
    stats['vv']+=torch.einsum('hq,hqi,hqj->hij',w,v,v).cpu()
    # Fit scalar affine calibration of per-page log partition; teacher only in training.
    x=torch.stack((s['zp'],torch.ones_like(s['zp'])),-1).double()
    weight=e['z'].softmax(-1)[...,:338].double()*s['tail']
    y=e['z'][...,:338].double()
    stats['xx']+=torch.einsum('hqp,hqpi,hqpj->hij',weight,x,x).cpu()
    stats['xy']+=torch.einsum('hqp,hqpi,hqp->hi',weight,x,y).cpu()

@torch.no_grad()
def fit(root):
    files=sorted((root/'train').glob('*.pt'))
    total={l:empty_stats() for l in LAYERS}
    for p in files:
        for l,st in torch.load(p,weights_only=True).items():
            for key in st:total[l][key]+=st[key]
    models={}
    for l,st in total.items():
        assert (st['n']>0).all()
        center=st['vsum']/st['n'][:,None]
        cov=st['vv']/st['n'][:,None,None]-center[:,:,None]*center[:,None,:]
        eig,basis=torch.linalg.eigh((cov+cov.transpose(-1,-2))/2)
        basis=basis[:,:,-8:]
        ridge=st['xx'].diagonal(dim1=-2,dim2=-1).sum(-1)*1e-6+1e-8
        prior=torch.tensor([1.,0.],dtype=torch.float64).expand(12,2)
        ab=torch.linalg.solve(st['xx']+ridge[:,None,None]*torch.eye(2,dtype=torch.float64),st['xy']+ridge[:,None]*prior)
        ab[:,0].clamp_(.5,1.5);ab[:,1].clamp_(-4,4)
        models[l]=dict(center=center.float(),basis=basis.float(),ab=ab.float(),captured_energy=(eig[:,-8:].clamp_min(0).sum(-1)/eig.clamp_min(0).sum(-1).clamp_min(1e-20)).float())
    atomic(root/'model.pt',dict(models=models,training_sources={p.name:digest(p) for p in files}),True)

@torch.no_grad()
def evaluate(e,s,m):
    truez=e['z'][...,:338]
    mean_z,mean_v=approx(s,s['zp'],s['vm'])
    _,true_mass_mean=approx(s,truez,s['vm'])
    center=m['center'].to(e['q'].device);basis=m['basis'].to(e['q'].device)
    # Mimic stored BF16 decoder and BF16 page codes; FP32 accumulation.
    basis=basis.to(torch.bfloat16).float()
    code=((s['vm']-center[:,None])@basis).to(torch.bfloat16).float()
    lowv=center[:,None]+code@basis.transpose(1,2)
    a,b=m['ab'].to(e['q'].device).unbind(-1)
    cal=s['zp']*a[:,None,None]+b[:,None,None]
    cal_z,cal_v=approx(s,cal,s['vm']);low_z,low_v=approx(s,cal,lowv)
    rawlow_z,rawlow_v=approx(s,s['zp'],lowv)
    dynamic=s['vm'].mean(1)
    dynamic_code=((s['vm']-dynamic[:,None])@basis).to(torch.bfloat16).float()
    dynamic_v=dynamic[:,None]+dynamic_code@basis.transpose(1,2)
    dyn_z,dyn_v=approx(s,s['zp'],dynamic_v)
    # Full-dimensional mean variants and teacher-assisted variants are diagnostics only.
    outputs={'hard_drop':s['ys'],'oracle_both':blend(s,s['zt'],s['yt']),
      'oracle_mass_meanV':blend(s,s['zt'],true_mass_mean),
      'estimated_mass_oracleV':blend(s,mean_z,s['yt']),
      'raw_meanV':blend(s,mean_z,mean_v),'calibrated_meanV':blend(s,cal_z,cal_v),
      'raw_rank8V':blend(s,rawlow_z,rawlow_v),'calibrated_rank8V':blend(s,low_z,low_v),
      'oracle_mass_rank8V':blend(s,s['zt'],approx(s,truez,lowv)[1]),
      'denominator_only_oracle':(1-s['rho'][...,None])*s['ys']}
    outputs['dynamic_rank8V']=blend(s,dyn_z,dyn_v)
    for name in ['raw_rank8V','dynamic_rank8V']:
        delta=outputs[name]-s['ys']
        for strength in [.25,.5]:outputs[name+f'_g{strength}']=s['ys']+strength*delta
    target=e['target'];den=target.square().sum(-1).clamp_min(1e-8)
    assert torch.allclose(outputs['oracle_both'],target,atol=3e-5,rtol=3e-5),'exact decomposition failed'
    base=(outputs['hard_drop']-target).square().sum(-1)/den
    result={}
    for name,out in outputs.items():
        err=(out-target).square().sum(-1)/den
        assert torch.isfinite(err).all(),name
        result[name]=dict(mean_head_nmse=err.mean(1).cpu().tolist(),query_nmse=err.cpu().tolist(),worse_fraction=(err>base).float().mean(1).cpu().tolist())
    return dict(methods=result,rho=s['rho'].mean(1).cpu().tolist(),raw_mass_mae=((mean_z-s['zs']).sigmoid()-s['rho']).abs().mean(1).cpu().tolist(),cal_mass_mae=((cal_z-s['zs']).sigmoid()-s['rho']).abs().mean(1).cpu().tolist())

def report(root):
    records=[r for p in sorted((root/'val').glob('*.json')) for r in json.loads(p.read_text())['rows']]
    if not records:return
    names=list(records[0]['methods']);summary={}
    for name in names:
        vals=torch.tensor([v for r in records for head in r['methods'][name]['query_nmse'] for v in head])
        summary[name]=dict(nmse=vals.mean().item(),p95=vals.quantile(.95).item(),p99=vals.quantile(.99).item(),max=vals.max().item())
    atomic(root/'summary.json',dict(events=len(records),query_head_samples=len(vals),methods=summary,scope='Fixed original mass_var mask; dense replay; full29 layers; teacher matched physical338; diagnostics not video acceptance'))
    lines=['# 未选页补偿：固定mask诊断','','所有结果来自相同338历史页、相同72页选择、相同query。完整V均值和oracle均只作诊断；rank8为预算候选。dense轨迹离线结果不代表视频改善。','',f'事件数：{len(records)}；query/head样本：{len(vals)}。','','| 方法 | NMSE均值 | P95 | P99 | 最大值 |','|---|---:|---:|---:|---:|']
    for name,x in summary.items():lines.append(f"| {name} | {x['nmse']:.6f} | {x['p95']:.6f} | {x['p99']:.6f} | {x['max']:.6f} |")
    lines+=['','每层、每头、每步及逐query结果见val/*.json。当前未宣布任何方案通过端到端质量速度验收。']
    (root/'report.zh.md.tmp').write_text('\n'.join(lines)+'\n');os.replace(root/'report.zh.md.tmp',root/'report.zh.md')

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--train-cases',type=int,default=8);parser.add_argument('--val-cases',type=int,default=4);parser.add_argument('--chunks',type=int,nargs='+',default=list(CHUNKS));args=parser.parse_args()
    root=args.root;lock=RunLock(root)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=cases('train')[:args.train_cases];val=cases('val')[:args.val_cases]
        identity=dict(code={f:digest(Path('local_experiments')/f) for f in ['compensation_probe.py','improve_models.py','broad_common.py']},weights=digest('artifacts/mass_variance_20261030/train_00_30/best.pt'),dictionary=digest(DATA/'dictionary.pt'),train=[c['id'] for c in train],val=[c['id'] for c in val],chunks=args.chunks,layers=LAYERS,rank=8,data={c['id']:{p.name:digest(p) for p in sorted((DATA/'trajectories'/c['id']).glob('*.pt'))} for c in train+val})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity,'Changed sources/protocol: use new root'
        else:atomic(root/'identity.json',identity)
        old=torch.load('artifacts/mass_variance_20261030/train_00_30/best.pt',weights_only=False,map_location='cpu')['models']['mass_var']
        weights={l:old[l]['weight'].cuda() for l in LAYERS};alphas={l:old[l]['alpha'].cuda() for l in LAYERS}
        engine=Engine();events=Events(LAYERS)
        for phase,items in [('train',train),('val',val)]:
            if phase=='val':
                fit(root);model=torch.load(root/'model.pt',weights_only=True)['models']
            for case in items:
                paths={c:root/phase/f'{case["id"]}.c{c}.{"pt" if phase=="train" else "json"}' for c in args.chunks}
                if all(p.exists() for p in paths.values()):continue
                engine.reset();events.reset();engine.callback=None
                emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                for chunk in range(max(args.chunks)+1):
                    if STOP:atomic(root/'status.json',dict(stage='interrupted',phase=phase,case=case['id'],next_chunk=chunk));return
                    data=torch.load(DATA/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
                    if chunk in paths and not paths[chunk].exists():
                        rows=[];stats={l:empty_stats() for l in LAYERS}
                        atomic(root/'status.json',dict(stage='running',phase=phase,case=case['id'],chunk=chunk,pid=os.getpid()))
                        def callback(current,q,k,v):
                            l=current['layer']
                            if l not in LAYERS:return
                            ids=torch.linspace(0,4607,data['queries'].shape[-2],device=q.device).long()
                            assert torch.equal(q[0,ids].transpose(0,1).cpu(),data['queries'][current['step'],l]),'dense replay mismatch'
                            e=events.event(current,q,k,v);pk=proto(e['h']['k'],weights[l]);s=state(e,pk,alphas[l])
                            if phase=='train':accumulate(stats[l],e,s)
                            else:rows.append(dict(case=case['id'],layer=l,chunk=chunk,step=current['step'],**evaluate(e,s,model[l])))
                        engine.callback=callback
                        for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                        engine.callback=None
                        if phase=='train':atomic(paths[chunk],stats,True)
                        else:
                            assert len(rows)==29*4
                            atomic(paths[chunk],dict(rows=rows,replay_exact=True));report(root)
                        print(json.dumps(dict(phase=phase,case=case['id'],chunk=chunk,committed=True)),flush=True)
                        disk_guard(root,50,2)
                    engine.forward(data['clean'].cuda(),emb,chunk,4)
        report(root);atomic(root/'status.json',dict(stage='complete',pid=os.getpid()))
    except Exception as exc:atomic(root/'status.json',dict(stage='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
