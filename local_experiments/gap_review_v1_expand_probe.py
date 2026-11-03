"""Resumable same-state real-model ablations; no QKV dumps, no speed claim."""
import argparse,gc,json,os,time
from pathlib import Path
import torch
from .broad_common import Engine,ROOT as DATA,cases
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .review_v1.ops import Config,Summary,SummaryOps,RouteOps,AttentionOps
ROOT=Path('artifacts/review_v1_20261102/expanded_probe')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',default='bird_flight');ap.add_argument('--summary-checkpoint');ap.add_argument('--tag',required=True);ap.add_argument('--save-labels',action='store_true');args=ap.parse_args()
    root=ROOT/args.tag/args.case;lock=RunLock(root)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(source=digest('local_experiments/review_v1/ops.py'),runner=digest(__file__),checkpoint=digest('artifacts/four_gaps_20261031/shared/epoch1.pt'),summary_checkpoint=digest(args.summary_checkpoint) if args.summary_checkpoint else None,case=args.case,chunks=[2,5,8],layers=list(range(1,30)),steps=[0,1,2,3],tiles='(17*layer+13*chunk+7*step)%72',save_labels=args.save_labels,probes=16,seed=202)
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        engine=Engine();rt=GapGainVector64Runtime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',weights_only=False,map_location='cpu')['model'])
        candidate=None
        if args.summary_checkpoint:
            from .review_v1.model import Model
            candidate=Model('artifacts/four_gaps_20261031/shared/epoch1.pt')
            candidate.load_state_dict(torch.load(args.summary_checkpoint,weights_only=False,map_location='cuda')['model'])
        physical_factors={}
        if args.summary_checkpoint:
            metadata=Path(args.summary_checkpoint).with_name('result.json')
            if metadata.exists():
                for row in json.loads(metadata.read_text()).get('rows',[]):
                    if 'physical_log_scale' in row:
                        physical_factors[row['layer']]=(torch.tensor(row['physical_log_scale'],device='cuda',dtype=torch.float64)-torch.tensor(row['applied_scale'],device='cuda',dtype=torch.float64).log()).exp()[:,None,None]
        original=rt.decision;tiles=torch.tensor([0,36],device='cuda');offsets=torch.linspace(0,63,16,device='cuda').round().long();cfg=Config()
        def decision(l,q,pk,base,detail,cov,vvar):
            result=original(l,q,pk,base,detail,cov,vvar)
            chunk=engine.current['chunk'];step=engine.current['step'];path=root/f'chunk{chunk}_layer{l}_step{step}.json'
            if chunk not in [2,5,8] or step not in [0,1,2,3] or path.exists():return result
            tiles=torch.tensor([(17*l+13*chunk+7*step)%72],device='cuda')
            disk_guard(root,50,1);atomic(root/'status.json',dict(status='evaluating',chunk=chunk,layer=l,pid=os.getpid()))
            p=pk.shape[1];history=p-72;oldmodel=rt.model[l];m=oldmodel if candidate is None else candidate[l]
            k=engine.cache[l]['k'][0,:p*64].transpose(0,1).reshape(12,p,64,128)
            v=engine.cache[l]['v'][0,:p*64].transpose(0,1).reshape_as(k)
            qs=q[0].transpose(0,1).reshape(12,72,64,128).index_select(1,tiles).index_select(2,offsets).flatten(1,2)
            oldids=result[0][0].index_select(1,tiles).long()
            oldmask=torch.zeros(12,1,p,device='cuda',dtype=torch.bool).scatter(-1,oldids,True)
            expand=lambda mask:mask[:,:,None].expand(-1,-1,16,-1).reshape(12,16,p)
            hs=SummaryOps.build(m,k[:,:history],v[:,:history]);cs=SummaryOps.build(m,k[:,history:],v[:,history:],cheap=True)
            s=Summary(*(torch.cat((getattr(hs,n),getattr(cs,n)),1) for n in ('keys','mean','detail','stats')))
            # Old mean remains FP32 here: this is an algebraic old-summary reference,
            # not the old custom kernel or its numerical reduction order.
            oldmean=oldmodel.center[:,None]+torch.einsum('hpk,hdk->hpd',base.float(),oldmodel.basis.bfloat16().float())
            old=Summary(pk,oldmean,detail,s.stats)
            target,zd=AttentionOps.evaluate(qs,k,v,s,m.basis,torch.ones(12,16,p,device='cuda',dtype=torch.bool))
            rows={}
            def evaluate(name,summary,mask,basis=None):
                out,z=AttentionOps.evaluate(qs,k,v,summary,m.basis if basis is None else basis,expand(mask))
                per=(out-target).square().sum((1,2))/target.square().sum((1,2)).clamp_min(1e-12)
                rows[name]=dict(nmse=float((out-target).square().sum()/target.square().sum()),head_nmse=per.tolist(),denominator_relative_mae=float(((z-zd).exp()-1).abs().mean()),selected=int(mask.sum()),candidates=mask.numel())
            evaluate('old_summary_old_mask_reference',old,oldmask,oldmodel.basis)
            evaluate('fullmean_old_mask',s,oldmask)
            f,logz=RouteOps.features(qs,s);theta=qs.new_tensor([1,1,1,.5],dtype=torch.float32) if candidate is None else candidate[l].theta[:,None,None]
            score=(f*theta).sum(-1).reshape(12,1,16,p).mean(2)
            own=(tiles+history)[None].expand(12,-1);selected=RouteOps.select(score,own,cfg)
            evaluate('fullmean_risk_fixed10',s,selected['mask'])
            labels=AttentionOps.residual_labels(qs,k,v,s,m.basis,target,zd)
            # Save scalar training features, NOT raw Q/K/V or residual vectors.
            if args.save_labels:atomic(root/f'calibration_chunk{chunk}_layer{l}_step{step}.pt',dict(features=f.cpu(),labels=labels.cpu(),layer=l,chunk=chunk,step=step,tiles=tiles.cpu(),summary_identity=identity),True)
            pred=(f.double()*theta.double()).sum(-1)*physical_factors.get(l,1.)
            row=dict(chunk=chunk,layer=l,step=step,pages=p,rows=rows,label_mean=float(labels.mean()),initial_proxy_mean=float(pred.mean()),initial_proxy_mae=float((pred-labels).abs().mean()),summary_bytes=s.bytes(),proxy_units='physical residual units restored from calibration metadata; not a strict bound',scope='Sampled 16 rotating real queries/head on accepted-model trajectory; references, not closed-loop video or kernel fidelity.')
            atomic(path,row);print(json.dumps(row),flush=True)
            return result
        rt.decision=decision
        try:
            emb=torch.load(DATA/'embeddings'/f'{args.case}.pt',weights_only=True).cuda()
            for c in range(9):generate_chunk(engine,emb,202,c)
            files=list(root.glob('chunk*_layer*_step*.json'));assert len(files)==348,len(files)
            atomic(root/'complete.json',dict(status='complete',events=len(files),identity=identity))
        finally:rt.close()
    except BaseException as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close();gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
