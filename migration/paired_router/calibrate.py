"""Train-only Q/K normalization and common LogZ prefit for the 26 new layers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import torch
from migration.paired_router import runtime
from streaming_sparse.v08.common import atomic_json,paired_seed,save,sha,source_identity
from streaming_sparse.v08.attention import block_logz
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.capture.capture_runner import sdpa_backend


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve()
    manifest,spec,_=runtime.configuration(root,'dense0');rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    paired_seed(0);torch.cuda.set_device(0)
    all_new=[i for i in range(1,30)if i not in [4,14,24]];layers=all_new[rank::8];pairs=[(l-1)*12+h for l in layers for h in range(12)]
    router=runtime.BaseRouter('K',pairs=348).cuda();router.requires_grad_(False)
    shared=torch.load(root/'three/runs/seed0/warmstart_K/best.pt',map_location='cpu',weights_only=True)
    router.sketch.cal_scale.copy_(shared['router']['sketch.cal_scale'])
    ds=ReplayDataset('train',require_complete=True);pids=sorted(ds.registry['splits']['train'])[:manifest['calibration_prompts']]
    events=[i for i,r in enumerate(ds.rows)if r['prompt_id']in pids and r['generation_seed']==pids.index(r['prompt_id'])%2]
    normal=[]
    for j,pid in enumerate(pids):normal.append(next(i for i in events if ds.rows[i]['prompt_id']==pid and ds.rows[i]['history_tokens']==[4680,9360,14040][j%3]and ds.rows[i]['step']==j%4))
    out=Path(spec['directory'])/'calibration';out.mkdir(exist_ok=True)
    progress=dict(rank=rank,layers=layers,phase='loading',source_sha256=source_identity(),calibration_code_sha256=sha(__file__),test_used=False)
    def status(**kw):
        progress.update(kw,updated_at=datetime.now(timezone.utc).isoformat());atomic_json(out/f'rank{rank}.json',progress);print(json.dumps(kw),flush=True)
    sums={i:torch.zeros(6,device='cuda',dtype=torch.float64)for i in pairs};squares={i:torch.zeros_like(sums[i])for i in pairs};counts={i:0 for i in pairs}
    options={}
    for pair in pairs:
        params=list(router.sketch.assignment[pair].parameters())+list(router.sketch.calibrator.networks[pair].parameters())
        for parameter in params:parameter.requires_grad_(True)
        options[pair]=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    phase='normalization';event_index=0;history_start=0
    class Capture(runtime.BaseNetwork):
        def _block(self,*args,**kwargs):
            block,layer,x,e0,grid,freqs=args[:6];hk=args[7];start=args[9]
            if layer in layers and start==history_start:
                from wan.modules.causal_model import causal_rope_apply
                frames=e0.shape[1];tokens=x.shape[1]//frames;e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
                mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2);att=block.self_attn
                q=att.norm_q(att.q(mod)).view(1,len(mod[0]),12,128)
                q=causal_rope_apply(q,grid,freqs,start_frame=start).to(mod.dtype)[0].detach()
                key_id=ds.rows[event_index]['sample_id']
                repeats=1 if phase=='normalization'else manifest['calibration_updates_per_event']
                for update in range(repeats):
                    seed=int.from_bytes(hashlib.sha256(f'{key_id}:{phase}:{update}'.encode()).digest()[:8],'little')%(2**63-1)
                    ix=torch.randperm(len(q),generator=torch.Generator().manual_seed(seed))[:manifest['calibration_queries']].to('cuda')
                    selected=q[ix]
                    for h in range(12):
                        pair=(layer-1)*12+h;qq=selected[:,h].detach();kk=hk[0,:,h].detach()
                        if phase=='normalization':
                            st=router.sketch.build(kk,pair);_,_,features,_=router.sketch.raw(qq,st);features=features.reshape(-1,6).double()
                            sums[pair]+=features.sum(0);squares[pair]+=features.square().sum(0);counts[pair]+=len(features)
                        else:
                            with torch.enable_grad():
                                optimizer=options[pair];optimizer.zero_grad(set_to_none=True)
                                # Every optimizer update rebuilds differentiable summaries.
                                st=router.sketch.build(kk,pair);raw,_,features,_=router.sketch.raw(qq,st)
                                norm=(features-router.sketch.feature_mean[pair])/router.sketch.feature_std[pair].clamp_min(1e-3)
                                logz=raw+router.sketch.calibrator(norm,pair).squeeze(-1)
                                with torch.no_grad():truth=block_logz(qq,kk)
                                loss=torch.nn.functional.huber_loss(logz,truth)/router.sketch.cal_scale.clamp_min(1e-8)
                                assert torch.isfinite(loss);loss.backward()
                                torch.nn.utils.clip_grad_norm_(list(router.sketch.assignment[pair].parameters())+list(router.sketch.calibrator.networks[pair].parameters()),1.,error_if_nonfinite=True)
                                optimizer.step()
            return super()._block(*args,**kwargs)
    net=Capture(runtime.load_teacher('cuda'),None,layers=list(range(1,30))).eval();versions=tuple(p._version for p in net.model.parameters())
    with torch.no_grad(),sdpa_backend():
        for phase,indices in [('normalization',normal),('logz_prefit',events)]:
            if phase=='logz_prefit':
                for pair in pairs:
                    mean=sums[pair]/counts[pair];var=(squares[pair]/counts[pair]-mean.square()).clamp_min(0)
                    router.sketch.feature_mean[pair].copy_(mean.float());router.sketch.feature_std[pair].copy_(var.sqrt().clamp_min(1e-3).float());router.sketch.feature_count[pair]=counts[pair]
                router.sketch.normalization_locked.fill_(True)
            for n,event_index in enumerate(indices,1):
                sample=ds.load(event_index);history_start=sample['history'].shape[1]
                result=net.replay(sample,sparse=False,checkpoint=False,collect_aux=False,recent_gradient=False);del result
                status(phase=phase,completed=n,total=len(indices),sample_id=sample['sample_id'])
    assert tuple(p._version for p in net.model.parameters())==versions
    payload=dict(router=router.cpu().state_dict(),pairs=pairs,layers=layers,feature_counts=counts,
        normalization_sample_ids=[ds.rows[i]['sample_id']for i in normal],prefit_sample_ids=[ds.rows[i]['sample_id']for i in events],
        updates_per_pair=len(events)*manifest['calibration_updates_per_event'],train_fingerprint=ds.fingerprint(),
        qk_only=True,backbone_frozen=True,source_sha256=source_identity())
    save(out/f'rank{rank}.pt',payload);status(phase='completed',passed=True,updates_per_pair=payload['updates_per_pair'])


if __name__=='__main__':main()
