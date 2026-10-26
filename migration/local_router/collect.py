"""One frozen Dense replay pass per event; immutable raw-KV/cache statistics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from streaming_sparse.v08.common import atomic_json,save,paired_seed,source_identity
from streaming_sparse.v08.replay import ReplayDataset
from streaming_sparse.v08.backbone import FunctionalBackbone,load_teacher
from streaming_sparse.capture.capture_runner import sdpa_backend
from .core import page_statistics,output


def event_key(row):return hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]
def history_key(row):return hashlib.sha256(f"{row['trajectory_id']}:{row['history_tokens']}".encode()).hexdigest()[:24]
def cpu_dict(values):return{k:v.detach().cpu().contiguous().clone()if torch.is_tensor(v)else v for k,v in values.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    root=Path(a.directory).resolve();m=json.loads((root/'experiment.json').read_text());assert m['source_sha256']==source_identity()
    cache=Path(m['cache']);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0);torch.cuda.set_device(0)
    trajectories=sorted({r['trajectory_id']for r in m['rows']});mine=set(trajectories[rank::world])
    rows=[r for r in m['rows']if r['trajectory_id']in mine]
    if a.pilot:rows=rows[:1]
    datasets={s:ReplayDataset(s,require_complete=True)for s in ['train','val']};row=None;start_frame=None;written=[]
    phase='pilot'if a.pilot else'collection';out=root/phase;out.mkdir(exist_ok=True)
    def progress(**kw):
        atomic_json(out/f'rank{rank}.json',dict(rank=rank,updated_at=datetime.now(timezone.utc).isoformat(),**kw));print(json.dumps(kw),flush=True)
    class Capture(FunctionalBackbone):
        def _block(self,*args,**kwargs):
            block,layer,x,e0,grid,freqs=args[:6];hk,hv,start=args[7:10]
            if start==start_frame and layer in m['layers']:
                ek=event_key(row);hp=cache/'history'/history_key(row)/f'layer{layer:02d}.pt';ep=cache/'events'/ek/f'layer{layer:02d}.pt'
                if not ep.exists()or torch.load(ep,map_location='cpu',weights_only=True,mmap=True).get('cache_format')!=2:
                    from wan.modules.causal_model import causal_rope_apply
                    frames=e0.shape[1];tokens=x.shape[1]//frames;e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
                    mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2);att=block.self_attn
                    q=att.norm_q(att.q(mod)).view(1,mod.shape[1],12,128)
                    k=att.norm_k(att.k(mod)).view_as(q);v=att.v(mod).view_as(q)
                    q=causal_rope_apply(q,grid,freqs,start_frame=start).to(v.dtype)[0]
                    k=causal_rope_apply(k,grid,freqs,start_frame=start).to(v.dtype)[0]
                    ix=torch.tensor(row['query_indices'],device=x.device)
                    qq=q[ix].transpose(0,1).contiguous();kk=k.transpose(0,1).contiguous();vv=v[0].transpose(0,1).contiguous()
                    kh=hk[0].transpose(0,1).contiguous();vh=hv[0].transpose(0,1).contiguous()
                    values=page_statistics(qq,kh,vh,kk,vv)
                    values['target']=output(values,torch.ones_like(values['page_logz']))
                    # Independent direct raw-token softmax on a few queries.
                    keys=torch.cat((kh,kk),1);vals=torch.cat((vh,vv),1)
                    ll=torch.bmm(qq.float(),keys.float().transpose(1,2))/128**.5
                    direct=torch.bmm(ll[:,:2].double().softmax(-1),vals.double()).float()
                    torch.testing.assert_close(values['target'][:,:2],direct,atol=5e-5,rtol=5e-5)
                    probe={}
                    if not durations:
                        # Direct token-mask oracle; no model/Router graph.
                        with torch.enable_grad():
                            small={name:value[:,:4]for name,value in values.items()}
                            gate=torch.linspace(.1,.9,values['page_logz'].shape[-1],device=x.device).expand(12,4,-1).clone().requires_grad_()
                            cached=output(small,gate)
                            token_mask=torch.cat((gate.repeat_interleave(64,-1)[...,:kh.shape[1]],gate.new_ones(12,4,kk.shape[1])),-1).double()
                            ex=(ll[:,:4].double()-ll[:,:4].double().amax(-1,keepdim=True)).exp()
                            weighted=ex*token_mask
                            raw=(torch.bmm(weighted,vals.double())/weighted.sum(-1,keepdim=True)).float()
                            torch.testing.assert_close(cached,raw,atol=5e-5,rtol=5e-5)
                            upstream=torch.ones_like(raw)/raw.numel()
                            gc=torch.autograd.grad(cached,gate,upstream)[0];gr=torch.autograd.grad(raw,gate,upstream)[0]
                            torch.testing.assert_close(gc,gr,atol=2e-5,rtol=2e-4)
                            probe=dict(raw_output_max_abs=float((cached-raw).abs().max()),gate_gradient_max_abs=float((gc-gr).abs().max()))
                    values.update(q=qq,layer=layer,sample_id=row['sample_id'],history_file=str(hp),cache_format=2,
                        query_indices=row['query_indices'],source_sha256=m['source_sha256'],teacher_sha256=m['teacher_sha256'],
                        step=row['step'],split=row['split'],prompt_id=row['prompt_id'],history_tokens=row['history_tokens'])
                    if not hp.exists():save(hp,cpu_dict(dict(kh=kh,vh=vh,source_sha256=m['source_sha256'],trajectory_id=row['trajectory_id'],history_tokens=row['history_tokens'],layer=layer)))
                    save(ep,cpu_dict(values));written.append(dict(layer=layer,file=str(ep),bytes=ep.stat().st_size,
                        dense_direct_max_abs=float((values['target'][:,:2]-direct).abs().max()),real_probe=probe))
                    del values,keys,vals,ll,direct,qq,kk,vv,kh,vh
            return super()._block(*args,**kwargs)
    progress(status='loading',events_total=len(rows),layers=m['layers'])
    net=Capture(load_teacher('cuda'),None,layers=m['layers']).eval();versions=tuple(p._version for p in net.model.parameters());begin=time.monotonic();durations=[]
    with torch.no_grad(),sdpa_backend():
        for i,row in enumerate(rows,1):
            if(root/'STOP').exists():raise RuntimeError('Stop requested')
            ek=event_key(row);receipt=cache/'receipts'/f'{ek}.json'
            if receipt.exists():progress(status='collecting',completed=i,total=len(rows),reused=True);continue
            sample=datasets[row['split']].load(row['replay_index']);assert sample['sample_id']==row['sample_id'];start_frame=sample['history'].shape[1]
            tick=time.monotonic();written=[]
            result=net.replay(sample,sparse=False,checkpoint=False,collect_aux=False,recent_gradient=False);del result
            assert all((cache/'events'/ek/f'layer{l:02d}.pt').exists()for l in m['layers'])
            torch.cuda.synchronize();durations.append(time.monotonic()-tick)
            atomic_json(receipt,dict(sample_id=row['sample_id'],source_sha256=m['source_sha256'],query_indices=row['query_indices'],layers=m['layers'],files=written))
            progress(status='collecting',completed=i,total=len(rows),seconds_last=durations[-1],seconds_mean=sum(durations)/len(durations),elapsed_seconds=time.monotonic()-begin,sample_id=row['sample_id'])
    assert tuple(p._version for p in net.model.parameters())==versions
    progress(status='completed',completed=len(rows),total=len(rows),events_total=len(rows),elapsed_seconds=time.monotonic()-begin,
        backbone_trainable=sum(p.numel()for p in net.model.parameters()if p.requires_grad),parameter_versions_unchanged=True)


if __name__=='__main__':main()
