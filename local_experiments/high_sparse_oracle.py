"""Diagnostic-only dense teacher on actual high-sparsity rollout inputs."""
import json,math,signal
from pathlib import Path
import torch
from .high_sparse_runtime import HighSparseRuntime
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .learned_cache.io import RunLock,atomic,digest,disk_guard
ROOT=Path('artifacts/high_sparse_20261030/oracle')
@torch.no_grad()
def evaluate(rt,q,k,v,lut):
    l=rt.active_layer;c=rt.engine.current;p=rt.active_pk.shape[1];device=q.device
    tiles=torch.linspace(0,71,8,device=device).long();ids=(tiles[:,None]*64+torch.tensor([0,21,42,63],device=device)).flatten()
    qs=q[0].transpose(0,1)[:,ids].float();kk=k[0].transpose(0,1).float().reshape(12,p,64,128);vv=v[0].transpose(0,1).float().reshape_as(kk)
    z=torch.einsum('hqd,hptd->hqpt',qs,kk)/math.sqrt(128)
    logz=z.logsumexp(-1);values=torch.einsum('hqpt,hptd->hqpd',z.softmax(-1),vv);del z,kk,vv
    weights=logz.softmax(-1);target=(weights[...,None]*values).sum(2)
    az=(qs@rt.active_pk.flatten(1,2).transpose(1,2)/math.sqrt(128)).reshape(12,32,p,4).logsumexp(-1)+math.log(16)
    aw=(az-logz.logsumexp(-1,keepdim=True)).exp()
    av=rt.centers[l][:,None]+rt.active_codes.float()@rt.basis[l].float().transpose(1,2)
    chosen=lut[0,:,tiles].long();mask=torch.zeros(12,8,p,device=device,dtype=torch.bool).scatter_(-1,chosen,True)
    energy=target.square().sum(-1).clamp_min(1e-8)
    def err(out):return float(((out-target).square().sum(-1)/energy).mean())
    def output(m,tail=True,oracle_mass=False):
        m=m.repeat_interleave(4,1);w=weights*m
        if not tail:return (w[...,None]*values).sum(2)/w.sum(-1,keepdim=True).clamp_min(1e-20)
        a=(weights if oracle_mass else aw)*(~m)
        return ((w[...,None]*values).sum(2)+a@av)/(w.sum(-1)+a.sum(-1))[...,None].clamp_min(1e-20)
    scores=weights.reshape(12,8,4,p).mean(2)
    own=(p-72+tiles)[None,:,None].expand(12,8,1)
    oscores=scores.clone();oscores.scatter_(-1,own,float('inf'))
    omask=torch.zeros_like(mask).scatter_(-1,oscores.topk(chosen.shape[-1],-1).indices,True)
    w=weights.reshape(12,8,4,p);a=aw.reshape_as(w);val=values.reshape(12,8,4,p,128)
    dw=w-a;dn=w[...,None]*val-a[...,None]*av[:,None,None]
    N=(a[...,None]*av[:,None,None]).sum(3)+(dn*mask[:,:,None,:,None]).sum(3)
    Z=a.sum(3)+(dw*mask[:,:,None]).sum(3)
    dr=scores.masked_fill(~mask,float('inf'));dr.scatter_(-1,own,float('inf'))
    drop=dr.topk(min(8,chosen.shape[-1]-1),-1,largest=False).indices
    add=scores.masked_fill(mask,-float('inf')).topk(8,-1).indices
    def gather(x,indices):
        idx=indices[:,:,None].expand(-1,-1,4,-1)
        return x.gather(3,idx[...,None].expand(-1,-1,-1,-1,128)) if x.ndim==5 else x.gather(3,idx)
    nr,na=gather(dn,drop),gather(dn,add);zr,za=gather(dw,drop),gather(dw,add)
    newN=N[:,:,:,None,None]-nr[:,:,:,:,None]+na[:,:,:,None,:]
    newZ=Z[:,:,:,None,None]-zr[:,:,:,:,None]+za[:,:,:,None,:]
    ys=newN/newZ[...,None].clamp_min(1e-20);tar=target.reshape(12,8,4,128)
    errors=((ys-tar[:,:,:,None,None]).square().sum(-1)/energy.reshape(12,8,4)[:,:,:,None,None]).mean(2).flatten(-2)
    initial=((N/Z[...,None]-tar).square().sum(-1)/energy.reshape(12,8,4)).mean(2)
    best,which=errors.min(-1);improved=best<initial
    corrected=mask.clone();corrected.scatter_(-1,drop.gather(-1,(which//8)[...,None]),~improved[...,None]);corrected.scatter_(-1,add.gather(-1,(which%8)[...,None]),improved[...,None])
    assert torch.equal(corrected.sum(-1),mask.sum(-1)) and corrected.gather(-1,own).all()
    return dict(layer=l,chunk=c['chunk'],step=c['step'],pages=p,keep=chosen.shape[-1],sampled_queries=32,heads=12,selected_current_fraction=float((chosen>=p-72).float().mean()),retained_mass=float((weights*mask.repeat_interleave(4,1)).sum(-1).mean()),nmse=dict(drop=err(output(mask,False)),comp=err(output(mask)),oracle_mass_with_same_V=err(output(mask,True,True)),oracle_mass_route_drop=err(output(omask,False)),oracle_mass_route_comp=err(output(omask)),oracle_residual_one_swap=err(output(corrected))),swap_fraction=float(improved.float().mean()),warning='Dense supervision diagnostics only. One-swap examines up to 8x8 candidates with no-op; not global optimality or deployable routing. Local errors are not video quality.')
class Observer:
    def __init__(self,rt,base,folder):self.rt=rt;self.base=base;self.folder=folder
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            r=self.rt;c=r.engine.current
            if r.active_layer in [4,14,24] and c['chunk'] in [2,5,8] and c['step'] in [0,3]:
                path=self.folder/f"c{c['chunk']}_l{r.active_layer}_s{c['step']}.json"
                if not path.exists():
                    row=evaluate(r,args[0],args[1],args[2],args[5]);atomic(path,row);print(json.dumps(row),flush=True)
            self.base[grid](*args,**kwargs)
        return call
@torch.no_grad()
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity={s:digest(Path('local_experiments')/s) for s in ['high_sparse_oracle.py','high_sparse_runtime.py']}
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine()
        for case in ['basketball_court','butterfly_garden']:
            folder=ROOT/case;folder.mkdir(exist_ok=True)
            if len(list(folder.glob('c*.json')))==18:continue
            if stop[0]:return
            disk_guard(ROOT,50,.2)
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda();rt=HighSparseRuntime(engine,'global',.9,True);rt.kernel=Observer(rt,rt.kernel,folder)
            try:
                for c in range(9):generate_chunk(engine,emb,1001,c)
            finally:rt.close();del rt
        rows=[json.loads(p.read_text()) for p in ROOT.glob('*/c*.json')]
        atomic(ROOT/'summary.json',dict(units=len(rows),expected=36,means={key:sum(r['nmse'][key] for r in rows)/len(rows) for key in rows[0]['nmse']},rows=rows))
    finally:lock.close()
if __name__=='__main__':main()
