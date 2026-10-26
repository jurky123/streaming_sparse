"""Warm optimized end-to-end timings and an independent actual-trajectory audit."""
import argparse,gc,json,time
from pathlib import Path
import torch
from .common import config,atomic_json,update,paired_seed,sha
from .fast_network import load_fast
from .speed_profile import digest,gpu_info


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',default='K32-Set');p.add_argument('--backend',default='triton')
    p.add_argument('--batch',type=int,default=512);p.add_argument('--group',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--audit',action='store_true')
    p.add_argument('--no-compile',action='store_true');p.add_argument('--graph',action='store_true')
    p.add_argument('--fused',action='store_true');p.add_argument('--query-stride',type=int,default=1)
    p.add_argument('--history-cache',action='store_true');p.add_argument('--compiled-build',action='store_true');a=p.parse_args()
    root,m=config(a.directory);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**kw):
        update(out/'status.json',status=stage,mode=a.mode,**kw);print(stage,kw,flush=True)
    status('loading');paired_seed(0)
    net,_=load_fast(root,compile=not a.no_compile,batch=a.batch,attention=a.backend,group=a.group,graph=a.graph,
        fused=a.fused,query_stride=a.query_stride,history_cache=a.history_cache,compiled_build=a.compiled_build)
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from utils.wan_wrapper import WanVAEWrapper
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=ReplayDataset('val',require_complete=True);pid=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'][0]
    sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
    embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
    original=tuple(p._version for p in net.parameters());records=[];reference=None
    with sdpa_backend():
        for iteration in range(a.repeats+1):
            net.set_mode(a.mode);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            status('warming/compiling' if iteration==0 else 'benchmark',iteration=iteration)
            begin=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,a.mode!='Dense')
            torch.cuda.synchronize();generated=time.perf_counter();latent=torch.cat(chunks,1)
            pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu()
            vae.model.clear_cache();torch.cuda.synchronize();end=time.perf_counter()
            row=dict(iteration=iteration,generation_seconds=generated-begin,decode_seconds=end-generated,total_seconds=end-begin,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,latent_sha256=digest(latent),first_chunk_sha256=digest(chunks[0]),gpu=gpu_info())
            if iteration==0:reference=latent.clone()
            else:assert torch.equal(reference,latent)
            records.append(row);atomic_json(out/'timings.json',dict(options=vars(a),records=records,prompt=pid));print(json.dumps(row),flush=True)
            if iteration==a.repeats:
                torch.save(dict(latent=latent,pixels=pixels),out/'output.pt')
                values=torch.stack([torch.stack([r['retained_H'],r['retained_C']])for r in net.routing_records]).cpu() if net.routing_records else None
                if values is not None:
                    routed=[]
                    for r,retained in zip(net.routing_records,values):
                        item={k:v for k,v in r.items()if not isinstance(v,torch.Tensor)}
                        item.update(retained_H=float(retained[0]),retained_C=float(retained[1]));routed.append(item)
                    atomic_json(out/'routing.json',routed)
            del pixels,latent,chunks;gc.collect()
        if a.audit and a.mode!='Dense':
            status('independent_actual_block_audit')
            def inspect(layer,start,step,q,k,v,sketch,prepared,mask,y):
                if (layer,start,step)!=(24,9,3):return
                ref=torch.cat([net.router.route(layer,q[:,i:i+128],sketch,step) for i in range(0,q.shape[1],128)],1)
                from .fast_kernels import sparse_attention
                from .fast_audit import timed
                results=[]
                for group in [16,32,64]:
                    yy,counts=sparse_attention(q,k,v,mask,group,True)
                    results.append(dict(group=group,ms=timed(lambda:sparse_attention(q,k,v,mask,group),10),
                        union_fraction=float(counts.float().mean()/mask.shape[-1])))
                truth=torch.cat([torch.nn.functional.scaled_dot_product_attention(q[:,i:i+128][None],k[None],v[None],
                    attn_mask=mask[:,i:i+128].repeat_interleave(sketch.costs.long(),-1)[None])[0]for i in range(0,q.shape[1],128)],1)
                error=(y.float()-truth.float())
                result=dict(mask_different_bits=int((mask!=ref).sum()),mask_different_queries=int((mask!=ref).any(-1).sum()),
                    attention_nmse=float(error.square().sum()/truth.float().square().sum()),attention_max_abs=float(error.abs().max()),
                    attention_benchmarks=results,logical_keep=float((mask*sketch.costs).sum()/(q.shape[0]*q.shape[1]*k.shape[1])))
                atomic_json(out/'actual_block_audit.json',result);print(json.dumps(result),flush=True)
                torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),mask=mask.cpu(),reference_mask=ref.cpu(),
                    sketch=sketch,prepared=prepared,head=net.router.heads[f'{layer}_{a.mode}'].state_dict(),beta=net.router.beta),out/'actual_block.pt')
            net.inspect_callback=inspect;net.set_mode(a.mode)
            chunks,_=trajectory(net,embeds,pid,0,True)
            assert torch.equal(torch.cat(chunks,1),reference)
    assert tuple(p._version for p in net.parameters())==original
    status('completed',backbone_and_head_unchanged=True)

if __name__=='__main__':main()
