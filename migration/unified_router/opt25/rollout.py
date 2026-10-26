"""Exact causal prefix reuse; RNG calls and scheduler order match v08 rollout."""
import torch
from streaming_sparse.v08.common import protocol,nmse
from streaming_sparse.v08.backbone import native_steps
from streaming_sparse.v08.cache import CacheIdentity,LayerKV,build_history,validate

@torch.no_grad()
def trajectory(network,embeds,pid,generation_seed,sparse,teacher_on_student=None,incremental=True,write_cache=False,height=720,width=1280,num_chunks=4):
    p=protocol();device=next(network.model.parameters()).device;dtype=next(network.model.parameters()).dtype
    steps,_=native_steps(network.wrapper)
    shape=(p['latent_shape'][0],height//8,width//8);seg=3*(height//16)*(width//16)
    rng=torch.Generator(device=device).manual_seed(generation_seed)
    noise=torch.randn((1,num_chunks*3,*shape),generator=rng,device=device,dtype=dtype)
    # Preserve the native sampler's separate RNG calls, not one concatenated
    # random draw whose CUDA Philox offsets could differ.
    transitions=torch.stack([torch.stack([torch.randn((3,*shape),generator=rng,device=device,dtype=dtype)
        .unflatten(0,(1,3)) for _ in range(3)]) for _ in range(num_chunks)])
    history=noise[:,:0];chunks=[];diagnostics=[];prefix_cache=()
    for chunk in range(num_chunks):
        identity=CacheIdentity(network.model_version,network.router_version,f'{pid}.seed{generation_seed}',
                               f'rollout.chunk{chunk}',0,chunk*seg,p['context_noise'],owner='student' if sparse else 'teacher')
        if incremental:
            # A frozen causal prefix is independent of later chunks. Only the
            # newly completed clean chunk is encoded at context_noise. Never
            # reuse noisy denoising K/V as clean-history K/V.
            cache=tuple(LayerKV(x.k,x.v,identity,x.layer) for x in prefix_cache)
            if chunk:
                result=network.forward_chunk(history[:,-3:].detach(),embeds,p['context_noise'],cache,(chunk-1)*3,
                    sparse=False,step=0,identity=identity,checkpoint=False,collect_aux=False,write_cache=True)
                cache=result['cache']
            validate(cache,identity,chunk*seg);prefix_cache=cache
        else:cache=build_history(network,history,embeds,identity,recent_gradient=False,checkpoint=False)
        current=noise[:,chunk*3:(chunk+1)*3]
        for step,timestep in enumerate(steps):
            result=network.forward_chunk(current,embeds,timestep,cache,chunk*3,sparse=sparse,step=step,
                identity=identity,checkpoint=False,collect_aux=False,write_cache=write_cache)
            x0=result['x0']
            if not torch.isfinite(x0).all():raise FloatingPointError('Nonfinite closed-loop latent')
            item=dict(chunk=chunk,step=step,timestep=timestep,latent_norm=float(x0.float().norm()),
                      latent_abs_max=float(x0.float().abs().max()))
            if teacher_on_student is not None:
                sample=dict(sample_id=f'rollout.chunk{chunk}.step{step}',trajectory_id=f'{pid}.seed{generation_seed}',
                    history_tokens=chunk*seg,context_noise=p['context_noise'],step=step,timestep=timestep,
                    noisy=current,history=history,prompt_embeds=embeds)
                reference=teacher_on_student.replay(sample,sparse=False,checkpoint=False,collect_aux=False)['x0']
                item['teacher_on_student_external_state_nmse']=float(nmse(x0,reference))
            diagnostics.append(item)
            if step<3:
                current=network.wrapper.scheduler.add_noise(x0.flatten(0,1),transitions[chunk,step].flatten(0,1),
                    torch.full((3,),steps[step+1],device=device)).unflatten(0,(1,3))
        chunks.append(x0.cpu());history=torch.cat((history,x0),1)
    return chunks,diagnostics

