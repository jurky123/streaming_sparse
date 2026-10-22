"""Fail-closed interface, information and branch-specific gradient audits."""
import argparse
import copy
import json
import math
from dataclasses import replace
import torch
from torch import nn
from .common import HERE,protocol,source_identity,atomic_json,paired_seed,nmse,ROOT
from .attention import attend,reference,costs,masks,BudgetGate
from .router import InformationRouter
from .lora import inject,configure,lora_parameters,base_digest
from .cache import CacheIdentity,LayerKV,validate


def raw_checks(device='cpu',real_event=None):
    torch.manual_seed(9)
    if real_event is None:
        q=torch.randn(3,128,device=device,dtype=torch.float64)*.3
        kh=torch.randn(137,128,device=device,dtype=torch.float64)*.3
        vh=torch.randn_like(kh);kc=torch.randn(11,128,device=device,dtype=torch.float64)*.3;vc=torch.randn_like(kc)
    else:
        e=real_event;q=e['q'][:2].float();kh=e['k_hist'].float();vh=e['v_hist'].float();kc=e['k_current'].float();vc=e['v_current'].float()
    page_cost=costs(len(kh),64,q.device)
    errors=[];gradient_errors=[]
    for keep in (0.,.1,.2,.3,1.):
        hard=masks(torch.randn(len(q),len(page_cost),device=device),page_cost,keep)[1].to(q.dtype)
        raw=[x.detach().clone().requires_grad_() for x in (q,kh,vh,kc,vc)]
        direct=[x.detach().clone().requires_grad_() for x in (q,kh,vh,kc,vc)]
        a=attend(*raw,hard);b=reference(*direct,hard)
        tol=protocol()['audit_tolerances']
        torch.testing.assert_close(a,b,atol=tol['raw_fp32_atol'],rtol=tol['raw_fp32_rtol'])
        errors.append(float((a-b).abs().max()))
        g=torch.randn_like(a)
        ga=torch.autograd.grad((a*g).sum(),raw);gb=torch.autograd.grad((b*g).sum(),direct)
        for aa,bb in zip(ga,gb):
            torch.testing.assert_close(aa,bb,atol=tol['raw_fp32_atol'],rtol=tol['raw_fp32_rtol'])
            gradient_errors.append(float((aa-bb).abs().max()))
        assert ((hard*page_cost).sum(-1)<=math.floor(keep*len(kh))).all()
    return dict(max_output_abs=max(errors),max_qkv_gradient_abs=max(gradient_errors),all5budgets=True,
                history_tokens=len(kh),current_tokens=len(kc),queries=len(q))


def unit_checks():
    paired_seed(0);checks={}
    checks['raw_attention']=raw_checks()
    # Native BF16 uses PyTorch's standard BF16 dtype tolerances, not a relaxed
    # version of the strong FP32 math checks above. Same query count, raw gather.
    args=[(torch.randn(*shape)*.1).bfloat16().requires_grad_() for shape in
          [(3,128),(137,128),(137,128),(11,128),(11,128)]]
    clones=[x.detach().clone().requires_grad_() for x in args]
    fixed=torch.tensor([[0.,0.,1.]]).expand(3,-1)
    actual=attend(*args,fixed)
    qh,kh,vh,kc,vc=clones
    key=torch.cat((kh[128:],kc));value=torch.cat((vh[128:],vc))
    expected=torch.nn.functional.scaled_dot_product_attention(qh[None,None],key[None,None],value[None,None])[0,0]
    torch.testing.assert_close(actual,expected)
    w=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*w).sum(),args);gb=torch.autograd.grad((expected*w).sum(),clones)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
    checks['native_BF16_hard_output_and_QKV_gradients_vs_raw_gather']=dict(max_output_abs=float((actual-expected).abs().max()),
        max_gradient_abs=max(float((a-b).abs().max()) for a,b in zip(ga,gb)),tolerance='torch.testing default BF16; FP32 checks unchanged')
    q=torch.randn(2,8,dtype=torch.float64)*.1;k=torch.randn(9,8,dtype=torch.float64)*.1
    v=torch.randn_like(k);kc=torch.randn(3,8,dtype=torch.float64);vc=torch.randn_like(kc)
    soft=torch.full((2,3),.4,dtype=torch.float64,requires_grad=True)
    args=tuple(x.requires_grad_() for x in (q,k,v,kc,vc))+(soft,)
    assert torch.autograd.gradcheck(lambda *a:attend(*a,size=4),args,atol=2e-5,rtol=2e-4)
    checks['continuous_mask_and_raw_qkv_gradcheck']=True
    score=torch.randn(3,4,dtype=torch.float64,requires_grad=True)*.1
    cost=torch.tensor([4.,4.,4.,1.],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda s:BudgetGate.apply(s,cost,7,.2),(score,),atol=2e-5,rtol=2e-4)
    g=torch.autograd.grad(BudgetGate.apply(score,cost,7,.2).sum(),score)[0]
    torch.testing.assert_close(g.sum(-1),torch.zeros(3,dtype=g.dtype),atol=1e-9,rtol=0)
    checks['implicit_cost_budget_derivative_and_score_shift_invariance']=True
    s=torch.randn(41,5,requires_grad=True);c=torch.tensor([64.,64.,64.,64.,8.])
    full=masks(s,c,.2)
    partial=[masks(s[i:i+16],c,.2) for i in range(0,len(s),16)]
    assert torch.equal(full[1],torch.cat([x[1] for x in partial]))
    torch.testing.assert_close(full[2],torch.cat([x[2] for x in partial]),atol=1e-7,rtol=1e-6)
    weight=torch.randn_like(s)
    gf=torch.autograd.grad((full[0]*weight).sum(),s,retain_graph=True)[0]
    gp=torch.autograd.grad((torch.cat([x[0] for x in partial])*weight).sum(),s)[0]
    torch.testing.assert_close(gf,gp,atol=2e-6,rtol=2e-5)
    checks['all_query_batched_budget_equals_independent_microbatches']=True
    rk=InformationRouter('K',pairs=1);rv=InformationRouter('V',pairs=1);rv.load_state_dict(rk.state_dict())
    q=torch.randn(5,128)*.1;kh=torch.randn(384,128)*.1;vh=torch.randn_like(kh)
    kc=torch.randn(7,128)*.1;vc=torch.randn_like(kc)
    ak=rk(q,kc,vc,rk.build(kh,vh,0));bk=rk(q,kc,vc*100+3,rk.build(kh,-vh*300,0))
    assert torch.equal(ak['scores'],bk['scores']) and torch.equal(ak['hard'],bk['hard'])
    assert torch.equal(ak['scores'],ak['estimate']['logz'])
    checks['RK_V_replacement_with_sketch_rebuild_exact_invariance']=True
    assert sum(p.numel() for p in rk.parameters())==sum(p.numel() for p in rv.parameters())
    with torch.no_grad():rv.head.networks[0][-1].weight.normal_()
    av=rv(q,kc,vc,rv.build(kh,vh,0));bv=rv(q,kc,-vc*10,rv.build(kh,vh*100+7,0))
    assert (av['scores']-bv['scores']).abs().max()>1e-5
    checks['RV_nonzero_residual_responds_to_V']=float((av['scores']-bv['scores']).abs().max())
    checks['parameter_capacity_equal']=sum(p.numel() for p in rk.parameters())
    # Initial final head zeros but outside scale nonzero; no dead-gate init.
    initial=ak['mask'].sum();initial.backward()
    checks['independent_final_head_initialization']=True
    ident=CacheIdentity('m0','r0','t0','s0',0,3,0.)
    entry=LayerKV(torch.zeros(1,3,1,128),torch.zeros(1,3,1,128),ident,0)
    try:validate((entry,),replace(ident,model_version='m1'),3)
    except ValueError:checks['stale_model_cache_rejected']=True
    else:raise AssertionError('Stale cache accepted')
    from .metrics import factorial
    models={}
    for seed in range(3):
        for group,value in {'F-K':2.,'F-V':1.8,'A-K':1.3,'A-V':1.2,'J-K':1.5,'J-V':1.}.items():
            models[seed,group]=dict(prompts={str(i):value+i*.1 for i in range(8)})
    f=factorial(models)
    assert abs(f['G_F']['mean']-.2)<1e-10 and abs(f['G_J']['mean']-.5)<1e-10 and abs(f['I']['mean']-.3)<1e-10
    checks['prompt_cluster_bootstrap_and_difference_in_differences']=True
    return checks


class TinyWrapper(nn.Module):
    """Real Wan modules on tiny spatial inputs; explicitly NOT production data."""
    def __init__(self):
        super().__init__()
        from wan.modules.causal_model import CausalWanModel
        from utils.scheduler import FlowMatchScheduler
        self.model=CausalWanModel(dim=256,ffn_dim=512,num_heads=2,num_layers=3,
                                 text_dim=32,text_len=16,in_dim=16,out_dim=16)
        self.scheduler=FlowMatchScheduler(shift=5.,sigma_min=0.,extra_one_step=True)
        self.scheduler.set_timesteps(1000,training=True)
        with torch.no_grad():self.model.head.head.weight.normal_(std=.02)
    def _convert_flow_pred_to_x0(self,*args,**kwargs):
        from utils.wan_wrapper import WanDiffusionWrapper
        return WanDiffusionWrapper._convert_flow_pred_to_x0(self,*args,**kwargs)
    def forward(self,noisy,conditional,timestep,**kwargs):
        flow=self.model(noisy.permute(0,2,1,3,4),t=timestep,context=conditional['prompt_embeds'],seq_len=32760,**kwargs).permute(0,2,1,3,4)
        return flow,self._convert_flow_pred_to_x0(flow.flatten(0,1),noisy.flatten(0,1),timestep.flatten()).unflatten(0,flow.shape[:2])


def native_replay(wrapper,sample):
    from .backbone import native_cache
    device=next(wrapper.parameters()).device;dtype=next(wrapper.parameters()).dtype
    h=sample['history'].to(device=device,dtype=dtype);x=sample['noisy'].to(device=device,dtype=dtype)
    emb=sample['prompt_embeds'].to(device=device,dtype=dtype)
    frame_tokens=sample['history_tokens']//h.shape[1]
    cache=native_cache(wrapper,device,dtype);cross=[None]*len(cache)
    with torch.no_grad():
        for start in range(0,h.shape[1],3):
            wrapper(h[:,start:start+3],{'prompt_embeds':emb},torch.full((1,3),sample['context_noise'],device=device),
                    kv_cache=cache,crossattn_cache=cross,current_start=start*frame_tokens)
        return wrapper(x,{'prompt_embeds':emb},torch.full((1,3),sample['timestep'],device=device),
                       kv_cache=cache,crossattn_cache=cross,current_start=h.shape[1]*frame_tokens)[1]


def network_checks(wrapper,sample,layers,production=False):
    from .backbone import FunctionalBackbone
    from ..capture.capture_runner import sdpa_backend
    device=next(wrapper.parameters()).device;dtype=next(wrapper.parameters()).dtype
    wrapper.requires_grad_(False).eval()
    router=InformationRouter('V',pairs=len(layers)*wrapper.model.num_heads).to(device)
    student=copy.deepcopy(wrapper)
    expected_base=base_digest(wrapper.model)
    inject(student.model,layers=layers,seed=0)
    assert base_digest(student.model)==expected_base
    network=FunctionalBackbone(student,router,layers=layers)
    checks={}
    def progress(label):
        atomic_json(HERE/'reports/audit_progress.json',dict(stage=label,production=production,source_sha256=source_identity()))
        print('audit: '+label,flush=True)
    with sdpa_backend():
        progress('native replay and full-keep equivalence')
        original=native_replay(wrapper,sample)
        if 'target' in sample:
            target=sample['target'].to(device)
            target_error=float(nmse(original,target));target_abs=float((original-target).abs().max())
            checks['replayed_native_target_identity']=dict(nmse=target_error,max_abs=target_abs)
            tolerance=protocol()['audit_tolerances']
            assert target_error<=tolerance['full_bf16_nmse'] and target_abs<=tolerance['full_bf16_max_abs']
        with torch.no_grad():
            functional=FunctionalBackbone(wrapper,None,layers=layers).replay(sample,sparse=False,checkpoint=False)['x0']
            zero=network.replay(sample,sparse=False,checkpoint=False)['x0']
            allkeep=network.replay(sample,keep=1.,checkpoint=False,collect_aux=False)['x0']
        tol=protocol()['audit_tolerances']
        for name,pred in [('functional_vs_original',functional),('zero_lora_vs_original',zero),('all_keep_vs_original',allkeep)]:
            err=float(nmse(pred,original));absolute=float((pred-original).abs().max())
            checks[name]=dict(nmse=err,max_abs=absolute)
            assert err<=tol['full_bf16_nmse'] and absolute<=tol['full_bf16_max_abs'],(name,checks[name])
        configure(student.model,router,'J-V')
        progress('public x0, auxiliary isolation and recent history branch gradients')
        result=network.replay(sample,audit=True,collect_aux=True)
        loss=nmse(result['x0'],original)
        rp=[p for p in router.parameters() if p.requires_grad]
        lp=[p for _,p in lora_parameters(student.model)]
        rg=torch.autograd.grad(loss,rp,retain_graph=True,allow_unused=True)
        lg=torch.autograd.grad(loss,lp,retain_graph=True,allow_unused=True)
        rnorm=sum(float(g.float().square().sum()) for g in rg if g is not None)**.5
        lnorm=sum(float(g.float().square().sum()) for g in lg if g is not None)**.5
        assert rnorm>0 and lnorm>0
        checks['Lout_router_and_raw_attention_LoRA_gradient']=dict(router=rnorm,lora=lnorm)
        auxiliary=result['att_aux']+result['cal_aux']
        auxiliary_lora=torch.autograd.grad(auxiliary,lp,retain_graph=True,allow_unused=True)
        assert all(g is None or torch.equal(g,torch.zeros_like(g)) for g in auxiliary_lora)
        checks['cal_and_att_auxiliary_LoRA_gradient_zero']=True
        branch=[]
        for layer in layers:
            entry=result['history_cache'][layer];branch.extend((entry.recent_k,entry.recent_v))
        bg=torch.autograd.grad(loss,branch,retain_graph=True,allow_unused=True)
        branch_norm=[float(g.float().norm()) if g is not None else 0. for g in bg]
        assert all(v>0 for v in branch_norm)
        branch_to_lora=torch.autograd.grad(branch,lp,grad_outputs=[g if g is not None else torch.zeros_like(x) for x,g in zip(branch,bg)],
                                          retain_graph=True,allow_unused=True)
        write_norm=sum(float(g.float().square().sum()) for g in branch_to_lora if g is not None)**.5
        assert write_norm>0
        checks['future_to_recent_KV_and_write_LoRA']=dict(recent_kv=branch_norm,write_lora=write_norm)
        del result,loss,auxiliary,rg,lg,bg,branch_to_lora
        detached=network.replay(sample,recent_gradient=False,audit=True,collect_aux=False)
        assert all(not e.k.requires_grad and not e.v.requires_grad for e in detached['history_cache'])
        checks['H0_history_branch_has_no_autograd_edge']=True
        del detached
        # Exercise actual nonzero LoRA A/B paths after a nondegenerate update.
        progress('nonzero adapter paths and checkpoint recomputation')
        with torch.no_grad():
            for n,p in lora_parameters(student.model):
                if n.endswith('lora_B'):p.normal_(std=1e-3)
        post=network.replay(sample,collect_aux=False)
        pg=torch.autograd.grad(nmse(post['x0'],original),lp,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in pg)
        checks['after_nondegenerate_update_LoRA_A_B_gradients_finite_nonzero']=True
        del post,pg
        # Checkpoint mode must reproduce immutable-cache values and gradients.
        no_checkpoint=network.replay(sample,checkpoint=False,collect_aux=False)['x0']
        checkpointed=network.replay(sample,checkpoint=True,collect_aux=False)['x0']
        torch.testing.assert_close(no_checkpoint,checkpointed,atol=0,rtol=0)
        a=torch.autograd.grad(nmse(no_checkpoint,original),lp,allow_unused=True)
        b=torch.autograd.grad(nmse(checkpointed,original),lp,allow_unused=True)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-3)
        checks['checkpoint_recompute_no_cache_mutation']=True
        manifests={}
        for group in ('F-V','A-V','J-V'):
            manifest=configure(student.model,router,group)
            assert bool(manifest['router'])==(not group.startswith('A'))
            assert bool(manifest['backbone'])==group.startswith(('A','J'))
            manifests[group]=manifest
        rk=InformationRouter('K',pairs=len(layers)*wrapper.model.num_heads).to(device)
        for group in ('F-K','A-K','J-K'):
            manifest=configure(student.model,rk,group)
            assert bool(manifest['router'])==(not group.startswith('A'))
            assert bool(manifest['backbone'])==group.startswith(('A','J'))
            manifests[group]=manifest
        k_network=FunctionalBackbone(student,rk,layers=layers)
        progress('six freeze manifests and genuine K-only output gradient')
        k_result=k_network.replay(sample,collect_aux=False)
        parameters=list(rk.parameters())+[p for _,p in lora_parameters(student.model)]
        k_grads=torch.autograd.grad(nmse(k_result['x0'],original),parameters,allow_unused=True)
        assert any(g is not None and g.norm()>0 for g in k_grads[:len(list(rk.parameters()))])
        checks['RK_public_output_loss_has_router_gradient']=True
        progress('actual optimizer update invalidates KV and Sketch versions')
        with torch.no_grad():
            k_network.replay(sample,collect_aux=False)
            old_identity=k_network._eval_history[0]
            old_cache=k_network._eval_history[1]
        optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
        for parameter,gradient in zip(parameters,k_grads):parameter.grad=gradient
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():k_network.replay(sample,collect_aux=False)
        assert old_identity!=k_network._eval_history[0]
        assert old_cache[layers[0]].k.data_ptr()!=k_network._eval_history[1][layers[0]].k.data_ptr()
        checks['optimizer_step_rebuilds_KV_and_Sketch']=True
        checks['freeze_manifests']=manifests
        checks['frozen_base_digest_unchanged']=base_digest(student.model)==expected_base
        assert checks['frozen_base_digest_unchanged']
    checks['production_model']=production
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--real',action='store_true')
    parser.add_argument('--tiny-network',action='store_true');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();paired_seed(0)
    name='phase0_real.json' if args.real else 'phase0_unit.json'
    report=dict(source_sha256=source_identity(),passed=False,checks={},teacher_sha256=protocol()['teacher_sha256'],
                protocol_sha256=__import__('hashlib').sha256((HERE/'configs/protocol.json').read_bytes()).hexdigest())
    try:
        report['checks']['unit']=unit_checks()
        if args.tiny_network:
            wrapper=TinyWrapper().to(device=args.device,dtype=torch.bfloat16).eval()
            sample=dict(sample_id='synthetic',trajectory_id='synthetic',history_tokens=144,
                context_noise=0.,step=1,timestep=937.5,history=torch.randn(1,9,16,8,8),
                noisy=torch.randn(1,3,16,8,8),prompt_embeds=torch.randn(1,4,32))
            report['checks']['tiny_network']=network_checks(wrapper,sample,(0,1,2))
        if args.real:
            from .backbone import load_teacher
            from .replay import ReplayDataset
            from ..v04.data import SnapshotData
            ds=ReplayDataset('train')
            if not len(ds):raise RuntimeError('Collect at least one real training trajectory before real audit')
            raw=SnapshotData(protocol()['warmstart']['legacy_qkv_root'],'train')
            replay_checks=[]
            for pair in sorted(raw.strata):
                version=sorted(raw.strata[pair])[0]
                for event in raw.load(version,args.device):replay_checks.append(dict(pair=list(pair),step=event['step_id'],**raw_checks(args.device,event)))
                print('audit: raw pair '+str(pair)+' passed',flush=True)
            report['checks']['raw36pairs_all4steps']=replay_checks
            index=next(i for i,r in enumerate(ds.rows) if r['history_tokens']==max(protocol()['histories']))
            report['replay_fingerprint']=ds.fingerprint();report['replay_sample_id']=ds.rows[index]['sample_id']
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            report['checks']['real_network']=network_checks(load_teacher(args.device),ds.load(index),tuple(protocol()['layers']),True)
            report['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
        report['passed']=True
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        atomic_json(HERE/'reports'/name,report)
        print(json.dumps(dict(passed=report['passed'],report=str(HERE/'reports'/name),error=report.get('error')),ensure_ascii=False))


if __name__=='__main__':main()
