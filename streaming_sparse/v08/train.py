"""Paired F/A/J training with fixed replay x0 targets and separate parameter paths."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
from .common import HERE,protocol,paired_seed,source_identity,atomic_json,save,sha,output,require_audit,nmse,tensor_digest
from .replay import ReplayDataset,BalancedReplay
from .router import InformationRouter
from .lora import inject,configure,lora_parameters,base_digest,effective_update_ratios
from .backbone import FunctionalBackbone,load_teacher
from .evaluate import evaluate


class PilotView:
    """Temporary optimization check, complete inputs but a declared small subset."""
    def __init__(self,ds,count=8):
        self.ds=ds;self.split=ds.split
        self.indices=torch.linspace(0,len(ds)-1,min(count,len(ds))).long().unique().tolist()
        self.rows=[ds.rows[i] for i in self.indices]
    def __len__(self):return len(self.indices)
    def load(self,i):return self.ds.load(self.indices[i])
    def fingerprint(self):
        import hashlib
        return hashlib.sha256(json.dumps([self.ds.fingerprint(),self.indices]).encode()).hexdigest()


def run(stage,seed,kind=None,group=None,device='cuda',pilot=False,pilot_updates=4,resume=None,diagnostic=None):
    require_audit();paired_seed(seed);p=protocol()
    if stage not in ('heads','main'):raise ValueError(stage)
    recent_gradient=diagnostic!='H0';router_input_gradient=diagnostic=='R1'
    if diagnostic:
        effect=json.loads((HERE/'reports/final_test.json').read_text())['comparisons']['best']['20']
        if not effect['investment_gate'] or effect['I']['prompt_paired_95CI'][0]<=0:
            raise ValueError('No locked stable main interaction effect; H0/R1 remain disabled')
        if stage!='main' or group not in ('J-K','J-V'):raise ValueError('Diagnostics only compare J-K/J-V')
        if diagnostic=='R1' and not pilot:raise ValueError('R1 is a separately labelled small-scale diagnostic')
    if stage=='heads':
        if kind not in ('K','V'):raise ValueError('Head warm-start needs information K/V')
        group='F-'+kind
    else:
        if group not in p['groups']:raise ValueError('Choose F-K/F-V/A-K/A-V/J-K/J-V')
        kind=group[-1]
    train=ReplayDataset('train',require_complete=not pilot)
    val=ReplayDataset('val',require_complete=not pilot)
    if not len(train) or not len(val):raise ValueError('Collect real Train and Val replay inputs first')
    if pilot:train=PilotView(train);val=PilotView(val)
    root=HERE/'diagnostics'/diagnostic if diagnostic else HERE/'diagnostics' if pilot else HERE/'runs'
    directory=output(root/f'seed{seed}'/(f'warmstart_{kind}' if stage=='heads' else group))
    if any(directory.iterdir()) and resume is None:raise FileExistsError(directory)
    wrapper=load_teacher(device);initial_base=base_digest(wrapper.model)
    inject(wrapper.model,tuple(p['layers']),seed)
    assert base_digest(wrapper.model)==initial_base
    router=InformationRouter(kind).to(device)
    if stage=='heads':
        parent=HERE/f'runs/seed{seed}/calibration/best.pt'
        state=torch.load(parent,map_location='cpu',weights_only=True)
        if state['source_sha256']!=source_identity() or state['kind']!='calibration':raise ValueError('Incompatible calibration')
        router.load_state_dict(state['router']);router.reset_residual()
    else:
        parent=HERE/f'runs/seed{seed}/warmstart_{kind}/best.pt'
        state=torch.load(parent,map_location='cpu',weights_only=True)
        if state['source_sha256']!=source_identity() or state['stage']!='heads':raise ValueError('Incompatible warm-start')
        router.load_state_dict(state['router'])
        params=dict(wrapper.model.named_parameters())
        with torch.no_grad():
            for n,v in state['lora'].items():params[n].copy_(v)
    if not router.sketch.normalization_locked:raise ValueError('Train-only normalization has not been locked')
    manifest=configure(wrapper.model,router,group)
    if stage=='heads':
        router.sketch.requires_grad_(False)
        manifest['router']=[dict(name=n,shape=list(v.shape),count=v.numel()) for n,v in router.named_parameters() if v.requires_grad]
    network=FunctionalBackbone(wrapper,router).eval()
    settings=p['main'];updates=pilot_updates if pilot else settings['updates'] if stage=='main' else p['warmstart']['residual_updates']
    validate_every=1 if pilot else settings['validate_every']
    learning_router=settings['router_lr'] if stage=='main' else p['warmstart']['residual_lr']
    rparams=[v for v in router.parameters() if v.requires_grad]
    lparams=[v for _,v in lora_parameters(wrapper.model) if v.requires_grad]
    groups=[]
    if rparams:groups.append(dict(params=rparams,lr=learning_router,initial_lr=learning_router,name='router'))
    if lparams:groups.append(dict(params=lparams,lr=settings['lora_lr'],initial_lr=settings['lora_lr'],name='lora'))
    optimizer=torch.optim.AdamW(groups,weight_decay=settings['wd'])
    schedule=BalancedReplay(train,seed)
    frozen_versions={n:v._version for n,v in wrapper.model.named_parameters() if not v.requires_grad}
    frozen_router_digest=tensor_digest(router.state_dict().items()) if group.startswith('A') else None
    normalizer_digest=tensor_digest((n,b) for n,b in router.sketch.named_buffers())
    initial_lora={n:v.detach().clone() for n,v in lora_parameters(wrapper.model)}
    provenance=dict(source_sha256=source_identity(),teacher_sha256=p['teacher_sha256'],base_digest=initial_base,
        stage=stage,group=group,information=kind,seed=seed,pilot=pilot,updates=updates,settings=settings,
        initialization_checkpoint=str(parent),initialization_sha256=sha(parent),
        initial_router_digest=tensor_digest(router.state_dict().items()),
        initial_lora_digest=tensor_digest(initial_lora.items()),normalizer_digest=normalizer_digest,
        train_fingerprint=train.fingerprint(),val_fingerprint=val.fingerprint(),trainable_manifest=manifest,
        recent_history_gradient=recent_gradient,router_input_stop_gradient=not router_input_gradient,diagnostic=diagnostic,
        objective_parameterization='x0',torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if device.startswith('cuda') else device,
        protocol_sha256=sha(HERE/'configs/protocol.json'))
    provenance['allocator_conf']=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    atomic_json(directory/'provenance.json',provenance)
    best=float('inf');best_step=0;curve=[];begin=0
    if resume:
        saved=torch.load(resume,weights_only=True,map_location='cpu')
        for key in ('source_sha256','group','seed','stage'):
            expected=source_identity() if key=='source_sha256' else provenance[key]
            if saved[key]!=expected:raise ValueError(f'Resume {key} mismatch')
        router.load_state_dict(saved['router']);params=dict(wrapper.model.named_parameters())
        with torch.no_grad():
            for n,v in saved['lora'].items():params[n].copy_(v)
        optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['sampler'])
        begin=saved['updates'];best=saved['best'];best_step=saved['best_step'];curve=saved['curve']
        frozen_versions={n:v._version for n,v in wrapper.model.named_parameters() if not v.requires_grad}
    def snapshot(step):
        return dict(source_sha256=source_identity(),teacher_sha256=p['teacher_sha256'],base_digest=initial_base,
            seed=seed,information=kind,group=group,stage=stage,pilot=pilot,updates=step,
            router=router.state_dict(),lora={n:v.detach().cpu() for n,v in lora_parameters(wrapper.model)},
            optimizer=optimizer.state_dict(),sampler=schedule.state_dict(),coverage=schedule.coverage(),
            best=best,best_step=best_step,curve=curve,normalizer_digest=normalizer_digest,
            provenance=provenance,torch_rng=torch.get_rng_state())
    def verify_frozen():
        if any(v._version!=frozen_versions[n] for n,v in wrapper.model.named_parameters() if n in frozen_versions):
            raise AssertionError('A frozen backbone parameter was modified')
        if frozen_router_digest and tensor_digest(router.state_dict().items())!=frozen_router_digest:
            raise AssertionError('A-group Router changed')
        if tensor_digest((n,b) for n,b in router.sketch.named_buffers())!=normalizer_digest:
            raise AssertionError('Train normalization drift')
    def validation(step):
        nonlocal best,best_step
        verify_frozen()
        network.model_version=f'{group}.seed{seed}.update{step}'
        network.router_version=network.model_version
        metrics=evaluate(network,val,directory/f'validation/update{step:06d}',.2)
        score=metrics['prompt_equal_error'];curve.append(dict(updates=step,**metrics))
        if score<best:
            best=score;best_step=step;save(directory/'best.pt',snapshot(step))
            atomic_json(directory/'coverage_best.json',schedule.coverage())
        atomic_json(directory/'validation_curve.json',curve)
        atomic_json(directory/'status.json',dict(status='running',stage=stage,group=group,updates=step,total=updates,
                    best=best,best_step=best_step,pilot=pilot,lora_update_to_base_norm=effective_update_ratios(wrapper.model)))
        save(directory/'latest.pt',snapshot(step))
    from ..capture.capture_runner import sdpa_backend
    with sdpa_backend():
        if begin==0:save(directory/'initial.pt',snapshot(0));validation(0)
        for step in range(begin+1,updates+1):
            optimizer.zero_grad(set_to_none=True)
            progress=min(1.,max(0.,(step-settings['warmup'])/max(1,updates-settings['warmup'])))
            multiplier=min(1.,step/settings['warmup'])*(.05+.95*(1+math.cos(math.pi*progress))/2)
            for pg in optimizer.param_groups:pg['lr']=pg['initial_lr']*multiplier
            totals=dict(output=0.,anchor=0.,att_aux=0.,cal_aux=0.)
            examples=schedule.sample(settings['accumulation'])
            for index in examples:
                sample=train.load(index)
                result=network.replay(sample,collect_aux=bool(rparams) and stage=='main',checkpoint=True,
                    recent_gradient=recent_gradient,detach_router=not router_input_gradient)
                expected=p['heads']*len(p['layers'])*p['frames_per_chunk']*p['frame_tokens']
                if int(result['diagnostics'][1])!=expected:
                    raise AssertionError('Not every current Query/pair was routed')
                target=sample['target'].to(device)
                out_loss=nmse(result['x0'],target)
                att_weight=.1*max(0.,1-(step-1)/p['loss']['att_decay_updates']) if stage=='main' else 0.
                cal_weight=p['loss']['cal_weight'] if rparams and stage=='main' else 0.
                objective=out_loss+att_weight*result['att_aux']+cal_weight*result['cal_aux']
                if not torch.isfinite(objective):raise FloatingPointError('Network output objective is not finite')
                (objective/settings['accumulation']).backward()
                for key,value in [('output',out_loss),('att_aux',result['att_aux']),('cal_aux',result['cal_aux'])]:
                    totals[key]+=float(value.detach())/settings['accumulation']
                del result,objective,out_loss
                if stage=='main' and step%p['loss']['anchor_every']==0:
                    dense=network.replay(sample,sparse=False,checkpoint=True,collect_aux=False,recent_gradient=recent_gradient)
                    anchor=nmse(dense['x0'],target)
                    weight=p['loss']['anchor_weight']*p['loss']['anchor_multiplier']/settings['accumulation']
                    if anchor.requires_grad:(weight*anchor).backward()
                    totals['anchor']+=float(anchor.detach())/settings['accumulation']
                    del dense,anchor
            norms={}
            for label,params in [('router',rparams),('lora',lparams)]:
                norms[label]=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)) if params else 0.
            optimizer.step();verify_frozen()
            network.model_version=f'{group}.seed{seed}.update{step}';network.router_version=network.model_version
            item=dict(updates=step,losses=totals,gradient_norm_before_clip=norms,inputs=[train.rows[i]['sample_id'] for i in examples],
                lrs={pg['name']:pg['lr'] for pg in optimizer.param_groups},recent_history_gradient=recent_gradient,
                router_input_stop_gradient=not router_input_gradient)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            if step%validate_every==0 or step==updates:validation(step)
        verify_frozen()
        final_base=base_digest(wrapper.model)
        if final_base!=initial_base:raise AssertionError('Frozen base hash changed')
        save(directory/'last.pt',snapshot(updates));atomic_json(directory/'coverage_last.json',schedule.coverage())
        lora_relative={n:float((v-initial_lora[n]).norm()/initial_lora[n].norm().clamp_min(1e-8)) for n,v in lora_parameters(wrapper.model)}
        atomic_json(directory/'completed.json',dict(source_sha256=source_identity(),stage=stage,group=group,seed=seed,
            updates=updates,best_step=best_step,best=best,base_digest_before=initial_base,base_digest_after=final_base,
            lora_relative_parameter_change=lora_relative,pilot=pilot,
            lora_update_to_base_norm=effective_update_ratios(wrapper.model),
            note='Pilot weights are temporary and cannot initialize formal experiments' if pilot else 'No early stopping; best and fixed last retained'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['heads','main'],required=True)
    p.add_argument('--kind',choices=['K','V']);p.add_argument('--group',choices=protocol()['groups'])
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--pilot',action='store_true');p.add_argument('--pilot-updates',type=int,default=4)
    p.add_argument('--resume')
    p.add_argument('--diagnostic',choices=['H0','R1'])
    a=p.parse_args();run(a.stage,a.seed,a.kind,a.group,a.device,a.pilot,a.pilot_updates,a.resume,a.diagnostic)


if __name__=='__main__':main()
