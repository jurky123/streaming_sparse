"""Explicit architecture/run-root amendment; retain original objective and math."""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from streaming_sparse.v08.common import atomic_json,source_identity,sha,protocol
from streaming_sparse.v08.router import InformationRouter as BaseRouter
from streaming_sparse.v08.backbone import FunctionalBackbone as BaseNetwork,load_teacher
from streaming_sparse.v08.lora import inject


def identity():return sha(__file__)


def configuration(root,stage):
    root=Path(root).resolve();manifest=json.loads((root/'experiment.json').read_text())
    assert manifest['source_sha256']==source_identity()
    spec=manifest['stages'][stage];p=Path(spec['directory'])/'configs/protocol.json'
    assert sha(p)==spec['protocol_sha256']
    return manifest,spec,json.loads(p.read_text())


def load_checkpoint(path,layers):
    st=torch.load(path,map_location='cpu',weights_only=True)
    assert st['source_sha256']==source_identity()and st['group'].startswith('F-')
    wrapper=load_teacher('cuda');inject(wrapper.model,layers,st['seed'])
    parameters=dict(wrapper.model.named_parameters())
    with torch.no_grad():
        for n,v in st['lora'].items():parameters[n].copy_(v)
    wrapper.requires_grad_(False)
    router=BaseRouter(st['information'],pairs=len(layers)*12).cuda();router.load_state_dict(st['router'])
    network=BaseNetwork(wrapper,router,layers=layers).eval();network.model_version=sha(path);network.router_version=network.model_version
    return network,st


def install_training(root,stage,group):
    from streaming_sparse.v08 import train
    manifest,spec,p=configuration(root,stage);directory=Path(spec['directory']);layers=spec['layers']
    train.HERE=directory;train.protocol=lambda:copy.deepcopy(p)
    train.InformationRouter=lambda kind:BaseRouter(kind,pairs=len(layers)*12)
    train.FunctionalBackbone=lambda wrapper,router:BaseNetwork(wrapper,router,layers=layers)
    # Add a declared paired stopping decision after a complete validation; final
    # snapshots must use the actual stop step, while the LR cap stays4000.
    tree=ast.parse(inspect.getsource(train.run));count=0
    class Change(ast.NodeTransformer):
        def visit_If(self,node):
            nonlocal count
            self.generic_visit(node)
            if ast.unparse(node.test)=='step % validate_every == 0 or step == updates':
                node.body+=ast.parse('if _paired_stop(step,group,directory,curve):\n    updates=step\n    break').body;count+=1
            return node
    tree=Change().visit(tree);ast.fix_missing_locations(tree);assert count==1
    train.__dict__['_paired_stop']=lambda step,group,directory,curve:paired_stop(root,stage,step,group,directory,curve)
    generated=directory/'train_budget.generated.py';text=ast.unparse(tree)+'\n'
    tmp=generated.with_name(generated.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(generated)
    exec(compile(text,str(generated),'exec'),train.__dict__)
    return manifest,spec


def paired_stop(root,stage,step,group,directory,curve):
    manifest,spec,_=configuration(root,stage)
    if step<spec['min_updates']:return False
    rank=dist.get_rank();result=[None];root=Path(root);stage_dir=Path(spec['directory'])
    if rank==0:
        old=min(r['prompt_equal_error']for r in curve[:-3]);recent=min(r['prompt_equal_error']for r in curve[-3:])
        relative=max(0.,old-recent)/max(old,1e-12);plateau=relative<spec['plateau_relative_improvement']
        marker=stage_dir/'paired_convergence'/f'{step:06d}.{group}.json'
        atomic_json(marker,dict(updates=step,group=group,old_best=old,recent_best=recent,relative_improvement=relative,plateau=plateau))
        other=stage_dir/'paired_convergence'/f'{step:06d}.{"F-V"if group=="F-K"else"F-K"}.json'
        start=time.monotonic()
        while not other.exists():
            if(root/'STOP').exists():raise RuntimeError('User stop requested while awaiting paired validation')
            if time.monotonic()-start>6*3600:raise TimeoutError('Other paired training did not reach the same validation boundary')
            atomic_json(directory/'paired_wait.json',dict(status='waiting_for_paired_validation',step=step,seconds=time.monotonic()-start))
            time.sleep(2)
        other_value=json.loads(other.read_text());stop=(plateau and other_value['plateau'])or step>=spec['max_updates']
        result[0]=stop
        if stop:atomic_json(directory/'stop_decision.json',dict(updates=step,both_plateau=plateau and other_value['plateau'],cap_reached=step>=spec['max_updates'],criterion=manifest['convergence']))
    dist.broadcast_object_list(result,src=0);return result[0]


PREFIXES=['sketch.assignment.','sketch.calibrator.networks.','content.','norm.','head.networks.']


def copy_pair(destination,source,old,new):
    for prefix in PREFIXES:
        for name,value in source.items():
            marker=prefix+str(old)+'.'
            if name.startswith(marker):destination[prefix+str(new)+'.'+name[len(marker):]].copy_(value)
    for name in ['scale','sketch.feature_mean','sketch.feature_std','sketch.feature_count']:
        destination[name][new].copy_(source[name][old])
