"""Profiler-only annotations; do not change tensors, math, or trainable state."""
import ast
import functools
import inspect
import textwrap
import torch

_restore=[]


def wrap(owner,name,label):
    original=getattr(owner,name)
    @functools.wraps(original)
    def call(*args,**kwargs):
        with torch.profiler.record_function(label):return original(*args,**kwargs)
    _restore.append((owner,name,original));setattr(owner,name,call)


def annotate(owner,name,mapping):
    original=getattr(owner,name)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Annotator(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            names={n.id for target in node.targets for n in ast.walk(target)if isinstance(n,ast.Name)}
            label=next((value for key,value in mapping.items()if key in names),None)
            if label:
                context=ast.parse(f'_profile_range({label!r})',mode='eval').body
                return ast.With(items=[ast.withitem(context_expr=context)],body=[node])
            return node
    tree=Annotator().visit(tree);ast.fix_missing_locations(tree)
    namespace={**original.__globals__,'_profile_range':torch.profiler.record_function}
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    _restore.append((owner,name,original));setattr(owner,name,namespace[original.__name__])


def install(network):
    from streaming_sparse.v08 import router,cache,attention,backbone
    from streaming_sparse.v08.execution import accelerate
    from streaming_sparse.v05.primitives import BudgetGate
    annotate(router.KSketch,'raw',dict(logits='router.prototype_logits',weights='router.prototype_reductions',
        raw='router.prototype_reductions',entropy='router.entropy',qnorm='router.features',direction='router.features',
        prototype_count='router.features',finite_logits='router.features',logit_mean='router.features',
        logit_var='router.features',kvar='router.directional_variance',features='router.features'))
    annotate(router.KSketch,'estimate',dict(norm='router.calibrator',logz='router.calibrator',
        current='router.current_logits',lc='router.current_reductions',wc='router.current_reductions',
        mu='router.prototype_content',oc='router.current_content',total='router.global_reference',
        mass='router.global_reference',mh='router.global_reference',mc='router.global_reference',
        dense='router.global_reference',residual='router.global_reference'))
    annotate(router.InformationRouter,'score',dict(n='router.residual_norm',projection='router.content_projection',
        scalar='router.head_features',feature='router.head_features',delta='router.scoring_head',scores='router.scoring_head'))
    wrap(router.InformationRouter,'build','router.cache_build')
    wrap(router.InformationRouter,'score','router.score')
    wrap(router.KSketch,'calibration_loss','aux.calibration')
    wrap(cache,'build_history','cache.history_build')
    wrap(accelerate,'page_select','router.hard_selection')
    # Function.apply expects static methods rather than bound class methods.
    original=BudgetGate.forward
    def gate(*args):
        with torch.profiler.record_function('router.soft_gate'):return original(*args)
    _restore.append((BudgetGate,'forward',staticmethod(original)));BudgetGate.forward=staticmethod(gate)
    original_back=attention.RawMultiAttention.backward
    source=textwrap.dedent(inspect.getsource(accelerate.multi_backward))
    tree=ast.parse(source)
    function=tree.body[0]
    def ranged(node,label):return ast.With(items=[ast.withitem(context_expr=ast.parse(f'_profile_range({label!r})',mode='eval').body)],body=[node])
    function.body=[ranged(node,'attention.native_qkv_backward')if isinstance(node,ast.With)else ranged(node,'attention.gate_vjp')if isinstance(node,ast.For)else node for node in function.body]
    ast.fix_missing_locations(tree);namespace={**accelerate.__dict__,'_profile_range':torch.profiler.record_function}
    exec(compile(tree,accelerate.__file__,'exec'),namespace)
    attention.RawMultiAttention.backward=staticmethod(namespace['multi_backward'])
    _restore.append((attention.RawMultiAttention,'backward',staticmethod(original_back)))
    wrap(attention,'_native_multi','attention.native_masked_forward')
    original_single=attention.RawAttention.backward
    def single(*args):
        with torch.profiler.record_function('aux.attention_backward'):return original_single(*args)
    _restore.append((attention.RawAttention,'backward',staticmethod(original_single)));attention.RawAttention.backward=staticmethod(single)
    for block in network.model.blocks:
        wrap(block.ffn,'forward','backbone.ffn')
        wrap(block.cross_attn,'forward','backbone.cross_attention')


def uninstall():
    for owner,name,original in reversed(_restore):setattr(owner,name,original)
    _restore.clear()


def summarize(profiler):
    events=list(profiler.events());forward={}
    backwards=[(e.time_range.start,e.time_range.end)for e in events if e.name=='train.backward']
    def ancestors(event):
        result=[]
        while event is not None:result.append(event);event=event.cpu_parent
        return result
    def direct(chain):
        for event in chain:
            if event.name.startswith(('router.','aux.','attention.','backbone.')):return event.name
        return None
    for event in events:
        if event.device_type==torch.autograd.DeviceType.CPU and event.sequence_nr>=0 and event.scope!=1:
            category=direct(ancestors(event))
            if category:forward[event.sequence_nr]=category
    groups={};total=0.
    for event in events:
        if event.device_type!=torch.autograd.DeviceType.CPU:continue
        microseconds=event.self_device_time_total
        if microseconds<=0:continue
        chain=ancestors(event);names=[parent.name for parent in chain]
        phase='backward'if any(a<=event.time_range.start<=b for a,b in backwards)else'forward'
        category=direct(chain)
        if category is None:
            category=next((forward[parent.sequence_nr]for parent in chain if parent.scope==1 and parent.sequence_nr in forward),None)
        if category is None:
            category='backbone.other'
        context='history'if 'cache.history_build'in names else'current'
        key=(phase,context,category);groups[key]=groups.get(key,0.)+microseconds/1e6;total+=microseconds/1e6
    rows=[dict(phase=p,context=c,component=n,gpu_seconds=value,gpu_percent=100*value/max(total,1e-12))for(p,c,n),value in sorted(groups.items(),key=lambda item:-item[1])]
    return dict(gpu_kernel_seconds=total,components=rows,
        accounting='Exclusive CUDA time attributed through CPU ranges and autograd sequence numbers; checkpoint recomputation is counted in backward; not additive to wall time.')
