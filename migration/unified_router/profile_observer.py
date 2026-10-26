"""Observer-only ranges; original source/checkpoints remain unchanged."""
import ast
import contextlib
import csv
import gzip
import inspect
import json
from pathlib import Path
import textwrap
import time
import types
from collections import defaultdict
import torch

from .common import atomic_json

PREFIX='SSPROF::'
_MISSING=object()


def annotated_block(function,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    class ExplicitSuper(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='super'and not node.args:
                node.args=[ast.Name(id='Network',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
            return node
    tree=ExplicitSuper().visit(tree)
    original_ast=ast.dump(tree,include_attributes=False)
    def annotate(body):
        result=[]
        for node in body:
            for attr in ['body','orelse','finalbody']:
                if isinstance(getattr(node,attr,None),list):setattr(node,attr,annotate(getattr(node,attr)))
            label=None
            if isinstance(node,ast.Assign):
                target=ast.unparse(node.targets[0])
                if target=='visible':label='runtime.mask_expand'
                elif target in ['raw_k','raw_v']:label='runtime.raw_kv_concat'
                elif target=='y':label='runtime.output_assembly'
                elif target=='count':label='runtime.counters'
                elif target=='(rh, rc)':label='runtime.scalar_sync'
            elif isinstance(node,ast.AugAssign):
                if ast.unparse(node.target).startswith(('retained[','nq')):label='runtime.counters'
            if label:
                context=ast.Call(func=ast.Name(id='_profile_scope',ctx=ast.Load()),args=[ast.Constant(label)],keywords=[])
                node=ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)],body=[node]),node)
            result.append(node)
        return result
    tree.body[0].body=annotate(tree.body[0].body)
    # Verify that removing our ranges recovers every original expression.
    import copy
    class RemoveScopes(ast.NodeTransformer):
        def visit_With(self,node):
            self.generic_visit(node)
            c=node.items[0].context_expr
            if isinstance(c,ast.Call)and isinstance(c.func,ast.Name)and c.func.id=='_profile_scope':return node.body
            return node
    assert ast.dump(RemoveScopes().visit(copy.deepcopy(tree)),include_attributes=False)==original_ast
    ns=namespace.copy();exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(function),'exec'),ns)
    return ns[function.__name__]


class Observer:
    def __init__(self,net,directory):
        self.net=net;self.directory=Path(directory);self.restores=[]
        self.recording=False;self.cross=False;self.phase='denoise';self.prof=None
        self.forwards=[];self.sdpa_calls=[]

    def scope(self,name):
        return torch.profiler.record_function(PREFIX+name)if self.recording else contextlib.nullcontext()

    def replace(self,obj,name,value):
        self.restores.append((obj,name,obj.__dict__.get(name,_MISSING)));setattr(obj,name,value)

    def wrap(self,obj,name,label):
        old=getattr(obj,name)
        def wrapped(*args,**kwargs):
            if not self.recording:return old(*args,**kwargs)
            with self.scope(label):return old(*args,**kwargs)
        self.replace(obj,name,wrapped)

    def __enter__(self):
        from . import network,core
        import streaming_sparse.v08.rollout as rollout
        import wan.modules.causal_model as cm
        import torch.nn.functional as F
        for name,label in [('build','router.build'),('scores','router.score'),('initial_mask','router.threshold'),
                ('candidates','router.candidates'),('features','router.features'),('choose','router.choose'),('apply_action','router.apply')]:
            self.wrap(network,name,label)
        part=core.part
        def prototype(*args,**kwargs):
            if not self.recording:return part(*args,**kwargs)
            p=args[2].shape[1]
            with self.scope('router.prototype.current_P16'if p==16 else f'router.prototype.history_P{p}'):
                return part(*args,**kwargs)
        self.replace(core,'part',prototype)
        self.wrap(cm,'causal_rope_apply','backbone.rope')
        sdpa=F.scaled_dot_product_attention
        def attention(q,k,v,*args,**kwargs):
            if not self.recording:return sdpa(q,k,v,*args,**kwargs)
            kind='cross'if self.cross else 'self'
            mask=kwargs.get('attn_mask')
            self.sdpa_calls.append(dict(kind=kind,q_shape=list(q.shape),k_shape=list(k.shape),
                v_shape=list(v.shape),mask_shape=list(mask.shape)if mask is not None else None,dtype=str(q.dtype)))
            with self.scope('backbone.cross_kernel'if self.cross else 'backbone.self_attention'):
                return sdpa(q,k,v,*args,**kwargs)
        self.replace(F,'scaled_dot_product_attention',attention)
        for block in self.net.model.blocks:
            for name in ['q','k','v']:self.wrap(getattr(block.self_attn,name),'forward','backbone.qkv_projection')
            self.wrap(block.self_attn.o,'forward','backbone.output_projection')
            for obj in [block.norm1,block.norm2,block.norm3,block.self_attn.norm_q,block.self_attn.norm_k]:
                self.wrap(obj,'forward','backbone.norm')
            self.wrap(block.ffn,'forward','backbone.ffn')
            old=block.cross_attn.forward
            def cross_factory(old):
                def cross(*args,**kwargs):
                    if not self.recording:return old(*args,**kwargs)
                    previous=self.cross;self.cross=True
                    try:
                        with self.scope('backbone.cross_attention'):return old(*args,**kwargs)
                    finally:self.cross=previous
                return cross
            self.replace(block.cross_attn,'forward',cross_factory(old))
        for head in self.net.router.heads.values():self.wrap(head,'forward','router.mlp')
        # Clone syntax, adding only scopes. Explicit super preserves fallback.
        ns=network.__dict__.copy();ns['_profile_scope']=self.scope
        cloned=types.MethodType(annotated_block(network.Network._block,ns),self.net)
        old_block=self.net._block
        def block(*args,**kwargs):
            if args[1]!=24 or args[9]!=9 or args[11]!=3:return old_block(*args,**kwargs)
            assert self.prof is None,'Target block was unexpectedly repeated'
            torch.cuda.synchronize()
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False)
            with profiler:
                self.recording=True
                try:
                    with self.scope('block.total'):result=cloned(*args,**kwargs)
                finally:self.recording=False
            self.prof=profiler
            return result
        self.replace(self.net,'_block',block)
        original_history=rollout.build_history
        def history(*args,**kwargs):
            previous=self.phase;self.phase='context'
            try:return original_history(*args,**kwargs)
            finally:self.phase=previous
        self.replace(rollout,'build_history',history)
        original_forward=self.net.forward_chunk
        def forward(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            result=original_forward(*args,**kwargs)
            torch.cuda.synchronize()
            self.forwards.append(dict(phase=self.phase,start_frame=int(args[4]),step=kwargs.get('step',0),
                seconds=time.perf_counter()-start,contains_profiler=self.phase=='denoise'and args[4]==9 and kwargs.get('step')==3))
            return result
        self.replace(self.net,'forward_chunk',forward)
        return self

    def __exit__(self,*exc):
        self.recording=False
        for obj,name,old in reversed(self.restores):
            if old is _MISSING:delattr(obj,name)
            else:setattr(obj,name,old)
        self.restores=[]

    def export(self):
        assert self.prof is not None,'Target block was not profiled'
        trace=self.directory/'trace.json';self.prof.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents'];kernels=defaultdict(lambda:[0,0.])
        gpu_copy_us=0.
        for event in events:
            if event.get('cat')=='kernel':
                entry=kernels[event['name']];entry[0]+=1;entry[1]+=event.get('dur',0.)
            elif event.get('cat')in ['gpu_memcpy','gpu_memset']:gpu_copy_us+=event.get('dur',0.)
        assert kernels,'No CUDA kernel events captured'
        kernel_rows=sorted([dict(name=k,count=v[0],cuda_ms=v[1]/1000)for k,v in kernels.items()],key=lambda r:-r['cuda_ms'])
        averages=self.prof.key_averages(group_by_input_shape=True)
        ops=[dict(name=e.key,count=e.count,cpu_ms=e.cpu_time_total/1000,self_cpu_ms=e.self_cpu_time_total/1000,
            cuda_ms=e.device_time_total/1000,self_cuda_ms=e.self_device_time_total/1000,
            self_cuda_memory_bytes=e.self_device_memory_usage,input_shapes=e.input_shapes)for e in averages if e.device_type==torch.autograd.DeviceType.CPU]
        modules=[dict(name=e.key.removeprefix(PREFIX),count=e.count,cpu_ms=e.cpu_time_total/1000,
            self_cpu_ms=e.self_cpu_time_total/1000,cuda_ms=e.device_time_total/1000,
            self_cuda_memory_bytes=e.self_device_memory_usage)for e in self.prof.key_averages()if e.key.startswith(PREFIX)and e.device_type==torch.autograd.DeviceType.CPU]
        atomic_json(self.directory/'module_profile.json',dict(scope='Layer24, fourth chunk, step3; profiler pass, not clean speed timing',
            ast_scopes_only_verified=True,
            modules=modules,kernels=kernel_rows,operators=ops,attention_shapes=self.sdpa_calls,
            kernel_sum_ms=sum(r['cuda_ms']for r in kernel_rows),gpu_copy_ms=gpu_copy_us/1000,
            coarse_forwards=self.forwards,coarse_warning='Forward boundary synchronizations and one traced block perturb this pass.',
            memory_warning='Operator memory columns are cumulative allocation deltas, not peak VRAM.'))
        for name,rows in [('modules',modules),('kernels',kernel_rows),('operators',sorted(ops,key=lambda x:-x['self_cuda_ms']))]:
            with(self.directory/f'{name}.csv').open('w')as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with trace.open('rb')as source,gzip.open(self.directory/'trace.json.gz','wb',compresslevel=3)as target:
            import shutil
            shutil.copyfileobj(source,target)
        trace.unlink()
