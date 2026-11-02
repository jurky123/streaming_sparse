"""Attribute eager summary clone/conversion shapes using real trained cache data."""
import torch,json,ast,inspect
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_input_runtime import GapInputRuntime
from . import gap_center_ops as module
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();rt=GapInputRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    try:
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        l=14;p=rt.n[l];cache=engine.cache[l];m=rt.model[l]
        k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128);v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k)
        tree=ast.parse(inspect.getsource(module.centered_pool))
        class Annotate(ast.NodeTransformer):
            def visit_Assign(self,node):
                label=ast.unparse(node)[:160]
                return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='mark',ctx=ast.Load()),args=[ast.Constant(value=label)],keywords=[]))],body=[node])
        tree=ast.fix_missing_locations(Annotate().visit(tree));env=module.__dict__.copy();env['mark']=torch.profiler.record_function
        exec(compile(tree,'<annotated pool>','exec'),env);fn=env['centered_pool']
        for _ in range(3):m.pool(k.float(),v.float(),cheap=True);m.covariance(k.float(),v.float())
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            with torch.profiler.record_function('input FP32 conversion'):kf=k.float();vf=v.float()
            fn(m,kf,vf,cheap=True)
            with torch.profiler.record_function('covariance'):m.covariance(kf,vf)
        torch.cuda.synchronize();rows=[]
        for e in prof.key_averages(group_by_input_shape=True):
            rows.append(dict(name=e.key,shapes=e.input_shapes,calls=e.count,device_us=e.device_time_total,self_device_us=e.self_device_time_total))
        rows.sort(key=lambda x:-x['device_us'])
        atomic('artifacts/gap_broad_opt_20261031/summary_eager_ops.json',dict(rows=rows))
        for x in rows:
            if 'copy' in x['name'] or 'clone' in x['name'] or x['name'] in ['input FP32 conversion','covariance'] or not (x['name'].startswith('aten::') or x['name'].startswith('void ')):
                print(json.dumps(x),flush=True)
    finally:rt.close()
if __name__=='__main__':main()
