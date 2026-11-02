"""Real closed-loop audit, separately from acceptance speed measurements."""
import gc,json
from pathlib import Path
import torch
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .improve_runtime import FastRuntime
from .compensation_runtime import CompensationRuntime
from .compensation_probe import ROOT
from .learned_cache.io import atomic,digest,RunLock
@torch.no_grad()
def main():
    root=ROOT/'deployment';lock=RunLock(root)
    try:
        selection=json.loads((ROOT/'selection.json').read_text())
        identity=dict(model=digest(ROOT/'model.pt'),selection=digest(ROOT/'selection.json'),code={s:digest(Path('local_experiments')/s) for s in ['compensation_runtime.py','compensation_kernels.py','compensation_deployment_checks.py']})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'gate.json').exists():assert json.loads((root/'gate.json').read_text())['passed'];return
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        engine=Engine();emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();first=None
        for name in ['baseline','zero','raw']:
            if (root/f'{name}.json').exists():continue
            rt=FastRuntime(engine) if name=='baseline' else CompensationRuntime(engine,ROOT/'model.pt',selection,mode=name,audit=name=='raw')
            try:
                rt.reset();latents=[]
                for c in range(41):latents.append(generate_chunk(engine,emb,1101,c).cpu())
                value=torch.cat(latents,1);assert torch.isfinite(value).all()
                if name=='baseline':atomic(root/'baseline.pt',value,True)
                if name=='zero':assert torch.equal(value,torch.load(root/'baseline.pt',weights_only=True)),'zero-strength must preserve full video exactly'
                if name=='raw':
                    ref=torch.load(root/'baseline.pt',weights_only=True);assert torch.equal(value[:,:3],ref[:,:3])
                    assert len(rt.comp_checks)==29*4*2,len(rt.comp_checks)
                result=rt.finish()|dict(variant=name,passed=True,full_latents_exact_to_baseline=name in ['baseline','zero'])
                atomic(root/f'{name}.json',result);print(json.dumps(dict(variant=name,passed=True,checks=len(result.get('compensation_checks',[])))),flush=True)
            finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
        atomic(root/'gate.json',dict(passed=True,identity=identity,checks='Full41 chunks gamma0 bit-exact; raw compensation finite and 232 real query-tile checks across all29 layers, all12 heads, chunks2/5/20/40 steps0/3. These checks do not imply video quality improves.'))
    finally:lock.close()
if __name__=='__main__':main()
