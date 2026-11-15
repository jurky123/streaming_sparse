"""Only callable after all preregistered development gates pass."""
from .core import *
@torch.no_grad()
def main():
 gate=json.loads((E29/'test_gate.json').read_text());assert gate['passed'],'Test sealed: development gate failed'
 dest=E29/'test';dest.mkdir(exist_ok=True)
 assert not (dest/'started.json').exists(),'Test is a once-only frozen evaluation'
 dump(dest/'started.json',dict(gate_sha=sha(E29/'test_gate.json'),selected={str(s):sha(E29/f's{s}'/'selected.pt') for s in [17,29]}))
 for seed in [17,29]:
  model=LocalReader().cuda();model.load_state_dict(torch.load(E29/f's{seed}'/'selected.pt',map_location='cuda',weights_only=False)['model']);model.eval();phi=load_phi(seed);bundle=bundle_for('C',seed);rows=evaluate(model,phi,bundle,seed,files('test'));dump(dest/f's{seed}.json',rows)
 dump(dest/'complete.json',dict(seeds=[17,29],events_per_seed=576,model_selection_after_test=False))
if __name__=='__main__':main()
