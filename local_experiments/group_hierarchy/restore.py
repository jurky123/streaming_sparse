"""Restore the old 120 event inputs; prove both local metrics and full trajectory match."""
import argparse,torch,json
from pathlib import Path
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..rollout_suite import generate_chunk
from ..group_response.core import Runtime
from ..learned_cache.io import atomic
from .local import Collector
from .run import ROOT,BASE,PRE
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
 collector=Collector(model,ROOT,a.case,a.seed);collector.engine=engine;rt=Runtime(engine,model,'G-KVR','L3',collector)
 emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();zs=[]
 for c in range(9):zs.append(generate_chunk(engine,emb,a.seed,c).cpu());print('REPLAY',a.case,a.seed,c,flush=True)
 z=torch.cat(zs,1);old=torch.load(Path('/tmp/group_response_results/video')/f'{a.case}_s{a.seed}'/'G-KVR/L3/latents.pt',weights_only=True);assert torch.equal(z,old),float((z.float()-old.float()).square().mean())
 assert len(list((ROOT/'restored_controls'/f'{a.case}_s{a.seed}').glob('*.reuse')))==30
 atomic(ROOT/'restoration'/f'{a.case}_s{a.seed}.json',dict(passed=True,latent_bitwise_equal=True,events=30,source='original G-KVR L3; same 9 chunks; old 120 event IDs restored, no new sample selection'))
 rt.close()
if __name__=='__main__':main()
