"""Fresh labels from actual student trajectories after downstream training."""
import argparse,json,os,time
from pathlib import Path
import torch
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .review_v1.model import Model
from .review_v1.observed_runtime import ObservedRuntime
from .review_v1.ops import AttentionOps,RouteOps
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/review_v1_20261102/expanded_probe/calibration')
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['city_walk','reef_fish'],required=True);ap.add_argument('--checkpoint',required=True);a=ap.parse_args()
    root=ROOT/a.case;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(summary_checkpoint=digest(a.checkpoint),source=digest(__file__),runtime=digest('local_experiments/review_v1/observed_runtime.py'),case=a.case,seed=202,chunks=[2,5,8],steps=[0,1,2,3],layers=list(range(1,30)),trajectory='student closed loop; dense teacher uses same student raw KV',probes=16)
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        engine=Engine();engine.wrapper.requires_grad_(False);m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['model'])
        def observe(l,q,k,v,s,selection):
            chunk=engine.current['chunk'];step=engine.current['step'];path=root/f'chunk{chunk}_layer{l}_step{step}.json'
            if chunk not in [2,5,8] or step==4 or path.exists():return
            tile=(17*l+13*chunk+7*step)%72;offset=torch.linspace(0,63,16,device=q.device).round().long();qs=q[:,tile*64:(tile+1)*64].index_select(1,offset);p=k.shape[1]
            target,zd=AttentionOps.evaluate(qs,k,v,s,m[l].basis,torch.ones(12,16,p,device=q.device,dtype=torch.bool))
            labels=AttentionOps.residual_labels(qs,k,v,s,m[l].basis,target,zd);f,_=RouteOps.features(qs,s)
            mask=selection['mask'][:,tile:tile+1].expand(-1,16,-1)
            out,z=AttentionOps.evaluate(qs,k,v,s,m[l].basis,mask)
            atomic(root/f'calibration_chunk{chunk}_layer{l}_step{step}.pt',dict(features=f.cpu(),labels=labels.cpu(),layer=l,chunk=chunk,step=step,tiles=torch.tensor([tile]),summary_identity=identity),True)
            atomic(path,dict(layer=l,chunk=chunk,step=step,pages=p,nmse=float((out-target).square().sum()/target.square().sum()),denominator_mae=float(((z-zd).exp()-1).abs().mean()),trajectory='student'))
        rt=ObservedRuntime(engine,m,observer=observe);emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda()
        for chunk in range(9):
            disk_guard(ROOT,50,2);path=root/f'clean{chunk}.pt'
            atomic(root/'status.json',dict(status='student_rollout',chunk=chunk,pid=os.getpid(),updated=time.time()))
            if path.exists():engine.forward(torch.load(path,map_location='cuda',weights_only=True),emb,chunk,4)
            else:atomic(path,generate_chunk(engine,emb,202,chunk).cpu(),True)
        assert len(list(root.glob('calibration_chunk*_layer*_step*.pt')))==348
        atomic(root/'complete.json',dict(status='complete',identity=identity))
    except BaseException as e:atomic(root/'status.json',dict(status='failed',error=repr(e)));raise
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
