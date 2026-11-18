"""Compare each clean grouping on identical real inputs along a saved baseline trajectory."""
import argparse,json
import torch
from .accepted_tiers import config
from .deployment import configure,load_weights,make_runtime
from .grouping_dataflow import GraphGrouping
from .weights import ROOT
from ..broad_common import Engine
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tier',choices=['base','step'],required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=config(a.tier,grouping_dataflow=False,grouping_softmax=False,local_assets='/tmp/streaming_sparse-0/runtime');configure(cfg)
 model,reader,phi,B,_=load_weights(cfg);engine=Engine();engine.wrapper.requires_grad_(False);rt=make_runtime(engine,model,reader,phi,B,cfg)
 emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda();root=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1'
 z=torch.load(root/'video'/(a.tier+'_old')/'latents.pt',weights_only=True).cuda();baseline=rt.group_graph;candidate=GraphGrouping(rt.config,contiguous_softmax=True);rows=[]
 def check(m,k,v):
  ref=baseline(m,k,v);got=candidate(m,k,v);n=int((ref[2]!=got[2]).sum())
  rows.append(dict(layer=next(l for l in range(1,30) if m is model[l]),chunk=engine.current['chunk'],different_token_positions=n,kv_equal=torch.equal(ref[0],got[0]) and torch.equal(ref[1],got[1])))
  return ref
 rt.group_graph=check
 for c in range(9):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 result=dict(tier=a.tier,events=len(rows),changed=sum(r['different_token_positions']>0 for r in rows),rows=rows,scope='same clean input and model for both groupers, saved baseline trajectory; arithmetic FP32 and stable sort retained, softmax layout/reduction changed')
 (root/(a.tier+'_order_audit.json')).write_text(json.dumps(result,indent=2)+'\n');print(result['events'],result['changed'],[r for r in rows if r['different_token_positions']][:5]);rt.close()
if __name__=='__main__':main()
