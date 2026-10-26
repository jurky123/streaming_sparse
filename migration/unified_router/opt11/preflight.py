import argparse,json,time
from pathlib import Path
import torch
from ..common import config,load_raw,atomic_json,paired_seed
from .data import Builder,initial_head,load_group,attend,dense,labels
from .head import AlignedHead,exact,apply,subset,actions,surrogate
@torch.no_grad()
def checks(root,m):
 d=torch.load(root/'capture/full_query_rank0.pt',map_location='cpu',weights_only=True);row=d['row'];layer=d['layer'];q=d['q'].cuda();raw=load_raw(m['parent_experiment'],row,layer);ids=row['group_query_indices'];group=load_group(root,row,layer);assert torch.equal(group,q[:,ids])
 builder=Builder(m['parent_experiment'],layer,m['monitor_beta']);head=initial_head(m['parent_experiment'],layer);full,aux=builder.inputs(q,raw,row['step']);small,saux=builder.inputs(group,raw,row['step']);fullmask=apply(full,exact(head,full).argmax(-1));mask=apply(small,exact(head,small).argmax(-1));changed=int((fullmask[:,ids]!=mask).any(-1).sum());assert changed==0,changed
 sketch,prepared,state=saux;direct=builder.route.actions(group,sketch,small['score'],small['base_mask'],small['rem'],small['add'],m['monitor_beta'],group.new_tensor(float(row['step']),dtype=torch.float32),small['stats'],small['cv'],head);assert torch.equal(direct,mask)
 from ..opt9.skip_attention import attention as oldskip
 from .attention import attention as newattn
 from ..opt9.segmented_attention import attention as memory
 a=memory(group,raw['kh'],raw['vh'],raw['kc'],raw['vc'],mask,pack_v=True,fuse_pack=True)
 b=newattn(group,raw['kh'],raw['vh'],raw['kc'],raw['vc'],mask,pvfp16=False,threshold=-float('inf'));assert torch.equal(a,b)
 c=oldskip(group,torch.cat((raw['kh'],raw['kc']),1),torch.cat((raw['vh'],raw['vc']),1),mask,threshold=0.,pvfp16=True,warps=2)
 e=attend('Aggressive',group,raw,mask);assert torch.equal(c,e)
 for variant in ['Compact','Aggressive']:
  whole=attend(variant,q,raw,fullmask);part=attend(variant,group,raw,mask);assert torch.equal(whole[:,ids],part),(variant,'group semantics mismatch')
 truth=dense(group,raw);start=time.perf_counter();losses,energy,_=labels(head,small,group,raw,truth,row['target_query_positions']);torch.cuda.synchronize();elapsed=time.perf_counter()-start
 x=subset(small,row['target_query_positions']);model=AlignedHead(head.state_dict()).cuda();model.load_state_dict(head.state_dict())
 with torch.enable_grad():
  logits=model(x);real=exact(model,x);assert torch.equal(logits,real)
  objective=(logits.softmax(-1)*losses['Aggressive'].detach()).sum()/energy;objective.backward();grad=sum(float(p.grad.float().square().sum())for p in model.parameters());assert grad>0 and all(torch.isfinite(p.grad).all()for p in model.parameters())
 result=dict(passed=True,group_queries_equal_original=True,whole_vs_group_router_changed_queries=changed,logits_mask_equals_deployment=True,segmented_no_skip_equals_opt9=True,segmented_skip_fp16_equals_old_skip=True,group_attention_bit_identical_to_full=True,training_forward_bit_identical=True,finite_nonzero_gradient=True,gradient_squared_norm=grad,counterfactual_seconds_per_event_two_variants=elapsed,initial_expected_aggressive_nmse=float(objective),train_event=row['sample_id'],layer=layer)
 atomic_json(root/'preflight.json',result);print(json.dumps(result,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory);paired_seed(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 while True:
  f=root/'capture/rank0.json'
  if f.exists()and json.loads(f.read_text()).get('status')=='completed':break
  time.sleep(3)
 checks(root,m)
if __name__=='__main__':main()
