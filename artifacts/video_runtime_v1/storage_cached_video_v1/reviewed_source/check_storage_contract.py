"""Check stored values, FP64 consumer boundary, all scores and accepted actions."""
import json
import torch
from .weights import ROOT,load
from .deployment import current,configure
from . import reader_reduce as rr
from .risk_projected import risk
from .logmass_cached import normalized
from .tensor_bmm import bf16_inputs_bmm
from .scores import candidate_pool
from .profile_route import capture
from ..review_v1.ops import Summary,RouteOps,Config
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;configure(current())
 _,_,phi,_,_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
 e=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True);q=e['q'].cuda().index_select(2,e['offsets'].cuda());s=Summary(*(e['summary'][n].cuda() for n in ('keys','mean','detail','stats')));en={k:v.cuda() for k,v in e['en'].items()};h,t,m,d=q.shape;p=s.mean.shape[1];b=h*t
 z=bf16_inputs_bmm(q.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,m,p,4)/(d**.5);lm=normalized(z);u=phi.query(q);rv=risk(q,z,s,e['theta'].cuda());own=e['own'].cuda();mask=RouteOps.select(rv,own,Config(),check_valid=False)['mask'];r,a=candidate_pool(rv,mask,own)
 mapping=torch.full((b,p),-1,device='cuda',dtype=torch.int32);mapping=mapping.scatter(1,r.reshape(b,16),torch.arange(16,device='cuda',dtype=torch.int32)[None].expand(b,-1));mapping=mapping.scatter(1,a.reshape(b,32),torch.arange(16,48,device='cuda',dtype=torch.int32)[None].expand(b,-1))
 rows={};base=None;score_base=None
 for name in ['baseline','output32','coeff','den','combined','combined_cached']:
  rr.OUTPUT32=name in ['output32','combined','combined_cached'];rr.ENDPOINT_DEN=name in ['den','combined','combined_cached'];enc={k:(v.bfloat16() if name in ['coeff','combined','combined_cached'] and k in ['co','A'] else v) for k,v in en.items()}
  from . import endpoint_den
  endpoint_den.CACHED_GROUPS=name=='combined_cached'
  fields=rr.reader_reduce(u,enc['co'],enc['A'],enc['mean'],enc['detail'],z,lm,mask.reshape(b,p),mapping,48,128,4,True,1,False,rr.OUTPUT32)
  if base is None:base=tuple(x.clone() for x in fields)
  raw_equal=[torch.equal(x.double(),y.double()) for x,y in zip(base,fields)]
  fn=torch.compile(lambda:rr.compact_scores(u,enc,z,lm,mask,r,a),fullgraph=True,options={'triton.cudagraphs':False});g,value=capture(fn);value=value.clone()
  if score_base is None:score_base=value.clone()
  rows[name]=dict(reader_values_equal=raw_equal,reader_max_abs=[float(torch.nan_to_num(x.double()-y.double()).abs().max()) for x,y in zip(base,fields)],output_bytes=sum(x.numel()*x.element_size() for x in fields),coeff_roundtrip_equal=all(torch.equal(enc[k].float(),en[k]) for k in ['co','A']),scores_equal=torch.equal(value,score_base),score_max_abs=float((value-score_base).abs().max()),actions_equal=torch.equal(value.flatten(1).argmax(1),score_base.flatten(1).argmax(1)),noop_equal=torch.equal(value.flatten(1).amax(1)>0,score_base.flatten(1).amax(1)>0))
  print(name,rows[name],flush=True)
 (ROOT/'artifacts/video_runtime_v1/storage_review_v1/contracts_cached.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
