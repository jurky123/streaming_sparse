"""Real-QKV operator scaling; frozen Tile20, forced FlashAttention reference."""
import argparse,gc,json,math,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
import triton as tr
from .setup import ROOT,TRAINING,tag
from .pipeline import Pipeline
from .attention import attention
from .route import scores,_fine_age
from ..common import config,atomic_json,sha
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..opt16.grouping import order,gather
from ..opt19.tile import select
from ..opt5.probe import capture,timing


def nmse(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum().clamp_min(1e-20))

def reference_check(q,out,aux,pipe):
 # Independent PyTorch implementation of selected, quantized attention. Global
 # softmax versus split softmax permits BF16 probability-rounding differences.
 ids=aux['selection'];ix=aux['q_order'];qq=aux['q_packed'];n=q.shape[1];seg=pipe.seg;pages=tr.cdiv(seg,64);errors=[]
 for h in [0,5,11]:
  for tile in [0,ids.shape[1]//2,ids.shape[1]-1]:
   positions=torch.arange(tile*32,min((tile+1)*32,n),device=q.device)
   qp=qq[h,positions].float();scale=qp.abs().max().clamp_min(1.27e-8)/127.;qi=(qp/scale).round()
   selected=ids[h,tile].long();indices=selected[:,None]//pages*seg+selected[:,None]%pages*64+torch.arange(64,device=q.device)[None]
   valid=(selected[:,None]%pages*64+torch.arange(64,device=q.device)[None])<seg
   kval=pipe.quant.k[h,indices[valid]].float();ks=pipe.quant.scale[h,selected][:,None].expand(-1,64)[valid]
   logits=(qi@kval.T)*(scale*ks[None])/(128**.5)
   expected=logits.softmax(-1).to(torch.bfloat16).float()@aux['kv'].v[h,indices[valid]].float()
   actual=out[h,ix[h,positions].long()];errors.append(nmse(actual,expected))
 assert max(errors)<1e-4,errors
 return dict(sampled_tiles=9,max_nmse=max(errors),mean_nmse=statistics.mean(errors),tolerance=1e-4)


def flop_stats(q,aux,seg):
 h,n,d=q.shape;ids=aux['selection'];candidates=aux['candidates'];b=aux['kv'].state['key'].shape[1];nk=aux['kv'].k.shape[1];g=tr.cdiv(n,32);pages=tr.cdiv(seg,64)
 lengths=(seg-(ids%pages)*64).clamp(max=64).float();qweights=(n-torch.arange(g,device=q.device)*32).clamp(max=32).float()
 keep=float((lengths.sum(-1)*qweights).sum()/(h*n*nk));executed=ids.numel()*32*64/(h*n*nk)
 age=tr.cdiv(b,pages)-1-candidates//pages;p=torch.where(age==0,16,torch.where(age==1,8,torch.where(age==2,4,2)))
 logical_fine=float((p.sum(-1)*qweights).sum())*2*64
 padded=F.pad(p,(0,(-p.shape[-1])%8)).reshape(h,g,-1,8).amax(-1)
 executed_fine=float(padded.sum())*2*32*64*8
 logical_coarse=0;executed_coarse=0
 for _,lc in aux['kv'].state['parts']:
  np,protos=lc.shape[1:];gp=128//protos
  logical_coarse+=2*h*g*8*64*np*protos
  executed_coarse+=2*h*g*16*64*tr.cdiv(np,gp)*gp*protos
 projection=2*h*n*128*64;dense=4*h*n*nk*128
 return dict(true_keep=keep,executed_fraction=executed,selected_pages=ids.shape[-1],candidate_pages=candidates.shape[-1],dense_matmul_flops=dense,sparse_attention_matmul_flops=dense*executed,router_matmul_logical_flops=projection+logical_coarse+logical_fine,router_matmul_padded_flops=projection+executed_coarse+executed_fine,router_fine_padding=executed_fine/logical_fine,route_score_bytes=aux['fine'].numel()*4,exclusions='Router FLOPs exclude exp/log/reductions/top-k/data movement; padded FLOPs count tensor-core padding, not branch/control/memory overhead')

@torch.no_grad()
def run(a):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 label=tag(a.height,a.width,a.chunks);outdir=ROOT/'bench'/label;outdir.mkdir(parents=True,exist_ok=True)
 def status(s,**kw):atomic_json(outdir/f'layer{a.layer:02d}.status.json',dict(status=s,**kw));print(s,kw,flush=True)
 status('loading');source=ROOT/'capture'/label/f'layer{a.layer:02d}.pt';raw=torch.load(source,map_location='cuda',weights_only=False);q=raw['q'];seg=raw['chunk_tokens']
 _,m=config(TRAINING);model=TileRouter(m,a.layer,'Tile20').cuda().eval();ckpt=TRAINING/f'training/layer{a.layer:02d}/Tile20/best.pt';model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 pipe=Pipeline(model,raw,seg);status('compiling')
 y,aux=pipe.run(q,raw['kc'],raw['vc'],True);assert torch.isfinite(y).all();validation=reference_check(q,y,aux,pipe)
 if seg==4680:
  from ..opt21.pipeline import Pipeline as Old
  old=Old(model,raw,32,.2,'shared','fp16');yy,dd=old.run(q,raw['kc'],raw['vc'],True)
  assert torch.equal(y,yy) and torch.equal(aux['selection'],dd['selection']),'Generalized implementation must equal current production'
  validation['production_bit_exact']=True;del old,yy,dd
 stats=flop_stats(q,aux,seg);kv=aux['kv'];qq=aux['q_packed'];ids=aux['selection'];ix=aux['q_order'];b=kv.state['key'].shape[1]
 k=torch.cat((raw['kh'],raw['kc']),1).contiguous();v=torch.cat((raw['vh'],raw['vc']),1).contiguous();oracle_n=math.ceil(k.shape[1]*.2);oracle_k=k[:,:oracle_n].contiguous();oracle_v=v[:,:oracle_n].contiguous()
 def dense():return F.scaled_dot_product_attention(q[None],k[None],v[None])[0]
 def oracle():return F.scaled_dot_product_attention(q[None],oracle_k[None],oracle_v[None])[0]
 def att():return attention(qq,kv,ids,pipe.quant,seg,ix)
 def route():
  f,c=scores(qq,kv.state,.35,8,seg);return select(f,c,b,32,.2)
 functions=dict(dense_flash=dense,oracle_contiguous20_flash=oracle,sparse_attention=att,router=route,sparse_pipeline=lambda:pipe.run(q,raw['kc'],raw['vc']))
 graphs={};outputs={}
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
  for name,fn in functions.items():
   graphs[name],outputs[name]=capture(fn);graphs[name].replay()
  torch.cuda.synchronize()
  assert torch.equal(outputs['sparse_pipeline'],y);validation['graph_exact']=True;validation['attention_nmse_vs_dense']=nmse(y,outputs['dense_flash'])
  status('timing')
  for graph in graphs.values():timing(graph,20)
  samples={name:[] for name in graphs}
  for repeat in range(5):
   names=list(graphs);names=names[repeat%len(names):]+names[:repeat%len(names)]
   if repeat%2:names.reverse()
   for name in names:samples[name].append(timing(graphs[name],a.reps))
  ms={name:statistics.median(values) for name,values in samples.items()}
  # Separate component timings; scopes do not sum exactly due graph scheduling.
  components={}
  def current():
   result=pipe.cache.current(raw['kc'],raw['vc']);pipe.quant.current();return result
  def grouping():
   indices=order(q,model.tree('q'));return gather(q,indices),indices
  def history():return Pipeline(model,raw,seg)
  for name,fn in [('current_kv_group_prototype_quant',current),('query_group_gather',grouping),('tile_selection',lambda:select(aux['fine'],aux['candidates'],b,32,.2)),('history_prepare',history)]:
   cg,co=capture(fn);timing(cg,10);components[name]=statistics.median(timing(cg,max(5,a.reps//2)) for _ in range(3));del cg,co;gc.collect()
  # Profile warmed kernels once to identify hotspots, not to compare latency.
  profile=[]
  if a.layer==24:
   torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    pipe.run(q,raw['kc'],raw['vc']);torch.cuda.synchronize()
   profile=[dict(name=e.key,ms=e.device_time_total/1000,calls=e.count) for e in prof.key_averages() if e.device_type==torch.autograd.DeviceType.CUDA]
 result=dict(case=label,height=a.height,width=a.width,chunks=a.chunks,history_chunks=a.chunks-1,layer=a.layer,query_tokens=seg,kv_tokens=k.shape[1],scope='Warm attention operator from actual frozen Dense closed-loop QKV; not sparse high-resolution video evaluation',gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,torch=torch.__version__,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),source=str(source),source_sha256=sha(source),validation=validation,work=stats,median_ms=ms,samples_ms=samples,components_ms=components,kernel_profile=profile,attention_time_fraction=ms['sparse_attention']/ms['dense_flash'],pipeline_time_fraction=ms['sparse_pipeline']/ms['dense_flash'],attention_speedup=ms['dense_flash']/ms['sparse_attention'],pipeline_speedup=ms['dense_flash']/ms['sparse_pipeline'],oracle_time_fraction=ms['oracle_contiguous20_flash']/ms['dense_flash'],peak_gb=torch.cuda.max_memory_allocated()/1e9)
 atomic_json(outdir/f'layer{a.layer:02d}.json',result);status('completed',ms=ms);print(json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser()
 for k in ['height','width','chunks','layer']:p.add_argument('--'+k,type=int,required=True)
 p.add_argument('--reps',type=int,default=40);a=p.parse_args();run(a)
if __name__=='__main__':main()
