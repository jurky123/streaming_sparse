import argparse,json,torch
from .setup import ROOT,tag
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=832);p.add_argument('--chunks',type=int,default=4);a=p.parse_args();name=tag(a.height,a.width,a.chunks)
 raw=torch.load(ROOT/'capture'/name/'layer24.pt',map_location='cuda',weights_only=False);q=raw['q'];k=torch.cat((raw['kh'],raw['kc']),1);v=torch.cat((raw['vh'],raw['vc']),1)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
  for _ in range(5):out=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   out=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None]);torch.cuda.synchronize()
 rows=[dict(name=e.key,ms=e.device_time_total/1000) for e in prof.key_averages() if e.device_type==torch.autograd.DeviceType.CUDA]
 atomic_json(ROOT/f'dense_profile_{name}.json',rows);print(json.dumps(rows))
if __name__=='__main__':main()
