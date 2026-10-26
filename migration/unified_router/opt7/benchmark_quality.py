import argparse,json
from pathlib import Path
import torch,lpips
from ..common import atomic_json
from migration.frozen_quality.worker import pixel_metrics,preview
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization7/benchmark'
 metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval()
 dense=torch.load(out/'Dense/output.pt',weights_only=True)['pixels']
 prior=torch.load(root/'optimization6/guard_benchmark/Guarded-K32/output.pt',weights_only=True)['pixels']
 rows={};panels={'Dense':dense,'Opt6':prior}
 for folder in sorted(out.iterdir()):
  if not folder.is_dir()or not(folder/'output.pt').exists():continue
  data=torch.load(folder/'output.pt',weights_only=True);pix=data['pixels']
  rows[folder.name]=dict(vs_dense=pixel_metrics(dense,pix,metric),vs_opt6=pixel_metrics(prior,pix,metric))
  if (folder/'routing.json').exists():
   records=json.load(open(folder/'routing.json'));rows[folder.name]['keep']=sum(r['retained_H']+r['retained_C']for r in records)/sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in records)
  print(folder.name,rows[folder.name],flush=True)
  if folder.name=='ScoreReuse0Split':panels[folder.name]=pix
 atomic_json(root/'optimization7/benchmark_quality.json',rows)
 if len(panels)==3:preview(root/'optimization7/first_prompt_Dense_Opt6_ScoreReuse0Split.mp4',panels,list(panels))
if __name__=='__main__':main()
