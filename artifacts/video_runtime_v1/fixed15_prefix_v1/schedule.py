from test import *
from local_experiments.video_runtime_v1.page_prefetch import PagePipeline
scheduled=PagePipeline(specialized.fn,aligned_coefficients=True)
def launch():return run(False,True,scheduled)
if __name__=='__main__':
 a=base();b=launch()
 print('scheduled', '%waitv =' in b.asm['ttgir'],flush=True)
 if args.profile:
  for _ in range(50): launch()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();launch();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 else:
  ts=times([timed(base),timed(launch)])
  delta=new.float()-old.float()
  row=dict(scheduled='%waitv =' in b.asm['ttgir'],ms=list(map(median,ts)),speedup=median(x/y for x,y in zip(*ts)),bitwise=torch.equal(new.view(torch.uint8),old.view(torch.uint8)),max_abs=float(delta.abs().max()),samples=ts)
  (root/f'scheduled_{p}.json').write_text(json.dumps(row,indent=2));print(str(row)[:160])
