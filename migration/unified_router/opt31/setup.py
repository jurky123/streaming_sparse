from pathlib import Path
import json,hashlib,time
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt31_priority_long_video_20261026'
PARENT=ROOT.parent/'opt30_behavior_grouping_20261026'
PREVIOUS=ROOT.parent/'opt29_query_aggregation_20261025'
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if(ROOT/'manifest.json').exists():return
 methods={}
 for name,root,variant,why in [('Best',PARENT,'q32_full','Lowest native96-Val NMSE among currently completed all29 Opt30 methods; quality-oriented candidate, long-video quality unverified'),('FastReference',PREVIOUS,'q64_mean8','Previous measured fastest720p fully trained method; speed-quality tradeoff reference')]:
  checkpoints={}
  for layer in range(1,30):
   folder=root/f'training/layer{layer:02d}/{variant}';assert json.loads((folder/'completed.json').read_text())['updates']>0
   path=folder/'best.pt';checkpoints[str(layer)]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  methods[name]=dict(variant=variant,checkpoints=checkpoints,selection=why)
 manifest=dict(created=time.time(),configuration=str(PARENT/'experiment.json'),methods=methods,fps=16,durations=[5,15,30],resolutions=[[480,832],[720,1280],[512,768]],configurations=[dict(height=h,width=w,seconds=s)for h,w,ss in [(480,832,[5,30]),(720,1280,[5,30]),(512,768,[5,15])]for s in ss],configuration_rule='Only explicit configurations, not a Cartesian product',cases=4,history_chunks=3,history_policy='Rolling latest3 complete chunks, same Dense/sparse; absolute RoPE positions preserved, first chunk evicted normally',frame_policy='Native chunks generate12n-3 frames; generate81/249/489 then crop to80/240/480 for5/15/30s. Timing includes same padding for Dense and sparse.',quality='PSNR and LPIPS on uncompressed RGB; full, post-first9, per-second and final5s',timing='Warm complete generation including clean-history encoding and rolling eviction; excludes VAE/load/compile. VAE and generation+VAE reported separately.',priority='Runs on GPUs released by Opt30, before further Opt30 claims and action/V-aware experiments')
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(ROOT)
if __name__=='__main__':main()
