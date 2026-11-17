"""Summarize frozen-baseline batch quality and render existing encoded comparisons."""
import argparse,csv,json,math,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .weights import ROOT

def metrics(q,start=0,stop=80):
 pf=q['per_frame'];mse=statistics.mean(pf['mse'][start:stop]);return dict(lpips=statistics.mean(pf['lpips'][start:stop]),ssim=statistics.mean(pf['ssim'][start:stop]),psnr=100. if mse==0 else -10*math.log10(mse))
def render(folder):
 import imageio.v2 as imageio
 from ..stream_eval_common import save_video
 target=folder/'comparison.mp4'
 if target.exists():return str(target)
 readers=[imageio.get_reader(folder/m/'video.mp4',input_params=['-threads','1']) for m in ['dense','c_risk','c_pset']];count=[0]
 def batches():
  buf=[]
  for frames in zip(*readers):
   buf.append(np.concatenate(frames,axis=1));count[0]+=1
   if len(buf)==8:yield np.stack(buf);buf=[]
  if buf:yield np.stack(buf)
 try:save_video(target,batches())
 finally:
  for r in readers:r.close()
 assert count[0]==105,count
 (folder/'comparison.json').write_text(json.dumps(dict(frames=105,left_to_right=['Dense','C+Risk','C+Risk+Full_Pset'],VAE_decodes=0),indent=2));return str(target)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(ROOT/'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1'));ap.add_argument('--comparisons',action='store_true');args=ap.parse_args();root=Path(args.root)
 manifest=json.loads((root/'manifest.json').read_text());status=json.loads((root/'batch/status.json').read_text());assert status['stage']=='complete'
 identity=json.loads((root/'execution_identity.json').read_text());assert identity['deployment']['tile_batch']==72 and identity['deployment']['history_pages']==360
 rows=[]
 for job in manifest:
  folder=root/'videos'/f"{job['case']}_s{job['noise_seed']}"/'chunks9'/job['method'];p=json.loads((folder/'postprocess.json').read_text());g=json.loads((folder/'generation.json').read_text());assert p['job']==job and g['job']==job
  assert g['runtime']['counters'][2]==0
  assert g['weights']['deployment']['tile_batch']==72 and not g['weights']['placeholder']
  if job['method']=='dense':continue
  q=p['quality']['early'];assert q['window']==[21,101] and all(len(v)==80 for v in q['per_frame'].values())
  row=dict(case=job['case'],seed=job['noise_seed'],method=job['method'])
  for name,start,stop in [('all',0,80),('early',0,40),('late',40,80)]:
   for k,v in metrics(q,start,stop).items():row[f'{name}_{k}']=v
  rows.append(row)
 summary={}
 for m in ['c_risk','c_pset']:
  rs=[r for r in rows if r['method']==m];summary[m]={k:statistics.mean(r[k] for r in rs) for k in rs[0] if k not in ['case','seed','method']}
 pairs=[]
 for r in rows:
  if r['method']!='c_risk':continue
  p=next(x for x in rows if x['method']=='c_pset' and (x['case'],x['seed'])==(r['case'],r['seed']))
  pairs.append(dict(case=r['case'],seed=r['seed'],lpips_delta=p['all_lpips']-r['all_lpips'],ssim_delta=p['all_ssim']-r['all_ssim'],psnr_delta=p['all_psnr']-r['all_psnr']))
 wins=sum(p['lpips_delta']<0 for p in pairs);byseed={s:{m:statistics.mean(r['all_lpips'] for r in rows if r['seed']==s and r['method']==m) for m in summary} for s in [701,702]}
 (root/'summary.json').write_text(json.dumps(dict(jobs=len(manifest),summary=summary,paired=pairs,lpips_wins=wins,by_seed=byseed,windows=dict(all=[21,101],early=[21,61],late=[61,101]),identity=identity),indent=2))
 with (root/'per_video.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 lines=['# 9d4d33d 基线批量短视频质量评估','','14个已有场景×seed701/702×Dense/C+Risk/C+Full_Pset，共84个9-chunk视频；28组严格配对。768×512、105帧、16FPS、history360、tile72。GPU0–4每卡一个常驻进程；质量清单冻结9d4d33d代码，仅修正batch参数透传和README。P2/scan候选没有进入本清单。','','源代码快照见 frozen_source.json，执行身份见 execution_identity.json。checkpoint内容哈希、同prompt/seed/长度/生成配置一致；全覆盖C_s17与Full_Pset_s17。14个场景来自现有embedding库，不称为独立未见场景测试。21–100帧的80帧指标用于主比较；前段为[21,61)，后段为[61,101)，不是长视频后段。','','## 配对结果','','|方法|整体LPIPS↓|前段LPIPS↓|后段LPIPS↓|整体SSIM↑|整体PSNR/dB↑|','|---|---:|---:|---:|---:|---:|','|Dense|—|—|—|—|—|']
 for m,label in [('c_risk','C摘要＋Risk'),('c_pset','C摘要＋Risk＋Full_Pset')]:
  r=summary[m];lines.append(f'|{label}|{r["all_lpips"]:.4f}|{r["early_lpips"]:.4f}|{r["late_lpips"]:.4f}|{r["all_ssim"]:.4f}|{r["all_psnr"]:.3f}|')
 diff=summary['c_pset']['all_lpips']-summary['c_risk']['all_lpips']
 lines+=['',f'Full_Pset在{wins}/28个配对视频上降低LPIPS。整体平均变化{diff:+.6f}（相对C+Risk约{diff/summary["c_risk"]["all_lpips"]:+.2%}），均值收益很小、场景差异明显，不能把此前局部Attention收益直接视为视频质量同比例收益。SSIM平均变化{summary["c_pset"]["all_ssim"]-summary["c_risk"]["all_ssim"]:+.6f}，PSNR平均变化{summary["c_pset"]["all_psnr"]-summary["c_risk"]["all_psnr"]:+.3f}dB。','','|seed|C+Risk LPIPS|Full_Pset LPIPS|','|---|---:|---:|']
 for s,r in byseed.items():lines.append(f'|{s}|{r["c_risk"]:.4f}|{r["c_pset"]:.4f}|')
 lines+=['','## 变化最大的配对视频','','下表delta为Full_Pset−C+Risk，LPIPS负值表示更接近Dense。对应三列视频从左到右Dense、C+Risk、Full_Pset。','','|场景|seed|LPIPS delta|SSIM delta|PSNR delta/dB|并排视频|','|---|---:|---:|---:|---:|---|']
 for p in sorted(pairs,key=lambda x:x['lpips_delta'])[:3]+sorted(pairs,key=lambda x:x['lpips_delta'])[-3:]:
  link=f"videos/{p['case']}_s{p['seed']}/chunks9/comparison.mp4";lines.append(f'|{p["case"]}|{p["seed"]}|{p["lpips_delta"]:+.4f}|{p["ssim_delta"]:+.4f}|{p["psnr_delta"]:+.3f}|[查看]({link})|')
 lines+=['','## 测量和产物说明','','LPIPS为AlexNet、SSIM在256×384双线性缩放后计算；PSNR使用原分辨率逐帧MSE。先算每个视频指标，再对28视频等权平均；PSNR不是所有帧MSE合并后再取log。前后段直接从保存的逐帧指标切分，不重新解码。Dense在每个prompt/seed上只生成和解码一次，其像素缓存供两种稀疏方法复用。','','所有84生成和84后处理任务完成，无nonfinite评分。保存clean latent、最终视频、逐帧指标和每方法对Dense的双列视频；comparison.mp4从已有编码视频拼接，无额外VAE解码。耗时含冷编译、IO和不同任务顺序，且多GPU并行运行，因此本清单不报告单视频加速比。','','这批结果主要回答短视频闭环差异；主体/运动/闪烁的因果结论不能单凭LPIPS推出。先检查改善和变差最大的并排视频，再决定后续质量测试范围。没有启动15s或30s视频。','','启动时冻结目录的wan_models符号链接导致local-assets相对路径解析错误；在生成任何视频前修正为真实目录＋模型子目录链接后重试，未更改算法或权重。最终展开的batch命令均显式tile72/history360，既有identity记录实际配置。']
 (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary,wins=wins,by_seed=byseed)),flush=True)
 if args.comparisons:
  folders=sorted((root/'videos').glob('*/chunks9'))
  with ThreadPoolExecutor(max_workers=4) as pool:outputs=list(pool.map(render,folders))
  (root/'comparisons.json').write_text(json.dumps(outputs,indent=2));print('Rendered comparisons',len(outputs),flush=True)
if __name__=='__main__':main()
