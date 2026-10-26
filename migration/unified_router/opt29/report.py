import json,statistics,time,html,math
from pathlib import Path
from .setup import ROOT,VARIANTS,PILOTS
from .io import atomic_json

def read(p):
 try:return json.loads(p.read_text())
 except(FileNotFoundError,json.JSONDecodeError):return {}
def generate(state=None):
 state=state or read(ROOT/'queue.json');runs=[]
 for p in sorted((ROOT/'training').glob('*/*/status.json')):
  s=read(p);curve=read(p.with_name('curve.json'))or[];best=read(p.with_name('best_validation.json'));tail=curve[-3:]
  runs.append(dict(layer=int(p.parent.parent.name[5:]),variant=p.parent.name,status=s.get('status'),updates=s.get('updates',0),total=s.get('total',0),initial=s.get('initial'),best=best.get('nmse',s.get('best')),best_step=best.get('updates'),recent_val=[x['nmse']for x in tail],convergence=read(p.with_name('completed.json')).get('convergence','running'),error=s.get('error'),folder=str(p.parent.relative_to(ROOT))))
 frontier=[]
 for phase in ['pilot','final']:
  for name in VARIANTS:
   vals=[read(p)for p in(ROOT/f'bench/{phase}').glob(f'layer*/{name}.json')];vals=[x for x in vals if x.get('completed')]
   if vals:
    ms=statistics.mean(x['ms']for x in vals);base=statistics.mean(x['baseline_ms']for x in vals);frontier.append(dict(variant=name,phase=phase,layers=len(vals),nmse=statistics.mean(x['best_validation']['nmse']for x in vals),ms=ms,speedup=base/ms))
 videos=[dict(path=str(p.relative_to(ROOT)),**read(p))for p in sorted((ROOT/'e2e').glob('*/*/result.json'))]
 video_comparison=[]
 for resolution in sorted({(v['height'],v['width'])for v in videos}):
  cases=[v for v in videos if(v['height'],v['width'])==resolution];names=sorted({n for v in cases for n in v.get('variants',{})if n!='Dense'})
  for name in names:
   paired=[v for v in cases if name in v['variants']and 'quality'in v['variants'][name]and 'quality'in v['variants'].get('FixedQ32',{})]
   if not paired:continue
   quality=[v['variants'][name]['quality']['sparse_frames_only']for v in paired];baseline=[v['variants']['FixedQ32']['quality']['sparse_frames_only']for v in paired];mse=statistics.mean(x['mse_rgb01']for x in quality);bmse=statistics.mean(x['mse_rgb01']for x in baseline);lp=statistics.mean(x['lpips']for x in quality);blp=statistics.mean(x['lpips']for x in baseline);seconds=statistics.mean(v['variants'][name]['seconds']for v in paired);dense=statistics.mean(v['variants']['Dense']['seconds']for v in paired)
   video_comparison.append(dict(height=resolution[0],width=resolution[1],variant=name,cases=len(paired),psnr_db=-10*math.log10(mse)if mse else None,lpips=lp,psnr_delta_vs_fixed_q32=10*math.log10(bmse/mse)if mse and bmse else None,lpips_delta_vs_fixed_q32=lp-blp,seconds=seconds,speedup_vs_dense=dense/seconds,both_quality_metrics_no_worse=(mse<=bmse and lp<=blp),quality_scope='Uncompressed RGB frames9..44; PSNR from meanMSE across matched cases; LPIPS mean across matched frames'))
 summary=dict(video_comparison=video_comparison,updated=time.time(),stage=state.get('stage'),jobs={k:sum(j['status']==k for j in state.get('jobs',[]))for k in ['pending','running','done','failed','blocked','orphaned']},runs=runs,frontier=frontier,promotion=read(ROOT/'promotion.json'),videos=videos,training_scope='Q aggregation + KV prototypes/projection; optional Train-only Q/K tree updates; backbone fixed',target='Native attention output NMSE; best by96 Val events; firstchunk/layer0 dense; actual~20% execution',next='8 GPU dynamic queue, no automatic retry loops; full29 finalists then decoded480p/720p comparisons')
 atomic_json(ROOT/'summary.json',summary)
 esc=html.escape
 lines=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Opt29 夜间探索</title><style>body{font:15px system-ui;margin:28px;background:#f6f8fb;color:#182234}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border-bottom:1px solid #dde3ee;text-align:left}th{background:#e7edf6}a{color:#1764ab}.bad{color:#bc3030}.muted{color:#5b677b}</style>',
 '<h1>Opt29｜Query 聚合夜间探索</h1>',f'<p>阶段：<b>{esc(str(summary["stage"]))}</b>　刷新时间 UTC：{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}</p>',f'<p>任务计数：{esc(str(summary["jobs"]))}。Q32×K64冻结对照与继续训练对照均保留。</p>',
 '<p>训练Q聚合权重、KV原型与投影；分组更新单独消融，backbone固定。Loss/best均依据实际量化attention输出误差。部分层结果不能当成完整视频质量。</p>',
 '<p><a href="PLAN.txt">实验计划</a> · <a href="promotion.json">晋级依据</a> · <a href="VIDEO_INDEX.html">视频索引</a> · <a href="summary.json">完整数据</a></p>',
 '<h2>训练与收敛</h2><table><tr><th>路线</th><th>层</th><th>状态</th><th>进度</th><th>初始</th><th>Best实际NMSE</th><th>Best步</th><th>最近3次Val</th><th>曲线/日志</th></tr>']
 ranking=['<h2>实际误差与完整模块速度</h2><table><tr><th>路线</th><th>阶段/已有层数</th><th>Val NMSE</th><th>模块ms</th><th>相对冻结Q32加速</th></tr>']
 for f in frontier:ranking.append(f'<tr><td>{esc(f["variant"])}</td><td>{f["phase"]} / {f["layers"]}</td><td>{f["nmse"]:.6f}</td><td>{f["ms"]:.3f}</td><td>{f["speedup"]:.3f}×</td></tr>')
 ranking.append('</table><p>pilot共4层、final共29层；未齐的数据仅供观察。模块计时不等于整段视频加速。</p>')
 lines[-1:-1]=ranking
 fmt=lambda x:'—'if x is None else f'{x:.6f}'
 for r in sorted(runs,key=lambda x:(x['variant'],x['layer'])):
  f=r['folder'];vals=', '.join(fmt(x)for x in r['recent_val']);lines.append(f'<tr><td>{esc(r["variant"])}</td><td>{r["layer"]}</td><td>{esc(str(r["status"]))}</td><td>{r["updates"]}/{r["total"]}</td><td>{fmt(r["initial"])}</td><td>{fmt(r["best"])}</td><td>{r["best_step"]}</td><td>{vals}</td><td><a href="{f}/loss_curve.png">曲线</a> · <a href="{f}/train.jsonl">训练</a></td></tr>')
 lines+=['</table><h2>失败与阻断（不自动无限重试）</h2><ul>']
 for j in state.get('jobs',[]):
  if j['status']in ['failed','blocked','orphaned']:
   log=Path(j.get('log','')).name;lines.append(f'<li class="bad">{esc(j["id"])}：{esc(j["status"])}，exit={j.get("exit_code")} <a href="logs/{log}">日志</a></li>')
 lines+=['</ul><p class="muted">到步数上限不等于收敛。查看最近验证趋势与best所在步；最终视频只使用完成全部29层的路线。</p>'];(ROOT/'index.html').write_text('\n'.join(lines))
 vi=['<!doctype html><meta charset="utf-8"><title>Opt29 视频对比</title><h1>Opt29 视频与端到端结果</h1><p><a href="index.html">返回状态</a></p>']
 vi.append('<p>质量同时看PSNR（越高越好）和LPIPS（越低越好）。主表只统计首个dense chunk之后36帧；完整与逐chunk指标在JSON中。若两项有取舍，会分别列出，不合成单一分数。</p><table border="1" cellpadding="8"><tr><th>分辨率</th><th>路线</th><th>案例数</th><th>PSNR dB ↑</th><th>LPIPS ↓</th><th>ΔPSNR vs Q32</th><th>ΔLPIPS vs Q32</th><th>生成秒</th><th>相对Dense加速</th></tr>')
 for r in video_comparison:
  vi.append(f'<tr><td>{r["height"]}x{r["width"]}</td><td>{esc(r["variant"])}</td><td>{r["cases"]}/4</td><td>{r["psnr_db"]}</td><td>{r["lpips"]:.6f}</td><td>{r["psnr_delta_vs_fixed_q32"]}</td><td>{r["lpips_delta_vs_fixed_q32"]:+.6f}</td><td>{r["seconds"]:.3f}</td><td>{r["speedup_vs_dense"]:.3f}×</td></tr>')
 vi.append('</table>')
 for v in videos:
  folder=str(Path(v['path']).parent);vi.append(f'<h2>{esc(folder)}</h2><p>每条三栏视频：Dense / 冻结Q32×K64 / 候选。指标使用未压缩RGB后36帧。</p><ul>')
  for name,r in v.get('variants',{}).items():
   if name=='Dense':continue
   quality=r.get('quality',{}).get('sparse_frames_only',{});vi.append(f'<li>{esc(name)}：{r.get("seconds",0):.3f}s，PSNR={quality.get("psnr_db")} dB，LPIPS={quality.get("lpips")} <a href="{folder}/Dense_FixedQ32_{name}.mp4">对比视频</a></li>')
  vi+=['</ul>']
 if not videos:vi+=['<p>视频阶段尚未完成；不会把部分层或未训练样本标为最终结果。</p>']
 (ROOT/'VIDEO_INDEX.html').write_text('\n'.join(vi))
if __name__=='__main__':generate()
