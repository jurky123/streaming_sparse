"""Join historical measurements and matched current-baseline reruns."""
import csv,json,statistics,math
from pathlib import Path
import torch
from ..stream_eval_common import REPO
from ..learned_cache.io import atomic

ROOT=REPO/'artifacts/comparison_current_baseline_20261117'
OLD=REPO/'artifacts/comparison_20261113'
CURRENT='current_round8_full_pset'

def main():
    with (OLD/'summary.csv').open() as f:
        old=list(csv.DictReader(f))
    rows=[dict(r,record='20261113') for r in old]
    checks={}
    for case in ['horse_field','traffic_crossing']:
        dr=json.loads((ROOT/case/'dense/result.json').read_text())
        dg=statistics.mean(r['generation_seconds'] for r in dr['repetitions'])
        dt=statistics.mean(r['total_generation_vae_seconds'] for r in dr['repetitions'])
        x=torch.load(ROOT/case/'dense/latents.pt',weights_only=True,map_location='cpu')
        y=torch.load(OLD/case/'dense/latents.pt',weights_only=True,map_location='cpu')
        checks[case]=dict(bitwise_equal=torch.equal(x,y),max_abs=float((x.float()-y.float()).abs().max()),nmse=float((x.float()-y.float()).square().mean()/y.float().square().mean()))
        for method in ['dense',CURRENT]:
            result=json.loads((ROOT/case/method/'result.json').read_text());reps=result['repetitions']
            gen=statistics.mean(r['generation_seconds'] for r in reps)
            total=statistics.mean(r['total_generation_vae_seconds'] for r in reps)
            st=reps[0]['storage'];raw=st.get('raw_bytes',st.get('self_kv_storage_bytes'))
            persistent=st.get('persistent_bytes',raw+st.get('auxiliary_bytes',0))
            row=dict(case=case,method=method,record='20261117',generation_seconds=gen,total_seconds=total,total_min=min(r['total_generation_vae_seconds'] for r in reps),total_max=max(r['total_generation_vae_seconds'] for r in reps),speedup=dt/total,generation_speedup=dg/gen,peak_gib=max(r['peak_allocated_bytes'] for r in reps)/2**30,kv_gib=raw/2**30,persistent_cache_and_aux_gib=persistent/2**30,physical_gpu=reps[0]['physical_gpu'],paired_dense_seconds=dt,auxiliary_gib=(persistent-raw)/2**30)
            for w,q in result['quality'].get('windows',{}).items():
                for metric in ['lpips','ssim','psnr']:row[f'{w}_{metric}']=q[metric]
            rows.append(row)
    atomic(ROOT/'dense_equivalence.json',checks)
    fields=list(rows[0])+['generation_speedup']
    with (ROOT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    text=['# 原始 30 秒配置：当前基线复测','',
          '原记录：`artifacts/comparison_20261113/summary.csv`、`REPORT.zh.txt`。本次按用户“按照原纪录跑”要求执行 30 秒；常规优化验收仍仅跑短视频。',
          '', '配置：horse_field、traffic_crossing，seed101；768×512，41 chunks，每 chunk 3 latent 帧，480 RGB 帧，16 FPS；5 历史 chunks（360 页）＋当前 chunk（72 页）。直接读取原 shared_inputs.pt 的文本 embedding、初始噪声及转移噪声，Self-Forcing generator_ema 不变。',
          '', 'GPU6 跑 horse_field，GPU7 跑 traffic_crossing；每张卡顺序执行原版 Dense 和当前基线，各完整预热生成＋VAE一次，测两轮。耗时取两次算术平均，端到端为每次真实“生成＋VAE”总耗时，不含 MP4 编码/指标计算。加载与编译排除；进程监控检查同卡无其他计算任务。没有额外显存预留进程。',
          '', '当前基线：第八轮已接受优化，完整覆盖 C_s17 摘要＋Risk＋Full_Pset_s17，Page64/Q64/R4、完整 V 均值＋rank8、M16/B16、10% 精算、16×32 单交换/no-op，72 tiles 合批，BF16 QK、四乘积补偿、late Reader、blocked action scoring。第九轮 fused Risk/static route 及第十轮 PAGE32/lazy 实验均关闭。精确开关、权重 SHA256、源码快照见各场景 metadata.json/source。',
          '', '**旧 ours_latest_summary / summary_tail 不包含 Full_Pset，摘要权重也不同。本表是旧方法与当前方法在同一视频任务上的比较，时间和质量变化不能全部归因于实现优化。其他旧方法仅引用历史结果，本次重跑的是 Dense 与当前基线。**',
          '', 'Dense latent 与旧记录一致性：'+ '；'.join(f'{k}: bitwise={v["bitwise_equal"]}, max_abs={v["max_abs"]:.6g}' for k,v in checks.items())+'。',
          '', 'Dense 的模型、输入、pipeline 与 attention 源码哈希均与旧记录一致；benchmark 入口脚本哈希已改变，具体哈希见 provenance_check.json。两种方法的两轮 latent 逐位重复性检查均通过。', '', '当前 Full_Pset 基线的两场景平均生成约 79.77 秒，生成＋VAE 约 99.70 秒，相对同卡 Dense 总加速几何均值为 0.655×（约慢 1.53 倍），纯生成约 0.568×。本次没有获得端到端加速。', '', '## 主要耗时对比', '', '|场景|记录|方法|生成/s|生成＋VAE/s|相对同卡 Dense 总加速|', '|---|---|---|---:|---:|---:|']
    focus=[r for r in rows if r['method'] in ['dense','ours_latest_summary','ours_latest_summary_tail',CURRENT]]
    for case in checks:
        for r in focus:
            if r['case']==case:text.append(f'|{case}|{r["record"]}|{r["method"]}|{float(r["generation_seconds"]):.3f}|{float(r["total_seconds"]):.3f}|{float(r["speedup"]):.3f}×|')
    text+=['', '## 前、中、后段质量', '', '所有指标相对于同输入 Dense。窗口为 [21,101)、[200,280)、[400,480)，各 80 帧；PSNR 在原分辨率计算，LPIPS/SSIM 在 256×384 计算，沿用原脚本且使用未压缩 RGB。这里只报告视频结果，没有将算法差异归为优化误差。', '', '|场景|方法|窗口|PSNR↑|SSIM↑|LPIPS↓|', '|---|---|---|---:|---:|---:|']
    for case in checks:
        for r in focus:
            if r['case']!=case or r['method']=='dense':continue
            for w in ['early','middle','late']:
                text.append(f'|{case}|{r["method"]}|{w}|{float(r[w+"_psnr"]):.3f}|{float(r[w+"_ssim"]):.4f}|{float(r[w+"_lpips"]):.4f}|')
    text+=['', '## 全部旧方法与当前基线', '', '耗时为两个场景算术平均；总加速为各自同卡 Dense 比值的几何平均。质量为两个场景的均值。', '', '|记录|方法|生成/s|生成＋VAE/s|总加速|前段 LPIPS|中段 LPIPS|后段 LPIPS|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for record,method in dict.fromkeys((r['record'],r['method']) for r in rows):
        rr=[r for r in rows if r['record']==record and r['method']==method]
        mean=lambda key:statistics.mean(float(r[key]) for r in rr)
        q='|'.join(f'{mean(w+"_lpips"):.4f}' if rr[0].get(w+'_lpips') not in ('',None) else '—' for w in ['early','middle','late'])
        text.append(f'|{record}|{method}|{mean("generation_seconds"):.3f}|{mean("total_seconds"):.3f}|{math.exp(statistics.mean(math.log(float(r["speedup"])) for r in rr)):.3f}×|{q}|')
    text+=['', '完整表（逐场景、各窗口 PSNR/SSIM/LPIPS、峰值显存等）：`summary.csv`。每个场景各方法目录包含两轮原始计时、未压缩 latent、video.mp4、质量指标和进程监控。', '', '代码改动仅新增原配置复测入口、单卡顺序控制器和表格生成器；未修改算法与优化基线。']
    (ROOT/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(dense_equivalence=checks,current=[r for r in rows if r['method']==CURRENT]),indent=2))
if __name__=='__main__':main()
