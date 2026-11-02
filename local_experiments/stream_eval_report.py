"""CPU-only resumable summary; same-case dense denominators, explicit exceptions."""
import json,statistics,math
from pathlib import Path
from .stream_eval_common import ROOT
from .learned_cache.io import atomic

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text());rows=[]
    for case in manifest['cases']:
        dense=ROOT/case/'dense/result.json'
        if not dense.exists():continue
        dr=json.loads(dense.read_text());dt=statistics.mean(r['total_generation_vae_seconds'] for r in dr['repetitions'])
        for method in manifest['methods']:
            p=ROOT/case/method/'result.json'
            if not p.exists():continue
            r=json.loads(p.read_text());total=statistics.mean(v['total_generation_vae_seconds'] for v in r['repetitions']);generation=statistics.mean(v['generation_seconds'] for v in r['repetitions'])
            q=r['quality'];aux=[];kv=[]
            for rep in r['repetitions']:
                v=rep['storage'];kv.append(v.get('self_kv_storage_bytes',v.get('persistent_self_kv_and_aux_bytes')));aux.append(v.get('auxiliary_bytes'))
            rows.append(dict(case=case,method=method,total_seconds=total,generation_seconds=generation,speedup=dt/total,quality=q,storage_bytes=kv,aux_bytes=aux))
    atomic(ROOT/'summary.json',dict(completed=len(rows),expected=len(manifest['cases'])*len(manifest['methods']),rows=rows))
    lines=['# 流式视频方法同模型比较','','统一512×768、480帧/16FPS、41个3-latent-frame chunk、最多5个历史chunk。相同Self-Forcing EMA骨干、输入噪声和文本嵌入，统一原生VAE。每种完整预热一次后计时两轮；加载和首次编译不计入，输出编码与质量计算不计入生成＋VAE。GPU7串行，计算和图形上下文监控排除经过核实的空闲父进程显存预留。', '',f'当前完成 {len(rows)}/{len(manifest["cases"])*len(manifest["methods"])} 个场景-方法单元。未完成项不作为失败或零分。','','|场景|方法|生成＋VAE/s|生成/s|相对同场景Dense加速|前段LPIPS↓|前段SSIM↑|前段PSNR↑|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        q=r['quality'];fmt=lambda k: '-' if k not in q else f'{q[k]:.4f}'
        lines.append(f"|{r['case']}|{r['method']}|{r['total_seconds']:.3f}|{r['generation_seconds']:.3f}|{r['speedup']:.3f}×|{fmt('lpips')}|{fmt('ssim')}|{fmt('psnr')}|")
    lines+=['','前段质量窗口RGB[21,101)，PSNR全分辨率、LPIPS/SSIM在256×384计算，均基于未压缩RGB。Dense一致性指标不是绝对质量排名。后续前段DINO/运动/亮度指标作为辅助，不能单独以一致性高或运动低判定质量好。只覆盖三个场景、一个seed，不能外推论文总体质量。','','## 配置与方法限制','']+[f'- {x}' for x in manifest['exceptions']]
    if len(rows)==len(manifest['cases'])*len(manifest['methods']):
        lines+=['','## 三场景汇总','','加速比取逐场景比值的几何平均；质量指标为三个场景前段指标的算术平均。','','|方法|平均生成＋VAE/s|几何平均加速|前段LPIPS↓|前段SSIM↑|前段PSNR↑|','|---|---:|---:|---:|---:|---:|']
        aggregate=[]
        for method in manifest['methods']:
            rr=[r for r in rows if r['method']==method]
            q={k:statistics.mean(r['quality'][k] for r in rr) for k in ['lpips','ssim','psnr']} if method!='dense' else {}
            speed=math.exp(statistics.mean(math.log(r['speedup']) for r in rr));total=statistics.mean(r['total_seconds'] for r in rr)
            values='|'.join(f'{q[k]:.4f}' if k in q else '-' for k in ['lpips','ssim','psnr'])
            lines.append(f'|{method}|{total:.3f}|{speed:.3f}×|{values}|')
            aggregate.append(dict(method=method,total_seconds=total,geomean_speedup=speed,quality_mean=q))
        atomic(ROOT/'aggregate.json',aggregate)
    lines+=['','## 实际内存','', '|场景|方法|自注意力KV或我方KV+辅助/MiB|官方额外辅助/MiB|','|---|---|---:|---:|']
    for r in rows:
        vals=[x for x in r['storage_bytes'] if x is not None];aux=[x for x in r['aux_bytes'] if x is not None]
        lines.append(f"|{r['case']}|{r['method']}|{max(vals)/2**20:.2f}|{max(aux)/2**20 if aux else 0:.2f}|")
    lines+=['','我方列出KV与摘要合计，官方方法列出KV及额外辅助两列；官方warm_audit.json还记录预热时峰值。进程峰值显存包含模型/工作区；父进程占卡预留不属于模型部署内存。','', '各方法视频位于 `<场景>/<方法>/video.mp4`。全部结果原子落盘，已完成方法可直接跳过。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
