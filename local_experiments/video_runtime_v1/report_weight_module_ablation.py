"""Aggregate the weight/module factorial experiment in the requested table format."""
import csv,json,math,statistics
from pathlib import Path
from ..stream_eval_common import REPO
ROOT=REPO/'artifacts/weight_module_ablation_20261117'
METHODS=['old_summary_risk','new_summary_risk','old_summary_full_pset','current_round8_full_pset']
LABELS={
 'dense':'Dense（原记录）','ours':'Ours 原始参考实现（ours）','ours_latest_summary':'Ours 最新 summary（原记录）','ours_latest_summary_tail':'Ours 最新 summary_tail（原记录）',
 'solattn':'Sol-Attn','svgear':'SVG-EAR','lightforcing':'LightForcing','forcingkv':'Forcing-KV','dasa':'DASA','pisa':'PISA',
 'old_summary_risk':'消融 A：旧摘要权重＋Risk','new_summary_risk':'消融 B：新摘要权重＋Risk','old_summary_full_pset':'消融 C：旧摘要权重＋Risk＋Full_Pset','current_round8_full_pset':'当前方法 D：新摘要权重＋Risk＋Full_Pset'}

def mean(xs):return statistics.mean(xs)
def geomean(xs):return math.exp(mean([math.log(x) for x in xs]))
def read(path):return json.loads(path.read_text())
def summarize(path):
    r=read(path/'result.json');reps=r['repetitions']
    return dict(generation_seconds=mean([x['generation_seconds'] for x in reps]),total_seconds=mean([x['total_generation_vae_seconds'] for x in reps]),total_min=min(x['total_generation_vae_seconds'] for x in reps),total_max=max(x['total_generation_vae_seconds'] for x in reps),physical_gpu=reps[0]['physical_gpu'],quality=r['quality'])

def main():
    detail=[]
    for case in ['horse_field','traffic_crossing']:
        for method in METHODS:
            path=ROOT/case/method;row=summarize(path)
            pair=read(path/'paired_dense.json')['path'] if (path/'paired_dense.json').exists() else str(ROOT/case/'dense')
            dense=summarize(Path(pair));assert row['physical_gpu']==dense['physical_gpu']
            row.update(case=case,method=method,paired_dense_seconds=dense['total_seconds'],speedup=dense['total_seconds']/row['total_seconds'],generation_speedup=dense['generation_seconds']/row['generation_seconds'],paired_dense_path=pair)
            q=row.pop('quality')
            for w in ['early','middle','late']:
                for k in ['lpips','ssim','psnr']:row[f'{w}_{k}']=q['windows'][w][k]
            detail.append(row)
    with (ROOT/'detail.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    with (REPO/'artifacts/comparison_20261113/summary.csv').open() as f:old=list(csv.DictReader(f))
    aggregate=[]
    for method in ['dense','ours','ours_latest_summary','ours_latest_summary_tail',*METHODS,'solattn','svgear','lightforcing','forcingkv','dasa','pisa']:
        rr=[r for r in (detail if method in METHODS else old) if r['method']==method]
        aggregate.append(dict(method=method,label=LABELS[method],source='current_runtime_ablation' if method in METHODS else '20261113_historical',total_seconds=mean([float(r['total_seconds']) for r in rr]),speedup=geomean([float(r['speedup']) for r in rr]),early_lpips=mean([float(r['early_lpips']) for r in rr]) if method!='dense' else None,late_lpips=mean([float(r['late_lpips']) for r in rr]) if method!='dense' else None))
    with (ROOT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    table=['| 方法 | 平均耗时/s↓ | 加速比↑ | 前段 LPIPS↓ | 后段 LPIPS↓ |','|---|---:|---:|---:|---:|']
    for r in aggregate:
        fmt=lambda k:'—' if r[k] is None else f'{r[k]:.4f}'
        table.append(f'| {r["label"]} | {r["total_seconds"]:.2f} | {r["speedup"]:.3f}× | {fmt("early_lpips")} | {fmt("late_lpips")} |')
    (ROOT/'TABLE.zh.md').write_text('\n'.join(table)+'\n')
    effects=[]
    for before,after,label in [('old_summary_risk','new_summary_risk','替换摘要权重（关闭 Full_Pset）：B−A'),('old_summary_risk','old_summary_full_pset','新增 Full_Pset（旧摘要）：C−A'),('new_summary_risk','current_round8_full_pset','新增 Full_Pset（新摘要）：D−B'),('old_summary_full_pset','current_round8_full_pset','替换摘要权重（开启 Full_Pset）：D−C')]:
        for case in ['horse_field','traffic_crossing','mean']:
            aa=[r for r in detail if r['method']==before and (case=='mean' or r['case']==case)]
            bb=[r for r in detail if r['method']==after and (case=='mean' or r['case']==case)]
            delta={k:mean([r[k] for r in bb])-mean([r[k] for r in aa]) for k in ['total_seconds','generation_seconds','early_lpips','middle_lpips','late_lpips','early_ssim','middle_ssim','late_ssim','early_psnr','middle_psnr','late_psnr']}
            effects.append(dict(comparison=label,case=case,**delta))
    (ROOT/'effects.json').write_text(json.dumps(effects,ensure_ascii=False,indent=2)+'\n')
    lines=['# 摘要权重与新增 Full_Pset 的消融','',
           '四组固定使用已接受的第八轮运行时。A/B 关闭 Full_Pset，C/D 开启；A/C 用旧 summary 权重，B/D 用新 C_s17 权重。未调整其他超参数，未训练新模型。',
           '', '权重核查：只有 29 层的 pool_k、pool_v、pool_feature 改变；bank、group_v、group_gate、rank8 basis 和 Risk logtheta 逐位相同。详见 weight_audit.json。Full_Pset 开启时统一使用当前 Full_Pset_s17 Reader、Adaptive φ 和 B16 投影基，因此 C 是固定修正器配旧摘要的交叉组合，并未专门适配旧摘要。',
           '', '输入沿用原记录：horse_field / traffic_crossing，seed101，768×512，41 chunks，480帧/16FPS（30秒），5个历史 chunks（360页）。固定 Page64/Q64/R4、完整V均值＋rank8、M16/B16、10%精算、单次512候选交换＋no-op。此次继承用户明确指定的原记录质量比较；没有改变常规短视频验收规则。',
           '', '八个单视频组合分配到八张 GPU：A 在6/7，B 在0/1，C 在2/3，D 在4/5。新增使用的0–5卡各自先重测 Dense，并检查其 latent 与原参考逐位一致；A复用刚完成的同卡Dense。D也重新测量。各组完整预热一次，计时两次，重复视频须逐位一致。',
           '', '平均耗时包含生成＋VAE，不含MP4编码与质量计算，先对同场景两次求平均，再对两场景求平均；加速比为逐场景同物理卡Dense配对比值的几何平均。历史行沿用其原始同卡Dense分母，表中Dense行是原记录均值，并非所有行的统一分母。',
           '', '前段为[21,101)，后段为[400,480)，均80帧；LPIPS在256×384计算，使用未压缩RGB。SSIM与PSNR及中段指标保存在detail.csv。所有指标衡量相对同输入Dense的差异。', '', *table,
           '', '## 配对消融结果','', '差值为后者减前者；LPIPS 负值表示更接近 Dense。耗时差跨卡比较，应同时参考各自同卡Dense配对加速比。', '', '| 消融 | 平均耗时差/s | 前段 LPIPS 差 | 中段 LPIPS 差 | 后段 LPIPS 差 |','|---|---:|---:|---:|---:|']
    for r in effects:
        if r['case']=='mean':lines.append(f'| {r["comparison"]} | {r["total_seconds"]:+.3f} | {r["early_lpips"]:+.4f} | {r["middle_lpips"]:+.4f} | {r["late_lpips"]:+.4f} |')
    lines+=['', '## 结论', '', '单换新摘要权重（A→B）几乎不增加计算成本；前段平均LPIPS基本持平，后段由0.6046降到0.5537，改善0.0509。这两个场景的后段收益主要来自新摘要权重。', '', '在新摘要上加入Full_Pset（B→D）后，前段LPIPS从0.3341升到0.3551，后段从0.5537升到0.5581；中段从0.5659降到0.5297。新增模块改善中段，但前后段平均指标没有收益，同时明显增加耗时。', '', '固定旧摘要时加入Full_Pset（A→C），前段LPIPS变差，后段有所改善；因此模块效果随摘要权重和视频阶段变化。C使用同一套当前Reader，不代表重新适配旧摘要后可达到的结果。', '', '本轮重测的D与上一轮完整基线在两个场景上latent逐位一致；新增使用的六张卡Dense也全部逐位一致。变量核对见controlled_variables.json，跨卡基线复核见full_baseline_equivalence.json。', '', '旧 summary 原记录与 A 的权重和算法相同，但运行时不同：RoPE、Risk/QK、Attention与补偿的执行及数值路径有差异，因此历史行与 A 的差值不属于训练权重或新增模块消融。严格归因只使用 A/B/C/D。',
            '', '代码改动：comparison_original.py 增加 summary 选择与 Full_Pset 关闭开关；新增八卡任务入口和本报告生成脚本。运行时/kernel未修改。每组 metadata.json 固定开关与权重SHA，source/保存代码快照；各rep保存GPU进程检查。']
    (ROOT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(table));print(json.dumps([x for x in effects if x['case']=='mean'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
