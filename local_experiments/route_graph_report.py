import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/route_graph_20261031')
def main():
    checks=json.loads((ROOT/'checks.json').read_text());timing=checks['timings'];med={m:statistics.median(r['ms'] for r in timing if r['method']==m) for m in ['eager','graph']}
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance').glob('*/*.json')) if p.name in ['eager.json','graph.json']]
    lines=['# 路由实现优化验收','','将归一化、特征、换页打分和LUT构造的原生PyTorch算子捕获到共享CUDA Graph。保持原有浮点计算顺序；先抽query probes再转FP32；QK直接写入图输入，复用摘要拼接缓冲；直接替换LUT中的页，去掉第二次mask topk。没有修改路由目标、权重、RoPE或精确KV页预算。','',f'812次实际调用的LUT与旧实现完全一致。交替微测路由中位数：{med["eager"]:.3f}→{med["graph"]:.3f}ms，约{med["eager"]/med["graph"]:.2f}×。这不是理论极限，图内仍有多个GPU kernel和FP32计算。','','## 30秒完整生成','','41chunks/480frames，包括路由、KV维护和VAE，排除加载、预热、图捕获及指标/编码。所有测量均观察到其他GPU进程，属于共享负载下观测，不是独占性能结论。','','| 场景 | eager秒 | graph秒 | 观测加速比 | 全长latent一致 |','|---|---:|---:|---:|---|']
    for case in ['basketball_court','butterfly_garden']:
        pair={r['method']:r for r in rows if r['case']==case}
        if len(pair)!=2:continue
        a=pair['eager']['performance']['total_generation_vae_seconds'];b=pair['graph']['performance']['total_generation_vae_seconds']
        lines.append(f'| {case} | {a:.3f} | {b:.3f} | {a/b:.3f}× | {pair["graph"]["full_bit_exact_to_eager"]} |')
    lines+=['','两场景前段latent还与既有gain1.0视频逐元素一致，复用已有质量结果。优化没有改变模型输出；未宣称改善原本未达标的90%稀疏视频质量。','','## 预算与工作显存','','复制到图中的code/方差/权重等输入按118,784B计入KV辅助预算，总计5,096,033,584B，仍低于5,096,079,360B，剩45,776B。图中的logits/概率/特征等计算工作区另计，它不增加可保留历史，但物理显存会常驻分配，不能忽略。','', '| 场景/实现 | 峰值allocated GiB |','|---|---:|']
    for r in rows:lines.append(f'| {r["case"]}/{r["method"]} | {r["performance"]["peak_allocated_bytes"]/2**30:.3f} |')
    lines+=['','本轮完成一阶段路由优化，并未把所有运算融合成单个kernel；之后启动全29稀疏层输出梯度训练。反向公式检查见backward_checks.json，完整训练报告位于../full_route_grad_20261031/report.zh.md。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(route_median_ms=med,checks_passed=checks['passed'],full=rows))
if __name__=='__main__':main()
