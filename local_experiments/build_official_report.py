"""Render a local Markdown research report from completed measurements."""
import argparse,html,json
from pathlib import Path
from statistics import mean
from .report_markdown import html_to_markdown
METHODS=['dense','lightforcing','dasa','forcingkv']
NAMES=['Dense','Light Forcing','DASA','Forcing-KV']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');a=p.parse_args();root=Path(a.experiment)
 summary=json.loads((root/'analysis/summary.json').read_text());rows=summary['rows'];manifest=json.loads((root/'manifest.json').read_text())
 table='<table><tr><th>方法</th><th>KV GiB</th><th>辅助状态 MiB</th><th>生成 s</th><th>chunk 中位 s</th><th>亮度变化 pp ↓</th><th>最大短暗降 pp ↓</th><th>自身开头色度漂移 ↓</th><th>光流 px/s</th><th>DINO 邻秒 cosine</th><th>逻辑 QK 量 / 满预算</th></tr>'
 metrics=['max_kv_gib','auxiliary_mib','generation_seconds','steady_chunk_seconds_median','brightness_step_mean_pp','dark_dip_max_pp','chroma_drift_from_own_opening_mean','flow_mean_pixels_per_second','dino_adjacent_1s_cosine','attention_pairs_vs_full_cap']
 for m,n in zip(METHODS,NAMES):
  data=[r for r in rows if r['method']==m];table+='<tr><td>'+n+'</td>'+''.join(f'<td>{mean(r[k] for r in data):.3f}</td>' for k in metrics)+'</tr>'
 table+='</table>'
 gallery=''
 for case in manifest['cases']:
  prompt=(root/case/'prompt.txt').read_text()
  gallery+=f'<h3>{case}</h3><details><summary>完整 prompt</summary><pre>{html.escape(prompt)}</pre></details><video controls preload="none" poster="{case}/contact_sheet.png" src="{case}/comparison.mp4"></video><p>'
  gallery+=' · '.join(f'<a href="{case}/{m}.mp4">{n} 原视频</a>' for m,n in zip(METHODS,NAMES))+'</p>'
  gallery+=f'<img src="{case}/contact_sheet.png" alt="{case} 场景的四行时间采样图，行顺序与方法表一致，列为 0 至 29.94 秒。">'
 versions='<table><tr><th>来源</th><th>commit</th><th>使用状态</th></tr><tr><td>streaming_sparse / dense</td><td>30ae30521450467f1dcbd5ec848f14d4a8ea89cc</td><td>运行</td></tr>'
 for name,commit in manifest['source_commits'].items():versions+=f'<tr><td>{name}</td><td><code>{commit}</code></td><td>'+('仅调研/源码下载' if name=='DeepForcing' else '运行')+'</td></tr>'
 versions+='</table>'
 findings_path=root/'analysis/findings.html'
 findings=findings_path.read_text() if findings_path.exists() else '<p>所有结果已生成。请结合下表、逐场景视频及实现限制评估；此新运行尚未附加人工帧图观察。</p>'
 template=(Path(__file__).parent/'baseline_survey/report_template.html').read_text()
 if not (root/'streaming_sparse_comparability.zh.md').exists():
  template=__import__('re').sub(r'<p>补充核查：.*?streaming_sparse_comparability\.zh\.md.*?</p>\n?', '', template)
 for key,value in [('FINDINGS',findings),('METRIC_TABLE',table),('VIDEO_GALLERY',gallery),('VERSION_TABLE',versions)]:template=template.replace('{{'+key+'}}',value)
 (root/'report.zh.md').write_text(html_to_markdown(template, root))
 print(root/'report.zh.md')
if __name__=='__main__':main()
