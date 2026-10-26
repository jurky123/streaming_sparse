"""Aggregate paired per-prompt medians, keeping generation and VAE scopes explicit."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt25_720p_e2e_20261025'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 rows=[]
 for i in range(8):
  p=ROOT/'720x1280'/f'case{i}';done=read(p/'completed.json');assert done['completed'] and done['first_chunk_exact']
  t=read(p/'timing.json');d=read(p/'decode.json');assert all(v['repeat_exact'] for v in t['results'].values())
  rows.append(dict(case=i,prompt_id=t['prompt_id'],generation=t['results'],decode=d))
 labels=['Dense','Opt22','Sparge32'];summary={}
 for label in labels:
  gen=statistics.mean(r['generation'][label]['seconds'] for r in rows);dec=statistics.mean(r['decode'][label]['seconds'] for r in rows)
  summary[label]=dict(generation_seconds=gen,decode_seconds=dec,generation_plus_decode_seconds=gen+dec,peak_generation_gb=max(r['generation'][label]['peak_gb'] for r in rows))
 for label in labels:
  summary[label]['generation_speedup_vs_dense']=summary['Dense']['generation_seconds']/summary[label]['generation_seconds']
  summary[label]['with_decode_speedup_vs_dense']=summary['Dense']['generation_plus_decode_seconds']/summary[label]['generation_plus_decode_seconds']
  summary[label]['paired_speedups']=[r['generation']['Dense']['seconds']/r['generation'][label]['seconds'] for r in rows]
 for label in labels[1:]:
  summary[label]['keep']=statistics.mean(r['generation'][label]['keep'] for r in rows)
  summary[label]['execution']=statistics.mean(r['generation'][label]['execution'] for r in rows)
 data=dict(completed=True,scope='A100-SXM4-40GB, B=1,720x1280,45 frames,4 chunks,4 denoising steps/chunk,8 Val prompts,seed0,frozen checkpoint,layer0 and first chunk Dense,other29 layers20% budget,Q32K64',timing='Per-case median of5 warmed complete rollouts; aggregate arithmetic mean across8 cases. Each case runs all3 variants on same GPU; variant order rotates. Includes clean-history encoding and preparation, router, model projections/FFN,scheduler,conditioning,latent transfers and diagnostics. Excludes initial model load/compilation/graph capture, video encoding and disk IO. VAE is measured separately (2 warmed repeats) and added for with-decode totals.',quality='Finite outputs, repeated rollout bit-exactness, first dense chunk equality checked. No 720p perceptual quality score in this speed benchmark.',summary=summary,cases=rows)
 write(ROOT/'summary.json',data)
 lines=['720p完整视频生成测速（Opt25）','',data['scope'],data['timing'],'','方案         生成(s)    VAE(s)    合计(s)    生成加速    含VAE加速']
 for k,v in summary.items():lines.append(f'{k:10} {v["generation_seconds"]:9.4f} {v["decode_seconds"]:9.4f} {v["generation_plus_decode_seconds"]:9.4f} {v["generation_speedup_vs_dense"]:9.4f}x {v["with_decode_speedup_vs_dense"]:9.4f}x')
 lines+=['','逐提示词：Dense / Opt22 / Sparge32，单位秒']
 for r in rows:lines.append(f'case{r["case"]}: '+ ' / '.join(f'{r["generation"][k]["seconds"]:.4f}' for k in labels))
 lines+=['','保留Q32原路由语义；本次将硬编码480p尺寸参数化到720p。Opt22/24原默认入口与checkpoint未修改。所有方案采用相同的有界历史KV缓存，稀疏方案在不同chunk图之间复用页缓存；原720p入口曾因重复持有KV显存不足，该失败不计入结果。','480p回归验证：Dense、Opt22、Sparge32三者均与各自已保存45帧latent逐元素完全一致。','720p各方案5次计时重复输出完全一致，首chunk Dense完全一致。没有把算子加速当作端到端加速。','这是速度测试，尚未新增720p视频感知质量评估。','查看状态：./opt25_status.sh','复现：CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt25.evaluate --case 0']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 checks=read(ROOT.parent/'opt24_attention_backends_20261025/FINAL_VERIFICATION.json')
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 unchanged=all(sha(p)==s for p,s in checks['files'].items());assert unchanged
 assert read(ROOT/'480x832/case0/completed.json')['regression_480p']
 locked=read(ROOT.parent/'opt20_all29_tile_finetune_20261025/chosen_checkpoints.json');assert len(locked)==58 and all(sha(p)==s for p,s in locked.items())
 old=read(ROOT.parent/'opt22_tile20_backbone_fusion_20261025/FINAL_VERIFICATION.json');assert all(sha(p)==s for p,s in old['files'].items())
 write(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,cases=8,regression_480p=True,first_chunk_exact=True,deterministic_repeats=True,opt24_sources_unchanged=unchanged,all58_checkpoints_unchanged=True,opt22_sources_unchanged=True,source_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
