"""Publish byte-identical trained checkpoints and prompt embeddings (no backbone)."""
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/weights_20261113'
ASSETS={
 'summary_pretrained':'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',
 'full_calibrated':'artifacts/review_v1_20261102/route_expanded/model.pt',
 'two_step_distilled':'artifacts/review_v1_20261102/distill/city_walk/latest.pt',
 'downstream_router_pilot':'artifacts/review_v1_20261102/route_downstream/fit/model.pt',
 'single_step_flow':'artifacts/flow_single_20261104/train/reef_fish/model.pt',
 'legacy_sparse':'artifacts/four_gaps_20261031/shared/epoch1.pt',
 'legacy_route_initializer':'artifacts/full_route_grad_20261031/epoch1.pt',
 'legacy_compensation':'artifacts/compensation_probe_20261030/model.pt',
 'legacy_dictionary':'artifacts/broad_coverage_20261030/dictionary.pt',
 'legacy_pool':'artifacts/mass_variance_20261030/train_00_30/best.pt',
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);assets=[]
 sources=dict(ASSETS)
 for case in ['city_walk','reef_fish','horse_field','traffic_crossing']:
  sources['embedding_'+case]=f'artifacts/broad_coverage_20261030/embeddings/{case}.pt'
 for name,rel in sources.items():
  src=ROOT/rel;out=OUT/(name+'.pt.gz')
  with src.open('rb') as inp,out.open('wb') as raw:
   with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=6) as z:shutil.copyfileobj(inp,z)
  assert out.stat().st_size<50*2**20
  assets.append(dict(name=name,path=rel,payload=out.name,bytes=src.stat().st_size,sha256=sha(src),compressed_bytes=out.stat().st_size,compressed_sha256=sha(out)))
  print(name,out.stat().st_size,flush=True)
 manifest=dict(schema=1,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),selected='summary_pretrained',assets=assets,
  protocol=dict(height=512,width=768,history_pages=338,layer0_history_pages=360,page_tokens=64,probes=16,exact_density=.1,early_window=[21,101],seed=101,backbone_frozen=True),
  scope='Exact original checkpoint bytes, including optimizer states when present; inference/ablation reproduction, not complete historical training resume state.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
