"""Verify complete latent and encoded-video equivalence to preceding fused path."""
import json,hashlib,time,torch,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--method',default='ours_latest_stable');p.add_argument('--reference',default='ours_latest_fused');p.add_argument('--output-name',default='equivalence_vs_fused.json');a=p.parse_args()
root=Path('/tmp/streaming_sparse-0/runtime/artifacts/comparison_20261113');torch.set_num_threads(4)
for case in ['horse_field','traffic_crossing']:
 folder=root/case/a.method;ref=root/case/a.reference
 while not (folder/'video.mp4').exists():time.sleep(2)
 actual=torch.load(folder/'latents.pt',map_location='cpu',weights_only=True);b=torch.load(ref/'latents.pt',map_location='cpu',weights_only=True)
 delta=actual.float()-b.float();equal=torch.equal(actual,b)
 row=dict(reference=a.reference,bitwise_equal=equal,shape=list(actual.shape),latent_nmse=float(delta.square().mean()/b.float().square().mean()),max_abs=float(delta.abs().max()),video_sha256_equal=hashlib.sha256((folder/'video.mp4').read_bytes()).digest()==hashlib.sha256((ref/'video.mp4').read_bytes()).digest(),scope='Both complete saved latent tensor and encoded video compared; same model, inputs, native VAE, and matched GPU type')
 (folder/a.output_name).write_text(json.dumps(row,indent=2));print(case,row,flush=True)
