"""Verify completion, protocol budgets, shared inputs and stored artifacts."""
import csv
import json
import math
from pathlib import Path
import subprocess
import shutil
import torch
from .run_kv_history import ROOT,sha,write_json,inspect_video
from .run_cache_video_suite import ARMS


def main():
    root=ROOT/'artifacts/cache_research_20261029';checks=[];cap=5096079360
    for case in ['sailboat','skateboard','dog']:
        capture=json.loads((root/'capture'/case/'research.json').read_text())
        assert capture['identical_to_previous_dense'] and capture['unchanged_history_checks']==36
        inputs=ROOT/'artifacts/ar_official_baselines_20261029'/case/'shared_inputs.pt'
        for area,methods in [('video',list(ARMS)),('followup',['k_guard'])]:
            folder=root/area/case
            for method in methods:
                result=json.loads((folder/f'{method}.result.json').read_text());research=result['research']
                assert result['completed'] and result['chunks']==41
                assert result['checkpoint_sha256']=='a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3'
                assert result['shared_inputs_sha256']==sha(inputs)
                assert result['max_self_kv_storage_bytes']==cap
                assert result['actual_transition_noise_calls']==123
                assert research['first_chunk_identical'] and research['sparse_calls']==4640
                assert research['persistent_summary_or_duplicate_kv_bytes']==0
                assert all(r['exact_history_tokens']+r['summary_equivalent_tokens']<=r['history_budget_tokens'] for r in research['budget_trace'])
                if method!='spatial':
                    assert research['preprocess_calls']==1160
                    assert min(research['reuse_counts'])==4 and max(research['reuse_counts'])==20
                    assert sum(research['reuse_counts'])/len(research['reuse_counts'])==19
                info=inspect_video(folder/f'{method}.mp4')
                assert (info['width'],info['height'],info['frames'],info['fps'],info['duration'])==(768,512,480,16.,30.)
                checks.append(dict(case=case,method=method,video=info,video_sha256=sha(folder/f'{method}.mp4'),
                    kv_bytes=result['max_self_kv_storage_bytes'],aux_bytes=research['persistent_auxiliary_permutation_peak_bytes']))
    screen=list(csv.DictReader((root/'screen/metrics.csv').open()));assert len(screen)==7776
    diagnosis=list(csv.DictReader((root/'diagnostic/metrics.csv').open()));assert len(diagnosis)==16704
    assert all(math.isfinite(float(r['nmse'])) for r in screen+diagnosis)
    assert all(math.isfinite(float(r['tail_mass_mae'])) for r in diagnosis if r['mode']=='var_tail')
    assert json.loads((root/'diagnostic/research.json').read_text())['dense_latent_identical']
    assert len(json.loads((root/'analysis/video_summary.json').read_text())['rows'])==18
    assert len(json.loads((root/'analysis/followup_summary.json').read_text())['rows'])==3
    source_names=['cache_research.py','cache_grouping.py','cache_sparse_prototype.py','screen_cache_research.py',
      'diagnose_cache_layers.py','run_cache_video_suite.py','run_cache_followup.py','benchmark_cache_interleaved.py',
      'analyze_cache_research.py','analyze_cache_followup.py','probe_rope_statistics.py','run_official_baseline.py']
    sources={name:sha(ROOT/'local_experiments'/name) for name in source_names}
    snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in source_names:shutil.copyfile(ROOT/'local_experiments'/name,snapshot/name)
    (root/'hardware_at_audit.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True))
    result=dict(passed=True,new_videos=18,reused_dense_videos=3,screen_rows=7776,diagnosis_rows=16704,
        unchanged_history_checks=108,screen_unique_query_instances=20736,methods=list(ARMS)+['k_guard'],
        torch_version=torch.__version__,source_sha256=sources,videos=checks)
    write_json(root/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['videos','source_sha256']}))


if __name__=='__main__':main()
