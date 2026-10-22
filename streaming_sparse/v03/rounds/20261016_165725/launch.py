import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
project=root.parents[3]
if (root/'launcher.json').exists():
    raise RuntimeError('This fresh round has already been launched; inspect status.json')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'logs/runner.log').open('w') as stream:
    child=subprocess.Popen([sys.executable,'-u','-m','streaming_sparse.v03.run_round','--root',str(root)],cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launcher.json').write_text(json.dumps({'pid':child.pid,'gpu_index':0,'cwd':str(project),'root':str(root)},indent=2))
print(json.dumps({'pid':child.pid,'root':str(root)}))
