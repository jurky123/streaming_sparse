"""Fetch pinned official baselines without resetting existing user changes."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args,**kwargs):return subprocess.run(list(args),check=True,**kwargs)
def main():
    base=ROOT/'external_baselines';lock=json.loads((base/'sources.lock.json').read_text())
    for name,spec in lock.items():
        path=base/name
        if not path.exists():
            run('git','clone','--no-checkout',spec['url'],str(path))
            run('git','-C',str(path),'checkout','--detach',spec['commit'])
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if head!=spec['commit']:raise RuntimeError(f'{name}: existing HEAD differs; preserve local checkout and resolve explicitly')
        patch=base/'patches'/f'{name}.patch'
        if patch.stat().st_size:
            applied=subprocess.run(['git','-C',str(path),'apply','--reverse','--check',str(patch)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
            if not applied:
                run('git','-C',str(path),'apply','--check',str(patch))
                run('git','-C',str(path),'apply',str(patch))
        code=path/'DASA' if name=='DASA' else path
        link=code/'wan_models'
        if not link.exists() and not link.is_symlink():link.symlink_to(ROOT/'wan_models',target_is_directory=True)
        print(name,head,'ready')
if __name__=='__main__':main()
