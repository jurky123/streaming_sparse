"""Reuse the historical 54-point quota scan for matched short-window calibration."""
import importlib.util,sys
from pathlib import Path
here=Path(__file__).resolve().parent
clean=here.parent/'clean_forward'
cell,prompt,seed=sys.argv[1:]
spec=importlib.util.spec_from_file_location('budget_grid',clean/'scan.py')
m=importlib.util.module_from_spec(spec)
src=Path(spec.origin).read_text().split("if __name__==")[0]
src=src.replace("    install(rt,'frame_fixed15')","    install(rt,'frame_fixed15')\n    configure_pages(rt,cell)")
src=src.replace("        if st != 4: return engine.dense(q,k,v)","")
src=src.replace('torch.zeros(41,1,29','torch.zeros(7,5,29').replace('range(41)','range(7)').replace('data[c,0,l-1]','data[c,st,l-1]').replace('denom[c,0,l-1]','denom[c,st,l-1]')
from cells import install as configure_pages
m.__dict__.update(configure_pages=configure_pages,cell=cell)
exec(compile(src,spec.origin,'exec'),m.__dict__)
m.ROOT=here/'calibration'/cell
m.ROOT.mkdir(parents=True,exist_ok=True)
m.main(prompt,int(seed))
