"""Run the same P432 tensor fixture and profiler region used for page prefetch."""
import runpy
import sys
from measure import bench,folder
bench.folder=folder
sys.modules['measure']=bench
runpy.run_path(str(folder.parent/'warp_rows_v1/check_hazards.py'),run_name='__main__')
