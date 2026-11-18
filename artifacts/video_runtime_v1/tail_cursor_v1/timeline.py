"""Reuse the nine-chunk CUDA timeline with the selected mask source."""
import importlib.util
from pathlib import Path
import sys
import ctypes

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("timeline", root.parent / "history_summary_v1/timeline.py")
timeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline)
variant = sys.argv.pop(1)
make_runtime = timeline.make_runtime

def runtime(*args):
    rt = make_runtime(*args)
    if variant=='maskreg':
        fn=ctypes.CDLL('/tmp/jzk_page_maskreg.so').run
        fn.argtypes=rt.packed_attention.kernel.argtypes
        rt.packed_attention.kernel=fn
    return rt

timeline.make_runtime = runtime
timeline.main()
