"""Use the accepted tier CLI, substituting the tested candidate library for base."""
import ctypes
from local_experiments.video_runtime_v1 import accepted_tiers, fixed15_attention

if __name__=='__main__':
    reference=fixed15_attention.library()
    candidate=ctypes.CDLL('/tmp/jzk_slope_last_page.so')
    for name in ('run','prepare_fixed15','commit_fixed15'):
        getattr(candidate,name).argtypes=getattr(reference,name).argtypes
    fixed15_attention.library=lambda: candidate
    accepted_tiers.main()
