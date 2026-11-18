"""Run the normal tier benchmark with per-layer summary caching enabled for base."""
from adapter import CachedAttention
from local_experiments.video_runtime_v1 import accepted_tiers, fixed15_attention

if __name__=='__main__':
    fixed15_attention.PackedAttention=CachedAttention
    accepted_tiers.main()
