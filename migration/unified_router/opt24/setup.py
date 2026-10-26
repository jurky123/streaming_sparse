from pathlib import Path
BASE=Path(__file__).resolve().parents[3]
ROOT=BASE/'streaming_sparse/v10/experiments/opt24_attention_backends_20261025'
CAPTURE=BASE/'streaming_sparse/v10/experiments/opt23_tile20_scaling_20261025/capture'
TRAINING=BASE/'streaming_sparse/v10/experiments/opt20_all29_tile_finetune_20261025'
ROOT.mkdir(exist_ok=True,parents=True)
def tag(h,w,c):return f'{h}x{w}_chunks{c}'
