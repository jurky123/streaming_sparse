from pathlib import Path
from ..opt20.setup import ROOT as TRAINING
from ..opt22.setup import ROOT as PARENT
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt23_tile20_scaling_20261025'
ROOT.mkdir(parents=True,exist_ok=True)
CASES=[(480,832,4),(480,832,7),(720,1280,4),(720,1280,7),(960,1664,4),(960,1664,7)]
def tag(h,w,chunks):return f'{h}x{w}_chunks{chunks}'
