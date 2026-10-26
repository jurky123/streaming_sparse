from pathlib import Path
from ..opt21.setup import ROOT as PARENT
from ..opt20.setup import ROOT as TRAINING
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt22_tile20_backbone_fusion_20261025'
ROOT.mkdir(parents=True,exist_ok=True)
