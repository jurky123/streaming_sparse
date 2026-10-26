from pathlib import Path
from ..opt20.setup import ROOT as PARENT
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt21_tile20_runtime_20261025'
ROOT.mkdir(parents=True,exist_ok=True)
