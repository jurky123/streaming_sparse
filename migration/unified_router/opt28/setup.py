from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt28_hardware_tiles_20261025'
ROOT.mkdir(parents=True,exist_ok=True)
from ..opt24.setup import tag,ROOT as VENDOR
CONFIGS=[(16,64,16),(32,64,16),(32,64,32),(64,64,16),(64,64,32),(64,64,64),(128,64,16),(128,64,32),(128,64,64),(256,64,32),(256,64,64),(16,128,16),(32,128,16),(32,128,32),(64,128,16),(64,128,32),(64,128,64),(128,128,32),(128,128,64),(256,128,64)]

CONFIGS += [(128,128,16),(256,128,32)]
