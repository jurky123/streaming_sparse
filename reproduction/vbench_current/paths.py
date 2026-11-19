from pathlib import Path
import os
CODE=Path(__file__).resolve().parent
REPO=CODE.parents[1]
RUN=Path(os.environ['VBENCH_RUN']).expanduser().resolve()
