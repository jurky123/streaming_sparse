"""Reuse the validated pixel/video evaluator with declared layers and labels."""
import argparse
import json
import os
from pathlib import Path
import sys
from migration.paired_router import runtime


def main():
    os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[2]/'.cache/torch'))
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    manifest,spec,_=runtime.configuration(a.directory,a.stage)
    from streaming_sparse.v08 import evaluate,router
    from migration.frozen_quality import worker
    evaluate.load_network=lambda path:runtime.load_checkpoint(path,spec['layers'])
    router.InformationRouter=lambda kind:runtime.BaseRouter(kind,pairs=len(spec['layers'])*12)
    source=Path(worker.__file__).read_text().replace('K200','Klast').replace('V200','Vlast').replace('Kpaused','Vbest')
    scope={'__name__':'paired_quality_evaluator','__file__':worker.__file__};exec(compile(source,'<declared paired quality labels>','exec'),scope)
    sys.argv=[sys.argv[0],'--directory',str(Path(spec['directory'])/'quality')];scope['main']()


if __name__=='__main__':main()
