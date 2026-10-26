"""Eight CPU processes exercise checkpoint steps 10 and 20, with real AdamW."""
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration.runtime import multigpu
from streaming_sparse.v08.common import save,atomic_json


def worker(rank,directory,barrier):
    torch.set_num_threads(1);multigpu.rank=rank
    native=torch.optim.AdamW;calls=[]
    def checked_save(path,value):
        assert rank==0,'A nonzero rank reached the real checkpoint writer'
        calls.append(value['updates']);save(path,value)
    def checked_json(path,value):
        assert rank==0,'A nonzero rank reached the real JSON writer'
        atomic_json(path,value)
    train=SimpleNamespace(save=checked_save,atomic_json=checked_json)
    multigpu.install_checkpoint_io(train,{'test_execution':True})
    assert torch.optim.AdamW is native
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.001)
    assert type(optimizer)is native
    for step in range(1,21):
        optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
        barrier.wait(timeout=60)
        train.save(directory/'recovery.pt',dict(source_sha256='test',updates=step,
            parameter=parameter.detach(),optimizer=optimizer.state_dict(),writer_rank=rank))
        train.atomic_json(directory/'recovery.json',dict(updates=step,writer_rank=rank))
        barrier.wait(timeout=60)
    (directory/f'rank{rank}_writes.json').write_text(json.dumps(calls))


def main():
    context=mp.get_context('fork')
    with tempfile.TemporaryDirectory(prefix='checkpoint_io_')as tmp:
        directory=Path(tmp);barrier=context.Barrier(8)
        children=[context.Process(target=worker,args=(rank,directory,barrier))for rank in range(8)]
        for child in children:child.start()
        for child in children:
            child.join(timeout=120);assert child.exitcode==0,(child.pid,child.exitcode)
        state=torch.load(directory/'recovery.pt',weights_only=True)
        assert state['updates']==20 and state['writer_rank']==0 and state['execution']['test_execution']
        assert int(state['optimizer']['state'][0]['step'])==20
        assert json.loads((directory/'rank0_writes.json').read_text())==list(range(1,21))
        for rank in range(1,8):assert json.loads((directory/f'rank{rank}_writes.json').read_text())==[]
        assert not list(directory.glob('*.tmp'))
    print(json.dumps(dict(passed=True,processes=8,updates=20,writer_rank=0,
        native_adamw_unchanged=True,nonzero_rank_writes=0,steps_10_and_20_passed=True)))


if __name__=='__main__':main()
