"""Full-12-head views; one aligned query tile for all four diffusion steps."""
import random
from ..v03.data import SnapshotData as LegacySnapshotData


class SnapshotData(LegacySnapshotData):
    def __init__(self,root,split,require_complete=True):
        super().__init__(root,split,heads=tuple(range(12)),require_complete=require_complete)


class BalancedSchedule:
    """Cover snapshots without replacement; rotate through eight tiles per snapshot.

    Kept separate from model RNG. At 6000x4 draws, the 3456 training snapshots
    receive 6/7 visits, not a claim of exhaustive eight-tile training coverage.
    """
    def __init__(self,versions,seed):
        self.versions=sorted(versions)
        if not self.versions:
            raise ValueError('Empty schedule')
        self.rng=random.Random(seed)
        self.pool=[]
        self.visits={v:0 for v in self.versions}
        self.offset={v:self.rng.randrange(8) for v in self.versions}

    def sample(self,count=4):
        if not 0<count<=len(self.versions):raise ValueError('Batch must fit unique snapshot set')
        chosen=[]
        used=set()
        for _ in range(count):
            if not self.pool:
                self.pool=list(self.versions)
                self.rng.shuffle(self.pool)
            index=next(i for i in range(len(self.pool)-1,-1,-1) if self.pool[i] not in used)
            version=self.pool.pop(index)
            used.add(version)
            tile=(self.offset[version]+self.visits[version])%8
            self.visits[version]+=1
            chosen.append((version,tile))
        return chosen

    def state_dict(self):
        return dict(rng=self.rng.getstate(),pool=self.pool,visits=self.visits,offset=self.offset,
                    versions=self.versions)

    def load_state_dict(self,value):
        if value['versions']!=self.versions:
            raise ValueError('Schedule dataset changed')
        self.rng.setstate(value['rng'])
        self.pool=list(value['pool'])
        self.visits=dict(value['visits'])
        self.offset=dict(value['offset'])
