"""Hard routing calibration policies, no differentiable top-k surrogate."""
from .high_sparse_gain_runtime import GainRuntime
POLICIES={'off':(0.,0.,0.),'half':(.5,.5,.5),'all':(1.,1.,1.),'early':(1.,0.,0.),'middle':(0.,1.,0.),'late':(0.,0.,1.)}
class OutputRuntime(GainRuntime):
    def __init__(self,engine,checkpoint,policy='off'):
        super().__init__(engine,checkpoint,0.);self.policy_name=policy
        self.persistent_bytes+=12;assert self.persistent_bytes<=self.budget
    def route(self,l,q,p):
        self.strength=POLICIES[self.policy_name][min((l-1)//10,2)]
        return super().route(l,q,p)
    def finish(self):return super().finish()|dict(output_policy=self.policy_name,layer_band_gates=POLICIES[self.policy_name],scope='Output-supervised hard-policy calibration; frozen gain router, V and video model; sparse-call budget 90%')
