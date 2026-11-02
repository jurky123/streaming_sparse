from .high_sparse_output_runtime import OutputRuntime,POLICIES
class TwoStepRuntime(OutputRuntime):
    def __init__(self,engine,checkpoint,policies):
        assert len(policies)==2 and all(p in POLICIES for p in policies)
        self.step_policies=tuple(policies)
        super().__init__(engine,checkpoint,policies[0])
        self.persistent_bytes+=12;assert self.persistent_bytes<=self.budget
    def route(self,l,q,p):
        self.policy_name=self.step_policies[self.engine.current['step']//2]
        return super().route(l,q,p)
    def finish(self):return super().finish()|dict(step_policies=self.step_policies,scope='Two-step x0 supervised hard gate schedule; frozen video model and router coefficients; same 41/410 sparse-call budget')
