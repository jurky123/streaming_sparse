"""Commit-only fused history features on the staged-route SingleTail baseline."""
import types
from .gap_staged_single_runtime import GapStagedSingleRuntime
from .gap_group_fused_features import group_features

class GapStagedHistoryRuntime(GapStagedSingleRuntime):
    def commit(self, l, *args, **kwargs):
        if self.training_event:
            return super().commit(l, *args, **kwargs)
        model = self.model[l]
        original = model.group_features
        try:
            model.group_features = types.MethodType(group_features, model)
            return super().commit(l, *args, **kwargs)
        finally:
            model.group_features = original

    def finish(self):
        return super().finish() | dict(history_features='Fused spatial response/RMS/gate; approximate projection and reduction; no extra retained history')
