"""Run the existing numerical contracts against the conditional action kernel."""
from . import validate
from .scores_action import fused_action_scores
if __name__=='__main__':
    validate.grouped_scores=fused_action_scores
    validate.run('artifacts/video_runtime_v1/hotspot_round4/validation')
