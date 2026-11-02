"""Profile the accepted probe-projection runtime in the reserved session."""
import sys
from pathlib import Path
from . import gap_staged_history_profile as profile
from .gap_route_qp_runtime import GapRouteQpRuntime
if __name__ == '__main__':
    profile.GapFeatureRuntime = GapRouteQpRuntime
    profile.ROOT = Path('artifacts/gap_route_qp_20261102/profile_gpu7')
    sys.argv = ['gap_route_qp_profile', '--variant', 'features']
    profile.main()
