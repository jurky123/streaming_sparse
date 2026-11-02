"""Profile composed runtime in the reserved experiment sequence."""
import sys
from .gap_staged_history_profile import main
if __name__=='__main__':
    sys.argv=['gap_staged_history_profile','--variant','features']
    main()
