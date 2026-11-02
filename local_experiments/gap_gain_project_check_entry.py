"""Reload only this isolated experimental runtime in the persistent session."""
import importlib, runpy
from . import gap_gain_project_runtime
if __name__ == '__main__':
    importlib.reload(gap_gain_project_runtime)
    runpy.run_module('local_experiments.gap_gain_project_check', run_name='__main__')
