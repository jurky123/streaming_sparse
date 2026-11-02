"""Run resumable generation first; evaluate only after all methods complete."""
import runpy
if __name__=='__main__':
    runpy.run_module('local_experiments.gap_stream_eval_suite',run_name='__main__')
    runpy.run_module('local_experiments.gap_stream_eval_quality',run_name='__main__')
