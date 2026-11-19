cd /home/shuchenweng/jzk/streaming_sparse
exp=/mnt/data/vbench_run_20261118/previews/rebased_retrain_v1
prev=/mnt/data/vbench_run_20261118/previews/rebased_budget_ablation_v1
export PYTHONPATH="$exp:$prev:/mnt/data/vbench_run_20261118:$PWD"
py=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
