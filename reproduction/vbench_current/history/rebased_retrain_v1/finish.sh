set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
cd "$exp"
while [ ! -f quality_0.json ] || [ ! -f quality_1.json ]; do sleep 10; done
bash compare.sh
while [ "$(find . -maxdepth 1 -name '*_repeat*.json' | wc -l)" -ne 12 ] || [ "$(find . -maxdepth 1 -name 'reader_only_quality*.json' | wc -l)" -ne 4 ] || [ "$(find budget* -name dev_reader_only.json | wc -l)" -ne 2 ]; do sleep 10; done
python ledger.py
(cd /home/shuchenweng/jzk/streaming_sparse && "$py" "$exp/data_audit.py")
"$py" final_audit.py
python report_results.py
