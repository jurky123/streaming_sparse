# E36 Query coverage

Run A with the frozen E29/E30 full64 caches and canonical dev data:

```bash
python -m local_experiments.e36_query_coverage.run --seed 17
python -m local_experiments.e36_query_coverage.run --seed 29
python -m local_experiments.e36_query_coverage.report
python -m local_experiments.e36_query_coverage.audit
python -m local_experiments.e36_query_coverage.finish
```

A gate failed in this run. B is not executed. `--stage B` requires the persisted A gate to pass. Rerunning a completed run requires a separate output directory; no overwrite of complete runs. Full panels are in A_s17/A_s29; compact review tar contains reports/CSVs/audits/source only. Development has 288 events total per model seed, 144 per trajectory. Test remains sealed.
