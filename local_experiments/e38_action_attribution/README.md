# E38 action attribution

Run B for seeds17/29, then report --stage B; run A for seeds17/29, report --stage A; audit; finish. Modules under local_experiments.e38_action_attribution. CUDA FP64 and canonical E24/E30 caches required. B reconstructs frozen summary and true local response from existing dev KV with explicit user authorization; A uses cache. No new capture or Test.

Protocol is fixed before runs. C and D calibration gates failed; no extreme statistics or threshold training performed. Finished runs refuse overwrite. Pilot --limit does not mark completion. Report archive contains tables/source/audit, not full QKV or large per-event targets. All gains in attribution use a common Q64 full-energy normalizer; M16 selection replay is unchanged.
