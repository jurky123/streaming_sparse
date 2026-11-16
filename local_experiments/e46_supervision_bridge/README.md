# E46 supervision and scope bridge

Run `core --seed 17/29 --mode align` and `core --seed 17/29 --mode layer`.
Run `evaluate --seed 17/29 --phase frozen`, then `--phase trained` after training.
Run `audit --seed 17/29`, then `report`.
Module prefix: `python -m local_experiments.e46_supervision_bridge.`

M16 set targets retain the original Q64 common scoring normalization and Q64 gain scale. Only target changes. Layer split has three Readers initialized from E29; same2304 total scheduled updates, 768 per layer. Shared T0 reused from E39, full baseline frozen evaluation reused from E42. Test sealed. Original artifacts are required; code alone does not regenerate data/models.
