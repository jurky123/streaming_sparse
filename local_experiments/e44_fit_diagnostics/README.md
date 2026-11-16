# E44 controlled E43 completion

`core --seed 17/29`: verify identities and train six fixed arms, 2304 updates each.
`physics --seed 17/29`: frozen layer29 teacher four-grid, actual phi, output and parameter gradients.
`report`: summarize completed final experiments and preserve 96x512 action matrices.

Use `python -m local_experiments.e44_fit_diagnostics.<module>` from repo root.

Sparse L uses one Adam per physical page group, exactly96 active updates/group. This avoids dense Adam's inactive clocks, moment decay and updates, but is not equivalent to changing LR by24; neither guarantees convergence. Group members cover all pages of the active two-tile snapshot, and coefficients remain shared across both tiles/states.

Contrast is a fixed diagnostic, not a tuned method: uniform weights, four role means plus R/A centered residuals, both coefficients1; original bs/floors remain. No new inference information or architecture. Teacher and physical identity checks use only original training snapshots; Test sealed.
