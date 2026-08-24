# Paired-seed statistical revision

Starting commit: `1982f26a52ed44474cc13af78b91a83fea7136b4`

Branch: `stats/paired-seed-manuscript-revision`

The revision preserves `analysis/statistics_revision/seed_block/` and writes all new results here. Source inputs were hashed before statistical code changes and are rechecked at completion.

The primary estimand is the marginal contrast among each protocol's QC-qualified outputs under a joint bootstrap of nominal seed labels. Each sampled label is used in every compared condition, while each condition retains its own surviving AF2-model strata. Sensitivities restrict to common contributing seeds and common model-seed survivors. Actual RNG integers were not recoverable; numeric seed labels are reported as nominal design keys rather than verified random-seed equality.

Continuous outcomes reduce retained recycles to prespecified trajectory medians, weight available AF2 models equally within seed, and weight contributing seeds equally. Categorical outcomes use named trajectory fractions or earliest/latest retained snapshots. Tetramer subunits are summarized before inference.

Focal W1 uncertainty recomputes W1 within joint seed-bootstrap replicates; permutation tests swap complete condition blocks within nominal seed labels. Full-panel outputs remain effect-size discovery results without mass-univariate P or q values.

QC retention uses the five nominal AF2 model strata per seed, with missing or failed trajectories contributing zero. Geometry among survivors and QC-adjusted usable target yield are reported separately.

Publication mode uses 2,000 whole-seed bootstrap replicates, 9,999 focal W1 permutations, and deterministic named RNG streams derived from base seed `20260824`.
