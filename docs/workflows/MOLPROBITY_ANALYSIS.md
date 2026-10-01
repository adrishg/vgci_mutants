# Local MolProbity analysis

The dedicated notebooks are:

- `cav12/Cav12_MolProbity_clashscore_analysis.ipynb`
- `kv21/Kv21_MolProbity_clashscore_analysis.ipynb`
- `nav15/Nav15_MolProbity_clashscore_analysis.ipynb`

Inputs remain in `molprobity_download/<run>/<run>.csv`. Set
`VGIC_MOLPROBITY_ROOT` to an alternative downloaded directory. The single
`load_molprobity` function in `shared/molprobity_analysis.py` is the migration
boundary: after an explicitly requested dataset migration, substitute an
explicit file manifest and `shared.data_access.resolve` calls there. No data
upload, relocation, or change to the existing Hugging Face mapping is performed.

Existing convergence manifests are read from the repository. Kv2.1's final
structural-QC distance tables and chain-resolved conformational table use the
existing local/cache/Hugging Face resolver. They were already locally available
for this execution. A fresh checkout also needs those existing analysis inputs;
the resolver can obtain them from the companion dataset when network access is
enabled. `VGIC_OFFLINE=1` forbids fetching, as elsewhere in this repository.

Each notebook writes only `<channel>/dataExtra/molprobity/tables/` and
`<channel>/dataExtra/molprobity/figures/`. The tables include source CSV SHA-256
hashes, per-ensemble audit, explicit row anomalies, missing snapshots, manifest
discrepancies, QC selection counts, N/median/quartiles/IQR/mean/SD/range,
seed-aware contrasts, pairing sensitivities, AF-model/recycle adjustments,
AF-model-specific and leave-one-AF-model-out effects, and software versions.
Kv2.1 additionally writes exact conformation joins, experimental rotation
references, correlations, adjusted regressions, and slope-interaction tests.
Figures use the project's shared palettes/style and PNG (300 dpi) plus PDF.

## Run and regenerate

Open any notebook from the repository root or its channel directory and run all
cells. Dependencies extend the existing scientific Python environment with
SciPy, statsmodels, nbformat and IPython.

To execute all three without requiring Jupyter kernel TCP ports:

```bash
VGIC_OFFLINE=1 python scripts/run_molprobity_notebooks.py
```

The runner uses an in-process IPython shell in a fresh Python subprocess per
notebook, executes every code cell, stores rich notebook outputs and stops on an
error. It intentionally launches from each channel directory to check relative
paths. Normal Jupyter/nbclient execution is also supported.

To recreate notebook source cells (clears their generated outputs first):

```bash
python scripts/generate_molprobity_notebooks.py
python scripts/run_molprobity_notebooks.py
python -m pytest tests/test_molprobity_analysis.py -q
```

## Scientific choices

- Audit all recorded models first. Failed and malformed rows are never silently
  discarded; ambiguous identities stop execution.
- Every available final/base record has exactly the same three exported metrics
  as r10. Exclude final copies from pooled recycle inference, retaining them as a
  separately reported sensitivity. Metric equality is not a coordinate checksum.
- Use the existing 3 Å convergence subset for Cav1.2/Nav1.5 and the established
  structural/interface/alignment subset for Kv2.1. No clashscore threshold is
  introduced. Raw, selected, final-only and selected-r10 results remain distinct.
- Average recycles within AF model, then AF models within seed. Bootstrap seeds,
  not snapshots. Complete seed-label matches allow nominal paired inference;
  partially overlapping ranges use Welch seed-summary tests with paired-common-
  label sensitivity. Independent-seed sensitivity is reported throughout.
  Numeric labels alone do not prove identical random-number streams.
- BH families include all planned contrasts and three metrics per channel/cohort.
  Association methods and slope-interaction tests have separate explicit families.
  Report effect sizes and CIs alongside p/q values, not biological conclusions
  based on significance alone.
- The S6 projection uses the four existing canonical-chain rotation coordinates
  and their experimental 8SD3→8SDA vector. It is a directional summary, not a
  state assignment. Unsigned F412-position displacement is reported separately.
  Adjusted and marginal associations differ and cannot establish an energetic
  barrier or causation.

## Initial audit and interpretation

The initial download has 19 CSVs / 113,994 successful records: 36,000 Cav1.2,
35,994 Kv2.1 and 42,000 Nav1.5. All rows have finite primary and complementary
metrics, with no duplicated within-ensemble model identifiers. Kv2.1 L403A
vanilla lacks seed 154 / AF model 5 / r6–r10 and final; exactly the same six
snapshots are absent from the pre-existing structural manifest. Its rank field
is absent throughout, consistently with its unranked filenames.

The main scientific signals are modest G406R global differences, stronger
protocol-associated shifts (especially Kv2.1 F412L and Nav1.5 masking), and a
Kv2.1 S6 association that depends on AF model, recycle, and QC selection.
The executed notebooks provide current numerical findings and uncertainty.
Global CSVs cannot assign clashes to R406 or to S6/interface atoms. Atom-level
cluster logs, run settings and coordinate provenance remain the appropriate
next inputs for mechanistic attribution.
