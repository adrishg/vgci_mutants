"""Build the self-contained, executable Kv2.1 sampling-breadth statistics notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "analysis/statistics_revision/Kv21_sampling_breadth_statistics.ipynb"


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    markdown(r"""
# Reproducible Kv2.1 sampling-breadth statistics

This notebook recalculates the uncertainty estimates used for the Kv2.1 sampling analyses. It keeps the statistical unit explicit and writes every result to versioned CSV tables.

Three analyses are kept separate:

1. **Nominal first 100 trajectories (L403A):** seed-level normalized global breadth, with a bootstrap over the 20 independent seeds.
2. **Full-QC S6 breadth (WT, L403A, F412L):** trajectory-block bootstrap, keeping all retained recycles from a `(seed, model)` trajectory together.
3. **Full-QC RMSF (WT, L403A, F412L):** trajectory-block bootstrap from the aligned Cα coordinate arrays.

Important provenance note: the current executable, chain-label-invariant S6 estimator does **not** reproduce the historical manuscript point values 3.50, 2.08, and 1.92. Its results below are therefore explicitly labeled **revised current estimates**, not confidence intervals for those historical values.
"""),
    code(r"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
while not (ROOT / "shared").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Run this notebook from within the vgci_mutants repository")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_kv21_sampling_breadth_uncertainty import (
    ALL_DISTANCE, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, OUT, run
)

pd.set_option("display.max_columns", 50)
results = run()
results
"""),
    markdown(r"""
## 1. Nominal first 100 trajectories: global breadth

The cohort is fixed before QC: the first 20 seeds × 5 model trajectories = 100 nominal trajectories per protocol. Failed trajectories are **not replaced**. For each retained seed, the global breadth is calculated across the complete distance panel; the 20 seed summaries are the independent units.

The reported effect is

\[
G = \frac{\mathrm{median}(B_{\mathrm{masked},s})}
         {\mathrm{median}(B_{\mathrm{vanilla},s})}.
\]

The 95% percentile interval resamples seeds independently within protocol 10,000 times. The Mann–Whitney test and rank-biserial effect size compare the two sets of 20 seed-level breadth values.
"""),
    code(r"""
first100 = pd.read_csv(ALL_DISTANCE / "l403a_first100_seed_level_global_breadth_summary.csv")
first100_boot = pd.read_csv(ALL_DISTANCE / "l403a_first100_seed_level_global_breadth_ratio_bootstrap.csv")
retention = pd.read_csv(ALL_DISTANCE / "l403a_first100_nominal_trajectory_qc_summary.csv")
display(first100)
display(retention)
print(f"Stored bootstrap replicates: {len(first100_boot):,}")
print("Recomputed percentile CI:", first100_boot["masked_over_vanilla_median_ratio"].quantile([.025, .5, .975]).to_dict())
"""),
    markdown(r"""
## 2. Revised current S6 breadth estimator

For each retained structure and each of six model-numbered S6 levels (400, 403, 404, 405, 407, 411), the coordinate is the **maximum of the six inter-subunit Cα ring spans**. Taking a maximum makes this coordinate invariant to arbitrary A/B/C/D chain ordering.

For level `r`, breadth is summarized by

\[
R_r = \frac{SD(D_{r,\mathrm{masked}})}{SD(D_{r,\mathrm{vanilla}})},
\qquad R = \mathrm{median}_{r=1}^{6}(R_r).
\]

`R > 1` means the masked ensemble is broader for the typical S6 level; `R = 1` means equal SD; `R < 1` means vanilla is broader.

The primary bootstrap samples complete `(seed, model)` trajectories with replacement and retains all their QC-passing recycle snapshots as a block. A sensitivity calculation first selects the latest QC representative per trajectory and then resamples those representatives. Both use 10,000 replicates and seed 20260803.
"""),
    code(r"""
s6 = pd.read_csv(OUT / "kv21_s6_masked_vs_vanilla_breadth_bootstrap.csv")
s6_levels = pd.read_csv(OUT / "kv21_s6_breadth_source_audit.csv")
s6_reps = pd.read_csv(OUT / "kv21_s6_breadth_bootstrap_replicates.csv")

display(s6[[
    "sequence_background", "reported_median_SD_ratio",
    "closest_current_reproducible_median_SD_ratio",
    "bootstrap_95CI_low", "bootstrap_95CI_high",
    "fraction_bootstrap_ratio_gt_1", "representative_only_median_SD_ratio",
    "representative_bootstrap_95CI_low", "representative_bootstrap_95CI_high",
    "representative_percent_difference_from_primary",
    "representative_sensitivity_exceeds_10pct",
    "reported_value_reproduced"
]])
display(s6_levels[[
    "sequence_background", "s6_coordinate_alias", "s6_coordinate_name",
    "vanilla_SD_A", "masked_SD_A", "masked_over_vanilla_SD_ratio",
    "representative_only_SD_ratio"
]])
print(f"Stored S6 bootstrap replicates: {len(s6_reps):,}")
"""),
    markdown(r"""
### S6 interpretation guardrail

The percentile intervals quantify uncertainty for the **revised current estimator shown here**. They must not be attached to the historical 3.50/2.08/1.92 values because those point estimates are not reproduced by the current source tables and executable selector. The representative-only result is a sensitivity analysis, not a replacement estimand.
"""),
    markdown(r"""
## 3. RMSF masked-minus-vanilla differences

RMSF is recomputed from aligned Cα coordinate arrays after every bootstrap draw. Complete `(seed, model)` trajectories are sampled as blocks, preserving all retained recycles. The effect is the median residue-wise difference

\[
\Delta = \mathrm{median}(RMSF_{masked} - RMSF_{vanilla})
\]

reported separately for directly masked positions and positions outside the direct mask. Positive values mean greater RMSF under masking.
"""),
    code(r"""
rmsf = pd.read_csv(OUT / "kv21_rmsf_trajectory_block_bootstrap.csv")
rmsf_audit = pd.read_csv(OUT / "kv21_rmsf_bootstrap_source_audit.csv")
display(rmsf)
display(rmsf_audit)
"""),
    markdown("## 4. Reproducibility checks and output inventory"),
    code(r"""
assert BOOTSTRAP_REPLICATES == 10_000
assert BOOTSTRAP_SEED == 20260803
assert len(first100_boot) == 10_000
assert len(s6_reps) == 3 * 2 * 10_000
assert not s6["reported_value_reproduced"].any()
assert (s6["fraction_bootstrap_ratio_gt_1"] == 1).all()
assert (s6["qualitative_conclusion_agrees_ratio_gt_1"]).all()

outputs = sorted(path.relative_to(ROOT).as_posix() for path in OUT.glob("kv21_*"))
print("All checks passed.\n")
print("\n".join(outputs))
"""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
