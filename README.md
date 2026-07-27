# Voltage-gated ion-channel mutant ensembles

This repository compares AlphaFold2 conformational ensembles of three voltage-gated ion channels:
Cav1.2, Kv2.1, and Nav1.5. The central question is whether sequence masking broadens the structural
ensemble and improves sampling of experimentally observed channel geometries.

The analysis focuses on Cα distance distributions at the selectivity filter, intracellular gate,
voltage sensors, mutation-adjacent regions, and selected coupling motifs. Predicted distributions are
compared with distances calculated from experimental structures whenever an appropriate residue
correspondence is available.

The ensembles analyzed here are generated with
[targetedMasking](https://github.com/adrishg/targetedMasking), a companion workflow that selectively
masks user-defined columns of an AlphaFold2/ColabFold multiple sequence alignment while retaining the
remaining evolutionary context. This repository is the downstream validation layer: it asks whether
the resulting structural variability is localized, reproducible, physically plausible, and compatible
with known channel structures.

## Why targeted MSA masking?

AlphaFold2 has learned broadly transferable rules of protein structure, but large multidomain proteins
still benefit strongly from the residue-coupling information supplied by an MSA. For voltage-gated ion
channels, that information is valuable for maintaining the overall fold, pore architecture, and
relationships between distant domains. The same evolutionary signal may also favor the dominant
sequence-compatible geometry and reduce sampling around mutation sites or regions capable of adopting
more than one functional conformation.

Targeted masking is designed as a compromise between two extremes:

- an intact MSA, which provides strong global structural context but may constrain a selected dynamic
  region toward one dominant geometry;
- broad MSA reduction, which may increase variability but also discard information needed to model a
  large channel accurately.

The targetedMasking workflow keeps the query sequence and the MSA information outside the selected
region intact, while masking chosen alignment positions in homologous sequences. Mutation sites,
voltage-sensor segments, gates, linkers, or other hypothesis-driven regions can therefore receive less
local evolutionary constraint without removing global MSA support from the rest of the protein.

The working hypothesis is that this intervention can expand sampling in the selected region while
preserving the accuracy contributed by the unmasked MSA elsewhere. It does not force a particular
structure or guarantee a new functional state. Every masked ensemble must still be checked for
convergence, structural integrity, reproducibility across seeds, and agreement with experimental
landmarks.

## Connected workflow

```text
sequence or multimer FASTA
          │
          ▼
ColabFold MSA generation
          │
          ├──────────────► vanilla prediction
          │
          ▼
targetedMasking
selected mutation/dynamic-region columns masked
query and remaining MSA context retained
          │
          ▼
targeted-masked prediction
          │
          ▼
convergence and structural QC
          │
          ▼
this repository
distance distributions · experimental overlays · mutation-site analysis
```

Use [adrishg/targetedMasking](https://github.com/adrishg/targetedMasking) to create the vanilla and
targeted-masked prediction ensembles. Use this repository to compare those ensembles and test whether
any additional variability is localized and experimentally meaningful.

## Channels and constructs

| Channel | Constructs | Experimental references | Main question |
|---|---|---|---|
| Cav1.2 | WT, G402S, G406R | 8FD7, 8HLP, 8WE6 | Does masking broaden voltage-sensor or gate geometry near disease-associated mutations? |
| Kv2.1 | WT, L403A, F412L | 8SD3, 8SDA; 9O10–9O13 as state references | Does masking expand the S6, gate, and voltage-sensor ensemble while preserving an assembled pore? |
| Nav1.5 | WT, IFM→QQQ | 6UZ3, 7FBS, 8VYJ, 8VYK | Does masking alter pore geometry or IFM-coupled inactivation contacts? |

## Preliminary conclusions

The current distance analysis indicates that the effect of masking is channel and region dependent.

- **Kv2.1 shows the strongest effect.** Approximately 94–95% of shared distance observables become more
  variable under masking. Broadening is most pronounced in S6, the intracellular gate, and the voltage
  sensors. WT S6 geometry remains close to the WT 8SD3 structure, although L403A does not reproduce the
  median S6 contraction observed in 8SDA.

- **Cav1.2 shows modest, region-selective broadening.** The clearest increases occur in voltage-sensor
  distances and the mutant intracellular gate. Selectivity-filter geometry remains comparatively
  constrained.

- **Nav1.5 does not show global broadening with the current masks.** Most shared distributions remain
  similar or become narrower. The original QQQ mask produces localized intracellular-gate broadening,
  but this response is reduced in masked v2.

These conclusions are preliminary. Broader one-dimensional distance distributions indicate greater
geometric heterogeneity, but they do not by themselves establish additional independent conformational
states. Seed-level RMSD analysis and structural clustering are the next required tests.

Channel-specific interpretations are available here:

- [Cav1.2 preliminary results](cav12/Cav12_preliminary_results.txt)
- [Kv2.1 preliminary results](kv21/Kv21_preliminary_results.txt)
- [Nav1.5 preliminary results](nav15/Nav15_preliminary_results.txt)

## Repository layout

```text
.
├── cav12/
│   ├── Cav12_distanceDistribution_vsExperimental.ipynb
│   ├── Cav12_G402S_mutationSite_analysis.ipynb
│   ├── Cav12_G406R_mutationSite_analysis.ipynb
│   ├── dataDistances/
│   └── experimentals/
├── kv21/
│   ├── Kv21_distanceDistribution_vsExperimental.ipynb
│   ├── Kv21_L403A_mutationSite_analysis.ipynb
│   ├── Kv21_F412L_mutationSite_analysis.ipynb
│   ├── dataDistances/
│   └── experimental/
├── nav15/
│   ├── Nav15_distanceDistribution_vsExperimental.ipynb
│   ├── Nav15_QQQ_mutationSite_analysis.ipynb
│   ├── Nav15_IFM_latching_analysis.ipynb
│   ├── dataDistances/
│   └── experimental/
├── shared/
│   ├── dataset_selection.py
│   ├── experimental_overlays.py
│   ├── mutation_site_analysis.py
│   └── plotting.py
├── filter_all_distance_csvs_from_rmsd.py
└── refilter_distances_at_custom_rmsd.py
```

## Dataset conventions

The notebooks use explicit dataset selectors rather than silently replacing one ensemble with another.

| Selector | Meaning |
|---|---|
| `all` | Complete distance table before convergence filtering |
| `all_ok_3` | Models retained using the 3 Å successive stable-core RMSD criterion |
| `all_ok_3_structural_qc` | Kv2.1 3 Å subset followed by selectivity-filter tetramer integrity filtering |
| `all_ok_3_structural_interface_qc` | Kv2.1 structural-QC subset followed by trajectory-level pore–VSD interface filtering |

The principal Cav1.2 and Nav1.5 notebooks currently use `all_ok_3`. The principal Kv2.1 notebook uses
`all_ok_3_structural_interface_qc` because a low successive RMSD can retain a converged but structurally
disrupted tetramer. These persisted CSVs include both the selectivity-filter check and a trajectory-level
interface check. A trajectory is rejected when any K427/E423/K420-to-N179/V182 Cα distance exceeds 27 Å.
This cutoff is above the largest corresponding 8SD3/8SDA distance (25.51 Å) and also removes one
isolated L403A recycle excursion at 27.35 Å.

CSV filenames are dated and should be treated as versioned inputs. In particular, the current Cav1.2
G406R analysis uses the complete `26-07-25_Cav1.2_g406r_*` tables and their corresponding 3 Å subsets.
The Nav1.5 analyses use the expanded `26-07-27_*_all_all.csv` tables and their cleaned
`26-07-27_*_all_ok_rmsd_3A.csv` subsets. These versions include both sequence-verified IFM-latch
coordinates, F/Q1170–N1343 and F/Q1170–N1449 in construct numbering.

## Residue numbering

Figure labels follow the numbering used by the relevant experimental paper whenever the model construct
uses a different numbering scheme.

For rat Kv2.1, the AlphaFold2 model and CSV columns are offset by two residues relative to 8SD3 and
8SDA. The underlying CSV names remain unchanged for data lookup, but visible aliases use experimental
numbering.

| Model or CSV | Experimental or paper label |
|---|---|
| G377 | G375 |
| F238–R291 | F236–R289 |
| F238–R310 | F236–R308 |
| A404 | A402 |
| L405A | L403A |
| F414L | F412L |

The Kv2.1 S6 profile therefore appears as V398, I401, A402, L403/A403, I405, and V409 even though the
corresponding CSV columns use positions 400, 403, 404, 405, 407, and 411.

## Main analyses

### Predicted versus experimental distances

Each channel-level notebook plots predicted distance distributions with PDB-specific experimental
markers. Marker overlap with the central violin density indicates local geometric compatibility.
A marker outside the distribution identifies a regional mismatch, not necessarily a global structural
failure.

### Mutation-site analyses

Mutation-focused notebooks rank nearby distance changes and compare WT and mutant ensembles separately
for vanilla and masked protocols. This separation helps distinguish mutation-associated changes from
masking-associated changes.

### Kv2.1 S6 profile

The Kv2.1 notebook calculates a chain-order-independent cross-pore proxy at several positions along S6.
For each position and model, it uses the maximum of the six intersubunit Cα ring distances. The same
measurement is calculated directly from experimental coordinates.

### Nav1.5 IFM analysis

The IFM-latching notebook examines the WT IFM motif and the IFM→QQQ mutant using explicitly defined
receptor contacts. These local coordinates are interpreted alongside pore and gate measurements and
are not treated as sufficient evidence for an open or inactivated state on their own.

## Running the notebooks

Start Jupyter from the repository root so imports from `shared/` and relative paths to data and
experimental structures resolve consistently.

```bash
jupyter lab
```

Recommended order:

1. Run the channel-level distance notebook.
2. Confirm the selected dataset printed by the loading cell.
3. Run the corresponding mutation-site notebook.
4. For Nav1.5, run the IFM-latching notebook after the main channel analysis.
5. Review warnings for missing columns or unresolved experimental residues before interpreting plots.

The notebooks require Python with pandas, NumPy, Matplotlib, Seaborn, and Biopython.

## Filtering and reproducibility

`refilter_distances_at_custom_rmsd.py` rebuilds filtered distance tables from stored RMSD manifests
without modifying the original CSVs.

Example:

```bash
python refilter_distances_at_custom_rmsd.py \
  --distances-root cav12/dataDistances \
  --rmsd-root cav12/rmsd_convergence_filtering \
  --output-root rmsd_threshold_sensitivity \
  --threshold 3 \
  --channel Cav12
```

For Kv2.1, structural QC should be retained when making the primary comparison. Convergence alone does
not guarantee that the selectivity-filter ring remains assembled.

## Interpretation limits

- Distance distributions describe local or regional geometry and are invariant to rigid-body alignment.
- Rows from related recycles or models should not automatically be treated as statistically independent.
- Greater variance does not prove the existence of distinct conformational basins.
- Targeted masking reduces selected MSA constraints; it does not define the direction of a structural
  change or prove that a sampled geometry is biologically occupied.
- A useful result should combine local variability with preserved global channel integrity.
- Experimental overlap at one landmark does not establish global structural agreement.
- F412L currently lacks a mutation-matched experimental Kv2.1 structure.
- The human Kv2.1 structures are state references and should not be treated as direct rat mutant matches.
- RMSD clustering, seed-level summaries, and state occupancy estimates remain necessary before making
  strong claims about conformational-state sampling.

## Current next steps

1. Complete seed-level and domain-specific RMSD analyses.
2. Cluster ensembles using independent seed-level representatives.
3. Quantify state occupancy rather than relying only on marginal distance widths.
4. Test whether masking enriches experimentally compatible states.
5. Integrate the RMSD results with the channel-specific preliminary interpretations.
