# CaV1.2 package validation

Validation used the uploaded files:

```text
cav12_wt_short_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_159.r3.pdb
8FD7.pdb
8HLP.pdb
8WE6.pdb
```

## Uploaded AlphaFold model

```text
chain: A
raw residue range: 1–1685
observed Cα atoms: 1685
WT identities: G402, G406, and G490
```

## Experimental α1 chains

| Reference | α1 chain | Observed Cα | Sequence identity over aligned observed residues | Mapped raw range | Core Cα used | Core RMSD to 8WE6 |
|---|---:|---:|---:|---:|---:|---:|
| 8WE6 | A | 1260 | 1.0000 | 118–1649 | 976 | 0.000 Å |
| 8HLP | A | 1253 | 1.0000 | 114–1655 | 951 | 2.102 Å |
| 8FD7 | K | 1270 | 1.0000 | 112–1658 | 941 | 3.371 Å |

All references exceeded the configured minimum of 850 matched core Cα atoms and 80% core coverage.

## End-to-end integration test

A temporary validation tree included all eight configured datasets, one structure per dataset, four 1685-residue FASTAs, and all three experimental references. The complete workflow succeeded:

```text
input inspection
reference mapping and common-frame alignment
model alignment
shard merge
per-residue profile calculation
protocol-versus-vanilla comparison
experimental-reference metrics
```

Results:

```text
8 configured datasets tested
8 successful alignments
0 failed alignments
3 automated unit tests passed
```

The uploaded model aligned to the 8WE6 core with a Cα RMSD of approximately 3.953 Å over the complete configured core correspondence.

The annotation test confirmed:

```text
raw 402: WT G, G402S S
raw 406: WT G, G406R R
raw 490: WT G, G490R R
```

The validation mutant datasets reused the uploaded WT coordinate model solely to test workflow execution; actual cluster mutant models will supply the corresponding mutant residue identities in the aligned-coordinate metadata.
