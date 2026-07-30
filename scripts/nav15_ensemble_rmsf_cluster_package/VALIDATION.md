# Validation Report

Validation used the uploaded files supplied with the request:

```text
nav15_wt_6uz0_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_051.r5.pdb
6UZ3.pdb
7DTC.pdb
7FBS.pdb
8T6L.pdb
8VYJ.pdb
8VYK.pdb
```

## Uploaded AlphaFold model

```text
Protein chains: A
Observed raw residues: 1–1572
Cα atoms: 1572
```

## Experimental mapping to the 1–1572 raw construct

| Reference | Observed Cα | Sequence identity over aligned observed residues | Mapped raw range | Core Cα used | Core RMSD to 6UZ3 |
|---|---:|---:|---:|---:|---:|
| 6UZ3 | 1126 | 1.0000 | 121–1462 | 946 | 0.000 Å |
| 7FBS | 1118 | 1.0000 | 119–1459 | 938 | 1.106 Å |
| 8T6L | 1237 | 1.0000 | 11–1462 | 942 | 1.370 Å |
| 7DTC | 1151 | 0.9783 | 120–1465 | 946 | 1.534 Å |
| 8VYJ | 1395 | 0.9606 | 12–1566 | 946 | 2.079 Å |
| 8VYK | 1395 | 0.9606 | 12–1566 | 946 | 2.073 Å |

The human references are intentionally labeled cross-species references rather than matched WT/QQQ structures.

## End-to-end integration test

A local seven-dataset test was constructed with one uploaded AlphaFold model in each configured dataset. The full workflow completed:

```text
input inspection
reference mapping and alignment
single-task model alignment
shard merge
all-model per-residue profiles
protocol-versus-vanilla comparisons
```

Results:

```text
Manifest rows: 7
Successful alignments: 7
Failed alignments: 0
Coordinate shape: 7 × 1572 × 3
Per-residue profile rows: 11,004
Protocol-versus-vanilla rows: 7,860
```

The uploaded AlphaFold model aligned to the 6UZ3 core with:

```text
Core Cα RMSD: 2.377 Å
Matched core Cα: 946
Core coverage: 1.000
```

Python compilation and shell syntax validation also completed successfully.
