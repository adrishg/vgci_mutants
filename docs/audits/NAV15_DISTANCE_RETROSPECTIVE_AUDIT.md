# Nav1.5 retrospective validation: distance-only audit

This audit separates analyses that can be performed from the existing
3 Å convergence-filtered distance CSVs from analyses that require a new pass
over the original AlphaFold PDB coordinates. The 2025 structures are treated
as retrospective experimental references; the predicted structures must not
be relaxed, morphed, or filtered using agreement with these measurements.

## Sequence-number correspondence

The 1,572-residue AlphaFold construct was globally aligned to the 2,016-residue
full-length human Nav1.5 sequence represented by 8VYJ. The alignment reproduces
the independently verified IFM mapping (canonical F1486 = model F1170).

| Feature | Canonical residue | Model residue |
|---|---:|---:|
| Upper gate DI | L409 | L410 |
| Lower gate DI | A413 | A414 |
| Upper gate DII | L935 | L741 |
| Lower gate DII | L938 | L744 |
| Upper gate DIII | I1466 | I1150 |
| Lower gate DIII | I1470 | I1154 |
| F1473 | F1473 | F1157 |
| Q1476 | Q1476 | Q1160 |
| D1484 | D1484 | D1168 |
| IFM | I1485–F1486–M1487 | I1169–F1170–M1171 |
| K1492 | K1492 | K1176 |
| K1504–K1505–P1506 | K1504–K1505–P1506 | K1188–K1189–P1190 |
| R1512 / F1522 | R1512 / F1522 | R1196 / F1206 |
| M1320 / L1327 | M1320 / L1327 | M1004 / L1011 |
| F1648 / M1652 | F1648 / M1652 | F1332 / M1336 |
| N1659 / I1660 | N1659 / I1660 | N1343 / I1344 |
| N1765 | N1765 | N1449 |
| Upper gate DIV | I1768 | I1452 |
| Lower gate DIV | I1771 | I1455 |
| E1773 / E1788 | E1773 / E1788 | E1457 / E1472 |
| E1867 | E1867 | E1551 |

Rat structures 6UZ3 and 8T6L use a +2 offset for the IFM-pocket residue set.
The engineered III–IV linker/QQQ segment is unresolved in 7FBS, so IFM-pocket
distances must not be fabricated for that structure.

## What the current CSVs can answer

| Analysis | Current status |
|---|---|
| Four-domain intracellular-gate size and square/rectangle geometry | Complete in `Nav15_pore_shape_analysis.ipynb` |
| Central IFM/QQQ residue to N1659 and N1765, Cα and shortest distance | Complete in `Nav15_IFM_latching_analysis.ipynb` |
| Whole IFM/QQQ motif contacts to F1473, Q1476, M1320, M1652, N1659 and I1660 | Available in the current CSVs |
| Whole IFM contact to N1765 | Only the central F/Q1170 contact is currently present |
| DIII and DIV upper/lower gate Cα neighborhoods | Present |
| Exact two-tier side-chain gate radii/diameters | New PDB pass required; exact DI/DII residues are not in the CSVs |
| D1484–K1492 salt bridge | New PDB pass required |
| K1504–E1867 and K1505–E1788 salt bridges | New PDB pass required |
| R1512–F1522 cation–π geometry | New PDB pass required |
| Linker–CTD total contacts and interface center-of-mass distance | New PDB pass required |
| VSD Cα landmark distances | Partially available |
| Gating-charge axial positions, S4 translation and tilt | Core alignment and a new PDB pass required |
| HOLE/CHAP pore radius | New PDB pass and external program required |
| CTD/S0IV orientation and all RMSDs | Core alignment required |
| Buried surface areas and solvent exposure | New PDB pass with an SASA calculation required |

## Distance-only observations

The full-length open structures do not show complete IFM release. Their
whole-motif minimum heavy-atom distances remain short for several receptor
residues:

| PDB | F1473 | Q1476 | M1320 | M1652 | N1659 | I1660 | N1765 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6UZ3 | 4.67 | 3.33 | 7.63 | 4.34 | 2.90 | 2.42 | 3.87 |
| 7DTC | 4.15 | 4.42 | 7.17 | 5.90 | 2.80 | 4.86 | 3.00 |
| 8VYJ | 3.44 | 4.24 | 8.56 | 6.03 | 4.07 | 5.29 | 3.05 |
| 8VYK | 3.35 | 5.19 | 8.30 | 5.49 | 3.52 | 4.23 | 3.19 |
| 8T6L | 4.23 | 4.16 | 7.81 | 5.43 | 3.04 | 3.52 | 3.52 |

Values are Å and were recalculated directly from the downloaded PDB files.
8T6L is a toxin-bound comparator, not a native open-state target. 7FBS is
excluded from this table because its engineered QQQ linker is unresolved.

The contrast between 6UZ3/7DTC and 8VYJ/8VYK is therefore a redistribution of
contacts rather than a binary bound/unbound event. In the 2025 open structures,
F1473 and N1765 remain close, whereas contacts around M1652 and I1660 are
somewhat looser. This pocket fingerprint should be compared residue by residue;
a single IFM Cα distance is insufficient.

Among the AlphaFold ensembles, WT vanilla is the only condition whose median
IFM-pocket distances remain relatively receptor-proximal. Its median
whole-motif shortest distances are 4.5 Å to F1473, 6.8 Å to Q1476, 9.7 Å to
M1652, 9.6 Å to N1659, 6.3 Å to I1660, and 13.8 Å for the available central
F1170–N1765 contact. The masked WT and all QQQ conditions have substantially
larger median pocket separations (typically about 13–32 Å for this residue
set). This supports strong protocol-dependent sampling of linker placement,
but it does not yet show that the separated configurations form a native open
state.

The gate and IFM coordinates are only weakly coupled in most ensembles. The
largest current relationship is in QQQ masked: the Spearman correlation
between mean gate diagonal and the central Q1170–N1765 shortest distance is
approximately -0.36. Thus masking can enlarge and square the pore while moving
the central QQQ residue closer to N1765, but the relationship is moderate and
is best interpreted together with the full pocket fingerprint and aligned
structural measurements.

## Additional structural measurements

An additional coordinate pass could complement the current distance analysis
without modifying any structure. The informative measurements are:

1. Exact upper- and lower-gate side-chain distances for all four domains.
2. All three IFM/QQQ residues against the complete receptor list.
3. D1484–K1492, K1504–E1867 and K1505–E1788 shortest N–O distances.
4. R1512 guanidinium-to-F1522 ring-centroid distance and angle.
5. Linker–CTD contact counts using one fixed heavy-atom cutoff.

The natural data representation is one tidy row per original PDB, joined to
the independent convergence manifests. Experimental-distance agreement is not
an appropriate model-quality filter.

## Template provenance

No AlphaFold template-hit files or recorded maximum-template-date setting were
found in the repository or the local analysis folder inspected here. The
absence of 8VYJ/8VYK from the template set is therefore unverified, so the
analysis does not establish blind or post-training validation. That claim
would require the template-hit files or the original AlphaFold run
configuration.
