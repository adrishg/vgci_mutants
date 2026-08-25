# Known issues

- The exact production FASTA and A3M files are absent from this repository. Original absolute paths point to unavailable group storage, so query identity, starting/masked A3M hashes, homolog-row correspondence, and historical masking impact cannot be verified.
- Actual AlphaFold random-seed integer values are not recorded in the available run metadata. `seed_###` labels are retained as nominal design keys, but their equality is not described as verified RNG equality.
- The companion `targetedMasking` repository previously applied query raw-string indices to homolog A3M rows. Regression tests demonstrated incorrect behavior when homolog rows contain lowercase insertions. Branch `fix/a3m-match-state-masking` corrects this, but production impact cannot be determined without the missing A3Ms.
- Kv2.1 experimental files 8SD3/8SDA are rat KCNB1 (P15387), while the exact modeled FASTA is absent. The modeled construct organism therefore remains unresolved rather than being inferred from the experimental reference.
- Historical S6 breadth ratios lack complete executable provenance. Revised IQR/MAD and whole-seed intervals are authoritative for the paired-seed revision.
- The registered WT `nav15_standard` focal table lacks the six gate-span columns; its motif-separation contrast is estimable, but the corresponding gate-span contrast is not.
- F412L side-chain shortest-distance outcomes have no atom-matched WT analogue. Mutation-specific interactions are limited to common C-alpha outcomes when available.
- G402 has no non-hydrogen side-chain atom. The prior `shortest_GLY402-*` wording is invalid as a side-chain definition; the paired revision uses position-402 C-alpha-to-C-alpha distances for WT-mutant comparisons.
