# Unresolved blockers

## Production FASTA and A3M provenance

The exact production FASTAs and vanilla/masked A3Ms are not available in this checkout. Original metadata points to unavailable `/quobyte/yarovoygrp/...` storage. Consequently, the revision cannot verify query identity, WT/mutant homolog-row matching, intended-versus-observed X replacements, starting/masked A3M hashes, or historical prediction impact of the companion masking defect. Expected paths include the Kv2.1 files and Nav1.5/Cav1.2 globs listed in `scripts/ensemble_rmsf_analysis/config/mask_definitions.yaml`.

## Random-seed provenance

Run configurations establish 100 nominal seed labels and five AF2 parameterizations but do not record the actual RNG integer assigned to each label. Joint resampling therefore uses nominal design labels. Equality of actual RNG values across conditions is unavailable, not asserted.

## Kv2.1 modeled organism

8SD3 and 8SDA are rat KCNB1 structures (P15387), and unavailable FASTA paths report 600-residue modeled sequences. Without the model FASTA, the organism and accession of the modeled construct cannot be resolved from primary sequence evidence.

## F412L mutation-matched interaction

The focal F412L table contains L412 side-chain shortest-heavy-atom distances. WT has a different Phe atom set, so these are not used as atom-matched mutation interactions. A side-chain mutation-specific interaction is not reported.

## Nav1.5 standard WT gate span

The registered `nav15_standard` WT focal distance table contains the two motif–receptor coordinates but not the six gate-span coordinates. The standard-mask motif outcome is estimable; its gate-span contrast is recorded as unavailable rather than substituted from another mask.

## Historical breadth provenance

Some historical manuscript S6 SD ratios lack a complete executable derivation. Revised whole-seed IQR/MAD/W1 summaries are authoritative; historical values are not treated as reproduced.
