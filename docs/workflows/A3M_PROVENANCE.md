# Production A3M provenance status

On 2026-07-29, a recursive case-insensitive search of the repository found no
`.a3m` files. A broader search of the local LabNotebook and Repositories trees
found unrelated A3Ms but none of the named production Kv2.1, Nav1.5, or
CaV1.2 masked inputs.

Copy or link the production A3M tree so that it contains the relative paths in
`mask_definitions.yaml`, then run:

```bash
python -m scripts.ensemble_rmsf_analysis.extract_masks \
  --a3m-root /path/to/production/vgic_mutants
```

The RMSF analysis itself is no longer blocked. Its direct-mask annotations now
use `authoritative_mask_definitions.yaml`, transcribed from the user-supplied
authoritative RMSF mask table in 1-based raw AlphaFold query/model numbering.
These annotations must not be described as programmatically extracted from
A3Ms.

The A3M command still fails loudly when the production files are absent. If
they become available, use it to compare the extracted exact position sets
against the supplied table.
