# Production A3M provenance status

On 2026-07-29, a recursive case-insensitive search of the repository found no
`.a3m` files. A broader search of the local LabNotebook and Repositories trees
found unrelated A3Ms but none of the named production Kv2.1, Nav1.5, or
CaV1.2 masked inputs.

Programmatic comparison with the original alignments requires a production A3M
tree containing the relative paths in `mask_definitions.yaml`. With that tree,
the comparison command is:

```bash
python -m scripts.ensemble_rmsf_analysis.extract_masks \
  --a3m-root /path/to/production/vgic_mutants
```

The RMSF analysis uses direct-mask annotations from
`authoritative_mask_definitions.yaml`, transcribed from the user-supplied
authoritative RMSF mask table in 1-based raw AlphaFold query/model numbering.
Their provenance is transcription from that table rather than programmatic
extraction from A3Ms.

The A3M command exits with an error when the production files are absent. Its
output provides a direct comparison between extracted position sets and the
supplied table when the production tree is present.
