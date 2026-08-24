# Production A3M archival checklist

The production A3Ms are not present locally. Archive them without renaming or reconstructing them.

For every vanilla and masked ensemble:

- Upload the exact starting A3M, exact masked A3M, query FASTA, masking command/log, and prediction run configuration.
- Preserve original basenames and record channel, sequence background, exact mask ID, creation date, script commit, prediction commit, and storage accession.
- Compute SHA-256 hashes before upload and again after download.
- Confirm the query row is identical between starting and masked files.
- Match homolog rows by stable header and record missing, added, duplicated, or reordered rows.
- Audit X replacements by A3M match-state position. Lowercase insertions must not advance numbering; `-` must remain `-`.
- Record intended and observed replacements by row, target position, and chain for multimers.
- Keep a small GitHub manifest/checksum table; deposit large A3Ms in the archival data repository rather than Git history.

Expected paths/basenames currently referenced by repository configuration include:

- `Kv2.1/wt/masked/kv21_wt_mask_test.repaired.a3m`
- `Kv2.1/l403a/masked/kv21_l403a_masked.a3m`
- `Kv2.1/f412l/masked/kv21_f412l_masked.a3m`
- The exact Nav1.5 and Cav1.2 vanilla/masked paths or globs listed in `scripts/ensemble_rmsf_analysis/config/mask_definitions.yaml`.

Do not mark the A3M provenance audit complete until every production file and hash is available.
