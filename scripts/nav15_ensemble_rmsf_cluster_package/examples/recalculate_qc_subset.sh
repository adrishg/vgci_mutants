#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/config/nav15_hive.yaml}"
OUTPUT_ROOT="/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH:-}"
python "${PACKAGE_ROOT}/calculate_nav15_ensemble_profiles.py" \
  --config "$CONFIG" \
  --merged-dir "${OUTPUT_ROOT}/merged" \
  --references "${OUTPUT_ROOT}/references/nav15_aligned_references.npz" \
  --annotations "${OUTPUT_ROOT}/references/nav15_residue_annotations.csv" \
  --output-dir "${OUTPUT_ROOT}/profiles" \
  --subset-name earliest_converged \
  --subset-manifest /path/to/Nav15_QC_manifest.csv \
  --subset-column earliest_converged_selected \
  --subset-key pdb_file
