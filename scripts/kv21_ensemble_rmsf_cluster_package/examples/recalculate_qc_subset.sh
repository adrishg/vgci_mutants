#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/config/kv21_hive.yaml}"
OUTPUT_ROOT="/quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results"
QC_MANIFEST="/replace/with/the/correct/Kv2.1_qc_manifest.csv"

export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH:-}"

python "${PACKAGE_ROOT}/calculate_kv21_ensemble_profiles.py" \
  --config "$CONFIG" \
  --merged-dir "${OUTPUT_ROOT}/merged" \
  --references "${OUTPUT_ROOT}/references/kv21_aligned_references.npz" \
  --annotations "${OUTPUT_ROOT}/references/kv21_residue_annotations.csv" \
  --output-dir "${OUTPUT_ROOT}/profiles/earliest_converged" \
  --subset-name earliest_converged \
  --subset-manifest "$QC_MANIFEST" \
  --subset-column earliest_converged_selected \
  --subset-key pdb_file
