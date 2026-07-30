#!/usr/bin/env bash
#SBATCH --job-name=kv21_rmsf_profiles
#SBATCH --partition=high
#SBATCH --account=genome-center-grp
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kv21_rmsf_profiles_%j.out
#SBATCH --error=logs/kv21_rmsf_profiles_%j.err

set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/config/kv21_hive.yaml}"
MAMBA_ENV="${MAMBA_ENV:-bioadri}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV}"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_ROOT=$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["project"]["output_root"])
PY
)

python "${PACKAGE_ROOT}/calculate_kv21_ensemble_profiles.py" \
  --config "$CONFIG" \
  --merged-dir "${OUTPUT_ROOT}/merged" \
  --references "${OUTPUT_ROOT}/references/kv21_aligned_references.npz" \
  --annotations "${OUTPUT_ROOT}/references/kv21_residue_annotations.csv" \
  --output-dir "${OUTPUT_ROOT}/profiles" \
  --subset-name all_models
