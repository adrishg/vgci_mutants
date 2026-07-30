#!/usr/bin/env bash
#SBATCH --job-name=kv21_rmsf_merge
#SBATCH --partition=high
#SBATCH --account=genome-center-grp
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/kv21_rmsf_merge_%j.out
#SBATCH --error=logs/kv21_rmsf_merge_%j.err

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

python "${PACKAGE_ROOT}/merge_kv21_alignment_shards.py" \
  --manifest "${OUTPUT_ROOT}/inspection/kv21_model_manifest.csv" \
  --parts-dir "${OUTPUT_ROOT}/parts" \
  --output-dir "${OUTPUT_ROOT}/merged" \
  --overwrite
