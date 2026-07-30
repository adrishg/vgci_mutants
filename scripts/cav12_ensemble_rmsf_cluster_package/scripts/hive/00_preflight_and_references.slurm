#!/usr/bin/env bash
#SBATCH --job-name=cav12_rmsf_preflight
#SBATCH --partition=high
#SBATCH --account=genome-center-grp
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cav12_rmsf_preflight_%j.out
#SBATCH --error=logs/cav12_rmsf_preflight_%j.err

set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/config/cav12_hive.yaml}"
MAMBA_ENV="${MAMBA_ENV:-bioadri}"
mkdir -p "${PACKAGE_ROOT}/logs"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV}"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH:-}"
OUTPUT_ROOT=$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["project"]["output_root"])
PY
)
mkdir -p "${OUTPUT_ROOT}/inspection" "${OUTPUT_ROOT}/references" "${OUTPUT_ROOT}/parts" "${OUTPUT_ROOT}/merged" "${OUTPUT_ROOT}/profiles"
python "${PACKAGE_ROOT}/inspect_cav12_inputs.py" --config "$CONFIG" --output-dir "${OUTPUT_ROOT}/inspection" --max-models-per-dataset 3
python "${PACKAGE_ROOT}/prepare_cav12_references.py" --config "$CONFIG" --output-dir "${OUTPUT_ROOT}/references"
printf '\nPreflight complete. Review:\n'
printf '  %s\n' "${OUTPUT_ROOT}/inspection/cav12_dataset_counts.csv"
printf '  %s\n' "${OUTPUT_ROOT}/inspection/cav12_dataset_path_status.csv"
printf '  %s\n' "${OUTPUT_ROOT}/references/cav12_reference_alignment_report.csv"
