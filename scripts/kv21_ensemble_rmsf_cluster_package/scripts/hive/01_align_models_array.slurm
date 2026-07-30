#!/usr/bin/env bash
#SBATCH --job-name=kv21_rmsf_align
#SBATCH --partition=high
#SBATCH --account=genome-center-grp
#SBATCH --array=0-99%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/kv21_rmsf_align_%A_%a.out
#SBATCH --error=logs/kv21_rmsf_align_%A_%a.err

set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/config/kv21_hive.yaml}"
MAMBA_ENV="${MAMBA_ENV:-bioadri}"
TASK_COUNT="${TASK_COUNT:-100}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV}"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH:-}"

OUTPUT_ROOT=$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["project"]["output_root"])
PY
)
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/inspection/kv21_model_manifest.csv}"
REFERENCES="${OUTPUT_ROOT}/references/kv21_aligned_references.npz"
PARTS_DIR="${OUTPUT_ROOT}/parts"
mkdir -p "$PARTS_DIR" "${PACKAGE_ROOT}/logs"

PART_NPZ=$(printf '%s/aligned_part_%04d.npz' "$PARTS_DIR" "$SLURM_ARRAY_TASK_ID")
PART_CSV=$(printf '%s/aligned_part_%04d_metadata.csv' "$PARTS_DIR" "$SLURM_ARRAY_TASK_ID")
if [[ -s "$PART_NPZ" && -s "$PART_CSV" ]]; then
  if python - "$PART_NPZ" "$MANIFEST" "$REFERENCES" "$SLURM_ARRAY_TASK_ID" "$TASK_COUNT" <<'PY'
import hashlib
import sys
import numpy as np

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

part_path, manifest_path, references_path, task_id, task_count = sys.argv[1:]
part = np.load(part_path, allow_pickle=False)
ok = (
    "manifest_sha256" in part.files
    and "references_sha256" in part.files
    and str(part["manifest_sha256"].item()) == sha256(manifest_path)
    and str(part["references_sha256"].item()) == sha256(references_path)
    and int(part["task_id"].item()) == int(task_id)
    and int(part["task_count"].item()) == int(task_count)
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "Shard ${SLURM_ARRAY_TASK_ID} matches the current manifest and references; skipping safely."
    exit 0
  fi
  echo "Shard ${SLURM_ARRAY_TASK_ID} is stale; replacing it."
fi

python "${PACKAGE_ROOT}/align_kv21_models.py" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --references "$REFERENCES" \
  --output-dir "$PARTS_DIR" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --task-count "$TASK_COUNT" \
  --overwrite
