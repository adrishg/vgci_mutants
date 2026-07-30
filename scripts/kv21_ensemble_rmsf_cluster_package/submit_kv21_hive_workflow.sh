#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PACKAGE_ROOT
cd "$PACKAGE_ROOT"
mkdir -p logs

PREFLIGHT_JOB=$(sbatch --parsable scripts/hive/00_preflight_and_references.slurm)
ALIGN_JOB=$(sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB}" scripts/hive/01_align_models_array.slurm)
MERGE_JOB=$(sbatch --parsable --dependency="afterok:${ALIGN_JOB}" scripts/hive/02_merge_alignment_shards.slurm)
PROFILE_JOB=$(sbatch --parsable --dependency="afterok:${MERGE_JOB}" scripts/hive/03_calculate_all_model_profiles.slurm)

cat <<EOF
Submitted Kv2.1 ensemble-RMSF workflow:
  preflight: ${PREFLIGHT_JOB}
  alignment array: ${ALIGN_JOB}
  merge: ${MERGE_JOB}
  all-model profiles: ${PROFILE_JOB}

Track with:
  squeue -j ${PREFLIGHT_JOB},${ALIGN_JOB},${MERGE_JOB},${PROFILE_JOB}
EOF
