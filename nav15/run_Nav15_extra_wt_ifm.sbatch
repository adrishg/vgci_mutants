#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH --account=yarovoygrp
#SBATCH -t 1-48:00:00
#SBATCH --job-name=dist_Nav15_wt_IFM
#SBATCH --mem=25G
#SBATCH --mail-user=ahgonzalez@ucdavis.edu
#SBATCH --mail-type=END

module load gcc/13.2.0
source /share/yarovlab/ahgz/.bashrc

project_path="/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5"

# WT vanilla
python3 getDistances_Nav15_extra_wt_ifm.py \
    --folder-path "$project_path/wt/vanilla/" \
    --output-csv "$project_path/dataDistances/26-07-26_Nav15_wt_vanilla_IFM_extra.csv"

# WT original masked
python3 getDistances_Nav15_extra_wt_ifm.py \
    --folder-path "$project_path/wt/masked/models/" \
    --output-csv "$project_path/dataDistances/26-07-26_Nav15_wt_masked_IFM_extra.csv"

# WT masked v2
python3 getDistances_Nav15_extra_wt_ifm.py \
    --folder-path "$project_path/wt/masked_v2/models/" \
    --output-csv "$project_path/dataDistances/26-07-26_Nav15_wt_maskedv2_IFM_extra.csv"

# WT masked v2 with IFM excluded from the input mask
python3 getDistances_Nav15_extra_wt_ifm.py \
    --folder-path "$project_path/wt/masked_v2_noIFM/models/" \
    --output-csv "$project_path/dataDistances/26-07-26_Nav15_wt_maskedv2_noIFM_IFM_extra.csv"
