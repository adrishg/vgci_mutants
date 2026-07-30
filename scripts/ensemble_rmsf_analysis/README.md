# Ensemble RMSF analysis

Run from the repository root:

```bash
python -m scripts.ensemble_rmsf_analysis.audit
python -m scripts.ensemble_rmsf_analysis.extract_masks --a3m-root /path/to/production/vgic_mutants
python -m scripts.ensemble_rmsf_analysis.generate_notebooks
```

The second command is mandatory. Long ranges in `config/mask_definitions.yaml`
are validation checkpoints only. Plotting uses the generated A3M residue
tables and has no manual-range fallback.

After the A3Ms validate, execute:

```bash
jupyter nbconvert --to notebook --execute kv21/Kv21_ensemble_RMSF.ipynb --inplace
jupyter nbconvert --to notebook --execute nav15/Nav15_ensemble_RMSF.ipynb --inplace
jupyter nbconvert --to notebook --execute cav12/Cav12_ensemble_RMSF.ipynb --inplace
```
