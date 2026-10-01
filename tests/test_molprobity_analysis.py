"""Scientific failure modes: audit exclusions, identity and pseudoreplication."""
import numpy as np
import pandas as pd
import pytest

from shared import molprobity_analysis as mp


def audited_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, 'ROOT', tmp_path)
    out = mp.output_dir('cav12')
    name = 'cav12_wt_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_091.r0.pdb'
    raw = pd.DataFrame([dict(model_name=name, source_path='/cluster/'+name,
        rank=1, af_model=1, seed=91, recycle=0, snapshot='r0', clashscore=30.,
        n_bad_clashes=800, max_overlap=1.6, status='ok', phenix_exit_code=0,
        elapsed_seconds=1., ensemble='WT | vanilla', genotype='WT', protocol='vanilla')])
    folder = tmp_path/'cav12/rmsd_convergence_filtering/Cav12_rmsd_convergence_wt_vanilla'
    folder.mkdir(parents=True)
    pd.DataFrame({'pdb_basename':[name]}).to_csv(folder/'all_models_manifest.csv', index=False)
    return raw, out


def test_clean_audit_has_no_phantom_anomaly_rows(tmp_path, monkeypatch):
    raw,out = audited_fixture(tmp_path, monkeypatch)
    data,audit = mp.audit_data(raw, 'cav12', out)
    assert data.analysis_eligible.all()
    assert pd.read_csv(out/'tables/row_anomalies.csv').empty
    assert audit.ok_fraction.iloc[0] == 1


def test_failures_and_nan_are_preserved_and_explicitly_excluded(tmp_path, monkeypatch):
    raw,out = audited_fixture(tmp_path, monkeypatch)
    raw.loc[0,['status','phenix_exit_code','clashscore']] = ['failed',1,np.nan]
    data,audit = mp.audit_data(raw, 'cav12', out)
    assert len(data) == 1 and not data.analysis_eligible.any()
    assert audit.failed.iloc[0] == 1 and audit.missing_clashscore.iloc[0] == 1
    anomalies = pd.read_csv(out/'tables/row_anomalies.csv')
    assert len(anomalies) == 1 and anomalies.invalid_metric.all()


def test_filename_metadata_mismatch_stops_analysis(tmp_path, monkeypatch):
    raw,out = audited_fixture(tmp_path, monkeypatch)
    raw.loc[0,'seed'] = 92
    with pytest.raises(ValueError, match='Malformed or ambiguous'):
        mp.audit_data(raw, 'cav12', out)
    assert pd.read_csv(out/'tables/row_anomalies.csv').invalid_identity.all()


def test_seed_estimand_is_invariant_to_repeating_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, 'BOOTSTRAPS', 40)
    monkeypatch.setattr(mp, 'ROOT', tmp_path)
    out = mp.output_dir('cav12')
    rows=[]
    for seed in range(10):
        for model in [1,2]:
            for protocol in ['vanilla','masked']:
                value=20+seed+model+(1+seed/20 if protocol=='masked' else 0)
                rows.append(dict(ensemble='WT | '+protocol, seed=seed, af_model=model,
                                 model_number=model,snapshot='r10',clashscore=value,
                                 n_bad_clashes=value*25,max_overlap=value/20))
    frame=pd.DataFrame(rows)
    def run(d):
        cohorts={name:d for name in ['all_recycles','repository_QC','final_only','r10_repository_QC']}
        return mp.contrast_statistics(cohorts,'cav12',out)
    result=run(frame)
    repeated=pd.concat([frame,frame[(frame.seed==0)&(frame.af_model==1)]],ignore_index=True)
    second=run(repeated)
    np.testing.assert_allclose(result.seed_mean_difference,second.seed_mean_difference)
    np.testing.assert_allclose(result.CI_low,second.CI_low)
    assert result[result.metric.eq('clashscore')].seed_mean_difference.iloc[0] == pytest.approx(1.225)
