from pathlib import Path

import pytest

from shared import data_access


def test_channel_paths_map_to_dataset_buckets():
    cases = {
        "kv21/dataDistances/26-02-11_Kv2.1_wt_maskedAF2_distances_all.csv":
            "data/Kv2.1/distances/26-02-11_Kv2.1_wt_maskedAF2_distances_all.csv",
        "cav12/dataRMSD/Cav12_all_models_all_references_RMSD_contacts.csv":
            "data/Cav1.2/rmsd/Cav12_all_models_all_references_RMSD_contacts.csv",
        "nav15/dataRMSF/merged/nav15_aligned_ca_coordinates.npy":
            "data/Nav1.5/rmsf/merged/nav15_aligned_ca_coordinates.npy",
        "kv21/rmsd_threshold_sensitivity/3p5A/WT/masked/x.csv":
            "data/Kv2.1/sensitivity/3p5A/WT/masked/x.csv",
    }
    for repo_path, expected in cases.items():
        assert data_access.hf_path(repo_path) == expected


def test_filenames_are_never_rewritten():
    repo_path = "kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_v5.csv"
    assert Path(data_access.hf_path(repo_path)).name == Path(repo_path).name


def test_unknown_bucket_fails_loudly():
    with pytest.raises(ValueError, match="No dataset bucket"):
        data_access.hf_path("kv21/someNewDirectory/table.csv")


def test_non_channel_paths_keep_their_layout():
    repo_path = "analysis/statistics_revision/tables/summary.csv"
    assert data_access.hf_path(repo_path) == repo_path


def test_small_provenance_files_stay_in_git():
    for repo_path in [
        "kv21/rmsd_convergence_filtering/Kv21_x/all_ok_models.csv",
        "docs/tables/sampling_depth/summary.csv",
        "analysis/statistics_revision/paired_seed_v2/run_summary.json",
        "kv21/experimental/8SD3.pdb",
    ]:
        assert data_access.is_kept_in_repo(repo_path)


def test_large_derived_tables_are_published():
    for repo_path in [
        "kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_v5.csv",
        "cav12/dataRMSF/merged/cav12_aligned_ca_coordinates.npy",
        "nav15/dataDistances/26-07-27_Nav15_wt_vanillaAF2_distances_all.csv",
    ]:
        assert not data_access.is_kept_in_repo(repo_path)


def test_working_tree_copy_wins(tmp_path, monkeypatch):
    target = tmp_path / "kv21/dataDistances/example.csv"
    target.parent.mkdir(parents=True)
    target.write_text("pdb_file,value\na.pdb,1\n")

    monkeypatch.setattr(data_access, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data_access, "EXTRA_ROOTS", [])
    assert data_access.resolve("kv21/dataDistances/example.csv") == target


def test_extra_root_is_searched_before_the_network(tmp_path, monkeypatch):
    target = tmp_path / "cluster/kv21/dataDistances/example.csv"
    target.parent.mkdir(parents=True)
    target.write_text("pdb_file\na.pdb\n")

    monkeypatch.setattr(data_access, "REPO_ROOT", tmp_path / "empty")
    monkeypatch.setattr(data_access, "EXTRA_ROOTS", [tmp_path / "cluster"])
    assert data_access.resolve("kv21/dataDistances/example.csv") == target


def test_lfs_pointer_is_not_mistaken_for_content(tmp_path, monkeypatch):
    target = tmp_path / "kv21/dataDistances/example.csv"
    target.parent.mkdir(parents=True)
    target.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 100\n"
    )

    monkeypatch.setattr(data_access, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data_access, "EXTRA_ROOTS", [])
    monkeypatch.setattr(data_access, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("VGIC_OFFLINE", "1")

    with pytest.raises(data_access.DataUnavailable):
        data_access.resolve("kv21/dataDistances/example.csv")


def test_offline_mode_names_the_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "REPO_ROOT", tmp_path / "empty")
    monkeypatch.setattr(data_access, "EXTRA_ROOTS", [])
    monkeypatch.setattr(data_access, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("VGIC_OFFLINE", "1")

    with pytest.raises(data_access.DataUnavailable, match="missing.csv"):
        data_access.resolve("kv21/dataDistances/missing.csv")
