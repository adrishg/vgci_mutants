from analysis.statistics_revision.scripts.run_seed_block_distance_panel import (
    comparison_registry,
)


def test_full_panel_registry_is_unique_and_labels_nonfactorial_masked_contrasts():
    registry = comparison_registry()
    assert len(registry) == 37
    assert registry["comparison_id"].is_unique

    masked_sequence = registry[
        registry["comparison_class"].eq("sequence_effect")
        & registry["protocol_A"].ne("vanilla")
    ]
    nonfactorial = masked_sequence[masked_sequence["channel"].isin(["cav12", "nav15"])]
    assert not nonfactorial.empty
    assert nonfactorial["inferential_role"].eq("protocol_specific_exploration").all()

    kv_factorial = masked_sequence[masked_sequence["channel"].eq("kv21")]
    assert not kv_factorial.empty
    assert kv_factorial["inferential_role"].eq("primary").all()
