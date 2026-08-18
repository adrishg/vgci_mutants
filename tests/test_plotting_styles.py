from shared.plotting import (
    CHANNEL_ENSEMBLE_PALETTES,
    NAV15_EXPERIMENTAL_STYLES,
    RMSD_REFERENCE_STYLES,
    ensemble_protocol_palette,
    experimental_reference_style,
)


def test_every_ensemble_uses_its_canonical_condition_palette():
    for (channel, condition), expected in CHANNEL_ENSEMBLE_PALETTES.items():
        assert ensemble_protocol_palette(channel, condition) == expected


def test_experimental_styles_are_preserved_in_decorated_labels():
    canonical = {**RMSD_REFERENCE_STYLES, **NAV15_EXPERIMENTAL_STYLES}
    for pdb_id, expected in canonical.items():
        assert experimental_reference_style(pdb_id) == expected
        assert experimental_reference_style(f"Experimental | {pdb_id}: reference") == expected


def test_experimental_structures_have_unique_color_marker_pairs_within_channel():
    groups = (
        tuple(RMSD_REFERENCE_STYLES[pdb_id] for pdb_id in ("8SD3", "8SDA", "9O10", "9O11", "9O12", "9O13")),
        tuple(RMSD_REFERENCE_STYLES[pdb_id] for pdb_id in ("8HLP", "8HMA", "8HMB", "8WEA", "8WE9", "8WE8", "8WE7", "8WE6", "8FD7", "8EOG")),
        tuple(NAV15_EXPERIMENTAL_STYLES.values()),
    )
    for styles in groups:
        encodings = {(style["color"], style["marker"]) for style in styles}
        assert len(encodings) == len(styles)
