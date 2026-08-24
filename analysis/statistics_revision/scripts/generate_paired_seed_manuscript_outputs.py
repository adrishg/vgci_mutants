#!/usr/bin/env python3
"""Create authoritative manuscript numbers, crosswalk, figures, and report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame.estimand_id.eq("primary_joint_nominal_seed")] if "estimand_id" in frame else frame


def fmt(value, digits=1):
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = []

    def add(claim_id, section, channel, contrast, outcome, source, record,
            point="estimate_B_minus_A", low="CI_low_B_minus_A", high="CI_high_B_minus_A",
            scale=1, digits=1, old="", status="verified"):
        estimate = record.get(point, float("nan")) * scale
        lower = record.get(low, float("nan")) * scale
        upper = record.get(high, float("nan")) * scale
        rows.append({
            "claim_id": claim_id, "manuscript_section": section, "channel": channel,
            "contrast": contrast, "outcome": outcome,
            "estimand": record.get("estimand", record.get("analysis_level", "joint nominal-seed contrast")),
            "analysis_unit": "recorded nominal seed label; AF2 model fixed within-seed stratum",
            "trajectory_reduction": "prespecified median or fraction",
            "model_weighting": "equal surviving AF2 models within seed",
            "seed_resampling": "joint nominal-seed bootstrap; actual RNG equality unverified",
            "primary_or_sensitivity": "primary" if status == "verified" else "qualified",
            "source_script": "analysis/statistics_revision/scripts/run_paired_seed_v2.py",
            "source_table": str(source.relative_to(ROOT)),
            "point_estimate_unrounded": estimate, "ci_low_unrounded": lower, "ci_high_unrounded": upper,
            "display_point_estimate": fmt(estimate, digits), "display_ci_low": fmt(lower, digits),
            "display_ci_high": fmt(upper, digits), "rounding_rule": f"{digits} decimal places",
            "status": status, "old_display_text": old,
        })

    l403_path = out / "l403a_focal_paired_seed.csv"
    l403 = primary(l403_path)
    for claim, outcome, scale, old in (
        ("L403_MAX", "maximum_E423_N179_CA_distance_A", 1, ""),
        ("L403_SHIFT", "fraction_any_subunit_at_experiment_anchored_cutoff", 100, "10.040–14.892"),
        ("L403_RMSE_8SDA", "ordered_vector_RMSE_to_8SDA_A", 1, ""),
        ("L403_CLOSER_8SDA", "fraction_closer_to_8SDA_ordered_vector", 100, "5.476%"),
    ):
        add(claim, "Kv2.1 L403A results", "Kv2.1", "L403A masked - vanilla", outcome,
            l403_path, l403[l403.outcome.eq(outcome)].iloc[0], scale=scale, old=old)
    w1_path = out / "l403a_w1_paired_seed.csv"; w1 = pd.read_csv(w1_path).query("analysis_level == 'primary_joint_nominal_seed'").iloc[0]
    add("L403_W1", "Kv2.1 L403A results", "Kv2.1", "L403A masked - vanilla", "W1 of maximum E423-N179 distance", w1_path, w1,
        point="seed_balanced_W1_A", low="seed_balanced_W1_CI_low_A", high="seed_balanced_W1_CI_high_A")
    interaction_path = out / "kv21_interactions_paired_seed.csv"; interaction = pd.read_csv(interaction_path).iloc[0]
    add("L403_INTERACTION", "Kv2.1 L403A results", "Kv2.1", "masking-by-L403A", interaction.outcome,
        interaction_path, interaction, point="masking_by_sequence_interaction", low="interaction_CI_low", high="interaction_CI_high",
        old="interaction contrast was 0.1 Å, and its 95% CI included zero", status="pairing-unverified sensitivity")

    f_path = out / "f412l_contacts_paired_seed.csv"; f = primary(f_path)
    for claim, outcome, old in (
        ("F412_L316_4A", "fraction_L412_L316_A_within_4.0A", "11.564–15.321"),
        ("F412_L403_4A", "fraction_L412_L403_A_within_4.0A", "−14.026 to −9.346"),
        ("F412_L329_CONT", "continuous_L412_L329_A", ""),
    ):
        scale = 100 if "fraction" in outcome else 1
        add(claim, "Kv2.1 F412L results", "Kv2.1", "F412L masked - vanilla", outcome,
            f_path, f[f.outcome.eq(outcome)].iloc[0], scale=scale, old=old)

    nav_path = out / "nav15_focal_paired_seed.csv"; nav = primary(nav_path)
    for claim, contrast, outcome, old in (
        ("NAV_QQQ_WT", "vanilla QQQ - vanilla WT", "motif_receptor_A", "11.9 Å in WT to 29.0 Å in QQQ"),
        ("NAV_WT_STANDARD", "WT nav15_standard - vanilla", "motif_receptor_A", ""),
        ("NAV_WT_V2", "WT nav15_v2 - vanilla", "motif_receptor_A", ""),
        ("NAV_WT_NOIFM", "WT nav15_v2_noIFM - vanilla", "motif_receptor_A", ""),
        ("NAV_QQQ_STANDARD", "QQQ nav15_standard_plus_IFM - vanilla", "motif_receptor_A", ""),
        ("NAV_QQQ_V2_GATE", "QQQ nav15_v2 - vanilla", "gate_span_A", ""),
    ):
        record = nav[(nav.contrast.eq(contrast)) & (nav.outcome.eq(outcome))].iloc[0]
        add(claim, "Nav1.5 results", "Nav1.5", contrast, outcome, nav_path, record, old=old)

    g402_path = out / "g402s_focal_paired_seed.csv"; g402 = primary(g402_path)
    record = g402[(g402.contrast.eq("G402S - WT under vanilla")) & g402.outcome.eq("nearest_CA_distance_A")].iloc[0]
    add("G402_CA", "Cav1.2 results", "Cav1.2", "G402S - WT vanilla", "nearest atom-matched position-402 C-alpha distance", g402_path, record,
        digits=2, old="G402 shortest non-H side-chain distance")
    partner_path = out / "g402s_partner_distribution.csv"; partner = pd.read_csv(partner_path).iloc[0]
    add("G402_TV", "Cav1.2 results", "Cav1.2", "G402S masked - vanilla", "C-alpha nearest-partner total-variation distance", partner_path, partner,
        point="total_variation_distance", low="total_variation_CI_low", high="total_variation_CI_high", digits=4,
        old="I1523 decreased from 60.8% to 47.7%")

    g406_path = out / "g406r_overlap_and_contacts.csv"; g406 = primary(g406_path)
    for claim, outcome, scale, old in (
        ("G406_PASS", "overlap_pass", 100, "−45.533 percentage points"),
        ("G406_ANY", "fraction_nominal_trajectories_with_any_overlap_pass_snapshot", 100, "−43.4 percentage points"),
        ("G406_COND_1528", "conditional_ASP1528", 100, "−17.389 percentage points"),
        ("G406_COND_1533", "conditional_ASP1533", 100, "−7.870 percentage points"),
        ("G406_UNCOND_1528", "unconditional_ASP1528", 100, ""),
        ("G406_UNCOND_1533", "unconditional_ASP1533", 100, ""),
    ):
        add(claim, "Cav1.2 results", "Cav1.2", "G406R masked - vanilla", outcome,
            g406_path, g406[g406.outcome.eq(outcome)].iloc[0], scale=scale, old=old)

    retention_path = out / "paired_retention_contrasts.csv"; retention = pd.read_csv(retention_path)
    for background in ("WT", "L403A", "F412L"):
        record = retention[(retention.channel.eq("kv21")) & retention.sequence_background.eq(background) & retention.stage.eq("analysis_final")].iloc[0]
        add(f"KV_RET_{background}", "QC results", "Kv2.1", f"{background} masked - vanilla", "analysis-final nominal trajectory retention", retention_path, record, scale=100)

    table = pd.DataFrame(rows)
    table.to_csv(out / "manuscript_numbers.csv", index=False)
    crosswalk = table[[
        "claim_id", "manuscript_section", "old_display_text", "display_point_estimate",
        "display_ci_low", "display_ci_high", "source_table", "status",
    ]].copy()
    crosswalk.to_csv(out / "MANUSCRIPT_NUMBER_CROSSWALK.tsv", sep="\t", index=False)

    figure_dir = out / "figures"; figure_dir.mkdir(exist_ok=True)
    occupancy = pd.read_csv(out / "l403a_zero_to_four_subunits.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    x = occupancy.category.to_numpy(); width = .36
    ax.bar(x - width/2, 100*occupancy.probability_A, width, label="Vanilla", color="#4C78A8")
    ax.bar(x + width/2, 100*occupancy.probability_B, width, label="Masked", color="#E45756")
    ax.set(xlabel="Interfaces above experiment-anchored 12.8 Å cutoff", ylabel="Seed-balanced protocol sampling frequency (%)", xticks=x)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "l403a_zero_to_four_subunits.png", dpi=300)
    fig.savefig(figure_dir / "l403a_zero_to_four_subunits.pdf")
    plt.close(fig)

    report = f"""# Paired-seed statistical revision report

The authoritative analysis uses joint resampling of recorded nominal seed labels while retaining each condition's own QC-qualified model strata. Common-seed and common model-seed outputs are separate sensitivities. Numeric seed-label ranges differ across several conditions and actual RNG equality is not independently recorded; paired-label interaction results are therefore qualified rather than treated as confirmed biological interactions.

Headline results include an L403A maximum E423-N179 shift of {table.loc[table.claim_id.eq('L403_MAX'),'display_point_estimate'].iloc[0]} Å and a {table.loc[table.claim_id.eq('L403_SHIFT'),'display_point_estimate'].iloc[0]}-percentage-point increase at the experiment-anchored cutoff. Analysis-final nominal-trajectory retention decreased for all three Kv2.1 backgrounds. F412L remained contact-specific, Nav1.5 masking moved WT motif geometry away from the receptor-supported regime, atom-matched G402S C-alpha nearest-partner categories were nearly invariant, and G406R masking strongly reduced R406-centered overlap-pass yield.

The source tables, full precision, estimands, rounding, and qualified statuses are in `manuscript_numbers.csv`. Missing production FASTA/A3M and seed provenance are listed in `UNRESOLVED_BLOCKERS.md`.
"""
    (out / "ANALYSIS_REVISION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
