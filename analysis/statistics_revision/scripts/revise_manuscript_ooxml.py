#!/usr/bin/env python3
"""Apply targeted paired-seed revisions while preserving the DOCX package."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path, PurePosixPath
import struct
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
XML = "http://www.w3.org/XML/1998/namespace"
for prefix, uri in (("w", W), ("r", R), ("wp", WP), ("a", A), ("pic", PIC)):
    ET.register_namespace(prefix, uri)


def q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def paragraph_text(paragraph) -> str:
    return "".join(node.text or "" for node in paragraph.iter(q(W, "t")))


def replace_paragraph(paragraph, new_text: str) -> str:
    old = paragraph_text(paragraph)
    nodes = list(paragraph.iter(q(W, "t")))
    if not nodes:
        run = ET.SubElement(paragraph, q(W, "r")); nodes = [ET.SubElement(run, q(W, "t"))]
    nodes[0].text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        nodes[0].set(q(XML, "space"), "preserve")
    for node in nodes[1:]:
        node.text = ""
    return old


def table_cell(text: object, *, header: bool = False):
    cell = ET.Element(q(W, "tc"))
    props = ET.SubElement(cell, q(W, "tcPr")); ET.SubElement(props, q(W, "tcW"), {q(W, "w"): "0", q(W, "type"): "auto"})
    paragraph = ET.SubElement(cell, q(W, "p")); run = ET.SubElement(paragraph, q(W, "r"))
    run_props = ET.SubElement(run, q(W, "rPr")); ET.SubElement(run_props, q(W, "sz"), {q(W, "val"): "11"})
    if header: ET.SubElement(run_props, q(W, "b"))
    ET.SubElement(run, q(W, "t")).text = str(text)
    return cell


def cohort_table(flow: pd.DataFrame):
    table = ET.Element(q(W, "tbl"))
    props = ET.SubElement(table, q(W, "tblPr")); ET.SubElement(props, q(W, "tblStyle"), {q(W, "val"): "TableGrid"})
    headers = ["Condition", "Seeds", "Nominal traj.", "Mapping", "Converged", "Final QC", "Interface QC", "Analysis final", "Snapshots", "Main exclusion"]
    row = ET.SubElement(table, q(W, "tr"))
    for header in headers: row.append(table_cell(header, header=True))
    short_reason = {
        "did_not_pass_recycle_convergence": "recycle convergence",
        "did_not_pass_structural_integrity_QC": "structural QC",
        "not_in_analysis_specific_final_distance_cohort": "analysis-specific exclusion",
        "none": "none",
    }
    for record in flow.itertuples(index=False):
        row = ET.SubElement(table, q(W, "tr"))
        values = [
            record.ensemble_id.replace("|", " "), record.nominal_seeds,
            record.nominal_model_seed_trajectories, record.mapping_qc_trajectories,
            record.converged_trajectories, record.final_structural_qc_trajectories,
            record.channel_interface_qc_trajectories, record.analysis_final_trajectories,
            record.retained_snapshots, short_reason.get(record.primary_exclusion_reason, record.primary_exclusion_reason),
        ]
        for value in values: row.append(table_cell(value))
    return table


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n": raise ValueError(f"Not PNG: {path}")
    return struct.unpack(">II", data[16:24])


def drawing_paragraph(paragraph, relationship_id: str, name: str, image_path: Path, doc_pr_id: int) -> None:
    for child in list(paragraph):
        if child.tag != q(W, "pPr"): paragraph.remove(child)
    ppr = paragraph.find(q(W, "pPr"))
    if ppr is None: ppr = ET.SubElement(paragraph, q(W, "pPr"))
    ET.SubElement(ppr, q(W, "jc"), {q(W, "val"): "center"})
    width_px, height_px = png_size(image_path)
    cx = int(6.5 * 914400); cy = int(cx * height_px / width_px)
    if cy > int(8.0 * 914400): cy = int(8.0 * 914400); cx = int(cy * width_px / height_px)
    run = ET.SubElement(paragraph, q(W, "r")); drawing = ET.SubElement(run, q(W, "drawing"))
    inline = ET.SubElement(drawing, q(WP, "inline"), {"distT": "0", "distB": "0", "distL": "0", "distR": "0"})
    ET.SubElement(inline, q(WP, "extent"), {"cx": str(cx), "cy": str(cy)})
    ET.SubElement(inline, q(WP, "docPr"), {"id": str(doc_pr_id), "name": name})
    graphic = ET.SubElement(inline, q(A, "graphic")); graphic_data = ET.SubElement(graphic, q(A, "graphicData"), {"uri": PIC})
    pic = ET.SubElement(graphic_data, q(PIC, "pic")); nv = ET.SubElement(pic, q(PIC, "nvPicPr"))
    ET.SubElement(nv, q(PIC, "cNvPr"), {"id": "0", "name": name}); ET.SubElement(nv, q(PIC, "cNvPicPr"))
    fill = ET.SubElement(pic, q(PIC, "blipFill")); ET.SubElement(fill, q(A, "blip"), {q(R, "embed"): relationship_id})
    stretch = ET.SubElement(fill, q(A, "stretch")); ET.SubElement(stretch, q(A, "fillRect"))
    shape = ET.SubElement(pic, q(PIC, "spPr")); transform = ET.SubElement(shape, q(A, "xfrm"))
    ET.SubElement(transform, q(A, "off"), {"x": "0", "y": "0"}); ET.SubElement(transform, q(A, "ext"), {"cx": str(cx), "cy": str(cy)})
    geometry = ET.SubElement(shape, q(A, "prstGeom"), {"prst": "rect"}); ET.SubElement(geometry, q(A, "avLst"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    args = parser.parse_args()
    base = args.analysis_dir if args.analysis_dir.is_absolute() else ROOT / args.analysis_dir
    original = base / "manuscript/original/2026-08-22_VGCImutantsPaperDraft2_moreStats_corrected_Cav12pair.docx"
    revised = base / "manuscript/2026-08-22_VGCImutantsPaperDraft2_pairedSeedStats_REVISED.docx"
    flow = pd.read_csv(base / "master_cohort_flow_summary.csv")
    reduced = pd.read_csv(base / "reduced_depth_paired_summary.csv").set_index("metric")
    panel_summary = pd.read_json(base / "full_panel/run_summary.json", typ="series")

    with zipfile.ZipFile(original) as archive:
        infos = archive.infolist(); content = {info.filename: archive.read(info.filename) for info in infos}
    document = ET.fromstring(content["word/document.xml"])
    paragraphs = list(document.iter(q(W, "p")))
    originals = {}

    replacements = {
        0: "Targeted MSA Masking Redistributes AlphaFold2 Structural Sampling in Wild-Type and Mutant Voltage-Gated Ion Channels",
        1: "Adriana Hernandez-Gonzalez, Diego Lopez-Mateos, Brandon Harris, Kush Narang, Aleena Siritanapivat, Vladimir Yarov-Yarovoy",
        4: "Disease-associated voltage-gated ion-channel mutations can alter state-dependent local structure while default AlphaFold2 predictions remain close to wild type. We tested targeted multiple-sequence-alignment (MSA) masking in wild-type and mutant Kv2.1, Nav1.5, and Cav1.2 ensembles. Analyses reduced recycles within model-seed trajectories, weighted available AlphaFold2 models equally within recorded seed labels, and separated geometry among quality-qualified outputs from nominal-denominator retention and usable target yield. Masking redistributed rather than universally broadened predicted geometries. In Kv2.1 L403A, masking increased the maximum E423–N179 Cα distance by 1.1 Å and increased sampling beyond a prediction-independent, experiment-anchored cutoff by 12.3 percentage points, but it did not reproduce the coordinated 8SDA rearrangement. A recorded-label overlap interaction sensitivity was positive, but equality of actual random seeds was not independently verified and therefore did not establish a mutation-specific masking response. Kv2.1 masking also reduced analysis-final trajectory retention and redistributed F412L contacts without a mutation-matched structure. In Nav1.5, masking shifted wild-type predictions away from the experimentally supported IFM–receptor arrangement; complete IFM disengagement was not treated as an open-state marker. Atom-matched Cav1.2 G402S Cα analysis supported a small local packing change but not the earlier side-chain nearest-partner claim. G406R masking strongly reduced R406-centered local-overlap pass yield. These prediction frequencies generate structural hypotheses; they do not establish equilibrium occupancy, biological states, or unique mutant structures.",
        30: "We generated AlphaFold2 ensembles for wild-type and mutant channel sequences under vanilla and targeted-MSA protocols. Kv2.1 WT, L403A, and F412L used the shared primary-frozen mask `kv21_common`. Nav1.5 analyses retained exact mask identities (`nav15_standard`, `nav15_standard_plus_IFM`, `nav15_v2`, and the mechanistic follow-up `nav15_v2_noIFM`), and Cav1.2 used condition-specific primary masks. Only the shared Kv2.1 mask supports a directly comparable sequence-by-mask design. Nav1.5 and Cav1.2 masked WT–mutant contrasts remain protocol-specific explorations. Exact status and positions are registered in Supplementary Table S1 and `docs/MASK_REGISTRY.tsv`.",
        36: "For each sequence and masking condition, five AlphaFold2 model parameterizations were evaluated across 100 recorded numeric seed labels, producing 500 nominal model-seed trajectories per condition. The numeric label ranges differed across several conditions, and the available run records did not independently expose the underlying RNG mapping; label overlap is therefore reported separately from confirmed random-seed pairing. Numbered recycle outputs r0–r10 and a final prediction yielded up to 12 saved structures per trajectory. Recycles and model parameterizations sharing one recorded seed label were not treated as independent replicates.",
        56: "For primary two-condition inference, retained recycles were reduced within model-seed trajectory, available AlphaFold2 model parameterizations were weighted equally within each recorded seed label, and recorded labels were resampled jointly across conditions while preserving condition-specific quality-control survivors. Common-contributing-label and common model-seed survivor analyses were reported as sensitivities. Because actual RNG equality was not independently recoverable, these are nominal-label design analyses rather than proof of paired random-number generation. Four-cell Kv2.1 interaction sensitivities used complete overlapping recorded labels and within-label differences.",
        60: "The primary L403A pore–VSD outcome was the continuous maximum E423–N179 Cα distance across the four subunits. The secondary 12.8 Å cutoff was prediction independent and experiment anchored: it was calculated from unrounded 8SD3 and 8SDA coordinates. Threshold summaries retained the absolute percentage-point difference as primary and called the secondary ratio a ratio of seed-balanced protocol sampling fractions; no pseudocount was added, and nonfinite bootstrap replicates were recorded. G406R severe overlap was defined by an R406-centered shortest-heavy distance below 2 Å, with 1.8 and 2.2 Å sensitivities. Sampling frequencies are protocol frequencies, not occupancies.",
        62: f"The all-distance discovery panel reported {int(panel_summary['coordinate_by_comparison_effect_estimates']):,} coordinate-by-comparison effect estimates. Retained recycles were reduced to trajectory medians, available models were weighted equally within recorded label, and labels were weighted equally. Raw W1, signed median change, IQR ratios, common-survivor sensitivities, leave-one-model ranges, and whole-label rank recurrence were retained. Normalized W1 was flagged when the pooled IQR was below 0.05, 0.10, or 0.25 Å. No mass-univariate P or q values were calculated, and high rank recurrence was not treated as independent validation.",
        100: "At the prediction-independent, experiment-anchored 12.8 Å cutoff, the seed-balanced within-trajectory shifted-interface fraction increased from 0.6% in vanilla to 13.0% after masking, an absolute increase of 12.3 percentage points (95% joint nominal-seed interval, 9.8–14.9 points). The secondary ratio of seed-balanced protocol sampling fractions was 21.4 (95% interval, 9.5–145.1); 14 bootstrap replicates were nonfinite because the vanilla denominator was zero, and no pseudocount was used. Earliest- and latest-retained-recycle differences were +19.1 and +9.4 percentage points.",
        101: "The increase was dominated by tetramers with all four interfaces above the cutoff: this category increased from 0.6% to 10.8%, whereas exactly two shifted interfaces increased from 0.0% to 0.8%. Predicted chain identities were not mapped to experimental chain A–D. The masked geometry was therefore usually more symmetric than the two-shifted/two-WT-like arrangement in 8SDA. Ordered-vector RMSE to 8SDA decreased from 3.7 to 3.0 Å (difference −0.7 Å; 95% interval, −0.8 to −0.7 Å), and the fraction closer to 8SDA than 8SD3 increased from 0.5% to 5.5%.",
        102: "Across 1,000 draws of 20 common recorded seed labels, the continuous and shifted-interface effects retained their direction in every draw, with median relative errors of 6.6% and 13.7%. Paired inner-bootstrap subset intervals contained the full-ensemble point estimate in 96.1% and 95.2% of draws, respectively. This is retrospective stability, not frequentist coverage or a prospective stopping rule (Supplementary Table S6).",
        113: "F412L masking produced contact-specific redistribution (Fig. 5B). The seed-balanced L412–L316 within-4-Å frequency increased from 0.5% to 13.9%, a difference of 13.4 percentage points (95% joint nominal-seed interval, 11.6–15.3). L412–L329 remained outside 4 Å, although its continuous distance increased by 0.64 Å (0.51–0.77 Å). L412–L403 within-4-Å proximity decreased from 95.8% to 84.1%, a difference of −11.7 points (−13.8 to −9.7). These are proximity criteria, not bonds. At the L412–L403 measure, the fraction below 2 Å increased to 4.9%, so unrelaxed short-distance models were retained as a severe-overlap flag rather than plausible packing.",
        124: "Using seed-balanced trajectory summaries, vanilla QQQ minus vanilla WT increased mean motif–receptor separation from 18.9 to 28.3 Å, a signed change of 9.4 Å (95% joint nominal-seed interval, 8.4–10.5 Å; W1 9.4 Å, permutation P = 0.0001). The gate-span change was only 0.06 Å (−0.18 to 0.29 Å). Greater motif–receptor separation was not consistently associated with greater intracellular-gate opening across predictions, and these geometries were not interpreted as progression through physical time.",
        126: "Masking also shifted wild-type motif geometry away from the experimentally supported receptor-proximal arrangement. Relative to WT vanilla, motif–receptor separation increased by 9.87 Å with `nav15_standard`, 10.02 Å with `nav15_v2`, and 10.02 Å with the `nav15_v2_noIFM` mechanistic follow-up. Leaving the IFM columns unmasked therefore did not restore receptor-proximal geometry when surrounding constraints were weakened. In QQQ, `nav15_standard_plus_IFM` changed the signed motif separation by only 0.26 Å (95% interval included zero), whereas `nav15_v2` increased it by 0.71 Å. Exact masks were not pooled into a factorial WT–QQQ interpretation.",
        140: "The prior G402/S402 shortest-heavy metric was audited before reanalysis. Glycine has no non-hydrogen side-chain atom, so `shortest_GLY402-*` cannot support the manuscript's former WT side-chain wording. The atomically matched primary comparison instead used position-402 Cα-to-partner-Cα distances. Under vanilla prediction, the nearest distance increased from 5.00 Å in WT to 5.07 Å in G402S, a change of 0.07 Å (95% joint nominal-seed interval, 0.07–0.08 Å). Within G402S, M1524 remained the nearest Cα partner in 100.0% of vanilla and 99.98% of masked outputs; total-variation distance was 0.0002 (0–0.0006). Thus, the earlier side-chain nearest-partner redistribution is not supported by the atom-matched Cα analysis. S402 shortest-heavy proximities remain mutant-specific descriptive geometry and are not called hydrogen bonds.",
        141: "The strongest G406R result was the R406-centered local-overlap pass status. The seed-balanced pass fraction fell from 67.4% to 21.9% after masking, a difference of −45.5 percentage points (95% interval, −47.2 to −43.8). The fraction of nominal model-seed trajectories containing any pass snapshot fell from 95.0% to 51.6% (difference −43.4 points; −46.8 to −39.6). Among pass outputs, R406–D1528 and R406–D1533 within-4-Å proximity decreased by 17.4 points (−19.7 to −14.8) and 7.9 points (−11.9 to −3.8), respectively. Unconditionally, overlap-pass within-4-Å yield decreased from 6.8% to 1.1% for D1528 and from 28.5% to 6.4% for D1533. These unrelaxed proximity criteria do not establish stable salt bridges or equilibrium occupancies.",
        151: "Kv2.1 distinguished increased access to one experimental coordinate from recovery of a broader transition. L403A masking increased maximum E423–N179 separation and threshold-positive interfaces, but shifted tetramers were usually more symmetric than 8SDA and the coordinated PIP/S6, I401, I405, S6-rotation, S4–S5, and F412-centered program was not recovered. A complete recorded-label interaction sensitivity was +0.28 Å (95% interval, +0.14 to +0.41 Å), but actual RNG equality across the four cells was not independently verified; it therefore does not establish a mutation-specific masking response. F412L remained an unresolved, contact-specific redistribution without a mutation-matched structure.",
        157: "Cav1.2 showed local, outcome-dependent changes. Atom-matched G402S Cα distances supported a small WT–mutant packing change, whereas the formerly claimed masking-dependent side-chain nearest-partner redistribution did not survive the valid glycine-aware definition. G406R showed a large masking-dependent reduction in R406-centered local-overlap pass status. Conditional acidic-residue proximities were secondary to this protocol-dependent pass event, and unconditional usable yields also decreased. These unrelaxed geometries do not identify a preferred stable salt bridge.",
        161: "Several limitations affect interpretation. Recycles and subunits are correlated, so analyses reduce within trajectory and resample complete recorded seed labels. However, numeric seed ranges differed across conditions and actual RNG equality was not independently recoverable; recorded-label overlap analyses and four-cell interactions are therefore pairing-unverified sensitivities. Intervals quantify prediction-run variation under fixed inputs, not biological or thermodynamic uncertainty. QC retention is protocol dependent and is reported separately from geometry and QC-adjusted usable yield. Only Kv2.1 used a shared mask. Production FASTAs and A3Ms were unavailable, so query identity, exact historical masking, and the impact of the companion A3M indexing defect remain unresolved. Frequencies depend on cutoffs and are protocol sampling frequencies, not equilibrium occupancy. Targeted masking was not compared with matched random masks, and representative models are illustrations rather than recovered structures.",
        163: "In 1,000 repeated draws of 20 common recorded labels, continuous maximum-distance and shifted-interface effects retained their direction in every draw; median relative errors were 6.6% and 13.7%. Paired inner-bootstrap subset intervals contained the full-ensemble point estimates in 96.1% and 95.2% of draws. These retrospective values are not coverage probabilities or stopping criteria and do not establish that 100 trajectories suffice for other coordinates or rare coordinated alternatives.",
        171: "Analysis code and versioned derived outputs are maintained in the project repository. The paired-seed revision, source hashes, registries, authoritative manuscript numbers, and unresolved blockers are under `analysis/statistics_revision/paired_seed_v2/`. Masking code is maintained separately in `targetedMasking`; its match-state indexing correction is on branch `fix/a3m-match-state-masking`. Exact production FASTAs and A3Ms are not deposited in the current checkout and must be archived with checksums before A3M provenance can be considered complete.",
        579: "Supplementary Table S5 reports the complete cohort flow generated from the authoritative paired-seed registry. Mapping, convergence, structural, interface, and analysis-specific stages are shown separately; values are nominal model-seed trajectories except retained snapshots.",
        584: "Subset interval contained full-ensemble point estimate",
        588: "6.6%", 589: "96.1%", 593: "13.7%", 594: "95.2%",
        601: "",
        602: f"Supplementary Figure S6. Coordinate-by-comparison effect estimates under the paired-seed revision. Heatmaps show raw seed-balanced W1 in ångströms for the largest registered effects. Trajectory medians, equal available-model weights within recorded label, and equal label weights were used. The full source table contains signed median changes, IQR ratios, normalized W1 with pooled-IQR flags at 0.05, 0.10, and 0.25 Å, common-survivor sensitivities, leave-one-model ranges, and whole-label top-rank recurrence across 200 bootstrap replicates. Primary vanilla sequence contrasts, shared-mask Kv2.1 comparisons, and protocol-specific Nav1.5/Cav1.2 masked comparisons remain distinct. No mass-univariate P or q values were calculated.",
        605: "",
        606: "Supplementary Figure S7. Subunit-resolved Kv2.1 L403A experimental-coordinate analysis. Panel E shows the seed-balanced zero-to-four distribution at the prediction-independent, experiment-anchored 12.8 Å E423–N179 cutoff with whole-label intervals. Both protocols contained 100 recorded seed labels; 484 vanilla and 459 masked model-seed trajectories contributed. Predicted chain letters were not mapped directly to experimental chain identities. Masked threshold-positive tetramers were dominated by four elongated interfaces rather than the two-shifted/two-WT-like 8SDA arrangement. Other panels show that PIP/S6, I401, I405, whole-S6 rotation, S4–S5 displacement, and F412-centered changes were not coordinately recovered. The analysis supports increased sampling in one experimental direction, not recovery of a complete state.",
    }

    # Paragraphs with embedded comments are revised by the same text-node-only
    # operation; comment range/reference elements remain untouched.
    replacements[99] = "Using the continuous maximum E423–N179 Cα distance as primary, masking increased the seed-balanced trajectory-median mean from 10.2 to 11.3 Å, a difference of 1.1 Å (95% joint nominal-seed interval, 1.0–1.2 Å). W1 was 1.1 Å (1.0–1.2 Å; paired-label permutation P = 0.0001). The complete recorded-label masking-by-L403A sensitivity was +0.28 Å (+0.14 to +0.41 Å), but actual RNG equality was not independently verified; this result is not sufficient to establish a sequence-specific masking effect."

    for index, new_text in replacements.items():
        originals[index] = replace_paragraph(paragraphs[index], new_text)

    # Replace resolved workflow markers with final figures.
    image_specs = [
        (598, ROOT / "analysis/statistics_revision/seed_block/nav15_regional_rmsd/Figure_S5_Nav15_regional_RMSD_seed_block.png", "Figure_S5_paired_seed.png"),
    ]
    rel_root = ET.fromstring(content["word/_rels/document.xml.rels"])
    existing_ids = [int(rel.get("Id", "rId0").replace("rId", "")) for rel in rel_root if rel.get("Id", "").startswith("rId") and rel.get("Id", "rId0")[3:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    added_media = {}
    for number, (paragraph_index, image_path, media_name) in enumerate(image_specs, start=1):
        if not image_path.exists(): raise FileNotFoundError(image_path)
        relationship_id = f"rId{next_id}"; next_id += 1
        ET.SubElement(rel_root, q(REL, "Relationship"), {
            "Id": relationship_id,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": f"media/{media_name}",
        })
        originals[paragraph_index] = paragraph_text(paragraphs[paragraph_index])
        drawing_paragraph(paragraphs[paragraph_index], relationship_id, media_name, image_path, 1000 + number)
        added_media[f"word/media/{media_name}"] = image_path.read_bytes()

    # Deliberately replace the two resolved draft figures in place, retaining
    # their existing relationship IDs and package part names.
    replacement_figures = [
        (600, "rId20", "word/media/image1.png", base / "full_panel/all_distance_seed_block_top_effects.png", "Figure_S6_paired_seed.png", 1002),
        (604, "rId21", "word/media/image3.png", ROOT / "docs/figures/supplementary_figure_s7/Figure_S7_Kv21_L403A_subunit_resolved_experimental_signatures.png", "Figure_S7_paired_seed.png", 1003),
    ]
    for paragraph_index, relationship_id, media_part, image_path, image_name, doc_pr_id in replacement_figures:
        if not image_path.exists(): raise FileNotFoundError(image_path)
        drawing_paragraph(paragraphs[paragraph_index], relationship_id, image_name, image_path, doc_pr_id)
        content[media_part] = image_path.read_bytes()

    body = document.find(q(W, "body")); placeholder = paragraphs[579]
    insert_at = list(body).index(placeholder) + 1
    body.insert(insert_at, cohort_table(flow))

    content["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    content["word/_rels/document.xml.rels"] = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)
    content.update(added_media)
    revised.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(revised, "w") as archive:
        for info in infos:
            archive.writestr(info, content[info.filename])
        for name, data in added_media.items():
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)

    lines = ["# Manuscript changelog", "", f"Original: `{original.name}`", f"Revised: `{revised.name}`", ""]
    resolved_markers = {1, 579, 598, 601, 605}
    for index in sorted(originals):
        original_display = "[resolved draft marker]" if index in resolved_markers else originals[index]
        revised_text = paragraph_text(paragraphs[index])
        revised_display = revised_text or ("[draft marker removed]" if index in {601, 605} else "[figure inserted]")
        lines.extend([f"## Paragraph P{index:04d}", "", f"Original: {original_display}", "", f"Revised: {revised_display}", ""])
    lines.extend([
        "## Supplementary Table S5", "", "Inserted the 21-row machine-generated cohort-flow table after paragraph P0579.", "",
        "## Supplementary Figures S6 and S7", "", "Replaced the two resolved draft graphics in place with the final full-panel S6 figure and the regenerated S7 figure containing the seed-balanced zero-to-four panel E.", "",
    ])
    (base / "MANUSCRIPT_CHANGELOG.md").write_text("\n".join(lines))
    print(revised)


if __name__ == "__main__":
    main()
