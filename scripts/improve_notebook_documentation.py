#!/usr/bin/env python3
"""Standardize scientific documentation in the channel-facing notebooks."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


INTRODUCTIONS = {
    "kv21/Kv21_F412L_mutationSite_analysis.ipynb": [
        (
            1,
            """## Comparison design

The analysis separates two effects that should not be conflated. WT and F412L
are compared within vanilla and masked protocols to estimate the mutation
effect, while vanilla and masked ensembles are compared within WT and F412L to
estimate the masking effect. The same ranked Cα-distance panel is retained in
all four comparisons, and 8SD3 WT and 8SDA L403A values provide fixed
experimental state references."""
        ),
        (
            3,
            """## S6-neighborhood distributions

Candidate coordinates are ranked by the largest absolute WT-to-F412L median
shift observed in either protocol. Split violins show the complete retained
ensemble rather than only summary statistics. The experimental 8SDA points test
whether F412L samples an L403A-like S6 direction; they are not a
mutation-matched F412L validation structure."""
        ),
    ],
    "nav15/Nav15_QQQ_mutationSite_analysis.ipynb": [
        (
            1,
            """## Comparison design

The six Cα distances describe the four-residue selectivity-filter geometry, not
the IFM/QQQ mutation site itself. Vanilla is compared separately with the
original mask and masked v2, and the two masked protocols are also compared
directly. Experimental points from 7FBS, 6UZ3, 8VYJ, and 8VYK provide pore-state
context without being treated as QQQ-latch measurements."""
        ),
        (
            3,
            """## Selectivity-filter distributions

Each split violin retains the same residue-pair order so that protocol effects
can be compared directly. Differences in these distal pore coordinates indicate
an associated change in filter geometry, but they do not identify the IFM
engagement state; that question is addressed in the dedicated latching
notebook."""
        ),
    ],
    "cav12/Cav12_G402S_mutationSite_analysis.ipynb": [
        (
            1,
            """## Comparison design

WT and G402S are compared separately in vanilla and masked ensembles. Nearby
coordinates are ranked by their WT-to-mutant median shift, while experimental
values from 8WE6, 8HLP, and 8FD7 remain fixed reference points. This separates
the sequence effect from the effect of masking."""
        ),
        (
            3,
            """## Mutation-centered distances and contacts

The first panels summarize Cα geometry around residue 402. The contact panels
then use shortest heavy-atom distances to test whether the introduced serine
forms recurrent local interactions. A distance at or below 4 Å is treated as a
candidate contact; values below 2 Å are considered atomic overlap rather than
evidence of stronger binding."""
        ),
    ],
    "cav12/Cav12_G406R_mutationSite_analysis.ipynb": [
        (
            1,
            """## Comparison design

WT and G406R are compared within vanilla and masked protocols using the same
mutation-centered coordinate set and the same 8WE6, 8HLP, and 8FD7 references.
This design distinguishes mutation-dependent repacking from a general masking
response."""
        ),
        (
            3,
            """## Mutation-centered distances and contacts

Cα panels describe the local S6-interface geometry, whereas shortest
heavy-atom panels test possible R406 partners. Acidic D1528 and D1533 contacts
are chemically plausible. The R406–T1531 distribution is retained as a quality
control because its sub-2 Å population represents steric overlap and must not
be interpreted as a stabilizing interaction."""
        ),
    ],
}


CONCLUSIONS = {}


def standardize_reader_language(nb):
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        cell.source = (
            cell.source
            .replace("# Kv2.1", r"# $\mathrm{K}_{\mathrm{V}}2.1$")
            .replace("# Nav1.5", r"# $\mathrm{Na}_{\mathrm{V}}1.5$")
            .replace("# Cav1.2", r"# $\mathrm{Ca}_{\mathrm{V}}1.2$")
            .replace("remains the audit view", "provides the complete chain-resolved view")
            .replace("## Experimental audit and explicit distances", "## Experimental validation and explicit distances")
            .replace("This audit ranks", "This analysis ranks")
            .replace("kept as an audit output", "retained as a provenance table")
            .replace("as a mask-design audit", "to show sensitivity to mask design")
            .replace("## Near-mutation distance checks", "## Near-mutation distance distributions")
            .replace("experimental double-check", "experimental correspondence validation")
            .replace("## Experimental marker visibility", "## Experimental marker encoding")
            .replace("## Split-violin comparison alternatives", "## Additional split-violin views")
        )


def document_compact_notebook(relative: str):
    path = ROOT / relative
    nb = nbformat.read(path, as_version=4)
    standardize_reader_language(nb)
    existing_headings = {
        line.strip()
        for cell in nb.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.lstrip().startswith("#")
    }
    offset = 0
    for index, text in INTRODUCTIONS.get(relative, []):
        heading = text.strip().splitlines()[0]
        if heading not in existing_headings:
            nb.cells.insert(index + offset, md(text))
            offset += 1
    conclusion = CONCLUSIONS.get(relative)
    if conclusion and conclusion.strip().splitlines()[0] not in existing_headings:
        nb.cells.append(md(conclusion))
    nbformat.write(nb, path)


def clean_all_reader_language():
    for channel in ("kv21", "nav15", "cav12"):
        for path in (ROOT / channel).glob("*.ipynb"):
            nb = nbformat.read(path, as_version=4)
            standardize_reader_language(nb)
            nbformat.write(nb, path)


def main():
    clean_all_reader_language()
    for relative in INTRODUCTIONS:
        document_compact_notebook(relative)


if __name__ == "__main__":
    main()
