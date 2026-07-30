"""Read the validated experimental-distance literals from channel notebooks.

This keeps the combined masked-versus-vanilla notebook synchronized with the
channel-specific experimental analyses without copying hundreds of values.
Only literal assignments are read; notebook code is never executed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from shared.plotting import experimental_reference_style

EXPERIMENTAL_COLORS = (
    "#E57373", "#F48FB1", "#F06292", "#FF8A65", "#FFB74D",
    "#FFCC99", "#FFD89A", "#FFE082", "#FFF176", "#FFF6B3",
)

# The NaV1.5 state notebook retains one historical label using the older
# domain-II residue name. The plotted model alias uses the corrected K1103
# correspondence, so normalize it at the overlay boundary.
NAV15_ALIAS_CORRECTIONS = {"E704-E1135": "E704-K1103"}


def _normalize_nav15_map(distance_map):
    return {NAV15_ALIAS_CORRECTIONS.get(key, key): values for key, values in distance_map.items()}


def _literal_assignments(path: Path, variable: str):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for cell in notebook["cells"]:
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                try:
                    values.append(ast.literal_eval(node.value))
                except (TypeError, ValueError):
                    pass
    return values


def _best_distance_list(path: Path, aliases: set[str]):
    candidates = _literal_assignments(path, "exp_distances_list")
    scored = []
    for candidate in candidates:
        if isinstance(candidate, list) and candidate and all(isinstance(item, dict) for item in candidate):
            keys = set().union(*(item.keys() for item in candidate))
            scored.append((len(keys & aliases), candidate))
    return max(scored, default=(0, []), key=lambda item: item[0])[1]


def _rows(distances, labels):
    rows = []
    for index, (distance_map, label) in enumerate(zip(distances, labels)):
        structure = label.replace("Experimental | ", "")
        style = experimental_reference_style(structure, index)
        for alias, values in distance_map.items():
            for value in values:
                rows.append({
                    "Alias": alias,
                    "Distance": float(value),
                    "Structure": structure,
                    "Color": style["color"],
                })
    return rows


def _cav12(repo_root: Path, aliases: set[str]):
    path = repo_root / "cav12" / "Cav12_distanceDistribution_vsExperimental.ipynb"
    distances = _best_distance_list(path, aliases)
    pdb_ids = ("8HLP", "8HMA", "8HMB", "8WEA", "8WE9", "8WE8", "8WE7", "8WE6", "8FD7", "8EOG")
    return _rows(distances, [f"Experimental | {pdb_id}" for pdb_id in pdb_ids])


def _cav12_all(repo_root: Path, aliases: set[str]):
    path = repo_root / "cav12" / "Cav12_distanceDistribution_vsExperimental.ipynb"
    pdb_ids = ("8HLP", "8HMA", "8HMB", "8WEA", "8WE9", "8WE8", "8WE7", "8WE6", "8FD7", "8EOG")
    combined = []
    for candidate in _literal_assignments(path, "exp_distances_list"):
        if not isinstance(candidate, list) or not candidate or not all(isinstance(item, dict) for item in candidate):
            continue
        filtered = [{key: values for key, values in item.items() if key in aliases} for item in candidate]
        combined.extend(_rows(filtered, [f"Experimental | {pdb_id}" for pdb_id in pdb_ids]))
    return _deduplicate(combined)


KV_ALIAS_MAP = {
    "G377_A-G377B": "G375 A-B", "G377_A-G377C": "G375 A-C",
    "G377_A-G377D": "G375 A-D", "G377_B-G377C": "G375 B-C",
    "G377_B-G377D": "G375 B-D", "G377C-G377D": "G375 C-D",
    "A_A404-B_A404": "A402 A-B", "A_A404-C_A404": "A402 A-C",
    "A_A404-D_A404": "A402 A-D", "B_A404-C_A404": "A402 B-C",
    "B_A404-D_A404": "A402 B-D", "C_A404-D_A404": "A402 C-D",
    "A_F238-A_R291": "F236-R289 A", "B_F238-B_R291": "F236-R289 B",
    "C_F238-C_R291": "F236-R289 C", "D_F238-D_R291": "F236-R289 D",
    "A_F238-A_R310": "F236-R308 A", "B_F238-B_R310": "F236-R308 B",
    "C_F238-C_R310": "F236-R308 C", "D_F238-D_R310": "F236-R308 D",
}


def _kv21(repo_root: Path, region: str, aliases: set[str]):
    path = repo_root / "kv21" / "Kv21_distanceDistribution_vsExperimental.ipynb"
    collections = _literal_assignments(path, "KV21_EXPERIMENTAL_CA")
    if not collections:
        return []
    data = collections[0]
    region_key = "voltage_sensor" if region == "vsds" else region
    distances, labels = [], []
    for pdb_id, entry in data.items():
        mapped = {
            KV_ALIAS_MAP[key]: values for key, values in entry.get(region_key, {}).items()
            if key in KV_ALIAS_MAP and KV_ALIAS_MAP[key] in aliases
        }
        distances.append(mapped)
        labels.append(f"Experimental | {pdb_id}: {entry['state']}")
    return _rows(distances, labels)


def _nav15(repo_root: Path, region: str, aliases: set[str]):
    path = repo_root / "nav15" / "Nav15_distanceDistribution_vsExperimental.ipynb"
    distances = [_normalize_nav15_map(item) for item in _best_distance_list(path, aliases)[:2]]
    labels = ["Experimental | 7FBS: QQQ mutant", "Experimental | 6UZ3: WT"]
    states = _literal_assignments(path, "NAV15_8VY_EXPERIMENTAL")
    if states:
        region_key = "vsd" if region == "vsds" else region
        for pdb_id, entry in states[0].items():
            normalized = _normalize_nav15_map(entry.get(region_key, {}))
            distances.append({key: value for key, value in normalized.items() if key in aliases})
            labels.append(f"Experimental | {pdb_id}: {entry['state']}")
    return _rows(distances, labels)


def _nav15_all(repo_root: Path, aliases: set[str]):
    path = repo_root / "nav15" / "Nav15_distanceDistribution_vsExperimental.ipynb"
    labels = ["Experimental | 7FBS: QQQ mutant", "Experimental | 6UZ3: WT"]
    combined = []
    for candidate in _literal_assignments(path, "exp_distances_list"):
        if isinstance(candidate, list) and len(candidate) >= 2 and all(isinstance(item, dict) for item in candidate):
            filtered = [{key: values for key, values in _normalize_nav15_map(item).items() if key in aliases} for item in candidate[:2]]
            combined.extend(_rows(filtered, labels))
    states = _literal_assignments(path, "NAV15_8VY_EXPERIMENTAL")
    if states:
        distances, state_labels = [], []
        for pdb_id, entry in states[0].items():
            merged = {}
            for region_key in ("selectivity_filter", "intracellular_gate", "vsd"):
                normalized = _normalize_nav15_map(entry.get(region_key, {}))
                merged.update({key: value for key, value in normalized.items() if key in aliases})
            distances.append(merged)
            state_labels.append(f"Experimental | {pdb_id}: {entry['state']}")
        combined.extend(_rows(distances, state_labels))
    return _deduplicate(combined)


def _deduplicate(rows):
    seen, unique = set(), []
    for row in rows:
        key = (row["Alias"], row["Distance"], row["Structure"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def experimental_rows(repo_root: str | Path, channel: str, region: str, aliases):
    """Return tidy experimental markers for one plotted structural region."""
    root, alias_set = Path(repo_root), set(aliases)
    if region == "overall":
        if channel == "Cav1.2":
            return _cav12_all(root, alias_set)
        if channel == "Kv2.1":
            return _deduplicate(sum((_kv21(root, item, alias_set) for item in ("selectivity_filter", "intracellular_gate", "vsds")), []))
        if channel == "Nav1.5":
            return _nav15_all(root, alias_set)
        return []
    if channel == "Cav1.2":
        return _cav12(root, alias_set)
    if channel == "Kv2.1":
        return _kv21(root, region, alias_set)
    if channel == "Nav1.5":
        return _nav15(root, region, alias_set)
    return []


def nav15_state_experimentals(
    region,
    *,
    experimental_states,
    colors,
    pdb_ids=("8VYJ", "8VYK"),
):
    """Return Nav1.5 state-reference maps, labels, and colors for one region."""
    distances = [experimental_states[pdb_id][region] for pdb_id in pdb_ids]
    labels = [
        f"Experimental | {pdb_id}: {experimental_states[pdb_id]['state']}"
        for pdb_id in pdb_ids
    ]
    return distances, labels, list(colors)
