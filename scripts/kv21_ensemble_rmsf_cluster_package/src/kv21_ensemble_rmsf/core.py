from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from Bio import Align, SeqIO
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1_extended


@dataclass(frozen=True)
class Residue:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    one_letter: str
    atoms: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Chain:
    chain_id: str
    residues: tuple[Residue, ...]

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.residues)


@dataclass(frozen=True)
class Structure:
    path: str
    chains: Mapping[str, Chain]


@dataclass(frozen=True)
class Fit:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float


@dataclass(frozen=True)
class MappingFit:
    shift: int
    orientation: str
    moving_ring_order: tuple[str, ...]
    fixed_ring_order: tuple[str, ...]
    mapping: Mapping[str, str]
    fit: Fit | None
    matched_atoms: int
    requested_atoms: int
    coverage: float
    valid: bool
    reason: str


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML configuration must be a mapping: {path}")
    return cfg


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def one_letter(resname: str) -> str:
    """Return a one-letter amino-acid code using uppercase Bio.PDB keys.

    The earlier project implementation title-cased three-letter names before
    dictionary lookup. Bio.PDB uses uppercase keys, so that converted ordinary
    amino acids to X and invalidated sequence verification. This implementation
    intentionally normalizes to uppercase first.
    """
    key = resname.strip().upper()
    aliases = {"MSE": "M", "SEC": "U", "PYL": "O"}
    return protein_letters_3to1_extended.get(key, aliases.get(key, "X"))


def load_structure(path: str | Path) -> Structure:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure(path.stem, str(path))
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"No coordinate model found in {path}") from exc

    chains: dict[str, Chain] = {}
    for chain in model:
        residues: list[Residue] = []
        for residue in chain:
            if not is_aa(residue, standard=False):
                continue
            hetflag, resseq, icode = residue.id
            if str(hetflag).strip() and residue.resname.strip().upper() != "MSE":
                continue
            atoms = {atom.get_name().strip(): np.asarray(atom.get_coord(), dtype=np.float64) for atom in residue.get_atoms()}
            if not atoms:
                continue
            residues.append(
                Residue(
                    chain_id=str(chain.id),
                    resseq=int(resseq),
                    icode=str(icode).strip(),
                    resname=residue.resname.strip().upper(),
                    one_letter=one_letter(residue.resname),
                    atoms=atoms,
                )
            )
        if residues:
            chains[str(chain.id)] = Chain(str(chain.id), tuple(residues))
    if not chains:
        raise ValueError(f"No protein chains found in {path}")
    return Structure(str(path.resolve()), chains)


def read_fasta(path: str | Path) -> str:
    """Read one monomer sequence from a FASTA or AlphaFold-multimer FASTA.

    AlphaFold-multimer inputs may store a homotetramer as one record with
    colon-separated chains:

        chain_A:chain_B:chain_C:chain_D

    Kv2.1 residue mapping is performed against one 600-residue subunit, so
    this function validates that all colon-separated chains are identical
    and returns one monomer sequence.
    """
    path = Path(path)
    records = list(SeqIO.parse(str(path), "fasta"))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    raw_sequence = (
        str(records[0].seq)
        .replace("-", "")
        .replace(" ", "")
        .replace("\n", "")
        .upper()
    )

    if not raw_sequence:
        raise ValueError(f"Empty FASTA sequence in {path}")

    chain_sequences = [
        sequence
        for sequence in raw_sequence.split(":")
        if sequence
    ]

    if not chain_sequences:
        raise ValueError(f"No chain sequences found in {path}")

    if len(chain_sequences) > 1:
        lengths = [len(sequence) for sequence in chain_sequences]

        if len(set(chain_sequences)) != 1:
            raise ValueError(
                f"Expected an identical-chain homomultimer FASTA in {path}, "
                f"but colon-separated chains differ; lengths={lengths}"
            )

    return chain_sequences[0]


def make_aligner(cfg: Mapping[str, Any] | None = None) -> Align.PairwiseAligner:
    cfg = dict(cfg or {})
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = float(cfg.get("match_score", 2.0))
    aligner.mismatch_score = float(cfg.get("mismatch_score", -1.0))
    aligner.open_gap_score = float(cfg.get("open_gap_score", -10.0))
    aligner.extend_gap_score = float(cfg.get("extend_gap_score", -0.5))
    end_gap = float(cfg.get("end_gap_score", 0.0))
    aligner.end_insertion_score = end_gap
    aligner.end_deletion_score = end_gap
    return aligner


@lru_cache(maxsize=4096)
def _sequence_pairs_cached(
    moving: str,
    fixed: str,
    match_score: float,
    mismatch_score: float,
    open_gap_score: float,
    extend_gap_score: float,
    end_gap_score: float,
) -> tuple[tuple[int, int], ...]:
    aligner = make_aligner(
        {
            "match_score": match_score,
            "mismatch_score": mismatch_score,
            "open_gap_score": open_gap_score,
            "extend_gap_score": extend_gap_score,
            "end_gap_score": end_gap_score,
        }
    )
    alignments = aligner.align(moving, fixed)
    if len(alignments) == 0:
        return tuple()
    alignment = alignments[0]
    pairs: list[tuple[int, int]] = []
    moving_blocks, fixed_blocks = alignment.aligned
    for (m0, m1), (f0, f1) in zip(moving_blocks, fixed_blocks):
        block_length = min(int(m1 - m0), int(f1 - f0))
        pairs.extend((int(m0 + i), int(f0 + i)) for i in range(block_length))
    return tuple(pairs)


def sequence_pairs(moving: str, fixed: str, cfg: Mapping[str, Any] | None = None) -> tuple[tuple[int, int], ...]:
    cfg = dict(cfg or {})
    return _sequence_pairs_cached(
        moving,
        fixed,
        float(cfg.get("match_score", 2.0)),
        float(cfg.get("mismatch_score", -1.0)),
        float(cfg.get("open_gap_score", -10.0)),
        float(cfg.get("extend_gap_score", -0.5)),
        float(cfg.get("end_gap_score", 0.0)),
    )


def sequence_identity(moving: str, fixed: str, cfg: Mapping[str, Any] | None = None) -> tuple[float, int, int]:
    pairs = sequence_pairs(moving, fixed, cfg)
    if not pairs:
        return math.nan, 0, 0
    matches = sum(moving[i] == fixed[j] for i, j in pairs)
    return matches / len(pairs), matches, len(pairs)


def chain_to_raw_index_map(chain: Chain, fasta_sequence: str, cfg: Mapping[str, Any] | None = None) -> dict[int, int]:
    """Map chain residue-list indices to 1-based raw FASTA positions.

    Direct residue numbering is used only when it is complete and sequence-
    consistent. Otherwise a global sequence alignment supplies the mapping.
    """
    direct: dict[int, int] = {}
    direct_valid = True
    for index, residue in enumerate(chain.residues):
        raw = residue.resseq
        if raw < 1 or raw > len(fasta_sequence):
            direct_valid = False
            break
        if fasta_sequence[raw - 1] != residue.one_letter and residue.one_letter != "X":
            direct_valid = False
            break
        direct[index] = raw
    if direct_valid and direct:
        return direct
    return {chain_index: fasta_index + 1 for chain_index, fasta_index in sequence_pairs(chain.sequence, fasta_sequence, cfg)}


def structure_to_raw_ca(
    structure: Structure,
    chain_ids: Sequence[str],
    fasta_sequence: str,
    raw_start: int,
    raw_end: int,
    aligner_cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_residues = raw_end - raw_start + 1
    coords = np.full((len(chain_ids), n_residues, 3), np.nan, dtype=np.float64)
    present = np.zeros((len(chain_ids), n_residues), dtype=bool)
    identities = np.full((len(chain_ids), n_residues), "", dtype="U1")
    report: dict[str, Any] = {}

    for chain_index, chain_id in enumerate(chain_ids):
        chain = structure.chains.get(chain_id)
        if chain is None:
            report[f"chain_{chain_id}_status"] = "missing"
            continue
        raw_map = chain_to_raw_index_map(chain, fasta_sequence, aligner_cfg)
        identity, matches, aligned = sequence_identity(chain.sequence, fasta_sequence, aligner_cfg)
        report[f"chain_{chain_id}_sequence_identity"] = identity
        report[f"chain_{chain_id}_sequence_matches"] = matches
        report[f"chain_{chain_id}_sequence_aligned_positions"] = aligned
        report[f"chain_{chain_id}_observed_residues"] = len(chain.residues)
        mapped_ca = 0
        for chain_residue_index, raw_position in raw_map.items():
            if raw_position < raw_start or raw_position > raw_end:
                continue
            residue = chain.residues[chain_residue_index]
            axis_index = raw_position - raw_start
            identities[chain_index, axis_index] = residue.one_letter
            ca = residue.atoms.get("CA")
            if ca is not None:
                coords[chain_index, axis_index] = ca
                present[chain_index, axis_index] = True
                mapped_ca += 1
        report[f"chain_{chain_id}_mapped_ca"] = mapped_ca
        report[f"chain_{chain_id}_status"] = "ok"
    return coords, present, identities, report


def cyclic_mappings(moving_chains: Sequence[str], canonical_chains: Sequence[str]) -> list[dict[str, str]]:
    """Return the four rotations for two already ordered pore-ring chain lists."""
    if len(moving_chains) != 4 or len(canonical_chains) != 4:
        raise ValueError("Kv2.1 cyclic mapping requires exactly four moving and four canonical chains")
    return [
        {moving_chains[i]: canonical_chains[(i + shift) % 4] for i in range(4)}
        for shift in range(4)
    ]


def infer_ring_order(
    coords: np.ndarray,
    present: np.ndarray,
    chain_ids: Sequence[str],
    selection_mask: np.ndarray,
) -> tuple[str, ...]:
    """Infer the undirected four-subunit pore cycle from chain centroids.

    PDB chain labels are not guaranteed to be listed in pore order. For four
    subunits, the physically adjacent cycle has the smallest centroid perimeter;
    diagonal jumps make the alternative cycles longer. The first configured
    chain is fixed as the cycle start to make the result deterministic.
    """
    from itertools import permutations

    if len(chain_ids) != 4:
        raise ValueError("Ring-order inference requires exactly four chains")
    centroids: list[np.ndarray] = []
    for chain_index, chain_id in enumerate(chain_ids):
        valid = selection_mask & present[chain_index]
        if int(valid.sum()) < 3:
            raise ValueError(f"Insufficient coordinates to infer ring order for chain {chain_id}")
        centroids.append(np.mean(coords[chain_index, valid], axis=0))
    centroids_array = np.asarray(centroids, dtype=np.float64)
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for remainder in permutations(range(1, 4)):
        cycle = (0,) + remainder
        perimeter = sum(
            float(np.linalg.norm(centroids_array[cycle[i]] - centroids_array[cycle[(i + 1) % 4]]))
            for i in range(4)
        )
        candidates.append((perimeter, cycle))
    minimum = min(value for value, _ in candidates)
    tolerance = max(1e-8, abs(minimum) * 1e-10)
    equivalent = [cycle for value, cycle in candidates if abs(value - minimum) <= tolerance]
    chosen = min(equivalent, key=lambda cycle: tuple(str(chain_ids[i]) for i in cycle))
    return tuple(str(chain_ids[i]) for i in chosen)


def ring_preserving_mappings(
    moving_ring_order: Sequence[str],
    fixed_ring_order: Sequence[str],
) -> list[tuple[str, int, dict[str, str]]]:
    """Return eight adjacency-preserving mappings: 4 rotations × 2 traversals."""
    moving_forward = tuple(moving_ring_order)
    moving_reverse = (moving_forward[0],) + tuple(reversed(moving_forward[1:]))
    results: list[tuple[str, int, dict[str, str]]] = []
    for orientation, moving_order in (("forward", moving_forward), ("reverse", moving_reverse)):
        for shift, mapping in enumerate(cyclic_mappings(moving_order, fixed_ring_order)):
            results.append((orientation, shift, mapping))
    return results


def ranges_to_mask(raw_start: int, raw_end: int, ranges: Iterable[Sequence[int]]) -> np.ndarray:
    mask = np.zeros(raw_end - raw_start + 1, dtype=bool)
    for start, end in ranges:
        start, end = sorted((int(start), int(end)))
        lo = max(start, raw_start)
        hi = min(end, raw_end)
        if lo <= hi:
            mask[lo - raw_start : hi - raw_start + 1] = True
    return mask


def kabsch_fit(moving: np.ndarray, fixed: np.ndarray) -> Fit:
    moving = np.asarray(moving, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.float64)
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(f"Kabsch arrays must have identical shape (N, 3); got {moving.shape} and {fixed.shape}")
    if moving.shape[0] < 3:
        raise ValueError("At least three coordinate pairs are required")
    moving_centroid = moving.mean(axis=0)
    fixed_centroid = fixed.mean(axis=0)
    moving_centered = moving - moving_centroid
    fixed_centered = fixed - fixed_centroid
    covariance = moving_centered.T @ fixed_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = fixed_centroid - moving_centroid @ rotation
    transformed = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - fixed) ** 2, axis=1))))
    return Fit(rotation=rotation, translation=translation, rmsd=rmsd)


def apply_fit(coords: np.ndarray, fit: Fit) -> np.ndarray:
    output = np.asarray(coords, dtype=np.float64).copy()
    flat = output.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    flat[valid] = flat[valid] @ fit.rotation + fit.translation
    return output


def evaluate_cyclic_raw_fits(
    moving_coords: np.ndarray,
    moving_present: np.ndarray,
    moving_chains: Sequence[str],
    fixed_coords: np.ndarray,
    fixed_present: np.ndarray,
    canonical_chains: Sequence[str],
    core_mask: np.ndarray,
    minimum_matched_ca: int,
    minimum_coverage: float,
) -> list[MappingFit]:
    requested = int(core_mask.sum() * len(canonical_chains))
    moving_index = {chain: i for i, chain in enumerate(moving_chains)}
    fixed_index = {chain: i for i, chain in enumerate(canonical_chains)}
    moving_ring_order = infer_ring_order(moving_coords, moving_present, moving_chains, core_mask)
    fixed_ring_order = infer_ring_order(fixed_coords, fixed_present, canonical_chains, core_mask)
    candidates: list[MappingFit] = []
    for orientation, shift, mapping in ring_preserving_mappings(moving_ring_order, fixed_ring_order):
        moving_pairs: list[np.ndarray] = []
        fixed_pairs: list[np.ndarray] = []
        for moving_chain, canonical_chain in mapping.items():
            mi = moving_index[moving_chain]
            fi = fixed_index[canonical_chain]
            valid = core_mask & moving_present[mi] & fixed_present[fi]
            if valid.any():
                moving_pairs.append(moving_coords[mi, valid])
                fixed_pairs.append(fixed_coords[fi, valid])
        if moving_pairs:
            moving_array = np.concatenate(moving_pairs, axis=0)
            fixed_array = np.concatenate(fixed_pairs, axis=0)
        else:
            moving_array = np.empty((0, 3), dtype=np.float64)
            fixed_array = np.empty((0, 3), dtype=np.float64)
        matched = moving_array.shape[0]
        coverage = matched / requested if requested else math.nan
        valid_candidate = True
        reason = "ok"
        fit: Fit | None = None
        if matched < minimum_matched_ca:
            valid_candidate = False
            reason = f"matched_ca<{minimum_matched_ca}"
        elif not math.isfinite(coverage) or coverage < minimum_coverage:
            valid_candidate = False
            reason = f"coverage<{minimum_coverage:.3f}"
        else:
            try:
                fit = kabsch_fit(moving_array, fixed_array)
            except Exception as exc:
                valid_candidate = False
                reason = f"fit_error:{type(exc).__name__}:{exc}"
        candidates.append(
            MappingFit(
                shift=shift,
                orientation=orientation,
                moving_ring_order=tuple(moving_ring_order),
                fixed_ring_order=tuple(fixed_ring_order),
                mapping=mapping,
                fit=fit,
                matched_atoms=matched,
                requested_atoms=requested,
                coverage=coverage,
                valid=valid_candidate,
                reason=reason,
            )
        )
    return candidates


def choose_best_mapping(candidates: Sequence[MappingFit]) -> tuple[MappingFit, float]:
    valid = [candidate for candidate in candidates if candidate.valid and candidate.fit is not None]
    if not valid:
        details = "; ".join(f"{candidate.orientation}_shift{candidate.shift}:{candidate.reason}" for candidate in candidates)
        raise ValueError(f"No valid cyclic mapping. {details}")
    ordered = sorted(valid, key=lambda candidate: candidate.fit.rmsd if candidate.fit is not None else math.inf)
    gap = math.nan if len(ordered) < 2 else float(ordered[1].fit.rmsd - ordered[0].fit.rmsd)
    return ordered[0], gap


def canonicalize_transformed(
    transformed_moving: np.ndarray,
    moving_present: np.ndarray,
    moving_identities: np.ndarray,
    moving_chains: Sequence[str],
    canonical_chains: Sequence[str],
    mapping: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_residues = transformed_moving.shape[1]
    coords = np.full((len(canonical_chains), n_residues, 3), np.nan, dtype=np.float64)
    present = np.zeros((len(canonical_chains), n_residues), dtype=bool)
    identities = np.full((len(canonical_chains), n_residues), "", dtype="U1")
    moving_index = {chain: i for i, chain in enumerate(moving_chains)}
    canonical_index = {chain: i for i, chain in enumerate(canonical_chains)}
    for moving_chain, canonical_chain in mapping.items():
        mi = moving_index[moving_chain]
        ci = canonical_index[canonical_chain]
        coords[ci] = transformed_moving[mi]
        present[ci] = moving_present[mi]
        identities[ci] = moving_identities[mi]
    return coords, present, identities


def mapping_text(mapping: Mapping[str, str]) -> str:
    return ";".join(f"{moving}->{canonical}" for moving, canonical in mapping.items())


def parse_af2_filename(filename: str) -> dict[str, Any]:
    name = Path(filename).name
    stem = re.sub(r"\.(pdb|cif|mmcif)$", "", name, flags=re.IGNORECASE)
    recycle_match = re.search(r"\.r(?P<recycle>\d+)$", stem)
    recycle_index: int | None = None
    recycle_label = "final"
    base_stem = stem
    if recycle_match:
        recycle_index = int(recycle_match.group("recycle"))
        recycle_label = f"r{recycle_index}"
        base_stem = stem[: recycle_match.start()]
    parsed: dict[str, Any] = {
        "pdb_file": name,
        "filename_stem": stem,
        "trajectory_id": base_stem,
        "recycle_label": recycle_label,
        "recycle_index": recycle_index,
        "is_final_model": recycle_index is None,
        "rank": None,
        "model_number": None,
        "seed": None,
        "relaxation": "unrelaxed" if "_unrelaxed_" in name else ("relaxed" if "_relaxed_" in name else "unknown"),
        "af2_protocol": None,
    }
    patterns = {
        "rank": r"_rank_(\d+)",
        "model_number": r"_model_(\d+)_seed_",
        "seed": r"_seed_(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            parsed[key] = int(match.group(1))
    protocol_match = re.search(r"alphafold2_(.+?)_model_\d+_seed_\d+", name)
    if protocol_match:
        parsed["af2_protocol"] = protocol_match.group(1)
    return parsed


def discover_structures(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files: list[Path] = []
    for pattern in ("*.pdb", "*.cif", "*.mmcif"):
        files.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted({path.resolve() for path in files if path.is_file()})


def finite_rmsd(coords: np.ndarray, reference: np.ndarray, present: np.ndarray, reference_present: np.ndarray) -> tuple[float, int, float]:
    valid = present & reference_present
    count = int(valid.sum())
    requested = int(reference_present.sum())
    coverage = count / requested if requested else math.nan
    if count == 0:
        return math.nan, 0, coverage
    diff = coords[valid] - reference[valid]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), count, coverage
