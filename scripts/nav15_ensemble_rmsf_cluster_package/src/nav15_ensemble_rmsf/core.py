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
        return "".join(residue.one_letter for residue in self.residues)


@dataclass(frozen=True)
class Structure:
    path: str
    chains: Mapping[str, Chain]


@dataclass(frozen=True)
class Fit:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML configuration must be a mapping: {path}")
    return cfg


def one_letter(resname: str) -> str:
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
            atoms = {
                atom.get_name().strip(): np.asarray(atom.get_coord(), dtype=np.float64)
                for atom in residue.get_atoms()
            }
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
    """Read one Nav1.5 model sequence.

    A colon-separated AlphaFold multimer record is accepted only when all
    components are identical; one monomer is returned. Multiple FASTA records
    are not concatenated.
    """
    path = Path(path)
    records = list(SeqIO.parse(str(path), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    raw = str(records[0].seq).replace("-", "").replace(" ", "").upper()
    components = [component for component in raw.split(":") if component]
    if not components:
        raise ValueError(f"Empty FASTA sequence in {path}")
    if len(components) > 1 and len(set(components)) != 1:
        raise ValueError(f"Colon-separated FASTA components differ in {path}")
    return components[0]


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
    return {
        chain_index: fasta_index + 1
        for chain_index, fasta_index in sequence_pairs(chain.sequence, fasta_sequence, cfg)
    }


def chain_to_raw_ca(
    structure: Structure,
    chain_id: str,
    fasta_sequence: str,
    raw_start: int,
    raw_end: int,
    aligner_cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_residues = raw_end - raw_start + 1
    coords = np.full((n_residues, 3), np.nan, dtype=np.float64)
    present = np.zeros(n_residues, dtype=bool)
    identities = np.full(n_residues, "", dtype="U1")
    pdb_numbers = np.full(n_residues, -1, dtype=np.int32)
    report: dict[str, Any] = {"chain_id": chain_id}
    chain = structure.chains.get(chain_id)
    if chain is None:
        report["status"] = "missing_chain"
        return coords, present, identities, pdb_numbers, report

    raw_map = chain_to_raw_index_map(chain, fasta_sequence, aligner_cfg)
    identity, matches, aligned = sequence_identity(chain.sequence, fasta_sequence, aligner_cfg)
    report.update(
        {
            "status": "ok",
            "sequence_identity": identity,
            "sequence_matches": matches,
            "sequence_aligned_positions": aligned,
            "observed_residues": len(chain.residues),
        }
    )
    mapped_ca = 0
    for chain_residue_index, raw_position in raw_map.items():
        if raw_position < raw_start or raw_position > raw_end:
            continue
        residue = chain.residues[chain_residue_index]
        axis_index = raw_position - raw_start
        identities[axis_index] = residue.one_letter
        pdb_numbers[axis_index] = residue.resseq
        ca = residue.atoms.get("CA")
        if ca is not None:
            coords[axis_index] = ca
            present[axis_index] = True
            mapped_ca += 1
    report["mapped_ca"] = mapped_ca
    mapped_positions = np.flatnonzero(present)
    report["mapped_raw_start"] = int(mapped_positions.min() + raw_start) if mapped_positions.size else math.nan
    report["mapped_raw_end"] = int(mapped_positions.max() + raw_start) if mapped_positions.size else math.nan
    return coords, present, identities, pdb_numbers, report


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
    valid = np.isfinite(output).all(axis=-1)
    output[valid] = output[valid] @ fit.rotation + fit.translation
    return output


def finite_rmsd(
    coords: np.ndarray,
    reference: np.ndarray,
    present: np.ndarray,
    reference_present: np.ndarray,
) -> tuple[float, int, float]:
    valid = np.asarray(present, dtype=bool) & np.asarray(reference_present, dtype=bool)
    count = int(valid.sum())
    requested = int(np.asarray(reference_present, dtype=bool).sum())
    coverage = count / requested if requested else math.nan
    if count == 0:
        return math.nan, 0, coverage
    delta = np.asarray(coords)[valid] - np.asarray(reference)[valid]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))), count, coverage


def parse_af2_filename(filename: str) -> dict[str, Any]:
    name = Path(filename).name
    stem = name
    for suffix in (".pdb", ".cif", ".mmcif"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    recycle_label = "final"
    recycle_index: int | None = None
    match = re.search(r"\.r(\d+)$", stem)
    if match:
        recycle_index = int(match.group(1))
        recycle_label = f"r{recycle_index}"
        trajectory_id = stem[: match.start()]
    else:
        trajectory_id = stem
    def integer(pattern: str) -> int | None:
        found = re.search(pattern, stem)
        return int(found.group(1)) if found else None
    relaxation_match = re.search(r"_(unrelaxed|relaxed)_", f"_{stem}_")
    protocol_match = re.search(r"(alphafold2_(?:multimer_v\d+|ptm))", stem)
    return {
        "pdb_file": name,
        "filename_stem": stem,
        "trajectory_id": trajectory_id,
        "recycle_label": recycle_label,
        "recycle_index": recycle_index,
        "is_final_model": recycle_index is None,
        "rank": integer(r"rank_(\d+)"),
        "model_number": integer(r"model_(\d+)"),
        "seed": integer(r"seed_(\d+)"),
        "relaxation": relaxation_match.group(1) if relaxation_match else "",
        "af2_protocol": protocol_match.group(1) if protocol_match else "",
    }


def discover_structures(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    patterns = ("*.pdb", "*.cif", "*.mmcif")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(path.resolve() for path in paths if path.is_file())
