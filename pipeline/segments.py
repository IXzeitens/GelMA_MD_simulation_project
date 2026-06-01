"""Segment planning and PDB splitting (replaces Packmol_process.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .pdb_utils import ATOM_RECORD_PREFIXES

log = logging.getLogger(__name__)

WATER_MOLECULES_PER_CHUNK = 10_000
ATOMS_PER_WATER_MOLECULE = 3
WATER_CHUNK_ATOMS = WATER_MOLECULES_PER_CHUNK * ATOMS_PER_WATER_MOLECULE
WATER_SEGMENT_START_INDEX = 3  # Naming begins at WT3 per project convention


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    name: str


def _chain_name(index: int) -> str:
    return f"G{chr(ord('A') + index)}" if index < 26 else f"G{index}"


def plan_segments(
    chain_count: int, atoms_per_chain: int, total_atoms: int
) -> list[Segment]:
    """Split atom range into chain segments (GA, GB, …) and water chunks (WT3, …)."""
    segments: list[Segment] = []
    cursor = 1

    for i in range(chain_count):
        end = cursor + atoms_per_chain - 1
        segments.append(Segment(cursor, end, _chain_name(i)))
        cursor = end + 1

    chunk_idx = WATER_SEGMENT_START_INDEX
    while cursor <= total_atoms:
        end = min(cursor + WATER_CHUNK_ATOMS - 1, total_atoms)
        segments.append(Segment(cursor, end, f"WT{chunk_idx}"))
        cursor = end + 1
        chunk_idx += 1

    return segments


def _segment_for(atom_id: int, segments: list[Segment]) -> str:
    for seg in segments:
        if seg.start <= atom_id <= seg.end:
            return seg.name
    return "UNKN"


def split_pdb(pdb_path: Path, segments: list[Segment], output_dir: Path) -> None:
    """Rewrite column-72 segname and split the PDB into per-segment files."""
    buckets: dict[str, list[str]] = {}
    atom_id = 0
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith(ATOM_RECORD_PREFIXES):
                continue
            atom_id += 1
            seg_name = _segment_for(atom_id, segments)
            padded = line.rstrip("\n").ljust(76)
            new_line = padded[:72] + seg_name.ljust(4) + padded[76:] + "\n"
            buckets.setdefault(seg_name, []).append(new_line)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in buckets.items():
        (output_dir / f"{name}.pdb").write_text("".join(lines), encoding="utf-8")
        log.info("Created: %s.pdb", name)
