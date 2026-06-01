"""Low-level PDB helpers."""
from __future__ import annotations

from pathlib import Path

ATOM_RECORD_PREFIXES = ("ATOM", "HETATM")


def count_atoms(pdb_path: Path) -> int:
    """Return the number of ATOM / HETATM records in a PDB file."""
    if not pdb_path.exists():
        return 0
    with pdb_path.open() as f:
        return sum(1 for line in f if line.startswith(ATOM_RECORD_PREFIXES))


def element_of(line: str) -> str:
    """Extract element symbol from a PDB ATOM/HETATM line."""
    element = line[76:78].strip().upper()
    if element:
        return element
    name = line[12:16].strip()
    return name[0].upper() if name else ""
