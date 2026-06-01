"""Physics calculation for the gelatin / water system."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .pdb_utils import ATOM_RECORD_PREFIXES, element_of

AVOGADRO = 6.02214076e23
MW_WATER_G_MOL = 18.015
ATOMIC_MASS_G_MOL = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
    "P": 30.974,
}


@dataclass(frozen=True)
class SystemParams:
    molecular_weight: float        # g/mol, rounded to 3 decimals
    n_water: int
    box_length_angstrom: int       # even integer


def molecular_weight(pdb_path: Path) -> float:
    total = 0.0
    with pdb_path.open() as f:
        for line in f:
            if line.startswith(ATOM_RECORD_PREFIXES):
                total += ATOMIC_MASS_G_MOL.get(element_of(line), 0.0)
    return total


def compute_system_params(pdb_path: Path, cfg: PipelineConfig) -> SystemParams:
    mw = molecular_weight(pdb_path)
    wt_percent = cfg.concentration

    mass_polymer = (mw * cfg.chain_count) / AVOGADRO
    mass_water = mass_polymer * ((100.0 - wt_percent) / wt_percent)
    n_water = int(round((mass_water * AVOGADRO) / MW_WATER_G_MOL))

    volume_cm3 = (mass_polymer / cfg.polymer_density_g_cm3) + (
        mass_water / cfg.water_density_g_cm3
    )
    length_exact = math.pow(volume_cm3 * 1e24, 1.0 / 3.0)
    length_expanded = length_exact * cfg.packmol_expansion_factor
    length_even = int(math.ceil(length_expanded / 2.0)) * 2

    return SystemParams(round(mw, 3), n_water, length_even)
