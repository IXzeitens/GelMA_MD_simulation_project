"""Pipeline configuration schema + (de)serialization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class SoftwarePaths:
    packmol: str = "packmol.exe"
    vmd: str = "vmd"
    namd3: str = "namd3"


@dataclass
class PipelineConfig:
    chain_count: int = 3
    concentration: float = 7.4
    polymer_density_g_cm3: float = 1.3
    water_density_g_cm3: float = 0.997
    packmol_expansion_factor: float = 1.05
    software_paths: SoftwarePaths = field(default_factory=SoftwarePaths)

    # Populated by Step 1 (calculation)
    calculated_M: Optional[float] = None
    calculated_n_water: Optional[int] = None
    calculated_box_L_Angstrom: Optional[int] = None
    atoms_per_chain: Optional[int] = None

    # Populated by Step 4 (segment planning)
    segments: dict[str, list[int]] = field(default_factory=dict)

    # Populated by Step 5.5 (autoionize neutralisation)
    calculated_n_sodium: Optional[int] = None
    calculated_n_chloride: Optional[int] = None

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        sp_data = raw.pop("software_paths", None) or {}
        known = {f.name for f in fields(cls) if f.name != "software_paths"}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(software_paths=SoftwarePaths(**sp_data), **filtered)

    def save(self, path: Path) -> None:
        payload = json.dumps(asdict(self), indent=4, ensure_ascii=False) + "\n"
        path.write_text(payload, encoding="utf-8")
