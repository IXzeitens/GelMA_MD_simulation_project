"""Minimal, dependency-free demo: compute the water-box parameters for one
GelMA chain at a target concentration — no Packmol / NAMD / GPU required.

This exercises the pure-Python core of Workflow-1 (Step 1: box sizing) so a
newcomer can verify the install in seconds. It reproduces the deterministic
formula documented in docs/PIPELINE_DESIGN.md and methods.md.

Run:
    python example/demo_calc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package root importable (example/ is a sibling of pipeline/)
_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from pipeline.calculation import compute_system_params, molecular_weight  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402

PDB = Path(__file__).resolve().parent / "Gel3MA.pdb"


def main() -> int:
    if not PDB.exists():
        print(f"[FATAL] example PDB not found: {PDB}")
        return 1

    # 12 chains at 7.4 wt% (matches the production Gel3MA system).
    cfg = PipelineConfig(
        chain_count=12,
        concentration=7.4,
        polymer_density_g_cm3=1.3,
        water_density_g_cm3=0.997,
        packmol_expansion_factor=1.20,
    )

    mw = molecular_weight(PDB)
    params = compute_system_params(PDB, cfg)

    L = params.box_length_angstrom
    L_exact = L / cfg.packmol_expansion_factor

    print("=== GelMA box-sizing demo (Gel3MA, DS=100%) ===")
    print(f"  Single-chain molecular weight : {mw:,.1f} g/mol")
    print(f"  Chains                        : {cfg.chain_count}")
    print(f"  Target concentration          : {cfg.concentration} wt%")
    print(f"  Water molecules needed        : {params.n_water:,}")
    print(f"  Ideal box edge (length_exact) : {L_exact:.1f} A")
    print(f"  Initial Packmol box edge      : {L} A  (x{cfg.packmol_expansion_factor} expansion)")
    print()
    print("  After NPT equilibration the box collapses back to ~length_exact")
    print("  (~84-86 A for this system); see docs/PIPELINE_DESIGN.md.")
    print()
    print("OK — pure-Python core works. External tools (Packmol/VMD/NAMD)")
    print("are only needed for the actual MD stages (see README quick-start).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
