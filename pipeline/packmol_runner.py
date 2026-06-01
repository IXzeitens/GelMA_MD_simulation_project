"""Packmol input generation + execution."""
from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

PACKMOL_TOLERANCE_ANGSTROM = 2.5
# At chain_count=12 in a ~90 Å box the default nloop=20 is too few — Packmol
# can leave 5-10 Å constraint violations. Bumping to 200 lets it actually
# converge; for the small (3-chain) box it still finishes in seconds.
PACKMOL_NLOOP = 200


def write_input_file(
    inp_path: Path,
    chain_pdb: Path,
    chain_count: int,
    water_pdb: Path,
    n_water: int,
    output_pdb: Path,
    box_half: int,
) -> None:
    box = f"-{box_half} -{box_half} -{box_half} {box_half} {box_half} {box_half}"
    # writebad yes: emit a PDB even if Packmol doesn't fully converge — the
    # all-together optimisation is allowed to bottom out with small residual
    # overlaps (<~1 Å) since NAMD minimisation will iron them out anyway.
    # Without this flag Packmol can exit empty-handed on tight 12-chain boxes.
    content = dedent(f"""\
        seed -1
        tolerance {PACKMOL_TOLERANCE_ANGSTROM}
        nloop {PACKMOL_NLOOP}
        writebad yes
        filetype pdb
        output {output_pdb.as_posix()}

        structure {chain_pdb.as_posix()}
          number {chain_count}
          inside box {box}
        end structure

        structure {water_pdb.as_posix()}
          number {n_water}
          inside box {box}
        end structure
    """)
    inp_path.write_text(content, encoding="utf-8")


def run_packmol(packmol_exe: Path, inp_path: Path) -> None:
    with inp_path.open() as stdin:
        subprocess.run(
            [str(packmol_exe)],
            stdin=stdin,
            cwd=inp_path.parent,
            check=True,
        )
