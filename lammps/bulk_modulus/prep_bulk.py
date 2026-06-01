"""Prepare LAMMPS data + coeffs for all four short-chain systems' bulk modulus runs.

For each system in {Gelatin, Gel1MA, Gel2MA, Gel3MA}:
  1. Read production/<sys>/Output/debug_1.psf + system_npt_part3.restart.coor
     (NAMD binary) via MDAnalysis -> write npt3_final.pdb
  2. Parse system_npt_part3.restart.xsc -> final box (LX, LY, LZ)
  3. Call lammps_workflow/tools/parmed_emit_lammps.py with the PDB + box +
     CHARMM topology/parameters from Gel3MA/script/ (shared FF set)
  4. Output: lammps_workflow/4_bulk/<sys>/{npt3_final.pdb, project.data, project.in.coeffs}

Run from WSL (parmed lives there):
    ~/miniconda3/bin/python /mnt/c/Users/User/Desktop/Work/0510/lammps_workflow/4_bulk/prep_bulk.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Repo root in WSL view
REPO = Path("/mnt/c/Users/User/Desktop/Work/0510")
ARCHIVE = REPO / "production"
TOOLS = REPO / "lammps_workflow" / "tools"
OUT_ROOT = REPO / "lammps_workflow" / "4_bulk"

SYSTEMS = ["Gelatin", "Gel1MA", "Gel2MA", "Gel3MA"]

# Shared CHARMM force field set lives in Gel3MA/script/
FF_DIR = ARCHIVE / "Gel3MA" / "script"
TOP = FF_DIR / "top_all36_prot_HYP_caf_3.txt"
PARS = [
    FF_DIR / "par_all36m_prot_hyp.prm",
    FF_DIR / "par_all36_carb.prm",
    FF_DIR / "par_all36_cgenff.prm",
    FF_DIR / "par_all36_lipid.prm",
    FF_DIR / "par_all36_na.prm",
    FF_DIR / "lma.prm",
    FF_DIR / "par_ions.prm",
]


def parse_xsc(xsc: Path) -> tuple[float, float, float]:
    """NAMD .xsc last data line: step a_x a_y a_z b_x b_y b_z c_x c_y c_z ..."""
    last = None
    for line in xsc.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        last = line
    if last is None:
        raise RuntimeError(f"no data line in {xsc}")
    tok = last.split()
    a_x, b_y, c_z = float(tok[1]), float(tok[5]), float(tok[9])
    return a_x, b_y, c_z


def namdbin_to_pdb(psf: Path, coor: Path, out_pdb: Path) -> int:
    """Use MDAnalysis to convert NAMD binary restart to PDB. Returns atom count."""
    import MDAnalysis as mda
    u = mda.Universe(str(psf), str(coor), format="NAMDBIN", topology_format="PSF")
    u.atoms.write(str(out_pdb))
    return len(u.atoms)


def run_parmed_emit(pdb: Path, psf: Path, box: tuple[float, float, float],
                    out_data: Path, out_coeffs: Path) -> None:
    cmd = [
        sys.executable,
        str(TOOLS / "parmed_emit_lammps.py"),
        "--psf", str(psf),
        "--pdb", str(pdb),
        "--top", str(TOP),
        "--par", *[str(p) for p in PARS],
        "--box", f"{box[0]:.6f}", f"{box[1]:.6f}", f"{box[2]:.6f}",
        "--out-data", str(out_data),
        "--out-coeffs", str(out_coeffs),
    ]
    print(f"  + {' '.join(cmd[:3])} ... --box {box[0]:.2f} {box[1]:.2f} {box[2]:.2f}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"parmed_emit failed for {pdb.parent.name}")
    # Last few lines of parmed_emit stdout are the summary
    for line in r.stdout.splitlines()[-8:]:
        print(f"    {line}")


def main() -> int:
    for sys_name in SYSTEMS:
        print(f"\n=== {sys_name} ===")
        src = ARCHIVE / sys_name / "Output"
        psf = src / "debug_1.psf"
        coor = src / "system_npt_part3.restart.coor"
        xsc = src / "system_npt_part3.restart.xsc"
        for f in (psf, coor, xsc):
            if not f.exists():
                print(f"  [SKIP] missing {f.name}")
                return 1

        out_dir = OUT_ROOT / sys_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb = out_dir / "npt3_final.pdb"
        data = out_dir / "project.data"
        coeffs = out_dir / "project.in.coeffs"

        box = parse_xsc(xsc)
        print(f"  box (Å): {box[0]:.3f} x {box[1]:.3f} x {box[2]:.3f}")

        n_atoms = namdbin_to_pdb(psf, coor, pdb)
        print(f"  PDB written: {pdb.name} ({n_atoms} atoms)")

        run_parmed_emit(pdb, psf, box, data, coeffs)
        print(f"  -> {data.name} ({data.stat().st_size:,} B), {coeffs.name} ({coeffs.stat().st_size:,} B)")

    print("\nALL DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
