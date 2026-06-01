"""Dynamic PSFGEN TCL generation + VMD execution.

Also exposes :func:`run_autoionize` for the post-psfgen neutralisation step
(replaces N water molecules with N Na+ or Cl- ions via the VMD autoionize
plugin). The pipeline calls it after psfgen so charged systems
(Gel1/2/3MA) become electroneutral before NAMD picks them up.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .segments import Segment

log = logging.getLogger(__name__)

PSFGEN_OUT_PDB = "debug_1.pdb"
PSFGEN_OUT_PSF = "debug_1.psf"
TCL_FILENAME = "run_psfgen.tcl"
IONIZE_TCL_FILENAME = "run_autoionize.tcl"
IONIZE_OUT_PREFIX = "debug_1_ion"


def _segment_block(seg: Segment) -> str:
    pdb = f"{seg.name}.pdb"
    if seg.name.startswith("WT"):
        return f"segment {seg.name} {{\n  auto none\n  pdb {pdb}\n}}"
    return f"segment {seg.name} {{\n  pdb {pdb}\n}}"


def write_tcl(temp_dir: Path, segments: list[Segment], top_file: str) -> Path:
    lines: list[str] = [
        "package require psfgen",
        "resetpsf",
        f"topology {top_file}",
        "",
    ]
    lines.extend(_segment_block(s) for s in segments)
    lines.append("")
    lines.extend(f"coordpdb {s.name}.pdb {s.name}" for s in segments)
    lines.extend([
        "",
        "guesscoord",
        "regenerate angles dihedrals",
        f"writepdb {PSFGEN_OUT_PDB}",
        f"writepsf {PSFGEN_OUT_PSF}",
        "resetpsf",
        "exit",
    ])

    tcl_path = temp_dir / TCL_FILENAME
    tcl_path.write_text("\n".join(lines), encoding="utf-8")
    return tcl_path


def run_vmd(vmd_exe: str, tcl_path: Path) -> None:
    subprocess.run(
        [vmd_exe, "-dispdev", "text", "-e", tcl_path.name],
        cwd=tcl_path.parent,
        check=True,
    )


def run_autoionize(vmd_exe: str, output_dir: Path) -> dict:
    """Neutralise the system at *output_dir* by replacing waters with ions.

    Calls VMD ``autoionize`` on ``debug_1.{psf,pdb}``, asks for net-zero
    charge via Na+ / Cl- (depending on sign), then overwrites the original
    PSF/PDB with the ionised versions and removes intermediates. Returns a
    dict with ``n_sod`` / ``n_cla`` actually placed (parsed from output PDB).

    Idempotent: if no ions are needed (charge ≈ 0), autoionize is still
    invoked but writes 0 ions, and the original files are simply
    overwritten with chemically equivalent content. To skip the call
    entirely use ``ionize=False`` at the pipeline level.
    """
    psf = output_dir / PSFGEN_OUT_PSF
    pdb = output_dir / PSFGEN_OUT_PDB
    if not (psf.exists() and pdb.exists()):
        raise FileNotFoundError(
            f"autoionize needs both {psf.name} and {pdb.name} in {output_dir}"
        )

    tcl = f"""package require autoionize
set rc [catch {{
    autoionize -psf {PSFGEN_OUT_PSF} -pdb {PSFGEN_OUT_PDB} \\
        -neutralize -cation SOD -anion CLA -seg ION \\
        -o {IONIZE_OUT_PREFIX}
}} msg]
if {{[file exists {IONIZE_OUT_PREFIX}.psf] && [file exists {IONIZE_OUT_PREFIX}.pdb]}} {{
    puts "AUTOIONIZE_OK"
}} else {{
    puts "AUTOIONIZE_FAIL: $msg"
    exit 1
}}
exit 0
"""
    tcl_path = output_dir / IONIZE_TCL_FILENAME
    tcl_path.write_text(tcl, encoding="utf-8")

    proc = subprocess.run(
        [vmd_exe, "-dispdev", "text", "-e", tcl_path.name],
        cwd=output_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if "AUTOIONIZE_OK" not in (proc.stdout or ""):
        log.error("autoionize stderr tail:\n%s", "\n".join((proc.stderr or "").splitlines()[-20:]))
        log.error("autoionize stdout tail:\n%s", "\n".join((proc.stdout or "").splitlines()[-20:]))
        raise RuntimeError("VMD autoionize failed; see logs above.")

    ion_psf = output_dir / f"{IONIZE_OUT_PREFIX}.psf"
    ion_pdb = output_dir / f"{IONIZE_OUT_PREFIX}.pdb"

    # Count ions placed (for the pipeline log + config.json record)
    n_sod = n_cla = 0
    with ion_pdb.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name == "SOD":
                n_sod += 1
            elif name == "CLA":
                n_cla += 1

    # Atomically replace originals
    ion_psf.replace(psf)
    ion_pdb.replace(pdb)

    log.info("Autoionize placed Na+=%d, Cl-=%d; debug_1.{psf,pdb} updated.",
             n_sod, n_cla)
    return {"n_sod": n_sod, "n_cla": n_cla}
