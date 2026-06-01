"""End-to-end orchestration of the gelatin MD workflow."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .calculation import SystemParams, compute_system_params
from .config import PipelineConfig
from .namd_runner import NAMD_STAGES, prepare_configs, run_stage
from .packmol_runner import run_packmol, write_input_file
from .paths import ProjectPaths
from .pdb_utils import count_atoms
from .psfgen_runner import (
    PSFGEN_OUT_PDB,
    PSFGEN_OUT_PSF,
    run_autoionize,
    run_vmd,
    write_tcl,
)
from .segments import Segment, plan_segments, split_pdb

log = logging.getLogger(__name__)

TOPOLOGY_FILE = "top_all36_prot_HYP_caf_3.txt"
PACKMOL_INPUT_FILENAME = "packmol_config.inp"


def prompt_user_settings(cfg: PipelineConfig) -> PipelineConfig:
    """Step 0: interactive chain_count / concentration prompt."""
    raw = input(f"[Input] Gel-chain 數量 (default={cfg.chain_count}): ").strip()
    if raw:
        cfg.chain_count = int(raw)
    raw = input(f"[Input] 目標濃度 wt%% (default={cfg.concentration}): ").strip()
    if raw:
        cfg.concentration = float(raw)
    return cfg


def _pick_input_pdb(input_dir: Path) -> Path:
    candidates = sorted(input_dir.glob("*.pdb"))
    if not candidates:
        raise FileNotFoundError(f"No .pdb file found in {input_dir}")
    return candidates[0]


def _record_calculation(cfg: PipelineConfig, params: SystemParams, atoms_per_chain: int) -> None:
    cfg.calculated_M = params.molecular_weight
    cfg.calculated_n_water = params.n_water
    cfg.calculated_box_L_Angstrom = params.box_length_angstrom
    cfg.atoms_per_chain = atoms_per_chain


def _record_segments(cfg: PipelineConfig, segments: list[Segment]) -> None:
    cfg.segments = {s.name: [s.start, s.end] for s in segments}


def _move_psfgen_outputs(temp_dir: Path, output_dir: Path) -> None:
    for name in (PSFGEN_OUT_PDB, PSFGEN_OUT_PSF):
        src = temp_dir / name
        if src.exists():
            shutil.move(str(src), output_dir / name)


def run(paths: ProjectPaths, *, interactive: bool = True) -> None:
    paths.ensure_working_dirs()
    if not paths.config.exists():
        raise FileNotFoundError(f"config.json not found at {paths.config}")

    cfg = PipelineConfig.load(paths.config)

    # Step 0
    if interactive:
        cfg = prompt_user_settings(cfg)
        cfg.save(paths.config)

    target_pdb = _pick_input_pdb(paths.input_dir)
    log.info("Processing model: %s", target_pdb.name)

    # Step 1
    params = compute_system_params(target_pdb, cfg)
    atoms_per_chain = count_atoms(target_pdb)
    log.info(
        "M=%s g/mol | n_water=%s | L=%s Å | atoms/chain=%s",
        params.molecular_weight, params.n_water, params.box_length_angstrom, atoms_per_chain,
    )
    _record_calculation(cfg, params, atoms_per_chain)
    cfg.save(paths.config)

    # Step 2 & 3
    inp_path = paths.packmol_dir / PACKMOL_INPUT_FILENAME
    wb_pdb = paths.temp_dir / f"{target_pdb.stem}_wb.pdb"
    write_input_file(
        inp_path=inp_path,
        chain_pdb=target_pdb,
        chain_count=cfg.chain_count,
        water_pdb=paths.packmol_dir / "water.pdb",
        n_water=params.n_water,
        output_pdb=wb_pdb,
        box_half=params.box_length_angstrom // 2,
    )
    log.info("Executing Packmol...")
    run_packmol(paths.packmol_dir / cfg.software_paths.packmol, inp_path)

    # Step 4
    total_atoms = count_atoms(wb_pdb)
    segments = plan_segments(cfg.chain_count, atoms_per_chain, total_atoms)
    _record_segments(cfg, segments)
    cfg.save(paths.config)
    log.info("Splitting PDB into %d segments...", len(segments))
    split_pdb(wb_pdb, segments, paths.temp_dir)

    # Step 5
    shutil.copy(paths.script_dir / TOPOLOGY_FILE, paths.temp_dir / TOPOLOGY_FILE)
    tcl_path = write_tcl(paths.temp_dir, segments, TOPOLOGY_FILE)
    log.info("Running VMD PSFGEN...")
    run_vmd(cfg.software_paths.vmd, tcl_path)
    _move_psfgen_outputs(paths.temp_dir, paths.output_dir)

    # Step 5.5: neutralise net charge by replacing waters with Na+/Cl- ions.
    # Required because Gel1/2/3MA loses LYS+ when converted to LMA (neutral amide),
    # leaving the box negatively charged. PME's uniform background otherwise
    # masks this with a non-physical plasma. No-op for Gelatin (charge = 0).
    log.info("Running VMD autoionize for charge neutralisation...")
    ion_counts = run_autoionize(cfg.software_paths.vmd, paths.output_dir)
    cfg.calculated_n_sodium = ion_counts["n_sod"]
    cfg.calculated_n_chloride = ion_counts["n_cla"]
    cfg.save(paths.config)

    # Step 6
    pme = prepare_configs(paths.script_dir, paths.output_dir, params.box_length_angstrom)
    log.info("NAMD configs updated (L=%s, PME=%s)", params.box_length_angstrom, pme)

    for stage in NAMD_STAGES:
        log.info("Starting %s simulation...", stage.label)
        try:
            run_stage(cfg.software_paths.namd3, stage, paths.output_dir)
        except subprocess.CalledProcessError:
            log.error(
                "%s simulation failed. Check %s for details.",
                stage.label, paths.output_dir / stage.log,
            )
            raise
        log.info("%s simulation completed.", stage.label)

    log.info("Workflow completed. Outputs in %s", paths.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(ProjectPaths(Path(__file__).resolve().parent.parent))
