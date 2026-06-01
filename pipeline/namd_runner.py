"""NAMD configuration patching + simulation execution."""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PME_SMOOTH_FACTORS = (2, 3, 5)

# NAMD 3 (multicore-CUDA build) launch flags.
# +p<N>             : CPU worker threads. With CUDASOAintegrate enabled in
#                    the .conf, the GPU handles integration; 4 workers is the
#                    sweet spot for single-GPU runs on modern hardware
#                    (more threads compete for cache without doing useful work).
# +setcpuaffinity   : pin workers to cores (recommended on Windows for stable perf).
# +devices <ids>    : comma-separated CUDA device list.
NAMD3_LAUNCH_FLAGS: tuple[str, ...] = ("+p4", "+setcpuaffinity", "+devices", "0")


@dataclass(frozen=True)
class NamdStage:
    conf: str
    log: str
    label: str


NAMD_STAGES: tuple[NamdStage, ...] = (
    NamdStage("NVT.conf", "nvt.log", "NVT"),
    NamdStage("npt_1.conf", "npt_1.log", "NPT Part 1"),
    NamdStage("npt_2.conf", "npt_2.log", "NPT Part 2"),
)


def smallest_smooth_int(target: int) -> int:
    """Smallest integer >= target whose only prime factors are 2, 3, 5 (good for NAMD PME)."""
    def is_smooth(n: int) -> bool:
        for p in PME_SMOOTH_FACTORS:
            while n % p == 0:
                n //= p
        return n == 1

    val = int(target)
    while not is_smooth(val):
        val += 1
    return val


def _patch_text(text: str, box_L: int, pme_size: int) -> str:
    text = re.sub(r"(cellBasisVector1\s+)[\d.]+", rf"\g<1>{box_L}.0", text)
    text = re.sub(r"(cellBasisVector2\s+0\.0\s+)[\d.]+", rf"\g<1>{box_L}.0", text)
    text = re.sub(r"(cellBasisVector3\s+0\.0\s+0\.0\s+)[\d.]+", rf"\g<1>{box_L}.0", text)
    text = re.sub(r"(PMEGridSize[XYZ]\s+)\d+", rf"\g<1>{pme_size}", text)
    text = re.sub(r"parameters\s+(?!\.\./)([\w.\-]+)", r"parameters          ../script/\1", text)
    return text


def prepare_configs(script_dir: Path, output_dir: Path, box_L: int) -> int:
    """Copy NAMD templates into output_dir with PBC/PME/parameters patched. Returns PME grid size."""
    pme_size = smallest_smooth_int(box_L)
    for stage in NAMD_STAGES:
        src = script_dir / stage.conf
        if not src.exists():
            log.warning("%s not found in %s", stage.conf, script_dir)
            continue
        patched = _patch_text(src.read_text(encoding="utf-8"), box_L, pme_size)
        (output_dir / stage.conf).write_text(patched, encoding="utf-8")
    return pme_size


def run_stage(namd_exe: str, stage: NamdStage, output_dir: Path) -> None:
    """Launch NAMD 3 for a single stage; stdout/stderr captured to stage.log."""
    log_path = output_dir / stage.log
    cmd = [namd_exe, *NAMD3_LAUNCH_FLAGS, stage.conf]
    log.info("Launching: %s", " ".join(cmd))
    with log_path.open("w") as log_file:
        subprocess.run(
            cmd,
            cwd=output_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
