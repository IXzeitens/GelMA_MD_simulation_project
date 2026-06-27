"""Shared utilities for Workflow 2 root-level scripts.

Workflow 2 lives at the project root and operates on the four self-contained
subprojects under ``0510/``: Gelatin, Gel1MA, Gel2MA, Gel3MA. Each subproject
owns its own Workflow 1 outputs (``Output/debug_1.{pdb,psf}``,
``Output/npt_*.conf``, restart files, ...) and a ``script/`` package that
includes ``namd_runner``.

This module provides a uniform ``SubSystem`` view so that ``NPT_conti.py`` and
``auto_extend.py`` can iterate over the subprojects without duplicating the
discovery / config-loading boilerplate.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator

#: Base subsystem names (rep1). Top-level replicates ``<name>_rep<N>`` are
#: auto-discovered at runtime — see :func:`discover_target_systems`.
BASE_SYSTEMS: tuple[str, ...] = ("Gelatin", "Gel1MA", "Gel2MA", "Gel3MA")

#: Back-compat alias. New code should use :func:`discover_target_systems`
#: (which includes any ``<base>_rep<N>`` clones produced by clone_subproject.py).
TARGET_SYSTEMS: tuple[str, ...] = BASE_SYSTEMS

DEFAULT_NAMD_EXE = "namd3"
NAMD_KEY_PRIORITY: tuple[str, ...] = ("namd3", "namd2", "namd")


def discover_target_systems(root_dir: Path) -> tuple[str, ...]:
    """Return BASE_SYSTEMS + sibling dirs matching ``<base>_rep<N>`` or
    ``<base>_c<conc>_<n>`` (the mechanics-arm concentration clones).

    Used by Workflow 2-B / 2-A / refresh-confs / nvt-thermo so adding a new
    replica or concentration clone (via ``clone_subproject.py``) is picked up
    without editing this file. ``--systems`` still lets callers target a subset.
    """
    found: list[str] = []
    seen: set[str] = set()
    for base in BASE_SYSTEMS:
        if (root_dir / base).is_dir() and base not in seen:
            found.append(base); seen.add(base)
        # Replicas (Gelatin_rep2, ...) and mechanics clones (Gel3MA_c10_0, ...).
        for pattern in (f"{base}_rep*", f"{base}_c*"):
            for p in sorted(root_dir.glob(pattern)):
                if p.is_dir() and p.name not in seen:
                    found.append(p.name); seen.add(p.name)
    return tuple(found)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubSystem:
    """A single Workflow 1 subproject ready for Workflow 2 operations."""

    name: str
    root: Path
    namd_exe: str
    box_length: int
    namd_runner: ModuleType

    @property
    def script_dir(self) -> Path:
        return self.root / "script"

    @property
    def output_dir(self) -> Path:
        return self.root / "Output"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"


def _read_namd_exe(software_paths: dict) -> str:
    # NAMD 3 is canonical; older configs may still use `namd2` / `namd`.
    for key in NAMD_KEY_PRIORITY:
        value = software_paths.get(key)
        if value:
            return value
    return DEFAULT_NAMD_EXE


def _read_box_length(config: dict) -> int:
    box = config.get("calculated_box_L_Angstrom") or config.get("box_L")
    if not box:
        raise ValueError("calculated_box_L_Angstrom not found in config.json")
    return int(box)


def _load_namd_runner(system_name: str, script_dir: Path) -> ModuleType:
    runner_path = script_dir / "namd_runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"namd_runner.py missing: {runner_path}")

    # Import as a uniquely-named module so each subsystem's namd_runner stays
    # isolated (avoids sys.path pollution and cross-subsystem caching).
    module_name = f"workflow2._namd_runner_{system_name}"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load namd_runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_subsystem(root_dir: Path, name: str) -> SubSystem:
    system_root = root_dir / name
    config_path = system_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing for {name}: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    namd_exe = _read_namd_exe(config.get("software_paths") or {})
    box_length = _read_box_length(config)
    namd_runner = _load_namd_runner(name, system_root / "script")

    return SubSystem(
        name=name,
        root=system_root,
        namd_exe=namd_exe,
        box_length=box_length,
        namd_runner=namd_runner,
    )


def iter_subsystems(root_dir: Path) -> Iterator[SubSystem]:
    """Yield each loadable subsystem (incl. ``*_rep*`` clones).

    Discovery is dynamic — ``clone_subproject.py`` outputs are picked up
    automatically without editing ``workflow2.BASE_SYSTEMS``.
    """
    for name in discover_target_systems(root_dir):
        try:
            yield load_subsystem(root_dir, name)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            log.warning("Skipping %s: %s", name, exc)
