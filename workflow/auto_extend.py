"""Workflow 2-B: auto-extend NPT simulation across all subprojects.

For each subsystem (Gelatin, Gel1MA, Gel2MA, Gel3MA):

  1. Detect the highest ``system_npt_partN.restart.coor`` already in
     ``Output/`` (must be N >= 2; Workflow 1 produces parts 1 and 2).
  2. Generate ``npt_{N+1}.conf`` by rewriting the previous part's template
     — only the TCL ``set inputname`` / ``set outputname``, the ``run`` step
     count, and ``CUDASOAintegrate`` change. PBC, PME and parameter lines
     are inherited as-is so trajectory continuity is preserved.
  3. Run NAMD 3 with the same full-speed flags as Workflow 1.

Run mode is always **full speed**: ``CUDASOAintegrate on`` + the standard
``NAMD3_LAUNCH_FLAGS`` (defined in each subsystem's ``script/namd_runner.py``).
The patch step rewrites any stale ``CUDASOAintegrate off`` it inherits, so a
.conf that came from a previous throttled run is auto-normalised on the next
extension.

Safety: refuses to overwrite an existing ``system_npt_part{N+1}.dcd`` or
``restart.coor``; an orphan ``npt_{N+1}.conf`` from a prior failed launch is
allowed to be overwritten.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from workflow2 import SubSystem, iter_subsystems

log = logging.getLogger(__name__)

EXTEND_STEPS = 5_000_000   # 10 ns at 2 fs timestep (was 20 ns; shortened for the
                           # convergence top-up of the big-box 4-system batch)
MIN_BASE_PART = 2          # Workflow 1 ends after part 2; extension starts at 3
CUDA_SOA_INTEGRATE = "on"  # always-on; NAMD 3 GPU-resident integrator

_RESTART_RE = re.compile(r"^system_npt_part(\d+)\.restart\.coor$")


def _detect_latest_part(output_dir: Path) -> int:
    parts = [
        int(match.group(1))
        for path in output_dir.glob("system_npt_part*.restart.coor")
        for match in [_RESTART_RE.match(path.name)]
        if match
    ]
    return max(parts, default=0)


def _patch_conf_text(template: str, current_part: int, next_part: int) -> str:
    """Rewrite TCL set-variables, run count, and the CUDASOAintegrate line.

    The .conf files reference ``$inputname`` / ``$outputname``, so updating
    the two ``set ...`` lines automatically propagates to ``binCoordinates``,
    ``binVelocities``, ``extendedSystem`` and ``outputName``.

    ``CUDASOAintegrate`` is forced to ``on`` (added if missing) — this
    normalises any stale ``off`` value that may have been inherited from a
    previous throttled run.
    """
    text = re.sub(
        r"^(\s*set\s+inputname\s+)\S+",
        rf"\g<1>system_npt_part{current_part}",
        template, flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s*set\s+outputname\s+)\S+",
        rf"\g<1>system_npt_part{next_part}",
        text, flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s*run\s+)\d+",
        rf"\g<1>{EXTEND_STEPS}",
        text, flags=re.MULTILINE,
    )

    if re.search(r"^\s*CUDASOAintegrate\s+", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*CUDASOAintegrate\s+)\S+",
            rf"\g<1>{CUDA_SOA_INTEGRATE}",
            text, flags=re.MULTILINE,
        )
    else:
        text = text.rstrip() + (
            f"\n\n# NAMD 3 GPU-resident integrator.\n"
            f"CUDASOAintegrate    {CUDA_SOA_INTEGRATE}\n"
        )
    return text


def _generate_next_conf(output_dir: Path, current_part: int, next_part: int) -> Path:
    template_path = output_dir / f"npt_{current_part}.conf"
    if not template_path.exists():
        raise FileNotFoundError(f"Template conf not found: {template_path}")

    # Safety lock only on real simulation artefacts; an orphan .conf from a
    # previously-failed launch is harmless and gets overwritten on retry.
    next_dcd = output_dir / f"system_npt_part{next_part}.dcd"
    next_coor = output_dir / f"system_npt_part{next_part}.restart.coor"
    if next_dcd.exists() or next_coor.exists():
        raise FileExistsError(
            f"Part {next_part} simulation output already exists in {output_dir}; "
            "refusing to overwrite."
        )

    patched = _patch_conf_text(
        template_path.read_text(encoding="utf-8"),
        current_part, next_part,
    )
    next_conf = output_dir / f"npt_{next_part}.conf"
    next_conf.write_text(patched, encoding="utf-8")
    return next_conf


def _prepare_extension(system: SubSystem,
                       max_part: int | None = None) -> tuple[Path, int] | None:
    """Return (next_conf_path, next_part), or None if this subsystem is skipped.

    Raises ``FileExistsError`` if the safety lock fires (caller treats as warning).
    Raises ``FileNotFoundError`` if the previous-part template is missing.

    ``max_part`` (optional): refuse to extend if doing so would create a part
    strictly greater than this cap. Used to align replicas at a target length
    (e.g. ``--max-part 3`` keeps every subsystem at 40 ns and prevents
    accidentally pushing rep1 past rep2/rep3).
    """
    current_part = _detect_latest_part(system.output_dir)
    if current_part < MIN_BASE_PART:
        log.warning(
            "[%s] need at least part %d completed (found %d); skipping.",
            system.name, MIN_BASE_PART, current_part,
        )
        return None

    next_part = current_part + 1
    if max_part is not None and next_part > max_part:
        log.info(
            "[%s] already at part %d (>= max-part %d); skipping.",
            system.name, current_part, max_part,
        )
        return None

    log.info(
        "[%s] latest=part %d → preparing part %d.",
        system.name, current_part, next_part,
    )
    conf_path = _generate_next_conf(system.output_dir, current_part, next_part)
    log.info("[%s] wrote %s.", system.name, conf_path.name)
    return conf_path, next_part


def _run_extension(system: SubSystem, conf_path: Path, next_part: int) -> None:
    runner = system.namd_runner
    stage = runner.NamdStage(
        conf=conf_path.name,
        log=f"npt_{next_part}.log",
        label=f"{system.name} NPT Part {next_part}",
    )
    log.info("[%s] starting %s...", system.name, stage.label)
    runner.run_stage(system.namd_exe, stage, system.output_dir)
    log.info("[%s] %s completed.", system.name, stage.label)


def _matches_rep_filter(sys_name: str, rep_filter: str | None) -> bool:
    """Decide whether ``sys_name`` belongs to the requested ``--rep`` cohort.

    ``rep_filter`` can be:
      * ``None``          — include everything (legacy behaviour)
      * ``"rep1"``        — bare base subsystem names (no ``_rep`` suffix)
      * ``"rep2"`` / etc. — names ending in ``_<rep>``
    """
    if rep_filter is None:
        return True
    if rep_filter == "rep1":
        return "_rep" not in sys_name
    return sys_name.endswith(f"_{rep_filter}")


def run_all(root_dir: Path,
            rep_filter: str | None = None,
            max_part: int | None = None,
            systems_filter: list[str] | None = None) -> list[str]:
    """Extend every (filtered) subsystem under ``root_dir`` by one NPT part.

    Args:
      rep_filter: optionally restrict to a single replica cohort. See
        :func:`_matches_rep_filter`.
      max_part: optional upper bound — subsystems already at this part are
        skipped. Critical when aligning replicas at a target length so this
        script does not accidentally push the most-advanced one past the
        others.
      systems_filter: optional explicit list of subsystem names to process.
        When given, ``rep_filter`` still applies (compose narrow filters) but
        names not in the list are silently skipped. Use to single-system
        extend in a multi-system cohort (e.g. ``--rep rep2 --systems
        Gel1MA_rep2`` extends only Gel1MA_rep2 inside the rep2 cohort).

    Returns the list of subsystem names that failed (empty list = success).
    """
    failures: list[str] = []
    matched = 0
    sys_filter_set = set(systems_filter) if systems_filter else None
    for system in iter_subsystems(root_dir):
        if not _matches_rep_filter(system.name, rep_filter):
            continue
        if sys_filter_set is not None and system.name not in sys_filter_set:
            continue
        matched += 1
        log.info("=== %s ===", system.name)

        try:
            prepared = _prepare_extension(system, max_part=max_part)
        except FileExistsError as exc:
            log.warning("[%s] %s", system.name, exc)
            continue
        except FileNotFoundError as exc:
            log.warning("[%s] template missing: %s", system.name, exc)
            continue

        if prepared is None:
            continue

        conf_path, next_part = prepared
        try:
            _run_extension(system, conf_path, next_part)
        except FileNotFoundError as exc:
            failures.append(system.name)
            log.error(
                "[%s] NAMD executable not found: %s — fix `software_paths.namd3` in %s.",
                system.name, exc, system.config_path,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(system.name)
            log.error(
                "[%s] NAMD failed (exit %s). Check %s.",
                system.name, exc.returncode, system.output_dir / f"npt_{next_part}.log",
            )
        except Exception as exc:
            failures.append(system.name)
            log.error("[%s] unexpected error: %s", system.name, exc)

    if matched == 0:
        log.warning("No subsystems matched filter rep=%r — nothing to do.", rep_filter)
    return failures


def _parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="auto_extend.py",
        description="Extend each (filtered) subsystem's NPT trajectory by one part.",
    )
    parser.add_argument(
        "--rep", default=None,
        help="Restrict to one replica cohort: "
             "'rep1' = base subsystems (no _rep suffix), "
             "'rep2' = *_rep2 subprojects, 'rep3' = *_rep3, etc. "
             "Default: process every subsystem (legacy behaviour).",
    )
    parser.add_argument(
        "--max-part", type=int, default=None,
        help="Cap the extension target. If a subsystem already has parts up "
             "to this number, it is skipped. Use this to align replicas at "
             "a target trajectory length (e.g. --max-part 3 → cap at 40 ns).",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Explicit subsystem-name list (e.g. Gel1MA_rep2). When given, "
             "only matching names are extended; --rep filter still applies "
             "(both filters are AND'd). Use for single-system extension in "
             "a multi-system cohort.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List which subsystems would be extended, then exit.",
    )
    return parser.parse_args(argv)


def _dry_run_report(root_dir: Path,
                    rep_filter: str | None,
                    max_part: int | None,
                    systems_filter: list[str] | None = None) -> int:
    matched: list[tuple[str, int, int | str]] = []
    sys_filter_set = set(systems_filter) if systems_filter else None
    for system in iter_subsystems(root_dir):
        if not _matches_rep_filter(system.name, rep_filter):
            continue
        if sys_filter_set is not None and system.name not in sys_filter_set:
            continue
        current = _detect_latest_part(system.output_dir)
        if current < MIN_BASE_PART:
            target = "skip (no part 2)"
        elif max_part is not None and current + 1 > max_part:
            target = f"skip (already at part {current}, cap = {max_part})"
        else:
            target = current + 1
        matched.append((system.name, current, target))

    if not matched:
        log.warning("No subsystems matched filter rep=%r systems=%s.",
                    rep_filter, systems_filter)
        return 1

    log.info("Plan (%d subsystem(s); filter rep=%s, systems=%s, max_part=%s):",
             len(matched), rep_filter, systems_filter, max_part)
    for name, cur, tgt in matched:
        log.info("  %-22s  latest=part %d  ->  %s", name, cur, tgt)
    log.info("Dry-run only — nothing executed.")
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args(argv)
    # This copy lives in `production/sim_scripts/`; the system folders
    # (Gelatin/, Gel1MA/, ...) are one level up under `production/`.
    root_dir = Path(__file__).resolve().parent.parent

    if args.dry_run:
        sys.exit(_dry_run_report(root_dir, args.rep, args.max_part, args.systems))

    failures = run_all(root_dir, rep_filter=args.rep, max_part=args.max_part,
                       systems_filter=args.systems)
    if failures:
        log.error("Workflow 2-B finished with failures: %s", ", ".join(failures))
        sys.exit(1)
    log.info("Workflow 2-B completed.")


if __name__ == "__main__":
    main()
