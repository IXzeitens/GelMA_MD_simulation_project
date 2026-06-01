"""One-shot batch driver for Workflow 1 (main.py) over a replica cohort.

What this does in a single invocation:
  1. Auto-clones any missing ``<base>_<rep>`` subprojects via
     ``sim_scripts/clone_subproject.py`` (unless --no-clone is set).
  2. For each system in the cohort, runs ``main.py`` (NVT 1 ns + NPT
     part 1 + part 2 = 20 ns) unless ``system_npt_part2.restart.coor``
     already exists (idempotent — safe to re-run after interruption).
  3. Prints a wall-clock summary with NVT / NPT1 / NPT2 status per
     system.

Sequential GPU execution — single NAMD3 process saturates the device,
parallel launches would just contend.

Usage
-----
    # rep1 cohort (the original 4 base systems)
    python run_batch_workflow1.py                              # all 4 base
    python run_batch_workflow1.py --systems Gel2MA             # one base
    python run_batch_workflow1.py --dry-run

    # rep2 cohort — one-shot: clones if missing, then runs main.py per system
    python run_batch_workflow1.py --rep rep2 --seed-base 20260615

    # Same idea for rep3+ (auto seed = seed-base + 100 × rep number)
    python run_batch_workflow1.py --rep rep3 --seed-base 20260615
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("batch")

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_SYSTEMS = ("Gelatin", "Gel1MA", "Gel2MA", "Gel3MA")
CLONE_SCRIPT = ROOT / "sim_scripts" / "clone_subproject.py"
DEFAULT_CLONE_SEED_BASE = 20260000     # matches clone_subproject SEED_BASE_DEFAULT
CLONE_SEED_STRIDE = 100                # matches clone_subproject SEED_STRIDE
NPT2_RESTART_FILE = "system_npt_part2.restart.coor"


def _systems_for_rep(rep: str) -> list[str]:
    """Translate base-system names into the requested replica cohort.

    rep1   → bare base names (Gelatin, Gel1MA, ...)
    rep2+  → "<base>_<rep>" names matching clone_subproject.py output
    """
    if rep == "rep1":
        return list(DEFAULT_BASE_SYSTEMS)
    return [f"{base}_{rep}" for base in DEFAULT_BASE_SYSTEMS]


def _is_npt2_done(sys_dir: Path) -> bool:
    """A subproject is considered done when NPT part 2 restart file exists."""
    return (sys_dir / "Output" / NPT2_RESTART_FILE).exists()


def _seed_for_rep(rep: str, seed_base: int) -> int:
    """Mirror clone_subproject._seed_for: base + 100 × rep_number."""
    m = re.search(r"\d+", rep)
    n = int(m.group(0)) if m else 0
    return seed_base + n * CLONE_SEED_STRIDE


def _auto_clone_missing(target_names: list[str], rep: str,
                        seed_base: int | None) -> None:
    """Invoke clone_subproject.py per missing ``<base>_<rep>`` target.

    Skipped entirely when rep == 'rep1' (base systems are the originals,
    never auto-created). Uses single-clone --src/--dest mode so partial
    cohorts work cleanly (clone_subproject's bulk --rep mode would error
    on whichever sibling dir already exists).
    """
    if rep == "rep1":
        return
    needs = [n for n in target_names if not (ROOT / n / "main.py").exists()]
    if not needs:
        return
    if not CLONE_SCRIPT.exists():
        log.error("clone_subproject.py not found at %s — cannot auto-clone.", CLONE_SCRIPT)
        raise SystemExit(1)

    base_seed = seed_base if seed_base is not None else DEFAULT_CLONE_SEED_BASE
    seed = _seed_for_rep(rep, base_seed)
    suffix = f"_{rep}"

    log.info("Auto-clone: %d %s cohort dir(s) missing — invoking clone_subproject.py",
             len(needs), rep)
    for name in needs:
        if not name.endswith(suffix):
            log.warning("  cannot derive base name for %s (no %s suffix); skipping clone",
                        name, suffix)
            continue
        base_name = name[: -len(suffix)]
        cmd = [
            sys.executable, str(CLONE_SCRIPT),
            "--src", base_name, "--dest", name,
            "--seed", str(seed),
        ]
        log.info("  -> cloning %s → %s (seed %d)", base_name, name, seed)
        subprocess.run(cmd, check=True, cwd=ROOT)


def _run_one(sys_dir: Path) -> dict:
    """Execute main.py in a subproject, piping `\\n\\n` to its stdin."""
    log.info("=" * 60)
    log.info("Starting %s ...", sys_dir.name)
    log.info("=" * 60)

    t0 = time.time()
    completed = {"nvt": False, "npt_part1": False, "npt_part2": False}
    try:
        # Pipe \n\n so Step 0's two `input(...)` calls fall back to config.json
        subprocess.run(
            [sys.executable, "main.py"],
            cwd=sys_dir,
            input="\n\n",
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.error("%s failed with exit code %d", sys_dir.name, e.returncode)
    elapsed = time.time() - t0

    out = sys_dir / "Output"
    completed["nvt"]       = (out / "system_nvt.restart.coor").exists()
    completed["npt_part1"] = (out / "system_npt_part1.restart.coor").exists()
    completed["npt_part2"] = (out / "system_npt_part2.restart.coor").exists()
    return {
        "system": sys_dir.name,
        "elapsed_h": elapsed / 3600.0,
        **completed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch Workflow 1 over a replica cohort.")
    ap.add_argument("--rep", default="rep1",
                    help="Replica cohort. 'rep1' = base systems (default). 'rep2' / 'rep3' / "
                         "etc. = '<base>_<rep>' cohorts produced by clone_subproject.py.")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Override target list. If omitted, all 4 systems of the chosen --rep "
                         "cohort are run. Mix with --rep to target a specific system in a "
                         "cohort (e.g. --rep rep2 --systems Gel2MA_rep2).")
    ap.add_argument("--seed-base", type=int, default=None,
                    help=f"Base seed for auto-cloning missing rep cohort dirs (default: "
                         f"{DEFAULT_CLONE_SEED_BASE}). Effective seed = base + 100 × rep_number "
                         f"so rep2 with base 20260615 yields seed 20260815.")
    ap.add_argument("--no-clone", action="store_true",
                    help="Skip auto-cloning missing rep cohort dirs (just warn).")
    ap.add_argument("--force-rerun", action="store_true",
                    help="Re-run main.py even on systems where NPT part 2 already finished. "
                         "Default: those are skipped (idempotent re-run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s %(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Decide target list: explicit --systems wins, else derive from --rep.
    target_names = args.systems if args.systems is not None else _systems_for_rep(args.rep)

    # Auto-clone missing rep cohort dirs (unless --no-clone or rep == rep1).
    if not args.no_clone and not args.dry_run:
        try:
            _auto_clone_missing(target_names, args.rep, args.seed_base)
        except subprocess.CalledProcessError as exc:
            log.error("Auto-clone failed (exit code %d). Aborting.", exc.returncode)
            return 1

    # Re-scan after possible cloning.
    targets: list[Path] = []
    missing: list[str] = []
    skipped: list[str] = []
    for name in target_names:
        d = ROOT / name
        if not d.is_dir() or not (d / "main.py").exists():
            missing.append(name)
            continue
        if not args.force_rerun and _is_npt2_done(d):
            skipped.append(name)
            continue
        targets.append(d)

    if missing:
        log.warning("Missing %d system(s) (no main.py after auto-clone): %s",
                    len(missing), missing)
        if args.no_clone and args.rep != "rep1":
            log.warning(
                "If these are missing clones, first run: "
                "python sim_scripts/clone_subproject.py --rep %s", args.rep,
            )
    if skipped:
        log.info("Already complete (NPT2 done), skipping: %s", skipped)

    if not targets:
        if skipped:
            log.info("All targets already complete. Use --force-rerun to redo.")
            return 0
        log.error("No valid targets.")
        return 1

    log.info("Plan (--rep %s): %s", args.rep, " -> ".join(t.name for t in targets))
    if args.dry_run:
        return 0

    t_batch = time.time()
    results = [_run_one(d) for d in targets]
    total_h = (time.time() - t_batch) / 3600.0

    # Widen system column to fit "Gelatin_rep2" (12 chars) etc.
    log.info("")
    log.info("=" * 64)
    log.info(" BATCH SUMMARY  (total wall = %.2f h)", total_h)
    log.info("=" * 64)
    log.info(f"  {'system':14s}  {'wall(h)':>8s}  {'NVT':>4s}  {'NPT1':>4s}  {'NPT2':>4s}")
    for r in results:
        log.info(f"  {r['system']:14s}  {r['elapsed_h']:8.2f}"
                 f"  {'OK' if r['nvt'] else 'NO':>4s}"
                 f"  {'OK' if r['npt_part1'] else 'NO':>4s}"
                 f"  {'OK' if r['npt_part2'] else 'NO':>4s}")
    log.info("=" * 64)

    failures = [r for r in results if not r["npt_part2"]]
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
