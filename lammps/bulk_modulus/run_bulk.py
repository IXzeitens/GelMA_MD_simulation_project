"""Bulk-modulus driver: fan out (system × pressure) LAMMPS runs and fit K.

Layout produced under lammps_workflow/4_bulk/<system>/P<bar>/:
    in.bulk          -> rendered template
    *.vp.log         -> ave/time output: step Pinst Vinst Tinst Dinst
    *.final.data, *.lammpstrj, *.log etc.

After all runs finish, --analyze parses each vp.log, computes <V> over the
production phase, fits P vs <V> for each system, and writes:
    lammps_workflow/4_bulk/results/<system>_bulk.csv      raw points
    lammps_workflow/4_bulk/results/bulk_summary.csv       K per system
    lammps_workflow/4_bulk/results/bulk_K_vs_DS.png       Chiu Fig 7 mirror

Subcommands:
    smoke     short run, single (Gel3MA, 1 bar) — verify the LAMMPS input
    run       full batch (4 systems × N pressures)
    analyze   post-process all vp.log under the tree
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Auto-detect: WSL/Linux uses /mnt/c/...; Windows native uses C:\...
if os.name == "posix":
    REPO = Path("/mnt/c/Users/User/Desktop/Work/0510")
else:
    REPO = Path(r"C:\Users\User\Desktop\Work\0510")
BULK_DIR = REPO / "lammps_workflow" / "4_bulk"
TEMPLATE = BULK_DIR / "in.bulk.template"

SYSTEMS = ["Gelatin", "Gel1MA", "Gel2MA", "Gel3MA"]

# Degree of substitution per system, from chain stoichiometry
# 0 / 1 / 2 / 3 MA per 24-res chain (3 LYS sites), expressed as % of LYS modified
DS_PCT = {"Gelatin": 0.0, "Gel1MA": 33.3, "Gel2MA": 66.7, "Gel3MA": 100.0}

# Production pressures (bar). 1 bar reference + three at higher P to give a
# clear linear slope. Avoid negative pressure (cavitation risk).
DEFAULT_PRESSURES = [1.0, 500.0, 1000.0, 2000.0]


def wsl_path(p: Path) -> str:
    s = str(p).replace("\\", "/")
    if s[1:3] == ":/":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def render_input(out_path: Path) -> None:
    """Copy template to in.bulk in run dir. -var passed on the command line."""
    shutil.copyfile(TEMPLATE, out_path)


def run_one(system: str, pressure: float, eq_steps: int, prod_steps: int,
            timestep_fs: float, temperature: float, sample_every: int,
            backend: str = "cpu", lmp_bin: str = "/usr/bin/lmp", np: int = 1,
            dry_run: bool = False) -> Path:
    """Stage and launch one (system, pressure) LAMMPS run. Returns run dir.

    backend: 'cpu' (serial), 'mpi' (mpirun -np N), 'gpu' (KOKKOS Kokkos cuda + 1 MPI rank)
    """
    sys_dir = BULK_DIR / system
    p_tag = f"P{int(round(pressure))}bar"
    run_dir = sys_dir / p_tag
    # WSL DrvFs (Windows mount) sometimes raises FileExistsError even with
    # exist_ok=True and even when the dir is not visible to stat. Use os.makedirs
    # directly and swallow the error — writes below will surface any real issue.
    try:
        os.makedirs(str(run_dir), exist_ok=True)
    except FileExistsError:
        pass

    # Stage data + coeffs (symlink would be cleaner but Windows perms are messy)
    for fname in ("project.data", "project.in.coeffs"):
        src = sys_dir / fname
        dst = run_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"missing {src} — run prep_bulk.py first")
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copyfile(src, dst)

    render_input(run_dir / "in.bulk")

    out_prefix = f"bulk_{system}_{p_tag}"
    log_file = run_dir / f"{out_prefix}.lmp.log"

    # Build LAMMPS invocation based on backend
    common_vars = (
        f"-in in.bulk "
        f"-var TEMPERATURE {temperature} "
        f"-var PRESSURE_BAR {pressure} "
        f"-var TIMESTEP_FS {timestep_fs} "
        f"-var EQ_STEPS {eq_steps} "
        f"-var PROD_STEPS {prod_steps} "
        f"-var SAMPLE_EVERY {sample_every} "
        f"-var OUT_PREFIX {out_prefix} "
        f"-log {out_prefix}.lmp.log"
    )
    if backend == "gpu":
        # KOKKOS GPU: 1 MPI rank per GPU, -k on g 1, -sf kk.
        # dihedral_charmm/kk requires half neighbor list; use newton on.
        invoke = (
            f"mpirun -np 1 {lmp_bin} "
            f"-k on g 1 -sf kk -pk kokkos neigh half newton on "
            f"{common_vars}"
        )
    elif backend == "mpi":
        invoke = f"mpirun -np {np} {lmp_bin} {common_vars}"
    else:  # cpu serial
        invoke = f"{lmp_bin} {common_vars}"
    lmp_cmd = f"cd {wsl_path(run_dir)} && {invoke}"

    print(f"[{system}/{p_tag}] backend={backend} eq={eq_steps} prod={prod_steps} ts={timestep_fs}fs")
    if dry_run:
        print(f"  DRY: {lmp_cmd}")
        return run_dir

    # In WSL/Linux: call bash directly. In Windows: shell out to wsl.
    if os.name == "posix":
        r = subprocess.run(["bash", "-c", lmp_cmd])
    else:
        r = subprocess.run(["wsl", "-e", "bash", "-c", lmp_cmd])
    if r.returncode != 0:
        raise RuntimeError(f"LAMMPS failed for {system} {p_tag}; see {log_file}")
    print(f"  done -> {log_file.name}")
    return run_dir


def parse_vp_log(vp_log: Path, skip_frac: float = 0.1) -> dict:
    """Read fix ave/time output: '# Time-averaged data for fix fLOG' header,
    then 'TimeStep v_Pinst v_Vinst v_Tinst v_Dinst' header, then rows."""
    P, V, T, D = [], [], [], []
    with vp_log.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 5:
                continue
            try:
                _step, p, v, t, d = (float(x) for x in tok[:5])
            except ValueError:
                continue
            P.append(p); V.append(v); T.append(t); D.append(d)
    if not P:
        raise RuntimeError(f"no data parsed from {vp_log}")
    n_skip = int(len(P) * skip_frac)
    P, V, T, D = P[n_skip:], V[n_skip:], T[n_skip:], D[n_skip:]
    mean = lambda xs: sum(xs) / len(xs)
    var = lambda xs, m: sum((x - m) ** 2 for x in xs) / len(xs)
    Pm, Vm, Tm, Dm = mean(P), mean(V), mean(T), mean(D)
    return {
        "n": len(P),
        "P_mean_bar": Pm, "P_std_bar": math.sqrt(var(P, Pm)),
        "V_mean_A3": Vm, "V_std_A3": math.sqrt(var(V, Vm)),
        "T_mean_K": Tm, "D_mean_gcm3": Dm,
    }


def fit_K(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Linear regression P vs V: K = -V_ref * dP/dV.
    Returns (K_bar, R2). V_ref taken as V at lowest P."""
    if len(points) < 2:
        raise ValueError("need >= 2 (P, V) points")
    pts = sorted(points, key=lambda x: x[0])
    V_ref = pts[0][1]
    n = len(pts)
    sx = sum(p[1] for p in pts)
    sy = sum(p[0] for p in pts)
    sxx = sum(p[1]**2 for p in pts)
    sxy = sum(p[0]*p[1] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        raise ValueError("singular regression")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # R^2
    ymean = sy / n
    ss_tot = sum((p[0] - ymean)**2 for p in pts)
    ss_res = sum((p[0] - (slope * p[1] + intercept))**2 for p in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    K_bar = -V_ref * slope
    return K_bar, r2


def analyze(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for system in SYSTEMS:
        sys_dir = BULK_DIR / system
        if not sys_dir.exists():
            continue
        points = []
        per_sys_rows = []
        for p_dir in sorted(sys_dir.glob("P*bar")):
            vp_logs = list(p_dir.glob("*.vp.log"))
            if not vp_logs:
                print(f"  [skip] {p_dir.name}: no vp.log")
                continue
            stats = parse_vp_log(vp_logs[0])
            p_nominal = float(p_dir.name[1:-3])
            row = {"system": system, "P_nominal_bar": p_nominal, **stats}
            per_sys_rows.append(row)
            points.append((stats["P_mean_bar"], stats["V_mean_A3"]))
            print(f"  {system}/{p_dir.name}: <P>={stats['P_mean_bar']:.1f}±{stats['P_std_bar']:.1f} bar  "
                  f"<V>={stats['V_mean_A3']:.0f}±{stats['V_std_A3']:.0f} Å³")
        if not per_sys_rows:
            continue
        # Per-system CSV
        with (results_dir / f"{system}_bulk.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sys_rows[0].keys()))
            w.writeheader(); w.writerows(per_sys_rows)
        if len(points) >= 2:
            K_bar, r2 = fit_K(points)
            K_MPa = K_bar * 1e-1  # 1 bar = 0.1 MPa
            summary_rows.append({
                "system": system,
                "DS_pct": DS_PCT[system],
                "n_pressures": len(points),
                "K_MPa": K_MPa,
                "R2": r2,
            })
            print(f"  -> {system}: K = {K_MPa:.0f} MPa  (R²={r2:.4f}, {len(points)} points)")
        else:
            print(f"  [skip] {system}: only {len(points)} point(s), need ≥2 for K")

    # Summary CSV
    if summary_rows:
        with (results_dir / "bulk_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSummary -> {results_dir / 'bulk_summary.csv'}")
        # Optional: plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ds = [r["DS_pct"] for r in summary_rows]
            K  = [r["K_MPa"] for r in summary_rows]
            chiu_ds = [16.7, 33.3, 50.0]
            chiu_K  = [2824, 2970, 2845]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ds, K, "o-", label="this work (pre-network)")
            ax.plot(chiu_ds, chiu_K, "s--", color="gray",
                    label="Chiu 2024 (post-crosslink GGMA)")
            ax.set_xlabel("Degree of substitution (%)")
            ax.set_ylabel("Bulk modulus K (MPa)")
            ax.set_title("Pre-network bulk modulus vs DS")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(results_dir / "bulk_K_vs_DS.png", dpi=150)
            print(f"Plot   -> {results_dir / 'bulk_K_vs_DS.png'}")
        except ImportError:
            print("matplotlib not available, skipping plot")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_backend(sp):
        sp.add_argument("--backend", choices=["cpu", "mpi", "gpu"], default="cpu")
        sp.add_argument("--lmp-bin", default="/usr/bin/lmp",
                        help="LAMMPS binary path. GPU: ~/miniconda3/envs/lammps_gpu/bin/lmp")
        sp.add_argument("--np", type=int, default=8, help="MPI ranks (mpi backend only)")

    p_smoke = sub.add_parser("smoke", help="short run on Gel3MA P=1 bar to verify input")
    p_smoke.add_argument("--system", default="Gel3MA", choices=SYSTEMS)
    p_smoke.add_argument("--pressure", type=float, default=1.0)
    p_smoke.add_argument("--eq-steps", type=int, default=5000)
    p_smoke.add_argument("--prod-steps", type=int, default=5000)
    p_smoke.add_argument("--sample-every", type=int, default=100)
    p_smoke.add_argument("--timestep-fs", type=float, default=1.0)
    p_smoke.add_argument("--temperature", type=float, default=310.0)
    add_backend(p_smoke)

    p_run = sub.add_parser("run", help="full batch run")
    p_run.add_argument("--systems", nargs="+", default=SYSTEMS)
    p_run.add_argument("--pressures", nargs="+", type=float, default=DEFAULT_PRESSURES)
    p_run.add_argument("--eq-steps", type=int, default=1000000, help="1 ns at 1 fs")
    p_run.add_argument("--prod-steps", type=int, default=4000000, help="4 ns at 1 fs")
    p_run.add_argument("--sample-every", type=int, default=100)
    p_run.add_argument("--timestep-fs", type=float, default=1.0)
    p_run.add_argument("--temperature", type=float, default=310.0)
    p_run.add_argument("--dry-run", action="store_true")
    add_backend(p_run)

    p_an = sub.add_parser("analyze", help="post-process vp.log files into K")

    args = ap.parse_args()

    if args.cmd == "smoke":
        run_one(args.system, args.pressure,
                eq_steps=args.eq_steps, prod_steps=args.prod_steps,
                timestep_fs=args.timestep_fs, temperature=args.temperature,
                sample_every=args.sample_every,
                backend=args.backend, lmp_bin=args.lmp_bin, np=args.np)
    elif args.cmd == "run":
        for sys_name in args.systems:
            for p in args.pressures:
                run_one(sys_name, p,
                        eq_steps=args.eq_steps, prod_steps=args.prod_steps,
                        timestep_fs=args.timestep_fs, temperature=args.temperature,
                        sample_every=args.sample_every,
                        backend=args.backend, lmp_bin=args.lmp_bin, np=args.np,
                        dry_run=args.dry_run)
    elif args.cmd == "analyze":
        analyze(BULK_DIR / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
