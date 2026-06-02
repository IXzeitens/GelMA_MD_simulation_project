"""Compute bulk modulus K from NAMD npt .xst trajectories via the
NPT volume-fluctuation method, no LAMMPS needed.

K_T = <V> k_B T / <(δV)^2>

For each system in production/{Gelatin,Gel1MA,Gel2MA,Gel3MA}/Output/,
concatenate npt_1, npt_2, npt_3 .xst files (after skipping each part's
equilibration tail), compute K_T, and write a CSV + plot.

Pre-network short-chain gelatin/GelMA — pair to Chiu 2024 Fig 7
(post-crosslink GGMA): expected K ~ 2-3 GPa (water-dominated).

Usage:
    python bulk_from_xst.py
    python bulk_from_xst.py --skip-frac 0.5 --parts 2 3   # use only npt_2/3, second halves
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

# 1 Å³ * 1 J/K * 1 K / 1 Å^6 = 1 J/Å³ = 1e30 J/m³ = 1e30 Pa
# K [MPa] = V [Å³] * kB * T / σ²_V [Å^6] * 1e24
KB_J_PER_K = 1.380649e-23
PA_TO_MPA = 1e-6
SCALE = 1e30 * KB_J_PER_K * PA_TO_MPA  # = 1.380649e1

# Simulation data lives alongside this repo; override with GELMA_REPO if elsewhere.
REPO = Path(os.environ.get("GELMA_REPO") or Path(__file__).resolve().parents[3])
ARCHIVE = REPO / "production"
OUT_DIR = REPO / "lammps_workflow" / "4_bulk" / "results_xst"

SYSTEMS = ["Gelatin", "Gel1MA", "Gel2MA", "Gel3MA"]
DS_PCT = {"Gelatin": 0.0, "Gel1MA": 33.3, "Gel2MA": 66.7, "Gel3MA": 100.0}
T_K = 310.0


def parse_xst(xst: Path) -> tuple[list[int], list[float]]:
    """Return (steps, volumes_A3). Assumes orthorhombic box.
    .xst columns: step a_x a_y a_z b_x b_y b_z c_x c_y c_z ..."""
    steps, vols = [], []
    for line in xst.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if len(tok) < 10:
            continue
        step = int(float(tok[0]))
        a_x, b_y, c_z = float(tok[1]), float(tok[5]), float(tok[9])
        V = a_x * b_y * c_z
        steps.append(step)
        vols.append(V)
    return steps, vols


def block_average_K(V: list[float], n_blocks: int = 5) -> tuple[float, float]:
    """Split V into n_blocks, compute K per block, return (mean K, std K)."""
    if len(V) < n_blocks * 5:
        return float("nan"), float("nan")
    bs = len(V) // n_blocks
    Ks = []
    for i in range(n_blocks):
        block = V[i*bs:(i+1)*bs]
        m = sum(block) / len(block)
        var = sum((x - m)**2 for x in block) / len(block)
        if var <= 0:
            continue
        K_MPa = SCALE * m * T_K / var
        Ks.append(K_MPa)
    if not Ks:
        return float("nan"), float("nan")
    Km = sum(Ks) / len(Ks)
    Kstd = math.sqrt(sum((k - Km)**2 for k in Ks) / max(1, len(Ks) - 1))
    return Km, Kstd


def analyze_system(name: str, parts: list[int], skip_frac: float) -> dict:
    """Concatenate selected npt parts and compute K."""
    out_dir = ARCHIVE / name / "Output"
    all_V = []
    per_part = []
    for p in parts:
        xst = out_dir / f"system_npt_part{p}.xst"
        if not xst.exists():
            print(f"  [skip] {name} part{p}: no .xst")
            continue
        _steps, V = parse_xst(xst)
        n_skip = int(len(V) * skip_frac)
        kept = V[n_skip:]
        per_part.append({"part": p, "n_total": len(V), "n_kept": len(kept),
                         "V_mean": sum(kept)/len(kept) if kept else float("nan")})
        all_V.extend(kept)

    if not all_V:
        return {"system": name, "error": "no data"}

    V_mean = sum(all_V) / len(all_V)
    V_var = sum((v - V_mean)**2 for v in all_V) / len(all_V)
    V_std = math.sqrt(V_var)
    K_total_MPa = SCALE * V_mean * T_K / V_var if V_var > 0 else float("nan")
    K_block, K_std = block_average_K(all_V)

    rho_proxy = V_mean ** (1/3)  # box side, for sanity check
    return {
        "system": name,
        "DS_pct": DS_PCT[name],
        "n_frames": len(all_V),
        "V_mean_A3": V_mean,
        "V_std_A3": V_std,
        "V_rel_fluct_pct": 100 * V_std / V_mean,
        "box_side_A": rho_proxy,
        "K_pooled_MPa": K_total_MPa,
        "K_blockmean_MPa": K_block,
        "K_blockstd_MPa": K_std,
        "per_part": per_part,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3],
                    help="Which npt parts to include (default: all 3)")
    ap.add_argument("--skip-frac", type=float, default=0.2,
                    help="Skip first fraction of each part as eq tail (default: 0.2)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Parts used: {args.parts}, skip first {args.skip_frac*100:.0f}% of each\n")
    print(f"{'sys':<8} {'DS%':>5} {'N':>5} {'<V>/A3':>10} {'sV/V':>7} {'L/A':>7} {'K_pool/MPa':>10} {'K_block+/-s/MPa':>18}")
    rows = []
    for sys_name in SYSTEMS:
        r = analyze_system(sys_name, args.parts, args.skip_frac)
        if "error" in r:
            print(f"{sys_name:<8} {r['error']}")
            continue
        print(f"{r['system']:<8} {r['DS_pct']:>5.1f} {r['n_frames']:>5d} "
              f"{r['V_mean_A3']:>10.0f} {r['V_rel_fluct_pct']:>6.3f}% "
              f"{r['box_side_A']:>7.2f} "
              f"{r['K_pooled_MPa']:>10.0f} "
              f"{r['K_blockmean_MPa']:>7.0f}±{r['K_blockstd_MPa']:.0f}")
        rows.append({k: v for k, v in r.items() if k != "per_part"})

    if not rows:
        print("No data, nothing to write")
        return 1

    # CSV
    csv_path = OUT_DIR / "bulk_from_xst.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {csv_path}")

    # Plot — paper style (mirror production/plot/_shared.apply_paper_style)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({
            "font.family":       "DejaVu Sans",
            "font.size":         10,
            "axes.titlesize":    11,
            "axes.labelsize":    10,
            "xtick.labelsize":   9,
            "ytick.labelsize":   9,
            "legend.fontsize":   9,
            "legend.frameon":    False,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.linewidth":    0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi":       300,
            "savefig.bbox":      "tight",
        })

        ds   = [r["DS_pct"] for r in rows]
        K    = [r["K_blockmean_MPa"] for r in rows]
        Kerr = [r["K_blockstd_MPa"] for r in rows]

        THIS_COLOR = "#1e3f6e"   # matches DS=100 dark blue in plot_merged.DS_COLORS

        fig, ax = plt.subplots(figsize=(5.8, 4.0))

        ax.errorbar(ds, K, yerr=Kerr, fmt="o-", color=THIS_COLOR,
                    markersize=6, linewidth=1.4, elinewidth=0.9, capsize=3)

        ax.set_xlabel("Degree of substitution (%)")
        ax.set_ylabel("Bulk modulus  K  (MPa)")
        ax.set_title("Bulk modulus vs DS", loc="left", fontweight="bold", pad=8)

        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 33, 67, 100])

        # subtle horizontal reference only (no full grid)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        ax.tick_params(direction="out", length=4, width=0.6)

        # Full frame (override paper style's top/right=off)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(0.8)

        fig.tight_layout()
        png = OUT_DIR / "bulk_K_vs_DS_xst.png"
        fig.savefig(png, dpi=300)
        print(f"-> {png}")
    except ImportError:
        print("(matplotlib not found, skip plot)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
