"""Per-system time-series figures from the existing per-frame CSV.

For each of the 4 systems, generates a single multi-panel overview figure
(saved to `production/plot/<sys>/Figure_<sys>_overview.png` + .pdf) plus
group-focused figures (conformational / SASA / H-bond / MA cluster).

Pulls directly from `<sys>/GelMA_analysis_rep1.csv` (time-series, 301 frames
× 30 ns), not from the equilibrium summary — for per-system plots we want
to SEE the trajectory and the equilibrium window highlighted.

Usage:
    python plot_per_system.py                     # all 4 systems, all groups
    python plot_per_system.py --systems Gel3MA    # one system
    python plot_per_system.py --groups conformational hbond
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    EQ_WINDOW_NS,
    METRICS,
    SYSTEM_ORDER,
    SYSTEMS,
    apply_paper_style,
    metrics_in_group,
    per_frame_csv,
    per_system_plot_dir,
    save_fig,
)

GROUPS_DEFAULT = ("conformational", "sasa", "hbond", "ma_cluster", "other")


def _load_per_frame(sys_name: str) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv(sys_name))
    df["Time_ns"] = pd.to_numeric(df["Time_ns"], errors="coerce")
    return df.dropna(subset=["Time_ns"]).sort_values("Time_ns").reset_index(drop=True)


def _highlight_eq_window(ax, t_max: float) -> None:
    ax.axvspan(t_max - EQ_WINDOW_NS, t_max, color="grey", alpha=0.15,
               label="eq window" if not ax.get_legend_handles_labels()[1] else None)


def _plot_group(df: pd.DataFrame, sys_name: str, group: str, color: str) -> Path | None:
    spec = SYSTEMS[sys_name]
    metrics = [m for m in metrics_in_group(group)
               if m.csv_col in df.columns and not (m.ma_only and not spec.has_ma)]
    if not metrics:
        return None

    n = len(metrics)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows),
                             squeeze=False, sharex=True)
    axes_flat = axes.flatten()
    t_max = float(df["Time_ns"].max())

    for ax, m in zip(axes_flat, metrics):
        y = pd.to_numeric(df[m.csv_col], errors="coerce")
        ax.plot(df["Time_ns"], y, color=color, lw=1.0, alpha=0.6, label="raw")
        # Prefer a Rolling-smoothed column if available
        roll_col = f"{m.csv_col}_Rolling"
        if roll_col in df.columns:
            yr = pd.to_numeric(df[roll_col], errors="coerce")
            ax.plot(df["Time_ns"], yr, color=color, lw=1.8, label="smoothed")
        _highlight_eq_window(ax, t_max)
        unit_str = f" ({m.unit})" if m.unit else ""
        ax.set_ylabel(f"{m.label}{unit_str}")
        ax.set_xlabel("Time (ns)")
        ax.grid(alpha=0.3)

    # Hide unused axes if metric count is odd
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"{spec.label.replace(chr(10), ' ')} — {group}", y=1.02, fontsize=12, color=color)
    fig.tight_layout()

    out = per_system_plot_dir(sys_name) / f"Figure_{sys_name}_{group}.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


def _plot_overview(df: pd.DataFrame, sys_name: str, color: str) -> Path:
    """Single multi-panel overview: pick one representative metric per group."""
    representative = [
        ("Rg_mean_A",                 "$R_g$ (Å)"),
        ("Ree_mean_A",                "$R_{ee}$ (Å)"),
        ("SASA_Global_A2",            "Total SASA (Å²)"),
        ("Hb_Inter_Strict",           "Inter-chain H-bonds"),
        ("Hb_PW_Strict",              "Polymer–water H-bonds"),
        ("MA_Inter_Cluster_Count",    "MA cluster count"),
    ]
    spec = SYSTEMS[sys_name]
    # Filter to columns present + applicable
    panels = []
    for col, label in representative:
        if col not in df.columns:
            continue
        if not spec.has_ma and col.startswith(("MA_", "SASA_MA", "Hb_MA")):
            continue
        panels.append((col, label))

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows),
                             squeeze=False, sharex=True)
    axes_flat = axes.flatten()
    t_max = float(df["Time_ns"].max())

    for ax, (col, label) in zip(axes_flat, panels):
        y = pd.to_numeric(df[col], errors="coerce")
        ax.plot(df["Time_ns"], y, color=color, lw=0.9, alpha=0.55)
        roll_col = f"{col}_Rolling"
        if roll_col in df.columns:
            yr = pd.to_numeric(df[roll_col], errors="coerce")
            ax.plot(df["Time_ns"], yr, color=color, lw=1.8)
        _highlight_eq_window(ax, t_max)
        ax.set_ylabel(label)
        ax.set_xlabel("Time (ns)")
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{spec.label.replace(chr(10), ' ')} — overview", y=1.02,
                 fontsize=12, color=color, fontweight="bold")
    fig.tight_layout()
    out = per_system_plot_dir(sys_name) / f"Figure_{sys_name}_overview.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEM_ORDER))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS_DEFAULT))
    ap.add_argument("--no-overview", action="store_true",
                    help="Skip the multi-group overview figure")
    args = ap.parse_args()

    apply_paper_style()
    n_plots = 0
    for sys_name in args.systems:
        if sys_name not in SYSTEMS:
            print(f"  [skip] unknown system: {sys_name}")
            continue
        spec = SYSTEMS[sys_name]
        df = _load_per_frame(sys_name)
        print(f"\n{sys_name}  (DS={spec.ds_pct}%, {len(df)} frames)")

        if not args.no_overview:
            p = _plot_overview(df, sys_name, spec.color)
            print(f"  → {p.relative_to(p.parents[2])}")
            n_plots += 1

        for group in args.groups:
            p = _plot_group(df, sys_name, group, spec.color)
            if p is not None:
                print(f"  → {p.relative_to(p.parents[2])}")
                n_plots += 1

    print(f"\nwrote {n_plots} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
