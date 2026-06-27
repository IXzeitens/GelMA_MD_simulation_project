"""Unified cross-system figure suite for the pre-crosslink characterization.

Single entry point that produces every *cross-system composite* ("組圖") figure
in the production plot suite. Consolidates what used to be five standalone
scripts (`plot_merged.py`, `plot_hbond.py`, `plot_timeseries.py`,
`plot_rmsf_residue.py`, `plot_ma_contact.py`) plus the convergence facet that
`time_block_analysis.py` used to draw. Output filenames are unchanged so the
paper / slides keep referencing the same PNG/PDF names.

Pick what to render with --figs (default = the full paper-relevant set):

  M-series (bar / RDF vs DS, reads pre_crosslink_summary.csv)
    M1          Rg / Ree / Lp / RMSD                 → Figure_M1_Conformational
    M2          4 H-bond + salt bridge + contacts    → Figure_M2_Interactions
    M3          MA–MA NN + MA–MA g(r)                → Figure_M3_MAstructure
    M3B         per-DS MA–MA g(r) detail             → Figure_M3b_RDF_detail
    M1P/M2P/M3P per-replica paired bars (need rep2)  → Figure_M*_paired

  H-bond decomposition (reads *_HBondGroups.csv)
    HB          functional-group stacked + Lys→MA    → Figure_HBondGroups
    HBPAIRS     donor–acceptor pair decomposition    → Figure_HBondPairs
    HBTARGET    one group's partners (--target)      → Figure_HBondPairs_<target>
    HBREP       rep1 vs rep2 Lys→MA (needs rep2)     → Figure_HBondGroups_RepCompare

  Time series (reads per-frame rep CSV)
    TSCONF      Rg/Ree/Lp/RMSD vs time               → Figure_TimeSeries_Conformational
    TSINTER     H-bonds/salt/MA-NN vs time           → Figure_TimeSeries_Interactions

  Other composites
    RMSF        per-residue RMSF, 3-panel mechanism  → Figure_RMSF_residue
    MACONTACT   MA–MA contact map + collapse/aggr.   → Figure_M5_MAcontact
    CONVERGENCE time-block facet (reads time_block_summary.csv) → Figure_TimeBlock_Convergence

Usage:
    python plot_merge.py                       # full default set
    python plot_merge.py --figs M1 M2 M3 M3B
    python plot_merge.py --figs HB HBPAIRS
    python plot_merge.py --figs HBTARGET --target carboxylate --scope intra
    python plot_merge.py --figs M1 M2 --rep rep1     # single-replica M-series
    python plot_merge.py --list                # list all tags and exit

Pipeline: run data_analysis.py → hbond_groups.py (×systems) → data_collect.py
→ time_block_analysis.py (for CONVERGENCE) BEFORE this script. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    DATA_ROOT,
    DATA_PREFIX,
    EQ_WINDOW_NS,
    METRICS,
    PLOT_ROOT,
    PRODUCTION,
    REP_TAG,
    SUMMARY_CSV,
    SYSTEM_ORDER,
    SYSTEMS,
    apply_paper_style,
    save_fig,
)

try:
    from scipy.signal import savgol_filter
except Exception:  # scipy optional — time-series smoothing degrades gracefully
    savgol_filter = None


# =============================================================================
# Shared styling / registries
# =============================================================================
# DS gradient — Chiu Fig 7 convention (light→dark blue for MA-bearing levels);
# Gelatin (DS=0%) gets a warm gold to flag it as the MA-free control. Single
# source of truth for every panel here (was duplicated across the old scripts).
DS_COLORS: dict[int, str] = {
    0:   "#e8a948",   # warm gold — Gelatin control (no MA chemistry)
    33:  "#a8d0f0",   # light blue
    67:  "#5a9fd4",   # medium blue
    100: "#1e3f6e",   # dark blue
}
PANEL_LETTERS = "abcdefghijklmnop"

# Single 7.5 wt% concentration in current production. When the high-conc scan
# (10/15/20%) returns, add grouped bars without rewriting the panel functions.
CONCENTRATIONS: list[tuple[str, str]] = [("7.5 wt%", "#5a9fd4")]

# Per-rep mode (set via --rep on the M-series): figures read one replica's
# per-frame CSVs and are saved with a "_<rep>" filename suffix. Empty = default
# ensemble behaviour (reads pre_crosslink_summary.csv, averages RDF over reps).
_OUT_SUFFIX: str = ""
_REP_FILTER: str | None = None


def _enable_chiu_rcparams() -> None:
    """Chiu house style overlay: Arial, full black box, on top of paper style."""
    mpl.rcParams.update({
        "font.family":       ["Arial", "DejaVu Sans"],
        "axes.spines.top":   True,
        "axes.spines.right": True,
    })


def chiu_box(ax) -> None:
    """Full black frame + inward ticks on all sides, no grid (Chiu house style)."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.8)
    ax.grid(False)


def _fig_path(stem: str) -> Path:
    """Output path for a figure stem, honoring the active per-rep suffix."""
    return PLOT_ROOT / f"{stem}{_OUT_SUFFIX}.png"


def _ds_label(ds_pct: int) -> str:
    return f"{ds_pct}%"


def detect_window(rep: str) -> str:
    """Analysis-window label for titles: the equilibrium window is the last 5 ns."""
    csv_path = DATA_ROOT / "Gelatin" / f"{DATA_PREFIX}_{rep}.csv"
    if not csv_path.exists():
        return "last 5 ns"
    try:
        t = pd.to_numeric(pd.read_csv(csv_path)["Time_ns"], errors="coerce").dropna()
        return f"{t.max() - 5.0:.0f}–{t.max():.0f} ns"
    except Exception:
        return "last 5 ns"


# =============================================================================
# Section A — M-series: bar / RDF figures vs DS
#   (ported from plot_merged.py; output filenames unchanged)
# =============================================================================
def _load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found — run data_collect.py first")
    return pd.read_csv(SUMMARY_CSV)


def _build_rep_summary(rep_label: str, window_ns: float = 5.0,
                       stationary_thresh: float = 5.0) -> pd.DataFrame:
    """Build a summary dataframe (same schema as pre_crosslink_summary.csv)
    from a SINGLE replica's per-frame CSVs, so the M-series figure functions can
    render single-replica figures without the cross-replica ensemble averaging
    baked into data_collect.py."""
    rows: list[dict] = []
    for sys_name, spec in SYSTEMS.items():
        csv_path = DATA_ROOT / sys_name / f"{DATA_PREFIX}_{rep_label}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["Time_ns"] = pd.to_numeric(df["Time_ns"], errors="coerce")
        df = df.dropna(subset=["Time_ns"]).sort_values("Time_ns")
        t_all = df["Time_ns"].to_numpy(dtype=float)
        t_max = float(t_all.max()) if len(t_all) else 0.0
        eq = df[df["Time_ns"] >= t_max - window_ns]
        n_eq = len(eq)
        for m in METRICS:
            col = m.csv_col
            base = {
                "system": sys_name, "ds_pct": spec.ds_pct, "has_ma": spec.has_ma,
                "metric": col, "label": m.label, "unit": m.unit, "group": m.group,
                "n_eq_frames": n_eq, "eq_t_start_ns": t_max - window_ns,
                "mean_stable": float("nan"), "sem_stable": float("nan"),
                "stable_t0_ns": float("nan"), "stable_t1_ns": float("nan"),
            }
            if (m.ma_only and not spec.has_ma) or col not in df.columns:
                base.update(mean_eq=float("nan"), sem_eq=float("nan"),
                            std_eq_across_frames=float("nan"),
                            drift_pct_per_window=float("nan"), stationary=True)
                rows.append(base)
                continue
            vals = pd.to_numeric(eq[col], errors="coerce").dropna()
            mean_eq = float(vals.mean()) if len(vals) else float("nan")
            std_across = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            sem_eq = (std_across / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            y_all = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            post = t_all >= 5.0
            if post.sum() >= 3 and not math.isnan(mean_eq) and abs(mean_eq) > 1e-12:
                slope = float(np.polyfit(t_all[post], y_all[post], 1)[0])
                drift = abs(slope) * window_ns / abs(mean_eq) * 100.0
            else:
                drift = float("nan")
            stationary = (not math.isnan(drift)) and drift < stationary_thresh
            base.update(mean_eq=mean_eq, sem_eq=sem_eq,
                        std_eq_across_frames=std_across,
                        drift_pct_per_window=drift, stationary=stationary)
            rows.append(base)
    return pd.DataFrame(rows)


def _discover_replicas(sys_name: str) -> list[tuple[str, Path]]:
    """Return [(rep_label, csv_path)] for every replica CSV available."""
    import re
    csv_pat = re.compile(rf"^{re.escape(DATA_PREFIX)}_(rep\d+)\.csv$")
    out: list[tuple[str, Path]] = []
    for p in sorted((DATA_ROOT / sys_name).glob(f"{DATA_PREFIX}_rep*.csv")):
        m = csv_pat.match(p.name)
        if m:
            out.append((m.group(1), p))
    return out


def _matched_window_from_reps(rep_csvs: list[Path],
                              window_ns: float = 5.0) -> tuple[float, float] | None:
    """Matched last-N-ns window that all replicas cover (hi = min t_max)."""
    if not rep_csvs:
        return None
    tmaxes: list[float] = []
    for p in rep_csvs:
        df = pd.read_csv(p, usecols=["Time_ns"])
        tmaxes.append(float(df["Time_ns"].max()))
    hi = min(tmaxes)
    lo = max(0.0, hi - window_ns)
    return lo, hi


def _paired_bar_panel(ax, metric: str, panel_letter: str | None = None,
                      subtitle: str | None = None, window_ns: float = 5.0,
                      ma_only: bool = False, show_legend: bool = False,
                      show_window: bool = False):
    """Per-replica paired bars per DS, computed from raw per-frame CSVs."""
    ds_list = sorted({spec.ds_pct for spec in SYSTEMS.values()
                      if (spec.has_ma or not ma_only)})
    rows: list[dict] = []
    for ds in ds_list:
        sys_name = next((n for n, s in SYSTEMS.items() if s.ds_pct == ds), None)
        if sys_name is None:
            continue
        spec = SYSTEMS[sys_name]
        if ma_only and not spec.has_ma:
            continue
        reps = _discover_replicas(sys_name)
        if not reps:
            continue
        win = _matched_window_from_reps([p for _, p in reps], window_ns)
        if win is None:
            continue
        lo, hi = win
        for rep_label, csv_path in reps:
            df = pd.read_csv(csv_path)
            if metric not in df.columns:
                continue
            sub = df[(df["Time_ns"] >= lo) & (df["Time_ns"] <= hi)][metric].dropna()
            if sub.empty:
                continue
            rows.append({"ds": ds, "rep": rep_label,
                         "mean": float(sub.mean()),
                         "sem": float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0})
    if not rows:
        ax.text(0.5, 0.5, f"no data for\n{metric}", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_xticks([]); ax.set_yticks([])
        return

    df_rows = pd.DataFrame(rows)
    rep_labels = sorted(df_rows["rep"].unique())
    n_rep = len(rep_labels)
    ds_present = sorted(df_rows["ds"].unique())
    x = np.arange(len(ds_present))
    width = min(0.8 / max(n_rep, 1), 0.38)

    for ri, rep in enumerate(rep_labels):
        offsets = x + (ri - (n_rep - 1) / 2) * width
        means, sems = [], []
        for d in ds_present:
            row = df_rows[(df_rows["ds"] == d) & (df_rows["rep"] == rep)]
            means.append(float(row["mean"].iloc[0]) if len(row) else float("nan"))
            sems.append(float(row["sem"].iloc[0]) if len(row) else 0.0)
        base_colors = [DS_COLORS.get(d, "#999") for d in ds_present]
        if rep == "rep1":
            fills, hatches, alphas = base_colors, [None] * len(ds_present), [1.0] * len(ds_present)
        else:
            fills, hatches, alphas = base_colors, ["///"] * len(ds_present), [0.85] * len(ds_present)
        for j, (xj, m, s, fill, hatch, alpha) in enumerate(
            zip(offsets, means, sems, fills, hatches, alphas)
        ):
            ax.bar(xj, m, width, yerr=s, color=fill, edgecolor="black",
                   lw=0.7, hatch=hatch, alpha=alpha,
                   error_kw=dict(elinewidth=0.9, capsize=3, ecolor="black"),
                   label=f"{rep}" if j == 0 else None)

    ens_means: list[float] = []
    ens_sems: list[float] = []
    for d in ds_present:
        vals = df_rows[df_rows["ds"] == d]["mean"].tolist()
        if len(vals) >= 2:
            ens_means.append(float(np.mean(vals)))
            ens_sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        else:
            ens_means.append(float("nan"))
            ens_sems.append(0.0)
    valid = ~np.isnan(ens_means)
    if any(valid):
        ax.errorbar(x[valid], np.array(ens_means)[valid], yerr=np.array(ens_sems)[valid],
                    fmt="ko", markersize=5, capsize=5, elinewidth=1.2,
                    label=f"N={n_rep} mean ± SEM", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(d) for d in ds_present])
    ax.set_xlabel("Degree of substitution")
    win = _matched_window_from_reps(
        [p for ds in ds_present
         for nm, sp in SYSTEMS.items() if sp.ds_pct == ds
         for _, p in _discover_replicas(nm)], window_ns)
    win_str = f"matched {win[0]:.0f}–{win[1]:.0f} ns" if win else ""

    row_meta = (_load_summary()[lambda d: d["metric"] == metric].iloc[0:1])
    if len(row_meta):
        label = row_meta["label"].iloc[0]
        unit = row_meta["unit"].iloc[0]
        ax.set_ylabel(f"{label}" + (f" ({unit})" if unit else ""))

    header = (f"({panel_letter}) " if panel_letter else "") + (subtitle or "")
    if header:
        ax.set_title(header, loc="left", fontweight="bold", fontsize=11, pad=8)
    if show_window and win_str:
        ax.text(0.99, 0.99, win_str, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#666",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                          boxstyle="round,pad=0.2"))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(False)  # no background gridlines
    for s in ("top", "right"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=False)
    return n_rep, win_str


def _bar_panel(ax, df: pd.DataFrame, metric: str, panel_letter: str | None = None,
               subtitle: str | None = None, show_legend: bool = False) -> None:
    """One Chiu-style bar panel: x = DS%, one bar per system, DS-gradient fill.

    Always plots mean_eq (last 5 ns) so every system shares the SAME sampling
    window. Non-stationary metrics get diagonal hatching but their bar value is
    NOT swapped to mean_stable (different window = different statistic)."""
    row_map = {r["system"]: r for _, r in df[df["metric"] == metric].iterrows()}
    systems_present = [s for s in SYSTEM_ORDER
                       if s in row_map and not pd.isna(row_map[s]["mean_eq"])]
    if not systems_present:
        ax.text(0.5, 0.5, f"no data for\n{metric}", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return

    ds_vals = [SYSTEMS[s].ds_pct for s in systems_present]
    x = np.arange(len(systems_present))
    means, sems, hatches = [], [], []
    for s in systems_present:
        r = row_map[s]
        is_stat = bool(r.get("stationary", True))
        means.append(float(r["mean_eq"]))
        sems.append(float(r["sem_eq"]) if not pd.isna(r["sem_eq"]) else 0.0)
        hatches.append(None if is_stat else "///")

    bar_colors = [DS_COLORS.get(d, "#999999") for d in ds_vals]
    bars = ax.bar(x, means, width=0.65, yerr=sems, color=bar_colors,
                  edgecolor="black", lw=0.8,
                  error_kw=dict(elinewidth=1.2, capsize=5, ecolor="black"))
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h); bar.set_edgecolor("black")

    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(d) for d in ds_vals])
    ax.set_xlabel("Degree of substitution", fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)
    row0 = row_map[systems_present[0]]
    unit = f" ({row0['unit']})" if row0["unit"] else ""
    ax.set_ylabel(f"{row0['label']}{unit}", fontsize=13.5)
    header = (f"({panel_letter}) " if panel_letter else "") + (subtitle or "")
    if header:
        ax.set_title(header, loc="left", fontweight="bold", fontsize=15, pad=8)
    _ = show_legend


def _load_ma_rdf_ensemble(sys_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read all replica MA-RDF CSVs for sys_name and return (r, g_avg)."""
    if _REP_FILTER is not None:
        csvs = [DATA_ROOT / sys_name / f"{DATA_PREFIX}_{_REP_FILTER}_MA_ensemble_RDF.csv"]
        csvs = [p for p in csvs if p.exists()]
    else:
        csvs = sorted((DATA_ROOT / sys_name).glob(f"{DATA_PREFIX}_rep*_MA_ensemble_RDF.csv"))
    if not csvs:
        return None
    accum, bins, n = None, None, 0
    for p in csvs:
        df = pd.read_csv(p)
        if "r_A" not in df.columns or "g_r" not in df.columns:
            continue
        if bins is None:
            bins = df["r_A"].to_numpy()
        g = df["g_r"].to_numpy()
        accum = g.copy() if accum is None else accum + g
        n += 1
    if n == 0 or bins is None:
        return None
    return bins, accum / n


def _load_ma_rdf_one_rep(sys_name: str, rep: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a single replica's MA-RDF CSV → (r, g) (no cross-rep averaging)."""
    p = DATA_ROOT / sys_name / f"{DATA_PREFIX}_{rep}_MA_ensemble_RDF.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "r_A" not in df.columns or "g_r" not in df.columns:
        return None
    return df["r_A"].to_numpy(), df["g_r"].to_numpy()


def _rdf_panel(ax, panel_letter: str | None = None, subtitle: str | None = None,
               r_max_A: float = 15.0) -> None:
    """Line-plot panel: MA–MA g(r), one line per MA-bearing system."""
    plotted = False
    for ds in (33, 67, 100):
        sys_name = next((n for n, spec in SYSTEMS.items() if spec.ds_pct == ds), None)
        if sys_name is None:
            continue
        rdf = _load_ma_rdf_ensemble(sys_name)
        if rdf is None:
            continue
        r, g = rdf
        mask = r <= r_max_A
        ax.plot(r[mask], g[mask], color=DS_COLORS.get(ds, "#999999"),
                lw=1.6, label=f"DS = {ds}%")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no MA RDF data", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return

    ax.axhline(1.0, color="#888888", lw=0.7, ls=":")
    for rcut, _label in ((5.0, "5 Å (pre-network)"), (7.0, "7 Å (Polymatic)")):
        ax.axvline(rcut, color="#444444", lw=0.7, ls="--")
        ax.text(rcut, ax.get_ylim()[1] * 0.95, f" {rcut:.0f} Å",
                fontsize=7, color="#444444", va="top", ha="left")
    ax.set_xlim(0, r_max_A)
    ax.set_xlabel("MA–MA distance r (Å)")
    ax.set_ylabel("g(r)")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(False)  # no background gridlines
    for s in ("top", "right"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)
    header = (f"({panel_letter}) " if panel_letter else "") + (subtitle or "")
    if header:
        ax.set_title(header, loc="left", fontweight="bold", fontsize=11, pad=8)


def _add_figure_legend(fig, df: pd.DataFrame) -> None:
    """One bottom legend decoding the DS gradient + hatching convention."""
    handles = [Patch(facecolor=DS_COLORS[d], edgecolor="black", linewidth=0.6,
                     label=f"DS = {d}%") for d in (0, 33, 67, 100)]
    any_nonstat = bool(((~df["stationary"].astype(bool)) & df["mean_eq"].notna()).any())
    if any_nonstat:
        handles.append(Patch(
            facecolor="white", edgecolor="black", hatch="///", linewidth=0.6,
            label="non-stationary (drift > 5% per 5 ns) —\n          bar value still uses last 5 ns for comparability"))
    handles.append(Patch(facecolor="none", edgecolor="none",
                         label=f"Concentration: {CONCENTRATIONS[0][0]}"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, 0.045), fontsize=12, frameon=False,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.6)


def _multi_panel(metrics_with_subtitles: list[tuple[str, str]], df: pd.DataFrame,
                 title: str, out: Path, ncols: int = 2,
                 panel_w: float = 5.0, panel_h: float = 4.2) -> None:
    n = len(metrics_with_subtitles)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for i, (ax, (metric, subtitle)) in enumerate(zip(axes_flat, metrics_with_subtitles)):
        _bar_panel(ax, df, metric, panel_letter=PANEL_LETTERS[i], subtitle=subtitle)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0.12, 1, 0.985))
    _add_figure_legend(fig, df)
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


def figure_M1_conformational(df: pd.DataFrame) -> None:
    _multi_panel(
        [("Rg_mean_A", "Radius of gyration"),
         ("Ree_mean_A", "End-to-end distance"),
         ("Persistence_Length_mean_A", "Persistence length"),
         ("RMSD_mean_A", "Backbone RMSD")],
        df=df, title="Chain conformation versus degree of substitution",
        out=_fig_path("Figure_M1_Conformational"), ncols=2)


def figure_M2_interactions(df: pd.DataFrame) -> None:
    # 5-panel set (a–e). The old (f) "Inter-residue contacts"
    # (Residue_Min_Contact_Total) was dropped per request 2026-06-22.
    _multi_panel(
        [("Hb_Intra_Strict", "Intra-chain H-bonds"),
         ("Hb_Inter_Strict", "Inter-chain H-bonds"),
         ("Hb_PW_Strict", "Polymer–water H-bonds"),
         ("Hb_MA_Wat_Total", "MA–water H-bonds"),
         ("Salt_Bridges", "Salt bridges")],
        df=df, title="Interaction network versus degree of substitution",
        out=_fig_path("Figure_M2_Interactions"), ncols=3)


def figure_M2_chiu(df: pd.DataFrame) -> None:
    """Chiu-comparable M2: the 4 H-bond panels only.

    Drops Salt Bridges and Inter-Residue Contacts — both are GelMA-specific
    (gellan gum has no Lys/Arg charged network, so Chiu's Fig 4 has no salt-
    bridge or charged-contact panel). This variant maps 1:1 onto Chiu Fig 4.
    """
    _multi_panel(
        [("Hb_Intra_Strict", "Intramolecular H-Bonds"),
         ("Hb_Inter_Strict", "Intermolecular H-Bonds"),
         ("Hb_PW_Strict", "Polymer–Water H-Bonds"),
         ("Hb_MA_Wat_Total", "MA–Water H-Bonds")],
        df=df, title="Hydrogen-Bond Interactions",
        out=_fig_path("Figure_M2_Chiu"), ncols=2)


def figure_M3_MAstructure(df: pd.DataFrame) -> None:
    """2-panel MA structure figure: MA–MA NN bar + MA–MA g(r) line."""
    out = _fig_path("Figure_M3_MAstructure")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), squeeze=False)
    axes_flat = axes.flatten()
    _bar_panel(axes_flat[0], df, "MA_Inter_MinDist_NN_A",
               panel_letter="a", subtitle="MA–MA Nearest Neighbour")
    _rdf_panel(axes_flat[1], panel_letter="b", subtitle="MA–MA Radial Distribution")
    fig.suptitle("MA-Group Geometry & Clustering",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0.12, 1, 0.985))
    _add_figure_legend(fig, df)
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


def _read_box_edge_A(sys_name: str) -> float | None:
    """Equilibrated cubic box edge (Å) from the latest .xsc, else config.json."""
    sys_root = DATA_ROOT.parent / sys_name
    xsc_files = sorted(sys_root.glob("Output/system_npt_part*.xsc"))
    if xsc_files:
        try:
            with xsc_files[-1].open() as f:
                last = [ln for ln in f if not ln.startswith("#")][-1]
            return float(last.split()[1])
        except Exception:
            pass
    cfg = sys_root / "config.json"
    if cfg.exists():
        import json
        with cfg.open() as f:
            return float(json.load(f).get("calculated_box_L_Angstrom", float("nan")))
    return None


def figure_M3b_RDF_detail(df: pd.DataFrame) -> None:
    """Per-DS MA–MA radial distribution panels with N(7 Å) inset.

    rep1 and rep2 are emitted as SEPARATE figures (one clean single curve per
    panel) instead of overlaid — two wiggly curves on one axis are unreadable.
    Output files: Figure_M3b_RDF_detail_rep1.png / _rep2.png.
    """
    R_MAX, POLYMATIC_R, N_CHAINS = 10.0, 7.0, 12
    n_ma_per_chain_by_ds = {33: 1, 67: 2, 100: 3}
    ma_systems = sorted(((spec.ds_pct, name) for name, spec in SYSTEMS.items()
                         if spec.has_ma), key=lambda x: x[0])
    n_panels = len(ma_systems)
    if n_panels == 0:
        return

    # running coordination number N(7 Å) for one (r, g) curve
    def _coord_N7(r, g, L, n_ma_per):
        if not (L and n_ma_per):
            return None
        rho = (N_CHAINS - 1) * n_ma_per / (L ** 3)
        integrand = 4.0 * np.pi * r ** 2 * g
        Nr = rho * np.array([np.trapezoid(integrand[: i + 1], r[: i + 1])
                             for i in range(len(r))])
        return float(Nr[int(np.searchsorted(r, POLYMATIC_R))])

    # one separate figure per replica
    for rep in ("rep1", "rep2"):
        # skip a replica that has no RDF data at all
        if not any(_load_ma_rdf_one_rep(name, rep) is not None
                   for _, name in ma_systems):
            continue
        fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.0),
                                 squeeze=False, sharey=False)
        for ax, (ds, sys_name) in zip(axes.flatten(), ma_systems):
            rdf = _load_ma_rdf_one_rep(sys_name, rep)
            if rdf is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            r, g = rdf
            mask = r <= R_MAX
            colour = DS_COLORS.get(ds, "#5a9fd4")
            ax.plot(r[mask], g[mask], color=colour, lw=1.8)
            ax.axvline(POLYMATIC_R, color="#bbb", lw=0.6, ls=":", zorder=0)
            ax.set_xlim(0, R_MAX)
            ax.set_xlabel("distance (Å)")
            ax.set_ylabel("RDF")
            ax.set_title(f"{sys_name} (DS = {ds}%)", loc="center",
                         fontsize=11, fontweight="bold", pad=8)
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.8)
            ax.tick_params(direction="out", length=4, width=0.6)
            n7 = _coord_N7(r, g, _read_box_edge_A(sys_name),
                           n_ma_per_chain_by_ds.get(ds))
            if n7 is not None:
                ax.text(0.97, 0.97, f"$N(7\\,\\mathrm{{\\AA}})$ = {n7:.2f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color="#333",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                                  boxstyle="round,pad=0.3"))
        fig.suptitle(f"MA–MA Radial Distribution — {rep}",
                     fontsize=13, fontweight="bold", y=1.00)
        fig.text(0.995, 0.005, "vinyl C8–C8", ha="right", va="bottom",
                 fontsize=7.5, color="#999")
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        out = _fig_path(f"Figure_M3b_RDF_detail_{rep}")
        save_fig(fig, out)
        plt.close(fig)
        print(f"  → {out.relative_to(out.parents[1])}")


def _paired_multi_panel(metrics_with_subtitles: list[tuple[str, str]],
                        title: str, out: Path, ncols: int = 3,
                        panel_w: float = 5.0, panel_h: float = 4.2) -> None:
    n = len(metrics_with_subtitles)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    n_rep_max, win_str = 1, ""
    for i, (ax, (metric, subtitle)) in enumerate(zip(axes_flat, metrics_with_subtitles)):
        res = _paired_bar_panel(ax, metric, panel_letter=PANEL_LETTERS[i],
                                subtitle=subtitle,
                                ma_only=(metric in {"SASA_MA_A2", "Hb_MA_Wat_Total",
                                                    "MA_Inter_MinDist_NN_A"}))
        if res:
            n_rep_max = max(n_rep_max, res[0])
            win_str = win_str or res[1]
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    handles = [Patch(facecolor="#c7c7c7", edgecolor="black", lw=0.7, label="rep1")]
    if n_rep_max >= 2:
        handles.append(Patch(facecolor="#c7c7c7", edgecolor="black", lw=0.7,
                             hatch="///", label="rep2"))
        handles.append(Line2D([0], [0], color="k", marker="o", lw=0, markersize=5,
                              label=f"N={n_rep_max} mean ± SEM"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.015), frameon=False, fontsize=9,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.6)
    if win_str:
        sup = title.replace("(matched window)", f"({win_str})")
        if sup == title:
            sup = f"{title} — {win_str}"
    else:
        sup = title
    fig.suptitle(sup, fontsize=14, fontweight="bold", y=1.005)
    bottom = 0.12 if nrows == 1 else 0.06
    fig.tight_layout(rect=(0, bottom, 1, 0.985))
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


def figure_M1_paired(df: pd.DataFrame) -> None:
    _paired_multi_panel(
        [("Rg_mean_A", "Radius of Gyration"),
         ("Ree_mean_A", "End-to-End Distance"),
         ("Persistence_Length_mean_A", "Persistence Length"),
         ("RMSD_mean_A", "Backbone RMSD")],
        title="Conformational Properties — rep1 vs rep2 (matched window)",
        out=PLOT_ROOT / "Figure_M1_paired.png", ncols=2)


def figure_M2_paired(df: pd.DataFrame) -> None:
    """Inter/intra interaction metrics, rep1 vs rep2 — 5 panels.

    Custom layout (not the generic grid): panels (a)(b)(c) fill the top row and
    (d)(e) are *centred* on the bottom row.  Implemented on a 12-column GridSpec
    so each 4-column panel can be offset to sit centred (d -> cols 2–6,
    e -> cols 6–10, leaving equal side margins).

    Panel (f) "Inter-Residue Contacts" was intentionally dropped: it is an
    undifferentiated all-atom residue-contact count dominated by the initial
    packmol packing (rep1/rep2 diverge ~2x at DS100) and carries no mechanistic
    meaning — Salt Bridges (e) is its chemistry-specific replacement.  The
    figure is also enlarged for slide readability.  Output filename unchanged.
    """
    # (metric key, panel subtitle, ma_only = MA-bearing systems only)
    metrics = [
        ("Hb_Intra_Strict", "Intramolecular H-Bonds", False),
        ("Hb_Inter_Strict", "Intermolecular H-Bonds", False),
        ("Hb_PW_Strict",    "Polymer–Water H-Bonds",  False),
        ("Hb_MA_Wat_Total", "MA–Water H-Bonds",       True),
        ("Salt_Bridges",    "Salt Bridges",           False),
    ]
    fig = plt.figure(figsize=(18.0, 10.5))            # enlarged for slides
    gs = fig.add_gridspec(2, 12, hspace=0.34, wspace=1.7)
    # top row a/b/c span the full width; bottom row d/e centred
    slots = [gs[0, 0:4], gs[0, 4:8], gs[0, 8:12], gs[1, 2:6], gs[1, 6:10]]

    n_rep_max, win_str = 1, ""
    for i, ((metric, subtitle, ma_only), slot) in enumerate(zip(metrics, slots)):
        ax = fig.add_subplot(slot)
        res = _paired_bar_panel(ax, metric, panel_letter=PANEL_LETTERS[i],
                                subtitle=subtitle, ma_only=ma_only)
        if res:                                       # res = (n_rep, window_str)
            n_rep_max = max(n_rep_max, res[0])
            win_str = win_str or res[1]

    # shared legend: rep1 solid / rep2 hatched / N=2 mean ± SEM
    handles = [Patch(facecolor="#c7c7c7", edgecolor="black", lw=0.7, label="rep1")]
    if n_rep_max >= 2:
        handles.append(Patch(facecolor="#c7c7c7", edgecolor="black", lw=0.7,
                             hatch="///", label="rep2"))
        handles.append(Line2D([0], [0], color="k", marker="o", lw=0, markersize=5,
                              label=f"N={n_rep_max} mean ± SEM"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=11,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.6)

    base = "Inter- and Intra-molecular Interactions — rep1 vs rep2"
    sup = f"{base} ({win_str})" if win_str else base
    fig.suptitle(sup, fontsize=16, fontweight="bold", y=0.99)
    # explicit margins (GridSpec spanning is incompatible with tight_layout)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08)
    save_fig(fig, PLOT_ROOT / "Figure_M2_paired.png")
    plt.close(fig)
    print("  → Figure_M2_paired.png")


def figure_M3_paired(df: pd.DataFrame) -> None:
    _paired_multi_panel(
        [("MA_Inter_MinDist_NN_A", "MA–MA Nearest Neighbour")],
        title="MA-Group Geometry — rep1 vs rep2 (matched window)",
        out=PLOT_ROOT / "Figure_M3_paired.png", ncols=1, panel_w=6.0)


# =============================================================================
# Section B — H-bond decomposition by functional group
#   (ported from plot_hbond.py; output filenames unchanged)
# =============================================================================
HB_SYS = [("Gelatin", 0, "DS = 0%"), ("Gel1MA", 33, "DS = 33%"),
          ("Gel2MA", 67, "DS = 67%"), ("Gel3MA", 100, "DS = 100%")]

# alias, csv_key, display, stack_color
GROUPS = [
    ("backbone",    "backbone(N-H/C=O)",   "backbone",       "#9e9e9e"),
    ("carboxylate", "carboxylate(D/E)",    "carboxylate",    "#1b9e77"),
    ("guanidinium", "guanidinium(Arg)",    "guanidinium",    "#7570b3"),
    ("hydroxyl",    "hydroxyl(S/T/Y/Hyp)", "hydroxyl",       "#66a61e"),
    ("amine",       "amine(Lys)",          "amine (Lys)",    "#2b6cb0"),
    ("MA_group",    "MA_group(LMA)",       "MA group (LMA)", "#d62728"),
    ("amide_sc",    "amide_sc(Asn/Gln)",   "amide (N/Q)",    "#e7ba52"),
    ("imidazole",   "imidazole(His)",      "imidazole (H)",  "#ce6dbd"),
    ("other",       "other",               "other",          "#d9d9d9"),
]
ALIAS_TO_CSV = {a: c for a, c, _, _ in GROUPS}
CSV_TO_DISPLAY = {c: d for _, c, d, _ in GROUPS}
PAIR_PRIORITY = {c: i for i, (_, c, _, _) in enumerate(GROUPS)}
SCOPES = [("intra", "intramolecular"), ("inter", "intermolecular")]


def _hb_rows(rep: str, sysn: str):
    f = DATA_ROOT / sysn / f"{DATA_PREFIX}_{rep}_HBondGroups.csv"
    if not f.exists():
        return
    for row in csv.DictReader(open(f, encoding="utf-8")):
        d, a, sc = row["donor_group"], row["acceptor_group"], row["scope"]
        if d == "water" or a == "water":
            continue
        yield d, a, sc, float(row["avg_per_frame"])


def participation(rep: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sysn, _, _ in HB_SYS:
        part: dict[str, float] = defaultdict(float)
        for d, a, _sc, v in _hb_rows(rep, sysn):
            part[d] += v
            part[a] += v
        out[sysn] = part
    return out


def group_participation(rep: str, csv_key: str) -> np.ndarray:
    p = participation(rep)
    return np.array([p[s].get(csv_key, 0.0) for s, _, _ in HB_SYS])


def pair_table(rep: str) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = {}
    for sysn, _, _ in HB_SYS:
        acc: dict[tuple[str, str], float] = defaultdict(float)
        for d, a, sc, v in _hb_rows(rep, sysn):
            if sc not in ("intra", "inter") or d not in PAIR_PRIORITY or a not in PAIR_PRIORITY:
                continue
            g = sorted((d, a), key=lambda k: PAIR_PRIORITY[k])
            label = f"{CSV_TO_DISPLAY[g[0]]}–{CSV_TO_DISPLAY[g[1]]}"
            acc[(label, sc)] += v
        out[sysn] = acc
    return out


def target_table(rep: str, target_csv: str) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = {}
    for sysn, _, _ in HB_SYS:
        acc: dict[tuple[str, str], float] = defaultdict(float)
        for d, a, sc, v in _hb_rows(rep, sysn):
            if sc not in ("intra", "inter") or target_csv not in (d, a):
                continue
            partner = a if d == target_csv else d
            acc[(partner, sc)] += v
        out[sysn] = acc
    return out


def figure_hb_groups(rep: str) -> Path:
    part = participation(rep)
    ds = [d for _, d, _ in HB_SYS]
    x = np.arange(len(HB_SYS))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    bottom = np.zeros(len(HB_SYS))
    for _alias, csv_key, display, color in GROUPS:
        vals = np.array([part[s].get(csv_key, 0.0) for s, _, _ in HB_SYS])
        if vals.sum() < 1e-6:
            continue
        axA.bar(x, vals, width=0.62, bottom=bottom, color=color,
                edgecolor="white", lw=0.6, label=display)
        bottom += vals
    axA.set_xticks(x)
    axA.set_xticklabels([f"{d}%" for d in ds])
    axA.set_xlim(-0.6, len(HB_SYS) - 0.4)
    axA.set_ylim(0, bottom.max() * 1.32)
    axA.set_xlabel("Degree of substitution")
    axA.set_ylabel("H-bond participation per frame\n(donor + acceptor, water excluded)")
    axA.set_title("(a) Functional-group H-bond contribution", loc="left",
                  fontweight="bold", fontsize=11, pad=8)
    chiu_box(axA)
    axA.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2,
               fontsize=7.5, frameon=False, columnspacing=1.1,
               handlelength=1.1, handletextpad=0.5, labelspacing=0.35)

    lys = group_participation(rep, "amine(Lys)")
    ma = group_participation(rep, "MA_group(LMA)")
    dsf = np.array(ds, dtype=float)
    ref = lys[0] * (1.0 - dsf / 100.0)
    axB.plot(dsf, ref, "--", color="#2b6cb0", lw=1.2, alpha=0.6, zorder=1,
             label=r"Lys$_{0}\,(1-\mathrm{DS})$")
    axB.plot(dsf, lys, "o-", color="#2b6cb0", lw=1.8, ms=8, zorder=3, label="amine (Lys)")
    axB.plot(dsf, ma, "s-", color="#d62728", lw=1.8, ms=8, zorder=3, label="MA group (LMA)")
    ymax = max(lys.max(), ma.max(), 1.0)
    for xi, yi in zip(dsf, lys):
        dy = -13 if yi > 0.85 * ymax else 9
        axB.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=8, color="#2b6cb0")
    for xi, yi in zip(dsf, ma):
        dy = 9 if yi < 0.15 * ymax else -14
        axB.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=8, color="#d62728")
    axB.set_xticks(dsf)
    axB.set_xticklabels([f"{int(d)}%" for d in dsf])
    axB.set_xlim(-8, 108)
    axB.set_ylim(-6, ymax * 1.18)
    axB.set_xlabel("Degree of substitution")
    axB.set_ylabel("H-bond participation per frame")
    axB.set_title("(b) Methacrylation signature: Lys → MA", loc="left",
                  fontweight="bold", fontsize=11, pad=8)
    chiu_box(axB)
    axB.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), fontsize=8.5,
               frameon=False, labelspacing=0.35)

    fig.suptitle("H-Bonds by Functional Group",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.995, 0.005, f"7.5 wt%, {rep}, {detect_window(rep)}",
             ha="right", va="bottom", fontsize=7.5, color="#999")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    out = PLOT_ROOT / "Figure_HBondGroups.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


def _ds_grouped_panels(tab: dict[str, dict[tuple[str, str], float]],
                       cols_csv: list[str], col_labels: list[str],
                       scopes: list[tuple[str, str]], xlabel: str,
                       title: str, out: Path, legend_in_first: bool = True,
                       provenance: str | None = None) -> Path:
    n_ds = len(HB_SYS)
    x = np.arange(len(cols_csv))
    bar_w = 0.85 / n_ds
    fig, axes = plt.subplots(len(scopes), 1,
                             figsize=(max(8.6, 1.08 * len(cols_csv)),
                                      3.8 * len(scopes) + 0.6),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for ai, (sc, sclabel) in enumerate(scopes):
        ax = axes[ai]
        ymax = 0.0
        for bi, (sysn, ds, _) in enumerate(HB_SYS):
            vals = np.array([tab[sysn].get((c, sc), 0.0) for c in cols_csv])
            offset = (bi - (n_ds - 1) / 2.0) * bar_w
            ax.bar(x + offset, vals, width=bar_w, color=DS_COLORS.get(ds, "#888"),
                   edgecolor="black", lw=0.4,
                   label=f"DS = {ds}%" if (ai == 0 and legend_in_first) else None)
            ymax = max(ymax, float(vals.max(initial=0.0)))
        ax.set_ylabel(f"Number of\n{sclabel} H-bonds")
        ax.set_xlim(-0.55, len(cols_csv) - 0.45)
        ax.set_ylim(0, max(ymax * 1.22, 0.5))
        ax.text(0.012, 0.92, f"({chr(ord('a') + ai)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold")
        ax.text(0.985, 0.92, sclabel, transform=ax.transAxes,
                ha="right", va="top", fontsize=10.5, fontweight="bold")
        chiu_box(ax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(col_labels, fontsize=9)
    axes[-1].set_xlabel(xlabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=n_ds, frameon=False, fontsize=10, columnspacing=1.5,
               handlelength=1.2, handletextpad=0.5)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    if provenance:
        fig.text(0.995, 0.005, provenance, ha="right", va="bottom",
                 fontsize=7.5, color="#999")
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    save_fig(fig, out)
    plt.close(fig)
    return out


def figure_hb_pairs(rep: str, top: int, scope: str) -> Path:
    tab = pair_table(rep)
    scopes = SCOPES if scope == "both" else [(s, n) for s, n in SCOPES if s == scope]
    totals: dict[str, float] = defaultdict(float)
    for sysn, _, _ in HB_SYS:
        for (pair, sc), v in tab[sysn].items():
            if sc in {s for s, _ in scopes}:
                totals[pair] += v
    pairs = [p for p, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:top]
    if not pairs:
        sys.exit("[ERROR] no pairs — run hbond_groups.py for each system first.")
    labels = [p.replace("–", "–\n") for p in pairs]
    return _ds_grouped_panels(
        tab, pairs, labels, scopes,
        xlabel="Hydrogen-bond donor–acceptor pair (functional group)",
        title="H-Bond Pairs by Functional Group",
        out=PLOT_ROOT / f"Figure_HBondPairs_{rep}.png",
        provenance=f"7.5 wt%, {rep}, {detect_window(rep)}")


def _available_hb_reps() -> list[str]:
    """Replicas that have HBondGroups CSVs for at least one system."""
    reps = [r for r in ("rep1", "rep2", "rep3")
            if any((DATA_ROOT / s / f"{DATA_PREFIX}_{r}_HBondGroups.csv").exists()
                   for s, _, _ in HB_SYS)]
    return reps or ["rep1"]


def figure_hb_pairs_allreps(top: int, scope: str) -> None:
    """Render the donor–acceptor pair decomposition for EVERY replica with data,
    each to its own rep-suffixed file (Figure_HBondPairs_<rep>.png)."""
    for rep in _available_hb_reps():
        out = figure_hb_pairs(rep, top, scope)
        print(f"  → {out.relative_to(out.parents[1])}")


def figure_hb_target(rep: str, target: str, top: int, scope: str) -> Path:
    target_csv = ALIAS_TO_CSV[target]
    target_disp = CSV_TO_DISPLAY[target_csv]
    tab = target_table(rep, target_csv)
    scopes = SCOPES if scope == "both" else [(s, n) for s, n in SCOPES if s == scope]
    totals: dict[str, float] = defaultdict(float)
    for sysn, _, _ in HB_SYS:
        for (partner, sc), v in tab[sysn].items():
            if sc in {s for s, _ in scopes}:
                totals[partner] += v
    partners = [p for p, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:top]
    if not partners:
        sys.exit(f"[ERROR] no partners for target={target}.")
    labels = [f"{CSV_TO_DISPLAY.get(p, p)}\n(self)" if p == target_csv
              else CSV_TO_DISPLAY.get(p, p) for p in partners]
    return _ds_grouped_panels(
        tab, partners, labels, scopes,
        xlabel=f"Partner functional group (paired with {target_disp})",
        title=f"{target_disp} H-Bond Partners",
        out=PLOT_ROOT / f"Figure_HBondPairs_{target}.png",
        provenance=f"7.5 wt%, {rep}, {detect_window(rep)}")


def figure_hb_repcompare() -> Path | None:
    has_rep2 = any((DATA_ROOT / s / f"{DATA_PREFIX}_rep2_HBondGroups.csv").exists()
                   for s, _, _ in HB_SYS)
    if not has_rep2:
        print("[skip] HBREP: no rep2 HBondGroups.csv found (N=1). "
              "Run hbond_groups.py --rep rep2 once rep2 trajectories exist.")
        return None
    ds = np.array([d for _, d, _ in HB_SYS], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for rep, ls, mk in (("rep1", "-", "o"), ("rep2", "--", "D")):
        lys = group_participation(rep, "amine(Lys)")
        ma = group_participation(rep, "MA_group(LMA)")
        ax.plot(ds, lys, ls, color="#2b6cb0", marker=mk, ms=7, lw=1.6,
                label=f"amine (Lys) — {rep}")
        ax.plot(ds, ma, ls, color="#d62728", marker=mk, ms=7, lw=1.6,
                label=f"MA group (LMA) — {rep}")
    lys0 = 0.5 * (group_participation("rep1", "amine(Lys)")[0]
                  + group_participation("rep2", "amine(Lys)")[0])
    ax.plot(ds, lys0 * (1 - ds / 100), ":", color="#555", lw=1.4,
            label=r"Lys$_0\,(1-\mathrm{DS})$ reference")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{int(d)}%" for d in ds])
    ax.set_xlabel("Degree of substitution")
    ax.set_ylabel("H-bond participation per frame")
    ax.set_title("Methacrylation signature Lys → MA — cross-seed (rep1 vs rep2)",
                 loc="left", fontweight="bold", fontsize=12, pad=8)
    ax.grid(False)  # no background gridlines
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)
    ax.legend(loc="center left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = PLOT_ROOT / "Figure_HBondGroups_RepCompare.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


# =============================================================================
# Section C — Time-series overlays (per-frame trajectory)
#   (ported from plot_timeseries.py; output filenames unchanged)
# =============================================================================
TS_SYS = [("Gelatin", 0, "#e8a948"), ("Gel1MA", 33, "#a8d0f0"),
          ("Gel2MA", 67, "#5a9fd4"), ("Gel3MA", 100, "#1e3f6e")]
TS_EQ_LO, TS_EQ_HI = 45.0, 50.0

TS_CONF = [
    ("Rg_mean_A",                 "Radius of gyration $R_g$ (Å)", False),
    ("Ree_mean_A",                "End-to-end $R_{ee}$ (Å)",      False),
    ("Persistence_Length_mean_A", "Persistence length $L_p$ (Å)", False),
    ("RMSD_mean_A",               "Backbone RMSD (Å)",            False),
]
TS_INTER = [
    ("Hb_Intra_Strict",       "Intra-chain H-bonds",   False),
    ("Hb_Inter_Strict",       "Inter-chain H-bonds",   False),
    ("Hb_PW_Strict",          "Polymer–water H-bonds", False),
    ("Hb_MA_Wat_Total",       "MA–water H-bonds",      True),
    ("Salt_Bridges",          "Salt bridges",          False),
    ("MA_Inter_MinDist_NN_A", "MA–MA NN distance (Å)", True),
]


def _ts_smooth(y):
    n = len(y)
    if n < 11 or savgol_filter is None:
        return y
    w = max(11, (n // 20) | 1)
    w = min(w, n if n % 2 else n - 1)
    try:
        return savgol_filter(y, w, 3)
    except Exception:
        return y


def _ts_panel(ax, rep, col, ylabel, ma_only, panel_letter):
    for sysn, ds, color in TS_SYS:
        if ma_only and ds == 0:
            continue
        csv_path = DATA_ROOT / sysn / f"{DATA_PREFIX}_{rep}.csv"
        if not csv_path.exists():
            continue
        d = pd.read_csv(csv_path)
        if col not in d.columns:
            continue
        t = pd.to_numeric(d["Time_ns"], errors="coerce").values
        y = pd.to_numeric(d[col], errors="coerce").values
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        if len(t) < 3:
            continue
        ax.plot(t, y, color=color, lw=0.5, alpha=0.20)
        ax.plot(t, _ts_smooth(y), color=color, lw=1.6, label=f"{ds}%")
    ax.axvspan(TS_EQ_LO, TS_EQ_HI, color="#888", alpha=0.10, zorder=0)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel_letter}) {ylabel.split('(')[0].strip()}",
                 loc="left", fontweight="bold", fontsize=10.5, pad=6)
    ax.grid(False)  # no background gridlines
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)


def _ts_figure(metrics, title, out, rep, ncols):
    n = len(metrics)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    axf = axes.flatten()
    letters = "abcdefgh"
    for i, (col, ylabel, ma_only) in enumerate(metrics):
        _ts_panel(axf[i], rep, col, ylabel, ma_only, letters[i])
    for ax in axf[n:]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], color=c, lw=2, label=f"DS = {ds}%")
               for _, ds, c in TS_SYS]
    handles.append(Patch(facecolor="#888", alpha=0.10,
                         label=f"eq window ({TS_EQ_LO:.0f}–{TS_EQ_HI:.0f} ns)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015),
               handlelength=1.6, columnspacing=1.4)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.995, 0.005, f"{rep}, 50 ns", ha="right", va="bottom",
             fontsize=7.5, color="#999")
    bottom = 0.10 if nrows == 1 else 0.06
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


def figure_ts_conformational(rep: str) -> None:
    sfx = "" if rep == "rep1" else f"_{rep}"
    _ts_figure(TS_CONF, "Conformational Observables vs Time",
               PLOT_ROOT / f"Figure_TimeSeries_Conformational{sfx}.png", rep, ncols=2)


def figure_ts_interactions(rep: str) -> None:
    sfx = "" if rep == "rep1" else f"_{rep}"
    _ts_figure(TS_INTER, "Interaction Observables vs Time",
               PLOT_ROOT / f"Figure_TimeSeries_Interactions{sfx}.png", rep, ncols=3)


# =============================================================================
# Section D — Per-residue RMSF (local vs global collapse decomposition)
#   (ported from plot_rmsf_residue.py; output filename unchanged)
# =============================================================================
MA_SITES = [6, 15, 23]
OFF_SITE = [r for r in range(1, 25) if all(abs(r - s) >= 2 for s in MA_SITES)]


def _chain_mod_sites(sys_name: str) -> dict[str, set[int]]:
    psf = PRODUCTION / sys_name / "Output" / "debug_1.psf"
    out: dict[str, set[int]] = {}
    in_atoms = False
    for line in psf.read_text().splitlines():
        if "!NATOM" in line:
            in_atoms = True
            continue
        if in_atoms:
            p = line.split()
            if len(p) < 6:
                if line.strip() == "":
                    continue
                break
            seg, resid, resname = p[1], p[2], p[3]
            if resname == "LMA":
                out.setdefault(seg, set()).add(int(resid))
    return out


def _load_rmsf(sys_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / sys_name / f"{DATA_PREFIX}_{REP_TAG}_RMSF.csv")


def _rmsf_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("resid")["rmsf_A"]
    return pd.DataFrame({"resid": g.mean().index, "mean": g.mean().values,
                         "sem": g.sem().values, "n": g.count().values})


def figure_rmsf_residue() -> None:
    raw = {s: _load_rmsf(s) for s in SYSTEM_ORDER}
    mods = {s: _chain_mod_sites(s) for s in SYSTEM_ORDER}
    profiles = {s: _rmsf_profile(raw[s]) for s in SYSTEM_ORDER}

    fig = plt.figure(figsize=(7.4, 9.2))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    for s in SYSTEM_ORDER:
        spec, p = SYSTEMS[s], profiles[s]
        ax1.plot(p["resid"], p["mean"], "-o", ms=3, lw=1.4,
                 color=spec.color, label=spec.label.replace("\n", " "))
        ax1.fill_between(p["resid"], p["mean"] - p["sem"], p["mean"] + p["sem"],
                         color=spec.color, alpha=0.15, lw=0)
    ymax = max(p["mean"].max() for p in profiles.values())
    for r in MA_SITES:
        ax1.axvline(r, color="0.5", ls=":", lw=0.8, zorder=0)
        ax1.text(r, 1.2, f"site {r}", ha="center", va="bottom", fontsize=7, color="0.4")
    ax1.set_xlabel("Residue number")
    ax1.set_ylabel(r"RMSF (Å)")
    ax1.set_title("(a)  Per-residue RMSF — mean ± SEM over 12 chains")
    ax1.set_xticks(range(1, 25, 2))
    ax1.set_ylim(0.8, ymax * 1.28)
    ax1.legend(ncol=2, loc="upper right", fontsize=8)

    rows = []
    for s in ["Gel1MA", "Gel2MA"]:
        df = raw[s]
        for site in MA_SITES:
            sub = df[df["resid"] == site]
            mo = sub[sub["chain"].apply(lambda c: site in mods[s].get(c, set()))]["rmsf_A"]
            un = sub[sub["chain"].apply(lambda c: site not in mods[s].get(c, set()))]["rmsf_A"]
            rows.append({"system": s, "site": site, "delta": mo.mean() - un.mean(),
                         "mod": mo.mean(), "unmod": un.mean()})
    tb = pd.DataFrame(rows)

    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhspan(-0.5, 0.5, color="0.85", alpha=0.5, lw=0, zorder=0)
    site_x = {6: 0, 15: 1, 23: 2}
    offset = {"Gel1MA": -0.12, "Gel2MA": 0.12}
    for s in ["Gel1MA", "Gel2MA"]:
        sub = tb[tb["system"] == s]
        xs = [site_x[v] + offset[s] for v in sub["site"]]
        ax2.scatter(xs, sub["delta"], s=70, color=SYSTEMS[s].color,
                    label=SYSTEMS[s].label.replace("\n", " "), zorder=3)
    ax2.set_xticks(list(site_x.values()))
    ax2.set_xticklabels(["res 6\n(interior)", "res 15\n(interior)", "res 23\n(C-term)"])
    ax2.set_ylabel(r"$\Delta$RMSF, mod − unmod (Å)")
    ax2.set_title("(b)  Mechanism B test — locking would give Δ < 0 (grey = ±0.5 Å)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.annotate("locking\n(not seen)", (0.0, -0.9), ha="center", va="top",
                 fontsize=7, color="0.4")

    ds, means, sems = [], [], []
    for s in SYSTEM_ORDER:
        p = profiles[s].set_index("resid")
        off = p.loc[[r for r in OFF_SITE if r in p.index]]
        ds.append(SYSTEMS[s].ds_pct)
        means.append(off["mean"].mean())
        sems.append(off["mean"].std(ddof=1) / np.sqrt(len(off)))
    order = np.argsort(ds)
    ds = np.array(ds)[order]; means = np.array(means)[order]; sems = np.array(sems)[order]
    ax3.errorbar(ds, means, yerr=sems, fmt="-o", color="0.2", capsize=3, lw=1.4)
    for x, y, s in zip(ds, means, [SYSTEM_ORDER[i] for i in order]):
        ax3.scatter(x, y, s=45, color=SYSTEMS[s].color, zorder=3)
    ax3.set_xlabel("Degree of substitution (%)")
    ax3.set_ylabel(r"RMSF, off-site residues (Å)")
    ax3.set_title("(c)  Mechanism A — residues far from any MA site, vs DS")
    ax3.set_xticks([0, 33, 67, 100])

    fig.tight_layout(h_pad=2.0)
    out = PLOT_ROOT / "Figure_RMSF_residue.png"
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")

    print("  --- (a) whole / at-site / off-site mean RMSF (Å) ---")
    for s in SYSTEM_ORDER:
        p = profiles[s].set_index("resid")["mean"]
        at = p.loc[[r for r in MA_SITES if r in p.index]]
        off = p.loc[[r for r in OFF_SITE if r in p.index]]
        at_str = f"{at.mean():.2f}" if len(at) else "n/a"
        print(f"    {s:8s} whole={p.mean():.2f}  at-site={at_str}  off-site={off.mean():.2f}")


# =============================================================================
# Section E — MA–MA contact map + collapse/aggregation (M5)
#   (ported from plot_ma_contact.py; output filename unchanged)
# =============================================================================
def _ma_contact_load(sys_name):
    m = np.load(DATA_ROOT / sys_name / f"{DATA_PREFIX}_{REP_TAG}_MAContactProb.npy")
    lab = pd.read_csv(DATA_ROOT / sys_name / f"{DATA_PREFIX}_{REP_TAG}_MAContactLabels.csv")
    chains = lab["chain"].to_numpy()
    n = len(chains)
    same = chains[:, None] == chains[None, :]
    iu = np.triu_indices(n, k=1)
    v, su = m[iu], same[iu]
    intra = v[su].mean() if su.any() else np.nan
    inter = v[~su].mean() if (~su).any() else np.nan
    bnd = [i for i in range(1, n) if chains[i] != chains[i - 1]]
    return {"m": m, "n": n, "bnd": bnd, "intra": intra, "inter": inter}


def figure_ma_contact() -> None:
    ma_systems = [s for s in SYSTEMS if SYSTEMS[s].has_ma]
    data = {s: _ma_contact_load(s) for s in ma_systems}
    vmax = max(d["m"].max() for d in data.values())
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    im = None
    for ax, s in zip(axes[:3], ma_systems):
        d = data[s]
        im = ax.imshow(d["m"], cmap="magma", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(f"{SYSTEMS[s].label.replace(chr(10), ' ')}\n{d['n']} MA groups",
                     fontsize=9)
        for b in d["bnd"]:
            ax.axhline(b - 0.5, color="cyan", lw=0.4, alpha=0.45)
            ax.axvline(b - 0.5, color="cyan", lw=0.4, alpha=0.45)
        ax.set_xlabel("MA group index")
    axes[0].set_ylabel("MA group index")
    cbar = fig.colorbar(im, ax=axes[:3], location="bottom", shrink=0.55, aspect=40, pad=0.02)
    cbar.set_label("P(MA–MA contact, C8–C8 ≤ 7 Å)")

    ax = axes[3]
    ds = [SYSTEMS[s].ds_pct for s in ma_systems]
    intra = [data[s]["intra"] for s in ma_systems]
    inter = [data[s]["inter"] for s in ma_systems]
    ax.plot(ds, intra, "-o", color="#d1495b", label="intra-chain (collapse)")
    ax.plot(ds, inter, "-s", color="#30638e", label="inter-chain (aggregation)")
    ax.set_xlabel("Degree of substitution (%)")
    ax.set_ylabel("mean P(MA–MA contact ≤ 7 Å)")
    ax.set_title("(d)  collapse vs aggregation")
    ax.set_xticks(ds)
    ax.margins(x=0.12, y=0.18)
    ax.annotate("Gel1MA: no intra pair\n(1 MA / chain)", xy=(33, inter[0]),
                xytext=(0, 12), textcoords="offset points",
                fontsize=7, color="#d1495b", ha="left")
    ax.legend(fontsize=8, frameon=False, loc="center right")

    fig.suptitle("MA–MA Contact Map (C8–C8 ≤ 7 Å)", fontsize=12, fontweight="bold")
    fig.text(0.995, 0.01, "Polymatic criterion", ha="right", va="bottom",
             fontsize=7.5, color="#999")
    out = PLOT_ROOT / "Figure_M5_MAcontact.png"
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


# =============================================================================
# Section F — Time-block convergence facet (reads time_block_summary.csv)
#   (figure side of time_block_analysis.py; that script still does the
#    numeric grading + writes the CSV this reads.)
# =============================================================================
TIME_BLOCK_CSV = PLOT_ROOT / "time_block_summary.csv"


def figure_convergence() -> None:
    if not TIME_BLOCK_CSV.exists():
        print(f"[skip] CONVERGENCE: {TIME_BLOCK_CSV.name} not found — "
              "run time_block_analysis.py first.")
        return
    df = pd.read_csv(TIME_BLOCK_CSV)
    if df.empty:
        print("[skip] CONVERGENCE: time_block_summary.csv is empty.")
        return
    n_blocks = int(df["n_blocks"].iloc[0])
    metrics = list(dict.fromkeys(df["metric"].tolist()))
    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.9))
    axes_flat = np.array(axes).flatten()
    for ax, met in zip(axes_flat, metrics):
        sub = df[df["metric"] == met]
        if sub.empty:
            ax.set_visible(False)
            continue
        for _, r in sub.iterrows():
            t = [r[f"block_{i + 1}_t_ns"] for i in range(n_blocks)]
            y = [r[f"block_{i + 1}_mean"] for i in range(n_blocks)]
            s = [r[f"block_{i + 1}_sem"] for i in range(n_blocks)]
            ax.errorbar(t, y, yerr=s, color=SYSTEMS[r["system"]].color,
                        marker="o", markersize=4, capsize=2, lw=1.2,
                        label=f"{r['system']} [{r['grade']}]")
        r0 = sub.iloc[0]
        ax.set_title(r0["label"], fontsize=9, loc="left", fontweight="bold")
        ax.set_xlabel("Time (ns)", fontsize=8)
        ax.set_ylabel(r0["unit"] if isinstance(r0["unit"], str) and r0["unit"] else "value",
                      fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)  # no background gridlines
        ax.legend(fontsize=6, loc="best", frameon=False)
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Time-Block Convergence", fontsize=12, fontweight="bold", y=1.005)
    fig.text(0.995, 0.005, f"{n_blocks} blocks, post-warmup", ha="right",
             va="bottom", fontsize=7.5, color="#999")
    fig.tight_layout()
    out = PLOT_ROOT / "Figure_TimeBlock_Convergence.png"
    save_fig(fig, out)
    plt.close(fig)
    print(f"  → {out.relative_to(out.parents[1])}")


# =============================================================================
# CLI dispatcher
# =============================================================================
# tag -> (callable, needs_summary_df, chiu_style, help)
def _build_registry(args):
    return {
        "M1":  (lambda df: figure_M1_conformational(df), True,  False, "Rg/Ree/Lp/RMSD bars"),
        "M2":  (lambda df: figure_M2_interactions(df),   True,  False, "H-bond/salt/contact bars"),
        "M2C": (lambda df: figure_M2_chiu(df),           True,  False, "M2 H-bond panels only (no salt/contact)"),
        "M3":  (lambda df: figure_M3_MAstructure(df),    True,  False, "MA–MA NN + g(r)"),
        "M3B": (lambda df: figure_M3b_RDF_detail(df),    True,  False, "per-DS MA–MA g(r) detail"),
        "M1P": (lambda df: figure_M1_paired(df),         True,  False, "M1 paired bars (rep2)"),
        "M2P": (lambda df: figure_M2_paired(df),         True,  False, "M2 paired bars (rep2)"),
        "M3P": (lambda df: figure_M3_paired(df),         True,  False, "M3 paired bars (rep2)"),
        "HB":       (lambda df: figure_hb_groups(args.rep),                   False, True,  "functional-group H-bond"),
        "HBPAIRS":  (lambda df: figure_hb_pairs_allreps(args.top, args.scope), False, True, "donor–acceptor pairs"),
        "HBTARGET": (lambda df: figure_hb_target(args.rep, args.target, args.top, args.scope), False, True, "one group's partners"),
        "HBREP":    (lambda df: figure_hb_repcompare(),                       False, True,  "rep1 vs rep2 Lys→MA"),
        "TSCONF":  (lambda df: figure_ts_conformational(args.rep), False, False, "conformational vs time"),
        "TSINTER": (lambda df: figure_ts_interactions(args.rep),   False, False, "interactions vs time"),
        "RMSF":      (lambda df: figure_rmsf_residue(), False, False, "per-residue RMSF 3-panel"),
        "MACONTACT": (lambda df: figure_ma_contact(),   False, False, "MA–MA contact map M5"),
        "CONVERGENCE": (lambda df: figure_convergence(), False, False, "time-block convergence facet"),
    }


# Default = the full paper-relevant cross-system set. The M-series uses the
# PAIRED (rep1 vs rep2 + N=2 mean ± SEM) variants M1P/M2P/M3P — production now
# has rep2 for every system. The old single-bar M1/M2/M3 (one averaged bar that
# reads as a single replica) are intentionally NOT in the default; they remain
# available via explicit `--figs M2` / `--figs M2 --rep rep2` for single-replica
# inspection only.
DEFAULT_FIGS = ("M1P", "M2P", "M3P", "M3B", "HB", "HBPAIRS",
                "TSCONF", "TSINTER", "RMSF", "MACONTACT", "CONVERGENCE")

# Chiu 2026 pre-crosslink counterpart for each tag — lets us pick the figures
# that have a direct GGMA-paper analogue (Chiu does these BEFORE crosslinking)
# vs the ones that are our GelMA-specific additions or have no Chiu equivalent.
#   ✓ = direct Chiu pre-crosslink figure
#   ~ = partial (Chiu has the H-bond half but not the GelMA-unique charged panels,
#       or it's a derived/extra view of a Chiu figure)
#   ✗ = no Chiu pre-crosslink counterpart (our addition, needs rep2, or post-only)
CHIU_PRECROSSLINK = {
    "M1":          "✓ Chiu Fig 2 — Rg, Ree (+our Lp, RMSD)",
    "M2":          "~ Chiu Fig 4 H-bonds; salt-bridge/contact = GelMA-only (no Lys in GG)",
    "M2C":         "✓ Chiu Fig 4 — H-bond panels only (salt/contact dropped)",
    "M3":          "✓ Chiu Fig 6 precursor — MA–MA proximity",
    "M3B":         "✓ Chiu Fig 6 precursor — MA–MA g(r) detail",
    "M1P":         "✓ = M1 (needs rep2)",
    "M2P":         "~ = M2 (needs rep2)",
    "M3P":         "✓ = M3 (needs rep2)",
    "HB":          "✓ Chiu Fig 4 — functional-group H-bond",
    "HBPAIRS":     "✓ Chiu Fig 4 / S6 — donor–acceptor pairs",
    "HBTARGET":    "~ Chiu Fig 4 derived (one group's partners)",
    "HBREP":       "✗ Chiu is N=1 (no rep compare)",
    "TSCONF":      "~ Chiu equilibration RMSD-vs-time (partial)",
    "TSINTER":     "✗ our addition (Chiu has no interactions-vs-time)",
    "RMSF":        "✗ Chiu uses chain RMSD, not per-residue RMSF",
    "MACONTACT":   "~ Chiu Fig 6 area (MA–MA contact map)",
    "CONVERGENCE": "✓ Chiu equilibration check (RMSD plateau; ours is stricter)",
    # Not a plot_merge tag — generated by Data_scripts/nvt_thermo.py:
    "NVT*":        "✓ Chiu equilibration (NVT thermo) — run Data_scripts/nvt_thermo.py",
}


def main() -> int:
    global _OUT_SUFFIX, _REP_FILTER
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", nargs="+", default=list(DEFAULT_FIGS),
                    help=f"Figure tags to render. Default: {' '.join(DEFAULT_FIGS)}")
    ap.add_argument("--rep", default="rep1",
                    help="Replica for M-series/H-bond/time-series (default rep1). "
                         "For M-series, --rep also switches to single-replica "
                         "summary built from that replica's per-frame CSVs "
                         "(figures get a '_<rep>' suffix).")
    ap.add_argument("--target", default="carboxylate",
                    choices=[a for a, _, _, _ in GROUPS],
                    help="(HBTARGET) functional group to dissect.")
    ap.add_argument("--scope", default="both", choices=["both", "intra", "inter"],
                    help="(HBPAIRS/HBTARGET) render both panels or one.")
    ap.add_argument("--top", type=int, default=8,
                    help="(HBPAIRS/HBTARGET) number of x-axis columns.")
    ap.add_argument("--list", action="store_true", help="List figure tags and exit.")
    args = ap.parse_args()

    # Windows consoles default to cp950 here; figure labels print fine into the
    # PNG/PDF but the status lines contain Å / → / – which cp950 can't encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    registry = _build_registry(args)
    if args.list:
        print(f"{'tag':<12} {'figure':<34} {'default':<8} Chiu pre-crosslink counterpart")
        print("-" * 100)
        for tag, (_fn, _ndf, _chiu, helptext) in registry.items():
            mark = "default" if tag in DEFAULT_FIGS else ""
            chiu = CHIU_PRECROSSLINK.get(tag, "")
            print(f"  {tag:<10} {helptext:<34} {mark:<8} {chiu}")
        # Standalone (not a plot_merge --figs tag) but part of the Chiu set:
        print(f"  {'NVT*':<10} {'NVT equilibration (T + Epot)':<34} {'':<8} "
              f"{CHIU_PRECROSSLINK['NVT*']}")
        print("\n✓ = direct Chiu pre-crosslink figure | ~ = partial / derived | "
              "✗ = our addition / rep2 / post-only")
        return 0

    apply_paper_style()

    requested = [t.upper() for t in args.figs]
    unknown = [t for t in requested if t not in registry]
    for t in unknown:
        print(f"  [skip] unknown figure tag: {t}")
    requested = [t for t in requested if t in registry]
    if not requested:
        print("nothing to do."); return 1

    # M-series single-replica mode (only when an explicit --rep other than the
    # ensemble default is requested AND an M-series tag is present).
    m_tags = {"M1", "M2", "M3", "M3B"}
    rep_mode = args.rep != "rep1" and any(t in m_tags for t in requested)

    summary_df = None
    if any(registry[t][1] for t in requested):
        if rep_mode:
            _REP_FILTER = args.rep
            _OUT_SUFFIX = f"_{args.rep}"
            summary_df = _build_rep_summary(args.rep)
            if summary_df.empty:
                print(f"  [error] no per-frame CSVs found for {args.rep}")
                return 1
            print(f"built single-replica summary for {args.rep}: {len(summary_df)} rows")
        else:
            summary_df = _load_summary()
            print(f"loaded {len(summary_df)} rows from {SUMMARY_CSV.name}")

    # Apply Chiu rcParams only if a Chiu-style figure is requested; restore
    # paper style afterwards so non-Chiu figures keep top/right spines off.
    for tag in requested:
        fn, needs_df, chiu, _help = registry[tag]
        if chiu:
            _enable_chiu_rcparams()
        else:
            apply_paper_style()
        print(f"[{tag}]")
        ret = fn(summary_df)
        # H-bond figures return their output Path instead of printing it.
        if isinstance(ret, Path):
            print(f"  → {ret.relative_to(ret.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
