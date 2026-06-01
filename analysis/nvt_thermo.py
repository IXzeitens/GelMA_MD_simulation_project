"""NVT-stage thermodynamic convergence analysis (root-level batch).

For each subsystem in ``analysis_config.SYSTEMS``:
  1. Parse ``<sys>/Output/nvt.log`` ENERGY lines into a tidy CSV.
  2. Compute converged-mean / std for the trailing 50 % of the NVT run.
  3. Emit a publication-grade 1×2 convergence figure (Temperature + Potential
     energy) under ``plot/<sys>/`` plus a cross-system overlay under ``plot/``.

Outputs:

    Data/<sys>/GelMA_analysis_NVT_thermo.csv         # parsed step → T / E
    Data/<sys>/GelMA_analysis_NVT_convergence.csv    # stable-region stats
    plot/<sys>/Figure_NVT_Convergence.{png,pdf}      # per-system 1×2 figure
    plot/Figure_NVT_Convergence_AllSystems.{png,pdf} # 4-system overlay

Production-aligned copy. The original lived at
``_legacy/legacy_scripts/nvt_thermo.py`` and inherited its rcParams from
``0511_plot.py``; here we instead reuse :func:`apply_paper_style` from
``production/plot/_shared.py`` so the figure stays visually consistent with
the M-series (``plot_merged.py``).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import (
    DATA_PREFIX,
    DATA_ROOT,
    PLOT_ROOT,
    ROOT_DIR,
    SYSTEMS,
)

log = logging.getLogger(__name__)

#: NAMD ``timestep`` in fs (matches ``NVT.conf``).
NVT_TIMESTEP_FS: float = 2.0

#: Fraction of trajectory considered "stable" for convergence statistics.
STABLE_FRACTION: float = 0.5

#: Target temperature from NVT.conf (Langevin set point).
TARGET_TEMPERATURE_K: float = 310.0

#: Save .pdf next to every .png so the figure is paper-ready as a vector too.
SAVE_VECTOR_PDF: bool = True

# ---------------------------------------------------------------------------
# Reuse the journal-grade rcParams from production/plot/_shared.py so this
# figure stays visually consistent with M1/M2/M3. Non-fatal if missing
# (e.g. someone runs this script standalone before plot/ is set up).
# ---------------------------------------------------------------------------
_PLOT_DIR = ROOT_DIR / "plot"
if _PLOT_DIR.is_dir() and str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))
try:
    from _shared import apply_paper_style  # type: ignore  # noqa: E402

    apply_paper_style()
except Exception as exc:  # pragma: no cover — styling is non-critical
    log.warning("Could not apply paper rcParams from %s/_shared.py: %s",
                _PLOT_DIR, exc)


# ===========================================================================
# Data model
# ===========================================================================
@dataclass(frozen=True)
class ConvergenceStats:
    """Stable-region (last STABLE_FRACTION of trajectory) thermodynamic stats."""
    temperature_mean_K: float
    temperature_std_K: float
    potential_mean_kcal: float
    potential_std_kcal: float
    total_mean_kcal: float
    total_std_kcal: float
    stable_start_ns: float
    stable_end_ns: float
    n_stable_records: int


# ===========================================================================
# Parsing
# ===========================================================================
def parse_namd_log_to_df(log_path: Path, timestep_fs: float = NVT_TIMESTEP_FS) -> pd.DataFrame:
    """Extract NAMD ``ENERGY:`` lines from a log file into a DataFrame.

    Returns columns: ``Step`` (int), ``Time_ns`` (float), ``Temperature_K``,
    ``Potential_Energy_kcal_mol``, ``Total_Energy_kcal_mol``.

    Raises ``FileNotFoundError`` if the log doesn't exist; returns an empty
    DataFrame if no ENERGY lines are present.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"NAMD log not found: {log_path}")

    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ENERGY:"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                step = int(parts[1])
                total_E = float(parts[11])
                temp = float(parts[12])
                pot_E = float(parts[13])
            except (ValueError, IndexError):
                continue
            rows.append({
                "Step": step,
                "Time_ns": (step * timestep_fs) / 1_000_000.0,
                "Temperature_K": temp,
                "Potential_Energy_kcal_mol": pot_E,
                "Total_Energy_kcal_mol": total_E,
            })
    return pd.DataFrame(rows)


def compute_convergence_stats(df: pd.DataFrame) -> ConvergenceStats | None:
    """Mean ± std over the trailing :data:`STABLE_FRACTION` of the trajectory."""
    if df.empty:
        return None
    max_time = float(df["Time_ns"].max())
    stable_start = max_time * (1.0 - STABLE_FRACTION)
    stable = df[df["Time_ns"] >= stable_start]
    if stable.empty:
        return None
    return ConvergenceStats(
        temperature_mean_K=float(stable["Temperature_K"].mean()),
        temperature_std_K=float(stable["Temperature_K"].std()),
        potential_mean_kcal=float(stable["Potential_Energy_kcal_mol"].mean()),
        potential_std_kcal=float(stable["Potential_Energy_kcal_mol"].std()),
        total_mean_kcal=float(stable["Total_Energy_kcal_mol"].mean()),
        total_std_kcal=float(stable["Total_Energy_kcal_mol"].std()),
        stable_start_ns=stable_start,
        stable_end_ns=max_time,
        n_stable_records=int(len(stable)),
    )


# ===========================================================================
# Plotting (per-system)
# ===========================================================================
def _save(fig, path: Path) -> None:
    fig.savefig(path)
    if SAVE_VECTOR_PDF:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("  → %s%s", path.name, " (+ pdf)" if SAVE_VECTOR_PDF else "")


def _full_frame(*axes, linewidth: float = 0.8) -> None:
    """Restore all four spines on each axes — overrides the global
    ``axes.spines.top/right = False`` set by :func:`apply_paper_style`
    for the M-series figures. NVT QC plots read better fully enclosed."""
    for ax in axes:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(linewidth)


def _frame_legends(*axes, edgecolor: str = "#666", framealpha: float = 0.95) -> None:
    """Re-enable the legend bounding box on each axes — overrides the
    global ``legend.frameon = False`` set by :func:`apply_paper_style`.
    Pair with :func:`_full_frame` so NVT QC plots have a closed-box
    feel both inside (axes) and over the legend.
    """
    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        leg.set_frame_on(True)
        frame = leg.get_frame()
        frame.set_edgecolor(edgecolor)
        frame.set_linewidth(0.6)
        frame.set_alpha(framealpha)


def plot_per_system_convergence(
    df: pd.DataFrame,
    stats: ConvergenceStats,
    plot_dir: Path,
    label: str,
    color: str,
    target_temp_K: float = TARGET_TEMPERATURE_K,
) -> None:
    """1×2 NVT convergence figure: Temperature (a) + Potential energy (b)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3)

    time_ns = df["Time_ns"]
    pretty_label = label.replace("\n", " ")

    # ---------- (a) Temperature ----------
    ax = axes[0]
    ax.plot(time_ns, df["Temperature_K"], color=color, linewidth=0.8, alpha=0.6,
            label="Instantaneous")
    ax.axhline(
        target_temp_K, color="#999999", linestyle=":", linewidth=1.5, zorder=4,
        label=f"Target = {target_temp_K:.0f} K",
    )
    ax.axhline(
        stats.temperature_mean_K, color="black", linestyle="--", linewidth=1.8, zorder=5,
        label=rf"$\bar{{T}}$ = {stats.temperature_mean_K:.1f} ± {stats.temperature_std_K:.1f} K",
    )
    ax.set_title("(a) Temperature Convergence", loc="left")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Temperature, $T$ (K)")
    ax.set_xlim(0, float(time_ns.max()))
    # Tight convergence range: ±4σ around stable-region mean (typically ~±10 K).
    t_margin = max(stats.temperature_std_K * 4.0, 5.0)
    ax.set_ylim(stats.temperature_mean_K - t_margin, stats.temperature_mean_K + t_margin)
    ax.legend(loc="lower right", fontsize=9)

    # ---------- (b) Potential energy ----------
    ax = axes[1]
    ax.plot(time_ns, df["Potential_Energy_kcal_mol"], color=color, linewidth=0.8, alpha=0.6,
            label="Instantaneous")
    ax.axhline(
        stats.potential_mean_kcal, color="black", linestyle="--", linewidth=1.8, zorder=5,
        label=(
            rf"$\bar{{E}}_{{\rm pot}}$ = {stats.potential_mean_kcal:,.0f} ± "
            rf"{stats.potential_std_kcal:,.0f} kcal mol$^{{-1}}$"
        ),
    )
    ax.set_title("(b) Potential Energy", loc="left")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"$E_{\rm pot}$ (kcal mol$^{-1}$)")
    ax.set_xlim(0, float(time_ns.max()))
    pe_margin = max(stats.potential_std_kcal * 4.0, 200.0)
    ax.set_ylim(stats.potential_mean_kcal - pe_margin, stats.potential_mean_kcal + pe_margin)
    ax.legend(loc="lower right", fontsize=9)

    _full_frame(*axes)
    _frame_legends(*axes)
    fig.suptitle(pretty_label, fontsize=12, fontweight="bold", y=0.99)
    _save(fig, plot_dir / "Figure_NVT_Convergence.png")


# ===========================================================================
# Plotting (cross-system overlay)
# ===========================================================================
def plot_cross_system_overlay(
    all_dfs: dict[str, pd.DataFrame],
    all_stats: dict[str, ConvergenceStats],
    target_temp_K: float = TARGET_TEMPERATURE_K,
    output_suffix: str = "",
) -> None:
    """Single overlay figure with all 4 subsystems' NVT temperature traces."""
    if not all_dfs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3)

    # ---------- (a) Temperature overlay ----------
    ax = axes[0]
    for name, df in all_dfs.items():
        spec = SYSTEMS[name]
        ax.plot(
            df["Time_ns"], df["Temperature_K"],
            color=spec.color, linewidth=0.7, alpha=0.5,
            label=spec.label.replace("\n", " "),
        )
    ax.axhline(target_temp_K, color="black", linestyle=":", linewidth=1.5,
               label=f"Target = {target_temp_K:.0f} K")
    ax.set_title("(a) NVT Temperature, All Subsystems", loc="left")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Temperature, $T$ (K)")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=8)

    # ---------- (b) Potential energy overlay ----------
    # Replaces the previous T̄-bar panel: temperature is thermostat-clamped, so
    # showing its converged value adds no information beyond panel (a). Potential
    # energy, by contrast, is an emergent observable whose plateau directly
    # demonstrates structural equilibration (Allen & Tildesley §6.3).
    #
    # `all_stats` is still accepted for API stability but unused here — each
    # system's PE plateau is visually evident from its line trace.
    ax = axes[1]
    warmup_ns = 0.05   # drop initial minimisation-era spike (PE drops from
                       # ~10⁷ → plateau within the first ~50 ps); excluding it
                       # lets matplotlib auto-scale show the plateau details.
    for name, df in all_dfs.items():
        spec = SYSTEMS[name]
        sub = df[df["Time_ns"] >= warmup_ns]
        ax.plot(
            sub["Time_ns"], sub["Potential_Energy_kcal_mol"],
            color=spec.color, linewidth=0.7, alpha=0.6,
            label=spec.label.replace("\n", " "),
        )
    ax.set_title("(b) NVT Potential Energy, All Subsystems", loc="left")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Potential energy, $E_\mathrm{pot}$ (kcal mol$^{-1}$)")
    ax.set_xlim(left=0)
    ax.legend(loc="best", fontsize=8)
    _ = target_temp_K  # silence "unused" — kept in signature for API stability

    _full_frame(*axes)
    _frame_legends(*axes)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    out = PLOT_ROOT / f"Figure_NVT_Convergence_AllSystems{output_suffix}.png"
    _save(fig, out)


# ===========================================================================
# Top-level driver
# ===========================================================================
def _resolve_nvt_log(base_name: str, rep: str) -> Path | None:
    """Locate ``nvt.log`` for one (base subsystem, rep) pair.

    Mirrors ``0511_data.resolve_replica_input_dir`` — rep1 lives in
    ``<base>/Output/``, rep2+ in ``<base>_<rep>/Output/``.
    """
    if rep == "rep1":
        p = ROOT_DIR / base_name / "Output" / "nvt.log"
    else:
        p = ROOT_DIR / f"{base_name}_{rep}" / "Output" / "nvt.log"
    return p if p.exists() else None


def process_subsystem(name: str, rep: str) -> tuple[pd.DataFrame, ConvergenceStats] | None:
    """Parse one (subsystem, rep) NVT log, write CSVs, draw per-system figure."""
    spec = SYSTEMS[name]
    log_path = _resolve_nvt_log(name, rep)
    if log_path is None:
        log.warning("[%s/%s] nvt.log not found; skipping.", name, rep)
        return None

    df = parse_namd_log_to_df(log_path)
    if df.empty:
        log.warning("[%s/%s] no ENERGY records in %s.", name, rep, log_path.name)
        return None

    stats = compute_convergence_stats(df)
    if stats is None:
        log.warning("[%s/%s] could not compute convergence stats.", name, rep)
        return None

    data_dir = DATA_ROOT / name
    plot_dir = PLOT_ROOT / name / (rep if rep != "rep1" else "")
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if rep == "rep1" else f"_{rep}"
    df.to_csv(data_dir / f"{DATA_PREFIX}_NVT_thermo{suffix}.csv", index=False)

    pd.DataFrame([{
        "subsystem": name, "replica": rep,
        "stable_start_ns": stats.stable_start_ns,
        "stable_end_ns": stats.stable_end_ns,
        "n_stable_records": stats.n_stable_records,
        "temperature_mean_K": stats.temperature_mean_K,
        "temperature_std_K": stats.temperature_std_K,
        "potential_mean_kcal_mol": stats.potential_mean_kcal,
        "potential_std_kcal_mol": stats.potential_std_kcal,
        "total_mean_kcal_mol": stats.total_mean_kcal,
        "total_std_kcal_mol": stats.total_std_kcal,
    }]).to_csv(data_dir / f"{DATA_PREFIX}_NVT_convergence{suffix}.csv", index=False)

    log.info(
        "[%s/%s] %d records | stable %.2f–%.2f ns | T̄=%.1f±%.1f K",
        name, rep, len(df), stats.stable_start_ns, stats.stable_end_ns,
        stats.temperature_mean_K, stats.temperature_std_K,
    )

    plot_per_system_convergence(df, stats, plot_dir, spec.label, spec.color)
    return df, stats


def _parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="nvt_thermo.py",
        description="NVT convergence audit per replica.",
    )
    parser.add_argument(
        "--rep", default="rep1",
        help="Replica to analyse: rep1 (default, base subsystem Output/), "
             "rep2 / rep3 (clone subproject Output/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args(argv)
    log.info("Project root: %s | replica = %s", ROOT_DIR, args.rep)

    all_dfs: dict[str, pd.DataFrame] = {}
    all_stats: dict[str, ConvergenceStats] = {}
    for name in SYSTEMS:
        log.info("=== %s (%s) ===", name, args.rep)
        result = process_subsystem(name, args.rep)
        if result is None:
            continue
        df, stats = result
        all_dfs[name] = df
        all_stats[name] = stats

    if len(all_dfs) >= 2:
        log.info("=== Cross-system overlay ===")
        suffix = "" if args.rep == "rep1" else f"_{args.rep}"
        plot_cross_system_overlay(all_dfs, all_stats, output_suffix=suffix)

    log.info("NVT thermo pipeline complete.")


if __name__ == "__main__":
    main()
