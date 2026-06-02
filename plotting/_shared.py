"""Shared paths, system definitions, and plot styling for the production
pre-crosslink analysis suite.

Single source of truth — data_collect.py and the plot_* scripts all import
from here so DS labels, colors, and selection rules stay aligned.

Maps over the existing `production/Data/<sys>/GelMA_analysis_*` outputs
written by the legacy `_legacy/legacy_scripts/0511_data.py` driver
(last regenerated 2026-05-21). Does NOT re-run analysis — only consumes
the existing CSVs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# =============================================================================
# Paths
# =============================================================================
# Simulation data (production/) lives alongside this repo, not inside it.
# Default: the workspace dir that contains gelma_md/ (= repo root's parent).
# Override with the GELMA_REPO env var if your data sits elsewhere.
REPO = Path(os.environ.get("GELMA_REPO") or Path(__file__).resolve().parents[2])

PRODUCTION = REPO / "production"
DATA_ROOT = PRODUCTION / "Data"
PLOT_ROOT = PRODUCTION / "plot"
SUMMARY_CSV = PLOT_ROOT / "pre_crosslink_summary.csv"

DATA_PREFIX = "GelMA_analysis"      # CSV name prefix per replica
REP_TAG = "rep1"                    # single replica in current production


# =============================================================================
# System registry — mirrors _legacy/legacy_scripts/analysis_config.SYSTEMS
# but limited to the 4 base big-box subprojects in current production/.
# =============================================================================
@dataclass(frozen=True)
class SubsystemSpec:
    name: str
    label: str           # for plot legend / x-axis tick
    color: str           # matplotlib color, matches legacy convention
    ds_pct: int          # degree of substitution (% lysine modified)
    has_ma: bool         # whether the PSF contains LMA residues


SYSTEMS: dict[str, SubsystemSpec] = {
    "Gelatin": SubsystemSpec("Gelatin", "Gelatin\n(DS=0%)",   "#d62728", ds_pct=0,   has_ma=False),
    "Gel1MA":  SubsystemSpec("Gel1MA",  "Gel1MA\n(DS=33%)",   "#1f77b4", ds_pct=33,  has_ma=True),
    "Gel2MA":  SubsystemSpec("Gel2MA",  "Gel2MA\n(DS=67%)",   "#ff7f0e", ds_pct=67,  has_ma=True),
    "Gel3MA":  SubsystemSpec("Gel3MA",  "Gel3MA\n(DS=100%)",  "#2ca02c", ds_pct=100, has_ma=True),
}
SYSTEM_ORDER: tuple[str, ...] = tuple(SYSTEMS.keys())


# =============================================================================
# Equilibrium window — matches _legacy analysis_config.EQ_WINDOW_NS
# =============================================================================
DT_PS = 100.0                # ps per dcd frame (dcdfreq 50000 × 2 fs)
EQ_WINDOW_NS = 5.0           # last 5 ns of each trajectory = "equilibrium sample"


# =============================================================================
# Metric registry — what to extract from Ensemble_Summary / per-frame CSV
# and how to label/group in plots.
# =============================================================================
@dataclass(frozen=True)
class MetricSpec:
    csv_col: str             # column name in per-frame rep1.csv (also used to derive _mean / _sem in Ensemble)
    label: str               # human-readable label for plot axis
    unit: str                # unit suffix for plot axis
    group: str               # logical group: conformational | sasa | hbond | ma_cluster | other
    ma_only: bool = False    # True → skip Gelatin in plots (NaN for no-MA system)


METRICS: list[MetricSpec] = [
    # Conformational
    MetricSpec("Rg_mean_A",                 "Radius of gyration $R_g$",       "Å",           "conformational"),
    MetricSpec("Ree_mean_A",                "End-to-end distance $R_{ee}$",   "Å",           "conformational"),
    MetricSpec("Persistence_Length_mean_A", "Persistence length $L_p$",       "Å",           "conformational"),
    MetricSpec("RMSD_mean_A",               "Backbone RMSD",                  "Å",           "conformational"),

    # Surface area
    MetricSpec("SASA_Global_A2",            "Total SASA",                     "Å²",          "sasa"),
    MetricSpec("SASA_MA_A2",                "MA-group SASA",                  "Å²",          "sasa", ma_only=True),

    # Hydrogen bonds (4 categories)
    MetricSpec("Hb_Intra_Strict",           "Intra-chain H-bonds",            "count",       "hbond"),
    MetricSpec("Hb_Inter_Strict",           "Inter-chain H-bonds",            "count",       "hbond"),
    MetricSpec("Hb_PW_Strict",              "Polymer–water H-bonds",          "count",       "hbond"),
    MetricSpec("Hb_MA_Wat_Total",           "MA–water H-bonds",               "count",       "hbond", ma_only=True),

    # MA-group geometry (only meaningful for MA-bearing systems)
    # Cluster count / max-cluster-size / cluster-Rg were dropped from the
    # downstream summary on 2026-05-27: at the Polymatic-aligned 7 Å cutoff
    # the system is ~90% MA singletons, the "largest cluster Rg" metric had
    # std > mean (transient bridges dominate the statistic), and the same
    # physics is captured cutoff-free by the MA–MA RDF panel in M3 (c).
    # The per-frame columns are still computed in data_analysis.py for
    # diagnostics but no longer surface in pre_crosslink_summary.csv.
    MetricSpec("MA_Inter_MinDist_NN_A",     "MA–MA nearest neighbour",        "Å",           "ma_cluster", ma_only=True),

    # MA–MA radial distribution function — characteristic distance + peak height.
    # Sourced from `<sys>/GelMA_analysis_rep1_MA_ensemble_RDF.csv` (not the
    # Ensemble_Summary), computed by the legacy `_compute_rdfs()` driver.
    # data_collect.py detects the "MA_RDF_" prefix and routes to RDF reader
    # instead of the standard ensemble lookup. Stationarity check is skipped
    # (RDF is already a 30-ns ensemble average — single number per system).
    MetricSpec("MA_RDF_FirstPeak_r_A",      "MA–MA RDF first peak position",  "Å",           "ma_cluster", ma_only=True),
    MetricSpec("MA_RDF_FirstPeak_g_r",      "MA–MA RDF first peak height",    "",            "ma_cluster", ma_only=True),

    # Other
    MetricSpec("Contact_Survival_Rate",     "Contact survival rate",          "",            "other"),
    MetricSpec("Residue_Min_Contact_Total", "Inter-residue contacts",         "count",       "other"),
    MetricSpec("Salt_Bridges",              "Salt bridges",                   "count",       "other"),
]


def metrics_in_group(group: str) -> list[MetricSpec]:
    return [m for m in METRICS if m.group == group]


# =============================================================================
# Path resolution helpers
# =============================================================================
def per_frame_csv(sys_name: str, rep: str = REP_TAG) -> Path:
    """Per-frame time-series CSV (one row per frame, 301 frames for 30 ns)."""
    return DATA_ROOT / sys_name / f"{DATA_PREFIX}_{rep}.csv"


def ensemble_csv(sys_name: str) -> Path:
    """Cross-replica ensemble summary CSV (mean ± SEM per metric per Time_ns)."""
    return DATA_ROOT / sys_name / f"{DATA_PREFIX}_Ensemble_Summary.csv"


def per_system_plot_dir(sys_name: str) -> Path:
    """Subfolder under production/plot/ for individual-system figures."""
    d = PLOT_ROOT / sys_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def iter_systems(filter_ma: bool | None = None) -> Iterable[SubsystemSpec]:
    """Yield SubsystemSpec in canonical order. filter_ma=True skips Gelatin."""
    for name in SYSTEM_ORDER:
        spec = SYSTEMS[name]
        if filter_ma is True and not spec.has_ma:
            continue
        if filter_ma is False and spec.has_ma:
            continue
        yield spec


# =============================================================================
# Plot styling (apply early in each plot script)
# =============================================================================
def apply_paper_style() -> None:
    """Matplotlib rcParams for paper-grade output (small but readable)."""
    import matplotlib as mpl
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
        "figure.dpi":        120,
    })


def save_fig(fig, path: Path, also_pdf: bool = True) -> None:
    """Save .png; optionally also .pdf alongside (paper-ready vector)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf and path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
