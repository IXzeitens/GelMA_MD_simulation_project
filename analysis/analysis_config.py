"""Shared configuration for the post-MD analysis & plotting scripts.

All three root-level analysis scripts (`0511_data.py`, `0511_plot.py`,
`0508_new_merge.py`) import their constants, paths and selection rules from
here so that:

* SYSTEMS list / DS% labels stay consistent across scripts.
* Data path (``Data/``) and plot path (``plot/``) are defined once.
* Topology selectors (CHARMM atom names) live in one place — change them
  here and every script picks up the new chemistry.
* Smoothing / preprocessing helpers are shared rather than re-implemented
  in each script (previously two slightly-different `apply_smoothing`).

This module imports nothing heavy at load time (only pandas, numpy, scipy).
The actual MD analysis (MDAnalysis) is imported by `0511_data.py` only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)


# =============================================================================
# Directory layout
# =============================================================================
# This copy lives in `production/Data_scripts/`. System folders (Gelatin/,
# Gel1MA/, ...) and canonical `Data/` + `plot/` directories are one level up
# under `production/`, so we point ROOT_DIR at the parent.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = ROOT_DIR / "Data"            # production/Data/<sys>/
PLOT_ROOT: Path = ROOT_DIR / "plot"            # production/plot/<sys>/
DATA_PREFIX: str = "GelMA_analysis"            # CSV name prefix per replica
ENSEMBLE_SUFFIX: str = "Ensemble_Summary"      # cross-replica aggregate suffix


# =============================================================================
# Subsystem registry — one source of truth for labels / colors / DS%
# =============================================================================
@dataclass(frozen=True)
class SubsystemSpec:
    name: str
    label: str
    color: str
    ds_pct: int           # degree of substitution (% lysine modified)
    has_ma: bool          # whether the PSF contains LMA residues


# DS values for Gel*MA are the chemistry-of-record. Gel2MA was reported as
# both 66% and 67% across the older scripts; we standardise on 67% here.
# Edit if your lab convention differs — every script reads this dict.
SYSTEMS: dict[str, SubsystemSpec] = {
    "Gelatin": SubsystemSpec("Gelatin", "Gelatin\n(Control)",   "#d62728", ds_pct=0,   has_ma=False),
    "Gel1MA":  SubsystemSpec("Gel1MA",  "Gel1MA\n(DS=33%)",     "#1f77b4", ds_pct=33,  has_ma=True),
    "Gel2MA":  SubsystemSpec("Gel2MA",  "Gel2MA\n(DS=67%)",     "#ff7f0e", ds_pct=67,  has_ma=True),
    "Gel3MA":  SubsystemSpec("Gel3MA",  "Gel3MA\n(DS=100%)",    "#2ca02c", ds_pct=100, has_ma=True),
}

# Default ordered list for iteration (Gelatin first as the control).
DEFAULT_SUBSYSTEM_NAMES: tuple[str, ...] = tuple(SYSTEMS.keys())


def discover_replica_dirs(base_name: str) -> tuple[str, ...]:
    """Return every directory belonging to ``base_name`` ensemble (rep1 + clones).

    Order: original base first, then ``<base>_rep2``, ``<base>_rep3``, ... sorted
    alphabetically. ``clone_subproject.py`` outputs are picked up automatically.
    """
    dirs: list[str] = []
    if (ROOT_DIR / base_name).is_dir():
        dirs.append(base_name)
    for p in sorted(ROOT_DIR.glob(f"{base_name}_rep*")):
        if p.is_dir():
            dirs.append(p.name)
    return tuple(dirs)


# =============================================================================
# Physical parameters (sampling rate, smoothing, equilibrium windows)
# =============================================================================
DT_PS: float = 100.0                # ps per dcd frame (dcdfreq 50000 × 2 fs)
SMOOTHING_RATIO: float = 0.03       # Savitzky-Golay window = ratio × n_frames
BLOCK_SIZE_NS: float = 5.0          # time-block size for stationarity check
EQ_WINDOW_NS: float = 5.0           # canonical last-5-ns equilibrium sample

# =============================================================================
# Trajectory parts to include in analysis
# =============================================================================
# Post-HYP-fix (2026-06-08): part 1 = extended equilibration window and is
# dropped from every analysis pipeline. Production trajectories therefore
# start at `system_npt_part2.dcd`. Change ANALYSIS_SKIP_PARTS here and all
# DCD-consuming scripts (see list_production_dcds() below) follow.
ANALYSIS_SKIP_PARTS: frozenset[int] = frozenset({1})


def list_production_dcds(
    output_dir: Path,
    *,
    skip_parts: frozenset[int] = ANALYSIS_SKIP_PARTS,
) -> list[Path]:
    """Return ``system_npt_part<N>.dcd`` paths in numeric order, excluding
    ``skip_parts``.

    Natural-sort (part2 < part10) is enforced because the legacy lexical
    ``sorted(glob(...))`` reorders to part1/part10/part2/... once the
    trajectory reaches part 10. Every analysis driver should consume the
    output of this helper instead of calling glob directly so the
    "drop part 1" convention is enforced from a single place.
    """
    import re
    pat = re.compile(r"system_npt_part(\d+)\.dcd$")
    pairs: list[tuple[int, Path]] = []
    for p in output_dir.glob("system_npt_part*.dcd"):
        m = pat.match(p.name)
        if not m:
            continue
        part_n = int(m.group(1))
        if part_n in skip_parts:
            continue
        pairs.append((part_n, p))
    return [p for _, p in sorted(pairs)]


def _resolve_output_dir(sys_or_output_dir: Path) -> Path:
    """Accept either `<sys>/` or `<sys>/Output/`; return the Output dir.

    Detection is by name (`.name == "Output"`) — keeps the helper callable
    from any layer of the production tree.
    """
    return (sys_or_output_dir
            if sys_or_output_dir.name == "Output"
            else sys_or_output_dir / "Output")


def load_production_universe(
    sys_or_output_dir: Path,
    *,
    parts: str | int = "all",
    skip_parts: frozenset[int] = ANALYSIS_SKIP_PARTS,
    psf_name: str = "debug_1.psf",
):
    """Return an MDAnalysis ``Universe`` for a production system, with the
    post-HYP-fix part-skip rule enforced.

    Parameters
    ----------
    sys_or_output_dir
        Either ``production/<sys>/`` (Output/ is auto-appended) or
        ``production/<sys>/Output/`` itself.
    parts
        - ``"all"`` (default): every DCD that survives ``skip_parts``,
          concatenated into one Universe.
        - ``"last"``: only the latest DCD (largest part number) — useful
          for single-frame extractors and end-of-trajectory analyses.
        - ``int N``: just ``system_npt_part{N}.dcd``. Raises if N is in
          ``skip_parts``.
    skip_parts
        Defaults to ``ANALYSIS_SKIP_PARTS`` (drops part 1). Pass
        ``frozenset()`` to include every chunk.
    psf_name
        Override the default ``debug_1.psf`` topology filename.

    Raises
    ------
    FileNotFoundError
        If PSF is missing or zero DCDs survive the filter.
    ValueError
        If ``parts`` is an int that's also in ``skip_parts``, or any other
        unrecognised value.

    Notes
    -----
    MDAnalysis is imported lazily inside this function so that ``import
    analysis_config`` stays cheap for plot scripts that never load a
    trajectory.

    NOT used by ``data_analysis.py`` because that driver needs an extra
    mdtraj header validation pass (``_filter_readable_dcds``) to skip
    DCDs that NAMD is still actively writing.
    """
    import MDAnalysis as mda

    out = _resolve_output_dir(sys_or_output_dir)
    psf = out / psf_name
    if not psf.exists():
        raise FileNotFoundError(f"missing PSF: {psf}")

    dcds = list_production_dcds(out, skip_parts=skip_parts)
    if isinstance(parts, int):
        if parts in skip_parts:
            raise ValueError(
                f"requested parts={parts} is in skip_parts={sorted(skip_parts)}; "
                "override skip_parts=frozenset() if you really need this chunk")
        dcds = [p for p in dcds if p.stem == f"system_npt_part{parts}"]
    elif parts == "last":
        dcds = dcds[-1:] if dcds else []
    elif parts != "all":
        raise ValueError(
            f"parts must be 'all', 'last', or an int; got {parts!r}")

    if not dcds:
        raise FileNotFoundError(
            f"no DCDs under {out} (skip_parts={sorted(skip_parts)}, parts={parts!r})")

    return mda.Universe(str(psf), *[str(d) for d in dcds])


# =============================================================================
# Hydrogen-bond detection
# =============================================================================
# Relaxed from the original "chemistry-strict" 3.0 Å on 2026-05-28:
#   • 3.0 Å rejected transient/longer-range H-bonds that other MD studies
#     (gellan gum: Chiu 2026; polymer general: McGreevy review 2018) include
#     using 3.5 Å. The strict value was over-counting in the wrong direction:
#     it inflated cross-system relative differences by dropping the partially
#     formed bonds disproportionately from systems with looser packing
#     (Gel1MA/Gel3MA), making Gel2MA's 67% rebound look anomalous.
#   • 150° angle is kept — that's the standard for "well-aligned" H-bond
#     geometry and the gain from loosening to 120° is mostly bent/forked
#     bonds which add noise.
HB_DIST_STRICT_A: float = 3.5       # donor-acceptor cutoff (Å)
HB_ANGLE_DEG: float = 150.0         # donor-H-acceptor angle cutoff (deg)
CONTACT_CUTOFF_A: float = 4.5       # residue-pair contact cutoff (Å)

# MA-MA distance criterion — single unified cutoff aligned with the
# downstream Polymatic / LAMMPS crosslink decision.
#
# 7 Å : "kinetically reactive distance" — the Polymatic crosslink criterion.
#       Origin: Abbott 2013 (Polymatic) and Rukmani 2019 (PEGDA nanogels);
#       adopted directly by Chiu 2026 §2.4 for GG-MA crosslinking. Physical
#       breakdown:
#           • newly formed C-C bond              ~1.54 Å
#           • two vdW radii (sp² carbon, 1.7 Å)  ~3.40 Å
#           • radical diffusive search in 0.1 ns ~2.00 Å
#           • Σ                                  ~6.94 Å  → rounded to 7 Å
#       Any two LMA-C8 atoms on different chains within 7 Å during the
#       0.1 ns NPT relaxation per Polymatic iteration are considered to react.
#       Directly transferable from GGMA → Gel-MA because the reactive C=C
#       head is chemically identical; only the linker (amide vs ester) differs
#       by ~0.04 Å, well inside the cutoff tolerance.
#
# Earlier revisions used a separate 5 Å "tight-contact pre-network" cutoff
# for the structural cluster analysis, distinct from the 7 Å reaction
# cutoff. That split caused the pre-crosslink cluster numbers to under-
# count MA pairs sitting in the 5–7 Å window — pairs the RDF clearly shows
# (first peak ~5–6 Å) and which Polymatic will subsequently convert into
# covalent bonds. We now use the same 7 Å for both descriptors so the
# "structural cluster" picture maps directly onto the predicted crosslink
# topology. The cluster Rg metric also stabilises (boundary no longer
# cuts through the first RDF peak).
MA_CLUSTER_CUTOFF_A: float    = 7.0   # cluster analysis (Polymatic-aligned)
MA_CROSSLINK_CUTOFF_A: float  = 7.0   # Polymatic / fix bond/react decision


# =============================================================================
# Reactive-site topology (CHARMM atom names for LMA residue)
# =============================================================================
MA_RESNAME: str = "LMA"
MA_REACTIVE_C: str = "C8"           # terminal vinyl C for MA-MA proximity


# =============================================================================
# Topology selectors — kept here so chemistry edits propagate to all scripts.
#
# 2026-05-28: tightened PROTEIN_ACCEPTOR_SEL to drop generic backbone amide
# N (`name N*` / `type N*`). Backbone N is a weak H-bond acceptor because
# its lone pair is delocalised into the adjacent C=O resonance; including
# it was systematically over-counting protein H-bonds by ~10-20% and
# muddied the Gel2MA 67% rebound interpretation. We now restrict to
# carbonyl/hydroxyl O plus the imidazole side-chain N (His ND1/NE2).
# =============================================================================
# 2026-06-03: include LMA so the methacrylated-Lys residue's backbone H-bonds are
# counted in the polymer metrics. LMA is a non-standard residue → the bare `protein`
# macro silently excluded it, undercounting protein H-bonds ∝ DS (see
# _docs/METHOD_FIX_LOG_2026-06-03_LMA.md). CAVEAT: Hb_PW now includes LMA's MA-water
# bonds, so Hb_MA_Wat_Total is a SUBSET of Hb_PW (overlap, not additive).
PROTEIN_DONOR_SEL: str = "(protein or resname LMA) and (name N* or name OG* or name OH*)"
PROTEIN_HYDROGEN_SEL: str = "(protein or resname LMA) and (name H* or name HN* or type H*)"
PROTEIN_ACCEPTOR_SEL: str = "(protein or resname LMA) and (name O* or name ND1 or name NE2 or type O*)"

# CHARMM TIP3 water: oxygen is "OH2", hydrogens are "H1"/"H2".
WATER_O_SEL: str = "(resname TIP3 or resname WAT or resname HOH) and (name O* or type O*)"
WATER_H_SEL: str = "(resname TIP3 or resname WAT or resname HOH) and (name H* or type H*)"

MA_DONOR_SEL: str = f"resname {MA_RESNAME} and name N4"
MA_HYDROGEN_SEL: str = f"resname {MA_RESNAME} and (name H38 or name H37)"
# Exclude amide N4 as acceptor (donor only — resonance with carbonyl).
MA_ACCEPTOR_SEL: str = f"resname {MA_RESNAME} and (name O2 or name O5)"


# =============================================================================
# Shared helpers
# =============================================================================
def time_axis_from_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric ``Time_ns`` column exists, derived from ``Frame`` if needed.

    Replaces the legacy ``dt = 21.0 / (n-1)`` hardcoded scheme: time is now
    ``Frame * DT_PS / 1000``, so trajectories of any length (post-auto_extend)
    get correct ns labels.
    """
    if "Time_ns" in df.columns:
        df = df.copy()
        df["Time_ns"] = pd.to_numeric(df["Time_ns"], errors="coerce")
    elif "Frame" in df.columns:
        df = df.copy()
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Time_ns"] = df["Frame"] * DT_PS / 1000.0
    else:
        raise ValueError("DataFrame must contain a 'Time_ns' or 'Frame' column.")
    return df.dropna(subset=["Time_ns"])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline-standard preprocess: time axis → numeric coerce → dedupe → sort."""
    df = time_axis_from_frames(df)
    # numeric_only=True skips string columns like 'Replica' cleanly
    df = df.groupby("Time_ns").mean(numeric_only=True).reset_index()
    return df.sort_values(by="Time_ns").reset_index(drop=True)


def apply_smoothing(
    series: pd.Series,
    *,
    ratio: float = SMOOTHING_RATIO,
    min_window: int = 11,
    polyorder: int = 3,
) -> np.ndarray | pd.Series:
    """Adaptive Savitzky-Golay smoothing with NaN handling.

    Window is ``max(min_window, int(n * ratio) | 1)`` — odd, scales with trace
    length so smoothing degree is roughly constant across short/long runs.
    Falls back to a rolling mean when the series is shorter than the window.
    """
    series = pd.to_numeric(series, errors="coerce").ffill().bfill()
    n = len(series)
    if n == 0:
        return series
    window = max(min_window, int(n * ratio) | 1)  # `| 1` forces odd
    if n >= window:
        return savgol_filter(series, window_length=window, polyorder=polyorder)
    return series.rolling(window=5, min_periods=1).mean()


def rolling_average(series: pd.Series, window: int) -> pd.Series:
    """Centered rolling-mean wrapper used by the per-frame analysis pipeline."""
    return pd.Series(series).rolling(window=window, center=True).mean()
