"""Pre-crosslink equilibrium summary collector.

Reads `production/Data/<sys>/GelMA_analysis_Ensemble_Summary.csv` for all 4
systems, extracts the last EQ_WINDOW_NS (= 5 ns) as the equilibrium window,
and writes a single paper-ready summary CSV.

Why a separate collector script:
  * The Ensemble_Summary CSVs hold the per-time-point mean ± SEM trajectory.
  * For paper tables / merged bar plots we need a single number per
    (system, metric): mean over the equilibrium window + propagated error.
  * Avoids hidden conventions in each plot script — they all read the same
    `pre_crosslink_summary.csv` and never touch the raw per-frame data.

Output schema (`production/plot/pre_crosslink_summary.csv`):
    system, ds_pct, has_ma, metric, label, unit, group,
    mean_eq, sem_eq, std_eq_across_frames, n_eq_frames, eq_t_start_ns,
    drift_pct_per_window, stationary, mean_stable, sem_stable, stable_window_ns

Stationarity check (per metric per system):
  * Fit linear regression to the post-warmup trajectory (default ns ≥ 5).
  * drift_pct_per_window = |slope| × EQ_WINDOW_NS / |mean| × 100
  * stationary = drift_pct_per_window < STATIONARY_THRESHOLD_PCT (default 5%)
  * For non-stationary metrics, also report `mean_stable` over a middle
    plateau window (ns 10 to TMAX-5) — useful when the last 5 ns captures
    a transient rearrangement (observed for Salt_Bridges in Gel2MA).

Window choice matches Chiu et al. 2026 §2.5 ("last 5 ns of each NPT").

Usage:
    python data_collect.py                          # all 4 systems
    python data_collect.py --window 5               # eq window
    python data_collect.py --stationary-thresh 5    # drift % threshold
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `python data_collect.py` work both as a script and as `python -m`
sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    DATA_PREFIX,
    DATA_ROOT,
    DT_PS,
    EQ_WINDOW_NS,
    METRICS,
    REP_TAG,
    SUMMARY_CSV,
    SYSTEMS,
    SYSTEM_ORDER,
    ensemble_csv,
)

# Stationarity check parameters
POST_WARMUP_NS = 5.0              # exclude first 5 ns from drift fit
STABLE_TAIL_BUFFER_NS = 5.0       # for mean_stable, also exclude last 5 ns
STATIONARY_THRESHOLD_PCT = 5.0    # |slope|*EQ_WINDOW / |mean| × 100 < this → stationary

# RDF-derived metric handling
RDF_METRIC_PREFIX = "MA_RDF_"
RDF_PEAK_R_MIN_A = 2.0            # ignore noise / overlap region below 2 Å
RDF_PEAK_R_MAX_A = 12.0           # search range upper bound (LJ-relevant)


def _ma_rdf_csv(sys_name: str) -> Path:
    """Path to the per-system MA-MA pair RDF file (written by 0511_data.py)."""
    return DATA_ROOT / sys_name / f"{DATA_PREFIX}_{REP_TAG}_MA_ensemble_RDF.csv"


def _compute_ma_rdf_peak(sys_name: str) -> tuple[float, float]:
    """Return (r_peak_A, g_r_peak) of the dominant MA–MA RDF peak.

    Strategy: global argmax of g(r) in r ∈ [RDF_PEAK_R_MIN_A, RDF_PEAK_R_MAX_A].
    This avoids the noisy r < 2 Å region (which is zero by exclusion anyway)
    and the long-r asymptote (~1 by normalisation). Returns NaN for systems
    without an MA RDF file (Gelatin).
    """
    csv_path = _ma_rdf_csv(sys_name)
    if not csv_path.exists():
        return float("nan"), float("nan")
    df = pd.read_csv(csv_path)
    if "r_A" not in df.columns or "g_r" not in df.columns:
        return float("nan"), float("nan")
    in_window = (df["r_A"] >= RDF_PEAK_R_MIN_A) & (df["r_A"] <= RDF_PEAK_R_MAX_A)
    sub = df[in_window]
    if sub.empty:
        return float("nan"), float("nan")
    i_peak = sub["g_r"].idxmax()
    return float(sub.loc[i_peak, "r_A"]), float(sub.loc[i_peak, "g_r"])


def equilibrium_mask(time_ns: pd.Series, window_ns: float) -> pd.Series:
    """Boolean mask: True for the last `window_ns` worth of frames."""
    t_max = float(time_ns.max())
    return time_ns >= (t_max - window_ns)


def integrated_autocorr_time_frames(x: np.ndarray) -> float:
    """Integrated autocorrelation time τ_int in units of frames.

    τ_int = 1 + 2·Σ_{k≥1} ρ(k), truncated at the first non-positive ρ(k)
    (the standard "initial positive sequence" estimator). For an
    uncorrelated series τ_int = 1; effective sample size N_eff = N / τ_int.

    Why this matters: consecutive MD frames (here 0.1 ns apart) are NOT
    independent. The naive SEM = std/√N treats N correlated frames as N
    independent samples and *underestimates* the error by √(τ_int). Standard
    MD practice corrects for this (Flyvbjerg & Petersen 1989; Sokal 1997).

    Capped at N/4 because τ_int is unreliable once it approaches the series
    length.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return 1.0
    # ρ(k) via unbiased autocovariance / variance
    acf = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * var)
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return float(min(max(tau, 1.0), n / 4.0))


def _drift_and_stable(t: np.ndarray, y: np.ndarray, eq_window_ns: float,
                      mean_eq: float) -> tuple[float, float, float, float, float]:
    """Per-metric stationarity diagnostic.

    Returns:
        drift_pct: |slope| * eq_window / |mean| * 100
        mean_stable: average over ns POST_WARMUP_NS .. (TMAX − STABLE_TAIL_BUFFER_NS)
                     (i.e. middle plateau, avoids both startup and tail transients)
        sem_stable:  std/sqrt(n) within the stable middle window
        stable_t0:   start of stable window (ns)
        stable_t1:   end of stable window (ns)
    """
    if len(t) < 3 or math.isnan(mean_eq) or abs(mean_eq) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Linear regression over the post-warmup full trajectory
    mask_full = t >= POST_WARMUP_NS
    if mask_full.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    slope, _intercept = np.polyfit(t[mask_full], y[mask_full], 1)
    drift_pct = abs(slope) * eq_window_ns / abs(mean_eq) * 100.0

    # Middle plateau (mean_stable)
    t_max = float(np.nanmax(t))
    stable_t0 = POST_WARMUP_NS
    stable_t1 = t_max - STABLE_TAIL_BUFFER_NS
    if stable_t1 <= stable_t0:
        return drift_pct, float("nan"), float("nan"), float("nan"), float("nan")
    mask_stable = (t >= stable_t0) & (t <= stable_t1)
    if mask_stable.sum() < 3:
        return drift_pct, float("nan"), float("nan"), stable_t0, stable_t1
    yv = y[mask_stable]
    mean_stable = float(np.nanmean(yv))
    sem_stable = float(np.nanstd(yv, ddof=1) / math.sqrt(len(yv))) if len(yv) > 1 else float("nan")
    return drift_pct, mean_stable, sem_stable, stable_t0, stable_t1


def collect_one_system(sys_name: str, window_ns: float,
                       stationary_thresh: float) -> list[dict]:
    """Per-system list of rows (one row per metric)."""
    spec = SYSTEMS[sys_name]
    csv_path = ensemble_csv(sys_name)
    if not csv_path.exists():
        print(f"  [WARN] {sys_name}: missing {csv_path.name}, skipping", file=sys.stderr)
        return []

    df = pd.read_csv(csv_path)
    if "Time_ns" not in df.columns:
        print(f"  [WARN] {sys_name}: no Time_ns column, skipping", file=sys.stderr)
        return []

    df["Time_ns"] = pd.to_numeric(df["Time_ns"], errors="coerce")
    df = df.dropna(subset=["Time_ns"]).sort_values("Time_ns")
    eq = df[equilibrium_mask(df["Time_ns"], window_ns)].copy()
    n_eq = len(eq)
    t_start = float(eq["Time_ns"].min()) if n_eq else float("nan")

    t_all = df["Time_ns"].to_numpy(dtype=float)

    # Pre-compute RDF-derived scalars once per system (avoids re-reading the
    # RDF CSV for each MA_RDF_* metric inside the loop).
    rdf_peak_r, rdf_peak_g = _compute_ma_rdf_peak(sys_name) if spec.has_ma \
        else (float("nan"), float("nan"))

    rows: list[dict] = []
    for m in METRICS:
        # RDF-derived metrics: skip the ensemble-summary lookup and stationarity
        # check entirely (RDF is a 30-ns ensemble average — single scalar).
        if m.csv_col.startswith(RDF_METRIC_PREFIX):
            if m.csv_col == "MA_RDF_FirstPeak_r_A":
                val = rdf_peak_r
            elif m.csv_col == "MA_RDF_FirstPeak_g_r":
                val = rdf_peak_g
            else:
                val = float("nan")
            if m.ma_only and not spec.has_ma:
                val = float("nan")
            rows.append({
                "system": sys_name, "ds_pct": spec.ds_pct, "has_ma": spec.has_ma,
                "metric": m.csv_col, "label": m.label, "unit": m.unit, "group": m.group,
                "mean_eq": val,
                "sem_eq": float("nan"),                   # RDF ensemble has no per-frame SEM
                "std_eq_across_frames": float("nan"),
                "n_eq_frames": n_eq, "eq_t_start_ns": t_start,
                "drift_pct_per_window": float("nan"),
                "stationary": True,                       # ensemble-averaged by construction
                "mean_stable": val,
                "sem_stable": float("nan"),
                "stable_t0_ns": float("nan"), "stable_t1_ns": float("nan"),
            })
            continue

        mean_col = f"{m.csv_col}_mean"
        sem_col = f"{m.csv_col}_sem"

        if mean_col not in eq.columns:
            mean_eq = float("nan")
            std_across = float("nan")
        else:
            vals = pd.to_numeric(eq[mean_col], errors="coerce").dropna()
            mean_eq = float(vals.mean()) if len(vals) else float("nan")
            std_across = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")

        # SEM: take the average per-frame SEM (if columns exist) over the eq window.
        # In single-replica case the Ensemble SEM column is empty → fall back to
        # the AUTOCORRELATION-CORRECTED within-trajectory SEM:
        #     SEM = std_across / sqrt(N_eff),  N_eff = n_eq / τ_int
        # NOT std/sqrt(n_eq): consecutive 0.1-ns frames are correlated
        # (τ_int ≈ 0.1–0.7 ns here → N_eff ≈ 7–40, not 51), so the naive form
        # underestimates the error by ≈ √τ_int (~1.1–2.7×, worst for slow
        # metrics like Rg). See integrated_autocorr_time_frames().
        n_eff = float("nan")
        if sem_col in eq.columns:
            sem_vals = pd.to_numeric(eq[sem_col], errors="coerce").dropna()
            sem_eq = float(sem_vals.mean()) if len(sem_vals) else float("nan")
        else:
            sem_eq = float("nan")
        if math.isnan(sem_eq) and not math.isnan(std_across) and n_eq > 1:
            tau = integrated_autocorr_time_frames(vals.to_numpy())
            n_eff = n_eq / tau
            sem_eq = std_across / math.sqrt(n_eff)

        # MA-only metrics are physically zero/undefined for Gelatin (no LMA
        # residues). Mark them NaN rather than 0 so plots skip cleanly.
        if m.ma_only and not spec.has_ma:
            rows.append({
                "system": sys_name, "ds_pct": spec.ds_pct, "has_ma": spec.has_ma,
                "metric": m.csv_col, "label": m.label, "unit": m.unit, "group": m.group,
                "mean_eq": float("nan"), "sem_eq": float("nan"),
                "std_eq_across_frames": float("nan"),
                "n_eq_frames": n_eq, "eq_t_start_ns": t_start,
                "drift_pct_per_window": float("nan"),
                "stationary": True,  # n/a — treat as fine
                "mean_stable": float("nan"), "sem_stable": float("nan"),
                "stable_t0_ns": float("nan"), "stable_t1_ns": float("nan"),
            })
            continue

        # Stationarity diagnostic on full-trajectory mean column
        if mean_col in df.columns:
            y_all = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
            drift_pct, mean_stable, sem_stable, st0, st1 = _drift_and_stable(
                t_all, y_all, window_ns, mean_eq
            )
        else:
            drift_pct = mean_stable = sem_stable = st0 = st1 = float("nan")
        stationary = (not math.isnan(drift_pct)) and (drift_pct < stationary_thresh)

        rows.append({
            "system": sys_name,
            "ds_pct": spec.ds_pct,
            "has_ma": spec.has_ma,
            "metric": m.csv_col,
            "label": m.label,
            "unit": m.unit,
            "group": m.group,
            "mean_eq": mean_eq,
            "sem_eq": sem_eq,
            "std_eq_across_frames": std_across,
            "n_eq_frames": n_eq,
            "eq_t_start_ns": t_start,
            "drift_pct_per_window": drift_pct,
            "stationary": stationary,
            "mean_stable": mean_stable,
            "sem_stable": sem_stable,
            "stable_t0_ns": st0,
            "stable_t1_ns": st1,
        })
    return rows


def print_overview(all_rows: list[dict]) -> None:
    """Compact tabular summary on stdout — sanity check.
    Cells marked with `!` if the metric failed the stationarity test."""
    print("\n=== Equilibrium summary (last 5 ns; `!` = non-stationary) ===")
    by_metric: dict[str, dict[str, str]] = {}
    for r in all_rows:
        if r["metric"] not in by_metric:
            by_metric[r["metric"]] = {"label": r["label"], "unit": r["unit"]}
        m = r["mean_eq"]
        s = r["sem_eq"]
        flag = "" if r.get("stationary", True) else "!"
        if math.isnan(m):
            cell = "        n/a"
        elif math.isnan(s):
            cell = f"{m:7.2f}    {flag}"
        else:
            cell = f"{m:7.2f}±{s:5.2f}{flag}"
        by_metric[r["metric"]][r["system"]] = cell

    header = f"{'metric':<35}" + "".join(f"{sys:>16}" for sys in SYSTEM_ORDER)
    print(header)
    print("-" * len(header))
    for metric, row in by_metric.items():
        label = row["label"][:33]
        unit = f" [{row['unit']}]" if row["unit"] else ""
        line = f"{(label + unit)[:35]:<35}" + "".join(f"{row.get(s, ''):>16}" for s in SYSTEM_ORDER)
        print(line)

    # Highlight non-stationary cells separately for readability
    print("\n=== Non-stationary metrics (mean_eq replaced with mean_stable) ===")
    non_stat = [(r["system"], r["metric"], r["label"], r["mean_eq"], r["mean_stable"],
                 r["drift_pct_per_window"], r["stable_t0_ns"], r["stable_t1_ns"])
                for r in all_rows
                if not r.get("stationary", True) and not math.isnan(r.get("mean_eq", float("nan")))]
    if not non_stat:
        print("  (all metrics stationary)")
    else:
        print(f"  {'system':<8} {'metric':<28} {'mean_eq':>9} → {'mean_stable':>11}  "
              f"{'drift_%':>8}  stable_window")
        for sysn, met, lbl, meq, mst, drift, t0, t1 in non_stat:
            meq_s = f"{meq:.2f}" if not math.isnan(meq) else "n/a"
            mst_s = f"{mst:.2f}" if not math.isnan(mst) else "n/a"
            print(f"  {sysn:<8} {met:<28} {meq_s:>9} → {mst_s:>11}  "
                  f"{drift:>7.1f}%  ns {t0:.0f}-{t1:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=EQ_WINDOW_NS,
                    help="Equilibrium window in ns (last N ns of trajectory)")
    ap.add_argument("--stationary-thresh", type=float, default=STATIONARY_THRESHOLD_PCT,
                    help="drift %% threshold below which a metric is considered stationary")
    ap.add_argument("--out", type=Path, default=SUMMARY_CSV)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for sys_name in SYSTEM_ORDER:
        rows = collect_one_system(sys_name, args.window, args.stationary_thresh)
        all_rows.extend(rows)
        if not args.quiet:
            n_filled = sum(1 for r in rows if not math.isnan(r["mean_eq"]))
            n_nonstat = sum(1 for r in rows
                            if not r.get("stationary", True)
                            and not math.isnan(r.get("mean_eq", float("nan"))))
            print(f"  {sys_name}: {n_filled}/{len(rows)} metrics in window "
                  f"(n_eq_frames={rows[0]['n_eq_frames'] if rows else 0}, "
                  f"non_stationary={n_nonstat})")

    if not all_rows:
        print("ERROR: no data collected", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    # encoding="utf-8" — labels contain Å, ², subscripts etc. that the Windows
    # cp950 default can't encode.
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows → {args.out}")

    if not args.quiet:
        print_overview(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
