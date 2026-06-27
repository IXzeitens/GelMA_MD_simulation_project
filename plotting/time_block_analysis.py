"""Time-block convergence analysis.

For each (system, metric) reads the full-trajectory ``GelMA_analysis_Ensemble
_Summary.csv``, splits the post-warmup portion (default ns >= 5) into N equal
blocks, and grades convergence by three independent criteria:

  * drift_pct_per_5ns   — |linear slope through block means| * 5 ns / |mean| * 100
  * block_scatter_pct   — stddev(block_means) / |mean| * 100
  * halves_diff_pct     — |mean(first N/2 blocks) - mean(last N/2 blocks)| / |mean| * 100

Each criterion is intentionally redundant — slope catches monotonic drift,
scatter catches noisy oscillation, halves catches step-shaped non-stationarity.
A metric passes only if ALL THREE are small.

Grade rubric (worst of the three):
  A   < 2%    well converged, paper-ready
  B   < 5%    marginal, still defensible if drift sign is benign
  C   < 10%   not converged, needs more sampling
  D   >= 10%  far from plateau, extend significantly

Outputs:
  * time_block_summary.csv             one row per (system, metric)

The per-metric facet figure (Figure_TimeBlock_Convergence.png) is drawn by
`plot_merge.py --figs CONVERGENCE`, which reads the CSV written here. This
script is the numeric half (grade + CSV); plot_merge owns all the figures.

Usage:
  python time_block_analysis.py                    # default 5 blocks, warmup 5 ns
  python time_block_analysis.py --n-blocks 8       # finer blocks for long traj
  python time_block_analysis.py --warmup 10        # skip first 10 ns instead

Reads from production/Data/<sys>/GelMA_analysis_Ensemble_Summary.csv (regenerated
by data_analysis.py). Does NOT touch the raw trajectory.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    METRICS,
    PLOT_ROOT,
    SYSTEMS,
    SYSTEM_ORDER,
    ensemble_csv,
)

POST_WARMUP_NS_DEFAULT = 5.0
N_BLOCKS_DEFAULT = 5

# Grade thresholds applied to the WORST of the three convergence criteria.
GRADE_THRESHOLDS = [(2.0, "A"), (5.0, "B"), (10.0, "C")]   # else "D"


def _grade(*pcts: float) -> str:
    worst = max((abs(x) for x in pcts if x is not None and not math.isnan(x)),
                default=float("nan"))
    if math.isnan(worst):
        return "?"
    for thr, letter in GRADE_THRESHOLDS:
        if worst < thr:
            return letter
    return "D"


def _block_means(time_ns: np.ndarray, vals: np.ndarray, edges: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (centers, means, sems) for N blocks defined by edges (N+1)."""
    n = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n, np.nan)
    sems = np.full(n, np.nan)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        # closed-right only on last block so we don't drop the final frame
        if i < n - 1:
            mask = (time_ns >= lo) & (time_ns < hi)
        else:
            mask = (time_ns >= lo) & (time_ns <= hi)
        v = vals[mask]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        means[i] = float(v.mean())
        sems[i] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return centers, means, sems


def analyze_one_system(sys_name: str, warmup_ns: float, n_blocks: int
                       ) -> list[dict]:
    csv_path = ensemble_csv(sys_name)
    if not csv_path.exists():
        print(f"  [WARN] {sys_name}: missing {csv_path.name}, skipping",
              file=sys.stderr)
        return []
    df = pd.read_csv(csv_path)
    if "Time_ns" not in df.columns:
        print(f"  [WARN] {sys_name}: no Time_ns column, skipping",
              file=sys.stderr)
        return []
    df["Time_ns"] = pd.to_numeric(df["Time_ns"], errors="coerce")
    df = df.dropna(subset=["Time_ns"]).sort_values("Time_ns").reset_index(drop=True)
    df = df[df["Time_ns"] >= warmup_ns].copy()
    if df.empty:
        print(f"  [WARN] {sys_name}: no frames past warmup={warmup_ns} ns",
              file=sys.stderr)
        return []
    t_min, t_max = float(df["Time_ns"].min()), float(df["Time_ns"].max())
    if t_max - t_min < n_blocks * 0.5:
        print(f"  [WARN] {sys_name}: only {t_max - t_min:.1f} ns post-warmup, "
              f"too short for {n_blocks} blocks", file=sys.stderr)
        return []
    edges = np.linspace(t_min, t_max, n_blocks + 1)
    block_width = (t_max - t_min) / n_blocks

    spec = SYSTEMS[sys_name]
    t_arr = df["Time_ns"].to_numpy(dtype=float)
    rows: list[dict] = []
    for m in METRICS:
        mean_col = f"{m.csv_col}_mean"
        if mean_col not in df.columns:
            continue
        # Skip MA-only metrics for the Gelatin control
        if m.ma_only and not spec.has_ma:
            continue
        vals = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
        # Skip metrics that are identically zero or all-NaN in this trajectory
        if np.all(np.isnan(vals)) or np.nanmax(np.abs(vals)) < 1e-12:
            continue

        centers, b_means, b_sems = _block_means(t_arr, vals, edges)
        total_mean = float(np.nanmean(b_means))
        if abs(total_mean) < 1e-12:
            continue

        # --- 3 independent convergence criteria ---
        valid = ~np.isnan(b_means)
        if valid.sum() >= 2:
            slope = float(np.polyfit(centers[valid], b_means[valid], 1)[0])
            drift_pct = abs(slope) * 5.0 / abs(total_mean) * 100.0
        else:
            slope = float("nan")
            drift_pct = float("nan")

        if valid.sum() >= 2:
            scatter_pct = float(np.nanstd(b_means, ddof=1)) / abs(total_mean) * 100.0
        else:
            scatter_pct = float("nan")

        h = n_blocks // 2
        first_half = float(np.nanmean(b_means[:h])) if h > 0 else float("nan")
        second_half = float(np.nanmean(b_means[-h:])) if h > 0 else float("nan")
        if math.isnan(first_half) or math.isnan(second_half):
            halves_pct = float("nan")
        else:
            halves_pct = abs(second_half - first_half) / abs(total_mean) * 100.0

        grade = _grade(drift_pct, scatter_pct, halves_pct)

        row = {
            "system": sys_name,
            "ds_pct": spec.ds_pct,
            "metric": m.csv_col,
            "label": m.label,
            "unit": m.unit,
            "group": m.group,
            "n_blocks": n_blocks,
            "block_width_ns": block_width,
            "warmup_ns": warmup_ns,
            "trajectory_end_ns": t_max,
            "total_mean": total_mean,
            "slope_per_ns": slope,
            "drift_pct_per_5ns": drift_pct,
            "block_scatter_pct": scatter_pct,
            "halves_diff_pct": halves_pct,
            "grade": grade,
        }
        for i in range(n_blocks):
            row[f"block_{i + 1}_t_ns"] = float(centers[i])
            row[f"block_{i + 1}_mean"] = float(b_means[i]) if not math.isnan(b_means[i]) else float("nan")
            row[f"block_{i + 1}_sem"] = float(b_sems[i]) if not math.isnan(b_sems[i]) else float("nan")
        rows.append(row)
    return rows


def print_grade_overview(df: pd.DataFrame) -> None:
    print("\n=== Convergence grades (worst of drift / scatter / halves) ===")
    metrics = list(dict.fromkeys(df["metric"].tolist()))
    header = f"{'metric':<32}" + "".join(f"{s:>10}" for s in SYSTEM_ORDER)
    print(header)
    print("-" * len(header))
    for met in metrics:
        line = f"{met[:32]:<32}"
        for s in SYSTEM_ORDER:
            row = df[(df["metric"] == met) & (df["system"] == s)]
            if row.empty:
                line += f"{'-':>10}"
            else:
                line += f"{row.iloc[0]['grade']:>10}"
        print(line)

    # Per-system grade tally for quick triage
    print("\n=== Per-system grade tally ===")
    print(f"{'system':<10}" + f"{'A':>5}{'B':>5}{'C':>5}{'D':>5}{'?':>5}{'metrics':>10}")
    for s in SYSTEM_ORDER:
        sub = df[df["system"] == s]
        if sub.empty:
            continue
        counts = sub["grade"].value_counts().to_dict()
        line = f"{s:<10}"
        for g in ("A", "B", "C", "D", "?"):
            line += f"{counts.get(g, 0):>5}"
        line += f"{len(sub):>10}"
        print(line)

    # Worst offenders — show only C/D
    bad = df[df["grade"].isin(("C", "D"))].copy()
    if bad.empty:
        print("\n  All metrics graded A or B.")
        return
    bad = bad.sort_values(
        by=["grade", "drift_pct_per_5ns"], ascending=[False, False])
    print("\n=== Metrics needing more sampling (grade C/D) ===")
    print(f"  {'system':<8} {'metric':<28} {'grade':>5} {'drift%':>7} "
          f"{'scatter%':>8} {'halves%':>7}")
    for _, r in bad.iterrows():
        print(f"  {r['system']:<8} {r['metric'][:28]:<28} {r['grade']:>5} "
              f"{r['drift_pct_per_5ns']:>7.2f} "
              f"{r['block_scatter_pct']:>8.2f} "
              f"{r['halves_diff_pct']:>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=float, default=POST_WARMUP_NS_DEFAULT,
                    help="Skip first N ns (equilibration window). Default 5.")
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS_DEFAULT,
                    help="Number of equal-time blocks. Default 5.")
    ap.add_argument("--out-csv", type=Path,
                    default=PLOT_ROOT / "time_block_summary.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for sys_name in SYSTEM_ORDER:
        rows = analyze_one_system(sys_name, args.warmup, args.n_blocks)
        all_rows.extend(rows)
        if not args.quiet:
            n_total = len(rows)
            grades = {r["grade"] for r in rows}
            print(f"  {sys_name}: {n_total} metrics analysed "
                  f"(grades present: {','.join(sorted(grades))})")

    if not all_rows:
        print("ERROR: no data collected", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nwrote {len(df)} rows -> {args.out_csv.name}")
    print("  (figure: run  python plot_merge.py --figs CONVERGENCE)")

    if not args.quiet:
        print_grade_overview(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
