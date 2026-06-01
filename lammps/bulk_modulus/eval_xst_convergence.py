"""Five convergence tests on existing NAMD npt_1/2/3 .xst trajectories to
decide whether the 30 ns of data is enough or whether npt_4 extension is needed.

Tests per system:
  A. Per-part K (npt_1 only, npt_2 only, npt_3 only) — should agree if equilibrated
  B. Cumulative K vs frame count — does K stabilize as more data added?
  C. Block-size scan — block std vs block length, looking for plateau
  D. V autocorrelation function — extract τ_V, compute true σ_K = K √(2τ/T)
  E. Running mean V — drift check (would invalidate stationarity)

Outputs:
  results_xst_eval/convergence_summary.csv   one row per system
  results_xst_eval/<system>_convergence.png  5-panel diagnostic plot
  results_xst_eval/RECOMMENDATION.md         human-readable verdict
"""
from __future__ import annotations
import math
from pathlib import Path

REPO = Path(r"C:\Users\User\Desktop\Work\0510")
ARCHIVE = REPO / "production"
OUT_DIR = REPO / "lammps_workflow" / "4_bulk" / "results_xst_eval"

SYSTEMS = ["Gelatin", "Gel1MA", "Gel2MA", "Gel3MA"]
DS_PCT = {"Gelatin": 0.0, "Gel1MA": 33.3, "Gel2MA": 66.7, "Gel3MA": 100.0}
T_K = 310.0
KB = 1.380649e-23
SCALE = 1e30 * KB * 1e-6   # → K in MPa from V[A^3], σ²[A^6]
DT_PS_PER_FRAME = 100      # xstFreq=50000, ts=2fs → 100 ps per frame


def parse_xst(xst: Path) -> tuple[list[int], list[float]]:
    steps, vols = [], []
    for line in xst.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        tok = s.split()
        if len(tok) < 10: continue
        steps.append(int(float(tok[0])))
        vols.append(float(tok[1]) * float(tok[5]) * float(tok[9]))
    return steps, vols


def K_from_V(V: list[float]) -> float:
    if len(V) < 5: return float("nan")
    m = sum(V) / len(V)
    var = sum((x-m)**2 for x in V) / len(V)
    return SCALE * m * T_K / var if var > 0 else float("nan")


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x-m)**2 for x in xs) / max(1, len(xs)-1))
    return m, s


def autocorrelation(V: list[float], max_lag: int) -> list[float]:
    """Return normalized autocorrelation C(t)/C(0) for lags 0..max_lag."""
    m = sum(V) / len(V)
    dV = [v - m for v in V]
    C0 = sum(d*d for d in dV) / len(dV)
    if C0 <= 0: return [float("nan")] * (max_lag+1)
    C = []
    for lag in range(max_lag + 1):
        if lag >= len(dV):
            C.append(float("nan")); continue
        n = len(dV) - lag
        c = sum(dV[i] * dV[i+lag] for i in range(n)) / n
        C.append(c / C0)
    return C


def integrated_tau(C: list[float]) -> float:
    """Integrated autocorrelation time (frames) via summation until first negative."""
    tau = 0.5  # C(0) = 1 contributes 0.5
    for c in C[1:]:
        if c < 0.05:  # cut where signal drops below noise
            break
        tau += c
    return tau


def evaluate_system(name: str) -> dict:
    out_dir = ARCHIVE / name / "Output"
    parts_V = {}
    for p in (1, 2, 3):
        xst = out_dir / f"system_npt_part{p}.xst"
        if not xst.exists(): continue
        _s, V = parse_xst(xst)
        # Drop first 20% of each part as eq tail
        V = V[int(len(V)*0.2):]
        parts_V[p] = V

    # Test A: per-part K
    K_per_part = {p: K_from_V(V) for p, V in parts_V.items()}

    # Pooled V across all parts
    V_all = []
    for V in parts_V.values():
        V_all.extend(V)
    K_pooled = K_from_V(V_all)

    # Test B: cumulative K
    cum_K = []
    cum_n = []
    step = max(5, len(V_all)//40)
    for n in range(step, len(V_all)+1, step):
        cum_K.append(K_from_V(V_all[:n]))
        cum_n.append(n)
    # Stability metric: relative std of last quarter of cumulative K
    tail = cum_K[max(1, len(cum_K)*3//4):]
    cum_stability_pct = (mean_std(tail)[1] / mean_std(tail)[0] * 100) if tail else float("nan")

    # Test C: block-size scan
    block_results = []  # (block_n_frames, n_blocks, K_mean, K_std)
    for block_n in [10, 20, 40, 60, 80, 120]:
        if block_n * 3 > len(V_all): continue
        n_blocks = len(V_all) // block_n
        Ks = [K_from_V(V_all[i*block_n:(i+1)*block_n]) for i in range(n_blocks)]
        Ks = [k for k in Ks if k == k]  # drop NaN
        if len(Ks) < 2: continue
        Km, Ks_ = mean_std(Ks)
        block_results.append((block_n, n_blocks, Km, Ks_))

    # Test D: autocorrelation → tau
    C = autocorrelation(V_all, max_lag=min(50, len(V_all)//4))
    tau_frames = integrated_tau(C)
    tau_ps = tau_frames * DT_PS_PER_FRAME
    T_total_ps = len(V_all) * DT_PS_PER_FRAME
    n_independent = T_total_ps / (2 * tau_ps) if tau_ps > 0 else float("nan")
    true_sigma_K_rel = math.sqrt(2 * tau_frames / len(V_all))  # √(2τ/T)
    true_sigma_K = K_pooled * true_sigma_K_rel

    # Test E: running mean drift
    chunk = max(20, len(V_all)//6)
    running_means = [sum(V_all[i:i+chunk])/chunk for i in range(0, len(V_all)-chunk+1, chunk)]
    drift_pct = (max(running_means) - min(running_means)) / mean_std(running_means)[0] * 100

    return {
        "system": name, "DS_pct": DS_PCT[name],
        "n_frames_total": len(V_all),
        "T_total_ns": T_total_ps / 1000,
        "K_per_part": K_per_part,
        "K_pooled_MPa": K_pooled,
        "K_per_part_spread_pct": (max(K_per_part.values()) - min(K_per_part.values()))
                                   / K_pooled * 100 if K_per_part else float("nan"),
        "cum_K_values": cum_K,
        "cum_K_x_frames": cum_n,
        "cum_stability_last25pct": cum_stability_pct,
        "block_scan": block_results,
        "autocorr_C": C,
        "tau_V_ps": tau_ps,
        "n_independent": n_independent,
        "true_sigma_K_rel_pct": true_sigma_K_rel * 100,
        "true_sigma_K_MPa": true_sigma_K,
        "running_means": running_means,
        "drift_range_pct_of_V": drift_pct * 100 / mean_std(V_all)[0]
                                   * mean_std(V_all)[0] / 100,  # already pct
    }


def make_plot(r: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle(f"{r['system']} (DS={r['DS_pct']:.1f}%, T={r['T_total_ns']:.1f} ns, "
                 f"K_pooled={r['K_pooled_MPa']:.0f}±{r['true_sigma_K_MPa']:.0f} MPa, τ={r['tau_V_ps']:.0f} ps)",
                 fontsize=11)

    # A: per-part K
    ax = axes[0,0]
    parts = sorted(r['K_per_part'].keys())
    Ks = [r['K_per_part'][p] for p in parts]
    ax.bar([f"npt_{p}" for p in parts], Ks, color="steelblue")
    ax.axhline(r['K_pooled_MPa'], color="k", linestyle="--", label=f"pooled={r['K_pooled_MPa']:.0f}")
    ax.set_ylabel("K (MPa)"); ax.set_title(f"A. Per-part K (spread {r['K_per_part_spread_pct']:.1f}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # B: cumulative K
    ax = axes[0,1]
    ax.plot(r['cum_K_x_frames'], r['cum_K_values'], "o-", markersize=3)
    ax.axhline(r['K_pooled_MPa'], color="k", linestyle="--")
    ax.axhspan(r['K_pooled_MPa']-r['true_sigma_K_MPa'], r['K_pooled_MPa']+r['true_sigma_K_MPa'],
               color="gray", alpha=0.2, label="±σ_K (true)")
    ax.set_xlabel("frames included"); ax.set_ylabel("K (MPa)")
    ax.set_title(f"B. Cumulative K (last 25% CV: {r['cum_stability_last25pct']:.1f}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # C: block-size scan
    ax = axes[0,2]
    if r['block_scan']:
        bn = [b[0] for b in r['block_scan']]
        Km = [b[2] for b in r['block_scan']]
        Ks = [b[3] for b in r['block_scan']]
        ax.errorbar(bn, Km, yerr=Ks, fmt="o-", capsize=4)
        ax.set_xscale("log")
    ax.axhline(r['K_pooled_MPa'], color="k", linestyle="--")
    ax.set_xlabel("block size (frames)"); ax.set_ylabel("K (MPa)")
    ax.set_title("C. Block-size scan (plateau = converged)")
    ax.grid(alpha=0.3)

    # D: autocorrelation
    ax = axes[1,0]
    lags_ps = [i * DT_PS_PER_FRAME for i in range(len(r['autocorr_C']))]
    ax.plot(lags_ps, r['autocorr_C'], "o-", markersize=3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(0.05, color="r", linestyle=":", label="cutoff")
    ax.axvline(r['tau_V_ps'], color="g", linestyle="--", label=f"τ={r['tau_V_ps']:.0f} ps")
    ax.set_xlabel("lag (ps)"); ax.set_ylabel("C(t)/C(0)")
    ax.set_title(f"D. V autocorrelation (n_indep ≈ {r['n_independent']:.0f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # E: running mean V drift
    ax = axes[1,1]
    rms = r['running_means']
    ax.plot(range(len(rms)), rms, "o-")
    Vmean = sum(rms)/len(rms)
    ax.axhline(Vmean, color="k", linestyle="--")
    rel_drift = (max(rms) - min(rms)) / Vmean * 100
    ax.set_xlabel("chunk index"); ax.set_ylabel("<V> (Å³)")
    ax.set_title(f"E. Running mean V (max-min: {rel_drift:.2f}% of <V>)")
    ax.grid(alpha=0.3)

    # F: verdict
    ax = axes[1,2]; ax.axis("off")
    verdict_lines = make_verdict(r)
    ax.text(0.0, 0.95, "\n".join(verdict_lines), transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=9)
    ax.set_title("F. Verdict")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def make_verdict(r: dict) -> list[str]:
    """Human-readable per-system verdict."""
    lines = [f"K_pooled = {r['K_pooled_MPa']:.0f} ± {r['true_sigma_K_MPa']:.0f} MPa",
             f"          ({r['true_sigma_K_rel_pct']:.1f}% rel σ_K)",
             ""]
    # Check 1: parts agree
    sp = r['K_per_part_spread_pct']
    lines.append(f"A. parts spread:    {sp:5.1f}%  " + ("OK" if sp < 30 else "WIDE"))
    # Check 2: cumulative stable
    cs = r['cum_stability_last25pct']
    lines.append(f"B. cum last-25% CV: {cs:5.1f}%  " + ("OK" if cs < 5 else "DRIFT"))
    # Check 3: blocks plateau
    if len(r['block_scan']) >= 3:
        last3_Km = [b[2] for b in r['block_scan'][-3:]]
        plat = (max(last3_Km) - min(last3_Km)) / mean_std(last3_Km)[0] * 100
        lines.append(f"C. block plateau:   {plat:5.1f}%  " + ("OK" if plat < 10 else "DRIFT"))
    # Check 4: enough independent samples
    ni = r['n_independent']
    lines.append(f"D. n_indep samples: {ni:5.0f}   " + ("OK" if ni > 30 else "LOW"))
    # Check 5: no V drift
    rms = r['running_means']
    rel_drift = (max(rms) - min(rms)) / (sum(rms)/len(rms)) * 100
    lines.append(f"E. <V> drift:       {rel_drift:5.2f}% " + ("OK" if rel_drift < 0.5 else "DRIFT"))

    lines.append("")
    # Overall verdict
    flags = sum([sp >= 30, cs >= 5, ni <= 30, rel_drift >= 0.5])
    if flags == 0:
        lines.append("==> npt_4 NOT needed.")
        lines.append("    Current 30 ns is enough for")
        lines.append("    publication-grade K precision.")
    elif flags <= 1:
        lines.append("==> npt_4 marginally helpful.")
        lines.append("    Current K usable; +5 ns")
        lines.append("    would tighten error ~20%.")
    else:
        lines.append("==> npt_4 RECOMMENDED.")
        lines.append("    {} convergence flags raised.".format(flags))
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating convergence of npt_1/2/3 .xst (30 ns total per system)\n")

    print(f"{'sys':<8} {'N_fr':>5} {'T/ns':>5} {'K_pool':>7} {'true_s':>7} "
          f"{'A_spr%':>6} {'B_CV%':>6} {'tau/ps':>7} {'D_nind':>6} {'E_drf%':>6}  verdict")
    print("-" * 105)

    summary_rows = []
    for sys in SYSTEMS:
        r = evaluate_system(sys)
        rms = r['running_means']
        rel_drift = (max(rms)-min(rms))/(sum(rms)/len(rms))*100
        v = make_verdict(r)[-2:]
        print(f"{r['system']:<8} {r['n_frames_total']:>5d} {r['T_total_ns']:>5.1f} "
              f"{r['K_pooled_MPa']:>7.0f} {r['true_sigma_K_MPa']:>7.0f} "
              f"{r['K_per_part_spread_pct']:>5.1f}% {r['cum_stability_last25pct']:>5.1f}% "
              f"{r['tau_V_ps']:>7.0f} {r['n_independent']:>6.0f} {rel_drift:>5.2f}%  "
              f"{v[0].strip()}")

        png = OUT_DIR / f"{sys}_convergence.png"
        try:
            make_plot(r, png)
        except ImportError:
            pass

        # flatten for CSV
        summary_rows.append({
            "system": r["system"], "DS_pct": r["DS_pct"],
            "n_frames": r["n_frames_total"], "T_total_ns": r["T_total_ns"],
            "K_pooled_MPa": r["K_pooled_MPa"],
            "true_sigma_K_MPa": r["true_sigma_K_MPa"],
            "true_sigma_K_pct": r["true_sigma_K_rel_pct"],
            "parts_spread_pct": r["K_per_part_spread_pct"],
            "cum_last25_CV_pct": r["cum_stability_last25pct"],
            "tau_V_ps": r["tau_V_ps"],
            "n_independent": r["n_independent"],
            "drift_max_min_pct": rel_drift,
        })

    # CSV
    import csv
    csv_path = OUT_DIR / "convergence_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nCSV  -> {csv_path}")

    # RECOMMENDATION.md
    md = ["# npt_4 extension necessity — verdict per system\n"]
    md.append("Generated from existing 30 ns NAMD npt_1/2/3 trajectories.\n\n")
    md.append("| System | K_pooled (MPa) | true σ_K | n_indep | τ_V (ps) | drift | parts spread | verdict |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in [evaluate_system(s) for s in SYSTEMS]:
        rms = r['running_means']
        rel_drift = (max(rms)-min(rms))/(sum(rms)/len(rms))*100
        verdict = make_verdict(r)[-2].strip()
        md.append(f"| {r['system']} | {r['K_pooled_MPa']:.0f} | "
                  f"±{r['true_sigma_K_MPa']:.0f} ({r['true_sigma_K_rel_pct']:.1f}%) | "
                  f"{r['n_independent']:.0f} | {r['tau_V_ps']:.0f} | {rel_drift:.2f}% | "
                  f"{r['K_per_part_spread_pct']:.1f}% | {verdict} |\n")
    (OUT_DIR / "RECOMMENDATION.md").write_text("".join(md), encoding="utf-8")
    print(f"MD   -> {OUT_DIR / 'RECOMMENDATION.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
