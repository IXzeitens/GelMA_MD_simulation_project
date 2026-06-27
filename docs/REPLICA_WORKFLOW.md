# Replica Workflow (rep2)

End-to-end procedure to add a second independent replica (`rep2`) on top of
the existing 4 rep1 subprojects (Gelatin, Gel1MA, Gel2MA, Gel3MA), so that
ensemble averages can be reported as **mean ± SEM (N=2 independent
Packmol seeds)**.

> `<workspace>` below = the directory that holds both `gelma_md/` and the
> simulation `production/` tree (the same path you can export as `GELMA_REPO`).

Total wall-clock estimate on the RTX 5060 Ti: **~50 GPU hours**
(~2.5 days) end to end if rep1 is already at 60 ns.

---

## Prerequisites (do these first)

1. **rep1 has converged at 60 ns** — re-run `auto_extend.py` once on top of
   the current 50 ns. Then:
   ```powershell
   cd <workspace>\production\Data_scripts
   python data_analysis.py
   cd ..\plot
   python data_collect.py
   ```
   Open `pre_crosslink_summary.csv` and confirm:
   * No metric in tier T1 (conformational) shows drift > 5%
   * No metric in tier T2 (H-bond) shows drift > 10%
   * No metric in tier T3 (inter-chain) shows drift > 15%

   Specifically watch: `Gel2MA Hb_Inter_Strict`, `Gel2MA Salt_Bridges`,
   `Gel3MA Salt_Bridges` — these were the 3 still drifting at 50 ns.

2. **Disk space** — each clone ≈ 3 GB after 60 ns of trajectory; need at
   least 15 GB free under `production/`.

3. **GPU available exclusively** — single-GPU sequential workflow; close
   anything else using the 5060 Ti.

---

## Phase 1 — Clone the four subprojects (minutes)

Use the production-aligned `clone_subproject.py`:

```powershell
cd <workspace>
python production/sim_scripts/clone_subproject.py --rep rep2 --seed-base 20260615
```

This creates four new sibling directories in `production/`:

```
production/
├── Gelatin/           (rep1, untouched)
├── Gelatin_rep2/      <-- NEW
├── Gel1MA/
├── Gel1MA_rep2/       <-- NEW
├── Gel2MA/
├── Gel2MA_rep2/       <-- NEW
├── Gel3MA/
└── Gel3MA_rep2/       <-- NEW
```

The clone:
* Copies `config.json`, `main.py`, `input/`, `packmol/`, `script/`
* Excludes `Output/`, `temp/`, all `.dcd/.coor/.vel/.xst/.log/.BAK/.old`
  files (these get regenerated)
* Injects `seed 20260815` into each clone's `script/NVT.conf` (base +
  100 × 2; deterministic NAMD velocity init per replica)
* Leaves `packmol_config.inp` with `seed -1` — **deliberate**: this gives
  fresh independent chain packing per replica, which is the whole point

### Verify clone

```powershell
# 4 new dirs exist
ls production | Select-String "_rep2"

# Each has essential parts
ls production/Gel1MA_rep2 | Select-String "config|main|input|packmol|script"

# NVT.conf has seed injected
Get-Content production/Gel1MA_rep2/script/NVT.conf | Select-String "seed"
```

Expected: each clone has all 5 essentials, NVT.conf shows
`seed                20260815` (or whatever seed_base + offset you picked).

---

## Phase 2 — Run Workflow 1 on each rep2 clone (~20 hours)

Each clone needs to go through: `packmol → psfgen → autoionize → minimize
→ NVT (1 ns) → NPT part 1 + 2 (= 20 ns)`. This is what `main.py` does for
the original 4 systems.

### Heads-up: `main.py` is interactive

`Gel1MA_rep2/main.py` calls `pipeline.run(paths)` which defaults to
`interactive=True`, prompting:

```
[Input] Gel-chain 數量 (default=12):
[Input] 目標濃度 wt% (default=7.5):
```

Two options:

**Option A — Pipe blank stdin to accept defaults (no code change):**
```powershell
foreach ($sys in @("Gelatin_rep2","Gel1MA_rep2","Gel2MA_rep2","Gel3MA_rep2")) {
    Write-Host "=== Running main.py for $sys ==="
    Push-Location "production/$sys"
    "`r`n`r`n" | python main.py
    Pop-Location
}
```

**Option B — Edit `pipeline.py` in each clone to use `interactive=False`:**
Change line 74 of each `production/<sys>_rep2/script/pipeline.py`:
```python
if interactive:           →     if False:
    cfg = prompt_user_settings(cfg)
```
(Then revert later if you want main.py interactive again for new subprojects.)

### Expected per-clone timing

* Packmol + psfgen + autoionize: ~30 sec
* Energy minimization: ~5 sec (NAMD3 + CUDA)
* NVT 1 ns: ~15 min
* NPT part 1 + part 2 (10 ns each = 20 ns total): ~5 hours each

Total per clone: **~5 hours**. Four clones sequentially: **~20 hours**.

### Verify after each clone runs

```powershell
ls production/Gel1MA_rep2/Output | Select-String "system_npt_part2"
# expect: system_npt_part2.dcd, .coor, .restart.coor, .vel, .xsc
```

---

## Phase 3 — Extend rep2 to 60 ns (~40 hours)

`auto_extend.py` supports a `--rep` filter that restricts to one replica
cohort. Each call extends by one 10-ns NPT chunk.

```powershell
# Confirm plan first
python production/sim_scripts/auto_extend.py --rep rep2 --dry-run

# Run 4 extensions to go from part 2 (20 ns) to part 6 (60 ns)
# Each call is ~10 hours wall clock; can be invoked back-to-back.
python production/sim_scripts/auto_extend.py --rep rep2 --max-part 3
python production/sim_scripts/auto_extend.py --rep rep2 --max-part 4
python production/sim_scripts/auto_extend.py --rep rep2 --max-part 5
python production/sim_scripts/auto_extend.py --rep rep2 --max-part 6
```

`--max-part` caps the target — so even if you run `--max-part 3` twice by
accident, the second call is a no-op (everyone's already at part 3).

### Verify all rep2 reach 60 ns

```powershell
foreach ($sys in @("Gelatin_rep2","Gel1MA_rep2","Gel2MA_rep2","Gel3MA_rep2")) {
    $latest = ls "production/$sys/Output/system_npt_part*.restart.coor" |
              Sort-Object Name -Descending | Select-Object -First 1
    Write-Host "$sys -> $($latest.Name)"
}
# expect: each shows system_npt_part6.restart.coor
```

---

## Phase 4 — Re-run analysis with rep2 included

⚠ **The current `data_analysis.py` iterates `analysis_config.SYSTEMS`,
which only contains the 4 base systems.** Out of the box it will analyse
rep1 only — rep2 will be silently skipped.

### Fix: extend SYSTEMS dict

Edit `production/Data_scripts/analysis_config.py` and add 4 entries:

```python
SYSTEMS: dict[str, SubsystemSpec] = {
    "Gelatin":      SubsystemSpec("Gelatin",      "Gelatin\n(Control)",   "#d62728", ds_pct=0,   has_ma=False),
    "Gel1MA":       SubsystemSpec("Gel1MA",       "Gel1MA\n(DS=33%)",     "#1f77b4", ds_pct=33,  has_ma=True),
    "Gel2MA":       SubsystemSpec("Gel2MA",       "Gel2MA\n(DS=67%)",     "#ff7f0e", ds_pct=67,  has_ma=True),
    "Gel3MA":       SubsystemSpec("Gel3MA",       "Gel3MA\n(DS=100%)",    "#2ca02c", ds_pct=100, has_ma=True),
    # === rep2 additions ===
    "Gelatin_rep2": SubsystemSpec("Gelatin_rep2", "Gelatin rep2",         "#d62728", ds_pct=0,   has_ma=False),
    "Gel1MA_rep2":  SubsystemSpec("Gel1MA_rep2",  "Gel1MA rep2",          "#1f77b4", ds_pct=33,  has_ma=True),
    "Gel2MA_rep2":  SubsystemSpec("Gel2MA_rep2",  "Gel2MA rep2",          "#ff7f0e", ds_pct=67,  has_ma=True),
    "Gel3MA_rep2":  SubsystemSpec("Gel3MA_rep2",  "Gel3MA rep2",          "#2ca02c", ds_pct=100, has_ma=True),
}
```

**Note**: this is per-subproject-as-system. The cleaner alternative is to
treat rep2 as a true replica of the same system (so analysis goes to
`Data/Gel1MA/GelMA_analysis_rep2.csv` instead of
`Data/Gel1MA_rep2/GelMA_analysis_rep1.csv`). The legacy
`resolve_replica_input_dir` already understands this nested form. If you
prefer that route, leave SYSTEMS unchanged and instead pass `rep=rep2` to
the analysis pipeline — but the current `data_analysis.py` driver only
iterates `REPLICAS = ("rep1", "rep2", "rep3")`, so this MIGHT already
work if `<sys>_rep2/Output/` is treated as `<sys>/rep2`. **Check before
running** by looking at `resolve_replica_input_dir` in data_analysis.py.

Then:
```powershell
cd <workspace>\production\Data_scripts
python data_analysis.py
```

This:
* Iterates 8 subsystems (4 rep1 + 4 rep2)
* Writes per-frame CSVs into `Data/<sys>/` and `Data/<sys>_rep2/`
* Auto-aggregates `<base>_rep1`/`<base>_rep2` into ensemble summary
  (depends on which SYSTEMS-dict route you took)

---

## Phase 5 — Refresh summary + plots

```powershell
cd ..\plot
python data_collect.py     # rebuilds pre_crosslink_summary.csv
python plot_merge.py       # M1, M2, M3 with N=2 error bars
```

If you used the per-subproject-as-system route (Phase 4 option 1), `data_collect.py` will produce **8 rows per metric** (one per subsystem). To get **N=2 ensemble means**, you'll need a small wrapper that groups by `ds_pct` and re-aggregates. Easiest: edit
`production/plot/data_collect.py` to add a `--ensemble-by ds_pct` flag, or
just do it ad-hoc in a notebook.

If you used the nested-replica route (Phase 4 option 2),
`Ensemble_Summary.csv` per base system already contains rep1+rep2 means
with proper SEM, and downstream tooling Just Works.

---

## Phase 6 — Sanity check the ensemble

After plots are regenerated, look for:

* **Error bars should be larger** than rep1-alone — this is real
  cross-seed variance becoming visible
* **Means should be similar to rep1 alone** for metrics that were robust
  (Rg, Ree, Lp, MA SASA, MA-MA NN)
* **Means may shift for metrics that were marginal** in rep1 (Inter Hb,
  Salt bridges) — this is the new variance being captured properly

If rep2 shows wildly different chain organisation from rep1 (e.g.
inter-residue contacts off by > 2× SEM), that's the cross-seed variance
the paper now has the right to report.

---

## Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot to update SYSTEMS dict | `data_analysis.py` finishes in ~5 sec | Edit `analysis_config.py` per Phase 4 |
| main.py prompts hang the batch | First clone never finishes | Use stdin pipe (Option A) or edit pipeline.py (Option B) |
| Disk full during Phase 3 | NAMD writes truncate, restart files missing | Free up ≥ 15 GB before starting |
| Wrong seed-base picked | Two replicas with the same NAMD seed | Choose seed-base that doesn't clash; `20260615` is the recommended value here (gives 20260815 for rep2, 20260915 for rep3) |
| Re-ran `auto_extend` past 60 ns | Trajectories now mismatched between rep1 and rep2 | Both `--max-part 6` cap; or trim analysis to 60 ns window |

---

## When to skip rep2 and ship

If after the 60 ns rep1 extension all three drift-flagged metrics fall
below tier thresholds (Gel2MA Inter Hb < 10%, both Salt bridges < 15%),
**you can defensibly publish single-replica results with a Methods note**
that N=1 was used and Packmol seed was random (`seed -1`), consistent
with Chiu 2026 §2.2. rep2 then becomes a reviewer-response artefact
rather than a publication requirement.

The decision criterion is essentially: "is the 60-ns rep1 already inside
the polymer-MD-publication norm?" — if yes, ship; if no, rep2.
