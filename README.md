# GelMA MD Simulation Project

End-to-end molecular-dynamics pipeline for gelatin methacrylate (GelMA)
hydrogels — from chain modeling and water-box assembly through NPT
equilibration, structural/interaction analysis, photo-crosslinking, and
bulk-modulus measurement.

The project studies how the **degree of methacrylate substitution (DS)**
controls the conformation, interaction network, and mechanical properties
of gelatin hydrogels. Four DS levels are modeled: **0 % (Gelatin), 33 %
(Gel1MA), 67 % (Gel2MA), 100 % (Gel3MA)**, each as 12 chains in a
~7.4 wt% water box.

```
 ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
 │  BUILD   │ → │ SIMULATE │ → │  ANALYZE  │ → │   PLOT   │ → │  LAMMPS  │
 └──────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────┘
  Packmol +      NAMD 3 GPU      MDAnalysis      publication     crosslink +
  psfgen box     NVT→NPT₁→₂→₃    metrics +       figures         bulk modulus
  assembly       (CHARMM36)      block stats                     (Polymatic)
```

---

## Repository layout

```
gelma_md/
├── pipeline/        Workflow-1 core package (box calc, Packmol, psfgen, NAMD)
├── workflow/        Entry points: main.py, batch driver, replica/extend tools
├── analysis/        MDAnalysis post-processing (metrics, NVT thermo, block stats)
├── plotting/        Publication figure generation (per-system + cross-system)
├── lammps/          Crosslinking + mechanical-property workflow
│   ├── tools/         CHARMM→LAMMPS converter, crosslink-fix generator,
│   │                  Polymatic driver, GPU build script
│   ├── bulk_modulus/  K via NPT volume fluctuation + multi-pressure templates
│   └── crosslink/     fix bond/create config + per-iteration relaxation
├── forcefield/      CHARMM36 + custom LMA parameters + topology
├── example/         One runnable subsystem (Gel3MA single-chain PDB + config + NAMD confs)
├── render/          VMD snapshot script (auto_render.tcl)
├── docs/            Pipeline design, subsystem layout, replica workflow
├── methods.md       Camera-ready Methods parameter table
├── requirements.txt
└── LICENSE          MIT
```

---

## Installation

### Python environment
```bash
pip install -r requirements.txt
```
Core dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `MDAnalysis`,
`mdtraj`, `statsmodels`, `PyYAML`. Optional: `freesasa` (SASA), `parmed`
(install via `conda install -c conda-forge parmed` for the LAMMPS converter).

### External software (not bundled)
| Tool | Role | Tested version |
|---|---|---|
| **NAMD 3** | MD engine (GPU-resident integrator) | 3.0.2 Win64-multicore-CUDA |
| **VMD** | psfgen topology + autoionize + rendering | 2.0 |
| **Packmol** | water-box packing | (any recent) |
| **LAMMPS** | crosslinking + mechanical tests | source-built, Kokkos 5.1.99 + CUDA |

Set paths in each subsystem's `config.json` under `software_paths`.

> **Data location:** the analysis / plotting / bulk-modulus scripts read the
> simulation `production/` tree, which lives *alongside* this repo, not inside
> it. By default they assume `production/` sits in the parent of `gelma_md/`.
> If your data is elsewhere, point them at it with the `GELMA_REPO` env var:
> `export GELMA_REPO=/path/to/workspace` (PowerShell: `$env:GELMA_REPO="..."`).

> **LAMMPS for RTX 50-series (Blackwell sm_120):** the conda-forge `lammps`
> package ships Kokkos 4.3.1 which does **not** recognize Blackwell GPUs.
> Build from source with `lammps/tools/build_lammps_gpu.sh`
> (`-DKokkos_ARCH_BLACKWELL120=ON`).

---

## Quick start

### 1. Build + simulate one system
```bash
cd example
# edit config.example.json -> config.json (set chain_count, concentration, paths)
python ../workflow/main.py        # box calc → Packmol → psfgen → NVT → NPT₁ → NPT₂
```
`main.py` is idempotent — re-running skips completed stages.

### 2. Batch all four DS systems (+ replicas)
```bash
python workflow/run_batch_workflow1.py            # Gelatin / Gel1MA / Gel2MA / Gel3MA
python workflow/auto_extend.py --max-part 3       # extend each NPT by one part (+10 ns)
```

### 3. Analyze
```bash
python analysis/data_analysis.py     # per-frame metrics → Data/<sys>/*.csv
python analysis/nvt_thermo.py        # NVT convergence check
```
Produces per-system CSVs (Rg, Ree, persistence length, H-bonds, salt bridges,
contact maps, RDF, DSSP, SASA, density profiles) + block-averaged statistics.

### 4. Plot
```bash
python plotting/data_collect.py      # aggregate cross-system summary
python plotting/plot_per_system.py   # 7 themed figures per subsystem
python plotting/plot_merged.py       # cross-system comparison figures
```
All figures output as 300 dpi PNG + vector PDF (Okabe-Ito colorblind-safe palette).

### 5. Crosslink + bulk modulus (LAMMPS)
```bash
# Convert NAMD-equilibrated structure to LAMMPS
python lammps/tools/parmed_emit_lammps.py --psf debug_1.psf --pdb npt3_final.pdb \
    --top forcefield/top_all36_prot_HYP_caf_3.txt --par forcefield/par_*.prm \
    --box 86 86 86 --out-data project.data --out-coeffs project.in.coeffs

# Photo-crosslink (Polymatic-style, 7 Å C=C cutoff)
python lammps/tools/generate_xlink_fix.py --psf debug_1.psf --top ... --par ... \
    --data project.data --cutoff 7.0 --prob 1.0 --out-fix crosslink.fix
python lammps/tools/polymatic_driver.py --data project.data --coeffs project.in.coeffs \
    --fix crosslink.fix --config lammps/crosslink/crosslink_config_chiu.yaml \
    --step-template lammps/crosslink/step.lmps.template --lmp <lmp> --workdir Gel3MA_xlink/

# Bulk modulus from NPT volume fluctuation (no LAMMPS needed for pre-network)
python lammps/bulk_modulus/bulk_from_xst.py --parts 2 3 --skip-frac 0.1
```

---

## Methodology summary

| Stage | Method | Key parameters |
|---|---|---|
| Box assembly | Packmol + VMD autoionize | 7.4 wt%, 1.20 expansion factor, Na⁺ neutralized |
| Equilibration | NAMD 3 NPT, Langevin piston | 310 K, 1 atm, 2 fs, rigidBonds, CUDASOAintegrate |
| Force field | CHARMM36m + CGenFF (LMA) + TIP3P | 12 Å cutoff, PME 1.0 Å grid |
| Bulk modulus | NPT volume fluctuation: K = ⟨V⟩·k_B·T / σ²_V | 310 K, autocorrelation-based σ_K |
| Crosslink | Polymatic-style `fix bond/create` | 7 Å C=C cutoff (Abbott 2013), iterative NPT relax |
| Persistence length | Kratky-Porod: L_p = −L / ln⟨cos θ⟩ | Cα-Cα bond vectors |

Full parameter table: [`methods.md`](methods.md). Pipeline internals:
[`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md).

---

## Representative results (pre-network, 12-chain box)

Bulk modulus from NPT volume fluctuation over ~28 ns NAMD trajectories:

| System  | DS%  | Bulk modulus K (MPa) |
|---------|------|----------------------|
| Gelatin | 0    | 2148 ± 150           |
| Gel1MA  | 33   | 1850 ± 130           |
| Gel2MA  | 67   | 2451 ± 170           |
| Gel3MA  | 100  | 1738 ± 120           |

Persistence length rises monotonically with DS (4.56 → 5.93 Å, +30 %),
consistent with more extended conformations at higher substitution.

---

## System design rationale

This project migrated from a 3-chain (~56 Å) to a 12-chain (~84 Å
equilibrated) box because inter-chain observables (H-bonds, salt bridges)
were dominated by seed noise in the small system (3 inter-chain pairs only;
relative SEM up to 100 % across replicas). The 12-chain box provides
**66 inter-chain pairs (22×)** and enough LMA reactive sites (36 at DS=100%
vs 9) to form an analyzable crosslink network. See
[`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md).

---

## Force-field licensing

`lma.prm` (custom LMA parameters) is released under MIT with this repository.
The CHARMM36 parameter/topology files (`par_all36*.prm`,
`top_all36_prot_HYP_caf_3.txt`) are redistributed for reproducibility under
MacKerell Lab's academic-use terms — verify before commercial use:
http://mackerell.umaryland.edu/charmm_ff.shtml

---

## Hardware tested

- NAMD 3.0.2 Win64-multicore-CUDA on RTX 5060 Ti — ~84 ns/day (60k atoms)
- LAMMPS (source, Kokkos 5.1.99 + CUDA 12.9, Blackwell sm_120) — ~2 ns/day
- WSL Ubuntu 24.04, parmed 4.3.1, MDAnalysis 2.10
- VMD 2.0 (psfgen, autoionize, topotools)

---

## Citation

If you use this pipeline, please cite (preprint forthcoming) and the
underlying tools:

- Phillips et al., *J. Comput. Chem.* **26**, 1781 (2005) — NAMD
- Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022) — LAMMPS
- Martínez et al., *J. Comput. Chem.* **30**, 2157 (2009) — Packmol
- Michaud-Agrawal et al., *J. Comput. Chem.* **32**, 2319 (2011) — MDAnalysis
- Huang & MacKerell, *J. Comput. Chem.* **34**, 2135 (2013) — CHARMM36
- Abbott, Hart & Colina, *Theor. Chem. Acc.* **132**, 1 (2013) — Polymatic 7 Å cutoff
- Chiu et al., *Carbohydr. Polym. Technol. Appl.* **13**, 101057 (2026) — GGMA reference

---

## License

MIT — see [LICENSE](LICENSE). (CHARMM36 force-field files excepted; see above.)
