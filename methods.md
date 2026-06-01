# Methods (camera-ready parameter table)

Placeholder. Intended to be a copy-pasteable Methods section listing:

## Force field
- Protein: CHARMM36m (`par_all36m_prot_hyp.prm`, `top_all36_prot_HYP_caf_3.txt`)
- Carbohydrates: CHARMM36 (`par_all36_carb.prm`)
- CGenFF (LMA analogy): `par_all36_cgenff.prm`
- Lipids: `par_all36_lipid.prm`
- Nucleic acids: `par_all36_na.prm` (carried, not used)
- LMA custom: `lma.prm` (backbone retyped to standard protein atoms)
- Ions: `par_ions.prm` (Beglov–Roux Na⁺/Cl⁻)
- Water: TIP3P

## System construction
- Base PDB: 24-residue gelatin chain with 3 LYS at residues 6, 15, 23
- LMA grafting: per-chain Bernoulli RNG with DS = 0 / 33 / 67 / 100 %
  → Gelatin, Gel1MA, Gel2MA, Gel3MA respectively
- 12 chains per system
- Packmol packing to ~7.4 wt% polymer (cubic box, water-fill)
- Ion balancing: VMD autoionize, neutralize + 0.15 M NaCl

## Simulation protocol (NAMD 3.0.2)
- Integrator: GPU-resident (CUDASOAintegrate on), rigid bonds, 2 fs timestep
- PME: grid spacing 1.0 Å, order 4, tolerance 1e-6
- Cutoff 12 Å with switching from 10 Å, pairlist 14 Å
- Thermostat: Langevin, 310 K, 1 ps⁻¹ damping (heavy atoms only)
- Barostat: Langevin piston, 1.01325 bar, period 100 fs, decay 50 fs
- Equilibration:
  * NVT 1 ns (heating to 310 K)
  * NPT₁ 2 ns (gentle barostat ramp)
  * NPT₂ 18 ns (production)
  * NPT₃ 10 ns (extension, dense xstFreq for K analysis)
- Output: dcdfreq=50000 (100 ps), xstFreq=50000 (100 ps) for npt_1/2/3

## Bulk modulus
- Method: NPT volume fluctuation, K_T = ⟨V⟩·k_B·T / σ²_V
- Sampling: npt_2 + npt_3 trajectories combined (~28 ns total)
- Real σ_K estimated via autocorrelation time τ_V (∼50–100 ps),
  giving σ_K / K ≈ √(2τ_V / T) ≈ 6–7 % per system
- Validation: per-part K agreement (npt_2 vs npt_3) within ~10%

## LAMMPS conversion (Stage 5)
- Tool: parmed_emit_lammps.py (in-house, this repo)
- Pair: lj/charmm/coul/long, cutoff 12/10 Å, PPPM tol 1e-5
- Styles: harmonic bond, charmm angle (UB zeroed), charmm dihedral,
  harmonic improper
- Known gap: CMAP cross-terms not emitted (≤5% backbone energy off,
  no effect on K)

## Crosslink (Stage 5)
- LAMMPS `fix bond/create` between MA-MA C=C atoms
- Cutoff 7 Å (Polymatic convention, Abbott et al. 2013)
- Atom-type retype: CG2DC3 → CG321 on bond formation
- Production: 5 ns NPT post-crosslink
