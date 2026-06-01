# Force field

CHARMM36 + CGenFF + custom LMA parameters + topology used by all four
GelMA subsystems.

| File | Role |
|---|---|
| `top_all36_prot_HYP_caf_3.txt` | Protein topology (with HYP hydroxyproline + project additions) |
| `par_all36m_prot_hyp.prm` | CHARMM36m protein parameters (with HYP) |
| `par_all36m_prot.prm` | CHARMM36m protein parameters (base) |
| `par_all36_carb.prm` | Carbohydrate parameters |
| `par_all36_cgenff.prm` | CGenFF small-molecule parameters (LMA analogy) |
| `par_all36_lipid.prm` | Lipid parameters (carried, unused at runtime) |
| `par_all36_na.prm` | Nucleic-acid parameters (carried, unused) |
| `par_ions.prm` | Beglov–Roux Na⁺ / Cl⁻ ion parameters |
| `lma.prm` | **Custom** LMA (lysine-methacrylamide) parameters |

Water: TIP3P (built into NAMD/CHARMM, no separate file).

## Usage

All NAMD `.conf` files reference these via relative paths
(`parameters ../script/<file>`); the LAMMPS converter
(`lammps/tools/parmed_emit_lammps.py`) takes them via `--par`/`--top`.

## Licensing

- **`lma.prm`** — custom LMA parameters developed for this project,
  released under MIT alongside the repository.
- **CHARMM36 files** (`par_all36*.prm`, `top_all36_prot_HYP_caf_3.txt`) —
  redistributed for reproducibility under MacKerell Lab's academic-use
  terms. Verify before commercial use:
  http://mackerell.umaryland.edu/charmm_ff.shtml
