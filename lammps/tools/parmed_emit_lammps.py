"""Convert a CHARMM PSF + PDB + parameter set to LAMMPS data + coefficient include.

parmed has no built-in LAMMPS writer, so we walk its internal structures
(atoms / bonds / angles / dihedrals / impropers / atom_type, bond_type,
angle_type, dihedral_type, improper_type) and emit the two files LAMMPS
needs to run:

    project.data        — atoms / bonds / angles / dihedrals / impropers (style "full")
    project.in.coeffs   — pair_coeff / bond_coeff / angle_coeff / dihedral_coeff / improper_coeff

The LAMMPS input file should then load both:
    units real ; atom_style full
    pair_style    lj/charmm/coul/long 10.0 12.0
    bond_style    harmonic
    angle_style   charmm                # CHARMM angles carry Urey-Bradley
    dihedral_style charmm
    improper_style harmonic
    read_data     project.data
    include       project.in.coeffs

Multi-term CHARMM dihedrals (n=1,2,3,4,…) are *expanded* into one LAMMPS
dihedral type per term, all assigned to the same physical dihedral via
multiple lines in the Dihedrals section. This matches what NAMD does
internally and what `dihedral_style charmm` expects.

CMAP cross-terms (protein φ-ψ correction) are written as comments only —
including them properly requires `fix cmap` plus a CMAP data block; that's
left as TODO since the impact on bulk modulus / Young's modulus is small.

Usage (inside WSL, where parmed is installed):

    ~/miniconda3/bin/python parmed_emit_lammps.py \\
        --psf  debug_1.psf  --pdb debug_1.pdb \\
        --top  top_all36_prot_HYP_caf_3.txt \\
        --par  par_all36m_prot_hyp.prm par_all36_carb.prm par_all36_cgenff.prm \\
               par_all36_lipid.prm par_all36_na.prm lma.prm par_ions.prm \\
        --box  102 102 102 \\
        --out-data    project.data \\
        --out-coeffs  project.in.coeffs
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", required=True, type=Path)
    ap.add_argument("--pdb", required=True, type=Path)
    ap.add_argument("--top", required=True, type=Path,
                    help="CHARMM topology (.rtf / .str). .txt accepted via internal rename.")
    ap.add_argument("--par", required=True, nargs="+", type=Path,
                    help="One or more CHARMM parameter (.prm) files.")
    ap.add_argument("--box", nargs=3, type=float, default=None, metavar=("LX", "LY", "LZ"),
                    help="Cubic box dimensions in A. If omitted, derive from PDB extents.")
    ap.add_argument("--out-data", required=True, type=Path)
    ap.add_argument("--out-coeffs", required=True, type=Path)
    args = ap.parse_args()

    try:
        import parmed as pmd
        from parmed.charmm import CharmmPsfFile, CharmmParameterSet
    except ImportError as e:
        sys.exit(f"[FATAL] parmed not available: {e}")

    print(f"[parmed {pmd.__version__}]")

    # --- Load PSF + PDB coords ---------------------------------------------
    psf = CharmmPsfFile(str(args.psf))
    pdb = pmd.load_file(str(args.pdb))
    if len(pdb.atoms) != len(psf.atoms):
        sys.exit(f"[FATAL] atom count mismatch: PSF={len(psf.atoms)} PDB={len(pdb.atoms)}")
    psf.coordinates = pdb.coordinates
    print(f"loaded {len(psf.atoms)} atoms")

    # --- Load params (parmed needs .rtf for topology auto-detect) ----------
    scratch = Path(tempfile.mkdtemp(prefix="parmed_ff_"))
    try:
        top_canonical = scratch / (args.top.stem + ".rtf")
        shutil.copy(args.top, top_canonical)
        param_files = [str(top_canonical)] + [str(p) for p in args.par]
        params = CharmmParameterSet(*param_files)
        psf.load_parameters(params)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"applied: {len(psf.bond_types)} bond_types, "
          f"{len(psf.angle_types)} angle_types, "
          f"{len(psf.dihedral_types)} dihedral_types, "
          f"{len(psf.improper_types)} improper_types")

    # --- Decide box --------------------------------------------------------
    if args.box is not None:
        box = list(args.box)
    elif pdb.box is not None:
        box = list(pdb.box[:3])
    else:
        import numpy as np
        coords = np.asarray(psf.coordinates)
        L = float((coords.max(axis=0) - coords.min(axis=0)).max() + 5.0)
        box = [L, L, L]
    print(f"box = {box}")

    # =====================================================================
    # Build LAMMPS-id maps. Order = order of first appearance in parmed's
    # iteration → keeps things deterministic across runs.
    # =====================================================================

    # 1. Atom types (used by pair_coeff). Each parmed AtomType has eps, rmin,
    #    eps_14, rmin_14, mass, atomic_number, name.
    atype_order: "OrderedDict[str, object]" = OrderedDict()
    for atom in psf.atoms:
        if atom.atom_type.name not in atype_order:
            atype_order[atom.atom_type.name] = atom.atom_type
    atype_id = {name: i + 1 for i, name in enumerate(atype_order)}

    # 2. Bond types
    btype_order = []
    btype_id: "dict[int, int]" = {}
    for bt in psf.bond_types:
        btype_order.append(bt)
        btype_id[id(bt)] = len(btype_order)

    # 3. Angle types
    atype_a_order = []
    atype_a_id: "dict[int, int]" = {}
    for at in psf.angle_types:
        atype_a_order.append(at)
        atype_a_id[id(at)] = len(atype_a_order)

    # 4. Dihedral types — MUST expand multi-term CHARMM dihedrals.
    #    parmed represents a dihedral type either as a single DihedralType
    #    (one term, attrs phi_k/per/phase) or DihedralTypeList of these.
    #    LAMMPS dihedral_style charmm: one coeff line per (k, n, delta, wt).
    dtype_records: "list[tuple[float, int, int, float]]" = []
    # Key = (round(phi_k,6), n, round(phase,3), round(weighting,3))
    dtype_seen: "dict[tuple, int]" = {}
    # For each physical dihedral, store list of LAMMPS dihedral type IDs.
    dihedral_to_lammps_types: "list[list[int]]" = []

    for dihedral in psf.dihedrals:
        dt = dihedral.type
        # Normalise to a list of single terms.
        if hasattr(dt, "__iter__"):
            terms = list(dt)
        else:
            terms = [dt]

        type_ids_for_this: list[int] = []
        for term in terms:
            phi_k     = float(term.phi_k)
            per       = int(term.per)
            phase_deg = float(term.phase)
            # Weighting: 1.0 unless it's a 1-4 special (parmed sets scee=1/scnb=1)
            # CHARMM default is 1.0 for non-1-4 atoms; LAMMPS will multiply.
            weight    = 1.0
            key = (round(phi_k, 6), per, round(phase_deg, 3), round(weight, 3))
            if key not in dtype_seen:
                dtype_seen[key] = len(dtype_records) + 1
                dtype_records.append((phi_k, per, int(round(phase_deg)), weight))
            type_ids_for_this.append(dtype_seen[key])
        dihedral_to_lammps_types.append(type_ids_for_this)

    # 5. Improper types — CHARMM impropers are harmonic.
    itype_order = []
    itype_id: "dict[int, int]" = {}
    for it in psf.improper_types:
        itype_order.append(it)
        itype_id[id(it)] = len(itype_order)

    # =====================================================================
    # Write the LAMMPS data file
    # =====================================================================
    args.out_data.parent.mkdir(parents=True, exist_ok=True)

    # Expand "physical" dihedrals into the LAMMPS entries each one maps to
    expanded_dihedrals: "list[tuple[int, int, int, int, int]]" = []
    for phys_idx, dihedral in enumerate(psf.dihedrals):
        a1 = dihedral.atom1.idx + 1
        a2 = dihedral.atom2.idx + 1
        a3 = dihedral.atom3.idx + 1
        a4 = dihedral.atom4.idx + 1
        for ltype in dihedral_to_lammps_types[phys_idx]:
            expanded_dihedrals.append((ltype, a1, a2, a3, a4))

    n_atoms = len(psf.atoms)
    n_bonds = len(psf.bonds)
    n_angles = len(psf.angles)
    n_dihedrals = len(expanded_dihedrals)
    n_impropers = len(psf.impropers)

    with args.out_data.open("w", encoding="utf-8") as f:
        f.write("LAMMPS data file emitted by parmed_emit_lammps.py (atom_style full)\n\n")
        f.write(f"{n_atoms:>10d}  atoms\n")
        f.write(f"{n_bonds:>10d}  bonds\n")
        f.write(f"{n_angles:>10d}  angles\n")
        f.write(f"{n_dihedrals:>10d}  dihedrals\n")
        f.write(f"{n_impropers:>10d}  impropers\n\n")

        f.write(f"{len(atype_order):>10d}  atom types\n")
        f.write(f"{len(btype_order):>10d}  bond types\n")
        f.write(f"{len(atype_a_order):>10d}  angle types\n")
        f.write(f"{len(dtype_records):>10d}  dihedral types\n")
        f.write(f"{len(itype_order):>10d}  improper types\n\n")

        f.write(f"{-box[0]/2:15.6f} {box[0]/2:15.6f} xlo xhi\n")
        f.write(f"{-box[1]/2:15.6f} {box[1]/2:15.6f} ylo yhi\n")
        f.write(f"{-box[2]/2:15.6f} {box[2]/2:15.6f} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        for name, atype in atype_order.items():
            f.write(f"{atype_id[name]:>5d}  {atype.mass:>12.6f}  # {name}\n")
        f.write("\n")

        # Atoms (full: atom_id mol_id type charge x y z)
        f.write("Atoms\n\n")
        coords = psf.coordinates  # list of (x,y,z)
        # parmed centers around (0,0,0)? Not necessarily — wrap relative to box.
        for atom in psf.atoms:
            i = atom.idx
            mol_id = atom.residue.chain_idx if hasattr(atom.residue, "chain_idx") else 1
            mol_id = 1  # keep simple; all in one molecule for LAMMPS bookkeeping
            t_id = atype_id[atom.atom_type.name]
            q = atom.charge
            x, y, z = coords[i]
            f.write(f"{i+1:>7d} {mol_id:>5d} {t_id:>4d} {q:>10.6f} "
                    f"{x:>12.6f} {y:>12.6f} {z:>12.6f}\n")
        f.write("\n")

        # Bonds
        f.write("Bonds\n\n")
        for j, bond in enumerate(psf.bonds, start=1):
            t = btype_id[id(bond.type)]
            f.write(f"{j:>7d} {t:>4d} {bond.atom1.idx+1:>7d} {bond.atom2.idx+1:>7d}\n")
        f.write("\n")

        # Angles
        f.write("Angles\n\n")
        for j, ang in enumerate(psf.angles, start=1):
            t = atype_a_id[id(ang.type)]
            f.write(f"{j:>7d} {t:>4d} {ang.atom1.idx+1:>7d} "
                    f"{ang.atom2.idx+1:>7d} {ang.atom3.idx+1:>7d}\n")
        f.write("\n")

        # Dihedrals (expanded for multi-term)
        f.write("Dihedrals\n\n")
        for j, (t, a1, a2, a3, a4) in enumerate(expanded_dihedrals, start=1):
            f.write(f"{j:>7d} {t:>4d} {a1:>7d} {a2:>7d} {a3:>7d} {a4:>7d}\n")
        f.write("\n")

        # Impropers
        if n_impropers:
            f.write("Impropers\n\n")
            for j, imp in enumerate(psf.impropers, start=1):
                t = itype_id[id(imp.type)]
                f.write(f"{j:>7d} {t:>4d} {imp.atom1.idx+1:>7d} "
                        f"{imp.atom2.idx+1:>7d} {imp.atom3.idx+1:>7d} "
                        f"{imp.atom4.idx+1:>7d}\n")
            f.write("\n")

    print(f"[OK] wrote {args.out_data}  ({n_atoms} atoms, {len(atype_order)} types, "
          f"{n_dihedrals} expanded dihedrals)")

    # =====================================================================
    # Write the LAMMPS coefficient include file
    # =====================================================================
    args.out_coeffs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_coeffs.open("w", encoding="utf-8") as f:
        f.write("# LAMMPS coefficient include file from parmed_emit_lammps.py\n")
        f.write("# Pair: lj/charmm/coul/long  inner 10.0  outer 12.0\n")
        f.write("# Bond: harmonic ; Angle: charmm (with Urey-Bradley) ;\n")
        f.write("# Dihedral: charmm ; Improper: harmonic\n\n")

        # pair_coeff: each unique atom type i, write its self-LJ.
        # parmed stores rmin (= sigma * 2^(1/6) / 2 * 2 = sigma * 2^(1/6))
        # CHARMM convention: rmin/2 in .prm; parmed exposes the full rmin
        # via atype.rmin. Convert to LAMMPS sigma = rmin / 2^(1/6).
        import math
        SIG_FACTOR = 2.0 ** (1.0 / 6.0)  # rmin -> sigma scaling for "rmin" in CHARMM
        # NOTE: in CHARMM/NAMD convention, Rmin/2 (half-min-distance) is what's
        # tabulated; parmed stores `rmin` = half-min-distance for the SELF
        # interaction (sometimes labelled `rmin_2` to disambiguate). Pair-style
        # `lj/charmm/coul/long` uses (eps, sigma); LAMMPS expects sigma in
        # `sig` units → sigma = 2 * (rmin/2) / 2^(1/6). So Rmin/2 → 2*Rmin/2 / 2^(1/6).
        # parmed's `atype.rmin` already equals Rmin/2 (half), so multiplier is 2.
        f.write("# pair_coeff  type_i  type_j  eps  sig  eps_14  sig_14\n")
        for name, atype in atype_order.items():
            i = atype_id[name]
            eps     = abs(atype.epsilon)
            sigma   = 2.0 * atype.rmin / SIG_FACTOR
            eps14   = abs(atype.epsilon_14) if atype.epsilon_14 is not None else eps
            sigma14 = 2.0 * atype.rmin_14 / SIG_FACTOR if atype.rmin_14 is not None else sigma
            f.write(f"pair_coeff {i:>4d} {i:>4d} {eps:>10.6f} {sigma:>10.6f} "
                    f"{eps14:>10.6f} {sigma14:>10.6f}  # {name}\n")
        f.write("\n")

        # bond_coeff harmonic:  K  r0    (parmed stores k as kcal/mol/A² and req as A,
        # both matching LAMMPS units real → no conversion needed.)
        f.write("# bond_coeff  type  K(kcal/mol/A^2)  r0(A)\n")
        for bt in btype_order:
            t = btype_id[id(bt)]
            f.write(f"bond_coeff {t:>4d} {bt.k:>10.4f} {bt.req:>8.4f}\n")
        f.write("\n")

        # angle_coeff charmm: K theta0 UB_K UB_r0.
        # parmed stores Urey-Bradley as a SEPARATE list; each UB carries only
        # the two end atoms (1 and 3 of the underlying angle). Build a lookup
        # by (min_idx, max_idx) pair, then for each angle_type pick UB params
        # via any one of its angle instances.
        ub_by_pair: "dict[tuple[int, int], tuple[float, float]]" = {}
        if getattr(psf, "urey_bradleys", None):
            for ub in psf.urey_bradleys:
                if ub.type is None:
                    continue
                pair = tuple(sorted([ub.atom1.idx, ub.atom2.idx]))
                ub_by_pair[pair] = (float(ub.type.k), float(ub.type.req))

        type_ub: "dict[int, tuple[float, float]]" = {}
        for ang in psf.angles:
            tid = id(ang.type)
            if tid in type_ub:
                continue
            pair = tuple(sorted([ang.atom1.idx, ang.atom3.idx]))
            type_ub[tid] = ub_by_pair.get(pair, (0.0, 0.0))

        f.write("# angle_coeff  type  K  theta0  UB_K  UB_r0\n")
        for at in atype_a_order:
            t = atype_a_id[id(at)]
            ub_k, ub_r = type_ub.get(id(at), (0.0, 0.0))
            f.write(f"angle_coeff {t:>4d} {at.k:>10.4f} {at.theteq:>8.3f} "
                    f"{ub_k:>10.4f} {ub_r:>8.4f}\n")
        f.write("\n")

        # dihedral_coeff charmm:  K  n  delta(int deg)  weighting
        f.write("# dihedral_coeff  type  K(kcal/mol)  n  delta(deg)  weighting\n")
        for t, (k, n, delta, wt) in enumerate(dtype_records, start=1):
            f.write(f"dihedral_coeff {t:>4d} {k:>10.4f} {n:>2d} {delta:>5d} {wt:>5.2f}\n")
        f.write("\n")

        # improper_coeff harmonic:  K  X0  (psi_eq usually 0 for sp2)
        f.write("# improper_coeff  type  K  X0(deg)\n")
        for it in itype_order:
            t = itype_id[id(it)]
            f.write(f"improper_coeff {t:>4d} {it.psi_k:>10.4f} {it.psi_eq:>8.3f}\n")
        f.write("\n")

        # CMAP — note only; full support would need fix cmap and a CMAP section.
        if psf.cmaps:
            f.write(f"# NOTE: {len(psf.cmaps)} CMAP cross-term corrections present.\n")
            f.write("# These are NOT included as coefficients here. To add them in LAMMPS,\n")
            f.write("# use 'fix cmap charmmfix.cmap' with the standard CHARMM36 CMAP data file.\n")

    print(f"[OK] wrote {args.out_coeffs} "
          f"({len(atype_order)} pair, {len(btype_order)} bond, "
          f"{len(atype_a_order)} angle, {len(dtype_records)} dihedral, "
          f"{len(itype_order)} improper)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
