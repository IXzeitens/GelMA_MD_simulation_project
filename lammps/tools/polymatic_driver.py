"""Polymatic-style iterative crosslink driver for Gel-MA (LAMMPS).

Implements the classic Polymatic loop (Abbott 2013) on top of a plain LAMMPS
binary — no REACTION package needed:

    for iter in 1..MAX_ITER:
        1. read current LAMMPS data file
        2. identify "original chains" via connected components of the bond graph
           (frozen at iter 0, so inter-chain stays meaningful after bonds form)
        3. find all C8-C8 pairs on DIFFERENT chains within `cutoff` (PBC)
        4. for each candidate, with probability `prob`, form a new C-C bond and
           retype both reactive carbons CG2DC3 (sp2 vinyl) -> CG321 (sp3 CH2)
        5. write the modified data file
        6. run `step.lmps` for `relax_steps` of NPT relaxation
        7. read relaxed coords back; log to crosslink_history.json
        8. stop when no new bonds form for `converge_patience` consecutive iters
                                                          or MAX_ITER reached

Type IDs are parsed from the pre-generated `crosslink.fix` line, so this driver
needs no parmed import.

Usage (inside WSL where lmp lives):
    python polymatic_driver.py \\
        --data       project.data \\
        --coeffs     project.in.coeffs \\
        --fix        crosslink.fix \\
        --config     crosslink_config.yaml \\
        --step-template step.lmps.template \\
        --lmp        /usr/bin/lmp \\
        --workdir    Gel3MA_xlink/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# =============================================================================
# LAMMPS data-file parser / writer (atom_style full)
# =============================================================================
def parse_data(path: Path) -> dict:
    """Parse a LAMMPS data file into a structured dict."""
    text = path.read_text(encoding="utf-8").splitlines()
    d = {
        "header": [],
        "counts": {}, "type_counts": {},
        "box": {}, "masses": [], "sections": {},
        "atoms": [], "bonds": [], "velocities": [],
    }
    i = 0
    d["title"] = text[0] if text else "LAMMPS data"
    # ---- header ----
    section_names = ("Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
                     "Dihedral Coeffs", "Improper Coeffs", "Atoms", "Velocities",
                     "Bonds", "Angles", "Dihedrals", "Impropers")
    body_start = None
    for idx, line in enumerate(text[1:], start=1):
        s = line.strip()
        if any(s == name or s.startswith(name + " ") for name in section_names):
            body_start = idx
            break
        m = re.match(r"^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)\b", s)
        if m:
            d["counts"][m.group(2)] = int(m.group(1)); continue
        m = re.match(r"^(\d+)\s+(atom|bond|angle|dihedral|improper)\s+types\b", s)
        if m:
            d["type_counts"][m.group(2)] = int(m.group(1)); continue
        m = re.match(r"^([-\d.eE]+)\s+([-\d.eE]+)\s+(xlo xhi|ylo yhi|zlo zhi)", s)
        if m:
            d["box"][m.group(3)] = (float(m.group(1)), float(m.group(2))); continue

    # ---- body sections ----
    cur = None
    for line in text[body_start:]:
        s = line.strip()
        if not s:
            continue
        header_match = next((name for name in section_names
                             if s == name or s.startswith(name + " ")), None)
        if header_match:
            cur = header_match
            continue
        if cur == "Masses":
            d["masses"].append(line)
        elif cur == "Atoms":
            d["atoms"].append(line)
        elif cur == "Velocities":
            d["velocities"].append(line)
        elif cur == "Bonds":
            d["bonds"].append(line)
        else:
            d["sections"].setdefault(cur, []).append(line)
    return d


def atom_fields(line: str):
    """atom_style full: id mol type q x y z [...]. Returns parsed tuple."""
    p = line.split()
    return (int(p[0]), int(p[1]), int(p[2]), float(p[3]),
            float(p[4]), float(p[5]), float(p[6]), p[7:])


def write_data(path: Path, d: dict) -> None:
    """Serialise the dict back to a LAMMPS data file."""
    L = [d["title"], ""]
    L.append(f"{d['counts'].get('atoms',0)} atoms")
    L.append(f"{d['counts'].get('bonds',0)} bonds")
    for k in ("angles", "dihedrals", "impropers"):
        if d["counts"].get(k):
            L.append(f"{d['counts'][k]} {k}")
    L.append("")
    L.append(f"{d['type_counts'].get('atom',0)} atom types")
    L.append(f"{d['type_counts'].get('bond',0)} bond types")
    for k in ("angle", "dihedral", "improper"):
        if d["type_counts"].get(k):
            L.append(f"{d['type_counts'][k]} {k} types")
    L.append("")
    for tag in ("xlo xhi", "ylo yhi", "zlo zhi"):
        lo, hi = d["box"][tag]
        L.append(f"{lo:.6f} {hi:.6f} {tag}")
    L.append("")
    L.append("Masses"); L.append("")
    L.extend(d["masses"]); L.append("")
    # Coeff sections (pass-through)
    for name in ("Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
                 "Dihedral Coeffs", "Improper Coeffs"):
        if d["sections"].get(name):
            L.append(name); L.append("")
            L.extend(d["sections"][name]); L.append("")
    L.append("Atoms"); L.append("")
    L.extend(d["atoms"]); L.append("")
    if d["velocities"]:
        L.append("Velocities"); L.append("")
        L.extend(d["velocities"]); L.append("")
    L.append("Bonds"); L.append("")
    L.extend(d["bonds"]); L.append("")
    for name in ("Angles", "Dihedrals", "Impropers"):
        if d["sections"].get(name):
            L.append(name); L.append("")
            L.extend(d["sections"][name]); L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# =============================================================================
# Chain identity (connected components) + candidate detection
# =============================================================================
def connected_components(n_atoms: int, bonds: list[str]) -> dict[int, int]:
    """Union-find over bond graph → {atom_id: component_id}."""
    parent = list(range(n_atoms + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for line in bonds:
        p = line.split()
        union(int(p[2]), int(p[3]))
    return {a: find(a) for a in range(1, n_atoms + 1)}


def min_image_dist(a, b, box) -> float:
    lx = box["xlo xhi"][1] - box["xlo xhi"][0]
    ly = box["ylo yhi"][1] - box["ylo yhi"][0]
    lz = box["zlo zhi"][1] - box["zlo zhi"][0]
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    dx -= lx * round(dx / lx); dy -= ly * round(dy / ly); dz -= lz * round(dz / lz)
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def find_candidates(d: dict, chain_of: dict[int, int], c8_type: int,
                    cutoff: float, already_bonded: set[int]) -> list[tuple[int, int, float]]:
    """All C8-C8 pairs on different original chains within cutoff, not yet reacted."""
    c8 = []
    for line in d["atoms"]:
        aid, mol, typ, q, x, y, z, rest = atom_fields(line)
        if typ == c8_type and aid not in already_bonded:
            c8.append((aid, (x, y, z)))
    out = []
    for i in range(len(c8)):
        for j in range(i + 1, len(c8)):
            ai, aj = c8[i][0], c8[j][0]
            if chain_of[ai] == chain_of[aj]:
                continue                       # same original chain → skip
            dist = min_image_dist(c8[i][1], c8[j][1], d["box"])
            if dist <= cutoff:
                out.append((ai, aj, dist))
    out.sort(key=lambda t: t[2])               # closest first
    return out


def apply_crosslink(d: dict, i: int, j: int, bond_type: int, post_type: int) -> None:
    """Add a new bond i-j and retype both atoms to post_type (sp3)."""
    # retype
    new_atoms = []
    for line in d["atoms"]:
        p = line.split()
        if int(p[0]) in (i, j):
            p[2] = str(post_type)
            line = " ".join(p)
        new_atoms.append(line)
    d["atoms"] = new_atoms
    # add bond
    next_bid = d["counts"]["bonds"] + 1
    d["bonds"].append(f"{next_bid} {bond_type} {i} {j}")
    d["counts"]["bonds"] = next_bid


# =============================================================================
# Type-ID extraction from crosslink.fix
# =============================================================================
def parse_fix(fix_path: Path) -> dict:
    """Extract c8_type / bond_type / post_type from the fix bond/create line."""
    for line in fix_path.read_text(encoding="utf-8").splitlines():
        m = re.search(
            r"fix\s+\S+\s+\S+\s+bond/create\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+).*?iparam\s+\d+\s+(\d+)",
            line)
        if m:
            return {"nevery": int(m.group(1)), "itype": int(m.group(2)),
                    "cutoff": float(m.group(4)), "bond_type": int(m.group(5)),
                    "post_type": int(m.group(6))}
    sys.exit(f"[FATAL] could not parse fix bond/create line from {fix_path}")


# =============================================================================
# LAMMPS relaxation
# =============================================================================
def run_relax(lmp: str, step_template: Path, workdir: Path, iter_n: int,
              cfg: dict, in_data: Path) -> Path:
    """Materialise step.lmps for this iter, run LAMMPS NPT, return relaxed data path."""
    relax = cfg["relaxation"]
    tmpl = step_template.read_text(encoding="utf-8")
    tmpl = (tmpl
            .replace("{{ITER}}", str(iter_n))
            .replace("{{TEMPERATURE}}", str(relax["temperature_K"]))
            .replace("{{PRESSURE_BAR}}", str(relax["pressure_bar"]))
            .replace("{{STEPS}}", str(relax["steps"]))
            .replace("{{TIMESTEP}}", str(relax["timestep_fs"])))
    step_path = workdir / f"step_{iter_n:02d}.lmps"
    step_path.write_text(tmpl, encoding="utf-8")
    # step.lmps reads iter_{ITER}_in.data — stage it
    shutil.copy(in_data, workdir / f"iter_{iter_n}_in.data")
    # ``lmp`` may be a multi-token launcher, e.g. "mpirun -np 6 /path/lmp_mpi".
    # shlex.split keeps a bare path working while enabling MPI invocation.
    proc = subprocess.run(shlex.split(lmp) + ["-in", step_path.name],
                          cwd=workdir, capture_output=True, text=True)
    (workdir / f"iter_{iter_n}.log").write_text(proc.stdout + "\n" + proc.stderr,
                                                encoding="utf-8")
    out_data = workdir / f"iter_{iter_n}_out.data"
    if proc.returncode != 0 or not out_data.exists():
        sys.exit(f"[FATAL] LAMMPS relax failed at iter {iter_n}; see iter_{iter_n}.log")
    return out_data


# =============================================================================
# Main loop
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--coeffs", required=True, type=Path)
    ap.add_argument("--fix", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--step-template", required=True, type=Path)
    ap.add_argument("--lmp", default="/usr/bin/lmp")
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect candidates for iter 1 only, no LAMMPS run.")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fix = parse_fix(args.fix)
    cutoff = cfg.get("crosslink_cutoff_A", fix["cutoff"])
    prob = cfg.get("self_avoidance", True) and cfg.get("prob", 0.5) or cfg.get("prob", 0.5)
    prob = cfg.get("prob", 0.5)
    max_iter = cfg["iteration"]["max_iter"]
    patience = cfg["iteration"].get("converge_when_new_bonds_below", 1)
    seed = cfg.get("seed", 12345)
    rng = random.Random(seed)

    args.workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.coeffs, args.workdir / args.coeffs.name)

    d = parse_data(args.data)
    n_atoms = d["counts"]["atoms"]
    # Freeze original-chain identity at iter 0.
    chain_of = connected_components(n_atoms, d["bonds"])
    n_chains = len(set(v for a, v in chain_of.items()
                       if any(int(l.split()[0]) == a for l in d["atoms"][:1]))) if False else None

    print(f"[init] atoms={n_atoms} bonds={d['counts']['bonds']} "
          f"C8 type={fix['itype']} -> post {fix['post_type']} via bond {fix['bond_type']}")
    print(f"[init] cutoff={cutoff} A  prob={prob}  max_iter={max_iter}")

    cands0 = find_candidates(d, chain_of, fix["itype"], cutoff, set())
    print(f"[init] initial inter-chain C8-C8 candidates within {cutoff} A: {len(cands0)}")
    if args.dry_run:
        for i, j, dist in cands0[:20]:
            print(f"    {i:6d} - {j:6d}   {dist:.2f} A")
        return 0

    history = {"config": str(args.config), "cutoff": cutoff, "prob": prob,
               "iterations": []}
    reacted: set[int] = set()
    stale = 0
    cur_data = args.data

    for it in range(1, max_iter + 1):
        d = parse_data(cur_data)
        cands = find_candidates(d, chain_of, fix["itype"], cutoff, reacted)
        new = []
        for (i, j, dist) in cands:
            if i in reacted or j in reacted:
                continue                       # self-avoidance: 1 bond per C8
            if rng.random() < prob:
                apply_crosslink(d, i, j, fix["bond_type"], fix["post_type"])
                reacted.add(i); reacted.add(j)
                new.append((i, j, round(dist, 3)))
        print(f"[iter {it:02d}] candidates={len(cands)}  new_bonds={len(new)}  total={len(reacted)//2}")
        history["iterations"].append({"iter": it, "n_candidates": len(cands),
                                      "n_new": len(new), "new_bonds": new,
                                      "total_xlinks": len(reacted) // 2})

        if not new:
            stale += 1
            if stale >= max(1, 3):     # 3 consecutive empty iters → converged
                print(f"[done] converged: no new bonds for {stale} iters")
                break
        else:
            stale = 0

        in_data = args.workdir / f"iter_{it}_pre.data"
        write_data(in_data, d)
        out_data = run_relax(args.lmp, args.step_template, args.workdir, it, cfg, in_data)
        cur_data = out_data

    # Final
    final = args.workdir / "final_crosslinked.data"
    shutil.copy(cur_data, final)
    (args.workdir / "crosslink_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] total crosslinks = {len(reacted)//2}")
    print(f"[done] final data -> {final}")
    print(f"[done] history -> {args.workdir / 'crosslink_history.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
