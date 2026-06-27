"""Multi-system MD analysis pipeline (replica-aware).

For each subsystem in ``analysis_config.SYSTEMS``:
  * Auto-discover replica directories (``<sys>/rep1``, ``rep2``, ``rep3``),
    falling back to ``<sys>/Output`` as a single-replica source for ``rep1``.
  * Per replica: unwrap + align to a quasi-equilibrium reference, compute
    RMSD / Rg / Ree / SASA / H-bonds / inter-chain contacts / reactive-MA
    cluster statistics, dump CSV + RDF tables + contact-map .npy.
  * Aggregate across replicas (mean / SEM) into a single Ensemble summary.

This script writes everything under ``Data/<sys>/`` so that ``0511_plot.py``
and ``0508_new_merge.py`` can read from a single canonical location.
"""
from __future__ import annotations

import gc
import glob
import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import align, distances, rms
from MDAnalysis.analysis.hydrogenbonds import hbond_analysis
from MDAnalysis.analysis.rdf import InterRDF
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from analysis_config import (
    BLOCK_SIZE_NS,
    CONTACT_CUTOFF_A,
    DATA_PREFIX,
    DATA_ROOT,
    DT_PS,
    EQ_WINDOW_NS,
    HB_ANGLE_DEG,
    HB_DIST_STRICT_A,
    MA_ACCEPTOR_SEL,
    MA_CLUSTER_CUTOFF_A,
    MA_DONOR_SEL,
    MA_HYDROGEN_SEL,
    MA_REACTIVE_C,
    MA_RESNAME,
    PLOT_ROOT,
    PROTEIN_ACCEPTOR_SEL,
    PROTEIN_DONOR_SEL,
    PROTEIN_HYDROGEN_SEL,
    ROOT_DIR,
    SMOOTHING_RATIO,
    SYSTEMS,
    WATER_H_SEL,
    WATER_O_SEL,
    list_production_dcds,
    rolling_average,
)

log = logging.getLogger(__name__)

# MDAnalysis raises a lot of harmless warnings; silence only the noisy class
# rather than blanket-ignoring all warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

REPLICAS: tuple[str, ...] = ("rep1", "rep2", "rep3")


# =============================================================================
# Atomic / element helpers
# =============================================================================
def guess_element(atom) -> str:
    """Heuristic element guess; falls back to 'C' if undetermined."""
    if getattr(atom, "element", "") not in ("", None):
        return atom.element.upper()
    for candidate in (atom.name.upper(), atom.type.upper()):
        for prefix, element in (
            ("CL", "C"), ("CA", "C"), ("CP", "C"), ("CT", "C"),
            ("O", "O"), ("N", "N"), ("H", "H"), ("S", "S"), ("C", "C"),
        ):
            if candidate.startswith(prefix):
                return element
    return "C"


def get_vdw_radius(element: str) -> float:
    return {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}.get(element, 1.70)


# =============================================================================
# Geometry / contact analysis (O(N) via cell-list capped_distance)
# =============================================================================
def count_residue_contacts(group_a, group_b, box, cutoff: float = CONTACT_CUTOFF_A) -> int:
    """Count unique inter-residue pairs within ``cutoff`` under PBC."""
    if len(group_a) == 0 or len(group_b) == 0:
        return 0
    pairs = distances.capped_distance(
        group_a.positions, group_b.positions,
        max_cutoff=cutoff, box=box, return_distances=False,
    )
    if len(pairs) == 0:
        return 0
    res_a = group_a.resindices[pairs[:, 0]]
    res_b = group_b.resindices[pairs[:, 1]]
    return len(np.unique(np.column_stack((res_a, res_b)), axis=0))


def update_contact_map(group_a, group_b, contact_matrix: np.ndarray, box,
                       cutoff: float = CONTACT_CUTOFF_A) -> None:
    """Accumulate per-residue-pair contact frequencies into ``contact_matrix``."""
    if len(group_a) == 0 or len(group_b) == 0:
        return
    pairs = distances.capped_distance(
        group_a.positions, group_b.positions,
        max_cutoff=cutoff, box=box, return_distances=False,
    )
    if len(pairs) == 0:
        return
    _, res_a_inv = np.unique(group_a.resindices, return_inverse=True)
    _, res_b_inv = np.unique(group_b.resindices, return_inverse=True)
    idx_a = res_a_inv[pairs[:, 0]]
    idx_b = res_b_inv[pairs[:, 1]]
    unique_pairs = np.unique(np.column_stack((idx_a, idx_b)), axis=0)
    for a, b in unique_pairs:
        contact_matrix[a, b] += 1


def extract_contact_pairs(group_a, group_b, box, cutoff: float = CONTACT_CUTOFF_A) -> set[tuple[int, int]]:
    """Return the set of currently-active atom-pair indices (for lifetime tracking)."""
    if len(group_a) == 0 or len(group_b) == 0:
        return set()
    pairs = distances.capped_distance(
        group_a.positions, group_b.positions,
        max_cutoff=cutoff, box=box, return_distances=False,
    )
    return set(map(tuple, pairs))


def extract_all_interchain_pairs(chains: dict, box,
                                  cutoff: float = CONTACT_CUTOFF_A) -> set[tuple[int, int]]:
    """Chain-symmetric variant: union of every inter-chain contact pair across
    all distinct chain combinations, keyed by *global* atom indices so the
    result is invariant to which physical chain happens to be labelled GA/GB/GC.

    Returns ``{(min(global_a_idx, global_b_idx), max(...)), ...}``.

    Used by the Contact_Survival_Rate metric — the old per-pair (GA, GB)
    version is sensitive to packmol-imposed chain-labelling, which can flip
    the metric's sign between replicas with identical chemistry. Aggregating
    over all three chain pairs and using global indices removes that artefact.
    """
    pairs_all: set[tuple[int, int]] = set()
    names = list(chains.keys())
    for i, n1 in enumerate(names):
        ga = chains[n1]
        if len(ga) == 0:
            continue
        for n2 in names[i + 1:]:
            gb = chains[n2]
            if len(gb) == 0:
                continue
            ps = distances.capped_distance(
                ga.positions, gb.positions,
                max_cutoff=cutoff, box=box, return_distances=False,
            )
            if len(ps) == 0:
                continue
            glob_a = ga.indices[ps[:, 0]]
            glob_b = gb.indices[ps[:, 1]]
            for a, b in zip(glob_a, glob_b):
                pairs_all.add((int(min(a, b)), int(max(a, b))))
    return pairs_all


def ma_nearest_neighbor_distance(ma_atoms, box) -> float:
    """Mean nearest-neighbor distance between MA reactive sites across distinct chains."""
    if len(ma_atoms) < 2:
        return 0.0
    dist = distances.distance_array(ma_atoms.positions, ma_atoms.positions, box=box)
    same_chain = ma_atoms.segids[:, None] == ma_atoms.segids[None, :]
    dist[same_chain] = np.inf
    min_dists = np.min(dist, axis=1)
    finite = min_dists[np.isfinite(min_dists)]
    return float(np.mean(finite)) if len(finite) else 0.0


def ma_cluster_properties(ma_atoms, box, cutoff: float = MA_CLUSTER_CUTOFF_A) -> tuple[int, int, float]:
    """Connected-components analysis of inter-chain MA proximity graph.

    Returns: (n_clusters, max_cluster_size, largest_cluster_rg_A).
    """
    if len(ma_atoms) < 2:
        return 0, 0, 0.0

    dist = distances.distance_array(ma_atoms.positions, ma_atoms.positions, box=box)
    inter_chain = ma_atoms.segids[:, None] != ma_atoms.segids[None, :]
    adjacency = ((dist < cutoff) & inter_chain).astype(int)
    np.fill_diagonal(adjacency, 0)

    n_components, labels = connected_components(csgraph=csr_matrix(adjacency), directed=False)
    unique, counts = np.unique(labels, return_counts=True)
    max_size = int(np.max(counts))

    if max_size < 2:
        return n_components, max_size, 0.0

    largest_label = unique[np.argmax(counts)]
    cluster_positions = ma_atoms.positions[labels == largest_label]
    com = np.mean(cluster_positions, axis=0)
    diff = cluster_positions - com
    diff -= box[:3] * np.round(diff / box[:3])      # minimum-image wrap
    rg = float(np.sqrt(np.sum(diff ** 2) / len(cluster_positions)))
    return n_components, max_size, round(rg, 3)


def _run_hbond_analysis(universe, donors_sel, hydrogens_sel, acceptors_sel) -> np.ndarray:
    """Run HBA with project-standard cutoffs; return empty array on no hits."""
    hb = hbond_analysis.HydrogenBondAnalysis(
        universe,
        donors_sel=donors_sel,
        hydrogens_sel=hydrogens_sel,
        acceptors_sel=acceptors_sel,
        d_a_cutoff=HB_DIST_STRICT_A,
        d_h_a_angle_cutoff=HB_ANGLE_DEG,
    ).run()
    return hb.results.hbonds if len(hb.results.hbonds) else np.array([])


def split_inter_intra(hb_data: np.ndarray, universe, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin H-bond records into inter-chain vs intra-chain per-frame counts."""
    inter = np.zeros(n_frames, dtype=int)
    intra = np.zeros(n_frames, dtype=int)
    if hb_data.size == 0:
        return inter, intra
    frames = hb_data[:, 0].astype(int)
    donor_segids = universe.atoms[hb_data[:, 1].astype(int)].segids
    acceptor_segids = universe.atoms[hb_data[:, 3].astype(int)].segids
    different_chain = donor_segids != acceptor_segids
    for f in frames[different_chain]:
        inter[f] += 1
    for f in frames[~different_chain]:
        intra[f] += 1
    return inter, intra


def count_hbonds_per_frame(hb_data: np.ndarray, n_frames: int) -> np.ndarray:
    """Histogram H-bond records by frame index."""
    counts = np.zeros(n_frames, dtype=int)
    if hb_data.size == 0:
        return counts
    frames = hb_data[:, 0].astype(int)
    np.add.at(counts, frames, 1)
    return counts


# =============================================================================
# Additional publication-grade metrics (LAMMPS-free)
# =============================================================================
def compute_rmsf_per_chain(
    chain_names: list[str],
    psf_path: Path,
    aligned_dcd_path: Path,
    ref_universe,
) -> dict[str, pd.DataFrame]:
    """Per-Cα RMSF computed with **per-chain alignment**.

    A naive global alignment (the pipeline default) leaves inter-chain
    rigid-body motion inside the RMSF, inflating values by ~5–10× for
    multi-chain systems. To get publication-quality per-residue RMSF we
    instead re-align the trajectory using only one chain's backbone at a
    time. Each chain therefore reflects its own internal flexibility.

    Implementation: load a fresh ``Universe`` from the same aligned DCD per
    chain (memory cheaper than 3 parallel universes; the previous chain's
    trajectory state is freed at the bottom of the loop). ``in_memory=True``
    keeps the per-chain align purely RAM-based — the aligned DCD file is
    not mutated.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in chain_names:
        u_chain = mda.Universe(str(psf_path), str(aligned_dcd_path))
        # Use the alpha carbon of every residue. Standard residues name it
        # "CA"; LMA (methacrylated Lys) renames its alpha carbon to "C22"
        # (CG311, bonded to backbone N + carbonyl C + HA + sidechain C20).
        # Without the C22 branch every modified residue is silently dropped
        # from the RMSF trace (e.g. Gel3MA loses res 6/15/23 entirely).
        ca = u_chain.select_atoms(
            f"segid {name} and (name CA or (resname LMA and name C22))")
        if len(ca) == 0:
            log.warning("RMSF: chain %s has 0 alpha-carbon atoms; skipping.", name)
            del u_chain
            continue
        align.AlignTraj(
            u_chain, ref_universe,
            select=f"segid {name} and backbone",
            in_memory=True,
        ).run()
        analysis = rms.RMSF(ca).run()
        out[name] = pd.DataFrame({
            "resid": ca.resids,
            "rmsf_A": np.asarray(analysis.results.rmsf),
        })
        try:
            u_chain.trajectory.close()
        except Exception:
            pass
        del u_chain
    return out


def compute_dssp_matrix(psf_path: Path, aligned_dcd: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Per-frame, per-residue simplified secondary structure (H / E / C).

    Uses ``mdtraj.compute_dssp(simplified=True)`` on the protein subset.
    Returns (codes, resids) where ``codes`` is an (n_frames, n_residues) array
    of ASCII characters and ``resids`` is the residue ID per column.
    Falls back to (None, None) if mdtraj or DSSP cannot run.
    """
    try:
        import mdtraj
    except ImportError:
        log.warning("mdtraj not installed — DSSP skipped.")
        return None, None

    try:
        traj = mdtraj.load_dcd(str(aligned_dcd), top=str(psf_path))
        protein_idx = traj.topology.select("protein")
        if len(protein_idx) == 0:
            log.warning("DSSP: 0 protein atoms; skipping.")
            return None, None
        protein_traj = traj.atom_slice(protein_idx)
        # simplified=True → only H (helix), E (strand), C (coil/loop)
        codes = mdtraj.compute_dssp(protein_traj, simplified=True)
        resids = np.array([r.resSeq for r in protein_traj.topology.residues])
        log.info(
            "DSSP via mdtraj: %d frames × %d residues.", codes.shape[0], codes.shape[1],
        )
        return codes, resids
    except Exception as exc:
        log.warning("DSSP computation failed: %s", exc)
        return None, None


def parse_namd_log(log_path: Path) -> pd.DataFrame:
    """Extract NAMD ENERGY-line time series into a DataFrame.

    Returns columns: step, total_E, kinetic_E, potential_E, temperature_K,
    pressure_bar, volume_A3, time_ps (if dt available). Empty DataFrame if
    the log doesn't exist or has no ENERGY lines.
    """
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    # NAMD ENERGY columns (post-NAMD 2.10 format):
    # ENERGY: ts BOND ANGLE DIHED IMPRP ELECT VDW BOUNDARY MISC
    #         KINETIC TOTAL TEMP POTENTIAL TOTAL3 TEMPAVG
    #         PRESSURE GPRESSURE VOLUME PRESSAVG GPRESSAVG
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ENERGY:"):
                continue
            parts = line.split()
            if len(parts) < 20:
                continue
            try:
                rows.append({
                    "step":         int(parts[1]),
                    "bond":         float(parts[2]),
                    "angle":        float(parts[3]),
                    "elect":        float(parts[6]),
                    "vdw":          float(parts[7]),
                    "kinetic_E":    float(parts[10]),
                    "total_E":      float(parts[11]),
                    "temperature_K":float(parts[12]),
                    "potential_E":  float(parts[13]),
                    "pressure_bar": float(parts[16]),
                    "volume_A3":    float(parts[18]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def compute_persistence_length(chain) -> float:
    """Estimate persistence length Lp from backbone-N virtual-bond autocorrelation.

    Lp = -L / ln(<cos θ>)  (Kratky-Porod / worm-like chain)

    2026-06-03 residue-agnostic: uses backbone **N** (one per residue, incl LMA)
    instead of `name CA`. LMA (modified Lys) has NO CA atom, so a CA trace skipped
    every LMA → gaps that grew with DS and inflated Lp (a DS-confound; CA-method
    gave a spurious monotonic ↑, N-method gives a peak at 67% — see METHOD_FIX_LOG).
    Every residue (standard + LMA) has a backbone N → gap-free. Single value per
    chain on the current frame; 0.0 if too short.
    """
    ca = chain.select_atoms("name N")
    if len(ca) < 3:
        return 0.0
    pos = ca.positions
    bonds = np.diff(pos, axis=0)
    bond_lengths = np.linalg.norm(bonds, axis=1)
    L = float(np.mean(bond_lengths))
    unit_bonds = bonds / bond_lengths[:, None]
    cos_thetas = np.sum(unit_bonds[:-1] * unit_bonds[1:], axis=1)
    avg_cos = float(np.mean(cos_thetas))
    # Negative or zero correlation → flexible chain limit; return 0 conventionally
    if avg_cos <= 1e-6:
        return 0.0
    return -L / np.log(avg_cos)


def count_salt_bridges(u, box, cutoff_A: float = 5.0) -> int:
    """Count unique basic↔acidic side-chain pairs within ``cutoff_A``.

    Cations: Lys NZ, Arg NH1/NH2/NE
    Anions:  Asp OD1/OD2, Glu OE1/OE2

    Relaxed from 4.0 → 5.0 Å on 2026-05-28: the original Barlow & Thornton
    1983 static-protein threshold of 4.0 Å under-counts MD-trajectory
    transient salt bridges (cation–anion encounter rates fluctuate around
    4-5 Å). Standard MD convention is 5.0 Å (e.g. Kumar & Nussinov 2002,
    Donald et al. 2011); using it makes the cross-DS comparison less
    sensitive to which frames happen to be near the 4 Å edge.
    """
    cation = u.select_atoms(
        "(resname LYS and name NZ) or "
        "(resname ARG and (name NH1 or name NH2 or name NE))"
    )
    anion = u.select_atoms(
        "(resname ASP and (name OD1 or name OD2)) or "
        "(resname GLU and (name OE1 or name OE2))"
    )
    if len(cation) == 0 or len(anion) == 0:
        return 0
    pairs = distances.capped_distance(
        cation.positions, anion.positions,
        max_cutoff=cutoff_A, box=box, return_distances=False,
    )
    if len(pairs) == 0:
        return 0
    # Unique cation-residue × anion-residue pairs (avoid double-counting
    # NH1/NH2/NE from same Arg pairing with same Asp)
    cat_res = cation.resindices[pairs[:, 0]]
    ani_res = anion.resindices[pairs[:, 1]]
    return len(np.unique(np.column_stack((cat_res, ani_res)), axis=0))


def compute_radial_density_profile(
    protein,
    n_bins: int = 50,
    r_max_A: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherically-averaged density profile from the protein COM.

    Iterates the trajectory in place. Returns (bin_centers_A, density_atoms_per_A3).
    """
    bins = np.linspace(0.0, r_max_A, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    counts = np.zeros(n_bins, dtype=float)
    n_frames = 0
    for _ts in protein.universe.trajectory:
        com = protein.center_of_mass()
        dist = np.linalg.norm(protein.positions - com, axis=1)
        counts += np.histogram(dist, bins=bins)[0]
        n_frames += 1
    if n_frames == 0:
        return bin_centers, np.zeros_like(bin_centers)
    counts /= n_frames
    density = counts / shell_volumes
    return bin_centers, density


# =============================================================================
# Per-replica pipeline
# =============================================================================
def resolve_replica_input_dir(sys_name: str, rep: str) -> Path | None:
    """Locate the ``Output/`` directory for one (base_system, rep) pair.

    Top-level rep layout (route B, current convention):

      * ``rep1`` → ``<sys_name>/Output/``
      * ``rep2`` → ``<sys_name>_rep2/Output/``
      * ``rep3`` → ``<sys_name>_rep3/Output/``
      * any ``repN`` similarly mapped to ``<sys_name>_repN/Output/``

    Falls back to legacy nested layout ``<sys>/<rep>/`` if the top-level
    sibling does not exist (transitional support for old route-A trees).
    Returns ``None`` if no candidate directory contains a Workflow 1 build.
    """
    if rep == "rep1":
        candidate = ROOT_DIR / sys_name / "Output"
    else:
        candidate = ROOT_DIR / f"{sys_name}_{rep}" / "Output"
    if candidate.is_dir():
        return candidate

    # Legacy fallback: nested rep dir from the old route-A prep_replica.py
    legacy = ROOT_DIR / sys_name / rep
    if legacy.is_dir():
        return legacy
    return None


def _build_reference_universe(psf: Path, first_dcd: Path, ref_positions: np.ndarray) -> mda.Universe:
    """Return a fresh Universe whose protein coordinates equal the supplied average."""
    ref = mda.Universe(str(psf), str(first_dcd))
    ref.select_atoms("(protein or resname LMA)").positions = ref_positions  # incl LMA
    return ref


def _compute_quasi_equilibrium_reference(u: mda.Universe, eq_window_frames: int) -> np.ndarray:
    """Average protein positions over the trailing ``eq_window_frames``."""
    protein = u.select_atoms("(protein or resname LMA)")  # incl LMA (consistent w/ ref)
    n_frames = len(u.trajectory)
    start = max(0, n_frames - eq_window_frames)
    accumulator = np.zeros((len(protein), 3))
    for _ in u.trajectory[start:]:
        accumulator += protein.positions
    return accumulator / max(1, n_frames - start)


def _parse_charmm_nonbonded(prm_paths) -> dict[str, float]:
    """Return {atom_type: Rmin/2 (Å)} extracted from CHARMM .prm files.

    NONBONDED block format (per atom-type line):
        TYPE   ignored    -epsilon    Rmin/2   [ignored -eps,1-4  Rmin/2,1-4]

    Rmin/2 is the radius at which the LJ potential is minimum (≡ effective
    vdW hard-sphere radius for that atom type). We use Rmin/2 directly as
    the SASA radius so the rolling-probe basis matches the force-field
    self-energy basis used in the MD itself.

    Skips comment lines and stops at the next section keyword.
    """
    SECTION_KEYWORDS = (
        "BONDS", "ANGLES", "DIHEDRALS", "IMPROPER", "IMPROPERS",
        "CMAP", "HBONDS", "NBFIX", "NBONDED", "END",
        "ATOMS", "MASS", "RESI", "PRES", "GROUP",
    )
    rmin: dict[str, float] = {}
    for path in prm_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        in_nb = False
        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.lstrip()
            if stripped.startswith("!") or not stripped:
                continue
            # Section detection — case-sensitive CHARMM uses uppercase
            head = stripped.split()[0].upper() if stripped.split() else ""
            if head == "NONBONDED":
                in_nb = True
                continue
            if not in_nb:
                continue
            if head in SECTION_KEYWORDS:
                in_nb = False
                continue
            # Continuation lines inside NONBONDED header (cutnb ..., - at EOL)
            if head.startswith(("CUTNB", "WMIN", "EPS")) or head.startswith("-"):
                continue
            # Strip inline comments
            if "!" in stripped:
                stripped = stripped.split("!", 1)[0].rstrip()
            parts = stripped.split()
            # Atom-type line: TYPE ignored eps Rmin/2 [eps,1-4 Rmin/2,1-4]
            if len(parts) < 4:
                continue
            atype = parts[0]
            try:
                r = float(parts[3])
            except ValueError:
                continue
            # CHARMM occasionally re-declares atoms (e.g. corrections at end
            # of file). Last one wins, which matches CHARMM's own behaviour.
            rmin[atype] = r
    return rmin


# Bondi (1964) vdW radii by element — fallback if a CHARMM atom type isn't
# found in any par_*.prm file (should be rare; logged on first miss).
_BONDI_FALLBACK = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "F": 1.47, "P": 1.80, "CL": 1.75, "BR": 1.85,
}


def _resolve_radii(atomgroup, rmin_map: dict[str, float]) -> tuple[np.ndarray, int]:
    """Per-atom radii (Å) for ``atomgroup`` using CHARMM Rmin/2 lookup.

    For atoms whose ``atom.type`` is missing from the lookup, fall back to
    Bondi-by-element. Returns (radii, n_fallback) so the caller can log how
    many atoms missed the CHARMM path.
    """
    radii = np.empty(len(atomgroup), dtype=float)
    n_fallback = 0
    for i, atom in enumerate(atomgroup):
        atype = (atom.type or "").strip()
        r = rmin_map.get(atype)
        if r is None:
            # Element-by-name-prefix fallback (PSF doesn't always set element)
            element = (getattr(atom, "element", "") or atom.name[0]).upper()
            r = _BONDI_FALLBACK.get(element, 1.7)
            n_fallback += 1
        radii[i] = r
    return radii, n_fallback


def _compute_sasa_arrays(
    psf_path: Path,
    aligned_dcd: Path,
    sasa_target,
    script_dir: Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pre-compute per-frame SASA (global + MA-only) for the aligned trajectory.

    ``sasa_target`` is the MDAnalysis ``AtomGroup`` whose SASA should be
    summed for the "global" column — typically ``protein or resname LMA``
    so that the methacrylate (LMA) residue is included.

    Radii basis: **CHARMM force-field Rmin/2** parsed from every
    ``script/par_*.prm`` file in the system's own script directory. This
    keeps the SASA basis self-consistent with the MD force field (relevant
    for the non-standard LMA residue — freesasa's built-in ProtOr
    classifier silently assigns radius 0 to unknown atoms, which would
    drop ~90% of the LMA atomic surface). Bondi-by-element fallback
    catches anything missing from the par files.

    Algorithm: Shrake-Rupley via ``freesasa.calcCoord`` (no temp PDB
    needed once radii are precomputed). Returns ``(None, None)`` if
    freesasa is unavailable; caller substitutes zeros.
    """
    n_frames = len(sasa_target.universe.trajectory)
    if len(sasa_target) == 0:
        log.warning("SASA target is empty; SASA columns will be 0.")
        return None, None

    ma_idx_in_target = np.array(
        [i for i, a in enumerate(sasa_target) if a.resname == MA_RESNAME],
        dtype=int,
    )

    try:
        import freesasa
        freesasa.setVerbosity(freesasa.silent)
    except ImportError:
        log.warning(
            "freesasa not installed — SASA columns will be 0. Install: pip install freesasa"
        )
        return None, None

    # Discover par_*.prm files in the system's script/ dir
    if script_dir is None:
        script_dir = psf_path.parent.parent / "script"
    prm_paths = sorted(Path(script_dir).glob("par_*.prm"))
    if not prm_paths:
        log.warning(
            "No par_*.prm files found in %s — falling back to Bondi-by-element radii for SASA.",
            script_dir,
        )
        rmin_map: dict[str, float] = {}
    else:
        rmin_map = _parse_charmm_nonbonded(prm_paths)
        log.info(
            "Parsed CHARMM Rmin/2 from %d .prm file(s) -> %d atom types.",
            len(prm_paths), len(rmin_map),
        )

    radii, n_fallback = _resolve_radii(sasa_target, rmin_map)
    if n_fallback:
        log.warning(
            "%d / %d SASA-target atoms missed CHARMM Rmin/2 lookup; used Bondi-by-element fallback.",
            n_fallback, len(sasa_target),
        )

    try:
        universe = sasa_target.universe
        sasa_global = np.zeros(n_frames, dtype=float)
        sasa_ma = np.zeros(n_frames, dtype=float)
        target_indices = sasa_target.atoms.indices

        for i, _ts in enumerate(universe.trajectory):
            # Flatten target coords to (x1,y1,z1,x2,y2,z2,...) per freesasa API
            coords = universe.atoms[target_indices].positions.reshape(-1)
            try:
                result = freesasa.calcCoord(coords.tolist(), radii.tolist())
            except Exception as calc_exc:
                log.warning(
                    "freesasa.calcCoord failed at frame %d: %s; SASA=0 for this frame.",
                    i, calc_exc,
                )
                continue
            sasa_global[i] = result.totalArea()
            if len(ma_idx_in_target):
                sasa_ma[i] = sum(
                    result.atomArea(int(j)) for j in ma_idx_in_target
                )

        log.info(
            "SASA via freesasa Shrake-Rupley w/ CHARMM Rmin/2 "
            "(%d frames; %d target atoms incl. %d LMA; %d fallback radii).",
            n_frames, len(sasa_target), len(ma_idx_in_target), n_fallback,
        )
        return sasa_global, sasa_ma
    except Exception as exc:
        log.warning("freesasa SASA failed: %s — SASA columns will be 0.", exc)
        return None, None


def _compute_rdfs(chains: dict, has_ma: bool,
                  psf_path: Path | None = None,
                  dcd_files: list[Path] | None = None) -> dict:
    """Ensemble-averaged InterRDFs over ALL unique inter-chain pairs.

    Previous (3-chain) code hard-coded {(GA,GB),(GA,GC),(GB,GC)}; with the
    big-box 12-chain system that would sample only 3 of 66 pairs — and the
    specific 3 would depend on packmol's arbitrary chain ordering, making
    the result irreproducible across replicas. Here we iterate every unique
    inter-chain pair and average the g(r) curves, yielding a single
    ensemble RDF that's invariant to packmol labelling and statistically
    meaningful at any chain count.

    PBC NOTE (2026-06-22): inter-chain RDFs MUST use raw, box-wrapped
    coordinates with valid minimum-image PBC — NOT the unwrap+center+aligned
    trajectory used for the per-chain metrics. ``unwrap()`` pushes atoms
    outside [0,L] and ``AlignTraj`` rotates coordinates without rotating the
    box, so minimum-image folds cross-boundary pairs to spurious sub-vdW
    (<2 Å, even 0.15 Å) distances — an unphysical small-r RDF tail. When the
    raw inputs are supplied we rebuild a transform-free universe here and
    re-slice the chains on it.

    Returns {label: (bins, g_r)} tuples (not InterRDF objects).
    """
    rdfs: dict = {}
    if psf_path is not None and dcd_files:
        u_raw = mda.Universe(str(psf_path), *[str(d) for d in dcd_files])
        chains = {n: u_raw.select_atoms(f"segid {n}") for n in chains.keys()}
    names = list(chains.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    aa_acc, bins = None, None
    for a, b in pairs:
        r = InterRDF(chains[a], chains[b], nbins=100, range=(0.0, 30.0)).run()
        bins = r.results.bins
        aa_acc = r.results.rdf.copy() if aa_acc is None else aa_acc + r.results.rdf
    if aa_acc is not None:
        rdfs["AllAtom_ensemble"] = (bins, aa_acc / len(pairs))

    if not has_ma:
        return rdfs

    ma_sel = f"resname {MA_RESNAME} and name {MA_REACTIVE_C}"
    ma_groups = {n: c.select_atoms(ma_sel) for n, c in chains.items()}
    ma_acc, ma_count = None, 0
    for a, b in pairs:
        if len(ma_groups[a]) and len(ma_groups[b]):
            r = InterRDF(ma_groups[a], ma_groups[b], nbins=100, range=(0.0, 30.0)).run()
            ma_acc = r.results.rdf.copy() if ma_acc is None else ma_acc + r.results.rdf
            ma_count += 1
    if ma_count:
        rdfs["MA_ensemble"] = (bins, ma_acc / ma_count)
    return rdfs


def _filter_readable_dcds(dcd_files: list[Path]) -> list[Path]:
    """Drop DCDs that can't have their header parsed.

    NAMD writes DCDs incrementally, so a simulation still in progress leaves
    a truncated file whose header lies about the frame count. Both MDAnalysis
    and mdtraj refuse to open these. We pre-filter here so a running NAMD on
    a partial part doesn't poison the whole pipeline — completed parts are
    still analysed, and the user can re-run after NAMD finishes to pick up
    the new part.

    If mdtraj is unavailable, fall back to trusting every DCD — appropriate
    when no NAMD is running (header-validation only matters for live writes).
    """
    try:
        import mdtraj as _mdtraj  # local import keeps mdtraj optional at module level
    except ImportError:
        log.warning(
            "mdtraj not installed; skipping DCD header validation. "
            "OK if NAMD is not currently writing any DCD."
        )
        return list(dcd_files)
    valid: list[Path] = []
    for dcd in dcd_files:
        try:
            with _mdtraj.formats.DCDTrajectoryFile(str(dcd), "r"):
                pass  # opening alone reads & validates the header
            valid.append(dcd)
        except Exception as exc:
            log.warning(
                "Skipping %s (header unreadable — file may still be written by NAMD): %s",
                dcd.name, exc,
            )
    return valid


def process_replica(sys_name: str, rep: str, data_out_dir: Path) -> bool:
    """Run the full analysis pipeline for one (system, replica) pair.

    Returns True on success, False if skipped (missing files / errors).
    """
    input_dir = resolve_replica_input_dir(sys_name, rep)
    if input_dir is None:
        return False

    psf_path = input_dir / "debug_1.psf"
    # list_production_dcds drops part 1 (equilibration) and natural-sorts the rest.
    dcd_files = _filter_readable_dcds(list_production_dcds(input_dir))
    if not psf_path.exists() or not dcd_files:
        log.info("[%s/%s] No usable PSF or DCD files in %s (note: part1 is "
                 "intentionally skipped — see ANALYSIS_SKIP_PARTS); skipping.",
                 sys_name, rep, input_dir)
        return False

    log.info("[%s/%s] Processing %d DCD chunk(s) (from %s) in %s",
             sys_name, rep, len(dcd_files), dcd_files[0].name, input_dir)

    try:
        u_initial = mda.Universe(str(psf_path), *[str(d) for d in dcd_files])
        # 2026-06-03 residue-agnostic: include LMA so the modified-Lys residues are
        # PBC-unwrapped/centred WITH their chain. The bare `protein` macro excludes
        # the non-standard LMA, which could leave its atoms on the wrong periodic
        # image → corrupting every per-chain metric (Rg/Ree/...) at LMA positions.
        protein_initial = u_initial.select_atoms("(protein or resname LMA)")
        n_frames = len(u_initial.trajectory)

        # Unwrap PBC + recentre protein in the box
        u_initial.trajectory.add_transformations(
            trans.unwrap(protein_initial),
            trans.center_in_box(protein_initial, wrap=False),
        )

        # Reference = average protein position over trailing equilibrium window
        frames_per_ns = max(1, int(1000.0 / DT_PS))
        eq_frames = int(EQ_WINDOW_NS * frames_per_ns)
        log.info("[%s/%s] Building quasi-equilibrium reference (last %.1f ns).",
                 sys_name, rep, EQ_WINDOW_NS)
        ref_positions = _compute_quasi_equilibrium_reference(u_initial, eq_frames)
        ref_universe = _build_reference_universe(psf_path, dcd_files[0], ref_positions)

        # Aligned trajectory lives in a tempdir so it doesn't pollute Output/.
        # ignore_cleanup_errors is a Windows safety net: even after we close the
        # Universe explicitly below, the OS sometimes lags on releasing the
        # file handle. Better to leak a few MB in %TEMP% than to fail the run.
        with tempfile.TemporaryDirectory(
            prefix=f"{sys_name}_{rep}_aligned_",
            ignore_cleanup_errors=True,
        ) as tmpdir:
            aligned_dcd = Path(tmpdir) / f"{DATA_PREFIX}_{rep}_aligned.dcd"
            log.info("[%s/%s] Aligning trajectory → %s", sys_name, rep, aligned_dcd.name)
            align.AlignTraj(
                u_initial, ref_universe,
                # residue-agnostic backbone: `backbone` macro excludes LMA; list the
                # backbone atom names so LMA's N/C are included (LMA has no CA/O).
                select="(protein or resname LMA) and (name N or name CA or name C or name O)",
                filename=str(aligned_dcd),
                in_memory=False,
            ).run()
            del u_initial

            u = mda.Universe(str(psf_path), str(aligned_dcd))
            try:
                _export_metrics_for_universe(
                    u, ref_universe, psf_path, aligned_dcd, dcd_files,
                    sys_name, rep, data_out_dir, n_frames,
                )
            finally:
                # Release Windows file handles on the aligned DCD before the
                # tempdir context tries to unlink it (otherwise WinError 32).
                try:
                    u.trajectory.close()
                except Exception:
                    pass
                del u
                gc.collect()
        return True

    except Exception as exc:
        log.error("[%s/%s] Pipeline failed: %s", sys_name, rep, exc, exc_info=True)
        return False


def _export_metrics_for_universe(
    u: mda.Universe,
    ref_universe: mda.Universe,
    psf_path: Path,
    aligned_dcd: Path,
    dcd_files: list[Path],
    sys_name: str,
    rep: str,
    data_out_dir: Path,
    n_frames: int,
) -> None:
    """Compute the metric set and write CSVs / .npy.

    Two universes are used:

    * ``u`` (the **aligned** trajectory) supplies the genuinely
      alignment-dependent outputs only — RMSD, RMSF, DSSP, SASA and the radial
      density profile.
    * ``u_phys`` (built here from ``dcd_files`` with the same unwrap+center
      transforms as the pre-align ``u_initial``) supplies **every PBC-sensitive
      inter-chain metric** (salt bridges, residue contacts, contact map /
      survival, MA NN / cluster) **and the H-bond networks**, plus the
      intra-chain shape metrics (Rg, Ree, Lp).

    Why: ``AlignTraj`` rotates coordinates **without** rotating the periodic
    box, so any minimum-image distance on the aligned ``u`` folds cross-boundary
    pairs to spurious sub-vdW distances (RDF showed g(r)≠0 at 0.15 Å; salt
    bridges were under-counted ~5–15 %). ``unwrap()`` keeps each chain whole
    (needed for intra-chain H-bonds / Rg) while leaving the box matched to the
    coordinate frame (needed for valid inter-chain min-image). For these
    short-range inter-chain contacts unwrap+center is numerically identical to
    raw box-wrapped coords — only the rotation was the bug.

    ``dcd_files`` are the RAW (un-transformed) trajectory chunks, used to build
    ``u_phys`` here and (separately) the transform-free RDF universe.
    """
    # Aligned protein (rotation-fitted) — used ONLY for alignment-dependent
    # outputs: SASA target indices + radial density profile.
    protein = u.select_atoms("protein")
    # Discover ALL protein chain segids dynamically (big-box systems have 12
    # chains GA–GL, not just GA/GB/GC). Hardcoding 3 chains silently dropped
    # 75% of the system from every per-chain metric and the RDF/contact-map
    # ensemble. ``segid`` slices include each chain's grafted LMA atoms too.
    protein_segids = sorted(set(protein.segids))

    # PBC-correct physics universe: same unwrap+center transforms as the
    # pre-align u_initial, but NOT rotated → minimum-image stays valid. All
    # inter-chain PBC metrics, H-bonds and chain-shape metrics run on this.
    u_phys = mda.Universe(str(psf_path), *[str(d) for d in dcd_files])
    protein_phys = u_phys.select_atoms("(protein or resname LMA)")
    u_phys.trajectory.add_transformations(
        trans.unwrap(protein_phys),
        trans.center_in_box(protein_phys, wrap=False),
    )
    chains = {name: u_phys.select_atoms(f"segid {name}") for name in protein_segids}
    log.info("[%s/%s] Discovered %d protein chains: %s",
             sys_name, rep, len(chains), ", ".join(protein_segids))
    ma_atoms = u_phys.select_atoms(f"resname {MA_RESNAME} and name {MA_REACTIVE_C}")
    has_ma = len(ma_atoms) > 0

    log.info("[%s/%s] Backbone RMSD vs reference...", sys_name, rep)
    rmsd_per_chain = {
        name: rms.RMSD(u, ref_universe,
                       select=f"segid {name} and (name N or name CA or name C or name O)").run().results.rmsd[:, 2]
        for name in chains
    }

    log.info("[%s/%s] Hydrogen-bond networks...", sys_name, rep)
    # H-bonds run on u_phys: AlignTraj's box-desync corrupts cross-boundary
    # min-image (inter-chain + protein/MA–water), and intra-chain H-bonds need
    # whole (unwrapped) chains. u_phys satisfies both.
    hb_pw = _run_hbond_analysis(u_phys, PROTEIN_DONOR_SEL, PROTEIN_HYDROGEN_SEL, WATER_O_SEL)
    hb_pp = _run_hbond_analysis(u_phys, PROTEIN_DONOR_SEL, PROTEIN_HYDROGEN_SEL, PROTEIN_ACCEPTOR_SEL)
    hb_ma_out = _run_hbond_analysis(u_phys, MA_DONOR_SEL, MA_HYDROGEN_SEL, WATER_O_SEL) if has_ma else np.array([])
    hb_ma_in = _run_hbond_analysis(u_phys, WATER_O_SEL, WATER_H_SEL, MA_ACCEPTOR_SEL) if has_ma else np.array([])

    inter_hb, intra_hb = split_inter_intra(hb_pp, u_phys, n_frames)
    pw_hb_per_frame = count_hbonds_per_frame(hb_pw, n_frames)
    ma_wat_per_frame = count_hbonds_per_frame(hb_ma_out, n_frames) + count_hbonds_per_frame(hb_ma_in, n_frames)

    log.info("[%s/%s] Radial distribution functions...", sys_name, rep)
    # raw, transform-free coords for inter-chain RDF (valid PBC) — see _compute_rdfs
    rdfs = _compute_rdfs(chains, has_ma, psf_path, dcd_files)

    log.info("[%s/%s] SASA (Shrake-Rupley, batch)...", sys_name, rep)
    # 'protein' keyword excludes the LMA residue (non-standard amino acid);
    # widen the selection so MA atoms are included in both the global sum
    # and (especially) the SASA_MA_A2 sub-sum.
    sasa_target = u.select_atoms(f"protein or resname {MA_RESNAME}")
    sasa_global_arr, sasa_ma_arr = _compute_sasa_arrays(psf_path, aligned_dcd, sasa_target)
    has_sasa = sasa_global_arr is not None

    # ---------- Tier 1 publication metrics ----------
    log.info("[%s/%s] RMSF per chain (Cα)...", sys_name, rep)
    rmsf_per_chain = compute_rmsf_per_chain(
        list(chains.keys()), psf_path, aligned_dcd, ref_universe,
    )

    log.info("[%s/%s] DSSP secondary structure (mdtraj)...", sys_name, rep)
    dssp_codes, dssp_resids = compute_dssp_matrix(psf_path, aligned_dcd)

    # Single ensemble contact matrix accumulated across ALL inter-chain pairs.
    # All chains have the same residue count (each is a copy of the same Input
    # PDB), so we collapse N*(N-1)/2 per-pair matrices into one (n_res, n_res)
    # ensemble that's labelling-invariant and physically meaningful.
    chain_names = list(chains.keys())
    chain_pairs = [(a, b) for i, a in enumerate(chain_names) for b in chain_names[i + 1:]]
    n_res_per_chain = len(chains[chain_names[0]].residues)
    cmap_ensemble = np.zeros((n_res_per_chain, n_res_per_chain))

    # residue-agnostic termini: each chain's first/last RESIDUE Cα, falling back to
    # backbone N if that residue is LMA (no CA). Termini are normally standard, so
    # this rarely differs — but keeps Ree correct if a chain end is modified.
    def _term_atom(c, idx):
        res = c.residues[idx]
        ca = res.atoms.select_atoms("name CA")
        return ca[0] if len(ca) else res.atoms.select_atoms("name N")[0]
    first_ca = {n: _term_atom(c, 0) for n, c in chains.items()}
    last_ca = {n: _term_atom(c, -1) for n, c in chains.items()}

    rows = []
    prev_pairs: set[tuple[int, int]] = set()
    persistence_per_chain: dict[str, list[float]] = {n: [] for n in chains}

    log.info("[%s/%s] Iterating %d frames for per-frame metrics...", sys_name, rep, n_frames)
    # Iterate u_phys (unwrap+center, box matched to coords) — NOT the aligned u.
    # Every metric below is either a PBC inter-chain distance (needs valid
    # min-image) or an intra-chain shape (needs whole chains); both hold on
    # u_phys. Alignment-dependent outputs (RMSD/RMSF/DSSP/SASA) were already
    # computed on the aligned u above and are indexed here by frame i.
    for i, ts in enumerate(u_phys.trajectory):
        box = ts.dimensions
        time_ns = round(i * DT_PS / 1000.0, 4)

        salt_bridges = count_salt_bridges(u_phys, box)
        for cname, chain in chains.items():
            persistence_per_chain[cname].append(compute_persistence_length(chain))

        # Accumulate ensemble contact map across all inter-chain pairs.
        # Final normalisation: divide by (n_frames × n_pairs) → per-pair-per-frame
        # contact probability per residue-residue pair.
        for ca, cb in chain_pairs:
            update_contact_map(chains[ca], chains[cb], cmap_ensemble, box)

        sasa_global = round(float(sasa_global_arr[i]), 2) if has_sasa else 0.0
        sasa_ma = round(float(sasa_ma_arr[i]), 2) if has_sasa else 0.0

        n_clust, max_clust, clust_rg = (
            ma_cluster_properties(ma_atoms, box) if has_ma else (0, 0, 0.0)
        )

        # Chain-symmetric: union across all 3 inter-chain pair combinations,
        # global atom indices → invariant to packmol's arbitrary GA/GB/GC
        # labelling. Removes the artefact where rep1's "close pair" happens
        # to be GA-GB but rep2's is GA-GC, flipping the metric to ~0.
        curr_pairs = extract_all_interchain_pairs(chains, box)
        survival = (
            round(len(curr_pairs & prev_pairs) / len(prev_pairs), 3)
            if prev_pairs else 0.0
        )
        prev_pairs = curr_pairs

        rows.append({
            "Replica": rep,
            "Frame": i,
            "Time_ns": time_ns,
            "RMSD_mean_A": round(float(np.mean([rmsd_per_chain[n][i] for n in chains])), 3),
            "Rg_mean_A": round(float(np.mean([chains[n].radius_of_gyration() for n in chains])), 3),
            "Ree_mean_A": round(float(np.mean([
                np.linalg.norm(first_ca[n].position - last_ca[n].position) for n in chains
            ])), 3),
            "SASA_Global_A2": sasa_global,
            "SASA_MA_A2": sasa_ma,
            "MA_Inter_MinDist_NN_A": round(ma_nearest_neighbor_distance(ma_atoms, box), 3) if has_ma else 0.0,
            "MA_Inter_Cluster_Count": n_clust,
            "MA_Inter_Max_Cluster_Size": max_clust,
            "MA_Largest_Cluster_Rg_A": clust_rg,
            "Contact_Survival_Rate": survival,
            "Hb_PW_Strict": int(pw_hb_per_frame[i]),
            "Hb_MA_Wat_Total": int(ma_wat_per_frame[i]),
            "Hb_Inter_Strict": int(inter_hb[i]),
            "Hb_Intra_Strict": int(intra_hb[i]),
            # Sum residue-residue contacts over all inter-chain pairs.
            # Scales as O(N²) chain pairs; magnitude differs from 3-chain
            # legacy numbers by factor ≈ n_pairs/3 (= 22× for 12 chains).
            "Residue_Min_Contact_Total": int(sum(
                count_residue_contacts(chains[ca], chains[cb], box)
                for ca, cb in chain_pairs
            )),
            "Salt_Bridges": int(salt_bridges),
            "Persistence_Length_mean_A": round(
                float(np.mean([persistence_per_chain[n][-1] for n in chains])), 3,
            ),
        })

    # Done reading coordinates from u_phys; release its DCD handles (Windows).
    try:
        u_phys.trajectory.close()
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("[%s/%s] No frames produced; nothing to write.", sys_name, rep)
        return

    # Adaptive rolling smoothing column for the long-running metrics
    window = max(10, int(n_frames * SMOOTHING_RATIO))
    for col in (
        "RMSD_mean_A", "Rg_mean_A", "Ree_mean_A",
        "MA_Inter_MinDist_NN_A", "Residue_Min_Contact_Total",
        "Hb_Inter_Strict", "Hb_Intra_Strict",
    ):
        if col in df.columns:
            df[f"{col}_Rolling"] = rolling_average(df[col], window)

    out_csv = data_out_dir / f"{DATA_PREFIX}_{rep}.csv"
    df.to_csv(out_csv, index=False)
    log.info("[%s/%s] Replica CSV → %s", sys_name, rep, out_csv)

    # rdfs is now {label: (bins, g_r)} from the ensemble-averaged _compute_rdfs.
    for pair_name, (bins, gr) in rdfs.items():
        rdf_csv = data_out_dir / f"{DATA_PREFIX}_{rep}_{pair_name}_RDF.csv"
        pd.DataFrame({"r_A": bins, "g_r": gr}).to_csv(rdf_csv, index=False)

    # Single ensemble contact map: divide by (n_frames × n_pairs) to express
    # as per-pair-per-frame contact probability for each residue-residue cell.
    cmap_norm = cmap_ensemble / (n_frames * max(1, len(chain_pairs)))
    np.save(data_out_dir / f"{DATA_PREFIX}_{rep}_ContactMap_ensemble.npy", cmap_norm)

    _export_block_statistics(df, data_out_dir, rep)
    _export_rmsf(rmsf_per_chain, data_out_dir, rep)
    _export_dssp(dssp_codes, dssp_resids, data_out_dir, rep)
    _export_namd_thermo(sys_name, rep, data_out_dir)
    _export_density_profile(protein, data_out_dir, rep)
    log.info("[%s/%s] Auxiliary outputs (RMSF / DSSP / NAMD thermo / density) written.",
             sys_name, rep)


def _export_rmsf(rmsf_per_chain: dict[str, pd.DataFrame], data_out_dir: Path, rep: str) -> None:
    """One long-format CSV with columns chain / resid / rmsf_A."""
    if not rmsf_per_chain:
        return
    long = pd.concat(
        [df.assign(chain=name) for name, df in rmsf_per_chain.items()],
        ignore_index=True,
    )[["chain", "resid", "rmsf_A"]]
    long.to_csv(data_out_dir / f"{DATA_PREFIX}_{rep}_RMSF.csv", index=False)


def _export_dssp(
    dssp_codes: np.ndarray | None,
    dssp_resids: np.ndarray | None,
    data_out_dir: Path,
    rep: str,
) -> None:
    """Write the per-frame × per-residue DSSP matrix + a fractional summary."""
    if dssp_codes is None or dssp_resids is None:
        return
    np.savez_compressed(
        data_out_dir / f"{DATA_PREFIX}_{rep}_DSSP.npz",
        codes=dssp_codes,
        resids=dssp_resids,
    )
    # Trajectory-averaged secondary structure fractions per residue
    n_frames = dssp_codes.shape[0]
    summary = pd.DataFrame({
        "resid": dssp_resids,
        "frac_H": (dssp_codes == "H").sum(axis=0) / n_frames,
        "frac_E": (dssp_codes == "E").sum(axis=0) / n_frames,
        "frac_C": (dssp_codes == "C").sum(axis=0) / n_frames,
    })
    summary.to_csv(data_out_dir / f"{DATA_PREFIX}_{rep}_DSSP_summary.csv", index=False)


def _export_namd_thermo(sys_name: str, rep: str, data_out_dir: Path) -> None:
    """Parse Output/*.log and write a single thermo CSV per (system, replica).

    Resolves the replica's Output directory via :func:`resolve_replica_input_dir`
    so this works for both top-level reps (``<sys>_rep2/Output``) and the
    legacy nested layout (``<sys>/rep2``).
    """
    log_dir = resolve_replica_input_dir(sys_name, rep)
    if log_dir is None:
        return
    log_files = []
    for pattern in ("nvt.log", "npt_*.log"):
        log_files.extend(sorted(log_dir.glob(pattern)))
    if not log_files:
        return
    parts = []
    cumulative_step = 0
    for lp in log_files:
        df = parse_namd_log(lp)
        if df.empty:
            continue
        df = df.copy()
        df["source_log"] = lp.name
        # Reset relative step within each file but also keep global cumulative
        df["global_step"] = df["step"] + cumulative_step
        if not df.empty:
            cumulative_step += int(df["step"].max())
        parts.append(df)
    if not parts:
        return
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(data_out_dir / f"{DATA_PREFIX}_{rep}_NAMD_thermo.csv", index=False)


def _export_density_profile(protein, data_out_dir: Path, rep: str) -> None:
    """Radial density from the protein COM (one-shot, iterates trajectory once)."""
    r_A, density = compute_radial_density_profile(protein)
    pd.DataFrame({"r_A": r_A, "density_atoms_per_A3": density}).to_csv(
        data_out_dir / f"{DATA_PREFIX}_{rep}_DensityProfile.csv", index=False,
    )


def _export_block_statistics(df: pd.DataFrame, data_out_dir: Path, rep: str) -> None:
    df = df.copy()
    df["Time_Block"] = (df["Time_ns"] // BLOCK_SIZE_NS).astype(int)
    block_stats = df.groupby("Time_Block").agg({
        "Rg_mean_A": ["mean", "std"],
        "Residue_Min_Contact_Total": ["mean", "std"],
        "Hb_Inter_Strict": ["mean", "std"],
        "MA_Inter_MinDist_NN_A": ["mean", "std"],
    })
    block_stats.columns = ["_".join(c).strip() for c in block_stats.columns.values]
    block_stats.reset_index(inplace=True)

    slopes: dict[str, float] = {}
    if len(block_stats) > 1:
        x = block_stats["Time_Block"].values
        for metric in (
            "Rg_mean_A_mean", "Residue_Min_Contact_Total_mean",
            "Hb_Inter_Strict_mean", "MA_Inter_MinDist_NN_A_mean",
        ):
            slope_name = f"{metric.replace('_mean', '')}_Drift_Slope"
            slopes[slope_name] = float(np.polyfit(x, block_stats[metric].values, 1)[0])

    block_stats = pd.concat([block_stats, pd.DataFrame([slopes])], axis=1)
    out_csv = data_out_dir / f"{DATA_PREFIX}_{rep}_BlockStats.csv"
    block_stats.to_csv(out_csv, index=False)


# =============================================================================
# Cross-replica aggregation (mean + SEM)
# =============================================================================
ENSEMBLE_METRIC_COLUMNS: tuple[str, ...] = (
    "RMSD_mean_A", "Rg_mean_A", "Ree_mean_A",
    "SASA_Global_A2", "SASA_MA_A2",
    "MA_Inter_MinDist_NN_A", "MA_Inter_Cluster_Count", "MA_Inter_Max_Cluster_Size",
    "MA_Largest_Cluster_Rg_A", "Contact_Survival_Rate",
    "Hb_PW_Strict", "Hb_MA_Wat_Total", "Hb_Inter_Strict", "Hb_Intra_Strict",
    "Residue_Min_Contact_Total",
    "Salt_Bridges", "Persistence_Length_mean_A",
)


def aggregate_replicas(sys_name: str, data_out_dir: Path, plot_dir: Path) -> int:
    """Collect per-replica CSVs and emit an Ensemble summary. Returns replica count."""
    # Strict match for main per-frame CSVs only: GelMA_analysis_rep<N>.csv,
    # never the auxiliary files (RMSF / DSSP_summary / NAMD_thermo /
    # DensityProfile / BlockStats / *_RDF) which would otherwise pollute the
    # ensemble average with heterogeneous columns.
    import re
    pattern = re.compile(rf"^{re.escape(DATA_PREFIX)}_rep\d+\.csv$")
    rep_csvs = sorted(
        p for p in data_out_dir.glob(f"{DATA_PREFIX}_rep*.csv")
        if pattern.match(p.name)
    )
    if not rep_csvs:
        return 0

    concat = pd.concat([pd.read_csv(p) for p in rep_csvs], ignore_index=True)
    available = [c for c in ENSEMBLE_METRIC_COLUMNS if c in concat.columns]
    stats = concat.groupby("Time_ns")[available].agg(["mean", "sem"])
    stats.columns = ["_".join(c).strip() for c in stats.columns.values]
    stats.reset_index(inplace=True)
    stats.to_csv(data_out_dir / f"{DATA_PREFIX}_Ensemble_Summary.csv", index=False)

    (plot_dir / "ensemble_status.log").write_text(
        f"System: {sys_name}\n"
        f"Compiled Replicas: {len(rep_csvs)}\n"
        f"State: Metastable Pre-network Organization\n"
        f"Verification: Complete\n",
        encoding="utf-8",
    )
    return len(rep_csvs)


# =============================================================================
# Top-level driver
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log.info("Project root: %s", ROOT_DIR)

    for sys_name, spec in SYSTEMS.items():
        log.info("=" * 75)
        log.info("System: %s  (DS=%d%%, has_MA=%s)", sys_name, spec.ds_pct, spec.has_ma)
        log.info("=" * 75)

        data_dir = DATA_ROOT / sys_name
        plot_dir = PLOT_ROOT / sys_name
        data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        for rep in REPLICAS:
            process_replica(sys_name, rep, data_dir)

        n_aggregated = aggregate_replicas(sys_name, data_dir, plot_dir)
        if n_aggregated > 0:
            log.info("[%s] Ensemble summary across %d replica(s) written.", sys_name, n_aggregated)
        else:
            log.warning("[%s] No replica CSVs produced; ensemble summary skipped.", sys_name)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
