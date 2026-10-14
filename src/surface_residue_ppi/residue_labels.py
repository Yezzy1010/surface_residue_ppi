"""Diagnostic protein-only contact reconstruction with explicit residue IDs.

The canonical-only scope is provisional, not a finalized training protocol.
Observed sequence is explicitly not a full biological/polymer sequence.
"""
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))
IDENTITY_COLUMNS = ["structure", "model", "chain", "hetero", "residue", "insertion_code"]
INPUT_COLUMNS = IDENTITY_COLUMNS + ["resname", "element", "name", "altloc", "x", "y", "z"]


def canonical_monomer(frame):
    """Create partner-independent inputs; unknown chemistry is counted, not mapped.

    Retain the original selected alternate atom when there is one atom per name.
    Ambiguous duplicate atom sites or multiple residue names at one identity fail
    explicitly. Do not delete a residue from labels because it lacks a C alpha.
    """
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing monomer fields: %s" % sorted(missing))
    # Only the allowed columns can participate in the monomer content hash.
    atoms = frame[INPUT_COLUMNS].copy()
    if len(atoms[["structure", "model", "chain"]].drop_duplicates()) != 1:
        raise ValueError("Expected one structure/model/chain")
    for column in ("hetero", "insertion_code", "resname", "element", "name", "altloc"):
        if atoms[column].isna().any():
            raise ValueError("Missing identity/atom value: %s" % column)
        atoms[column] = atoms[column].astype(str).str.strip()
    keep = atoms["hetero"].eq("") & atoms["resname"].isin(AA)
    excluded_atoms = int((~keep).sum())
    atoms = atoms[keep].copy()
    if atoms.empty:
        raise ValueError("No canonical nonhetero amino acids")
    xyz = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite canonical coordinates")
    keys, names, index_by_key = [], [], {}
    atom_to_residue, atom_ids = [], set()
    ca_counts = []
    for row in atoms.itertuples(index=False, name=None):
        key = tuple(row[:len(IDENTITY_COLUMNS)])
        resname, element, atom_name, altloc = row[len(IDENTITY_COLUMNS):len(IDENTITY_COLUMNS)+4]
        if key not in index_by_key:
            index_by_key[key] = len(keys)
            keys.append(key)
            names.append(resname)
            ca_counts.append(0)
        index = index_by_key[key]
        if names[index] != resname:
            raise ValueError("Multiple residue names at one full residue identity")
        site = (key, atom_name)
        if site in atom_ids:
            raise ValueError("Ambiguous duplicate/alternate atom site; explicit resolution required")
        atom_ids.add(site)
        atom_to_residue.append(index)
        ca_counts[index] += int(atom_name == "CA" and element.upper() == "C")
    ca_valid = np.asarray(ca_counts, dtype=np.int64) == 1
    payload = atoms.to_dict(orient="split")
    payload.pop("index")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(residue_keys=keys, residue_names=names, observed_sequence="".join(AA[n] for n in names),
                sequence_scope="observed_canonical_residues_only_not_full_polymer_sequence",
                xyz=xyz, elements=atoms["element"].to_numpy(),
                atom_to_residue=np.asarray(atom_to_residue, dtype=np.int64),
                ca_valid=ca_valid, excluded_atoms=excluded_atoms,
                selected_altloc_atoms=int(atoms["altloc"].ne("").sum()),
                monomer_input_sha256=hashlib.sha256(encoded.encode()).hexdigest())


def protein_contacts(left, right, cutoff=6.0):
    """Strict upstream distance rule, using complete full-identity residue pairs."""
    if cutoff <= 0:
        raise ValueError("Cutoff must be positive")
    # Upstream excludes H exactly; do not silently change isotope treatment.
    masks = [side["elements"] != "H" for side in (left, right)]
    coords = [side["xyz"][mask] for side, mask in zip((left, right), masks)]
    indices = [side["atom_to_residue"][mask] for side, mask in zip((left, right), masks)]
    pairs = set()
    if len(coords[0]) and len(coords[1]):
        tree = cKDTree(coords[1])
        for start in range(0, len(coords[0]), 256):
            for offset, neighbors in enumerate(tree.query_ball_point(coords[0][start:start+256], cutoff)):
                i = start+offset
                for j in neighbors:
                    if np.linalg.norm(coords[0][i]-coords[1][j]) < cutoff:
                        pairs.add((int(indices[0][i]), int(indices[1][j])))
    labels = [np.zeros(len(side["residue_keys"]), dtype=np.bool_) for side in (left, right)]
    for i, j in pairs:
        labels[0][i] = True
        labels[1][j] = True
    return pairs, labels
