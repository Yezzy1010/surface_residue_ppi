"""Parse a single cleaned monomer, preserving explicit residue identities."""
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def read_monomer(path):
    models = list(PDBParser(QUIET=True).get_structure("monomer", str(path)))
    if len(models) != 1 or len(list(models[0])) != 1:
        raise ValueError("Smoke input must contain exactly one model and one chain")
    keys, ca, coords, atom_to_residue, atom_types, letters = [], [], [], [], [], []
    elements = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
    for chain in models[0]:
        for residue in chain:
            if residue.id[0] != " ":
                raise ValueError("Unresolved nonstandard residue %s" % (residue.id,))
            if any(atom.is_disordered() for atom in residue):
                raise ValueError("Resolve alternate locations before extraction")
            index = len(keys)
            keys.append((chain.id, residue.id[1], residue.id[2].strip(), residue.resname))
            letters.append(seq1(residue.resname))
            ca.append(residue["CA"].coord if "CA" in residue else [np.nan] * 3)
            for atom in residue:
                if atom.element in ("H", "D"):
                    continue
                if atom.element not in elements:
                    raise ValueError("Unsupported atom element %s" % atom.element)
                coords.append(atom.coord)
                atom_to_residue.append(index)
                atom_types.append(elements[atom.element])
    coordinates = np.asarray(coords, dtype=np.float32)
    if not len(coordinates) or len(set(keys)) != len(keys) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid/duplicate residue identities or coordinates")
    center = coordinates.mean(0)
    return dict(sequence="".join(letters), residue_keys=keys,
                sequence_scope="observed_residues_only_engineering_smoke",
                atom_xyz=coordinates - center,
                atom_to_residue=np.asarray(atom_to_residue, dtype=np.int64),
                atom_types=np.eye(6, dtype=np.float32)[atom_types],
                ca_xyz=np.asarray(ca, dtype=np.float32) - center,
                center=center)
