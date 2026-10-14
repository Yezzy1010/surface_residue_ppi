"""Audit released contact labels without feeding partner geometry to encoders.

This module has no model dependencies. Contact reconstruction follows the
pinned ATOM3D neighbors.py: element != 'H', strict distance < cutoff, and
deduplication using its six identity fields. Insertion-code collisions are
reported separately; they must be resolved before residue targets are made.
"""
import gzip
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


KEY_COLUMNS = ["ensemble", "subunit", "structure", "model", "chain", "residue"]


def decode(value, fmt):
    if value is None:
        raise ValueError("Missing LMDB value")
    if fmt == "json":
        return json.loads(value)
    if fmt == "pkl":
        # Caller must first verify the official archive checksum.
        return pickle.loads(value)
    if fmt == "msgpack":
        import msgpack
        return msgpack.unpackb(value, raw=False)
    raise ValueError("Unsupported serialization format: %s" % fmt)


class ReleasedLMDB:
    """Read ATOM3D's gzip records without importing PyTorch or model code."""

    def __init__(self, path):
        self.path = Path(path)
        self.env = lmdb.open(str(path), readonly=True, lock=False,
                             readahead=False, meminit=False, max_readers=2)
        with self.env.begin() as txn:
            self.count = int(txn.get(b"num_examples"))
            self.format = txn.get(b"serialization_format").decode()
            self.id_to_idx = decode(txn.get(b"id_to_idx"), self.format)
        if len(self.id_to_idx) != self.count:
            self.close()
            raise ValueError("LMDB id map count differs from num_examples")
        if set(self.id_to_idx.values()) != set(range(self.count)):
            self.close()
            raise ValueError("LMDB id map does not cover contiguous record indices")

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        with self.env.begin() as txn:
            value = txn.get(str(index).encode())
        if value is None:
            raise ValueError("Missing record %d" % index)
        item = decode(gzip.decompress(value), self.format)
        if not isinstance(item, dict):
            raise ValueError("Expected a dictionary record")
        for key, typename in item.get("types", {}).items():
            if typename == str(pd.DataFrame) and isinstance(item.get(key), dict):
                item[key] = pd.DataFrame(**item[key])
        return item

    def close(self):
        self.env.close()


def residue_keys(frame):
    return list(frame[KEY_COLUMNS].itertuples(index=False, name=None))


def heavy_contacts(left, right, cutoff=6.0):
    """Complete positives, without an all-atoms dense distance matrix."""
    result = set()
    left = left[left["element"] != "H"]
    right = right[right["element"] != "H"]
    xyz0 = left[["x", "y", "z"]].to_numpy(dtype=np.float64)
    xyz1 = right[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz0).all() or not np.isfinite(xyz1).all():
        raise ValueError("Non-finite atom coordinates")
    if not len(left) or not len(right):
        return result
    tree = cKDTree(xyz1)
    keys0, keys1 = residue_keys(left), residue_keys(right)
    for start in range(0, len(left), 256):
        for offset, neighbors in enumerate(tree.query_ball_point(xyz0[start:start + 256], cutoff)):
            i = start + offset
            for j in neighbors:
                # query_ball_point includes the boundary; upstream uses <.
                if np.linalg.norm(xyz0[i] - xyz1[j]) < cutoff:
                    result.add((keys0[i], keys1[j]))
    return result


def audit_record(item):
    atoms, labels = item["atoms_pairs"], item["atoms_neighbors"]
    if not isinstance(atoms, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise ValueError("Expected atoms_pairs and atoms_neighbors DataFrames")
    required = KEY_COLUMNS + ["element", "name", "x", "y", "z"]
    missing = set(required) - set(atoms.columns)
    if missing:
        raise ValueError("Missing atom fields: %s" % sorted(missing))
    if not np.isfinite(atoms[["x", "y", "z"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite coordinates")
    subunits = list(atoms["subunit"].unique())
    if len(subunits) != 2:
        raise ValueError("DIPS audit requires two bound subunits, got %d" % len(subunits))
    sides = [atoms[atoms["subunit"] == value] for value in subunits]
    keys = [set(residue_keys(side)) for side in sides]
    if "label" not in labels or not labels["label"].eq(1).all():
        raise ValueError("Release labels contain non-positive or absent label values")
    pairs0 = labels[[c + "0" for c in KEY_COLUMNS]].itertuples(index=False, name=None)
    pairs1 = labels[[c + "1" for c in KEY_COLUMNS]].itertuples(index=False, name=None)
    released = set()
    for a, b in zip(pairs0, pairs1):
        if a in keys[0] and b in keys[1]:
            released.add((a, b))
        elif b in keys[0] and a in keys[1]:
            released.add((b, a))
        else:
            raise ValueError("Contact references an absent or ambiguous subunit residue")
    reconstructed = heavy_contacts(*sides)
    issues, details = [], []
    for side, side_keys in zip(sides, keys):
        full_key = KEY_COLUMNS + [c for c in ("hetero", "insertion_code", "resname") if c in side]
        collision_count = len(side[full_key].drop_duplicates()) - len(side_keys)
        atom_identity = KEY_COLUMNS + [c for c in ("hetero", "insertion_code", "name", "altloc") if c in side]
        duplicate_atoms = int(side.duplicated(atom_identity).sum())
        ca_keys = set(residue_keys(side[side["name"] == "CA"]))
        altlocs = (int(side["altloc"].fillna("").astype(str).str.strip().ne("").sum())
                   if "altloc" in side else None)
        if collision_count:
            issues.append("upstream_residue_key_collision")
        if duplicate_atoms:
            issues.append("duplicate_atom_identity")
        if len(side[["structure", "model", "chain"]].drop_duplicates()) != 1:
            issues.append("subunit_not_single_chain_model")
        details.append(dict(atoms=len(side), residues=len(side_keys),
                            residues_without_ca=len(side_keys - ca_keys),
                            upstream_key_collisions=collision_count,
                            duplicate_atoms=duplicate_atoms, alternate_atoms=altlocs,
                            deuterium_atoms=int(side["element"].eq("D").sum())))
    missing, extra = reconstructed - released, released - reconstructed
    if missing or extra:
        issues.append("released_contacts_differ_from_heavy_lt_6A")
    if len(labels) != len(released):
        issues.append("duplicate_released_contacts")
    def examples(values):
        return [[list(a), list(b)] for a, b in sorted(values, key=repr)[:5]]
    return dict(id=item["id"], atoms_columns=list(atoms.columns),
                labels_columns=list(labels.columns), subunits=list(map(str, subunits)),
                sides=details, released_positive_rows=len(labels),
                unique_released_positives=len(released), reconstructed_positives=len(reconstructed),
                missing_released_positives=len(missing), extra_released_positives=len(extra),
                missing_examples=examples(missing), extra_examples=examples(extra),
                complete_candidate_pairs=len(keys[0]) * len(keys[1]),
                interface_residue_counts=[len({pair[i] for pair in released}) for i in (0, 1)],
                operational_negative_pairs=(len(keys[0]) * len(keys[1]) - len(released)
                                            if not issues else None),
                issues=sorted(set(issues)))
