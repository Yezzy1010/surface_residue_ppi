"""Sparse point-to-residue adapters with identical B/C neighborhoods."""
from typing import NamedTuple
import math

import torch
from torch import nn
from torch_scatter import scatter_min, scatter_softmax


class Neighborhood(NamedTuple):
    residue: torch.Tensor
    point: torch.Tensor
    distance: torch.Tensor
    residue_count: int


@torch.no_grad()
def neighborhoods(atoms, atom_to_residue, points, residue_count, r_max=4.0,
                  epsilon=1.0, chunk_size=256):
    """Distances use every heavy atom; stable ties go to the lowest residue index."""
    if r_max <= 0 or epsilon < 0 or chunk_size < 1 or residue_count < 1:
        raise ValueError("Invalid neighborhood configuration")
    if atoms.ndim != 2 or atoms.shape[1] != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Coordinates must be N x 3")
    if atom_to_residue.shape != (len(atoms),) or len(atoms) == 0:
        raise ValueError("An atom-to-residue index is required for every atom")
    if atom_to_residue.dtype != torch.long:
        raise ValueError("Residue indices must be int64")
    if atom_to_residue.min() < 0 or atom_to_residue.max() >= residue_count:
        raise ValueError("Residue index out of range")
    if not torch.isfinite(atoms).all() or not torch.isfinite(points).all():
        raise ValueError("Nonfinite coordinates")
    blocks_a, blocks_b, owners = [], [], []
    present = torch.bincount(atom_to_residue, minlength=residue_count) > 0
    for start in range(0, len(points), chunk_size):
        distances = torch.cdist(atoms, points[start:start + chunk_size],
                                compute_mode="donot_use_mm_for_euclid_dist")
        per_residue = scatter_min(distances, atom_to_residue[:, None],
                                  dim=0, dim_size=residue_count)[0]
        per_residue[~present] = float("inf")
        nearest, owner = per_residue.min(dim=0)
        owners.append(owner)
        legal = (per_residue <= r_max) & (per_residue - nearest[None] <= epsilon)
        rr, pp = legal.nonzero(as_tuple=True)
        blocks_b.append((rr, pp + start, per_residue[rr, pp]))
        keep = nearest <= r_max
        pp_a = torch.arange(len(owner), device=points.device)[keep]
        blocks_a.append((owner[keep], pp_a + start, nearest[keep]))

    def combine(blocks):
        if not blocks:
            index = atom_to_residue.new_empty(0)
            return Neighborhood(index, index.clone(), points.new_empty(0), residue_count)
        return Neighborhood(*(torch.cat([b[i] for b in blocks]) for i in range(3)),
                            residue_count)

    owner = torch.cat(owners) if owners else atom_to_residue.new_empty(0)
    return combine(blocks_a), combine(blocks_b), owner


class SurfaceAdapter(nn.Module):
    def __init__(self, mode, hidden_dim, surface_dim, sigma=2.0):
        super().__init__()
        if mode not in ("A", "B", "C") or sigma <= 0:
            raise ValueError("Invalid adapter mode or sigma")
        self.mode, self.sigma = mode, sigma
        self.value = nn.Linear(surface_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        if mode == "C":
            self.query = nn.Sequential(nn.LayerNorm(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.key = nn.Sequential(nn.LayerNorm(surface_dim),
                                     nn.Linear(surface_dim, hidden_dim, bias=False))

    def forward(self, residue_features, point_features, neighborhood):
        rr, pp, distance, count = neighborhood
        if count != len(residue_features):
            raise ValueError("Neighborhood/residue count mismatch")
        if len(rr) == 0:
            return residue_features
        logits = -distance.square() / (2 * self.sigma ** 2)
        if self.mode == "C":
            logits = logits + (self.query(residue_features)[rr] *
                                self.key(point_features)[pp]).sum(-1) / math.sqrt(
                                    residue_features.shape[-1])
        weights = scatter_softmax(logits, rr, dim=0)
        values = self.value(point_features)[pp] * weights[:, None]
        pooled = residue_features.new_zeros(residue_features.shape).index_add(0, rr, values)
        valid = (torch.bincount(rr, minlength=count) > 0)[:, None]
        gate = torch.sigmoid(self.gate(torch.cat([residue_features, pooled], dim=-1)))
        return residue_features + valid * gate * self.output(pooled)
