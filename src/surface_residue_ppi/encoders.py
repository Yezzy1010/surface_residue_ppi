"""Strict, frozen wrappers around the three upstream encoders."""
from types import SimpleNamespace
import sys
import torch
from torch import nn


class FrozenGearNet(nn.Module):
    output_dim = 3072

    def __init__(self, checkpoint):
        super().__init__()
        from torchdrug import layers, models
        from torchdrug.layers import geometry
        self.model = models.GearNet(input_dim=21, hidden_dims=[512] * 6,
            num_relation=7, edge_input_dim=59, num_angle_bin=8, batch_norm=True,
            concat_hidden=True, short_cut=True, readout="sum")
        self.graph = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[geometry.SequentialEdge(max_distance=2),
                geometry.SpatialEdge(radius=10., min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5)], edge_feature="gearnet")
        self.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, path):
        from rdkit import Chem
        from torchdrug import data
        molecule = Chem.MolFromPDBFile(str(path), proximityBonding=False)
        if molecule is None:
            raise ValueError("RDKit cannot parse %s" % path)
        protein = data.Protein.from_molecule(molecule, atom_feature=None, bond_feature=None)
        graph = self.graph(data.Protein.pack([protein]).to(next(self.model.parameters()).device))
        graph.view = "residue"
        output = self.model(graph, graph.node_feature.float())["node_feature"]
        keys = [(data.Protein.id2alphabet[int(c)], int(n),
                 data.Protein.id2alphabet[int(i)].strip(), data.Protein.id2residue[int(t)])
                for c, n, i, t in zip(graph.chain_id.cpu(), graph.residue_number.cpu(),
                                     graph.insertion_code.cpu(), graph.residue_type.cpu())]
        return output, keys, graph.node_position


def load_surface(root, config):
    sys.path.insert(0, str(root / "third_party/dMaSIF"))
    from model import dMaSIF
    checkpoint = root / "third_party/dMaSIF/models/dMaSIF_search_3layer_12A_16dim"
    model = dMaSIF(SimpleNamespace(**config))
    saved = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.requires_grad_(False)
    return model.cuda().eval(), checkpoint


@torch.no_grad()
def surface_features(model, atoms, types, seed):
    # no_grad allows the upstream surface generator's local enable_grad block.
    # inference_mode would prevent the coordinate derivatives it needs.
    # KeOps 1.x rejects strided tensors; LMDB/DataFrame filtering can produce
    # non-contiguous views even when the underlying NumPy array looks dense.
    atoms = atoms.contiguous()
    types = types.contiguous()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    protein = dict(atoms=atoms, atom_xyz=atoms, atomtypes=types,
                   batch_atoms=torch.zeros(len(atoms), device=atoms.device, dtype=torch.long).contiguous(),
                   mesh_labels=None)
    model.preprocess_surface(protein)
    model.embed(protein)
    return {key: protein[key].detach() for key in
            ("xyz", "normals", "input_features", "embedding_1", "embedding_2")}
