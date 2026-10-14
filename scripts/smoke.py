"""Real-chain engineering validation; no supervised task labels are consumed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import time
import traceback

import numpy as np
import torch
from torch import nn
import esm

from surface_residue_ppi.encoders import FrozenGearNet, load_surface, surface_features
from surface_residue_ppi.structures import read_monomer
from surface_residue_ppi.adapters import neighborhoods, SurfaceAdapter
from audit_gpu import sha256


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    socket.setdefaulttimeout(60)
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    if args.limit:
        config["sample_count"] = args.limit
    out = root / "runs/smoke" / str(config["seed"]) / os.environ["SLURM_JOB_ID"]
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    report = dict(status="running", job_id=os.environ["SLURM_JOB_ID"], host=platform.node(),
                  started=time.time(), records=[], failures=[], stage="initialization",
                  purpose=config["purpose"], source_hashes={str(p.relative_to(root)): sha256(p)
                  for directory in ("src", "scripts", "tests", "slurm", "configs")
                  for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in str(p)})
    try:
        source = Path(config["source_manifest"])
        available = [record for record in json.loads(source.read_text())["records"]
                     if "/train/" in record["source"] and config["min_length"] <=
                     record["num_residues"] <= config["max_length"]]
        available.sort(key=lambda row: (row["num_residues"], row["protein_id"]))
        if len(available) < config["sample_count"]:
            raise ValueError("Insufficient source training examples")
        indices = np.linspace(0, len(available) - 1, config["sample_count"]).astype(int)
        selected = [available[i] for i in indices]
        dump(out / "manifest.json", dict(source=str(source), source_sha256=sha256(source),
                                         selection="length quantiles of source training chains", records=selected))
        report["stage"] = "surface_load"
        surface, surface_checkpoint = load_surface(root, config["dmasif"])
        report["dmasif_sha256"] = sha256(surface_checkpoint)
        report["dmasif_config_status"] = "architecture_strict_loaded; geometry from upstream benchmark/name/defaults"
        report["stage"] = "gearnet_load"
        gearnet = FrozenGearNet(config["gearnet_checkpoint"]).cuda()
        report["gearnet_sha256"] = sha256(config["gearnet_checkpoint"])
        # Validate the surface before downloading the sequence model.
        report["stage"] = "surface_first_sample"
        first = Path(config["source_project"]) / selected[0]["pdb"]
        parsed_first = read_monomer(first)
        first_surface = surface_features(surface, torch.from_numpy(parsed_first["atom_xyz"]).cuda(),
            torch.from_numpy(parsed_first["atom_types"]).cuda(), config["seed"])
        report["first_surface_shape"] = list(first_surface[config["dmasif_branch"]].shape)
        print("FIRST_SURFACE", report["first_surface_shape"], flush=True)
        report["stage"] = "esm_load"
        sequence_model, alphabet = getattr(esm.pretrained, config["esm_variant"])()
        sequence_model = sequence_model.cuda().eval().requires_grad_(False)
        converter = alphabet.get_batch_converter()
        esm_path = Path(torch.hub.get_dir()) / "checkpoints" / (config["esm_variant"] + ".pt")
        report["esm_sha256"] = sha256(esm_path)
        all_backbones = [surface, gearnet, sequence_model]
        buffers_before = [{k: v.clone() for k, v in model.named_buffers()} for model in all_backbones]
        report["stage"] = "sample_loop"
        for index, record in enumerate(selected):
            key = record["protein_id"]
            started = time.time()
            try:
                path = Path(config["source_project"]) / record["pdb"]
                assert sha256(path) == record["clean_sha256"]
                parsed = parsed_first if index == 0 else read_monomer(path)
                assert parsed["sequence"] == record["sequence"]
                atoms = torch.from_numpy(parsed["atom_xyz"]).cuda()
                atom_ids = torch.from_numpy(parsed["atom_to_residue"]).cuda()
                point_data = first_surface if index == 0 else surface_features(surface,
                    atoms, torch.from_numpy(parsed["atom_types"]).cuda(), config["seed"] + index)
                f = point_data[config["dmasif_branch"]]
                with torch.no_grad():
                    raw_gear, keys, positions = gearnet(path)
                    lookup = {key: i for i, key in enumerate(parsed["residue_keys"])}
                    if len(set(keys)) != len(keys):
                        raise ValueError("Duplicate GearNet residue identities")
                    mapping = torch.tensor([lookup[key] for key in keys], device="cuda")
                    ca = torch.from_numpy(parsed["ca_xyz"]).cuda()
                    assert torch.allclose(positions - torch.from_numpy(parsed["center"]).cuda(),
                                          ca[mapping], atol=1e-4)
                    length = len(parsed["sequence"])
                    g = torch.zeros(length, raw_gear.shape[1], device="cuda")
                    g[mapping] = raw_gear
                    valid = torch.zeros(length, device="cuda", dtype=torch.bool)
                    valid[mapping] = True
                    _, _, tokens = converter([(key, parsed["sequence"])])
                    e = sequence_model(tokens.cuda(), repr_layers=[config["esm_layer"]],
                                       return_contacts=False)["representations"][config["esm_layer"]][0, 1:length + 1]
                a, b, owner = neighborhoods(atoms, atom_ids, point_data["xyz"], length, **config["mapping"])
                projection = nn.Linear(e.shape[1] + g.shape[1], config["hidden_dim"]).cuda()
                h = projection(torch.cat([e, g], dim=-1))
                outputs = {}
                for mode, edges in [("A", a), ("B", b), ("C", b)]:
                    adapter = SurfaceAdapter(mode, config["hidden_dim"], f.shape[1], config["sigma"]).cuda()
                    augmented = adapter(h, f, edges)
                    assert torch.isfinite(augmented).all()
                    empty = torch.bincount(edges.residue, minlength=length) == 0
                    assert torch.equal(augmented[empty], h[empty])
                    # A numerical gradient check, not a task loss or model training result.
                    augmented[valid].square().mean().backward(retain_graph=True)
                    assert adapter.value.weight.grad is not None
                    assert torch.isfinite(adapter.value.weight.grad).all()
                    assert adapter.value.weight.grad.abs().sum() > 0
                    outputs[mode] = dict(shape=list(augmented.shape),
                        trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                        empty_residues=int(empty.sum()))
                assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
                cache = dict(esm=e.cpu().numpy(), gearnet=g.cpu().numpy(),
                    common_valid=valid.cpu().numpy(), point_owner=owner.cpu().numpy(),
                    atom_xyz=parsed["atom_xyz"], atom_to_residue=parsed["atom_to_residue"],
                    **{name: value.cpu().numpy() for name, value in point_data.items()})
                for name, edges in [("A", a), ("BC", b)]:
                    cache[name + "_residue"] = edges.residue.cpu().numpy()
                    cache[name + "_point"] = edges.point.cpu().numpy()
                    cache[name + "_distance"] = edges.distance.cpu().numpy()
                cache_path = out / (key + ".npz")
                np.savez_compressed(cache_path, **cache)
                with np.load(cache_path, allow_pickle=False) as replay:
                    assert all(np.array_equal(replay[name], value) for name, value in cache.items())
                dump(out / (key + ".json"), dict(residue_keys=parsed["residue_keys"],
                    sequence=parsed["sequence"], sequence_scope=parsed["sequence_scope"],
                    structure_sha256=sha256(path), sampling_seed=config["seed"] + index,
                    coordinate_center=parsed["center"].tolist(), cache_sha256=sha256(cache_path)))
                result = dict(protein_id=key, residues=length, common_valid=int(valid.sum()),
                    surface_points=len(f), esm_dim=e.shape[1], gearnet_dim=g.shape[1],
                    surface_dim=f.shape[1], adapters=outputs, seconds=time.time() - started,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), cache_bytes=cache_path.stat().st_size)
                report["records"].append(result)
                print("SAMPLE", json.dumps(result), flush=True)
            except Exception:
                report["failures"].append(dict(protein_id=key, error=traceback.format_exc()))
                print(report["failures"][-1], flush=True)
        for model, before in zip(all_backbones, buffers_before):
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(torch.equal(before[k], v) for k, v in model.named_buffers())
        report["status"] = "passed" if not report["failures"] else "partial"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["finished"] = time.time()
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(out / "report.json", report)
        print("REPORT", out / "report.json", report["status"], flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
