"""Inspect installed encoders and their checkpoints on a Slurm GPU node."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit with sbatch on a GPU node")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(status="running", job_id=os.environ["SLURM_JOB_ID"],
                  host=platform.node(), python=sys.version, executable=sys.executable,
                  started=time.time(), imports={}, checkpoints={})
    try:
        import torch
        torch.set_num_threads(4)
        assert torch.cuda.is_available()
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        tensor = torch.arange(16, device="cuda", dtype=torch.float32)
        assert tensor.sum().item() == 120
        for name in ["torchdrug", "esm", "torch_scatter", "torch_cluster",
                     "torch_sparse", "torch_spline_conv", "torch_geometric",
                     "pykeops", "Bio", "plyfile", "pyvtk"]:
            try:
                module = importlib.import_module(name)
                report["imports"][name] = dict(version=getattr(module, "__version__", None),
                                                path=module.__file__)
            except Exception:
                report["imports"][name] = dict(error=traceback.format_exc())
            print(name, report["imports"][name], flush=True)
        for name, path in {
            "gearnet": args.root.parent / "Surface-GearNet/checkpoints/mc_gearnet_edge.pth",
            "dmasif": args.root / "third_party/dMaSIF/models/dMaSIF_search_3layer_12A_16dim",
        }.items():
            checkpoint = torch.load(str(path), map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint)
            report["checkpoints"][name] = dict(path=str(path), sha256=sha256(path),
                top_keys=list(checkpoint), tensors={key: list(value.shape)
                for key, value in state.items() if hasattr(value, "shape")})
        sys.path.insert(0, str(args.root / "third_party/dMaSIF"))
        import model
        report["dmasif_model_import"] = model.__file__
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["finished"] = time.time()
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("checkpoints", "imports")}, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
