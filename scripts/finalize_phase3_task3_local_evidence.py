"""Export missing representation/mask provenance using fixed trained checkpoints only."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from train_phase2_task1 import sha256
from train_phase3_task3_local import (ATOL, RTOL, CELLS, PairModel, batch_rows,
                                     load_data, local_logmeanexp, pair_feature, save_json)

RUNS = ("276590_20260927", "276592_20260928", "276589_20260929")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "Slurm GPU required")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path.cwd()
    out = root / "runs/phase3_task3_local_evidence" / os.environ["SLURM_JOB_ID"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = root / "runs/phase3_task3_protocol/20260928/structure_v2"
    require(sha256(protocol / "report.json") == "ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593", "Protocol changed")
    pr = json.loads((protocol / "report.json").read_text())
    for name, digest in pr["artifact_sha256"].items():
        require(sha256(protocol / name) == digest, "Manifest changed: " + name)
    train = json.loads((protocol / "train.json").read_text())
    val = json.loads((protocol / "val.json").read_text())
    examples, digests, backbone = load_data(root, protocol, train, val)
    mask_manifest = {k: dict(length=len(v["mask"]), valid=int(v["mask"].sum()),
                            sha256_uint8=hashlib.sha256(v["mask"].astype(np.uint8).tobytes()).hexdigest())
                     for k, v in examples.items()}
    save_json(out / "common_masks.json", mask_manifest)
    historic = root / "runs/phase3_task3_structure_training/276331_20260927"
    norm_hash = sha256(historic / "normalization.json")
    features_hash = sha256(historic / "feature_hashes.json")
    require(norm_hash == "740b96b1585febb2c3be8c451f8ac445214cb5cf46228463bce36aece99044ca", "Historical normalization changed")
    require(json.loads((historic / "feature_hashes.json").read_text()) == digests, "Cache provenance changed")
    stats = {k: {q: np.asarray(v, np.float32) if q != "count" else v for q, v in item.items()}
             for k, item in json.loads((historic / "normalization.json").read_text()).items()}
    results = []
    try:
        for run in RUNS:
            directory = root / "runs/phase3_task3_local" / run
            report = json.loads((directory / "report.json").read_text())
            require(report["status"] == "passed" and len(report["results"]) == 4, "Incomplete run")
            require(sha256(directory / "normalization.json") == norm_hash, "Normalization changed")
            require(sha256(directory / "feature_hashes.json") == features_hash, "Feature list changed")
            require(report["source_sha256"] == sha256(root / "scripts/train_phase3_task3_local.py"), "Training source changed")
            for item in report["results"]:
                seed, cell = report["seed"], item["cell"]
                require(cell in CELLS, "Unexpected cell")
                cp = Path(item["checkpoint"])
                require(sha256(cp) == item["checkpoint_sha256"], "Checkpoint changed")
                payload = torch.load(cp, map_location="cpu")
                require(payload["seed"] == seed and payload["cell"] == cell and payload["epoch"] == item["best_epoch"], "Checkpoint metadata mismatch")
                variant, pooling = cell.rsplit("-", 1)
                model = PairModel(variant, pooling).cuda().eval()
                model.load_state_dict(payload["state_dict"], strict=True)
                zab, zba, sab, sba = [], [], [], []
                with torch.no_grad():
                    for start in range(0, len(val), 24):
                        batch, lengths, ix, jx, _, _ = batch_rows(val[start:start+24], examples, stats, "cuda")
                        proteins = model.encode(batch, lengths)
                        for i, j in zip(ix.tolist(), jx.tolist()):
                            a, b = proteins[i], proteins[j]
                            if pooling == "MEAN":
                                ab = pair_feature(a.mean(0), b.mean(0))
                                ba = pair_feature(b.mean(0), a.mean(0))
                            else:
                                ab, ba = local_logmeanexp(a, b), local_logmeanexp(b, a)
                            zab.append(ab.cpu().numpy()); zba.append(ba.cpu().numpy())
                            sab.append(model.relation(ab.unsqueeze(0)).item())
                            sba.append(model.relation(ba.unsqueeze(0)).item())
                zab, zba, sab, sba = map(np.asarray, (zab, zba, sab, sba))
                original_path = Path(item["predictions"])
                require(sha256(original_path) == item["predictions_sha256"], "Prediction changed")
                with np.load(original_path, allow_pickle=False) as archive:
                    original = {k: archive[k] for k in archive.files}
                rows = np.asarray([r["source_row"] for r in val])
                na = np.asarray([mask_manifest[r["a"]]["valid"] for r in val])
                nb = np.asarray([mask_manifest[r["b"]]["valid"] for r in val])
                require(np.array_equal(original["source_rows"], rows), "Validation order changed")
                require(np.array_equal(original["candidate_counts"], na * nb), "Candidate counts changed")
                require(np.array_equal(original["labels"], [r["label"] for r in val]), "Labels changed")
                for side in ("a", "b"):
                    require(np.array_equal(original[side], [r[side] for r in val]), "Protein IDs changed")
                require(all(np.isfinite(x).all() for x in (zab, zba, sab, sba)), "Nonfinite evidence")
                representation_pass = np.allclose(zab, zba, atol=ATOL, rtol=RTOL)
                score_pass = np.allclose(sab, sba, atol=ATOL, rtol=RTOL)
                original_pass = np.allclose(sab, original["logits"], atol=ATOL, rtol=RTOL)
                path = out / f"{seed}_{cell.lower().replace('-', '_')}_details.npz"
                np.savez_compressed(path, **original, valid_a=na, valid_b=nb,
                                    representation_ab=zab, representation_ba=zba,
                                    score_ab=sab, score_ba=sba,
                                    probability_ab=1/(1+np.exp(-sab)), probability_ba=1/(1+np.exp(-sba)),
                                    predicted_label_ab=(sab >= 0).astype(np.int8),
                                    representation_mean=zab.mean(1), representation_std=zab.std(1),
                                    representation_min=zab.min(1), representation_max=zab.max(1),
                                    representation_l2=np.linalg.norm(zab, axis=1))
                result = dict(seed=seed, cell=cell, file=str(path), sha256=sha256(path),
                              checkpoint_sha256=sha256(cp), strict_load=True,
                              representation_shape=list(zab.shape), representation_pass=bool(representation_pass),
                              representation_max_abs=float(np.abs(zab-zba).max()), score_pass=bool(score_pass),
                              score_max_abs=float(np.abs(sab-sba).max()), original_score_pass=bool(original_pass),
                              original_score_max_abs=float(np.abs(sab-original["logits"]).max()),
                              relation_parameters=sum(p.numel() for p in model.relation.parameters()))
                results.append(result)
                print(json.dumps(result), flush=True)
                require(representation_pass and score_pass and original_pass, "Fixed tolerance failed; stop interpretation")
        files = ["scripts/train_phase3_task3_local.py", "scripts/train_phase2_task1.py", "scripts/phase2_training_metrics.py",
                 "src/surface_residue_ppi/adapters.py", "scripts/finalize_phase3_task3_local_evidence.py"]
        env = dict(python=platform.python_version(), executable=os.sys.executable, torch=torch.__version__,
                   numpy=np.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   packages={name: importlib.metadata.version(name) for name in ("scikit-learn", "torch-scatter", "scipy")},
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                   num_threads=torch.get_num_threads(),
                   nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True))
        save_json(out / "report.json", dict(status="passed", results=results, job_id=os.environ["SLURM_JOB_ID"],
                  host=platform.node(), started_at=started, ended_at=time.time(), environment=env,
                  normalization_sha256=norm_hash, feature_hashes_sha256=features_hash,
                  common_masks_sha256=sha256(out / "common_masks.json"), backbone_sha256=backbone,
                  source_hashes={name:sha256(root/name) for name in files},
                  protocol_hashes={name:sha256(protocol/name) for name in ("report.json", "train.json", "val.json", "proteins.json")},
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), atol=ATOL, rtol=RTOL,
                  test_records_read=0, test_labels_read=0))
    except Exception:
        import traceback
        save_json(out / "report.json", dict(status="failed", results=results, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
