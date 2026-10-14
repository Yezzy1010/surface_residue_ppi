"""Phase 3 task three: frozen ESM2 protein-pair classification on D-SCRIPT.

The official test file must not be opened by this training entry point.
Earlier sequence pilot artifacts are retained with a recorded protocol deviation.  A deterministic held-out slice of the
official training subset is used for model selection; the same small
symmetrical relation head is used for every seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phase2_training_metrics import binary_metrics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_pairs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            a, b, y = line.split("\t")
            rows.append({"a": a, "b": b, "label": int(y)})
    return rows


def split_train(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Stable 80/20 split; the labels are not used to choose records."""
    scored = []
    for row in rows:
        a, b = sorted((row["a"], row["b"]))
        key = f"{seed}:{a}:{b}"
        scored.append((hashlib.sha256(key.encode()).hexdigest(), row))
    scored.sort(key=lambda x: x[0])
    cut = int(round(len(scored) * 0.8))
    train, val = [x[1] for x in scored[:cut]], [x[1] for x in scored[cut:]]
    if {r["label"] for r in train} != {0, 1} or {r["label"] for r in val} != {0, 1}:
        raise ValueError("train-only split lost a class")
    return train, val


def features(rows: list[dict], emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for row in rows:
        u, v = emb[row["a"]], emb[row["b"]]
        # This is invariant to swapping the two proteins and has no
        # structure-derived input.
        x.append(np.concatenate((u + v, np.abs(u - v), u * v)).astype(np.float32))
        y.append(row["label"])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


class SymmetricRelation(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate(model, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x).to(device)).cpu().numpy()
    m = binary_metrics(y.astype(np.int64), scores)
    m["records"] = int(len(y))
    return m, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--protocol", type=Path, default=Path("runs/phase3_task3_protocol/20260928/subset_v1"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    root = args.root.resolve()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU Slurm required")
    protocol = root / args.protocol
    job_id = os.environ["SLURM_JOB_ID"]
    out = root / "runs/phase3_task3_training" / f"{job_id}_{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "job_id": job_id, "seed": args.seed,
        "test_records_read": 0, "test_labels_read": 0,
        "protocol_sha256": sha(protocol / "report.json"), "started_at": time.time(),
        "failures": [], "model": "E", "relation_head": "symmetric_sum_absdiff_product_mlp",
    }
    save(out / "report.json", report)
    try:
        arrays = np.load(protocol / "esm_embeddings" / "embeddings.npz", allow_pickle=False)
        keys = arrays["keys"].astype(str).tolist()
        values = arrays["embeddings"].astype(np.float32)
        if values.ndim != 2 or len(keys) != len(values) or values.shape[1] != 480:
            raise ValueError(f"unexpected ESM cache shape {values.shape}")
        emb = {k: values[i] for i, k in enumerate(keys)}
        train_rows = read_pairs(protocol / "train.tsv")
        if any(r["a"] not in emb or r["b"] not in emb for r in train_rows):
            raise ValueError("training pair missing from ESM cache")
        fit_rows, val_rows = split_train(train_rows, 20260928)
        x_fit, y_fit = features(fit_rows, emb)
        x_val, y_val = features(val_rows, emb)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = SymmetricRelation(x_fit.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(1, int(y_fit.sum())); neg = max(1, len(y_fit) - pos)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
        best_ap, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
        batch_size = 256
        for epoch in range(1, args.epochs + 1):
            model.train(); order = np.random.default_rng(args.seed + epoch).permutation(len(y_fit)); losses=[]
            for start in range(0, len(order), batch_size):
                ix = order[start:start + batch_size]
                logits = model(torch.from_numpy(x_fit[ix]).to(device))
                loss = loss_fn(logits, torch.from_numpy(y_fit[ix]).to(device))
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(float(loss.detach().cpu()))
            vm, _ = evaluate(model, x_val, y_val, device)
            row = {"epoch": epoch, "train_bce": float(np.mean(losses)), "validation": vm}
            history.append(row); save(out / "E_history.json", history)
            print("EPOCH", args.seed, json.dumps(row), flush=True)
            if vm["average_precision"] > best_ap + 1e-4:
                best_ap = vm["average_precision"]; best_epoch = epoch; stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 5:
                break
        if best_state is None:
            raise ValueError("no task-three checkpoint")
        model.load_state_dict(best_state)
        checkpoint = out / "E.pt"; torch.save({"seed": args.seed, "epoch": best_epoch, "model": best_state}, checkpoint)
        val_metrics, val_scores = evaluate(model, x_val, y_val, device)
        np.savez_compressed(out / "val_predictions.npz", labels=y_val, scores=val_scores)
        report.update({
            "status": "passed", "scope": "developmental validation only",
            "train_records": len(fit_rows), "validation_records": len(val_rows),
            "official_train_records": len(train_rows), "test_records_read": 0,
            "test_labels_read": 0, "validation": val_metrics,
            "best_epoch": best_epoch, "epochs_run": len(history),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": sha(checkpoint), "embedding_sha256": sha(protocol / "esm_embeddings" / "embeddings.npz"),
        })
    except Exception:
        report["status"] = "failed"; report["failures"].append(traceback.format_exc()); print(report["failures"][-1], flush=True)
    finally:
        report.update(finished_at=time.time(), wall_seconds=time.time() - report["started_at"], peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        save(out / "report.json", report); print("REPORT", out / "report.json", report["status"], flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
