"""Strict-load and exchange audit for completed task3 local-readout runs."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from train_phase2_task1 import fit_statistics, sha256
from train_phase3_task3_local import (
    ATOL, CELLS, RTOL, SEEDS, PairModel, batch_rows, load_data,
)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def audit_one(model, rows, examples, stats, device, batch_size):
    model.eval()
    labels, scores_ab, scores_ba, counts = [], [], [], []
    aa, bb, source_rows = [], [], []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch, lengths, ix, jx, y, _ = batch_rows(selected, examples, stats, device)
        proteins = model.encode(batch, lengths)
        ab = torch.stack([model.score_embeddings(proteins[int(i)], proteins[int(j)])
                          for i, j in zip(ix.tolist(), jx.tolist())])
        ba = torch.stack([model.score_embeddings(proteins[int(j)], proteins[int(i)])
                          for i, j in zip(ix.tolist(), jx.tolist())])
        labels.extend(y.cpu().numpy().tolist())
        scores_ab.extend(ab.cpu().numpy().tolist())
        scores_ba.extend(ba.cpu().numpy().tolist())
        counts.extend([int(examples[r["a"]]["mask"].sum()) *
                       int(examples[r["b"]]["mask"].sum()) for r in selected])
        aa.extend(r["a"] for r in selected)
        bb.extend(r["b"] for r in selected)
        source_rows.extend(int(r["source_row"]) for r in selected)
    y = np.asarray(labels, dtype=np.int8)
    ab = np.asarray(scores_ab, dtype=np.float64)
    ba = np.asarray(scores_ba, dtype=np.float64)
    delta = np.abs(ab - ba)
    return dict(
        labels=y, score_ab=ab, score_ba=ba, candidate_counts=np.asarray(counts, dtype=np.int64),
        a=np.asarray(aa), b=np.asarray(bb), source_rows=np.asarray(source_rows, dtype=np.int64),
        symmetry=dict(
            max_abs=float(delta.max(initial=0.0)), mean_abs=float(delta.mean()),
            median_abs=float(np.median(delta)), q95=float(np.quantile(delta, .95)),
            q99=float(np.quantile(delta, .99)), pass_=bool(np.allclose(ab, ba, atol=ATOL, rtol=RTOL)),
            atol=ATOL, rtol=RTOL,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path,
                    default=Path("runs/phase3_task3_protocol/20260928/structure_v2"))
    ap.add_argument("--runs", type=Path, default=Path("runs/phase3_task3_local"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/phase3_task3_local_exchange_audit"))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(".").resolve()
    report = json.loads((args.protocol / "report.json").read_text())
    if report.get("version") != "task3-structure-v2" or report.get("test_records_read") or report.get("test_labels_read"):
        raise ValueError("Invalid protocol/test access")
    for name, digest in report["artifact_sha256"].items():
        if sha256(args.protocol / name) != digest:
            raise ValueError("Protocol artifact changed: " + name)
    train_rows = json.loads((args.protocol / "train.json").read_text())
    val_rows = json.loads((args.protocol / "val.json").read_text())
    examples, digests, backbone = load_data(root, args.protocol, train_rows, val_rows)
    stats = fit_statistics([examples[k] for k in sorted({x for r in train_rows for x in (r["a"], r["b"])})])
    device = torch.device("cuda")
    out = args.out / os.environ["SLURM_JOB_ID"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    results = []
    for seed in SEEDS:
        candidates = sorted(args.runs.glob(f"*_{seed}/report.json"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one report for seed {seed}, found {candidates}")
        seed_report_path = candidates[0]
        seed_report = json.loads(seed_report_path.read_text())
        if seed_report.get("status") != "passed" or int(seed_report.get("seed")) != seed:
            raise ValueError("Invalid training report: " + str(seed_report_path))
        for item in seed_report["results"]:
            cell = item["cell"]
            if cell not in CELLS:
                raise ValueError("Unexpected cell: " + cell)
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            if not checkpoint.is_file() or sha256(checkpoint) != item["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch: " + str(checkpoint))
            variant, pooling = cell.rsplit("-", 1)
            model = PairModel(variant, pooling, hidden=report["training"]["hidden_dim"]).to(device)
            payload = torch.load(checkpoint, map_location=device)
            if int(payload.get("seed")) != seed or payload.get("cell") != cell:
                raise ValueError("Checkpoint metadata mismatch: " + str(checkpoint))
            model.load_state_dict(payload["state_dict"], strict=True)
            audit = audit_one(model, val_rows, examples, stats, device, report["training"]["batch_size"])
            expected_a = [r["a"] for r in val_rows]
            expected_b = [r["b"] for r in val_rows]
            expected_rows = [int(r["source_row"]) for r in val_rows]
            if audit["a"].tolist() != expected_a or audit["b"].tolist() != expected_b or audit["source_rows"].tolist() != expected_rows:
                raise ValueError("Validation order mismatch")
            if not np.array_equal(audit["labels"], np.asarray([r["label"] for r in val_rows], dtype=np.int8)):
                raise ValueError("Validation labels mismatch")
            pred = out / f"{seed}_{cell.lower().replace('-', '_')}_exchange.npz"
            np.savez_compressed(pred, labels=audit["labels"], score_ab=audit["score_ab"],
                                score_ba=audit["score_ba"], candidate_counts=audit["candidate_counts"],
                                source_rows=audit["source_rows"], a=audit["a"], b=audit["b"])
            results.append(dict(seed=seed, cell=cell, checkpoint=str(checkpoint),
                                checkpoint_sha256=sha256(checkpoint), strict_load=True,
                                predictions=str(pred), predictions_sha256=sha256(pred),
                                symmetry=audit["symmetry"], labels_count=int(len(audit["labels"])),
                                test_records_read=0, test_labels_read=0))
    final = dict(status="passed", kind="exchange_audit", job_id=os.environ["SLURM_JOB_ID"],
                 host=platform.node(), gpu=torch.cuda.get_device_name(), protocol_sha256=sha256(args.protocol / "report.json"),
                 source_sha256=sha256(Path(__file__)), feature_hashes_sha256=hashlib_sha(digests),
                 backbone_sha256=backbone, results=results, atol=ATOL, rtol=RTOL,
                 train_pairs=len(train_rows), val_pairs=len(val_rows), test_records_read=0,
                 test_labels_read=0, started_at=started, wall_seconds=time.time() - started,
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save_json(out / "report.json", final)
    print(json.dumps(final, indent=2), flush=True)


def hashlib_sha(digests):
    import hashlib
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    main()
