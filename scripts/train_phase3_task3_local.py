"""Task3 ES/ES-B mean versus complete local-pair readouts."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, random, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from train_phase2_task1 import Readout, fit_statistics, make_batch, sha256
from phase2_training_metrics import binary_metrics

SEEDS = (20260927, 20260928, 20260929)
CELLS = ("ES-MEAN", "ES-LOCAL", "ES-B-MEAN", "ES-B-LOCAL")
ATOL, RTOL, TAU, LOCAL_BLOCK_PAIRS = 1e-6, 1e-5, 1.0, 32768

def pair_feature(left, right):
    return torch.cat((left + right, torch.abs(left - right), left * right), dim=-1)

def local_logmeanexp(left, right, tau=TAU, block_pairs=LOCAL_BLOCK_PAIRS):
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]: raise ValueError("Local inputs must be residue matrices with equal width")
    na, nb = int(left.shape[0]), int(right.shape[0])
    if na < 1 or nb < 1 or tau <= 0 or block_pairs < 1: raise ValueError("Invalid local candidate set")
    running, count = None, 0
    block_rows = max(1, block_pairs // nb)
    for i0 in range(0, na, block_rows):
        phi = pair_feature(left[i0:i0 + block_rows, None, :], right[None, :, :]).reshape(-1, left.shape[1] * 3)
        n = int(phi.shape[0]); term = torch.logsumexp(phi / tau, dim=0)
        running = term if running is None else torch.logaddexp(running, term); count += n
    return tau * (running - math.log(count))

def local_logmeanexp_flat(left, right, tau=TAU):
    phi = pair_feature(left[:, None, :], right[None, :, :]).reshape(-1, left.shape[1] * 3)
    return tau * torch.logsumexp(phi / tau, dim=0) - tau * math.log(phi.shape[0])

class PairModel(nn.Module):
    def __init__(self, variant, pooling, hidden=256):
        super().__init__()
        if variant not in ("ES", "ES-B") or pooling not in ("MEAN", "LOCAL"): raise ValueError("Invalid cell")
        self.variant, self.pooling = variant, pooling
        self.encoder = Readout(variant, hidden); self.encoder.head = nn.Identity()
        self.relation = nn.Sequential(nn.Linear(3 * hidden, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def encode(self, batch, lengths):
        enc = self.encoder
        h = enc.project(batch["esm"] if enc.variant == "E" else torch.cat((batch["esm"], batch["gearnet"]), dim=-1))
        if enc.adapter is not None: h = enc.adapter(h, batch["surface"], batch["B"])
        result, offset = [], 0
        for n in lengths:
            mask = batch["mask"][offset:offset + n]
            if not bool(mask.any()): raise ValueError("Empty common-valid mask")
            result.append(h[offset:offset + n][mask]); offset += n
        return result
    def score_embeddings(self, left, right):
        rep = pair_feature(left.mean(0, keepdim=True), right.mean(0, keepdim=True)) if self.pooling == "MEAN" else local_logmeanexp(left, right).unsqueeze(0)
        return self.relation(rep).squeeze()
    def forward(self, batch, lengths, ix, jx):
        proteins = self.encode(batch, lengths)
        return torch.stack([self.score_embeddings(proteins[int(i)], proteins[int(j)]) for i, j in zip(ix.tolist(), jx.tolist())])

def batch_rows(rows, examples, stats, device):
    keys = list(dict.fromkeys(k for r in rows for k in (r["a"], r["b"]))); bykey = {k: i for i, k in enumerate(keys)}; ex = [examples[k] for k in keys]
    b = make_batch(ex, stats, device); ix = torch.tensor([bykey[r["a"]] for r in rows], device=device); jx = torch.tensor([bykey[r["b"]] for r in rows], device=device); y = torch.tensor([r["label"] for r in rows], device=device, dtype=torch.float32)
    return b, [len(x["mask"]) for x in ex], ix, jx, y, keys

@torch.no_grad()
def evaluate(model, rows, examples, stats, device, batch_size, symmetry=False):
    model.eval(); labels, scores, swapped, counts = [], [], [], []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]; b, lengths, ix, jx, y, _ = batch_rows(selected, examples, stats, device); proteins = model.encode(b, lengths)
        logits = torch.stack([model.score_embeddings(proteins[int(i)], proteins[int(j)]) for i, j in zip(ix.tolist(), jx.tolist())])
        if symmetry: swapped.extend([model.score_embeddings(proteins[int(j)], proteins[int(i)]).item() for i, j in zip(ix.tolist(), jx.tolist())])
        labels.extend(y.cpu().numpy().tolist()); scores.extend(logits.cpu().numpy().tolist()); counts.extend([int(examples[r["a"]]["mask"].sum()) * int(examples[r["b"]]["mask"].sum()) for r in selected])
    y, s = np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64); metrics = binary_metrics(y, s)
    if abs(metrics["average_precision"] - average_precision_score(y, s)) > 1e-10 or abs(metrics["auroc"] - roc_auc_score(y, s)) > 1e-10: raise ValueError("Metric implementation mismatch")
    result = dict(metrics=metrics, labels=y, logits=s, candidate_counts=np.asarray(counts, dtype=np.int64))
    if symmetry:
        rev = np.asarray(swapped, dtype=np.float64); delta = np.abs(s - rev); result["symmetry"] = dict(max_abs=float(delta.max(initial=0.0)), mean_abs=float(delta.mean()), pass_=bool(np.allclose(s, rev, atol=ATOL, rtol=RTOL)), atol=ATOL, rtol=RTOL)
    return result

def load_data(root, protocol, train_rows, val_rows):
    proteins = {r["id"]: r for r in json.loads((protocol / "proteins.json").read_text())}; ids = {x for row in train_rows + val_rows for x in (row["a"], row["b"])}; examples, digests, backbone = {}, {}, None; base = root / "runs/phase3_task3_features/structure_v1/cache"
    for key in sorted(ids):
        cache = base / (key + ".npz"); meta = json.loads(cache.with_suffix(".json").read_text())
        if sha256(cache) != meta["cache_sha256"] or key not in proteins or meta["pdb_sha256"] != proteins[key]["pdb_sha256"] or meta["sequence_sha256"] != proteins[key]["sequence_sha256"]: raise ValueError("Feature provenance mismatch: " + key)
        with np.load(cache, allow_pickle=False) as z: a = {name: z[name] for name in z.files}
        n = len(proteins[key]["sequence"])
        required = ("esm", "gearnet", "surface", "common_valid", "A_residue", "A_point", "A_distance", "BC_residue", "BC_point", "BC_distance")
        if any(name not in a for name in required) or a["esm"].shape != (n, 480) or a["gearnet"].shape != (n, 3072) or a["common_valid"].shape != (n,) or not a["common_valid"].any(): raise ValueError("Feature dimension/mask mismatch: " + key)
        if not all(np.isfinite(a[name]).all() for name in ("esm", "gearnet", "surface")): raise ValueError("Nonfinite features: " + key)
        if backbone is None: backbone = meta["checkpoint_sha256"]
        if meta["checkpoint_sha256"] != backbone: raise ValueError("Backbone mismatch")
        examples[key] = dict(esm=a["esm"].astype(np.float32), gearnet=a["gearnet"].astype(np.float32), surface=a["surface"].astype(np.float32), mask=a["common_valid"].astype(bool), labels=np.zeros(n, dtype=np.float32), A_residue=a["A_residue"].astype(np.int64), A_point=a["A_point"].astype(np.int64), A_distance=a["A_distance"].astype(np.float32), BC_residue=a["BC_residue"].astype(np.int64), BC_point=a["BC_point"].astype(np.int64), BC_distance=a["BC_distance"].astype(np.float32)); digests[key] = meta["cache_sha256"]
    return examples, digests, backbone

def save_json(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def train_cell(cell, seed, train_rows, val_rows, examples, stats, cfg, device, out):
    variant, pooling = cell.rsplit("-", 1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model = PairModel(variant, pooling, cfg["hidden_dim"]).to(device); opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]); positives = sum(r["label"] for r in train_rows); loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_rows) - positives) / positives, device=device)); best, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); order = list(range(len(train_rows))); random.shuffle(order); losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            selected = [train_rows[i] for i in order[start:start + cfg["batch_size"]]]; b, lengths, ix, jx, y, _ = batch_rows(selected, examples, stats, device); loss = loss_fn(model(b, lengths, ix, jx), y)
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
        ev = evaluate(model, val_rows, examples, stats, device, cfg["batch_size"], symmetry=True); row = dict(epoch=epoch, train_bce=float(np.mean(losses)), validation=ev["metrics"], symmetry=ev["symmetry"]); history.append(row); save_json(out / (cell.lower().replace("-", "_") + "_history.json"), history); print("EPOCH", cell, seed, json.dumps(row), flush=True)
        if ev["metrics"]["average_precision"] > best + cfg["min_delta"]: best, best_epoch, stale, best_state = ev["metrics"]["average_precision"], epoch, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= cfg["patience"]: break
    if best_state is None: raise ValueError("No checkpoint selected")
    model.load_state_dict(best_state); checkpoint = out / (cell.lower().replace("-", "_") + ".pt"); torch.save(dict(seed=seed, cell=cell, epoch=best_epoch, state_dict=best_state), checkpoint); ev = evaluate(model, val_rows, examples, stats, device, cfg["batch_size"], symmetry=True); pred = out / (cell.lower().replace("-", "_") + "_val_predictions.npz"); np.savez_compressed(pred, labels=ev["labels"], logits=ev["logits"], candidate_counts=ev["candidate_counts"], source_rows=np.asarray([r["source_row"] for r in val_rows]), a=np.asarray([r["a"] for r in val_rows]), b=np.asarray([r["b"] for r in val_rows]))
    return dict(cell=cell, seed=seed, validation=ev["metrics"], symmetry=ev["symmetry"], best_epoch=best_epoch, epochs_run=len(history), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), predictions=str(pred), predictions_sha256=sha256(pred), trainable_parameters=sum(p.numel() for p in model.parameters()))

def preflight(root, protocol, train_rows, val_rows, out, device):
    examples, digests, backbone = load_data(root, protocol, train_rows, val_rows); counts = []
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            na, nb = int(examples[row["a"]]["mask"].sum()), int(examples[row["b"]]["mask"].sum()); counts.append(dict(split=split, source_row=row["source_row"], a=row["a"], b=row["b"], valid_a=na, valid_b=nb, candidates=na * nb))
    save_json(out / "candidate_counts.json", counts); torch.manual_seed(7); left, right = torch.randn(5, 7, device=device), torch.randn(4, 7, device=device); full, chunked = local_logmeanexp_flat(left, right), local_logmeanexp(left, right, block_pairs=6); toy_delta = float((full - chunked).abs().max().item())
    if not torch.allclose(full, chunked, atol=ATOL, rtol=RTOL): raise ValueError("Toy mismatch")
    stats = fit_statistics([examples[k] for k in sorted({x for r in train_rows[:4] for x in (r["a"], r["b"])})]); cfg = dict(hidden_dim=256, learning_rate=1e-3, weight_decay=1e-4, batch_size=2, epochs=1, patience=1, min_delta=1e-4)
    for cell in CELLS:
        variant, pooling = cell.rsplit("-", 1); model = PairModel(variant, pooling, hidden=256).to(device); b, lengths, ix, jx, y, _ = batch_rows(train_rows[:2], examples, stats, device); logits = model(b, lengths, ix, jx); loss = nn.BCEWithLogitsLoss()(logits, y); loss.backward()
        if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise ValueError("Preflight nonfinite: " + cell)
    report = dict(status="passed", kind="preflight", job_id=os.environ.get("SLURM_JOB_ID"), host=platform.node(), gpu=torch.cuda.get_device_name(), protocol_sha256=sha256(protocol / "report.json"), source_sha256=sha256(Path(__file__)), train_pairs=len(train_rows), val_pairs=len(val_rows), train_proteins=len({x for r in train_rows for x in (r["a"], r["b"])}), val_proteins=len({x for r in val_rows for x in (r["a"], r["b"])}), feature_hashes_sha256=hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(), backbone_sha256=backbone, candidate_count=len(counts), max_candidates=max(x["candidates"] for x in counts), mean_candidates=float(np.mean([x["candidates"] for x in counts])), toy_max_abs_delta=toy_delta, atol=ATOL, rtol=RTOL, tau=TAU, local_block_pairs=LOCAL_BLOCK_PAIRS, test_records_read=0, test_labels_read=0, peak_gpu_bytes=torch.cuda.max_memory_allocated()); save_json(out / "preflight_report.json", report); print(json.dumps(report, indent=2), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed-index", type=int); ap.add_argument("--preflight", action="store_true"); ap.add_argument("--protocol", type=Path, default=Path("runs/phase3_task3_protocol/20260928/structure_v2")); args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(): raise RuntimeError("Slurm GPU required")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; root, protocol = Path(".").resolve(), args.protocol; report = json.loads((protocol / "report.json").read_text())
    if report.get("version") != "task3-structure-v2" or report.get("test_records_read") or report.get("test_labels_read"): raise ValueError("Invalid protocol/test access")
    for name, digest in report["artifact_sha256"].items():
        if sha256(protocol / name) != digest: raise ValueError("Protocol artifact changed: " + name)
    train_rows, val_rows = json.loads((protocol / "train.json").read_text()), json.loads((protocol / "val.json").read_text()); seed = None if args.preflight else SEEDS[args.seed_index]; out = root / "runs/phase3_task3_local" / ((os.environ["SLURM_JOB_ID"] + "_preflight") if args.preflight else (os.environ["SLURM_JOB_ID"] + "_" + str(seed))); out.mkdir(parents=True, exist_ok=False); device = torch.device("cuda"); started = time.time()
    try:
        if args.preflight: preflight(root, protocol, train_rows, val_rows, out, device); return
        examples, digests, backbone = load_data(root, protocol, train_rows, val_rows); stats = fit_statistics([examples[k] for k in sorted({x for r in train_rows for x in (r["a"], r["b"])})]); save_json(out / "normalization.json", {k: {q: (v.tolist() if isinstance(v, np.ndarray) else v) for q, v in d.items()} for k, d in stats.items()}); save_json(out / "feature_hashes.json", digests); results = [train_cell(cell, seed, train_rows, val_rows, examples, stats, report["training"], device, out) for cell in CELLS]; final = dict(status="passed", kind="training", seed=seed, job_id=os.environ["SLURM_JOB_ID"], host=platform.node(), gpu=torch.cuda.get_device_name(), protocol_sha256=sha256(protocol / "report.json"), source_sha256=sha256(Path(__file__)), feature_hashes_sha256=sha256(out / "feature_hashes.json"), backbone_sha256=backbone, train_pairs=len(train_rows), val_pairs=len(val_rows), results=results, tau=TAU, local_block_pairs=LOCAL_BLOCK_PAIRS, atol=ATOL, rtol=RTOL, test_records_read=0, test_labels_read=0, started_at=started, wall_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_gpu_bytes=torch.cuda.max_memory_allocated(), training=report["training"]); save_json(out / "report.json", final); print(json.dumps(final, indent=2), flush=True)
    except Exception:
        save_json(out / "report.json", dict(status="failed", seed=seed, job_id=os.environ.get("SLURM_JOB_ID"), traceback=traceback.format_exc(), started_at=started, wall_seconds=time.time()-started)); raise

if __name__ == "__main__": main()
