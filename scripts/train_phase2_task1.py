"""Train Phase 2 task-one readouts on frozen train/validation feature caches.

The test split is never opened.  All backbone features are precomputed and
frozen; only the shared projection, optional surface adapter, and task head
are optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surface_residue_ppi.adapters import Neighborhood, SurfaceAdapter
from phase2_training_metrics import binary_metrics, macro_metrics


VARIANTS = ("E", "ES", "S", "ES-A", "ES-B", "ES-C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_manifest(path: Path, expected_split: str) -> dict:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Invalid {expected_split} manifest: {path}")
    return {(row["split"], int(row["lmdb_index"])): row for row in rows}


def load_examples(root: Path, feature_report: dict, pilot_manifest: list, train_manifest: dict,
                  val_manifest: dict, strict_val_manifest: dict):
    if feature_report.get("status") != "passed" or feature_report.get("test_records_read") != 0:
        raise ValueError("Final feature report is not a passed train/val-only run")
    if feature_report.get("records_requested") != len(feature_report.get("records", [])):
        raise ValueError("Feature report record count is incomplete")
    pilot_by_key = {(row["split"], int(row["lmdb_index"])): row for row in pilot_manifest}
    allowed = set(train_manifest) | set(val_manifest)
    if allowed != set(pilot_by_key):
        raise ValueError("Train/val manifests do not exactly cover the frozen pilot manifest")
    if set(val_manifest) != set(strict_val_manifest):
        raise ValueError("The frozen validation set is not the strict cluster-disjoint validation set")
    train_clusters = {cluster for row in train_manifest.values() for cluster in row["cluster_pair"]}
    val_clusters = {cluster for row in val_manifest.values() for cluster in row["cluster_pair"]}
    if train_clusters & val_clusters:
        raise ValueError("Train/validation cluster overlap detected")

    report_records = {(row["split"], int(row["lmdb_index"])): row
                      for row in feature_report["records"]}
    if set(report_records) != allowed:
        raise ValueError("Feature records do not match the frozen train/val manifests")
    examples = {"train": [], "val": []}
    positive_counts = {"train": 0, "val": 0}
    valid_counts = {"train": 0, "val": 0}
    record_counts = {"train": 0, "val": 0}
    for key in sorted(allowed):
        split, index = key
        manifest_record = pilot_by_key[key]
        feature_record = report_records[key]
        if feature_record["id"] != manifest_record["id"]:
            raise ValueError(f"Feature record ID mismatch at {key}")
        record_counts[split] += 1
        for side_index, manifest_side in enumerate(manifest_record["sides"]):
            side_report = feature_record["sides"][side_index]
            expected_key = manifest_side.get('seqres_key', f"{manifest_record['pdb'].lower()}_{manifest_side['chain']}")
            if side_report["seqres_key"] != expected_key:
                raise ValueError(f"SEQRES key mismatch at {key}, side {side_index}")
            cache_path = root / side_report["cache"]
            label_path = cache_path.with_name(cache_path.stem + "_labels.npy")
            if not cache_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing feature/label pair: {cache_path}")
            with np.load(cache_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            labels = np.load(label_path, allow_pickle=False)
            length = int(side_report["sequence_length"])
            mask = arrays["common_valid"].astype(np.bool_, copy=False)
            if arrays["esm"].shape != (length, 480):
                raise ValueError(f"Invalid ESM cache shape at {cache_path}: {arrays['esm'].shape}")
            if arrays["gearnet"].shape != (length, 3072):
                raise ValueError(f"Invalid GearNet cache shape at {cache_path}: {arrays['gearnet'].shape}")
            if arrays["surface"].ndim != 2 or arrays["surface"].shape[1] != 16:
                raise ValueError(f"Invalid surface cache shape at {cache_path}: {arrays['surface'].shape}")
            if labels.shape != (length,) or mask.shape != (length,):
                raise ValueError(f"Feature/label sequence-length mismatch at {cache_path}")
            if int(mask.sum()) != int(side_report["common_valid"]):
                raise ValueError(f"Common-mask mismatch at {cache_path}")
            expected_mask = np.zeros(length, dtype=np.bool_)
            expected_labels = np.zeros(length, dtype=np.int8)
            for residue in manifest_side['common_valid']:
                position = int(residue['seq_id']) - 1
                if not 0 <= position < length or expected_mask[position]:
                    raise ValueError(f'Invalid/duplicate sequence index in manifest at {cache_path}')
                expected_mask[position] = True
                expected_labels[position] = int(residue['positive'])
            if not np.array_equal(mask, expected_mask) or not np.array_equal(labels, expected_labels):
                raise ValueError(f'Cache mask/labels differ from frozen residue identities at {cache_path}')
            if np.any(labels[~mask] != 0) or np.any((labels != 0) & (labels != 1)):
                raise ValueError(f"Labels outside the common mask or nonbinary labels at {cache_path}")
            for residue_name, point_name, distance_name in (("A_residue", "A_point", "A_distance"),
                                                            ("BC_residue", "BC_point", "BC_distance")):
                residue_edges = arrays[residue_name]
                point_edges = arrays[point_name]
                distance_edges = arrays[distance_name]
                if not (len(residue_edges) == len(point_edges) == len(distance_edges)):
                    raise ValueError(f"Neighborhood edge arrays differ in length at {cache_path}")
                if (len(residue_edges) and (residue_edges.min() < 0 or residue_edges.max() >= length
                        or point_edges.min() < 0 or point_edges.max() >= len(arrays["surface"]))):
                    raise ValueError(f"Neighborhood index outside cache bounds at {cache_path}")
                if not np.isfinite(distance_edges).all():
                    raise ValueError(f"Nonfinite neighborhood distances at {cache_path}")
            positives = int(labels[mask].sum())
            if positives != int(manifest_side["common_positive_count"]):
                raise ValueError(f"Label count mismatch at {cache_path}: {positives}")
            for name in ("esm", "gearnet", "surface", "surface_xyz"):
                if not np.isfinite(arrays[name]).all():
                    raise ValueError(f"Nonfinite {name} values at {cache_path}")
            example = dict(split=split, key=key, side=side_index, cache=cache_path,
                           esm=arrays["esm"].astype(np.float32, copy=False),
                           gearnet=arrays["gearnet"].astype(np.float32, copy=False),
                           surface=arrays["surface"].astype(np.float32, copy=False),
                           mask=mask, labels=labels.astype(np.float32, copy=False),
                           A_residue=arrays["A_residue"].astype(np.int64, copy=False),
                           A_point=arrays["A_point"].astype(np.int64, copy=False),
                           A_distance=arrays["A_distance"].astype(np.float32, copy=False),
                           BC_residue=arrays["BC_residue"].astype(np.int64, copy=False),
                           BC_point=arrays["BC_point"].astype(np.int64, copy=False),
                           BC_distance=arrays["BC_distance"].astype(np.float32, copy=False))
            examples[split].append(example)
            positive_counts[split] += positives
            valid_counts[split] += int(mask.sum())
    return examples, dict(record_counts=record_counts, chain_examples={k: len(v) for k, v in examples.items()},
                          valid_residues=valid_counts, positive_residues=positive_counts,
                          train_cluster_count=len(train_clusters), val_cluster_count=len(val_clusters),
                          cluster_overlap_count=0)


def fit_statistics(examples: list) -> dict:
    sums = {name: None for name in ("esm", "gearnet", "surface")}
    squares = {name: None for name in sums}
    counts = {name: 0 for name in sums}
    for example in examples:
        for name, values, select in (("esm", example["esm"], example["mask"]),
                                     ("gearnet", example["gearnet"], example["mask"]),
                                     ("surface", example["surface"],
                                      np.ones(len(example["surface"]), dtype=np.bool_))):
            selected = values[select].astype(np.float64, copy=False)
            if not len(selected):
                continue
            subtotal = selected.sum(axis=0, dtype=np.float64)
            subsquare = np.square(selected).sum(axis=0, dtype=np.float64)
            sums[name] = subtotal if sums[name] is None else sums[name] + subtotal
            squares[name] = subsquare if squares[name] is None else squares[name] + subsquare
            counts[name] += len(selected)
    result = {}
    for name in sums:
        mean = sums[name] / counts[name]
        variance = np.maximum(squares[name] / counts[name] - mean * mean, 1e-8)
        result[name] = dict(mean=mean.astype(np.float32), std=np.sqrt(variance).astype(np.float32),
                            count=int(counts[name]))
    return result


def make_batch(examples: list, stats: dict, device: torch.device) -> dict:
    residue_offset = 0
    point_offset = 0
    result = {name: [] for name in ("esm", "gearnet", "surface", "mask", "labels",
                                    "A_r", "A_p", "A_d", "B_r", "B_p", "B_d")}
    for example in examples:
        result["esm"].append((example["esm"] - stats["esm"]["mean"]) / stats["esm"]["std"])
        result["gearnet"].append((example["gearnet"] - stats["gearnet"]["mean"]) / stats["gearnet"]["std"])
        result["surface"].append((example["surface"] - stats["surface"]["mean"]) / stats["surface"]["std"])
        result["mask"].append(example["mask"])
        result["labels"].append(example["labels"])
        result["A_r"].append(example["A_residue"] + residue_offset)
        result["A_p"].append(example["A_point"] + point_offset)
        result["A_d"].append(example["A_distance"])
        result["B_r"].append(example["BC_residue"] + residue_offset)
        result["B_p"].append(example["BC_point"] + point_offset)
        result["B_d"].append(example["BC_distance"])
        residue_offset += len(example["esm"])
        point_offset += len(example["surface"])
    def floating(name):
        return torch.from_numpy(np.concatenate(result[name], axis=0)).to(device=device, dtype=torch.float32)
    def integer(name):
        return torch.from_numpy(np.concatenate(result[name], axis=0)).to(device=device, dtype=torch.long)
    batch = dict(esm=floating("esm"), gearnet=floating("gearnet"), surface=floating("surface"),
                 mask=torch.from_numpy(np.concatenate(result["mask"])).to(device=device, dtype=torch.bool),
                 labels=floating("labels"), A=Neighborhood(integer("A_r"), integer("A_p"), floating("A_d"), residue_offset),
                 B=Neighborhood(integer("B_r"), integer("B_p"), floating("B_d"), residue_offset))
    return batch


class Readout(nn.Module):
    def __init__(self, variant: str, hidden: int = 256):
        super().__init__()
        self.variant = variant
        self.surface_mode = variant[-1] if variant.startswith("ES-") else ("A" if variant == "S" else None)
        if variant == "E":
            input_dim = 480
        elif variant in ("ES", "ES-A", "ES-B", "ES-C"):
            input_dim = 480 + 3072
        else:
            input_dim = None
        self.project = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)) if input_dim else None
        self.adapter = SurfaceAdapter(self.surface_mode, hidden, 16, sigma=2.0) if self.surface_mode else None
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        if self.project is None:
            h = batch["esm"].new_zeros((len(batch["esm"]), 256))
        elif self.variant == "E":
            h = self.project(batch["esm"])
        else:
            h = self.project(torch.cat((batch["esm"], batch["gearnet"]), dim=-1))
        if self.adapter is not None:
            edges = batch["A"] if self.surface_mode == "A" else batch["B"]
            h = self.adapter(h, batch["surface"], edges)
        return self.head(h).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, examples: list, stats: dict, batch_size: int, device: torch.device) -> dict:
    model.eval()
    labels, scores, instance_metrics = [], [], []
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        batch = make_batch(selected, stats, device)
        logits = model(batch)
        mask = batch["mask"]
        batch_labels = batch["labels"].cpu().numpy().astype(np.int8)
        batch_scores = logits.cpu().numpy()
        residue_offset = 0
        for example in selected:
            length = len(example["labels"])
            example_mask = example["mask"]
            example_labels = batch_labels[residue_offset:residue_offset + length][example_mask]
            example_scores = batch_scores[residue_offset:residue_offset + length][example_mask]
            row = binary_metrics(example_labels, example_scores)
            instance_metrics.append(row)
            labels.append(example_labels)
            scores.append(example_scores)
            residue_offset += length
    pooled = binary_metrics(np.concatenate(labels), np.concatenate(scores))
    return macro_metrics(instance_metrics, pooled)


def train_variant(name: str, train_examples: list, val_examples: list, stats: dict, args,
                  pos_weight: float, device: torch.device, out: Path) -> dict:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Readout(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    best_ap = -float("inf")
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.shuffle(order)
        loss_numerator = 0.0
        seen_residues = 0
        for start in range(0, len(order), args.batch_size):
            selected = [train_examples[i] for i in order[start:start + args.batch_size]]
            batch = make_batch(selected, stats, device)
            logits = model(batch)
            mask = batch["mask"]
            losses = criterion(logits, batch["labels"])
            loss = losses[mask].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_numerator += float(loss.detach()) * int(mask.sum())
            seen_residues += int(mask.sum())
        val_metrics = evaluate(model, val_examples, stats, args.eval_batch_size, device)
        train_loss = loss_numerator / max(1, seen_residues)
        history.append(dict(epoch=epoch, train_weighted_bce=train_loss, validation=val_metrics))
        print("EPOCH", name, json.dumps(history[-1]), flush=True)
        score = val_metrics["average_precision"]
        if score > best_ap + args.min_delta:
            best_ap = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No valid best checkpoint for {name}")
    model.load_state_dict(best_state)
    torch.save(dict(variant=name, seed=seed, epoch=best_epoch, state_dict=best_state), out / f"{name.replace('-', '_')}.pt")
    return dict(variant=name, seed=seed, best_epoch=best_epoch, best_validation=best_metrics,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                epochs_run=len(history), seconds=time.time() - started, history=history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--feature-run", default="runs/phase2_features/275876_final_v2")
    parser.add_argument("--protocol-dir", default="runs/phase2_protocol/20260925/local_20250925_phase2_protocol_01")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Submit Phase 2 training through Slurm on a GPU node")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    feature_dir = root / args.feature_run
    protocol_dir = root / args.protocol_dir
    feature_report_path = feature_dir / "report.json"
    feature_report = json.loads(feature_report_path.read_text())
    pilot = json.loads((protocol_dir / "pilot_manifest.json").read_text())
    train_manifest = load_split_manifest(protocol_dir / "train_manifest.json", "train")
    val_manifest = load_split_manifest(protocol_dir / "val_manifest.json", "val")
    strict_val = load_split_manifest(protocol_dir / "strict_val_manifest.json", "val")
    job = os.environ["SLURM_JOB_ID"]
    out = root / "runs/phase2_task1_training" / job
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", job_id=job, host=os.uname().nodename, started_at=time.time(),
                  seed=args.seed, variants=list(VARIANTS), feature_run=args.feature_run,
                  metric_aggregation="macro_average_over_validation_chains",
                  checkpoint_selection_metric="macro_average_precision",
                  pooled_residue_metrics="diagnostic_only",
                  feature_report_sha256=sha256(feature_report_path), protocol_dir=args.protocol_dir,
                  train_manifest_sha256=sha256(protocol_dir / "train_manifest.json"),
                  val_manifest_sha256=sha256(protocol_dir / "val_manifest.json"),
                  strict_val_manifest_sha256=sha256(protocol_dir / "strict_val_manifest.json"),
                  test_records_read=0, test_labels_read=0, backbone_features_frozen=True,
                  labels_separate_from_feature_arrays=True,
                  config={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, dataset={},
                  normalization_fit_split="train", results=[], failures=[])
    try:
        examples, dataset_report = load_examples(root, feature_report, pilot, train_manifest, val_manifest, strict_val)
        report["dataset"] = dataset_report
        print("DATASET", json.dumps(dataset_report), flush=True)
        stats = fit_statistics(examples["train"])
        report["normalization"] = {key: {"count": value["count"], "mean": value["mean"].tolist(),
                                             "std": value["std"].tolist()} for key, value in stats.items()}
        train_positive = dataset_report["positive_residues"]["train"]
        train_valid = dataset_report["valid_residues"]["train"]
        pos_weight = (train_valid - train_positive) / max(1, train_positive)
        report["positive_class_weight"] = pos_weight
        for variant in VARIANTS:
            result = train_variant(variant, examples["train"], examples["val"], stats,
                                  args, pos_weight, device, out)
            report["results"].append(result)
            (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["finished_at"] = time.time()
        report["wall_seconds"] = report["finished_at"] - report["started_at"]
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("REPORT", out / "report.json", report["status"], flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
