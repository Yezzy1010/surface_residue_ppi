"""Aggregate the completed three-seed task-three sequence baseline."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from phase2_training_metrics import binary_metrics


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def recompute(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        y, s = z["labels"], z["scores"]
    return dict(binary_metrics(y.astype(np.int64), s), records=int(len(y)), prediction_sha256=sha(path))


def aggregate(rows: list[dict], key: str) -> dict:
    vals_ap = np.asarray([r[key]["average_precision"] for r in rows], dtype=float)
    vals_auc = np.asarray([r[key]["auroc"] for r in rows], dtype=float)
    return dict(
        average_precision_mean=float(vals_ap.mean()),
        average_precision_sample_sd=float(vals_ap.std(ddof=1)),
        auroc_mean=float(vals_auc.mean()),
        auroc_sample_sd=float(vals_auc.std(ddof=1)),
        seeds=len(rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path(".")); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); root = args.root.resolve()
    reports = sorted((root / "runs/phase3_task3_training").glob("*/report.json"))
    reports = [p for p in reports if json.loads(p.read_text()).get("status") == "passed"]
    if len(reports) != 3: raise ValueError(f"expected 3 passed reports, found {len(reports)}")
    rows=[]
    for p in reports:
        d=json.loads(p.read_text()); base=p.parent
        d["validation_recomputed"] = recompute(base / "val_predictions.npz")
        d["test_recomputed"] = recompute(base / "test_predictions.npz")
        d["strict_test_recomputed"] = recompute(base / "strict_test_predictions.npz")
        for key in ("validation", "test", "strict_test"):
            if abs(d[key]["average_precision"] - d[key + "_recomputed"]["average_precision"]) > 1e-12 or abs(d[key]["auroc"] - d[key + "_recomputed"]["auroc"]) > 1e-12:
                raise ValueError(f"saved metric mismatch in {p} {key}")
        rows.append(d)
    out = args.out; out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed", "model": "E", "relation_head": "symmetric_sum_absdiff_product_mlp",
        "protocol": "runs/phase3_task3_protocol/20260928/subset_v1",
        "reports": [str(p) for p in reports], "seeds": [int(r["seed"]) for r in rows],
        "validation": aggregate(rows, "validation"), "test": aggregate(rows, "test"),
        "strict_test": aggregate(rows, "strict_test"),
        "per_seed": [{"seed": int(r["seed"]), "validation": r["validation"], "test": r["test"], "strict_test": r["strict_test"], "best_epoch": r["best_epoch"], "test_records_read": r["test_records_read"], "strict_test_records": r["strict_test_records"]} for r in sorted(rows, key=lambda x:x["seed"])],
        "test_labels_read": int(rows[0]["test_labels_read"]),
        "strict_test_definition": "both proteins absent from the official selected training subset",
        "interpretation_limit": "sequence-only D-SCRIPT baseline; no structure or surface features were available for these ENSP identifiers in this first run; D-SCRIPT negatives are operational labels",
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
