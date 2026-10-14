"""Task 2 symmetric pair-head retraining under the v1.1 sampler protocol."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn

from train_phase3_task2 import (
    PROTOCOL_VERSION, encode, fit_statistics, load_pair_records, sample_pairs,
    stable_seed, sha256,
)
from train_phase2_task1 import Readout, load_examples, load_split_manifest, make_batch
from phase2_training_metrics import binary_metrics

VARIANTS = ("ES-SYM", "ES-B-SYM", "ES-C-SYM")
BASE_VARIANTS = {"ES-SYM": "ES", "ES-B-SYM": "ES-B", "ES-C-SYM": "ES-C"}
SEEDS = (20260927, 20260928, 20260929)
SEED_JOB = {20260927: 276315, 20260928: 276316, 20260929: 276314}
ABS_TOL = 1e-6
REL_TOL = 1e-5


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


class SymmetricPairHead(nn.Module):
    """g(x,y)=0.5*(f(x,y)+f(y,x)) using one ordered parameterization."""
    def __init__(self, hidden=256, pair_hidden=128):
        super().__init__()
        self.left = nn.Linear(hidden, pair_hidden, bias=False)
        self.right = nn.Linear(hidden, pair_hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def ordered(self, left, right):
        return (self.left(left) * self.right(right)).sum(-1) / np.sqrt(self.left.out_features) + self.bias

    def forward(self, left, right):
        return 0.5 * (self.ordered(left, right) + self.ordered(right, left))


def head_formula():
    return "g_theta(x,y)=0.5*(f_theta(x,y)+f_theta(y,x)); f_theta=sum(left(x)*right(y))/sqrt(pair_hidden)+bias"


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stream_digest(record_key, pairs, labels):
    h = hashlib.sha256()
    key = f"{record_key[0]}:{int(record_key[1])}".encode()
    h.update(len(key).to_bytes(4, "little")); h.update(key)
    p = np.ascontiguousarray(pairs, dtype=np.int64)
    y = np.ascontiguousarray(labels, dtype=np.float32)
    h.update(p.shape[0].to_bytes(8, "little")); h.update(p.tobytes()); h.update(y.tobytes())
    return h


def generate_stream(records, seed, epoch, ratio=3):
    h = hashlib.sha256(); total = pos = neg = 0; rows = []
    for record in records:
        pairs, labels = sample_pairs(record, seed, epoch, ratio)
        if pairs.dtype != np.int64 or labels.dtype != np.float32:
            raise ValueError("Unexpected sampler dtype")
        if len(pairs) != len(labels):
            raise ValueError("Pair/label length mismatch")
        if not np.array_equal(labels[:len(record["positives"])], np.ones(len(record["positives"]), np.float32)):
            raise ValueError("Sampler positive prefix changed")
        rh = stream_digest(record["key"], pairs, labels)
        h.update(rh.digest())
        total += len(labels); pos += int(labels.sum()); neg += int(len(labels) - labels.sum())
        rows.append((pairs, labels))
    return rows, dict(epoch=epoch, complexes=len(records), pairs=total, positives=pos, negatives=neg,
                      sha256=h.hexdigest(), ratio=ratio)


def fixed_validation_source(root: Path, seed: int):
    job = SEED_JOB[seed]
    return root / f"runs/phase3_task2_training/{job}_{seed}/ES_val_predictions.npz"


def load_fixed_validation(root: Path, seed: int, val_rows):
    source = fixed_validation_source(root, seed)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as z:
        required = {"labels", "pairs", "offsets", "lmdb_indices"}
        if not required.issubset(set(z.files)):
            raise ValueError(f"Missing validation fields in {source}: {z.files}")
        labels = np.asarray(z["labels"], dtype=np.float32)
        pairs_1 = np.asarray(z["pairs"], dtype=np.int64)
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        lmdb_indices = np.asarray(z["lmdb_indices"], dtype=np.int64)
    if pairs_1.ndim != 2 or pairs_1.shape[1] != 2 or len(labels) != len(pairs_1):
        raise ValueError("Invalid validation pair/label shapes")
    if len(offsets) != len(val_rows) + 1 or len(lmdb_indices) != len(val_rows):
        raise ValueError("Validation offsets/indices do not match validation rows")
    expected_indices = np.asarray([int(r["key"][1]) for r in val_rows], dtype=np.int64)
    if not np.array_equal(lmdb_indices, expected_indices):
        raise ValueError("Validation lmdb order mismatch")
    if np.any(pairs_1 < 1):
        raise ValueError("Validation pairs are not 1-based")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Nonbinary validation labels")
    pairs = pairs_1 - 1
    rows = []
    for i, r in enumerate(val_rows):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        p, y = pairs[lo:hi], labels[lo:hi]
        if len(p) == 0 or int(y.sum()) == 0:
            raise ValueError(f"Invalid validation row {r['key']}")
        rows.append((p, y))
    return rows, dict(source=str(source.resolve()), source_sha256=sha256(source), pairs=int(len(pairs)),
                      positives=int(labels.sum()), negatives=int(len(labels)-labels.sum()),
                      offsets_sha256=array_sha256(offsets), lmdb_indices_sha256=array_sha256(lmdb_indices),
                      pairs_sha256=array_sha256(pairs_1), labels_sha256=array_sha256(labels))


def assert_validation_rows(rows, val_records):
    if len(rows) != len(val_records):
        raise ValueError("Validation row count mismatch")
    for (pairs, labels), r in zip(rows, val_records):
        if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= r["left_length"]):
            raise ValueError("Validation left pair outside residue bounds")
        if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= r["right_length"]):
            raise ValueError("Validation right pair outside residue bounds")
        li = np.asarray(r["left_valid"], dtype=np.int64)
        ri = np.asarray(r["right_valid"], dtype=np.int64)
        if not np.isin(pairs[:, 0], li).all() or not np.isin(pairs[:, 1], ri).all():
            raise ValueError("Validation pair outside common mask")


@torch.no_grad()
def evaluate_fixed(model, pair_head, records, validation_rows, stats, device, export_path=None):
    model.eval(); pair_head.eval()
    labels_all=[]; score_ab_all=[]; score_ba_all=[]; pairs_all=[]; offsets=[0]; keys=[]
    complex_rows=[]
    for record, (pairs, labels) in zip(records, validation_rows):
        batch = make_batch([record["left"], record["right"]], stats, device)
        h = encode(model, model.variant, batch)
        n0 = len(record["left"]["labels"]); n1 = len(record["right"]["labels"])
        h0, h1 = h[:n0], h[n0:n0+n1]
        ix = torch.from_numpy(pairs[:,0]).to(device); jx = torch.from_numpy(pairs[:,1]).to(device)
        ab = pair_head(h0[ix], h1[jx])
        ba = pair_head(h1[jx], h0[ix])
        a = ab.cpu().numpy(); b = ba.cpu().numpy(); y = labels.astype(np.float32, copy=False)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("NaN/Inf score")
        delta = np.abs(a-b)
        tol = ABS_TOL + REL_TOL*np.maximum(np.abs(a), np.abs(b))
        m_ab = binary_metrics(y, a); m_ba = binary_metrics(y, b)
        complex_rows.append(dict(split="val", lmdb_index=int(record["key"][1]), key=str(record["key"]),
                                 pairs=int(len(y)), positives=int(y.sum()), negatives=int(len(y)-y.sum()),
                                 ap_canonical=float(m_ab["average_precision"]), auroc_canonical=float(m_ab["auroc"]),
                                 ap_swapped=float(m_ba["average_precision"]), auroc_swapped=float(m_ba["auroc"]),
                                 abs_delta_mean=float(delta.mean()), abs_delta_median=float(np.median(delta)),
                                 abs_delta_max=float(delta.max()), tolerance_pass=int((delta<=tol).sum()),
                                 tolerance_total=int(len(delta))))
        labels_all.append(y); score_ab_all.append(a); score_ba_all.append(b); pairs_all.append(pairs+1); offsets.append(offsets[-1]+len(y)); keys.append(record["key"])
    labels = np.concatenate(labels_all); ab=np.concatenate(score_ab_all); ba=np.concatenate(score_ba_all); pairs=np.concatenate(pairs_all)
    abs_delta=np.abs(ab-ba); rel_delta=abs_delta/np.maximum(np.maximum(np.abs(ab),np.abs(ba)), 1e-12)
    tol=ABS_TOL+REL_TOL*np.maximum(np.abs(ab),np.abs(ba)); passed=abs_delta<=tol
    def macro(field):
        vals=[r[field] for r in complex_rows if np.isfinite(r[field])]
        return float(np.mean(vals)) if vals else float("nan")
    result=dict(split="val", complexes=len(complex_rows), pairs=int(len(labels)), positives=int(labels.sum()), negatives=int(len(labels)-labels.sum()),
                average_precision_canonical=macro("ap_canonical"), auroc_canonical=macro("auroc_canonical"),
                average_precision_swapped=macro("ap_swapped"), auroc_swapped=macro("auroc_swapped"),
                abs_delta_mean=float(abs_delta.mean()), abs_delta_median=float(np.median(abs_delta)), abs_delta_max=float(abs_delta.max()),
                abs_delta_q50=float(np.quantile(abs_delta,.50)), abs_delta_q95=float(np.quantile(abs_delta,.95)), abs_delta_q99=float(np.quantile(abs_delta,.99)),
                relative_delta_mean=float(rel_delta.mean()), relative_delta_median=float(np.median(rel_delta)), relative_delta_max=float(rel_delta.max()),
                relative_delta_q50=float(np.quantile(rel_delta,.50)), relative_delta_q95=float(np.quantile(rel_delta,.95)), relative_delta_q99=float(np.quantile(rel_delta,.99)),
                tolerance_abs=ABS_TOL, tolerance_rel=REL_TOL, tolerance_pass=int(passed.sum()), tolerance_total=int(len(passed)),
                tolerance_fraction=float(passed.mean()), symmetry_pass=bool(passed.all()), metrics_by_complex=complex_rows)
    if export_path is not None:
        np.savez_compressed(export_path, score_ab=ab, score_ba_transposed=ba, abs_delta=abs_delta,
                            relative_delta=rel_delta, labels=labels, pairs=pairs, offsets=np.asarray(offsets,dtype=np.int64),
                            lmdb_indices=np.asarray([int(k[1]) for k in keys],dtype=np.int64))
        result["prediction_path"]=str(export_path); result["prediction_sha256"]=sha256(export_path)
    return result


def train_variant(name, records_train, records_val, val_rows, stats, cfg, seed, device, output, pos_weight, expected_stream):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    base = BASE_VARIANTS[name]
    model = Readout(base, cfg["hidden_dim"]).to(device)
    pair_head = SymmetricPairHead(cfg["hidden_dim"]).to(device)
    params=list(model.parameters())+list(pair_head.parameters())
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    best=-1; best_state=None; best_epoch=0; stale=0; history=[]; stream_checks=[]
    for epoch in range(1,cfg["epochs"]+1):
        stream, meta = generate_stream(records_train,seed,epoch,3)
        if meta["sha256"] != expected_stream[str(epoch)]["sha256"] or meta["pairs"] != expected_stream[str(epoch)]["pairs"]:
            raise ValueError(f"Sampler stream mismatch at epoch {epoch} for {name}")
        stream_checks.append(meta)
        model.train(); pair_head.train(); order=list(range(len(records_train))); random.shuffle(order); losses=[]
        for start in range(0,len(order),cfg["batch_size"]):
            selected=[records_train[i] for i in order[start:start+cfg["batch_size"]]]
            examples=sum(([r["left"],r["right"]] for r in selected),[])
            batch=make_batch(examples,stats,device); h=encode(model,base,batch); logits_all=[]; labels_all=[]; offset=0
            for record,(pairs,y) in zip(selected,[stream[i] for i in order[start:start+cfg["batch_size"]]]):
                n0=len(record["left"]["labels"]); n1=len(record["right"]["labels"]); h0=h[offset:offset+n0]; h1=h[offset+n0:offset+n1+n0]; offset += n0+n1
                ix=torch.from_numpy(pairs[:,0]).to(device); jx=torch.from_numpy(pairs[:,1]).to(device)
                logits_all.append(pair_head(h0[ix],h1[jx])); labels_all.append(torch.from_numpy(y).to(device))
            loss=criterion(torch.cat(logits_all),torch.cat(labels_all)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); losses.append(float(loss.detach()))
        val=evaluate_fixed(model,pair_head,records_val,val_rows,stats,device)
        row=dict(epoch=epoch,train_bce=float(np.mean(losses)),validation={k:v for k,v in val.items() if k!="metrics_by_complex"},sampler=meta); history.append(row); save_json(output/(name.replace("-","_")+"_history.json"),history)
        print("EPOCH",name,seed,json.dumps(row),flush=True)
        score=val["average_precision_canonical"]
        if score > best + cfg["min_delta"]:
            best=score; best_epoch=epoch; stale=0; best_state={"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"pair_head":{k:v.detach().cpu().clone() for k,v in pair_head.state_dict().items()}}
        else: stale+=1
        if stale >= cfg["patience"]: break
    if best_state is None: raise ValueError("No symmetric checkpoint")
    model.load_state_dict(best_state["model"]); pair_head.load_state_dict(best_state["pair_head"])
    checkpoint=output/(name.replace("-","_")+".pt"); torch.save(dict(variant=name,base_variant=base,seed=seed,epoch=best_epoch,model=best_state["model"],pair_head=best_state["pair_head"],head_formula=head_formula()),checkpoint)
    metrics=evaluate_fixed(model,pair_head,records_val,val_rows,stats,device,output/(name.replace("-","_")+"_predictions.npz"))
    save_json(output/(name.replace("-","_")+"_metrics_by_complex.json"),metrics["metrics_by_complex"])
    return dict(variant=name,base_variant=base,seed=seed,best_epoch=best_epoch,epochs_run=len(history),validation={k:v for k,v in metrics.items() if k!="metrics_by_complex"},checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),history_path=str(output/(name.replace("-","_")+"_history.json")),sampler_stream_checks=stream_checks,trainable_parameters=sum(p.numel() for p in params))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(".")); ap.add_argument("--seed-index",type=int,required=True); ap.add_argument("--config",type=Path,default=Path("configs/experiments/phase3_20260927.json")); args=ap.parse_args(); root=args.root.resolve()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(): raise RuntimeError("GPU Slurm required")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_num_threads(4); device=torch.device("cuda")
    cfg=json.loads((root/args.config).read_text()); seed=cfg["seeds"][args.seed_index]; protocol=root/cfg["protocol_dir"]; pilot=json.loads((protocol/"pilot_manifest.json").read_text()); splits={s:load_split_manifest(protocol/(s+"_manifest.json"),s) for s in ("train","val")}; strict=load_split_manifest(protocol/"strict_val_manifest.json","val"); features=json.loads((root/"runs/phase3_features/expanded_v1/report.json").read_text())
    job=os.environ["SLURM_JOB_ID"]; out=root/"runs/phase3_task2_sym_training"/f"{job}_{seed}"; out.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",task_id="SRPPI-T2-SYM-001",task_version="v1.1",job_id=job,array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),seed=seed,host=platform.node(),started_at=time.time(),test_records_read=0,test_labels_read=0,protocol_version=PROTOCOL_VERSION,head_formula=head_formula(),validation_protocol="historical task2 validation pair stream; dynamic training sampler revision",source_sha256=sha256(Path(__file__)),ordered_source_sha256=sha256(root/"scripts/train_phase3_task2.py"),config_sha256=sha256(root/args.config),protocol_sha256=sha256(protocol/"report.json"),train_manifest_sha256=sha256(protocol/"train_manifest.json"),val_manifest_sha256=sha256(protocol/"val_manifest.json"),strict_val_manifest_sha256=sha256(protocol/"strict_val_manifest.json"),feature_report_sha256=sha256(root/"runs/phase3_features/expanded_v1/report.json"),failures=[])
    save_json(out/"report.json",report)
    try:
        examples,dataset=load_examples(root,features,pilot,splits["train"],splits["val"],strict); stats=fit_statistics(examples["train"]); train_manifest=[r for r in pilot if r["split"]=="train"]; val_manifest=[r for r in pilot if r["split"]=="val"]
        train_rows=load_pair_records(root,train_manifest,examples); val_rows=load_pair_records(root,val_manifest,examples); validation_rows,val_meta=load_fixed_validation(root,seed,val_rows); assert_validation_rows(validation_rows,val_rows)
        pos_weight=3.0; epochs=int(cfg["training"]["epochs"]); expected={}
        for epoch in range(1,epochs+1):
            _,meta=generate_stream(train_rows,seed,epoch,3); expected[str(epoch)]=meta
        sampler_meta=dict(seed=seed,ratio=3,epoch_range=[1,epochs],record_order_sha256=array_sha256(np.asarray([int(r["key"][1]) for r in train_rows],dtype=np.int64)),epochs=expected,source_sha256=sha256(root/"scripts/train_phase3_task2.py"))
        save_json(out/"sampler_stream_manifest.json",sampler_meta)
        report.update(dataset=dataset,validation_source=val_meta,sampler_stream_manifest=str(out/"sampler_stream_manifest.json"),normalization_fit_split="train",normalization={k:{q:(v.tolist() if isinstance(v,np.ndarray) else v) for q,v in d.items()} for k,d in stats.items()},results=[]); save_json(out/"report.json",report)
        for variant in VARIANTS:
            result=train_variant(variant,train_rows,val_rows,validation_rows,stats,cfg["training"],seed,device,out,pos_weight,expected); report["results"].append(result); save_json(out/"report.json",report)
        report["status"]="passed"
    except Exception:
        report["status"]="failed"; report["failures"].append(traceback.format_exc()); print(report["failures"][-1],flush=True)
    finally:
        report.update(finished_at=time.time(),wall_seconds=time.time()-report["started_at"],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_gpu_bytes=torch.cuda.max_memory_allocated()); save_json(out/"report.json",report); print("REPORT",out/"report.json",report["status"],flush=True)
    if report["status"]!="passed": raise SystemExit(1)

if __name__=="__main__": main()
