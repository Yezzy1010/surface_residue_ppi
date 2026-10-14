"""Fixed task3 2x2 contrasts; prediction-only hpc job, no model or test data."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from summarize_phase3_task3_structures import weighted_metrics
from surface_residue_ppi.paired_statistics import component_ids

SEEDS = (20260927, 20260928, 20260929)
CELLS = ("ES-MEAN", "ES-LOCAL", "ES-B-MEAN", "ES-B-LOCAL")
RUNS = ("276590_20260927", "276592_20260928", "276589_20260929")
CONTRASTS = {"readout_ES": [-1, 1, 0, 0], "readout_ES_B": [0, 0, -1, 1],
             "surface_MEAN": [-1, 0, 1, 0], "surface_LOCAL": [0, -1, 0, 1],
             "difference_in_differences": [1, -1, -1, 1]}
METRICS = ("average_precision", "auroc")
ATOL, RTOL = 1e-6, 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tsv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def delta_stats(a, b):
    delta = np.abs(a-b)
    relative = delta / np.maximum(np.abs(b), 1e-12)
    stat = lambda x: dict(mean=float(x.mean()), median=float(np.median(x)), max=float(x.max()),
                          q50=float(np.quantile(x, .50)), q95=float(np.quantile(x, .95)), q99=float(np.quantile(x, .99)))
    threshold = ATOL + RTOL*np.abs(b)
    return dict(absolute=stat(delta), relative=stat(relative),
                relative_definition="abs(a-b)/max(abs(b),1e-12); reporting only",
                pass_=bool((delta <= threshold).all()), failures=int((delta > threshold).sum()),
                max_tolerance_ratio=float((delta/threshold).max()), atol=ATOL, rtol=RTOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True, type=Path)
    args = ap.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Slurm CPU job required")
    root = Path.cwd(); started = time.time()
    out = root / "runs/phase3_task3_local_summary" / os.environ["SLURM_JOB_ID"]
    out.mkdir(parents=True, exist_ok=False)
    protocol = root / "runs/phase3_task3_protocol/20260928/structure_v2"
    require(digest(protocol/"report.json") == "ffaadc6e790accf4b119676edc8bb9a22803caae4dbaad59fe5c4fbd6c136593", "Protocol changed")
    pr = json.loads((protocol/"report.json").read_text())
    for name, value in pr["artifact_sha256"].items():
        require(digest(protocol/name) == value, "Manifest changed")
    rows = json.loads((protocol/"val.json").read_text()); y = np.asarray([r["label"] for r in rows])
    components = component_ids(rows); ncomp = int(components.max())+1
    require(ncomp == 109 and len(rows) == 194 and int(y.sum()) == 124, "Frozen evaluation size changed")
    detail_report = json.loads((args.details/"report.json").read_text())
    require(detail_report["status"] == "passed" and len(detail_report["results"]) == 12, "Missing representation evidence")
    require(not detail_report["test_records_read"] and not detail_report["test_labels_read"], "Test access")
    details = {(r["seed"], r["cell"]):r for r in detail_report["results"]}
    scores = np.empty((3,4,len(y)), np.float64)
    metrics = np.empty((3,4,2), np.float64)
    inventory, metrics_rows, pair_rows, symmetries, source = [], [], [], [], []
    for si, (seed, run) in enumerate(zip(SEEDS, RUNS)):
        directory = root/"runs/phase3_task3_local"/run
        rep = json.loads((directory/"report.json").read_text())
        require(rep["seed"] == seed and rep["status"] == "passed", "Invalid training report")
        require(rep["protocol_sha256"] == digest(protocol/"report.json") and rep["training"] == pr["training"], "Protocol/config changed")
        require(rep["source_sha256"] == "ddee972007cffd9525b54ce62ce193bb15b28c01d25a74cd00d7cd6d99d5987b", "Training source changed")
        require(not rep["test_records_read"] and not rep["test_labels_read"], "Test access")
        for name in ("normalization", "feature_hashes"):
            require(digest(directory/(name+".json")) == detail_report[name+"_sha256"], "Input hashes differ")
        source.append(dict(path=str(directory/"report.json"), sha256=digest(directory/"report.json")))
        for ci, cell in enumerate(CELLS):
            item = next(r for r in rep["results"] if r["cell"] == cell)
            cp, pp = Path(item["checkpoint"]), Path(item["predictions"])
            require(digest(cp) == item["checkpoint_sha256"] and digest(pp) == item["predictions_sha256"], "Artifact hash mismatch")
            with np.load(pp, allow_pickle=False) as archive:
                a = {k:archive[k] for k in archive.files}
            require(np.array_equal(a["labels"], y), "Labels differ")
            for key in ("a", "b"):
                require(a[key].tolist() == [r[key] for r in rows], "Pair IDs differ")
            require(a["source_rows"].tolist() == [r["source_row"] for r in rows], "Row order differs")
            score = a["logits"]; require(np.isfinite(score).all(), "Nonfinite scores")
            m = weighted_metrics(y, score, np.ones(len(y)))
            reference = [average_precision_score(y,score), roc_auc_score(y,score)]
            require(np.allclose(m,reference,atol=1e-12,rtol=0), "Independent metrics disagree")
            require(np.allclose(m,[item["validation"][k] for k in METRICS],atol=1e-12,rtol=0), "Reported metrics disagree")
            history = json.loads((directory/(cell.lower().replace("-","_")+"_history.json")).read_text())
            best, epoch, stale = -1.0, 0, 0
            for h in history:
                v = h["validation"]["average_precision"]
                if v > best+pr["training"]["min_delta"]:
                    best, epoch, stale = v, h["epoch"], 0
                else: stale += 1
            require(epoch == item["best_epoch"] and len(history) == item["epochs_run"], "Selection protocol mismatch")
            require(stale == 5 or len(history) == 30, "Unexpected stopping rule")
            de = details[seed,cell]; dp = Path(de["file"])
            require(de["strict_load"] and de["representation_pass"] and de["score_pass"] and de["original_score_pass"], "Evidence failure")
            require(digest(dp) == de["sha256"] and de["checkpoint_sha256"] == item["checkpoint_sha256"], "Evidence hash mismatch")
            with np.load(dp,allow_pickle=False) as archive:
                d = {k:archive[k] for k in archive.files}
            require(np.array_equal(d["logits"],score) and np.array_equal(d["source_rows"],a["source_rows"]), "Evidence identity mismatch")
            require(np.array_equal(d["candidate_counts"],d["valid_a"]*d["valid_b"]), "Candidate count mismatch")
            sym = dict(seed=seed,cell=cell,score=delta_stats(d["score_ab"],d["score_ba"]),
                       representation=delta_stats(d["representation_ab"],d["representation_ba"]),
                       original_replay=delta_stats(score,d["score_ab"]))
            require(sym["score"]["pass_"] and sym["representation"]["pass_"] and sym["original_replay"]["pass_"], "Tolerance failure")
            sym["canonical_metrics"] = dict(zip(METRICS,weighted_metrics(y,d["score_ab"],np.ones(len(y))).tolist()))
            sym["swapped_metrics"] = dict(zip(METRICS,weighted_metrics(y,d["score_ba"],np.ones(len(y))).tolist()))
            symmetries.append(sym)
            scores[si,ci], metrics[si,ci] = score, m
            inventory.append(dict(seed=seed,cell=cell,path=str(cp),bytes=cp.stat().st_size,sha256=digest(cp),
                                  best_epoch=epoch,epochs_run=len(history),strict_load=True,strict_load_job=detail_report["job_id"],
                                  training_job=rep["job_id"],trainable_parameters=item["trainable_parameters"],
                                  relation_parameters=de["relation_parameters"]))
            metrics_rows.append(dict(seed=seed,cell=cell,AP=m[0],AUROC=m[1],positives=124,negatives=70,
                                     best_epoch=epoch,epochs_run=len(history),prediction_path=str(pp),sha256=digest(pp)))
            for i,r in enumerate(rows):
                pair_rows.append(dict(seed=seed,cell=cell,validation_index=i,source_row=r["source_row"],a=r["a"],b=r["b"],label=r["label"],
                                      valid_a=int(d["valid_a"][i]),valid_b=int(d["valid_b"][i]),N=int(d["candidate_counts"][i]),
                                      original_logit=float(score[i]),score_ab=float(d["score_ab"][i]),score_ba=float(d["score_ba"][i]),
                                      abs_delta=float(abs(d["score_ab"][i]-d["score_ba"][i])),
                                      probability_ab=float(d["probability_ab"][i]),predicted_label_ab=int(d["predicted_label_ab"][i]),
                                      representation_mean=float(d["representation_mean"][i]),representation_std=float(d["representation_std"][i]),
                                      representation_min=float(d["representation_min"][i]),representation_max=float(d["representation_max"][i]),
                                      representation_l2=float(d["representation_l2"][i]),
                                      representation_swap_max_abs=float(np.abs(d["representation_ab"][i]-d["representation_ba"][i]).max()),
                                      component=int(components[i])))
    rng = np.random.default_rng(20260928)
    draws = np.empty((2000,ncomp),np.int64); boot = np.empty((2000,3,4,2),np.float64)
    for n in range(2000):
        draws[n] = np.bincount(rng.integers(ncomp,size=ncomp),minlength=ncomp)
        weight = draws[n][components]
        require((y*weight).sum() and ((1-y)*weight).sum(), "Undefined bootstrap draw; stop")
        for si in range(3):
            for ci in range(4):
                boot[n,si,ci] = weighted_metrics(y,scores[si,ci],weight)
    aggregate, contrasts, contrast_rows = {}, {}, []
    for mi,m in enumerate(METRICS):
        aggregate[m] = {cell:dict(per_seed=metrics[:,ci,mi].tolist(),mean=float(metrics[:,ci,mi].mean()),
                                 sample_sd=float(metrics[:,ci,mi].std(ddof=1))) for ci,cell in enumerate(CELLS)}
    for name,coef in CONTRASTS.items():
        point = np.einsum("scm,c->sm",metrics,np.array(coef))
        distribution = np.einsum("bcm,c->bm",boot.mean(axis=1),np.array(coef))
        contrasts[name] = {}
        for mi,m in enumerate(METRICS):
            lo, hi = np.quantile(distribution[:,mi],[.025,.975])
            entry = dict(per_seed=point[:,mi].tolist(),mean=float(point[:,mi].mean()),
                         sample_sd=float(point[:,mi].std(ddof=1)),low=float(lo),high=float(hi))
            contrasts[name][m] = entry
            contrast_rows.append(dict(contrast=name,metric=m,seed_20260927=entry["per_seed"][0],
                                      seed_20260928=entry["per_seed"][1],seed_20260929=entry["per_seed"][2],
                                      mean=entry["mean"],sample_sd=entry["sample_sd"],ci95_low=lo,ci95_high=hi))
    tsv(out/"checkpoint_inventory.tsv",inventory); tsv(out/"metrics_by_seed.tsv",metrics_rows)
    tsv(out/"pair_predictions.tsv",pair_rows); tsv(out/"paired_effects.tsv",contrast_rows)
    save(out/"symmetry_summary.json",symmetries)
    np.savez_compressed(out/"bootstrap.npz",component_ids=components,component_multiplicity=draws,
                        per_seed_metrics=boot,original_metrics=metrics,seeds=SEEDS,cells=CELLS,metrics=METRICS)
    candidates = json.loads((root/"runs/phase3_task3_local/276583_preflight/candidate_counts.json").read_text())
    require(len(candidates)==1306,"Candidate list incomplete"); tsv(out/"candidate_counts.tsv",candidates)
    final = dict(status="passed",seeds=SEEDS,cells=CELLS,aggregate=aggregate,contrasts=contrasts,source=source,
                 dataset=pr["splits"],details_report=str(args.details/"report.json"),details_report_sha256=digest(args.details/"report.json"),
                 bootstrap=dict(components=ncomp,replicates=2000,seed=20260928,confidence=.95,undefined_draws=0,
                                method="paired connected-component percentile bootstrap of metrics averaged over three fixed fitted seeds",
                                component_sizes=sorted(np.bincount(components).tolist(),reverse=True)),
                 scientific_limits=pr["limits"]+["Intervals condition on validation-selected checkpoints and fixed seeds; no model-selection or seed-population correction."],
                 formulas=CONTRASTS,score_source="original saved training predictions; replay only for validation and symmetry",
                 job_id=os.environ["SLURM_JOB_ID"],host=platform.node(),started_at=started,ended_at=time.time(),
                 environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
                 script_sha256=digest(__file__),test_records_read=0,test_labels_read=0,
                 artifacts={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/"report.json",final)
    print(json.dumps(dict(aggregate=aggregate,contrasts=contrasts),indent=2),flush=True)


if __name__ == "__main__":
    main()
