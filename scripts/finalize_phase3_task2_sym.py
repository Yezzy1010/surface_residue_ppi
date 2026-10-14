from __future__ import annotations
import csv, hashlib, json, os, platform, statistics, subprocess, time
from pathlib import Path
import numpy as np
import torch
from train_phase3_task2 import Readout, sha256
from train_phase3_task2_sym import SymmetricPairHead, head_formula, SEED_JOB, SEEDS, VARIANTS, ABS_TOL, REL_TOL

ROOT=Path('/datapool/home/zhangye/projects/surface-residue-ppi')
OUT=ROOT/'runs/phase3_task2_sym_training/summary_v1.1'
OUT.mkdir(parents=True,exist_ok=False)

def jwrite(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')
def q(vals):
 a=np.asarray(vals,dtype=float); return {k:float(np.quantile(a,x)) for k,x in [('q50',.5),('q95',.95),('q99',.99)]}
def sd(vals): return float(np.std(np.asarray(vals,dtype=float),ddof=1)) if len(vals)>1 else 0.0

def report_for(seed):
 return ROOT/f'runs/phase3_task2_sym_training/{os.environ.get("JOBMAP_%d"%seed, "")}_{seed}/report.json'
# fixed job IDs are recorded by the submitting script; avoid environment dependence
jobmap=SEED_JOB
reports={s:json.loads((ROOT/f'runs/phase3_task2_sym_training/{jobmap[s]}_{s}/report.json').read_text()) for s in SEEDS}
for s,d in reports.items():
 if d.get('status')!='passed' or len(d.get('results',[]))!=3: raise RuntimeError(f'incomplete report seed {s}')

inventory=[]; rows=[]; metrics_by_seed={}
for seed in SEEDS:
 d=reports[seed]; metrics_by_seed[seed]={}
 for res in d['results']:
  v=res['variant']; base=res['base_variant']; ck=Path(res['checkpoint']); pred=Path(res['validation']['prediction_path'])
  if not ck.is_file() or not pred.is_file(): raise FileNotFoundError((ck,pred))
  # strict load of newly trained model/head
  state=torch.load(ck,map_location='cpu',weights_only=False)
  model=Readout(base,256); head=SymmetricPairHead(256)
  model.load_state_dict(state['model'],strict=True); head.load_state_dict(state['pair_head'],strict=True)
  z=np.load(pred,allow_pickle=False); req={'score_ab','score_ba_transposed','abs_delta','relative_delta','labels','pairs','offsets','lmdb_indices'}
  if not req.issubset(set(z.files)): raise RuntimeError(f'{pred} missing {req-set(z.files)}')
  if len(z['score_ab']) != 669235: raise RuntimeError(f'{pred} pair count {len(z["score_ab"])}')
  # independently recompute summary from saved predictions
  ab=z['score_ab'].astype(float); ba=z['score_ba_transposed'].astype(float); delta=np.abs(ab-ba); rel=delta/np.maximum(np.maximum(np.abs(ab),np.abs(ba)),1e-12); tol=ABS_TOL+REL_TOL*np.maximum(np.abs(ab),np.abs(ba))
  if not np.array_equal(delta.astype(np.float32),z['abs_delta']): raise RuntimeError(f'abs delta mismatch {pred}')
  if not np.isfinite(ab).all() or not np.isfinite(ba).all(): raise RuntimeError(f'nonfinite {pred}')
  row=dict(seed=seed,variant=v,base_variant=base,job_id=d['job_id'],host=d['host'],partition='a100',gpu='A100',checkpoint=str(ck.resolve()),checkpoint_size=ck.stat().st_size,checkpoint_sha256=sha256(ck),strict_load=True,prediction_path=str(pred.resolve()),prediction_size=pred.stat().st_size,prediction_sha256=sha256(pred),pairs=int(len(ab)),complexes=int(len(z['lmdb_indices'])),canonical_ap=res['validation']['average_precision_canonical'],canonical_auroc=res['validation']['auroc_canonical'],swapped_ap=res['validation']['average_precision_swapped'],swapped_auroc=res['validation']['auroc_swapped'],abs_delta_mean=float(delta.mean()),abs_delta_median=float(np.median(delta)),abs_delta_max=float(delta.max()),abs_delta_q50=float(np.quantile(delta,.5)),abs_delta_q95=float(np.quantile(delta,.95)),abs_delta_q99=float(np.quantile(delta,.99)),relative_delta_mean=float(rel.mean()),relative_delta_median=float(np.median(rel)),relative_delta_max=float(rel.max()),relative_delta_q50=float(np.quantile(rel,.5)),relative_delta_q95=float(np.quantile(rel,.95)),relative_delta_q99=float(np.quantile(rel,.99)),tolerance_abs=ABS_TOL,tolerance_rel=REL_TOL,tolerance_pass=int((delta<=tol).sum()),tolerance_fraction=float((delta<=tol).mean()),best_epoch=res['best_epoch'],epochs_run=res['epochs_run'],trainable_parameters=res['trainable_parameters'])
  rows.append(row); inventory.append(row); metrics_by_seed[seed][v]=json.loads((OUT.parent/f'{d["job_id"]}_{seed}/{v.replace("-","_")}_metrics_by_complex.json').read_text())

# paired complex-level surface differences
surface=[]
for seed in SEEDS:
 by={v:{int(x['lmdb_index']):x for x in metrics_by_seed[seed][v]} for v in VARIANTS}
 for v in ('ES-B-SYM','ES-C-SYM'):
  for idx in sorted(by['ES-SYM']):
   a=by[v][idx]; b=by['ES-SYM'][idx]
   surface.append(dict(seed=seed,variant=v,lmdb_index=idx,ap_delta=float(a['ap_canonical']-b['ap_canonical']),auroc_delta=float(a['auroc_canonical']-b['auroc_canonical']),ap_swapped_delta=float(a['ap_swapped']-b['ap_swapped']),auroc_swapped_delta=float(a['auroc_swapped']-b['auroc_swapped'])))
summary=[]
for seed in SEEDS:
 for v in VARIANTS:
  summary.append(next(r for r in rows if r['seed']==seed and r['variant']==v))
for v in ('ES-B-SYM','ES-C-SYM'):
 for metric in ('ap_delta','auroc_delta','ap_swapped_delta','auroc_swapped_delta'):
  vals=[x[metric] for x in surface if x['variant']==v]; summary.append(dict(seed='across_seeds',variant=f'{v}-{"ES-SYM"}',metric=metric,mean=float(np.mean(vals)),sample_sd=sd(vals),q=q(vals),n=len(vals)))

# provenance
jwrite(OUT/'symmetry_summary.json',dict(task_id='SRPPI-T2-SYM-001',version='v1.1',status='passed',head_formula=head_formula(),tolerance=dict(abs=ABS_TOL,rel=REL_TOL),rows=rows,surface_delta_summary=summary,validation_scope='sampled-negative developmental validation; not official benchmark',test_records_read=0,test_labels_read=0,train_pair_protocol='dynamic sample_pairs(seed,epoch,ratio=3), deterministic and shared across variants; not historical exact replay'))
with (OUT/'symmetry_summary.tsv').open('w',newline='') as f:
 fields=list(rows[0].keys()); w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows(rows)
with (OUT/'surface_paired_deltas.tsv').open('w',newline='') as f:
 fields=list(surface[0].keys()); w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows(surface)
jwrite(OUT/'surface_delta_summary.json',[x for x in summary if x.get('seed')=='across_seeds'])
with (OUT/'checkpoint_inventory.tsv').open('w',newline='') as f:
 fields=['seed','variant','checkpoint','checkpoint_size','checkpoint_sha256','strict_load','job_id','host','partition','gpu']; w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in inventory])
# job provenance from sacct
sacct=subprocess.check_output(['sacct','-j','276497,276498','--format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,AllocTRES,NodeList,MaxRSS,Start,End','-P'],text=True)
(OUT/'slurm_jobs.txt').write_text(sacct)
versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,host=platform.node(),generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jobs={'preflight':276497,'training_array':276498},resources={'partition':'a100','gres':'gpu:1','cpus_per_task':4,'mem':'96G','time':'08:00:00'},code_sha256={'train_phase3_task2_sym.py':sha256(ROOT/'scripts/train_phase3_task2_sym.py'),'preflight_phase3_task2_sym.py':sha256(ROOT/'scripts/preflight_phase3_task2_sym.py'),'ordered_train_phase3_task2.py':sha256(ROOT/'scripts/train_phase3_task2.py'),'config':sha256(ROOT/'configs/experiments/phase3_20260927.json'),'protocol_report':sha256(ROOT/'runs/phase3_protocol/20260927/expanded_v1/report.json'),'val_manifest':sha256(ROOT/'runs/phase3_protocol/20260927/expanded_v1/val_manifest.json'),'train_manifest':sha256(ROOT/'runs/phase3_protocol/20260927/expanded_v1/train_manifest.json'),'feature_report':sha256(ROOT/'runs/phase3_features/expanded_v1/report.json')})
jwrite(OUT/'environment_provenance.json',versions)
jwrite(OUT/'run_config.json',dict(task_id='SRPPI-T2-SYM-001',version='v1.1',head_formula=head_formula(),variants=list(VARIANTS),seeds=list(SEEDS),protocol_version='task2-common-mask-v2',validation_pairs=669235,validation_complexes=467,training_sampler=dict(function='sample_pairs',ratio=3,negative_universe='Cartesian product of common valid masks',deterministic=True,shared_stream=True,exact_historical_replay=False),tolerances={'absolute':ABS_TOL,'relative':REL_TOL},test_records_read=0,test_labels_read=0,code_sha256=versions['code_sha256'],jobs=versions['jobs'],resources=versions['resources']))
# evidence index
with (OUT/'evidence_index.tsv').open('w') as f:
 f.write('artifact\tpath\tsha256\tstatus\n')
 for p in [OUT/'run_config.json',OUT/'environment_provenance.json',OUT/'symmetry_summary.tsv',OUT/'symmetry_summary.json',OUT/'surface_paired_deltas.tsv',OUT/'surface_delta_summary.json',OUT/'checkpoint_inventory.tsv',OUT/'slurm_jobs.txt']:
  f.write(f'{p.name}\t{p.resolve()}\t{sha256(p)}\tverified\n')
print(json.dumps({'status':'passed','out':str(OUT),'rows':len(rows),'surface_rows':len(surface),'torch':torch.__version__,'numpy':np.__version__},indent=2))
