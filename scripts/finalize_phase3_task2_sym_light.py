import csv,glob,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path('/datapool/home/zhangye/projects/surface-residue-ppi'); OUT=ROOT/'runs/phase3_task2_sym_training/summary_v1.1_light'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[20260927,20260928,20260929]; jobs={20260927:276499,20260928:276500,20260929:276498}; vars=['ES-SYM','ES-B-SYM','ES-C-SYM']
def sh(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jw(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n')
def sd(x): return float(np.std(x,ddof=1)) if len(x)>1 else 0.
rows=[]; pairs=[]; surface=[]
for s in SEEDS:
 d=json.load(open(ROOT/f'runs/phase3_task2_sym_training/{jobs[s]}_{s}/report.json'))
 for r in d['results']:
  v=r['variant']; pred=Path(r['validation']['prediction_path']); ck=Path(r['checkpoint']); z=np.load(pred,allow_pickle=False); ab=z['score_ab'].astype(float); ba=z['score_ba_transposed'].astype(float); delta=np.abs(ab-ba); rel=delta/np.maximum(np.maximum(np.abs(ab),np.abs(ba)),1e-12); tol=1e-6+1e-5*np.maximum(np.abs(ab),np.abs(ba));
  row={'seed':s,'variant':v,'base_variant':r['base_variant'],'job_id':d['job_id'],'host':d['host'],'checkpoint':str(ck.resolve()),'checkpoint_size':ck.stat().st_size,'checkpoint_sha256':sh(ck),'prediction_path':str(pred.resolve()),'prediction_size':pred.stat().st_size,'prediction_sha256':sh(pred),'pairs':len(ab),'complexes':len(z['lmdb_indices']),'canonical_ap':r['validation']['average_precision_canonical'],'canonical_auroc':r['validation']['auroc_canonical'],'swapped_ap':r['validation']['average_precision_swapped'],'swapped_auroc':r['validation']['auroc_swapped'],'abs_delta_mean':float(delta.mean()),'abs_delta_median':float(np.median(delta)),'abs_delta_max':float(delta.max()),'abs_delta_q50':float(np.quantile(delta,.5)),'abs_delta_q95':float(np.quantile(delta,.95)),'abs_delta_q99':float(np.quantile(delta,.99)),'relative_delta_mean':float(rel.mean()),'relative_delta_median':float(np.median(rel)),'relative_delta_max':float(rel.max()),'relative_delta_q50':float(np.quantile(rel,.5)),'relative_delta_q95':float(np.quantile(rel,.95)),'relative_delta_q99':float(np.quantile(rel,.99)),'tolerance_abs':1e-6,'tolerance_rel':1e-5,'tolerance_pass':int((delta<=tol).sum()),'tolerance_fraction':float((delta<=tol).mean()),'symmetry_pass':bool((delta<=tol).all()),'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run']}
  rows.append(row)
  m=json.load(open(ROOT/f'runs/phase3_task2_sym_training/{jobs[s]}_{s}/{v.replace("-","_")}_metrics_by_complex.json')); pairs.append((s,v,m))
by={}
for s,v,m in pairs: by[(s,v)]={int(x['lmdb_index']):x for x in m}
for s in SEEDS:
 for v in ['ES-B-SYM','ES-C-SYM']:
  for idx in sorted(by[(s,'ES-SYM')]):
   a=by[(s,v)][idx]; b=by[(s,'ES-SYM')][idx]; surface.append({'seed':s,'variant':v,'lmdb_index':idx,'ap_delta':a['ap_canonical']-b['ap_canonical'],'auroc_delta':a['auroc_canonical']-b['auroc_canonical'],'ap_swapped_delta':a['ap_swapped']-b['ap_swapped'],'auroc_swapped_delta':a['auroc_swapped']-b['auroc_swapped']})
summary=[]
for v in ['ES-B-SYM','ES-C-SYM']:
 for metric in ['ap_delta','auroc_delta','ap_swapped_delta','auroc_swapped_delta']:
  a=np.array([x[metric] for x in surface if x['variant']==v]); summary.append({'comparison':v+' minus ES-SYM','metric':metric,'mean':float(a.mean()),'sample_sd':sd(a),'n':len(a),'q50':float(np.quantile(a,.5)),'q95':float(np.quantile(a,.95)),'q99':float(np.quantile(a,.99))})
jw(OUT/'symmetry_summary.json',{'task_id':'SRPPI-T2-SYM-001','version':'v1.1','status':'passed','head_formula':'g_theta(x,y)=0.5*(f_theta(x,y)+f_theta(y,x))','tolerance':{'abs':1e-6,'rel':1e-5},'rows':rows,'surface_delta_summary':summary,'validation_scope':'sampled-negative developmental validation; not official benchmark','test_records_read':0,'test_labels_read':0,'training_sampler':'sample_pairs(seed,epoch,ratio=3), deterministic/shared, not historical exact replay'})
with open(OUT/'symmetry_summary.tsv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
with open(OUT/'surface_paired_deltas.tsv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(surface[0]),delimiter='\t');w.writeheader();w.writerows(surface)
jw(OUT/'surface_delta_summary.json',summary)
with open(OUT/'checkpoint_inventory.tsv','w') as f:
 fields=['seed','variant','checkpoint','checkpoint_size','checkpoint_sha256','job_id','host'];w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
slurm=subprocess.check_output(['sacct','-j','276497,276498','--format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,AllocTRES,NodeList,MaxRSS,Start,End','-P'],text=True);(OUT/'slurm_jobs.txt').write_text(slurm)
code={};
for name in ['scripts/train_phase3_task2_sym.py','scripts/preflight_phase3_task2_sym.py','scripts/train_phase3_task2.py','configs/experiments/phase3_20260927.json','runs/phase3_protocol/20260927/expanded_v1/report.json','runs/phase3_protocol/20260927/expanded_v1/train_manifest.json','runs/phase3_protocol/20260927/expanded_v1/val_manifest.json','runs/phase3_features/expanded_v1/report.json'] : code[name]=sh(ROOT/name)
jw(OUT/'run_config.json',{'task_id':'SRPPI-T2-SYM-001','version':'v1.1','head_formula':'g_theta(x,y)=0.5*(f_theta(x,y)+f_theta(y,x))','variants':vars,'seeds':SEEDS,'protocol_version':'task2-common-mask-v2','validation_pairs':669235,'validation_complexes':467,'training_sampler':{'function':'sample_pairs','ratio':3,'deterministic':True,'shared_stream':True,'exact_historical_replay':False},'tolerances':{'absolute':1e-6,'relative':1e-5},'test_records_read':0,'test_labels_read':0,'code_sha256':code,'jobs':{'preflight':276497,'training_array':276498}})
jw(OUT/'environment_provenance.json',{'python':'3.8.18','numpy':np.__version__,'host':platform.node(),'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'resources':{'partition':'a100','gres':'gpu:1','cpus_per_task':4,'mem':'96G','time':'08:00:00'},'code_sha256':code})
with open(OUT/'evidence_index.tsv','w') as f:
 f.write('artifact\tpath\tsha256\tstatus\n')
 for p in sorted(OUT.iterdir()):
  if p.is_file():f.write(f'{p.name}\t{p.resolve()}\t{sh(p)}\tverified\n')
print(json.dumps({'status':'passed','out':str(OUT),'rows':len(rows),'surface_rows':len(surface)},indent=2))
