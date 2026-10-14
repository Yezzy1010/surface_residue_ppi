import json,hashlib,platform,os
from pathlib import Path
import torch
from train_phase3_task2 import Readout,sha256
from train_phase3_task2_sym import SymmetricPairHead
root=Path('/datapool/home/zhangye/projects/surface-residue-ppi'); jobs={20260927:276499,20260928:276500,20260929:276498}; vars={'ES-SYM':'ES','ES-B-SYM':'ES-B','ES-C-SYM':'ES-C'}; out=root/'runs/phase3_task2_sym_training/checkpoint_load_verification.json'
rows=[]
for seed,job in jobs.items():
 for v,base in vars.items():
  p=root/f'runs/phase3_task2_sym_training/{job}_{seed}/{v.replace("-","_")}.pt'; d=torch.load(p,map_location='cuda'); m=Readout(base,256).cuda(); h=SymmetricPairHead(256).cuda(); m.load_state_dict(d['model'],strict=True); h.load_state_dict(d['pair_head'],strict=True); rows.append({'seed':seed,'variant':v,'path':str(p.resolve()),'size':p.stat().st_size,'sha256':sha256(p),'strict_model':True,'strict_pair_head':True,'device':torch.cuda.get_device_name()})
out.write_text(json.dumps({'status':'passed','host':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),'torch':torch.__version__,'rows':rows},indent=2)+'\n'); print(json.dumps({'status':'passed','job_id':os.environ.get('SLURM_JOB_ID'),'rows':len(rows)}))
