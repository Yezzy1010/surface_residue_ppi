"""Collect the authorized arrays once they pass; never submit or rerun training."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from continue_phase3 import command, wait_array, REMOTE, REMOTE_ROOT


def main():
    if Path('.').resolve()!=Path('/home/zy/projects/Surface-residue'):raise ValueError('Use authorized workstation')
    folder=Path('runs/phase3_orchestration/20260928/collect_v2');folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=folder/'state.json'
    state=json.loads(path.read_text()) if path.exists() else {'created_at':time.time()}
    def record(**updates):
        state.update(updates,updated_at=time.time(),pid=os.getpid())
        tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
        print(json.dumps(updates),flush=True)
    work=[
        (2,276314,'runs/phase3_task2_training','scripts/summarize_phase3_task2.py',
         'runs/phase3_task2_summary/20260928/common_mask_v2/report.json'),
        (3,276327,'runs/phase3_task3_structure_training','scripts/summarize_phase3_task3_structures.py',
         'runs/phase3_task3_summary/20260928/structure_v2/report.json')]
    try:
        for task,job,relative,script,summary in work:
            if Path(summary).exists():
                if json.loads(Path(summary).read_text())['status']!='passed':raise ValueError('Existing summary did not pass')
            else:
                record(status='waiting',task=task,array=job)
                wait_array(job,3,record)
                record(status='downloading',task=task)
                Path(relative).mkdir(parents=True,exist_ok=True)
                command(['rsync','-rt','--include=*/','--include=*.json','--include=*predictions.npz','--exclude=*',
                         REMOTE+':'+REMOTE_ROOT+'/'+relative+'/',relative+'/'],timeout=1800)
                record(status='summarizing',task=task)
                output=command([sys.executable,script,'--out',summary],timeout=1800)
                (folder/('task'+str(task)+'_summary.log')).write_text(output)
            command(['rsync','-rt','--relative',summary,REMOTE+':'+REMOTE_ROOT+'/'],timeout=600)
            record(**{'task'+str(task)+'_summary':summary},status='collected',task=task)
        record(status='both_main_comparisons_collected',test_records_read_by_collector=0)
    except Exception:
        record(status='needs_review',error=traceback.format_exc());raise


if __name__=='__main__':main()
