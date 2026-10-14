"""Continue authorized Phase 3 gates on the workstation while Slurm runs models.

Wait for extraction, assemble complete metadata, submit three-seed training,
then recompute saved predictions. Any failed job or validation stops progression.
Submission intent is persisted before sbatch; ambiguous submission is never retried.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


REMOTE = 'zhangye@10.9.9.66'
REMOTE_ROOT = '/datapool/home/zhangye/projects/surface-residue-ppi'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def command(argv, timeout=180):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(str(argv[:2])+': '+result.stdout[-3000:]+result.stderr[-3000:])
    return result.stdout


def ssh(cmd):
    return command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',REMOTE,cmd])


def wait_array(job, count, checkpoint):
    expected = {str(job)+'_'+str(i) for i in range(count)}
    network_errors = 0
    previous = None
    while True:
        try:
            output = ssh('sacct -X -j '+str(job)+' --noheader --parsable2 --format=JobID%40,State%30,ExitCode')
            network_errors = 0
        except Exception:
            network_errors += 1
            if network_errors >= 5:
                raise
            time.sleep(30)
            continue
        states = {}
        for line in output.splitlines():
            pieces = line.strip().split('|')
            if len(pieces)>=3 and pieces[0] in expected:
                states[pieces[0]] = dict(state=pieces[1].split()[0],exit_code=pieces[2])
        if states != previous:
            checkpoint(job_states=states)
            previous = states
        failures = {k:v for k,v in states.items() if v['state'] in
                    ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED')}
        if failures:
            raise RuntimeError('Slurm jobs need review; no downstream submission: '+str(failures))
        if set(states)==expected and all(v['state']=='COMPLETED' and v['exit_code']=='0:0' for v in states.values()):
            return
        if any(v['state']=='COMPLETED' and v['exit_code']!='0:0' for v in states.values()):
            raise RuntimeError('Nonzero Slurm exit code')
        time.sleep(30)


def download_reports(relative, predictions=False):
    args=['rsync','-rt','--include=*/','--include=report.json']
    if predictions:
        args += ['--include=*_val_predictions.npz','--include=*_val_predictions.json','--include=config.json']
    args += ['--exclude=*',REMOTE+':'+REMOTE_ROOT+'/'+relative+'/',relative+'/']
    Path(relative).mkdir(parents=True,exist_ok=True)
    command(args,timeout=1800)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extraction-job',required=True,type=int)
    parser.add_argument('--state-dir',type=Path,default=Path('runs/phase3_orchestration/20260927/run_v1'))
    args=parser.parse_args()
    if Path('.').resolve()!=Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Use the authorized workstation')
    args.state_dir.mkdir(parents=True,exist_ok=True)
    lock=(args.state_dir/'lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.state_dir/'state.json'
    config=Path('configs/experiments/phase3_20260927.json')
    state=read(path) if path.exists() else dict(extraction_job=args.extraction_job,
        config_sha256=digest(config),created_at=time.time())
    if state['extraction_job']!=args.extraction_job or state['config_sha256']!=digest(config):
        raise ValueError('Continuation inputs differ')
    if state.get('submission_intent') and not state.get('training_job'):
        raise RuntimeError('Prior sbatch may have submitted: reconcile job queue before retrying')
    if state.get('status')=='main_comparison_complete':
        print('Already complete; nothing resubmitted')
        return

    def checkpoint(**updates):
        state.update(updates,updated_at=time.time(),pid=os.getpid())
        temporary=path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(path)
        print(json.dumps(updates),flush=True)

    try:
        merged=Path('runs/phase3_features/expanded_v1/report.json')
        if not state.get('training_job'):
            checkpoint(status='waiting_features')
            wait_array(args.extraction_job,4,checkpoint)
            download_reports('runs/phase3_features')
            shards=[]
            for report in Path('runs/phase3_features').glob('*/report.json'):
                if read(report).get('array_job_id')==str(args.extraction_job):
                    shards.append(report)
            if len(shards)!=4:
                raise ValueError('Expected exactly four feature shard reports')
            checkpoint(status='assembling_features',shard_reports=[str(p) for p in shards])
            if not merged.exists():
                command([sys.executable,'scripts/merge_phase3_features.py',
                    '--protocol','runs/phase3_protocol/20260927/expanded_v1/report.json',
                    '--reuse','runs/phase2_features/275876_final_v2/report.json',
                    '--shards',*[str(p) for p in sorted(shards)],'--out',str(merged)],timeout=1800)
            combined=read(merged)
            if combined['status']!='passed' or combined['failures'] or combined['records_requested']!=5328:
                raise ValueError('Merged feature manifest is incomplete')
            actual_sources={s['sha256'] for s in combined['sources'] if s['role']=='new_phase3_cache'}
            if actual_sources!={digest(p) for p in shards}:
                raise ValueError('Existing merged report belongs to different shards')
            command(['rsync','-rt','--relative',str(merged),REMOTE+':'+REMOTE_ROOT+'/'],timeout=600)
            verified=ssh('cd '+REMOTE_ROOT+' && sha256sum runs/phase3_features/expanded_v1/report.json'
                         ' && sha256sum -c reports/phase3_launch_code_20260927.sha256')
            if digest(merged)+'  runs/phase3_features/expanded_v1/report.json' not in verified:
                raise ValueError('Cluster merged report hash mismatch')
            (args.state_dir/'training_preflight.txt').write_text(verified)
            checkpoint(status='submitting_training',submission_intent=time.time(),feature_report_sha256=digest(merged))
            output=ssh('cd '+REMOTE_ROOT+' && sbatch --parsable slurm/train_phase3_task1.sbatch')
            (args.state_dir/'training_submission.txt').write_text(output)
            ids=[line.strip().split(';')[0] for line in output.splitlines()
                 if re.fullmatch(r'\d+(;[^\s]+)?',line.strip())]
            if len(ids)!=1:
                raise ValueError('Cannot determine submitted training job; do not resubmit blindly')
            checkpoint(training_job=int(ids[0]),status='waiting_training')
        wait_array(state['training_job'],3,checkpoint)
        download_reports('runs/phase3_task1_training',predictions=True)
        reports=[p for p in Path('runs/phase3_task1_training').glob('*/report.json')
                 if read(p).get('feature_report_sha256')==state['feature_report_sha256']]
        if len(reports)!=3:
            raise ValueError('Expected exactly three training reports for the frozen cache')
        checkpoint(status='summarizing_validation',training_reports=[str(p) for p in reports])
        summary=Path('runs/phase3_task1_summary/20260927/main_v1/report.json')
        if not summary.exists():
            command([sys.executable,'scripts/summarize_phase3_task1.py','--reports',
                     *[str(p) for p in sorted(reports)],'--out',str(summary)],timeout=1800)
        if read(summary)['status']!='passed':
            raise ValueError('Summary did not pass validation')
        command(['rsync','-rt','--relative',str(summary),REMOTE+':'+REMOTE_ROOT+'/'],timeout=600)
        checkpoint(status='main_comparison_complete',summary=str(summary),
                   next='Review paired results, then implement predefined diagnostics; test remains sealed')
    except Exception:
        checkpoint(status='needs_review',error=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
