#!/usr/bin/env python
"""Re-extract uni2 features for the 185 re-patched 40x slides.

Their old (wrong-scale) patches + features are in _BACKUP_wrong_40x_20260902/;
patches_0_224 now holds the MPP-calibrated coords (508px / 456px). This runs
extract_features_fp_fast.py --model uni2 per pool, sharded across GPUs. It skips
any slide whose .pt already exists, so re-running resumes.
"""
import argparse
import csv
import os
import subprocess

import pandas as pd

PRE = '/NAS2/Data1/lbliao/Code-195/PrePATH'
MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
PY = '/home/lbliao/anaconda3/envs/clam/bin/python'
FR = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
WR = '/NAS145/linboliao/Data/迈新生物'
SCAN = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/mag_scan/slides_40x.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='uni2')
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    df = pd.read_csv(SCAN)
    df = df[df.has_h5 == True]  # noqa: E712
    shard_root = f'{PRE}/csv/ProstateDiagnosis_{a.model}_repatch40x'
    log_dir = f'/home/lbliao/mil_runs/reextract_{a.model}_40x'
    os.makedirs(shard_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = PRE + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = '/home/lbliao/anaconda3/envs/clam/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['HF_HOME'] = f'{PRE}/models/ckpts/huggingface'
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    env['HF_HUB_OFFLINE'] = '1'

    gi = 0
    for pool, g in df.groupby('pool'):
        coors_dir = f'{FR}/{pool}/patches_0_224'
        slide_dir = f'{WR}/{pool}'
        feat_dir = f'{FR}/{pool}/feat_0_224'
        done_dir = f'{feat_dir}/pt_files/{a.model}'
        done = set(f[:-3] for f in os.listdir(done_dir)) if os.path.isdir(done_dir) else set()
        todo = [str(s) for s in g.slide_id if str(s) not in done
                and os.path.exists(f'{coors_dir}/patches/{s}.h5')]
        if not todo:
            print(f'{pool}: nothing to do')
            continue
        n = min(len(a.gpus), max(1, len(todo)))
        shards = [todo[i::n] for i in range(n)]
        print(f'{pool}: {len(todo)} slides -> {n} shards')
        for j, chunk in enumerate(shards):
            if not chunk:
                continue
            gpu = a.gpus[(gi) % len(a.gpus)]
            gi += 1
            part = f'{shard_root}/{pool}_gpu{gpu}_{j}.csv'
            with open(part, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['case_id', 'slide_id'])
                for s in chunk:
                    w.writerow([s, s])
            log = f'{log_dir}/{pool}_gpu{gpu}_{j}.log'
            cmd = [PY, '-u', 'extract_features_fp_fast.py',
                   '--data_coors_dir', coors_dir,
                   '--data_slide_dir', slide_dir,
                   '--slide_ext', '.svs;.kfb;.ndpi;.tif;.tiff',
                   '--batch_size', str(a.batch_size),
                   '--csv_path', part,
                   '--feat_dir', feat_dir,
                   '--model', a.model]
            print(f'  gpu {gpu}: {len(chunk)} slides -> {log}')
            if a.dry_run:
                print('    ' + ' '.join(cmd))
                continue
            e = env.copy()
            e['CUDA_VISIBLE_DEVICES'] = str(gpu)
            with open(log, 'w') as lf:
                subprocess.Popen(cmd, cwd=PRE, env=e, stdin=subprocess.DEVNULL,
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    if not a.dry_run:
        print(f'\nlaunched. logs: {log_dir}/')


if __name__ == '__main__':
    main()
