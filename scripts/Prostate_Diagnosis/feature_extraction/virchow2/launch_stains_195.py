#!/usr/bin/env python
"""Re-extract features for the 301 (+ ynzl) external cohorts WITH stain
normalisation, on the Code-195 box.  --model uni2|virchow2|h-optimus-1  --method macenko|reinhard|vahadane

Why: Group-B / Group-C showed the 301 specificity collapse is a feature-level
covariate shift that per-dim BN and CORAL can't fix. This maps 301 patches to
the TRAINING staining (reference montage from build_stain_reference.py) before
virchow2, then the frozen 5-fold AB_MIL ensemble is re-evaluated.

  - method   : macenko (default; --method reinhard|vahadane to compare)
  - output   : <FEAT_ROOT>/MIL外部测试/feat_0_224_stains/<Method>/{pt_files,h5_files}/<model>/
               (kept fully separate from the raw feat_0_224 tree)
  - cohorts  : 301 (147) + ynzl (the 136 with WSI+coords) ; ynzl is the safety control
  - resume   : re-run this; extract_features_fp_stains.py skips existing .pt

env handled here: PYTHONPATH, LD_LIBRARY_PATH (clam libstdc++), HF cache offline,
STAIN_REF_IMG, CUDA_VISIBLE_DEVICES.
"""
import argparse
import csv
import glob
import os
import subprocess

PRE = '/NAS2/Data1/lbliao/Code-195/PrePATH'
MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
PY = '/home/lbliao/anaconda3/envs/clam/bin/python'
CLAM_LIB = '/home/lbliao/anaconda3/envs/clam/lib'
FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
WSI_ROOT = '/NAS145/linboliao/Data/迈新生物'
POOL = 'MIL外部测试'


def wsi_index():
    idx = {}
    for f in glob.glob(os.path.join(WSI_ROOT, POOL, '**'), recursive=True):
        if f.lower().endswith('.svs'):
            idx['.'.join(os.path.basename(f).split('.')[:-1])] = f
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='uni2', help='feature encoder (uni2 / virchow2 / h-optimus-1 ...)')
    ap.add_argument('--method', default='macenko', choices=['macenko', 'reinhard', 'vahadane'])
    ap.add_argument('--ref', default=os.path.join(FEAT_ROOT, POOL, 'feat_0_224_stains',
                                                  'stain_reference_train_montage.png'))
    ap.add_argument('--cohorts', nargs='+', default=['301', 'ynzl'])
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    assert os.path.exists(a.ref), f'reference image not found: {a.ref}  (run build_stain_reference.py)'
    method_dir = a.method.capitalize()
    feat_dir = os.path.join(FEAT_ROOT, POOL, 'feat_0_224_stains', method_dir)
    coors_dir = os.path.join(FEAT_ROOT, POOL, 'patches_0_224')
    slide_dir = os.path.join(WSI_ROOT, POOL)
    shard_root = os.path.join(PRE, 'csv', f'ProstateDiagnosis_{a.model}_stains_{a.method}')
    log_dir = os.path.join(PRE, 'logs', f'{a.model}_stains_{a.method}')
    os.makedirs(shard_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    widx = wsi_index()
    done_dir = os.path.join(feat_dir, 'pt_files', a.model)
    done = set(f[:-3] for f in os.listdir(done_dir)) if os.path.isdir(done_dir) else set()

    # collect slide_ids that have coords + WSI + not done
    todo = []
    for c in a.cohorts:
        d = list(csv.DictReader(open(os.path.join(MIL, f'datasets/ProstateDiagnosis/external_test_{c}.csv'))))
        for row in d:
            sid = str(row['slide_id'])
            has_h5 = os.path.exists(os.path.join(coors_dir, 'patches', f'{sid}.h5'))
            if sid in widx and has_h5 and sid not in done:
                todo.append(sid)
    todo = sorted(set(todo))
    print(f'model={a.model}  method={a.method}  ref={a.ref}')
    print(f'feat_dir={feat_dir}')
    print(f'slides to (re)extract: {len(todo)}   already done: {len(done)}')
    if not todo:
        print('nothing to do')
        return

    n = len(a.gpus)
    shards = [todo[i::n] for i in range(n)]

    env_base = os.environ.copy()
    env_base['PYTHONPATH'] = PRE + ':' + env_base.get('PYTHONPATH', '')
    env_base['LD_LIBRARY_PATH'] = CLAM_LIB + ':' + env_base.get('LD_LIBRARY_PATH', '')
    env_base['HF_HOME'] = os.path.join(PRE, 'models/ckpts/huggingface')
    env_base['HF_ENDPOINT'] = 'https://hf-mirror.com'
    env_base['HF_HUB_OFFLINE'] = '1'
    env_base['STAIN_REF_IMG'] = a.ref
    env_base['TOKENIZERS_PARALLELISM'] = 'false'

    for gi, chunk in enumerate(shards):
        if not chunk:
            continue
        gpu = a.gpus[gi]
        part = os.path.join(shard_root, f'part_gpu{gpu}.csv')
        with open(part, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case_id', 'slide_id'])
            for sid in chunk:
                w.writerow([sid, sid])
        log = os.path.join(log_dir, f'gpu{gpu}.log')
        cmd = [PY, '-u', 'extract_features_fp_stains.py',
               '--data_coors_dir', coors_dir,
               '--data_slide_dir', slide_dir,
               '--slide_ext', '.svs;.kfb;.ndpi',
               '--batch_size', str(a.batch_size),
               '--csv_path', part,
               '--feat_dir', feat_dir,
               '--model', a.model,
               '--stain_method', a.method,
               '--stain_ref', a.ref]
        print(f'gpu {gpu}: {len(chunk)} slides -> {log}')
        if a.dry_run:
            print('   ', ' '.join(cmd))
            continue
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with open(log, 'w') as lf:
            subprocess.Popen(cmd, cwd=PRE, env=env, stdin=subprocess.DEVNULL,
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    if not a.dry_run:
        print(f'\nlaunched on GPUs {a.gpus[:len(shards)]}. tail: {log_dir}/gpu*.log')


if __name__ == '__main__':
    main()
