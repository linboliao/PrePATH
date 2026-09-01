import csv
import os
import subprocess
import time

REPO = '/NAS3/lbliao/Code-138/PrePATH'
PY = '/data12/jing/anaconda3/envs/PrePATH/bin/python'
MANIFEST = os.path.join(REPO, 'csv/Prostate_Diagnosis_hoptimus1/shard_manifest.csv')
FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
WSI_ROOT = '/NAS145/linboliao/Data/迈新生物'
LOG_DIR = os.path.join(REPO, 'logs/hoptimus1_extract')
os.makedirs(LOG_DIR, exist_ok=True)

with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))

env_base = os.environ.copy()
env_base['PYTHONPATH'] = REPO + ':' + env_base.get('PYTHONPATH', '')
env_base['LD_LIBRARY_PATH'] = os.path.expanduser('~/miniconda3/envs/PrePATH/lib') + ':' + env_base.get('LD_LIBRARY_PATH', '')

for row in rows:
    pool = row['pool']
    gpu = row['gpu']
    csv_path = row['csv_path']
    shard = row['global_shard']
    coors_dir = os.path.join(FEAT_ROOT, pool, 'patches_0_224')
    slide_dir = os.path.join(WSI_ROOT, pool)
    feat_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224')
    log_path = os.path.join(LOG_DIR, f'shard{shard}_{pool}_gpu{gpu}.log')

    cmd = [
        PY, '-u', 'extract_features_fp_fast.py',
        '--data_coors_dir', coors_dir,
        '--data_slide_dir', slide_dir,
        '--slide_ext', '.svs;.kfb;.ndpi',
        '--batch_size', '64',
        '--csv_path', csv_path,
        '--feat_dir', feat_dir,
        '--model', 'h-optimus-1',
    ]
    env = env_base.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu

    with open(log_path, 'w') as logf:
        subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                          stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'launched shard {shard} ({pool}, {row["n_slides"]} slides) on gpu {gpu} -> {log_path}')
    time.sleep(5)  # stagger startup to avoid the HF stored_tokens write race seen with virchow2

print('\nAll 6 shards launched.')
