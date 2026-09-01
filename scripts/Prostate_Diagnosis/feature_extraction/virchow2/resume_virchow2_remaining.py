import os
import csv
import subprocess

REPO = '/NAS3/lbliao/Code-138/PrePATH'
PY = '/data12/jing/anaconda3/envs/PrePATH/bin/python'
FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
WSI_ROOT = '/NAS145/linboliao/Data/迈新生物'
SHARD_ROOT = os.path.join(REPO, 'csv/Prostate_Diagnosis_virchow2_resume')
LOG_DIR = os.path.join(REPO, 'logs/virchow2_extract')
os.makedirs(SHARD_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# free GPUs to use for finishing the remaining work
FREE_GPUS = [2, 3, 4, 5]

POOLS = ['MIL训练数据', 'MIL测试数据', 'MIL外部测试']

remaining_by_pool = {}
for pool in POOLS:
    uni2_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224', 'pt_files', 'uni2')
    virchow2_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224', 'pt_files', 'virchow2')
    all_ids = set(f[:-3] for f in os.listdir(uni2_dir) if f.endswith('.pt'))
    done_ids = set(f[:-3] for f in os.listdir(virchow2_dir) if f.endswith('.pt')) if os.path.isdir(virchow2_dir) else set()
    remaining = sorted(all_ids - done_ids)
    remaining_by_pool[pool] = remaining
    print(f'{pool}: {len(all_ids)} total, {len(done_ids)} done, {len(remaining)} remaining')

total_remaining = sum(len(v) for v in remaining_by_pool.values())
print(f'\nTotal remaining: {total_remaining}')

if total_remaining == 0:
    print('Nothing left to do.')
    raise SystemExit(0)

# Build one shard CSV per (pool, gpu) that has remaining work, distributing
# each pool's remaining slides round-robin-ish across the free GPUs
# proportional to how much remains in that pool.
n_gpu = len(FREE_GPUS)
env_base = os.environ.copy()
env_base['PYTHONPATH'] = REPO + ':' + env_base.get('PYTHONPATH', '')
env_base['LD_LIBRARY_PATH'] = os.path.expanduser('~/miniconda3/envs/PrePATH/lib') + ':' + env_base.get('LD_LIBRARY_PATH', '')

launched = 0
gpu_cursor = 0
for pool, remaining in remaining_by_pool.items():
    if not remaining:
        continue
    n = len(remaining)
    # split this pool's remaining slides across as many of the free GPUs as makes sense
    n_shards = min(n_gpu, max(1, n))
    base, rem = divmod(n, n_shards)
    start = 0
    pool_dir = os.path.join(SHARD_ROOT, pool)
    os.makedirs(pool_dir, exist_ok=True)
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        chunk = remaining[start:start + size]
        start += size
        gpu = FREE_GPUS[gpu_cursor % n_gpu]
        gpu_cursor += 1
        part_path = os.path.join(pool_dir, f'resume_part_{i}.csv')
        with open(part_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case_id', 'slide_id'])
            for sid in chunk:
                w.writerow([sid, sid])

        coors_dir = os.path.join(FEAT_ROOT, pool, 'patches_0_224')
        slide_dir = os.path.join(WSI_ROOT, pool)
        feat_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224')
        log_path = os.path.join(LOG_DIR, f'resume_{pool}_part{i}_gpu{gpu}.log')

        cmd = [
            PY, '-u', 'extract_features_fp_fast.py',
            '--data_coors_dir', coors_dir,
            '--data_slide_dir', slide_dir,
            '--slide_ext', '.svs;.kfb;.ndpi',
            '--batch_size', '64',
            '--csv_path', part_path,
            '--feat_dir', feat_dir,
            '--model', 'virchow2',
        ]
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with open(log_path, 'w') as logf:
            subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                              stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'launched {pool} resume_part_{i} ({len(chunk)} slides) on gpu {gpu} -> {log_path}')
        launched += 1

print(f'\nLaunched {launched} resume shards on GPUs {FREE_GPUS}.')
