import os
import csv

FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
SHARD_ROOT = '/NAS3/lbliao/Code-138/PrePATH/csv/Prostate_Diagnosis_hoptimus1_16way'
FREE_GPUS = list(range(8))

os.makedirs(SHARD_ROOT, exist_ok=True)

remaining_by_pool = {}
for pool in ['MIL训练数据', 'MIL测试数据', 'MIL外部测试']:
    uni2_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224', 'pt_files', 'uni2')
    done_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224', 'pt_files', 'h-optimus-1')
    all_ids = set(f[:-3] for f in os.listdir(uni2_dir) if f.endswith('.pt'))
    done_ids = set(f[:-3] for f in os.listdir(done_dir) if f.endswith('.pt')) if os.path.isdir(done_dir) else set()
    remaining = sorted(all_ids - done_ids)
    remaining_by_pool[pool] = remaining
    print(f'{pool}: {len(all_ids)} total, {len(done_ids)} done, {len(remaining)} remaining')

total_remaining = sum(len(v) for v in remaining_by_pool.values())
print(f'\nTotal remaining: {total_remaining}')

# 16 shards total, proportional to each pool's remaining count
n_shards_target = 16
sizes = {p: len(v) for p, v in remaining_by_pool.items()}
raw = {p: n_shards_target * n / total_remaining for p, n in sizes.items()}
shard_counts = {p: max(1, round(v)) if sizes[p] > 0 else 0 for p, v in raw.items()}
# fix rounding so the total is exactly 16
diff = n_shards_target - sum(shard_counts.values())
# adjust the largest pool to absorb the rounding difference
largest_pool = max(sizes, key=lambda p: sizes[p])
shard_counts[largest_pool] += diff
print('shard_counts:', shard_counts)

shard_index = 0
manifest_rows = []
for pool, remaining in remaining_by_pool.items():
    n_shards = shard_counts[pool]
    if n_shards == 0 or not remaining:
        continue
    n = len(remaining)
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
        gpu = FREE_GPUS[shard_index % len(FREE_GPUS)]
        part_path = os.path.join(pool_dir, f'part_{i}.csv')
        with open(part_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case_id', 'slide_id'])
            for sid in chunk:
                w.writerow([sid, sid])
        manifest_rows.append({
            'global_shard': shard_index,
            'gpu': gpu,
            'pool': pool,
            'csv_path': part_path,
            'n_slides': len(chunk),
        })
        print(f'  {pool} part_{i}.csv: {len(chunk)} slides -> gpu {gpu}')
        shard_index += 1

manifest_path = os.path.join(SHARD_ROOT, 'shard_manifest.csv')
with open(manifest_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['global_shard', 'gpu', 'pool', 'csv_path', 'n_slides'])
    w.writeheader()
    w.writerows(manifest_rows)
print('\nWrote shard manifest:', manifest_path)
print('Total shards:', shard_index, '| total slides:', sum(r['n_slides'] for r in manifest_rows))
