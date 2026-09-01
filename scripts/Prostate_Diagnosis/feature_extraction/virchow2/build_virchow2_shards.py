import os
import csv

FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
SHARD_ROOT = '/NAS3/lbliao/Code-138/PrePATH/csv/Prostate_Diagnosis_virchow2'

# proportional to pool size, summing to 16
POOLS = [
    ('MIL训练数据', 11),
    ('MIL测试数据', 2),
    ('MIL外部测试', 3),
]

os.makedirs(SHARD_ROOT, exist_ok=True)

shard_index = 0
manifest_rows = []
for pool, n_shards in POOLS:
    pt_dir = os.path.join(FEAT_ROOT, pool, 'feat_0_224', 'pt_files', 'uni2')
    slide_ids = sorted(f[:-3] for f in os.listdir(pt_dir) if f.endswith('.pt'))
    print(f'{pool}: {len(slide_ids)} slides -> {n_shards} shards')
    pool_dir = os.path.join(SHARD_ROOT, pool)
    os.makedirs(pool_dir, exist_ok=True)
    # contiguous chunking, near-equal sizes
    n = len(slide_ids)
    base, rem = divmod(n, n_shards)
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        chunk = slide_ids[start:start + size]
        start += size
        part_path = os.path.join(pool_dir, f'part_{i}.csv')
        with open(part_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case_id', 'slide_id'])
            for sid in chunk:
                w.writerow([sid, sid])
        gpu = shard_index % 8
        manifest_rows.append({
            'global_shard': shard_index,
            'gpu': gpu,
            'pool': pool,
            'csv_path': part_path,
            'n_slides': len(chunk),
        })
        print(f'  part_{i}.csv: {len(chunk)} slides -> gpu {gpu}')
        shard_index += 1

manifest_path = os.path.join(SHARD_ROOT, 'shard_manifest.csv')
with open(manifest_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['global_shard', 'gpu', 'pool', 'csv_path', 'n_slides'])
    w.writeheader()
    w.writerows(manifest_rows)
print('\nWrote shard manifest:', manifest_path)
print('Total shards:', shard_index, '| total slides:', sum(r['n_slides'] for r in manifest_rows))
