#!/usr/bin/env bash
# Re-extract uni2 for ALL slides using the MPP-calibrated coords under
#   /NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis/<pool>/patches_0_224
# 16 shards total, 2 shards per GPU (8x RTX 3080 Ti, 12 GB). auto_skip keeps the
# 185 already-present 40x .pt files. BS kept low so 2 uni2(ViT-h fp32) fit in 12 GB.
set -uo pipefail
PRE=/NAS2/Data1/lbliao/Code-195/PrePATH
PY=/home/lbliao/anaconda3/envs/clam/bin/python
FR=/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis
WR=/NAS145/linboliao/Data/迈新生物
SHARD=$PRE/csv/ProstateDiagnosis_uni2_full16
LOGD=/home/lbliao/mil_runs/reextract_uni2_prostatediag
BS=32
NSHARD=8
NGPU=8
EXT='.svs;.kfb;.ndpi;.tif;.tiff'
mkdir -p "$SHARD" "$LOGD"
cd "$PRE"
export PYTHONPATH=.:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$PRE/models/ckpts/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for pool in "MIL外部测试" "MIL训练数据" "MIL测试数据"; do
  echo "=== $pool  $(date '+%F %H:%M') ==="
  coors=$FR/$pool/patches_0_224
  feat=$FR/$pool/feat_0_224
  slidedir=$WR/$pool

  "$PY" - "$coors" "$feat" "$SHARD" "$pool" "$NSHARD" <<'PYEOF'
import sys, os, csv
coors, feat, shard, pool, nshard = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
pdir = f"{coors}/patches"
allsl = sorted(f[:-3] for f in os.listdir(pdir) if f.endswith(".h5"))
ptdir = f"{feat}/pt_files/uni2"
done = set(f[:-3] for f in os.listdir(ptdir)) if os.path.isdir(ptdir) else set()
todo = [s for s in allsl if s not in done]
for g in range(nshard):
    with open(f"{shard}/{pool}_s{g}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["case_id", "slide_id"])
        for s in todo[g::nshard]:
            w.writerow([s, s])
print(f"{pool}: {len(allsl)} coords, {len(done)} already done, {len(todo)} to extract")
PYEOF

  pids=()
  for s in $(seq 0 $((NSHARD-1))); do
    part=$SHARD/${pool}_s${s}.csv
    [ "$(wc -l < "$part")" -le 1 ] && continue
    gpu=$(( s % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u extract_features_fp_fast.py \
      --data_coors_dir "$coors" --data_slide_dir "$slidedir" \
      --slide_ext "$EXT" --batch_size $BS \
      --csv_path "$part" --feat_dir "$feat" --model uni2 \
      > "$LOGD/${pool}_s${s}.log" 2>&1 &
    pids+=($!)
  done
  echo "  ${#pids[@]} shards launched (1/GPU); waiting..."
  wait "${pids[@]}"
  nowpt=$(ls "$feat/pt_files/uni2" 2>/dev/null | wc -l)
  echo "  $pool done $(date '+%F %H:%M')  uni2 pt now=$nowpt"
done
echo "ALL UNI2 RE-EXTRACT DONE $(date '+%F %H:%M')"
