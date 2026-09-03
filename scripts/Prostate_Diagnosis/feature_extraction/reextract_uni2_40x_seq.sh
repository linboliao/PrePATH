#!/usr/bin/env bash
# Re-extract uni2 for the 185 re-patched 40x slides. One pool at a time, one
# shard per GPU (previous run OOM'd by putting 3 pools' shards on the same GPU).
set -uo pipefail
PRE=/NAS2/Data1/lbliao/Code-195/PrePATH
MIL=/NAS2/Data1/lbliao/Code-195/MIL_BASELINE
PY=/home/lbliao/anaconda3/envs/clam/bin/python
FR=/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis
WR=/NAS145/linboliao/Data/迈新生物
SCAN=$MIL/result/ProstateDiagnosis/DataAnalysis/mag_scan/slides_40x.csv
SHARD=$PRE/csv/ProstateDiagnosis_uni2_repatch40x
LOGD=/home/lbliao/mil_runs/reextract_uni2_40x
BS=48
NGPU=8
mkdir -p "$SHARD" "$LOGD"
cd "$PRE"
export PYTHONPATH=.:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$PRE/models/ckpts/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

for pool in "MIL外部测试" "MIL训练数据" "MIL测试数据"; do
  echo "=== $pool  $(date +%H:%M) ==="
  coors=$FR/$pool/patches_0_224
  feat=$FR/$pool/feat_0_224
  slidedir=$WR/$pool
  # build the per-GPU shard CSVs (only slides not already done, with an h5)
  "$PY" - "$pool" "$coors" "$feat" "$SHARD" "$NGPU" <<'PYEOF'
import sys, os, csv, pandas as pd
pool, coors, feat, shard, ngpu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
d = pd.read_csv("/NAS2/Data1/lbliao/Code-195/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis/mag_scan/slides_40x.csv")
d = d[(d.pool == pool) & (d.has_h5 == True)]
done = set(f[:-3] for f in os.listdir(f"{feat}/pt_files/uni2")) if os.path.isdir(f"{feat}/pt_files/uni2") else set()
todo = [str(s) for s in d.slide_id if str(s) not in done and os.path.exists(f"{coors}/patches/{s}.h5")]
for g in range(ngpu):
    p = f"{shard}/{pool}_g{g}.csv"
    chunk = todo[g::ngpu]
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id", "slide_id"])
        for s in chunk: w.writerow([s, s])
print(f"{pool}: {len(todo)} to extract")
PYEOF
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    part=$SHARD/${pool}_g${g}.csv
    [ "$(wc -l < "$part")" -le 1 ] && continue
    CUDA_VISIBLE_DEVICES=$g nohup "$PY" -u extract_features_fp_fast.py \
      --data_coors_dir "$coors" --data_slide_dir "$slidedir" \
      --slide_ext '.svs;.kfb;.ndpi;.tif;.tiff' --batch_size $BS \
      --csv_path "$part" --feat_dir "$feat" --model uni2 \
      > "$LOGD/${pool}_g${g}.log" 2>&1 &
    pids+=($!)
  done
  echo "  ${#pids[@]} shards on GPUs; waiting..."
  wait "${pids[@]}"
  echo "  $pool done $(date +%H:%M)"
done
echo "ALL UNI2 RE-EXTRACT DONE $(date +%H:%M)"
