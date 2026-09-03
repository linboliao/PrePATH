#!/usr/bin/env bash
# Re-patch the 40x slides whose wrong-scale (224px) patches were moved to backup.
# create_patches_fp.py is now MPP-calibrated -> 40x (mpp ~0.22) slides get ~509px
# patches; auto_skip leaves every still-present 20x slide untouched.
set -uo pipefail
cd /NAS2/Data1/lbliao/Code-195/PrePATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=.:${PYTHONPATH:-}
PY=/home/lbliao/anaconda3/envs/clam/bin/python
WR=/NAS145/linboliao/Data/迈新生物
FR=/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis
LOGD=/home/lbliao/mil_runs/repatch_40x
mkdir -p "$LOGD"

for pool in "MIL外部测试" "MIL训练数据" "MIL测试数据"; do
  echo "=== re-patch $pool  ($(date +%H:%M)) ==="
  "$PY" -u create_patches_fp.py \
    --source "$WR/$pool" \
    --save_dir "$FR/$pool/patches_0_224" \
    --patch_size 224 --step_size 224 --patch_level 0 \
    --preset maixin.csv \
    --wsi_format "svs;kfb;tif;tiff" \
    --seg --patch --stitch --use_mp \
    > "$LOGD/$pool.log" 2>&1
  echo "  $pool done ($(date +%H:%M))"
done
echo "ALL RE-PATCH DONE $(date +%H:%M)"
