#!/bin/bash

# enable color correction, remove if you do not need this
export COLOR_CORRECTION_FLAG="TRUE"


# --- You Can Change Following Parameters ----
TASK_NAME=test_acrobat2023  # task name
wsi_dir=/jhcnas7/Pathology/original_data/Breast/ACROBAT2023/tiff
slide_ext=.tiff
feat_dir=/jhcnas7/Pathology/code/PrePATH/temp_results #path to save feature
coors_dir=/jhcnas7/Pathology/code/PrePATH/temp_results  # path where the coors files are saved
model="resnet50" # foundation models to be used
batch_size=32

# ----------------------------------------------

# ----DO NOT CHANGE THE FOLLOWING CODE----
# Unless you know what you are doing ----

csv_path=csv/$TASK_NAME
log_dir=scripts/extract_feature/logs
mkdir -p $log_dir


python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path

nohup python extract_features_fp_fast.py \
    --model $model \
    --csv_path $csv_path/part_$part.csv \
    --data_coors_dir $coors_dir \
    --data_slide_dir $wsi_dir \
    --feat_dir $feat_dir \
    --ignore_partial yes \
    --batch_size $batch_size \
    --datatype auto \
    --slide_ext $slide_ext \
    --save_storage "yes" > $log_dir/${TASK_NAME}_${model}_${part}.log 2>&1 &
        
 