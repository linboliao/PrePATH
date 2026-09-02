#!/bin/bash

# enable color correction, remove if you do not need this
export COLOR_CORRECTION_FLAG="TRUE"


# --- You Can Change Following Parameters ----
TASK_NAME=test_acrobat2023  # task name
image_h5_dir=/packed_h5/temp
slide_ext=.tiff
feat_dir=/jhcnas7/Pathology/code/PrePATH/temp_results #path to save feature
model="resnet50" # foundation models to be used
batch_size=32

# ----------------------------------------------

# ----DO NOT CHANGE THE FOLLOWING CODE----
# Unless you know what you are doing ----

csv_path=csv/$TASK_NAME
log_dir=scripts/extract_feature/logs
mkdir -p $log_dir


python scripts/extract_feature/generate_csv.py --h5_dir $image_h5_dir --num 1 --root $csv_path

nohup python extract_features_fp_from_packed_h5.py \
            --model $model \
            --csv_path $csv_path/part_$part.csv \
            --image_h5_dir $image_h5_dir \
            --feat_dir $feat_dir \
            --batch_size $batch_size > $log_dir/${TASK_NAME}_${model}_${part}.log 2>&1 &
        
        
 