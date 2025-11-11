export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=~/miniconda3/envs/PrePATH/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

model=h-optimus-1
slide_ext='.svs;.kfb'  # The extension of the WSI files, remeber to keep the `.` in front
batch_size=32
patch_size=448
coors_dir=/NAS145/Data/CRC/TCGA_COAD/patch_0_448
wsi_dir=

feat_dir=/NAS145/Data/CRC/TCGA_COAD/feat_0_448
csv_path=csv/extract_features_$model

#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 3 --root $csv_path
CUDA_VISIBLE_DEVICES=3 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_2.csv --feat_dir $feat_dir --model $model
