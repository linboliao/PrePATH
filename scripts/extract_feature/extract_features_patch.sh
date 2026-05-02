export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=~/miniconda3/envs/PrePATH/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

model=dinov3
batch_size=32
patch_size=448

patch_img_dir=/NAS145/Data/LS/xh/image
feat_dir=/NAS145/Data/CRC/协和/feat_0_$patch_size
data_h5_dir=/NAS145/Data/CRC/协和/patches_0_$patch_size
csv_path=csv/extract_features_$model

#python scripts/extract_feature/generate_csv.py --h5_dir $data_h5_dir/patches --num 4 --root $csv_path
CUDA_VISIBLE_DEVICES=7 python extract_features_fp_from_patch.py --data_h5_dir $data_h5_dir --patch_img_dir $patch_img_dir --batch_size $batch_size --csv_path $csv_path/part_3.csv --feat_dir $feat_dir --model $model
