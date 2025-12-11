cd ../../
export LD_LIBRARY_PATH=~/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=.:$PYTHONPATH

model=h-optimus-1
slide_ext='.svs;.kfb'  # The extension of the WSI files, remeber to keep the `.` in front
batch_size=32
patch_size=448
wsi_dir=/NAS145/Data/MXB/CLS测试 # The directory where the WSI files are stored
feat_dir=/NAS145/Data/MXB/CLS测试/feat_stains_0_$patch_size # path to save feature
coors_dir=/NAS145/Data/MXB/CLS测试/patches_0_$patch_size # path where the coors files are saved
csv_path=csv/extract_features_$model

#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 4 --root $csv_path
CUDA_VISIBLE_DEVICES=2 python extract_features_fp_stains.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_3.csv --feat_dir $feat_dir --model $model
