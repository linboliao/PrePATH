cd ../../

export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export CUDA_LAUNCH_BLOCKING=1

model=h-optimus-1
slide_ext='.svs;.kfb;.ndpi'  # The extension of the WSI files, remeber to keep the `.` in front
batch_size=64
patch_size=448

coors_dir=/NAS145/liaolinbo/Data/MXB/test/patches_0_$patch_size
wsi_dir=/NAS145/liaolinbo/Data/MXB/test/slides

feat_dir=/NAS145/liaolinbo/Data/MXB/test/feat_0_$patch_size
csv_path=csv/Contrast
part=0
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=6 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_$part.csv --feat_dir $feat_dir --model $model
