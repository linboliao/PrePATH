export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=~/miniconda3/envs/PrePATH/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

#model=h-optimus-1
#slide_ext='.svs;.kfb;.ndpi'  # The extension of the WSI files, remeber to keep the `.` in front
#batch_size=64
#patch_size=448
#
#coors_dir=/NAS145/Data/CRC/协和/patches_0_$patch_size
#wsi_dir=/NAS4/llb/协和医院结直肠癌数据
#
#feat_dir=/NAS145/Data/CRC/协和/feat_0_$patch_size
#csv_path=csv/extract_features_xh_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=1 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#coors_dir=/NAS145/Data/CRC/中日/patches_0_$patch_size
#wsi_dir=/NAS4/llb/中日友好医院结直肠癌数据
#
#feat_dir=/NAS145/Data/CRC/中日/feat_0_$patch_size
#csv_path=csv/extract_features_zz_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=1 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#coors_dir=/NAS145/Data/CRC/浙肿/patches_0_$patch_size
#wsi_dir=/NAS4/llb/浙江省肿瘤医院结直肠癌数据
#
#feat_dir=/NAS145/Data/CRC/浙肿/feat_0_$patch_size
#csv_path=csv/extract_features_zz_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=1 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model



model=h-optimus-1
slide_ext='.svs;.kfb;.ndpi'  # The extension of the WSI files, remeber to keep the `.` in front
batch_size=64
patch_size=448

coors_dir=/NAS145/Data/MXB_gleason/patches_0_$patch_size
wsi_dir=/NAS145/Data/MXB_gleason/slides

feat_dir=/NAS145/Data/MXB_gleason/feat_0_$patch_size
csv_path=csv/extract_features_gl_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=6 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
