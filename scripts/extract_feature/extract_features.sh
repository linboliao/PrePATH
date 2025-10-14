cd ../../
export LD_LIBRARY_PATH=~/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=.:$PYTHONPATH

model=virchow2
slide_ext='.svs;.kfb'  # The extension of the WSI files, remeber to keep the `.` in front
batch_size=128
wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/slides # The directory where the WSI files are stored
feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/feat_0_224 # path to save feature
coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/patches_0_224 # path where the coors files are saved
csv_path=csv/extract_features_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/测试一/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/测试一/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/测试一/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治2/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治2/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治2/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治3/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治3/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治3/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
#
#wsi_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/slides
#feat_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/feat_0_224
#coors_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/patches_0_224
#csv_path=csv/extract_features_$model
#
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
#CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model

wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌一/slides
feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌一/feat_0_224
coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌一/patches_0_224
csv_path=csv/extract_features_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model

wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌二/slides
feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌二/feat_0_224
coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌二/patches_0_224
csv_path=csv/extract_features_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model

wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌三/slides
feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌三/feat_0_224
coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌三/patches_0_224
csv_path=csv/extract_features_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model

wsi_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌四/slides
feat_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌四/feat_0_224
coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/无癌四/patches_0_224
csv_path=csv/extract_features_$model

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 1 --root $csv_path
CUDA_VISIBLE_DEVICES=5 python extract_features_fp_fast.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --batch_size $batch_size --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model $model
