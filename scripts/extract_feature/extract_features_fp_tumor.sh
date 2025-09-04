export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
#export CUDA_LAUNCH_BLOCKING=1
cd ../../
coors_dir=/NAS2/Data1/lbliao/Data/CRC/协和/patches
wsi_dir=/NAS2/Data4/llb/协和医院结直肠癌数据/slides
feat_dir=/NAS2/Data1/lbliao/Data/CRC/协和/tumor_feat
csv_path=csv/extract_features_tumor
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 5 --root $csv_path
CUDA_VISIBLE_DEVICES=1 python extract_features_fp_tumor.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext '.ndpi;.kfb' --csv_path $csv_path/part_1.csv --feat_dir $feat_dir --model uni